package asset

import (
	"fmt"
	"strings"

	"github.com/rclancey/earcut"
	"github.com/wltechblog/beam/geom"
	"seehuhn.de/go/geom/path"
	"seehuhn.de/go/geom/vec"
	"seehuhn.de/go/sfnt"
	"seehuhn.de/go/sfnt/glyf"
)

// TextMesh generates extruded 3D text standing upright in the XY plane,
// reading left-to-right, extruded along +Z. size is the em height in world
// units; spacing is extra letter spacing in em units.
func TextMesh(text, fontPath string, size, depth, spacing float64) (*geom.TriangleMesh, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text: empty string")
	}
	if size <= 0 {
		size = 1
	}
	f, err := sfnt.ReadFile(fontPath)
	if err != nil {
		return nil, fmt.Errorf("text: load font %q: %w", fontPath, err)
	}
	glyfOut, ok := f.Outlines.(*glyf.Outlines)
	if !ok {
		return nil, fmt.Errorf("text: font %q: CFF outlines not supported (use a TrueType font)", fontPath)
	}
	sub, err := f.CMapTable.GetBest()
	if err != nil {
		return nil, fmt.Errorf("text: font %q has no usable cmap: %w", fontPath, err)
	}

	upem := float64(f.UnitsPerEm)
	if upem == 0 {
		upem = 1000
	}
	scale := size / upem

	m := &geom.TriangleMesh{}
	penX := 0.0 // advance in font units

	for _, r := range text {
		gid := sub.Lookup(r)
		adv := f.GlyphWidth(gid)
		if r == ' ' {
			penX += adv
			continue
		}
		gp := glyfOut.Path(gid)
		contours := contoursOf(gp, upem) // font units, y-down converted to y-up baseline 0
		if len(contours) > 0 {
			addExtrudedGlyph(m, contours, penX, scale, depth)
		}
		penX += adv
		if spacing != 0 {
			penX += spacing * upem
		}
	}
	if len(m.Tris) == 0 {
		return nil, fmt.Errorf("text: no renderable glyphs in %q (font may lack coverage)", text)
	}

	// Beam cameras conventionally sit on the -Z side looking at the origin,
	// where world +X appears screen-LEFT. Bake a 180° Y rotation so text
	// faces -Z and reads correctly for those cameras, and centre the run on
	// the origin. rotY(180) is a proper rotation, so winding is preserved.
	w := penX * scale
	for i := range m.Verts {
		m.Verts[i].X = w/2 - m.Verts[i].X
		m.Verts[i].Z = -m.Verts[i].Z
	}
	return m, nil
}

// contoursOf flattens a glyph path into closed 2D polygons (font units,
// y-flipped so the baseline is 0 and up is +y).
func contoursOf(p path.Path, upem float64) [][][2]float64 {
	var contours [][][2]float64
	var cur [][2]float64
	var last [2]float64
	started := false

	// Font path is already y-up (baseline 0, ascender +y): identity map.
	flip := func(v vec.Vec2) [2]float64 {
		return [2]float64{v.X, v.Y}
	}

	p(func(cmd path.Command, pts []vec.Vec2) bool {
		switch cmd {
		case path.CmdMoveTo:
			if len(cur) > 2 {
				contours = append(contours, cur)
			}
			cur = [][2]float64{flip(pts[0])}
			last = flip(pts[0])
			started = true
		case path.CmdLineTo:
			if started {
				pt := flip(pts[0])
				cur = append(cur, pt)
				last = pt
			}
		case path.CmdQuadTo:
			if started {
				c := flip(pts[0])
				e := flip(pts[1])
				cur = append(cur, flattenQuad(last, c, e, 8)...)
				last = e
			}
		case path.CmdCubeTo:
			if started {
				c1 := flip(pts[0])
				c2 := flip(pts[1])
				e := flip(pts[2])
				cur = append(cur, flattenCubic(last, c1, c2, e, 10)...)
				last = e
			}
		case path.CmdClose:
			if len(cur) > 2 {
				contours = append(contours, cur)
			}
			cur = nil
			started = false
		}
		return true
	})
	if len(cur) > 2 {
		contours = append(contours, cur)
	}
	return contours
}

func flattenQuad(p0, p1, p2 [2]float64, steps int) [][2]float64 {
	var out [][2]float64
	for i := 1; i <= steps; i++ {
		t := float64(i) / float64(steps)
		mt := 1 - t
		x := mt*mt*p0[0] + 2*mt*t*p1[0] + t*t*p2[0]
		y := mt*mt*p0[1] + 2*mt*t*p1[1] + t*t*p2[1]
		out = append(out, [2]float64{x, y})
	}
	return out
}

func flattenCubic(p0, p1, p2, p3 [2]float64, steps int) [][2]float64 {
	var out [][2]float64
	for i := 1; i <= steps; i++ {
		t := float64(i) / float64(steps)
		mt := 1 - t
		x := mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0]
		y := mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1]
		out = append(out, [2]float64{x, y})
	}
	return out
}

// addExtrudedGlyph triangulates one glyph's contour set (with holes) and
// extrudes it. penX is the advance offset in font units; scale converts font
// units to world units; depth is the extrusion in world units.
func addExtrudedGlyph(m *geom.TriangleMesh, contours [][][2]float64, penX, scale, depth float64) {
	// Build the flat polygon + hole index list for earcut.
	var data []float64
	var holeIdx []int
	for _, c := range contours {
		if len(c) < 3 {
			continue
		}
		if len(data) > 0 {
			holeIdx = append(holeIdx, len(data)/2)
		}
		for _, pt := range c {
			data = append(data, pt[0], pt[1])
		}
	}
	if len(data) < 6 {
		return
	}
	tris, err := earcut.Earcut(data, holeIdx, 2)
	if err != nil || len(tris) == 0 {
		return // skip malformed glyph; the rest of the text still renders
	}

	// Front face vertices (z=0): world X = (glyphX + penX) * scale, Y = glyphY * scale.
	nPts := len(data) / 2
	base := len(m.Verts)
	for i := 0; i < nPts; i++ {
		x := (data[2*i] + penX) * scale
		y := data[2*i+1] * scale
		m.Verts = append(m.Verts, geom.Vec3{X: x, Y: y, Z: 0})
	}
	// Back face vertices (z=depth).
	for i := 0; i < nPts; i++ {
		x := (data[2*i] + penX) * scale
		y := data[2*i+1] * scale
		m.Verts = append(m.Verts, geom.Vec3{X: x, Y: y, Z: depth})
	}

	// Front face triangles: earcut emits indices CCW in y-up space; front
	// faces point toward -Z, so emit (a,c,b) to keep them outward.
	for i := 0; i+2 < len(tris); i += 3 {
		a, b, c := tris[i], tris[i+1], tris[i+2]
		m.Tris = append(m.Tris, [3]int32{int32(base + a), int32(base + c), int32(base + b)})
	}
	// Back face triangles: opposite winding, offset by nPts.
	for i := 0; i+2 < len(tris); i += 3 {
		a, b, c := tris[i], tris[i+1], tris[i+2]
		m.Tris = append(m.Tris, [3]int32{int32(base + nPts + a), int32(base + nPts + b), int32(base + nPts + c)})
	}

	// Side walls: walk each contour's original boundary.
	off := 0
	for _, c := range contours {
		n := len(c)
		if n < 3 {
			continue
		}
		for i := 0; i < n; i++ {
			j := (i + 1) % n
			f0 := base + off + i // front i
			f1 := base + off + j // front j
			b0 := f0 + nPts      // back i
			b1 := f1 + nPts      // back j
			// One quad per edge, wound so the outward normal points right of
			// travel: outer rings are CCW after the y-flip (interior left),
			// hole rings CW (hole interior right) — "right of travel" is
			// outward for both. cross(f1-f0, b0-f0) = (dy,-dx,0)·d. ✓
			m.Tris = append(m.Tris, [3]int32{int32(f0), int32(f1), int32(b0)})
			m.Tris = append(m.Tris, [3]int32{int32(f1), int32(b1), int32(b0)})
		}
		off += n
	}
}
