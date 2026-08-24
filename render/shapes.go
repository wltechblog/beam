package render

import (
	"math"

	"github.com/wltechblog/beam/geom"
)

// BoxMesh builds a unit box mesh between min and max with per-vertex normals.
func BoxMesh(min, max geom.Vec3) *geom.TriangleMesh {
	cx := (min.X + max.X) / 2
	cy := (min.Y + max.Y) / 2
	cz := (min.Z + max.Z) / 2
	hx := (max.X - min.X) / 2
	hy := (max.Y - min.Y) / 2
	hz := (max.Z - min.Z) / 2
	return CubeMesh(geom.Vec3{X: cx, Y: cy, Z: cz}, geom.Vec3{X: hx, Y: hy, Z: hz})
}

// CubeMesh builds an axis-aligned box centered at c with half-extents h.
func CubeMesh(c, h geom.Vec3) *geom.TriangleMesh {
	// 8 corners
	v := make([]geom.Vec3, 8)
	for i := 0; i < 8; i++ {
		x := -h.X
		if i&1 != 0 {
			x = h.X
		}
		y := -h.Y
		if i&2 != 0 {
			y = h.Y
		}
		z := -h.Z
		if i&4 != 0 {
			z = h.Z
		}
		v[i] = c.Add(geom.Vec3{X: x, Y: y, Z: z})
	}
	// 6 faces: quad corner indices (CCW from outside) + face normal
	faces := [][4]int{
		{0, 2, 3, 1}, // -z back
		{4, 5, 7, 6}, // +z front
		{0, 1, 5, 4}, // -y bottom
		{2, 6, 7, 3}, // +y top
		{0, 4, 6, 2}, // -x left
		{1, 3, 7, 5}, // +x right
	}
	normals := []geom.Vec3{{X: 0, Y: 0, Z: -1}, {X: 0, Y: 0, Z: 1}, {X: 0, Y: -1, Z: 0}, {X: 0, Y: 1, Z: 0}, {X: -1, Y: 0, Z: 0}, {X: 1, Y: 0, Z: 0}}
	m := &geom.TriangleMesh{}
	for fi, f := range faces {
		base := len(m.Verts)
		for _, ci := range f {
			m.Verts = append(m.Verts, v[ci])
			m.Norms = append(m.Norms, normals[fi])
		}
		m.Tris = append(m.Tris, [3]int32{int32(base), int32(base + 1), int32(base + 2)})
		m.Tris = append(m.Tris, [3]int32{int32(base), int32(base + 2), int32(base + 3)})
	}
	return m
}

// SphereMesh tessellates a sphere (UV parametrization) with smooth normals.
func SphereMesh(c geom.Vec3, r float64, seg, rings int) *geom.TriangleMesh {
	m := &geom.TriangleMesh{}
	for j := 0; j <= rings; j++ {
		v := float64(j) / float64(rings)
		phi := v * math.Pi
		for i := 0; i <= seg; i++ {
			u := float64(i) / float64(seg)
			theta := u * 2 * math.Pi
			n := geom.Vec3{
				X: math.Sin(phi) * math.Cos(theta),
				Y: math.Cos(phi),
				Z: math.Sin(phi) * math.Sin(theta),
			}
			m.Verts = append(m.Verts, c.Add(n.MulS(r)))
			m.Norms = append(m.Norms, n)
		}
	}
	row := seg + 1
	for j := 0; j < rings; j++ {
		for i := 0; i < seg; i++ {
			a := int32(j*row + i)
			b := a + 1
			ci := a + int32(row)
			d := ci + 1
			if j == 0 {
				m.Tris = append(m.Tris, [3]int32{a, d, b}) // pole fix: keep outward
				continue
			}
			if j == rings-1 {
				m.Tris = append(m.Tris, [3]int32{a, ci, d})
				continue
			}
			m.Tris = append(m.Tris, [3]int32{a, ci, b})
			m.Tris = append(m.Tris, [3]int32{b, ci, d})
		}
	}
	return m
}
