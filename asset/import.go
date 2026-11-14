package asset

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/wltechblog/beam/geom"
)

// LoadMesh loads a mesh by file extension: .obj/.gltf/.glb/.stl.
// Everything resolves to geom.TriangleMesh — the universal object form.
func LoadMesh(path string) (*geom.TriangleMesh, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".obj":
		return LoadOBJ(path)
	case ".stl":
		return LoadSTL(path)
	case ".gltf", ".glb":
		return LoadGLTF(path)
	default:
		// Try glTF first (most common), then OBJ, then STL.
		if m, err := LoadGLTF(path); err == nil {
			return m, nil
		}
		if m, err := LoadOBJ(path); err == nil {
			return m, nil
		}
		return LoadSTL(path)
	}
}

// LoadOBJ parses a minimal Wavefront OBJ (v lines, f lines, vn optional).
// Polygon faces with >3 vertices are fan-triangulated.
func LoadOBJ(path string) (*geom.TriangleMesh, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	m := &geom.TriangleMesh{}
	var texCoords []geom.UV  // raw vt list
	var rawNorms []geom.Vec3 // raw vn list
	type corner struct{ v, vt, vn int }
	var faces [][3]corner
	useUV, useVN := false, false

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		tok := strings.Fields(sc.Text())
		if len(tok) == 0 {
			continue
		}
		switch tok[0] {
		case "v":
			if len(tok) < 4 {
				continue
			}
			x, _ := strconv.ParseFloat(tok[1], 64)
			y, _ := strconv.ParseFloat(tok[2], 64)
			z, _ := strconv.ParseFloat(tok[3], 64)
			m.Verts = append(m.Verts, geom.Vec3{X: x, Y: y, Z: z})
		case "vt":
			if len(tok) < 3 {
				continue
			}
			u, _ := strconv.ParseFloat(tok[1], 64)
			v, _ := strconv.ParseFloat(tok[2], 64)
			texCoords = append(texCoords, geom.UV{U: u, V: v})
		case "vn":
			if len(tok) < 4 {
				continue
			}
			x, _ := strconv.ParseFloat(tok[1], 64)
			y, _ := strconv.ParseFloat(tok[2], 64)
			z, _ := strconv.ParseFloat(tok[3], 64)
			rawNorms = append(rawNorms, geom.Vec3{X: x, Y: y, Z: z}.Norm())
		case "f":
			var cns []corner
			for _, t := range tok[1:] {
				v, vt, vn, ok := parseCorner(t, len(m.Verts), len(texCoords), len(rawNorms))
				if !ok {
					continue
				}
				cns = append(cns, corner{v: v, vt: vt, vn: vn})
				useUV = useUV || vt >= 0
				useVN = useVN || vn >= 0
			}
			if len(cns) < 3 {
				continue
			}
			for k := 2; k < len(cns); k++ {
				faces = append(faces, [3]corner{cns[0], cns[k-1], cns[k]})
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(faces) == 0 {
		return nil, fmt.Errorf("%s: no faces", path)
	}

	// Weld corners: the same (v, vt, vn) triple maps to ONE vertex. When a
	// position index appears with a DIFFERENT vt/vn across faces (the
	// classic UV-seam case), split it into a new vertex — geometry grows
	// slightly, per-vertex UVs/normals stay exact.
	nVerts0 := len(m.Verts)
	normFor := make([]int32, nVerts0) // position -> vn index, -1 none
	uvFor := make([]int32, nVerts0)   // position -> vt index, -1 none
	for i := range normFor {
		normFor[i], uvFor[i] = -1, -1
	}
	// grow the per-position tables as seam splits append vertices
	growTables := func(n int) {
		for len(normFor) < n {
			normFor = append(normFor, -1)
			uvFor = append(uvFor, -1)
		}
	}
	remap := make(map[[3]int32]int32, len(faces)*3)
	m.Tris = make([][3]int32, 0, len(faces))
	for _, fce := range faces {
		var tri [3]int32
		for ci, c := range fce {
			key := [3]int32{int32(c.v), int32(c.vt), int32(c.vn)}
			if vi, ok := remap[key]; ok {
				tri[ci] = vi
				continue
			}
			vi := int32(c.v)
			if (c.vt >= 0 && uvFor[vi] != -1 && uvFor[vi] != int32(c.vt)) ||
				(c.vn >= 0 && normFor[vi] != -1 && normFor[vi] != int32(c.vn)) {
				// seam split: duplicate the position, carry no old UV/normal
				m.Verts = append(m.Verts, m.Verts[vi])
				vi = int32(len(m.Verts) - 1)
				growTables(len(m.Verts))
				// uvFor/normFor for the NEW index default to -1 (fresh)
			}
			if c.vt >= 0 {
				uvFor[vi] = int32(c.vt)
			}
			if c.vn >= 0 {
				normFor[vi] = int32(c.vn)
			}
			remap[key] = vi
			tri[ci] = vi
		}
		m.Tris = append(m.Tris, tri)
	}
	if useUV {
		m.UVs = make([]geom.UV, len(m.Verts))
		for i := range m.UVs {
			if uvFor[i] >= 0 && uvFor[i] < int32(len(texCoords)) {
				m.UVs[i] = texCoords[uvFor[i]]
			}
		}
	}
	if useVN {
		m.Norms = make([]geom.Vec3, len(m.Verts))
		for i := range m.Norms {
			if normFor[i] >= 0 && normFor[i] < int32(len(rawNorms)) {
				m.Norms[i] = rawNorms[normFor[i]]
			}
		}
	}
	return m, nil
}

// parseCorner parses one OBJ face token — v, v/vt, v//vn, v/vt/vn — into
// 0-based indices (vt/vn = -1 when absent). 1-based and negative
// (relative) indices per the OBJ spec.
func parseCorner(t string, nVerts, nVT, nVN int) (v, vt, vn int, ok bool) {
	parts := strings.Split(t, "/")
	n, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, -1, -1, false
	}
	v = relIdx(n, nVerts)
	if v < 0 {
		return 0, -1, -1, false
	}
	vt, vn = -1, -1
	if len(parts) > 1 && parts[1] != "" {
		if n, err := strconv.Atoi(parts[1]); err == nil {
			if r := relIdx(n, nVT); r >= 0 {
				vt = r
			}
		}
	}
	if len(parts) > 2 && parts[2] != "" {
		if n, err := strconv.Atoi(parts[2]); err == nil {
			if r := relIdx(n, nVN); r >= 0 {
				vn = r
			}
		}
	}
	return v, vt, vn, true
}

func relIdx(n, total int) int {
	if n < 0 {
		n = total + n + 1
	}
	if n < 1 || n > total {
		return -1
	}
	return n - 1
}

// faceIndices converts OBJ face tokens (v, v/vt, v//vn, v/vt/vn — 1-based,
// negative = relative) to plain vertex indices.
func faceIndices(tok []string, nVerts int) []int {
	var out []int
	for _, t := range tok {
		parts := strings.SplitN(t, "/", 2)
		n, err := strconv.Atoi(parts[0])
		if err != nil {
			continue
		}
		if n < 0 {
			n = nVerts + n + 1
		}
		if n < 1 || n > nVerts {
			continue
		}
		out = append(out, n-1)
	}
	return out
}

// LoadSTL parses binary STL (80-byte header + triangle count + records).
// ASCII STL is detected (header starting with "solid" + parseable facets)
// and handled too.
func LoadSTL(path string) (*geom.TriangleMesh, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// Binary STL: 84 + 50*count == len (± tolerance for some writers).
	if len(raw) >= 84 {
		n := binary.LittleEndian.Uint32(raw[80:84])
		if int(84+50*int(n)) == len(raw) && n > 0 {
			return parseBinarySTL(raw, int(n))
		}
	}
	return parseASCIISTL(raw)
}

func parseBinarySTL(raw []byte, n int) (*geom.TriangleMesh, error) {
	m := &geom.TriangleMesh{}
	m.Verts = make([]geom.Vec3, 0, 3*n)
	m.Tris = make([][3]int32, 0, n)
	for i := 0; i < n; i++ {
		o := 84 + 50*i
		a := geom.Vec3{X: readLE32(raw, o+12), Y: readLE32(raw, o+16), Z: readLE32(raw, o+20)}
		b := geom.Vec3{X: readLE32(raw, o+24), Y: readLE32(raw, o+28), Z: readLE32(raw, o+32)}
		c := geom.Vec3{X: readLE32(raw, o+36), Y: readLE32(raw, o+40), Z: readLE32(raw, o+44)}
		base := int32(len(m.Verts))
		m.Verts = append(m.Verts, a, b, c)
		m.Tris = append(m.Tris, [3]int32{base, base + 1, base + 2})
	}
	return m, nil
}

func readLE32(b []byte, o int) float64 {
	return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[o:])))
}

func parseASCIISTL(raw []byte) (*geom.TriangleMesh, error) {
	m := &geom.TriangleMesh{}
	sc := bufio.NewScanner(strings.NewReader(string(raw)))
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var face []int
	for sc.Scan() {
		tok := strings.Fields(sc.Text())
		if len(tok) == 0 {
			continue
		}
		if tok[0] == "vertex" && len(tok) >= 4 {
			x, _ := strconv.ParseFloat(tok[1], 64)
			y, _ := strconv.ParseFloat(tok[2], 64)
			z, _ := strconv.ParseFloat(tok[3], 64)
			face = append(face, len(m.Verts))
			m.Verts = append(m.Verts, geom.Vec3{X: x, Y: y, Z: z})
			if len(face) == 3 {
				m.Tris = append(m.Tris, [3]int32{int32(face[0]), int32(face[1]), int32(face[2])})
				face = nil
			}
		} else if tok[0] == "endfacet" {
			face = nil
		}
	}
	if len(m.Tris) == 0 {
		return nil, fmt.Errorf("stl: no facets (not a valid binary or ASCII STL)")
	}
	return m, nil
}
