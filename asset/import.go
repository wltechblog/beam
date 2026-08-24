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
		case "vn":
			if len(tok) < 4 {
				continue
			}
			x, _ := strconv.ParseFloat(tok[1], 64)
			y, _ := strconv.ParseFloat(tok[2], 64)
			z, _ := strconv.ParseFloat(tok[3], 64)
			m.Norms = append(m.Norms, geom.Vec3{X: x, Y: y, Z: z}.Norm())
		case "f":
			idx := faceIndices(tok[1:], len(m.Verts))
			if len(idx) < 3 {
				continue
			}
			for k := 2; k < len(idx); k++ {
				m.Tris = append(m.Tris, [3]int32{int32(idx[0]), int32(idx[k-1]), int32(idx[k])})
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(m.Tris) == 0 {
		return nil, fmt.Errorf("%s: no faces", path)
	}
	return m, nil
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
