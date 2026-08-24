package asset

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/qmuntal/gltf"
	"github.com/wltechblog/beam/geom"
)

// LoadGLTF decodes glTF 2.0 / GLB meshes via the pure-Go qmuntal/gltf
// library. POSITION + indices are extracted; NORMAL attributes are kept
// when present for smooth shading, otherwise the renderer falls back to
// face normals. Skinning/morph targets are ignored (static meshes).
func LoadGLTF(path string) (*geom.TriangleMesh, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	doc := &gltf.Document{}
	if err := gltf.NewDecoder(f).Decode(doc); err != nil {
		return nil, fmt.Errorf("gltf decode: %w", err)
	}
	if len(doc.Meshes) == 0 {
		return nil, fmt.Errorf("%s: no meshes", path)
	}

	m := &geom.TriangleMesh{}
	for _, mesh := range doc.Meshes {
		for _, prim := range mesh.Primitives {
			if err := appendPrimitive(m, doc, prim); err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
		}
	}
	if len(m.Tris) == 0 {
		return nil, fmt.Errorf("%s: no triangles", path)
	}
	return m, nil
}

// accessorView is a decoded accessor slice: data, per-element stride.
type accessorView struct {
	data      []byte
	stride    int   // bytes per element (incl. bufferView padding)
	compSize  int   // bytes per scalar component
	nComp     int   // scalar components per element
	offset    int   // start of first element
	scalarFmt uint8 // 0 float32, 1 uint16, 2 uint32
}

func viewOf(doc *gltf.Document, ai int) (*accessorView, error) {
	if ai < 0 || ai >= len(doc.Accessors) {
		return nil, fmt.Errorf("accessor %d out of range", ai)
	}
	acc := doc.Accessors[ai]
	if acc.BufferView == nil {
		return nil, fmt.Errorf("accessor %d has no buffer view", ai)
	}
	view := doc.BufferViews[*acc.BufferView]
	if view.Buffer >= len(doc.Buffers) {
		return nil, fmt.Errorf("buffer view references missing buffer %d", view.Buffer)
	}
	buf := doc.Buffers[view.Buffer].Data

	v := &accessorView{
		offset:    int(view.ByteOffset) + int(acc.ByteOffset),
		nComp:     nComponents(acc.Type),
		scalarFmt: 0,
	}
	switch acc.ComponentType {
	case gltf.ComponentFloat:
		v.compSize = 4
		v.scalarFmt = 0
	case gltf.ComponentUshort:
		v.compSize = 2
		v.scalarFmt = 1
	case gltf.ComponentUint:
		v.compSize = 4
		v.scalarFmt = 2
	default:
		return nil, fmt.Errorf("unsupported component type %v", acc.ComponentType)
	}
	v.stride = v.compSize * v.nComp
	if s := int(view.ByteStride); s > v.stride {
		v.stride = s
	}
	if v.offset+acc.Count*v.stride > len(buf)+v.stride { // tolerate 1-elem overrun guard below
		return nil, fmt.Errorf("accessor %d out of bounds", ai)
	}
	v.data = buf
	return v, nil
}

func (v *accessorView) count() int { return len(v.data) / v.stride } // upper bound

func (v *accessorView) floatElem(i, c int) float64 {
	o := v.offset + i*v.stride + c*v.compSize
	switch v.scalarFmt {
	case 1:
		return float64(binary.LittleEndian.Uint16(v.data[o:]))
	case 2:
		return float64(binary.LittleEndian.Uint32(v.data[o:]))
	default:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(v.data[o:])))
	}
}

func nComponents(t gltf.AccessorType) int {
	switch t {
	case gltf.AccessorScalar:
		return 1
	case gltf.AccessorVec3:
		return 3
	case gltf.AccessorVec2:
		return 2
	case gltf.AccessorVec4:
		return 4
	case gltf.AccessorMat2:
		return 4
	case gltf.AccessorMat3:
		return 9
	case gltf.AccessorMat4:
		return 16
	}
	return 0
}

func appendPrimitive(m *geom.TriangleMesh, doc *gltf.Document, prim *gltf.Primitive) error {
	posRef, ok := prim.Attributes["POSITION"]
	if !ok {
		return fmt.Errorf("primitive missing POSITION")
	}
	pv, err := viewOf(doc, posRef)
	if err != nil {
		return fmt.Errorf("POSITION: %w", err)
	}
	count := doc.Accessors[posRef].Count

	vertBase := len(m.Verts)
	for i := 0; i < count; i++ {
		x := pv.floatElem(i, 0)
		y := pv.floatElem(i, 1)
		z := pv.floatElem(i, 2)
		m.Verts = append(m.Verts, geom.Vec3{X: x, Y: y, Z: z})
	}

	// Optional per-vertex normals.
	if nRef, ok := prim.Attributes["NORMAL"]; ok {
		nv, err := viewOf(doc, nRef)
		if err == nil && doc.Accessors[nRef].Count == count {
			for i := 0; i < count; i++ {
				m.Norms = append(m.Norms, geom.Vec3{X: nv.floatElem(i, 0), Y: nv.floatElem(i, 1), Z: nv.floatElem(i, 2)})
			}
		}
	}

	if prim.Indices != nil {
		iv, err := viewOf(doc, *prim.Indices)
		if err != nil {
			return fmt.Errorf("indices: %w", err)
		}
		n := doc.Accessors[*prim.Indices].Count
		for k := 2; k < n; k += 3 {
			a, b, c := int(iv.floatElem(k-2, 0)), int(iv.floatElem(k-1, 0)), int(iv.floatElem(k, 0))
			if a == b || b == c || a == c {
				continue // degenerate
			}
			m.Tris = append(m.Tris, [3]int32{int32(vertBase + a), int32(vertBase + b), int32(vertBase + c)})
		}
		return nil
	}
	// Non-indexed: sequential triples.
	for k := 2; k < count; k += 3 {
		m.Tris = append(m.Tris, [3]int32{int32(vertBase + k - 2), int32(vertBase + k - 1), int32(vertBase + k)})
	}
	return nil
}
