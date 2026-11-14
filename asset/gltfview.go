package asset

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/qmuntal/gltf"
)

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
		offset: int(view.ByteOffset) + int(acc.ByteOffset),
		nComp:  nComponents(acc.Type),
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
	if v.offset+acc.Count*v.stride > len(buf)+v.stride {
		return nil, fmt.Errorf("accessor %d out of bounds", ai)
	}
	v.data = buf
	return v, nil
}

func (v *accessorView) count() int { return len(v.data) / v.stride }

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
