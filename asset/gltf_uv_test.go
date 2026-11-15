package asset

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestGLTFUVVFlip pins the V-coordinate convention: glTF stores UVs with a
// top-left origin (v-down), while beam textures sample v-up (OBJ convention,
// asset/texture.go fy=(1-v)*H). LoadModel must flip V at import so texel
// lookups land on the authored position. Without the flip, full-image maps
// render vertically mirrored (easy to miss) and texture ATLAS lookups land in
// dead zones — the Kenney city rendered as a fleet of flat-white trucks for
// exactly this reason.
func TestGLTFUVVFlip(t *testing.T) {
	// Minimal GLB with one triangle carrying TEXCOORD_0 (u, v) pairs.
	// Written as a hand-rolled GLB: JSON chunk + BIN chunk.
	glb := buildUVTestGLB(t)
	dir := t.TempDir()
	p := filepath.Join(dir, "uv.glb")
	if err := os.WriteFile(p, glb, 0o644); err != nil {
		t.Fatal(err)
	}
	lm, err := LoadModel(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(lm.Parts) == 0 || len(lm.Parts[0].Mesh.UVs) != 3 {
		t.Fatalf("expected 3 UVs, got %d parts", len(lm.Parts))
	}
	// Authored glTF UVs (top-left origin): v=0.25, 0.25, 0.75.
	// beam-space (v-up) must be flipped: 0.75, 0.75, 0.25.
	want := []float64{0.75, 0.75, 0.25}
	for i, uv := range lm.Parts[0].Mesh.UVs {
		if diff := uv.V - want[i]; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("UV[%d].V = %f, want %f (flipped from glTF v)", i, uv.V, want[i])
		}
	}
}

func buildUVTestGLB(t *testing.T) []byte {
	t.Helper()
	// Minimal but valid glTF 2.0 document: 3 positions, 3 uvs, 3 indices.
	jsonStr := `{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"scene":0,` +
		`"nodes":[{"mesh":0}],` +
		`"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]}],` +
		`"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1]}}],` +
		`"accessors":[` +
		`{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","max":[1,1,1],"min":[0,0,0]},` +
		`{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"},` +
		`{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR","max":[2],"min":[0]}],` +
		`"bufferViews":[` +
		`{"buffer":0,"byteOffset":0,"byteLength":36},` +
		`{"buffer":0,"byteOffset":36,"byteLength":24},` +
		`{"buffer":0,"byteOffset":60,"byteLength":6}],` +
		`"buffers":[{"byteLength":66}]}`

	// 3 positions (12 bytes each)
	pos := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
	// 3 UVs (8 bytes each) — v values 0.25, 0.25, 0.75
	uvs := []float32{0.5, 0.25, 0.5, 0.25, 0.5, 0.75}
	// indices
	idx := []uint16{0, 1, 2}

	bin := make([]byte, 0, 66)
	for _, f := range pos {
		b := f32bytes(f)
		bin = append(bin, b...)
	}
	for _, f := range uvs {
		b := f32bytes(f)
		bin = append(bin, b...)
	}
	bin = append(bin, byte(idx[0]), byte(idx[0]>>8), byte(idx[1]), byte(idx[1]>>8), byte(idx[2]), byte(idx[2]>>8))

	return wrapGLB(t, []byte(jsonStr), bin)
}

func f32bytes(f float32) []byte {
	b := make([]byte, 4)
	u := math.Float32bits(f)
	binary.LittleEndian.PutUint32(b, u)
	return b
}

// wrapGLB packs JSON + BIN chunks into a GLB container.
func wrapGLB(t *testing.T, jsonStr, bin []byte) []byte {
	t.Helper()
	var out bytes.Buffer
	binary.Write(&out, binary.LittleEndian, uint32(0x46546C67)) // magic
	binary.Write(&out, binary.LittleEndian, uint32(2))          // version

	pad := func(n int) int { return (4 - n%4) % 4 }
	jsonPad, binPad := pad(len(jsonStr)), pad(len(bin))
	total := 12 + 8 + len(jsonStr) + jsonPad + 8 + len(bin) + binPad
	binary.Write(&out, binary.LittleEndian, uint32(total))

	binary.Write(&out, binary.LittleEndian, uint32(len(jsonStr)+jsonPad))
	binary.Write(&out, binary.LittleEndian, uint32(0x4E4F534A)) // JSON
	out.Write(jsonStr)
	if jsonPad > 0 {
		out.Write(make([]byte, jsonPad))
	}

	binary.Write(&out, binary.LittleEndian, uint32(len(bin)+binPad))
	binary.Write(&out, binary.LittleEndian, uint32(0x004E4942)) // BIN
	out.Write(bin)
	if binPad > 0 {
		out.Write(make([]byte, binPad))
	}
	return out.Bytes()
}
