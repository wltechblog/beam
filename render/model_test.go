package render

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/qmuntal/gltf"
	"github.com/wltechblog/beam/scene"
)

// buildMiniGLB writes a one-material GLB: a unit quad on XZ with UVs,
// textured, emissive — exercising the PBR mapping and texture-table path.
func buildMiniGLB(t *testing.T, dir string) string {
	var buf bytes.Buffer
	var views []*gltf.BufferView
	var accs []*gltf.Accessor

	addView := func(data []byte) int {
		off := buf.Len()
		buf.Write(data)
		views = append(views, &gltf.BufferView{Buffer: 0, ByteOffset: off, ByteLength: len(data)})
		return len(views) - 1
	}
	f32 := func(v ...float32) []byte {
		out := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(x))
		}
		return out
	}
	u16 := func(v ...uint16) []byte {
		out := make([]byte, 2*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint16(out[i*2:], x)
		}
		return out
	}

	// 2x2 red/green checker PNG.
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.NRGBA{R: 255, A: 255})
	img.Set(1, 1, color.NRGBA{R: 255, A: 255})
	img.Set(1, 0, color.NRGBA{G: 255, A: 255})
	img.Set(0, 1, color.NRGBA{G: 255, A: 255})
	var pb bytes.Buffer
	if err := png.Encode(&pb, img); err != nil {
		t.Fatal(err)
	}
	imgView := addView(pb.Bytes())

	pos := addView(f32(-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1))
	nrm := addView(f32(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0))
	uv := addView(f32(0, 0, 1, 0, 1, 1, 0, 1))
	idx := addView(u16(0, 1, 2, 0, 2, 3))

	acc := func(view int, ct gltf.ComponentType, at gltf.AccessorType, count int) int {
		accs = append(accs, &gltf.Accessor{BufferView: &view, ComponentType: ct, Type: at, Count: count})
		return len(accs) - 1
	}
	posAcc := acc(pos, gltf.ComponentFloat, gltf.AccessorVec3, 4)
	nrmAcc := acc(nrm, gltf.ComponentFloat, gltf.AccessorVec3, 4)
	uvAcc := acc(uv, gltf.ComponentFloat, gltf.AccessorVec2, 4)
	idxAcc := acc(idx, gltf.ComponentUshort, gltf.AccessorScalar, 6)

	doc := &gltf.Document{
		Asset: gltf.Asset{Version: "2.0"},
		Images: []*gltf.Image{{
			BufferView: &imgView, MimeType: "image/png", Name: "checker",
		}},
		Textures: []*gltf.Texture{{Source: ptrGI(0)}},
		Materials: []*gltf.Material{{
			Name: "TexMat",
			PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
				BaseColorTexture: &gltf.TextureInfo{Index: 0},
				MetallicFactor:   ptrF(0.0),
			},
		}},
		Meshes: []*gltf.Mesh{{
			Name: "quad",
			Primitives: []*gltf.Primitive{{
				Attributes: gltf.PrimitiveAttributes{
					"POSITION": posAcc, "NORMAL": nrmAcc, "TEXCOORD_0": uvAcc,
				},
				Indices: &idxAcc, Material: ptrGI(0),
			}},
		}},
		Nodes:  []*gltf.Node{{Name: "n", Mesh: ptrGI(0)}},
		Scenes: []*gltf.Scene{{Nodes: []int{0}}},
	}
	sc := 0
	doc.Scene = &sc
	doc.BufferViews = views
	doc.Accessors = accs
	doc.Buffers = []*gltf.Buffer{{ByteLength: buf.Len(), Data: buf.Bytes()}}

	path := filepath.Join(dir, "mini.glb")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := gltf.NewEncoder(f).Encode(doc); err != nil {
		t.Fatal(err)
	}
	return path
}

func ptrGI(v int) *int        { return &v }
func ptrF(v float64) *float64 { return &v }

// TestCompileModelObject drives scene JSON → Compile → World: one model
// object expands to mesh objects with imported materials, the texture
// lands in the world texture table, and scaleHint normalizes size.
func TestCompileModelObject(t *testing.T) {
	dir := t.TempDir()
	glb := buildMiniGLB(t, dir)

	scJSON := `{
		"version": 1,
		"camera": {"pos": [0,3,-4], "lookAt": [0,0,0], "fov": 50},
		"objects": [
			{"type": "model", "src": "` + filepath.Base(glb) + `", "scaleHint": 4}
		],
		"lights": [{"type": "point", "pos": [2,4,-2], "color": [1,1,1], "intensity": 1}]
	}`
	sp := filepath.Join(dir, "scene.json")
	if err := os.WriteFile(sp, []byte(scJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	s, err := scene.Load(sp)
	if err != nil {
		t.Fatal(err)
	}
	w, err := Compile(s, NewFileResolver(dir))
	if err != nil {
		t.Fatal(err)
	}

	if len(w.Objects) != 1 {
		t.Fatalf("objects = %d, want 1", len(w.Objects))
	}
	obj := w.Objects[0]
	if obj.Kind != 'm' {
		t.Fatalf("kind = %c", obj.Kind)
	}
	if len(obj.Mesh.Tris) != 2 {
		t.Errorf("tris = %d", len(obj.Mesh.Tris))
	}
	// scaleHint 4: quad spans [-2..2] on X.
	minX, maxX := math.Inf(1), math.Inf(-1)
	for _, v := range obj.Mesh.Verts {
		minX = math.Min(minX, v.X)
		maxX = math.Max(maxX, v.X)
	}
	if math.Abs(minX+2) > 1e-6 || math.Abs(maxX-2) > 1e-6 {
		t.Errorf("scaleHint extent = [%v, %v], want [-2, 2]", minX, maxX)
	}

	// Material table: [default, TexMat]; object uses TexMat.
	if len(w.Materials) != 2 {
		t.Fatalf("materials = %d, want 2 (default + imported)", len(w.Materials))
	}
	if obj.MatIdx != 1 {
		t.Fatalf("obj material = %d, want 1", obj.MatIdx)
	}
	m := w.Materials[1]
	if m.Kind != 'm' {
		t.Errorf("imported material kind = %c, want m (dielectric)", m.Kind)
	}
	// Texture registered in the world table and referenced.
	if len(w.Textures) != 1 {
		t.Fatalf("textures = %d, want 1", len(w.Textures))
	}
	if m.TexIdx != 0 || m.Tex != w.Textures[0] {
		t.Errorf("material tex idx = %d, want 0", m.TexIdx)
	}
	if w.Textures[0].W != 2 || w.Textures[0].H != 2 {
		t.Errorf("texture dims = %dx%d", w.Textures[0].W, w.Textures[0].H)
	}
}

// TestCompileModelOverride verifies materialOverride replaces every part.
func TestCompileModelOverride(t *testing.T) {
	dir := t.TempDir()
	glb := buildMiniGLB(t, dir)

	scJSON := `{
		"version": 1,
		"camera": {"pos": [0,3,-4], "lookAt": [0,0,0], "fov": 50},
		"materials": {"paint": {"type": "metal", "color": [0.9, 0.7, 0.2]}},
		"objects": [
			{"type": "model", "src": "` + filepath.Base(glb) + `",
			 "materialOverride": "paint"}
		],
		"lights": [{"type": "point", "pos": [2,4,-2], "color": [1,1,1], "intensity": 1}]
	}`
	sp := filepath.Join(dir, "scene.json")
	if err := os.WriteFile(sp, []byte(scJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := scene.Load(sp)
	if err != nil {
		t.Fatal(err)
	}
	w, err := Compile(s, NewFileResolver(dir))
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Objects) != 1 || w.Objects[0].MatIdx != 1 {
		t.Fatalf("override failed: objs=%d matIdx=%d", len(w.Objects), w.Objects[0].MatIdx)
	}
	if m := w.Materials[w.Objects[0].MatIdx]; m.Kind != 'M' {
		t.Errorf("override material kind = %c, want M (metal)", m.Kind)
	}
}
