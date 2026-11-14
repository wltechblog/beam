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

// buildColoredGLB writes a one-material GLB whose base-color texture is a
// solid color quad. color 0 = pure red, 1 = pure blue — distinct enough to
// catch cross-model texture collisions.
func buildColoredGLB(t *testing.T, dir, name string, which int, metallic, roughness *float64) string {
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

	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			if which == 0 {
				img.Set(x, y, color.NRGBA{R: 255, A: 255})
			} else {
				img.Set(x, y, color.NRGBA{B: 255, A: 255})
			}
		}
	}
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
			BufferView: &imgView, MimeType: "image/png", Name: "solid",
		}},
		Textures: []*gltf.Texture{{Source: ptrGI(0)}},
		Materials: []*gltf.Material{{
			Name: "SolidMat",
			PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
				BaseColorTexture: &gltf.TextureInfo{Index: 0},
				MetallicFactor:   metallic,
				RoughnessFactor:  roughness,
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

	path := filepath.Join(dir, name)
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

// TestModelTextureNoCrossModelCollision pins the corridor bug: two models
// in one scene, each with its own base texture (image index 0 in BOTH
// files). The old dedupe key ("model:img0") made the second model reuse
// the first model's texture — the lantern rendered with the duck's
// yellow map. Each model must get its OWN texture-table entry.
func TestModelTextureNoCrossModelCollision(t *testing.T) {
	dir := t.TempDir()
	redGLB := buildColoredGLB(t, dir, "red.glb", 0, ptrF(0.0), nil)
	blueGLB := buildColoredGLB(t, dir, "blue.glb", 1, ptrF(0.0), nil)

	scJSON := `{
		"version": 1,
		"camera": {"pos": [0, 3, -6], "lookAt": [0, 0, 0], "fov": 40},
		"environment": {"color": [0.1, 0.1, 0.15], "ambient": 0.2},
		"objects": [
			{"type": "model", "src": REDFILE, "name": "m1", "scaleHint": 1,
			 "transform": {"translate": [-2, 0, 0]}},
			{"type": "model", "src": BLUEFILE, "name": "m2", "scaleHint": 1,
			 "transform": {"translate": [2, 0, 0]}}
		],
		"lights": [{"type": "point", "pos": [0, 4, -3], "intensity": 1.0}]
	}`
	scJSON = replaceJSONPlaceholders2(scJSON, map[string]string{
		"REDFILE":  redGLB,
		"BLUEFILE": blueGLB,
	})

	s, err := scene.Parse([]byte(scJSON))
	if err != nil {
		t.Fatal(err)
	}
	w, err := Compile(s, NewFileResolver(dir))
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Textures) != 2 {
		t.Fatalf("textures = %d, want 2 (one per model — collision means the second model reuses the first's)", len(w.Textures))
	}
	// The two textures must be DIFFERENT: tex0 predominantly red, tex1 blue.
	// Pix is linear 0..1.
	for ti, want := range []struct{ r, b float64 }{{1, 0}, {0, 1}} {
		tex := w.Textures[ti]
		if tex == nil {
			t.Fatalf("texture %d nil", ti)
		}
		rSum, bSum := 0.0, 0.0
		for i := 0; i < tex.W*tex.H; i++ {
			rSum += float64(tex.Pix[i*3])
			bSum += float64(tex.Pix[i*3+2])
		}
		n := float64(tex.W * tex.H)
		if rSum/n < want.r*0.9 || bSum/n < want.b*0.9 {
			t.Errorf("texture %d mean (r=%.2f b=%.2f), want (r≈%.2f b≈%.2f) — textures collided or swapped",
				ti, rSum/n, bSum/n, want.r, want.b)
		}
	}
	// Both models' materials must reference DIFFERENT texture indices.
	var seen []int
	for _, m := range w.Materials {
		if m.Tex != nil && m.TexIdx >= 0 {
			seen = append(seen, m.TexIdx)
		}
	}
	if len(seen) != 2 || seen[0] == seen[1] {
		t.Errorf("model material TexIdx values = %v, want two distinct indices", seen)
	}
}

// TestGLTFFullyRoughMetalIsMatte pins the lantern mapping: a glTF material
// with metallic=1 and roughness omitted (=1.0, the spec default — the
// Khronos Lantern ships exactly this) must NOT become a beam metal. A
// roughness-1 'M' hemisphere-scatters reflections (fog/glow from every
// direction) and renders as washed-out white; it maps to dark textured
// matte, which is what a fully-rough conductor visually is.
func TestGLTFFullyRoughMetalIsMatte(t *testing.T) {
	dir := t.TempDir()
	glb := buildColoredGLB(t, dir, "rough.glb", 0, ptrF(1.0), nil)

	sJSON := `{ 
		"version": 1,
		"camera": {"pos": [0, 3, -6], "lookAt": [0, 0, 0], "fov": 40},
		"environment": {"color": [0.1, 0.1, 0.15], "ambient": 0.2},
		"objects": [{"type": "model", "src": GLBFILE, "name": "m", "scaleHint": 1}],
		"lights": [{"type": "point", "pos": [0, 4, -3], "intensity": 1.0}]
	}`
	sJSON = replaceJSONPlaceholders2(sJSON, map[string]string{"GLBFILE": glb})

	s, err := scene.Parse([]byte(sJSON))
	if err != nil {
		t.Fatal(err)
	}
	w, err := Compile(s, NewFileResolver(dir))
	if err != nil {
		t.Fatal(err)
	}
	// The imported material is the only textured entry in the table
	// (the default material carries TexIdx 0 with a nil Tex — filter on
	// Tex presence, not the index).
	for _, m := range w.Materials {
		if m.Tex == nil {
			continue
		}
		if m.Kind == 'M' {
			t.Errorf("metallic=1 roughness=1 (spec defaults) compiled to beam metal 'M' — renders washed-out white; want matte 'm'")
		}
		if m.Kind != 'm' && m.Kind != 'M' {
			t.Fatalf("unexpected kind %c", m.Kind)
		}
		if m.Albedo.X > 0.7 {
			t.Errorf("fully-rough conductor albedo %.2f too bright for a dark matte — was the darkening scale applied?", m.Albedo.X)
		}
		return
	}
	t.Fatal("no textured material found — model import produced nothing")
}

// replaceJSONPlaceholders2 substitutes src paths into scene JSON — the
// shared helper from model_anim_test.go has hardcoded keys; this variant
// takes a map.
func replaceJSONPlaceholders2(js string, replace map[string]string) string {
	out := make([]byte, 0, len(js))
	i := 0
	for i < len(js) {
		matched := false
		for k, v := range replace {
			if len(js)-i >= len(k) && js[i:i+len(k)] == k {
				out = append(out, '"')
				out = append(out, v...)
				out = append(out, '"')
				i += len(k)
				matched = true
				break
			}
		}
		if !matched {
			out = append(out, js[i])
			i++
		}
	}
	return string(out)
}
