package asset

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
	"github.com/wltechblog/beam/geom"
)

// glbBuilder assembles a minimal glTF document with one shared buffer.
type glbBuilder struct {
	buf     bytes.Buffer
	views   []*gltf.BufferView
	accs    []*gltf.Accessor
	doc     *gltf.Document
	updates []func()
}

func newGB() *glbBuilder {
	return &glbBuilder{doc: &gltf.Document{
		Asset: gltf.Asset{Version: "2.0"},
	}}
}

func (b *glbBuilder) addView(data []byte) *int {
	off := b.buf.Len()
	b.buf.Write(data)
	i := len(b.views)
	b.views = append(b.views, &gltf.BufferView{Buffer: 0, ByteOffset: off, ByteLength: len(data)})
	return &i
}

func (b *glbBuilder) addAccessor(view *int, ct gltf.ComponentType, at gltf.AccessorType, count int) *int {
	i := len(b.accs)
	b.accs = append(b.accs, &gltf.Accessor{
		BufferView: view, ComponentType: ct, Type: at, Count: count,
	})
	return &i
}

func f32e(v ...float32) []byte {
	out := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(x))
	}
	return out
}

func u16e(v ...uint16) []byte {
	out := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(out[i*2:], x)
	}
	return out
}

// writeGLB encodes the document as a .glb file.
func (b *glbBuilder) writeGLB(t *testing.T, path string) {
	buf := &gltf.Buffer{ByteLength: b.buf.Len(), Data: b.buf.Bytes()}
	b.doc.Buffers = []*gltf.Buffer{buf}
	b.doc.BufferViews = b.views
	b.doc.Accessors = b.accs
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := gltf.NewEncoder(f).Encode(b.doc); err != nil {
		t.Fatal(err)
	}
}

// buildFixtureGLB writes a two-material GLB:
//   - quad A at origin, matte red material (baseColorFactor)
//   - quad B under a parent node translated [2,0,0] + scaled 2x on X
//     (nested hierarchy: root -> group -> leaf), textured + emissive
func buildFixtureGLB(t *testing.T, dir string) string {
	b := newGB()

	// Tiny 2x2 PNG (red quadrant / green quadrant).
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.NRGBA{R: 255, A: 255})
	img.Set(1, 0, color.NRGBA{G: 255, A: 255})
	img.Set(0, 1, color.NRGBA{G: 255, A: 255})
	img.Set(1, 1, color.NRGBA{R: 255, A: 255})
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, img); err != nil {
		t.Fatal(err)
	}
	imgView := b.addView(pngBuf.Bytes())
	b.doc.Images = []*gltf.Image{{BufferView: imgView, MimeType: "image/png", Name: "checker"}}
	b.doc.Textures = []*gltf.Texture{{Source: ptr(0), Name: "checker"}}

	// Quad A: two triangles spanning [-1..1]², flat.
	posA := f32e(
		-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1,
	)
	normA := f32e(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0)
	uvA := f32e(0, 0, 1, 0, 1, 1, 0, 1)
	idxA := u16e(0, 1, 2, 0, 2, 3)
	pv := b.addView(posA)
	nv := b.addView(normA)
	uvv := b.addView(uvA)
	iv := b.addView(idxA)
	posAcc := b.addAccessor(pv, gltf.ComponentFloat, gltf.AccessorVec3, 4)
	nrmAcc := b.addAccessor(nv, gltf.ComponentFloat, gltf.AccessorVec3, 4)
	uvAcc := b.addAccessor(uvv, gltf.ComponentFloat, gltf.AccessorVec2, 4)
	idxAcc := b.addAccessor(iv, gltf.ComponentUshort, gltf.AccessorScalar, 6)

	// Both primitives share the same geometry accessors; B differs only
	// by material + node transform (keeps the fixture tiny).
	prim := func(mat int) *gltf.Primitive {
		return &gltf.Primitive{
			Attributes: gltf.PrimitiveAttributes{
				"POSITION":   *posAcc,
				"NORMAL":     *nrmAcc,
				"TEXCOORD_0": *uvAcc,
			},
			Indices:  idxAcc,
			Material: &mat,
		}
	}

	mat0 := &gltf.Material{
		Name: "RedMatte",
		PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
			BaseColorFactor: &[4]float64{0.9, 0.1, 0.1, 1},
			MetallicFactor:  ptr(0.0),
		},
	}
	mat1 := &gltf.Material{
		Name: "GreenTextured",
		PBRMetallicRoughness: &gltf.PBRMetallicRoughness{
			BaseColorTexture: &gltf.TextureInfo{Index: 0},
			MetallicFactor:   ptr(0.1),
		},
		EmissiveFactor: [3]float64{0.2, 0.8, 0.2},
	}
	b.doc.Materials = []*gltf.Material{mat0, mat1}
	b.doc.Meshes = []*gltf.Mesh{
		{Name: "quad", Primitives: []*gltf.Primitive{prim(0)}},
		{Name: "quadT", Primitives: []*gltf.Primitive{prim(1)}},
	}
	// Root -> group (translate [2,0,0], scale [2,1,1]) -> leaf (mesh 1).
	m0, m1 := 0, 1
	b.doc.Nodes = []*gltf.Node{
		{Name: "root", Children: []int{1, 3}},
		{Name: "group", Translation: [3]float64{2, 0, 0}, Scale: [3]float64{2, 1, 1}, Children: []int{2}},
		{Name: "leaf", Mesh: &m1},
		{Name: "flat", Mesh: &m0},
	}
	b.doc.Scenes = []*gltf.Scene{{Name: "S", Nodes: []int{0}}}
	sc := 0
	b.doc.Scene = &sc

	path := filepath.Join(dir, "fixture.glb")
	b.writeGLB(t, path)
	return path
}

func ptr[T any](v T) *T { return &v }

// geomCompose builds T*Rz*Ry*Rx*S via the real geom package.
func geomCompose(tx, ty, tz, rx, ry, rz, sx, sy, sz float64) geom.Mat4 {
	t := geom.Translate(tx, ty, tz).
		Mul(geom.RotateZ(rz)).Mul(geom.RotateY(ry)).Mul(geom.RotateX(rx)).
		Mul(geom.Scale(sx, sy, sz))
	return t
}

func TestLoadModelGeometry(t *testing.T) {
	dir := t.TempDir()
	path := buildFixtureGLB(t, dir)

	lm, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(lm.Parts) != 2 {
		t.Fatalf("want 2 parts, got %d", len(lm.Parts))
	}

	// Walk order: root children [1,3] — node 1 (group -> leaf, mat 1)
	// emits FIRST, then node 3 (flat, mat 0). So part 0 is the
	// transformed+textured quad, part 1 the flat red one.
	p0 := lm.Parts[0]
	if len(p0.Mesh.Tris) != 2 {
		t.Fatalf("part0 tris = %d", len(p0.Mesh.Tris))
	}
	if p0.MatIdx != 1 {
		t.Errorf("part0 material = %d", p0.MatIdx)
	}
	minX, maxX := math.Inf(1), math.Inf(-1)
	for _, v := range p0.Mesh.Verts {
		minX = math.Min(minX, v.X)
		maxX = math.Max(maxX, v.X)
	}
	if math.Abs(minX) > 1e-6 || math.Abs(maxX-4) > 1e-6 {
		t.Errorf("part0 x extent = [%v, %v], want [0, 4] (nested TRS)", minX, maxX)
	}
	p1 := lm.Parts[1]
	// bbox check: [-1..1] on X, y=0 (identity node)
	minX, maxX = math.Inf(1), math.Inf(-1)
	for _, v := range p1.Mesh.Verts {
		minX = math.Min(minX, v.X)
		maxX = math.Max(maxX, v.X)
	}
	if minX != -1 || maxX != 1 {
		t.Errorf("part1 x extent = [%v, %v], want [-1, 1]", minX, maxX)
	}
	if p1.MatIdx != 0 {
		t.Errorf("part1 material = %d", p1.MatIdx)
	}
	// UVs present
	if len(p1.Mesh.UVs) != 4 {
		t.Errorf("part1 uvs = %d", len(p1.Mesh.UVs))
	}
	// Normals present and up
	if len(p1.Mesh.Norms) != 4 || p1.Mesh.Norms[0].Y < 0.99 {
		t.Errorf("part1 normals wrong: %+v", p1.Mesh.Norms[0])
	}

	// Part 0 carries the nested transform group(translate [2,0,0],
	// scale [2,1,1]) applied to the same quad: x spans [-2+2 .. 2+2]
	// = [0, 4].
	// UVs on the textured part too.
	if len(p0.Mesh.UVs) != 4 {
		t.Errorf("part0 uvs = %d", len(p0.Mesh.UVs))
	}
	// Non-uniform scale on normals: inverse-transpose renormalizes; Y
	// normal unaffected (scale x only), stays unit.
	if n := p0.Mesh.Norms[0]; math.Abs(n.Y-1) > 1e-6 {
		t.Errorf("part0 normal = %+v, want Y=1", n)
	}

	// Images decoded: 1 image, 2x2
	if len(lm.Images) != 1 || lm.Images[0] == nil {
		t.Fatalf("images not decoded: %v", lm.Images)
	}
	if b := lm.Images[0].Bounds(); b.Dx() != 2 || b.Dy() != 2 {
		t.Errorf("image bounds = %v", b)
	}
}

func TestLoadModelNoMeshes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.glb")
	b := newGB()
	b.writeGLB(t, path)
	if _, err := LoadModel(path); err == nil {
		t.Fatal("expected error for mesh-less document")
	}
}

func TestMat4InverseRoundTrip(t *testing.T) {
	// glm-free sanity of the new geom helpers via a composite transform.
	m := geomCompose(2, 3, 4, 0.3, 0.7, 0.2, 2, 1, 0.5)
	inv := m.Inverse()
	id := m.Mul(inv)
	want := [16]float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	for i := range id {
		if math.Abs(id[i]-want[i]) > 1e-9 {
			t.Fatalf("inverse round trip failed at %d: %+v", i, id)
		}
	}
}
