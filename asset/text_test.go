package asset

import (
	"math"
	"os"
	"testing"

	"github.com/wltechblog/beam/geom"
)

const testFont = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

func loadFont(t *testing.T) string {
	t.Helper()
	if _, err := os.Stat(testFont); err != nil {
		t.Skipf("test font not available: %v", testFont)
	}
	return testFont
}

func meshBounds(m *geom.TriangleMesh) geom.Bounds {
	b := geom.Bounds{Min: geom.Vec3{X: math.Inf(1), Y: math.Inf(1), Z: math.Inf(1)}, Max: geom.Vec3{X: math.Inf(-1), Y: math.Inf(-1), Z: math.Inf(-1)}}
	for _, v := range m.Verts {
		b.Grow(v)
	}
	return b
}

func TestTextMeshCenteredAndSized(t *testing.T) {
	font := loadFont(t)
	const size = 2.0
	const depth = 0.4
	m, err := TextMesh("BEAM", font, size, depth, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Tris) == 0 {
		t.Fatal("no triangles")
	}
	b := meshBounds(m)
	// Centered on origin horizontally.
	if c := (b.Min.X + b.Max.X) / 2; c > size*0.05 || c < -size*0.05 {
		t.Errorf("text not centered: x center %.3f (bbox %.3f..%.3f)", c, b.Min.X, b.Max.X)
	}
	// Em height: cap tops near ~0.72 em, baseline 0; allow descender room.
	if b.Max.Y > size*1.05 || b.Max.Y < size*0.6 {
		t.Errorf("cap height out of em box: maxY %.3f (size %.1f)", b.Max.Y, size)
	}
	if b.Min.Y < -size*0.3 || b.Min.Y > 0 {
		t.Errorf("baseline/descender wrong: minY %.3f", b.Min.Y)
	}
	// Extrusion depth baked to [-depth, 0] by the 180° Y rotation.
	if d := b.Max.Z - b.Min.Z; d < depth-0.01 || d > depth+0.01 {
		t.Errorf("extrusion depth: %.4f want %.2f", d, depth)
	}
	if b.Max.Z > 0.001 || b.Min.Z > 0.001 {
		t.Errorf("depth should span [-d,0]: z %.3f..%.3f", b.Min.Z, b.Max.Z)
	}
}

func TestTextMeshFrontFacesCamera(t *testing.T) {
	// After the 180° bake, front faces point -Z (toward cameras on -Z).
	// A ray from -Z must hit a front face with outward normal N.Z < 0.
	font := loadFont(t)
	m, err := TextMesh("I", font, 1, 0.2, 0)
	if err != nil {
		t.Fatal(err)
	}
	bvh := geom.BuildMesh(m)
	// "I" stem spans the full cap height at x=0; aim mid-height.
	r := geom.Ray{Orig: geom.Vec3{X: 0, Y: 0.35, Z: -5}, Dir: geom.Vec3{Z: 1}}
	h := bvh.Intersect(r)
	if h == nil {
		t.Fatal("ray through I stem should hit")
	}
	if !h.Front {
		t.Errorf("front hit expected from -Z, got back face")
	}
	if h.N.Z >= 0 {
		t.Errorf("front normal should point -Z: %v", h.N)
	}
}

func TestTextMeshHoles(t *testing.T) {
	// The counter (hole) of "O" must be a real hole: a ray through the
	// counter passes clean through the extrusion — if hole triangulation
	// regressed to solid, it would hit.
	font := loadFont(t)
	m, err := TextMesh("O", font, 1, 0.3, 0)
	if err != nil {
		t.Fatal(err)
	}
	bvh := geom.BuildMesh(m)
	r := geom.Ray{Orig: geom.Vec3{X: 0, Y: 0.35, Z: -5}, Dir: geom.Vec3{Z: 1}}
	if h := bvh.Intersect(r); h != nil {
		t.Errorf("ray through O counter hit something (hole broken): T=%v", h.T)
	}
}

func TestTextMeshSpacingWidens(t *testing.T) {
	font := loadFont(t)
	tight, err := TextMesh("II", font, 1, 0.2, 0)
	if err != nil {
		t.Fatal(err)
	}
	spaced, err := TextMesh("II", font, 1, 0.2, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	wTight := meshBounds(tight).Max.X - meshBounds(tight).Min.X
	wSpaced := meshBounds(spaced).Max.X - meshBounds(spaced).Min.X
	// 0.5 em extra spacing between the two glyphs → ~0.5 world units wider.
	if wSpaced-wTight < 0.4 {
		t.Errorf("spacing not applied: tight %.3f spaced %.3f", wTight, wSpaced)
	}
}

func TestTextMeshMoreTrisThanBox(t *testing.T) {
	// Curved glyphs triangulate many more triangles than a boxy "I".
	font := loadFont(t)
	iMesh, err := TextMesh("I", font, 1, 0.2, 0)
	if err != nil {
		t.Fatal(err)
	}
	oMesh, err := TextMesh("O", font, 1, 0.2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(oMesh.Tris) <= len(iMesh.Tris) {
		t.Errorf("O (%d tris) should have more than I (%d)", len(oMesh.Tris), len(iMesh.Tris))
	}
}

func TestTextMeshErrors(t *testing.T) {
	font := loadFont(t)
	if _, err := TextMesh("", font, 1, 0.2, 0); err == nil {
		t.Error("empty text should error")
	}
	if _, err := TextMesh("hi", "/nonexistent/font.ttf", 1, 0.2, 0); err == nil {
		t.Error("missing font should error")
	}
}
