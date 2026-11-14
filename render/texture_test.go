package render

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/scene"
)

// quadTexture writes a four-quadrant PNG (image-space: TL red, TR green,
// BL blue, BR yellow) into dir and returns its file name.
func quadTexture(t *testing.T, dir string) string {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			var c color.NRGBA
			switch {
			case x < 32 && y < 32:
				c = color.NRGBA{R: 255, A: 255}
			case x >= 32 && y < 32:
				c = color.NRGBA{G: 255, A: 255}
			case x < 32 && y >= 32:
				c = color.NRGBA{B: 255, A: 255}
			default:
				c = color.NRGBA{R: 255, G: 255, A: 255}
			}
			img.SetNRGBA(x, y, c)
		}
	}
	f, err := os.Create(filepath.Join(dir, "quad.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	return "quad.png"
}

// TestTexturedQuadRenders pins the full texture path end to end:
// material.map loads through FileResolver, the mesh carries UVs, and the
// rendered quad shows the four quadrants in the orientation the OBJ
// convention dictates (u tracks world +X, v tracks up; beam cameras at
// -Z see world +X screen-LEFT, so image-right appears screen-left).
func TestTexturedQuadRenders(t *testing.T) {
	dir := t.TempDir()
	quadTexture(t, dir)
	obj := "v -1.5 -1 0\nv 1.5 -1 0\nv 1.5 2 0\nv -1.5 2 0\n" +
		"vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n" +
		"f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
	if err := os.WriteFile(filepath.Join(dir, "quad.obj"), []byte(obj), 0o644); err != nil {
		t.Fatal(err)
	}
	raw := fmt.Sprintf(`{
		"version": 1,
		"render": {"width": 96, "height": 72, "samples": 1, "maxDepth": 3, "seed": 7},
		"camera": {"pos": [0, 0.5, -5], "lookAt": [0, 0.5, 0], "fov": 40},
		"materials": {
			"board": {"type": "matte", "color": [1, 1, 1], "map": "quad.png"},
			"floor": {"type": "matte", "color": [0.3, 0.3, 0.32]}
		},
		"objects": [
			{"type": "plane", "point": [0, -1.5, 0], "normal": [0, 1, 0], "material": "floor"},
			{"type": "mesh", "src": "quad.obj", "material": "board"}
		],
		"lights": [{"type": "point", "pos": [-3, 4, -5], "intensity": 1.6, "falloff": "none"}],
		"environment": {"color": [0.05, 0.05, 0.08], "ambient": 0.15}
	}`)
	s, err := scene.Parse([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	s.ApplyDefaults()
	w, err := Compile(s, NewFileResolver(dir))
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Textures) != 1 {
		t.Fatalf("texture table = %d, want 1", len(w.Textures))
	}
	var board *CompiledMaterial
	for i := range w.Materials {
		if w.Materials[i].Tex != nil {
			board = &w.Materials[i]
		}
	}
	if board == nil {
		t.Fatal("no material carries a compiled texture")
	}
	out := filepath.Join(t.TempDir(), "out.png")
	path, err := RenderScene(s, Options{Dir: dir, Quiet: true, Output: out})
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	probe := func(fx, fy float64) (r, g, bl uint32) {
		x := b.Min.X + int(float64(b.Dx())*fx)
		y := b.Min.Y + int(float64(b.Dy())*fy)
		r0, g0, b0, _ := img.At(x, y).RGBA()
		return r0 >> 8, g0 >> 8, b0 >> 8
	}
	probes := []struct {
		fx, fy float64
		want   string
	}{
		{0.30, 0.36, "green"},  // screen top-left = image top-right
		{0.70, 0.36, "red"},    // screen top-right = image top-left
		{0.30, 0.74, "yellow"}, // screen bottom-left = image bottom-right
		{0.70, 0.74, "blue"},   // screen bottom-right = image bottom-left
	}
	for _, pr := range probes {
		r, g, bl := probe(pr.fx, pr.fy)
		var got string
		switch {
		case r > 120 && g > 120:
			got = "yellow"
		case r > 120:
			got = "red"
		case g > 120:
			got = "green"
		case bl > 120:
			got = "blue"
		}
		if got != pr.want {
			t.Errorf("quadrant at (%.2f,%.2f): got %s (RGB %d,%d,%d), want %s",
				pr.fx, pr.fy, got, r, g, bl, pr.want)
		}
	}
}
