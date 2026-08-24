package render

import (
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// fakeResolver returns a quad facing -Z for text/mesh objects in tests —
// no font or file dependencies.
type fakeResolver struct{}

func (fakeResolver) LoadMesh(src string) (*geom.TriangleMesh, error) {
	return CubeMesh(geom.Vec3{}, geom.Vec3{X: 0.5, Y: 0.5, Z: 0.5}), nil
}

func (fakeResolver) TextMesh(text, font string, size, depth, spacing float64) (*geom.TriangleMesh, error) {
	return CubeMesh(geom.Vec3{}, geom.Vec3{X: 0.5, Y: 0.2, Z: 0.1}), nil
}

// testScene builds a small scene with one sphere.
func testScene() *scene.Scene {
	s := scene.Defaults()
	s.Render = scene.RenderOpts{Width: 64, Height: 48, Samples: 2, MaxDepth: 3, Seed: 7, Exposure: 1}
	s.Materials = map[string]*scene.Material{
		"red": {Type: "matte", Color: []float64{0.8, 0.1, 0.1}},
	}
	s.Objects = []scene.Object{
		{Type: "sphere", Center: []float64{0, 0.5, 0}, Radius: 0.5, Material: "red"},
	}
	s.Lights = []scene.Light{
		{Type: "point", Pos: []float64{2, 3, -2}, Intensity: 10},
	}
	s.Environment = &scene.Environment{Color: []float64{0.05, 0.06, 0.08}, Ambient: 0.2}
	return s
}

func renderToImage(t *testing.T, s *scene.Scene) *image.RGBA {
	t.Helper()
	s.ApplyDefaults()
	world, err := Compile(s, fakeResolver{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	cam := NewCamera(
		vec3Of(s.Camera.Pos), vec3Of(s.Camera.LookAt), s.Camera.FOV,
		float64(s.Render.Width)/float64(s.Render.Height),
		s.Camera.Aperture, s.Camera.FocusDist)
	r := &Renderer{
		World: world, Camera: cam,
		Width: s.Render.Width, Height: s.Render.Height,
		Samples: s.Render.Samples, MaxDepth: s.Render.MaxDepth,
		Exposure: s.Render.Exposure, Seed: s.Render.Seed,
	}
	return r.Render(nil)
}

func TestRenderDeterministic(t *testing.T) {
	// The founding guarantee: same scene + seed → byte-identical pixels.
	a := renderToImage(t, testScene())
	b := renderToImage(t, testScene())
	if a.Rect != b.Rect {
		t.Fatalf("size mismatch: %v vs %v", a.Rect, b.Rect)
	}
	for y := a.Rect.Min.Y; y < a.Rect.Max.Y; y++ {
		for x := a.Rect.Min.X; x < a.Rect.Max.X; x++ {
			if a.RGBAAt(x, y) != b.RGBAAt(x, y) {
				t.Fatalf("pixel (%d,%d) differs: %v vs %v", x, y, a.RGBAAt(x, y), b.RGBAAt(x, y))
			}
		}
	}
}

func TestRenderSeedChangesImage(t *testing.T) {
	s1 := testScene()
	s2 := testScene()
	s2.Render.Seed = 99
	a := renderToImage(t, s1)
	b := renderToImage(t, s2)
	diff := 0
	for y := 0; y < 48; y++ {
		for x := 0; x < 64; x++ {
			if a.RGBAAt(x, y) != b.RGBAAt(x, y) {
				diff++
			}
		}
	}
	if diff == 0 {
		t.Fatal("different seeds produced identical images (seed not wired)")
	}
}

func TestRenderSphereSilhouette(t *testing.T) {
	// Physics sanity: a sphere of radius 0.5 at distance 5 with 50° fov
	// must subtend the correct angular size in pixels.
	s := testScene()
	s.Camera.Pos = []float64{0, 0.5, -5}
	s.Camera.LookAt = []float64{0, 0.5, 0}
	img := renderToImage(t, s)

	// half-angle of sphere = asin(r/dist); expected pixel radius =
	// tan(asin(0.1)) / tan(25°) * (W/2)
	expR := 0.10050378 / 0.46630766 * 32
	count := 0
	minX, maxX, minY, maxY := 64, 0, 48, 0
	for y := 0; y < 48; y++ {
		for x := 0; x < 64; x++ {
			c := img.RGBAAt(x, y)
			// Sphere is red; background is dark blue-grey.
			if c.R > 100 && c.R > 2*c.B {
				count++
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if count == 0 {
		t.Fatal("no sphere pixels found — sphere not rendered")
	}
	gotR := float64(maxX-minX+1) / 2
	if math.Abs(gotR-expR) > 2.5 {
		t.Errorf("silhouette radius: got %.1f want ~%.1f (bbox x[%d,%d] y[%d,%d], %d px)",
			gotR, expR, minX, maxX, minY, maxY, count)
	}
	// Centered on the image center (32, 24) — camera looks at sphere center.
	cx := float64(minX+maxX) / 2
	cy := float64(minY+maxY) / 2
	if math.Abs(cx-31.5) > 2 || math.Abs(cy-23.5) > 2 {
		t.Errorf("sphere center: (%.1f,%.1f), want ~(31.5,23.5)", cx, cy)
	}
}

func TestRenderShadows(t *testing.T) {
	// Shadow machinery, tested directly (test lives in package render):
	// a sphere hovering over a floor, light straight above. The floor
	// point under the sphere must be occluded; a distant floor point lit.
	s := testScene()
	s.Render.Samples = 1
	s.Materials["floor"] = &scene.Material{Type: "matte", Color: []float64{0.9, 0.9, 0.9}}
	s.Objects = []scene.Object{
		{Type: "sphere", Center: []float64{0, 1.5, 0}, Radius: 0.5, Material: "red"},
		{Type: "plane", Point: []float64{0, 0, 0}, Normal: []float64{0, 1, 0}, Material: "floor"},
	}
	s.Lights = []scene.Light{{Type: "point", Pos: []float64{0, 6, 0}, Intensity: 10}}
	s.Environment = &scene.Environment{Color: []float64{0.05, 0.05, 0.05}, Ambient: 0.2}

	s.ApplyDefaults()
	world, err := Compile(s, fakeResolver{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	r := &Renderer{World: world, MaxDepth: 4, Exposure: 1}

	up := geom.Vec3{Y: 1}
	// Occlusion: shadow ray from under the sphere is blocked.
	if trans := r.shadowTrans(geom.Vec3{Y: 1e-4, Z: 0.05}, up, 6); trans.X > 0.01 || trans.Y > 0.01 {
		t.Errorf("shadow under sphere should be black, got %v", trans)
	}
	// Distant floor point sees the light.
	if trans := r.shadowTrans(geom.Vec3{X: 3, Y: 1e-4}, up, 6); trans.X < 0.99 {
		t.Errorf("distant floor should be unoccluded, got %v", trans)
	}

	// End-to-end through trace(): floor hit inside the shadow is much
	// darker than the same floor lit, viewed from the same origin.
	lum := func(v geom.Vec3) float64 { return v.X + v.Y + v.Z }
	camO := geom.Vec3{Y: 1.2, Z: -5}
	inShadow := r.trace(geom.Ray{Orig: camO, Dir: geom.Vec3{Y: -1.2, Z: 5.1}.Norm()}, 4, NewRNG(1))
	lit := r.trace(geom.Ray{Orig: camO, Dir: geom.Vec3{X: 3, Y: -1.2, Z: 8}.Norm()}, 4, NewRNG(2))
	if lum(inShadow) >= 0.75*lum(lit) {
		t.Errorf("shadowed floor (%v) not darker than lit floor (%v)", inShadow, lit)
	}
}

func TestRenderGlassTransmits(t *testing.T) {
	// Behind a glass sphere, the environment must show through (tinted).
	s := testScene()
	s.Environment = &scene.Environment{
		Color:   []float64{0.1, 0.4, 0.9},
		Ambient: 0.2,
		Sky:     [2][]float64{{0.1, 0.4, 0.9}, {0.1, 0.4, 0.9}},
	}
	s.Materials["glass"] = &scene.Material{Type: "glass", Color: []float64{1, 1, 1}, IOR: 1.5}
	s.Objects = []scene.Object{
		{Type: "sphere", Center: []float64{0, 0.5, 0}, Radius: 0.5, Material: "glass"},
	}
	s.Lights = nil
	img := renderToImage(t, s)
	c := img.RGBAAt(32, 24) // through the sphere center
	// Blue must dominate: sky is (0.1,0.4,0.9) × exposure.
	if c.B < 60 || c.B <= c.R {
		t.Errorf("glass center should transmit blue sky: %+v", c)
	}
}

func TestRenderEmissive(t *testing.T) {
	// Emissive object returns albedo × power regardless of lights.
	s := testScene()
	s.Lights = nil
	s.Environment = &scene.Environment{Color: []float64{0, 0, 0}, Ambient: 0}
	s.Materials["em"] = &scene.Material{Type: "emissive", Color: []float64{1, 0, 0}, Power: 2}
	s.Objects = []scene.Object{
		{Type: "sphere", Center: []float64{0, 0.5, 0}, Radius: 0.5, Material: "em"},
	}
	img := renderToImage(t, s)
	c := img.RGBAAt(32, 24)
	// Reinhard: power 2 red -> 2/(1+2) = 2/3 -> 255*(2/3)+0.5 = 170.5 -> 170.
	if math.Abs(float64(c.R)-170) > 2 || c.G > 10 || c.B > 10 {
		t.Errorf("emissive red sphere: %+v (want R\u2248170 = 255\u00d72/3)", c)
	}
}

func TestRenderPreviewAndOutput(t *testing.T) {
	// End-to-end through RenderFile: file in → PNG out, defaults applied.
	dir := t.TempDir()
	body := `{
		"version": 1,
		"camera": {"pos": [0,0.5,-4], "lookAt": [0,0.5,0], "fov": 50},
		"materials": {"m": {"type": "matte", "color": [0.2,0.6,0.2]}},
		"objects": [{"type": "sphere", "center": [0,0.5,0], "radius": 0.5, "material": "m"}],
		"lights": [{"type": "directional", "dir": [0,-1,-1], "intensity": 1}],
		"render": {"width": 40, "height": 30, "samples": 2}
	}`
	p := filepath.Join(dir, "t.json")
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "t.png")
	got, err := RenderFile(p, Options{Output: out, Quiet: true, Dir: dir})
	if err != nil {
		t.Fatalf("RenderFile: %v", err)
	}
	if got != out {
		t.Errorf("output path: %v", got)
	}
	f, err := os.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cfg, err := png.DecodeConfig(f)
	if err != nil {
		t.Fatalf("decode png: %v", err)
	}
	if cfg.Width != 40 || cfg.Height != 30 {
		t.Errorf("png size: %dx%d", cfg.Width, cfg.Height)
	}
}
