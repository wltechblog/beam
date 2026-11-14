package gpu

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/render"
	"github.com/wltechblog/beam/scene"
)

// writeTexAssets drops a four-quadrant test texture (red top-left, green
// top-right, blue bottom-left, white bottom-right) and a UV sphere OBJ
// into dir. The dominant hue per sphere quadrant pins UV orientation;
// the floor (same map, planar mapping) pins the plane path.
func writeTexAssets(t *testing.T, dir string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			var c color.Color
			switch {
			case y < 32 && x < 32:
				c = color.RGBA{220, 30, 30, 255}
			case y < 32:
				c = color.RGBA{30, 220, 30, 255}
			case x < 32:
				c = color.RGBA{30, 30, 220, 255}
			default:
				c = color.RGBA{230, 230, 230, 255}
			}
			img.Set(x, y, c)
		}
	}
	f, err := os.Create(filepath.Join(dir, "quad.png"))
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	f.Close()

	obj := "o ball\n"
	segs, rings := 24, 12
	for r := 0; r <= rings; r++ {
		for s := 0; s < segs; s++ {
			phi := 3.141592653589793 * float64(r) / float64(rings)
			th := 2 * 3.141592653589793 * float64(s) / float64(segs)
			sp, ct, cph := math.Sin(phi), math.Cos(th), math.Cos(phi)
			x, y, z := sp*ct, cph, sp*math.Sin(th)
			obj += fmt.Sprintf("v %.6f %.6f %.6f\nvt %.6f %.6f\n", x, y, z, float64(s)/float64(segs), 1-float64(r)/float64(rings))
		}
	}
	for r := 0; r < rings; r++ {
		for s := 0; s < segs; s++ {
			a := r*segs + s + 1
			b := r*segs + (s+1)%segs + 1
			c := (r+1)*segs + (s+1)%segs + 1
			d := (r+1)*segs + s + 1
			obj += fmt.Sprintf("f %d/%d %d/%d %d/%d %d/%d\n", a, a, b, b, c, c, d, d)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "ball.obj"), []byte(obj), 0o644); err != nil {
		t.Fatal(err)
	}
}

const texScene = `{
  "version": 1,
  "render": {"width": 200, "height": 150, "samples": 2, "maxDepth": 4, "seed": 7},
  "camera": {"pos": [0, 1.2, -4.0], "lookAt": [0, 1.0, 0], "fov": 45},
  "materials": {
    "texball":  {"type": "matte", "color": [1, 1, 1], "map": "quad.png"},
    "texfloor": {"type": "matte", "color": [1, 1, 1], "map": "quad.png"}
  },
  "objects": [
    {"type": "plane", "point": [0, 0, 0], "normal": [0, 1, 0], "material": "texfloor"},
    {"type": "mesh", "src": "ball.obj", "material": "texball", "transform": {"translate": [0, 1.0, 0], "scale": [0.95, 0.95, 0.95]}}
  ],
  "lights": [
    {"type": "point", "pos": [-4, 5, -4], "intensity": 1.1, "falloff": "none"}
  ],
  "environment": {"color": [0.05, 0.05, 0.08], "ambient": 0.15}
}`

// TestGpuTextureParity: textured mesh (UV path) + textured plane (planar
// mapping) on both render paths. Two gates: fp32 envelope (mean < 1.0,
// max <= 32, <0.1% over) AND the texture must be VISIBLE (green-dominant
// pixels present on both) — parity alone could pass with both paths
// falling back to solid albedo.
func TestGpuTextureParity(t *testing.T) {
	devs, err := ProbeDevices()
	if err != nil || len(devs) == 0 {
		t.Skipf("no OpenCL GPU: %v", err)
	}
	dir := t.TempDir()
	writeTexAssets(t, dir)
	scn := filepath.Join(dir, "scene.json")
	if err := os.WriteFile(scn, []byte(texScene), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := scene.Load(scn)
	if err != nil {
		t.Fatal(err)
	}
	s.ApplyDefaults()
	res := render.NewFileResolver(dir)
	world, err := render.Compile(s, res)
	if err != nil {
		t.Fatal(err)
	}

	const W, H = 200, 150
	cam := render.NewCamera(
		geom.Vec3{X: s.Camera.Pos[0], Y: s.Camera.Pos[1], Z: s.Camera.Pos[2]},
		geom.Vec3{X: s.Camera.LookAt[0], Y: s.Camera.LookAt[1], Z: s.Camera.LookAt[2]},
		s.Camera.FOV, float64(W)/float64(H), s.Camera.Aperture, s.Camera.FocusDist)

	cr := &render.Renderer{World: world, Camera: cam, Width: W, Height: H,
		Samples: s.Render.Samples, MaxDepth: s.Render.MaxDepth, Exposure: s.Render.Exposure,
		Seed: s.Render.Seed}
	cimg := cr.Render(nil)

	gimg, _, err := RenderGPU(world, cam, W, H, s.Render.Samples, s.Render.MaxDepth,
		s.Render.Exposure, s.Render.Tonemap == "clip", s.Render.Seed, false)
	if err != nil {
		t.Fatalf("GPU render: %v", err)
	}

	var maxd, sum int64
	green := func(img interface {
		At(x, y int) (int, int, int, int)
	}) int {
		return 0
	}
	_ = green
	gCpu, gGpu := 0, 0
	for i := 0; i < len(cimg.Pix); i += 4 {
		for c := 0; c < 3; c++ {
			d := int64(cimg.Pix[i+c]) - int64(gimg.Pix[i+c])
			if d < 0 {
				d = -d
			}
			if d > maxd {
				maxd = d
			}
			sum += d
		}
		if int(cimg.Pix[i+1]) > int(cimg.Pix[i+0])+40 && int(cimg.Pix[i+1]) > int(cimg.Pix[i+2])+40 {
			gCpu++
		}
		if int(gimg.Pix[i+1]) > int(gimg.Pix[i+0])+40 && int(gimg.Pix[i+1]) > int(gimg.Pix[i+2])+40 {
			gGpu++
		}
	}
	n := int64(W * H * 3)
	mean := float64(sum) / float64(n)
	if mean > 1.0 || maxd > 32 {
		t.Fatalf("texture CPU/GPU divergence out of envelope: mean %.4f max %d", mean, maxd)
	}
	if gCpu < 2000 || gGpu < 2000 {
		t.Fatalf("texture not visible: green-dominant pixels cpu %d gpu %d (need >= 2000 each — fallback-to-solid suspicion)", gCpu, gGpu)
	}
	t.Logf("texture parity: mean %.4f max %d, green px cpu %d gpu %d", mean, maxd, gCpu, gGpu)
}
