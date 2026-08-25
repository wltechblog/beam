package gpu

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/render"
	"github.com/wltechblog/beam/scene"
)

func vec3(v []float64) geom.Vec3 {
	if len(v) != 3 {
		return geom.Vec3{X: 0, Y: 0, Z: 0}
	}
	return geom.Vec3{X: v[0], Y: v[1], Z: v[2]}
}

// TestRenderVsCPU renders simple-ab on both paths and compares stats.
func TestRenderVsCPU(t *testing.T) {
	devs, err := ProbeDevices()
	if err != nil || len(devs) == 0 {
		t.Skipf("no OpenCL GPU: %v", err)
	}

	sp := filepath.Join("..", "simple-ab", "simple.json")
	if _, err := os.Stat(sp); err != nil {
		t.Skip("no simple-ab scene")
	}
	s, err := scene.Load(sp)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	s.ApplyDefaults()
	world, err := render.Compile(s, render.NewFileResolver(filepath.Join("..", "simple-ab")))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	const W, H = 480, 270
	cam := render.NewCamera(
		vec3(s.Camera.Pos), vec3(s.Camera.LookAt),
		s.Camera.FOV, float64(W)/float64(H), s.Camera.Aperture, s.Camera.FocusDist)

	// CPU reference
	cr := &render.Renderer{World: world, Camera: cam, Width: W, Height: H,
		Samples: 2, MaxDepth: s.Render.MaxDepth, Exposure: s.Render.Exposure,
		ClipTonemap: s.Render.Tonemap == "clip", Seed: s.Render.Seed}
	cimg := cr.Render(nil)

	// GPU
	gimg, dt, err := RenderGPU(world, cam, W, H, 2, s.Render.MaxDepth,
		s.Render.Exposure, s.Render.Tonemap == "clip", s.Render.Seed, false)
	if err != nil {
		t.Fatalf("GPU render: %v", err)
	}
	t.Logf("GPU kernel: %v (%d×%d @2spp)", dt, W, H)

	// stats: max/mean abs diff
	var maxd, sum int64
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
	}
	n := int64(W * H * 3)
	t.Logf("CPU vs GPU(fp32): maxΔ=%d meanΔ=%.2f", maxd, float64(sum)/float64(n))

	// save for visual diff
	if err := png.Encode(writeTo(t, "/tmp/beam_cpu.png"), cimg); err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(writeTo(t, "/tmp/beam_gpu.png"), gimg); err != nil {
		t.Fatal(err)
	}

	// fp32 tolerance: mean should be small (1-2 units), max can spike on edges
	mean := float64(sum) / float64(n)
	if mean > 8 {
		t.Errorf("mean delta too high: %.2f — structural divergence", mean)
	}
}

func writeTo(t *testing.T, p string) *os.File {
	t.Helper()
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}
