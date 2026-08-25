package gpu

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/render"
	"github.com/wltechblog/beam/scene"
)

// TestGpuSmokeMeshless is the fast GPU-vs-CPU canary (~0.2s total).
//
// gpu-smoke/scene.json: 3 objects (plane + 2 spheres), NO meshes — the
// global BVH/triangle arrays are empty. This guards against zero-length
// buffer-upload panics (busted in RenderGPU before the padding fix) and
// confirms basic plane/sphere/metal/glass parity on solid surfaces.
//
// The floor is deliberately SOLID: a checker floor at grazing camera
// angles aliases sub-pixel and amplifies fp32-vs-fp64 jitter into
// mean-delta ~4 (boundary-straddling sample flips), which is expected
// fp32 behavior, not a regression. Solid surfaces give tight parity.
func TestGpuSmokeMeshless(t *testing.T) {
	devs, err := ProbeDevices()
	if err != nil || len(devs) == 0 {
		t.Skipf("no OpenCL GPU: %v", err)
	}

	sp := filepath.Join("..", "gpu-smoke", "scene.json")
	if _, err := os.Stat(sp); err != nil {
		t.Skip("no gpu-smoke scene")
	}
	s, err := scene.Load(sp)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	s.ApplyDefaults()
	world, err := render.Compile(s, render.NewFileResolver(filepath.Join("..", "gpu-smoke")))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	const W, H = 320, 180
	cam := render.NewCamera(vec3(s.Camera.Pos), vec3(s.Camera.LookAt),
		s.Camera.FOV, float64(W)/float64(H), s.Camera.Aperture, s.Camera.FocusDist)

	// CPU reference — would previously have been the only path that worked;
	// GPU below panicked on mesh-less scenes before the padding fix.
	cr := &render.Renderer{World: world, Camera: cam, Width: W, Height: H,
		Samples: s.Render.Samples, MaxDepth: s.Render.MaxDepth, Exposure: s.Render.Exposure,
		ClipTonemap: s.Render.Tonemap == "clip", Seed: s.Render.Seed}
	cimg := cr.Render(nil)

	gimg, dt, err := RenderGPU(world, cam, W, H, s.Render.Samples, s.Render.MaxDepth,
		s.Render.Exposure, s.Render.Tonemap == "clip", s.Render.Seed, false)
	if err != nil {
		t.Fatalf("GPU render: %v", err)
	}
	t.Logf("GPU kernel: %v (%d×%d @%dspp)", dt, W, H, s.Render.Samples)

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
	mean := float64(sum) / float64(W*H*3)
	t.Logf("CPU vs GPU(fp32): maxΔ=%d meanΔ=%.4f", maxd, mean)

	if err := png.Encode(writeTo(t, "/tmp/beam_smoke_cpu.png"), cimg); err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(writeTo(t, "/tmp/beam_smoke_gpu.png"), gimg); err != nil {
		t.Fatal(err)
	}

	// Solid surfaces: expect near-perfect parity (observed maxΔ=9, mean=0.000).
	// Anything above this means structural divergence (wrong shapes, wrong
	// materials, wrong lighting), not fp32 rounding noise.
	if mean > 1.0 {
		t.Errorf("mean delta too high: %.4f — structural divergence on solid surfaces", mean)
	}
	if maxd > 16 {
		t.Errorf("max delta too high: %d — check sphere silhouettes / shadows", maxd)
	}
}
