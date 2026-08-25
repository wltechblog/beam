package gpu

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/render"
	"github.com/wltechblog/beam/scene"
)

// TestBenchmarkPOV — the real target: 1080p 4spp, CPU baseline 18.9s.
func TestBenchmarkPOV(t *testing.T) {
	if os.Getenv("BENCH") == "" {
		t.Skip("set BENCH=1 to run the full benchmark")
	}
	devs, err := ProbeDevices()
	if err != nil || len(devs) == 0 {
		t.Skipf("no OpenCL GPU: %v", err)
	}

	sp := filepath.Join("..", "pov-benchmark", "beam-benchmark.json")
	s, err := scene.Load(sp)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	s.ApplyDefaults()
	dir := filepath.Join("..", "pov-benchmark")
	world, err := render.Compile(s, render.NewFileResolver(dir))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	const W, H = 1920, 1080
	cam := render.NewCamera(vec3(s.Camera.Pos), vec3(s.Camera.LookAt),
		s.Camera.FOV, float64(W)/float64(H), s.Camera.Aperture, s.Camera.FocusDist)

	// CPU
	cr := &render.Renderer{World: world, Camera: cam, Width: W, Height: H,
		Samples: s.Render.Samples, MaxDepth: s.Render.MaxDepth, Exposure: s.Render.Exposure,
		ClipTonemap: s.Render.Tonemap == "clip", Seed: s.Render.Seed}
	cimg := cr.Render(nil)

	// GPU
	gimg, dt, err := RenderGPU(world, cam, W, H, s.Render.Samples, s.Render.MaxDepth,
		s.Render.Exposure, s.Render.Tonemap == "clip", s.Render.Seed, false)
	if err != nil {
		t.Fatalf("GPU render: %v", err)
	}

	var maxd, sum int64
	bad := 0
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
			if d > 32 {
				bad++
			}
		}
	}
	n := int64(W * H * 3)
	t.Logf("samples=%d maxDepth=%d", s.Render.Samples, s.Render.MaxDepth)
	t.Logf("GPU kernel total: %v (incl. buffer upload + readback)", dt)
	t.Logf("CPU vs GPU(fp32): maxΔ=%d meanΔ=%.2f  pixels>32Δ: %.3f%%",
		maxd, float64(sum)/float64(n), 100*float64(bad)/float64(n))

	if err := png.Encode(writeTo(t, "/tmp/pov_cpu_1080.png"), cimg); err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(writeTo(t, "/tmp/pov_gpu_1080.png"), gimg); err != nil {
		t.Fatal(err)
	}
}
