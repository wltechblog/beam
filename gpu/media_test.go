package gpu

import (
	"testing"

	"github.com/wltechblog/beam/render"
	"github.com/wltechblog/beam/scene"
)

// mediaScene: solid floor + pillars + participating media with single
// scattering. No checker (grazing-angle aliasing would swamp the media
// signal) — this pins the media march itself.
const mediaScene = `{
  "version": 1,
  "camera": {"pos": [0, 2.2, -7.5], "lookAt": [0, 1.4, 0], "fov": 42},
  "materials": {
    "floor":  {"type": "matte", "color": [0.32, 0.30, 0.28]},
    "pillar": {"type": "matte", "color": [0.55, 0.53, 0.50]}
  },
  "objects": [
    {"type": "plane", "point": [0, 0, 0], "normal": [0, 1, 0], "material": "floor"},
    {"type": "box", "min": [-0.3, 0, -0.3], "max": [0.3, 3.2, 0.3], "material": "pillar"},
    {"type": "box", "min": [-2.4, 0, -0.3], "max": [-1.8, 3.2, 0.3], "material": "pillar"}
  ],
  "lights": [
    {"type": "point", "pos": [-3.5, 4.5, -4.0], "intensity": 3.0, "falloff": "none"}
  ],
  "environment": {
    "color": [0.02, 0.03, 0.05], "ambient": 0.06,
    "media": {"color": [0.55, 0.62, 0.78], "distance": 10.0, "samples": 8}
  },
  "render": {"width": 160, "height": 120, "samples": 2, "maxDepth": 4, "seed": 7, "exposure": 1.0}
}`

// TestGpuMediaParity: CPU vs GPU renders of a single-scattering scene.
// The media march is RNG-free (deterministic midpoint), so the only
// divergence source is fp32 rounding in the march arithmetic. Envelope:
// mean < 1.0, max <= 32, <0.1% of pixels over 32.
func TestGpuMediaParity(t *testing.T) {
	devs, err := ProbeDevices()
	if err != nil || len(devs) == 0 {
		t.Skipf("no OpenCL GPU: %v", err)
	}
	s, err := scene.Parse([]byte(mediaScene))
	if err != nil {
		t.Fatal(err)
	}
	s.ApplyDefaults()
	world, err := render.Compile(s, nil)
	if err != nil {
		t.Fatal(err)
	}
	if world.Media == nil {
		t.Fatal("media not compiled")
	}

	const W, H = 160, 120
	cam := render.NewCamera(
		vec3(s.Camera.Pos), vec3(s.Camera.LookAt),
		s.Camera.FOV, float64(W)/float64(H), s.Camera.Aperture, s.Camera.FocusDist)

	cr := &render.Renderer{World: world, Camera: cam, Width: W, Height: H,
		Samples: s.Render.Samples, MaxDepth: s.Render.MaxDepth, Exposure: s.Render.Exposure,
		ClipTonemap: s.Render.Tonemap == "clip", Seed: s.Render.Seed}
	cimg := cr.Render(nil)

	gimg, _, err := RenderGPU(world, cam, W, H, s.Render.Samples, s.Render.MaxDepth,
		s.Render.Exposure, s.Render.Tonemap == "clip", s.Render.Seed, false)
	if err != nil {
		t.Fatalf("GPU render: %v", err)
	}

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
	mean := float64(sum) / float64(n)
	over := 0
	for i := 0; i < len(cimg.Pix); i += 4 {
		for c := 0; c < 3; c++ {
			if abs64(int64(cimg.Pix[i+c])-int64(gimg.Pix[i+c])) > 32 {
				over++
			}
		}
	}
	if mean > 1.0 || maxd > 32 || int64(over) > n/1000 {
		t.Fatalf("media CPU/GPU divergence out of envelope: mean %.4f max %d over32 %d/%d", mean, maxd, over, n)
	}
	t.Logf("media parity: mean %.4f max %d", mean, maxd)
}

func abs64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}
