package render

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"time"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// Options tweak rendering independent of scene content.
type Options struct {
	Preview bool   // 320px wide, 1 sample
	Quiet   bool   // no progress output
	Output  string // output path override
	Dir     string // base dir for mesh/font resolution (defaults to scene dir)
	GPU     bool   // render on GPU via the registered hook (falls back to CPU)
}

// GPUHook, when set, renders a compiled scene on the GPU and returns the
// image. It exists as a hook so the render package carries no OpenCL
// dependency; the gpu package registers itself from main via
// render.SetGPUHook. Must not write the PNG — pipeline owns that.
var GPUHook func(w *World, cam *Camera, width, height, samples, maxDepth int,
	exposure float64, clip bool, seed int64) (*image.RGBA, time.Duration, error)

// SetGPUHook installs the GPU renderer (idempotent, called from main).
func SetGPUHook(f func(w *World, cam *Camera, width, height, samples, maxDepth int,
	exposure float64, clip bool, seed int64) (*image.RGBA, time.Duration, error)) {
	GPUHook = f
}

// RenderFile loads, compiles, and renders a scene file to PNG. Single entry
// point used by the CLI and (later) the MCP server.
func RenderFile(path string, opts Options) (string, error) {
	s, err := scene.Load(path)
	if err != nil {
		return "", err
	}
	if opts.Dir == "" {
		opts.Dir = filepath.Dir(path)
	}
	return RenderScene(s, opts)
}

// RenderScene compiles and renders an in-memory scene. Returns the output
// path.
func RenderScene(s *scene.Scene, opts Options) (string, error) {
	s.ApplyDefaults()

	resolver := NewFileResolver(opts.Dir)
	world, err := Compile(s, resolver)
	if err != nil {
		return "", err
	}

	width, height, samples := s.Render.Width, s.Render.Height, s.Render.Samples
	if opts.Preview {
		h := int(float64(height) * 320.0 / float64(width))
		if h < 1 {
			h = 1
		}
		width, height, samples = 320, h, 1
	}

	cam := NewCamera(
		vec3Of(s.Camera.Pos),
		vec3Of(s.Camera.LookAt),
		s.Camera.FOV,
		float64(width)/float64(height),
		s.Camera.Aperture,
		s.Camera.FocusDist,
	)

	r := &Renderer{
		World:       world,
		Camera:      cam,
		Width:       width,
		Height:      height,
		Samples:     samples,
		MaxDepth:    s.Render.MaxDepth,
		Exposure:    s.Render.Exposure,
		ClipTonemap: s.Render.Tonemap == "clip",
		Seed:        s.Render.Seed,
	}

	out := opts.Output
	if out == "" {
		out = s.Render.Output
	}
	if out == "" {
		out = "beam-out.png"
	}

	start := time.Now()
	var img *image.RGBA
	if opts.GPU {
		if GPUHook == nil {
			return "", fmt.Errorf("gpu render requested but no GPU hook is registered " +
				"(the gpu package was not linked into this binary)")
		}
		var gpuDT time.Duration
		img, gpuDT, err = GPUHook(world, cam, width, height, samples,
			r.MaxDepth, r.Exposure, r.ClipTonemap, r.Seed)
		if err != nil {
			return "", fmt.Errorf("gpu render failed: %w", err)
		}
		if !opts.Quiet {
			fmt.Fprintf(os.Stderr, "gpu kernel %s\n", gpuDT.Round(time.Millisecond))
		}
	}
	if img == nil {
		if opts.Quiet {
			img = r.Render(nil)
		} else {
			lastPct := -1
			img = r.Render(func(finished, total int) {
				pct := finished * 100 / total
				if pct != lastPct {
					lastPct = pct
					fmt.Fprintf(os.Stderr, "\rrender %3d%%", pct)
				}
			})
			fmt.Fprintln(os.Stderr)
		}
	}

	f, err := os.Create(out)
	if err != nil {
		return "", fmt.Errorf("create output: %w", err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return "", fmt.Errorf("encode png: %w", err)
	}
	if !opts.Quiet {
		fmt.Fprintf(os.Stderr, "wrote %s (%dx%d, %d samples) in %s\n",
			out, width, height, samples, time.Since(start).Round(time.Millisecond))
	}
	return out, nil
}

func vec3Of(v []float64) geom.Vec3 {
	if len(v) != 3 {
		return geom.Vec3{X: 0, Y: 0.8, Z: 0}
	}
	return geom.Vec3{X: v[0], Y: v[1], Z: v[2]}
}
