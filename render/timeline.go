package render

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/wltechblog/beam/asset"
	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// RenderTimelineFile loads a scene and renders its animation block to an
// MP4. The animation must validate (it did at Parse time).
func RenderTimelineFile(path string, opts Options, fpsOverride int) (string, error) {
	s, err := scene.Load(path)
	if err != nil {
		return "", err
	}
	if opts.Dir == "" {
		opts.Dir = filepath.Dir(path)
	}
	return RenderTimeline(s, opts, fpsOverride)
}

// RenderTimeline renders the scene's animation block: duration*fps frames,
// each frame sampling every track at its time t, recompiling the scene
// (transforms are baked at compile time), and assembling an MP4.
//
// Design notes:
//
//   - Recompile-per-frame is deliberate. Compile is milliseconds even on
//     the 11K-triangle teapot; render dominates by orders of magnitude.
//     Per-frame transform baking keeps CPU and GPU bit-compatible with a
//     still render of the same pose — frame 0 of a timeline whose tracks
//     all start at t=0 with the authored values is exactly a still render.
//   - Meshes are loaded once and cached by src (or text params): the
//     recompile must not re-read teapot.obj 120 times. The cached mesh is
//     the UNtransformed base; Compile clones + transforms it per frame.
//   - The seed stays fixed across frames (no noise shimmer).
//   - fpsOverride (from -fps) wins over the scene's animation.fps.
func RenderTimeline(s *scene.Scene, opts Options, fpsOverride int) (string, error) {
	a := s.Animation
	if a == nil {
		return "", fmt.Errorf("scene has no animation block")
	}
	fps := a.FPS
	if fpsOverride > 0 {
		fps = fpsOverride
	}
	if fps < 1 {
		fps = 30 // scene said nothing AND -fps unset: 30
	}
	frames := animFrameCount(a)

	s.ApplyDefaults()
	resolver := &cachingResolver{inner: NewFileResolver(opts.Dir)}

	width, height, samples := s.Render.Width, s.Render.Height, s.Render.Samples
	if opts.Preview {
		h := int(float64(height) * 320.0 / float64(width))
		if h < 1 {
			h = 1
		}
		width, height, samples = 320, h, 1
	}
	aspect := float64(width) / float64(height)

	out := timelineOutputPath(s, opts)

	framesDir, err := os.MkdirTemp("", "beam-anim-")
	if err != nil {
		return "", fmt.Errorf("temp frames dir: %w", err)
	}
	keepFrames := os.Getenv("BEAM_ORBIT_KEEP_FRAMES") == "1"
	if !keepFrames {
		defer os.RemoveAll(framesDir)
	}

	start := time.Now()
	for i := 0; i < frames; i++ {
		t := animTimeOf(a, i)

		// Deep-copy camera + objects + lights, sample tracks onto the
		// copy, compile THAT. The authored scene document stays pristine
		// (frame 0 must be reproducible regardless of frame order).
		frame := cloneSceneForFrame(s)
		if err := applyAnimationAt(frame, t); err != nil {
			return "", fmt.Errorf("frame %d (t=%v): %w", i, t, err)
		}

		world, err := Compile(frame, resolver)
		if err != nil {
			return "", fmt.Errorf("frame %d (t=%v): %w", i, t, err)
		}

		cam := NewCamera(
			vec3Of(frame.Camera.Pos),
			vec3Of(frame.Camera.LookAt),
			frame.Camera.FOV,
			aspect,
			frame.Camera.Aperture,
			frame.Camera.FocusDist,
		)

		r := &Renderer{
			World:       world,
			Camera:      cam,
			Width:       width,
			Height:      height,
			Samples:     samples,
			MaxDepth:    frame.Render.MaxDepth,
			Exposure:    frame.Render.Exposure,
			ClipTonemap: frame.Render.Tonemap == "clip",
			Seed:        s.Render.Seed,
		}
		if s.Render.MaxSeconds > 0 {
			td := time.Now().Add(time.Duration(s.Render.MaxSeconds * float64(time.Second)))
			r.Deadline = &td
		}

		img := renderOrbitFrame(r, opts, cam) // same GPU/CPU + deadline logic as orbit
		if img == nil {
			return "", errRenderDeadline
		}

		framePath := filepath.Join(framesDir, fmt.Sprintf("frame_%04d.png", i))
		if err := writeFramePNG(framePath, img); err != nil {
			return "", err
		}

		if !opts.Quiet {
			fmt.Fprintf(os.Stderr, "\ranim %d/%d", i+1, frames)
		}
	}
	if !opts.Quiet {
		fmt.Fprintln(os.Stderr)
		dt := time.Since(start).Round(time.Millisecond)
		fmt.Fprintf(os.Stderr, "rendered %d frames (%dx%d, %d samples) in %s — encoding\n",
			frames, width, height, samples, dt)
	}

	absOut, err := filepath.Abs(out)
	if err != nil {
		absOut = out
	}
	if err := encodeMP4(framesDir, absOut, fps, width, height, opts.Quiet); err != nil {
		fmt.Fprintf(os.Stderr, "beam: ffmpeg failed (%v) — raw frames kept at %s\n", err, framesDir)
		kept := framesDir + "-kept"
		if mvErr := os.Rename(framesDir, kept); mvErr != nil {
			return "", fmt.Errorf("encode failed: %w (frames still at %s)", err, framesDir)
		}
		return "", fmt.Errorf("encode failed: %w — raw frames kept at %s", err, kept)
	}

	if !opts.Quiet {
		fmt.Fprintf(os.Stderr, "wrote %s (%d frames @ %d fps, %.2fs) in %s\n",
			out, frames, fps, a.Duration, time.Since(start).Round(time.Millisecond))
		if keepFrames {
			fmt.Fprintf(os.Stderr, "raw frames kept at %s\n", framesDir)
		}
	}
	return out, nil
}

// timelineOutputPath mirrors the orbit output-path rules: -o wins, else
// <scene>.mp4 with a .png reflex-strip.
func timelineOutputPath(s *scene.Scene, opts Options) string {
	out := opts.Output
	if out == "" {
		out = strings.TrimSuffix(filepath.Base(s.Render.Output), ".png")
		if out == "" || out == "." {
			out = "beam-out"
		}
		out += ".mp4"
	} else if strings.HasSuffix(out, ".png") {
		out = strings.TrimSuffix(out, ".png") + ".mp4"
	}
	if dir := filepath.Dir(out); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	return out
}

// writeFramePNG encodes one frame to disk.
func writeFramePNG(path string, img *image.RGBA) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create frame: %w", err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return fmt.Errorf("encode frame: %w", err)
	}
	return nil
}

// cloneSceneForFrame deep-copies the parts a frame can mutate: camera
// fields, objects (transforms), and lights. Materials / environment /
// render opts are shared read-only — Compile does not mutate them.
func cloneSceneForFrame(s *scene.Scene) *scene.Scene {
	c := &scene.Scene{
		Version:     s.Version,
		Camera:      cloneCamera(&s.Camera),
		Materials:   s.Materials,
		Objects:     make([]scene.Object, len(s.Objects)),
		Lights:      make([]scene.Light, len(s.Lights)),
		Environment: s.Environment,
		Render:      s.Render,
		Animation:   s.Animation,
	}
	for i := range s.Objects {
		o := s.Objects[i]
		if o.Transform != nil {
			t := *o.Transform // clone the struct...
			// ...and the slices: applyObjectProp copies INTO existing
			// 3-vectors, so a shared backing array would let one
			// frame's values leak into the authored scene.
			t.Translate = append([]float64(nil), t.Translate...)
			t.Rotate = append([]float64(nil), t.Rotate...)
			t.Scale = append([]float64(nil), t.Scale...)
			o.Transform = &t
		}
		c.Objects[i] = o
	}
	for i := range s.Lights {
		l := s.Lights[i]
		l.Pos = append([]float64(nil), l.Pos...)
		l.Color = append([]float64(nil), l.Color...)
		c.Lights[i] = l
	}
	return c
}

func cloneCamera(c *scene.Camera) scene.Camera {
	out := scene.Camera{
		Pos:       append([]float64(nil), c.Pos...),
		LookAt:    append([]float64(nil), c.LookAt...),
		Up:        append([]float64(nil), c.Up...),
		FOV:       c.FOV,
		Aperture:  c.Aperture,
		FocusDist: c.FocusDist,
	}
	return out
}

// cachingResolver wraps a FileResolver, memoizing meshes by src and text
// meshes by their generation parameters. Returned meshes are the cached
// base objects — Compile clones before transforming (transformMesh always
// allocates fresh Verts/Norms slices), so per-frame mutation is safe.
type cachingResolver struct {
	inner    MeshResolver
	mesh     map[string]*geom.TriangleMesh
	textures map[string]*asset.Texture
	models   map[string]*asset.LoadedModel
}

func (c *cachingResolver) LoadMesh(src string) (*geom.TriangleMesh, error) {
	if m, ok := c.mesh[src]; ok {
		return m, nil
	}
	m, err := c.inner.LoadMesh(src)
	if err != nil {
		return nil, err
	}
	if c.mesh == nil {
		c.mesh = map[string]*geom.TriangleMesh{}
	}
	c.mesh[src] = m
	return m, nil
}

// LoadTexture memoizes textures by path — same policy as meshes. THIS is
// what makes animated textured scenes resolve: Compile only loads maps
// through a TextureResolver, and the still path's bare FileResolver has
// one while this wrapper previously did not.
func (c *cachingResolver) LoadTexture(path string) (*asset.Texture, error) {
	if t, ok := c.textures[path]; ok {
		return t, nil
	}
	tr, ok := c.inner.(TextureResolver)
	if !ok {
		return nil, fmt.Errorf("texture %q: resolver cannot load images", path)
	}
	t, err := tr.LoadTexture(path)
	if err != nil {
		return nil, err
	}
	if c.textures == nil {
		c.textures = map[string]*asset.Texture{}
	}
	c.textures[path] = t
	return t, nil
}

// LoadEnvMap memoizes env maps by path — same policy as textures. Keeps
// animated envMap scenes from re-decoding the HDR every frame.
func (c *cachingResolver) LoadEnvMap(path string) (*asset.Texture, error) {
	if t, ok := c.textures[path]; ok {
		return t, nil
	}
	er, ok := c.inner.(EnvMapResolver)
	if !ok {
		return nil, fmt.Errorf("envMap %q: resolver cannot load env maps", path)
	}
	t, err := er.LoadEnvMap(path)
	if err != nil {
		return nil, err
	}
	if c.textures == nil {
		c.textures = map[string]*asset.Texture{}
	}
	c.textures[path] = t
	return t, nil
}

// LoadModel memoizes imported models (glTF/GLB) by src — same policy as
// meshes and textures. Compile clones before recentering/transforming
// (compileModel allocates fresh Verts per part), so per-frame reuse is
// safe. Without this, animated scenes with model objects fail at frame
// 0 — the wrapper is the resolver Compile actually sees on the anim path.
func (c *cachingResolver) LoadModel(src string) (*asset.LoadedModel, error) {
	if m, ok := c.models[src]; ok {
		return m, nil
	}
	mr, ok := c.inner.(ModelResolver)
	if !ok {
		return nil, fmt.Errorf("model %q: resolver cannot import models", src)
	}
	m, err := mr.LoadModel(src)
	if err != nil {
		return nil, err
	}
	if c.models == nil {
		c.models = map[string]*asset.LoadedModel{}
	}
	c.models[src] = m
	return m, nil
}

func (c *cachingResolver) TextMesh(text, font string, size, depth, spacing float64) (*geom.TriangleMesh, error) {
	key := fmt.Sprintf("text\x00%s\x00%s\x00%v\x00%v\x00%v", text, font, size, depth, spacing)
	if m, ok := c.mesh[key]; ok {
		return m, nil
	}
	m, err := c.inner.TextMesh(text, font, size, depth, spacing)
	if err != nil {
		return nil, err
	}
	if c.mesh == nil {
		c.mesh = map[string]*geom.TriangleMesh{}
	}
	c.mesh[key] = m
	return m, nil
}
