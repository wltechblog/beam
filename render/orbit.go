package render

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// OrbitOpts configures a turntable orbit animation: the camera circles the
// scene's lookAt target at constant height and radius, starting at the
// scene's own camera position, covering a full 360° across Frames frames.
// Frame 0 is identical to a plain still render, and the last frame is the
// step just before returning to it — so the video loops seamlessly.
type OrbitOpts struct {
	Frames int // total frames across the 360° sweep (>= 2)
	FPS    int // output video framerate (>= 1)
}

// orbitCircle is the derived circle a camera travels: target center,
// radius, constant height offset, and starting azimuth (radians).
type orbitCircle struct {
	target geom.Vec3
	radius float64
	yOff   float64 // camera height minus target height (kept constant)
	az0    float64
}

// deriveOrbit computes the orbit circle from a scene camera. Frame 0 of the
// orbit reproduces the original camera position exactly.
func deriveOrbit(s *scene.Scene) orbitCircle {
	target := vec3Of(s.Camera.LookAt)
	pos := vec3Of(s.Camera.Pos)
	d := pos.Sub(target)
	az := math.Atan2(d.Z, d.X)
	return orbitCircle{
		target: target,
		radius: math.Hypot(d.X, d.Z), // horizontal radius: height is preserved separately
		yOff:   d.Y,
		az0:    az,
	}
}

// posAt returns the camera position at azimuth az (radians).
func (o orbitCircle) posAt(az float64) geom.Vec3 {
	return geom.Vec3{
		X: o.target.X + o.radius*math.Cos(az),
		Y: o.target.Y + o.yOff,
		Z: o.target.Z + o.radius*math.Sin(az),
	}
}

// RenderOrbitFile loads a scene and renders a 360° turntable orbit video.
func RenderOrbitFile(path string, opts Options, orbit OrbitOpts) (string, error) {
	s, err := scene.Load(path)
	if err != nil {
		return "", err
	}
	if opts.Dir == "" {
		opts.Dir = filepath.Dir(path)
	}
	return RenderOrbit(s, opts, orbit)
}

// RenderOrbit compiles the scene once, then renders Frames frames with the
// camera orbiting the lookAt target, and assembles them into an MP4 with
// ffmpeg. Returns the video path.
//
// The compiled World, resolution, sample count, seed and tonemap settings
// are shared by every frame — only the camera moves. The seed stays fixed
// across frames so the noise pattern does not shimmer between frames.
func RenderOrbit(s *scene.Scene, opts Options, orbit OrbitOpts) (string, error) {
	if orbit.Frames < 2 {
		return "", fmt.Errorf("orbit needs at least 2 frames, got %d", orbit.Frames)
	}
	if orbit.FPS < 1 {
		orbit.FPS = 30
	}

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
	aspect := float64(width) / float64(height)

	circ := deriveOrbit(s)

	// Output path: explicit -o wins; otherwise <scene>.mp4 next to nothing
	// fancy. Strip a .png suffix if the user passed one by reflex.
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

	framesDir, err := os.MkdirTemp("", "beam-orbit-")
	if err != nil {
		return "", fmt.Errorf("temp frames dir: %w", err)
	}
	// BEAM_ORBIT_KEEP_FRAMES=1 keeps the raw PNG frames after encoding
	// (path printed) — for frame-exact inspection or re-encoding.
	keepFrames := os.Getenv("BEAM_ORBIT_KEEP_FRAMES") == "1"
	if !keepFrames {
		defer os.RemoveAll(framesDir)
	}

	start := time.Now()
	for i := 0; i < orbit.Frames; i++ {
		az := circ.az0 + 2*math.Pi*float64(i)/float64(orbit.Frames)
		pos := circ.posAt(az)
		if i == 0 {
			pos = vec3Of(s.Camera.Pos) // exact: frame 0 == the scene's own camera
		}
		cam := NewCamera(
			pos,
			circ.target,
			s.Camera.FOV,
			aspect,
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
		if s.Render.MaxSeconds > 0 {
			t := time.Now().Add(time.Duration(s.Render.MaxSeconds * float64(time.Second)))
			r.Deadline = &t
		}

		img := renderOrbitFrame(r, opts, cam)
		if img == nil {
			return "", errRenderDeadline
		}

		framePath := filepath.Join(framesDir, fmt.Sprintf("frame_%04d.png", i))
		f, err := os.Create(framePath)
		if err != nil {
			return "", fmt.Errorf("create frame: %w", err)
		}
		if err := png.Encode(f, img); err != nil {
			f.Close()
			return "", fmt.Errorf("encode frame: %w", err)
		}
		f.Close()

		if !opts.Quiet {
			fmt.Fprintf(os.Stderr, "\rorbit %d/%d", i+1, orbit.Frames)
		}
	}
	if !opts.Quiet {
		fmt.Fprintln(os.Stderr)
	}

	if !opts.Quiet {
		dt := time.Since(start).Round(time.Millisecond)
		fmt.Fprintf(os.Stderr, "rendered %d frames (%dx%d, %d samples) in %s — encoding\n",
			orbit.Frames, width, height, samples, dt)
	}

	// ffmpeg runs with cmd.Dir = framesDir; a relative out would land
	// INSIDE the temp dir and be deleted by the cleanup below.
	absOut, err := filepath.Abs(out)
	if err != nil {
		absOut = out
	}
	if err := encodeMP4(framesDir, absOut, orbit.FPS, width, height, opts.Quiet); err != nil {
		// Keep the frames: the renders are the expensive part. Tell the
		// user where they are rather than deleting them.
		fmt.Fprintf(os.Stderr, "beam: ffmpeg failed (%v) — raw frames kept at %s\n", err, framesDir)
		kept := framesDir + "-kept"
		if mvErr := os.Rename(framesDir, kept); mvErr != nil {
			return "", fmt.Errorf("encode failed: %w (frames still at %s)", err, framesDir)
		}
		return "", fmt.Errorf("encode failed: %w — raw frames kept at %s", err, kept)
	}

	if !opts.Quiet {
		fmt.Fprintf(os.Stderr, "wrote %s (%d frames @ %d fps, %.1fs) in %s\n",
			out, orbit.Frames, orbit.FPS,
			float64(orbit.Frames)/float64(orbit.FPS),
			time.Since(start).Round(time.Millisecond))
		if keepFrames {
			fmt.Fprintf(os.Stderr, "raw frames kept at %s\n", framesDir)
		}
	}
	return out, nil
}

// renderOrbitFrame renders one frame via the GPU hook when requested, else
// the CPU renderer. Returns nil on deadline abort.
func renderOrbitFrame(r *Renderer, opts Options, cam *Camera) *image.RGBA {
	if !opts.GPU || GPUHook == nil {
		return r.Render(nil)
	}
	type gpuResult struct {
		img *image.RGBA
		err error
	}
	res := make(chan gpuResult, 1)
	go func() {
		im, _, e := GPUHook(r.World, cam, r.Width, r.Height, r.Samples,
			r.MaxDepth, r.Exposure, r.ClipTonemap, r.Seed)
		res <- gpuResult{im, e}
	}()
	var gr gpuResult
	if r.Deadline != nil {
		timer := time.NewTimer(time.Until(*r.Deadline))
		defer timer.Stop()
		select {
		case gr = <-res:
		case <-timer.C:
			return nil
		}
	} else {
		gr = <-res
	}
	if gr.err != nil {
		// Surface synchronously: a GPU failure must not be swallowed
		// into a "deadline" misdiagnosis.
		fmt.Fprintf(os.Stderr, "\nbeam: gpu frame failed: %v\n", gr.err)
		os.Exit(1)
	}
	return gr.img
}

// encodeMP4 assembles frame_%04d.png files into an H.264 MP4. Even
// dimensions are enforced for yuv420p compatibility.
func encodeMP4(framesDir, out string, fps, width, height int, quiet bool) error {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		return fmt.Errorf("ffmpeg not found in PATH: %w", err)
	}
	args := []string{
		"-y",
		"-framerate", strconv.Itoa(fps),
		"-i", "frame_%04d.png",
	}
	if width%2 != 0 || height%2 != 0 {
		args = append(args, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2")
	}
	args = append(args,
		"-c:v", "libx264",
		"-preset", "medium",
		"-crf", "16",
		"-pix_fmt", "yuv420p",
		"-movflags", "+faststart",
		out,
	)
	cmd := exec.Command(ffmpeg, args...)
	cmd.Dir = framesDir
	if quiet {
		cmd.Stdout = nil
		cmd.Stderr = nil
	} else {
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}
