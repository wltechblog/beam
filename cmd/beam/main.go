// beam renders JSON scene descriptions to PNG.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/wltechblog/beam/gpu"
	"os"
	"path/filepath"
	"time"

	"github.com/wltechblog/beam/render"
	"github.com/wltechblog/beam/scene"
)

func main() {
	preview := flag.Bool("preview", false, "fast low-res render (320px, 1 sample)")
	quiet := flag.Bool("q", false, "suppress progress output")
	out := flag.String("o", "", "output path override (default: <scene>.png)")
	fit := flag.Bool("fit", false, "pre-render fit check: project object bounds to screen, report clipping, exit 2 if anything clips")
	gpuFlag := flag.Bool("gpu", false, "render on GPU (OpenCL; hard-fails with diagnostics if no device)")
	orbit := flag.Int("orbit", 0, "render a 360° turntable orbit video: N frames around the lookAt target")
	anim := flag.Bool("anim", false, "render the scene's animation block (keyframed timeline) to an MP4")
	// 0 = unset: -anim then honors the scene's animation.fps (RenderTimeline
	// falls back to 30 only when neither is given). -orbit keeps 30 unless set.
	fps := flag.Int("fps", 0, "output video framerate for -orbit and -anim (0 = scene animation.fps for -anim)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: beam [options] scene.json\n\noptions:\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}

	if *fit {
		os.Exit(runFitCheck(flag.Arg(0), *out != ""))
	}

	if *orbit > 0 {
		if *gpuFlag {
			render.SetGPUHook(gpu.Hook)
		}
		os.Exit(runOrbit(flag.Arg(0), render.Options{
			Preview: *preview,
			Quiet:   *quiet,
			Output:  *out,
			GPU:     *gpuFlag,
		}, render.OrbitOpts{Frames: *orbit, FPS: *fps}))
	}

	if *anim {
		if *gpuFlag {
			render.SetGPUHook(gpu.Hook)
		}
		os.Exit(runAnim(flag.Arg(0), render.Options{
			Preview: *preview,
			Quiet:   *quiet,
			Output:  *out,
			GPU:     *gpuFlag,
		}, *fps))
	}

	start := time.Now()
	if *gpuFlag {
		render.SetGPUHook(gpu.Hook)
	}
	path, err := render.RenderFile(flag.Arg(0), render.Options{
		Preview: *preview,
		Quiet:   *quiet,
		Output:  *out,
		GPU:     *gpuFlag,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "beam:", err)
		os.Exit(1)
	}
	fmt.Printf("%s (%.1fs)\n", path, time.Since(start).Seconds())
}

// runFitCheck loads the scene, projects every object's bounds to screen
// space, and prints a JSON report. Exit codes: 0 = fits, 2 = clipping,
// 1 = scene failed to load/compile.
func runFitCheck(path string, quiet bool) int {
	s, err := scene.Load(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "beam:", err)
		return 1
	}
	dir := filepath.Dir(path)
	width, height := s.Render.Width, s.Render.Height
	if width == 0 {
		width = 800
	}
	if height == 0 {
		height = 600
	}
	report, err := render.ScreenFitCheck(s, render.NewFileResolver(dir), width, height)
	if err != nil {
		fmt.Fprintln(os.Stderr, "beam:", err)
		return 1
	}
	if !quiet {
		b, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(b))
	}
	if !report.Fitted {
		os.Exit(2)
	}
	return 0
}

// runOrbit renders a 360° turntable orbit video of the scene.
func runOrbit(path string, opts render.Options, orbit render.OrbitOpts) int {
	start := time.Now()
	videoPath, err := render.RenderOrbitFile(path, opts, orbit)
	if err != nil {
		fmt.Fprintln(os.Stderr, "beam:", err)
		return 1
	}
	fmt.Printf("%s (%.1fs)\n", videoPath, time.Since(start).Seconds())
	return 0
}

// runAnim renders the scene's animation block to an MP4.
func runAnim(path string, opts render.Options, fps int) int {
	start := time.Now()
	videoPath, err := render.RenderTimelineFile(path, opts, fps)
	if err != nil {
		fmt.Fprintln(os.Stderr, "beam:", err)
		return 1
	}
	fmt.Printf("%s (%.1fs)\n", videoPath, time.Since(start).Seconds())
	return 0
}
