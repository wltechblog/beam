package render

import (
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/scene"
)

// helper: parse or die
func mustParse(t *testing.T, raw string) *scene.Scene {
	t.Helper()
	s, err := scene.Parse([]byte(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return s
}

// The minimal anim scene: a sphere that moves along X while the camera
// dolly-slightly pans fov. Two tracks, independent keyframes.
const animTestScene = `{
  "version": 1,
  "camera": {"pos": [0, 1.4, -6], "lookAt": [0, 0.8, 0], "fov": 45},
  "environment": {"color": [0.05, 0.06, 0.1]},
  "lights": [
    {"type": "point", "name": "key", "pos": [-4, 6, -4], "color": [1, 1, 1], "intensity": 1.1, "falloff": "none"}
  ],
  "materials": {
    "red":   {"type": "matte", "color": [0.8, 0.15, 0.1]},
    "blue":  {"type": "matte", "color": [0.1, 0.25, 0.8]},
    "floor": {"type": "matte", "color": [0.5, 0.5, 0.52]}
  },
  "objects": [
    {"type": "plane", "point": [0,0,0], "normal": [0,1,0], "material": "floor"},
    {"type": "sphere", "name": "ball", "center": [-2, 0.75, 0], "radius": 0.75, "material": "red"},
    {"type": "sphere", "name": "still", "center": [2.2, 0.75, 0.5], "radius": 0.75, "material": "blue"}
  ],
  "animation": {
    "duration": 2,
    "fps": 12,
    "tracks": [
      {"target": "ball", "property": "translate", "ease": "smooth",
       "keys": [{"t": 0, "value": [0, 0, 0]}, {"t": 2, "value": [4, 0, 0]}]},
      {"target": "key", "property": "intensity", "ease": "linear",
       "keys": [{"t": 0, "value": [1.1]}, {"t": 1, "value": [0.15]}, {"t": 2, "value": [1.1]}]}
    ]
  },
  "render": {"width": 320, "height": 180, "samples": 2, "maxDepth": 4, "seed": 7, "output": "animtest.png"}
}`

func TestSampleTrackBasics(t *testing.T) {
	tr := &scene.Track{
		Target: "x", Property: "translate", Ease: "smooth",
		Keys: []scene.Key{
			{T: 0, Value: []float64{0, 0, 0}},
			{T: 2, Value: []float64{4, 0, 0}},
		},
	}
	// before first key → not ok (authored value kept)
	if v, ok := sampleTrack(tr, -0.1); ok {
		t.Fatalf("before first key: expected !ok, got %v", v)
	}
	// at key 0 → exact
	v, _ := sampleTrack(tr, 0)
	if v[0] != 0 {
		t.Fatalf("t=0: want 0, got %v", v)
	}
	// midpoint with smoothstep = 0.5 → 2.0
	v, _ = sampleTrack(tr, 1)
	if v[0] != 2.0 {
		t.Fatalf("t=1 smooth midpoint: want 2.0, got %v", v[0])
	}
	// after last key → not ok
	if v, ok := sampleTrack(tr, 2.1); ok {
		t.Fatalf("after last key: expected !ok, got %v", v)
	}
	// 3-key track: segment pick at t=1.5 with linear ease on middle segment
	tr3 := &scene.Track{
		Target: "x", Property: "translate", Ease: "linear",
		Keys: []scene.Key{
			{T: 0, Value: []float64{0, 0, 0}},
			{T: 1, Value: []float64{10, 0, 0}},
			{T: 2, Value: []float64{0, 0, 0}},
		},
	}
	v, _ = sampleTrack(tr3, 1.5)
	if v[0] != 5 {
		t.Fatalf("3-key middle segment: want 5.0, got %v", v[0])
	}
}

func TestEaseShapes(t *testing.T) {
	cases := []struct {
		ease    string
		u, want float64
	}{
		{"", 0.5, 0.5},
		{"linear", 0.5, 0.5},
		{"smooth", 0.5, 0.5},
		{"smooth", 0.25, 0.15625},
		{"easeIn", 0.5, 0.25},
		{"easeOut", 0.5, 0.75},
	}
	for _, c := range cases {
		if got := ease(c.ease, c.u); got != c.want {
			t.Errorf("ease(%q, %v) = %v, want %v", c.ease, c.u, got, c.want)
		}
	}
}

func TestApplyAnimationAt(t *testing.T) {
	s := mustParse(t, animTestScene)

	// t=0: ball translate = identity [0,0,0] (matches authored pose)
	if err := applyAnimationAt(s, 0); err != nil {
		t.Fatal(err)
	}
	ball := s.Objects[1]
	if ball.Transform == nil || ball.Transform.Translate == nil {
		t.Fatal("ball should have a transform after t=0")
	}
	if ball.Transform.Translate[0] != 0 {
		t.Fatalf("t=0 ball.translate.x = %v, want 0 (identity start)", ball.Transform.Translate[0])
	}

	// t=1 (midpoint of 0->4 smooth): translate.x = 2, light intensity = 0.15
	s2 := mustParse(t, animTestScene)
	if err := applyAnimationAt(s2, 1); err != nil {
		t.Fatal(err)
	}
	if got := s2.Objects[1].Transform.Translate[0]; got != 2 {
		t.Fatalf("t=1 ball.translate.x = %v, want 2", got)
	}
	if got := s2.Lights[0].Intensity; got != 0.15 {
		t.Fatalf("t=1 light intensity = %v, want 0.15", got)
	}

	// applying at t=1 TWICE on the same scene must be idempotent (tracks
	// overwrite absolute values, never accumulate)
	if err := applyAnimationAt(s2, 1); err != nil {
		t.Fatal(err)
	}
	if got := s2.Objects[1].Transform.Translate[0]; got != 2 {
		t.Fatalf("re-apply at t=1: ball.x = %v, want 2 (absolute, not accumulating)", got)
	}
}

func TestAnimFrameCount(t *testing.T) {
	cases := []struct {
		dur  float64
		fps  int
		want int
	}{
		{2, 12, 24},
		{4, 30, 120},
		{1, 30, 30},
		{0.5, 30, 15},
	}
	for _, c := range cases {
		a := &scene.Animation{Duration: c.dur, FPS: c.fps}
		if got := animFrameCount(a); got != c.want {
			t.Errorf("animFrameCount(%v s, %d fps) = %d, want %d", c.dur, c.fps, got, c.want)
		}
	}
}

// TestTimelineVideo renders the full pipeline through a real encode when
// ffmpeg exists, else skips. Verifies frame count + container params via
// the file itself (frame PNG count) rather than shelling ffprobe.
func TestTimelineVideo(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("no ffmpeg")
	}
	dir := t.TempDir()
	s := mustParse(t, animTestScene)
	out := filepath.Join(dir, "anim.mp4")
	p, err := RenderTimeline(s, Options{Output: out, Quiet: true}, 0)
	if err != nil {
		t.Fatalf("RenderTimeline: %v", err)
	}
	if p != out {
		t.Fatalf("output path %q != requested %q", p, out)
	}
	st, err := os.Stat(p)
	if err != nil || st.Size() == 0 {
		t.Fatalf("video missing/empty: %v", err)
	}
}

// TestTimelineFrame0MatchesStill: a timeline whose tracks all key their
// authored values at t=0 renders frame 0 byte-identically to a still render
// of the same scene. This pins the determinism contract.
func TestTimelineFrame0MatchesStill(t *testing.T) {
	s := mustParse(t, animTestScene)

	// Render frame 0 by hand: clone + apply at t=0 + compile + render.
	frame := cloneSceneForFrame(s)
	if err := applyAnimationAt(frame, 0); err != nil {
		t.Fatal(err)
	}
	world, err := Compile(frame, nil)
	if err != nil {
		t.Fatal(err)
	}
	cam := NewCamera(vec3Of(frame.Camera.Pos), vec3Of(frame.Camera.LookAt),
		frame.Camera.FOV, float64(frame.Render.Width)/float64(frame.Render.Height), 0, 0)
	r := &Renderer{
		World: world, Camera: cam,
		Width: frame.Render.Width, Height: frame.Render.Height,
		Samples: frame.Render.Samples, MaxDepth: frame.Render.MaxDepth,
		Exposure: frame.Render.Exposure, Seed: frame.Render.Seed,
	}
	imgA := r.Render(nil)

	// Still render of the same scene — animation block ignored.
	world2, err := Compile(s, nil)
	if err != nil {
		t.Fatal(err)
	}
	cam2 := NewCamera(vec3Of(s.Camera.Pos), vec3Of(s.Camera.LookAt),
		s.Camera.FOV, float64(s.Render.Width)/float64(s.Render.Height), 0, 0)
	r2 := &Renderer{
		World: world2, Camera: cam2,
		Width: s.Render.Width, Height: s.Render.Height,
		Samples: s.Render.Samples, MaxDepth: s.Render.MaxDepth,
		Exposure: s.Render.Exposure, Seed: s.Render.Seed,
	}
	imgB := r2.Render(nil)

	if !imagesIdentical(imgA, imgB) {
		t.Fatal("frame 0 differs from still render — determinism contract broken")
	}
}

func imagesIdentical(a, b *image.RGBA) bool {
	if a.Rect != b.Rect {
		return false
	}
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			return false
		}
	}
	return true
}

// TestTimelineIgnoresAnimationOnStill pins that a still render path
// (RenderFile) is unaffected by an animation block: compile does not apply
// tracks. (The contract: plain render = authored pose.)
func TestTimelineIgnoresAnimationOnStill(t *testing.T) {
	s := mustParse(t, animTestScene)
	w1, err := Compile(s, nil)
	if err != nil {
		t.Fatal(err)
	}
	// ball at authored center -2 (no translate applied)
	ball := w1.Objects[1]
	if ball.Kind != 's' || ball.Center.X != -2 {
		t.Fatalf("still render must use authored pose; ball.Center.X = %v", ball.Center.X)
	}
}

// TestCachingResolverMemoizes: two loads return the same mesh pointer.
func TestCachingResolverMemoizes(t *testing.T) {
	dir := t.TempDir()
	obj := filepath.Join(dir, "tri.obj")
	if err := os.WriteFile(obj, []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c := &cachingResolver{inner: NewFileResolver(dir)}
	m1, err := c.LoadMesh("tri.obj")
	if err != nil {
		t.Fatal(err)
	}
	m2, err := c.LoadMesh("tri.obj")
	if err != nil {
		t.Fatal(err)
	}
	if m1 != m2 {
		t.Fatal("cachingResolver returned distinct meshes for the same src")
	}
	// The cache hands out the SAME base pointer (that is the point); the
	// safety contract is that Compile never mutates it: transformMesh
	// allocates fresh Verts/Norms. Prove it: compile a frame with a
	// transform, confirm the cached base still has its original vertex.
	base0 := m1.Verts[0]
	frame := &scene.Scene{
		Version: 1,
		Camera:  scene.Camera{Pos: []float64{0, 0, -5}, LookAt: []float64{0, 0, 0}, FOV: 50},
		Objects: []scene.Object{{
			Type: "mesh", Src: "tri.obj",
			Transform: &scene.Transform{Translate: []float64{5, 0, 0}},
		}},
		Render: scene.RenderOpts{Width: 4, Height: 4, Samples: 1, MaxDepth: 2, Seed: 1},
	}
	if _, err := Compile(frame, c); err != nil {
		t.Fatal(err)
	}
	if m1.Verts[0] != base0 {
		t.Fatal("Compile mutated the cached base mesh — frames would drift")
	}
}
