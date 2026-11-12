package render

import (
	"os"
	"os/exec"
	"testing"

	"github.com/wltechblog/beam/scene"
)

func TestDeriveOrbitFrame0MatchesCamera(t *testing.T) {
	s := testOrbitScene()
	circ := deriveOrbit(s)

	pos := vec3Of(s.Camera.Pos)
	p0 := circ.posAt(circ.az0)
	if p0.Sub(pos).Len() > 1e-12 {
		t.Fatalf("frame-0 orbit position %v != camera pos %v", p0, pos)
	}

	// A quarter turn must move the camera a sensible distance: chord of
	// 90° on the orbit circle = radius * sqrt(2).
	p90 := circ.posAt(circ.az0 + 1.5707963267948966)
	chord := p90.Sub(pos).Len()
	want := circ.radius * 1.4142135623730951
	if chord < want*0.99 || chord > want*1.01 {
		t.Fatalf("90° chord %v, want ~%v (radius %v)", chord, want, circ.radius)
	}

	// Height stays constant across the sweep.
	if p90.Y != pos.Y {
		t.Fatalf("orbit changed height: %v vs %v", p90.Y, pos.Y)
	}
}

func TestOrbitVideo(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
	s := testOrbitScene()
	out := t.TempDir() + "/orbit.mp4"
	path, err := RenderOrbit(s, Options{Quiet: true, Dir: "."}, OrbitOpts{Frames: 4, FPS: 4})
	_ = out
	if err != nil {
		t.Fatalf("RenderOrbit: %v", err)
	}
	defer os.Remove(path)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("video missing: %v", err)
	}
	fi, err := os.Stat(path)
	if err != nil || fi.Size() < 1000 {
		t.Fatalf("video suspiciously small: %v bytes", fi.Size())
	}
}

func TestOrbitRejectsFewerThan2Frames(t *testing.T) {
	if _, err := RenderOrbit(testOrbitScene(), Options{Quiet: true, Dir: "."}, OrbitOpts{Frames: 1}); err == nil {
		t.Fatal("expected error for 1-frame orbit")
	}
}

func testOrbitScene() *scene.Scene {
	// two spheres side by side so the orbit has something to look at
	return &scene.Scene{
		Version: 1,
		Camera: scene.Camera{
			Pos: []float64{0, 2, -8}, LookAt: []float64{0, 1, 0}, FOV: 45,
		},
		Objects: nil,
		Render:  scene.RenderOpts{Width: 64, Height: 48, Samples: 1, MaxDepth: 2, Seed: 7},
	}
}
