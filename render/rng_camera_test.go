package render

import (
	"testing"

	"github.com/wltechblog/beam/geom"
)

func TestRNGDeterministic(t *testing.T) {
	// Same seed → identical sequence.
	a := NewRNG(42)
	b := NewRNG(42)
	for i := 0; i < 1000; i++ {
		if a.Float64() != b.Float64() {
			t.Fatalf("divergence at %d", i)
		}
	}
}

func TestRNGDifferentSeeds(t *testing.T) {
	a := NewRNG(1)
	b := NewRNG(2)
	diff := 0
	for i := 0; i < 100; i++ {
		if a.Float64() != b.Float64() {
			diff++
		}
	}
	if diff == 0 {
		t.Fatal("different seeds produced identical sequences")
	}
}

func TestRNGRange(t *testing.T) {
	r := NewRNG(7)
	for i := 0; i < 100000; i++ {
		v := r.Float64()
		if v < 0 || v >= 1 {
			t.Fatalf("Float64 out of [0,1): %v", v)
		}
	}
}

func TestSeedPixelDistinct(t *testing.T) {
	// Neighbouring pixels must get different seeds (AA + DOF quality).
	s := SeedPixel(42, 0, 0)
	if s == SeedPixel(42, 1, 0) || s == SeedPixel(42, 0, 1) {
		t.Error("adjacent pixels share a seed")
	}
	if s == SeedPixel(43, 0, 0) {
		t.Error("different render seeds collide")
	}
}

func TestCameraCenterRay(t *testing.T) {
	// Camera on -Z looking at origin: center pixel ray must go +Z.
	c := NewCamera(geom.Vec3{Z: -5}, geom.Vec3{}, 50, 16.0/9.0, 0, 0)
	r := c.Ray(0.5, 0.5, 0, 0, nil)
	if d := r.Dir.Norm().Sub(geom.Vec3{Z: 1}).Len(); d >= 1e-9 {
		t.Errorf("center ray direction off-axis by %v: %v", d, r.Dir)
	}
}

func TestCameraRayCorners(t *testing.T) {
	c := NewCamera(geom.Vec3{Z: -5}, geom.Vec3{}, 90, 1, 0, 0)
	// Beam convention (see renderTile + TextMesh): for a camera on -Z
	// looking at +Z, u = up × w points along -X, so world +X appears
	// screen-LEFT. px=0 (screen left) → +X-ish, px=1 (screen right) → -X-ish.
	r := c.Ray(1, 0.5, 0, 0, nil) // screen right
	if r.Dir.X >= 0 {
		t.Errorf("screen-right ray should point -X-ish: %v", r.Dir)
	}
	l := c.Ray(0, 0.5, 0, 0, nil) // screen left
	if l.Dir.X <= 0 {
		t.Errorf("screen-left ray should point +X-ish: %v", l.Dir)
	}
	// At fov 90, aspect 1: screen edges are ~45° off axis → |Dir.X/Dir.Z| ≈ 1.
	if mathAbsF(mathAbsF(l.Dir.X/(-l.Dir.Z))-1) > 1e-9 {
		t.Errorf("fov 90 screen-left should be ~45°: X/Z = %v", l.Dir.X/-l.Dir.Z)
	}
}

func TestCameraYFlip(t *testing.T) {
	// Camera.Ray's py is bottom-up (py=0 = bottom edge, py=1 = top edge);
	// the image-y flip happens in renderTile (py = 1 - y/H). Documented
	// here so the two-layer convention is testable at each layer.
	c := NewCamera(geom.Vec3{Z: -5}, geom.Vec3{}, 50, 1, 0, 0)
	top := c.Ray(0.5, 1, 0, 0, nil)
	bot := c.Ray(0.5, 0, 0, 0, nil)
	if top.Dir.Y <= bot.Dir.Y {
		t.Errorf("py=1 should aim higher than py=0: %v vs %v", top.Dir, bot.Dir)
	}
	if top.Dir.Y <= 0 || bot.Dir.Y >= 0 {
		t.Errorf("py extremes should straddle the horizon: %v vs %v", top.Dir, bot.Dir)
	}
}

func mathAbsF(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
