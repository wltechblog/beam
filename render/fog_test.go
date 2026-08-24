package render

import (
	"math"
	"testing"

	"github.com/wltechblog/beam/geom"
)

// numericOD integrates fog density numerically for cross-checking the
// closed form in opticalDepth.
func numericOD(f *FogDef, y0, dy, dist float64, steps int) float64 {
	if dist <= 0 {
		return 0
	}
	if dist > 1e6 {
		dist = 1e6
	}
	h := dist / float64(steps)
	total := 0.0
	for i := 0; i < steps; i++ {
		t := (float64(i) + 0.5) * h
		y := y0 + dy*t
		if y <= f.Alt {
			total += h
		} else if f.Off > 0 {
			total += h * math.Exp(-(y-f.Alt)/f.Off)
		}
	}
	return total
}

func TestOpticalDepthClosedForm(t *testing.T) {
	cases := []struct {
		name     string
		alt, off float64
		y0, dy   float64
		dist     float64
		tol      float64
	}{
		{"level-below", 1.5, 0, 0.0, 0, 10, 0},      // fully dense
		{"level-above-hard", 1.5, 0, 3.0, 0, 10, 0}, // zero density
		{"level-above-exp", 1.5, 40, 3.0, 0, 10, 1e-9},
		{"ascending", 1.5, 40, 0.0, 0.2, 100, 5e-3},
		{"descending", 1.5, 40, 5.0, -0.2, 100, 5e-3},
		{"ascending-hard", 1.5, 0, 0.0, 0.2, 100, 5e-3},
		{"descending-hard", 1.5, 0, 5.0, -0.2, 100, 5e-3},
		{"steep-up", 1.5, 10, 0.5, 2.0, 60, 5e-3},
		{"steep-down", 1.5, 10, 30.0, -2.0, 60, 5e-3},
		{"never-cross-up", 1.5, 40, 2.0, 0.01, 50, 5e-3},
		{"never-cross-down", 1.5, 40, 2.0, -0.01, 50, 5e-3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := &FogDef{Alt: c.alt, Off: c.off}
			got := opticalDepth(f, c.y0, c.dy, c.dist)
			if c.tol == 0 {
				if got != c.dist && c.name == "level-below" {
					t.Errorf("level below alt: got %v want %v", got, c.dist)
				}
				if c.name == "level-above-hard" && got != 0 {
					t.Errorf("level above hard cutoff: got %v want 0", got)
				}
				return
			}
			want := numericOD(f, c.y0, c.dy, c.dist, 200000)
			if math.Abs(got-want) > c.tol {
				t.Errorf("closed-form %v vs numeric %v (tol %v)", got, want, c.tol)
			}
		})
	}
}

func TestFogBlend(t *testing.T) {
	w := &World{
		Fog: &FogDef{Color: geom.Vec3{X: 1, Y: 1, Z: 1}, K: 1.0 / 10, Alt: 1000, Max: 1},
	}
	r := &Renderer{World: w}
	c := geom.Vec3{X: 0, Y: 0, Z: 0}
	// zero distance → no fog
	if got := r.fogBlend(c, geom.Vec3{Y: 0}, geom.Vec3{Y: 0}, 0); got != c {
		t.Errorf("dist 0: %v", got)
	}
	// full density, 10 units = one e-fold → 1-e^-1 fog
	b := r.fogBlend(c, geom.Vec3{Y: -1}, geom.Vec3{X: 1}, 10)
	want := 1 - math.Exp(-1)
	if math.Abs(b.X-want) > 1e-9 {
		t.Errorf("e-fold blend: got %v want %v", b.X, want)
	}
	// huge distance → fully fog color
	b = r.fogBlend(c, geom.Vec3{Y: -1}, geom.Vec3{X: 1}, 1e4)
	if math.Abs(b.X-1) > 1e-6 {
		t.Errorf("saturated: got %v want 1", b.X)
	}
	// MaxFog caps the blend
	w.Fog.Max = 0.5
	b = r.fogBlend(c, geom.Vec3{Y: -1}, geom.Vec3{X: 1}, 1e4)
	if math.Abs(b.X-0.5) > 1e-9 {
		t.Errorf("capped: got %v want 0.5", b.X)
	}
	// nil fog → passthrough
	r.World.Fog = nil
	if got := r.fogBlend(geom.Vec3{X: 3, Y: 4, Z: 5}, geom.Vec3{}, geom.Vec3{}, 100); got != (geom.Vec3{X: 3, Y: 4, Z: 5}) {
		t.Errorf("nil fog: %v", got)
	}
}

func TestSkyThreeStop(t *testing.T) {
	w := &World{
		SkyGradient: true,
		EnvTop:      geom.Vec3{X: 1, Y: 1, Z: 1},
		EnvMid:      geom.Vec3{X: 0.5, Y: 0.5, Z: 0.5},
		EnvBot:      geom.Vec3{X: 0, Y: 0, Z: 0},
	}
	r := &Renderer{World: w}
	up := r.envAt(geom.Vec3{Y: 1})
	dn := r.envAt(geom.Vec3{Y: -1})
	hz := r.envAt(geom.Vec3{X: 1})
	if up != (geom.Vec3{X: 1, Y: 1, Z: 1}) {
		t.Errorf("zenith: %v", up)
	}
	if dn != (geom.Vec3{X: 0, Y: 0, Z: 0}) {
		t.Errorf("below: %v", dn)
	}
	if hz != (geom.Vec3{X: 0.5, Y: 0.5, Z: 0.5}) {
		t.Errorf("horizon: %v", hz)
	}
}

func TestSkyCurve(t *testing.T) {
	w := &World{
		SkyGradient: true,
		EnvTop:      geom.Vec3{X: 0, Y: 0, Z: 4}, // deep blue zenith
		EnvMid:      geom.Vec3{X: 3, Y: 3, Z: 3}, // white horizon
	}
	r := &Renderer{World: w}

	// linear (no curve): t maps directly
	w.SkyCurve = 0
	lin := r.envAt(geom.Vec3{X: 1, Y: 0.25, Z: 0}.Norm())
	// dir.Y after Norm is slightly under 0.25; recompute expectation from dir
	d := geom.Vec3{X: 1, Y: 0.25, Z: 0}.Norm()
	exp := w.EnvMid.Add(w.EnvTop.Sub(w.EnvMid).MulS(d.Y))
	if lin != exp {
		t.Fatalf("linear: got %v want %v", lin, exp)
	}

	// curve 0.5: low elevations hold the zenith color longer
	w.SkyCurve = 0.5
	cur := r.envAt(d)
	t2 := math.Sqrt(d.Y)
	exp2 := w.EnvMid.Add(w.EnvTop.Sub(w.EnvMid).MulS(t2))
	if cur != exp2 {
		t.Fatalf("curve: got %v want %v", cur, exp2)
	}
	// bluer at low elevation = closer to zenith: R drops (top R=0), Z rises
	// (top Z=4 > mid Z=3)
	if cur.X >= lin.X || cur.Z <= lin.Z {
		t.Fatalf("curve 0.5 should keep sky bluer: cur X %v lin X %v, cur Z %v lin Z %v",
			cur.X, lin.X, cur.Z, lin.Z)
	}

	// curve=1 behaves linear
	w.SkyCurve = 1
	if got := r.envAt(d); got != lin {
		t.Fatalf("curve 1 must equal linear: got %v want %v", got, lin)
	}
}
