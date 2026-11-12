package render

import (
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// softSceneJSON builds the proven ball-over-plane scene as JSON (identical
// to the end-to-end CLI verification scene), varying only the light radius
// and shadowSamples.
func softSceneJSON(radius float64, shadowSamples int) []byte {
	return []byte(fmt.Sprintf(`{
  "version": 1,
  "render": {"width": 160, "height": 120, "samples": 2, "maxDepth": 4, "seed": 7, "shadowSamples": %d},
  "camera": {"pos": [0, 1.6, -5], "lookAt": [0, 0.6, 0], "fov": 45},
  "materials": {
    "floor": {"type": "matte", "color": [0.75, 0.73, 0.70]},
    "ball":  {"type": "matte", "color": [0.85, 0.25, 0.15]}
  },
  "objects": [
    {"type": "plane", "point": [0, 0, 0], "normal": [0, 1, 0], "material": "floor"},
    {"type": "sphere", "center": [0, 1, 0], "radius": 0.7, "material": "ball"}
  ],
  "lights": [
    {"type": "point", "pos": [2.5, 4, -2], "intensity": 14, "radius": %g}
  ],
  "environment": {"color": [0.03, 0.04, 0.06], "ambient": 0.15}
}`, shadowSamples, radius))
}

// renderSceneLums parses + renders the scene through the REAL pipeline and
// returns luminance rows sampled across the lower half (floor region).
func renderSceneLums(t *testing.T, radius float64, shadowSamples int) []int {
	t.Helper()
	sc, err := scene.Parse(softSceneJSON(radius, shadowSamples))
	if err != nil {
		t.Fatalf("scene parse: %v", err)
	}
	out := filepath.Join(t.TempDir(), fmt.Sprintf("soft-%v-%d.png", radius, shadowSamples))
	path, err := RenderScene(sc, Options{Quiet: true, Output: out})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open render output %s: %v", path, err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode png: %v", err)
	}
	b := img.Bounds()
	rows := []int{int(float64(b.Dy()) * 0.55), int(float64(b.Dy()) * 0.65), int(float64(b.Dy()) * 0.75)}
	var band []int
	for _, y := range rows {
		for x := 0; x < b.Dx(); x += 2 {
			cr, cg, cb, _ := img.At(x, y).RGBA()
			lum := (0.299*float64(cr) + 0.587*float64(cg) + 0.114*float64(cb)) / 257.0
			band = append(band, int(lum+0.5))
		}
	}
	mn, mx := band[0], band[0]
	for _, v := range band {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if mx-mn < 30 {
		t.Fatalf("rendered band has no light/shadow contrast (min=%d max=%d)", mn, mx)
	}
	return band
}

// midband returns the fraction of samples whose luminance lies strictly
// between 25% and 75% of the p95 — a penumbra detector. Hard shadows are
// bimodal (lit/umbra, tiny midband); penumbra samples populate the middle.
func midband(band []int) float64 {
	sorted := append([]int(nil), band...)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	p95 := sorted[len(sorted)*95/100]
	lo, hi := 0.25*float64(p95), 0.75*float64(p95)
	in := 0
	for _, v := range band {
		f := float64(v)
		if f > lo && f < hi {
			in++
		}
	}
	return float64(in) / float64(len(band))
}

// TestHardShadowPathUnchangedWithRadiusZero pins the compat contract:
// radius == 0 must produce EXACTLY the historic hard-shadow shading —
// no RNG draws, no averaging, identical bytes.
func TestHardShadowPathUnchangedWithRadiusZero(t *testing.T) {
	bandA := renderSceneLums(t, 0, 0)
	bandB := renderSceneLums(t, 0, 0) // fresh world, same seed
	for i := range bandA {
		if bandA[i] != bandB[i] {
			t.Fatalf("hard path non-deterministic or radius-sensitive at px %d: %d vs %d", i, bandA[i], bandB[i])
		}
	}
	// radius==0 must also equal a world where ShadowSamples is large (the
	// hard branch must not consult samples at all)
	bandC := renderSceneLums(t, 0, 16)
	for i := range bandA {
		if bandA[i] != bandC[i] {
			t.Fatalf("radius==0 path changed with ShadowSamples at px %d: %d vs %d", i, bandA[i], bandC[i])
		}
	}
}

// TestSoftShadowWidensMidband proves the penumbra exists: with radius > 0
// the mid-tone band between umbra and lit floor must grow vs the hard
// render of the same geometry.
func TestSoftShadowWidensMidband(t *testing.T) {
	hard := midband(renderSceneLums(t, 0, 0))
	soft := midband(renderSceneLums(t, 1.2, 4))
	if soft <= hard+0.005 {
		t.Fatalf("soft shadows did not widen the midband: hard=%.3f soft=%.3f", hard, soft)
	}
}

// TestSampleLightPointOnSphere verifies the disk-sampling math: every
// sampled point must lie exactly on the light sphere surface (within fp
// tolerance) for a spread of RNG draws.
func TestSampleLightPointOnSphere(t *testing.T) {
	l := &LightDef{
		Kind:   'p',
		Pos:    geom.Vec3{X: 2.5, Y: 4, Z: -2},
		Radius: 1.2,
	}
	p := geom.Vec3{X: 0.3, Y: 0, Z: 0.5}
	rng := NewRNG(42)
	for i := 0; i < 500; i++ {
		u1 := rng.Float64()
		u2 := rng.Float64()
		sp := sampleLightPoint(l, p, u1, u2)
		d := sp.Sub(l.Pos).Len()
		if math.Abs(d-l.Radius) > 1e-9 {
			t.Fatalf("sample %d off sphere surface: dist=%.12f want=%.12f", i, d, l.Radius)
		}
		// sample must be on the p-facing side (dot with p->light axis > 0)
		axis := l.Pos.Sub(p)
		if sp.Sub(p).Dot(axis) <= 0 {
			t.Fatalf("sample %d on far side of light", i)
		}
	}
}

// TestSoftShadowMonotonicPenumbra checks the physical scaling: a larger
// light radius must not shrink the penumbra midband (bigger source =
// wider transition, statistically).
func TestSoftShadowMonotonicPenumbra(t *testing.T) {
	small := midband(renderSceneLums(t, 0.4, 4))
	big := midband(renderSceneLums(t, 1.6, 4))
	if big+0.002 < small {
		t.Fatalf("bigger light produced narrower midband: small(r=0.4)=%.3f big(r=1.6)=%.3f", small, big)
	}
}
