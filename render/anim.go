package render

import (
	"fmt"
	"math"

	"github.com/wltechblog/beam/scene"
)

// applyAnimationAt samples every track at time t (seconds) and writes the
// results into sc in place. The scene is mutated: camera fields, object
// transforms, light fields. Scenarios:
//
//   - camera properties overwrite Camera fields directly
//   - object transform tracks COMPOSE with the object's own transform:
//     the track's value replaces that channel of the transform (a track
//     only animating rotate leaves translate/scale at their authored
//     values), so "spin in place while translating" = one rotate track +
//     one translate track, exactly as authored
//   - light tracks overwrite Pos / Color / Intensity directly
//
// Tracks were validated at Parse time; this function trusts them (keys
// sorted, arities correct, targets exist).
func applyAnimationAt(sc *scene.Scene, t float64) error {
	a := sc.Animation
	if a == nil {
		return nil
	}

	// Index named targets once per frame. Scenes are small; a linear scan
	// per track is fine and avoids map allocation churn.
	objByName := map[string]*scene.Object{}
	for i := range sc.Objects {
		if n := sc.Objects[i].Name; n != "" {
			objByName[n] = &sc.Objects[i]
		}
	}
	lightByName := map[string]*scene.Light{}
	for i := range sc.Lights {
		if n := sc.Lights[i].Name; n != "" {
			lightByName[n] = &sc.Lights[i]
		}
	}

	for i := range a.Tracks {
		tr := &a.Tracks[i]
		v, ok := sampleTrack(tr, t)
		if !ok {
			continue // property unanimated before first / after last key: keep authored value
		}
		switch {
		case tr.Target == "camera":
			applyCameraProp(&sc.Camera, tr.Property, v)
		case objByName[tr.Target] != nil:
			applyObjectProp(objByName[tr.Target], tr.Property, v)
		case lightByName[tr.Target] != nil:
			if err := applyLightProp(lightByName[tr.Target], tr.Property, v); err != nil {
				return fmt.Errorf("track %d (%s/%s): %w", i, tr.Target, tr.Property, err)
			}
		}
	}
	return nil
}

// sampleTrack evaluates a track at time t. Returns ok=false when t is
// outside the track's key span (the authored property value is kept —
// tracks do not extrapolate).
func sampleTrack(tr *scene.Track, t float64) ([]float64, bool) {
	keys := tr.Keys
	if t < keys[0].T {
		return nil, false
	}
	last := keys[len(keys)-1]
	if t > last.T {
		return nil, false
	}
	if len(keys[0].Value) == 0 {
		return nil, false
	}
	out := make([]float64, len(keys[0].Value))
	// find segment [j, j+1] with keys[j].T <= t <= keys[j+1].T
	j := 0
	for k := 1; k < len(keys); k++ {
		if keys[k].T <= t {
			j = k
		} else {
			break
		}
	}
	if keys[j].T == t || j == len(keys)-1 {
		copy(out, keys[j].Value)
		return out, true
	}
	k0, k1 := &keys[j], &keys[j+1]
	span := k1.T - k0.T
	if span <= 0 {
		copy(out, k0.Value)
		return out, true
	}
	u := (t - k0.T) / span
	u = ease(tr.Ease, u)
	for c := range out {
		out[c] = k0.Value[c] + u*(k1.Value[c]-k0.Value[c])
	}
	return out, true
}

// ease maps linear segment progress u in [0,1] to eased progress.
func ease(kind string, u float64) float64 {
	switch kind {
	case "smooth": // smoothstep: 3u² − 2u³, C¹ continuous
		return u * u * (3 - 2*u)
	case "easeIn": // quadratic accelerate
		return u * u
	case "easeOut": // quadratic decelerate
		return 1 - (1-u)*(1-u)
	default: // "" | "linear"
		return u
	}
}

func applyCameraProp(c *scene.Camera, prop string, v []float64) {
	switch prop {
	case "pos":
		copy(c.Pos, v)
	case "lookAt":
		copy(c.LookAt, v)
	case "fov":
		c.FOV = v[0]
	case "aperture":
		c.Aperture = v[0]
	}
}

func applyObjectProp(o *scene.Object, prop string, v []float64) {
	if o.Transform == nil {
		o.Transform = &scene.Transform{}
	}
	tr := o.Transform
	switch prop {
	case "translate":
		tr.Translate = appendVec3(tr.Translate, v)
	case "rotate":
		tr.Rotate = appendVec3(tr.Rotate, v)
	case "scale":
		tr.Scale = appendVec3(tr.Scale, v)
	}
}

func appendVec3(dst []float64, v []float64) []float64 {
	if len(dst) != 3 {
		dst = make([]float64, 3)
	}
	copy(dst, v)
	return dst
}

func applyLightProp(l *scene.Light, prop string, v []float64) error {
	switch prop {
	case "pos":
		if len(l.Pos) != 3 {
			l.Pos = make([]float64, 3)
		}
		copy(l.Pos, v)
	case "color":
		if len(l.Color) != 3 {
			l.Color = make([]float64, 3)
		}
		copy(l.Color, v)
	case "intensity":
		if v[0] < 0 {
			return fmt.Errorf("intensity went negative (%v) — clamp the keyframes", v[0])
		}
		l.Intensity = v[0]
	}
	return nil
}

// animFrameCount returns the number of frames the timeline renders:
// ceil(duration * fps). A 4s animation at 30fps = 120 frames; the last
// frame sits at t just under duration.
func animFrameCount(a *scene.Animation) int {
	n := int(math.Ceil(a.Duration * float64(a.FPS)))
	if n < 1 {
		n = 1
	}
	return n
}

// animTimeOf returns the sample time of frame i.
func animTimeOf(a *scene.Animation, i int) float64 {
	return float64(i) / float64(a.FPS)
}
