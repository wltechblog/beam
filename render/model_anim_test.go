package render

import (
	"testing"

	"github.com/wltechblog/beam/scene"
)

// TestTimelineModelCompilesPerFrame pins the anim-path model import: the
// animation path compiles through cachingResolver, and until it gained
// LoadModel (mirroring the LoadTexture fix), animated scenes with model
// objects failed at frame 0 — the wrapper is the resolver Compile sees.
func TestTimelineModelCompilesPerFrame(t *testing.T) {
	glb := buildMiniGLB(t, t.TempDir())

	scJSON := `{
		"version": 1,
		"camera": {"pos": [0, 1.5, -4], "lookAt": [0, 0.5, 0], "fov": 40},
		"environment": {"color": [0.1, 0.1, 0.15], "ambient": 0.2},
		"objects": [
			{"type": "model", "src": MODELGLB, "name": "hero", "scaleHint": 1.5}
		],
		"materials": {"floor": {"type": "matte", "color": [0.5, 0.5, 0.55]}},
		"lights": [{"type": "point", "name": "key", "pos": [0, 4, -3], "intensity": 1.0}],
		"animation": {
			"duration": 2,
			"fps": 2,
			"tracks": [
				{"target": "hero", "property": "translate",
				 "keys": [{"t": 0, "value": [0, 0, 0]}, {"t": 2, "value": [0, 0, 1]}]},
				{"target": "camera", "property": "pos",
				 "keys": [{"t": 0, "value": [0, 1.5, -4]}, {"t": 2, "value": [0, 1.5, -2]}]}
			]
		},
		"render": {"width": 48, "height": 32, "samples": 1, "maxDepth": 2, "seed": 3}
	}`
	scJSON = replaceJSONPlaceholders(scJSON, glb)

	s, err := scene.Parse([]byte(scJSON))
	if err != nil {
		t.Fatal(err)
	}

	// The anim path's resolver wrapper — this is the surface under test.
	resolver := NewFileResolver(t.TempDir() + "/nonexistent-scene-dir")
	bd := &cachingResolver{inner: resolver}

	for _, ft := range []float64{0, 0.5, 1.0, 1.5, 2.0} {
		frame := cloneSceneForFrame(s)
		if err := applyAnimationAt(frame, ft); err != nil {
			t.Fatalf("t=%v: %v", ft, err)
		}
		world, err := Compile(frame, bd)
		if err != nil {
			t.Fatalf("t=%v compile through cachingResolver: %v", ft, ft)
		}
		if len(world.Objects) == 0 {
			t.Fatalf("t=%v: model produced no objects", ft)
		}
	}
}

// replaceJSONPlaceholders substitutes src paths into the scene JSON —
// temp dirs differ per run.
func replaceJSONPlaceholders(js, glbPath string) string {
	out := make([]byte, 0, len(js))
	replace := map[string]string{"MODELGLB": glbPath}
	i := 0
	for i < len(js) {
		matched := false
		for k, v := range replace {
			if len(js)-i >= len(k) && js[i:i+len(k)] == k {
				out = append(out, '"')
				out = append(out, v...)
				out = append(out, '"')
				i += len(k)
				matched = true
				break
			}
		}
		if !matched {
			out = append(out, js[i])
			i++
		}
	}
	return string(out)
}
