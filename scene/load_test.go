package scene

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeScene(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "s.json")
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

func errPaths(errs []ValidationErr) []string {
	out := make([]string, len(errs))
	for i, e := range errs {
		out[i] = e.Path
	}
	return out
}

func hasPath(errs []ValidationErr, path string) bool {
	for _, e := range errs {
		if e.Path == path {
			return true
		}
	}
	return false
}

const minimalValid = `{
  "version": 1,
  "camera": {"pos": [0,1,-5], "lookAt": [0,0.8,0], "fov": 50},
  "materials": {"m": {"type": "matte", "color": [1,0,0]}},
  "objects": [{"type": "sphere", "center": [0,1,0], "radius": 1, "material": "m"}],
  "lights": [{"type": "point", "pos": [2,3,-2], "intensity": 1}],
  "render": {"width": 64, "height": 48}
}`

func TestLoadValidScene(t *testing.T) {
	s, err := Load(writeScene(t, minimalValid))
	if err != nil {
		t.Fatalf("valid scene rejected: %v", err)
	}
	if s.Version != 1 || len(s.Objects) != 1 || s.Render.Width != 64 {
		t.Errorf("scene parsed wrong: %+v", s)
	}
}

func TestLoadAppliesDefaults(t *testing.T) {
	s, err := Load(writeScene(t, minimalValid))
	if err != nil {
		t.Fatal(err)
	}
	if s.Render.Samples != 1 || s.Render.MaxDepth != 5 || s.Render.Exposure != 1 {
		t.Errorf("render defaults not applied: %+v", s.Render)
	}
	if s.Environment == nil {
		t.Errorf("environment default not applied")
	}
}

func TestLoadRejectsBadJSON(t *testing.T) {
	_, err := Load(writeScene(t, `{not json`))
	if err == nil || !strings.Contains(err.Error(), "parse scene JSON") {
		t.Errorf("expected parse error, got %v", err)
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load("/nonexistent/scene.json"); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestValidateVersion(t *testing.T) {
	s := Defaults()
	s.Version = 2
	errs := s.Validate()
	if !hasPath(errs, "/version") {
		t.Errorf("expected /version error, got %v", errPaths(errs))
	}
}

func TestValidateCamera(t *testing.T) {
	s := Defaults()
	s.Camera.FOV = 0
	s.Camera.Pos = []float64{1, 2} // not vec3
	s.Camera.LookAt = nil
	errs := s.Validate()
	if !hasPath(errs, "/camera/fov") || !hasPath(errs, "/camera/pos") || !hasPath(errs, "/camera/lookAt") {
		t.Errorf("camera errors: %v", errPaths(errs))
	}
	s.Camera.FOV = 180
	if errs := s.Validate(); !hasPath(errs, "/camera/fov") {
		t.Errorf("fov=180 should fail: %v", errPaths(errs))
	}
}

func TestValidateObjects(t *testing.T) {
	s := Defaults()
	s.Objects = []Object{
		{Type: "sphere", Center: []float64{0, 0, 0}, Radius: -1},
		{Type: "wedge"},
		{Type: "box", Min: []float64{1, 1, 1}, Max: []float64{0, 0, 0}},
		{Type: "text", Text: "hi", Size: 1, Depth: 1, Material: "nope"},
	}
	errs := s.Validate()
	for want := range map[string]bool{
		"/objects/0/radius":   true,
		"/objects/1/type":     true,
		"/objects/2/min":      true,
		"/objects/3/material": true,
	} {
		if !hasPath(errs, want) {
			t.Errorf("missing %s in %v", want, errPaths(errs))
		}
	}
}

func TestValidateEmptyObjects(t *testing.T) {
	s := Defaults()
	if errs := s.Validate(); !hasPath(errs, "/objects") {
		t.Errorf("empty objects should error: %v", errPaths(errs))
	}
}

func TestValidateLights(t *testing.T) {
	s := Defaults()
	s.Objects = []Object{{Type: "sphere", Center: []float64{0, 0, 0}, Radius: 1}}
	s.Lights = []Light{
		{Type: "spot", Pos: []float64{0, 5, 0}, Dir: []float64{0, -1, 0}, Angle: 95},
		{Type: "directional", Dir: []float64{0, 0, 0}}, // present-but-zero: presence check only
		{Type: "glow"},
	}
	errs := s.Validate()
	for _, want := range []string{"/lights/0/angle", "/lights/2/type"} {
		if !hasPath(errs, want) {
			t.Errorf("missing %s in %v", want, errPaths(errs))
		}
	}
}

func TestValidatePatterns(t *testing.T) {
	s := Defaults()
	s.Objects = []Object{{Type: "sphere", Center: []float64{0, 0, 0}, Radius: 1}}
	s.Materials = map[string]*Material{
		"bad":  {Type: "matte", Pattern: &Pattern{Type: "checker", Colors: [][]float64{{1, 1, 1}}}},             // needs 2
		"bad2": {Type: "matte", Pattern: &Pattern{Type: "gradient", Colors: [][]float64{{0, 0, 0}, {1, 1, 1}}}}, // needs axis
		"bad3": {Type: "matte", Pattern: &Pattern{Type: "plaid"}},                                               // unknown
		"bad4": {Type: "glass", Color: []float64{1, 1, 1}},                                                      // ior missing
	}
	errs := s.Validate()
	for _, want := range []string{
		"/materials/bad/pattern", "/materials/bad2/pattern/axis",
		"/materials/bad3/pattern/type", "/materials/bad4/ior",
	} {
		if !hasPath(errs, want) {
			t.Errorf("missing %s in %v", want, errPaths(errs))
		}
	}
}

func TestValidationErrorsMessage(t *testing.T) {
	s := Defaults()
	s.Version = 9
	errs := s.Validate()
	ve := &ValidationErrors{Errs: errs}
	msg := ve.Error()
	if !strings.Contains(msg, "/version") {
		t.Errorf("error message should list paths: %q", msg)
	}
}

// ---- animation validation ----

func animSceneBase() string {
	return `{
	  "version": 1,
	  "camera": {"pos": [0,1,-5], "lookAt": [0,0.8,0], "fov": 50},
	  "lights": [{"type": "point", "name": "key", "pos": [3,5,-3], "intensity": 1}],
	  "materials": {"m": {"type": "matte", "color": [0.8,0.2,0.2]}},
	  "objects": [
	    {"type": "sphere", "name": "ball", "center": [0,0.8,0], "radius": 0.8, "material": "m"}
	  ],
	  "render": {"width": 64, "height": 64, "samples": 1, "seed": 1}
	}`
}

func withAnim(base, anim string) string {
	// splice the animation block in before the closing brace
	return base[:len(base)-1] + ",\"animation\":" + anim + "}"
}

func TestValidateAnimOK(t *testing.T) {
	raw := withAnim(animSceneBase(), `{"duration":2,"fps":12,"tracks":[
	    {"target":"ball","property":"translate","ease":"smooth",
	     "keys":[{"t":0,"value":[0,0,0]},{"t":2,"value":[3,0,0]}]},
	    {"target":"key","property":"intensity",
	     "keys":[{"t":0,"value":[1]},{"t":2,"value":[0.2]}]},
	    {"target":"camera","property":"fov",
	     "keys":[{"t":0,"value":[50]},{"t":2,"value":[35]}]}
	  ]}`)
	if _, err := Parse([]byte(raw)); err != nil {
		t.Fatalf("valid animation rejected: %v", err)
	}
}

func TestValidateAnimErrors(t *testing.T) {
	cases := []struct {
		name    string
		anim    string
		wantSub string
	}{
		{"unknown target", `{"duration":1,"tracks":[
		    {"target":"ghost","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[1,0,0]}]}]}`,
			"unknown target"},
		{"camera bad property", `{"duration":1,"tracks":[
		    {"target":"camera","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[1,0,0]}]}]}`,
			"supports properties"},
		{"light bad property", `{"duration":1,"tracks":[
		    {"target":"key","property":"rotate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[0,90,0]}]}]}`,
			"supports properties"},
		{"unsorted keys", `{"duration":2,"tracks":[
		    {"target":"ball","property":"translate","keys":[{"t":1,"value":[0,0,0]},{"t":0.5,"value":[1,0,0]}]}]}`,
			"strictly increasing"},
		{"too few keys", `{"duration":1,"tracks":[
		    {"target":"ball","property":"translate","keys":[{"t":0,"value":[0,0,0]}]}]}`,
			"at least 2"},
		{"bad arity vec", `{"duration":1,"tracks":[
		    {"target":"ball","property":"translate","keys":[{"t":0,"value":[0]},{"t":1,"value":[1,0,0]}]}]}`,
			"3 value component"},
		{"bad arity scalar", `{"duration":1,"tracks":[
		    {"target":"key","property":"intensity","keys":[{"t":0,"value":[1,1]},{"t":1,"value":[0.5]}]}]}`,
			"1 value component"},
		{"keys exceed duration", `{"duration":1,"tracks":[
		    {"target":"ball","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":2,"value":[1,0,0]}]}]}`,
			"exceeds animation duration"},
		{"duplicate track", `{"duration":1,"tracks":[
		    {"target":"ball","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[1,0,0]}]},
		    {"target":"ball","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[2,0,0]}]}]}`,
			"already animated"},
		{"bad ease", `{"duration":1,"tracks":[
		    {"target":"ball","property":"translate","ease":"bounce",
		     "keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[1,0,0]}]}]}`,
			"easeIn | easeOut"},
		{"zero duration", `{"duration":0,"tracks":[
		    {"target":"ball","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[1,0,0]}]}]}`,
			"duration"},
		{"unnamed object target", `{"duration":1,"tracks":[
		    {"target":"ball2","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[1,0,0]}]}]}`,
			"unknown target"},
	}
	for _, c := range cases {
		raw := withAnim(animSceneBase(), c.anim)
		_, err := Parse([]byte(raw))
		if err == nil {
			t.Errorf("%s: expected error containing %q, got nil", c.name, c.wantSub)
			continue
		}
		if !strings.Contains(err.Error(), c.wantSub) {
			t.Errorf("%s: error %q does not contain %q", c.name, err.Error(), c.wantSub)
		}
	}
}

func TestValidateAnimDuplicateNames(t *testing.T) {
	base := `{
	  "version": 1,
	  "camera": {"pos": [0,1,-5], "lookAt": [0,0.8,0], "fov": 50},
	  "lights": [],
	  "materials": {"m": {"type": "matte", "color": [0.8,0.2,0.2]}},
	  "objects": [
	    {"type": "sphere", "name": "dup", "center": [0,0.8,0], "radius": 0.8, "material": "m"},
	    {"type": "sphere", "name": "dup", "center": [2,0.8,0], "radius": 0.8, "material": "m"}
	  ],
	  "render": {"width": 64, "height": 64, "samples": 1, "seed": 1}
	}`
	anim := `{"duration":1,"tracks":[
	    {"target":"dup","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":1,"value":[1,0,0]}]}]}`
	raw := base[:len(base)-1] + ",\"animation\":" + anim + "}"
	_, err := Parse([]byte(raw))
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("expected duplicate-name error, got %v", err)
	}
}

func TestValidateAnimDefaults(t *testing.T) {
	raw := withAnim(animSceneBase(), `{"duration":2,"tracks":[
	    {"target":"ball","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":2,"value":[1,0,0]}]}]}`)
	s, err := Parse([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if s.Animation.FPS != 30 {
		t.Fatalf("fps default = %d, want 30", s.Animation.FPS)
	}
}
