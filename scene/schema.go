// Package scene defines the Beam scene description schema (versioned JSON)
// and its validation. Scenes are pure data — no scripting — which makes them
// schema-validatable and trivially generatable by LLMs.
package scene

import "fmt"

// Scene is the root document.
type Scene struct {
	Version     int                  `json:"version"`
	Camera      Camera               `json:"camera"`
	Materials   map[string]*Material `json:"materials"`
	Objects     []Object             `json:"objects"`
	Lights      []Light              `json:"lights"`
	Environment *Environment         `json:"environment,omitempty"`
	Render      RenderOpts           `json:"render"`
}

// Camera is a perspective camera with optional thin-lens depth of field.
type Camera struct {
	Pos       []float64 `json:"pos"`
	LookAt    []float64 `json:"lookAt"`
	Up        []float64 `json:"up,omitempty"`
	FOV       float64   `json:"fov"`                 // horizontal field of view, degrees
	Aperture  float64   `json:"aperture,omitempty"`  // lens radius; 0 = pinhole
	FocusDist float64   `json:"focusDist,omitempty"` // 0 = autofocus on lookAt
}

// Pattern modifies material albedo as a function of world position.
// Built-in functions only — data, not shaders.
type Pattern struct {
	Type   string      `json:"type"` // solid | checker | gradient
	Colors [][]float64 `json:"colors"`
	Scale  float64     `json:"scale,omitempty"`
	Axis   string      `json:"axis,omitempty"` // gradient: x | y | z
	From   float64     `json:"from,omitempty"` // gradient range start
	To     float64     `json:"to,omitempty"`   // gradient range end
}

// Material describes surface response.
type Material struct {
	Type      string    `json:"type"` // matte | metal | glass | emissive
	Color     []float64 `json:"color,omitempty"`
	Pattern   *Pattern  `json:"pattern,omitempty"` // overrides Color for albedo
	Roughness float64   `json:"roughness,omitempty"`
	IOR       float64   `json:"ior,omitempty"`   // glass index of refraction
	Power     float64   `json:"power,omitempty"` // emissive multiplier
}

// Transform is translate * rotate(ZYX, degrees) * scale.
type Transform struct {
	Translate []float64 `json:"translate,omitempty"`
	Rotate    []float64 `json:"rotate,omitempty"` // degrees, XYZ euler
	Scale     []float64 `json:"scale,omitempty"`
}

// Object is a tagged union of shape kinds.
type Object struct {
	Type      string     `json:"type"` // sphere | plane | box | mesh | text
	Material  string     `json:"material,omitempty"`
	Transform *Transform `json:"transform,omitempty"`

	// sphere / plane / box
	Center []float64 `json:"center,omitempty"`
	Radius float64   `json:"radius,omitempty"`
	Point  []float64 `json:"point,omitempty"`
	Normal []float64 `json:"normal,omitempty"`
	Min    []float64 `json:"min,omitempty"`
	Max    []float64 `json:"max,omitempty"`

	// mesh
	Src string `json:"src,omitempty"` // OBJ / STL / GLB path, relative to scene

	// text
	Text    string  `json:"text,omitempty"`
	Size    float64 `json:"size,omitempty"`    // em height in world units
	Depth   float64 `json:"depth,omitempty"`   // extrusion depth
	Font    string  `json:"font,omitempty"`    // "regular" | "bold" | path to .ttf
	Spacing float64 `json:"spacing,omitempty"` // extra letter spacing (em units)
}

// Light is a tagged union of light kinds.
type Light struct {
	Type      string    `json:"type"` // point | directional | spot
	Pos       []float64 `json:"pos,omitempty"`
	Dir       []float64 `json:"dir,omitempty"`
	Color     []float64 `json:"color,omitempty"`
	Intensity float64   `json:"intensity,omitempty"`
	Angle     float64   `json:"angle,omitempty"` // spot half-angle, degrees
}

// Environment controls background and ambient light.
type Environment struct {
	Color   []float64 `json:"color"`
	Ambient float64   `json:"ambient,omitempty"`
	// Sky is an optional gradient sampled by miss rays, giving glass and
	// metal something believable to show. Two entries: [zenith, bottom].
	// Three entries: [zenith, horizon, belowHorizon] — the horizon stop
	// sits at dir.Y == 0.
	Sky [3][]float64 `json:"sky,omitempty"`
	// Fog blends distant surfaces toward Fog.Color with height falloff
	// (POV fog_type 2 style): fully dense below Altitude, falling off
	// exponentially above with scale Offset.
	Fog *Fog `json:"fog,omitempty"`
	// SkyCurve bends the gradient: t' = t^SkyCurve. 1 = linear;
	// <1 keeps the zenith color closer to the horizon (POV-style skies
	// that stay blue low down); >1 concentrates change near the horizon.
	SkyCurve float64 `json:"skyCurve,omitempty"`
}

// Fog describes exponential distance fog with optional ground-fog height
// falloff. Transmittance = exp(-k * integral of density along the ray),
// where k = 1/Distance and density = 1 below Altitude, exp(-(h-alt)/Offset)
// above.
type Fog struct {
	Color    []float64 `json:"color"`
	Distance float64   `json:"distance"`           // e-folding distance at full density
	Altitude float64   `json:"altitude,omitempty"` // dense below this height
	Offset   float64   `json:"offset,omitempty"`   // falloff scale above Altitude (0 = hard cutoff)
	MaxFog   float64   `json:"maxFog,omitempty"`   // blend cap, 0..1 (POV rgbt transparency)
}

// RenderOpts controls output.
type RenderOpts struct {
	Width    int     `json:"width,omitempty"`
	Height   int     `json:"height,omitempty"`
	Samples  int     `json:"samples,omitempty"`  // AA grid: samples x samples
	MaxDepth int     `json:"maxDepth,omitempty"` // ray recursion depth
	Seed     int64   `json:"seed,omitempty"`
	Exposure float64 `json:"exposure,omitempty"`
	Output   string  `json:"output,omitempty"`
}

// Default material used when an object references a missing material.
const DefaultMaterial = "__default"

// Defaults returns a Scene with all defaults populated.
func Defaults() *Scene {
	return &Scene{
		Version:   1,
		Camera:    Camera{Pos: []float64{0, 1, -5}, LookAt: []float64{0, 0.8, 0}, FOV: 50},
		Materials: map[string]*Material{},
		Render:    RenderOpts{Width: 800, Height: 600, Samples: 1, MaxDepth: 5, Seed: 42, Exposure: 1},
	}
}

// ApplyDefaults fills unset fields with defaults in place.
func (s *Scene) ApplyDefaults() {
	if len(s.Camera.Pos) == 0 {
		s.Camera.Pos = []float64{0, 1, -5}
	}
	if len(s.Camera.LookAt) == 0 {
		s.Camera.LookAt = []float64{0, 0.8, 0}
	}
	if s.Camera.FOV == 0 {
		s.Camera.FOV = 50
	}
	if s.Render.Width == 0 {
		s.Render.Width = 800
	}
	if s.Render.Height == 0 {
		s.Render.Height = 600
	}
	if s.Render.Samples == 0 {
		s.Render.Samples = 1
	}
	if s.Render.MaxDepth == 0 {
		s.Render.MaxDepth = 5
	}
	if s.Render.Exposure == 0 {
		s.Render.Exposure = 1
	}
	if s.Render.Seed == 0 {
		s.Render.Seed = 42
	}
	if s.Environment == nil {
		s.Environment = &Environment{Color: []float64{0.04, 0.05, 0.08}, Ambient: 0.2}
	}
}

// ValidationErr is one schema error with a JSON-pointer-ish path.
type ValidationErr struct {
	Path string `json:"path"`
	Msg  string `json:"msg"`
}

func (e *ValidationErr) Error() string { return fmt.Sprintf("%s: %s", e.Path, e.Msg) }

// ValidationErrors collects all schema problems (never fail on the first —
// agents self-correct faster from a complete list).
type ValidationErrors struct {
	Errs []ValidationErr
}

func (v *ValidationErrors) Error() string {
	msg := "scene validation failed:"
	for i, e := range v.Errs {
		if i >= 8 {
			msg += fmt.Sprintf(" (and %d more)", len(v.Errs)-i)
			break
		}
		msg += fmt.Sprintf("\n  %s: %s", e.Path, e.Msg)
	}
	return msg
}
