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
	// Animation is an optional keyframed timeline. When present,
	// `beam -anim` renders duration*fps frames, sampling every track at
	// each frame time. A plain still render ignores it.
	Animation *Animation `json:"animation,omitempty"`
}

// Animation is a keyframed timeline. All tracks are sampled independently
// at each frame time t (seconds); a still render ignores the block.
type Animation struct {
	Duration float64 `json:"duration"`      // seconds, > 0
	FPS      int     `json:"fps,omitempty"` // output framerate (default 30)
	Tracks   []Track `json:"tracks"`        // >= 1
}

// Track animates one property of one target over time. Targets are
// "camera", an object name, or a light name. Each (target, property)
// pair may appear at most once.
type Track struct {
	Target   string `json:"target"`         // "camera" | object name | light name
	Property string `json:"property"`       // see Validate for the per-kind list
	Keys     []Key  `json:"keys"`           // >= 2, strictly increasing t
	Ease     string `json:"ease,omitempty"` // "" | "linear" | "smooth" | "easeIn" | "easeOut" (default linear)
}

// Key is one keyframe: a time in seconds and the property's value.
// Vector properties take [x,y,z]; scalar properties take [v].
type Key struct {
	T     float64   `json:"t"`
	Value []float64 `json:"value"`
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
	Type    string    `json:"type"` // matte | metal | glass | emissive
	Color   []float64 `json:"color,omitempty"`
	Pattern *Pattern  `json:"pattern,omitempty"` // overrides Color for albedo
	// Map is an albedo texture (PNG/JPEG path relative to the scene or
	// absolute). Sampled at the mesh's UVs; multiplies Color (so a white
	// Color shows the map as-authored). Ignored on untextured geometry —
	// Color applies there.
	Map       string  `json:"map,omitempty"`
	Roughness float64 `json:"roughness,omitempty"`
	// Reflectivity raises glass Fresnel base; on metal it is the reflection
	// strength (POV `reflection` value, 0..1).
	Reflectivity float64 `json:"reflectivity,omitempty"`
	IOR          float64 `json:"ior,omitempty"`      // glass index of refraction
	Power        float64 `json:"power,omitempty"`    // emissive multiplier
	Diffuse      float64 `json:"diffuse,omitempty"`  // matte direct scale (POV finish diffuse)
	Specular     float64 `json:"specular,omitempty"` // matte highlight strength (POV finish specular)
	// Metallic=true (default) on metal: reflection tinted by pigment (conductor).
	// Metallic=false: UNTINTED reflection (POV-Ray `reflection` without `metallic`).
	Metallic bool `json:"metallic,omitempty"`
}

// Transform is translate * rotate(ZYX, degrees) * scale.
type Transform struct {
	Translate []float64 `json:"translate,omitempty"`
	Rotate    []float64 `json:"rotate,omitempty"` // degrees, XYZ euler
	Scale     []float64 `json:"scale,omitempty"`
}

// Object is a tagged union of shape kinds. Name is optional; named
// objects can be targeted by animation tracks.
type Object struct {
	Type      string     `json:"type"` // sphere | plane | box | mesh | text
	Name      string     `json:"name,omitempty"`
	Material  string     `json:"material,omitempty"`
	Transform *Transform `json:"transform,omitempty"`

	// sphere / plane / box
	Center []float64 `json:"center,omitempty"`
	Radius float64   `json:"radius,omitempty"`
	Point  []float64 `json:"point,omitempty"`
	Normal []float64 `json:"normal,omitempty"`
	Min    []float64 `json:"min,omitempty"`
	Max    []float64 `json:"max,omitempty"`

	// mesh / model
	Src string `json:"src,omitempty"` // OBJ / STL / GLTF / GLB path, relative to scene
	// Model import options (type "model"): PBR materials, textures and
	// node transforms are honored. MaterialOverride replaces every part's
	// material with a named scene material ("\_default" = plain matte).
	MaterialOverride string `json:"materialOverride,omitempty"`
	// ScaleHint rescales the imported model so its longest bounding-box
	// edge becomes this length (0 = use raw units).
	ScaleHint float64 `json:"scaleHint,omitempty"`

	// text
	Text    string  `json:"text,omitempty"`
	Size    float64 `json:"size,omitempty"`    // em height in world units
	Depth   float64 `json:"depth,omitempty"`   // extrusion depth
	Font    string  `json:"font,omitempty"`    // "regular" | "bold" | path to .ttf
	Spacing float64 `json:"spacing,omitempty"` // extra letter spacing (em units)
}

// Light is a tagged union of light kinds. Name is optional; named lights
// can be targeted by animation tracks.
type Light struct {
	Type       string    `json:"type"` // point | directional | spot
	Name       string    `json:"name,omitempty"`
	Pos        []float64 `json:"pos,omitempty"`
	Dir        []float64 `json:"dir,omitempty"`
	Color      []float64 `json:"color,omitempty"`
	Intensity  float64   `json:"intensity,omitempty"`
	Angle      float64   `json:"angle,omitempty"`   // spot half-angle, degrees
	Radius     float64   `json:"radius,omitempty"`  // spherical light radius; > 0 = soft shadows
	Falloff    string    `json:"falloff,omitempty"` // "" | "quadratic" (default) | "none"
	Shadowless bool      `json:"shadowless,omitempty"`
}

// ShadowSamples is the per-light shadow-ray count for radius lights,
// read from render.shadowSamples when the light itself doesn't set it.
// (Lives on RenderOpts so per-light override stays possible later; the
// light struct above keeps the hot fields ordered.)

// SkyStop is one entry of a piecewise sky gradient keyed by ray dir.Y
// (1 = zenith, 0 = horizon, -1 = nadir). Colors are linear radiance;
// segments interpolate linearly and ends clamp.
type SkyStop struct {
	Y     float64   `json:"y"`
	Color []float64 `json:"color"`
}

// Environment controls background and ambient light.
type Environment struct {
	Color   []float64 `json:"color"`
	Ambient float64   `json:"ambient,omitempty"`
	// AmbientColor overrides the ambient tint (default = color)
	AmbientColor []float64 `json:"ambientColor,omitempty"`
	// Sky is an optional gradient sampled by miss rays, giving glass and
	// metal something believable to show. Two entries: [zenith, bottom].
	// Three entries: [zenith, horizon, belowHorizon] — the horizon stop
	// sits at dir.Y == 0.
	Sky [3][]float64 `json:"sky,omitempty"`
	// SkyMap is a richer alternative to Sky: arbitrary piecewise stops
	// keyed by dir.Y (see SkyStop). Takes precedence over Sky when set.
	SkyMap []SkyStop `json:"skyMap,omitempty"`
	// Fog blends distant surfaces toward Fog.Color with height falloff
	// (POV fog_type 2 style): fully dense below Altitude, falling off
	// exponentially above with scale Offset.
	Fog *Fog `json:"fog,omitempty"`
	// Media is participating media with single scattering: extinction
	// (same height profile as fog) PLUS in-scattered light — shadow rays
	// from each march sample produce light shafts / god rays. When
	// present it replaces fog compositing (fog is the absorption-only
	// approximation of the same physics).
	Media *Media `json:"media,omitempty"`
	// SkyCurve bends the gradient: t' = t^SkyCurve. 1 = linear;
	// <1 keeps the zenith color closer to the horizon (POV-style skies
	// that stay blue low down); >1 concentrates change near the horizon.
	SkyCurve float64 `json:"skyCurve,omitempty"`
	// EnvMap is an equirectangular environment map (.hdr Radiance or
	// .png/.jpg LDR) that REPLACES the flat/gradient sky for miss rays
	// and reflections — real image-based lighting backgrounds. The map's
	// mean color also tints ambient when ambientColor is not set.
	EnvMap string `json:"envMap,omitempty"`
	// EnvScale multiplies env map radiance (exposure headroom for HDR
	// maps: values > 1 brighten, < 1 dim). Default 1.
	EnvScale float64 `json:"envScale,omitempty"`
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
	// Profile "atan" enables the POV-Ray fog_type 2 density function
	// exactly: density = 1/(u^2+1) with u = (y-offset)/alt (0 above
	// ground level), whose ray integral is the atan difference below.
	// "exp" (default) is the height-exponential profile described above.
	Profile string `json:"profile,omitempty"`
}

// Media describes participating media with single scattering. Extinction
// uses the same height profile as Fog (distance = e-folding at full
// density; altitude/offset/profile shape it). On top, render mediaSamples
// deterministic march steps along every camera/shading ray: each sample
// shadow-rays toward every light, and unshadowed light in-scatters,
// attenuated by transmittance to both the light and the viewer — that is
// what draws light shafts and god rays.
type Media struct {
	Color    []float64 `json:"color"`
	Distance float64   `json:"distance"` // e-folding extinction distance at full density
	// With no Altitude/Offset/Profile authored the media is UNIFORM:
	// dense everywhere (plain POV fog_type 1). Altitude > 0 opts into
	// the ground-fog height profile: dense below Altitude, exponential
	// falloff above with scale Offset (0 = hard cutoff).
	Altitude float64 `json:"altitude,omitempty"`
	Offset   float64 `json:"offset,omitempty"`
	Profile  string  `json:"profile,omitempty"` // "exp" (default) | "atan" — same profiles as fog
	Samples  int     `json:"samples,omitempty"` // march steps per ray segment (default 8, max 64)
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
	Tonemap  string  `json:"tonemap,omitempty"` // "" | "reinhard" (default) | "clip" (POV-style)

	// ShadowSamples is the per-light shadow-ray count for spherical area
	// lights (radius > 0). 0 = 1 (single jittered ray). Higher = smoother
	// penumbra at linear cost per radius light.
	ShadowSamples int `json:"shadowSamples,omitempty"`

	// MaxSeconds aborts the render (no PNG written) if it runs longer.
	// Guards against runaway scenes (huge sample counts, CPU fallbacks).
	MaxSeconds float64 `json:"maxSeconds,omitempty"`
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
	if s.Animation != nil && s.Animation.FPS == 0 {
		s.Animation.FPS = 30
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
