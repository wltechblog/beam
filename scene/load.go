package scene

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

func normalize3(v []float64) []float64 {
	l := math.Hypot(math.Hypot(v[0], v[1]), v[2])
	if l < 1e-12 {
		return []float64{0, 0, 1}
	}
	return []float64{v[0] / l, v[1] / l, v[2] / l}
}

// Load reads and validates a scene file. A scene that fails validation is
// never rendered — agents get the full error list at once.
func Load(path string) (*Scene, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read scene: %w", err)
	}
	var s Scene
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("parse scene JSON: %w", err)
	}
	if errs := s.Validate(); len(errs) > 0 {
		return nil, &ValidationErrors{Errs: errs}
	}
	s.ApplyDefaults()
	return &s, nil
}

// Validate returns all schema problems at once (empty = valid).
func (s *Scene) Validate() []ValidationErr {
	var errs []ValidationErr
	add := func(path, msg string, args ...interface{}) {
		errs = append(errs, ValidationErr{Path: path, Msg: fmt.Sprintf(msg, args...)})
	}

	if s.Version != 1 {
		add("/version", "must be 1 (got %d)", s.Version)
	}

	// Camera
	if !v3(s.Camera.Pos) {
		add("/camera/pos", "must be [x,y,z]")
	}
	if !v3(s.Camera.LookAt) {
		add("/camera/lookAt", "must be [x,y,z]")
	}
	if s.Camera.FOV <= 0 || s.Camera.FOV >= 180 {
		add("/camera/fov", "must be in (0,180) degrees")
	}
	if s.Camera.Aperture < 0 {
		add("/camera/aperture", "must be >= 0")
	}

	// Materials
	for name, m := range s.Materials {
		p := "/materials/" + name
		switch m.Type {
		case "matte", "metal", "glass", "emissive":
		case "":
			add(p+"/type", "required (matte | metal | glass | emissive)")
		default:
			add(p+"/type", "unknown material type %q", m.Type)
		}
		if m.Type == "glass" && m.IOR <= 0 {
			add(p+"/ior", "glass requires ior > 0 (default 1.5)")
		}
		if m.Pattern != nil {
			if err := m.Pattern.validate(p + "/pattern"); err != nil {
				errs = append(errs, *err)
			}
		}
	}

	// Objects
	if len(s.Objects) == 0 {
		add("/objects", "scene needs at least one object")
	}
	for i := range s.Objects {
		o := &s.Objects[i]
		p := fmt.Sprintf("/objects/%d", i)
		if o.Material != "" {
			if _, ok := s.Materials[o.Material]; !ok {
				add(p+"/material", "references undefined material %q", o.Material)
			}
		}
		switch o.Type {
		case "sphere":
			if !v3(o.Center) {
				add(p+"/center", "sphere requires center [x,y,z]")
			}
			if o.Radius <= 0 {
				add(p+"/radius", "sphere requires radius > 0")
			}
		case "plane":
			if !v3(o.Point) {
				add(p+"/point", "plane requires point [x,y,z]")
			}
			if !v3(o.Normal) {
				add(p+"/normal", "plane requires normal [x,y,z]")
			}
		case "box":
			if !v3(o.Min) || !v3(o.Max) {
				add(p+"/min", "box requires min and max [x,y,z]")
			} else {
				for a := 0; a < 3; a++ {
					if o.Min[a] >= o.Max[a] {
						add(p+"/min", "box min[%d] must be < max[%d]", a, a)
					}
				}
			}
		case "mesh":
			if o.Src == "" {
				add(p+"/src", "mesh requires src (OBJ/STL/GLB path)")
			}
		case "text":
			if o.Text == "" {
				add(p+"/text", "text requires text string")
			}
			if o.Size <= 0 {
				add(p+"/size", "text requires size > 0 (em height)")
			}
			if o.Depth <= 0 {
				add(p+"/depth", "text requires depth > 0 (extrusion)")
			}
		case "":
			add(p+"/type", "required (sphere | plane | box | mesh | text)")
		default:
			add(p+"/type", "unknown object type %q", o.Type)
		}
		if o.Transform != nil {
			for _, part := range [][2]interface{}{{"translate", o.Transform.Translate}, {"scale", o.Transform.Scale}, {"rotate", o.Transform.Rotate}} {
				name, vec := part[0].(string), part[1].([]float64)
				if len(vec) != 0 && !v3(vec) {
					add(p+"/transform/"+name, "must be [x,y,z]")
				}
			}
		}
	}

	// Lights
	for i := range s.Lights {
		l := &s.Lights[i]
		p := fmt.Sprintf("/lights/%d", i)
		switch l.Type {
		case "point":
			if !v3(l.Pos) {
				add(p+"/pos", "point light requires pos [x,y,z]")
			}
		case "directional":
			if !v3(l.Dir) {
				add(p+"/dir", "directional light requires dir [x,y,z]")
			}
		case "spot":
			if !v3(l.Pos) || !v3(l.Dir) {
				add(p+"/pos", "spot light requires pos and dir [x,y,z]")
			}
			if l.Angle <= 0 || l.Angle >= 90 {
				add(p+"/angle", "spot angle must be in (0,90) degrees")
			}
		case "":
			add(p+"/type", "required (point | directional | spot)")
		default:
			add(p+"/type", "unknown light type %q", l.Type)
		}
		if l.Intensity < 0 {
			add(p+"/intensity", "must be >= 0")
		}
	}

	// Environment
	if s.Environment != nil {
		if !v3(s.Environment.Color) {
			add("/environment/color", "must be [r,g,b]")
		}
		if s.Environment.Ambient < 0 {
			add("/environment/ambient", "must be >= 0")
		}
	}

	// Render
	if s.Render.Width < 0 || s.Render.Width > 8192 {
		add("/render/width", "must be in [0,8192] (0 = default 800)")
	}
	if s.Render.Height < 0 || s.Render.Height > 8192 {
		add("/render/height", "must be in [0,8192] (0 = default 600)")
	}
	if s.Render.Samples < 0 || s.Render.Samples > 16 {
		add("/render/samples", "must be in [0,16] (0 = default 1; grid = samples²)")
	}
	if s.Render.MaxDepth < 0 || s.Render.MaxDepth > 20 {
		add("/render/maxDepth", "must be in [0,20]")
	}
	if s.Render.Exposure < 0 {
		add("/render/exposure", "must be >= 0")
	}

	return errs
}

func v3(v []float64) bool { return len(v) == 3 }

func (p *Pattern) validate(path string) *ValidationErr {
	switch p.Type {
	case "solid", "checker":
		if len(p.Colors) != 2 && p.Type == "checker" {
			return &ValidationErr{path, "checker requires exactly 2 colors"}
		}
		for i, c := range p.Colors {
			if len(c) != 3 {
				return &ValidationErr{fmt.Sprintf("%s/colors/%d", path, i), "color must be [r,g,b]"}
			}
		}
	case "gradient":
		if p.Axis == "" {
			return &ValidationErr{path + "/axis", "gradient requires axis (x|y|z)"}
		}
		if p.Axis != "x" && p.Axis != "y" && p.Axis != "z" {
			return &ValidationErr{path + "/axis", "axis must be x, y, or z"}
		}
		if len(p.Colors) != 2 {
			return &ValidationErr{path, "gradient requires exactly 2 colors"}
		}
	default:
		return &ValidationErr{path + "/type", fmt.Sprintf("unknown pattern type %q (solid | checker | gradient)", p.Type)}
	}
	return nil
}
