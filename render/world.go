package render

import (
	"math"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// geomNormal aliases geom.Vec3 for readability in sampling code.
type geomNormal = geom.Vec3

// MeshResolver loads mesh files and generates text geometry. The CLI wires
// the real importer; tests use fakes.
type MeshResolver interface {
	LoadMesh(src string) (*geom.TriangleMesh, error)
	TextMesh(text, font string, size, depth, spacing float64) (*geom.TriangleMesh, error)
}

// World is the compiled scene: objects, materials, lights, environment.
type World struct {
	Objects     []*Object3D
	Materials   []CompiledMaterial
	EnvColor    geom.Vec3
	EnvTop      geom.Vec3
	EnvMid      geom.Vec3 // optional 3rd sky stop at the horizon
	EnvBot      geom.Vec3
	SkyGradient bool
	SkyCurve    float64 // exponent on gradient t; 0 or 1 = linear
	Ambient     float64
	Lights      []LightDef
	Fog         *FogDef
}

// FogDef is compiled fog parameters (exponential + ground-fog falloff).
type FogDef struct {
	Color geom.Vec3
	K     float64 // 1/e-folding distance
	Alt   float64 // dense below this world-space height
	Off   float64 // falloff scale above Alt; 0 = hard cutoff
	Max   float64 // maximum fog blend factor
}

// Object3D is one renderable object (primitive or mesh instance).
type Object3D struct {
	Kind   byte // 'p' plane, 's' sphere, 'm' mesh
	Center geom.Vec3
	Radius float64
	PlaneP geom.Vec3
	PlaneN geom.Vec3
	Mesh   *geom.TriangleMesh
	BVH    *geom.BVH
	MatIdx int
}

// LightDef is a compiled light.
type LightDef struct {
	Kind      byte // 'p' point, 'd' directional, 's' spot
	Pos, Dir  geom.Vec3
	Color     geom.Vec3
	Intensity float64
	CosAngle  float64 // spot: cosine of half-angle
}

// CompiledMaterial holds runtime material parameters.
type CompiledMaterial struct {
	Kind      byte // 'm' matte, 'M' metal, 'g' glass, 'e' emissive
	Albedo    geom.Vec3
	Pattern   *scene.Pattern // nil = solid albedo
	Roughness float64
	IOR       float64
	Power     float64
}

func rgb(c []float64) geom.Vec3 {
	if len(c) != 3 {
		return geom.Vec3{X: 0.8, Y: 0.8, Z: 0.8}
	}
	return geom.Vec3{X: c[0], Y: c[1], Z: c[2]}
}

func rgbv(v []float64) geom.Vec3 { return rgb(v) }

func normv(v []float64) geom.Vec3 {
	if len(v) != 3 {
		return geom.Vec3{X: 0, Y: 1, Z: 0}
	}
	return geom.Vec3{X: v[0], Y: v[1], Z: v[2]}.Norm()
}

func cosDeg(deg float64) float64 { return math.Cos(deg * math.Pi / 180) }

func defaultMat() CompiledMaterial {
	return CompiledMaterial{Kind: 'm', Albedo: geom.Vec3{X: 0.8, Y: 0.8, Z: 0.8}}
}

func compileMaterial(m *scene.Material) CompiledMaterial {
	cm := CompiledMaterial{Albedo: geom.Vec3{X: 0.8, Y: 0.8, Z: 0.8}}
	if m == nil {
		cm.Kind = 'm'
		return cm
	}
	if len(m.Color) == 3 {
		cm.Albedo = rgb(m.Color)
	}
	cm.Pattern = m.Pattern
	switch m.Type {
	case "metal":
		cm.Kind = 'M'
		cm.Roughness = m.Roughness
	case "glass":
		cm.Kind = 'g'
		cm.IOR = m.IOR
		if cm.IOR == 0 {
			cm.IOR = 1.5
		}
	case "emissive":
		cm.Kind = 'e'
		cm.Power = m.Power
		if cm.Power == 0 {
			cm.Power = 1
		}
	default: // matte
		cm.Kind = 'm'
	}
	return cm
}

// AlbedoAt evaluates the pattern (if any) at a world position.
func (cm *CompiledMaterial) AlbedoAt(p geom.Vec3) geom.Vec3 {
	if cm.Pattern == nil {
		return cm.Albedo
	}
	switch cm.Pattern.Type {
	case "checker":
		sc := cm.Pattern.Scale
		if sc <= 0 {
			sc = 1
		}
		cx := math.Floor(p.X / sc)
		cy := math.Floor(p.Y / sc)
		cz := math.Floor(p.Z / sc)
		if int64(cx+cy+cz)&1 == 0 {
			return rgb(cm.Pattern.Colors[0])
		}
		return rgb(cm.Pattern.Colors[1])
	case "gradient":
		var t float64
		from, to := cm.Pattern.From, cm.Pattern.To
		if to <= from {
			to = from + 1
		}
		switch cm.Pattern.Axis {
		case "y":
			t = (p.Y - from) / (to - from)
		case "z":
			t = (p.Z - from) / (to - from)
		default:
			t = (p.X - from) / (to - from)
		}
		t = math.Max(0, math.Min(1, t))
		a, b := rgb(cm.Pattern.Colors[0]), rgb(cm.Pattern.Colors[1])
		return a.Add(b.Sub(a).MulS(t))
	}
	return cm.Albedo
}
