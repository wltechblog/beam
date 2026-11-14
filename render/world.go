package render

import (
	"github.com/wltechblog/beam/asset"
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

// TextureResolver is an OPTIONAL MeshResolver extension for loading
// material maps. Resolvers that cannot load images simply don't
// implement it; materials with maps degrade to solid Color.
type TextureResolver interface {
	LoadTexture(path string) (*asset.Texture, error)
}

// ModelResolver is implemented by resolvers that can import full models
// (glTF/GLB with materials + textures). Compile type-asserts, like
// TextureResolver — stub resolvers in tests degrade models to nothing.
type ModelResolver interface {
	LoadModel(src string) (*asset.LoadedModel, error)
}

// EnvMapResolver is implemented by resolvers that can load equirectangular
// environment maps (.hdr Radiance / .png/.jpg LDR). Compile type-asserts;
// scenes degrade to the flat/gradient sky when absent.
type EnvMapResolver interface {
	LoadEnvMap(path string) (*asset.Texture, error)
}

// World is the compiled scene: objects, materials, lights, environment.
// SkyStop is one entry of the piecewise sky gradient (dir.Y keyed).
type SkyStop struct {
	Y     float64
	Color geom.Vec3
}

type World struct {
	Objects     []*Object3D
	Materials   []CompiledMaterial
	EnvColor    geom.Vec3
	EnvTop      geom.Vec3
	EnvMid      geom.Vec3 // optional 3rd sky stop at the horizon
	EnvBot      geom.Vec3
	SkyGradient bool
	SkyCurve    float64 // exponent on gradient t; 0 or 1 = linear
	// SkyStops is a piecewise gradient keyed by dir.Y, taking precedence
	// over the 2/3-stop Sky when non-empty.
	SkyStops     []SkyStop
	Ambient      float64
	AmbientColor geom.Vec3
	Lights       []LightDef
	// Textures is the deduped texture table; CompiledMaterial.TexIdx
	// indexes it. Order = first-use order at compile time.
	Textures []*asset.Texture
	// ShadowSamples is the per-light shadow-ray count for spherical area
	// lights (radius > 0); 0 = 1. Set from scene render.shadowSamples at
	// compile time so both the CPU renderer and the GPU hook see it.
	ShadowSamples int
	// EnvMap is the compiled equirectangular environment map; nil = use
	// flat/gradient sky. Rides the same Textures table as albedo maps so
	// the GPU upload is shared (EnvTexIdx is its index, -1 = none).
	EnvMap    *asset.Texture
	EnvTexIdx int
	// EnvScale multiplies env map radiance (HDR exposure headroom).
	EnvScale float64
	Fog      *FogDef
	// Media replaces Fog when set: extinction + single scattering.
	Media *MediaDef
}

// FogDef is compiled fog parameters (exponential + ground-fog falloff).
type FogDef struct {
	Color geom.Vec3
	K     float64 // 1/e-folding distance
	Alt   float64 // dense below this world-space height
	Off   float64 // falloff scale above Alt; 0 = hard cutoff
	Max   float64 // maximum fog blend factor
	// Atan switches density to POV-Ray fog_type 2: 1/(u^2+1) with
	// u = (y-Off)/Alt, whose segment integral is an atan difference.
	Atan bool
}

// MediaDef is compiled participating media: extinction with the same
// height profile as fog, PLUS single scattering (shadow rays from march
// samples → light shafts). Replaces FogDef when both are authored —
// media is the physically richer model of the same absorption.
type MediaDef struct {
	Color geom.Vec3
	K     float64 // extinction coefficient (1/distance)
	Alt   float64 // dense below this world-space height
	Off   float64 // falloff scale above Alt; 0 = hard cutoff
	Atan  bool    // POV fog_type 2 density profile
	// Single scattering: deterministic midpoint march of this many steps
	// per ray segment; each step shadow-rays toward every light.
	Steps int
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
	Kind       byte // 'p' point, 'd' directional, 's' spot, 'x' emissive mesh, 'X' emissive sphere
	Pos, Dir   geom.Vec3
	Color      geom.Vec3
	Intensity  float64
	CosAngle   float64 // spot: cosine of half-angle
	Radius     float64 // spherical light radius; > 0 = soft shadows
	NoFalloff  bool    // POV default: point lights do not attenuate with distance
	Shadowless bool    // skip shadow rays (POV shadowless keyword)

	// Mesh-light fields (Kind == 'x'). TriV is the flattened triangle
	// soup (9 floats/tri, world space, post-transform); TriCDF is the
	// cumulative distribution over triangle areas (len = nTris, last
	// entry = total area); TotalArea is that total. Sampling = binary
	// search the CDF with one uniform draw, then barycentric jitter
	// with two more — fixed 3 draws per mesh-light sample, so CPU and
	// GPU RNG streams stay in lockstep.
	TriV      []float64
	TriCDF    []float64
	TotalArea float64
}

// SampleMeshLight picks one point on the mesh light's surface from
// three uniform RNG draws: one for the area CDF (binary search), two
// for barycentric coordinates. Returns the point and its face normal.
// Exposed on LightDef so the GPU port can mirror it exactly.
func (l *LightDef) SampleMeshLight(u1, u2, u3 float64) (geom.Vec3, geom.Vec3) {
	// u1 selects a triangle by area
	x := u1 * l.TotalArea
	lo, hi := 0, len(l.TriCDF)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if l.TriCDF[mid] < x {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	tri := lo
	// barycentric jitter (uniform on the triangle): sqrt keeps the
	// distribution flat in area, not in parameter space
	su := math.Sqrt(u2)
	b0 := 1 - su
	b1 := su * (1 - u3)
	b2 := su * u3
	o := tri * 9
	tv := l.TriV
	p := geom.Vec3{
		X: tv[o+0]*b0 + tv[o+3]*b1 + tv[o+6]*b2,
		Y: tv[o+1]*b0 + tv[o+4]*b1 + tv[o+7]*b2,
		Z: tv[o+2]*b0 + tv[o+5]*b1 + tv[o+8]*b2,
	}
	n := geom.Vec3{
		X: tv[o+3+0] - tv[o+0], Y: tv[o+3+1] - tv[o+1], Z: tv[o+3+2] - tv[o+2],
	}.Cross(geom.Vec3{
		X: tv[o+6+0] - tv[o+0], Y: tv[o+6+1] - tv[o+1], Z: tv[o+6+2] - tv[o+2],
	})
	return p, n.Norm()
}

// CompiledMaterial holds runtime material parameters.
type CompiledMaterial struct {
	Kind    byte // 'm' matte, 'M' metal, 'g' glass, 'e' emissive
	Albedo  geom.Vec3
	Pattern *scene.Pattern // nil = solid albedo
	// Tex is the compiled albedo texture; nil = untextured. TexIdx is its
	// slot in the World's texture table (GPU upload order).
	Tex          *asset.Texture
	TexIdx       int // -1 when untextured
	Roughness    float64
	IOR          float64
	Reflectivity float64 // glass Fresnel base override (0 = pure Schlick)
	Power        float64
	Diffuse      float64 // matte direct-light scale (default 1)
	SpecularK    float64 // matte specular highlight strength (default 0)
	Untinted     bool    // reflection untinted (POV reflection w/o metallic)
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
	return CompiledMaterial{Kind: 'm', Albedo: geom.Vec3{X: 0.8, Y: 0.8, Z: 0.8}, Diffuse: 1}
}

func compileMaterial(m *scene.Material) CompiledMaterial {
	cm := CompiledMaterial{Albedo: geom.Vec3{X: 0.8, Y: 0.8, Z: 0.8}, Diffuse: 1}
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
		cm.Reflectivity = m.Reflectivity
		if cm.Reflectivity == 0 {
			cm.Reflectivity = 1 // default full-strength when unset
		}
		cm.Untinted = !m.Metallic
	case "glass":
		cm.Kind = 'g'
		cm.IOR = m.IOR
		cm.Reflectivity = m.Reflectivity
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
	cm.Diffuse = 1
	if m.Diffuse > 0 {
		cm.Diffuse = m.Diffuse
	}
	cm.SpecularK = m.Specular
	return cm
}

// AlbedoAtUV evaluates the albedo at a UV coordinate: texture sample
// multiplied by the material's base Albedo (white = map as-authored).
// Falls back to the pattern/solid path when untextured — call sites
// pass the hit's world position too so patterns keep working on
// textured-and-patterned materials.
func (cm *CompiledMaterial) AlbedoAtUV(u, v float64, p geom.Vec3) geom.Vec3 {
	if cm.Tex != nil {
		r, g, b := cm.Tex.SampleLinear(u, v)
		return geom.Vec3{
			X: float64(r) * cm.Albedo.X,
			Y: float64(g) * cm.Albedo.Y,
			Z: float64(b) * cm.Albedo.Z,
		}
	}
	return cm.AlbedoAt(p)
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
