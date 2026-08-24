package render

import (
	"fmt"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// Compile converts a scene into render-ready structures. Mesh and text
// objects are resolved through the MeshResolver (nil resolver skips them).
func Compile(s *scene.Scene, resolver MeshResolver) (*World, error) {
	w := &World{
		EnvColor: rgb(envColor(s)),
		Ambient:  0.2,
	}
	w.AmbientColor = w.EnvColor
	if s.Environment != nil && s.Environment.Ambient > 0 {
		w.Ambient = s.Environment.Ambient
	}
	if s.Environment != nil && len(s.Environment.AmbientColor) == 3 {
		w.AmbientColor = rgb(s.Environment.AmbientColor)
	}
	if s.Environment != nil && len(s.Environment.SkyMap) >= 2 {
		for _, st := range s.Environment.SkyMap {
			w.SkyStops = append(w.SkyStops, SkyStop{Y: st.Y, Color: vec3Of(st.Color)})
		}
		for i := 1; i < len(w.SkyStops); i++ { // insertion sort by Y
			for j := i; j > 0 && w.SkyStops[j-1].Y > w.SkyStops[j].Y; j-- {
				w.SkyStops[j-1], w.SkyStops[j] = w.SkyStops[j], w.SkyStops[j-1]
			}
		}
		w.SkyGradient = true
	} else if s.Environment != nil && len(s.Environment.Sky[0]) > 0 {
		w.SkyGradient = true
		w.SkyCurve = s.Environment.SkyCurve
		w.EnvTop = vec3Of(s.Environment.Sky[0])
		w.EnvBot = vec3Of(s.Environment.Sky[1])
		if s.Environment.Sky[2] != nil {
			w.EnvMid = vec3Of(s.Environment.Sky[2])
		}
	}
	if s.Environment != nil && s.Environment.Fog != nil {
		f := s.Environment.Fog
		w.Fog = &FogDef{
			Color: rgb(f.Color),
			K:     1.0 / f.Distance,
			Alt:   f.Altitude,
			Off:   f.Offset,
			Max:   f.MaxFog,
			Atan:  f.Profile == "atan",
		}
		if w.Fog.Max <= 0 {
			w.Fog.Max = 1
		}
	}

	matIdx := map[string]int{}
	mats := []CompiledMaterial{defaultMat()}
	for name, m := range s.Materials {
		matIdx[name] = len(mats)
		mats = append(mats, compileMaterial(m))
	}
	w.Materials = mats

	for i := range s.Objects {
		o := &s.Objects[i]
		mi, ok := matIdx[o.Material]
		if !ok {
			mi = 0 // "__default"
		}
		var obj Object3D
		switch o.Type {
		case "sphere":
			obj = Object3D{Kind: 's', Center: rgbv(o.Center), Radius: o.Radius, MatIdx: mi}
		case "plane":
			obj = Object3D{Kind: 'p', PlaneP: rgbv(o.Point), PlaneN: normv(o.Normal), MatIdx: mi}
		case "box":
			mesh := BoxMesh(rgbv(o.Min), rgbv(o.Max))
			if o.Transform != nil {
				mesh = transformMesh(mesh, o.Transform)
			}
			obj = Object3D{Kind: 'm', Mesh: mesh, BVH: geom.BuildMesh(mesh), MatIdx: mi}
		case "mesh":
			if resolver == nil {
				continue
			}
			m, err := resolver.LoadMesh(o.Src)
			if err != nil {
				return nil, fmt.Errorf("mesh %q: %w", o.Src, err)
			}
			if o.Transform != nil {
				m = transformMesh(m, o.Transform)
			}
			obj = Object3D{Kind: 'm', Mesh: m, BVH: geom.BuildMesh(m), MatIdx: mi}
		case "text":
			if resolver == nil {
				continue
			}
			m, err := resolver.TextMesh(o.Text, o.Font, o.Size, o.Depth, o.Spacing)
			if err != nil {
				return nil, fmt.Errorf("text %q: %w", o.Text, err)
			}
			if o.Transform != nil {
				m = transformMesh(m, o.Transform)
			}
			obj = Object3D{Kind: 'm', Mesh: m, BVH: geom.BuildMesh(m), MatIdx: mi}
		default:
			continue
		}
		if o.Transform != nil && obj.Kind != 'm' {
			obj = transformPrimitive(obj, o.Transform)
		}
		w.Objects = append(w.Objects, &obj)
	}

	for i := range s.Lights {
		l := &s.Lights[i]
		ld := LightDef{Pos: rgbv(l.Pos), Dir: normv(l.Dir), Intensity: l.Intensity}
		if ld.Intensity == 0 {
			ld.Intensity = 1
		}
		ld.Color = geom.Vec3{X: 1, Y: 1, Z: 1}
		if len(l.Color) == 3 {
			ld.Color = rgb(l.Color)
		}
		switch l.Type {
		case "point":
			ld.Kind = 'p'
		case "directional":
			ld.Kind = 'd'
		case "spot":
			ld.Kind = 's'
			ld.CosAngle = cosDeg(l.Angle)
		}
		w.Lights = append(w.Lights, ld)
	}

	return w, nil
}

func envColor(s *scene.Scene) []float64 {
	if s.Environment == nil || len(s.Environment.Color) != 3 {
		return []float64{0.04, 0.05, 0.08}
	}
	return s.Environment.Color
}

// transformMesh applies a transform to every vertex (normals recomputed as
// face normals if the transform has non-uniform scale; approximate: rotate
// normals, renormalized).
func transformMesh(m *geom.TriangleMesh, t *scene.Transform) *geom.TriangleMesh {
	tm, rm := buildMatrices(t)
	out := &geom.TriangleMesh{
		Verts:  make([]geom.Vec3, len(m.Verts)),
		Tris:   m.Tris,
		MatIdx: m.MatIdx,
	}
	for i, v := range m.Verts {
		out.Verts[i] = tm.Apply(v)
	}
	if len(m.Norms) > 0 {
		out.Norms = make([]geom.Vec3, len(m.Norms))
		for i, n := range m.Norms {
			out.Norms[i] = rm.ApplyDir(n).Norm()
		}
	}
	return out
}

// transformPrimitive bakes a transform into a primitive: rotation+translation
// for spheres/planes; non-uniform scale turns spheres into meshes (correct
// ellipsoids) and planes stay planes (scaled infinitely).
func transformPrimitive(obj Object3D, t *scene.Transform) Object3D {
	tm, rm := buildMatrices(t)
	switch obj.Kind {
	case 's':
		// uniform scale = still a sphere
		s := scaleOf(t)
		if s.X == s.Y && s.Y == s.Z {
			obj.Center = tm.Apply(obj.Center)
			obj.Radius *= mathAbs(s.X)
			return obj
		}
		// non-uniform: convert to mesh
		mesh := SphereMesh(obj.Center, obj.Radius, 48, 32)
		return meshObject(mesh, tm, obj.MatIdx)
	case 'p':
		obj.PlaneP = tm.Apply(obj.PlaneP)
		obj.PlaneN = rm.ApplyDir(obj.PlaneN).Norm()
		return obj
	case 'm':
		// handled by caller (transformMesh)
		return obj
	}
	return obj
}

func meshObject(m *geom.TriangleMesh, tm geom.Mat4, matIdx int) Object3D {
	for i := range m.Verts {
		m.Verts[i] = tm.Apply(m.Verts[i])
	}
	if len(m.Norms) > 0 {
		norms := make([]geom.Vec3, len(m.Verts))
		for ti := range m.Tris {
			a, b, c := m.TriVerts(ti)
			fn := b.Sub(a).Cross(c.Sub(a))
			for _, vi := range m.Tris[ti] {
				norms[vi] = norms[vi].Add(fn)
			}
		}
		for i := range norms {
			norms[i] = norms[i].Norm()
		}
		m.Norms = norms
	}
	return Object3D{Kind: 'm', Mesh: m, BVH: geom.BuildMesh(m), MatIdx: matIdx}
}

func buildMatrices(t *scene.Transform) (geom.Mat4, geom.Mat4) {
	tv := vecOf(t.Translate, 0, 0, 0)
	rv := vecOf(t.Rotate, 0, 0, 0)
	sv := vecOf(t.Scale, 1, 1, 1)
	tm := geom.ComposeTRS(tv, rv, sv)
	rm := geom.RotateZ(rv.Z).Mul(geom.RotateY(rv.Y)).Mul(geom.RotateX(rv.X))
	return tm, rm
}

func vecOf(v []float64, d0, d1, d2 float64) geom.Vec3 {
	if len(v) != 3 {
		return geom.Vec3{X: d0, Y: d1, Z: d2}
	}
	return geom.Vec3{X: v[0], Y: v[1], Z: v[2]}
}

func scaleOf(t *scene.Transform) geom.Vec3 {
	return vecOf(t.Scale, 1, 1, 1)
}

func mathAbs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
