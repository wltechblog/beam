package render

import (
	"fmt"
	"math"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// Fit-checking: project world-space geometry through the camera to pixel
// space and report per-object screen coverage. Catches clipping (objects
// extending past the frame) before a full render is burned — generic across
// every object type because it operates on compiled geometry.

// ObjectScreenReport describes where one object lands on the output image.
type ObjectScreenReport struct {
	Name    string // scene label ("object[3]")
	Kind    string // sphere / plane / mesh/text
	InFrame bool   // object is fully inside the frame
	// Pixel-space AABB of the object's projected bounds. X grows right,
	// Y grows down (image convention). Values may lie outside [0,W]x[0,H]
	// when the object extends beyond the frame.
	MinPx, MaxPx float64
	MinPy, MaxPy float64
	// Screen-space area fraction covered (0..1 of the full image),
	// estimated from the projected AABB.
	Coverage float64
}

// ScreenFit summarizes fit of every scene object in the camera frame.
type ScreenFit struct {
	Width, Height int
	// Per-object reports, one entry per compiled object.
	Objects []ObjectScreenReport
	// Fitted reports whether every bounded object is fully inside the
	// frame. Planes are inherently unbounded and excluded from the
	// verdict (report-only).
	Fitted bool
	// Problems lists human-readable issues (clipping, off-screen).
	Problems []string
}

// ScreenFitCheck compiles the scene (with resolver) and projects each
// bounded object's world AABB corners through the camera at the given
// output size. This is the pre-render guard: a scene that fails this check
// WILL clip in the render.
func ScreenFitCheck(s *scene.Scene, resolver MeshResolver, width, height int) (*ScreenFit, error) {
	s.ApplyDefaults()

	world, err := Compile(s, resolver)
	if err != nil {
		return nil, err
	}

	cam := NewCamera(
		vec3Of(s.Camera.Pos),
		vec3Of(s.Camera.LookAt),
		s.Camera.FOV,
		float64(width)/float64(height),
		0, 0, // pinhole: no DOF for projection math
	)

	fit := &ScreenFit{Width: width, Height: height, Fitted: true}

	for i, obj := range world.Objects {
		rep := ObjectScreenReport{Name: fmt.Sprintf("object[%d]", i), Kind: kindName(obj)}

		bounds, bounded := objectBounds(obj)
		if !bounded {
			// Planes and other unbounded shapes: report-only.
			rep.InFrame = true
			rep.Coverage = 1.0
			fit.Objects = append(fit.Objects, rep)
			continue
		}

		minU, maxU := math.Inf(1), math.Inf(-1)
		minV, maxV := math.Inf(1), math.Inf(-1)
		allBehind := true
		clipped := false

		for ci := 0; ci < 8; ci++ {
			p := corner(bounds, ci)
			u, v, ok := projectPoint(cam, p, width, height)
			if !ok {
				allBehind = false // ok=false only means behind; still count
				clipped = true
				continue
			}
			allBehind = false
			if u < 0 || u > float64(width) || v < 0 || v > float64(height) {
				clipped = true
			}
			minU = math.Min(minU, u)
			maxU = math.Max(maxU, u)
			minV = math.Min(minV, v)
			maxV = math.Max(maxV, v)
		}

		rep.MinPx, rep.MaxPx = minU, maxU
		rep.MinPy, rep.MaxPy = minV, maxV
		rep.Coverage = clamp01(((maxU - minU) * (maxV - minV)) / (float64(width) * float64(height)))
		rep.InFrame = !clipped && !allBehind

		if allBehind {
			fit.Fitted = false
			fit.Problems = append(fit.Problems, fmt.Sprintf(
				"%s (%s) is entirely behind the camera", rep.Name, rep.Kind))
		} else if clipped {
			fit.Fitted = false
			fit.Problems = append(fit.Problems, fmt.Sprintf(
				"%s (%s) is clipped: screen x[%.0f,%.0f] y[%.0f,%.0f] of %dx%d",
				rep.Name, rep.Kind, minU, maxU, minV, maxV, width, height))
		}
		fit.Objects = append(fit.Objects, rep)
	}

	return fit, nil
}

func clamp01(x float64) float64 { return math.Max(0, math.Min(1, x)) }

func kindName(obj *Object3D) string {
	switch obj.Kind {
	case 's':
		return "sphere"
	case 'p':
		return "plane"
	case 'm':
		return "mesh/text"
	}
	return "unknown"
}

// objectBounds returns the world-space AABB of a compiled object. Planes
// return bounded=false (infinite extent).
func objectBounds(obj *Object3D) (geom.Bounds, bool) {
	switch obj.Kind {
	case 's':
		r := math.Abs(obj.Radius)
		return geom.Bounds{
			Min: obj.Center.Add(geom.Vec3{X: -r, Y: -r, Z: -r}),
			Max: obj.Center.Add(geom.Vec3{X: r, Y: r, Z: r}),
		}, true
	case 'm':
		return meshBounds(obj.Mesh), true
	}
	return geom.Bounds{}, false
}

// meshBounds returns the AABB over all vertices.
func meshBounds(m *geom.TriangleMesh) geom.Bounds {
	var b geom.Bounds
	for i, v := range m.Verts {
		if i == 0 {
			b.Min, b.Max = v, v
			continue
		}
		b.Grow(v)
	}
	return b
}

// corner returns AABB corner ci ∈ [0,8).
func corner(b geom.Bounds, ci int) geom.Vec3 {
	p := geom.Vec3{X: b.Min.X, Y: b.Min.Y, Z: b.Min.Z}
	if ci&1 != 0 {
		p.X = b.Max.X
	}
	if ci&2 != 0 {
		p.Y = b.Max.Y
	}
	if ci&4 != 0 {
		p.Z = b.Max.Z
	}
	return p
}

// projectPoint maps a world point to pixel coordinates using the camera's
// pinhole basis. ok=false when the point is behind the camera.
func projectPoint(cam *Camera, p geom.Vec3, width, height int) (u, v float64, ok bool) {
	// Camera basis: w points backward (toward camera), u = right, v = up.
	d := p.Sub(cam.orig)
	fwd := -d.Dot(cam.w)
	if fwd <= 0 {
		return 0, 0, false
	}
	// halfW/halfH are the frustum half-extents at the focal plane
	// (tan(fov/2) and tan(fov/2)/aspect); horiz = u*2*halfW*focus.
	halfW := cam.horiz.Len() / (2 * cam.focus)
	halfH := cam.vert.Len() / (2 * cam.focus)
	su := d.Dot(cam.u) / (fwd * halfW)
	sv := d.Dot(cam.v) / (fwd * halfH)
	// Convert to pixels; py flips because image y grows downward.
	pu := (su + 1) / 2 * float64(width)
	pv := (1 - sv) / 2 * float64(height)
	return pu, pv, true
}
