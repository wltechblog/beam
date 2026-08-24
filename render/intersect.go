package render

import (
	"github.com/wltechblog/beam/geom"
)

// Intersect finds the nearest object hit for a ray, or nil.
func (w *World) Intersect(r geom.Ray) (*geom.Hit, *Object3D) {
	var best *geom.Hit
	var bestObj *Object3D
	for _, o := range w.Objects {
		var h *geom.Hit
		switch o.Kind {
		case 's':
			h = geom.IntersectSphere(r, o.Center, o.Radius)
		case 'p':
			h = geom.IntersectPlane(r, o.PlaneP, o.PlaneN)
		case 'm':
			h = o.BVH.Intersect(r)
			if h != nil && len(o.Mesh.Norms) > 0 {
				h.N = o.Mesh.TriNormal(h.Tri, 0.33, 0.33) // interpolated normal
			}
		}
		if h != nil && (best == nil || h.T < best.T) {
			best, bestObj = h, o
		}
	}
	return best, bestObj
}

// IntersectAny reports whether anything blocks the segment p→q (shadow test).
func (w *World) IntersectAny(r geom.Ray, maxT float64) bool {
	h, _ := w.Intersect(r)
	return h != nil && h.T < maxT
}
