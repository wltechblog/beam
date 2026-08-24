package geom

// Ray for intersection queries.
type Ray struct {
	Orig, Dir Vec3 // Dir is normalized
}

// Hit is an intersection result. Tri stores the triangle index for meshes.
type Hit struct {
	T     float64
	P     Vec3
	N     Vec3 // geometric normal, oriented against the ray
	Front bool // ray hit the front face
	Tri   int  // triangle index (mesh objects); -1 otherwise
}

// IntersectSphere returns the nearest hit or nil.
func IntersectSphere(r Ray, center Vec3, radius float64) *Hit {
	oc := r.Orig.Sub(center)
	b := oc.Dot(r.Dir)
	c := oc.Dot(oc) - radius*radius
	disc := b*b - c
	if disc < 0 {
		return nil
	}
	sq := sqrt(disc)
	t := -b - sq
	if t < 1e-9 {
		t = -b + sq
	}
	if t < 1e-9 {
		return nil
	}
	p := r.Orig.Add(r.Dir.MulS(t))
	n := p.Sub(center).MulS(1 / radius)
	front := r.Dir.Dot(n) < 0
	if !front {
		n = n.MulS(-1)
	}
	return &Hit{T: t, P: p, N: n, Front: front, Tri: -1}
}

// IntersectPlane returns the hit (any side) or nil.
func IntersectPlane(r Ray, point, normal Vec3) *Hit {
	denom := normal.Dot(r.Dir)
	if denom > -1e-9 && denom < 1e-9 {
		return nil
	}
	t := point.Sub(r.Orig).Dot(normal) / denom
	if t < 1e-9 {
		return nil
	}
	p := r.Orig.Add(r.Dir.MulS(t))
	n := normal
	front := r.Dir.Dot(n) < 0
	if !front {
		n = n.MulS(-1)
	}
	return &Hit{T: t, P: p, N: n, Front: front, Tri: -1}
}

// Triangle corners a-b-c (counter-clockwise seen from the front).
func IntersectTriangle(r Ray, a, b, c Vec3) *Hit {
	e1 := b.Sub(a)
	e2 := c.Sub(a)
	pvec := r.Dir.Cross(e2)
	det := e1.Dot(pvec)
	if det > -1e-12 && det < 1e-12 {
		return nil
	}
	inv := 1 / det
	tvec := r.Orig.Sub(a)
	u := tvec.Dot(pvec) * inv
	if u < 0 || u > 1 {
		return nil
	}
	qvec := tvec.Cross(e1)
	v := r.Dir.Dot(qvec) * inv
	if v < 0 || u+v > 1 {
		return nil
	}
	t := e2.Dot(qvec) * inv
	if t < 1e-9 {
		return nil
	}
	p := r.Orig.Add(r.Dir.MulS(t))
	n := e1.Cross(e2).Norm()
	front := det > 0
	if !front {
		n = n.MulS(-1)
	}
	return &Hit{T: t, P: p, N: n, Front: front, Tri: -1}
}
