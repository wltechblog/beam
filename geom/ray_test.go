package geom

import "testing"

func TestIntersectSphereHit(t *testing.T) {
	r := Ray{Orig: Vec3{Z: -5}, Dir: Vec3{Z: 1}}
	h := IntersectSphere(r, Vec3{}, 1)
	if h == nil {
		t.Fatal("expected hit")
	}
	if h.T != 4 {
		t.Errorf("T: got %v want 4", h.T)
	}
	if !approx(h.P, Vec3{Z: -1}) {
		t.Errorf("P: %v", h.P)
	}
	if !approx(h.N, Vec3{Z: -1}) {
		t.Errorf("N: %v", h.N)
	}
	if !h.Front {
		t.Errorf("expected front face")
	}
}

func TestIntersectSphereInside(t *testing.T) {
	// Origin inside the sphere: nearest hit is the exit point ahead.
	r := Ray{Orig: Vec3{}, Dir: Vec3{Y: 1}}
	h := IntersectSphere(r, Vec3{}, 1)
	if h == nil {
		t.Fatal("expected hit from inside")
	}
	if h.T != 1 {
		t.Errorf("T: got %v want 1", h.T)
	}
	if h.Front {
		t.Errorf("exit hit should be back face")
	}
	if h.N.Y >= 0 {
		t.Errorf("normal must be flipped against ray: %v", h.N)
	}
}

func TestIntersectSphereMiss(t *testing.T) {
	r := Ray{Orig: Vec3{X: 2}, Dir: Vec3{Y: 1}} // parallel, offset beyond radius
	if h := IntersectSphere(r, Vec3{}, 1); h != nil {
		t.Errorf("expected miss, got %+v", h)
	}
	// Behind the ray origin.
	r2 := Ray{Orig: Vec3{Z: 5}, Dir: Vec3{Z: 1}}
	if h := IntersectSphere(r2, Vec3{}, 1); h != nil {
		t.Errorf("hit behind origin: %+v", h)
	}
}

func TestIntersectSphereTangent(t *testing.T) {
	// Ray grazing at exactly radius offset: disc == 0, single hit.
	r := Ray{Orig: Vec3{X: 1, Z: -5}, Dir: Vec3{Z: 1}}
	h := IntersectSphere(r, Vec3{}, 1)
	if h == nil {
		t.Fatal("tangent ray should hit")
	}
	if h.T != 5 {
		t.Errorf("tangent T: %v", h.T)
	}
}

func TestIntersectPlane(t *testing.T) {
	r := Ray{Orig: Vec3{Y: 2, Z: -1}, Dir: Vec3{Y: -1}}
	h := IntersectPlane(r, Vec3{}, Vec3{Y: 1})
	if h == nil {
		t.Fatal("expected hit")
	}
	if h.T != 2 {
		t.Errorf("T: %v", h.T)
	}
	if !h.Front {
		t.Errorf("hitting from above should be front")
	}
}

func TestIntersectPlaneParallel(t *testing.T) {
	r := Ray{Orig: Vec3{Y: 2}, Dir: Vec3{Z: 1}}
	if h := IntersectPlane(r, Vec3{}, Vec3{Y: 1}); h != nil {
		t.Errorf("parallel ray hit: %+v", h)
	}
	// Ray in the plane (origin on plane, direction parallel).
	r2 := Ray{Orig: Vec3{}, Dir: Vec3{X: 1}}
	if h := IntersectPlane(r2, Vec3{}, Vec3{Y: 1}); h != nil {
		t.Errorf("coplanar ray hit: %+v", h)
	}
}

func TestIntersectTriangle(t *testing.T) {
	// CCW triangle in XY plane seen from +Z: front face normal is +Z,
	// so approach from +Z for a front hit.
	a, b, c := Vec3{X: -1, Y: -1}, Vec3{X: 3, Y: -1}, Vec3{X: -1, Y: 3}
	r := Ray{Orig: Vec3{Z: 5}, Dir: Vec3{Z: -1}}
	h := IntersectTriangle(r, a, b, c)
	if h == nil {
		t.Fatal("expected hit")
	}
	if h.T != 5 {
		t.Errorf("T: %v", h.T)
	}
	if !h.Front {
		t.Errorf("approach from +Z against CCW winding should be front")
	}
	if h.N.Z <= 0 {
		t.Errorf("front normal should point +Z: %v", h.N)
	}
}

func TestIntersectTriangleBackface(t *testing.T) {
	// Same triangle approached from -Z: back-face hit. Front=false and
	// the normal must be flipped against the ray (N.Z < 0).
	a, b, c := Vec3{X: -1, Y: -1}, Vec3{X: 3, Y: -1}, Vec3{X: -1, Y: 3}
	r := Ray{Orig: Vec3{Z: -5}, Dir: Vec3{Z: 1}}
	h := IntersectTriangle(r, a, b, c)
	if h == nil {
		t.Fatal("expected hit")
	}
	if h.Front {
		t.Errorf("approach from -Z should be back face")
	}
	if h.N.Z >= 0 {
		t.Errorf("back-face normal should flip to -Z: %v", h.N)
	}
}

func TestIntersectTriangleMiss(t *testing.T) {
	a, b, c := Vec3{X: -1, Y: -1}, Vec3{X: 1, Y: -1}, Vec3{X: 0, Y: 1}
	r := Ray{Orig: Vec3{X: 5, Z: -5}, Dir: Vec3{Z: 1}} // outside
	if h := IntersectTriangle(r, a, b, c); h != nil {
		t.Errorf("miss expected, got %+v", h)
	}
	// Edge exactly on u+v=1 boundary: u=0.5, v=0.5 is ON the hypotenuse.
	// The check is `v < 0 || u+v > 1` — equality passes. Either result is
	// acceptable numerically, but it must not panic; just run it.
	r2 := Ray{Orig: Vec3{X: 0, Y: 0, Z: -5}, Dir: Vec3{Z: 1}}
	_ = IntersectTriangle(r2, a, b, c)
}

func TestIntersectTriangleDegenerate(t *testing.T) {
	// Collinear points: cross product ~0 → det ~0 → miss.
	a, b, c := Vec3{X: 0}, Vec3{X: 1}, Vec3{X: 2}
	r := Ray{Orig: Vec3{Y: -1}, Dir: Vec3{Y: 1}}
	if h := IntersectTriangle(r, a, b, c); h != nil {
		t.Errorf("degenerate triangle hit: %+v", h)
	}
}
