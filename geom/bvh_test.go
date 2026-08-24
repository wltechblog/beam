package geom

import (
	"math"
	"math/rand"
	"testing"
)

// randomRays builds deterministic pseudo-random rays for fuzz-style tests.
func randomRays(n int, seed int64) []Ray {
	rng := rand.New(rand.NewSource(seed))
	rays := make([]Ray, n)
	for i := range rays {
		orig := Vec3{
			X: rng.Float64()*10 - 5,
			Y: rng.Float64()*10 - 5,
			Z: rng.Float64()*10 - 5,
		}
		dir := Vec3{
			X: rng.Float64() - 0.5,
			Y: rng.Float64() - 0.5,
			Z: rng.Float64() - 0.5,
		}.Norm()
		if dir.Len() < 1e-6 {
			dir = Vec3{Z: 1}
		}
		rays[i] = Ray{Orig: orig, Dir: dir}
	}
	return rays
}

// randomMesh builds a triangle soup with n overlapping triangles in a unit
// cube region (worst case for BVH median splits).
func randomMesh(n int, seed int64) *TriangleMesh {
	rng := rand.New(rand.NewSource(seed))
	m := &TriangleMesh{}
	for i := 0; i < n; i++ {
		a := Vec3{X: rng.Float64()*2 - 1, Y: rng.Float64()*2 - 1, Z: rng.Float64()*2 - 1}
		b := a.Add(Vec3{X: rng.Float64() - 0.5, Y: rng.Float64() - 0.5, Z: rng.Float64() - 0.5})
		c := a.Add(Vec3{X: rng.Float64() - 0.5, Y: rng.Float64() - 0.5, Z: rng.Float64() - 0.5})
		base := int32(len(m.Verts))
		m.Verts = append(m.Verts, a, b, c)
		m.Tris = append(m.Tris, [3]int32{base, base + 1, base + 2})
	}
	return m
}

// bruteForceIntersect intersects a ray against every triangle directly —
// the ground truth the BVH must reproduce exactly.
func bruteForceIntersect(m *TriangleMesh, r Ray) *Hit {
	var best *Hit
	for t := range m.Tris {
		a, b, c := m.TriVerts(t)
		h := IntersectTriangle(r, a, b, c)
		if h != nil && (best == nil || h.T < best.T) {
			h.Tri = t
			best = h
		}
	}
	return best
}

func TestBVHMatchesBruteForce(t *testing.T) {
	m := randomMesh(500, 1)
	bvh := BuildMesh(m)
	rays := randomRays(300, 2)
	mismatch, nilDiff := 0, 0
	for i, r := range rays {
		bvhHit := bvh.Intersect(r)
		want := bruteForceIntersect(m, r)
		switch {
		case bvhHit == nil && want == nil:
			// both miss: OK
		case bvhHit == nil && want != nil:
			nilDiff++
			if nilDiff <= 3 {
				t.Errorf("ray %d: BVH missed hit brute force found (T=%.6f, tri=%d)", i, want.T, want.Tri)
			}
		case bvhHit != nil && want == nil:
			nilDiff++
			if nilDiff <= 3 {
				t.Errorf("ray %d: BVH found phantom hit T=%.6f", i, bvhHit.T)
			}
		default:
			if bvhHit.Tri != want.Tri || math.Abs(bvhHit.T-want.T) > 1e-9 {
				mismatch++
				if mismatch <= 3 {
					t.Errorf("ray %d: BVH tri=%d T=%.9f vs brute tri=%d T=%.9f",
						i, bvhHit.Tri, bvhHit.T, want.Tri, want.T)
				}
			}
		}
	}
	if mismatch > 0 || nilDiff > 0 {
		t.Fatalf("BVH/brute-force divergence: %d mismatches, %d nil-diffs of 300 rays", mismatch, nilDiff)
	}
}

func TestBVHEmptyMesh(t *testing.T) {
	m := &TriangleMesh{}
	bvh := BuildMesh(m)
	if h := bvh.Intersect(Ray{Orig: Vec3{Z: -5}, Dir: Vec3{Z: 1}}); h != nil {
		t.Errorf("empty BVH returned hit: %+v", h)
	}
}

func TestBVHSingleTriangle(t *testing.T) {
	// Single triangle spanning the XY plane at z=0.
	m := &TriangleMesh{
		Verts: []Vec3{{X: -1, Y: -1}, {X: 1, Y: -1}, {X: 0, Y: 1}},
		Tris:  [][3]int32{{0, 1, 2}},
	}
	bvh := BuildMesh(m)
	// Ray through the interior point (0, -0.5).
	h := bvh.Intersect(Ray{Orig: Vec3{Y: -0.5, Z: -5}, Dir: Vec3{Z: 1}})
	if h == nil {
		t.Fatal("expected hit")
	}
	if h.Tri != 0 {
		t.Errorf("Tri index: %d", h.Tri)
	}
	if h.T != 5 {
		t.Errorf("T: %v", h.T)
	}
}

func TestBoundsOf(t *testing.T) {
	m := &TriangleMesh{
		Verts: []Vec3{{X: 1, Y: 2, Z: 3}, {X: -4, Y: 0, Z: 6}, {X: 0, Y: 5, Z: -2}},
		Tris:  [][3]int32{{0, 1, 2}},
	}
	b := m.BoundsOf([]int32{0})
	if b.Min != (Vec3{X: -4, Y: 0, Z: -2}) || b.Max != (Vec3{X: 1, Y: 5, Z: 6}) {
		t.Errorf("BoundsOf: %v %v", b.Min, b.Max)
	}
}

func TestBoundsUnionGrow(t *testing.T) {
	a := Bounds{Vec3{X: 0, Y: 0, Z: 0}, Vec3{X: 1, Y: 1, Z: 1}}
	b := Bounds{Vec3{X: -2, Y: 0.5, Z: 3}, Vec3{X: 0.5, Y: 2, Z: 4}}
	u := a.Union(b)
	if u.Min != (Vec3{X: -2, Y: 0, Z: 0}) || u.Max != (Vec3{X: 1, Y: 2, Z: 4}) {
		t.Errorf("Union: %v %v", u.Min, u.Max)
	}
	if !u.Valid() {
		t.Errorf("union must be valid")
	}
}

func TestBoundsHitRay(t *testing.T) {
	b := Bounds{Vec3{X: -1, Y: -1, Z: -1}, Vec3{X: 1, Y: 1, Z: 1}}
	// Straight through the middle.
	if !b.HitRay(Ray{Orig: Vec3{Z: -5}, Dir: Vec3{Z: 1}}, 1e-9, 1e30) {
		t.Errorf("center ray should hit box")
	}
	// Parallel, outside.
	if b.HitRay(Ray{Orig: Vec3{X: 5, Z: -5}, Dir: Vec3{Z: 1}}, 1e-9, 1e30) {
		t.Errorf("parallel outside ray should miss")
	}
	// Through the box but tmax before entry: miss within [tmin,tmax].
	if b.HitRay(Ray{Orig: Vec3{Z: -5}, Dir: Vec3{Z: 1}}, 1e-9, 1) {
		t.Errorf("ray segment ending before box should miss")
	}
}
