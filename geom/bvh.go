package geom

import (
	"math"
	"sort"
)

func sqrt(x float64) float64 { return math.Sqrt(x) }

// Bounds is an axis-aligned bounding box.
type Bounds struct {
	Min, Max Vec3
}

func (b Bounds) Valid() bool {
	return b.Max.X >= b.Min.X && b.Max.Y >= b.Min.Y && b.Max.Z >= b.Min.Z
}

func (b Bounds) Center() Vec3 {
	return Vec3{(b.Min.X + b.Max.X) / 2, (b.Min.Y + b.Max.Y) / 2, (b.Min.Z + b.Max.Z) / 2}
}

func (b *Bounds) Grow(p Vec3) {
	b.Min = Vec3{math.Min(b.Min.X, p.X), math.Min(b.Min.Y, p.Y), math.Min(b.Min.Z, p.Z)}
	b.Max = Vec3{math.Max(b.Max.X, p.X), math.Max(b.Max.Y, p.Y), math.Max(b.Max.Z, p.Z)}
}

func (b Bounds) GrowPad(eps float64) Bounds {
	return Bounds{b.Min.Add(Vec3{-eps, -eps, -eps}), b.Max.Add(Vec3{eps, eps, eps})}
}

func (b Bounds) Union(o Bounds) Bounds {
	r := b
	r.Grow(o.Min)
	r.Grow(o.Max)
	return r
}

// Hit reports whether the ray intersects the box within [tmin,tmax].
func (b Bounds) HitRay(r Ray, tmin, tmax float64) bool {
	inv := func(d float64) float64 {
		if d == 0 {
			return 1e30
		}
		return 1 / d
	}
	for a := 0; a < 3; a++ {
		invD := inv(r.Dir.Comp(a))
		t0 := (b.Min.Comp(a) - r.Orig.Comp(a)) * invD
		t1 := (b.Max.Comp(a) - r.Orig.Comp(a)) * invD
		if invD < 0 {
			t0, t1 = t1, t0
		}
		if t0 > tmin {
			tmin = t0
		}
		if t1 < tmax {
			tmax = t1
		}
		if tmax < tmin {
			return false
		}
	}
	return true
}

// LongestAxis returns 0, 1, or 2.
func (b Bounds) LongestAxis() int {
	dx := b.Max.X - b.Min.X
	dy := b.Max.Y - b.Min.Y
	dz := b.Max.Z - b.Min.Z
	switch {
	case dx >= dy && dx >= dz:
		return 0
	case dy >= dz:
		return 1
	}
	return 2
}

// TriangleMesh is an indexed triangle set — the universal object form in
// Beam (everything resolves to meshes) and the GPU flattening unit.
type TriangleMesh struct {
	Verts  []Vec3
	Tris   [][3]int32
	Norms  []Vec3 // per-vertex normals; empty = use face normals
	MatIdx []int32
}

func (m *TriangleMesh) NumTris() int { return len(m.Tris) }

// TriVerts returns the three corner positions.
func (m *TriangleMesh) TriVerts(t int) (Vec3, Vec3, Vec3) {
	idx := m.Tris[t]
	return m.Verts[idx[0]], m.Verts[idx[1]], m.Verts[idx[2]]
}

// TriNormal returns the interpolated vertex normal when available.
func (m *TriangleMesh) TriNormal(t int, u, v float64) Vec3 {
	if len(m.Norms) == 0 {
		a, b, c := m.TriVerts(t)
		return b.Sub(a).Cross(c.Sub(a)).Norm()
	}
	idx := m.Tris[t]
	w := 1 - u - v
	return m.Norms[idx[0]].MulS(w).
		Add(m.Norms[idx[1]].MulS(u)).
		Add(m.Norms[idx[2]].MulS(v)).Norm()
}

// BoundsOf computes the bounds of the given triangle indices.
func (m *TriangleMesh) BoundsOf(tris []int32) Bounds {
	b := Bounds{Vec3{math.Inf(1), math.Inf(1), math.Inf(1)}, Vec3{math.Inf(-1), math.Inf(-1), math.Inf(-1)}}
	for _, t := range tris {
		a, c, d := m.TriVerts(int(t))
		b.Grow(a)
		b.Grow(c)
		b.Grow(d)
	}
	return b
}

// BuildMesh builds a BVH over a mesh. The mesh's tris are reordered in place
// so leaves reference contiguous ranges.
func BuildMesh(m *TriangleMesh) *BVH {
	idx := make([]int32, len(m.Tris))
	for i := range idx {
		idx[i] = int32(i)
	}
	b := &BVH{mesh: m}
	b.build(idx)
	return b
}

// BVH is a median-split binary BVH.
type BVH struct {
	mesh *TriangleMesh
	// flattened nodes: internal nodes store child indices
	nodes []bvhNode
	left  []int32 // triangle start (leaf) or left child (internal)
	count []int32 // triangle count; 0 marks internal
	// triIndex maps flattened position -> original mesh triangle
	triOrder []int32
}

type bvhNode struct {
	bounds   Bounds
	right    int32
	internal bool
}

func (b *BVH) build(tris []int32) {
	b.triOrder = make([]int32, 0, len(tris))
	b.buildNode(tris, 0)
}

func (b *BVH) buildNode(tris []int32, depth int) int32 {
	bounds := b.mesh.BoundsOf(tris)

	if len(tris) <= 4 || depth > 32 {
		start := int32(len(b.triOrder))
		b.triOrder = append(b.triOrder, tris...)
		b.nodes = append(b.nodes, bvhNode{bounds: bounds, internal: false})
		b.left = append(b.left, start)
		b.count = append(b.count, int32(len(tris)))
		return int32(len(b.nodes) - 1)
	}

	axis := bounds.LongestAxis()
	sort.Slice(tris, func(i, j int) bool {
		a, _, _ := b.mesh.TriVerts(int(tris[i]))
		c, _, _ := b.mesh.TriVerts(int(tris[j]))
		return a.Comp(axis) < c.Comp(axis)
	})
	mid := len(tris) / 2

	b.nodes = append(b.nodes, bvhNode{}) // placeholder for internal node
	b.left = append(b.left, 0)
	b.count = append(b.count, 0)
	nodeIdx := int32(len(b.nodes) - 1)

	l := b.buildNode(tris[:mid], depth+1)
	r := b.buildNode(tris[mid:], depth+1)

	b.nodes[nodeIdx] = bvhNode{
		bounds:   b.mesh.BoundsOf(tris),
		internal: true,
		right:    r,
	}
	b.left[nodeIdx] = l
	return nodeIdx
}

// Intersect returns the nearest hit or nil.
func (b *BVH) Intersect(r Ray) *Hit {
	if len(b.nodes) == 0 {
		return nil
	}
	var best *Hit
	b.intersectNode(0, r, &best)
	return best
}

// FlatNode is one BVH node in array form, exported for GPU flattening.
type FlatNode struct {
	BMin, BMax Vec3
	Left       int32 // leaf: triangle start (index into triOrder); internal: left child node
	Right      int32 // internal: right child node
	Count      int32 // leaf: triangle count; 0 marks internal
}

// Flatten exports the BVH nodes and the leaf triangle order (indices into
// mesh.Tris) for GPU upload. Nodes are in the same array layout used by
// Intersect; leaf Left indexes triOrder.
func (b *BVH) Flatten() ([]FlatNode, []int32) {
	nodes := make([]FlatNode, len(b.nodes))
	for i, n := range b.nodes {
		nodes[i] = FlatNode{
			BMin:  n.bounds.Min,
			BMax:  n.bounds.Max,
			Left:  b.left[i],
			Right: n.right,
			Count: b.count[i],
		}
	}
	return nodes, b.triOrder
}

func (b *BVH) intersectNode(n int32, r Ray, best **Hit) {
	node := b.nodes[n]
	tmax := math.MaxFloat64
	if *best != nil {
		tmax = (*best).T
	}
	if !node.bounds.GrowPad(1e-4).HitRay(r, 1e-9, tmax) {
		return
	}
	if node.internal {
		b.intersectNode(b.left[n], r, best)
		b.intersectNode(node.right, r, best)
		return
	}
	for i := int32(0); i < b.count[n]; i++ {
		t := b.triOrder[b.left[n]+i]
		a, v0, v1 := b.mesh.TriVerts(int(t))
		h := IntersectTriangle(r, a, v0, v1)
		if h != nil && (*best == nil || h.T < (*best).T) {
			h.Tri = int(t)
			*best = h
		}
	}
}
