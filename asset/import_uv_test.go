package asset

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/geom"
)

// writeOBJ is a test helper: writes s to a temp .obj and loads it.
func writeOBJ(t *testing.T, s string) *geom.TriangleMesh {
	t.Helper()
	p := filepath.Join(t.TempDir(), "m.obj")
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadOBJ(p)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestOBJUVsAndSeamSplit pins the v/vt/vn loader: per-vertex UVs carried
// from face tokens, and a position appearing with TWO different vts gets
// split into two vertices (the classic UV-seam case) instead of one
// vertex whose UV flips per face.
func TestOBJUVsAndSeamSplit(t *testing.T) {
	m := writeOBJ(t, `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 0.25 0.5
f 1/1 2/2 3/3
f 1/3 3/3 4/1
`)
	if len(m.Tris) != 2 {
		t.Fatalf("tris = %d, want 2", len(m.Tris))
	}
	if len(m.UVs) == 0 {
		t.Fatal("no UVs parsed")
	}
	// Position 0 (index 0) is referenced with vt 0 (f1) AND vt 0.25/0.5 (f2)
	// -> seam split: one extra vertex, quad has 5 distinct welded vertices.
	if len(m.Verts) != 5 {
		t.Fatalf("verts = %d, want 5 (seam split fires)", len(m.Verts))
	}
	// The two triangles must reference DIFFERENT vertex indices for the
	// shared position (that's the point of the split).
	if m.Tris[0][0] == m.Tris[1][0] {
		t.Fatalf("both tris use vertex %d for the seam position — no split", m.Tris[0][0])
	}
	// And their UVs must be the distinct authored ones.
	a := m.UVs[m.Tris[0][0]]
	b := m.UVs[m.Tris[1][0]]
	if a.U != 0 || a.V != 0 {
		t.Errorf("f1 seam-corner UV = %v, want (0,0)", a)
	}
	if b.U != 0.25 || b.V != 0.5 {
		t.Errorf("f2 seam-corner UV = %v, want (0.25,0.5)", b)
	}
	// Split vertex must share the position with the original.
	p0 := m.Verts[m.Tris[0][0]]
	p1 := m.Verts[m.Tris[1][0]]
	if p0 != p1 {
		t.Errorf("seam-split vertices diverge in position: %v vs %v", p0, p1)
	}
}

// TestOBJNoUVs pins that position-only OBJs load exactly as before
// (teapot.obj is v//vn; UVs must stay empty, no phantom splits).
func TestOBJNoUVs(t *testing.T) {
	m := writeOBJ(t, `
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1
`)
	if len(m.UVs) != 0 {
		t.Errorf("UVs = %d, want 0 for v//vn mesh", len(m.UVs))
	}
	if len(m.Verts) != 3 {
		t.Errorf("verts = %d, want 3", len(m.Verts))
	}
	if len(m.Norms) != 3 {
		t.Errorf("norms = %d, want 3 (vn carried)", len(m.Norms))
	}
}
