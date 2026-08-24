package asset

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/geom"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadOBJBasic(t *testing.T) {
	p := writeTemp(t, "m.obj", `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vn 0 0 1
f 1 2 3
f 1 2 3 4
f 1/1 2/1 3/1
`)
	m, err := LoadMesh(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Verts) != 4 {
		t.Errorf("verts: %d want 4", len(m.Verts))
	}
	// tris: 1 + 2 (quad fan) + 1 = 4
	if len(m.Tris) != 4 {
		t.Errorf("tris: %d want 4", len(m.Tris))
	}
	if m.Verts[1] != (geom.Vec3{X: 1}) {
		t.Errorf("vert 1: %v", m.Verts[1])
	}
}

func TestLoadOBJNegativeIndices(t *testing.T) {
	// Negative OBJ indices are relative to the current vertex count.
	p := writeTemp(t, "m.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
	m, err := LoadMesh(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Tris) != 1 || m.Tris[0] != [3]int32{0, 1, 2} {
		t.Errorf("negative index face: %+v", m.Tris)
	}
}

func TestLoadOBJNoFaces(t *testing.T) {
	p := writeTemp(t, "m.obj", "v 0 0 0\nv 1 0 0\n")
	if _, err := LoadMesh(p); err == nil {
		t.Error("expected error for OBJ with no faces")
	}
}

func TestLoadOBJMissingFile(t *testing.T) {
	if _, err := LoadMesh(filepath.Join(t.TempDir(), "nope.obj")); err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadOBJFilenamesWithSpaces(t *testing.T) {
	// Regression for the rsync-style list parser fix: names with spaces.
	p := writeTemp(t, "m.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
	if _, err := LoadMesh(p); err != nil {
		t.Fatal(err)
	}
}

func TestLoadSTLBinary(t *testing.T) {
	// Build a binary STL with one triangle by hand.
	var b bytes.Buffer
	b.Write(make([]byte, 80))
	binary.Write(&b, binary.LittleEndian, uint32(1)) // 1 triangle
	rec := make([]byte, 50)
	putF32 := func(off int, v float32) {
		binary.LittleEndian.PutUint32(rec[off:], math.Float32bits(v))
	}
	// normal
	putF32(0, 0)
	putF32(4, 0)
	putF32(8, 1)
	// verts
	putF32(12, 0)
	putF32(16, 0)
	putF32(20, 0)
	putF32(24, 1)
	putF32(28, 0)
	putF32(32, 0)
	putF32(36, 0)
	putF32(40, 1)
	putF32(44, 0)
	b.Write(rec)

	p := filepath.Join(t.TempDir(), "t.stl")
	if err := os.WriteFile(p, b.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadMesh(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Tris) != 1 || len(m.Verts) != 3 {
		t.Fatalf("binary STL: %d tris %d verts", len(m.Tris), len(m.Verts))
	}
	if m.Verts[1] != (geom.Vec3{X: 1}) || m.Verts[2] != (geom.Vec3{Y: 1}) {
		t.Errorf("verts wrong: %+v", m.Verts)
	}
}

func TestLoadSTLASCII(t *testing.T) {
	p := writeTemp(t, "t.stl", `solid x
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid x
`)
	m, err := LoadMesh(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Tris) != 1 {
		t.Errorf("ascii STL tris: %d", len(m.Tris))
	}
}
