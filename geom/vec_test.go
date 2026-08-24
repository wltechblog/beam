package geom

import (
	"math"
	"testing"
)

const eps = 1e-12

func approx(a, b Vec3) bool {
	return math.Abs(a.X-b.X) < eps && math.Abs(a.Y-b.Y) < eps && math.Abs(a.Z-b.Z) < eps
}

func TestVec3Basics(t *testing.T) {
	a := Vec3{X: 1, Y: 2, Z: 3}
	b := Vec3{X: 4, Y: 5, Z: 6}
	if got := a.Add(b); got != (Vec3{X: 5, Y: 7, Z: 9}) {
		t.Errorf("Add: %v", got)
	}
	if got := a.Sub(b); got != (Vec3{X: -3, Y: -3, Z: -3}) {
		t.Errorf("Sub: %v", got)
	}
	if got := a.MulS(2); got != (Vec3{X: 2, Y: 4, Z: 6}) {
		t.Errorf("MulS: %v", got)
	}
	if got := a.Dot(b); got != 32 {
		t.Errorf("Dot: %v", got)
	}
	// Cross product handedness: x̂ × ŷ = ẑ
	if got := (Vec3{X: 1}).Cross(Vec3{Y: 1}); got != (Vec3{Z: 1}) {
		t.Errorf("Cross: %v", got)
	}
	if l := (Vec3{X: 3, Y: 4}).Len(); l != 5 {
		t.Errorf("Len: %v", l)
	}
}

func TestVec3Norm(t *testing.T) {
	if got := (Vec3{X: 0, Y: 0, Z: 2}).Norm(); got != (Vec3{Z: 1}) {
		t.Errorf("Norm: %v", got)
	}
	// Degenerate vector must return zero, not NaN.
	if got := (Vec3{}).Norm(); got != (Vec3{}) {
		t.Errorf("Norm of zero: %v", got)
	}
}

func TestVec3Comp(t *testing.T) {
	v := Vec3{X: 7, Y: 8, Z: 9}
	if v.Comp(0) != 7 || v.Comp(1) != 8 || v.Comp(2) != 9 {
		t.Errorf("Comp roundtrip failed")
	}
}

func TestMat4IdentityApply(t *testing.T) {
	m := Identity()
	p := Vec3{X: 1, Y: 2, Z: 3}
	if got := m.Apply(p); got != p {
		t.Errorf("identity changed point: %v", got)
	}
}

func TestMat4Translate(t *testing.T) {
	m := Translate(10, 20, 30)
	if got := m.Apply(Vec3{X: 1, Y: 1, Z: 1}); got != (Vec3{X: 11, Y: 21, Z: 31}) {
		t.Errorf("translate: %v", got)
	}
	// ApplyDir must NOT translate.
	if got := m.ApplyDir(Vec3{X: 1, Y: 0, Z: 0}); got != (Vec3{X: 1}) {
		t.Errorf("ApplyDir translated: %v", got)
	}
}

func TestMat4RotateZ90(t *testing.T) {
	m := RotateZ(90)
	// x̂ → ŷ
	got := m.Apply(Vec3{X: 1})
	if !approx(got, Vec3{Y: 1}) {
		t.Errorf("rotZ90 x̂: %v", got)
	}
}

func TestMat4RotateY180(t *testing.T) {
	m := RotateY(180)
	got := m.Apply(Vec3{X: 1})
	if !approx(got, Vec3{X: -1}) {
		t.Errorf("rotY180 x̂: %v", got)
	}
}

func TestMat4Scale(t *testing.T) {
	m := Scale(2, 3, 4)
	if got := m.Apply(Vec3{X: 1, Y: 1, Z: 1}); got != (Vec3{X: 2, Y: 3, Z: 4}) {
		t.Errorf("scale: %v", got)
	}
}

func TestComposeTRSTranslate(t *testing.T) {
	m := ComposeTRS(Vec3{X: 5, Y: 6, Z: 7}, Vec3{}, Vec3{X: 1, Y: 1, Z: 1})
	if got := m.Apply(Vec3{}); got != (Vec3{X: 5, Y: 6, Z: 7}) {
		t.Errorf("TRS translate: %v", got)
	}
}

func TestComposeTRSScale(t *testing.T) {
	m := ComposeTRS(Vec3{}, Vec3{}, Vec3{X: 2, Y: 2, Z: 2})
	if got := m.Apply(Vec3{X: 1, Y: -1, Z: 3}); got != (Vec3{X: 2, Y: -2, Z: 6}) {
		t.Errorf("TRS scale: %v", got)
	}
}

func TestComposeTRSRotate(t *testing.T) {
	// rotate 90° about Z maps x̂ to ŷ regardless of scale/translation order.
	m := ComposeTRS(Vec3{}, Vec3{Z: 90}, Vec3{X: 1, Y: 1, Z: 1})
	if got := m.Apply(Vec3{X: 1}); !approx(got, Vec3{Y: 1}) {
		t.Errorf("TRS rotate: %v", got)
	}
}

func TestMat4MulAssociative(t *testing.T) {
	a := Translate(1, 2, 3)
	b := RotateX(30)
	c := Scale(2, 2, 2)
	p := Vec3{X: 0.5, Y: -0.25, Z: 1}
	ab_c := a.Mul(b).Mul(c).Apply(p)
	a_bc := a.Mul(b.Mul(c)).Apply(p)
	if !approx(ab_c, a_bc) {
		t.Errorf("matrix multiply not associative: %v vs %v", ab_c, a_bc)
	}
}
