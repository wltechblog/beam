package geom

import "math"

// Vec3 is a double-precision 3-vector. Beam runs entirely in float64 for
// cross-platform bit-exactness; the GPU kernel layer will narrow to float32
// at the boundary.
type Vec3 struct{ X, Y, Z float64 }

func (a Vec3) Add(b Vec3) Vec3     { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func (a Vec3) Sub(b Vec3) Vec3     { return Vec3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func (a Vec3) MulS(s float64) Vec3 { return Vec3{a.X * s, a.Y * s, a.Z * s} }
func (a Vec3) Dot(b Vec3) float64  { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func (a Vec3) Cross(b Vec3) Vec3 {
	return Vec3{a.Y*b.Z - a.Z*b.Y, a.Z*b.X - a.X*b.Z, a.X*b.Y - a.Y*b.X}
}
func (a Vec3) Len() float64 { return math.Sqrt(a.Dot(a)) }

func (a Vec3) Norm() Vec3 {
	l := a.Len()
	if l < 1e-12 {
		return Vec3{}
	}
	return a.MulS(1 / l)
}

func (a Vec3) Comp(c int) float64 {
	switch c {
	case 0:
		return a.X
	case 1:
		return a.Y
	}
	return a.Z
}

// Mat4 is a 4x4 matrix, row-major.
type Mat4 [16]float64

func Identity() Mat4 {
	return Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
}

func (m Mat4) Mul(b Mat4) Mat4 {
	var r Mat4
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var s float64
			for k := 0; k < 4; k++ {
				s += m[i*4+k] * b[k*4+j]
			}
			r[i*4+j] = s
		}
	}
	return r
}

func (m Mat4) Apply(p Vec3) Vec3 {
	x := m[0]*p.X + m[1]*p.Y + m[2]*p.Z + m[3]
	y := m[4]*p.X + m[5]*p.Y + m[6]*p.Z + m[7]
	z := m[8]*p.X + m[9]*p.Y + m[10]*p.Z + m[11]
	w := m[12]*p.X + m[13]*p.Y + m[14]*p.Z + m[15]
	if w != 1 && w != 0 {
		return Vec3{x / w, y / w, z / w}
	}
	return Vec3{x, y, z}
}

// ApplyDir transforms a direction (no translation).
func (m Mat4) ApplyDir(v Vec3) Vec3 {
	return Vec3{
		m[0]*v.X + m[1]*v.Y + m[2]*v.Z,
		m[4]*v.X + m[5]*v.Y + m[6]*v.Z,
		m[8]*v.X + m[9]*v.Y + m[10]*v.Z,
	}
}

func Translate(x, y, z float64) Mat4 {
	m := Identity()
	m[3], m[7], m[11] = x, y, z
	return m
}

func Scale(x, y, z float64) Mat4 {
	m := Identity()
	m[0], m[5], m[10] = x, y, z
	return m
}

func RotateX(deg float64) Mat4 {
	c, s := math.Cos(deg*math.Pi/180), math.Sin(deg*math.Pi/180)
	return Mat4{1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1}
}

func RotateY(deg float64) Mat4 {
	c, s := math.Cos(deg*math.Pi/180), math.Sin(deg*math.Pi/180)
	return Mat4{c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1}
}

func RotateZ(deg float64) Mat4 {
	c, s := math.Cos(deg*math.Pi/180), math.Sin(deg*math.Pi/180)
	return Mat4{c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
}

// Transpose returns mᵀ.
func (m Mat4) Transpose() Mat4 {
	return Mat4{
		m[0], m[4], m[8], m[12],
		m[1], m[5], m[9], m[13],
		m[2], m[6], m[10], m[14],
		m[3], m[7], m[11], m[15],
	}
}

// Inverse returns m⁻¹ via the adjugate method. For affine transforms
// (the only kind beam composes) the result is exact.
func (m Mat4) Inverse() Mat4 {
	var inv Mat4
	inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10]
	inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10]
	inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9]
	inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9]
	inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10]
	inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10]
	inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9]
	inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9]
	inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6]
	inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6]
	inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5]
	inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5]
	inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6]
	inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6]
	inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5]
	inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5]

	det := m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12]
	if det == 0 {
		return Identity()
	}
	di := 1 / det
	for i := range inv {
		inv[i] *= di
	}
	return inv
}

// ComposeTRS builds translate * rotateZ * rotateY * rotateX * scale.
func ComposeTRS(t, r, s Vec3) Mat4 {
	m := Translate(t.X, t.Y, t.Z).
		Mul(RotateZ(r.Z)).Mul(RotateY(r.Y)).Mul(RotateX(r.X)).
		Mul(Scale(s.X, s.Y, s.Z))
	return m
}
