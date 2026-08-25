package render

import (
	"math"

	"github.com/wltechblog/beam/geom"
)

// Camera generates primary rays with optional thin-lens depth of field.
type Camera struct {
	orig, llc, horiz, vert, u, v, w geom.Vec3
	lens, focus                     float64
}

func NewCamera(pos, lookAt geom.Vec3, fovDeg float64, aspect, aperture, focus float64) *Camera {
	w := pos.Sub(lookAt).Norm() // points backward (toward camera)
	up := geom.Vec3{X: 0, Y: 1, Z: 0}
	if math.Abs(w.Dot(up)) > 0.999 {
		up = geom.Vec3{X: 0, Y: 0, Z: 1}
	}
	u := up.Cross(w).Norm() // right
	v := w.Cross(u)         // up in camera space

	theta := fovDeg * math.Pi / 180
	halfW := math.Tan(theta / 2)
	halfH := halfW / aspect

	if focus <= 0 {
		focus = pos.Sub(lookAt).Len() // autofocus
	}
	return &Camera{
		orig:  pos,
		llc:   pos.Sub(u.MulS(halfW * focus)).Sub(v.MulS(halfH * focus)).Sub(w.MulS(focus)),
		horiz: u.MulS(2 * halfW * focus),
		vert:  v.MulS(2 * halfH * focus),
		u:     u, v: v, w: w,
		lens:  aperture / 2,
		focus: focus,
	}
}

// Ray samples the pixel at (px,py) ∈ [0,1]² UV with jitter (jx,jy). rnd is
// only consulted when depth of field is active; every render is reproducible
// because each pixel owns an independent seeded RNG.
func (c *Camera) Ray(px, py, jx, jy float64, rnd *RNG) geom.Ray {
	target := c.llc.Add(c.horiz.MulS(px + jx)).Add(c.vert.MulS(py + jy))
	dir := target.Sub(c.orig).Norm()
	if c.lens <= 0 {
		return geom.Ray{Orig: c.orig, Dir: dir}
	}
	// Thin lens: find where the pinhole ray crosses the focal plane, then
	// shoot from a sampled lens point to that focal point.
	fwd := c.w.Dot(dir)
	if fwd > -1e-6 {
		return geom.Ray{Orig: c.orig, Dir: dir} // behind camera: no DOF
	}
	t := c.focus / -fwd
	fp := c.orig.Add(dir.MulS(t))
	ang := rnd.Float64() * 2 * math.Pi
	rad := math.Sqrt(rnd.Float64()) * c.lens
	offset := c.u.MulS(beamCos(ang) * rad).Add(c.v.MulS(beamSin(ang) * rad))
	orig := c.orig.Add(offset)
	return geom.Ray{Orig: orig, Dir: fp.Sub(orig).Norm()}
}

// GPUParams exports the precomputed camera basis for the GPU kernel. The
// basis is computed here once in float64, so the kernel consumes bit-ident
// values and never re-derives them.
// Layout: orig(3) llc(3) horiz(3) vert(3) u(3) v(3) w(3) lens focus = 23.
func (c *Camera) GPUParams() [23]float64 {
	var p [23]float64
	put := func(i int, v geom.Vec3) { p[i] = v.X; p[i+1] = v.Y; p[i+2] = v.Z }
	put(0, c.orig)
	put(3, c.llc)
	put(6, c.horiz)
	put(9, c.vert)
	put(12, c.u)
	put(15, c.v)
	put(18, c.w)
	p[21] = c.lens
	p[22] = c.focus
	return p
}
