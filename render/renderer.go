package render

import (
	"image"
	"image/color"
	"math"
	"runtime"
	"sync"

	"github.com/wltechblog/beam/geom"
)

// Renderer renders a compiled World with a Whitted-style integrator:
// direct lighting with shadows, mirror reflections with roughness
// perturbation, and Snell refraction with Fresnel + TIR, capped at
// MaxDepth bounces. One RNG per pixel, threaded through recursion —
// same scene + seed = byte-identical PNG on any machine.
type Renderer struct {
	World    *World
	Camera   *Camera
	Width    int
	Height   int
	Samples  int
	MaxDepth int
	Exposure float64
	Seed     int64
}

// Render produces the image using one worker goroutine per CPU. onTile
// (optional) receives (finished, total) after each completed tile.
func (r *Renderer) Render(onTile func(finished, total int)) *image.RGBA {
	const tileSize = 64
	cols := (r.Width + tileSize - 1) / tileSize
	rows := (r.Height + tileSize - 1) / tileSize
	total := cols * rows

	img := image.NewRGBA(image.Rect(0, 0, r.Width, r.Height))
	type tile struct{ row, col int }
	tasks := make(chan tile, total)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			tasks <- tile{row, col}
		}
	}
	close(tasks)

	var mu sync.Mutex
	finished := 0
	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range tasks {
				r.renderTile(img, t.row, t.col, tileSize)
				mu.Lock()
				finished++
				f, tot := finished, total
				mu.Unlock()
				if onTile != nil {
					onTile(f, tot)
				}
			}
		}()
	}
	wg.Wait()
	return img
}

func (r *Renderer) renderTile(img *image.RGBA, row, col, size int) {
	x0 := col * size
	y0 := row * size
	for y := y0; y < y0+size && y < r.Height; y++ {
		for x := x0; x < x0+size && x < r.Width; x++ {
			px := (float64(x) + 0.5) / float64(r.Width)
			py := 1 - (float64(y)+0.5)/float64(r.Height) // flip: image y grows down
			var acc geom.Vec3
			n := r.Samples
			rng := NewRNG(SeedPixel(r.Seed, x, y))
			for sy := 0; sy < n; sy++ {
				for sx := 0; sx < n; sx++ {
					jx := ((float64(sx) + rng.Float64()) / float64(n)) / float64(r.Width)
					jy := ((float64(sy) + rng.Float64()) / float64(n)) / float64(r.Height)
					ray := r.Camera.Ray(px, py, jx, jy, rng)
					acc = acc.Add(r.trace(ray, r.MaxDepth, rng))
				}
			}
			avg := acc.MulS(1 / float64(n*n))
			r8, g8, b8, a8 := tonemap(avg, r.Exposure)
			img.SetRGBA(x, y, color.RGBA{r8, g8, b8, a8})
		}
	}
}

// envAt returns environment radiance for a miss direction: flat color
// or the zenith/horizon gradient blended on dir.Y.
func (r *Renderer) envAt(dir geom.Vec3) geom.Vec3 {
	if !r.World.SkyGradient {
		return r.World.EnvColor
	}
	t := dir.Y*0.5 + 0.5 // -1..1 → 0..1
	return r.World.EnvBot.Add(r.World.EnvTop.Sub(r.World.EnvBot).MulS(t))
}

// trace follows one ray. depth caps reflection/refraction recursion.
func (r *Renderer) trace(ray geom.Ray, depth int, rng *RNG) geom.Vec3 {
	if depth <= 0 {
		return r.envAt(ray.Dir)
	}
	hit, obj := r.World.Intersect(ray)
	if hit == nil {
		return r.envAt(ray.Dir)
	}
	mat := &r.World.Materials[obj.MatIdx]
	albedo := mat.AlbedoAt(hit.P)
	n := hit.N // oriented against the ray
	p := hit.P

	switch mat.Kind {
	case 'e': // emissive
		return albedo.MulS(mat.Power)
	case 'g': // glass
		return r.glass(ray, hit, mat, depth, rng)
	case 'M': // metal
		var out geom.Vec3
		if depth > 1 {
			rd := reflectDir(ray.Dir, n)
			rd = perturb(rd, mat.Roughness, rng)
			refl := r.trace(offset(p, n, rd), depth-1, rng)
			out = mulV(refl, albedo)
		}
		// analytic highlights: point lights are not scene geometry, so
		// mirror reflections never catch them without this
		return out.Add(r.specular(p, n, ray.Dir.MulS(-1), mat, albedo, 1))
	}

	// matte: ambient + direct light with shadows
	out := mulV(albedo, r.World.EnvColor).MulS(r.World.Ambient)
	for i := range r.World.Lights {
		l := &r.World.Lights[i]
		ld, dist := lightDir(l, p)
		if ld == (geom.Vec3{}) {
			continue
		}
		ndotl := n.Dot(ld)
		if ndotl <= 0 {
			continue
		}
		shadowRay := offset(p, n, ld)
		trans := r.shadowTrans(shadowRay.Orig, ld, dist)
		if trans.X <= 0 && trans.Y <= 0 && trans.Z <= 0 {
			continue
		}
		atten := lightAttenuation(l, dist)
		lit := mulV(albedo, l.Color).MulS(ndotl * l.Intensity * atten)
		out = out.Add(mulV(lit, trans))
	}
	return out
}

// glass computes Fresnel-weighted reflection + refraction with TIR.
func (r *Renderer) glass(ray geom.Ray, hit *geom.Hit, mat *CompiledMaterial, depth int, rng *RNG) geom.Vec3 {
	albedo := mat.AlbedoAt(hit.P)
	if depth <= 1 {
		return mulV(r.World.EnvColor, albedo)
	}
	n := hit.N
	cosI := -ray.Dir.Dot(n) // >= 0 because hit.N opposes the ray
	etaI, etaT := 1.0, mat.IOR
	if !hit.Front {
		etaI, etaT = mat.IOR, 1.0
	}
	eta := etaI / etaT
	sin2T := eta * eta * (1 - cosI*cosI)

	// Fresnel (Schlick)
	r0 := (etaI - etaT) / (etaI + etaT)
	r0 *= r0
	fr := r0 + (1-r0)*math.Pow(1-cosI, 5)

	rd := reflectDir(ray.Dir, n)
	refl := r.trace(offset(hit.P, n, rd), depth-1, rng)

	if sin2T > 1 {
		return mulV(refl, albedo) // total internal reflection
	}
	cosT := math.Sqrt(1 - sin2T)
	refrDir := ray.Dir.MulS(eta).Add(n.MulS(eta*cosI - cosT)).Norm()
	refr := r.trace(geom.Ray{Orig: hit.P.Sub(n.MulS(1e-4)), Dir: refrDir}, depth-1, rng)
	base := mulV(refl.MulS(fr).Add(refr.MulS(1-fr)), albedo)
	return base.Add(r.specular(hit.P, n, ray.Dir.MulS(-1), mat, albedo, fr))
}

func reflectDir(d, n geom.Vec3) geom.Vec3 {
	return d.Sub(n.MulS(2 * d.Dot(n))).Norm()
}

// offset spawns a ray from p, nudged off the surface into the hemisphere
// that dir points toward.
func offset(p, n, dir geom.Vec3) geom.Ray {
	eps := 1e-4
	if dir.Dot(n) < 0 {
		eps = -eps
	}
	return geom.Ray{Orig: p.Add(n.MulS(eps)), Dir: dir}
}

// perturb roughens a mirror direction; deterministic via rng.
func perturb(dir geom.Vec3, rough float64, rng *RNG) geom.Vec3 {
	if rough <= 0 {
		return dir
	}
	for tries := 0; tries < 8; tries++ {
		v := geom.Vec3{X: rng.Float64()*2 - 1, Y: rng.Float64()*2 - 1, Z: rng.Float64()*2 - 1}
		l := v.Len()
		if l <= 1 && l > 1e-6 {
			cand := dir.Add(v.MulS(rough)).Norm()
			if cand.Dot(dir) > 0.5 { // stay within ~60° of the mirror dir
				return cand
			}
		}
	}
	return dir
}

// specular adds Blinn-Phong highlights for mirror-like materials so
// analytic (non-geometry) lights produce visible glints. k scales the
// contribution (Fresnel reflectance for glass).
func (r *Renderer) specular(p, n, viewDir geom.Vec3, mat *CompiledMaterial, albedo geom.Vec3, k float64) geom.Vec3 {
	sh := specShininess(mat.Roughness)
	var out geom.Vec3
	for i := range r.World.Lights {
		l := &r.World.Lights[i]
		ld, dist := lightDir(l, p)
		if ld == (geom.Vec3{}) {
			continue
		}
		h := ld.Add(viewDir).Norm()
		ndoth := n.Dot(h)
		if ndoth <= 0 {
			continue
		}
		atten := lightAttenuation(l, dist)
		spec := math.Pow(ndoth, sh) * l.Intensity * atten * k
		out = out.Add(mulV(l.Color, albedo).MulS(spec))
	}
	return out
}

// specShininess maps roughness to a Blinn-Phong exponent.
func specShininess(rough float64) float64 {
	if rough < 0.015 {
		rough = 0.015
	}
	sh := 2 / (rough * rough)
	if sh > 8192 {
		sh = 8192
	}
	return sh
}

// shadowTrans marches a shadow ray toward the light accumulating
// transmission: opaque hits return black; glass surfaces tint the
// light and let it through (fake, caustic-free transmission).
func (r *Renderer) shadowTrans(orig, dir geom.Vec3, dist float64) geom.Vec3 {
	trans := geom.Vec3{X: 1, Y: 1, Z: 1}
	for hop := 0; hop < 8; hop++ {
		h, obj := r.World.Intersect(geom.Ray{Orig: orig, Dir: dir})
		if h == nil || h.T >= dist {
			return trans
		}
		mat := &r.World.Materials[obj.MatIdx]
		if mat.Kind != 'g' {
			return geom.Vec3{}
		}
		trans = mulV(trans, mat.AlbedoAt(h.P)).MulS(0.85)
		orig = h.P.Add(dir.MulS(2e-4))
	}
	return trans
}

func mulV(a, b geom.Vec3) geom.Vec3 {
	return geom.Vec3{X: a.X * b.X, Y: a.Y * b.Y, Z: a.Z * b.Z}
}

func lightDir(l *LightDef, p geom.Vec3) (geom.Vec3, float64) {
	switch l.Kind {
	case 'd': // directional: dir points where light travels
		return l.Dir.MulS(-1), math.MaxFloat64
	case 's': // spot: dir points where light travels
		toL := l.Pos.Sub(p)
		dist := toL.Len()
		dir := toL.MulS(1 / dist)
		if dir.Dot(l.Dir.MulS(-1)) < l.CosAngle {
			return geom.Vec3{}, 0 // outside the cone
		}
		return dir, dist
	default: // point
		toL := l.Pos.Sub(p)
		dist := toL.Len()
		return toL.MulS(1 / dist), dist
	}
}

func lightAttenuation(l *LightDef, dist float64) float64 {
	if l.Kind == 'd' || dist == math.MaxFloat64 {
		return 1
	}
	return 1 / (1 + 0.09*dist + 0.032*dist*dist)
}

// tonemap applies exposure + Reinhard and returns 8-bit RGBA.
func tonemap(c geom.Vec3, exposure float64) (uint8, uint8, uint8, uint8) {
	f := func(x float64) uint8 {
		x *= exposure
		x = x / (1 + x) // Reinhard
		if x < 0 {
			x = 0
		}
		if x > 1 {
			x = 1
		}
		return uint8(x*255 + 0.5)
	}
	return f(c.X), f(c.Y), f(c.Z), 255
}
