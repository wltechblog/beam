package render

import (
	"errors"
	"image"
	"image/color"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/wltechblog/beam/geom"
)

// Renderer renders a compiled World with a Whitted-style integrator:
// direct lighting with shadows, mirror reflections with roughness
// perturbation, and Snell refraction with Fresnel + TIR, capped at
// MaxDepth bounces. One RNG per pixel, threaded through recursion —
// same scene + seed = byte-identical PNG on any machine.
type Renderer struct {
	World       *World
	Camera      *Camera
	Width       int
	Height      int
	Samples     int
	MaxDepth    int
	Exposure    float64
	ClipTonemap bool // POV-style: clip instead of Reinhard
	Seed        int64

	// Deadline aborts the render mid-flight when non-nil. Workers check
	// it between tiles (and between rows within a tile); the returned
	// image is nil when aborted.
	Deadline *time.Time
}

var errRenderDeadline = errors.New("render exceeded maxSeconds deadline — aborting " +
	"(scene likely oversized for the available hardware; lower samples/resolution)")

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
	aborted := false
	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range tasks {
				if aborted || (r.Deadline != nil && time.Now().After(*r.Deadline)) {
					aborted = true
					continue // drain remaining tasks without rendering
				}
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
	if aborted {
		return nil
	}
	return img
}

func (r *Renderer) renderTile(img *image.RGBA, row, col, size int) {
	x0 := col * size
	y0 := row * size
	for y := y0; y < y0+size && y < r.Height; y++ {
		if r.Deadline != nil && time.Now().After(*r.Deadline) {
			return
		}
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
			r8, g8, b8, a8 := tonemap(avg, r.Exposure, r.ClipTonemap)
			img.SetRGBA(x, y, color.RGBA{r8, g8, b8, a8})
		}
	}
}

// envAt returns environment radiance for a miss direction: flat color
// or a gradient on dir.Y. With a horizon stop (EnvMid), the ramp is
// zenith → horizon for dir.Y > 0 and horizon → below for dir.Y < 0.
func (r *Renderer) envAt(dir geom.Vec3) geom.Vec3 {
	if len(r.World.SkyStops) >= 2 {
		return skyMapAt(r.World.SkyStops, dir.Y)
	}
	if !r.World.SkyGradient {
		return r.World.EnvColor
	}
	t := dir.Y*0.5 + 0.5 // -1..1 → 0..1
	top, bot := r.World.EnvTop, r.World.EnvBot
	if r.World.EnvMid != (geom.Vec3{}) {
		if dir.Y >= 0 {
			top, bot = r.World.EnvTop, r.World.EnvMid
			t = dir.Y // 0..1 horizon→zenith
		} else {
			top, bot = r.World.EnvBot, r.World.EnvMid
			t = -dir.Y // 0..1 horizon→below
		}
		if c := r.World.SkyCurve; c > 0 && c != 1 {
			t = beamPow(t, c)
		}
	}
	return bot.Add(top.Sub(bot).MulS(t))
}

// skyMapAt samples the piecewise sky gradient at elevation dirY
// (-1..1). Segments interpolate linearly; ends clamp.
func skyMapAt(stops []SkyStop, dirY float64) geom.Vec3 {
	n := len(stops)
	if dirY <= stops[0].Y {
		return stops[0].Color
	}
	if dirY >= stops[n-1].Y {
		return stops[n-1].Color
	}
	for i := 1; i < n; i++ {
		if dirY <= stops[i].Y {
			a, b := stops[i-1], stops[i]
			t := (dirY - a.Y) / (b.Y - a.Y)
			return a.Color.Add(b.Color.Sub(a.Color).MulS(t))
		}
	}
	return stops[n-1].Color
}

// trace follows one ray, compositing fog/media over the traversed
// segment. depth caps reflection/refraction recursion.
func (r *Renderer) trace(ray geom.Ray, depth int, rng *RNG) geom.Vec3 {
	if depth <= 0 {
		return r.composite(r.envAt(ray.Dir), ray.Orig, ray.Dir, fogFar)
	}
	hit, obj := r.World.Intersect(ray)
	if hit == nil {
		return r.composite(r.envAt(ray.Dir), ray.Orig, ray.Dir, fogFar)
	}
	c := r.shade(ray, hit, obj, depth, rng)
	return r.composite(c, ray.Orig, ray.Dir, hit.T)
}

// composite applies whichever atmospheric model is authored: media
// (extinction + single scattering) when set, else fog (absorption only).
func (r *Renderer) composite(c geom.Vec3, orig, dir geom.Vec3, dist float64) geom.Vec3 {
	if r.World.Media != nil {
		return r.mediaComposite(c, orig, dir, dist)
	}
	return r.fogBlend(c, orig, dir, dist)
}

// fogFar caps fog integration for miss rays; with any sane fog distance
// the transmittance saturates long before this.
const fogFar = 1e5

// fogBlend applies exponential fog over the segment orig → orig+dir·dist.
// c is the unfogged segment radiance.
func (r *Renderer) fogBlend(c geom.Vec3, orig, dir geom.Vec3, dist float64) geom.Vec3 {
	f := r.World.Fog
	if f == nil {
		return c
	}
	od := opticalDepth(f, orig.Y, dir.Y, dist)
	tr := beamExp(-f.K * od)
	amt := 1 - tr
	if amt > f.Max {
		amt = f.Max
	}
	return c.MulS(1 - amt).Add(f.Color.MulS(amt))
}

// mediaComposite composites participating media over the segment orig →
// orig+dir·dist: in-scattered light accumulates along a deterministic
// midpoint march (Steps samples, zero RNG draws so CPU/GPU streams stay
// identical), and the surface/environment radiance c is attenuated by the
// closed-form segment transmittance. At each sample every light
// contributes lightColor·intensity·attenuation, occluded by the geometry
// shadow ray AND attenuated by the media along the light path — that
// light-path extinction is what turns occluders into visible shafts.
func (r *Renderer) mediaComposite(c geom.Vec3, orig, dir geom.Vec3, dist float64) geom.Vec3 {
	m := r.World.Media
	if dist > fogFar {
		dist = fogFar
	}
	// Cap the march at the visibility horizon: density <= 1 everywhere,
	// so transmittance >= exp(-K·t) — beyond 7/K (7 e-foldings ≈ 1e-3)
	// nothing can contribute. Without this, sky rays (dist = fogFar)
	// space their samples thousands of units apart and the visible
	// near segment where all the in-scatter lives is never sampled.
	marchDist := dist
	if cap := 7 / m.K; marchDist > cap {
		marchDist = cap
	}
	step := marchDist / float64(m.Steps)
	var acc geom.Vec3
	for i := 0; i < m.Steps; i++ {
		tMid := (float64(i) + 0.5) * step
		p := orig.Add(dir.MulS(tMid))
		sigma := mediaDensity(m, p.Y)
		if sigma <= 0 {
			continue
		}
		tTo := beamExp(-m.K * mediaOD(m, orig.Y, dir.Y, tMid))
		if tTo < 1e-4 {
			break // fully extinguished: later samples cannot contribute
		}
		slice := 1 - beamExp(-m.K*sigma*step) // extinction across this step
		insc := r.mediaInscatter(m, p)
		acc = acc.Add(mulV(insc, m.Color).MulS(tTo * slice))
	}
	tAll := beamExp(-m.K * mediaOD(m, orig.Y, dir.Y, dist))
	return acc.Add(c.MulS(tAll))
}

// mediaInscatter sums direct light arriving at p from every light,
// geometry-occluded and media-attenuated along the light path. Isotropic
// phase folded into the artist scale (beam lights are not physical
// units); the march slice weight lives in mediaComposite.
func (r *Renderer) mediaInscatter(m *MediaDef, p geom.Vec3) geom.Vec3 {
	var out geom.Vec3
	for i := range r.World.Lights {
		l := &r.World.Lights[i]
		if l.Kind == 'x' {
			// Mesh light in fog: shade from the emitter's area-weighted
			// centroid (compile-time Pos). The media march is
			// deterministic by design — zero RNG draws — so the direct
			// path's triangle sampling does not apply here; the centroid
			// gives the correct glow-scale in-scatter and ports to GPU
			// as a plain point-at-Pos light.
			toL := l.Pos.Sub(p)
			dist := toL.Len()
			if dist < 1e-9 {
				continue
			}
			sld := toL.MulS(1 / dist)
			o := p.Add(sld.MulS(1e-4))
			trans := r.shadowTrans(o, sld, dist)
			if trans.X <= 0 && trans.Y <= 0 && trans.Z <= 0 {
				continue
			}
			tm := beamExp(-m.K * mediaOD(m, p.Y, sld.Y, dist))
			out = out.Add(mulV(l.Color, trans).MulS(l.Intensity * tm))
			continue
		}
		ld, dist := lightDir(l, p)
		if ld == (geom.Vec3{}) {
			continue
		}
		o := p.Add(ld.MulS(1e-4))
		trans := r.shadowTrans(o, ld, dist)
		if trans.X <= 0 && trans.Y <= 0 && trans.Z <= 0 {
			continue
		}
		// Media transmittance along the light path. Directional lights
		// are infinitely distant — an infinite path through uniform
		// media would extinguish everything — so they skip it: model as
		// a collimated sun beam entering the scene (classic god rays).
		tm := 1.0
		if l.Kind != 'd' {
			tm = beamExp(-m.K * mediaOD(m, p.Y, ld.Y, dist))
		}
		atten := lightAttenuation(l, dist)
		out = out.Add(mulV(l.Color, trans).MulS(l.Intensity * atten * tm))
	}
	return out
}

// mediaDensity is the unitless density profile at height y — the same
// profile fog integrates in closed form (opticalDepth), evaluated at a
// point: 1 below Alt (or the POV atan ground-fog curve), exponential
// falloff above.
func mediaDensity(m *MediaDef, y float64) float64 {
	if m.Atan {
		if m.Alt == 0 {
			return 0
		}
		u := (y - m.Off) / m.Alt
		return 1 / (u*u + 1)
	}
	if y <= m.Alt {
		return 1
	}
	if m.Off > 0 {
		return beamExp(-(y - m.Alt) / m.Off)
	}
	return 0
}

// mediaOD is opticalDepth for a MediaDef (same closed-form integrals).
func mediaOD(m *MediaDef, y0, dy, dist float64) float64 {
	return opticalDepth(&FogDef{Alt: m.Alt, Off: m.Off, Atan: m.Atan}, y0, dy, dist)
}

// mediaTrans is segment transmittance through the media — used by the
// light path (shaft dimming) and by surface shadow rays (direct light
// through fog reaches surfaces dimmer).
func (r *Renderer) mediaTrans(orig, dir geom.Vec3, dist float64) float64 {
	m := r.World.Media
	if m == nil {
		return 1
	}
	return beamExp(-m.K * mediaOD(m, orig.Y, dir.Y, dist))
}

// opticalDepth integrates fog density along a segment of length dist
// starting at height y0 with vertical slope dy. Density is 1 below
// FogDef.Alt; above it, Off > 0 gives exponential falloff and Off == 0
// is a hard cutoff (POV fog_offset 0). Closed form — no sampling, so
// renders stay deterministic.
func opticalDepth(f *FogDef, y0, dy, dist float64) float64 {
	if dist <= 0 {
		return 0
	}
	const maxD = 1e6
	if dist > maxD {
		dist = maxD
	}
	alt := f.Alt
	var total float64
	addDense := func(a, b float64) {
		if a < 0 {
			a = 0
		}
		if b > dist {
			b = dist
		}
		if b > a {
			total += b - a
		}
	}
	addExp := func(a, b float64) {
		if a < 0 {
			a = 0
		}
		if b > dist {
			b = dist
		}
		if b <= a {
			return
		}
		if math.Abs(dy) < 1e-12 {
			total += (b - a) * beamExp(-(y0-alt)/f.Off)
			return
		}
		ga := beamExp(-(y0 + dy*a - alt) / f.Off)
		gb := beamExp(-(y0 + dy*b - alt) / f.Off)
		total += f.Off / dy * (ga - gb)
	}
	if math.Abs(dy) < 1e-12 {
		if y0 <= alt {
			return dist
		}
		if f.Off > 0 {
			return dist * beamExp(-(y0-alt)/f.Off)
		}
		return 0
	}
	if f.Atan {
		// POV fog_type 2: density 1/(u^2+1), u = (y-Off)/Alt. The exact
		// line integral is Alt*(atan(u2) - atan(u1))/dy — see POV-Ray
		// source/core/render/trace.cpp ComputeGroundFogDepth.
		u1 := (y0 - f.Off) / alt
		u2 := (y0 + dy*dist - f.Off) / alt
		if math.Abs(dy) < 1e-12 {
			return dist / (u1*u1 + 1)
		}
		return alt * (beamAtan(u2) - beamAtan(u1)) / dy
	}
	sStar := (alt - y0) / dy // parameter where y crosses Alt
	if f.Off > 0 {
		if dy > 0 { // ascending: dense below Alt, sparse above
			addDense(0, sStar)
			addExp(sStar, dist)
		} else { // descending: sparse above Alt, dense below
			addExp(0, sStar)
			addDense(sStar, dist)
		}
		return total
	}
	// hard cutoff: density only while below Alt
	if dy > 0 {
		addDense(0, sStar)
	} else {
		addDense(sStar, dist)
	}
	return total
}

// albedoAt resolves the surface albedo: texture at the hit's
// barycentric-interpolated UV when the mesh is textured, else the
// pattern/solid path. obj may be nil (glass path) — then patterns only.
func (r *Renderer) albedoAt(mat *CompiledMaterial, hit *geom.Hit, obj *Object3D) geom.Vec3 {
	if mat.Tex == nil || obj == nil {
		return mat.AlbedoAt(hit.P)
	}
	switch obj.Kind {
	case 'm':
		if obj.Mesh != nil && len(obj.Mesh.UVs) > 0 {
			if a, b, c, ok := obj.Mesh.TriUV(hit.Tri); ok {
				w0 := 1 - hit.U - hit.V
				u := a.U*w0 + b.U*hit.U + c.U*hit.V
				v := a.V*w0 + b.V*hit.U + c.V*hit.V
				return mat.AlbedoAtUV(u, v, hit.P)
			}
		}
	case 'p':
		// Planes have no authored UVs — texture them with automatic
		// planar mapping: project onto the plane's dominant axis and
		// use the hit position's other two coordinates (world units =
		// UV units, wrap-addressed by the sampler; tiled textures from
		// free-asset packs cover floors/walls with zero scene authoring).
		n := obj.PlaneN
		switch {
		case n.Y*n.Y >= n.X*n.X && n.Y*n.Y >= n.Z*n.Z: // floor/ceiling
			return mat.AlbedoAtUV(hit.P.X, hit.P.Z, hit.P)
		case n.X*n.X >= n.Z*n.Z: // wall facing x
			return mat.AlbedoAtUV(hit.P.Z, hit.P.Y, hit.P)
		default: // wall facing z
			return mat.AlbedoAtUV(hit.P.X, hit.P.Y, hit.P)
		}
	}
	return mat.AlbedoAt(hit.P)
}

// shade computes the surface response at a hit (no fog).
func (r *Renderer) shade(ray geom.Ray, hit *geom.Hit, obj *Object3D, depth int, rng *RNG) geom.Vec3 {
	mat := &r.World.Materials[obj.MatIdx]
	albedo := r.albedoAt(mat, hit, obj)
	n := hit.N // oriented against the ray
	p := hit.P

	switch mat.Kind {
	case 'e': // emissive
		return albedo.MulS(mat.Power)
	case 'g': // glass
		return r.glass(ray, hit, mat, depth, rng)
	case 'M': // metal
		var reflOut geom.Vec3
		if depth > 1 {
			rd := reflectDir(ray.Dir, n)
			rd = perturb(rd, mat.Roughness, rng)
			refl := r.trace(offset(p, n, rd), depth-1, rng)
			if mat.Untinted {
				reflOut = refl.MulS(mat.Reflectivity)
			} else {
				reflOut = mulV(refl, albedo).MulS(mat.Reflectivity)
			}
		}
		// POV default reflection (untinted) also shows diffuse under it,
		// scaled by remaining energy; keep specular highlight on top.
		var out geom.Vec3
		if mat.Untinted {
			// matte-style diffuse term scaled by (1 - reflectivity)
			out = r.directLight(p, n, albedo, mat.Diffuse, 1-mat.Reflectivity, rng)
		}
		out = out.Add(reflOut)
		if mat.SpecularK > 0 {
			out = out.Add(r.specular(p, n, ray.Dir.MulS(-1), mat, geom.Vec3{X: 1, Y: 1, Z: 1}, mat.SpecularK))
		} else {
			out = out.Add(r.specular(p, n, ray.Dir.MulS(-1), mat, geom.Vec3{X: 1, Y: 1, Z: 1}, 1))
		}
		return out
	}

	// matte: ambient + direct light with shadows
	out := r.directLight(p, n, albedo, mat.Diffuse, 1, rng)
	if mat.SpecularK > 0 {
		// POV finish specular: highlight takes light color, NOT pigment tint
		out = out.Add(r.specular(p, n, ray.Dir.MulS(-1), mat, geom.Vec3{X: 1, Y: 1, Z: 1}, mat.SpecularK))
	}
	return out
}

// directLight computes ambient + the direct-light sum with shadows for a
// diffuse surface. diffuseK scales the direct term (mat.Diffuse for matte,
// Diffuse*(1-Reflectivity) for untinted metal). Lights with Radius > 0 are
// spherical area lights: the shadow term is averaged over shadowSamples
// jittered points on the visible sphere; other values exactly preserve the
// historic hard-shadow path. Draw order is fixed (2 RNG draws per jitter)
// so CPU and GPU streams stay in lockstep.
func (r *Renderer) directLight(p, n, albedo geom.Vec3, diffuseK, energyScale float64, rng *RNG) geom.Vec3 {
	out := mulV(albedo, r.World.AmbientColor).MulS(r.World.Ambient * energyScale)
	for i := range r.World.Lights {
		l := &r.World.Lights[i]
		if l.Kind == 'x' {
			// Mesh light: area-weighted triangle sampling. Same budget
			// as spherical area lights (shadowSamples rays), same fixed
			// draw order (3 per sample) so CPU/GPU streams stay in
			// lockstep. ndotl uses the SAMPLE's face normal — the
			// emitter's back faces do not light the surface.
			ns := r.World.ShadowSamples
			if ns < 1 {
				ns = 1
			}
			var acc geom.Vec3
			var anyLit bool
			for js := 0; js < ns; js++ {
				u1 := rng.Float64()
				u2 := rng.Float64()
				u3 := rng.Float64()
				sp, sn := l.SampleMeshLight(u1, u2, u3)
				toL := sp.Sub(p)
				dist := toL.Len()
				if dist < 1e-9 {
					continue
				}
				sld := toL.MulS(1 / dist)
				if n.Dot(sld) <= 0 || sn.Dot(sld) >= 0 {
					continue // surface faces away, or sample faces away
				}
				shadowRay := offset(p, n, sld)
				trans := r.shadowTrans(shadowRay.Orig, sld, dist)
				if trans.X <= 0 && trans.Y <= 0 && trans.Z <= 0 {
					continue
				}
				atten := lightAttenuation(l, dist)
				lit := mulV(albedo, l.Color).MulS(n.Dot(sld) * l.Intensity * diffuseK * energyScale * atten)
				acc = acc.Add(mulV(lit, trans))
				anyLit = true
			}
			if anyLit {
				out = out.Add(acc.MulS(1 / float64(ns)))
			}
			continue
		}
		ld, dist := lightDir(l, p)
		if ld == (geom.Vec3{}) {
			continue
		}
		ndotl := n.Dot(ld)
		if ndotl <= 0 {
			continue
		}

		trans := geom.Vec3{X: 1, Y: 1, Z: 1}
		if !l.Shadowless {
			if l.Radius > 0 {
				nSamples := r.World.ShadowSamples
				if nSamples < 1 {
					nSamples = 1
				}
				var acc geom.Vec3
				for js := 0; js < nSamples; js++ {
					// Deterministic point on the light's visible disk: two
					// RNG draws -> concentric-map disk sample, projected to
					// the sphere cap facing p. Fixed draw count keeps the
					// CPU and GPU RNG streams identical.
					u1 := rng.Float64()
					u2 := rng.Float64()
					sp := sampleLightPoint(l, p, u1, u2)
					sld, sdist := toLightDir(sp, p)
					shadowRay := offset(p, n, sld)
					acc = acc.Add(r.shadowTrans(shadowRay.Orig, sld, sdist))
				}
				trans = acc.MulS(1 / float64(nSamples))
			} else {
				shadowRay := offset(p, n, ld)
				trans = r.shadowTrans(shadowRay.Orig, ld, dist)
			}
			// direct light also attenuates through the media layer on
			// its way to the surface (thick fog dims surfaces).
			// Directional lights skip it — same sun-beam semantics as
			// mediaInscatter.
			if r.World.Media != nil && l.Kind != 'd' {
				mt := r.mediaTrans(p, ld, dist)
				trans = trans.MulS(mt)
			}
			if trans.X <= 0 && trans.Y <= 0 && trans.Z <= 0 {
				continue
			}
		}
		atten := lightAttenuation(l, dist)
		lit := mulV(albedo, l.Color).MulS(ndotl * l.Intensity * diffuseK * energyScale * atten)
		out = out.Add(mulV(lit, trans))
	}
	return out
}

// toLightDir returns the unit direction and distance from p to q.
func toLightDir(q, p geom.Vec3) (geom.Vec3, float64) {
	d := q.Sub(p)
	l := d.Len()
	return d.MulS(1 / l), l
}

// sampleLightPoint picks a deterministic point on the spherical light l as
// seen from shading point p, from two uniform RNG draws. u1,u2 -> concentric
// square->disk map (no rejection loop) -> scaled by the angular radius of the
// light as seen from p -> rotated into the perpendicular frame around the
// light direction. The sample lands on the disk through the light CENTER
// facing p (cap approximation: for realistic light/scene scales the visible
// disk and the cap differ imperceptibly but the disk keeps the math exact,
// loop-free and CPU/GPU portable).
func sampleLightPoint(l *LightDef, p geom.Vec3, u1, u2 float64) geom.Vec3 {
	// frame around axis p -> light center
	axis := l.Pos.Sub(p)
	dist := axis.Len()
	axisN := axis.MulS(1 / dist)

	// visible angular radius of the sphere
	sinMax := l.Radius / dist
	if sinMax > 1 {
		sinMax = 1
	}
	// sample uniformly on the disk of that angular radius (concentric map)
	// r = sin(theta) gives uniform-on-disk in angle space
	r := sinMax * sqrtOneMinus(u1) // r = sqrt(1-u1) scaled — uniform disk
	phi := 2 * math.Pi * u2
	dx := r * math.Cos(phi)
	dy := r * math.Sin(phi)

	// perpendicular frame (axisN has one near-zero component; pick the
	// smallest to build a stable cross)
	var ref geom.Vec3
	if axisN.X < axisN.Y && axisN.X < axisN.Z {
		ref = geom.Vec3{X: 1, Y: 0, Z: 0}
	} else if axisN.Y < axisN.Z {
		ref = geom.Vec3{X: 0, Y: 1, Z: 0}
	} else {
		ref = geom.Vec3{X: 0, Y: 0, Z: 1}
	}
	u := axisN.Cross(ref).Norm()
	v := axisN.Cross(u)

	// direction from p through the sampled disk point. dir is unit by
	// construction: axis*cos(theta) + (u cos(phi) + v sin(phi)) * sin(theta)
	// with sin(theta) = r. The ray's perpendicular offset from the light
	// center is therefore dist*r (NOT dist*dx) — that sets the secant
	// discriminant below.
	cosT := sqrtOneMinus(r * r)
	dirN := axisN.MulS(cosT).Add(u.MulS(dx)).Add(v.MulS(dy)).Norm()
	// distance from p to the sphere surface along dirN (near secant root):
	// t = dist*cosT - sqrt(R^2 - (dist*r)^2)
	disc := dist * dist * r * r
	rad2 := l.Radius * l.Radius
	if disc > rad2 {
		disc = rad2
	}
	t := dist*cosT - math.Sqrt(rad2-disc)
	return p.Add(dirN.MulS(t))
}

// sqrtOneMinus(x) = sqrt(1-x) without cancellation near x=1.
func sqrtOneMinus(x float64) float64 {
	if x >= 1 {
		return 0
	}
	return math.Sqrt(1 - x)
}

// glass computes Fresnel-weighted reflection + refraction with TIR.
func (r *Renderer) glass(ray geom.Ray, hit *geom.Hit, mat *CompiledMaterial, depth int, rng *RNG) geom.Vec3 {
	albedo := r.albedoAt(mat, hit, nil)
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
	if mat.Reflectivity > r0 {
		r0 = mat.Reflectivity // POV-style reflection {min, max fresnel}
	}
	fr := r0 + (1-r0)*beamPow(1-cosI, 5)

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
		spec := beamPow(ndoth, sh) * l.Intensity * atten * k
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
		if mat.Kind == 'e' {
			// Emissive surfaces are LIGHT SOURCES, not occluders: a
			// shadow ray that hits one has reached (or passed through) an
			// emitter — e.g. the ray toward a mesh light's centroid or a
			// radius light's center always crosses the emitter's own
			// surface. Continue past it at full transmission.
			orig = h.P.Add(dir.MulS(2e-4))
			continue
		}
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
	if l.Kind == 'd' || dist == math.MaxFloat64 || l.NoFalloff {
		return 1
	}
	return 1 / (1 + 0.09*dist + 0.032*dist*dist)
}

// tonemap applies exposure + (Reinhard or clip) + sRGB encoding and returns 8-bit RGBA.
func tonemap(c geom.Vec3, exposure float64, clip bool) (uint8, uint8, uint8, uint8) {
	f := func(x float64) uint8 {
		x *= exposure
		if !clip {
			x = x / (1 + x) // Reinhard
		}
		if x < 0 {
			x = 0
		}
		if x > 1 {
			x = 1
		}
		x = linearToSRGB(x)
		return uint8(x*255 + 0.5)
	}
	return f(c.X), f(c.Y), f(c.Z), 255
}

// linearToSRGB encodes a linear-light value in [0,1] with the standard
// sRGB transfer function (matching POV-Ray's default output encoding).
func linearToSRGB(x float64) float64 {
	if x <= 0.0031308 {
		return 12.92 * x
	}
	return 1.055*beamPow(x, 1/2.4) - 0.055
}
