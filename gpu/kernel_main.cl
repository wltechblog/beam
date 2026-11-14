// kernel_main.cl — Beam GPU kernel, part 3: phase-machine DFS + entry point.
//
// The explicit frame stack replicates renderer.go recursion EXACTLY:
//   metal: perturb (RNG) → reflect trace (subtree RNG) → specular (no RNG)
//   glass: reflect trace → [TIR check] → refract trace → specular
//   matte: no RNG
// RNG is one stream per pixel threaded through the whole DFS — matching
// the CPU's single rng argument through recursion.

#define PH_ENTER 0
#define PH_METAL_WAIT 1
#define PH_GLASS_REFL_WAIT 2
#define PH_GLASS_REFR_WAIT 3

typedef struct {
	int phase;
	v3 ro, rd;
	int depth;
	// hit
	int hit;
	bd t;
	v3 p, n;
	int front;
	int matIdx;
	int objIdx;
	// scratch
	v3 albedo;
	v3 reflRes;   // glass: reflection result
	v3 childRes;  // result delivered by completed child frame
	v3 out;       // metal/matte partial output
} fframe;

// scene buffer bundle passed everywhere
typedef struct {
	__global const bd *objA;     // sphere center | plane point (3)
	__global const bd *objB;     // sphere radius (1) | plane normal (3)
	__global const int *objKind; // 0 s, 1 p, 2 m
	__global const int *objRoot; // mesh: first node index, -1 none
	__global const int *objNorms;
	__global const int *objMat;
	int nObjs;
	__global const bd *nodeBMin, *nodeBMax;
	__global const int *nodeLeft, *nodeRight, *nodeCount;
	__global const bd *triV; // 9/tri in flattened triOrder layout
	__global const bd *triN; // 9/tri vertex normals (flattened layout)
	__global const bd *matAlb;
	__global const bd *matP0, *matP1;
	__global const bd *matPar;   // 9: rough ior refl power diffuse specK pscale pfrom pto
	__global const int *matFlags; // 4: kind ptype axis untinted
	int nMats;
	__global const bd *lPos, *lDir, *lCol, *lPar; // lPar 2: intensity cosAngle
	__global const int *lFlags;                   // 3: kind noFalloff shadowless
	int nLights;
	// mesh lights (lFlags kind 3): concatenated triangle soups + area
	// CDFs + per-light total area + (float-start, tri-count) offsets
	__global const bd *mlTri;
	__global const bd *mlCDF;
	__global const bd *mlArea;
	__global const int *mlStart;
	// textures: per-tri UVs (6/tri in triOrder), texture table info
	// (w,h,pixOffset per texture), per-material texture index (-1 none),
	// concatenated texel data (linear rgb, row-major from top)
	__global const bd *triUV;
	__global const int *texInfo;
	__global const int *matTex;
	__global const bd *texPix;
	int shadowSamples; // per-light shadow rays for radius lights (cfg[8])
	__global const bd *skyStops; // 4/stop: y r g b
	int nStops;
	v3 envColor, envTop, envMid, envBot;
	int skyGradient;
	bd skyCurve;
	bd ambient;
	v3 ambientColor;
	int fogEnabled;
	v3 fogColor;
	bd fogK, fogAlt, fogOff, fogMax;
	int fogAtan;
	// participating media (single scattering). When mediaMode is set the
	// fogBuf/envFlags slots carry MEDIA parameters instead of fog
	// (compile.go nils Fog when Media is authored — slots never collide).
	int mediaMode;   // envFlags[5]
	int mediaAtan;   // envFlags[3]
	int mediaSteps;  // envFlags[4]
	v3 mediaColor;   // fogBuf[0..2]
	bd mediaK;       // fogBuf[3]
	bd mediaAlt;     // fogBuf[4]
	bd mediaOff;     // fogBuf[5]
} scenebuf;


// len_check: textured-material-on-untextured-mesh guard. A mesh without
// authored UVs uploads all-zero triUV rows; sampling at a constant (0,0)
// would paint one texel over the whole surface. Exactly-zero corner UVs
// for all three vertices are degenerate in texture space (a real UV map
// collapses the triangle to a point) — treat as "no UVs" and fall back to
// the untextured albedo. kind2==0 (mesh) is the only caller.
static int len_check(scenebuf *S, khit h) {
	int o6 = h.tri * 6;
	if (S->triUV[o6+0] == 0 && S->triUV[o6+1] == 0 &&
	    S->triUV[o6+2] == 0 && S->triUV[o6+3] == 0 &&
	    S->triUV[o6+4] == 0 && S->triUV[o6+5] == 0) return 0;
	return 1;
}

// sample_mesh_light: area-weighted triangle sampling for mesh lights —
// exact port of LightDef.SampleMeshLight. u1 = CDF binary search,
// u2/u3 = sqrt-barycentric jitter. Returns point + face normal.
static void sample_mesh_light(__global const bd *mlTri, __global const bd *mlCDF,
                              __global const bd *mlArea, __global const int *mlStart,
                              int li, bd u1, bd u2, bd u3, v3 *outP, v3 *outN) {
	int fstart = mlStart[li * 2];
	int ntri = mlStart[li * 2 + 1];
	bd x = u1 * mlArea[li];
	int lo = 0, hi = ntri - 1;
	while (lo < hi) {
		int mid = (lo + hi) >> 1;
		if (mlCDF[fstart + mid] < x) lo = mid + 1; else hi = mid;
	}
	int o = (fstart + lo) * 9;
	__global const bd *tv = mlTri;
	bd su = sqrt(u2);
	bd b0 = 1 - su, b1 = su * (1 - u3), b2 = su * u3;
	outP->x = tv[o+0]*b0 + tv[o+3]*b1 + tv[o+6]*b2;
	outP->y = tv[o+1]*b0 + tv[o+4]*b1 + tv[o+7]*b2;
	outP->z = tv[o+2]*b0 + tv[o+5]*b1 + tv[o+8]*b2;
	v3 e1 = v3_(tv[o+3]-tv[o+0], tv[o+4]-tv[o+1], tv[o+5]-tv[o+2]);
	v3 e2 = v3_(tv[o+6]-tv[o+0], tv[o+7]-tv[o+1], tv[o+8]-tv[o+2]);
	*outN = vnorm(vcross(e1, e2));
}

// ---------- participating media (exact port of renderer.go) ----------

// media_density: unitless density profile at height y. exp profile: 1
// below Alt, exp falloff above. atan profile: POV ground-fog curve
// 1/(u^2+1). (CPU mediaDensity handles both; so does this.)
static bd media_density(bd y, bd alt, bd off, int atanProfile) {
	if (atanProfile) {
		if (alt == 0) return 0;
		bd u = (y - off) / alt;
		return 1 / (u * u + 1);
	}
	if (y <= alt) return 1;
	if (off > 0) return gp_exp_use(-(y - alt) / off);
	return 0;
}

// media_inscatter: direct light at p from every light, geometry-occluded
// (8-hop glass shadow march) and media-attenuated along the light path.
// Directional lights skip light-path transmittance (sun-beam semantics).
static v3 media_inscatter(scenebuf *S, v3 p) {
	v3 out = v3_(0, 0, 0);
	for (int i = 0; i < S->nLights; i++) {
		klightdir L = light_dir(S->lPos, S->lDir, S->lPar, S->lFlags, i, p);
		if (!L.ok) continue;
		v3 o = vadd(p, vmuls(L.ld, 1e-4));
		v3 trans = v3_(1, 1, 1);
		for (int hop = 0; hop < 8; hop++) {
			khit h = world_intersect(o, L.ld, S->objA, S->objB, S->objKind, S->objRoot,
			                         S->objNorms, S->nObjs, S->nodeBMin, S->nodeBMax,
			                         S->nodeLeft, S->nodeRight, S->nodeCount, S->triV, S->triN);
			if (!h.hit || h.t >= L.dist) break;
			int m2 = S->objMat[h.objIdx];
			if (S->matFlags[m2 * 4 + 0] == 3) { // emissive: light source, not occluder
				o = vadd(h.p, vmuls(L.ld, 2e-4));
				continue;
			}
			if (S->matFlags[m2 * 4 + 0] != 2) { trans = v3_(0, 0, 0); break; }
			v3 alb2 = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, m2, h.p);
			trans = vmuls(vmul(trans, alb2), 0.85);
			o = vadd(h.p, vmuls(L.ld, 2e-4));
		}
		if (trans.x <= 0 && trans.y <= 0 && trans.z <= 0) continue;
		bd tm = 1;
		if (S->lFlags[i * 3] != 1) { // not directional
			tm = gp_exp_use(-S->mediaK * optical_depth(p.y, L.ld.y, L.dist, S->mediaAlt, S->mediaOff, S->mediaAtan));
		}
		bd atten = light_atten(S->lPar, S->lFlags, i, L.dist);
		v3 lc = v3_(S->lCol[i*3], S->lCol[i*3+1], S->lCol[i*3+2]);
		out = vadd(out, vmuls(vmul(lc, trans), S->lPar[i * 4 + 0] * atten * tm));
	}
	return out;
}

// media_composite: in-scatter accumulates along a deterministic midpoint
// march (mediaSteps samples, ZERO rng draws — CPU/GPU RNG streams stay
// identical), surface radiance c attenuated by closed-form transmittance.
// Includes the 7/K visibility-horizon march cap from the CPU version.
static v3 media_composite(scenebuf *S, v3 c, v3 orig, v3 dir, bd dist) {
	if (dist > (bd)100000.0) dist = (bd)100000.0;
	bd marchDist = dist;
	bd cap = 7 / S->mediaK;
	if (marchDist > cap) marchDist = cap;
	int steps = S->mediaSteps;
	if (steps < 1) steps = 1;
	bd step = marchDist / (bd)steps;
	v3 acc = v3_(0, 0, 0);
	for (int i = 0; i < steps; i++) {
		bd tMid = ((bd)i + 0.5) * step;
		v3 p = vadd(orig, vmuls(dir, tMid));
		bd sigma = media_density(p.y, S->mediaAlt, S->mediaOff, S->mediaAtan);
		if (sigma <= 0) continue;
		bd tTo = gp_exp_use(-S->mediaK * optical_depth(orig.y, dir.y, tMid, S->mediaAlt, S->mediaOff, S->mediaAtan));
		if (tTo < 1e-4) break; // fully extinguished: later samples cannot contribute
		bd slice = 1 - gp_exp_use(-S->mediaK * sigma * step);
		v3 insc = media_inscatter(S, p);
		acc = vadd(acc, vmuls(vmul(insc, S->mediaColor), tTo * slice));
	}
	bd tAll = gp_exp_use(-S->mediaK * optical_depth(orig.y, dir.y, dist, S->mediaAlt, S->mediaOff, S->mediaAtan));
	return vadd(acc, vmuls(c, tAll));
}

// apply_atmos dispatches to whichever atmospheric model is authored.
// Replaces every bare fog_blend call site in trace_stack.
static v3 apply_atmos(scenebuf *S, v3 c, v3 orig, v3 dir, bd dist) {
	if (S->mediaMode) return media_composite(S, c, orig, dir, dist);
	if (S->fogEnabled) return fog_blend(c, orig, dir, dist, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
	return c;
}

// forward decls
static v3 trace_stack(scenebuf *S, v3 ro, v3 rd, int depth, beam_rng *rng);

// diffuse+shadow loop shared by matte and untinted metal.
// energyScale multiplies the direct term ((1-reflectivity) for metal, 1 matte).
static v3 direct_light(scenebuf *S, v3 p, v3 n, v3 albedo, int m, bd diffuseK, bd energyScale, beam_rng *rng) {
	v3 out = vmul(albedo, S->ambientColor);
	out = vmuls(out, S->ambient * energyScale);
	for (int i = 0; i < S->nLights; i++) {
		if (S->lFlags[i * 3] == 3) { // mesh light: area-weighted triangle sampling
			int ns = S->shadowSamples;
			if (ns < 1) ns = 1;
			v3 acc = v3_(0, 0, 0);
			int anyLit = 0;
			for (int js = 0; js < ns; js++) {
				bd u1 = rng_float(rng);
				bd u2 = rng_float(rng);
				bd u3 = rng_float(rng);
				v3 sp, sn;
				sample_mesh_light(S->mlTri, S->mlCDF, S->mlArea, S->mlStart, i, u1, u2, u3, &sp, &sn);
				v3 toL = vsub(sp, p);
				bd sdist = sqrt(vdot(toL, toL));
				if (sdist < 1e-9) continue;
				v3 sld = vmuls(toL, 1 / sdist);
				if (vdot(n, sld) <= 0 || vdot(sn, sld) >= 0) continue;
				v3 orig = vadd(p, vmuls(n, vdot(sld, n) < 0 ? -1e-4 : 1e-4));
				v3 sampleTrans = v3_(1, 1, 1);
				for (int hop = 0; hop < 8; hop++) {
					khit h = world_intersect(orig, sld, S->objA, S->objB, S->objKind, S->objRoot,
					                         S->objNorms, S->nObjs, S->nodeBMin, S->nodeBMax,
					                         S->nodeLeft, S->nodeRight, S->nodeCount, S->triV, S->triN);
					if (!h.hit || h.t >= sdist) break;
					int m2 = S->objMat[h.objIdx];
					if (S->matFlags[m2 * 4 + 0] == 3) { // emissive: light source, not occluder
						orig = vadd(h.p, vmuls(sld, 2e-4));
						continue;
					}
					if (S->matFlags[m2 * 4 + 0] != 2) { sampleTrans = v3_(0,0,0); break; }
					v3 alb2 = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, m2, h.p);
					sampleTrans = vmuls(vmul(sampleTrans, alb2), 0.85);
					orig = vadd(h.p, vmuls(sld, 2e-4));
				}
				if (S->mediaMode) {
					bd mt = gp_exp_use(-S->mediaK * optical_depth(p.y, sld.y, sdist, S->mediaAlt, S->mediaOff, S->mediaAtan));
					sampleTrans = vmuls(sampleTrans, mt);
				}
				if (sampleTrans.x <= 0 && sampleTrans.y <= 0 && sampleTrans.z <= 0) continue;
				bd atten = light_atten(S->lPar, S->lFlags, i, sdist);
				bd scale = vdot(n, sld) * S->lPar[i * 4 + 0] * diffuseK * energyScale * atten;
				v3 lc = v3_(S->lCol[i*3], S->lCol[i*3+1], S->lCol[i*3+2]);
				acc = vadd(acc, vmuls(vmul(vmul(albedo, lc), sampleTrans), scale));
				anyLit = 1;
			}
			if (anyLit) out = vadd(out, vmuls(acc, 1.0 / (bd)ns));
			continue;
		}
		klightdir L = light_dir(S->lPos, S->lDir, S->lPar, S->lFlags, i, p);
		if (!L.ok) continue;
		bd ndotl = vdot(n, L.ld);
		if (ndotl <= 0) continue;
		bd radius = S->lPar[i * 4 + 2];
		int shadowless = S->lFlags[i * 3 + 2];
		v3 trans = v3_(1, 1, 1);
		if (!shadowless) {
			if (radius > 0) {
				// spherical area light: average shadow rays over jittered
				// points on the visible sphere (2 rng draws each)
				int ns = S->shadowSamples;
				if (ns < 1) ns = 1;
				v3 acc = v3_(0, 0, 0);
				v3 lightPos = v3_(S->lPos[i*3], S->lPos[i*3+1], S->lPos[i*3+2]);
				for (int js = 0; js < ns; js++) {
					bd u1 = rng_float(rng);
					bd u2 = rng_float(rng);
					v3 sp = sample_light_point(lightPos, radius, p, u1, u2);
					v3 toL = vsub(sp, p);
					bd sdist = sqrt(vdot(toL, toL));
					v3 sld = vmuls(toL, 1 / sdist);
					v3 orig = vadd(p, vmuls(n, vdot(sld, n) < 0 ? -1e-4 : 1e-4));
					v3 sampleTrans = v3_(1, 1, 1); // unobstructed = full
					for (int hop = 0; hop < 8; hop++) {
						khit h = world_intersect(orig, sld, S->objA, S->objB, S->objKind, S->objRoot,
						                         S->objNorms, S->nObjs, S->nodeBMin, S->nodeBMax,
						                         S->nodeLeft, S->nodeRight, S->nodeCount, S->triV, S->triN);
						if (!h.hit || h.t >= sdist) break;
						int m2 = S->objMat[h.objIdx];
												if (S->matFlags[m2 * 4 + 0] == 3) { // emissive: light source, not occluder
							orig = vadd(h.p, vmuls(sld, 2e-4));
							continue;
						}
						if (S->matFlags[m2 * 4 + 0] != 2) { sampleTrans = v3_(0,0,0); break; }
						v3 alb2 = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, m2, h.p);
						sampleTrans = vmuls(vmul(sampleTrans, alb2), 0.85);
						orig = vadd(h.p, vmuls(sld, 2e-4));
					}
					acc = vadd(acc, sampleTrans);
				}
				trans = vmuls(acc, 1.0 / (bd)ns);
			} else {
				v3 orig = vadd(p, vmuls(n, vdot(L.ld, n) < 0 ? -1e-4 : 1e-4));
				// shadow march
				for (int hop = 0; hop < 8; hop++) {
					khit h = world_intersect(orig, L.ld, S->objA, S->objB, S->objKind, S->objRoot,
					                         S->objNorms, S->nObjs, S->nodeBMin, S->nodeBMax,
					                         S->nodeLeft, S->nodeRight, S->nodeCount, S->triV, S->triN);
					if (!h.hit || h.t >= L.dist) break;
					int m2 = S->objMat[h.objIdx];
										if (S->matFlags[m2 * 4 + 0] == 3) { // emissive: light source, not occluder
						orig = vadd(h.p, vmuls(L.ld, 2e-4));
						continue;
					}
					if (S->matFlags[m2 * 4 + 0] != 2) { trans = v3_(0, 0, 0); break; }
					v3 alb2 = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, m2, h.p);
					trans = vmuls(vmul(trans, alb2), 0.85);
					orig = vadd(h.p, vmuls(L.ld, 2e-4));
				}
			}
			// direct light also attenuates through the media layer on
			// its way to the surface (thick fog dims surfaces).
			// Directional lights skip it — same sun-beam semantics as
			// media_inscatter.
			if (S->mediaMode && S->lFlags[i * 3] != 1) {
				bd mt = gp_exp_use(-S->mediaK * optical_depth(p.y, L.ld.y, L.dist, S->mediaAlt, S->mediaOff, S->mediaAtan));
				trans = vmuls(trans, mt);
			}
			if (trans.x <= 0 && trans.y <= 0 && trans.z <= 0) continue;
		}
		bd atten = light_atten(S->lPar, S->lFlags, i, L.dist);
		bd s = ndotl * S->lPar[i * 4] * diffuseK * energyScale * atten;
		v3 lit = vmuls(vmul(albedo, v3_(S->lCol[i*3], S->lCol[i*3+1], S->lCol[i*3+2])), s);
		out = vadd(out, vmul(lit, trans));
	}
	return out;
}

// specular with caller-supplied tint + strength k
static v3 spec_loop(scenebuf *S, v3 p, v3 n, v3 viewDir, int m, v3 tint, bd k) {
	bd rough = S->matPar[m * 9 + 0];
	if (rough < 0.015) rough = 0.015;
	bd sh = 2 / (rough * rough);
	if (sh > 8192) sh = 8192;
	v3 out = v3_(0, 0, 0);
	for (int i = 0; i < S->nLights; i++) {
		klightdir L = light_dir(S->lPos, S->lDir, S->lPar, S->lFlags, i, p);
		if (!L.ok) continue;
		v3 h = vnorm(vadd(L.ld, viewDir));
		bd ndoth = vdot(n, h);
		if (ndoth <= 0) continue;
		bd atten = light_atten(S->lPar, S->lFlags, i, L.dist);
		bd spec = gp_pow_use(ndoth, sh) * S->lPar[i * 4] * atten * k;
		out = vadd(out, vmuls(vmul(v3_(S->lCol[i*3], S->lCol[i*3+1], S->lCol[i*3+2]), tint), spec));
	}
	return out;
}

static v3 shade_result(scenebuf *S, fframe *f, v3 reflRes, v3 refrRes, int haveRefr, beam_rng *rng) {
	(void)rng;
	int m = f->matIdx;
	int kind = S->matFlags[m * 4 + 0];
	v3 albedo = f->albedo;
	v3 n = f->n;
	v3 p = f->p;
	v3 viewDir = vmuls(f->rd, -1);

	if (kind == 3) { // emissive
		return vmuls(albedo, S->matPar[m * 9 + 3]); // power
	}

	if (kind == 2) { // glass (dead path — combine inlined in state machine)
		v3 refl = reflRes;
		if (!haveRefr) return vmul(refl, albedo); // TIR
		bd fr = f->out.x;
		return vmul(vadd(vmuls(refl, fr), vmuls(refrRes, 1 - fr)), albedo);
	}

	if (kind == 1) { // metal
		v3 reflOut = reflRes;
		bd refl = S->matPar[m * 9 + 2];
		int untinted = S->matFlags[m * 4 + 3];
		v3 rOut;
		if (untinted) rOut = vmuls(reflOut, refl);
		else rOut = vmuls(vmul(reflOut, albedo), refl);
		v3 out = v3_(0, 0, 0);
		if (untinted) {
			out = direct_light(S, p, n, albedo, m, S->matPar[m * 9 + 4], 1 - refl, rng);
		}
		out = vadd(out, rOut);
		bd sk = S->matPar[m * 9 + 5];
		if (sk > 0) out = vadd(out, spec_loop(S, p, n, viewDir, m, v3_(1, 1, 1), sk));
		else out = vadd(out, spec_loop(S, p, n, viewDir, m, v3_(1, 1, 1), 1));
		return out;
	}

	// matte
	v3 out = direct_light(S, p, n, albedo, m, S->matPar[m * 9 + 4], 1, rng);
	bd sk = S->matPar[m * 9 + 5];
	if (sk > 0) out = vadd(out, spec_loop(S, p, n, viewDir, m, v3_(1, 1, 1), sk));
	return out;
}

static v3 trace_stack(scenebuf *S, v3 ro, v3 rd, int depth, beam_rng *rng) {
	fframe stack[MAX_FRAMES];
	int sp = 0;

	stack[sp].phase = PH_ENTER;
	stack[sp].ro = ro; stack[sp].rd = rd; stack[sp].depth = depth;
	sp++;

	v3 result = v3_(0, 0, 0);

	while (sp > 0) {
		fframe *f = &stack[sp - 1];
		v3 res;

		if (f->phase == PH_ENTER) {
			if (f->depth <= 0) {
				res = env_at(f->rd, S->skyStops, S->nStops, S->envColor, S->envTop, S->envMid, S->envBot, S->skyGradient, S->skyCurve);
				res = apply_atmos(S, res, f->ro, f->rd, (bd)100000.0);
				goto frame_done;
			}
			khit h = world_intersect(f->ro, f->rd, S->objA, S->objB, S->objKind, S->objRoot,
			                         S->objNorms, S->nObjs, S->nodeBMin, S->nodeBMax,
			                         S->nodeLeft, S->nodeRight, S->nodeCount, S->triV, S->triN);
			if (!h.hit) {
				res = env_at(f->rd, S->skyStops, S->nStops, S->envColor, S->envTop, S->envMid, S->envBot, S->skyGradient, S->skyCurve);
				res = apply_atmos(S, res, f->ro, f->rd, (bd)100000.0);
				goto frame_done;
			}
			f->hit = 1; f->t = h.t; f->p = h.p; f->n = h.n; f->front = h.front;
			f->matIdx = S->objMat[h.objIdx];
			// Textured albedo: exact port of Renderer.albedoAt. triUV carries
			// per-triangle corner UVs in triOrder — GLOBAL tri index (objRoot's
			// BVH leaf tri lists permute into a single triOrder, so h.tri is
			// already global). Barycentric interp mirrors positions (w0 = 1-u-v).
			{
				int texIdx = S->matTex[f->matIdx];
				if (texIdx >= 0) {
					v3 alb = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, f->matIdx, h.p);
					int kind2 = S->objKind[h.objIdx];
					bd tu = 0, tv = 0;
					int haveUV = 0;
					if (kind2 == 2 && len_check(S, h)) { // mesh (objKind: 0 sphere, 1 plane, 2 mesh)
						int o6 = h.tri * 6;
						tu = S->triUV[o6+0] * (1 - h.bu - h.bv) + S->triUV[o6+2] * h.bu + S->triUV[o6+4] * h.bv;
						tv = S->triUV[o6+1] * (1 - h.bu - h.bv) + S->triUV[o6+3] * h.bu + S->triUV[o6+5] * h.bv;
						haveUV = 1;
					} else if (kind2 == 1) { // plane: automatic planar mapping
						v3 pn = v3_(S->objB[h.objIdx*3+0], S->objB[h.objIdx*3+1], S->objB[h.objIdx*3+2]);
						if (pn.y*pn.y >= pn.x*pn.x && pn.y*pn.y >= pn.z*pn.z) { tu = h.p.x; tv = h.p.z; }
						else if (pn.x*pn.x >= pn.z*pn.z) { tu = h.p.z; tv = h.p.y; }
						else { tu = h.p.x; tv = h.p.y; }
						haveUV = 1;
					}
					if (haveUV) {
						v3 ts = tex_sample(S->texPix, S->texInfo, texIdx, tu, tv);
						f->albedo = vmul(ts, alb); // texture * base albedo (white = as-authored)
					} else {
						f->albedo = alb;
					}
				} else {
					f->albedo = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, f->matIdx, h.p);
				}
			}

			int kind = S->matFlags[f->matIdx * 4 + 0];
			if (kind == 3) { // emissive
				res = vmuls(f->albedo, S->matPar[f->matIdx * 9 + 3]);
				res = apply_atmos(S, res, f->ro, f->rd, h.t);
				goto frame_done;
			}
			if (kind == 1) { // metal
				if (f->depth > 1) {
					v3 rdir = reflect_dir(f->rd, f->n);
					rdir = perturb(rdir, S->matPar[f->matIdx * 9 + 0], rng);
					v3 orig = vadd(f->p, vmuls(f->n, vdot(rdir, f->n) < 0 ? -1e-4 : 1e-4));
					f->phase = PH_METAL_WAIT;
					stack[sp].phase = PH_ENTER;
					stack[sp].ro = orig; stack[sp].rd = rdir; stack[sp].depth = f->depth - 1;
					sp++;
					continue;
				}
				res = shade_result(S, f, v3_(0, 0, 0), v3_(0, 0, 0), 0, rng);
				res = apply_atmos(S, res, f->ro, f->rd, h.t);
				goto frame_done;
			}
			if (kind == 2) { // glass
				if (f->depth <= 1) {
					res = vmul(S->envColor, f->albedo);
					res = apply_atmos(S, res, f->ro, f->rd, h.t);
					goto frame_done;
				}
				// Fresnel prep (no RNG)
				v3 n = f->n;
				bd cosI = -vdot(f->rd, n);
				bd etaI = 1, etaT = S->matPar[f->matIdx * 9 + 1];
				if (!f->front) { etaI = S->matPar[f->matIdx * 9 + 1]; etaT = 1; }
				bd eta = etaI / etaT;
				bd sin2T = eta * eta * (1 - cosI * cosI);
				bd r0 = (etaI - etaT) / (etaI + etaT);
				r0 *= r0;
				bd reflOverride = S->matPar[f->matIdx * 9 + 2];
				if (reflOverride > r0) r0 = reflOverride;
				bd fr = r0 + (1 - r0) * gp_pow_use(1 - cosI, 5);
				f->out = v3_(fr, sin2T, cosI);
				v3 rdir = reflect_dir(f->rd, n);
				v3 orig = vadd(f->p, vmuls(n, vdot(rdir, n) < 0 ? -1e-4 : 1e-4));
				f->phase = PH_GLASS_REFL_WAIT;
				stack[sp].phase = PH_ENTER;
				stack[sp].ro = orig; stack[sp].rd = rdir; stack[sp].depth = f->depth - 1;
				sp++;
				continue;
			}
			// matte
			res = shade_result(S, f, v3_(0, 0, 0), v3_(0, 0, 0), 0, rng);
			res = apply_atmos(S, res, f->ro, f->rd, h.t);
			goto frame_done;
		}

		if (f->phase == PH_METAL_WAIT) {
			res = shade_result(S, f, f->childRes, v3_(0, 0, 0), 0, rng);
			res = apply_atmos(S, res, f->ro, f->rd, f->t);
			goto frame_done;
		}

		if (f->phase == PH_GLASS_REFL_WAIT) {
			f->reflRes = f->childRes;
			bd sin2T = f->out.y;
			if (sin2T > 1) { // TIR
				res = vmul(f->reflRes, f->albedo);
				res = apply_atmos(S, res, f->ro, f->rd, f->t);
				goto frame_done;
			}
			bd cosT = sqrt(1 - sin2T);
			bd cosI = f->out.z;
			// eta needed: recompute (cheap, exact same ops)
			bd etaI = 1, etaT = S->matPar[f->matIdx * 9 + 1];
			if (!f->front) { etaI = S->matPar[f->matIdx * 9 + 1]; etaT = 1; }
			bd eta = etaI / etaT;
			v3 refrDir = vnorm(vadd(vmuls(f->rd, eta), vmuls(f->n, eta * cosI - cosT)));
			v3 orig = vsub(f->p, vmuls(f->n, 1e-4));
			f->phase = PH_GLASS_REFR_WAIT;
			stack[sp].phase = PH_ENTER;
			stack[sp].ro = orig; stack[sp].rd = refrDir; stack[sp].depth = f->depth - 1;
			sp++;
			continue;
		}

		// PH_GLASS_REFR_WAIT
		{
			bd fr = f->out.x;
			v3 base = vadd(vmuls(f->reflRes, fr), vmuls(f->childRes, 1 - fr));
			v3 alb = f->albedo;
			v3 spec = spec_loop(S, f->p, f->n, vmuls(f->rd, -1), f->matIdx, alb, fr);
			res = vadd(vmul(base, alb), spec);
			res = apply_atmos(S, res, f->ro, f->rd, f->t);
			goto frame_done;
		}

	frame_done:
		sp--;
		result = res;
		if (sp > 0) {
			stack[sp - 1].childRes = res;
		}
	}
	return result;
}

// ---------- tonemap (exact port) ----------

static bd lin_to_srgb(bd x) {
	if (x <= 0.0031308) return 12.92 * x;
	return 1.055 * gp_pow_use(x, (bd)(1.0 / 2.4)) - 0.055;
}

static uchar tonemap_ch(bd x, bd exposure, int clip) {
	x *= exposure;
	if (!clip) x = x / (1 + x);
	if (x < 0) x = 0;
	if (x > 1) x = 1;
	x = lin_to_srgb(x);
	return (uchar)(x * 255 + 0.5);
}

// ---------- entry point ----------

__kernel void trace_pixel(__global const int *cfg,      // [0]=W [1]=H [2]=samples [3]=maxDepth [4]=clip
                          __global const bd *cam,       // orig(3) llc(3) horiz(3) vert(3) u(3) v(3) w(3) lens focus
                          __global const bd *objA, __global const bd *objB,
                          __global const int *objKind, __global const int *objRoot,
                          __global const int *objNorms, __global const int *objMat,
                          __global const bd *nodeBMin, __global const bd *nodeBMax,
                          __global const int *nodeLeft, __global const int *nodeRight,
                          __global const int *nodeCount,
                          __global const bd *triV, __global const bd *triN,
                          __global const bd *matAlb, __global const bd *matP0, __global const bd *matP1,
                          __global const bd *matPar, __global const int *matFlags,
                          __global const bd *lPos, __global const bd *lDir,
                          __global const bd *lCol, __global const bd *lPar,
                          __global const int *lFlags,
                          __global const bd *skyStops,
                          __global const bd *envBuf,   // envColor(3) top(3) mid(3) bot(3) curve(1) ambient(1) ambientColor(3)
                          __global const bd *fogBuf,   // color(3) k alt off max (atan folded into fogFlags)
                          __global const int *envFlags, // [0] skyGradient [1] nStops [2] fogEnabled [3] fogAtan
                          __global const long *seedBuf, // [0] = seed
                          __global const bd *exposureBuf,
                          __global const bd *mlTri, __global const bd *mlCDF,
                          __global const bd *mlArea, __global const int *mlStart,
                          __global const bd *triUV, __global const int *texInfo,
                          __global const int *matTex, __global const bd *texPix,
                          __global uchar *outRGBA) {
	int xi = get_global_id(0);
	int yi = get_global_id(1);
	int W = cfg[0], H = cfg[1];
	if (xi >= W || yi >= H) return;
	int n = cfg[2];
	int maxDepth = cfg[3];
	int clip = cfg[4];

	scenebuf S;
	S.objA = objA; S.objB = objB; S.objKind = objKind; S.objRoot = objRoot;
	S.objNorms = objNorms; S.objMat = objMat; S.nObjs = cfg[5];
	S.nodeBMin = nodeBMin; S.nodeBMax = nodeBMax;
	S.nodeLeft = nodeLeft; S.nodeRight = nodeRight; S.nodeCount = nodeCount;
	S.triV = triV; S.triN = triN;
	S.matAlb = matAlb; S.matP0 = matP0; S.matP1 = matP1; S.matPar = matPar; S.matFlags = matFlags;
	S.nMats = cfg[6];
	S.lPos = lPos; S.lDir = lDir; S.lCol = lCol; S.lPar = lPar; S.lFlags = lFlags;
	S.mlTri = mlTri; S.mlCDF = mlCDF; S.mlArea = mlArea; S.mlStart = mlStart;
	S.triUV = triUV; S.texInfo = texInfo; S.matTex = matTex; S.texPix = texPix;
	S.nLights = cfg[7];
	S.shadowSamples = cfg[8];
	S.skyStops = skyStops; S.nStops = envFlags[1];
	S.envColor = v3_(envBuf[0], envBuf[1], envBuf[2]);
	S.envTop = v3_(envBuf[3], envBuf[4], envBuf[5]);
	S.envMid = v3_(envBuf[6], envBuf[7], envBuf[8]);
	S.envBot = v3_(envBuf[9], envBuf[10], envBuf[11]);
	S.skyGradient = envFlags[0];
	S.skyCurve = envBuf[12];
	S.ambient = envBuf[13];
	S.ambientColor = v3_(envBuf[14], envBuf[15], envBuf[16]);
	S.fogEnabled = envFlags[2];
	S.fogColor = v3_(fogBuf[0], fogBuf[1], fogBuf[2]);
	S.fogK = fogBuf[3]; S.fogAlt = fogBuf[4]; S.fogOff = fogBuf[5]; S.fogMax = fogBuf[6];
	S.fogAtan = envFlags[3];
	S.mediaMode = envFlags[5];
	S.mediaAtan = envFlags[3]; // atan flag shares the slot (fog+media never both set)
	S.mediaSteps = envFlags[4];
	S.mediaColor = S.fogColor;
	S.mediaK = S.fogK; S.mediaAlt = S.fogAlt; S.mediaOff = S.fogOff;

	// camera — basis computed on host, bit-identical
	v3 orig = v3_(cam[0], cam[1], cam[2]);
	v3 llc = v3_(cam[3], cam[4], cam[5]);
	v3 horiz = v3_(cam[6], cam[7], cam[8]);
	v3 vert = v3_(cam[9], cam[10], cam[11]);
	v3 u = v3_(cam[12], cam[13], cam[14]);
	v3 v = v3_(cam[15], cam[16], cam[17]);
	v3 w = v3_(cam[18], cam[19], cam[20]);
	bd lens = cam[21];
	bd focus = cam[22];

	beam_rng rng;
	rng.s = seed_pixel(seedBuf[0], xi, yi);

	bd px = ((bd)xi + 0.5) / (bd)W;
	bd py = 1 - ((bd)yi + 0.5) / (bd)H;

	v3 acc = v3_(0, 0, 0);
	for (int sy = 0; sy < n; sy++) {
		for (int sx = 0; sx < n; sx++) {
			bd jx = (((bd)sx + rng_float(&rng)) / (bd)n) / (bd)W;
			bd jy = (((bd)sy + rng_float(&rng)) / (bd)n) / (bd)H;
			// Camera.Ray — exact op order
			v3 target = vadd(vadd(llc, vmuls(horiz, px + jx)), vmuls(vert, py + jy));
			v3 dir = vnorm(vsub(target, orig));
			v3 rorig = orig;
			if (lens > 0) {
				bd fwd = vdot(w, dir);
				if (fwd > -1e-6) {
					// behind camera — no RNG consumed
				} else {
					bd t = focus / -fwd;
					v3 fp = vadd(orig, vmuls(dir, t));
					bd ang = rng_float(&rng) * 2 * (bd)GP_PI;
					bd rad = sqrt(rng_float(&rng)) * lens;
					v3 offs = vadd(vmuls(u, gp_cos_use(ang) * rad), vmuls(v, gp_sin_use(ang) * rad));
					rorig = vadd(orig, offs);
					dir = vnorm(vsub(fp, rorig));
				}
			}
			v3 c = trace_stack(&S, rorig, dir, maxDepth, &rng);
			acc = vadd(acc, c);
		}
	}
	bd inv = 1 / (bd)(n * n);
	acc = vmuls(acc, inv);
	bd exposure = exposureBuf[0];
	uchar r = tonemap_ch(acc.x, exposure, clip);
	uchar g = tonemap_ch(acc.y, exposure, clip);
	uchar b = tonemap_ch(acc.z, exposure, clip);
	int idx = (yi * W + xi) * 4;
	outRGBA[idx + 0] = r;
	outRGBA[idx + 1] = g;
	outRGBA[idx + 2] = b;
	outRGBA[idx + 3] = 255;
}
