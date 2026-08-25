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
} scenebuf;

// forward decls
static v3 trace_stack(scenebuf *S, v3 ro, v3 rd, int depth, beam_rng *rng);

// diffuse+shadow loop shared by matte and untinted metal.
// energyScale multiplies the direct term ((1-reflectivity) for metal, 1 matte).
static v3 direct_light(scenebuf *S, v3 p, v3 n, v3 albedo, int m, bd diffuseK, bd energyScale) {
	v3 out = vmul(albedo, S->ambientColor);
	out = vmuls(out, S->ambient * energyScale);
	for (int i = 0; i < S->nLights; i++) {
		klightdir L = light_dir(S->lPos, S->lDir, S->lPar, S->lFlags, i, p);
		if (!L.ok) continue;
		bd ndotl = vdot(n, L.ld);
		if (ndotl <= 0) continue;
		v3 trans = v3_(1, 1, 1);
		if (!S->lFlags[i * 3 + 2]) { // shadowless
			v3 orig = vadd(p, vmuls(n, vdot(L.ld, n) < 0 ? -1e-4 : 1e-4));
			// shadow march
			for (int hop = 0; hop < 8; hop++) {
				khit h = world_intersect(orig, L.ld, S->objA, S->objB, S->objKind, S->objRoot,
				                         S->objNorms, S->nObjs, S->nodeBMin, S->nodeBMax,
				                         S->nodeLeft, S->nodeRight, S->nodeCount, S->triV, S->triN);
				if (!h.hit || h.t >= L.dist) break;
				// need object matIdx of the hit: recompute via object scan
				// (world_intersect tracks the winning object index in objWin)
				int m2 = S->objMat[h.objIdx];
				if (S->matFlags[m2 * 4 + 0] != 2) { // not glass
					trans = v3_(0, 0, 0);
					break;
				}
				v3 alb2 = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, m2, h.p);
				trans = vmuls(vmul(trans, alb2), 0.85);
				orig = vadd(h.p, vmuls(L.ld, 2e-4));
			}
			if (trans.x <= 0 && trans.y <= 0 && trans.z <= 0) continue;
		}
		bd atten = light_atten(S->lPar, S->lFlags, i, L.dist);
		bd s = ndotl * S->lPar[i * 2] * diffuseK * energyScale * atten;
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
		bd spec = gp_pow_use(ndoth, sh) * S->lPar[i * 2] * atten * k;
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
			out = direct_light(S, p, n, albedo, m, S->matPar[m * 9 + 4], 1 - refl);
		}
		out = vadd(out, rOut);
		bd sk = S->matPar[m * 9 + 5];
		if (sk > 0) out = vadd(out, spec_loop(S, p, n, viewDir, m, v3_(1, 1, 1), sk));
		else out = vadd(out, spec_loop(S, p, n, viewDir, m, v3_(1, 1, 1), 1));
		return out;
	}

	// matte
	v3 out = direct_light(S, p, n, albedo, m, S->matPar[m * 9 + 4], 1);
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
				if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, (bd)100000.0, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
				goto frame_done;
			}
			khit h = world_intersect(f->ro, f->rd, S->objA, S->objB, S->objKind, S->objRoot,
			                         S->objNorms, S->nObjs, S->nodeBMin, S->nodeBMax,
			                         S->nodeLeft, S->nodeRight, S->nodeCount, S->triV, S->triN);
			if (!h.hit) {
				res = env_at(f->rd, S->skyStops, S->nStops, S->envColor, S->envTop, S->envMid, S->envBot, S->skyGradient, S->skyCurve);
				if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, (bd)100000.0, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
				goto frame_done;
			}
			f->hit = 1; f->t = h.t; f->p = h.p; f->n = h.n; f->front = h.front;
			f->matIdx = S->objMat[h.objIdx];
			f->albedo = mat_albedo(S->matAlb, S->matP0, S->matP1, S->matPar, S->matFlags, f->matIdx, h.p);

			int kind = S->matFlags[f->matIdx * 4 + 0];
			if (kind == 3) { // emissive
				res = vmuls(f->albedo, S->matPar[f->matIdx * 9 + 3]);
				if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, h.t, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
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
				if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, h.t, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
				goto frame_done;
			}
			if (kind == 2) { // glass
				if (f->depth <= 1) {
					res = vmul(S->envColor, f->albedo);
					if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, h.t, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
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
			if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, h.t, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
			goto frame_done;
		}

		if (f->phase == PH_METAL_WAIT) {
			res = shade_result(S, f, f->childRes, v3_(0, 0, 0), 0, rng);
			if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, f->t, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
			goto frame_done;
		}

		if (f->phase == PH_GLASS_REFL_WAIT) {
			f->reflRes = f->childRes;
			bd sin2T = f->out.y;
			if (sin2T > 1) { // TIR
				res = vmul(f->reflRes, f->albedo);
				if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, f->t, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
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
			if (S->fogEnabled) res = fog_blend(res, f->ro, f->rd, f->t, S->fogColor, S->fogK, S->fogAlt, S->fogOff, S->fogMax, S->fogAtan);
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
	S.nLights = cfg[7];
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
