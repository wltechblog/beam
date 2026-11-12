// kernel_trace.cl — Beam GPU kernel, part 2: scene data layout + Whitted tracer.
// Iterative DFS with an explicit frame stack replicates renderer.go's recursion
// (left-first: reflection child consumes RNG before refraction child).
// fp64 build → bit-identical to CPU; fp32 build → fast approximate path.

#pragma OPENCL FP_CONTRACT OFF

#define MAX_FRAMES 40
#define MAX_NODES 128

typedef struct { bd x, y, z; } v3;

static v3 v3_(bd x, bd y, bd z) { v3 r; r.x = x; r.y = y; r.z = z; return r; }
static v3 vadd(v3 a, v3 b) { return v3_(a.x + b.x, a.y + b.y, a.z + b.z); }
static v3 vsub(v3 a, v3 b) { return v3_(a.x - b.x, a.y - b.y, a.z - b.z); }
static v3 vmul(v3 a, v3 b) { return v3_(a.x * b.x, a.y * b.y, a.z * b.z); }
static v3 vmuls(v3 a, bd s) { return v3_(a.x * s, a.y * s, a.z * s); }
static v3 vcross(v3 a, v3 b) {
	return v3_(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static bd vdot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static v3 vnorm(v3 a) {
	bd l = sqrt(vdot(a, a));
	if (l < 1e-12) return v3_(0, 0, 0);
	return vmuls(a, 1 / l);
}

// ---------- material accessors (SoA) ----------
// matPar stride 9: [0]rough [1]ior [2]refl [3]power [4]diffuse [5]specK [6]pscale [7]pfrom [8]pto
// matFlags stride 4: [0]kind(0 m,1 M,2 g,3 e) [1]ptype(0 none,1 checker,2 gradient) [2]axis [3]untinted

static v3 mat_albedo(__global const bd *matAlb, __global const bd *matP0,
                     __global const bd *matP1, __global const bd *matPar,
                     __global const int *matFlags, int m, v3 p) {
	int ptype = matFlags[m * 4 + 1];
	if (ptype == 0) return v3_(matAlb[m * 3], matAlb[m * 3 + 1], matAlb[m * 3 + 2]);
	if (ptype == 1) { // checker
		bd sc = matPar[m * 9 + 6];
		if (sc <= 0) sc = 1;
		bd cx = floor(p.x / sc);
		bd cy = floor(p.y / sc);
		bd cz = floor(p.z / sc);
		if (((long)(cx + cy + cz) & 1) == 0) {
			return v3_(matP0[m * 3], matP0[m * 3 + 1], matP0[m * 3 + 2]);
		}
		return v3_(matP1[m * 3], matP1[m * 3 + 1], matP1[m * 3 + 2]);
	}
	// gradient
	bd t;
	bd from = matPar[m * 9 + 7], to = matPar[m * 9 + 8];
	if (to <= from) to = from + 1;
	int axis = matFlags[m * 4 + 2];
	if (axis == 1) t = (p.y - from) / (to - from);
	else if (axis == 2) t = (p.z - from) / (to - from);
	else t = (p.x - from) / (to - from);
	if (t < 0) t = 0;
	if (t > 1) t = 1;
	v3 a = v3_(matP0[m * 3], matP0[m * 3 + 1], matP0[m * 3 + 2]);
	v3 b = v3_(matP1[m * 3], matP1[m * 3 + 1], matP1[m * 3 + 2]);
	return vadd(a, vmuls(vsub(b, a), t));
}

// ---------- intersection primitives (exact ports of geom/ray.go) ----------

typedef struct {
	int hit;
	bd t;
	v3 p, n;
	int front;
	int tri;
	int objIdx;
	bd bu, bv; // barycentric coords of a triangle hit (weights of b and c)
} khit;

static khit isect_sphere(v3 ro, v3 rd, v3 center, bd radius) {
	khit h; h.hit = 0; h.tri = -1; h.objIdx = -1; h.bu = 0; h.bv = 0;
	v3 oc = vsub(ro, center);
	bd b = vdot(oc, rd);
	bd c = vdot(oc, oc) - radius * radius;
	bd disc = b * b - c;
	if (disc < 0) return h;
	bd sq = sqrt(disc);
	bd t = -b - sq;
	if (t < 1e-9) t = -b + sq;
	if (t < 1e-9) return h;
	h.hit = 1; h.t = t;
	h.p = vadd(ro, vmuls(rd, t));
	h.n = vmuls(vsub(h.p, center), 1 / radius);
	h.front = vdot(rd, h.n) < 0;
	if (!h.front) h.n = vmuls(h.n, -1);
	return h;
}

static khit isect_plane(v3 ro, v3 rd, v3 point, v3 normal) {
	khit h; h.hit = 0; h.tri = -1; h.objIdx = -1; h.bu = 0; h.bv = 0;
	bd denom = vdot(normal, rd);
	if (denom > -1e-9 && denom < 1e-9) return h;
	bd t = vdot(vsub(point, ro), normal) / denom;
	if (t < 1e-9) return h;
	h.hit = 1; h.t = t;
	h.p = vadd(ro, vmuls(rd, t));
	h.n = normal;
	h.front = vdot(rd, h.n) < 0;
	if (!h.front) h.n = vmuls(h.n, -1);
	return h;
}

static khit isect_tri(v3 ro, v3 rd, v3 a, v3 b, v3 c) {
	khit h; h.hit = 0; h.tri = -1; h.objIdx = -1; h.bu = 0; h.bv = 0;
	v3 e1 = vsub(b, a);
	v3 e2 = vsub(c, a);
	v3 pvec = vcross(rd, e2);
	bd det = vdot(e1, pvec);
	if (det > -1e-12 && det < 1e-12) return h;
	bd inv = 1 / det;
	v3 tvec = vsub(ro, a);
	bd u = vdot(tvec, pvec) * inv;
	if (u < 0 || u > 1) return h;
	v3 qvec = vcross(tvec, e1);
	bd v = vdot(rd, qvec) * inv;
	if (v < 0 || u + v > 1) return h;
	bd t = vdot(e2, qvec) * inv;
	if (t < 1e-9) return h;
	h.hit = 1; h.t = t;
	h.p = vadd(ro, vmuls(rd, t));
	h.n = vnorm(vcross(e1, e2));
	h.front = det > 0;
	if (!h.front) h.n = vmuls(h.n, -1);
	h.bu = u; h.bv = v;
	return h;
}

// world_intersect: objects in declaration order; strict < for replacement
// (first hit wins exact ties). Mesh traversal left-first to match CPU
// recursion order; bounds pruning is safe because padded bounds can only
// prune subtrees with no strictly-closer triangles.
static khit world_intersect(v3 ro, v3 rd,
                            __global const bd *objA, __global const bd *objB,
                            __global const int *objKind, __global const int *objRoot,
                            __global const int *objNorms, int nObjs,
                            __global const bd *nodeBMin, __global const bd *nodeBMax,
                            __global const int *nodeLeft, __global const int *nodeRight,
                            __global const int *nodeCount,
                            __global const bd *triV, __global const bd *triN) {
	khit best; best.hit = 0; best.t = 0; best.objIdx = -1;
	int bestNorms = 0;

	for (int oi = 0; oi < nObjs; oi++) {
		int ok = objKind[oi];
		if (ok == 0) { // sphere
			khit h = isect_sphere(ro, rd, v3_(objA[oi*3], objA[oi*3+1], objA[oi*3+2]), objB[oi*3]);
			if (h.hit && (best.hit == 0 || h.t < best.t)) {
				best = h; bestNorms = 0; best.objIdx = oi;
			}
		} else if (ok == 1) { // plane
			khit h = isect_plane(ro, rd, v3_(objA[oi*3], objA[oi*3+1], objA[oi*3+2]),
			                     v3_(objB[oi*3], objB[oi*3+1], objB[oi*3+2]));
			if (h.hit && (best.hit == 0 || h.t < best.t)) {
				best = h; bestNorms = 0; best.objIdx = oi;
			}
		} else { // mesh
			int root = objRoot[oi];
			if (root < 0) continue;
			int stack[MAX_NODES];
			int sp = 0;
			stack[sp++] = root;
			while (sp > 0) {
				int n = stack[--sp];
				bd tmin = 1e-9;
				bd tmax = best.hit ? best.t : (bd)1.7976931348623157e+308;
				int miss = 0;
				for (int a = 0; a < 3; a++) {
					bd mn = nodeBMin[n*3+a] - 1e-4;
					bd mx = nodeBMax[n*3+a] + 1e-4;
					bd d = a == 0 ? rd.x : (a == 1 ? rd.y : rd.z);
					bd o = a == 0 ? ro.x : (a == 1 ? ro.y : ro.z);
					bd invD = d == 0 ? (bd)1e30 : 1 / d;
					bd t0 = (mn - o) * invD;
					bd t1 = (mx - o) * invD;
					if (invD < 0) { bd tmp = t0; t0 = t1; t1 = tmp; }
					if (t0 > tmin) tmin = t0;
					if (t1 < tmax) tmax = t1;
					if (tmax < tmin) { miss = 1; break; }
				}
				if (miss) continue;
				if (nodeCount[n] == 0) { // internal: push right then left → left popped first
					if (sp < MAX_NODES) stack[sp++] = nodeRight[n];
					if (sp < MAX_NODES) stack[sp++] = nodeLeft[n];
				} else {
					int start = nodeLeft[n];
					int cnt = nodeCount[n];
					for (int i = 0; i < cnt; i++) {
						int gt = start + i; // triOrder flattened: leaf order == global index
						v3 a = v3_(triV[gt*9+0], triV[gt*9+1], triV[gt*9+2]);
						v3 b = v3_(triV[gt*9+3], triV[gt*9+4], triV[gt*9+5]);
						v3 c = v3_(triV[gt*9+6], triV[gt*9+7], triV[gt*9+8]);
						khit th = isect_tri(ro, rd, a, b, c);
						if (th.hit && (best.hit == 0 || th.t < best.t)) {
							best = th; best.tri = gt; bestNorms = objNorms[oi]; best.objIdx = oi;
						}
					}
				}
			}
		}
	}
	if (best.hit && bestNorms && best.tri >= 0) {
		// interpolated normal at the actual hit point — matches World.Intersect
		bd u = best.bu, v = best.bv;
		bd w = 1 - u - v;
		int gt = best.tri;
		v3 n0 = v3_(triN[gt*9+0], triN[gt*9+1], triN[gt*9+2]);
		v3 n1 = v3_(triN[gt*9+3], triN[gt*9+4], triN[gt*9+5]);
		v3 n2 = v3_(triN[gt*9+6], triN[gt*9+7], triN[gt*9+8]);
		best.n = vnorm(vadd(vadd(vmuls(n0, w), vmuls(n1, u)), vmuls(n2, v)));
	}
	return best;
}

// ---------- lights ----------

typedef struct { v3 ld; bd dist; int ok; } klightdir;

// sample_light_point picks a deterministic point on a spherical light as
// seen from p, from two uniform RNG draws (concentric disk in angle space,
// exact port of render/renderer.go sampleLightPoint). Fixed 2-draw order
// keeps the CPU and GPU RNG streams identical.
static v3 sample_light_point(v3 lightPos, bd radius, v3 p, bd u1, bd u2) {
	v3 axis = vsub(lightPos, p);
	bd dist = sqrt(vdot(axis, axis));
	v3 axisN = vmuls(axis, 1 / dist);

	bd sinMax = radius / dist;
	if (sinMax > 1) sinMax = 1;
	bd omu1 = 1 - u1;
	if (omu1 < 0) omu1 = 0;
	bd r = sinMax * sqrt(omu1); // uniform on the angular disk
	bd phi = 2 * GP_PI * u2;
	bd dx = r * cos(phi);
	bd dy = r * sin(phi);

	// stable perpendicular frame
	v3 ref;
	if (axisN.x < axisN.y && axisN.x < axisN.z) ref = v3_(1, 0, 0);
	else if (axisN.y < axisN.z) ref = v3_(0, 1, 0);
	else ref = v3_(0, 0, 1);
	v3 u = vnorm(vcross(axisN, ref));
	v3 v = vcross(axisN, u);

	bd s2 = r * r; // sin(theta)^2 over the full disk radius
	bd om = 1 - s2;
	if (om < 0) om = 0;
	bd cosT = sqrt(om);
	v3 dirN = vnorm(vadd(vadd(vmuls(axisN, cosT), vmuls(u, dx)), vmuls(v, dy)));

	bd disc = dist * dist * s2;
	bd rad2 = radius * radius;
	if (disc > rad2) disc = rad2;
	bd t = dist * cosT - sqrt(rad2 - disc);
	return vadd(p, vmuls(dirN, t));
}

static klightdir light_dir(__global const bd *lPos, __global const bd *lDir,
                           __global const bd *lPar, __global const int *lFlags,
                           int i, v3 p) {
	klightdir r; r.ok = 0; r.dist = 0;
	int kind = lFlags[i * 3];
	if (kind == 1) { // directional: lDir points where light travels
		r.ld = vmuls(v3_(lDir[i*3], lDir[i*3+1], lDir[i*3+2]), -1);
		r.dist = (bd)1.7976931348623157e+308;
		r.ok = 1;
		return r;
	}
	if (kind == 2) { // spot
		v3 toL = vsub(v3_(lPos[i*3], lPos[i*3+1], lPos[i*3+2]), p);
		bd dist = sqrt(vdot(toL, toL));
		v3 dir = vmuls(toL, 1 / dist);
		bd cosA = lPar[i * 4 + 1];
		if (vdot(dir, vmuls(v3_(lDir[i*3], lDir[i*3+1], lDir[i*3+2]), -1)) < cosA) {
			r.ok = 0;
			return r;
		}
		r.ld = dir; r.dist = dist; r.ok = 1;
		return r;
	}
	// point
	v3 toL = vsub(v3_(lPos[i*3], lPos[i*3+1], lPos[i*3+2]), p);
	bd dist = sqrt(vdot(toL, toL));
	r.ld = vmuls(toL, 1 / dist);
	r.dist = dist;
	r.ok = 1;
	return r;
}

static bd light_atten(__global const bd *lPar, __global const int *lFlags, int i, bd dist) {
	if (lFlags[i * 3] == 1 || dist == (bd)1.7976931348623157e+308 || lFlags[i * 3 + 1]) {
		return 1;
	}
	return 1 / (1 + 0.09 * dist + 0.032 * dist * dist);
}

// ---------- fog (exact port of opticalDepth incl. addition order) ----------

static void add_dense(bd a, bd b, bd dist, bd *total) {
	if (a < 0) a = 0;
	if (b > dist) b = dist;
	if (b > a) *total += b - a;
}

static void add_exp(bd a, bd b, bd dist, bd y0, bd dy, bd alt, bd off, bd *total) {
	if (a < 0) a = 0;
	if (b > dist) b = dist;
	if (b <= a) return;
	if (fabs(dy) < 1e-12) {
		*total += (b - a) * gp_exp_use(-(y0 - alt) / off);
		return;
	}
	bd ga = gp_exp_use(-(y0 + dy * a - alt) / off);
	bd gb = gp_exp_use(-(y0 + dy * b - alt) / off);
	*total += off / dy * (ga - gb);
}

static bd optical_depth(bd y0, bd dy, bd dist, bd alt, bd off, int atanFog) {
	if (dist <= 0) return 0;
	const bd maxD = (bd)1000000.0;
	if (dist > maxD) dist = maxD;
	bd total = 0;
	if (fabs(dy) < 1e-12) {
		if (y0 <= alt) return dist;
		if (off > 0) return dist * gp_exp_use(-(y0 - alt) / off);
		return 0;
	}
	if (atanFog) {
		bd u1 = (y0 - off) / alt;
		bd u2 = (y0 + dy * dist - off) / alt;
		return alt * (gp_atan_use(u2) - gp_atan_use(u1)) / dy;
	}
	bd sStar = (alt - y0) / dy;
	if (off > 0) {
		if (dy > 0) {
			add_dense(0, sStar, dist, &total);
			add_exp(sStar, dist, dist, y0, dy, alt, off, &total);
		} else {
			add_exp(0, sStar, dist, y0, dy, alt, off, &total);
			add_dense(sStar, dist, dist, &total);
		}
		return total;
	}
	if (dy > 0) {
		add_dense(0, sStar, dist, &total);
	} else {
		add_dense(sStar, dist, dist, &total);
	}
	return total;
}

// ---------- env ----------

static v3 sky_map_at(__global const bd *stops, int nStops, bd dirY) {
	if (dirY <= stops[0]) return v3_(stops[1], stops[2], stops[3]);
	if (dirY >= stops[(nStops - 1) * 4]) {
		int q = (nStops - 1) * 4;
		return v3_(stops[q+1], stops[q+2], stops[q+3]);
	}
	for (int i = 1; i < nStops; i++) {
		if (dirY <= stops[i * 4]) {
			bd aY = stops[(i-1)*4], bY = stops[i*4];
			v3 a = v3_(stops[(i-1)*4+1], stops[(i-1)*4+2], stops[(i-1)*4+3]);
			v3 b = v3_(stops[i*4+1], stops[i*4+2], stops[i*4+3]);
			bd t = (dirY - aY) / (bY - aY);
			return vadd(a, vmuls(vsub(b, a), t));
		}
	}
	int q = (nStops - 1) * 4;
	return v3_(stops[q+1], stops[q+2], stops[q+3]);
}

static v3 env_at(v3 dir, __global const bd *stops, int nStops,
                 v3 envColor, v3 envTop, v3 envMid, v3 envBot,
                 int skyGradient, bd skyCurve) {
	if (nStops >= 2) return sky_map_at(stops, nStops, dir.y);
	if (!skyGradient) return envColor;
	bd t = dir.y * 0.5 + 0.5;
	v3 top = envTop, bot = envBot;
	if (!(envMid.x == 0 && envMid.y == 0 && envMid.z == 0)) {
		if (dir.y >= 0) {
			top = envTop; bot = envMid; t = dir.y;
		} else {
			top = envBot; bot = envMid; t = -dir.y;
		}
		if (skyCurve > 0 && skyCurve != 1) t = gp_pow_use(t, skyCurve);
	}
	return vadd(bot, vmuls(vsub(top, bot), t));
}

static v3 fog_blend(v3 c, v3 orig, v3 dir, bd dist,
                    v3 fogColor, bd fogK, bd fogAlt, bd fogOff, bd fogMax, int fogAtan) {
	bd od = optical_depth(orig.y, dir.y, dist, fogAlt, fogOff, fogAtan);
	bd tr = gp_exp_use(-fogK * od);
	bd amt = 1 - tr;
	if (amt > fogMax) amt = fogMax;
	return vadd(vmuls(c, 1 - amt), vmuls(fogColor, amt));
}

// ---------- frame stack ----------

#define PH_TRACE 0      // entry: intersect + dispatch
#define PH_METAL_DONE 1 // have reflOut → finish shading
#define PH_GLASS_REFL 2 // have refl → maybe TIR, else launch refr
#define PH_GLASS_REFR 3 // have refl+refr → combine

typedef struct {
	int phase;
	v3 ro, rd;
	int depth;
	// hit data
	int hit;
	bd t;
	v3 p, n;
	int front;
	int matIdx;
	// metal
	v3 reflOut;
	// glass
	bd fr, cosT, sin2T, eta, cosI;
	v3 refrDir;
	int tir;
	v3 glassAlbedo;
} frame;

static v3 offset_ray(v3 p, v3 n, v3 dir) {
	bd eps = 1e-4;
	if (vdot(dir, n) < 0) eps = -eps;
	return vadd(p, vmuls(n, eps)); // returns origin; dir passed separately
}

static v3 reflect_dir(v3 d, v3 n) {
	return vnorm(vsub(d, vmuls(n, 2 * vdot(d, n))));
}

static v3 perturb(v3 dir, bd rough, beam_rng *rng) {
	if (rough <= 0) return dir;
	for (int tries = 0; tries < 8; tries++) {
		v3 v = v3_(rng_float(rng) * 2 - 1, rng_float(rng) * 2 - 1, rng_float(rng) * 2 - 1);
		bd l = sqrt(vdot(v, v));
		if (l <= 1 && l > 1e-6) {
			v3 cand = vnorm(vadd(dir, vmuls(v, rough)));
			if (vdot(cand, dir) > 0.5) return cand;
		}
	}
	return dir;
}

static bd spec_shininess(bd rough) {
	if (rough < 0.015) rough = 0.015;
	bd sh = 2 / (rough * rough);
	if (sh > 8192) sh = 8192;
	return sh;
}

static v3 specular(v3 p, v3 n, v3 viewDir, int m,
                   __global const bd *matPar,
                   v3 albedo, bd k,
                   __global const bd *lPos, __global const bd *lDir,
                   __global const bd *lCol, __global const bd *lPar,
                   __global const int *lFlags, int nLights) {
	bd sh = spec_shininess(matPar[m * 9 + 0]);
	v3 out = v3_(0, 0, 0);
	for (int i = 0; i < nLights; i++) {
		klightdir L = light_dir(lPos, lDir, lPar, lFlags, i, p);
		if (!L.ok) continue;
		v3 h = vnorm(vadd(L.ld, viewDir));
		bd ndoth = vdot(n, h);
		if (ndoth <= 0) continue;
		bd atten = light_atten(lPar, lFlags, i, L.dist);
		bd spec = gp_pow_use(ndoth, sh) * lPar[i * 4] * atten * k;
		out = vadd(out, vmuls(vmul(v3_(lCol[i*3], lCol[i*3+1], lCol[i*3+2]), albedo), spec));
	}
	return out;
}
