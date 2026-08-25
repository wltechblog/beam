// kernel_math.cl — Beam GPU kernel, part 1: types, RNG, transcendental ports.
// Every function here replicates the exact algorithm in render/beammath.go so
// that fp64 renders are bit-identical to CPU renders.

#ifdef BEAM_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double bd;
#else
typedef float bd;
#endif

typedef ulong u64;
typedef long i64;

// math.Pi exactly (works for both fp32 and fp64 builds)
#define GP_PI 3.141592653589793115997963468544185161590576171875

// ---------- RNG: splitmix64, exact port of render/rng.go ----------

typedef struct { u64 s; } beam_rng;

static u64 rng_next(beam_rng *r) {
	r->s += 0x9E3779B97F4A7C15UL;
	u64 z = r->s;
	z ^= z >> 30;
	z *= 0xBF58476D1CE4E5B9UL;
	z ^= z >> 27;
	z *= 0x94D049BB133111EBUL;
	z ^= z >> 31;
	return z;
}

static u64 seed_pixel(i64 base, int x, int y) {
	u64 h = (u64)base * 0x9E3779B97F4A7C15UL + (u64)x * 0xBF58476D1CE4E5B9UL + (u64)y * 0x94D049BB133111EBUL;
	h ^= h >> 30;
	h *= 0xBF58476D1CE4E5B9UL;
	h ^= h >> 27;
	h *= 0x94D049BB133111EBUL;
	h ^= h >> 31;
	return h;
}

// Float64 conversion — CPU: float64(z>>11) / float64(1<<53).
#ifdef BEAM_DOUBLE
static bd rng_float(beam_rng *r) {
	return (bd)((u64)rng_next(r) >> 11) / (bd)9007199254740992.0L;
}
#else
static bd rng_float(beam_rng *r) {
	return (bd)(((u64)rng_next(r) >> 11) * (1.0 / 9007199254740992.0));
}
#endif

// ---------- bit helpers ----------

#ifdef BEAM_DOUBLE
static i64 d2bits(bd d) { return as_long(d); }
static bd bits2d(i64 b) { return as_double(b); }
#endif

// ---------- transcendental ports (fp64 only path uses these) ----------
// Verbatim ports of Go's portable math: exp.go, log.go, atan.go, pow.go,
// frexp.go, ldexp.go, modf.go, sin.go, trig_reduce.go.

#ifdef BEAM_DOUBLE

#define BD_INF INFINITY

#define GP_LN2HI 6.93147180369123816490e-01
#define GP_LN2LO 1.90821492927058770002e-10
#define GP_LOG2E 1.44269504088896338700e+00

static bd gp_frexp_norm(bd x, int *exp) {
	// returns normalized y; *exp = extra exponent (0 unless denormal)
	const bd SmallestNormal = 2.2250738585072014e-308;
	if (fabs(x) < SmallestNormal) {
		*exp = -52;
		return x * (bd)4503599627370496.0; // 2**52
	}
	*exp = 0;
	return x;
}

static bd gp_frexp_frac(bd f, int *eout) {
	// frac, exp as in Go Frexp
	if (f == 0.0 || isinf(f) || isnan(f)) {
		*eout = 0;
		return f;
	}
	int extra = 0;
	f = gp_frexp_norm(f, &extra);
	i64 x = d2bits(f);
	int exp = (int)((x >> 52) & 0x7FF) - 1023 + 1;
	x &= ~((i64)0x7FF << 52);
	x |= (i64)(-1 + 1023) << 52;
	*eout = exp + extra;
	return bits2d(x);
}

static bd gp_ldexp(bd frac, int exp) {
	int e2 = 0;
	frac = gp_frexp_norm(frac, &e2);
	exp += e2;
	i64 x = d2bits(frac);
	exp += (int)((x >> 52) & 0x7FF) - 1023;
	if (exp < -1075) {
		return copysign((bd)0.0, frac);
	}
	if (exp > 1023) {
		if (frac < 0) return -(BD_INF);
		return BD_INF;
	}
	bd m = (bd)1.0;
	if (exp < -1022) {
		exp += 53;
		m = (bd)1.0 / (bd)9007199254740992.0; // 2**-53
	}
	x &= ~((i64)0x7FF << 52);
	x |= (i64)(exp + 1023) << 52;
	return m * bits2d(x);
}

static bd gp_modf_int(bd f, bd *frac) {
	// Go Modf returns (int, frac)
	if (f < 1) {
		if (f < 0) {
			bd i, fr;
			i = gp_modf_int(-f, &fr);
			*frac = -fr;
			return -i;
		}
		if (f == 0) {
			*frac = f;
			return f;
		}
		*frac = f;
		return (bd)0.0;
	}
	i64 x = d2bits(f);
	uint e = (uint)((x >> 52) & 0x7FF) - 1023;
	if (e < 64 - 12) {
		i64 mask = ((i64)1 << (64 - 12 - e)) - 1;
		x &= ~mask;
	}
	bd i = bits2d(x);
	*frac = f - i;
	return i;
}

static bd gp_expmulti(bd hi, bd lo, int k) {
	const bd P1 = 1.66666666666666657415e-01;
	const bd P2 = -2.77777777770155933842e-03;
	const bd P3 = 6.61375632143793436117e-05;
	const bd P4 = -1.65339022054652515390e-06;
	const bd P5 = 4.13813679705723846039e-08;

	bd r = hi - lo;
	bd t = r * r;
	bd c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
	bd y = 1 - ((lo - (r * c) / (2 - c)) - hi);
	return gp_ldexp(y, k);
}

static bd gp_exp(bd x) {
	const bd Overflow = 7.09782712893383973096e+02;
	const bd Underflow = -7.45133219101941108420e+02;
	const bd NearZero = (bd)1.0 / (bd)268435456.0; // 2**-28

	if (isnan(x) || isinf(x)) {
		if (x < 0) return (bd)0.0;
		return x;
	}
	if (x > Overflow) return BD_INF;
	if (x < Underflow) return (bd)0.0;
	if (-NearZero < x && x < NearZero) return 1 + x;

	int k = 0;
	if (x < 0) {
		k = (int)(GP_LOG2E * x - 0.5);
	} else if (x > 0) {
		k = (int)(GP_LOG2E * x + 0.5);
	}
	bd hi = x - (bd)k * (bd)GP_LN2HI;
	bd lo = (bd)k * (bd)GP_LN2LO;
	return gp_expmulti(hi, lo, k);
}

static bd gp_log(bd x) {
	const bd L1 = 6.666666666666735130e-01;
	const bd L2 = 3.999999999940941908e-01;
	const bd L3 = 2.857142874366239149e-01;
	const bd L4 = 2.222219843214978396e-01;
	const bd L5 = 1.818357216161805012e-01;
	const bd L6 = 1.531383769920937332e-01;
	const bd L7 = 1.479819860511658591e-01;
	const bd Sqrt2Half = 0.70710678118654752440; // Sqrt2/2

	if (isnan(x)) return x;
	if (x < 0) return (bd)NAN;
	if (isinf(x)) return x;
	if (x == 0) return -(BD_INF);

	int ki = 0;
	bd f1 = gp_frexp_frac(x, &ki);
	if (f1 < Sqrt2Half) {
		f1 *= 2;
		ki--;
	}
	bd f = f1 - 1;
	bd k = (bd)ki;

	bd s = f / (2 + f);
	bd s2 = s * s;
	bd s4 = s2 * s2;
	bd t1 = s2 * (L1 + s4 * (L3 + s4 * (L5 + s4 * L7)));
	bd t2 = s4 * (L2 + s4 * (L4 + s4 * L6));
	bd R = t1 + t2;
	bd hfsq = 0.5 * f * f;
	return k * (bd)GP_LN2Hi - ((hfsq - (s * (hfsq + R) + k * (bd)GP_LN2Lo)) - f);
}

static bd gp_xatan(bd x) {
	const bd P0 = -8.750608600031904122785e-01;
	const bd P1 = -1.615753718733365076637e+01;
	const bd P2 = -7.500855792314704667340e+01;
	const bd P3 = -1.228866684490136173410e+02;
	const bd P4 = -6.485021904942025371773e+01;
	const bd Q0 = +2.485846490142306297962e+01;
	const bd Q1 = +1.650270098316988542046e+02;
	const bd Q2 = +4.328810604912902668951e+02;
	const bd Q3 = +4.853903996359136964868e+02;
	const bd Q4 = +1.945506571482613964425e+02;
	bd z = x * x;
	z = z * ((((P0 * z + P1) * z + P2) * z + P3) * z + P4) /
	    (((((z + Q0) * z + Q1) * z + Q2) * z + Q3) * z + Q4);
	z = x * z + x;
	return z;
}

static bd gp_satan(bd x) {
	const bd Morebits = 6.123233995736765886130e-17;
	const bd Tan3pio8 = 2.41421356237309504880;
	if (x <= 0.66) {
		return gp_xatan(x);
	}
	if (x > Tan3pio8) {
		return (bd)GP_PI / 2 - gp_xatan(1 / x) + Morebits;
	}
	return (bd)GP_PI / 4 + gp_xatan((x - 1) / (x + 1)) + 0.5 * Morebits;
}

static bd gp_atan(bd x) {
	if (x == 0) return x;
	if (x > 0) return gp_satan(x);
	return -gp_satan(-x);
}

static int gp_is_odd_int(bd x) {
	if (fabs(x) >= (bd)9007199254740992.0) { // 1<<53
		return 0;
	}
	bd frac;
	bd xi = gp_modf_int(x, &frac);
	return frac == 0 && ((i64)xi & 1) == 1;
}

static bd gp_pow(bd x, bd y) {
	if (y == 0 || x == 1) return 1;
	if (y == 1) return x;
	if (isnan(x) || isnan(y)) return (bd)NAN;
	if (x == 0) {
		if (y < 0) {
			if (signbit(x) && gp_is_odd_int(y)) return -(BD_INF);
			return BD_INF;
		}
		if (y > 0) {
			if (signbit(x) && gp_is_odd_int(y)) return x;
			return (bd)0.0;
		}
	}
	if (isinf(y)) {
		if (x == -1) return 1;
		if ((fabs(x) < 1) == (y > 0)) return (bd)0.0;
		return BD_INF;
	}
	if (isinf(x)) {
		if (isinf(x) && x < 0) {
			return gp_pow(1 / x, -y);
		}
		if (y < 0) return (bd)0.0;
		return BD_INF;
	}
	if (y == 0.5) return sqrt(x);
	if (y == -0.5) return 1 / sqrt(x);

	bd frac;
	bd yi_i = gp_modf_int(fabs(y), &frac);
	bd yf = frac;
	bd yi = yi_i;
	if (yf != 0 && x < 0) return (bd)NAN;
	if (yi >= (bd)9223372036854775808.0) { // 1<<63
		if (x == -1) return 1;
		if ((fabs(x) < 1) == (y > 0)) return (bd)0.0;
		return BD_INF;
	}

	bd a1 = 1.0;
	int ae = 0;

	if (yf != 0) {
		if (yf > 0.5) {
			yf--;
			yi++;
		}
		a1 = gp_exp(yf * gp_log(x));
	}

	int xe = 0;
	bd x1 = gp_frexp_frac(x, &xe);
	for (i64 i = (i64)yi; i != 0; i >>= 1) {
		if (xe < -4096 || 4096 < xe) {
			ae += xe;
			break;
		}
		if (i & 1 == 1) {
			a1 *= x1;
			ae += xe;
		}
		x1 *= x1;
		xe <<= 1;
		if (x1 < 0.5) {
			x1 += x1;
			xe--;
		}
	}

	if (y < 0) {
		a1 = 1 / a1;
		ae = -ae;
	}
	return gp_ldexp(a1, ae);
}

// ---- sin/cos with Payne-Hanek (trig_reduce.go) ----

__constant u64 gp_mPi4[20] = {
	0x0000000000000001UL, 0x45f306dc9c882a53UL, 0xf84eafa3ea69bb81UL,
	0xb6c52b3278872083UL, 0xfca2c757bd778ac3UL, 0x6e48dc74849ba5c0UL,
	0x0c925dd413a32439UL, 0xfc3bd63962534e7dUL, 0xd1046bea5d768909UL,
	0xd338e04d68befc82UL, 0x7323ac7306a673e9UL, 0x3908bf177bf25076UL,
	0x3ff12fffbc0b301fUL, 0xde5e2316b414da3eUL, 0xda6cfd9e4f96136eUL,
	0x9e8c7ecd3cbfd45aUL, 0xea4f758fd7cbe2f6UL, 0x7a0e73ef14a525d4UL,
	0xd7f6bf623f1aba10UL, 0xac06608df8f6d757UL,
};

__constant bd gp_sinC[6] = {
	1.58962301576546568060e-10,
	-2.50507477628578072866e-8,
	2.75573136213857245213e-6,
	-1.98412698295895385996e-4,
	8.33333333332211858878e-3,
	-1.66666666666666307295e-1,
};

__constant bd gp_cosC[6] = {
	-1.13585365213876817300e-11,
	2.08757008419747316778e-9,
	-2.75573141792967388112e-7,
	2.48015872888517045348e-5,
	-1.38888888888730564116e-3,
	4.16666666666665929218e-2,
};

#define GP_REDUCE_THRESHOLD (bd)536870912.0 // 1<<29

static void gp_trig_reduce(bd x, u64 *jo, bd *zo) {
	const bd PI4 = (bd)GP_PI / 4;
	if (x < PI4) {
		*jo = 0;
		*zo = x;
		return;
	}
	const int bias = 1023;
	const int shift = 52;
	const i64 mask = 0x7FF;

	i64 ix = d2bits(x);
	int exp = (int)((ix >> shift) & mask) - bias - shift;
	ix &= ~(mask << shift);
	ix |= (i64)1 << shift;

	uint digit = (uint)(exp + 61) / 64;
	uint bitshift = (uint)(exp + 61) % 64;

	u64 z0 = (gp_mPi4[digit] << bitshift) | (gp_mPi4[digit + 1] >> (64 - bitshift));
	u64 z1 = (gp_mPi4[digit + 1] << bitshift) | (gp_mPi4[digit + 2] >> (64 - bitshift));
	u64 z2 = (gp_mPi4[digit + 2] << bitshift) | (gp_mPi4[digit + 3] >> (64 - bitshift));

	// 64x64 -> high/low multiplies
	u64 ixu = (u64)ix;
	u64 z2hi = mul_hi(z2, ixu);
	u64 z1hi = mul_hi(z1, ixu);
	u64 z1lo = z1 * ixu;
	u64 z0lo = z0 * ixu;

	u64 lo = z1lo + z2hi;
	u64 hi = z0lo + z1hi + (lo < z1lo ? 1 : 0);
	u64 j = hi >> 61;

	hi = (hi << 3) | (lo >> 61);
	uint lz = clz(hi);
	u64 e = (u64)(bias - (int)(lz + 1));
	hi = (hi << (lz + 1)) | (lo >> (64 - (lz + 1)));
	hi >>= 64 - shift;
	hi |= e << shift;
	bd z = bits2d((i64)hi);

	if (j & 1 == 1) {
		j++;
		j &= 7;
		z--;
	}
	*jo = j;
	*zo = z * PI4;
}

static bd gp_sin_core(bd x) {
	const bd PI4A = 7.85398125648498535156e-1;
	const bd PI4B = 3.77489470793079817668e-8;
	const bd PI4C = 2.69515142907905952645e-15;

	if (x == 0 || isnan(x)) return x;
	if (isinf(x)) return (bd)NAN;

	bool sign = false;
	if (x < 0) {
		x = -x;
		sign = true;
	}
	u64 j;
	bd y, z;
	if (x >= GP_REDUCE_THRESHOLD) {
		gp_trig_reduce(x, &j, &z);
	} else {
		j = (u64)(x * (4 / (bd)GP_PI));
		y = (bd)j;
		if (j & 1 == 1) {
			j++;
			y++;
		}
		j &= 7;
		z = ((x - y * PI4A) - y * PI4B) - y * PI4C;
	}
	if (j > 3) {
		sign = !sign;
		j -= 4;
	}
	bd zz = z * z;
	if (j == 1 || j == 2) {
		y = 1.0 - 0.5 * zz + zz * zz * ((((((gp_cosC[0] * zz) + gp_cosC[1]) * zz + gp_cosC[2]) * zz + gp_cosC[3]) * zz + gp_cosC[4]) * zz + gp_cosC[5]);
	} else {
		y = z + z * zz * ((((((gp_sinC[0] * zz) + gp_sinC[1]) * zz + gp_sinC[2]) * zz + gp_sinC[3]) * zz + gp_sinC[4]) * zz + gp_sinC[5]);
	}
	if (sign) y = -y;
	return y;
}

static bd gp_cos_core(bd x) {
	const bd PI4A = 7.85398125648498535156e-1;
	const bd PI4B = 3.77489470793079817668e-8;
	const bd PI4C = 2.69515142907905952645e-15;

	if (isnan(x) || isinf(x)) return (bd)NAN;

	bool sign = false;
	x = fabs(x);
	u64 j;
	bd y, z;
	if (x >= GP_REDUCE_THRESHOLD) {
		gp_trig_reduce(x, &j, &z);
	} else {
		j = (u64)(x * (4 / (bd)GP_PI));
		y = (bd)j;
		if (j & 1 == 1) {
			j++;
			y++;
		}
		j &= 7;
		z = ((x - y * PI4A) - y * PI4B) - y * PI4C;
	}
	if (j > 3) {
		j -= 4;
		sign = !sign;
	}
	if (j > 1) {
		sign = !sign;
	}
	bd zz = z * z;
	if (j == 1 || j == 2) {
		y = z + z * zz * ((((((gp_sinC[0] * zz) + gp_sinC[1]) * zz + gp_sinC[2]) * zz + gp_sinC[3]) * zz + gp_sinC[4]) * zz + gp_sinC[5]);
	} else {
		y = 1.0 - 0.5 * zz + zz * zz * ((((((gp_cosC[0] * zz) + gp_cosC[1]) * zz + gp_cosC[2]) * zz + gp_cosC[3]) * zz + gp_cosC[4]) * zz + gp_cosC[5]);
	}
	if (sign) y = -y;
	return y;
}

#define gp_pow_use gp_pow
#define gp_exp_use gp_exp
#define gp_atan_use gp_atan
#define gp_sin_use gp_sin_core
#define gp_cos_use gp_cos_core

#else // !BEAM_DOUBLE — fp32 fast path (no parity requirement)

#define gp_pow_use pow
#define gp_exp_use exp
#define gp_atan_use atan
#define gp_sin_use sin
#define gp_cos_use cos

#endif // BEAM_DOUBLE
