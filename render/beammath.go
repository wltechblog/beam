package render

import (
	"math"
	"math/bits"
)

// Beam-owned pure-Go transcendental ports. These are verbatim ports of
// the Go standard library's portable algorithms (exp.go, log.go, atan.go,
// pow.go, frexp.go, ldexp.go, modf.go). Both the CPU renderer AND the GPU
// kernel (gpu/kernel.cl gp_* functions) run these exact algorithms, so a
// float64 GPU render can be bit-identical to the CPU render regardless of
// which platform assembly the host Go build would otherwise use.

const (
	beamLn2Hi = 6.93147180369123816490e-01
	beamLn2Lo = 1.90821492927058770002e-10
	beamLog2e = 1.44269504088896338700e+00
	beamPi    = 3.141592653589793115997963468544185161590576171875
)

// beamExp is the portable Go exp algorithm.
func beamExp(x float64) float64 {
	const (
		Overflow  = 7.09782712893383973096e+02
		Underflow = -7.45133219101941108420e+02
		NearZero  = 1.0 / (1 << 28) // 2**-28
	)
	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case math.IsInf(x, -1):
		return 0
	case x > Overflow:
		return math.Inf(1)
	case x < Underflow:
		return 0
	case -NearZero < x && x < NearZero:
		return 1 + x
	}
	var k int
	switch {
	case x < 0:
		k = int(beamLog2e*x - 0.5)
	case x > 0:
		k = int(beamLog2e*x + 0.5)
	}
	hi := x - float64(k)*beamLn2Hi
	lo := float64(k) * beamLn2Lo
	return beamExpmulti(hi, lo, k)
}

func beamExpmulti(hi, lo float64, k int) float64 {
	const (
		P1 = 1.66666666666666657415e-01  /* 0x3FC55555; 0x55555555 */
		P2 = -2.77777777770155933842e-03 /* 0xBF66C16C; 0x16BEBD93 */
		P3 = 6.61375632143793436117e-05  /* 0x3F11566A; 0xAF25DE2C */
		P4 = -1.65339022054652515390e-06 /* 0xBEBBBD41; 0xC5D26BF1 */
		P5 = 4.13813679705723846039e-08  /* 0x3E663769; 0x72BEA4D0 */
	)
	r := hi - lo
	t := r * r
	c := r - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
	y := 1 - ((lo - (r*c)/(2-c)) - hi)
	return beamLdexp(y, k)
}

// beamLog is the portable Go log algorithm.
func beamLog(x float64) float64 {
	const (
		L1 = 6.666666666666735130e-01
		L2 = 3.999999999940941908e-01
		L3 = 2.857142874366239149e-01
		L4 = 2.222219843214978396e-01
		L5 = 1.818357216161805012e-01
		L6 = 1.531383769920937332e-01
		L7 = 1.479819860511658591e-01
	)
	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(-1)
	}
	f1, ki := beamFrexp(x)
	if f1 < math.Sqrt2/2 {
		f1 *= 2
		ki--
	}
	f := f1 - 1
	k := float64(ki)

	s := f / (2 + f)
	s2 := s * s
	s4 := s2 * s2
	t1 := s2 * (L1 + s4*(L3+s4*(L5+s4*L7)))
	t2 := s4 * (L2 + s4*(L4+s4*L6))
	R := t1 + t2
	hfsq := 0.5 * f * f
	return k*beamLn2Hi - ((hfsq - (s*(hfsq+R) + k*beamLn2Lo)) - f)
}

// beamAtan is the portable Go atan algorithm (atan.go xatan/satan).
func beamAtan(x float64) float64 {
	if x == 0 {
		return x
	}
	if x > 0 {
		return beamSatan(x)
	}
	return -beamSatan(-x)
}

func beamXatan(x float64) float64 {
	const (
		P0 = -8.750608600031904122785e-01
		P1 = -1.615753718733365076637e+01
		P2 = -7.500855792314704667340e+01
		P3 = -1.228866684490136173410e+02
		P4 = -6.485021904942025371773e+01
		Q0 = +2.485846490142306297962e+01
		Q1 = +1.650270098316988542046e+02
		Q2 = +4.328810604912902668951e+02
		Q3 = +4.853903996359136964868e+02
		Q4 = +1.945506571482613964425e+02
	)
	z := x * x
	z = z * ((((P0*z+P1)*z+P2)*z+P3)*z + P4) / (((((z+Q0)*z+Q1)*z+Q2)*z+Q3)*z + Q4)
	z = x*z + x
	return z
}

func beamSatan(x float64) float64 {
	const (
		Morebits = 6.123233995736765886130e-17 // pi/2 = PIO2 + Morebits
		Tan3pio8 = 2.41421356237309504880      // tan(3*pi/8)
	)
	if x <= 0.66 {
		return beamXatan(x)
	}
	if x > Tan3pio8 {
		return beamPi/2 - beamXatan(1/x) + Morebits
	}
	return beamPi/4 + beamXatan((x-1)/(x+1)) + 0.5*Morebits
}

// beamPow is the portable Go pow algorithm.
func beamPow(x, y float64) float64 {
	switch {
	case y == 0 || x == 1:
		return 1
	case y == 1:
		return x
	case math.IsNaN(x) || math.IsNaN(y):
		return math.NaN()
	case x == 0:
		switch {
		case y < 0:
			if math.Signbit(x) && beamIsOddInt(y) {
				return math.Inf(-1)
			}
			return math.Inf(1)
		case y > 0:
			if math.Signbit(x) && beamIsOddInt(y) {
				return x
			}
			return 0
		}
	case math.IsInf(y, 0):
		switch {
		case x == -1:
			return 1
		case (math.Abs(x) < 1) == math.IsInf(y, 1):
			return 0
		default:
			return math.Inf(1)
		}
	case math.IsInf(x, 0):
		if math.IsInf(x, -1) {
			return beamPow(1/x, -y)
		}
		switch {
		case y < 0:
			return 0
		case y > 0:
			return math.Inf(1)
		}
	case y == 0.5:
		return math.Sqrt(x)
	case y == -0.5:
		return 1 / math.Sqrt(x)
	}

	yi, yf := beamModf(math.Abs(y))
	if yf != 0 && x < 0 {
		return math.NaN()
	}
	if yi >= 1<<63 {
		switch {
		case x == -1:
			return 1
		case (math.Abs(x) < 1) == (y > 0):
			return 0
		default:
			return math.Inf(1)
		}
	}

	a1 := 1.0
	ae := 0

	if yf != 0 {
		if yf > 0.5 {
			yf--
			yi++
		}
		a1 = beamExp(yf * beamLog(x))
	}

	x1, xe := beamFrexp(x)
	for i := int64(yi); i != 0; i >>= 1 {
		if xe < -1<<12 || 1<<12 < xe {
			ae += xe
			break
		}
		if i&1 == 1 {
			a1 *= x1
			ae += xe
		}
		x1 *= x1
		xe <<= 1
		if x1 < .5 {
			x1 += x1
			xe--
		}
	}

	if y < 0 {
		a1 = 1 / a1
		ae = -ae
	}
	return beamLdexp(a1, ae)
}

func beamIsOddInt(x float64) bool {
	if math.Abs(x) >= (1 << 53) {
		return false
	}
	xi, xf := beamModf(x)
	return xf == 0 && int64(xi)&1 == 1
}

// beamFrexp is the portable Go frexp (IEEE bit manipulation).
func beamFrexp(f float64) (frac float64, exp int) {
	switch {
	case f == 0:
		return f, 0
	case math.IsInf(f, 0) || math.IsNaN(f):
		return f, 0
	}
	f, exp = beamNormalize(f)
	x := math.Float64bits(f)
	exp += int((x>>52)&0x7FF) - 1023 + 1
	x &^= 0x7FF << 52
	x |= (-1 + 1023) << 52
	frac = math.Float64frombits(x)
	return
}

func beamNormalize(x float64) (y float64, exp int) {
	const SmallestNormal = 2.2250738585072014e-308
	if math.Abs(x) < SmallestNormal {
		return x * (1 << 52), -52
	}
	return x, 0
}

// beamLdexp is the portable Go ldexp.
func beamLdexp(frac float64, exp int) float64 {
	frac, e := beamNormalize(frac)
	exp += e
	x := math.Float64bits(frac)
	exp += int(x>>52)&0x7FF - 1023
	if exp < -1075 {
		return math.Copysign(0, frac) // underflow
	}
	if exp > 1023 { // overflow
		if frac < 0 {
			return math.Inf(-1)
		}
		return math.Inf(1)
	}
	var m float64 = 1
	if exp < -1022 { // denormal
		exp += 53
		m = 1.0 / (1 << 53) // 2**-53
	}
	x &^= 0x7FF << 52
	x |= uint64(exp+1023) << 52
	return m * math.Float64frombits(x)
}

// beamModf is the portable Go modf.
func beamModf(f float64) (intg float64, frac float64) {
	if f < 1 {
		switch {
		case f < 0:
			intg, frac = beamModf(-f)
			return -intg, -frac
		case f == 0:
			return f, f
		}
		return 0, f
	}
	x := math.Float64bits(f)
	e := uint(x>>52)&0x7FF - 1023
	if e < 64-12 {
		x &^= 1<<(64-12-e) - 1
	}
	intg = math.Float64frombits(x)
	frac = f - intg
	return
}

// ---- sin/cos: portable Go algorithms incl. Payne-Hanek reduction ----
// Needed because OpenCL sin()/cos() differ from Go's; DOF camera math
// consumes RNG, so lens-sample directions must match bit-exactly.

const beamReduceThreshold = 1 << 29

var beamMPi4 = [...]uint64{
	0x0000000000000001, 0x45f306dc9c882a53, 0xf84eafa3ea69bb81,
	0xb6c52b3278872083, 0xfca2c757bd778ac3, 0x6e48dc74849ba5c0,
	0x0c925dd413a32439, 0xfc3bd63962534e7d, 0xd1046bea5d768909,
	0xd338e04d68befc82, 0x7323ac7306a673e9, 0x3908bf177bf25076,
	0x3ff12fffbc0b301f, 0xde5e2316b414da3e, 0xda6cfd9e4f96136e,
	0x9e8c7ecd3cbfd45a, 0xea4f758fd7cbe2f6, 0x7a0e73ef14a525d4,
	0xd7f6bf623f1aba10, 0xac06608df8f6d757,
}

var beamSinC = [...]float64{
	1.58962301576546568060e-10,
	-2.50507477628578072866e-8,
	2.75573136213857245213e-6,
	-1.98412698295895385996e-4,
	8.33333333332211858878e-3,
	-1.66666666666666307295e-1,
}

var beamCosC = [...]float64{
	-1.13585365213876817300e-11,
	2.08757008419747316778e-9,
	-2.75573141792967388112e-7,
	2.48015872888517045348e-5,
	-1.38888888888730564116e-3,
	4.16666666666665929218e-2,
}

// beamTrigReduce is Go's trigReduce (Payne-Hanek, math/trig_reduce.go).
func beamTrigReduce(x float64) (j uint64, z float64) {
	const PI4 = beamPi / 4
	if x < PI4 {
		return 0, x
	}
	const (
		bias  = 1023
		shift = 52
		mask  = 0x7FF
	)
	ix := math.Float64bits(x)
	exp := int(ix>>shift&mask) - bias - shift
	ix &^= mask << shift
	ix |= 1 << shift
	digit, bitshift := uint(exp+61)/64, uint(exp+61)%64
	z0 := (beamMPi4[digit] << bitshift) | (beamMPi4[digit+1] >> (64 - bitshift))
	z1 := (beamMPi4[digit+1] << bitshift) | (beamMPi4[digit+2] >> (64 - bitshift))
	z2 := (beamMPi4[digit+2] << bitshift) | (beamMPi4[digit+3] >> (64 - bitshift))
	z2hi, _ := bits.Mul64(z2, ix)
	z1hi, z1lo := bits.Mul64(z1, ix)
	z0lo := z0 * ix
	lo, c := bits.Add64(z1lo, z2hi, 0)
	hi, _ := bits.Add64(z0lo, z1hi, c)
	j = hi >> 61
	hi = hi<<3 | lo>>61
	lz := uint(bits.LeadingZeros64(hi))
	e := uint64(bias - (lz + 1))
	hi = (hi << (lz + 1)) | (lo >> (64 - (lz + 1)))
	hi >>= 64 - shift
	hi |= e << shift
	z = math.Float64frombits(hi)
	if j&1 == 1 {
		j++
		j &= 7
		z--
	}
	return j, z * PI4
}

func beamSin(x float64) float64 {
	const (
		PI4A = 7.85398125648498535156e-1
		PI4B = 3.77489470793079817668e-8
		PI4C = 2.69515142907905952645e-15
	)
	switch {
	case x == 0 || math.IsNaN(x):
		return x
	case math.IsInf(x, 0):
		return math.NaN()
	}
	sign := false
	if x < 0 {
		x = -x
		sign = true
	}
	var j uint64
	var y, z float64
	if x >= beamReduceThreshold {
		j, z = beamTrigReduce(x)
	} else {
		j = uint64(x * (4 / beamPi))
		y = float64(j)
		if j&1 == 1 {
			j++
			y++
		}
		j &= 7
		z = ((x - y*PI4A) - y*PI4B) - y*PI4C
	}
	if j > 3 {
		sign = !sign
		j -= 4
	}
	zz := z * z
	if j == 1 || j == 2 {
		y = 1.0 - 0.5*zz + zz*zz*((((((beamCosC[0]*zz)+beamCosC[1])*zz+beamCosC[2])*zz+beamCosC[3])*zz+beamCosC[4])*zz+beamCosC[5])
	} else {
		y = z + z*zz*((((((beamSinC[0]*zz)+beamSinC[1])*zz+beamSinC[2])*zz+beamSinC[3])*zz+beamSinC[4])*zz+beamSinC[5])
	}
	if sign {
		y = -y
	}
	return y
}

func beamCos(x float64) float64 {
	const (
		PI4A = 7.85398125648498535156e-1
		PI4B = 3.77489470793079817668e-8
		PI4C = 2.69515142907905952645e-15
	)
	switch {
	case math.IsNaN(x) || math.IsInf(x, 0):
		return math.NaN()
	}
	sign := false
	x = math.Abs(x)
	var j uint64
	var y, z float64
	if x >= beamReduceThreshold {
		j, z = beamTrigReduce(x)
	} else {
		j = uint64(x * (4 / beamPi))
		y = float64(j)
		if j&1 == 1 {
			j++
			y++
		}
		j &= 7
		z = ((x - y*PI4A) - y*PI4B) - y*PI4C
	}
	if j > 3 {
		j -= 4
		sign = !sign
	}
	if j > 1 {
		sign = !sign
	}
	zz := z * z
	if j == 1 || j == 2 {
		y = z + z*zz*((((((beamSinC[0]*zz)+beamSinC[1])*zz+beamSinC[2])*zz+beamSinC[3])*zz+beamSinC[4])*zz+beamSinC[5])
	} else {
		y = 1.0 - 0.5*zz + zz*zz*((((((beamCosC[0]*zz)+beamCosC[1])*zz+beamCosC[2])*zz+beamCosC[3])*zz+beamCosC[4])*zz+beamCosC[5])
	}
	if sign {
		y = -y
	}
	return y
}
