package render

import (
	"math"
	"testing"
)

func TestBeamMathBreakdown(t *testing.T) {
	var e, l, a, p int
	for i := 0; i < 100000; i++ {
		x := float64(i)*0.0017 - 85.0
		if x <= 0 {
			x = 0.0001
		}
		if math.Float64bits(beamExp(x)) != math.Float64bits(math.Exp(x)) {
			e++
		}
		if math.Float64bits(beamLog(x)) != math.Float64bits(math.Log(x)) {
			l++
		}
		y := float64(i)*0.00003 - 1.5
		if math.Float64bits(beamAtan(y)) != math.Float64bits(math.Atan(y)) {
			a++
		}
		base := float64(i%997) / 996 * 1.2
		exp := float64(i % 8191)
		if math.Float64bits(beamPow(base, exp)) != math.Float64bits(math.Pow(base, exp)) {
			p++
		}
	}
	t.Logf("exp:%d log:%d atan:%d pow:%d (of 100k each)", e, l, a, p)
}

func TestBeamSinCos(t *testing.T) {
	d := 0
	for i := 0; i < 200000; i++ {
		x := float64(i)*0.0001 - 10.0
		if math.Float64bits(beamSin(x)) != math.Float64bits(math.Sin(x)) {
			d++
		}
		if math.Float64bits(beamCos(x)) != math.Float64bits(math.Cos(x)) {
			d++
		}
	}
	// large args (Payne-Hanek path)
	for i := 0; i < 5000; i++ {
		x := float64(i)*3e7 + 1e8
		if math.Float64bits(beamSin(x)) != math.Float64bits(math.Sin(x)) {
			d++
		}
	}
	t.Logf("sin/cos bit-mismatch: %d", d)
	if d != 0 {
		t.Errorf("sin/cos mismatch: %d", d)
	}
}
