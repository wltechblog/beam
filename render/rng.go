package render

// RNG is a deterministic splitmix64 generator. One instance per pixel,
// seeded from (seed, x, y) — the core of Beam's reproducibility guarantee:
// same scene + seed = byte-identical PNG on any machine.
type RNG struct{ s uint64 }

func NewRNG(seed uint64) *RNG { return &RNG{s: seed} }

// SeedPixel derives a per-pixel seed from the render seed + coordinates.
func SeedPixel(base int64, x, y int) uint64 {
	h := uint64(base)*0x9E3779B97F4A7C15 + uint64(x)*0xBF58476D1CE4E5B9 + uint64(y)*0x94D049BB133111EB
	h ^= h >> 30
	h *= 0xBF58476D1CE4E5B9
	h ^= h >> 27
	h *= 0x94D049BB133111EB
	h ^= h >> 31
	return h
}

func (r *RNG) Next() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z ^= z >> 30
	z *= 0xBF58476D1CE4E5B9
	z ^= z >> 27
	z *= 0x94D049BB133111EB
	z ^= z >> 31
	return z
}

func (r *RNG) Float64() float64 {
	return float64(r.Next()>>11) / float64(1<<53)
}
