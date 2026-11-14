// Package asset: Radiance (.hdr) HDR environment maps.
//
// Decodes 32-bit_rle_rgbe into the same linear-float Texture container
// used by albedo maps, so the GPU upload path is shared. Equirectangular
// layout: row 0 = zenith, last row = nadir, 2π horizontal wrap.

package asset

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/wltechblog/beam/geom"
)

// LoadEnvMap loads an equirectangular environment map. .hdr files decode
// through the Radiance reader (true HDR values); .png/.jpg load as LDR
// (sRGB decode, values 0..1) for convenience.
func LoadEnvMap(path string) (*Texture, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".hdr", ".pic":
		return LoadHDR(path)
	default:
		return LoadTexture(path)
	}
}

// LoadHDR decodes a Radiance RGBE .hdr file into a linear float Texture.
func LoadHDR(path string) (*Texture, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReader(f)

	format, W, H, flipY, flipX, err := parseHDRHeader(br)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if !strings.Contains(format, "32-bit_rle_rgbe") {
		return nil, fmt.Errorf("%s: unsupported FORMAT %q (want 32-bit_rle_rgbe)", path, format)
	}

	t := &Texture{W: W, H: H, Pix: make([]float32, W*H*3)}
	scan := make([]byte, W*4)
	for y := 0; y < H; y++ {
		if err := readScanline(br, scan, W); err != nil {
			return nil, fmt.Errorf("%s: scanline %d: %w", path, y, err)
		}
		row := y * W * 3
		for x := 0; x < W; x++ {
			r, g, b := rgbeToFloat(scan[x], scan[W+x], scan[2*W+x], scan[3*W+x])
			i := row + x*3
			t.Pix[i], t.Pix[i+1], t.Pix[i+2] = r, g, b
		}
	}
	if flipX {
		flipTextureX(t)
	}
	if flipY {
		flipTextureY(t)
	}
	return t, nil
}

// parseHDRHeader consumes the info header lines, then the blank
// terminator, then the resolution line (which follows the blank line
// per the Radiance spec).
func parseHDRHeader(br *bufio.Reader) (format string, W, H int, flipY, flipX bool, err error) {
	magic, err := br.ReadString('\n')
	if err != nil {
		return "", 0, 0, false, false, err
	}
	if !strings.HasPrefix(magic, "#?RADIANCE") && !strings.HasPrefix(magic, "#?RGBE") {
		return "", 0, 0, false, false, fmt.Errorf("not a Radiance file (magic %q)", strings.TrimSpace(magic))
	}
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return "", 0, 0, false, false, err
		}
		line = strings.TrimRight(line, "\r\n")
		if i := strings.Index(line, "FORMAT="); i >= 0 {
			format = strings.TrimSpace(line[i+len("FORMAT="):])
			continue
		}
		if line == "" {
			// Blank line ends the info header; the RESOLUTION line
			// follows immediately (e.g. "-Y 512 +X 1024").
			res, rerr := br.ReadString('\n')
			if rerr != nil {
				return "", 0, 0, false, false, rerr
			}
			return parseResolution(format, strings.TrimRight(res, "\r\n"))
		}
	}
}

// parseResolution decodes the "-Y H +X W" axis line.
func parseResolution(format, res string) (string, int, int, bool, bool, error) {
	fields := strings.Fields(res)
	if len(fields) != 4 || len(fields[0]) < 2 || len(fields[2]) < 2 {
		return "", 0, 0, false, false, fmt.Errorf("bad resolution line %q", res)
	}
	var W, H int
	flipY, flipX := false, false
	for k := 0; k < 4; k += 2 {
		negSign := fields[k][0] == '-'
		tag := fields[k][1]
		n := 0
		if _, err := fmt.Sscanf(fields[k+1], "%d", &n); err != nil {
			return "", 0, 0, false, false, fmt.Errorf("bad resolution %q", res)
		}
		switch tag {
		case 'Y':
			H = n
			flipY = !negSign // +Y = rows stored bottom-up; -Y = top-down (standard)
		case 'X':
			W = n
			flipX = negSign // -X = right-to-left
		}
	}
	if W <= 0 || H <= 0 {
		return "", 0, 0, false, false, fmt.Errorf("unresolved dimensions in %q", res)
	}
	return format, W, H, flipY, flipX, nil
}

// readScanline decompresses one new-style RLE scanline (or reads a
// plain old-style row of RGBE quads when the magic is absent).
func readScanline(br *bufio.Reader, dst []byte, W int) error {
	if W < 8 || W > 0x7fff {
		_, err := ioReadFull(br, dst[:W*4]) // old-style, unencoded
		return err
	}
	var hdr [4]byte
	if _, err := ioReadFull(br, hdr[:]); err != nil {
		return err
	}
	if hdr[0] == 2 && hdr[1] == 2 && int(hdr[2])<<8|int(hdr[3]) == W {
		// new-style RLE: one count stream per channel
		for ch := 0; ch < 4; ch++ {
			off := ch * W
			for pos := 0; pos < W; {
				b, err := br.ReadByte()
				if err != nil {
					return err
				}
				if b > 128 { // run of (b-128) identical bytes
					cnt := int(b) - 128
					if cnt == 0 || pos+cnt > W {
						return fmt.Errorf("bad run count")
					}
					v, err := br.ReadByte()
					if err != nil {
						return err
					}
					for i := 0; i < cnt; i++ {
						dst[off+pos+i] = v
					}
					pos += cnt
				} else { // b literal bytes follow
					cnt := int(b)
					if cnt == 0 || pos+cnt > W {
						return fmt.Errorf("bad literal count")
					}
					if _, err := ioReadFull(br, dst[off+pos:off+pos+cnt]); err != nil {
						return err
					}
					pos += cnt
				}
			}
		}
		return nil
	}
	// old-style: the 4 bytes already read are the first pixel
	copy(dst, hdr[:])
	if W*4 > 4 {
		if _, err := ioReadFull(br, dst[4:]); err != nil {
			return err
		}
	}
	return nil
}

func ioReadFull(br *bufio.Reader, p []byte) (int, error) {
	n := 0
	for n < len(p) {
		m, err := br.Read(p[n:])
		n += m
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// rgbeToFloat converts one RGBE quad to linear floats. E=0 is the
// canonical zero (black) marker.
func rgbeToFloat(r, g, b, e uint8) (float32, float32, float32) {
	if e == 0 {
		return 0, 0, 0
	}
	f := math.Ldexp(1.0, int(e)-128-8)
	rf := (float64(r) + 0.5) / 256 * f
	gf := (float64(g) + 0.5) / 256 * f
	bf := (float64(b) + 0.5) / 256 * f
	return float32(rf), float32(gf), float32(bf)
}

func flipTextureX(t *Texture) {
	for y := 0; y < t.H; y++ {
		row := y * t.W * 3
		for x := 0; x < t.W/2; x++ {
			a := row + x*3
			b := row + (t.W-1-x)*3
			for c := 0; c < 3; c++ {
				t.Pix[a+c], t.Pix[b+c] = t.Pix[b+c], t.Pix[a+c]
			}
		}
	}
}

func flipTextureY(t *Texture) {
	row := make([]float32, t.W*3)
	for y := 0; y < t.H/2; y++ {
		a := y * t.W * 3
		b := (t.H - 1 - y) * t.W * 3
		copy(row, t.Pix[a:a+t.W*3])
		copy(t.Pix[a:a+t.W*3], t.Pix[b:b+t.W*3])
		copy(t.Pix[b:b+t.W*3], row)
	}
}

// SampleDir samples the equirectangular map along dir.
//
//	u = atan2(dir.X, dir.Z) / 2π + 0.5   (seam at dir.Z < 0; wraps; +Z = image center)
//	v = acos(dir.Y) / π                  (v=0 zenith = image TOP row, v=1 nadir = bottom)
//
// Bilinear with horizontal wrap and vertical clamp. Orientation contract:
// standard equirectangular layout (Poly Haven etc.) — row 0 is the zenith.
// The GPU port (env_sample) matches modulo fp precision.
func (t *Texture) SampleDir(dir geom.Vec3) geom.Vec3 {
	u := math.Atan2(dir.X, dir.Z)/(2*math.Pi) + 0.5
	u -= math.Floor(u)
	v := math.Acos(math.Max(-1, math.Min(1, dir.Y))) / math.Pi
	v = math.Max(0, math.Min(1, v))

	fx := u*float64(t.W) - 0.5
	fy := v*float64(t.H) - 0.5
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	tx := fx - float64(x0)
	ty := fy - float64(y0)
	x1, y1 := x0+1, y0+1
	x0 = ((x0 % t.W) + t.W) % t.W
	x1 = ((x1 % t.W) + t.W) % t.W
	y0 = clampI(y0, 0, t.H-1)
	y1 = clampI(y1, 0, t.H-1)

	p00 := t.Pix[(y0*t.W+x0)*3:]
	p10 := t.Pix[(y0*t.W+x1)*3:]
	p01 := t.Pix[(y1*t.W+x0)*3:]
	p11 := t.Pix[(y1*t.W+x1)*3:]

	return geom.Vec3{
		X: bilin(p00[0], p10[0], p01[0], p11[0], tx, ty),
		Y: bilin(p00[1], p10[1], p01[1], p11[1], tx, ty),
		Z: bilin(p00[2], p10[2], p01[2], p11[2], tx, ty),
	}
}

func bilin(p00, p10, p01, p11 float32, tx, ty float64) float64 {
	top := float64(p00)*(1-tx) + float64(p10)*tx
	bot := float64(p01)*(1-tx) + float64(p11)*tx
	return top*(1-ty) + bot*ty
}

// MeanColor returns the unweighted mean of all pixels — the compile-time
// ambient tint for image-based-lighting-lite.
func (t *Texture) MeanColor() geom.Vec3 {
	var sum geom.Vec3
	n := float64(t.W * t.H)
	for i := 0; i < len(t.Pix); i += 3 {
		sum.X += float64(t.Pix[i])
		sum.Y += float64(t.Pix[i+1])
		sum.Z += float64(t.Pix[i+2])
	}
	return sum.MulS(1 / n)
}

func clampI(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
