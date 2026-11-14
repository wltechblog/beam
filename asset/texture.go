// Package asset: texture loading. PNG and JPEG via the Go stdlib —
// zero external dependencies. Images decode to linear-space RGB (sRGB
// -> linear 2.2 transfer, the same convention the tonemapper's sRGB
// encode expects), stored as planar float32 rows for the CPU sampler
// and uploaded flat to the GPU.
package asset

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Texture is a decoded image in linear RGB, top row first (image row
// order; texture V grows UP, samplers flip when indexing).
type Texture struct {
	W, H int
	// Pix is 3 floats/pixel, row-major from the top-left (y=0 = top).
	Pix []float32
}

// srgbToLinear approximates the sRGB transfer curve with the classic
// 2.2 gamma. Exact-piecewise would differ by <1.5% and Beam's checker/
// solid colors have always been authored as plain linear values — a
// single consistent convention beats two similar ones.
func srgbToLinear(c uint32) float32 {
	v := float64(c) / 65535.0
	return float32(math.Pow(v, 2.2))
}

// LoadTexture loads a PNG/JPEG texture and converts to linear float32.
func LoadTexture(path string) (*Texture, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var img image.Image
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png":
		img, err = png.Decode(f)
	case ".jpg", ".jpeg":
		img, err = jpeg.Decode(f)
	default:
		// decode by content sniffing
		_, err = f.Seek(0, 0)
		if err != nil {
			return nil, err
		}
		cfg, fmtName, err2 := image.DecodeConfig(f)
		if err2 != nil {
			return nil, err2
		}
		_ = cfg
		if fmtName != "png" && fmtName != "jpeg" {
			return nil, fmt.Errorf("%s: unsupported texture format %q (png/jpg only)", path, fmtName)
		}
		_, err = f.Seek(0, 0)
		if err != nil {
			return nil, err
		}
		img, _, err = image.Decode(f)
	}
	if err != nil {
		return nil, err
	}

	b := img.Bounds()
	t := &Texture{W: b.Dx(), H: b.Dy(), Pix: make([]float32, b.Dx()*b.Dy()*3)}
	i := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA()
			t.Pix[i+0] = srgbToLinear(r)
			t.Pix[i+1] = srgbToLinear(g)
			t.Pix[i+2] = srgbToLinear(bl)
			i += 3
		}
	}
	return t, nil
}

// SampleLinear bilinearly samples the texture at (u, v) with V UP
// (image row y=0 is the BOTTOM of texture space). Wrap addressing on
// both axes (tiling textures are the norm for free-asset packs).
func (t *Texture) SampleLinear(u, v float64) (r, g, b float32) {
	if t == nil || t.W == 0 || t.H == 0 {
		return 1, 1, 1
	}
	// wrap to [0,1)
	u = u - math.Floor(u)
	v = v - math.Floor(v)
	// v up -> row from bottom: row 0 = bottom = image row H-1
	fx := u * float64(t.W)
	fy := (1 - v) * float64(t.H)
	x0 := int(math.Floor(fx)) % t.W
	y0 := int(math.Floor(fy)) % t.H
	x1 := (x0 + 1) % t.W
	y1 := (y0 + 1) % t.H
	ax := float32(fx - math.Floor(fx))
	ay := float32(fy - math.Floor(fy))
	idx := func(x, y int) int { return (y*t.W + x) * 3 }
	p00 := t.Pix[idx(x0, y0):]
	p10 := t.Pix[idx(x1, y0):]
	p01 := t.Pix[idx(x0, y1):]
	p11 := t.Pix[idx(x1, y1):]
	w00 := (1 - ax) * (1 - ay)
	w10 := ax * (1 - ay)
	w01 := (1 - ax) * ay
	w11 := ax * ay
	return p00[0]*w00 + p10[0]*w10 + p01[0]*w01 + p11[0]*w11,
		p00[1]*w00 + p10[1]*w10 + p01[1]*w01 + p11[1]*w11,
		p00[2]*w00 + p10[2]*w10 + p01[2]*w01 + p11[2]*w11
}
