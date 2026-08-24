package render

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/wltechblog/beam/scene"
)

// TestScreenFitVertexCrossCheck projects EVERY mesh vertex (not just the 8
// AABB corners) and verifies the corner-based AABB projection is a valid
// superset. Guards against a loose-corners bug in the fit estimate.
func TestScreenFitVertexCrossCheck(t *testing.T) {
	path := filepath.Join("..", "full.json")
	if _, err := os.Stat(path); err != nil {
		t.Skip("full.json not present")
	}
	s, err := scene.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	s.ApplyDefaults()

	world, err := Compile(s, NewFileResolver(filepath.Dir(path)))
	if err != nil {
		t.Fatal(err)
	}
	w, h := s.Render.Width, s.Render.Height
	cam := NewCamera(vec3Of(s.Camera.Pos), vec3Of(s.Camera.LookAt), s.Camera.FOV,
		float64(w)/float64(h), 0, 0)

	for i, obj := range world.Objects {
		if obj.Kind != 'm' {
			continue
		}
		minU, maxU := math.Inf(1), math.Inf(-1)
		minV, maxV := math.Inf(1), math.Inf(-1)
		for _, v := range obj.Mesh.Verts {
			u, vv, ok := projectPoint(cam, v, w, h)
			if !ok {
				continue
			}
			minU = math.Min(minU, u)
			maxU = math.Max(maxU, u)
			minV = math.Min(minV, vv)
			maxV = math.Max(maxV, vv)
		}
		t.Logf("object[%d] vertex-exact: x[%.1f,%.1f] y[%.1f,%.1f] of %dx%d",
			i, minU, maxU, minV, maxV, w, h)

		b, _ := objectBounds(obj)
		cMinU, cMaxU := math.Inf(1), math.Inf(-1)
		cMinV, cMaxV := math.Inf(1), math.Inf(-1)
		for ci := 0; ci < 8; ci++ {
			u, vv, ok := projectPoint(cam, corner(b, ci), w, h)
			if !ok {
				continue
			}
			cMinU = math.Min(cMinU, u)
			cMaxU = math.Max(cMaxU, u)
			cMinV = math.Min(cMinV, vv)
			cMaxV = math.Max(cMaxV, vv)
		}
		// Corner hull must contain the vertex-exact hull (AABB ⊇ mesh).
		if cMinU > minU+1 || cMaxU < maxU-1 || cMinV > minV+1 || cMaxV < maxV-1 {
			t.Errorf("object[%d]: corner hull x[%.1f,%.1f] y[%.1f,%.1f] does not contain "+
				"vertex hull x[%.1f,%.1f] y[%.1f,%.1f]",
				i, cMinU, cMaxU, cMinV, cMaxV, minU, maxU, minV, maxV)
		}
	}
}
