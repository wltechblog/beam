package render

import (
	"fmt"
	"testing"

	"github.com/wltechblog/beam/scene"
)

// TestEnvProbeSkyTop prints actual env radiance seen by sky pixels
// (kept verbose for calibration work; no assertion).
func TestEnvProbeSkyTop(t *testing.T) {
	sc, err := scene.Load("../pov-benchmark/beam-benchmark.json")
	if err != nil {
		t.Skip("scene not present:", err)
	}
	w, err := Compile(sc, NewFileResolver("../pov-benchmark"))
	if err != nil {
		t.Fatal(err)
	}
	cam := NewCamera(vec3Of(sc.Camera.Pos), vec3Of(sc.Camera.LookAt), sc.Camera.FOV, 1920.0/1080.0, 0, 0)
	r := &Renderer{World: w, Camera: cam}
	// probe the reference rows: py = 1 - fy (image y grows down)
	for _, p := range [][2]float64{{.5, .997}, {.5, .99}, {.5, .96}, {.5, .92}, {.5, .88}, {.5, .84}, {.5, .80}} {
		ray := cam.Ray(p[0], p[1], 0, 0, nil) // NORMALIZED coords [0,1]
		env := r.envAt(ray.Dir)
		fmt.Printf("py=%.3f dir=(%.3f,%.3f,%.3f) env=(%.2f,%.2f,%.2f)\n",
			p[1], ray.Dir.X, ray.Dir.Y, ray.Dir.Z, env.X, env.Y, env.Z)
	}
}
