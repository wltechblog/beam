package gpu

import (
	"image"
	"time"

	"github.com/wltechblog/beam/render"
)

// Hook adapts RenderGPU to render.GPUHook's signature (fp32 build, first
// suitable device — probe ranking puts discrete Arc cards first).
func Hook(w *render.World, cam *render.Camera, width, height, samples, maxDepth int,
	exposure float64, clip bool, seed int64) (*image.RGBA, time.Duration, error) {
	return RenderGPU(w, cam, width, height, samples, maxDepth, exposure, clip, seed, false)
}
