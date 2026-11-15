package render

import (
	"sync"

	"github.com/wltechblog/beam/geom"
)

// bvhCache memoizes BVH construction by mesh pointer. transformMesh (and the
// model/text compile paths) always allocates a fresh mesh per Compile call,
// so pointer identity is a sound cache key for those. The animation
// timeline clones + re-transforms meshes every frame; without this cache
// each frame rebuilt a full BVH for every object (~100K-tri scenes wasted
// minutes of build time across thousands of frames). Meshes are immutable
// after compile, so cached BVHs are always valid.
var bvhCache sync.Map // *geom.TriangleMesh -> *geom.BVH

// buildMeshCached returns a cached BVH for m, building one on miss.
func buildMeshCached(m *geom.TriangleMesh) *geom.BVH {
	if v, ok := bvhCache.Load(m); ok {
		return v.(*geom.BVH)
	}
	b := geom.BuildMesh(m)
	bvhCache.Store(m, b)
	return b
}
