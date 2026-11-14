package render

import (
	"fmt"
	"github.com/qmuntal/gltf"
	"github.com/wltechblog/beam/asset"
	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/scene"
)

// compileModel imports a glTF/GLB model: one beam object per glTF
// primitive (material group), materials mapped from the glTF PBR
// metallic-roughness model, textures decoded from the document and added
// to the world texture table. Returns the objects; the caller appends
// them to w.Objects.
//
// Material mapping (glTF PBR → beam):
//   - metallic ≥ 0.5          → beam metal (tinted conductor; reflectivity
//     = 1 - roughness²·0.5, roughness passed through)
//   - alphaMode BLEND/MASK with texture  → approximated: still opaque
//     (no transmission without a glass kind)
//   - emissive-only materials  → beam emissive (power = mean factor × 4);
//     emissive with a PBR base renders shaded, glow dropped
//   - otherwise               → matte with baseColorFactor / base texture
//
// The scene-level MaterialOverride replaces every part's material with a
// named scene material (useful for stylization). ScaleHint rescales so
// the longest bbox edge = ScaleHint. The object's own Transform applies
// after import scaling.
func compileModel(
	lm *asset.LoadedModel,
	o *scene.Object,
	mats *[]CompiledMaterial,
	matIdx map[string]int,
	textures *[]*asset.Texture,
	texIdx map[string]int,
) []*Object3D {
	// Collect valid parts and compute the model-space bbox for ScaleHint.
	var parts []asset.ModelPart
	var min, max geom.Vec3
	first := true
	for _, p := range lm.Parts {
		if len(p.Mesh.Tris) == 0 {
			continue
		}
		parts = append(parts, p)
		for _, v := range p.Mesh.Verts {
			if first {
				min, max = v, v
				first = false
				continue
			}
			if v.X < min.X {
				min.X = v.X
			}
			if v.Y < min.Y {
				min.Y = v.Y
			}
			if v.Z < min.Z {
				min.Z = v.Z
			}
			if v.X > max.X {
				max.X = v.X
			}
			if v.Y > max.Y {
				max.Y = v.Y
			}
			if v.Z > max.Z {
				max.Z = v.Z
			}
		}
	}
	if len(parts) == 0 {
		return nil
	}

	// Import-space normalization: center on origin (XZ), drop to y=0
	// floor, and optionally scale to ScaleHint.
	cx := (min.X + max.X) / 2
	cz := (min.Z + max.Z) / 2
	scale := 1.0
	if o.ScaleHint > 0 {
		edge := max.X - min.X
		if y := max.Y - min.Y; y > edge {
			edge = y
		}
		if z := max.Z - min.Z; z > edge {
			edge = z
		}
		if edge > 0 {
			scale = o.ScaleHint / edge
		}
	}

	// Material cache: glTF material index → compiled beam material index.
	matCache := map[int]int{}
	modelMatIdx := func(gi int, name string) int {
		if o.MaterialOverride != "" {
			if mi, ok := matIdx[o.MaterialOverride]; ok {
				return mi
			}
		}
		if gi < 0 {
			return 0 // "__default"
		}
		if mi, ok := matCache[gi]; ok {
			return mi
		}
		var cm CompiledMaterial
		if gi < len(lm.Materials) && lm.Materials[gi] != nil {
			cm = compileGLTFMaterial(lm.Materials[gi], lm, textures, texIdx)
		} else {
			cm = defaultMat()
		}
		*mats = append(*mats, cm)
		mi := len(*mats) - 1
		matCache[gi] = mi
		return mi
	}

	var out []*Object3D
	for _, p := range parts {
		mesh := p.Mesh
		if scale != 1 || cx != 0 || cz != 0 {
			mesh = recenterMesh(mesh, cx, min.Y, cz, scale)
		}
		if o.Transform != nil {
			mesh = transformMesh(mesh, o.Transform)
		}
		mi := modelMatIdx(p.MatIdx, p.MatName)
		out = append(out, &Object3D{
			Kind: 'm', Mesh: mesh, BVH: geom.BuildMesh(mesh), MatIdx: mi,
		})
	}
	return out
}

// recenterMesh translates by (-cx, -minY, -cz) and scales by s (uniform).
func recenterMesh(m *geom.TriangleMesh, cx, minY, cz, s float64) *geom.TriangleMesh {
	out := &geom.TriangleMesh{
		Verts:  make([]geom.Vec3, len(m.Verts)),
		Tris:   m.Tris,
		Norms:  m.Norms, // uniform scale: directions unchanged
		MatIdx: m.MatIdx,
		UVs:    m.UVs,
	}
	for i, v := range m.Verts {
		out.Verts[i] = geom.Vec3{
			X: (v.X - cx) * s,
			Y: (v.Y - minY) * s,
			Z: (v.Z - cz) * s,
		}
	}
	return out
}

// compileGLTFMaterial maps one glTF material to a beam material.
func compileGLTFMaterial(gm *gltf.Material, lm *asset.LoadedModel, textures *[]*asset.Texture, texIdx map[string]int) CompiledMaterial {
	cm := CompiledMaterial{Albedo: geom.Vec3{X: 0.8, Y: 0.8, Z: 0.8}, Diffuse: 1, TexIdx: -1}

	var baseFactor [4]float64 = [4]float64{1, 1, 1, 1}
	var baseTex *gltf.TextureInfo
	var metallic, roughness float64 = 1, 1
	if gm.PBRMetallicRoughness != nil {
		pbr := gm.PBRMetallicRoughness
		if pbr.BaseColorFactor != nil {
			baseFactor = *pbr.BaseColorFactor
		}
		baseTex = pbr.BaseColorTexture
		metallic = pbr.MetallicFactorOrDefault()
		roughness = pbr.RoughnessFactorOrDefault()
	}

	// Base color texture: decode through the model's image table and add
	// to the world texture table (dedupe by image identity).
	if baseTex != nil && baseTex.Index >= 0 && baseTex.Index < len(lm.Images) && lm.Images[baseTex.Index] != nil {
		if t, i := modelTexture(lm, baseTex.Index, textures, texIdx); t != nil {
			cm.Tex = t
			cm.TexIdx = i
		}
	}

	cm.Albedo = geom.Vec3{X: baseFactor[0], Y: baseFactor[1], Z: baseFactor[2]}

	// Emissive: glTF semantics are a glow layer OVER PBR shading. Beam
	// has no per-material dual (shaded + glow), so emissive only takes
	// the material over entirely when it is the ONLY visual signal —
	// no base texture and a plain dielectric (neon signs, light
	// fixtures). Otherwise the shaded PBR path wins and the glow is
	// dropped (documented approximation; a future EmissiveAdd term
	// could restore it).
	var ef [3]float64 = [3]float64{gm.EmissiveFactor[0], gm.EmissiveFactor[1], gm.EmissiveFactor[2]}
	emissiveOnly := (ef[0] > 0 || ef[1] > 0 || ef[2] > 0) && cm.Tex == nil && metallic < 0.5
	if emissiveOnly {
		cm.Kind = 'e'
		cm.Power = (ef[0] + ef[1] + ef[2]) / 3 * 4
		// No base texture occupies the single Tex slot — bind the
		// emissive texture there so the glow is properly masked.
		// (Tex and TexIdx stay in sync; the earlier version swapped
		// Tex while leaving TexIdx on the base texture, desyncing
		// the CPU and GPU paths.)
		if gm.EmissiveTexture != nil && gm.EmissiveTexture.Index >= 0 && gm.EmissiveTexture.Index < len(lm.Images) && lm.Images[gm.EmissiveTexture.Index] != nil {
			if t, i := modelTexture(lm, gm.EmissiveTexture.Index, textures, texIdx); t != nil {
				cm.Tex = t
				cm.TexIdx = i
			}
		}
		return cm
	}

	if metallic >= 0.5 {
		if roughness >= 0.9 {
			// A fully-rough conductor scatters like a diffuse surface —
			// glTF spec defaults (metallic 1, roughness 1) hit this when a
			// material omits both factors. A roughness-1 beam metal is a
			// hemisphere-scattering mirror that samples fog/glow from every
			// direction and reads as washed-out white. Dark textured matte
			// is what the surface actually looks like.
			cm.Kind = 'm'
			cm.Albedo = cm.Albedo.MulS(0.55)
			return cm
		}
		cm.Kind = 'M'
		cm.Roughness = roughness
		r := 1 - roughness*roughness*0.5
		if r < 0.04 {
			r = 0.04
		}
		cm.Reflectivity = r
		// glTF metals are tinted conductors by default.
		return cm
	}

	// Dielectric: matte with a hint of metal darkening for mid values.
	cm.Kind = 'm'
	if metallic > 0 {
		cm.Albedo = cm.Albedo.MulS(1 - metallic*0.5)
	}
	return cm
}

// modelTexture converts a decoded model image into a beam texture and
// registers it in the world table. Dedupe identity is model-src + image
// index: different models' img0 are DIFFERENT images — a bare "model:imgN"
// key made every model after the first silently reuse the first model's
// texture (the corridor's lantern rendered with the duck's yellow map).
func modelTexture(lm *asset.LoadedModel, imgIdx int, textures *[]*asset.Texture, texIdx map[string]int) (*asset.Texture, int) {
	key := fmt.Sprintf("%s#img%d", lm.Src, imgIdx)
	if i, ok := texIdx[key]; ok {
		return (*textures)[i], i
	}
	img := lm.Images[imgIdx]
	t := asset.TextureFromImage(img)
	if t == nil {
		return nil, -1
	}
	*textures = append(*textures, t)
	i := len(*textures) - 1
	texIdx[key] = i
	return t, i
}
