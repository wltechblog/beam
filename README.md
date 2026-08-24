# Beam

Data-driven 3D scene renderer with an agent-first interface.

Beam renders scenes described as **versioned JSON** — no scripting language, no
Turing-complete scene DSL. Scenes are data: schema-validatable, diffable, and
trivially generatable by LLMs (an agent can emit a valid scene on the first try
given the schema).

## Design principles

1. **Scenes are data, not programs** — the POV-Ray lesson. Patterns are named
   built-in functions with parameters, not shaders with loops. This keeps
   scenes sandboxable, validatable, and cache-friendly for model generation.
2. **Deterministic by construction** — pure Go, strict IEEE-754 (Go forbids
   fused-multiply-add reassociation), per-pixel RNG seeded by coordinates.
   Same scene + seed = byte-identical PNG on any platform.
3. **CPU now, GPU-ready** — the renderer is embarrassingly parallel (tile
   scheduler + goroutines). The GPU boundary (OpenCL wavefront on the Arc
   A770) will consume flattened scene arrays; the JSON layer, scheduler, and
   output pipeline never change.
4. **Zero heavy dependencies** — stdlib + pure-Go libraries only (glTF loader,
   font outline reader). No CGO anywhere.

## CLI

```
beam render <scene.json> [-o out.png] [--preview] [--quiet]
beam validate <scene.json>
beam version
```

`--preview` renders at 320px wide, 1 sample — fast enough for an agent
write → preview → look → refine loop.

## Scene schema (v1)

```json
{
  "version": 1,
  "camera": {"pos": [0, 1.2, -5], "lookAt": [0, 0.8, 0], "fov": 42,
             "aperture": 0.06},
  "materials": {
    "floor": {"type": "matte", "pattern": {"type": "checker",
               "colors": [[0.92,0.92,0.9], [0.15,0.15,0.18]], "scale": 0.9}},
    "chrome": {"type": "metal", "color": [0.94,0.95,0.97], "roughness": 0.04},
    "glass": {"type": "glass", "color": [0.96,0.98,1.0], "ior": 1.5}
  },
  "objects": [
    {"type": "sphere", "center": [0, 0.9, 0], "radius": 0.9, "material": "glass"},
    {"type": "text", "text": "BEAM", "size": 1.4, "depth": 0.35,
     "material": "chrome", "transform": {"translate": [0, 1.6, -2.5]}}
  ],
  "lights": [{"type": "point", "pos": [5, 7, -4], "intensity": 1.6}],
  "environment": {"color": [0.04, 0.05, 0.08], "ambient": 0.25},
  "render": {"width": 1280, "height": 720, "samples": 2, "seed": 42}
}
```

Object types: `sphere`, `plane`, `box`, `mesh` (OBJ / STL / GLB), `text`
(3D extruded TrueType, embedded default font).

Materials: `matte`, `metal` (roughness), `glass` (IOR), `emissive` (power).
Patterns: `solid`, `checker`, `gradient`.

Lights: `point`, `directional`, `spot`. Camera: perspective with optional
thin-lens depth of field.

## Import formats

- **OBJ + MTL** (v, f, usemtl → auto-materials)
- **STL** (binary + ASCII)
- **glTF 2.0 / GLB** via `qmuntal/gltf` (positions/indices/normals)

## Text

Real 3D extruded text: TrueType glyph outlines (`go-sfnt`) → adaptive curve
flattening → robust ear-clip triangulation (Mapbox earcut algorithm, ported)
→ extruded solid with correct outward-facing normals. Bevel is on the roadmap.

## Layout

```
geom/      vectors, matrices, triangle meshes (GPU-flattenable data)
scene/     schema, validation (JSON-pointer errors), loading
render/    camera, BVH, Whitted integrator, tile scheduler, tonemap
asset/     OBJ / STL / glTF importers
text3d/    glyph outlines → earcut → extrusion
cmd/beam/  CLI
scenes/    example scenes
```

## Roadmap

- Phase 2: MCP server (`render` / `preview` / `validate` / `import`) for the
  agent iterate loop; golden-test ports of classic POV sample scenes
- Phase 3: OpenCL device behind the geom boundary (Intel Arc A770)
- Phase 4: timeline animation (keyframes, ease curves), volumetric
  fire/plasma via ray-marched fbm fields, particle instancing (rain)
- Later: WASM shader escape hatch, bevel text, CSG
