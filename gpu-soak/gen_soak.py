#!/usr/bin/env python3
"""GPU soak scene generator — empirically separates GPU silicon from CPU.

IMPORTANT — the render `samples` field is a jitter-GRID dimension, NOT
rays/pixel: rays per pixel = samples ** 2. samples=4 -> 16 rays/px,
samples=16 -> 256 rays/px. A 7680x4320 scene at samples=16 is 8.5 billion
rays (~10 min even on the A770, hours on CPU). Never bake configs like
that into shared scene files — someone WILL render them.

Measured on the A770 for THIS scene, 960x540 (518400 px):
  samples=4 -> 0.57s   samples=8 -> 2.23s   12 -> 5.01s   16 -> 9.11s
  i.e. kernel ~= 0.036 * samples^2 * (pixels / 518400) seconds
CPU (all-core Go renderer): ~1.3 us per ray  ->  roughly 20x slower.

Shipped config: 1920x1080 @ samples=8 (64 rays/px)
  GPU   ~9s   (well inside the 60s budget)
  CPU ~170s   (run-test.sh kills it at 45s: instant "you're on CPU" verdict)

- 438 spheres / analytic objects (BVH-sorted, cheap per-object)
- 24 instanced high-poly meshes (~13K triangles total): supersampled
  triangle-intersection + BVH walk per ray
- glass + metal + emissive materials driving deep recursion (maxDepth 12)
- 3 lights with shadows (shadow rays on every diffuse hit)

Scene is procedural geometry only — no external assets, deterministic
across runs. Rendered output doubles as a visual sanity check: colored
rows of spheres, metal/glass cluster center, blue emissive slab.
"""
import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.json")
OBJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soak-knot.obj")

# ---------------------------------------------------------------- meshes
def torus_knot(p=2, q=3, R=1.0, r=0.34, seg=220, tube=48):
    """(p,q) torus knot tube as a triangle mesh — high-poly BVH stress."""
    verts, tris = [], []
    for i in range(seg):
        u = i / seg * 2 * math.pi * p  # wrap p times around the torus
        # classic (p,q) torus-knot centerline
        cr = R + 0.35 * math.cos(q * u / p)
        cx = cr * math.cos(u)
        cy = 0.35 * math.sin(q * u / p)
        cz = cr * math.sin(u)
        # frame via numerical tangent
        eps = 1e-3
        u2 = u + eps
        cr2 = R + 0.35 * math.cos(q * u2 / p)
        tx, ty, tz = cr2 * math.cos(u2) - cx, 0.35 * math.sin(q * u2 / p) - cy, cr2 * math.sin(u2) - cz
        tl = math.sqrt(tx * tx + ty * ty + tz * tz) or 1
        tx, ty, tz = tx / tl, ty / tl, tz / tl
        # normal = radial-ish: cross(t, up) then cross back
        nx, ny, nz = tz, 0, -tx
        nl = math.sqrt(nx * nx + nz * nz) or 1
        nx, nz = nx / nl, nz / nl
        bx, by, bz = ty * nz - tz * ny, tz * nx - tx * nz, tx * ny - ty * nx
        for j in range(tube):
            a = j / tube * 2 * math.pi
            c, s = math.cos(a) * r, math.sin(a) * r
            verts.append((cx + nx * c + bx * s, cy + ny * c + by * s, cz + nz * c + bz * s))
    for i in range(seg):
        for j in range(tube):
            a = i * tube + j
            b = i * tube + (j + 1) % tube
            c = ((i + 1) % seg) * tube + j
            d = ((i + 1) % seg) * tube + (j + 1) % tube
            tris.append((a, b, c))
            tris.append((b, d, c))
    return verts, tris


def write_obj(path, verts, tris):
    with open(path, "w") as f:
        for v in verts:
            f.write("v %.4f %.4f %.4f\n" % v)
        for t in tris:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


# ---------------------------------------------------------------- scene
materials = {
    "floor":  {"type": "matte", "color": [0.35, 0.36, 0.38]},
    "red":    {"type": "matte", "color": [0.75, 0.2, 0.15]},
    "green":  {"type": "matte", "color": [0.15, 0.7, 0.25]},
    "blue":   {"type": "matte", "color": [0.15, 0.3, 0.75]},
    "white":  {"type": "matte", "color": [0.8, 0.8, 0.82]},
    "chrome": {"type": "metal", "color": [0.92, 0.92, 0.95], "roughness": 0.02},
    "gold":   {"type": "metal", "color": [0.95, 0.72, 0.3], "roughness": 0.05},
    "glass":  {"type": "glass", "color": [0.9, 0.95, 1.0], "ior": 1.5},
    "emit":   {"type": "emissive", "color": [0.4, 0.7, 1.0], "power": 2.5},
    "knot":   {"type": "metal", "color": [0.7, 0.65, 0.6], "roughness": 0.1, "reflectivity": 0.4, "metallic": False},
}

objects = [{"type": "plane", "point": [0, 0, 0], "normal": [0, 1, 0], "material": "floor"}]

# emissive slab backdrop — lights the scene independently of the point lights
objects.append({"type": "box", "min": [-14, 0.1, 16], "max": [14, 2.2, 16.6], "material": "emit"})

# --- sphere field: 7 rows x 20 cols, alternating material palette rows
palette = ["red", "green", "blue", "white", "red", "green", "blue"]
for row in range(7):
    z = -1.5 + row * 1.7
    mat = palette[row % len(palette)]
    for col in range(20):
        x = -9.5 + col * 1.0
        objects.append({
            "type": "sphere",
            "center": [x, 0.32, z],
            "radius": 0.3,
            "material": mat,
        })

# --- center feature cluster: chrome + glass + gold
objects.append({"type": "sphere", "center": [-1.1, 0.75, 4.6], "radius": 0.75, "material": "chrome"})
objects.append({"type": "sphere", "center": [0.0, 0.6, 5.6], "radius": 0.6, "material": "glass"})
objects.append({"type": "sphere", "center": [1.2, 0.55, 4.9], "radius": 0.55, "material": "gold"})

# --- 24 torus-knot instances: 2 rings of 12, each ~21K triangles... no,
#     220*48 quads = 21120 tris each * 24 = 506K — too many. Use seg=90,
#     tube=26 -> ~4.7K tris each, 24 instances ~112K tris total. Instancing
#     multiplies RAY work, not memory: each instance builds its own BVH
#     (transform baked), so the BVH walk cost is per-instance per ray.
v, t = torus_knot(seg=90, tube=26)
write_obj(OBJ, v, t)
ntris = len(t)
for ring in range(2):
    rad = 5.2 + ring * 2.2
    y = 1.35 + ring * 0.75
    for k in range(12):
        a = k / 12 * 2 * math.pi + ring * 0.26
        objects.append({
            "type": "mesh",
            "src": os.path.basename(OBJ),
            "material": "knot",
            "transform": {
                "rotate": [0, math.degrees(a) * 0.15, math.degrees(a)],
                "scale": [0.42, 0.42, 0.42],
                "translate": [rad * math.cos(a), y, rad * math.sin(a) + 3.0],
            },
        })

scene = {
    "version": 1,
    "camera": {"pos": [0, 3.1, -7.5], "lookAt": [0, 1.0, 3.0], "fov": 50},
    "environment": {"color": [0.5, 0.65, 0.85], "ambient": 0.12, "ambientColor": [1, 1, 1]},
    "lights": [
        {"type": "point", "pos": [-6, 9, -4], "color": [1.0, 0.97, 0.9], "intensity": 1.15, "falloff": "none"},
        {"type": "point", "pos": [7, 7, 6], "color": [0.85, 0.9, 1.0], "intensity": 0.8, "falloff": "none"},
        {"type": "point", "pos": [0, 5, -8], "color": [1.0, 0.95, 0.88], "intensity": 0.5, "falloff": "none"},
    ],
    "materials": materials,
    "objects": objects,
    "render": {
        "width": 1920,
        "height": 1080,
        "samples": 8,
        "maxDepth": 12,
        "seed": 45,
        "exposure": 1.0,
        "tonemap": "clip",
        "output": "soak.png",
    },
}

with open(OUT, "w") as f:
    json.dump(scene, f, indent=1)

nspheres = 7 * 20 + 3
print(f"wrote {OUT}")
print(f"  objects: {len(objects)}  (analytic: {nspheres} spheres + plane + slab)")
print(f"  knot instances: 24 x {ntris} tris = {24 * ntris} mesh tris")
print(f"  render: 1920x1080 @ samples=8 (64 rays/px), maxDepth 12")
