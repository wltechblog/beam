#!/usr/bin/env python3
"""Generate Beam scene JSON from the POV-Ray benchmark (benchmark.pov v2.03).

Coordinate system: POV is Z-up; Beam is Y-up. Map (x, y, z)_pov -> (x, -z, y)_beam.

Ported elements:
- camera (location/look_at/angle 45)
- sun directional light (150k-unit area light collapsed to directional) + gray fill
- sky sphere pigment_map -> environment sky gradient (zenith/horizon blend)
- sand floor plane (basin floor) + isosurface ridges (approximated, flat)
- water box slab: glass ior 1.31 with green fade tint
- basin: difference of two boxes -> 5-slab union
- 5 sockets: fluted cylinder pedestals -> procedural revolve meshes
- texts: "POV-Ray" and "Version 3.8" extruded, stone material
- glass sphere at Pos3
- POV logo at Pos1: sphere + cone + carved sphere -> sphere/cone/scaled-sphere stand-ins
- Pos2 orbit ring: superellipsoids + julia fractals -> small spheres at 6 positions
- Pos4: cylinder with sine-X displacement -> procedural mesh
- Pos5: lathe tower -> revolved profile mesh (FnA, 64x48)
- heightfield mountain -> 160x160 height mesh with noise-warped cone profile
- clouds (media) skipped: no volumetrics in Beam v1
"""
import json
import math
import random


def p2b(v):
    """POV Z-up -> Beam Y-up: (x, y, z)_pov -> (x, z, -y)_beam.

    POV +z (up) maps to beam +y (up); POV +y maps to beam -z.
    Right-handed -> right-handed (rotation about X by -90 degrees).
    """
    return [v[0], v[2], -v[1]]


def p2b_box(lo, hi):
    """Convert a POV box to Beam coords, re-normalizing min/max per axis
    (the z->-y flip swaps ordering)."""
    a = p2b(lo)
    b = p2b(hi)
    return [min(a[i], b[i]) for i in range(3)], [max(a[i], b[i]) for i in range(3)]


# ------------------------------------------------------------------ materials

MATERIALS = {
    "sand": {
        # lifted ~10% over POV's rgb<1,0.85,0.6> to stand in for POV's
        # radiosity floor bounce (Beam has no GI)
        "type": "matte", "color": [1.1, 0.95, 0.68],
    },
    "stone": {  # Stone_Tex: granite-ish
        "type": "matte", "color": [0.55, 0.52, 0.50],
        "pattern": {"type": "checker", "scale": 0.35,
                     "colors": [[0.55, 0.52, 0.50], [0.42, 0.40, 0.39]]},
    },
    "stone2": {  # Stone_Tex2 for basin/texts
        "type": "matte", "color": [0.50, 0.48, 0.46],
    },
    "water": {  # M_Watx4: reflection 0.1-0.95 fresnel, ior 1.31, green fade
        # POV fades to green with DEPTH (fade_distance 0.8); glass tints per
        # interface instead, so use a much lighter tint to net a similar look
        "type": "glass", "color": [0.65, 0.88, 0.65], "ior": 1.31,
    },
    "metal": {  # Metal_Texture: copper-ish metallic reflection 0.8
        "type": "metal", "color": [0.65, 0.55, 0.5], "roughness": 0.01,
    },
    "glass": {  # Mat_Glass
        "type": "glass", "color": [0.9, 0.95, 0.98], "ior": 1.5,
    },
    "flame": {  # isosurface at Pos1: orange
        "type": "matte", "color": [1.0, 0.45, 0.2],
    },
    "logo": {  # purple metallic logo
        "type": "metal", "color": [0.65, 0.55, 0.9], "roughness": 0.01,
    },
    "pink": {  # julia fractals
        "type": "matte", "color": [1.0, 0.4, 0.8],
    },
    "mountain": {  # heightfield bozo pigment
        "type": "matte", "color": [0.36, 0.36, 0.42],
        "pattern": {"type": "checker", "scale": 0.2,
                     "colors": [[0.36, 0.36, 0.42], [0.54, 0.36, 0.18]]},
    },
    "fnobj": {  # Pos4 cylinder
        "type": "matte", "color": [0.85, 0.80, 0.75],
    },
}


# ------------------------------------------------------------------ socket pedestal mesh

def socket_mesh(grooves=24, rings=10, height=5.0, top_z=-0.04, bot_z=-5.0,
                r_outer=0.8, r_groove=0.72, r_hole=0.3):
    """Fluted cylinder pedestal: outer wall with `grooves` vertical scallops,
    hollow center (r_hole), top disc at z=-0.04 plus base ring.

    POV Socket: cylinder(-5z, -0.04z, 0.8) minus cylinder(-6z, 0, 0.3) minus
    24 rotated groove boxes, plus 12 rods on rim and a top disc cylinder
    (-0.04z, 0, 0.86). We model: outer scalloped wall, inner hole wall, top
    annulus, and bottom annulus, in POV z-up space. Returned mesh is in POV
    coordinates (convert at placement).
    """
    verts = []
    tris = []

    def ring(z, radius_fn, n):
        base = len(verts)
        for i in range(n):
            ang = 2 * math.pi * i / n
            r = radius_fn(ang)
            verts.append((r * math.cos(ang), r * math.sin(ang), z))
        return base

    def outer_r(ang):
        # scallop: 24 grooves; groove floor near r_groove, flat r_outer
        wave = math.cos(ang * grooves) * 0.5 + 0.5  # 0..1
        return r_outer - (r_outer - r_groove) * (1.0 - wave)

    n = 96
    # outer wall rings from bottom to top
    ring_ids = []
    for k in range(rings + 1):
        z = bot_z + (top_z - bot_z) * k / rings
        ring_ids.append(ring(z, outer_r, n))
    # connect outer wall (quads -> 2 tris), CCW seen from outside
    for k in range(rings):
        a0 = ring_ids[k]
        a1 = ring_ids[k + 1]
        for i in range(n):
            j = (i + 1) % n
            tris.append([a0 + i, a0 + j, a1 + j])
            tris.append([a0 + i, a1 + j, a1 + i])
    # inner hole wall (radius r_hole), normals flipped
    ring_ids = []
    for k in range(rings + 1):
        z = bot_z + (top_z - bot_z) * k / rings
        ring_ids.append(ring(z, lambda a: r_hole, n))
    for k in range(rings):
        a0 = ring_ids[k]
        a1 = ring_ids[k + 1]
        for i in range(n):
            j = (i + 1) % n
            tris.append([a0 + i, a1 + j, a0 + j])
            tris.append([a0 + i, a1 + i, a1 + j])
    # top annulus (z=top_z) between r_hole and scalloped outer; disc extends to 0.86
    t_inner = ring(top_z, lambda a: r_hole, n)
    t_outer = ring(top_z, outer_r, n)
    for i in range(n):
        j = (i + 1) % n
        tris.append([t_inner + i, t_outer + j, t_inner + j])
        tris.append([t_inner + i, t_outer + i, t_outer + j])
    # bottom annulus (z=bot_z)
    b_inner = ring(bot_z, lambda a: r_hole, n)
    b_outer = ring(bot_z, outer_r, n)
    for i in range(n):
        j = (i + 1) % n
        tris.append([b_inner + i, b_inner + j, b_outer + j])
        tris.append([b_inner + i, b_outer + j, b_outer + i])
    return verts, tris


# ------------------------------------------------------------------ lathe tower (Pos5)

def fnA(X):
    if X < 0.13:
        return 0.5 + math.sin(X * 14) * 0.4
    return 0.5 + math.sin((X - 0.13) * 7.2) * 0.3


def lathe_tower(res_rot=72, res_spl=56):
    """Revolve the FnA profile around POV's Z axis (tower height axis)."""
    verts = []
    tris = []
    for i in range(res_rot + 1):
        u = i / res_rot
        ang = 2 * math.pi * u
        for j in range(res_spl + 1):
            v = j / res_spl  # 0..1 along profile
            r = fnA(v)
            verts.append((r * math.cos(ang), r * math.sin(ang), v))
    w = res_spl + 1
    for i in range(res_rot):
        for j in range(res_spl):
            a = i * w + j
            b = a + 1
            c = (i + 1) * w + j
            d = c + 1
            tris.append([a, b, d])
            tris.append([a, d, c])
    return verts, tris


# ------------------------------------------------------------------ Pos4 displaced cylinder

def fn4_mesh(res_u=48, res_v=24):
    """Cylinder r=2 (POV units, before scale 0.12) along POV Y axis from -2..2,
    radius displaced by the sine-X pattern in the SDL.

    The SDL: cylinder(-2y, 2y, 2) patterned with XsinPigm via cylindrical warp,
    then the difference cuts the inside. We approximate the *look*: a cylinder
    whose radius bulges following sin(u)^2 * sin(theta) waves.
    """
    verts = []
    tris = []
    for i in range(res_v + 1):
        v = i / res_v
        y = -2 + 4 * v
        # pattern value along axis
        s = math.sin(y * math.pi / 2)
        for k in range(res_u):
            u = k / res_u
            theta = 2 * math.pi * u
            # XsinPigm: xPigm(sin(x), sin(y)^2) with repeat warps -> periodic
            # lobes: radius modulated by |sin(3 theta) * s|
            bulge = abs(math.sin(theta * 3) * s)
            r = 1.55 + 0.45 * bulge
            verts.append((r * math.cos(theta), y, r * math.sin(theta)))
    w = res_u
    for i in range(res_v):
        for k in range(res_u):
            a = i * w + k
            b = i * w + (k + 1) % w
            c = (i + 1) * w + k
            d = (i + 1) * w + (k + 1) % w
            tris.append([a, b, d])
            tris.append([a, d, c])
    # end caps
    def cap(ysign, yy):
        base = len(verts)
        rmax = 1.55 + 0.45 * abs(math.sin(0) * math.sin(yy * math.pi / 2))
        verts.append((0, yy, 0))  # center
        for k in range(res_u):
            u = k / res_u
            theta = 2 * math.pi * u
            s = math.sin(yy * math.pi / 2)
            bulge = abs(math.sin(theta * 3) * s)
            r = 1.55 + 0.45 * bulge
            verts.append((r * math.cos(theta), yy, r * math.sin(theta)))
        for k in range(res_u):
            j = (k + 1) % w
            if ysign > 0:
                tris.append([base, base + 1 + j, base + 1 + k])
            else:
                tris.append([base, base + 1 + k, base + 1 + j])
    cap(+1, 2)
    cap(-1, -2)
    return verts, tris


# ------------------------------------------------------------------ heightfield mountain

def make_noise(seed):
    rng = random.Random(seed)
    g = [[rng.random() for _ in range(8)] for _ in range(8)]
    return g


def noise2(g, x, y):
    """Bilinear value noise, tiled 8x8 grid over [0,1)."""
    n = len(g)
    fx = x * n
    fy = y * n
    x0 = int(fx) % n
    y0 = int(fy) % n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    tx = fx - math.floor(fx)
    ty = fy - math.floor(fy)
    sx = tx * tx * (3 - 2 * tx)
    sy = ty * ty * (3 - 2 * ty)
    a = g[y0][x0] * (1 - sx) + g[y0][x1] * sx
    b = g[y1][x0] * (1 - sx) + g[y1][x1] * sx
    return a * (1 - sy) + b * sy


def mountain_mesh(n=160):
    """height_field: base cone 1-(sqrt(sqrt(x^2+y^2))-0.0001) in [0,1]^2 domain,
    water_level 0.02, warped by turbulence (approximated with 2-octave noise),
    then scaled <5,5,1.3>*2.6, rotated 90x then 43z, translated <-18,-13,-0.4>.

    We build in the heightfield's local X-Y plane (POV z-up), apply the
    transforms with a matrix helper at the end.
    """
    g = make_noise(45)
    verts = []
    tris = []
    water = 0.02
    for j in range(n):
        for i in range(n):
            x = i / (n - 1) - 0.5
            y = j / (n - 1) - 0.5
            d = math.sqrt(x * x + y * y)
            base = 1.0 - (min(math.sqrt(d), 1.0) - 0.0001)
            # turbulence warp approx: sample noise at displaced coords, 2 octaves
            t = noise2(g, (x + 0.5) * 3, (y + 0.5) * 3)
            t2 = noise2(g, (x + 0.5) * 6, (y + 0.5) * 6)
            h = base * (1 + 0.25 * (t - 0.5) + 0.125 * (t2 - 0.5))
            h = max(h, water)
            verts.append((x, y, h))
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            tris.append([a, c, d])
            tris.append([a, d, b])
    return verts, tris


def apply_pov_transforms(verts, ops):
    """Apply a list of POV transform ops to vertex list. Ops: ('scale', (sx,sy,sz)),
    ('rotz', deg), ('rotx', deg), ('translate', (tx,ty,tz))."""
    out = []
    for v in verts:
        x, y, z = v
        for op in ops:
            if op[0] == "scale":
                x, y, z = x * op[1][0], y * op[1][1], z * op[1][2]
            elif op[0] == "rotz":
                a = math.radians(op[1])
                x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
            elif op[0] == "rotx":
                a = math.radians(op[1])
                y, z = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)
            elif op[0] == "translate":
                x += op[1][0]; y += op[1][1]; z += op[1][2]
        out.append((x, y, z))
    return out


# ------------------------------------------------------------------ assembly

POV_POSITIONS = {
    "Pos1": (0.0, 0.0, 0.6),
    "Pos2": (-2.4, -0.8, 1.0),
    "Pos3": (-1.4, -2.8, 1.3),
    "Pos4": (1.4, 0.6, 0.5),
    "Pos5": (-10.0, -5.0, 1.5),
}


def build_scene():
    objects = []

    # ---- floor (sand plane at z=-0.3, clipped to basin exterior in POV;
    #      we keep the full plane as the surrounding sand)
    objects.append({
        "type": "plane",
        "point": p2b((0, 0, -0.3)),
        "normal": p2b((0, 0, 1)),
        "material": "sand",
    })

    # ---- basin: difference box(+-2.0,1.7 z .5..-6) - box(+-1.9,1.6 z 1..-5)
    #      -> 5 slabs: floor + 4 walls (stone2)
    # POV coords; y-up conversion at the end. Wall thickness 0.1, rim at z=0.5.
    basin_parts = [
        # (min corner, max corner) in POV space
        ((-2.0, -1.7, -6.0), (2.0, 1.7, -5.9)),    # bottom slab
        ((-2.0, -1.7, -6.0), (-1.9, 1.7, 0.5)),    # -x wall
        ((1.9, -1.7, -6.0), (2.0, 1.7, 0.5)),      # +x wall
        ((-1.9, -1.7, -6.0), (1.9, -1.6, 0.5)),    # -y wall
        ((-1.9, 1.6, -6.0), (1.9, 1.7, 0.5)),      # +y wall
    ]
    for lo, hi in basin_parts:
        bmin, bmax = p2b_box(lo, hi)
        objects.append({
            "type": "box",
            "min": bmin,
            "max": bmax,
            "material": "stone2",
        })

    # ---- water: box slab z from 0.42 down to -5.5 (POV), inner basin size
    wmin, wmax = p2b_box((-1.95, -1.65, -5.5), (1.95, 1.65, 0.42))
    objects.append({
        "type": "box",
        "min": wmin,
        "max": wmax,
        "material": "water",
    })

    # ---- sockets at Pos1..Pos5 (scale 1,1,0.8,0.8,0.4,2 -> per-object)
    sock_verts, sock_tris = socket_mesh()
    socket_scales = {"Pos1": 1.0, "Pos2": 1.0, "Pos3": 0.8, "Pos4": 0.4, "Pos5": 2.0}
    for name, s in socket_scales.items():
        verts = apply_pov_transforms(
            sock_verts,
            [("scale", (s, s, s)), ("translate", POV_POSITIONS[name])],
        )
        fname = f"obj-socket-{name}.obj"
        write_obj(fname, [p2b(v) for v in verts], sock_tris)
        objects.append({
            "type": "mesh",
            "src": fname,
            "material": "stone",
        })

    # ---- texts on basin wall (scale 0.3, rot 90x, rot -90z)
    objects.append({
        "type": "text",
        "text": "POV-Ray",
        "size": 0.3 * 0.7,
        "depth": 0.075,
        "material": "stone2",
        "transform": {
            "translate": p2b((-1.75, 0.995, 0.575)),
        },
    })
    objects.append({
        "type": "text",
        "text": "Version 3.8",
        "size": 0.3 * 0.7,
        "depth": 0.075,
        "material": "stone2",
        "transform": {
            "translate": p2b((-0.5, -1.5, 0.575)),
            "rotate": [0, 90, 0],
        },
    })

    # ---- glass sphere at Pos3
    objects.append({
        "type": "sphere",
        "center": p2b(tuple(a + b for a, b in zip(POV_POSITIONS["Pos3"], (0, 0, 0.24)))),
        "radius": 0.24,
        "material": "glass",
    })

    # ---- Pos1: metal ring (cylinder r=0.7 h=0.15 with hole) + orange isosurface
    #      approximated as torus-like: use flattened sphere + orange blob sphere
    objects.append({
        "type": "sphere",
        "center": p2b((POV_POSITIONS["Pos1"][0], POV_POSITIONS["Pos1"][1],
                        POV_POSITIONS["Pos1"][2] + 0.075)),
        "radius": 0.7,
        "transform": {"scale": [1.0, 0.214, 1.0]},
        "material": "metal",
    })
    objects.append({
        "type": "sphere",
        "center": p2b((POV_POSITIONS["Pos1"][0], POV_POSITIONS["Pos1"][1],
                        POV_POSITIONS["Pos1"][2] + 0.35)),
        "radius": 0.35,
        "transform": {"scale": [1.0, 0.8, 1.0]},
        "material": "flame",
    })

    # ---- Pos1: POV logo (purple metallic) - sphere on cone silhouette approx
    logo_scale = 0.4
    lx, ly, lz = POV_POSITIONS["Pos1"]
    objects.append({
        "type": "sphere",
        "center": p2b((lx - 0.2, ly - 0.2, lz + 0.96 + 0.5 * logo_scale)),
        "radius": 1.0 * logo_scale,
        "material": "logo",
    })
    objects.append({
        "type": "sphere",
        "center": p2b((lx - 0.2 + 0.6 * logo_scale, ly - 0.2, lz + 0.96)),
        "radius": 1.2 * logo_scale,
        "transform": {"scale": [2.0, 1.0, 0.5], "rotate": [0, 30, 0]},
        "material": "logo",
    })

    # ---- Pos2: orbit ring of 12 small spheres (6 superellipsoids + 6 julias)
    ox, oy, oz = POV_POSITIONS["Pos2"]
    for cnt in range(0, 360, 60):
        a = math.radians(cnt + 30)
        r = 0.6
        sx = ox + r * math.cos(a)
        sy = oy + r * math.sin(a)
        # superellipsoid (metal) and julia (pink) alternate at slight offsets
        objects.append({
            "type": "sphere",
            "center": p2b((sx, sy, oz + 0.12)),
            "radius": 0.12,
            "material": "metal",
        })
        objects.append({
            "type": "sphere",
            "center": p2b((ox + r * math.cos(math.radians(cnt)), oy + r * math.sin(math.radians(cnt)), oz + 0.28)),
            "radius": 0.10,
            "material": "pink",
        })

    # ---- Pos4: sine-displaced cylinder (scale 0.12)
    fn_verts, fn_tris = fn4_mesh()
    fn_verts = apply_pov_transforms(
        fn_verts,
        [
            ("translate", (0, 1.96, 0)),
            ("rotx", 90),
            ("scale", (0.12, 0.12, 0.12)),
            ("translate", POV_POSITIONS["Pos4"]),
        ],
    )
    fname = "obj-fn4.obj"
    write_obj(fname, [p2b(v) for v in fn_verts], fn_tris)
    objects.append({
        "type": "mesh",
        "src": fname,
        "material": "fnobj",
    })

    # ---- Pos5: lathe tower (FnLathe 360,100,100)
    tw_verts, tw_tris = lathe_tower()
    # my tower stands along POV +z already; SDL has rotate 90*x making it vertical
    # the same way. Scale <1.9,1.9,2.8>, translate Pos5.
    tw_verts = apply_pov_transforms(
        tw_verts,
        [("scale", (1.9, 1.9, 2.8)), ("translate", POV_POSITIONS["Pos5"])],
    )
    fname = "obj-tower.obj"
    write_obj(fname, [p2b(v) for v in tw_verts], tw_tris)
    objects.append({
        "type": "mesh",
        "src": fname,
        "material": "stone",
    })

    # ---- heightfield mountain (rotate 90x, rotate 43z, scale <5,5,1.3>*2.6, translate)
    m_verts, m_tris = mountain_mesh(160)
    m_verts = apply_pov_transforms(
        m_verts,
        [
            ("rotz", 43),
            ("scale", (5 * 2.6, 5 * 2.6, 1.3 * 2.6)),
            ("translate", (-18, -13, -0.4)),
        ],
    )
    fname = "obj-mountain.obj"
    write_obj(fname, [p2b(v) for v in m_verts], m_tris)
    objects.append({
        "type": "mesh",
        "src": fname,
        "material": "mountain",
    })

    # ---- sun + fill + sky
    sun_dir_pov = (-0.7, 0.83, 0.24)
    scene = {
        "version": 1,
        "camera": {
            "pos": p2b((3.2, 3.2, 1.8)),
            "lookAt": p2b((-1, -1, 0.9)),
            "fov": 45,
        },
        "environment": {
            "color": [0.62, 0.66, 0.78],
            "ambient": 0.75,
            # 3-stop sky: [zenith, horizon, below-horizon]. Stops solved by
            # inverting POV reference pixels through exposure 2.6 + Reinhard
            # (POV's display gamma brightens its declared values considerably:
            # grey 0.667 displays ~0.83, so our raw stops must be much higher).
            "sky": [
                [0.0, 0.05, 2.9],     # zenith (deep blue)
                [2.8, 2.95, 2.85],    # horizon (POV near-horizon white ~225)
                [2.87, 2.87, 2.83],   # below horizon
            ],
            # POV's sky stays blue until close to the horizon (its gradient
            # is non-linear); curve<1 holds zenith color low before snapping
            # to white. Solved against reference rows +4.3°..0°.
            "skyCurve": 0.5,
            # POV benchmark: fog_type 2, alt 1.5, offset 0, rgbt <0.75,0.80,0.86,0.2>
            # (t=0.2 → max 80% fog), distance 400. POV z-up 1.5 == Beam y 1.5.
            # Color lifted into radiance space to match POV's gamma-brightened
            # display (~225): raw 0.75 would display only 169 in our pipeline.
            "fog": {
                "color": [2.45, 3.05, 4.5],
                "distance": 12,
                "altitude": 1.5,
                "offset": 0,
                "maxFog": 0.8,
            },
        },
        "lights": [
            {
                "type": "directional",
                # POV light at sun_pos*150000 shines toward origin: travel dir = -pos
                "dir": [-sun_dir_pov[0], -sun_dir_pov[2], sun_dir_pov[1]],
                # Empirically calibrated (raw POV 3.43/2.87/1.95 is too dark for
                # Beam: POV's radiosity + media-attenuated sun ≈ our brighter direct)
                "color": [6.0, 5.0, 3.4],
            },
            {
                "type": "directional",
                "dir": [-0.9, -0.5, -0.6],
                "color": [0.6, 0.6, 0.6],
            },
        ],
        "materials": MATERIALS,
        "objects": objects,
        "render": {
            "width": 1920,
            "height": 1080,
            "samples": 4,
            "maxDepth": 12,
            "seed": 45,
            "exposure": 2.6,
            "output": "beam-benchmark.png",
        },
    }
    return scene


def write_obj(path, verts, tris):
    """Write a POV-space mesh to an OBJ file already converted to Beam coords."""
    with open(path, "w") as f:
        f.write("# beam pov-benchmark generated\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def main():
    scene = build_scene()
    with open("beam-benchmark.json", "w") as f:
        json.dump(scene, f, indent=1)
    nmesh = sum(1 for o in scene["objects"] if o["type"] == "mesh")
    print(f"wrote beam-benchmark.json: {len(scene['objects'])} objects ({nmesh} meshes)")


if __name__ == "__main__":
    main()
