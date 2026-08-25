#!/usr/bin/env python3
"""Generate Beam scene JSON from the POV-Ray benchmark (benchmark.pov v2.03).

Coordinate system: POV is Z-up LEFT-handed; Beam is Y-up right-handed. Map (x, y, z)_pov -> (-x, z, -y)_beam (improper, mirror).

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
    """POV (left-handed, Z-up) -> Beam (right-handed, Y-up).

    Map (x, y, z)_pov -> (-x, z, -y)_beam. This is an IMPROPER transform
    (det = -1): POV-Ray uses a left-handed coordinate system, so converting
    to a right-handed renderer requires one reflection to preserve the
    rendered image orientation. A proper rotation like (x, z, -y) mirrors
    the whole scene on screen. det = -1 also inverts triangle winding,
    fixed by swap_winding() at every write_obj call.
    """
    return [-v[0], v[2], -v[1]]


def swap_winding(tris):
    """Invert triangle winding to match the mirrored (improper) p2b map."""
    return [(a, c, b) for (a, b, c) in tris]


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
        "type": "matte", "color": [0.52, 0.44, 0.38],
    },
    "water": {  # M_Watx4: reflection 0.1-0.95 fresnel, ior 1.31, green fade
        # POV fades to green with DEPTH (fade_distance 0.8); glass tints per
        # interface instead, so use a much lighter tint to net a similar look
        "type": "glass", "color": [0.65, 0.88, 0.65], "ior": 1.31, "reflectivity": 0.1,
    },
    "metal": {  # Metal_Texture: metallic pigment <0.65,0.55,0.5>,
        # diffuse 0.15, specular 0.3, roughness 0.01, reflection 0.8 metallic
        "type": "metal", "color": [0.65, 0.55, 0.5], "roughness": 0.01,
        "diffuse": 0.15, "specular": 0.3, "reflectivity": 0.8, "metallic": True,
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
    "pink": {  # julia fractals: POV pigment <1,0.4,0.8>, diffuse 0.6,
        # specular 0.2, reflection 0.2 (untinted -> chrome-ish sparkle)
        "type": "matte", "color": [1.0, 0.4, 0.8],
        "diffuse": 0.6, "specular": 0.2,
        "reflectivity": 0.2, "metallic": False,
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



def socket_mesh(nseg=144, ncap=10):
    """Real POV Socket (benchmark.pov 646): cylinder(-5z..-0.04z, r=0.8)
    minus center hole cylinder(r=0.3) minus 12 rotated cutters (horizontal
    cylinder r=0.1 at z=-0.24 + box below it) => 12 through-wall SLOTS with
    rounded tops starting at z=-0.24. Union 12 rim rods (r=0.05 running
    z=-5..-0.04 at r=0.8, offset 15deg, capped by r=0.07 stub z=-0.04..-0.01),
    top disc cylinder(r=0.86, z=-0.04..0), torus collar (0.83, 0.03) at
    z=-0.04. POV z-up space; convert with p2b as usual.

    Explicit-surface construction (beam has no CSG at render time):
      - outer wall, segmented: solid above slot tops; below slot top only
        material in the angular gaps between cutters (slot windows open)
      - slot ceilings: half-cylinder r=0.1 arcs closing each slot top
      - inner hole wall, top annulus (r 0.3..0.8) + disc edge, bottom
        annulus, rim rods with caps, top disc top, torus collar.
    """
    verts = []
    tris = []

    def quad(a, b, c, d):
        tris.append([a, b, c]); tris.append([a, c, d])

    R_OUT, R_HOLE, R_DISC = 0.8, 0.3, 0.86
    Z_TOP, Z_BOT, Z_CUT = -0.04, -5.0, -0.24
    SLOT_R = 0.1
    N_CUT = 12

    def cutter_k(x, y):
        """Return cutter index whose footprint contains (x,y), else -1.
        Cutter k: rotate point by -30k deg; footprint = dx in [0,1],
        |dy| <= 0.1 (box) union circle centered (0.9? no: center on axis)
        -> circle center (1.0, 0)? The horizontal cylinder runs 0->1 along
        x at z=-0.24 with r=0.1: its top at z=-0.14. Footprint in xy is the
        BOX |dy|<=0.1, 0<=dx<=1 (cylinder within box in projection)."""
        for k in range(N_CUT):
            ca = math.radians(30.0 * k)
            dx = x * math.cos(ca) + y * math.sin(ca)
            dy = -x * math.sin(ca) + y * math.cos(ca)
            if 0.0 <= dx <= 1.0 and abs(dy) <= SLOT_R:
                return k
        return -1

    # ---- outer wall --------------------------------------------------
    # ring heights: top edge, slot top, slot bottom(cut), wall bottom
    n = nseg
    def outer_ring(z):
        base = len(verts)
        for i in range(n):
            ang = 2 * math.pi * i / n
            verts.append((R_OUT * math.cos(ang), R_OUT * math.sin(ang), z))
        return base

    r_top = outer_ring(Z_TOP)
    r_cap = outer_ring(Z_CUT + SLOT_R)   # slot ceiling crest
    r_cut = outer_ring(Z_CUT)
    r_bot = outer_ring(Z_BOT)

    def in_slot(ang):
        return cutter_k(math.cos(ang), math.sin(ang)) >= 0

    # wall bands:
    for i in range(n):
        j = (i + 1) % n
        ang_i = 2 * math.pi * i / n
        s = in_slot(ang_i)
        # top band [Z_TOP .. Z_CUT+SLOT_R]: always solid
        quad(r_top + i, r_top + j, r_cap + j, r_cap + i)
        # slot band [Z_CUT+SLOT_R .. Z_CUT]: only in gaps between slots
        if not s:
            quad(r_cap + i, r_cap + j, r_cut + j, r_cut + i)
        # below band [Z_CUT .. Z_BOT]: only where NOT inside a slot
        if not s:
            quad(r_cut + i, r_cut + j, r_bot + j, r_bot + i)

    # ---- slot ceilings (rounded tops) ---------------------------------
    # half-cylinder arc from wall (r=0.8) inward, at each cutter angle,
    # spanning the slot angular width; arc from z=Z_CUT+SLOT_R down to
    # Z_CUT as radius goes 0.8 -> 0.8-0.1
    for k in range(N_CUT):
        ca = math.radians(30.0 * k)
        half_w = math.asin(SLOT_R / R_OUT)   # angular half-width of slot
        steps = max(4, int(half_w / (2 * math.pi) * n))
        # left and right edge of slot (angular), walk ceiling arc
        aL = ca - half_w; aR = ca + half_w
        # rings of ceiling: outer at r=0.8 (already on wall), inner r=0.7
        # ceiling surface: quad strip between (r=0.8, z=Z_CUT+SLOT_R) and
        # (r=0.7, z=Z_CUT) following the cylinder silhouette
        ncs = ncap
        for s_i in range(ncs):
            t0 = s_i / ncs
            t1 = (s_i + 1) / ncs
            # param t: 0 at left edge z=Z_CUT+SLOT_R crest, 1 at right;
            # ceiling strip: from crest line down/right along arc
            # build as fan of quads across slot width for this arc step
            for w in range(steps):
                w0 = w / steps; w1 = (w + 1) / steps
                # two arc positions
                a0 = aL + (aR - aL) * w0
                a1 = aL + (aR - aL) * w1
                # arc height at parameter across width (cylinder cap:
                # z = Z_CUT + sqrt(SLOT_R^2 - (r*sin(theta))^2) approx)
                def ceil_z(a):
                    dy = abs(R_OUT * math.sin(a - ca))
                    if dy >= SLOT_R:
                        return Z_CUT + 1e-6
                    return Z_CUT + math.sqrt(SLOT_R * SLOT_R - dy * dy)
                # radial strip between wall radii at two angles
                p00 = (R_OUT * math.cos(a0), R_OUT * math.sin(a0), ceil_z(a0))
                p01 = (R_OUT * math.cos(a1), R_OUT * math.sin(a1), ceil_z(a1))
                # strip down to slot bottom plane
                p10 = (R_OUT * math.cos(a0), R_OUT * math.sin(a0), Z_COT if False else Z_CUT)
                p11 = (R_OUT * math.cos(a1), R_OUT * math.sin(a1), Z_CUT)
                i00 = len(verts); verts.append(p00)
                i01 = len(verts); verts.append(p01)
                i10 = len(verts); verts.append(p10)
                i11 = len(verts); verts.append(p11)
                quad(i00, i01, i11, i10)
        # (this creates the visible rounded lip; radial depth omitted - the
        #  slot reads as a recessed window with rounded top from outside)

    # ---- inner hole wall ----------------------------------------------
    def inner_ring(z):
        base = len(verts)
        for i in range(n):
            ang = 2 * math.pi * i / n
            verts.append((R_HOLE * math.cos(ang), R_HOLE * math.sin(ang), z))
        return base

    ir_top = inner_ring(Z_TOP)
    ir_bot = inner_ring(Z_BOT)
    for i in range(n):
        j = (i + 1) % n
        # winding: normal points toward hole axis? hole surface normal
        # points INTO the hole (toward -r); from outside solid => we want
        # normals facing the hole interior; verified by bbox later
        quad(ir_bot + i, ir_bot + j, ir_top + j, ir_top + i)

    # ---- top annulus r 0.3..0.8 at Z_TOP -------------------------------
    for i in range(n):
        j = (i + 1) % n
        # outer ring at Z_TOP is r_top; inner is ir_top
        quad(r_top + i, r_top + j, ir_top + j, ir_top + i)

    # ---- top disc edge (r 0.8..0.86) and top face ----------------------
    de_top = outer_ring.__doc__  # placeholder to keep locals tidy

    # disc side wall r=0.86 from Z_TOP to 0
    base = len(verts)
    for i in range(n):
        ang = 2 * math.pi * i / n
        verts.append((R_DISC * math.cos(ang), R_DISC * math.sin(ang), Z_TOP))
    d_bot = base
    base = len(verts)
    for i in range(n):
        ang = 2 * math.pi * i / n
        verts.append((R_DISC * math.cos(ang), R_DISC * sin(ang) if False else R_DISC * math.sin(ang), 0.0))
    d_top = base
    for i in range(n):
        j = (i + 1) % n
        quad(d_bot + i, d_bot + j, d_top + j, d_top + i)
    # top face disc (fan)
    c = len(verts); verts.append((0.0, 0.0, 0.0))
    for i in range(n):
        j = (i + 1) % n
        tris.append([c, d_top + i, d_top + j])
    # annulus between R_OUT and R_DISC at Z_TOP (visible collar shoulder)
    for i in range(n):
        j = (i + 1) % n
        quad(r_top + j, r_top + i, d_bot + i, d_bot + j)

    # ---- bottom annulus at Z_BOT --------------------------------------
    for i in range(n):
        j = (i + 1) % n
        quad(r_bot + j, r_bot + i, ir_bot + i, ir_bot + j)

    # ---- rim rods: 12, offset 15deg ------------------------------------
    for k in range(N_CUT):
        ca = math.radians(30.0 * k + 15.0)
        cx, cy = R_OUT * math.cos(ca), R_OUT * math.sin(ca)
        # rod r=0.05, z -5..-0.04; cap stub r=0.07, z -0.04..-0.01
        nr = 10
        for side in (-1, 1):
            pass
        def tube(cx, cy, r, z0, z1, nr, cap_top=False):
            ringA = len(verts)
            for i in range(nr):
                ang = 2 * math.pi * i / nr
                verts.append((cx + r * math.cos(ang), cy + r * math.sin(ang), z0))
            ringB = len(verts)
            for i in range(nr):
                ang = 2 * math.pi * i / nr
                verts.append((cx + r * math.cos(ang), cy + r * math.sin(ang), z1))
            for i in range(nr):
                j = (i + 1) % nr
                quad(ringA + i, ringA + j, ringB + j, ringB + i)
            if cap_top:
                c0 = len(verts); verts.append((cx, cy, z1))
                for i in range(nr):
                    j = (i + 1) % nr
                    tris.append([c0, ringB + i, ringB + j])
        tube(cx, cy, 0.05, Z_BOT, Z_TOP, 10)
        tube(cx, cy, 0.07, Z_TOP, -0.01, 10)
        # cap top of stub (visible top at z=-0.01)
        # re-emit ring at z=-0.01 and fan it
        ringB = len(verts)
        for i in range(10):
            ang = 2 * math.pi * i / 10
            verts.append((cx + 0.07 * math.cos(ang), cy + 0.07 * math.sin(ang), -0.01))
        c0 = len(verts); verts.append((cx, cy, -0.01))
        for i in range(10):
            j = (i + 1) % 10
            tris.append([c0, ringB + j, ringB + i])

    # ---- torus collar (0.83, 0.03) at z=-0.04 ---------------------------
    NT, NS = 96, 8
    R_T, R_S = 0.83, 0.03
    z0 = -0.04
    prev = None
    for t_i in range(NT + 1):
        ta = 2 * math.pi * t_i / NT
        ring = []
        for s_i in range(NS):
            sa = 2 * math.pi * s_i / NS
            # torus in xy-plane (after rotate 90x, POV torus default axis z)
            x = (R_T + R_S * math.cos(sa)) * math.cos(ta)
            y = (R_T + R_S * math.cos(sa)) * math.sin(ta)
            z = z0 + R_S * math.sin(sa)
            ring.append(len(verts))
            verts.append((x, y, z))
        if prev is not None:
            for s_i in range(NS):
                s_j = (s_i + 1) % NS
                quad(prev[s_i], prev[s_j], ring[s_j], ring[s_i])
        prev = ring

    return verts, tris



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


def make_noise3(seed, n=8):
    rng = random.Random(seed)
    return [[[rng.random() for _ in range(n)] for _ in range(n)] for _ in range(n)]


def noise3(g, x, y, z):
    """Trilinear value noise over tiled n^3 grid, domain [0,1)^3."""
    n = len(g)
    fx, fy, fz = x * n, y * n, z * n
    x0, y0, z0 = int(fx) % n, int(fy) % n, int(fz) % n
    x1, y1, z1 = (x0 + 1) % n, (y0 + 1) % n, (z0 + 1) % n
    tx, ty, tz = fx - math.floor(fx), fy - math.floor(fy), fz - math.floor(fz)
    sx = tx * tx * (3 - 2 * tx)
    sy = ty * ty * (3 - 2 * ty)
    sz = tz * tz * (3 - 2 * tz)
    def c(i, j, k):
        return g[k][j][i]
    c00 = c(x0, y0, z0) * (1 - sx) + c(x1, y0, z0) * sx
    c10 = c(x0, y1, z0) * (1 - sx) + c(x1, y1, z0) * sx
    c01 = c(x0, y0, z1) * (1 - sx) + c(x1, y0, z1) * sx
    c11 = c(x0, y1, z1) * (1 - sx) + c(x1, y1, z1) * sx
    c0 = c00 * (1 - sy) + c10 * sy
    c1 = c01 * (1 - sy) + c11 * sy
    return c0 * (1 - sz) + c1 * sz


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


# ------------------------------------------------------------------ Pos2 real shapes

def pov_rand_seq(seed=45):
    """Exact POV-Ray LCG: next = next*1812433253 + 12345 (mod 2^32).
    Verified against povray-src/source/core/math/randomsequence.cpp
    POV_rand(): seed(45) -> next_rand=45, rand() advances then divides."""
    n = seed
    def rand():
        nonlocal n
        n = (n * 1812433253 + 12345) & 0xFFFFFFFF
        return n / 0xFFFFFFFF
    return rand


def julia_field(x, y, z, c, max_iter=7, exit_value=4.0):
    """Quaternion julia field for surface extraction. POV semantics
    (quaternion.cpp JuliaFractalRules::Iterate): w=0 slice, z^2+c,
    escape when d+x2 > 4.0. Field: -1 inside, else +i/max_iter
    (iteration count to escape as smooth-outside proxy)."""
    w = 0.0
    for i in range(1, max_iter + 1):
        d = y * y + z * z + w * w
        x2 = x * x
        if d + x2 > exit_value:
            return i / max_iter
        xn = x2 - d + c[0]
        yn = 2.0 * x * y + c[1]
        zn = 2.0 * x * z + c[2]
        wn = 2.0 * x * w + c[3]
        x, y, z, w = xn, yn, zn, wn
    return -1.0


def superellipsoid_mesh(e, n, nlat=32, nlon=64):
    """Parametric superellipsoid matching POV's <e,n> implicit
    (|x|^(2/e)+|y|^(2/e))^(e/n)+|z|^(2/n)=1, unit extents (POV declares
    superellipsoid { <e,n> } as <-1,-1,-1>..<1,1,1>).

    Param: u = polar [0,pi], v = azimuth [0,2pi]:
      z = sign(cos u)|cos u|^n
      f(u) = |sin u|^n            (so (sum)^(e/n) = sin^2 u, |z|^(2/n)=cos^2 u)
      x = f * sign(cos v)|cos v|^e,  y = f * sign(sin v)|sin v|^e
    Returns (verts, tris) CCW seen from outside (u-major grid)."""
    verts = []
    for i in range(nlat + 1):
        u = math.pi * i / nlat
        fu = (abs(math.sin(u))) ** n
        z = _sgn(math.cos(u)) * (abs(math.cos(u))) ** n
        for j in range(nlon + 1):
            v = 2 * math.pi * j / nlon
            x = fu * _sgn(math.cos(v)) * (abs(math.cos(v))) ** e
            y = fu * _sgn(math.sin(v)) * (abs(math.sin(v))) ** e
            verts.append((x, y, z))
    tris = []
    def idx(i, j):
        return i * (nlon + 1) + j
    for i in range(nlat):
        for j in range(nlon):
            a, b, c_, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            # CCW seen from outside: (a, d, c_) / (a, c_, b) — the natural
            # (i,j)->(i,j+1)->(i+1,j+1) order faces INWARD for u=polar/v=azimuth
            tris.append((a, c_, b))
            tris.append((a, d, c_))
    return verts, tris


def _sgn(v):
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


def surface_nets(field_fn, lo, hi, res=44):
    """Naive dual-contouring (surface nets): vertex per sign-crossing cell at
    the mean of crossing-edge midpoints, quad per sign-crossing grid edge
    connecting the 4 adjacent cell vertices. No lookup tables, closed
    manifold-ish output. Winding fixed by centroid heuristic."""
    step = tuple((hi[k] - lo[k]) / res for k in range(3))
    # sample grid (res+1)^3
    F = [[[0.0] * (res + 1) for _ in range(res + 1)] for _ in range(res + 1)]
    P = [[[None] * res for _ in range(res)] for _ in range(res)]  # cell verts
    for i in range(res + 1):
        x = lo[0] + step[0] * i
        for j in range(res + 1):
            y = lo[1] + step[1] * j
            for k in range(res + 1):
                z = lo[2] + step[2] * k
                F[i][j][k] = field_fn(x, y, z)
    # cell vertices at mean of crossing-edge midpoints
    verts = []
    for i in range(res):
        for j in range(res):
            for k in range(res):
                # corners of cell (i,j,k)
                cs = (F[i][j][k], F[i+1][j][k], F[i][j+1][k], F[i+1][j+1][k],
                      F[i][j][k+1], F[i+1][j][k+1], F[i][j+1][k+1], F[i+1][j+1][k+1])
                if all(s <= 0 for s in cs) or all(s > 0 for s in cs):
                    continue
                pts = []
                # 12 edges of the cube (corner indices)
                edges = ((0,1),(0,2),(4,6),(5,7),(0,4),(1,5),(2,3),(6,7),(2,6),(3,7),(1,3),(4,5))
                for a, b in edges:
                    if (cs[a] <= 0) != (cs[b] <= 0):
                        pa = (lo[0]+step[0]*(i+(1 if a&1 else 0)), lo[1]+step[1]*(j+(1 if a>>1 else 0)), lo[2]+step[2]*(k+(1 if a>>2 else 0)))
                        pb = (lo[0]+step[0]*(i+(1 if b&1 else 0)), lo[1]+step[1]*(j+(1 if b>>1 else 0)), lo[2]+step[2]*(k+(1 if b>>2 else 0)))
                        # linear interpolation on field value
                        t = cs[a] / (cs[a] - cs[b]) if cs[a] != cs[b] else 0.5
                        pts.append(tuple(pa[m] + (pb[m]-pa[m]) * t for m in range(3)))
                if not pts:
                    continue
                cx = sum(p[0] for p in pts)/len(pts); cy = sum(p[1] for p in pts)/len(pts); cz = sum(p[2] for p in pts)/len(pts)
                P[i][j][k] = len(verts)
                verts.append((cx, cy, cz))
    # quads: for each grid edge with sign change, join 4 adjacent cells
    tris = []
    cx = sum(v[0] for v in verts)/max(1,len(verts))
    cy = sum(v[1] for v in verts)/max(1,len(verts))
    cz = sum(v[2] for v in verts)/max(1,len(verts))
    def quad(a, b, c_, d):
        if a is None or b is None or c_ is None or d is None:
            return
        # orient outward from centroid: full 3D cross product dotted with
        # (face center - centroid), not a scrambled projection
        va, vb, vc = verts[a], verts[b], verts[c_]
        u = (vb[0]-va[0], vb[1]-va[1], vb[2]-va[2])
        w = (vc[0]-va[0], vc[1]-va[1], vc[2]-va[2])
        nx = u[1]*w[2] - u[2]*w[1]
        ny = u[2]*w[0] - u[0]*w[2]
        nz = u[0]*w[1] - u[1]*w[0]
        mid = ((verts[a][0]+verts[c_][0])/2-cx, (verts[a][1]+verts[c_][1])/2-cy, (verts[a][2]+verts[c_][2])/2-cz)
        if nx*mid[0] + ny*mid[1] + nz*mid[2] > 0:
            tris.append((a, b, c_)); tris.append((a, c_, d))
        else:
            tris.append((a, c_, b)); tris.append((a, d, c_))
    for i in range(res + 1):
        for j in range(res + 1):
            for k in range(res + 1):
                # x-edge: cells (i,j-1,k-1),(i,j,k-1),(i,j-1,k),(i,j,k)
                if 0 < j < res and 0 < k < res and i < res and (F[i][j][k] <= 0) != (F[i+1][j][k] <= 0):
                    quad(P[i][j-1][k-1], P[i][j][k-1], P[i][j][k], P[i][j-1][k])
                # y-edge
                if 0 < i < res and 0 < k < res and j < res and (F[i][j][k] <= 0) != (F[i][j+1][k] <= 0):
                    quad(P[i-1][j][k-1], P[i-1][j][k], P[i][j][k], P[i][j][k-1])
                # z-edge
                if 0 < i < res and 0 < j < res and k < res and (F[i][j][k] <= 0) != (F[i][j][k+1] <= 0):
                    quad(P[i-1][j-1][k], P[i][j-1][k], P[i][j][k], P[i-1][j][k])
    return verts, tris


def julia_mesh(c, res=44):
    """Quaternion julia fractal mesh, EXACT POV bounds/escape (fractal.cpp
    Compute_BBox): R = min(2, 1+|c|) + tol, Exit_Value = R^2 + tol, set
    contained in sphere |p| <= R. Worst |c| here ~0.77 -> R ~1.77."""
    R = min(2.0, 1.0 + math.sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2] + c[3]*c[3]))
    exit_value = R * R + 1e-8
    return surface_nets(
        lambda x, y, z: julia_field(x, y, z, c, max_iter=7, exit_value=exit_value),
        (-R, -R, -R), (R, R, R), res=res,
    )


def bollard_mesh(noise_fn, res=40):
    """Isosurface bollard at Pos1: f = max(sqrt(x^2+y^2)-0.25, z-0.7) - pigm*0.07.
    POV contained_by box <-0.35,-0.35,0>,<0.35,0.35,0.8>; inside = f < 0.
    pigm approximated by POV-style turbulence noise (mountain uses same)."""
    def fld(x, y, z):
        r = math.sqrt(x*x + y*y)
        pigm = noise_fn(x, y, z)
        return max(r - 0.25, z - 0.7) - pigm * 0.07
    return surface_nets(fld, (-0.34, -0.34, -0.02), (0.34, 0.34, 0.74), res=res)


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
        write_obj(fname, [p2b(v) for v in verts], swap_winding(sock_tris))
        objects.append({
            "type": "mesh",
            "src": fname,
            "material": "stone",
        })

    # ---- texts on basin walls (ground-truthed 2026-08-24, probes/gt-beam)
    # POV net chains conjugated through p2b (det -1):
    #   POV_Text:     scale 0.3 -> Rx(90)Rz(-90), T(-1.97,0.995,0.575)
    #     => beam rotY(+90), west wall (x=1.9), face +x, upright
    #   Version_Text: scale 0.3 -> Rx(90)Rz(-90) then object rotZ(90) cancels
    #     the Rz => net Rx(90) only: beam NO rotation, south wall, face -z
    # anchor offsets tuned so centroids match POV reference within 0.02.
    objects.append({
        "type": "text",
        "text": "POV-Ray",
        "size": 0.3 * 0.7,
        "depth": 0.075,
        "material": "stone2",
        "transform": {
            "rotate": [0, 90, 0],
            "translate": [1.9, 0.575, -0.55],
        },
    })
    objects.append({
        "type": "text",
        "text": "Version 3.8",
        "size": 0.3 * 0.7,
        "depth": 0.075,
        "material": "stone2",
        "transform": {
            "translate": [-0.15, 0.575, 1.42],
        },
    })

    # ---- glass sphere at Pos3
    objects.append({
        "type": "sphere",
        "center": p2b(tuple(a + b for a, b in zip(POV_POSITIONS["Pos3"], (0, 0, 0.24)))),
        "radius": 0.24,
        "material": "glass",
    })

    # ---- Pos1: metal collar ring — POV difference { cylinder -0z..0.15z r0.7
    #      minus cylinder r0.68 } translate 0.6z: annulus at world z 0.6-0.75.
    #      Kept as flattened-sphere stand-in: r 0.7, y-scale 0.107 (thickness
    #      2*0.7*0.107 = 0.15), center z = 0.675 = Pos1_z + 0.075.
    objects.append({
        "type": "sphere",
        "center": p2b((POV_POSITIONS["Pos1"][0], POV_POSITIONS["Pos1"][1],
                        POV_POSITIONS["Pos1"][2] + 0.075)),
        "radius": 0.7,
        "transform": {"scale": [1.0, 0.107, 1.0]},
        "material": "metal",
    })
    #      orange isosurface bollard: max(sqrt(x^2+y^2)-0.25, z-0.7) - pigm*0.07,
    #      bozo-turbulence displaced cylinder, translated Pos1
    # 3D fbm (x,y AND z): no-z noise = unbroken vertical ridges = spline
    # shaft. POV bozo varies in 3D, so lumps must break up with height.
    _gs3 = [make_noise3(707), make_noise3(1337)]
    def _noise(x, y, z):
        u = (x + 0.35) / 0.7
        v = (y + 0.35) / 0.7
        w = (z + 0.02) / 0.76
        total = 0.0
        for o, g in enumerate(_gs3):
            f = 1.6 * (2 ** o)
            total += (noise3(g, u * f, v * f, w * f) - 0.5) / (2 ** o)
        return total + 0.5
    _bv, _bt = bollard_mesh(_noise)
    _bv = apply_pov_transforms(_bv, [("translate", POV_POSITIONS["Pos1"])])
    write_obj("obj-bollard.obj", [p2b(v) for v in _bv], swap_winding(_bt))
    objects.append({
        "type": "mesh",
        "src": "obj-bollard.obj",
        "material": "flame",
    })

    # ---- Pos1: POV logo (logo.inc CSG: sphere + carved cone + carved
    #      crescent), full object chain Rx(90) scale .4 T(0,-.2,0) Rz(-25)
    #      T(0,0,.96) T(Pos1) baked — verified vs POV render (probes/)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("logo_mesh", "probes/logo_mesh.py")
    _lm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_lm)
    _lv, _lt = _lm.logo_final()
    write_obj("obj-logo.obj", [p2b(v) for v in _lv], swap_winding(_lt))
    objects.append({
        "type": "mesh",
        "src": "obj-logo.obj",
        "material": "logo",
    })

    # ---- Pos2: orbit ring — 6 superellipsoids (metal) + 6 quaternion julia
    #      fractals (pink), exact POV rand() sequence seed(45), transforms
    #      baked per benchmark.pov: supers scale 0.12, T<0.6,0,0.12>, Rz(Cnt+30);
    #      julias scale 0.12, T<0.6,0,0.1>, Rz(Cnt). Then union T(Pos2).
    rand = pov_rand_seq(45)
    ox, oy, oz = POV_POSITIONS["Pos2"]
    for i, cnt in enumerate(range(0, 360, 60)):
        e = rand() * 2
        n_exp = rand() * 2
        cx = rand()
        cy = rand() * 0.6
        c = (cx, cy, -0.54, 0.2)

        # superellipsoid (metal) at (Cnt+30) deg
        sv, st = superellipsoid_mesh(e, n_exp)
        sv = apply_pov_transforms(
            sv,
            [("scale", (0.12, 0.12, 0.12)), ("translate", (0.6, 0.0, 0.12)),
             ("rotz", cnt + 30), ("translate", POV_POSITIONS["Pos2"])],
        )
        fname = f"obj-super-{i}.obj"
        write_obj(fname, [p2b(v) for v in sv], swap_winding(st))
        objects.append({"type": "mesh", "src": fname, "material": "metal"})

        # julia fractal (pink) at Cnt deg
        jv, jt = julia_mesh(c, res=40)
        jv = apply_pov_transforms(
            jv,
            [("scale", (0.12, 0.12, 0.12)), ("translate", (0.6, 0.0, 0.1)),
             ("rotz", cnt), ("translate", POV_POSITIONS["Pos2"])],
        )
        fname = f"obj-julia-{i}.obj"
        write_obj(fname, [p2b(v) for v in jv], swap_winding(jt))
        objects.append({"type": "mesh", "src": fname, "material": "pink"})

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
    write_obj(fname, [p2b(v) for v in fn_verts], swap_winding(fn_tris))
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
    write_obj(fname, [p2b(v) for v in tw_verts], swap_winding(tw_tris))
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
    write_obj(fname, [p2b(v) for v in m_verts], swap_winding(m_tris))
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
            "color": [0.290, 0.285, 0.298],
            "ambient": 0.38, "ambientColor": [0.257, 0.216, 0.161],
            # 3-stop sky: [zenith, horizon, below-horizon]. Stops solved by
            # inverting POV reference pixels through exposure 2.6 + Reinhard
            # (POV's display gamma brightens its declared values considerably:
            # grey 0.667 displays ~0.83, so our raw stops must be much higher).
            # skyMap: piecewise gradient keyed by ray dir.Y, stops solved
            # directly from POV reference rows (row -> camera elevation via
            # env probe, then invert exposure 2.6 + Reinhard). The reference
            # sky is far steeper near zenith than any 2-stop ramp can follow.
            "skyMap": [
                # y values = ACTUAL ray dir.Y per image row (corrected probe;
                # camera pitch -8.4deg, image top = dir.Y 0.077). Colors =
                # POV reference rows inverted through exposure 2.6 + Reinhard.
                {"y": 0.10,   "color": [0.679, 1.059, 5.162]},  # just above frame
                {"y": 0.074,  "color": [0.627, 0.951, 4.543]},  # fy=0.01 (198,217,247)
                {"y": 0.061,  "color": [0.468, 0.683, 2.061]},   # fy=0.04 (184,205,238)
                {"y": 0.043,  "color": [0.379, 0.528, 1.296]},   # fy=0.08 (173,194,229)
                {"y": 0.025,  "color": [0.426, 0.528, 0.925]},   # fy=0.12 (179,194,220)
                {"y": 0.007,  "color": [0.356, 0.359, 0.375]},   # fy=0.16 (169,174,183)
                {"y": -0.011, "color": [0.304, 0.229, 0.195]},   # fy=0.20 (160,148,145)
                {"y": -0.10,  "color": [0.271, 0.203, 0.172]},   # below frame
                {"y": -1.0,   "color": [0.211, 0.151, 0.122]},
            ],
            # POV benchmark: fog_type 2, alt 1.5, offset 0, rgbt <0.75,0.80,0.86,0.2>
            # (t=0.2 -> max 80% fog), distance 400. Profile "atan" = POV's exact
            # 1/(u^2+1) ground-fog density (trace.cpp ComputeGroundFogDepth).
            # POV distance 400 renders nearly invisible; 60 tuned as media
            # proxy (POV's scene includes cloud media we don't model).
            "fog": {
                "color": [0.211, 0.173, 0.134],
                "distance": 60,
                "altitude": 1.5,
                "offset": 0,
                "maxFog": 1.0,
                "profile": "atan",
            },
        },
        "lights": [
            {
                "type": "directional",
                # POV light at sun_pos*150000 shines toward origin: travel dir = -pos
                "dir": p2b([-sun_dir_pov[0], -sun_dir_pov[1], -sun_dir_pov[2]]),
                # Empirically calibrated (raw POV 3.43/2.87/1.95 is too dark for
                # Beam: POV's radiosity + media-attenuated sun ≈ our brighter direct)
                "color": [2.441, 1.909, 1.150],
            },
            {
                "type": "directional",
                "dir": [-0.9, -0.5, -0.6],
                "color": [0.281, 0.259, 0.229],
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
            "exposure": 1.0, "tonemap": "clip",
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
