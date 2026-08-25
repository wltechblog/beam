#!/usr/bin/env python3
"""Build the real POV-Ray logo (logo.inc Povray_Logo) as a triangle mesh.

CSG: merge { sphere 2y r=1
             difference { cone(2y,1,-4y,0) - sphere(2y,1.4, scale 1,1,2) }
             difference { sphere(0,1, scale 2.6,2.2,1) - sphere(0,1, scale 2.3,1.8,2)
                          translate -0.35x ; rotZ(30) ; translate 2y }
             rotZ(-25); translate(-0.5,-0.35,0); scale 1/4 }

Method: dense sphere-surface sampling with inside-tests (classic blob-erode):
- build each CSG solid as an inside() predicate
- march rays / sample the surface on a fine spherical grid... too slow in py.

Simpler robust method: UV-sphere marching on the OUTER sphere of the union,
plus for each CSG difference part, generate its surface and clip triangles
where inside the subtracted solid. Union = keep triangles not inside any
other solid.

We do triangle-level CSG approximation:
  part A: sphere r=1 @ (0,2,0)                      (keep all)
  part B: cone(2y,1 -> -4y,0) minus sphere(2y,1.4*<1,1,2>)
  part C: ellipsoid(2.6,2.2,1) minus ellipsoid(2.3,1.8,2)+t(-0.35,0,0),
          then rotZ(30), translate(0,2,0)
Final: rotZ(-25), translate(-0.5,-0.35,0), scale(1/4).

All math in POV coords; final verts converted p2b + winding swap at the end.
"""
import math
import numpy as np

def rotz(deg, p):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = p
    return (x * c - y * s, x * s + y * c, z)

def cone_surface(hi=(0, 2, 0), r_hi=1.0, lo=(0, -4, 0), r_lo=0.0, seg=64, rings=24):
    """Lateral cone/frustum surface between two apex points along y axis."""
    verts, tris = [], []
    # param along axis
    ax = np.array(hi) - np.array(lo)
    h = np.linalg.norm(ax)
    uy = ax / h
    # orthonormal basis
    ref = np.array([1.0, 0, 0]) if abs(uy[0]) < 0.9 else np.array([0, 1.0, 0])
    ux = np.cross(uy, ref); ux /= np.linalg.norm(ux)
    uz = np.cross(ux, uy)
    for i in range(rings + 1):
        t = i / rings
        p = np.array(lo) + t * ax
        r = r_lo + t * (r_hi - r_lo)
        for j in range(seg):
            a = 2 * math.pi * j / seg
            verts.append(tuple(p + r * (math.cos(a) * ux + math.sin(a) * uz)))
    for i in range(rings):
        for j in range(seg):
            j2 = (j + 1) % seg
            a, b, c, d = i * seg + j, i * seg + j2, (i + 1) * seg + j, (i + 1) * seg + j2
            tris.append((a, b, c)); tris.append((b, d, c))
    return verts, tris

def sphere_surface(center, radii=(1, 1, 1), seg=64, rings=32):
    verts, tris = [], []
    cx, cy, cz = center
    rx, ry, rz = radii
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(seg):
            th = 2 * math.pi * j / seg
            verts.append((cx + rx * math.sin(phi) * math.cos(th),
                          cy + ry * math.sin(phi) * math.sin(th),
                          cz + rz * math.cos(phi)))
    for i in range(rings):
        for j in range(seg):
            j2 = (j + 1) % seg
            a, b, c, d = i * seg + j, i * seg + j2, (i + 1) * seg + j, (i + 1) * seg + j2
            tris.append((a, b, c)); tris.append((b, d, c))
    return verts, tris

def inside_ellipsoid(p, center, radii):
    dx, dy, dz = p[0] - center[0], p[1] - center[1], p[2] - center[2]
    return (dx / radii[0]) ** 2 + (dy / radii[1]) ** 2 + (dz / radii[2]) ** 2 <= 1.0

def inside_cone(p, hi=(0, 2, 0), r_hi=1.0, lo=(0, -4, 0), r_lo=0.0):
    # distance along axis from lo
    ax = np.array(hi) - np.array(lo); h = np.linalg.norm(ax); uy = ax / h
    d = np.array(p) - np.array(lo)
    t = float(np.dot(d, uy))          # 0 at lo, h at hi
    if t < 0 or t > h:
        return False
    r = r_lo + t / h * (r_hi - r_lo)
    radial = np.linalg.norm(d - t * uy)
    return radial <= r

def clip_tri(verts, tri, inside_fn, eps=1e-9):
    """Keep the part of the triangle OUTSIDE the solid (inside_fn True = cut away).
    Returns list of triangles (vertex indices into verts)."""
    v = [verts[i] for i in tri]
    sig = [inside_fn(p) for p in v]
    if not any(sig):
        return [tri]
    if all(sig):
        return []
    # one or two vertices inside: split. Approximate: emit degenerate-safe
    # centroid-clip (cut at edge crossings).
    out_tris = []
    # edges with sign change -> intersection points
    pts = []  # (point, start_idx)
    for k in range(3):
        a, b = v[k], v[(k + 1) % 3]
        sa, sb = sig[k], sig[(k + 1) % 3]
        if sa and not sb:
            # a inside, b outside: crossing between a->b
            t = 0.5
            # bisect for accuracy
            for _ in range(24):
                m = tuple((a[q] + t * (b[q] - a[q])) for q in range(3))
                if inside_fn(m):
                    t = (t + 1) / 2
                else:
                    t = t / 2
            pts.append(tuple(a[q] + t * (b[q] - a[q]) for q in range(3)))
        elif sb and not sa:
            t = 0.5
            for _ in range(24):
                m = tuple((a[q] + t * (b[q] - a[q])) for q in range(3))
                if inside_fn(m):
                    t = (t + 1) / 2
                else:
                    t = t / 2
            pts.append(tuple(a[q] + t * (b[q] - a[q]) for q in range(3)))
    # the outside part is bounded by: outside vertices + crossing pts
    outside_idx = [k for k in range(3) if not sig[k]]
    if len(outside_idx) == 2 and len(pts) == 2:
        # quad: o1, o2, x2, x1
        verts.append(pts[0]); verts.append(pts[1])
        i1, i2 = len(verts) - 2, len(verts) - 1
        # quad o1, o2, x2, x1 — verify winding by keeping CCW order of original tri
        k0, k1 = outside_idx[0], outside_idx[1]
        if (k1 - k0) % 3 == 1:
            out_tris += [(tri[k0], tri[k1], i2), (tri[k0], i2, i1)]
        else:
            out_tris += [(tri[k1], tri[k0], i1), (tri[k1], i1, i2)]
    elif len(outside_idx) == 1 and len(pts) == 2:
        o = v[outside_idx[0]]
        verts.append(x1_ := pts[0]); verts.append(x2_ := pts[1])
        i1, i2 = len(verts) - 2, len(verts) - 1
        out_tris.append((tri[outside_idx[0]], i1, i2))
    return out_tris

def build_logo():
    # ---- part B: cone minus carved sphere
    vb, tb = cone_surface(rings=28)
    carveB = lambda p: inside_ellipsoid(p, (0, 2, 0), (1.4, 1.4, 2.8))
    # ---- part C: ellipsoid minus inner ellipsoid (pre-transform)
    vc, tc = sphere_surface((0, 0, 0), (2.6, 2.2, 1.0))
    carveC = lambda p: inside_ellipsoid(p, (-0.35, 0, 0), (2.3, 1.8, 2.0))
    # ---- part A: plain sphere
    va, ta = sphere_surface((0, 2, 0), (1, 1, 1))

    parts = []
    for verts, tris, carve, xform in (
        (vb, tb, carveB, lambda p: p),
        (vc, tc, carveC, lambda p: rotz(30, (p[0] + 0.0, p[1], p[2])) ),
    ):
        # part C translate -0.35x is inside carve; surface itself unshifted
        pass

    # clip B
    newv = list(vb)
    newt = []
    for tri in tb:
        newt += clip_tri(newv, tri, carveB)
    vb, tb = newv, newt
    # clip C
    newv = list(vc)
    newt = []
    for tri in tc:
        newt += clip_tri(newv, tri, carveC)
    vc, tc = newv, newt
    # apply C's chain: rotZ(30) then translate 2y  (surface was at origin)
    # NOTE: carve already applied in local coords; but carve translate -0.35x
    # was modeled directly in carveC center. good.
    vc = [ (p[0], p[1] + 2.0, p[2]) for p in vc ]
    # rotate C by rotZ(30): carve was defined pre-rotation (subtraction happens
    # BEFORE rotate in POV: difference { ... } rotate z*30 translate 2y })
    vc = [ rotz(30, p) for p in vc ]
    # hmm: rotating after translation double-transforms. POV order:
    #   difference { s1 - s2 translate <-0.35,0,0> } rotate z*30 translate 2y
    # The -0.35x translate belongs to s2 INSIDE the difference. Model:
    #   carve center (-0.35,0,0) (already done), then whole difference
    #   rotated 30 and lifted 2y. So: rotZ30 THEN translate 2y:
    vc = [ (p[0], p[1] - 2.0, p[2]) for p in vc ]  # undo bad lift first
    vc = [ rotz(30, p) for p in vc ]
    vc = [ (p[0], p[1] + 2.0, p[2]) for p in vc ]

    # merge parts
    verts = list(va)
    tris = [t for t in ta]
    off = len(verts)
    verts += vb
    tris += [(a + off, b + off, c + off) for a, b, c in tb]
    off = len(verts)
    verts += vc
    tris += [(a + len(verts) - len(vc), b + off, c + off) for a, b, c in tc]

    # final logo transform: rotZ(-25), translate(-0.5,-0.35,0), scale 1/4
    out = []
    for p in verts:
        p = rotz(-25, p)
        p = (p[0] - 0.5, p[1] - 0.35, p[2])
        p = (p[0] / 4, p[1] / 4, p[2] / 4)
        out.append(p)
    return out, tris

def rotx(deg, p):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    x, y, z = p
    return (x, y * c - z * s, y * s + z * c)

def logo_final():
    """Logo mesh with the benchmark OBJECT chain baked:
    Rx(90) -> scale 0.4 -> T(0,-0.2,0) -> Rz(-25) -> T(0,0,0.96) -> T(0,0,0.6)."""
    verts, tris = build_logo()
    out = []
    for p in verts:
        p = rotx(90, p)
        p = (p[0] * 0.4, p[1] * 0.4, p[2] * 0.4)
        p = (p[0], p[1] - 0.2, p[2])
        p = rotz(-25, p)
        p = (p[0], p[1], p[2] + 0.96)
        p = (p[0], p[1], p[2] + 0.6)
        out.append(p)
    return out, tris

if __name__ == "__main__":
    verts, tris = logo_final()
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    print(f"verts={len(verts)} tris={len(tris)}")
    print(f"POV final bounds: x[{min(xs):.3f},{max(xs):.3f}] y[{min(ys):.3f},{max(ys):.3f}] z[{min(zs):.3f},{max(zs):.3f}]")
