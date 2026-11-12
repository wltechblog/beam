#!/usr/bin/env python3
"""Tessellate the original Utah Teapot Bezier patch data into a Wavefront OBJ.

Source data: teapot_bezier — Martin Newell's 1975 teapot, 28 bicubic Bezier
patches (no bottom — authentic), as distributed by the Utah Graphics Lab
model repository (https://users.cs.utah.edu/~dejohnso/models/teapot.html).
The teapot data is universally treated as public domain.

Newell's data is already Y-up (lid rim: radius 1.4 at y=2.25; teapot stands
on y=0). Newell's patch grids do NOT share a uniform orientation — some
patches yield inward analytic normals — so we do not trust the analytic
normal sign at all. Instead:

  1. Evaluate every patch on an (N+1)x(N+1) grid; weld coincident vertices
     (1e-5 quantization) so shared patch boundaries are seamless.
  2. Drop degenerate triangles (patch corners such as the lid knob collapse
     to a point; their zero-area triangles poison adjacency).
  3. Orient triangles consistently per connected component by edge
     adjacency propagation.
  4. Flip each component so its surface points away from its centroid
     (sum of dot(face normal, vertex - centroid) > 0). Robust for the
     teapot's open tubes (spout, handle) where signed volume is ill-defined.
  5. Vertex normals are area-weighted sums of incident face normals —
     consistent with the winding by construction, and smooth at this
     tessellation density.

Usage: gen_teapot.py [teapot_bezier] [teapot.obj] [subdivisions]
"""

import math
import re
import sys
from collections import defaultdict, deque

BEZIER = sys.argv[1] if len(sys.argv) > 1 else "teapot_bezier"
OUT = sys.argv[2] if len(sys.argv) > 2 else "teapot.obj"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 14  # subdivisions per side


def bernstein(i, n, t):
    return math.comb(n, i) * t**i * (1 - t) ** (n - i)


def bernstein_deriv(i, n, t):
    if n == 0:
        return 0.0
    a = i * bernstein(i - 1, n - 1, t) if i > 0 else 0.0
    b = (n - i) * bernstein(i, n - 1, t) if i < n else 0.0
    return a - b


def eval_patch(P, u, v):
    """S(u,v) for a 4x4 control grid (normals are recomputed later)."""
    S = [0.0, 0.0, 0.0]
    for i in range(4):
        for j in range(4):
            w = bernstein(i, 3, u) * bernstein(j, 3, v)
            p = P[i][j]
            for k in range(3):
                S[k] += w * p[k]
    return S


def main():
    text = open(BEZIER).read()
    pts = [
        (float(x), float(y), float(z))
        for x, y, z in re.findall(
            r"pt\(\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*\)",
            text,
        )
    ]
    n_patches = len(pts) // 16
    assert len(pts) % 16 == 0 and 28 <= n_patches <= 36, (
        f"expected 448-576 control points (28-36 patches), got {len(pts)}"
    )
    patches = [
        [[pts[pi * 16 + row * 4 + col] for col in range(4)] for row in range(4)]
        for pi in range(n_patches)
    ]

    # Weld vertices at shared patch boundaries so the surface is seamless.
    weld = {}
    verts = []

    def vert_index(S):
        key = (round(S[0], 5), round(S[1], 5), round(S[2], 5))
        idx = weld.get(key)
        if idx is None:
            idx = len(verts)
            weld[key] = idx
            verts.append(S)
        return idx

    tris = []
    for P in patches:
        grid = [
            [vert_index(eval_patch(P, i / N, j / N)) for j in range(N + 1)]
            for i in range(N + 1)
        ]
        for i in range(N):
            for j in range(N):
                a = grid[i][j]
                b = grid[i][j + 1]
                c = grid[i + 1][j]
                d = grid[i + 1][j + 1]
                tris.append((a, c, b))
                tris.append((b, c, d))

    # Drop degenerate triangles (repeated indices or zero area).
    def cross(u, v):
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    clean = []
    for a, b, c in tris:
        if a == b or b == c or a == c:
            continue
        fn = cross(sub(verts[b], verts[a]), sub(verts[c], verts[a]))
        if fn[0] * fn[0] + fn[1] * fn[1] + fn[2] * fn[2] < 1e-16:
            continue
        clean.append((a, b, c))
    tris = clean

    # Consistent orientation per connected component (edge propagation).
    edge = defaultdict(list)
    for ti, (a, b, c) in enumerate(tris):
        edge[(a, b)].append(ti)
        edge[(b, c)].append(ti)
        edge[(c, a)].append(ti)

    flip = [False] * len(tris)
    seen = [False] * len(tris)
    components = 0
    for seed in range(len(tris)):
        if seen[seed]:
            continue
        components += 1
        seen[seed] = True
        q = deque([seed])
        while q:
            t = q.popleft()
            a, b, c = tris[t]
            if flip[t]:
                a, b, c = c, b, a
            for (p, r) in ((a, b), (b, c), (c, a)):
                for nb in edge[(p, r)] + edge[(r, p)]:
                    if nb == t or seen[nb]:
                        continue
                    na, nbv, nc = tris[nb]
                    if flip[nb]:
                        na, nbv, nc = nc, nbv, na
                    # consistent iff neighbour traverses shared edge
                    # in the opposite direction (r -> p)
                    consistent = (
                        (na, nbv) == (r, p)
                        or (nbv, nc) == (r, p)
                        or (nc, na) == (r, p)
                    )
                    flip[nb] = (not consistent) != flip[nb]
                    seen[nb] = True
                    q.append(nb)
    tris = [
        (c, b, a) if f else (a, b, c) for f, (a, b, c) in zip(flip, tris)
    ]

    # Per-component outward sign: sum over faces of dot(face normal,
    # vertex - centroid) must be positive for an outward-oriented surface.
    comp_id = [-1] * len(tris)
    cid = 0
    for seed in range(len(tris)):
        if comp_id[seed] >= 0:
            continue
        comp_id[seed] = cid
        q = deque([seed])
        while q:
            t = q.popleft()
            a, b, c = tris[t]
            for (p, r) in ((a, b), (b, c), (c, a)):
                for nb in edge[(p, r)] + edge[(r, p)]:
                    if comp_id[nb] < 0:
                        comp_id[nb] = cid
                        q.append(nb)
        cid += 1

    for ci in range(cid):
        idxs = [t for t in range(len(tris)) if comp_id[t] == ci]
        used = sorted({v for t in idxs for v in tris[t]})
        cx = sum(verts[v][0] for v in used) / len(used)
        cy = sum(verts[v][1] for v in used) / len(used)
        cz = sum(verts[v][2] for v in used) / len(used)
        acc = 0.0
        for t in idxs:
            a, b, c = tris[t]
            fn = cross(sub(verts[b], verts[a]), sub(verts[c], verts[a]))
            mid = (
                (verts[a][0] + verts[b][0] + verts[c][0]) / 3 - cx,
                (verts[a][1] + verts[b][1] + verts[c][1]) / 3 - cy,
                (verts[a][2] + verts[b][2] + verts[c][2]) / 3 - cz,
            )
            acc += fn[0] * mid[0] + fn[1] * mid[1] + fn[2] * mid[2]
        if acc < 0:
            for t in idxs:
                a, b, c = tris[t]
                tris[t] = (c, b, a)

    # Vertex normals: area-weighted incident face normals (face normal
    # cross product magnitude = 2 * triangle area, so the raw cross product
    # is already area-weighted). Consistent with winding by construction.
    norms = [[0.0, 0.0, 0.0] for _ in verts]
    for a, b, c in tris:
        fn = cross(sub(verts[b], verts[a]), sub(verts[c], verts[a]))
        for v in (a, b, c):
            norms[v][0] += fn[0]
            norms[v][1] += fn[1]
            norms[v][2] += fn[2]
    for nrm in norms:
        ln = math.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2) or 1.0
        nrm[0] /= ln
        nrm[1] /= ln
        nrm[2] /= ln

    # Guarantee: winding agrees with vertex normals everywhere.
    bad = 0
    for a, b, c in tris:
        fn = cross(sub(verts[b], verts[a]), sub(verts[c], verts[a]))
        avg = [norms[a][k] + norms[b][k] + norms[c][k] for k in range(3)]
        if fn[0] * avg[0] + fn[1] * avg[1] + fn[2] * avg[2] <= 0:
            bad += 1
    if bad:
        raise SystemExit(f"{bad} triangles disagree with vertex normals")

    with open(OUT, "w") as f:
        f.write("# Utah Teapot, tessellated from the original 1975 Martin Newell\n")
        f.write("# Bezier patch data (28 bicubic patches, no bottom, public domain).\n")
        f.write(f"# Generated by gen_teapot.py at {N}x{N} subdivisions per patch.\n")
        f.write(f"# {len(verts)} vertices, {len(tris)} triangles, "
                f"{cid} welded component(s)\n")
        for S in verts:
            f.write(f"v {S[0]:.5f} {S[1]:.5f} {S[2]:.5f}\n")
        for nrm in norms:
            f.write(f"vn {nrm[0]:.5f} {nrm[1]:.5f} {nrm[2]:.5f}\n")
        for a, b, c in tris:
            f.write(f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n")

    print(
        f"wrote {OUT}: {len(verts)} verts, {len(tris)} tris, "
        f"{cid} component(s); orientation verified"
    )


if __name__ == "__main__":
    main()
