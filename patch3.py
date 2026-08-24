import sys, re

# ---- scene/schema.go ----
s = open('scene/schema.go').read()
if 'Sky' not in s:
    old = "type Environment struct {\n\tColor    []float64 `json:\"color\"`\n\tAmbient  float64   `json:\"ambient,omitempty\"`\n}"
    if old not in s:
        # try to locate and show actual
        i = s.find('type Environment struct')
        print("ACTUAL:", repr(s[i:i+160])); sys.exit(1)
    new = ("type Environment struct {\n"
           "\tColor    []float64 `json:\"color\"`\n"
           "\tAmbient  float64   `json:\"ambient,omitempty\"`\n"
           "\t// Sky is an optional [zenith, horizon] gradient sampled by miss\n"
           "\t// rays, giving glass and metal something believable to show.\n"
           "\tSky      [2][]float64 `json:\"sky,omitempty\"`\n"
           "}")
    s = s.replace(old, new, 1)
    open('scene/schema.go', 'w').write(s)
    print("schema.go: Sky added")

# ---- render/world.go ----
w = open('render/world.go').read()
if 'SkyGradient' not in w:
    m = re.search(r'(EnvColor\s+geom\.Vec3[^\n]*\n)', w)
    if not m:
        i = w.find('EnvColor')
        print("world.go ACTUAL:", repr(w[max(0,i-80):i+120])); sys.exit(1)
    line = m.group(1)
    w = w.replace(line, line + "\tEnvTop      geom.Vec3\n\tEnvBot      geom.Vec3\n\tSkyGradient bool\n", 1)
    open('render/world.go', 'w').write(w)
    print("world.go: gradient fields added")

# ---- render/compile.go ----
c = open('render/compile.go').read()
if 'EnvTop' not in c:
    m = re.search(r'(\n[^\n]*EnvColor[^\n]*=[^\n]*\n)', c)
    if not m:
        i = c.find('EnvColor')
        print("compile.go ACTUAL:", repr(c[max(0,i-100):i+150])); sys.exit(1)
    line = m.group(1)
    add = ("\tif s.Environment != nil && len(s.Environment.Sky) == 2 {\n"
           "\t\tw.SkyGradient = true\n"
           "\t\tw.EnvTop = vec3Of(s.Environment.Sky[0])\n"
           "\t\tw.EnvBot = vec3Of(s.Environment.Sky[1])\n"
           "\t}\n")
    c = c.replace(line, line + add, 1)
    open('render/compile.go', 'w').write(c)
    print("compile.go: gradient wired")

# ---- render/renderer.go ----
t = open('render/renderer.go').read()
if 'func (r *Renderer) envAt' not in t:
    old = "\thit, obj := r.World.Intersect(ray)\n\tif hit == nil {\n\t\treturn r.World.EnvColor\n\t}"
    assert old in t, "miss return not found"
    t = t.replace(old, "\thit, obj := r.World.Intersect(ray)\n\tif hit == nil {\n\t\treturn r.envAt(ray.Dir)\n\t}", 1)

    old = "\tif depth <= 0 {\n\t\treturn r.World.EnvColor\n\t}"
    assert old in t, "depth return not found"
    t = t.replace(old, "\tif depth <= 0 {\n\t\treturn r.envAt(ray.Dir)\n\t}", 1)

    old = "// trace follows one ray. depth caps reflection/refraction recursion.\nfunc (r *Renderer) trace(ray geom.Ray, depth int, rng *RNG) geom.Vec3 {"
    assert old in t, "trace header not found"
    new = ("// envAt returns environment radiance for a miss direction: flat color\n"
           "// or the zenith/horizon gradient blended on dir.Y.\n"
           "func (r *Renderer) envAt(dir geom.Vec3) geom.Vec3 {\n"
           "\tif !r.World.SkyGradient {\n"
           "\t\treturn r.World.EnvColor\n"
           "\t}\n"
           "\tt := dir.Y*0.5 + 0.5 // -1..1 → 0..1\n"
           "\treturn r.World.EnvBot.Add(r.World.EnvTop.Sub(r.World.EnvBot).MulS(t))\n"
           "}\n\n" + old)
    t = t.replace(old, new, 1)

    old = "jx := (float64(sx) + 0.5) / float64(n) / float64(r.Width)"
    assert old in t, "jitter v2 not found"
    t = t.replace(old, "jx := ((float64(sx) + rng.Float64()) / float64(n)) / float64(r.Width)", 1)
    old = "jy := (float64(sy) + 0.5) / float64(n) / float64(r.Height)"
    assert old in t, "jitter v2 y not found"
    t = t.replace(old, "jy := ((float64(sy) + rng.Float64()) / float64(n)) / float64(r.Height)", 1)

    open('render/renderer.go', 'w').write(t)
    print("renderer.go: envAt + random AA applied")

print("OK")
