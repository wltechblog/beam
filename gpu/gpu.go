package gpu

// Host-side scene flattening + GPU render entry.
//
// The CPU World/Camera are flattened into SoA float64/int32 arrays matching
// kernel_main.cl's trace_pixel signature exactly. In the fp64 build every
// value crosses the boundary bit-identically, and the kernel replicates the
// CPU arithmetic order — the render is byte-identical to the CPU renderer.

import (
	_ "embed"
	"fmt"
	"image"
	"os"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/wltechblog/beam/geom"
	"github.com/wltechblog/beam/render"
)

//go:embed kernel_math.cl
var clMath string

//go:embed kernel_trace.cl
var clTrace string

//go:embed kernel_main.cl
var clMain string

// KernelSource returns the concatenated kernel source. Pass -D BEAM_DOUBLE=1
// as a build option for the bit-exact fp64 path.
func KernelSource() string {
	return clMath + clTrace + clMain
}

func put3(dst []float64, i int, v geom.Vec3) {
	dst[i], dst[i+1], dst[i+2] = v.X, v.Y, v.Z
}

func matKindNum(k byte) int32 {
	switch k {
	case 'M':
		return 1
	case 'g':
		return 2
	case 'e':
		return 3
	}
	return 0
}

func lightKindNum(k byte) int32 {
	switch k {
	case 'd':
		return 1
	case 's':
		return 2
	}
	return 0
}

func rgbDefault(c []float64) geom.Vec3 {
	if len(c) != 3 {
		return geom.Vec3{X: 0.8, Y: 0.8, Z: 0.8}
	}
	return geom.Vec3{X: c[0], Y: c[1], Z: c[2]}
}

// sceneArrays holds the flattened world in kernel layout.
type sceneArrays struct {
	objA, objB           []float64 // objA: center|planeP (3/obj); objB: radius|planeN
	objKind, objRoot     []int32   // objRoot: mesh root node (global), -1 none
	objNorms, objMat     []int32
	nodeBMin, nodeBMax   []float64 // 3/node
	nodeLeft, nodeRight  []int32   // leaf: Left=global tri start; internal: children nodes
	nodeCount            []int32   // leaf: count; internal: 0
	triV, triN           []float64 // 9/tri in flattened traversal order
	matAlb, matP0, matP1 []float64 // 3/mat
	matPar               []float64 // 9/mat
	matFlags             []int32   // 4/mat
	lPos, lDir, lCol     []float64 // 3/light
	lPar                 []float64 // 2/light: intensity, cosAngle
	lFlags               []int32   // 3/light: kind, noFalloff, shadowless
	skyStops             []float64 // 4/stop
	envBuf               []float64 // 17
	envFlags             []int32   // 4
	fogBuf               []float64 // 7
}

func flatten(w *render.World) (bufs sceneArrays) {
	n := len(w.Objects)
	bufs.objA = make([]float64, max(n, 1)*3)
	bufs.objB = make([]float64, max(n, 1)*3)
	bufs.objKind = make([]int32, max(n, 1))
	bufs.objRoot = make([]int32, max(n, 1))
	bufs.objNorms = make([]int32, max(n, 1))
	bufs.objMat = make([]int32, max(n, 1))

	for i, o := range w.Objects {
		switch o.Kind {
		case 's':
			bufs.objKind[i] = 0
			put3(bufs.objA, i*3, o.Center)
			bufs.objB[i*3] = o.Radius
		case 'p':
			bufs.objKind[i] = 1
			put3(bufs.objA, i*3, o.PlaneP)
			put3(bufs.objB, i*3, o.PlaneN)
		case 'm':
			bufs.objKind[i] = 2
			if o.Mesh != nil && o.BVH != nil {
				nodes, triOrder := o.BVH.Flatten()
				nodeBase := int32(len(bufs.nodeBMin) / 3)
				triBase := int32(len(bufs.triV) / 9)
				hasNorms := int32(0)
				if len(o.Mesh.Norms) > 0 {
					hasNorms = 1
				}
				bufs.objNorms[i] = hasNorms
				for _, nd := range nodes {
					bufs.nodeBMin = append(bufs.nodeBMin, nd.BMin.X, nd.BMin.Y, nd.BMin.Z)
					bufs.nodeBMax = append(bufs.nodeBMax, nd.BMax.X, nd.BMax.Y, nd.BMax.Z)
					if nd.Count == 0 { // internal: Left/Right are node indices in this mesh
						bufs.nodeLeft = append(bufs.nodeLeft, nodeBase+nd.Left)
						bufs.nodeRight = append(bufs.nodeRight, nodeBase+nd.Right)
						bufs.nodeCount = append(bufs.nodeCount, 0)
					} else { // leaf: Left is triOrder start in this mesh
						bufs.nodeLeft = append(bufs.nodeLeft, triBase+nd.Left)
						bufs.nodeRight = append(bufs.nodeRight, 0)
						bufs.nodeCount = append(bufs.nodeCount, nd.Count)
					}
				}
				bufs.objRoot[i] = nodeBase // CPU root is always node 0 → nodeBase
				for _, t := range triOrder {
					idx := o.Mesh.Tris[t]
					a, b, c := o.Mesh.Verts[idx[0]], o.Mesh.Verts[idx[1]], o.Mesh.Verts[idx[2]]
					bufs.triV = append(bufs.triV, a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z)
					if hasNorms == 1 {
						n0, n1, n2 := o.Mesh.Norms[idx[0]], o.Mesh.Norms[idx[1]], o.Mesh.Norms[idx[2]]
						bufs.triN = append(bufs.triN, n0.X, n0.Y, n0.Z, n1.X, n1.Y, n1.Z, n2.X, n2.Y, n2.Z)
					} else {
						bufs.triN = append(bufs.triN, 0, 0, 0, 0, 0, 0, 0, 0, 0)
					}
				}
			} else {
				bufs.objRoot[i] = -1
			}
		}
		bufs.objMat[i] = int32(o.MatIdx)
	}

	m := len(w.Materials)
	bufs.matAlb = make([]float64, max(m, 1)*3)
	bufs.matP0 = make([]float64, max(m, 1)*3)
	bufs.matP1 = make([]float64, max(m, 1)*3)
	bufs.matPar = make([]float64, max(m, 1)*9)
	bufs.matFlags = make([]int32, max(m, 1)*4)
	for i, mm := range w.Materials {
		put3(bufs.matAlb, i*3, mm.Albedo)
		bufs.matPar[i*9+0] = mm.Roughness
		bufs.matPar[i*9+1] = mm.IOR
		bufs.matPar[i*9+2] = mm.Reflectivity
		bufs.matPar[i*9+3] = mm.Power
		bufs.matPar[i*9+4] = mm.Diffuse
		bufs.matPar[i*9+5] = mm.SpecularK
		bufs.matFlags[i*4+0] = matKindNum(mm.Kind)
		if mm.Pattern != nil {
			var c0, c1 []float64
			if len(mm.Pattern.Colors) > 0 {
				c0 = mm.Pattern.Colors[0]
			}
			if len(mm.Pattern.Colors) > 1 {
				c1 = mm.Pattern.Colors[1]
			}
			put3(bufs.matP0, i*3, rgbDefault(c0))
			put3(bufs.matP1, i*3, rgbDefault(c1))
			bufs.matPar[i*9+6] = mm.Pattern.Scale
			bufs.matPar[i*9+7] = mm.Pattern.From
			bufs.matPar[i*9+8] = mm.Pattern.To
			switch mm.Pattern.Type {
			case "checker":
				bufs.matFlags[i*4+1] = 1
			case "gradient":
				bufs.matFlags[i*4+1] = 2
				switch mm.Pattern.Axis {
				case "y":
					bufs.matFlags[i*4+2] = 1
				case "z":
					bufs.matFlags[i*4+2] = 2
				}
			}
		}
		if mm.Untinted {
			bufs.matFlags[i*4+3] = 1
		}
	}

	l := len(w.Lights)
	bufs.lPos = make([]float64, max(l, 1)*3)
	bufs.lDir = make([]float64, max(l, 1)*3)
	bufs.lCol = make([]float64, max(l, 1)*3)
	bufs.lPar = make([]float64, max(l, 1)*2)
	bufs.lFlags = make([]int32, max(l, 1)*3)
	for i, lt := range w.Lights {
		put3(bufs.lPos, i*3, lt.Pos)
		put3(bufs.lDir, i*3, lt.Dir)
		put3(bufs.lCol, i*3, lt.Color)
		bufs.lPar[i*2+0] = lt.Intensity
		bufs.lPar[i*2+1] = lt.CosAngle
		bufs.lFlags[i*3+0] = lightKindNum(lt.Kind)
		if lt.NoFalloff {
			bufs.lFlags[i*3+1] = 1
		}
		if lt.Shadowless {
			bufs.lFlags[i*3+2] = 1
		}
	}

	if len(w.SkyStops) > 0 {
		bufs.skyStops = make([]float64, 0, len(w.SkyStops)*4)
		for _, st := range w.SkyStops {
			bufs.skyStops = append(bufs.skyStops, st.Y, st.Color.X, st.Color.Y, st.Color.Z)
		}
	} else {
		bufs.skyStops = make([]float64, 4) // dummy; nStops=0
	}

	bufs.envBuf = make([]float64, 17)
	put3(bufs.envBuf, 0, w.EnvColor)
	put3(bufs.envBuf, 3, w.EnvTop)
	put3(bufs.envBuf, 6, w.EnvMid)
	put3(bufs.envBuf, 9, w.EnvBot)
	bufs.envBuf[12] = w.SkyCurve
	bufs.envBuf[13] = w.Ambient
	put3(bufs.envBuf, 14, w.AmbientColor)

	bufs.envFlags = make([]int32, 4)
	if w.SkyGradient {
		bufs.envFlags[0] = 1
	}
	bufs.envFlags[1] = int32(len(w.SkyStops))
	if w.Fog != nil {
		bufs.envFlags[2] = 1
		if w.Fog.Atan {
			bufs.envFlags[3] = 1
		}
	}

	bufs.fogBuf = make([]float64, 7)
	if w.Fog != nil {
		put3(bufs.fogBuf, 0, w.Fog.Color)
		bufs.fogBuf[3] = w.Fog.K
		bufs.fogBuf[4] = w.Fog.Alt
		bufs.fogBuf[5] = w.Fog.Off
		bufs.fogBuf[6] = w.Fog.Max
	}
	return bufs
}

// Diagnostic runs the common no-device failure checks and returns a
// human-readable explanation of why the GPU is unusable.
func Diagnostic() string {
	_, err := ProbeDevices()
	if err == nil {
		return "GPU OK"
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("GPU unavailable: %v. ", err))
	if _, serr := os.Stat("/dev/dri"); serr != nil {
		b.WriteString("/dev/dri is not mounted in this container. " +
			"Add `devices: [/dev/dri:/dev/dri]` to the container config.")
		return b.String()
	}
	b.WriteString("/dev/dri exists. Checking device access:\n")
	entries, _ := os.ReadDir("/dev/dri")
	any := false
	allAccessible := true
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, "renderD") && !strings.HasPrefix(name, "card") {
			continue
		}
		any = true
		path := "/dev/dri/" + name
		f, oerr := os.OpenFile(path, os.O_RDWR, 0)
		if oerr == nil {
			f.Close()
			b.WriteString(fmt.Sprintf("  %s: accessible\n", path))
		} else {
			allAccessible = false
			b.WriteString(fmt.Sprintf("  %s: %v\n", path, oerr))
		}
	}
	if !any {
		b.WriteString("  /dev/dri contains no renderD*/card* nodes. " +
			"The container needs the host GPU devices; see the /dev/dri note below.")
		return b.String()
	}
	if !allAccessible {
		b.WriteString("Device nodes are not accessible to this user. General fixes, most common first:\n" +
			"  - ensure the runtime user is in the host's `video` and `render` groups " +
			"(in containers, pass them through: e.g. docker-compose `group_add: [video, render]`, " +
			"and make sure the group drop step — gosu/su-exec/setpriv — keeps supplementary groups)\n" +
			"  - or run the container with `user: root`, or `privileged: true` (broadest, least secure)\n" +
			"  - or, as a quick workaround: `chmod 666 /dev/dri/*` on the host " +
			"(reverts on reboot / device re-enumeration)")
		return b.String()
	}
	b.WriteString("All /dev/dri nodes are accessible, but no OpenCL platform was found. " +
		"The GPU driver/runtime is missing or broken:\n" +
		"  - install the vendor OpenCL runtime (e.g. Debian/Ubuntu: `apt install intel-opencl-icd`, " +
		"AMD: `rocm-opencl-runtime`, NVIDIA: `nvidia-opencl-cuda-tools`)\n" +
		"  - check that /etc/OpenCL/vendors/*.icd exists and points to an installed library\n" +
		"  - verify with `clinfo`; OCL_ICD_VENDORS env var can override the vendor directory")
	return b.String()
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// RenderGPU renders a compiled World on the first available GPU device.
// double=true selects the bit-exact fp64 path (cl_khr_fp64 required).
// Returns the RGBA image and wall time of the kernel.
func RenderGPU(w *render.World, cam *render.Camera, width, height, samples, maxDepth int,
	exposure float64, clip bool, seed int64, double bool) (*image.RGBA, time.Duration, error) {

	devs, err := ProbeDevices()
	if err != nil {
		return nil, 0, fmt.Errorf("%w\n%s", err, Diagnostic())
	}
	if sel := os.Getenv("BEAM_DEVICE"); sel != "" {
		// Explicit device selection: numeric index into the ranked list,
		// or a case-insensitive name substring match.
		found := false
		if idx, ierr := strconv.Atoi(sel); ierr == nil && idx >= 0 && idx < len(devs) {
			devs = []Device{devs[idx]}
			found = true
		} else {
			for _, d := range devs {
				if strings.Contains(strings.ToLower(d.Name), strings.ToLower(sel)) {
					devs = []Device{d}
					found = true
					break
				}
			}
		}
		if !found {
			names := make([]string, len(devs))
			for i, d := range devs {
				names[i] = d.Name
			}
			return nil, 0, fmt.Errorf("BEAM_DEVICE=%q matches no GPU device; available: %s",
				sel, strings.Join(names, ", "))
		}
	}
	var dev Device
	for _, d := range devs {
		if double && !HasFP64(d) {
			continue
		}
		dev = d
		break
	}
	if dev.ID == nil {
		return nil, 0, fmt.Errorf("no suitable GPU device (fp64=%v)", double)
	}

	ctx, err := NewContext(dev)
	if err != nil {
		return nil, 0, err
	}
	defer ctx.Release()

	// Device identity on stderr for every GPU render — makes "which silicon
	// actually ran this" visible (CLI and MCP), so a CPU-side OpenCL device
	// or a wrong-device selection can't masquerade as a GPU run.
	prec := "fp32"
	if double {
		prec = "fp64"
	}
	fmt.Fprintf(os.Stderr, "gpu device: %s (%s)\n", dev.Name, prec)

	opts := "-cl-std=CL1.2"
	if double {
		opts = "-cl-std=CL1.2 -D BEAM_DOUBLE=1"
	}
	if err := ctx.BuildProgram(KernelSource(), opts); err != nil {
		return nil, 0, err
	}
	if err := ctx.GetKernel("trace_pixel"); err != nil {
		return nil, 0, err
	}

	b := flatten(w)
	camP := cam.GPUParams()
	camSlice := camP[:]

	cfg := []int32{int32(width), int32(height), int32(samples), int32(maxDepth),
		boolInt(clip), int32(len(w.Objects)), int32(len(w.Materials)), int32(len(w.Lights))}
	seedArr := []int64{seed}
	expArr := []float64{exposure}
	pix := make([]byte, width*height*4)

	args := []struct {
		buf  *Buffer
		data any
		ro   bool
	}{}
	_ = args

	// cfg
	bcfg, err := ctx.CreateBufferRO(len(cfg)*4, unsafe.Pointer(&cfg[0]))
	if err != nil {
		return nil, 0, err
	}
	defer bcfg.Release()
	if err := ctx.SetArg(0, bcfg); err != nil {
		return nil, 0, err
	}

	i32 := func(idx int, v []int32) (*Buffer, error) {
		if len(v) == 0 {
			v = make([]int32, 1)
		}
		bb, e := ctx.CreateBufferRO(len(v)*4, unsafe.Pointer(&v[0]))
		if e != nil {
			return nil, e
		}
		return bb, ctx.SetArg(idx, bb)
	}
	// zero-length uploads are invalid in OpenCL — pad to one dummy element.
	// Kernel reads nothing from them (associated counts are 0).
	f64 := func(idx int, v []float64) (*Buffer, error) {
		if len(v) == 0 {
			v = make([]float64, 1)
		}
		if double {
			bb, e := ctx.CreateBufferRO(len(v)*8, unsafe.Pointer(&v[0]))
			if e != nil {
				return nil, e
			}
			return bb, ctx.SetArg(idx, bb)
		}
		// fp32 build: bd is float — upload as float32
		f32 := make([]float32, len(v))
		for i, x := range v {
			f32[i] = float32(x)
		}
		bb, e := ctx.CreateBufferRO(len(f32)*4, unsafe.Pointer(&f32[0]))
		if e != nil {
			return nil, e
		}
		return bb, ctx.SetArg(idx, bb)
	}
	i64 := func(idx int, v []int64) (*Buffer, error) {
		if len(v) == 0 {
			v = make([]int64, 1)
		}
		bb, e := ctx.CreateBufferRO(len(v)*8, unsafe.Pointer(&v[0]))
		if e != nil {
			return nil, e
		}
		return bb, ctx.SetArg(idx, bb)
	}

	handles := []*Buffer{}
	mk := func(bb *Buffer, e error) *Buffer {
		if e != nil {
			panic(e)
		}
		handles = append(handles, bb)
		return bb
	}

	mk(f64(1, camSlice))
	mk(f64(2, b.objA))
	mk(f64(3, b.objB))
	mk(i32(4, b.objKind))
	mk(i32(5, b.objRoot))
	mk(i32(6, b.objNorms))
	mk(i32(7, b.objMat))
	mk(f64(8, b.nodeBMin))
	mk(f64(9, b.nodeBMax))
	mk(i32(10, b.nodeLeft))
	mk(i32(11, b.nodeRight))
	mk(i32(12, b.nodeCount))
	mk(f64(13, b.triV))
	mk(f64(14, b.triN))
	mk(f64(15, b.matAlb))
	mk(f64(16, b.matP0))
	mk(f64(17, b.matP1))
	mk(f64(18, b.matPar))
	mk(i32(19, b.matFlags))
	mk(f64(20, b.lPos))
	mk(f64(21, b.lDir))
	mk(f64(22, b.lCol))
	mk(f64(23, b.lPar))
	mk(i32(24, b.lFlags))
	mk(f64(25, b.skyStops))
	mk(f64(26, b.envBuf))
	mk(f64(27, b.fogBuf))
	mk(i32(28, b.envFlags))
	mk(i64(29, seedArr))
	mk(f64(30, expArr))
	defer func() {
		for _, h := range handles {
			h.Release()
		}
	}()

	bout, err := ctx.CreateBufferWO(len(pix))
	if err != nil {
		return nil, 0, err
	}
	defer bout.Release()
	if err := ctx.SetArg(31, bout); err != nil {
		return nil, 0, err
	}

	t0 := time.Now()
	if err := ctx.Run2D(width, height); err != nil {
		return nil, 0, err
	}
	if err := ctx.Read(bout, pix); err != nil {
		return nil, 0, err
	}
	dt := time.Since(t0)

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	copy(img.Pix, pix)
	return img, dt, nil
}

func boolInt(b bool) int32 {
	if b {
		return 1
	}
	return 0
}
