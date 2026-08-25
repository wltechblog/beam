package gpu

/*
#cgo LDFLAGS: -lOpenCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Device describes one OpenCL GPU device.
type Device struct {
	Platform C.cl_platform_id
	ID       C.cl_device_id
	Name     string
}

// ProbeDevices lists GPU devices across all platforms.
func ProbeDevices() ([]Device, error) {
	var nplat C.cl_uint
	if rc := C.clGetPlatformIDs(0, nil, &nplat); rc != C.CL_SUCCESS {
		return nil, fmt.Errorf("clGetPlatformIDs: %d", int(rc))
	}
	if nplat == 0 {
		return nil, fmt.Errorf("no OpenCL platforms")
	}
	plats := make([]C.cl_platform_id, nplat)
	if rc := C.clGetPlatformIDs(nplat, &plats[0], nil); rc != C.CL_SUCCESS {
		return nil, fmt.Errorf("clGetPlatformIDs(2): %d", int(rc))
	}
	var devs []Device
	for _, p := range plats {
		var ndev C.cl_uint
		if rc := C.clGetDeviceIDs(p, C.CL_DEVICE_TYPE_GPU, 0, nil, &ndev); rc != C.CL_SUCCESS || ndev == 0 {
			continue
		}
		ids := make([]C.cl_device_id, ndev)
		if rc := C.clGetDeviceIDs(p, C.CL_DEVICE_TYPE_GPU, ndev, &ids[0], nil); rc != C.CL_SUCCESS {
			continue
		}
		for _, d := range ids {
			devs = append(devs, Device{Platform: p, ID: d, Name: deviceName(d)})
		}
	}
	return devs, nil
}

func deviceName(d C.cl_device_id) string {
	var sz C.size_t
	if rc := C.clGetDeviceInfo(d, C.CL_DEVICE_NAME, 0, nil, &sz); rc != C.CL_SUCCESS || sz == 0 {
		return "?"
	}
	buf := make([]byte, sz)
	if rc := C.clGetDeviceInfo(d, C.CL_DEVICE_NAME, sz, unsafe.Pointer(&buf[0]), nil); rc != C.CL_SUCCESS {
		return "?"
	}
	return trimNul(buf)
}

func trimNul(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}

// Context wraps device, context, queue, program, kernel.
type Context struct {
	Dev     Device
	ctx     C.cl_context
	queue   C.cl_command_queue
	program C.cl_program
	kernel  C.cl_kernel
}

// NewContext creates a context+queue on the given device.
func NewContext(dev Device) (*Context, error) {
	c := &Context{Dev: dev}
	var rc C.cl_int
	c.ctx = C.clCreateContext(nil, 1, &dev.ID, nil, nil, &rc)
	if rc != C.CL_SUCCESS {
		return nil, fmt.Errorf("clCreateContext: %d", int(rc))
	}
	c.queue = C.clCreateCommandQueue(c.ctx, dev.ID, 0, &rc)
	if rc != C.CL_SUCCESS {
		C.clReleaseContext(c.ctx)
		return nil, fmt.Errorf("clCreateCommandQueue: %d", int(rc))
	}
	return c, nil
}

// Release frees all OpenCL resources.
func (c *Context) Release() {
	if c.kernel != nil {
		C.clReleaseKernel(c.kernel)
		c.kernel = nil
	}
	if c.program != nil {
		C.clReleaseProgram(c.program)
		c.program = nil
	}
	if c.queue != nil {
		C.clReleaseCommandQueue(c.queue)
		c.queue = nil
	}
	if c.ctx != nil {
		C.clReleaseContext(c.ctx)
		c.ctx = nil
	}
}

// BuildProgram compiles source for this context's device; options are
// compiler flags (e.g. "-D BEAM_DOUBLE"). Build log included on failure.
func (c *Context) BuildProgram(src, options string) error {
	csrc := C.CString(src)
	defer C.free(unsafe.Pointer(csrc))
	lengths := []C.size_t{C.size_t(len(src))}
	cprog := C.clCreateProgramWithSource(c.ctx, 1,
		(**C.char)(unsafe.Pointer(&csrc)), &lengths[0], nil)
	if cprog == nil {
		return fmt.Errorf("clCreateProgramWithSource failed")
	}
	copts := C.CString(options)
	defer C.free(unsafe.Pointer(copts))
	rc := C.clBuildProgram(cprog, 1, &c.Dev.ID, copts, nil, nil)
	if rc != C.CL_SUCCESS {
		log := c.buildLog(cprog)
		C.clReleaseProgram(cprog)
		return fmt.Errorf("clBuildProgram: %d\n%s", int(rc), log)
	}
	c.program = cprog
	return nil
}

func (c *Context) buildLog(p C.cl_program) string {
	var sz C.size_t
	if rc := C.clGetProgramBuildInfo(p, c.Dev.ID, C.CL_PROGRAM_BUILD_LOG, 0, nil, &sz); rc != C.CL_SUCCESS || sz == 0 {
		return ""
	}
	buf := make([]byte, sz)
	if rc := C.clGetProgramBuildInfo(p, c.Dev.ID, C.CL_PROGRAM_BUILD_LOG, sz, unsafe.Pointer(&buf[0]), nil); rc != C.CL_SUCCESS {
		return ""
	}
	return trimNul(buf)
}

// GetKernel instantiates a kernel by name from the built program.
func (c *Context) GetKernel(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	var rc C.cl_int
	k := C.clCreateKernel(c.program, cname, &rc)
	if rc != C.CL_SUCCESS {
		return fmt.Errorf("clCreateKernel(%s): %d", name, int(rc))
	}
	c.kernel = k
	return nil
}

// Buffer wraps a cl_mem object.
type Buffer struct {
	mem  C.cl_mem
	size int
}

// CreateBufferRO allocates a read-only buffer and copies data into it.
func (c *Context) CreateBufferRO(size int, data unsafe.Pointer) (*Buffer, error) {
	var rc C.cl_int
	m := C.clCreateBuffer(c.ctx, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(size), data, &rc)
	if rc != C.CL_SUCCESS {
		return nil, fmt.Errorf("clCreateBuffer(RO,%d): %d", size, int(rc))
	}
	return &Buffer{mem: m, size: size}, nil
}

// CreateBufferWO allocates a write-only buffer (kernel output).
func (c *Context) CreateBufferWO(size int) (*Buffer, error) {
	var rc C.cl_int
	m := C.clCreateBuffer(c.ctx, C.CL_MEM_WRITE_ONLY,
		C.size_t(size), nil, &rc)
	if rc != C.CL_SUCCESS {
		return nil, fmt.Errorf("clCreateBuffer(WO,%d): %d", size, int(rc))
	}
	return &Buffer{mem: m, size: size}, nil
}

// Release frees the buffer.
func (b *Buffer) Release() {
	if b != nil && b.mem != nil {
		C.clReleaseMemObject(b.mem)
		b.mem = nil
	}
}

// SetArg binds a buffer to kernel argument idx.
func (c *Context) SetArg(idx int, b *Buffer) error {
	rc := C.clSetKernelArg(c.kernel, C.cl_uint(idx), C.size_t(unsafe.Sizeof(b.mem)),
		unsafe.Pointer(&b.mem))
	if rc != C.CL_SUCCESS {
		return fmt.Errorf("clSetKernelArg(%d): %d", idx, int(rc))
	}
	return nil
}

// Run2D enqueues the kernel over a 2D global size.
func (c *Context) Run2D(gx, gy int) error {
	g := [2]C.size_t{C.size_t(gx), C.size_t(gy)}
	rc := C.clEnqueueNDRangeKernel(c.queue, c.kernel, 2, nil, &g[0], nil, 0, nil, nil)
	if rc != C.CL_SUCCESS {
		return fmt.Errorf("clEnqueueNDRangeKernel: %d", int(rc))
	}
	return nil
}

// Read copies a buffer back to host memory and waits for completion.
func (c *Context) Read(b *Buffer, dst []byte) error {
	rc := C.clEnqueueReadBuffer(c.queue, b.mem, C.CL_TRUE, 0, C.size_t(b.size),
		unsafe.Pointer(&dst[0]), 0, nil, nil)
	if rc != C.CL_SUCCESS {
		return fmt.Errorf("clEnqueueReadBuffer: %d", int(rc))
	}
	return nil
}

// HasFP64 reports whether the device supports fp64 (cl_khr_fp64 / core).
func HasFP64(d Device) bool {
	const clDeviceDoubleFPConfig = 0x1032
	var cfg C.cl_device_fp_config
	rc := C.clGetDeviceInfo(d.ID, C.cl_device_info(clDeviceDoubleFPConfig),
		C.size_t(unsafe.Sizeof(cfg)), unsafe.Pointer(&cfg), nil)
	return rc == C.CL_SUCCESS && cfg != 0
}
