package gpu

import (
	"os"
	"testing"
)

func TestCompileKernel(t *testing.T) {
	devs, err := ProbeDevices()
	if err != nil || len(devs) == 0 {
		t.Skipf("no OpenCL GPU: %v", err)
	}
	for _, d := range devs {
		t.Logf("device: %s fp64=%v", d.Name, HasFP64(d))
	}
	dev := devs[0]
	ctx, err := NewContext(dev)
	if err != nil {
		t.Fatalf("context: %v", err)
	}
	defer ctx.Release()

	// fp32 compile
	err = ctx.BuildProgram(KernelSource(), "-cl-std=CL1.2")
	if err != nil {
		t.Errorf("fp32 build FAILED:\n%s", err)
	} else {
		t.Logf("fp32 build OK on %s", dev.Name)
	}
	if err := ctx.GetKernel("trace_pixel"); err != nil {
		t.Errorf("kernel create: %v", err)
	} else {
		t.Logf("trace_pixel kernel created")
	}
	if os.Getenv("KEEP") == "" {
		_ = os.WriteFile("/tmp/beam_kernel_fp32.log", []byte("fp32 ok"), 0644)
	}
}
