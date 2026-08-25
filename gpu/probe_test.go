package gpu

import "testing"

func TestProbeDevices(t *testing.T) {
	devs, err := ProbeDevices()
	if err != nil {
		t.Skipf("no OpenCL: %v", err)
	}
	t.Logf("devices: %+v", devs)
	if len(devs) == 0 {
		t.Skip("no GPU devices")
	}
	ctx, err := NewContext(devs[0])
	if err != nil {
		t.Fatalf("context: %v", err)
	}
	defer ctx.Release()
}
