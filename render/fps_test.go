package render

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"testing"
)

// TestTimelineHonorsSceneFPS pins the CLI contract: with no -fps override,
// the MP4 is encoded at the scene's animation.fps. Regression: the -fps
// flag default was 30, silently re-timing 24fps scenes 25% fast.
func TestTimelineHonorsSceneFPS(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("no ffmpeg")
	}
	sc := mustParse(t, animTestScene)
	sc.Animation.FPS = 12 // distinct from every default
	sc.Animation.Duration = 0.5

	out := t.TempDir() + "/fps-scene.mp4"
	p, err := RenderTimeline(sc, Options{Output: out, Quiet: true}, 0) // 0 = no override
	if err != nil {
		t.Fatal(err)
	}

	probeOut, err := exec.Command("ffprobe", "-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=r_frame_rate,nb_frames",
		"-show_entries", "format=duration",
		"-of", "json", p).Output()
	if err != nil {
		t.Fatalf("ffprobe: %v", err)
	}
	var probe struct {
		Streams []struct {
			RFrameRate string `json:"r_frame_rate"`
			NbFrames   string `json:"nb_frames"`
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	if err := json.Unmarshal(probeOut, &probe); err != nil {
		t.Fatalf("parse ffprobe json: %v\n%s", err, probeOut)
	}
	if len(probe.Streams) == 0 {
		t.Fatal("no video stream")
	}
	var num, den int
	if _, err := fmt.Sscanf(probe.Streams[0].RFrameRate, "%d/%d", &num, &den); err != nil || den == 0 {
		t.Fatalf("r_frame_rate %q: %v", probe.Streams[0].RFrameRate, err)
	}
	fps := float64(num) / float64(den)
	if fps != 12 {
		t.Fatalf("encoded %.0f fps, want the scene's 12", fps)
	}
	frames, _ := strconv.Atoi(probe.Streams[0].NbFrames)
	if frames != 6 {
		t.Fatalf("encoded %d frames, want duration*fps = 6", frames)
	}
	dur, _ := strconv.ParseFloat(probe.Format.Duration, 64)
	if dur < 0.45 || dur > 0.55 {
		t.Fatalf("duration %.2fs, want ~0.5s", dur)
	}
}
