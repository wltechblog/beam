package mcp

import (
	"bytes"
	"encoding/json"
	"io"
	"os/exec"
	"strings"
	"testing"
)

// pipe drives Serve over an in-memory pipe: write requests, read responses
// line by line.
type pipe struct {
	in  io.WriteCloser
	out io.ReadCloser
}

func newPipe(t *testing.T) *pipe {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	srv := &Server{out: outW}
	go func() {
		// serve directly on the pipes (mirrors Serve loop without scanner)
		sc := &lineScanner{r: inR}
		for sc.next() {
			var req rpcRequest
			if err := json.Unmarshal([]byte(sc.line), &req); err != nil {
				srv.replyError(nil, -32700, "parse error")
				continue
			}
			if req.ID == nil {
				continue
			}
			srv.dispatch(&req)
		}
	}()
	t.Cleanup(func() {
		inW.Close()
		outR.Close()
	})
	return &pipe{in: inW, out: outR}
}

// lineScanner is a minimal line reader for the test pipe.
type lineScanner struct {
	r    io.Reader
	buf  []byte
	line string
}

func (s *lineScanner) next() bool {
	for {
		i := bytes.IndexByte(s.buf, '\n')
		if i >= 0 {
			s.line = string(s.buf[:i])
			s.buf = s.buf[i+1:]
			return strings.TrimSpace(s.line) != ""
		}
		chunk := make([]byte, 4096)
		n, err := s.r.Read(chunk)
		s.buf = append(s.buf, chunk[:n]...)
		if err != nil {
			return false
		}
	}
}

func (p *pipe) send(t *testing.T, v interface{}) {
	t.Helper()
	b, _ := json.Marshal(v)
	if _, err := p.in.Write(append(b, '\n')); err != nil {
		t.Fatalf("write: %v", err)
	}
}

func (p *pipe) recv(t *testing.T) map[string]interface{} {
	t.Helper()
	var buf []byte
	for {
		i := bytes.IndexByte(buf, '\n')
		if i >= 0 {
			var m map[string]interface{}
			if err := json.Unmarshal(buf[:i], &m); err != nil {
				t.Fatalf("bad response line: %v", err)
			}
			return m
		}
		chunk := make([]byte, 4096)
		n, err := p.out.Read(chunk)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		buf = append(buf, chunk[:n]...)
	}
}

const goodScene = `{"version":1,"camera":{"pos":[0,1,-5],"lookAt":[0,0.8,0],"fov":50},` +
	`"materials":{"red":{"type":"matte","color":[0.8,0.1,0.1]}},` +
	`"objects":[{"type":"sphere","center":[0,0.5,0],"radius":0.5,"material":"red"}],` +
	`"render":{"width":64,"height":48,"output":"mcp-test-out.png"}}`

func TestInitialize(t *testing.T) {
	p := newPipe(t)
	p.send(t, map[string]interface{}{"jsonrpc": "2.0", "id": 1, "method": "initialize"})
	resp := p.recv(t)
	res := resp["result"].(map[string]interface{})
	si := res["serverInfo"].(map[string]interface{})
	if si["name"] != "beam-mcp" {
		t.Fatalf("serverInfo.name = %v", si["name"])
	}
	if res["protocolVersion"] != "2025-03-26" {
		t.Fatalf("protocolVersion = %v", res["protocolVersion"])
	}
}

func TestToolsList(t *testing.T) {
	p := newPipe(t)
	p.send(t, map[string]interface{}{"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
	resp := p.recv(t)
	tools := resp["result"].(map[string]interface{})["tools"].([]interface{})
	if len(tools) != 4 {
		t.Fatalf("expected 4 tools, got %d", len(tools))
	}
	want := map[string]bool{"beam_render": false, "beam_preview": false, "beam_validate": false, "beam_schema": false}
	for _, tv := range tools {
		name := tv.(map[string]interface{})["name"].(string)
		if _, ok := want[name]; !ok {
			t.Errorf("unexpected tool %s", name)
		}
		want[name] = true
	}
	for name, seen := range want {
		if !seen {
			t.Errorf("missing tool %s", name)
		}
	}
}

func TestCallToolUnknownMethod(t *testing.T) {
	p := newPipe(t)
	p.send(t, map[string]interface{}{"jsonrpc": "2.0", "id": 1, "method": "no/such"})
	resp := p.recv(t)
	errObj := resp["error"].(map[string]interface{})
	if code := errObj["code"].(float64); code != -32601 {
		t.Fatalf("expected -32601, got %v", code)
	}
}

func TestCallValidateInvalidScene(t *testing.T) {
	p := newPipe(t)
	// sphere with no center/radius: validation must list every problem
	p.send(t, map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]interface{}{
			"name":      "beam_validate",
			"arguments": map[string]interface{}{"scene": map[string]interface{}{"version": 1, "objects": []interface{}{map[string]interface{}{"type": "sphere"}}}},
		},
	})
	resp := p.recv(t)
	res := resp["result"].(map[string]interface{})
	if res["isError"] == true {
		t.Fatal("validation failure must be a successful call, not isError")
	}
	text := res["content"].([]interface{})[0].(map[string]interface{})["text"].(string)
	var out map[string]interface{}
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatalf("validate output not JSON: %v", err)
	}
	if out["valid"] != false {
		t.Fatal("expected valid=false")
	}
	errs := out["errors"].([]interface{})
	if len(errs) < 3 {
		t.Fatalf("expected camera+radius+center errors, got %d: %v", len(errs), errs)
	}
	// error entries must carry a path for agent self-correction
	for _, ev := range errs {
		e := ev.(map[string]interface{})
		if e["path"] == "" && strings.Contains(e["msg"].(string), "camera") {
			t.Fatalf("camera error missing path: %v", e)
		}
	}
}

func TestCallRenderRoundTrip(t *testing.T) {
	p := newPipe(t)
	p.send(t, map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]interface{}{
			"name":      "beam_render",
			"arguments": map[string]interface{}{"scene": json.RawMessage(goodScene)},
		},
	})
	resp := p.recv(t)
	res := resp["result"].(map[string]interface{})
	if res["isError"] == true {
		t.Fatalf("render failed: %v", res)
	}
	var out map[string]interface{}
	text := res["content"].([]interface{})[0].(map[string]interface{})["text"].(string)
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatalf("render output not JSON: %v (%s)", err, text)
	}
	if out["output"] != "mcp-test-out.png" {
		t.Fatalf("output = %v", out["output"])
	}
}

func TestCallSchema(t *testing.T) {
	p := newPipe(t)
	p.send(t, map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]interface{}{"name": "beam_schema", "arguments": map[string]interface{}{}},
	})
	resp := p.recv(t)
	text := resp["result"].(map[string]interface{})["content"].([]interface{})[0].(map[string]interface{})["text"].(string)
	for _, kw := range []string{"sphere", "materials", "lights", "environment", "render"} {
		if !strings.Contains(text, kw) {
			t.Errorf("schema missing %q", kw)
		}
	}
}

func TestCallRenderMissingScene(t *testing.T) {
	p := newPipe(t)
	p.send(t, map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]interface{}{
			"name":      "beam_render",
			"arguments": map[string]interface{}{},
		},
	})
	resp := p.recv(t)
	res := resp["result"].(map[string]interface{})
	if res["isError"] != true {
		t.Fatal("expected isError for missing scene")
	}
}

// animScene: goodScene + names + a minimal valid animation block.
const animScene = `{"version":1,"camera":{"pos":[0,1,-5],"lookAt":[0,0.8,0],"fov":50},` +
	`"materials":{"red":{"type":"matte","color":[0.8,0.1,0.1]}},` +
	`"objects":[{"type":"sphere","name":"ball","center":[0,0.5,0],"radius":0.5,"material":"red"}],` +
	`"lights":[{"type":"point","name":"key","pos":[3,5,-3]}],` +
	`"animation":{"duration":0.5,"fps":10,"tracks":[` +
	`{"target":"ball","property":"translate","keys":[{"t":0,"value":[0,0,0]},{"t":0.5,"value":[2,0,0]}]},` +
	`{"target":"key","property":"intensity","keys":[{"t":0,"value":[1]},{"t":0.5,"value":[0.3]}]}]},` +
	`"render":{"width":64,"height":48,"output":"mcp-test-anim.mp4"}}`

func TestCallRenderAnimate(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("no ffmpeg")
	}
	p := newPipe(t)
	p.send(t, map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]interface{}{
			"name": "beam_render",
			"arguments": map[string]interface{}{
				"scene":   json.RawMessage(animScene),
				"animate": true,
			},
		},
	})
	resp := p.recv(t)
	res := resp["result"].(map[string]interface{})
	if res["isError"] == true {
		t.Fatalf("animate render failed: %v", res)
	}
	text := res["content"].([]interface{})[0].(map[string]interface{})["text"].(string)
	if !strings.Contains(text, "mcp-test-anim.mp4") || !strings.Contains(text, "MP4") {
		t.Fatalf("unexpected animate output: %s", text)
	}
}

func TestCallRenderAnimateWithoutAnimation(t *testing.T) {
	p := newPipe(t)
	p.send(t, map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]interface{}{
			"name": "beam_render",
			"arguments": map[string]interface{}{
				"scene":   json.RawMessage(goodScene),
				"animate": true,
			},
		},
	})
	resp := p.recv(t)
	res := resp["result"].(map[string]interface{})
	if res["isError"] != true {
		t.Fatal("animate without an animation block must be an error")
	}
	text := res["content"].([]interface{})[0].(map[string]interface{})["text"].(string)
	if !strings.Contains(text, "no animation") {
		t.Fatalf("error should explain the missing animation block: %s", text)
	}
}
