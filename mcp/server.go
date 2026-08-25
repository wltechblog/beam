// Package mcp implements a Model Context Protocol server over stdio for
// Beam, exposing scene rendering as tools agents can call directly:
//
//	beam_render  — full-quality render (scene JSON in, PNG path out)
//	beam_preview — fast 320px preview for the iterate loop
//	beam_validate — schema check + optional fit-check, no render
//	beam_schema  — full scene schema reference for agents
//
// Wire format: newline-delimited JSON-RPC 2.0 on stdin/stdout, matching the
// MCP 2025-03-26 protocol used by Gino (initialize / tools/list / tools/call).
package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/wltechblog/beam/gpu"
	"github.com/wltechblog/beam/render"
	"github.com/wltechblog/beam/scene"
)

// Server runs the MCP stdio server loop.
type Server struct {
	out io.Writer
	mu  sync.Mutex
}

// Serve reads JSON-RPC requests from in and writes responses to out. It
// blocks until EOF. Logging goes to stderr so the stdout channel stays
// protocol-clean.
func Serve(in io.Reader, out io.Writer) {
	log.SetFlags(0)
	log.SetPrefix("[beam-mcp] ")
	s := &Server{out: out}
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1024*1024), 32*1024*1024) // scenes can be large
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var req rpcRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			s.replyError(nil, -32700, "parse error: %v", err)
			continue
		}
		if req.ID == nil {
			// Notification (e.g. notifications/initialized) — no response.
			continue
		}
		s.dispatch(&req)
	}
}

func (s *Server) dispatch(req *rpcRequest) {
	switch req.Method {
	case "initialize":
		s.replyResult(req.ID, map[string]interface{}{
			"protocolVersion": "2025-03-26",
			"serverInfo": map[string]string{
				"name":    "beam-mcp",
				"version": "0.1.0",
			},
			"capabilities": map[string]interface{}{
				"tools": map[string]interface{}{},
			},
		})
	case "ping":
		s.replyResult(req.ID, map[string]interface{}{})
	case "tools/list":
		s.replyResult(req.ID, map[string]interface{}{"tools": toolsList()})
	case "tools/call":
		var p struct {
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			s.replyError(req.ID, -32602, "invalid params: %v", err)
			return
		}
		text, isErr := callTool(p.Name, p.Arguments)
		if isErr {
			s.replyResult(req.ID, map[string]interface{}{
				"content": []map[string]string{{"type": "text", "text": text}},
				"isError": true,
			})
		} else {
			s.replyResult(req.ID, map[string]interface{}{
				"content": []map[string]string{{"type": "text", "text": text}},
			})
		}
	default:
		s.replyError(req.ID, -32601, "method not found: %s", req.Method)
	}
}

// callTool routes to a tool implementation. Returns the text result and
// whether it is a tool-level error (reported via isError, not a protocol
// error — agents need to read these to self-correct).
func callTool(name string, args map[string]interface{}) (string, bool) {
	switch name {
	case "beam_render":
		return toolRender(args, false)
	case "beam_preview":
		return toolRender(args, true)
	case "beam_validate":
		return toolValidate(args)
	case "beam_schema":
		return toolSchema()
	default:
		return fmt.Sprintf("unknown tool %q", name), true
	}
}

/*** tool implementations ***/

// renderArgs is the shared input shape for beam_render / beam_preview.
type renderArgs struct {
	Scene     json.RawMessage `json:"scene"`     // scene JSON (preferred)
	SceneFile string          `json:"sceneFile"` // path to .json on disk
	Output    string          `json:"output"`    // output PNG path (optional)
	Dir       string          `json:"dir"`       // base dir for mesh/font resolution
	GPU       bool            `json:"gpu"`       // render on GPU (OpenCL)
}

func decodeRenderArgs(args map[string]interface{}) (renderArgs, []byte, error) {
	b, _ := json.Marshal(args) // round-trip to use struct tags
	var ra renderArgs
	if err := json.Unmarshal(b, &ra); err != nil {
		return ra, nil, err
	}
	if len(ra.Scene) == 0 && ra.SceneFile == "" {
		return ra, nil, fmt.Errorf("provide scene (JSON) or sceneFile (path)")
	}
	var raw []byte
	if len(ra.Scene) > 0 {
		raw = ra.Scene
	} else {
		var err error
		raw, err = os.ReadFile(ra.SceneFile)
		if err != nil {
			return ra, nil, fmt.Errorf("read sceneFile: %w", err)
		}
	}
	return ra, raw, nil
}

func toolRender(args map[string]interface{}, preview bool) (string, bool) {
	ra, raw, err := decodeRenderArgs(args)
	if err != nil {
		return err.Error(), true
	}

	s, err := scene.Parse(raw)
	if err != nil {
		return "scene invalid — call beam_validate for the full error list: " + err.Error(), true
	}

	// renderArgs round-trip done; build render options
	if ra.Dir == "" && ra.SceneFile != "" {
		// default resolution dir: alongside the scene file
		dir := ra.SceneFile
		idx := strings.LastIndex(dir, string(os.PathSeparator))
		if idx < 0 {
			ra.Dir = "."
		} else {
			ra.Dir = dir[:idx]
		}
	}

	opts := render.Options{
		Preview: preview,
		Quiet:   true,
		Output:  ra.Output,
		Dir:     ra.Dir,
		GPU:     ra.GPU,
	}
	if ra.GPU {
		render.SetGPUHook(gpu.Hook)
	}
	start := time.Now()
	path, err := render.RenderScene(s, opts)
	if err != nil {
		return err.Error(), true
	}

	result := map[string]interface{}{
		"output":  path,
		"width":   s.Render.Width,
		"height":  s.Render.Height,
		"seconds": time.Since(start).Seconds(),
		"preview": preview,
	}
	if preview {
		result["note"] = "preview render — 320px wide, 1 sample. Verify composition, then call beam_render for full size."
	}
	b, _ := json.Marshal(result)
	return string(b), false
}

func toolValidate(args map[string]interface{}) (string, bool) {
	var a struct {
		Scene     json.RawMessage `json:"scene"`
		SceneFile string          `json:"sceneFile"`
		Dir       string          `json:"dir"`
		FitCheck  bool            `json:"fitCheck"`
	}
	b, _ := json.Marshal(args)
	if err := json.Unmarshal(b, &a); err != nil {
		return err.Error(), true
	}
	if len(a.Scene) == 0 && a.SceneFile == "" {
		return "provide scene (JSON) or sceneFile (path)", true
	}

	var raw []byte
	if len(a.Scene) > 0 {
		raw = a.Scene
	} else {
		var err error
		raw, err = os.ReadFile(a.SceneFile)
		if err != nil {
			return "read sceneFile: " + err.Error(), true
		}
	}

	s, err := scene.Parse(raw)
	if err != nil {
		// Validation failure is a successful validation call — the agent
		// gets the structured error list to fix, not a broken tool.
		out := map[string]interface{}{"valid": false, "errors": errList(err)}
		b, _ := json.Marshal(out)
		return string(b), false
	}

	out := map[string]interface{}{
		"valid":   true,
		"objects": len(s.Objects),
		"width":   s.Render.Width,
		"height":  s.Render.Height,
	}

	if a.FitCheck {
		dir := a.Dir
		if dir == "" && a.SceneFile != "" {
			idx := strings.LastIndex(a.SceneFile, string(os.PathSeparator))
			if idx >= 0 {
				dir = a.SceneFile[:idx]
			}
		}
		if dir == "" {
			dir = "."
		}
		fit, ferr := render.ScreenFitCheck(s, render.NewFileResolver(dir), s.Render.Width, s.Render.Height)
		if ferr != nil {
			out["fitCheckError"] = ferr.Error()
		} else {
			out["fit"] = fit
		}
	}

	fb, _ := json.Marshal(out)
	return string(fb), false
}

// errList extracts ValidationErr entries if err is a ValidationErrors.
func errList(err error) []scene.ValidationErr {
	if ve, ok := err.(*scene.ValidationErrors); ok {
		return ve.Errs
	}
	return []scene.ValidationErr{{Msg: err.Error()}}
}

func toolSchema() (string, bool) {
	b, _ := json.MarshalIndent(map[string]interface{}{
		"description": "Beam scene schema v1. All vectors are [x,y.z] floats. Cameras default to pos [0,1,-5] looking at [0,0.8,0] — -Z faces the camera, +X is screen-left.",
		"scene": map[string]interface{}{
			"version":     "1 (required)",
			"camera":      "pos [x,y,z], lookAt [x,y,z], fov degrees (horizontal), aperture, focusDist (0 = pinhole / autofocus)",
			"materials":   "map name -> {type: matte|metal|glass|emissive, color [r,g,b] 0..1, pattern, roughness, ior (glass), power (emissive)}",
			"pattern":     "{type: solid|checker|gradient, colors [[r,g,b],[r,g,b]], scale (checker), axis x|y|z + from/to (gradient)}",
			"objects":     "list of {type: sphere|plane|box|mesh|text, material, transform {translate, rotate(deg XYZ), scale}, ...shape fields}",
			"sphere":      "center [x,y,z], radius",
			"plane":       "point [x,y,z], normal [x,y,z] (normalized not required)",
			"box":         "min [x,y,z], max [x,y,z]",
			"mesh":        "src: path to OBJ/STL/GLB/GLTF relative to scene dir",
			"text":        "text, size (em height world units), depth (extrusion), font (regular|bold|serif|mono|path.ttf), spacing (em)",
			"lights":      "list of {type: point|directional|spot, pos/dir, color, intensity, angle (spot half-angle)}",
			"environment": "{color [r,g,b], ambient, sky: [zenith, horizon] colors for gradient background}",
			"render":      "{width, height, samples (NxN AA grid, max 16), maxDepth (recursion, max 20), seed, exposure, output}",
		},
		"tips": []string{
			"materials are referenced by name from objects",
			"use beam_preview first, inspect, adjust, then beam_render",
			"glass needs ior (1.5 typical); emissive needs power",
			"fitCheck on beam_validate reports clipping before rendering",
		},
	}, "", "  ")
	return string(b), false
}

/*** tool descriptors ***/

func toolsList() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"name":        "beam_render",
			"description": "Render a Beam scene to PNG at full quality. Input is the complete scene JSON (get the shape via beam_schema). Returns output path, dimensions, and render seconds. Renders are deterministic for a given scene+seed. gpu:true renders on OpenCL GPU (~10x faster, fp32).",
			"inputSchema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"scene":     map[string]interface{}{"type": "object", "description": "Scene JSON object (preferred over sceneFile)"},
					"sceneFile": map[string]interface{}{"type": "string", "description": "Path to a scene .json on disk"},
					"output":    map[string]interface{}{"type": "string", "description": "Output PNG path (default beam-out.png)"},
					"dir":       map[string]interface{}{"type": "string", "description": "Base dir for mesh/font resolution"},
				},
			},
		},
		{
			"name":        "beam_preview",
			"description": "Fast low-res preview render (320px, 1 sample) of a Beam scene — for the iterate loop: preview, inspect visually, adjust, then beam_render. Input matches beam_render. gpu:true renders on OpenCL GPU.",
			"inputSchema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"scene":     map[string]interface{}{"type": "object", "description": "Scene JSON object"},
					"sceneFile": map[string]interface{}{"type": "string", "description": "Path to a scene .json on disk"},
					"output":    map[string]interface{}{"type": "string", "description": "Output PNG path"},
					"dir":       map[string]interface{}{"type": "string", "description": "Base dir for mesh/font resolution"},
				},
			},
		},
		{
			"name":        "beam_validate",
			"description": "Validate scene JSON without rendering. Returns valid=true or the full error list with JSON paths (fix all errors at once). With fitCheck=true, also projects object bounds to screen space and reports clipping/off-screen objects.",
			"inputSchema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"scene":     map[string]interface{}{"type": "object", "description": "Scene JSON object"},
					"sceneFile": map[string]interface{}{"type": "string", "description": "Path to a scene .json on disk"},
					"dir":       map[string]interface{}{"type": "string", "description": "Base dir for mesh/font resolution"},
					"fitCheck":  map[string]interface{}{"type": "boolean", "description": "Also run screen-fit check (default false)"},
				},
			},
		},
		{
			"name":        "beam_schema",
			"description": "Full reference for the Beam scene JSON schema: object types, materials, patterns, lights, camera, render options, conventions. Call this first when composing a scene.",
			"inputSchema": map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		},
	}
}

/*** JSON-RPC plumbing ***/

type rpcRequest struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string           `json:"method"`
	Params  json.RawMessage  `json:"params,omitempty"`
}

func (s *Server) replyResult(id *json.RawMessage, result interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      id,
		"result":  result,
	})
	if err != nil {
		log.Printf("mcp: marshal result: %v", err)
		return
	}
	fmt.Fprintln(s.out, string(b))
}

func (s *Server) replyError(id *json.RawMessage, code int, format string, a ...interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      id,
		"error": map[string]interface{}{
			"code":    code,
			"message": fmt.Sprintf(format, a...),
		},
	})
	if err != nil {
		return
	}
	fmt.Fprintln(s.out, string(b))
}
