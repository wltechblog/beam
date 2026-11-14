package asset

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/qmuntal/gltf"
	"github.com/wltechblog/beam/geom"
)

// ModelPart is one material group of an imported model: a mesh plus the
// glTF material it should render with. glTF primitives each reference at
// most one material; one part is emitted per primitive (transformed), so
// beam's one-material-per-object model stays intact. MatIdx is the glTF
// materials[] index, -1 when the primitive set none.
type ModelPart struct {
	Mesh     *geom.TriangleMesh
	MatIdx   int
	MatName  string
	PartName string
}

// LoadedModel is the result of a glTF/GLB import: transformed parts plus
// the document's materials and decoded images for the compile step.
type LoadedModel struct {
	Parts      []ModelPart
	Materials  []*gltf.Material
	Images     []image.Image // entry i matches doc.Images[i]; nil = failed/absent
	ImageNames []string

	// Src is the path the model was loaded from — texture dedupe
	// identity at compile time (two models' img0 are different images).
	Src string
}

// LoadModel imports a glTF 2.0 / GLB file: geometry (POSITION, NORMAL,
// TEXCOORD_0, indices), the full node transform hierarchy (matrix + TRS,
// quaternions), materials, and textures (embedded GLB buffer views, data
// URLs, and external files relative to the model). Skinning/morph/
// animations are ignored — static display meshes.
func LoadModel(path string) (*LoadedModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	doc := &gltf.Document{}
	if err := gltf.NewDecoder(f).Decode(doc); err != nil {
		return nil, fmt.Errorf("gltf decode: %w", err)
	}

	lm := &LoadedModel{Materials: doc.Materials, Src: path}
	lm.decodeImages(doc, path)

	emitted := 0
	sceneIdx := 0
	if doc.Scene != nil {
		sceneIdx = *doc.Scene
	}
	if len(doc.Scenes) > 0 && sceneIdx < len(doc.Scenes) && len(doc.Scenes[sceneIdx].Nodes) > 0 {
		for _, ni := range doc.Scenes[sceneIdx].Nodes {
			emitted += lm.walkNode(doc, ni, geom.Identity())
		}
	} else {
		// Scene-less document: every mesh flat at identity.
		for mi := range doc.Meshes {
			for pi := range doc.Meshes[mi].Primitives {
				emitted += lm.appendPrimitive(doc, doc.Meshes[mi].Primitives[pi],
					fmt.Sprintf("mesh%d.%d", mi, pi), geom.Identity())
			}
		}
	}
	if emitted == 0 {
		return nil, fmt.Errorf("%s: no meshes found", path)
	}
	return lm, nil
}

// walkNode accumulates the full 4x4 world transform down the hierarchy
// (glTF nodes may carry a matrix OR TRS; TRS with quaternion rotation
// composes as T*R*S) and emits one part per mesh primitive.
func (lm *LoadedModel) walkNode(doc *gltf.Document, ni int, parent geom.Mat4) int {
	if ni < 0 || ni >= len(doc.Nodes) {
		return 0
	}
	n := doc.Nodes[ni]
	local := nodeMatrix(n)
	world := parent.Mul(local)

	out := 0
	if n.Mesh != nil && *n.Mesh < len(doc.Meshes) {
		mesh := doc.Meshes[*n.Mesh]
		for pi := range mesh.Primitives {
			name := fmt.Sprintf("node%d/%s#%d", ni, mesh.Name, pi)
			out += lm.appendPrimitive(doc, mesh.Primitives[pi], name, world)
		}
	}
	for _, ci := range n.Children {
		out += lm.walkNode(doc, ci, world)
	}
	return out
}

// nodeMatrix builds the node's local transform: matrix when genuinely
// non-identity, else T * R(quat) * S. The qmuntal decoder fills Matrix
// with the IDENTITY default (not zeros), so the zero-array check is
// useless — compare against identity instead.
func nodeMatrix(n *gltf.Node) geom.Mat4 {
	if n.Matrix != geom.Identity() {
		// glTF matrices are COLUMN-major; geom.Mat4 is row-major.
		var m geom.Mat4
		for col := 0; col < 4; col++ {
			for row := 0; row < 4; row++ {
				m[row*4+col] = n.Matrix[col*4+row]
				_ = row
			}
		}
		return m
	}
	t := n.Translation
	s := n.Scale
	qx, qy, qz, qw := n.Rotation[0], n.Rotation[1], n.Rotation[2], n.Rotation[3]

	// Rotation matrix from quaternion (row-major).
	xx, yy, zz := qx*qx, qy*qy, qz*qz
	xy, xz, yz := qx*qy, qx*qz, qy*qz
	wx, wy, wz := qw*qx, qw*qy, qw*qz
	r := geom.Mat4{
		1 - 2*(yy+zz), 2 * (xy - wz), 2 * (xz + wy), 0,
		2 * (xy + wz), 1 - 2*(xx+zz), 2 * (yz - wx), 0,
		2 * (xz - wy), 2 * (yz + wx), 1 - 2*(xx+yy), 0,
		0, 0, 0, 1,
	}
	return geom.Translate(t[0], t[1], t[2]).Mul(r).Mul(geom.Scale(s[0], s[1], s[2]))
}

// decodeImages decodes every image reference: embedded buffer views (GLB),
// data URLs, and external files relative to the model. Failures leave a
// nil entry — materials fall back to their factor colors.
func (lm *LoadedModel) decodeImages(doc *gltf.Document, modelPath string) {
	lm.Images = make([]image.Image, len(doc.Images))
	lm.ImageNames = make([]string, len(doc.Images))
	dir := filepath.Dir(modelPath)
	for i, img := range doc.Images {
		var data []byte
		switch {
		case img.BufferView != nil:
			if *img.BufferView < len(doc.BufferViews) {
				bv := doc.BufferViews[*img.BufferView]
				if int(bv.Buffer) < len(doc.Buffers) &&
					int(bv.ByteOffset)+int(bv.ByteLength) <= len(doc.Buffers[bv.Buffer].Data) {
					data = doc.Buffers[bv.Buffer].Data[bv.ByteOffset : bv.ByteOffset+bv.ByteLength]
				}
			}
		case strings.HasPrefix(img.URI, "data:"):
			if d, err := dataURLBytes(img.URI); err == nil {
				data = d
			}
		case img.URI != "":
			if d, err := os.ReadFile(filepath.Join(dir, img.URI)); err == nil {
				data = d
			}
		}
		name := img.Name
		if name == "" {
			name = fmt.Sprintf("img%d", i)
		}
		lm.ImageNames[i] = name
		if len(data) > 0 {
			if im, err := decodeImageBytes(data, img.MimeType); err == nil {
				lm.Images[i] = im
			}
		}
	}
}

func dataURLBytes(uri string) ([]byte, error) {
	const marker = ";base64,"
	i := strings.Index(uri, marker)
	if i < 0 {
		return nil, fmt.Errorf("data URL without base64 payload")
	}
	return base64.StdEncoding.DecodeString(uri[i+len(marker):])
}

func decodeImageBytes(data []byte, mime string) (image.Image, error) {
	m := strings.ToLower(mime)
	if strings.Contains(m, "png") || (m == "" && len(data) > 4 && data[0] == 0x89 && data[1] == 'P') {
		return png.Decode(bytes.NewReader(data))
	}
	if strings.Contains(m, "jpeg") || strings.Contains(m, "jpg") ||
		(len(data) > 2 && data[0] == 0xFF && data[1] == 0xD8) {
		return jpeg.Decode(bytes.NewReader(data))
	}
	// Last resort: try PNG then JPEG regardless of the declared mime.
	if im, err := png.Decode(bytes.NewReader(data)); err == nil {
		return im, nil
	}
	return jpeg.Decode(bytes.NewReader(data))
}

// appendPrimitive extracts one primitive into its own mesh part with the
// accumulated world transform applied to positions and normals (normals
// use the inverse-transpose upper 3x3, then renormalized).
func (lm *LoadedModel) appendPrimitive(doc *gltf.Document, prim *gltf.Primitive, partName string, world geom.Mat4) int {
	if _, ok := prim.Attributes["POSITION"]; !ok {
		return 0
	}
	m := &geom.TriangleMesh{}

	posRef := prim.Attributes["POSITION"]
	pv, err := viewOf(doc, posRef)
	if err != nil {
		return 0
	}
	count := doc.Accessors[posRef].Count
	for i := 0; i < count; i++ {
		v := world.Apply(geom.Vec3{
			X: pv.floatElem(i, 0),
			Y: pv.floatElem(i, 1),
			Z: pv.floatElem(i, 2),
		})
		m.Verts = append(m.Verts, v)
	}

	if nRef, ok := prim.Attributes["NORMAL"]; ok {
		if nv, err := viewOf(doc, nRef); err == nil && doc.Accessors[nRef].Count == count {
			inv := world.Inverse()
			for i := 0; i < count; i++ {
				n := inv.Transpose().ApplyDir(geom.Vec3{
					X: nv.floatElem(i, 0),
					Y: nv.floatElem(i, 1),
					Z: nv.floatElem(i, 2),
				})
				m.Norms = append(m.Norms, n.Norm())
			}
		}
	}

	if uvRef, ok := prim.Attributes["TEXCOORD_0"]; ok {
		if uv, err := viewOf(doc, uvRef); err == nil && doc.Accessors[uvRef].Count == count {
			for i := 0; i < count; i++ {
				m.UVs = append(m.UVs, geom.UV{
					U: uv.floatElem(i, 0),
					V: uv.floatElem(i, 1),
				})
			}
		}
	}

	if prim.Indices != nil {
		if iv, err := viewOf(doc, *prim.Indices); err == nil {
			n := doc.Accessors[*prim.Indices].Count
			for k := 2; k < n; k += 3 {
				a, b, c := int(iv.floatElem(k-2, 0)), int(iv.floatElem(k-1, 0)), int(iv.floatElem(k, 0))
				if a == b || b == c || a == c {
					continue // degenerate
				}
				m.Tris = append(m.Tris, [3]int32{int32(a), int32(b), int32(c)})
			}
		}
	} else if len(m.Verts) >= 3 {
		for k := 2; k < len(m.Verts); k += 3 {
			m.Tris = append(m.Tris, [3]int32{int32(k - 2), int32(k - 1), int32(k)})
		}
	}
	if len(m.Tris) == 0 {
		return 0
	}

	part := ModelPart{Mesh: m, PartName: partName, MatIdx: -1}
	if prim.Material != nil && *prim.Material >= 0 && *prim.Material < len(doc.Materials) {
		part.MatIdx = *prim.Material
		part.MatName = doc.Materials[*prim.Material].Name
	}
	lm.Parts = append(lm.Parts, part)
	return 1
}

// LoadGLTF is the legacy geometry-only entry (LoadMesh dispatch): all
// parts concatenated into one mesh. Textures/materials need the compile-
// time model path (scene object type "model").
func LoadGLTF(path string) (*geom.TriangleMesh, error) {
	lm, err := LoadModel(path)
	if err != nil {
		return nil, err
	}
	all := &geom.TriangleMesh{}
	for _, p := range lm.Parts {
		base := int32(len(all.Verts))
		all.Verts = append(all.Verts, p.Mesh.Verts...)
		all.Norms = append(all.Norms, p.Mesh.Norms...)
		all.UVs = append(all.UVs, p.Mesh.UVs...)
		for _, t := range p.Mesh.Tris {
			all.Tris = append(all.Tris, [3]int32{t[0] + base, t[1] + base, t[2] + base})
		}
	}
	if len(all.Tris) == 0 {
		return nil, fmt.Errorf("%s: no triangles", path)
	}
	return all, nil
}
