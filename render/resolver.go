package render

import (
	"os"
	"path/filepath"

	"github.com/wltechblog/beam/asset"
	"github.com/wltechblog/beam/geom"
)

// FileResolver resolves mesh files and fonts relative to a base directory.
// It implements MeshResolver; the CLI and MCP layer wire it into Compile.
type FileResolver struct {
	dir string
}

// NewFileResolver creates a resolver rooted at dir.
func NewFileResolver(dir string) *FileResolver {
	if dir == "" {
		dir = "."
	}
	return &FileResolver{dir: dir}
}

func (fr *FileResolver) LoadMesh(src string) (*geom.TriangleMesh, error) {
	return asset.LoadMesh(fr.resolve(src))
}

func (fr *FileResolver) TextMesh(text, font string, size, depth, spacing float64) (*geom.TriangleMesh, error) {
	return asset.TextMesh(text, fr.font(font), size, depth, spacing)
}

func (fr *FileResolver) resolve(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(fr.dir, p)
}

// font maps scene font names to system fonts, falling back to sane defaults.
// A name ending in .ttf/.otf resolves relative to the scene dir (or absolute).
func (fr *FileResolver) font(name string) string {
	if name == "" {
		name = "regular"
	}
	if filepath.Ext(name) == ".ttf" || filepath.Ext(name) == ".otf" {
		return fr.resolve(name)
	}
	candidates := map[string][]string{
		"regular": {
			"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
			"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
		},
		"bold": {
			"/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
			"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
		},
		"serif": {
			"/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
			"/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
		},
		"mono": {
			"/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
			"/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
		},
	}
	for _, c := range candidates[name] {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	for _, c := range candidates["regular"] {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return name // let TextMesh report the error
}
