// beam renders JSON scene descriptions to PNG.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/wltechblog/beam/render"
)

func main() {
	preview := flag.Bool("preview", false, "fast low-res render (320px, 1 sample)")
	quiet := flag.Bool("q", false, "suppress progress output")
	out := flag.String("o", "", "output path override (default: <scene>.png)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: beam [options] scene.json\n\noptions:\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}

	start := time.Now()
	path, err := render.RenderFile(flag.Arg(0), render.Options{
		Preview: *preview,
		Quiet:   *quiet,
		Output:  *out,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "beam:", err)
		os.Exit(1)
	}
	fmt.Printf("%s (%.1fs)\n", path, time.Since(start).Seconds())
}
