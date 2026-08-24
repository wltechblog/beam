// beam-mcp runs the Beam MCP server over stdio.
package main

import (
	"os"

	"github.com/wltechblog/beam/mcp"
)

func main() {
	mcp.Serve(os.Stdin, os.Stdout)
}
