package main

import (
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const chunkSize = 50

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run . <repository-path>")
	}
	repoRoot := os.Args[1]

	i := 1
	err := filepath.WalkDir(repoRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() && filepath.Ext(path) == ".go" {
			rel, err := filepath.Rel(repoRoot, path)
			if err != nil {
				return err
			}
			fmt.Printf("Go file %d found: %s\n", i, rel)
			i++
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
			j := 1
			for i := 0; i < len(lines); i += chunkSize {
				end := min(i+chunkSize, len(lines))
				chunk := strings.Join(lines[i:end], "\n")
				if strings.TrimSpace(chunk) == "" {
					continue
				}
				fmt.Printf("Chunk %d\n Chunk start: %d\n Chunk end: %d\n", j, i+1, end)
				j++
			}
		}

		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
