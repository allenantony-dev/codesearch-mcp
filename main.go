package main

import (
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
)

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
		}

		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
