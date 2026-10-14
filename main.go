package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const chunkSize = 50

type Chunk struct {
	text  string
	path  string
	start int
	end   int
}

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
			var chunks []Chunk
			for i := 0; i < len(lines); i += chunkSize {
				var chunk Chunk
				end := min(i+chunkSize, len(lines))
				chunk.text = strings.Join(lines[i:end], "\n")
				if strings.TrimSpace(chunk.text) == "" {
					continue
				}
				chunk.path = rel
				chunk.start = i + 1
				chunk.end = end
				chunks = append(chunks, chunk)
				fmt.Printf("Chunk %d\n Chunk start: %d\n Chunk end: %d\n", j, i+1, end)
				j++
			}

			texts := make([]string, len(chunks))
			for k, c := range chunks {
				texts[k] = c.text
			}

			emb, err := embed(texts)
			fmt.Println("Embedding: ", emb)
		}

		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}

func embed(chunks []string) ([][]float32, error) {
	type embedRequest struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}
	body, err := json.Marshal(embedRequest{
		Model: "nomic-embed-text",
		Input: chunks,
	})
	if err != nil {
		return nil, err
	}

	resp, err := http.Post("http://127.0.0.1:11434/api/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama: %s", resp.Status)
	}

	var embedResponse struct {
		Embeddings [][]float32 `json:"embeddings"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&embedResponse); err != nil {
		return nil, err
	}

	if len(embedResponse.Embeddings) != len(chunks) {
		return nil, fmt.Errorf(
			"ollama: expected %d embeddings, got: %d",
			len(chunks),
			len(embedResponse.Embeddings),
		)
	}

	return embedResponse.Embeddings, nil
}
