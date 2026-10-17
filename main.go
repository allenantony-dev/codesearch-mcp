package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
)

const chunkSize = 50

type Chunk struct {
	text  string
	path  string
	start int
	end   int
}

type Result struct {
	repo, path, code string
	start, end       int
	score            float64
}

func main() {
	if len(os.Args) < 3 {
		log.Fatal("Usage: go run . index <repository-path> | search|keyword <query> | eval <questions.json> [keyword]")
	}

	ctx := context.Background()
	conn, err := pgx.Connect(ctx, os.Getenv("DB_URL"))
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close(ctx)

	switch os.Args[1] {
	case "index":
		index(ctx, conn, os.Args[2])
	case "search", "keyword":
		for _, r := range search(ctx, conn, os.Args[2], os.Args[1] == "keyword") {
			fmt.Printf("%s/%s:%d-%d (score %.3f)\n%s\n\n", r.repo, r.path, r.start, r.end, r.score, r.code)
		}
	case "eval":
		eval(ctx, conn, os.Args[2], len(os.Args) > 3 && os.Args[3] == "keyword")
	default:
		log.Fatalf("unknown subcommand %q", os.Args[1])
	}
}

func index(ctx context.Context, conn *pgx.Conn, path string) {
	repoRoot := path
	abs, err := filepath.Abs(repoRoot)
	if err != nil {
		log.Fatal(err)
	}
	repo := filepath.Base(abs)

	var paths []string
	err = filepath.WalkDir(repoRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() && filepath.Ext(path) == ".go" {
			rel, err := filepath.Rel(repoRoot, path)
			if err != nil {
				return err
			}
			paths = append(paths, rel)

			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")

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
			}

			batch := &pgx.Batch{}
			batch.Queue(`DELETE FROM code_embeddings WHERE repo = $1 AND path = $2`, repo, rel)
			if len(chunks) > 0 {
				texts := make([]string, len(chunks))
				for k, c := range chunks {
					texts[k] = c.text
				}

				emb, err := embed(texts)
				if err != nil {
					return err
				}

				for k, c := range chunks {
					batch.Queue(`INSERT INTO code_embeddings (repo, code, path, start_line, end_line, embedding)
					VALUES ($1, $2, $3, $4, $5, $6::real[]::vector)`,
						repo, c.text, c.path, c.start, c.end, emb[k])
				}
			}
			return conn.SendBatch(ctx, batch).Close()
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	if len(paths) == 0 {
		log.Fatal("no .go files found. skipping cleanup")
	}
	if _, err := conn.Exec(ctx, `DELETE FROM code_embeddings WHERE repo = $1 AND path <> ALL($2)`, repo, paths); err != nil {
		log.Fatal(err)
	}
}

// search returns the top 5 chunks. Vector search scores by cosine distance
// (lower is closer); keyword search scores by ts_rank (higher is better).
func search(ctx context.Context, conn *pgx.Conn, query string, keyword bool) []Result {
	sql := `SELECT repo, path, start_line, end_line, code, embedding <=> $1::real[]::vector AS score
		 FROM code_embeddings
		 ORDER BY score
		 LIMIT 5`
	var arg any = query
	if keyword {
		sql = `SELECT repo, path, start_line, end_line, code, ts_rank(tsv, q) AS score
		 FROM code_embeddings, websearch_to_tsquery('simple', $1) q
		 WHERE tsv @@ q
		 ORDER BY score DESC
		 LIMIT 5`
	} else {
		emb, err := embed([]string{query})
		if err != nil {
			log.Fatal(err)
		}
		arg = emb[0]
	}
	rows, err := conn.Query(ctx, sql, arg)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	var results []Result
	for rows.Next() {
		var r Result
		if err := rows.Scan(&r.repo, &r.path, &r.start, &r.end, &r.code, &r.score); err != nil {
			log.Fatal(err)
		}
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
	return results
}

// eval runs each question and prints the rank of the expected file (- if not
// in the top 5), then recall@3 and recall@5.
func eval(ctx context.Context, conn *pgx.Conn, file string, keyword bool) {
	data, err := os.ReadFile(file)
	if err != nil {
		log.Fatal(err)
	}
	var questions []struct {
		Q      string `json:"q"`
		Expect string `json:"expect"`
	}
	if err := json.Unmarshal(data, &questions); err != nil {
		log.Fatal(err)
	}

	var hit3, hit5 int
	for _, q := range questions {
		rank := "-"
		for i, r := range search(ctx, conn, q.Q, keyword) {
			if r.repo+"/"+r.path == q.Expect {
				rank = strconv.Itoa(i + 1)
				if i < 3 {
					hit3++
				}
				hit5++
				break
			}
		}
		fmt.Printf("%s  %s\n", rank, q.Q)
	}
	n := float64(len(questions))
	fmt.Printf("\nrecall@3: %.2f  recall@5: %.2f  (%d questions)\n", float64(hit3)/n, float64(hit5)/n, len(questions))
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
