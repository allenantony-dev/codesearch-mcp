package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
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
	if len(os.Args) < 2 || (os.Args[1] != "mcp" && len(os.Args) < 3) {
		log.Fatal("Usage: go run . index <repository-path> | search|keyword|hybrid|ask <query> | eval <questions.json> [keyword|hybrid] | mcp")
	}

	ctx := context.Background()
	// A pool replaces dead connections, so the MCP server recovers when Postgres comes back.
	conn, err := pgxpool.New(ctx, os.Getenv("DB_URL"))
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	switch os.Args[1] {
	case "index":
		index(ctx, conn, os.Args[2])
	case "search", "keyword", "hybrid":
		results, err := find(ctx, conn, os.Args[2], os.Args[1], 5)
		if err != nil {
			log.Fatal(err)
		}
		for _, r := range results {
			fmt.Printf("%s/%s:%d-%d (score %.3f)\n%s\n\n", r.repo, r.path, r.start, r.end, r.score, r.code)
		}
	case "ask":
		ask(ctx, conn, os.Args[2])
	case "mcp":
		serve(ctx, conn)
	case "eval":
		mode := "search"
		if len(os.Args) > 3 {
			mode = os.Args[3]
		}
		eval(ctx, conn, os.Args[2], mode)
	default:
		log.Fatalf("unknown subcommand %q", os.Args[1])
	}
}

func index(ctx context.Context, conn *pgxpool.Pool, path string) {
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

// find returns the top limit chunks for a mode: search (vector), keyword or hybrid.
func find(ctx context.Context, conn *pgxpool.Pool, query, mode string, limit int) ([]Result, error) {
	switch mode {
	case "keyword":
		return search(ctx, conn, query, true, limit)
	case "hybrid":
		return hybrid(ctx, conn, query, limit)
	default:
		return search(ctx, conn, query, false, limit)
	}
}

// hybrid fuses the top 20 of vector and keyword search with Reciprocal Rank
// Fusion: each list adds 1/(60+rank) to a chunk, so only positions matter and
// the two incomparable score scales never meet.
func hybrid(ctx context.Context, conn *pgxpool.Pool, query string, limit int) ([]Result, error) {
	type key struct {
		repo, path string
		start      int
	}
	vector, err := search(ctx, conn, query, false, max(20, limit))
	if err != nil {
		return nil, err
	}
	keyword, err := search(ctx, conn, query, true, max(20, limit))
	if err != nil {
		return nil, err
	}
	fused := map[key]Result{}
	for _, list := range [][]Result{vector, keyword} {
		for i, r := range list {
			k := key{r.repo, r.path, r.start}
			r.score = fused[k].score + 1/float64(60+i+1)
			fused[k] = r
		}
	}

	results := slices.Collect(maps.Values(fused))
	// Ties are common (one list each at the same rank), so break them by
	// location to keep runs repeatable despite random map order.
	slices.SortFunc(results, func(a, b Result) int {
		return cmp.Or(
			cmp.Compare(b.score, a.score),
			cmp.Compare(a.repo, b.repo),
			cmp.Compare(a.path, b.path),
			cmp.Compare(a.start, b.start),
		)
	})
	return results[:min(limit, len(results))], nil
}

// search returns the top limit chunks. Vector search scores by cosine distance
// (lower is closer); keyword search scores by ts_rank (higher is better).
func search(ctx context.Context, conn *pgxpool.Pool, query string, keyword bool, limit int) ([]Result, error) {
	sql := `SELECT repo, path, start_line, end_line, code, embedding <=> $1::real[]::vector AS score
		 FROM code_embeddings
		 ORDER BY score
		 LIMIT $2`
	var arg any = query
	if keyword {
		sql = `SELECT repo, path, start_line, end_line, code, ts_rank(tsv, q) AS score
		 FROM code_embeddings, websearch_to_tsquery('simple', $1) q
		 WHERE tsv @@ q
		 ORDER BY score DESC
		 LIMIT $2`
	} else {
		emb, err := embed([]string{query})
		if err != nil {
			return nil, err
		}
		arg = emb[0]
	}
	rows, err := conn.Query(ctx, sql, arg, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Result
	for rows.Next() {
		var r Result
		if err := rows.Scan(&r.repo, &r.path, &r.start, &r.end, &r.code, &r.score); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, rows.Err()
}

// eval runs each question and prints the rank of the first chunk that contains
// the expected line (- if none in the top 10), then recall@3, @5 and @10, and
// recall within a fixed number of returned lines. The line budgets compare
// chunkings fairly: top 5 of big chunks returns more code than top 5 of small.
func eval(ctx context.Context, conn *pgxpool.Pool, file, mode string) {
	data, err := os.ReadFile(file)
	if err != nil {
		log.Fatal(err)
	}
	var questions []struct {
		Q      string `json:"q"`
		Expect string `json:"expect"`
		Line   int    `json:"line"`
	}
	if err := json.Unmarshal(data, &questions); err != nil {
		log.Fatal(err)
	}

	budgets := []int{125, 250}
	var hit3, hit5, hit10 int
	hitBudget := make([]int, len(budgets))
	for _, q := range questions {
		rank := "-"
		lines := 0
		results, err := find(ctx, conn, q.Q, mode, 30)
		if err != nil {
			log.Fatal(err)
		}
		for i, r := range results {
			lines += r.end - r.start + 1
			if r.repo+"/"+r.path != q.Expect || q.Line < r.start || r.end < q.Line {
				continue
			}
			if i < 10 {
				rank = strconv.Itoa(i + 1)
				hit10++
			}
			if i < 5 {
				hit5++
			}
			if i < 3 {
				hit3++
			}
			// A chunk counts only if it fits in the budget without going over.
			for b, budget := range budgets {
				if lines <= budget {
					hitBudget[b]++
				}
			}
			break
		}
		fmt.Printf("%s  %s\n", rank, q.Q)
	}
	n := float64(len(questions))
	fmt.Printf("\nrecall@3: %.2f  recall@5: %.2f  recall@10: %.2f  within 125 lines: %.2f  within 250 lines: %.2f  (%d questions)\n",
		float64(hit3)/n, float64(hit5)/n, float64(hit10)/n, float64(hitBudget[0])/n, float64(hitBudget[1])/n, len(questions))
}

// ask answers a question from the top 5 hybrid chunks with a local model,
// ASK_MODEL or qwen2.5-coder:7b. The default is ~2x slower than llama3.2:3b,
// which refused an answer it was given and misread the token bucket.
// 5 chunks, not 10 as in serve: a model on CPU is slow on long prompts.
func ask(ctx context.Context, conn *pgxpool.Pool, question string) {
	results, err := find(ctx, conn, question, "hybrid", 5)
	if err != nil {
		log.Fatal(err)
	}
	var b strings.Builder
	b.WriteString("Answer the question using only the code chunks below. After each claim, cite the chunk it comes from as repo/path:start-end. If the chunks don't contain the answer, say you don't know.\n\n")
	for _, r := range results {
		fmt.Fprintf(&b, "%s/%s:%d-%d\n```go\n%s\n```\n\n", r.repo, r.path, r.start, r.end, r.code)
	}
	fmt.Fprintf(&b, "Question: %s\n", question)

	body, err := json.Marshal(map[string]any{
		"model":  cmp.Or(os.Getenv("ASK_MODEL"), "qwen2.5-coder:7b"),
		"prompt": b.String(),
		"stream": false,
		// Ollama's default context is smaller than 5 chunks and silently cuts the prompt.
		"options": map[string]any{"num_ctx": 8192},
	})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := http.Post("http://127.0.0.1:11434/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Fatalf("ollama: %s", resp.Status)
	}
	var out struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		log.Fatal(err)
	}
	fmt.Println(out.Response)
	fmt.Println("\nretrieved:")
	for _, r := range results {
		fmt.Printf("  [%s/%s:%d-%d]\n", r.repo, r.path, r.start, r.end)
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

// serve runs an MCP server on stdin/stdout with one tool, search_code, which
// returns the top 10 hybrid results. 10 is chosen on cost, not on eval deltas
// (too noisy at this corpus size): 10 chunks are about 4K tokens, ~2% of the
// client's context, while a miss costs another round trip and a partial answer.
func serve(ctx context.Context, conn *pgxpool.Pool) {
	type args struct {
		Query string `json:"query" jsonschema:"what to look for: a question about the code or an exact identifier"`
	}
	server := mcp.NewServer(&mcp.Implementation{Name: "codesearch", Version: "0.1.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "search_code",
		Description: "Search the indexed Go repos (taskq, ratelimit-gateway) by meaning and exact identifiers. Returns the 10 best chunks, each headed by repo/path:start-end; read the file around a result for more context.",
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in args) (*mcp.CallToolResult, any, error) {
		results, err := find(ctx, conn, in.Query, "hybrid", 10)
		if err != nil {
			// The SDK sends this to the client as a tool error; the server keeps running.
			return nil, nil, fmt.Errorf("search unavailable: %w", err)
		}

		var b strings.Builder
		for _, r := range results {
			fmt.Fprintf(&b, "%s/%s:%d-%d\n```go\n%s\n```\n\n", r.repo, r.path, r.start, r.end, r.code)
		}
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: b.String()}}}, nil, nil
	})

	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}
