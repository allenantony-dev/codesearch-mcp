# codesearch-mcp

Code search over Go repos, built on Postgres and a local embedding model, and served to Claude as an MCP tool. I built it to learn how retrieval works, and how you tell whether a change to it actually helped.

The basic idea is simple. Cut every file into chunks, turn each chunk into a vector of 768 numbers that captures what it's about, and answer a question by finding the chunks whose vectors are closest to the question's. The hard part is knowing whether it works. A search always returns *something*, and it always looks plausible.

So most of this project is measurement. Every change below was scored against a set of questions with known answers, and the honest summary is short: **combining vector search with keyword search is the one change that held up. It wins mostly the questions that name an identifier, never loses one, and on questions it had never seen lifted recall@10 from 0.33 to 0.73. Almost everything I tried after that was noise on a corpus too small to tell the difference.** The sections show how I found that out, including the times I fooled myself first.

Each section is one problem and what I did about it, in order, with a link to the commit.

**The problems, in order:**

1. [Pointing a result back at the code](#pointing-a-result-back-at-the-code)
2. [Re-indexing a repo](#re-indexing-a-repo)
3. [Knowing whether search works](#knowing-whether-search-works)
4. [Vector search can't find identifiers](#vector-search-cant-find-identifiers)
5. [Combining the two](#combining-the-two)
6. [The eval was too generous](#the-eval-was-too-generous)
7. [Chunking by function instead of by line count](#chunking-by-function-instead-of-by-line-count)
8. [Letting Claude use it](#letting-claude-use-it)
9. [Postgres goes down](#postgres-goes-down)
10. [How many results to return](#how-many-results-to-return)
11. [Answering instead of retrieving](#answering-instead-of-retrieving)

## How it works

```
 index:  .go files ──> 50-line chunks ──> Ollama (nomic-embed-text) ──> Postgres + pgvector
                                                                         │
 search: question ──┬─> embed ──> nearest vectors  (top 20) ──┐          │
                    └─> full-text match on the code (top 20) ─┴─> fuse ──> top N chunks
```

One binary, seven subcommands:

| Subcommand | What it does |
|---|---|
| `index <repo>` | Walks the repo, chunks and embeds every `.go` file, replaces that repo's rows |
| `search` / `keyword` / `hybrid <query>` | Vector only, keyword only, or both fused. Prints the top 5 |
| `eval <questions.json> [mode]` | Scores a mode against questions with known answers |
| `mcp` | Serves `search_code` (hybrid, top 10) over stdio for Claude Code |
| `ask <question>` | Answers in plain English from the top 5 chunks with a local model |

The corpus is two of my own repos, [taskq](https://github.com/allenantony-dev/taskq) and [ratelimit-gateway](https://github.com/allenantony-dev/ratelimit-gateway): 41 chunks in all. That number matters more than anything else in this README, and it comes up again at the end.

## The design, problem by problem

### Pointing a result back at the code

[`12e2c39`](https://github.com/allenantony-dev/codesearch-mcp/commit/12e2c39) · [`928b698`](https://github.com/allenantony-dev/codesearch-mcp/commit/928b698) · [`339865a`](https://github.com/allenantony-dev/codesearch-mcp/commit/339865a) · [`de10a29`](https://github.com/allenantony-dev/codesearch-mcp/commit/de10a29) · [`282881e`](https://github.com/allenantony-dev/codesearch-mcp/commit/282881e)

A chunk is useless if you can't open the file where it came from, so every row stores the path relative to the repo root and the chunk's first and last line. Each file's chunks are embedded in one Ollama call and written in one `pgx.Batch`, so a file costs one round trip to each.

> Two off-by-one bugs on the first try. The end line was printed as `end+1`, then `end-1`, before landing on `end`. And a 341-line file produced a chunk ending at line 342, because `strings.Split` on a file ending in a newline returns an extra empty string at the end. `strings.TrimSuffix(data, "\n")` before splitting fixes it.

### Re-indexing a repo

[`a742fdc`](https://github.com/allenantony-dev/codesearch-mcp/commit/a742fdc) · [`ad7b3e2`](https://github.com/allenantony-dev/codesearch-mcp/commit/ad7b3e2) · [`d7b6626`](https://github.com/allenantony-dev/codesearch-mcp/commit/d7b6626)

Running `index` twice doubled every row. Now each file's batch starts with a `DELETE` for that file, and the delete and inserts go through together, so a file is replaced rather than duplicated.

That still left rows for files that had been deleted from the repo, which would come back as results pointing at nothing. After the walk, one query deletes every row whose path wasn't seen:

```sql
DELETE FROM code_embeddings WHERE repo = $1 AND path <> ALL($2)
```

If the walk finds no `.go` files at all, it stops before this query. An empty list would match every row and wipe the repo.

Both deletes are scoped by a `repo` column, the directory name. Without it, indexing a second repo deleted all of the first one's rows.

### Knowing whether search works

[`902942e`](https://github.com/allenantony-dev/codesearch-mcp/commit/902942e) · [`194a16c`](https://github.com/allenantony-dev/codesearch-mcp/commit/194a16c)

Search orders chunks by cosine distance (`<=>`) to the embedded question. Trying a few queries by hand told me nothing: every result looked reasonable.

`eval` runs 23 questions from `questions.json`, each with the file and line that answers it, and reports **recall@k**: the share of questions whose answer appears in the top k results. The questions mix plain-English ones ("why does a failed job wait a random time instead of the full backoff") with exact identifiers (`ErrPermanent`, `WORKER_CONCURRENCY`).

Vector search alone scored 0.57 at @3 and 0.78 at @5, counting a hit when the right *file* appeared.

> pgx sends a `[]float32` as a Postgres `real[]`, and pgvector won't read that directly. `$1::vector` fails with `invalid input syntax for type vector`. It needs `$1::real[]::vector`.

### Vector search can't find identifiers

[`babb5b3`](https://github.com/allenantony-dev/codesearch-mcp/commit/babb5b3)

Ask vector search for `ErrPermanent` and it returns code that is *about* errors, not the line that defines `ErrPermanent`. Embeddings capture meaning, and an identifier is a name, not a meaning.

A generated `tsvector` column with a GIN index gives Postgres full-text search over the code, matched with `websearch_to_tsquery` and ordered by `ts_rank`. It uses the `simple` configuration, which lowercases words but doesn't stem them or drop common words, because code identifiers shouldn't be stemmed.

On its own, keyword search scored 0.13/0.17. It found identifiers and almost nothing else.

### Combining the two

[`d4837fd`](https://github.com/allenantony-dev/codesearch-mcp/commit/d4837fd)

The two searches fail on different questions, so `hybrid` runs both and merges them with Reciprocal Rank Fusion. Each list gives a chunk `1/(60 + rank)`, and the scores add up. Only positions are used, so the two scoring scales, a distance and a text rank, never have to be compared.

| file-level | recall@3 | recall@5 |
|---|---|---|
| vector | 0.57 | 0.78 |
| keyword | 0.13 | 0.17 |
| **hybrid** | **0.74** | **0.91** |

Those are file-level numbers, the only metric at the time. [Section 6](#the-eval-was-too-generous) replaces it with a stricter, line-level one, and on that metric the gap on these 23 questions is smaller:

| line-level | recall@3 | recall@5 | recall@10 |
|---|---|---|---|
| vector | 0.30 | 0.43 | 0.48 |
| keyword | 0.13 | 0.17 | 0.17 |
| **hybrid** | **0.48** | **0.57** | **0.57** |

That's 3 questions at @5, about the size of the chunking differences in [section 7](#chunking-by-function-instead-of-by-line-count), which I call noise. What makes this one different is *which* 3: `ErrPermanent`, `PriorityNormal` and `WORKER_CONCURRENCY`, all identifiers, and hybrid loses no question that vector alone gets. The gain comes from a mechanism, so it should hold on any question that names something. On the 15 held-out questions in [section 10](#how-many-results-to-return), a third of which are identifiers, vector alone scored 0.07/0.13/0.33 against hybrid's 0.47/0.53/0.73. Of the 6 questions hybrid gained there at @5, 5 are identifiers, and again it lost none.

> The sixth gain looked like it broke the mechanism, since [section 10](#how-many-results-to-return) shows keyword search returning nothing for English questions. It's *"…can't complete, retry or heartbeat it?"*, and `websearch_to_tsquery` reads a bare `or` as its OR operator, so the query became `(… & 'retry') | ('heartbeat' & 'it')`. The second half matches exactly one chunk, the test holding the answer. So the line isn't really identifier against English. It's whether some group of the question's words can all match, and this question got section 10's OR by accident. No other question in either set contains `or`, or a `-` that would read as negation.

Ties are common, because a chunk ranked at the same position in only one of the lists gets the same score. They're broken by repo, path and line so runs are repeatable despite Go's random map order.

### The eval was too generous

[`afff3c2`](https://github.com/allenantony-dev/codesearch-mcp/commit/afff3c2)

The next experiment, chunking by function, looked like a win at file level: 0.78/0.83. But "the right file" is a weak test. A 340-line file split into seven chunks counts as a hit whichever chunk comes back.

Now a hit requires the returned chunk to contain the answer line. Every number dropped:

| line-level, hybrid | recall@3 | recall@5 |
|---|---|---|
| 50-line chunks | 0.48 | 0.57 |
| one chunk per declaration | 0.30 | 0.35 |

> I made two changes in one go, a new chunking and a stricter eval, and for a moment I couldn't say which one the numbers were measuring. Every measurement after this changes one thing at a time.

### Chunking by function instead of by line count

[`b3e708f`](https://github.com/allenantony-dev/codesearch-mcp/commit/b3e708f) · [`09ff7e1`](https://github.com/allenantony-dev/codesearch-mcp/commit/09ff7e1) · [`23ed272`](https://github.com/allenantony-dev/codesearch-mcp/commit/23ed272)

A 50-line chunk cuts functions in half. Using `go/parser` to make one chunk per top-level declaration keeps every chunk whole, which sounds strictly better.

The comparison above isn't fair, though. Top 5 of 50-line chunks is up to 250 lines of code, and top 5 small declarations might be 40. Of course the bigger chunks find more. So `eval` also counts a hit only if it arrives within a fixed number of returned lines:

| hybrid | within 125 lines | within 250 lines |
|---|---|---|
| 50-line | 0.39 | 0.57 |
| per declaration | 0.43 | 0.52 |
| declarations merged to ≥ 10 lines | 0.35 | 0.43 |
| merged to ≥ 20 | 0.30 | 0.48 |
| merged to ≥ 30 | 0.26 | 0.52 |

At equal amounts of code, declaration chunking ties 50-line chunks. The differences are one question either way. So I went back to 50 lines: simpler, and no worse.

### Letting Claude use it

[`aeb23c7`](https://github.com/allenantony-dev/codesearch-mcp/commit/aeb23c7)

`codesearch mcp` speaks MCP over stdin/stdout using the official [Go SDK](https://github.com/modelcontextprotocol/go-sdk), with one tool:

```
search_code(query)  →  top hybrid chunks, each headed repo/path:start-end
```

The tool description tells Claude what the tool is good at (meaning *and* exact identifiers) and that chunks are partial: *"read the file around a result for more context."* That turned out to be the most useful sentence in the tool. Claude found `Release` and `ReapExpired` by reading `postgres.go` around a result rather than from the chunks themselves.

### Postgres goes down

[`391d566`](https://github.com/allenantony-dev/codesearch-mcp/commit/391d566)

Search called `log.Fatal` on any error. That's fine for a CLI, but in the MCP server it meant one dead database killed the process. Claude lost the tool for the rest of the session and could only report `MCP server "codesearch" is not connected`.

Search now returns its errors. The CLI still exits on them, and the tool handler returns them, so Claude sees *"search unavailable: failed to connect … connection refused"* and the server keeps running.

That only covers half of it. Tested inside Claude Code:

| Postgres | What Claude got |
|---|---|
| up | results |
| stopped | a tool error naming the database and port; server still up |
| started again, no reconnect | the same results as before |

The third row needs `pgxpool`.

> Same lesson as [taskq](https://github.com/allenantony-dev/taskq#a-job-outlives-its-lease), learned twice. A `pgx.Conn` doesn't reconnect: once Postgres restarts, that connection is dead for the life of the process, and every later call fails even with the database healthy. A pool throws away dead connections and dials new ones. It's also safe to use from concurrent tool calls, so the mutex around the connection went too.

### How many results to return

[`6244c03`](https://github.com/allenantony-dev/codesearch-mcp/commit/6244c03) · [`0bd5688`](https://github.com/allenantony-dev/codesearch-mcp/commit/0bd5688)

The tool started at 5 results, because 250 lines scored 0.57. In real use, Claude kept searching twice and still said things like *"the search didn't return `isFinal` or `backoffFor`, so I haven't seen what counts as a final failure."* That's the 43% miss rate, seen from the client's side.

**Where the misses are.** Going from 5 to 10 results added nothing on the 23 questions: recall@10 was 0.57, the same as recall@5. So I looked at the 10 misses together, and they turned out to be one kind of question:

- **Every miss is a plain-English question.** Every question that names an identifier is answered.
- **Keyword search returns zero rows for all of them.** `websearch_to_tsquery` requires every word to match, and with the `simple` configuration that includes "how", "does" and "the". So for exactly the questions that need help, hybrid is vector search alone.
- **Vector search ranks the answer 11th to 23rd** for 8 of them, and outside the top 30 for the other 2. With 41 chunks, chance would put it around 21st.
- **Test files crowd out the code they test.** 10 of the 41 chunks are `_test.go`, and one is in the top 3 for 6 of the 10 misses. A test mentions everything the code does, so it's a near-perfect decoy.

**Two obvious fixes, measured one at a time:**

| | @3 | @5 | @10 | within 250 lines |
|---|---|---|---|---|
| hybrid, as is | 0.48 | 0.57 | 0.57 | 0.57 |
| hybrid, keyword words OR'd instead of AND'd | 0.39 | 0.39 | 0.70 | 0.39 |
| hybrid, test files weighted 0.75 | 0.48 | 0.57 | 0.65 | 0.61 |

OR on its own lifts keyword search from 0.17 to 0.52 at @5, nearly as good as hybrid. But fused into hybrid, its noisy list pulls good vector hits down: two questions that were 1st fell to 7th and 9th. Down-weighting tests moved one question.

**Checking the winner.** 0.70 at @10 was the best number I'd measured, but I'd picked it from about eight setups tried on the same 23 questions, so some of it could be the questions rewarding my choices rather than better retrieval. I wrote down recall@10 as the deciding measure first, then had a separate Claude session, one that had never seen any search output, write 15 new questions from the source. Each setup ran on them once:

| held out, 15 questions | @3 | @5 | @10 |
|---|---|---|---|
| hybrid, AND (current) | 0.47 | 0.53 | **0.73** |
| hybrid, OR | 0.67 | 0.67 | **0.73** |

A tie. The 0.70 against 0.57 was noise. OR was worse at @5 on one set and better on the other, and a ranking that flips between question sets can't be trusted at this size. I kept AND for a reason that doesn't depend on any question set: the `simple` parser splits on underscores, so OR turns `WORKER_CONCURRENCY` into "worker OR concurrency", which matches most of taskq. It dropped out of the top 10 on both sets.

The held-out run also showed the original 23 weren't easier. The current setup scored 0.53 against 0.57.

> Two things I got wrong here. I said "10 results add nothing" as if it were settled. It was true of the 23 questions I'd been tuning on, and false on the new ones, where the current setup gained 3 answers going from 5 to 10. And 8 of the 15 "new" questions turned out to ask about the same lines as old ones, sometimes in nearly the same words. Two separate writers landed on the same obvious features, which says this codebase has maybe a dozen genuinely distinct things to ask about. The held-out set is really 7 questions.

**So the tool returns 10, chosen on cost, not on a score.** A chunk averages 1,103 characters, so 10 results are about 4K tokens, roughly 2% of the client's context. A miss costs another round trip and an answer with holes in it. Those costs are lopsided enough that the decision doesn't need the noisy measurement. With 10, the same retry question was answered in one search, including `isFinal`, the backoff schedule and the test.

### Answering instead of retrieving

[`914eb44`](https://github.com/allenantony-dev/codesearch-mcp/commit/914eb44)

Through MCP, Claude does the answering. `ask` does it locally, so both halves run in this process. It takes the top 5 hybrid chunks and a prompt that says *answer only from these chunks, cite `repo/path:start-end` for each claim, and say you don't know if they don't contain the answer*, then sends that to a local model.

I expected the interesting failure to be made-up answers: ask about a feature that doesn't exist and get a confident description of it. It didn't happen. The interesting failures were about code the model *was* given. All five questions went to llama3.2:3b, and only the two it got wrong were re-run on qwen2.5-coder:7b:

| question | answer in the 5 chunks? | llama3.2:3b | qwen2.5-coder:7b |
|---|---|---|---|
| priority aging (doesn't exist) | no | "I don't know" ✓ | not run |
| gateway auth (doesn't exist) | no | "I don't know", plus a hedged guess ✓ | not run |
| why the reaper resets to pending | no, retrieval missed it | "I don't know" ✓ | not run |
| SIGTERM shutdown | **yes** | "not present in the given chunks" ✗ | right in outline, two steps wrong |
| token bucket refill | yes | "replenished every second" ✗, no citation | right, no citation |

- **The prompt stopped the made-up answers.** Every question whose answer wasn't in the chunks got "I don't know", including the one where retrieval had missed the answer.
- **Reading code it was given is a model problem, not a prompt problem.** The 3B model refused an answer that was in front of it and misread the token bucket: nothing refills every second; each request adds `(now - last) * rate` tokens. A 7B code-tuned model fixed both, at about 2× the time. That changed two things at once, size and code training, so the claim is about that model, not about size.
- **The false refusal is the dangerous one.** A wrong description of the token bucket gets caught by anyone who reads the code. "Not present in the given chunks", when it was, looks exactly like the careful behaviour I asked for. Nothing here measures it: `eval` scores retrieval, so a question that retrieves perfectly and then gets refused counts as a hit.
- **The citation miss survived the model swap.** Both models skipped the citation on the token bucket question, which points at that prompt or chunk rather than at the model.

> `num_ctx` is Ollama's whole context window, shared by the prompt and the answer, and a prompt that doesn't fit is cut with no error back to the caller. The model then answers from whatever survived, which looks like a normal answer. These prompts were 1,814 to 2,126 tokens. On the version I ran, 0.22.0, the default is 4,096, which leaves about 2,000 tokens for a reply, so they'd have fit even without the override. Older releases defaulted to 2,048, where the longest prompt overflows outright and the shortest leaves about 200 tokens to answer. `ask` sets `num_ctx: 8192` for headroom, and that was in place before the first run, so neither model saw a truncated prompt and the false refusal isn't truncation.

`ask` defaults to `qwen2.5-coder:7b`, about 3–4 minutes per answer on my CPU. A slow right answer beats a fast wrong one. `ASK_MODEL` picks another model.

## Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE code_embeddings (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo       TEXT NOT NULL,          -- directory name, scopes every delete
    code       TEXT NOT NULL,
    path       TEXT NOT NULL,          -- relative to the repo root
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    embedding  VECTOR(768) NOT NULL,   -- nomic-embed-text
    tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', code)) STORED
);
CREATE INDEX ON code_embeddings USING GIN (tsv);
```

There's no index on `embedding`. At 41 rows a full scan is instant.

## Running it

Postgres with pgvector, and Ollama:

```bash
docker run -d --name codesearch-postgres -p 5433:5432 \
  -e POSTGRES_USER=codesearch -e POSTGRES_PASSWORD=codesearch -e POSTGRES_DB=codesearch \
  -v codesearch-pgdata:/var/lib/postgresql/data pgvector/pgvector:pg17
docker exec -i codesearch-postgres psql -U codesearch -d codesearch < schema.sql

ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b     # only for ask
```

```bash
export DB_URL='postgres://codesearch:codesearch@localhost:5433/codesearch?sslmode=disable'

go run . index ~/projects/taskq
go run . index ~/projects/ratelimit-gateway

go run . hybrid "where are failed jobs retried"
go run . ask "how is the token bucket refilled between requests?"
```

Re-running `index` on a repo replaces its rows, and files deleted since the last run disappear.

### Eval

```bash
go run . eval questions.json hybrid     # or keyword; omit for vector
go run . eval heldout.json hybrid
```

Each line is the rank of the first chunk containing the answer line (`-` if not in the top 10), followed by recall@3/@5/@10 and recall within 125 and 250 returned lines.

`heldout.json` has been run once against each setup above, and once against vector alone for the comparison in [section 5](#combining-the-two). Tune against it and it stops being held out.

### Claude Code

```bash
go build -o ~/go/bin/codesearch .
claude mcp add codesearch -e DB_URL="$DB_URL" -- ~/go/bin/codesearch mcp
```

Then ask *"using codesearch, where does taskq retry failed jobs?"* After rebuilding, run `/mcp` → Reconnect, because Claude Code keeps the old process running.

The server is added in local scope, so it only appears in sessions started in the directory where you ran `claude mcp add`. Add `-s user` to have it everywhere.

## Known gaps

Deliberately deferred, with the trigger for each.

**The eval can't resolve small effects**
- 41 chunks and 38 questions, 8 of them near-duplicates. Big effects, like hybrid against either search alone, show up clearly. A difference of 2–3 questions doesn't. Indexing a larger repo is the fix, and the prerequisite for any further tuning.
- Nothing measures whether `ask` used what it was given. A false refusal scores as a hit. That needs its own labelled set of answers.

**Retrieval**
- English questions get no help from keyword search, because AND needs every word to match. OR wasn't the fix. The untested next step is dropping common words, and it isn't a one-word change. Switching only the query to `english` makes things worse: the query gets stemmed ("retries" → "retri") while the `tsv` column isn't, so fewer words match. It needs a second `tsvector` column built with `english`, searched alongside the `simple` one, which stays for identifiers.
- Test files rank as decoys for the code they test. Down-weighting them didn't move the numbers at this size.
- 50-line chunks split functions, and a file's last chunk can be four lines of closing braces (`producer/main.go:51-54`), which still takes a result slot.
- Questions whose answer is implied by boilerplate rather than stated (*"how often does the reaper run"*, an interval passed to a ticker) are missed by every setup.

**When the corpus grows**
- No vector index, so search scans every row. Add HNSW when that becomes slow.
- `index` re-embeds every file on every run. Skipping unchanged files needs a content hash per file.

**Small**
- Only `.go` files are indexed.
- The tool description names the two indexed repos by hand.
- `ask` takes 3–4 minutes on CPU and prints nothing until it's done.
