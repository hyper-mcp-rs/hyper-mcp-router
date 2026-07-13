# hyper-mcp-router

An adaptive, OpenAI Chat Completions compatible LLM router. It presents a single
virtual model to any client, classifies each incoming prompt with an **embedded**
zero-shot NLI model, and transparently forwards the request to the appropriate
backend model tier — with **modality-aware** routing across the full Chat
Completions v1 surface (text, image/audio/file input, audio/image output).

No external state, no database, no runtime model downloads.

## Highlights

- **Single self-contained binary** — the ONNX model and tokenizer are embedded
  at build time; nothing is fetched at runtime.
- **Zero customer-data collection** — the embedded model needs no fine-tuning
  and no telemetry. Operational logs contain routing metadata only, never user
  content.
- **Client-agnostic** — any OpenAI Chat Completions client works unmodified.
- **Correct SSE streaming** — raw byte passthrough, no buffering or re-parsing.
- **Modality-aware routing** — each request is sent to a backend that supports
  every modality it requires, independent of complexity.
- **Trivial-prompt fast path** — greetings and acknowledgements skip the model
  entirely and route to the Fast tier in microseconds (see
  [Performance & tuning](#performance--tuning)).
- **Adaptive inference concurrency** — the classifier auto-sizes a pool of
  inference sessions to the host's core count (container-aware), scaling
  throughput across cores with no configuration.

## ⚠️ Binary size

The embedded `model_quantized.onnx` is **~87 MB**, so the final binary is at
minimum this size. This is a deliberate design tradeoff: a single, fully
self-contained binary with no runtime downloads. The model is **not** compressed
or split.

## Building

`build.rs` downloads the model and tokenizer from HuggingFace into `OUT_DIR` on
the first build (network access required once); subsequent builds reuse the
cached files.

```sh
cargo build --release
```

## Running

```sh
hyper-mcp-router serve [--config <path>] [--log-stdout] \
    [--trivial-max-words <N>] [--inference-pool-size <N>] [--intra-op-threads <N>]
```

- `--config <path>` — explicit config file. If given it is used verbatim; a
  missing or unparseable file is fatal (no fallback).
- `--log-stdout` — write structured JSON logs to stdout instead of the
  well-known rolling file location (use this for Cloud Run / containers).
- `--trivial-max-words <N>` — length ceiling for the trivial-prompt fast path
  (default `6`; `0` disables it). See [Performance & tuning](#performance--tuning).
- `--inference-pool-size <N>` — concurrent inference sessions (default: auto,
  from the detected core count).
- `--intra-op-threads <N>` — ONNX Runtime intra-op threads per session
  (default: auto; `0` = runtime default).

The performance flags are optional overrides — omit them and the router sizes
itself to the machine. See [Performance & tuning](#performance--tuning).

### Config discovery

Without `--config`, the first existing file is used:

| OS | Search order (first match wins) |
|---|---|
| Linux | `$XDG_CONFIG_HOME/hyper-mcp-router/config.toml` (or `~/.config/...`), then `/etc/hyper-mcp-router/config.toml` |
| macOS | `~/Library/Application Support/hyper-mcp-router/config.toml`, then `/etc/hyper-mcp-router/config.toml` |
| Windows | `%APPDATA%\hyper-mcp-router\config.toml` |

A missing config is a fatal error. See [`config.toml`](./config.toml) for a
fully-annotated example, including plaintext / environment-variable / OS-keyring
API keys.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Classify, route, and proxy a chat request |
| `GET` | `/v1/models` | Advertise the single virtual model `hyper-mcp-router` (backends are never exposed) |
| `GET` | `/health` | Liveness probe (`200 {"status":"ok"}`) |

## How routing works

Each request resolves two axes:

- **Modality set** (hard constraint) — read deterministically from the request
  JSON (content-part types and the `modalities` field). `image-output` is the
  only inferred modality, via a hardened lexical-OR-NLI-threshold signal.
- **Complexity type** (preference) — the argmax of three complexity hypotheses,
  escalated by cheap message-history heuristics. Trivial turns skip the model
  via a fast path (see [Performance & tuning](#performance--tuning)).

The router selects the configured model whose declared modalities are a
**superset** of the required set, preferring the resolved complexity type
(exact → nearest higher → highest lower). If no single model covers the required
set it returns `415`.

## Performance & tuning

Every request that isn't short-circuited runs a single batched forward pass
through the embedded NLI model, and that pass is the router's main CPU cost. Two
mechanisms keep it off the hot path and let it scale across cores. **Both are
automatic** — the knobs below exist only for override.

### Trivial-prompt fast path

Terse turns — greetings, acknowledgements, `"ok"`, `"thanks"`, `"please
continue"` — skip the model entirely via a cheap lexical/length guard and route
directly to the **Fast** tier. The guard is deliberately conservative: a turn
must be short **and** free of reasoning cues (`prove`, `derive`, `analyze`, …)
**and** match an acknowledgement pattern, so a terse `"Prove P != NP."` is not
mistaken for filler. Image-generation intent always defers to the full path, and
history-based escalation still applies on top — a terse turn on a deep or
tool-calling thread is unaffected.

- Skipped turns cost roughly **0.3 ms instead of ~13 ms** and never occupy the
  inference pool.
- `--trivial-max-words <N>` (default `6`) sets the length ceiling; `0` disables
  the fast path.

### Adaptive inference concurrency (session pool)

`ort`'s `Session::run` requires exclusive access, so the classifier holds a
**pool** of independent sessions and leases one per request, allowing up to
`pool_size` inferences to run at once. At startup the router detects the cores
available to the process — honoring container CPU limits (cgroup quotas) as well
as CPU affinity — and sizes the pool and per-session thread count to match. The
startup log reports `detected_cores`, `pool_size`, and `intra_op_threads`.

| Flag | Default | Meaning |
|---|---|---|
| `--inference-pool-size <N>` | auto (`cores / 2`, min 1) | Concurrent inference sessions. Each is an independent in-memory copy of the model. |
| `--intra-op-threads <N>` | auto (`2`) | ONNX Runtime intra-op threads per session (`0` = runtime default). Keep `pool_size × intra_op_threads` near the core count to avoid oversubscription. |

Because each session is a full copy of the embedded model, a larger pool uses
proportionally more memory (the model is small, so this is usually negligible).

Measured on an 18-core host — per-request latency ~15 ms, essentially unchanged
by pool size — throughput scales with the pool until the CPU saturates:

| `pool_size` (intra-op 2) | throughput |
|---|---|
| 1 | ~68 req/s |
| 4 | ~230 req/s |
| 8 | ~390 req/s |

The auto-plan on that host (pool 9, intra-op 2) reaches roughly **6×** the
single-session ceiling. Lowering `intra_op_threads` trades a little per-request
latency for more concurrency; raise it if single-request latency matters more
than throughput.

### Measuring

An opt-in load test ramps concurrency and reports latency percentiles and
throughput:

```sh
cargo test --test api_routing -- --ignored --nocapture load_test_progressive_concurrency
```

Environment overrides: `LOAD_REQUESTS` (requests per concurrency level),
`LOAD_PROMPT` (fixed prompt — a trivial one measures the fast path),
`LOAD_POOL_SIZE` / `LOAD_INTRA_OP` (build a dedicated classifier to sweep pool
size).

## Logging

Structured JSON (NDJSON) always. Level via `RUST_LOG` (default `info`). Log
destination is the rolling daily file `{config dir}/hyper-mcp-router/logs/router.log`
(overridable with `ROUTER_LOG_PATH`) or stdout with `--log-stdout`.

## License

Apache-2.0. See [LICENSE](./LICENSE).
