# hyper-mcp-router

An adaptive, OpenAI Chat Completions compatible LLM router. It presents a single
virtual model to any client, classifies each incoming prompt with an **embedded**
zero-shot NLI model, and transparently forwards the request to the appropriate
backend model tier — with **modality-aware** routing across the full Chat
Completions v1 surface (text, image/audio/file input, audio/image output, and
tool calling).

No external state, no database, no runtime model downloads.

## Highlights

- **Single self-contained binary** — the ONNX model and tokenizer are embedded
  at build time; nothing is fetched at runtime.
- **Zero customer-data collection** — the default embedded model needs no
  fine-tuning and no telemetry, and classification is fully local.
  Operational logs contain routing metadata only, never user content.
- **Pluggable classification** — the classifier is a trait; models live one
  file per engine under `src/engines/` and are selected with the
  `[classifier] model` config setting (see
  [Classifier engines](#classifier-engines)).
- **Client-agnostic** — any OpenAI Chat Completions client works unmodified.
- **Correct SSE streaming** — raw byte passthrough, no buffering or re-parsing.
  Upstream response headers (request ids, rate-limit metadata) pass through on
  both the streaming and buffered paths.
- **Modality-aware routing** — each request is sent to a backend that supports
  every modality it requires (including tool calling), independent of
  complexity.
- **Context-aware complexity** — classified over a window of recent substantive
  user turns, so a terse follow-up inherits its context's difficulty; pure
  filler skips the model and routes to Fast (see
  [Performance & tuning](#performance--tuning)). No brittle history heuristic.
- **Adaptive inference concurrency** — the classifier auto-sizes a pool of
  inference sessions to the host's core count **and memory budget** (both
  container-aware: CPU quotas and cgroup memory limits), scaling throughput
  across cores with no configuration and without risking an OOM kill.

## ⚠️ Binary size

The embedded `model_quantized.onnx` is **~87 MB**, so the final binary is at
minimum this size. This is a deliberate design tradeoff: a single, fully
self-contained binary with no runtime downloads. The model is **not** compressed
or split.

## Building

`build.rs` downloads the model and tokenizer from HuggingFace into `OUT_DIR` on
the first build (network access required once); subsequent builds reuse the
cached files. Artifacts are fetched from a **pinned revision** and verified
against pinned SHA-256 digests — on download and on every cache hit — so a
partial download or upstream change can never be silently embedded.

```sh
cargo build --release
```

## Running

```sh
hyper-mcp-router serve [--config <path>] [--log-stdout]
```

- `--config <path>` — explicit config file. If given it is used verbatim; a
  missing or unparseable file is fatal (no fallback).
- `--log-stdout` — write structured JSON logs to stdout instead of the
  well-known rolling file location (use this for Cloud Run / containers).

That is the whole CLI, deliberately: everything else — including which
classifier model runs and its model-specific tuning — lives in the config
file. Different classifier models bring different configuration
(`[classifier.<model>]` tables), so the config file is the single source of
truth; see [Classifier engines](#classifier-engines) and
[Performance & tuning](#performance--tuning).

The server drains gracefully: on SIGTERM/SIGINT it stops accepting new
requests and lets in-flight ones (including open SSE streams) finish.

### Config discovery

Without `--config`, the first existing file is used:

| OS | Search order (first match wins) |
|---|---|
| Linux | `$XDG_CONFIG_HOME/hyper-mcp-router/config.toml` (or `~/.config/...`), then `/etc/hyper-mcp-router/config.toml` |
| macOS | `~/Library/Application Support/hyper-mcp-router/config.toml`, then `/etc/hyper-mcp-router/config.toml` |
| Windows | `%APPDATA%\hyper-mcp-router\config.toml` |

A missing config is a fatal error. See
[`config.example.toml`](./config.example.toml) for a fully-annotated example,
including plaintext / environment-variable / OS-keyring API keys and
`{ source = "google-adc" }` for Vertex-AI-hosted backends (per-request Google
OAuth tokens via Application Default Credentials, auto-refreshed).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Classify, route, and proxy a chat request |
| `GET` | `/v1/models` | Advertise the single virtual model `hyper-mcp-router` (backends are never *listed*; see note) |
| `GET` | `/health` | Liveness probe (`200 {"status":"ok"}`) |

Note on backend visibility: `/v1/models` only ever lists the virtual model, but
upstream response bodies pass through verbatim — so the `model` field of a
completion (and of every SSE chunk) names the backend that actually produced
it. The backend signs its work; treat backend names as visible to clients.

Request handling notes:

- `n > 1` (multiple choices) is rejected with `400` — the router serves exactly
  one completion per request and will not silently alter what you asked for.
  Everything else passes through byte-for-byte; only `model` is rewritten.
- Request bodies up to `server.max_body_bytes` (default 32 MiB) are accepted,
  comfortably covering base64 image/audio/file payloads.
- `server.request_timeout_secs` bounds **non-streaming** requests only;
  streaming responses have no total deadline and are guarded by the
  `server.stream_idle_timeout_secs` idle timeout instead, so long generations
  are never severed mid-stream.

## Classifier engines

Classification is pluggable behind the `ClassifierEngine` trait
(`src/classifier.rs`); concrete engines live in `src/engines/`, **one file
per model**, grouped by provider family (shared family plumbing lives in the
family's `mod.rs`, e.g. `engines/gemini/`). The active model is selected by
the `[classifier] model` config setting — **config-only, no CLI flag**: each
model brings its own settings in
a `[classifier.<model>]` table (e.g. `[classifier.deberta-v3-xsmall-zeroshot]`
holds `inference_pool_size` / `intra_op_threads`, which are meaningless for
other engines), so the config file is the single source of truth. Engine ids
name the **model**, never the technique — several engines can classify
zero-shot, but only one is `deberta-v3-xsmall-zeroshot`. Everything
model-specific is owned by the engine's file — how it is invoked (local
inference vs. a remote API), how many concurrent "sessions" it runs and how
they are sized, and how large its context window is. The proxy, window
construction, filler pruning, the lexical image prefilter, the
classification-skip optimisation, and the failure fallback are engine-agnostic
and never change when an engine is added.

| Model | File | Interaction | Sessions | Context budgets (window / current turn) |
|---|---|---|---|---|
| `deberta-v3-xsmall-zeroshot` (default) | `engines/deberta_v3_xsmall_zeroshot.rs` | embedded ONNX NLI, fully local | ORT session pool, auto-sized by CPU + memory | 1000 / 400 chars (512-token model) |
| `gemini-embedding-001` | `engines/gemini/embedding_001.rs` · `engines/vertex/gemini_embedding_001.rs` | remote anchor-prototype embeddings (Gemini API **or** Vertex AI — auth-selected) | concurrent embed calls (default 32) | 6000 / 2000 chars (2048-token model) |
| `gemini-embedding-2` | `engines/gemini/embedding_2.rs` · `engines/vertex/gemini_embedding_2.rs` | remote anchor-prototype embeddings (Gemini API **or** Vertex AI — auth-selected) | concurrent embed calls (default 32) | 24000 / 8000 chars (8192-token model) |
| `text-embedding-005` | `engines/vertex/text_embedding_005.rs` | remote anchor-prototype embeddings (Vertex AI) | concurrent embed calls (default 32) | 6000 / 2000 chars (2048-token model) |

Both budgets are trait methods (`context_char_budget`,
`current_turn_char_budget`), so an engine backed by a large-context model can
raise them — e.g. to see image-generation intent expressed deep in a long
prompt — without touching the routing core. They bound classifier input only;
forwarded requests are never truncated.

The remote engines (Gemini and Vertex AI families) classify by
**anchor prototypes** (shared, provider-neutral logic in
`engines/embedding.rs`): at startup they embed a curated exemplar set per
class (one batched call, failing fast on a bad credential or unreachable
endpoint), mean-pool them into prototype vectors, and per request cosine-score
the window and current turn against those prototypes. Each family's `mod.rs`
owns only its transport — wire format, auth, endpoint layout. The
`gemini-embedding` models are published on **both** Google surfaces; the
router treats these as two separate engines per model that happen to share a
name, and **the auth fields of the engine's config table pick which one
runs**: `api_key` (plaintext / `${ENV_VAR}` / OS keyring) selects the
Generative Language engine in `gemini/`, while `project` selects the Vertex
AI engine in `vertex/` (setting both is a startup error).
`text-embedding-005` is published only on **Vertex AI**. The `vertex/` family
takes a GCP `project` and `location` — both required, and `location` has
**no default**: it determines model availability, data residency, and the
endpoint host (a region like `us-central1`, or `global`) — plus an optional
`quota_project` for quota/billing attribution, authenticating via
**Application Default Credentials** (service-account key file, `gcloud auth application-default
login`, or the GCE/Cloud Run metadata server — token refresh handled by
`google-cloud-auth`), with an optional static `access_token` override for
quick tests. Per-request API failures degrade to
the balanced default like any engine failure.

**Privacy caveat**: the remote engines send prompt text (the classification
window and current turn) to their provider's API (Google). The
"zero customer-data collection, fully local" property in the highlights holds
**only for the default `deberta-v3-xsmall-zeroshot` engine**.

Adding an engine = one new file in `engines/`, one `ClassifierModel` variant,
and one `match` arm in `engines::build`. Nothing else changes.

## How routing works

Each request resolves two axes:

- **Modality set** (hard constraint) — read deterministically from the request
  JSON: content-part types (image/audio/file input), the `modalities` field
  (`"audio"` → audio output, `"image"` → image output), and tool signals — the
  `tools`/`functions` fields, or tool artifacts already in the transcript
  (`role: "tool"` messages, assistant `tool_calls`), so a tool-loop
  continuation stays on a tool-capable backend even if the follow-up omits
  `tools`.
- **Image-generation intent** (soft constraint) — when the request doesn't say
  `modalities: ["image"]` explicitly, image-output is *inferred* via a hardened
  lexical-OR-NLI-threshold signal. Because it is probabilistic, it is applied
  only if an image-capable backend exists — an inferred intent never makes a
  request unroutable; it degrades to a text route instead.
- **Complexity type** (preference) — the argmax of three complexity hypotheses,
  classified over a **window of recent substantive user turns** (see
  [Performance & tuning](#performance--tuning)), so a terse follow-up inherits
  the difficulty of its context. There is no message-history heuristic.

The router selects the configured model whose declared modalities are a
**superset** of the required set, preferring the resolved complexity type
(exact → nearest higher → highest lower). If no single model covers the required
set it returns `422`.

Complexity is only used to *rank among* candidates, so when at most one model can
serve the required modality set (e.g. a single-model deployment, or a request
for a modality only one backend provides) the router **skips classification
entirely** and routes directly — zero inference. (The NLI image-generation
signal is skipped along with it; the lexical signal and the explicit
`modalities` field still apply.)

## Performance & tuning

Complexity classification runs a single batched forward pass through the
embedded NLI model, and that pass is the router's main CPU cost. The mechanisms
below keep it accurate, off the hot path where possible, and scalable across
cores. **All are automatic** — the knobs exist only for override.

### Complexity window (context-aware, no history heuristic)

Complexity is classified over a **window of recent substantive user turns**, not
just the last message. Walking back from the current turn, the router:

- skips **trivially-simple** turns — greetings/acknowledgements like `"ok"`,
  `"thanks"`, `"please continue"` (a short turn that is also free of reasoning
  cues like `prove`/`derive`/`analyze` **and** matches an acknowledgement
  pattern, so a terse `"Prove P != NP."` is never mistaken for filler);
- skips **assistant/tool** messages entirely (usually the longest text), so the
  budget stretches across many turns of actual user intent;
- accumulates substantive user turns until the conversation start or a character
  budget (kept safely inside the model's 512-token limit).

Consequences:

- A **terse follow-up inherits its context**: `"ok, continue"` after a proof
  request classifies as hard, because the walk-back reaches the substantive turn
  behind it. Because filler is pruned, the window ages by *substantive* turns
  — a hard conversation stays hard until genuinely new (substantive) work pushes
  it out; the natural reset is a **new conversation**.
- A conversation of **pure filler** prunes to an empty window and routes to the
  **Fast** tier **without running the model** (~0.3 ms vs ~13 ms) — the old
  fast-path, now falling out naturally.
- Image-generation intent is judged on the **current turn only**, so an old
  "draw a cat" turn in the window can't misroute a later, unrelated request.

`[classifier] trivial_max_words` (default `6`) sets the filler-pruning length
ceiling; `0` disables pruning.

### Adaptive inference concurrency (session pool)

`ort`'s `Session::run` requires exclusive access, so the classifier holds a
**pool** of independent sessions and leases one per request, allowing up to
`pool_size` inferences to run at once. At startup the router detects the
resources available to the process and sizes the pool to fit **both**:

- **CPU** — cores from `available_parallelism()`, honoring container CPU
  limits (cgroup quotas) and CPU affinity: pool `cores / 2`.
- **Memory** — the budget from the **cgroup memory limit** when present
  (what Cloud Run gen2 / Kubernetes / Docker actually enforce — inside a
  container `/proc/meminfo` reports the *host's* memory, which would
  over-provision and get the instance OOM-killed), falling back to the
  host/VM total elsewhere: pool = whatever fits in 90% of the budget after
  the measured fixed baseline (~420 MB) at ~200 MB per session (see
  [Memory](#memory) below).

The plan takes the **minimum** of the two, min 1. The startup log reports
`detected_cores`, `memory_budget_mb`, `pool_size`, and `intra_op_threads`.
Explicit `[classifier.deberta-v3-xsmall-zeroshot]` settings are always
honored — but a configuration the host can't handle (thread oversubscription,
or estimated memory above the detected budget) logs a **warning** at startup
instead of failing.

| `[classifier.deberta-v3-xsmall-zeroshot]` setting | Default | Meaning |
|---|---|---|
| `inference_pool_size` | auto (min of `cores / 2` and what fits in memory; min 1) | Concurrent inference sessions. Each is an independent in-memory copy of the model. |
| `intra_op_threads` | auto (`2`) | ONNX Runtime intra-op threads per session (`0` = runtime default). Keep `pool_size × intra_op_threads` near the core count to avoid oversubscription. |

#### Memory

Because each session is a full in-memory copy of the embedded ~87 MB model, a
larger pool uses proportionally more memory. Measured
(`measure_session_memory.sh`, macOS, debug profile): **~105 MB per
session at startup** (weights), growing to **~190 MB per session under
sustained max-length load** — ONNX Runtime's arena allocator retains the
worst-case activation memory it has seen. Both scale linearly with pool size,
so budget `pool_size × ~190 MB` plus a ~420 MB fixed baseline. These measured
constants are exactly what the memory-aware auto-plan uses
(`planning::SESSION_MEMORY_BYTES` / `planning::BASELINE_MEMORY_BYTES`), so an
unconfigured router already fits its memory budget; an explicit
`inference_pool_size` beyond it logs a startup warning. Cap it in
`[classifier.deberta-v3-xsmall-zeroshot]` on memory-constrained hosts.

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
`LOAD_PROMPT` (fixed prompt — a trivial one measures the empty-window Fast path),
`LOAD_TURNS` (build N-user-turn conversations to measure how the windowed
classifier scales with conversation depth), `LOAD_POOL_SIZE` / `LOAD_INTRA_OP`
(build a dedicated classifier to sweep pool size).

Per-session **memory** is measured separately by
`measure_session_memory.sh` (repo root), which starts the router at two pool
sizes, records RSS after startup and after a burst of max-length requests,
and divides the deltas (fixed costs cancel out). Overrides: `PROFILE`
(`debug`/`release`), `POOLS` (e.g. `"1 4"`), `REQUESTS`, `PORT`.

The windowed classifier's per-request cost grows with conversation depth but is
**bounded** by the character budget — measured on an 18-core host (pool 8,
single request): ~15 ms (1 turn) → ~31 ms (4) → ~60 ms (8) → ~81 ms (16), then
**flat** (32 turns ≈ 16 turns), because the window saturates at the budget
rather than growing with the transcript.

## Logging

Structured JSON (NDJSON) always. Level via `RUST_LOG` (default `info`). Log
destination is the rolling daily file `{config dir}/hyper-mcp-router/logs/router.log`
(overridable with `ROUTER_LOG_PATH`) or stdout with `--log-stdout`.

## License

Apache-2.0. See [LICENSE](./LICENSE).
