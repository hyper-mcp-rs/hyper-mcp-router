# hyper-mcp-router

An adaptive, OpenAI Chat Completions compatible LLM router. It presents a single
virtual model to any client, classifies each incoming prompt — by default with
an **embedded** zero-shot NLI model; remote embedding engines are pluggable —
and transparently forwards the request to the appropriate backend model tier,
with **modality-aware** routing across the full Chat Completions v1 surface
(text, image/audio/file input, audio/image output, and tool calling) and
**context-window-aware** placement (a request is never knowingly sent to a
backend whose window it cannot fit).

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
- **Context-window-aware routing** — every model declares its `context_window`
  (required, in tokens); a request whose estimated size overflows a "fast"
  model's small window routes to a backend that can actually hold it (see
  [How routing works](#how-routing-works)).
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

## Docker

The [`Dockerfile`](./Dockerfile) builds the smallest image this router can
ship in: a **`scratch`** container holding a fully statically linked (musl)
executable and a CA bundle — nothing else, not even a shell (~200 MB on
disk, dominated by the embedded model). pyke publishes no musl ONNX Runtime
binaries, so stage 1 compiles ONNX Runtime from source as static musl
libraries — the first build takes ~30–60 min on native hardware; later
builds reuse the cached layer. A build-time `readelf` gate fails the build
if the executable picks up any dynamic dependency.

```sh
docker build -t hyper-mcp-router .                                         # native arch
docker buildx build --platform linux/amd64,linux/arm64 -t hyper-mcp-router . # both
```

Both `linux/amd64` and `linux/arm64` are supported; prefer **native
builders per architecture** — the ONNX Runtime compile under QEMU emulation
takes hours.

Mount the config at the well-known `/etc` path (no flags needed — config
discovery probes it) with `server.host = "0.0.0.0"` so the port mapping can
reach it; logs go to stdout:

```sh
docker run -p 8080:8080 \
  -v ./config.toml:/etc/hyper-mcp-router/config.toml:ro \
  hyper-mcp-router
```

For `{ source = "google-adc" }` backends and the Vertex AI classifier
engines, provide Application Default Credentials — there is no gcloud in
the image (on GCE/Cloud Run/GKE the metadata server provides ADC and no
mount is needed):

```sh
docker run -p 8080:8080 \
  -v ./config.toml:/etc/hyper-mcp-router/config.toml:ro \
  -v ~/.config/gcloud/application_default_credentials.json:/gcloud/adc.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud/adc.json \
  hyper-mcp-router
```

Container caveats: the image runs as a non-root numeric user with no
writable filesystem — always use `--log-stdout` (the default `CMD` does) —
and `api_key = { source = "keyring" }` cannot work without an OS secret
store; use `${ENV_VAR}`-expanded keys instead.

## Running

```sh
hyper-mcp-router serve [--config <path>] [--log-stdout]
hyper-mcp-router validate [--config <path>]
```

- `--config <path>` — explicit config file. If given it is used verbatim; a
  missing or unparseable file is fatal (no fallback). The format is chosen by
  file extension: `.toml`, `.yaml`/`.yml`, or `.json` (anything else is an
  error).
- `--log-stdout` — write structured JSON logs to stdout instead of the
  well-known rolling file location (use this for Cloud Run / containers).

`validate` loads and validates a config, prints the classifier ladder and
backend catalogue that `serve` would run, and exits — without starting the
server, initialising engines, or touching the network. It covers everything
`serve` checks at boot that can be checked offline: schema and field
validation, modality coverage, the engine auth-surface choice, required
Vertex `project`/`location` fields, and distinct capacity-ladder budgets.
Credentials are **not** exercised — a config can validate and still fail at
boot on ADC or keyring problems. A validation failure exits non-zero, so it
works as a CI/pre-deploy check.

That is the whole CLI, deliberately: everything else — including which
classifier model runs and its model-specific tuning — lives in the config
file. Different classifier models bring different configuration
(`[classifier.<model>]` tables), so the config file is the single source of
truth; see [Classifier engines](#classifier-engines) and
[Performance & tuning](#performance--tuning).

The server drains gracefully: on SIGTERM/SIGINT it stops accepting new
requests and lets in-flight ones (including open SSE streams) finish.

### Config discovery

Without `--config`, the first existing file is used. In each directory the
file names `config.toml`, `config.yaml`, `config.yml`, and `config.json` are
probed in that order:

| OS | Search order (first match wins) |
|---|---|
| Linux | `$XDG_CONFIG_HOME/hyper-mcp-router/config.{toml,yaml,yml,json}` (or `~/.config/...`), then `/etc/hyper-mcp-router/config.{toml,yaml,yml,json}` |
| macOS | `~/Library/Application Support/hyper-mcp-router/config.{toml,yaml,yml,json}`, then `/etc/hyper-mcp-router/config.{toml,yaml,yml,json}` |
| Windows | `%APPDATA%\hyper-mcp-router\config.{toml,yaml,yml,json}` |

A missing config is a fatal error. See
[`config.example.toml`](./config.example.toml) for a fully-annotated example
([`config.example.yaml`](./config.example.yaml) and
[`config.example.json`](./config.example.json) are exact equivalents — JSON
has no comments, so the TOML file is the canonical reference),
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
family's `mod.rs`, e.g. `engines/gemini/`). The active model(s) are selected
by the `[classifier] model` config setting — a single id, or a **list**
forming a [capacity ladder](#the-capacity-ladder-multiple-engines) —
**config-only, no CLI flag**: each
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

### The capacity ladder (multiple engines)

`[classifier] model` also accepts a **list**:

```toml
[classifier]
model = ["deberta-v3-xsmall-zeroshot", "gemini-embedding-2"]
```

The engines form a **capacity ladder** (`EngineRoster` in
`src/classifier.rs`), ordered by their `context_char_budget` — derived from
the engines, never declared in config. Per request the router builds the
classification window once at the *top* engine's budget and hands it to the
**smallest engine whose budget covers it**: short prompts stay on the cheap
(typically local) engine, long prompts escalate to the larger-context one
instead of being judged by their last 1000 characters, and only a window that
exceeds even the top budget is truncated (at the top budget). A single-engine
config behaves exactly as before.

Ladder rules, enforced at startup: budgets must be **unique** (the ladder
needs a total order; listing the same model twice is a config error), and
`current_turn_char_budget` must be **monotone** in ladder order (a
higher-capacity engine may never see *less* of the current turn). The startup
log prints one line per rung — engine, budgets, and a `local` marker.
Thresholds: `image_generation_threshold` scales are engine-specific, so with
several engines prefer the per-engine key in each `[classifier.<model>]`
table; the top-level key acts as the fallback for engines without their own.
A request-time engine failure still maps to the balanced default —
deliberately no retry on a lower rung. Note that a growing conversation can
cross a rung boundary mid-session and be judged by a different engine; the
classification is best-effort by design.

**Privacy**: with a mixed local/remote ladder (like the example above),
prompt **length** decides whether prompt text is sent to a remote provider
for classification. Configure only local engines if that must never happen.

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
**only for the default `deberta-v3-xsmall-zeroshot` engine** — and, on a
multi-engine ladder, only when **every** rung is local.

Adding an engine = one new file in `engines/`, one `ClassifierModel` variant,
and one `match` arm in `engines::build`. Nothing else changes (its unique
`context_char_budget` slots it into the ladder automatically).

## How routing works

Each request resolves three routing axes — capability (the modality set,
including an inferred image-generation signal), capacity (context fit), and
complexity:

- **Modality set** (hard constraint) — read deterministically from the request
  JSON: content-part types (image/audio/file input), the `modalities` field
  (`"audio"` → audio output, `"image"` → image output), and tool signals — the
  `tools`/`functions` fields, or tool artifacts already in the transcript
  (`role: "tool"` messages, assistant `tool_calls`), so a tool-loop
  continuation stays on a tool-capable backend even if the follow-up omits
  `tools`.
- **Image-generation intent** (soft addition to the modality set) — when the
  request doesn't say `modalities: ["image"]` explicitly, image-output is
  *inferred* via a hardened lexical-OR-NLI-threshold signal. Because it is
  probabilistic, it is applied only if an image-capable backend exists that
  also fits the request — an inferred intent never makes a request
  unroutable; it degrades to a text route instead.
- **Context fit** (strong preference) — the request's estimated context
  occupancy: the text of every message at ~4 characters per token, plus any
  requested `max_tokens`/`max_completion_tokens` completion budget. Every
  model declares its `context_window` in config (**required**, in tokens);
  candidates whose window cannot fit the estimate are avoided — "fast" models
  usually have far smaller windows than "frontier" ones, and a very large
  request sent to a small-window backend is a guaranteed upstream failure.
- **Complexity type** (preference) — the argmax of three complexity hypotheses,
  classified over a **window of recent substantive user turns** (see
  [Performance & tuning](#performance--tuning)), so a terse follow-up inherits
  the difficulty of its context. There is no message-history heuristic.

### Modality reference

The full modality vocabulary — the ids a model declares in its `modalities`
config list, what makes a request *require* each one, and what happens when
no configured model covers it:

| Modality id | Direction | A request requires it when… | If no model declares it |
|---|---|---|---|
| `text` | in/out | always — text I/O is in play for every chat request | Startup error: text is the fallback baseline, so at least one model (any tier) **must** declare it |
| `image-input` | input | any message has an `image_url` content part (`input_image` accepted as a newer alias) | `422` at request time |
| `audio-input` | input | any message has an `input_audio` content part | `422` at request time |
| `file-input` | input | any message has a `file` content part (documents, e.g. PDFs) | `422` at request time |
| `audio-output` | output | the request's `modalities` field contains `"audio"` | `422` at request time |
| `image-output` | output | **explicit**: the request's `modalities` field contains `"image"` (hard requirement) — or **inferred**: the lexical/NLI image-generation signal fires (soft) | Explicit: `422`. Inferred: the intent is dropped and the request degrades to a text route (also when no image-capable model fits the request's context estimate) |
| `tools` | capability | the request offers a non-empty `tools` (or legacy `functions`) array — or the transcript already carries tool artifacts: `role: "tool"` messages, assistant `tool_calls`, or a legacy `function_call` | `422` at request time |

Detection is deterministic, metadata-only, and never consults the classifier;
a model's declared set must be a **superset** of the required set to be a
candidate. Only `text` coverage is validated at startup — every other
modality is best-effort: configure it or requests requiring it are refused
with a `422` naming the uncovered set.

The router selects the configured model whose declared modalities are a
**superset** of the required set and whose context window fits the request,
preferring the resolved complexity type (exact → nearest higher → highest
lower). If no single model covers the required set it returns `422`. If
covering models exist but the request (by estimate) fits none of their
windows, it is still forwarded to the largest-window candidate as a best
effort — the estimate is a chars-per-token heuristic and the backend is the
authority — with a warning logged.

Complexity is only used to *rank among* candidates, so when at most one model
can serve the required modality set within its context window (e.g. a
single-model deployment, a request for a modality only one backend provides,
or a long transcript only one window can hold) the router **skips
classification entirely** and routes directly — zero inference. (The NLI
image-generation signal is skipped along with it; the lexical signal and the
explicit `modalities` field still apply.)

## Performance & tuning

What classification *costs* depends on the engine. With the default embedded
`deberta-v3-xsmall-zeroshot` engine it is a single batched ONNX forward pass
— the router's main **CPU** cost; with a remote embedding engine it is one
batched embed **API call** — a network round-trip and provider quota, with
essentially no local CPU or memory cost. The mechanisms below keep
classification accurate, off the hot path where possible, and scalable.
**All are automatic** — the knobs exist only for override.

Scope guide: the [complexity window](#complexity-window-context-aware-no-history-heuristic)
is engine-agnostic routing policy (only the character budget is
engine-specific). The [session pool](#adaptive-inference-concurrency-embedded-engine)
and [memory](#memory) sections — and every latency/throughput/memory number
in them — apply **only to the embedded engine**; the remote engines'
equivalents are covered in
[their own section](#remote-engines-gemini--vertex-ai).

### Complexity window (context-aware, no history heuristic)

Complexity is classified over a **window of recent substantive user turns**, not
just the last message. Walking back from the current turn, the router:

- skips **trivially-simple** turns — greetings/acknowledgements like `"ok"`,
  `"thanks"`, `"please continue"` (a short turn that is also free of reasoning
  cues like `prove`/`derive`/`analyze` **and** matches an acknowledgement
  pattern, so a terse `"Prove P != NP."` is never mistaken for filler);
- skips **assistant/tool** messages entirely (usually the longest text), so the
  budget stretches across many turns of actual user intent;
- accumulates substantive user turns until the conversation start or a
  character budget — engine-specific, sized to the classifier model's input
  limit (for the default embedded model, kept safely inside its 512-token
  limit; see the [capacity ladder](#the-capacity-ladder-multiple-engines) for
  how a multi-engine roster sizes the window).

Consequences:

- A **terse follow-up inherits its context**: `"ok, continue"` after a proof
  request classifies as hard, because the walk-back reaches the substantive turn
  behind it. Because filler is pruned, the window ages by *substantive* turns
  — a hard conversation stays hard until genuinely new (substantive) work pushes
  it out; the natural reset is a **new conversation**.
- A conversation of **pure filler** prunes to an empty window and routes to the
  **Fast** tier **without running the engine at all** (~0.3 ms vs ~13 ms for
  the embedded model) — the old fast-path, now falling out naturally. For a
  remote engine this saves a billable API call — and keeps that prompt's text
  from leaving the process at all.
- Image-generation intent is judged on the **current turn only**, so an old
  "draw a cat" turn in the window can't misroute a later, unrelated request.

`[classifier] trivial_max_words` (default `6`) sets the filler-pruning length
ceiling; `0` disables pruning.

### Adaptive inference concurrency (embedded engine)

**Embedded-engine only** — everything in this section (and [Memory](#memory)
below) concerns the local ONNX sessions of `deberta-v3-xsmall-zeroshot`; the
remote engines run no local inference and ignore these settings entirely.

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

### Remote engines (Gemini / Vertex AI)

The remote embedding engines have a different cost profile, and different
knobs (in their `[classifier.<model>]` tables):

- **Per-request cost is one batched embed call** — the classification window
  and the current turn embedded together (Gemini `batchEmbedContents`;
  Vertex `:predict` with true batching — except the gemini-* models on
  Vertex, whose API takes one text per request, so a "batch" fans out as up
  to two concurrent calls). Latency is the provider's API round-trip, not
  local CPU; cost is provider quota/billing.
- **`max_concurrency`** (default `32`) bounds in-flight embed calls — the
  remote analogue of the embedded engine's session pool, backed by a
  semaphore rather than ONNX sessions. There is no CPU/memory auto-plan to
  size it: raise it for burst throughput (within provider rate limits),
  lower it to smooth quota consumption.
- **`request_timeout_secs`** (default `10`) bounds each embed call; a
  timeout or API failure degrades that request to the balanced default,
  like any engine failure.
- **No memory planning** — `inference_pool_size` / `intra_op_threads` and
  the [Memory](#memory) numbers do not apply; the engines hold only
  prototype vectors.
- **Startup embeds the anchor set** in one batched call, failing fast on a
  bad credential or unreachable endpoint — a broken remote engine is a boot
  error, not a per-request surprise.

The engine-agnostic savings matter *more* here: the trivial fast-path and
the classification-skip optimisation don't just save milliseconds — each
skipped classification is an API call not billed and prompt text not sent to
the provider.

### Measuring

Both measuring tools below exercise the **embedded engine**; remote-engine
performance is dominated by the provider's API latency and rate limits, so
measure those against the provider's own quotas.

An opt-in load test ramps concurrency and reports latency percentiles and
throughput:

```sh
cargo test --test api_routing -- --ignored --nocapture load_test_progressive_concurrency
```

Environment overrides: `LOAD_REQUESTS` (requests per concurrency level),
`LOAD_PROMPT` (fixed prompt — a trivial one measures the empty-window Fast path),
`LOAD_TURNS` (build N-user-turn conversations to measure how the windowed
classifier scales with conversation depth), `LOAD_POOL_SIZE` / `LOAD_INTRA_OP`
(build a dedicated embedded classifier to sweep its pool size).

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

Structured JSON (NDJSON) always. Level via `RUST_LOG`; the default is `info`
with ONNX Runtime's verbose per-session logging quieted to `warn`
(`info,ort=warn`), and a set `RUST_LOG` overrides that entirely. Log
destination is the rolling daily file `{config dir}/hyper-mcp-router/logs/router.log`
(directory overridable with `ROUTER_LOG_PATH`) or stdout with `--log-stdout`.
Routing logs are metadata-only — modalities, tier, engine, prompt sizes,
estimated tokens, latency — never user content.

## License

Apache-2.0. See [LICENSE](./LICENSE).
