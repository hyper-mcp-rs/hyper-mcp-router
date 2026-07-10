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
hyper-mcp-router serve [--config <path>] [--log-stdout]
```

- `--config <path>` — explicit config file. If given it is used verbatim; a
  missing or unparseable file is fatal (no fallback).
- `--log-stdout` — write structured JSON logs to stdout instead of the
  well-known rolling file location (use this for Cloud Run / containers).

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
  escalated by cheap message-history heuristics.

The router selects the configured model whose declared modalities are a
**superset** of the required set, preferring the resolved complexity type
(exact → nearest higher → highest lower). If no single model covers the required
set it returns `415`.

## Logging

Structured JSON (NDJSON) always. Level via `RUST_LOG` (default `info`). Log
destination is the rolling daily file `{config dir}/hyper-mcp-router/logs/router.log`
(overridable with `ROUTER_LOG_PATH`) or stdout with `--log-stdout`.

## License

Apache-2.0. See [LICENSE](./LICENSE).
