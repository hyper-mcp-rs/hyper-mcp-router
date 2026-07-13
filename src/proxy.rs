//! HTTP surface and request forwarding.
//!
//! Requests and responses are handled as raw `serde_json::Value` throughout —
//! the router never deserialises into typed OpenAI structs, guaranteeing
//! byte-for-byte passthrough of every field the client sends. Only `messages`,
//! `model`, `stream`, and `n` are ever read or mutated.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};

use crate::classifier::{
    build_classification_window, detect_required_modalities, extract_prompt,
    looks_like_image_generation, truncate_prompt, Classification, Classifier, Modality,
    ModalitySet, ModelTier, CLASSIFICATION_CHAR_BUDGET,
};
use crate::config::RouterConfig;

/// Shared, cloneable server state. `Classifier` holds an unsynchronised
/// `Session` (ORT supports concurrent `run(&self)`); everything here is
/// `Send + Sync`.
#[derive(Clone)]
pub struct AppState {
    pub classifier: Arc<Classifier>,
    pub config: Arc<RouterConfig>,
    pub http: reqwest::Client,
}

impl AppState {
    /// Build the shared state, constructing the upstream HTTP client with the
    /// configured connect/request timeouts. **No retries** are configured — a
    /// retry could trigger duplicate, billable generations.
    pub fn new(classifier: Arc<Classifier>, config: Arc<RouterConfig>) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(config.server.connect_timeout_secs))
            .timeout(Duration::from_secs(config.server.request_timeout_secs))
            .build()?;
        Ok(AppState {
            classifier,
            config,
            http,
        })
    }
}

/// Build the axum router. `/health` is a liveness probe that touches no
/// backend; anything unmatched returns 404.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state)
}

// ───────────────────────────────────────────────────────────────────────────
// Simple handlers
// ───────────────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

/// The single virtual model id advertised to clients. The router deliberately
/// never exposes its configured backend models.
pub const ADVERTISED_MODEL: &str = "hyper-mcp-router";

/// `GET /v1/models`: advertises the single virtual model, never the backends.
async fn list_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": ADVERTISED_MODEL,
            "object": "model",
            "created": 1_700_000_000,
            "owned_by": "hyper-mcp-router",
        }]
    }))
}

// ───────────────────────────────────────────────────────────────────────────
// Chat completions proxy
// ───────────────────────────────────────────────────────────────────────────

async fn chat_completions(State(state): State<AppState>, raw: Bytes) -> Response {
    let started = Instant::now();

    // 1. Parse; reject non-object bodies with 400.
    let mut body: Value = match serde_json::from_slice(&raw) {
        Ok(v @ Value::Object(_)) => v,
        _ => return bad_request("request body must be a JSON object"),
    };

    // 2. Resolve the required modality set + complexity, then select a backend.
    let classification = classify_or_default(&state.classifier, &body).await;

    let mut required = detect_required_modalities(&body);
    if classification.image_generation {
        required.insert(Modality::ImageOutput);
    }

    // Complexity comes entirely from the windowed classification (recent
    // substantive user turns); there is no separate history heuristic.
    let complexity = classification.complexity;

    let prompt_len = extract_prompt(&body)
        .map(|p| p.chars().count())
        .unwrap_or(0);
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let backend = match state.config.select_model(&required, complexity) {
        Some(m) => m,
        None => {
            tracing::info!(
                modalities = ?required.to_kebab_vec(),
                complexity = ?complexity,
                status = 415u16,
                streaming,
                prompt_chars = prompt_len,
                "no backend covers the required modality set"
            );
            return unsupported_modality_error(&required);
        }
    };

    // 3. Rewrite the model field to the selected backend's configured name.
    body["model"] = Value::String(backend.name.clone());

    // 4. Sanitise: drop `n` unconditionally; forward everything else untouched.
    sanitise(&mut body);

    // Metadata-only routing log (no user content).
    let image_source = if classification.image_generation {
        if looks_like_image_generation(&truncate_prompt(&extract_prompt(&body).unwrap_or_default()))
        {
            Some("lexical")
        } else {
            Some("nli-threshold")
        }
    } else {
        None
    };
    tracing::info!(
        modalities = ?required.to_kebab_vec(),
        image_output_source = image_source,
        complexity = ?complexity,
        model = %backend.name,
        streaming,
        prompt_chars = prompt_len,
        "routing request"
    );

    // 5. Forward to `{base_url}/chat/completions`.
    let url = format!(
        "{}/chat/completions",
        backend.base_url.as_str().trim_end_matches('/')
    );
    // Keyless backends (no configured `api_key`) get no `Authorization` header.
    let mut request = state.http.post(&url).json(&body);
    if let Some(api_key) = &backend.api_key {
        request = request.bearer_auth(api_key);
    }
    let upstream = request.send().await;

    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            let (code, kind) = if e.is_timeout() {
                (StatusCode::GATEWAY_TIMEOUT, "upstream_timeout")
            } else {
                (StatusCode::BAD_GATEWAY, "upstream_unavailable")
            };
            tracing::warn!(error = %e, status = code.as_u16(), model = %backend.name, "upstream request failed");
            return upstream_error(code, kind);
        }
    };

    let status = resp.status();
    let latency_ms = started.elapsed().as_millis();
    tracing::info!(
        model = %backend.name,
        upstream_status = status.as_u16(),
        latency_ms = latency_ms as u64,
        streaming,
        "upstream responded"
    );

    // 6/7. Stream SSE bytes on success; otherwise forward the full body.
    if streaming && status.is_success() {
        stream_passthrough(resp)
    } else {
        buffered_passthrough(resp).await
    }
}

/// Classify a request's complexity from a **window of recent substantive user
/// turns** (see [`build_classification_window`]), mapping any model failure to
/// the balanced default.
///
/// - No user message at all → balanced default (unchanged).
/// - User turns exist but all are trivial (pure chit-chat) → the window is empty,
///   so route the baseline `Fast` *without* running the model.
/// - Otherwise classify the window once. The current turn (last user message)
///   drives the image-generation axis so an old image request in the window
///   can't misroute the present one.
async fn classify_or_default(classifier: &Arc<Classifier>, body: &Value) -> Classification {
    let window = build_classification_window(
        body,
        classifier.trivial_max_words(),
        CLASSIFICATION_CHAR_BUDGET,
    );
    let Some(window) = window else {
        // Nothing substantive to classify.
        return if extract_prompt(body).is_some() {
            // A user message exists but was trivial: pure chit-chat → Fast.
            Classification {
                complexity: ModelTier::Fast,
                image_generation: false,
            }
        } else {
            Classification::balanced_default()
        };
    };

    // Image-generation intent is a property of the *current* turn only.
    let current_turn = extract_prompt(body)
        .map(|p| truncate_prompt(&p))
        .unwrap_or_default();

    // Inference is CPU-bound (one batched forward pass): run it on the blocking
    // pool so it never stalls an async worker.
    let classifier = Arc::clone(classifier);
    match tokio::task::spawn_blocking(move || classifier.classify(&window, &current_turn)).await {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "classification failed; using balanced default");
            Classification::balanced_default()
        }
        Err(e) => {
            tracing::error!(error = %e, "classification task panicked; using balanced default");
            Classification::balanced_default()
        }
    }
}

/// Remove `body["n"]` unconditionally. This is the only field the router
/// strips; all others pass through untouched.
fn sanitise(body: &mut Value) {
    if let Value::Object(map) = body {
        map.remove("n");
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Response construction
// ───────────────────────────────────────────────────────────────────────────

/// Pipe raw upstream SSE bytes straight to the client. No parsing, buffering,
/// or model-field rewriting.
fn stream_passthrough(resp: reqwest::Response) -> Response {
    let status = resp.status();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(resp.bytes_stream()))
        .expect("valid streaming response")
}

/// Forward a non-streaming (or error) upstream response verbatim: same status,
/// same body, preserved content type.
async fn buffered_passthrough(resp: reqwest::Response) -> Response {
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| header::HeaderValue::from_static("application/json"));

    match resp.bytes().await {
        Ok(bytes) => Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(bytes))
            .expect("valid buffered response"),
        Err(e) => {
            tracing::warn!(error = %e, "failed to read upstream response body");
            upstream_error(StatusCode::BAD_GATEWAY, "upstream_body_error")
        }
    }
}

fn bad_request(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": { "message": message, "type": "invalid_request_error" } })),
    )
        .into_response()
}

/// 415 with a minimal JSON body naming the unsatisfiable modality set.
fn unsupported_modality_error(required: &ModalitySet) -> Response {
    let mods = required.to_kebab_vec();
    (
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        Json(json!({
            "error": {
                "message": format!("no configured model supports the required modality set: {mods:?}"),
                "type": "unsupported_modality",
                "modalities": mods,
            }
        })),
    )
        .into_response()
}

fn upstream_error(code: StatusCode, kind: &str) -> Response {
    (
        code,
        Json(json!({
            "error": { "message": "upstream request failed", "type": kind }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── list_models ───────────────────────────────────────────────────────
    #[tokio::test]
    async fn list_models_shape() {
        let v = list_models().await.0;
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "hyper-mcp-router");
    }

    #[tokio::test]
    async fn list_models_advertises_only_the_virtual_model() {
        let v = list_models().await.0;
        // Never the backend models — always the single virtual id.
        assert_eq!(v["data"][0]["id"], "hyper-mcp-router");
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
    }

    // ── modality resolution (route resolution composition) ─────────────────
    // Mirrors the proxy's Route Resolution without needing an ONNX session.
    fn resolve_required(body: &Value, image_generation: bool) -> ModalitySet {
        let mut required = detect_required_modalities(body);
        if image_generation {
            required.insert(Modality::ImageOutput);
        }
        required
    }

    #[test]
    fn resolution_image_input_adds_image_input() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}]});
        let set = resolve_required(&body, false);
        assert!(set.contains(Modality::ImageInput));
        assert!(!set.contains(Modality::ImageOutput));
    }

    #[test]
    fn resolution_audio_in_and_out() {
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}],
            "modalities": ["text", "audio"],
        });
        let set = resolve_required(&body, false);
        assert!(set.contains(Modality::AudioInput));
        assert!(set.contains(Modality::AudioOutput));
    }

    #[test]
    fn resolution_image_output_only_when_signal_fires() {
        let body = json!({"messages": [{"role": "user", "content": "make a logo"}]});
        assert!(!resolve_required(&body, false).contains(Modality::ImageOutput));
        assert!(resolve_required(&body, true).contains(Modality::ImageOutput));
    }

    #[test]
    fn resolution_complexity_never_changes_modalities() {
        let body = json!({"messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        ]});
        let before = resolve_required(&body, false);
        // The complexity tier and the modality set are independent axes.
        let after = resolve_required(&body, false);
        assert_eq!(before, after);
        assert!(before.contains(Modality::ImageInput));
    }

    // ── sanitise ──────────────────────────────────────────────────────────
    #[test]
    fn sanitise_removes_n_only() {
        let mut body = json!({
            "model": "x",
            "n": 4,
            "messages": [],
            "logprobs": true,
            "top_logprobs": 5,
            "custom_unknown_key": {"nested": [1, 2, 3]},
        });
        sanitise(&mut body);
        assert!(body.get("n").is_none());
        assert_eq!(body["logprobs"], true);
        assert_eq!(body["top_logprobs"], 5);
        assert_eq!(body["custom_unknown_key"]["nested"][2], 3);
        assert_eq!(body["model"], "x");
    }
}
