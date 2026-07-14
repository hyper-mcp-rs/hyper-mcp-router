//! HTTP surface and request forwarding.
//!
//! Requests and responses are handled as raw `serde_json::Value` throughout —
//! the router never deserialises into typed OpenAI structs, guaranteeing
//! byte-for-byte passthrough of every field the client sends. Only `messages`,
//! `model`, `stream`, and `n` are ever read; only `model` is rewritten
//! (`n > 1` is rejected up front rather than silently altered).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};

use anyhow::Context;

use crate::classifier::{Classification, ClassifierEngine, ModelTier};
use crate::config::{count_candidates, select_candidate, ModelApiKey, ModelConfig, RouterConfig};
use crate::gcp_auth::{self, AccessTokenCredentials};
use crate::modality::{detect_required_modalities, Modality, ModalitySet};
use crate::prompt::{
    build_classification_window, extract_prompt, has_nonempty_user_text,
    looks_like_image_generation, truncate_prompt,
};

/// Shared, cloneable server state. The classifier is a trait object — the
/// proxy is engine-agnostic; each engine synchronises its own "sessions"
/// internally (see `crate::engines`). `trivial_max_words` is routing policy
/// (window filler pruning), deliberately *not* an engine concern.
#[derive(Clone)]
pub struct AppState {
    pub classifier: Arc<dyn ClassifierEngine>,
    pub config: Arc<RouterConfig>,
    pub http: reqwest::Client,
    pub trivial_max_words: usize,
    /// The runtime model catalogue: each configured model paired with its
    /// resolved auth handle, exactly as they are paired in the config file.
    /// Built once at startup; the proxy routes over THIS, not raw config.
    pub models: Arc<[RoutedModel]>,
}

/// A backend model as the proxy actually routes to it: the static
/// configuration together with its resolved runtime auth. Mirrors the config
/// file, where a `[[models]]` entry carries its own `api_key` — the pairing
/// is never split across parallel structures.
pub struct RoutedModel {
    pub config: ModelConfig,
    pub auth: ModelAuth,
}

impl RoutedModel {
    /// Pair every configured model with its resolved auth handle, discovering
    /// ADC once iff some model asks for it — a missing/broken credential
    /// setup is a boot error here, not a per-request surprise. Configs
    /// without `google-adc` models never touch the host's credential
    /// environment.
    fn resolve_all(config: &RouterConfig) -> anyhow::Result<Vec<RoutedModel>> {
        let mut adc: Option<AccessTokenCredentials> = None;
        config
            .models
            .iter()
            .map(|m| {
                let auth = match &m.api_key {
                    None => ModelAuth::None,
                    Some(ModelApiKey::Static(key)) => ModelAuth::Static(key.clone()),
                    Some(ModelApiKey::GoogleAdc) => {
                        ModelAuth::GoogleAdc(shared_adc(&mut adc, &m.name)?)
                    }
                };
                Ok(RoutedModel {
                    config: m.clone(),
                    auth,
                })
            })
            .collect()
    }
}

/// A routed model's **runtime** authentication handle — the resolved form of
/// [`ModelApiKey`]. The `google-adc` variant *owns* its credential, so "an
/// ADC model without credentials" is unrepresentable; the proxy only ever
/// calls [`bearer`](Self::bearer).
pub enum ModelAuth {
    /// Keyless backend: no `Authorization` header.
    None,
    /// Static secret (plaintext / env / keyring), sent verbatim.
    Static(String),
    /// A current Google OAuth token per request via Application Default
    /// Credentials (cached/refreshed by the auth library). Every `google-adc`
    /// model shares one process-wide credential — ADC identifies the process
    /// principal, so per-model credentials would be identical.
    GoogleAdc(AccessTokenCredentials),
}

impl ModelAuth {
    /// The `Authorization` bearer for one request to this model: `None` for
    /// keyless backends, the static secret verbatim, or a current ADC token.
    /// A token-fetch failure yields a ready error `Response` (boxed — it is
    /// the rare, large arm) rather than limping along unauthenticated.
    async fn bearer(&self, model: &str) -> Result<Option<String>, Box<Response>> {
        match self {
            ModelAuth::None => Ok(None),
            ModelAuth::Static(key) => Ok(Some(key.clone())),
            ModelAuth::GoogleAdc(credentials) => adc_bearer(credentials, model).await.map(Some),
        }
    }
}

/// Memoized ADC discovery for [`RoutedModel::resolve_all`]: the first
/// `google-adc` model triggers discovery (a boot error when unavailable,
/// named after the model that asked); every later one clones the shared
/// credential.
fn shared_adc(
    adc: &mut Option<AccessTokenCredentials>,
    model: &str,
) -> anyhow::Result<AccessTokenCredentials> {
    if let Some(shared) = adc {
        return Ok(shared.clone());
    }
    let discovered = gcp_auth::adc_credentials().with_context(|| {
        format!(
            "model `{model}` sets api_key = {{ source = \"google-adc\" }}, \
             which requires Application Default Credentials"
        )
    })?;
    Ok(adc.insert(discovered).clone())
}

/// The `google-adc` arm of [`ModelAuth::bearer`]: fetch a current token from
/// the shared credential, mapping failure to a 502 response.
async fn adc_bearer(
    credentials: &AccessTokenCredentials,
    model: &str,
) -> Result<String, Box<Response>> {
    gcp_auth::bearer(credentials).await.map_err(|e| {
        tracing::warn!(
            error = %e,
            model,
            status = 502u16,
            "google-adc token fetch failed"
        );
        Box::new(upstream_error(
            StatusCode::BAD_GATEWAY,
            "upstream_auth_failed",
        ))
    })
}

impl AppState {
    /// Build the shared state, constructing the upstream HTTP client with the
    /// configured connect/idle timeouts. The **total** request timeout is
    /// applied per-request, and only to non-streaming requests — a client-wide
    /// total timeout would sever long-lived SSE streams mid-flight. **No
    /// retries** are configured — a retry could trigger duplicate, billable
    /// generations.
    pub fn new(
        classifier: Arc<dyn ClassifierEngine>,
        config: Arc<RouterConfig>,
        trivial_max_words: usize,
    ) -> anyhow::Result<Self> {
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(config.server.connect_timeout_secs));
        // Idle guard: abort when no bytes arrive for this long. This is what
        // bounds a stalled stream, since streams have no total deadline.
        if config.server.stream_idle_timeout_secs > 0 {
            builder =
                builder.read_timeout(Duration::from_secs(config.server.stream_idle_timeout_secs));
        }
        let http = builder.build()?;
        let models: Arc<[RoutedModel]> = RoutedModel::resolve_all(&config)?.into();

        Ok(AppState {
            classifier,
            config,
            http,
            trivial_max_words,
            models,
        })
    }
}

/// Build the axum router. `/health` is a liveness probe that touches no
/// backend; anything unmatched returns 404. The body limit is raised from
/// axum's 2 MB default — base64 image/audio/file payloads are far larger — to
/// the configured `server.max_body_bytes`.
pub fn build_router(state: AppState) -> Router {
    let max_body_bytes = state.config.server.max_body_bytes;
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

// ───────────────────────────────────────────────────────────────────────────
// Simple handlers
// ───────────────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

/// The single virtual model id advertised to clients. The router never
/// *advertises* its configured backend models; note however that upstream
/// response bodies pass through verbatim, so the `model` field of a completion
/// does name the backend that produced it.
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

    // 2. Reject multi-choice requests up front. The router serves exactly one
    //    completion per request; silently honoring only one of `n` requested
    //    choices would be a contract violation, so it is an explicit error
    //    instead of a silent mutation.
    if requests_multiple_choices(&body) {
        return bad_request("`n` > 1 is not supported; send one request per completion");
    }

    // 3/4. Resolve the required modality set and (when it can affect the
    //      choice) the complexity tier — see [`resolve_route`].
    let route = resolve_route(&state, &body).await;
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // The tier only ranks among >= 2 candidates; when skipped, any value selects
    // the sole (or zero) candidate.
    let complexity = route.classified.unwrap_or(ModelTier::Balanced);

    let backend = match select_candidate(
        state.models.iter(),
        |routed| &routed.config,
        &route.required,
        complexity,
    ) {
        Some(routed) => routed,
        None => {
            tracing::info!(
                modalities = ?route.required.to_kebab_vec(),
                complexity = ?route.classified,
                status = 422u16,
                streaming,
                prompt_chars = route.prompt_chars,
                "no backend covers the required modality set"
            );
            return unsupported_modality_error(&route.required);
        }
    };

    // 5. Rewrite the model field to the selected backend's configured name.
    //    Everything else is forwarded untouched.
    body["model"] = Value::String(backend.config.name.clone());

    // Metadata-only routing log (no user content).
    tracing::info!(
        modalities = ?route.required.to_kebab_vec(),
        image_output_source = route.image_source,
        complexity = ?route.classified,
        model = %backend.config.name,
        streaming,
        prompt_chars = route.prompt_chars,
        "routing request"
    );

    // 6. Forward to `{base_url}/chat/completions`. The total request timeout
    //    applies only to non-streaming requests; streams are bounded by the
    //    client-level idle (read) timeout instead.
    let url = format!(
        "{}/chat/completions",
        backend.config.base_url.as_str().trim_end_matches('/')
    );
    let mut request = state.http.post(&url).json(&body);
    if !streaming {
        request = request.timeout(Duration::from_secs(
            state.config.server.request_timeout_secs,
        ));
    }
    // Keyless backends get no `Authorization` header; google-adc backends
    // get a per-request token (see [`ModelAuth::bearer`]).
    match backend.auth.bearer(&backend.config.name).await {
        Ok(Some(token)) => request = request.bearer_auth(token),
        Ok(None) => {}
        Err(error_response) => return *error_response,
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
            tracing::warn!(error = %e, status = code.as_u16(), model = %backend.config.name, "upstream request failed");
            return upstream_error(code, kind);
        }
    };

    let status = resp.status();
    let latency_ms = started.elapsed().as_millis();
    tracing::info!(
        model = %backend.config.name,
        upstream_status = status.as_u16(),
        latency_ms = latency_ms as u64,
        streaming,
        "upstream responded"
    );

    // 7. Stream SSE bytes on success; otherwise forward the full body.
    if streaming && status.is_success() {
        stream_passthrough(resp)
    } else {
        buffered_passthrough(resp).await
    }
}

/// One request's routing decision: the resolved modality set, the complexity
/// tier (`None` when classification was skipped), and metadata for honest
/// logging.
struct RouteResolution {
    required: ModalitySet,
    classified: Option<ModelTier>,
    image_source: Option<&'static str>,
    prompt_chars: usize,
}

/// Resolve a request's route along both axes.
///
/// The modality set is read deterministically first — `image-output` may
/// already be present (explicit `modalities` field); otherwise it is
/// *inferred* (lexical, then NLI) and applied as a soft preference: never at
/// the cost of making the request unroutable.
///
/// Complexity is classified ONLY when it can affect the choice. With <= 1
/// model able to serve the required set there is nothing to rank, so the
/// (serialized) NLI pass is skipped entirely — single-model or
/// single-candidate deployments run zero inference. (That also skips the NLI
/// image-generation signal; the lexical signal still applies.) `classified:
/// None` records the skip, logged honestly rather than as a fabricated tier.
async fn resolve_route(state: &AppState, body: &Value) -> RouteResolution {
    let mut required = detect_required_modalities(body);
    // How much of the current turn the classifier (and the lexical prefilter)
    // sees is model-specific — the engine declares its budget.
    let current_turn = extract_prompt(body)
        .map(|p| truncate_prompt(&p, state.classifier.current_turn_char_budget()))
        .unwrap_or_default();
    let lexical_image = looks_like_image_generation(&current_turn);
    let mut image_source: Option<&'static str> = if required.contains(Modality::ImageOutput) {
        Some("explicit")
    } else {
        None
    };
    // Set once an inferred image intent proves unsatisfiable, so the NLI signal
    // doesn't retry (and re-log) the same dead end.
    let mut image_intent_dropped = false;
    if lexical_image && image_source.is_none() {
        if try_require_image_output(&state.config, &mut required) {
            image_source = Some("lexical");
        } else {
            image_intent_dropped = true;
        }
    }

    let classified: Option<ModelTier> =
        if count_candidates(state.models.iter(), |routed| &routed.config, &required) <= 1 {
            None
        } else {
            let classification =
                classify_or_default(state, body, &current_turn, lexical_image).await;
            if classification.image_generation
                && !required.contains(Modality::ImageOutput)
                && !image_intent_dropped
            {
                if try_require_image_output(&state.config, &mut required) {
                    image_source = Some(if lexical_image {
                        "lexical"
                    } else {
                        "nli-threshold"
                    });
                } else {
                    image_intent_dropped = true;
                }
            }
            Some(classification.complexity)
        };
    if image_intent_dropped {
        tracing::info!(
            modalities = ?required.to_kebab_vec(),
            "inferred image-generation intent, but no backend covers image-output; routing without it"
        );
    }

    RouteResolution {
        required,
        classified,
        image_source,
        prompt_chars: current_turn.chars().count(),
    }
}

/// Whether the request asks for more than one choice (`n > 1`). A non-numeric
/// `n` is left for the upstream to reject.
fn requests_multiple_choices(body: &Value) -> bool {
    body.get("n")
        .and_then(Value::as_f64)
        .is_some_and(|n| n > 1.0)
}

/// Soft-insert the **inferred** `image-output` modality: apply it only if some
/// configured model can still serve the request afterwards. Deterministic
/// modalities are hard constraints (422 when uncovered), but image intent is
/// probabilistic — degrading to a text route beats rejecting a servable
/// request over an inference. Returns whether it was applied.
fn try_require_image_output(config: &RouterConfig, required: &mut ModalitySet) -> bool {
    let mut with_image = *required;
    with_image.insert(Modality::ImageOutput);
    if config.candidate_count(&with_image) > 0 {
        *required = with_image;
        true
    } else {
        false
    }
}

/// Classify a request's complexity from a **window of recent substantive user
/// turns** (see [`build_classification_window`]), mapping any engine failure
/// to the balanced default. Engine-agnostic: the window budget comes from the
/// engine ([`ClassifierEngine::context_char_budget`]), and CPU-bound engines
/// handle their own blocking-thread hand-off.
///
/// - No usable user text at all (no user messages, or only empty/attachment-only
///   content) → balanced default: there is nothing to judge, so don't pretend
///   it's chit-chat.
/// - Non-empty user text exists but all of it is trivial (pure chit-chat) → the
///   window is empty, so route the baseline `Fast` *without* running the model.
/// - Otherwise classify the window once. The current turn (`current_turn`, the
///   last user message, already truncated) drives the image-generation axis so
///   an old image request in the window can't misroute the present one;
///   `lexical_image` is its precomputed lexical signal.
async fn classify_or_default(
    state: &AppState,
    body: &Value,
    current_turn: &str,
    lexical_image: bool,
) -> Classification {
    let window = build_classification_window(
        body,
        state.trivial_max_words,
        state.classifier.context_char_budget(),
    );
    let Some(window) = window else {
        // Nothing substantive to classify.
        return if has_nonempty_user_text(body) {
            // The user did say something, but all of it was trivial: pure
            // chit-chat → Fast.
            Classification {
                complexity: ModelTier::Fast,
                image_generation: false,
            }
        } else {
            Classification::balanced_default()
        };
    };

    match state
        .classifier
        .classify(&window, current_turn, lexical_image)
        .await
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "classification failed; using balanced default");
            Classification::balanced_default()
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Response construction
// ───────────────────────────────────────────────────────────────────────────

/// Whether an upstream response header may be forwarded to the client.
/// Hop-by-hop headers (RFC 9110 §7.6.1) describe the upstream connection, not
/// the payload; payload-framing headers (`content-length`,
/// `transfer-encoding`) are recomputed by axum for the response we build.
/// Everything else — `x-request-id`, rate-limit headers, `retry-after`, … —
/// passes through so clients can implement backoff and tracing.
fn is_end_to_end_header(name: &header::HeaderName) -> bool {
    !matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "content-length"
    )
}

/// Copy every end-to-end header from `src` into `dst` (preserving repeats).
fn copy_end_to_end_headers(src: &header::HeaderMap, dst: &mut header::HeaderMap) {
    for (name, value) in src {
        if is_end_to_end_header(name) {
            dst.append(name.clone(), value.clone());
        }
    }
}

/// Pipe raw upstream SSE bytes straight to the client. No parsing, buffering,
/// or model-field rewriting; upstream end-to-end headers are preserved, with
/// SSE defaults filled in only when the upstream omitted them.
fn stream_passthrough(resp: reqwest::Response) -> Response {
    let status = resp.status();
    let upstream_headers = resp.headers().clone();
    let mut builder = Response::builder().status(status);
    if let Some(dst) = builder.headers_mut() {
        copy_end_to_end_headers(&upstream_headers, dst);
    }
    if !upstream_headers.contains_key(header::CONTENT_TYPE) {
        builder = builder.header(header::CONTENT_TYPE, "text/event-stream");
    }
    if !upstream_headers.contains_key(header::CACHE_CONTROL) {
        builder = builder.header(header::CACHE_CONTROL, "no-cache");
    }
    builder
        .body(Body::from_stream(resp.bytes_stream()))
        .expect("valid streaming response")
}

/// Forward a non-streaming (or error) upstream response verbatim: same status,
/// same body, preserved end-to-end headers.
async fn buffered_passthrough(resp: reqwest::Response) -> Response {
    let status = resp.status();
    let upstream_headers = resp.headers().clone();

    match resp.bytes().await {
        Ok(bytes) => {
            let mut builder = Response::builder().status(status);
            if let Some(dst) = builder.headers_mut() {
                copy_end_to_end_headers(&upstream_headers, dst);
            }
            if !upstream_headers.contains_key(header::CONTENT_TYPE) {
                builder = builder.header(header::CONTENT_TYPE, "application/json");
            }
            builder
                .body(Body::from(bytes))
                .expect("valid buffered response")
        }
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

/// 422 with a minimal JSON body naming the unsatisfiable modality set. (422
/// Unprocessable Content, not 415: the request *media type* is fine — it is
/// the required capability combination no configured backend can serve.)
fn unsupported_modality_error(required: &ModalitySet) -> Response {
    let mods = required.to_kebab_vec();
    (
        StatusCode::UNPROCESSABLE_ENTITY,
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

    // ── model auth handles ──────────────────────────────────────────
    #[tokio::test]
    async fn model_auth_keyless_and_static_bearers() {
        assert_eq!(ModelAuth::None.bearer("m").await.unwrap(), None);
        assert_eq!(
            ModelAuth::Static("sk-static".into())
                .bearer("m")
                .await
                .unwrap(),
            Some("sk-static".to_string())
        );
    }

    #[test]
    fn routed_models_without_google_adc_never_touch_adc() {
        // Static and keyless models resolve to their handles without ADC
        // discovery — this test must pass on hosts with NO Google credential
        // environment at all (hermetic). Each model stays paired with its
        // own auth, exactly as in the config file.
        let keyless = model("keyless", &[Modality::Text]);
        let mut keyed = model("keyed", &[Modality::Text]);
        keyed.api_key = Some(crate::config::ModelApiKey::Static("sk-1".into()));

        let cfg = crate::config::RouterConfig {
            server: Default::default(),
            classifier: Default::default(),
            models: vec![keyless, keyed],
        };
        let routed = RoutedModel::resolve_all(&cfg).expect("no ADC needed");
        assert_eq!(routed[0].config.name, "keyless");
        assert!(matches!(routed[0].auth, ModelAuth::None));
        assert_eq!(routed[1].config.name, "keyed");
        assert!(matches!(&routed[1].auth, ModelAuth::Static(k) if k == "sk-1"));
    }

    // ── n validation ──────────────────────────────────────────────────
    #[test]
    fn multiple_choices_detected_only_above_one() {
        assert!(!requests_multiple_choices(&json!({"messages": []})));
        assert!(!requests_multiple_choices(&json!({"n": 1, "messages": []})));
        assert!(requests_multiple_choices(&json!({"n": 2, "messages": []})));
        assert!(requests_multiple_choices(
            &json!({"n": 4.0, "messages": []})
        ));
        // Non-numeric `n` is not ours to police — the upstream rejects it.
        assert!(!requests_multiple_choices(
            &json!({"n": "4", "messages": []})
        ));
    }

    // ── header passthrough ───────────────────────────────────────────
    #[test]
    fn end_to_end_headers_forwarded_hop_by_hop_dropped() {
        let mut src = header::HeaderMap::new();
        src.insert("x-request-id", "abc-123".parse().unwrap());
        src.insert("x-ratelimit-remaining-tokens", "99".parse().unwrap());
        src.insert(header::RETRY_AFTER, "5".parse().unwrap());
        src.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        src.insert(header::CONNECTION, "keep-alive".parse().unwrap());
        src.insert(header::TRANSFER_ENCODING, "chunked".parse().unwrap());
        src.insert(header::CONTENT_LENGTH, "42".parse().unwrap());

        let mut dst = header::HeaderMap::new();
        copy_end_to_end_headers(&src, &mut dst);

        assert_eq!(dst.get("x-request-id").unwrap(), "abc-123");
        assert_eq!(dst.get("x-ratelimit-remaining-tokens").unwrap(), "99");
        assert_eq!(dst.get(header::RETRY_AFTER).unwrap(), "5");
        assert_eq!(dst.get(header::CONTENT_TYPE).unwrap(), "application/json");
        assert!(dst.get(header::CONNECTION).is_none());
        assert!(dst.get(header::TRANSFER_ENCODING).is_none());
        assert!(dst.get(header::CONTENT_LENGTH).is_none());
    }

    // ── inferred image-output soft insertion ─────────────────────────────
    fn config_with(models: Vec<crate::config::ModelConfig>) -> RouterConfig {
        RouterConfig {
            server: Default::default(),
            classifier: Default::default(),
            models,
        }
    }

    fn model(name: &str, mods: &[Modality]) -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            name: name.to_string(),
            base_url: url::Url::parse("http://x").unwrap(),
            api_key: None,
            tier: ModelTier::Balanced,
            modalities: mods.to_vec(),
        }
    }

    #[test]
    fn inferred_image_output_applied_when_covered() {
        let cfg = config_with(vec![
            model("text", &[Modality::Text]),
            model("image", &[Modality::Text, Modality::ImageOutput]),
        ]);
        let mut required: ModalitySet = [Modality::Text].into_iter().collect();
        assert!(try_require_image_output(&cfg, &mut required));
        assert!(required.contains(Modality::ImageOutput));
    }

    #[test]
    fn inferred_image_output_dropped_when_uncovered() {
        // No image-capable backend: the inferred intent must degrade, keeping
        // the request routable, rather than turn into a 422.
        let cfg = config_with(vec![model("text", &[Modality::Text])]);
        let mut required: ModalitySet = [Modality::Text].into_iter().collect();
        assert!(!try_require_image_output(&cfg, &mut required));
        assert!(!required.contains(Modality::ImageOutput));
    }
}
