//! HTTP surface and request forwarding.
//!
//! Requests and responses are handled as raw `serde_json::Value` throughout —
//! the router never deserialises into typed OpenAI structs, guaranteeing
//! byte-for-byte passthrough of every field the client sends. Only `messages`,
//! `model`, `stream`, `n`, and the completion budget (`max_completion_tokens`
//! / `max_tokens`, for context-fit estimation) are ever read; only `model` is
//! rewritten (`n > 1` is rejected up front rather than silently altered).

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

use secrecy::{ExposeSecret, SecretString};

use crate::classifier::{Classification, ClassifierEngine, EngineRoster, ModelTier};
use crate::config::{ModelApiKey, ModelConfig, RouterConfig};
use crate::gcp_auth::{self, AccessTokenCredentials};
use crate::modality::{detect_required_modalities, Modality, ModalitySet};
use crate::prompt::{
    build_classification_window, estimate_request_tokens, extract_prompt, has_nonempty_user_text,
    looks_like_image_generation, truncate_prompt,
};
use crate::selection::{count_candidates, select_candidate};
use crate::telemetry::{attr, Metrics};
use crate::usage::report_upstream_usage;

use tracing::Instrument;

/// Shared, cloneable server state. The classifiers are trait objects on a
/// capacity ladder ([`EngineRoster`]) — the proxy is engine-agnostic; each
/// engine synchronises its own "sessions" internally (see `crate::engines`).
/// `trivial_max_words` is routing policy (window filler pruning),
/// deliberately *not* an engine concern.
#[derive(Clone)]
pub struct AppState {
    pub classifiers: EngineRoster,
    pub config: Arc<RouterConfig>,
    pub http: reqwest::Client,
    pub trivial_max_words: usize,
    /// The runtime model catalogue: each configured model paired with its
    /// resolved auth handle, exactly as they are paired in the config file.
    /// Built once at startup; the proxy routes over THIS, not raw config.
    pub models: Arc<[RoutedModel]>,
    /// OpenTelemetry instruments, bound once at startup. No-op recorders
    /// unless `[telemetry]` enabled the metrics signal (see
    /// [`crate::telemetry::Metrics`]).
    pub metrics: Arc<Metrics>,
    /// Whether upstream `usage` objects should be parsed for metrics even
    /// when debug logging is off — true iff `[telemetry]` metrics are on.
    pub usage_metrics: bool,
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
                    // Keyring references are resolved by `config::load`
                    // (RouterConfig::resolve_secrets) before the proxy is
                    // built; one surviving here means a caller skipped it.
                    Some(ModelApiKey::Keyring { service, user }) => anyhow::bail!(
                        "model `{}` still carries an unresolved keyring reference \
                         (service={service}, user={user}); call \
                         RouterConfig::resolve_secrets after parsing",
                        m.name
                    ),
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
    /// Static secret (plaintext / env / keyring), sent verbatim. Held
    /// redacted; the only exposure is building the `Authorization` header.
    Static(SecretString),
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
            ModelAuth::Static(key) => Ok(Some(key.expose_secret().to_owned())),
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
        classifiers: EngineRoster,
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
        let usage_metrics = config.telemetry.as_ref().is_some_and(|t| t.metrics);

        Ok(AppState {
            classifiers,
            config,
            http,
            trivial_max_words,
            models,
            metrics: Arc::new(Metrics::new()),
            usage_metrics,
        })
    }

    /// Convenience for a single-engine deployment (and tests): wrap one
    /// engine in a trivial roster.
    pub fn with_single_engine(
        classifier: Arc<dyn ClassifierEngine>,
        config: Arc<RouterConfig>,
        trivial_max_words: usize,
    ) -> anyhow::Result<Self> {
        Self::new(
            EngineRoster::new(vec![classifier])?,
            config,
            trivial_max_words,
        )
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

/// Span-wrapping entry point: opens the per-request `chat_completions` span
/// (parented to the caller's `traceparent` when one arrives and telemetry is
/// on) and runs the routing logic inside it. Every field is declared here as
/// [`tracing::field::Empty`] and recorded as routing progresses, so the
/// exported span carries the full decision — model selection, prompt
/// categorization, sizes, and token usage — exactly once per request.
async fn chat_completions(
    State(state): State<AppState>,
    request_headers: header::HeaderMap,
    raw: Bytes,
) -> Response {
    use tracing::field::Empty;
    let span = tracing::info_span!(
        "chat_completions",
        otel.kind = "server",
        model = Empty,
        complexity = Empty,
        classifier_engine = Empty,
        modalities = Empty,
        streaming = Empty,
        prompt_chars = Empty,
        window_chars = Empty,
        estimated_tokens = Empty,
        upstream_status = Empty,
        prompt_tokens = Empty,
        completion_tokens = Empty,
        total_tokens = Empty,
    );
    // Join the caller's trace. Errs harmlessly when no OTel layer is
    // installed (telemetry off) — the span still scopes logging.
    let _ = tracing_opentelemetry::OpenTelemetrySpanExt::set_parent(
        &span,
        crate::telemetry::extract_context(&request_headers),
    );
    chat_completions_inner(state, raw).instrument(span).await
}

async fn chat_completions_inner(state: AppState, raw: Bytes) -> Response {
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

    // Record the routing axes on the request span (sizes, categorization,
    // engine) — exported when telemetry is on, and echoed by the fmt layer's
    // span-close event either way.
    let span = tracing::Span::current();
    span.record("streaming", streaming);
    span.record("prompt_chars", route.prompt_chars as u64);
    span.record("window_chars", route.window_chars() as u64);
    span.record("estimated_tokens", route.estimated_tokens);
    span.record("classifier_engine", route.classifier_engine);
    span.record(
        "modalities",
        format!("{:?}", route.required.to_kebab_vec()).as_str(),
    );
    if let Some(tier) = route.classified {
        span.record("complexity", format!("{tier:?}").as_str());
    }

    // The tier only ranks among >= 2 candidates; when skipped, any value selects
    // the sole (or zero) candidate.
    let complexity = route.classified.unwrap_or(ModelTier::Balanced);

    let backend = match select_candidate(
        state.models.iter(),
        |routed| &routed.config,
        &route.required,
        complexity,
        route.estimated_tokens,
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
            record_request_metrics(&state.metrics, started, None, &route, streaming, "422");
            return unsupported_modality_error(&route.required);
        }
    };
    span.record("model", backend.config.name.as_str());

    // Context fit is a strong preference, not a hard gate: when even the
    // largest covering window is (by estimate) too small, the request is
    // still forwarded there — the estimate is a heuristic and the upstream
    // is the authority — but the overflow is logged honestly.
    if !backend.config.fits_context(route.estimated_tokens) {
        tracing::warn!(
            model = %backend.config.name,
            estimated_tokens = route.estimated_tokens,
            context_window = backend.config.context_window.get(),
            "request exceeds every capable backend's declared context window; \
             forwarding to the largest as a best effort"
        );
    }

    // 5. Rewrite the model field to the selected backend's configured name.
    //    Everything else is forwarded untouched.
    body["model"] = Value::String(backend.config.name.clone());

    // Metadata-only routing log (no user content).
    tracing::info!(
        modalities = ?route.required.to_kebab_vec(),
        image_output_source = route.image_source,
        complexity = ?route.classified,
        classifier_engine = route.classifier_engine,
        model = %backend.config.name,
        streaming,
        prompt_chars = route.prompt_chars,
        estimated_tokens = route.estimated_tokens,
        "routing request"
    );

    // Config-gated companion carrying the full prompt text — see
    // [`log_completion_request`].
    log_completion_request(
        &route,
        &backend.config.name,
        streaming,
        state.config.logging.log_prompts,
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

    // Client span for the upstream call; `traceparent` is injected from it so
    // the backend's own telemetry joins this trace. For streaming responses
    // the span covers time-to-response-headers (the body outlives it — the
    // request span carries the stream).
    let upstream_span = tracing::info_span!(
        "upstream_request",
        otel.kind = "client",
        model = %backend.config.name,
        http.response.status_code = tracing::field::Empty,
    );
    let mut trace_headers = header::HeaderMap::new();
    crate::telemetry::inject_context(&upstream_span, &mut trace_headers);
    if !trace_headers.is_empty() {
        request = request.headers(trace_headers);
    }
    let upstream = request.send().instrument(upstream_span.clone()).await;

    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            let (code, kind) = if e.is_timeout() {
                (StatusCode::GATEWAY_TIMEOUT, "upstream_timeout")
            } else {
                (StatusCode::BAD_GATEWAY, "upstream_unavailable")
            };
            tracing::warn!(error = %e, status = code.as_u16(), model = %backend.config.name, "upstream request failed");
            record_request_metrics(
                &state.metrics,
                started,
                Some(&backend.config.name),
                &route,
                streaming,
                kind,
            );
            return upstream_error(code, kind);
        }
    };

    let status = resp.status();
    upstream_span.record("http.response.status_code", status.as_u16());
    span.record("upstream_status", status.as_u16());
    let latency_ms = started.elapsed().as_millis();
    tracing::info!(
        model = %backend.config.name,
        upstream_status = status.as_u16(),
        latency_ms = latency_ms as u64,
        streaming,
        "upstream responded"
    );
    record_request_metrics(
        &state.metrics,
        started,
        Some(&backend.config.name),
        &route,
        streaming,
        status.as_str(),
    );

    // 7. Stream SSE bytes on success; otherwise forward the full body.
    if streaming && status.is_success() {
        stream_passthrough(resp)
    } else {
        let usage_metrics = state.usage_metrics.then_some(state.metrics.as_ref());
        buffered_passthrough(
            resp,
            &backend.config.name,
            route.estimated_tokens,
            usage_metrics,
        )
        .await
    }
}

/// Record the per-request counter and duration histogram. `status` is the
/// upstream HTTP status for forwarded requests, or a router outcome kind
/// (`422`, `upstream_timeout`, `upstream_unavailable`). Pre-routing 400s are
/// deliberately not counted — they never reached a routing decision.
fn record_request_metrics(
    metrics: &Metrics,
    started: Instant,
    model: Option<&str>,
    route: &RouteResolution,
    streaming: bool,
    status: &str,
) {
    let attrs = [
        attr("model", model.unwrap_or("none").to_string()),
        attr(
            "complexity",
            route
                .classified
                .map_or_else(|| "none".to_string(), |t| format!("{t:?}")),
        ),
        attr("streaming", streaming),
        attr("status", status.to_string()),
    ];
    metrics.requests.add(1, &attrs);
    metrics
        .request_duration
        .record(started.elapsed().as_secs_f64(), &attrs);
}

/// One request's routing decision: the resolved modality set, the complexity
/// tier (`None` when classification was skipped), and metadata for honest
/// logging.
struct RouteResolution {
    required: ModalitySet,
    classified: Option<ModelTier>,
    image_source: Option<&'static str>,
    /// The ENTIRE current-turn prompt (the last user message's text),
    /// untruncated — extracted once during resolution and reused by the
    /// debug-level request log. Empty when no user message exists.
    prompt: String,
    /// The compiled classification window exactly as fed to the complexity
    /// classifier: substantive user turns newest→oldest under the roster's
    /// top char budget (see [`build_classification_window`]). `None` when
    /// nothing substantive existed to classify. Kept for the debug-level
    /// request log — built once during resolution either way.
    window: Option<String>,
    /// Chars of the current turn AS THE CLASSIFIER SAW IT — i.e. after
    /// truncation to the selected engine's per-turn budget; may be shorter
    /// than `prompt`.
    prompt_chars: usize,
    /// Estimated context-window occupancy of the FULL request (all message
    /// text plus the requested completion budget), in tokens — see
    /// [`estimate_request_tokens`]. Candidates whose declared
    /// `context_window` cannot fit this are avoided.
    estimated_tokens: u64,
    /// The capacity-ladder engine selected for this request (by window
    /// length). Recorded even when classification is skipped — `classified:
    /// None` already says the engine did not run.
    classifier_engine: &'static str,
}

impl RouteResolution {
    /// Size of the compiled classification window in chars (0 when nothing
    /// substantive existed to classify).
    fn window_chars(&self) -> usize {
        self.window.as_deref().map_or(0, |w| w.chars().count())
    }
}

/// Prompt-content request log: the ENTIRE current-turn prompt (untruncated)
/// AND the compiled classification window — the pruned multi-turn text the
/// complexity classifier actually consumed — alongside the model selection
/// and routing metrics. Emitted at **info** level, gated by the
/// `[logging] log_prompts` config flag (default off) rather than a log
/// level: whether user content may appear in logs is a deployment policy,
/// decoupled from `RUST_LOG` verbosity — a deployment can log every prompt
/// without debug noise, or run full debug diagnostics without ever logging
/// a prompt. The unconditional "routing request" event stays metadata-only.
/// Both texts ride on [`RouteResolution`], produced once during route
/// resolution — nothing here re-reads the request body.
fn log_completion_request(route: &RouteResolution, model: &str, streaming: bool, enabled: bool) {
    if !enabled {
        return;
    }
    tracing::info!(
        model,
        modalities = ?route.required.to_kebab_vec(),
        complexity = ?route.classified,
        classifier_engine = route.classifier_engine,
        streaming,
        prompt_chars = route.prompt_chars,
        estimated_tokens = route.estimated_tokens,
        prompt = %route.prompt,
        classification_window = route.window.as_deref().unwrap_or(""),
        "completion request"
    );
}

/// Resolve a request's route along both axes.
///
/// The modality set is read deterministically first — `image-output` may
/// already be present (explicit `modalities` field); otherwise it is
/// *inferred* (lexical, then NLI) and applied as a soft preference: never at
/// the cost of making the request unroutable.
///
/// Complexity is classified ONLY when it can affect the choice. With <= 1
/// model able to serve the required set *within its context window* there is
/// nothing to rank, so the (serialized) NLI pass is skipped entirely —
/// single-model or single-candidate deployments run zero inference. (That
/// also skips the NLI image-generation signal; the lexical signal still
/// applies.) `classified: None` records the skip, logged honestly rather
/// than as a fabricated tier.
async fn resolve_route(state: &AppState, body: &Value) -> RouteResolution {
    let mut required = detect_required_modalities(body);
    // Estimated context occupancy of the FULL request — the third routing
    // axis besides modalities and complexity. Computed once and reused by
    // every candidate check below.
    let estimated_tokens = estimate_request_tokens(body);
    // Build the classification window ONCE, at the top of the capacity
    // ladder; its length then selects the smallest engine whose budget covers
    // it (see [`EngineRoster::select`]). Only a window that exceeds even the
    // top budget is truncated — and a single-engine roster reproduces the
    // previous one-engine behaviour exactly.
    let window = build_classification_window(
        body,
        state.trivial_max_words,
        state.classifiers.max_context_char_budget(),
    );
    let window_chars = window.as_deref().map_or(0, |w| w.chars().count());
    let engine = state.classifiers.select(window_chars);
    // The current turn is extracted ONCE: kept whole (in the resolution, for
    // the debug request log) and truncated to the selected engine's per-turn
    // budget for classification — how much of the turn the classifier (and
    // the lexical prefilter) sees is model-specific.
    let prompt = extract_prompt(body).unwrap_or_default();
    let current_turn = truncate_prompt(&prompt, engine.current_turn_char_budget());
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
        if try_require_image_output(&state.config, &mut required, estimated_tokens) {
            image_source = Some("lexical");
        } else {
            image_intent_dropped = true;
        }
    }

    // Size metrics for every request — the distribution of prompt/window/
    // context sizes is a routing-behavior signal independent of outcome.
    state
        .metrics
        .prompt_chars
        .record(current_turn.chars().count() as u64, &[]);
    state.metrics.window_chars.record(window_chars as u64, &[]);
    state.metrics.estimated_tokens.record(estimated_tokens, &[]);

    let candidates = count_candidates(
        state.models.iter(),
        |routed| &routed.config,
        &required,
        estimated_tokens,
    );
    let classified: Option<ModelTier> = if candidates <= 1 {
        tracing::debug!(
            candidates,
            "classification skipped: at most one candidate serves the request; nothing to rank"
        );
        state
            .metrics
            .classification_skipped
            .add(1, &[attr("reason", "single_candidate")]);
        None
    } else {
        if window.is_none() {
            // The model-free fast paths (see [`classify_without_model`]).
            let reason = if has_nonempty_user_text(body) {
                "trivial_fast_path"
            } else {
                "no_user_text"
            };
            state
                .metrics
                .classification_skipped
                .add(1, &[attr("reason", reason)]);
        }
        let classify_started = Instant::now();
        let classify_span = tracing::info_span!(
            "classify",
            engine = engine.name(),
            window_chars,
            complexity = tracing::field::Empty,
        );
        let classification = classify_or_default(
            engine.as_ref(),
            body,
            window.as_deref(),
            &current_turn,
            lexical_image,
        )
        .instrument(classify_span.clone())
        .await;
        classify_span.record(
            "complexity",
            format!("{:?}", classification.complexity).as_str(),
        );
        if window.is_some() {
            // The engine genuinely ran: record inference timing and the
            // categorization outcome (fast-path decisions are counted above).
            state.metrics.classification_duration.record(
                classify_started.elapsed().as_secs_f64(),
                &[attr("engine", engine.name())],
            );
            state.metrics.classified.add(
                1,
                &[
                    attr("complexity", format!("{:?}", classification.complexity)),
                    attr("engine", engine.name()),
                ],
            );
        }
        if classification.image_generation
            && !required.contains(Modality::ImageOutput)
            && !image_intent_dropped
        {
            if try_require_image_output(&state.config, &mut required, estimated_tokens) {
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
        prompt,
        window,
        estimated_tokens,
        classifier_engine: engine.name(),
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
/// configured model can still serve the request afterwards — covering its
/// modalities AND fitting it in the model's context window (an image model
/// whose window the request overflows is a guaranteed upstream failure, so
/// the inferred intent degrades to a text route instead). Deterministic
/// modalities are hard constraints (422 when uncovered), but image intent is
/// probabilistic — degrading to a text route beats rejecting a servable
/// request over an inference. Returns whether it was applied.
fn try_require_image_output(
    config: &RouterConfig,
    required: &mut ModalitySet,
    estimated_tokens: u64,
) -> bool {
    let mut with_image = *required;
    with_image.insert(Modality::ImageOutput);
    if count_candidates(config.models.iter(), |m| m, &with_image, estimated_tokens) > 0 {
        *required = with_image;
        true
    } else {
        false
    }
}

/// Classify a request's complexity from a **window of recent substantive user
/// turns** (already built by [`resolve_route`] at the capacity ladder's top
/// budget — the selected engine's own budget is guaranteed to cover it, see
/// [`EngineRoster::select`]). Engine-agnostic; CPU-bound engines handle
/// their own blocking-thread hand-off. An engine failure maps to the
/// balanced default — deliberately no retry on a lower rung, which would add
/// tail latency and blur the failure signal.
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
    engine: &dyn ClassifierEngine,
    body: &Value,
    window: Option<&str>,
    current_turn: &str,
    lexical_image: bool,
) -> Classification {
    let Some(window) = window else {
        // Nothing substantive to classify.
        return classify_without_model(body);
    };

    match engine.classify(window, current_turn, lexical_image).await {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                engine = engine.name(),
                error = %e,
                "classification failed; using balanced default"
            );
            Classification::balanced_default()
        }
    }
}

/// The model-free classification for an empty window, with the reason logged
/// at debug so an evaluation log distinguishes "the classifier judged this
/// Fast" from "the fast path fired without inference".
fn classify_without_model(body: &Value) -> Classification {
    if has_nonempty_user_text(body) {
        // The user did say something, but all of it was trivial: pure
        // chit-chat → Fast.
        tracing::debug!(
            "classification skipped: user text is all trivial filler; routing Fast without inference"
        );
        Classification {
            complexity: ModelTier::Fast,
            image_generation: false,
        }
    } else {
        tracing::debug!("classification skipped: no user text to judge; using balanced default");
        Classification::balanced_default()
    }
}

// ───────────────────────────────────────────────────────────────────────
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
/// same body, preserved end-to-end headers. Successful responses additionally
/// feed the usage report (see [`report_upstream_usage`]) — the body is
/// already fully buffered here, so the peek is free.
async fn buffered_passthrough(
    resp: reqwest::Response,
    model: &str,
    estimated_tokens: u64,
    usage_metrics: Option<&Metrics>,
) -> Response {
    let status = resp.status();
    let upstream_headers = resp.headers().clone();

    match resp.bytes().await {
        Ok(bytes) => {
            if status.is_success() {
                report_upstream_usage(model, estimated_tokens, &bytes, usage_metrics);
            }
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
            logging: Default::default(),
            telemetry: None,
            models: vec![keyless, keyed],
        };
        let routed = RoutedModel::resolve_all(&cfg).expect("no ADC needed");
        assert_eq!(routed[0].config.name, "keyless");
        assert!(matches!(routed[0].auth, ModelAuth::None));
        assert_eq!(routed[1].config.name, "keyed");
        assert!(matches!(&routed[1].auth, ModelAuth::Static(k) if k.expose_secret() == "sk-1"));
    }

    #[test]
    fn routed_models_reject_unresolved_keyring_references() {
        // An unresolved keyring marker surviving to proxy construction is a
        // programming error (config::load resolves them); it must be a loud
        // boot failure naming the model, never a silent keyless backend.
        let mut keyed = model("keyed", &[Modality::Text]);
        keyed.api_key = Some(crate::config::ModelApiKey::Keyring {
            service: "svc".into(),
            user: "u".into(),
        });
        let cfg = crate::config::RouterConfig {
            server: Default::default(),
            classifier: Default::default(),
            logging: Default::default(),
            telemetry: None,
            models: vec![keyed],
        };
        // (`match` rather than `unwrap_err`: `RoutedModel` has no `Debug` —
        // deliberately, it carries a resolved credential.)
        let err = match RoutedModel::resolve_all(&cfg) {
            Ok(_) => panic!("unresolved keyring reference must fail resolution"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("keyed") && err.contains("unresolved"),
            "got: {err}"
        );
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

    // ── debug completion-request logging ──────────────────────
    use crate::test_support::captured_log;

    fn route_resolution(
        classified: Option<ModelTier>,
        prompt: &str,
        window: Option<&str>,
    ) -> RouteResolution {
        RouteResolution {
            required: [Modality::Text].into_iter().collect(),
            classified,
            image_source: None,
            prompt: prompt.to_string(),
            window: window.map(str::to_string),
            prompt_chars: 12,
            estimated_tokens: 34,
            classifier_engine: "embedded-nli",
        }
    }

    #[test]
    fn completion_request_logged_with_full_prompt_when_enabled() {
        // With `[logging] log_prompts = true`, the event fires at INFO — a
        // deployment can log every prompt without enabling debug noise.
        let out = captured_log(tracing::Level::INFO, || {
            log_completion_request(
                &route_resolution(
                    Some(ModelTier::Frontier),
                    "summarize the plot of Moby-Dick in one sentence",
                    Some("earlier substantive turn\nsummarize the plot of Moby-Dick"),
                ),
                "backend-model",
                true,
                true,
            );
        });
        assert!(out.contains("completion request"), "got: {out}");
        // The ENTIRE current-turn prompt, verbatim.
        assert!(
            out.contains("summarize the plot of Moby-Dick in one sentence"),
            "got: {out}"
        );
        // The compiled window the complexity classifier consumed, verbatim.
        assert!(out.contains("earlier substantive turn"), "got: {out}");
        // Model selection and metrics ride along on the same event.
        assert!(out.contains("backend-model"), "got: {out}");
        assert!(out.contains("Frontier"), "got: {out}");
        assert!(out.contains("estimated_tokens=34"), "got: {out}");
    }

    #[test]
    fn completion_request_log_survives_missing_user_message() {
        // No user turn: resolution carries an empty prompt and no window; the
        // event still fires.
        let out = captured_log(tracing::Level::INFO, || {
            log_completion_request(
                &route_resolution(None, "", None),
                "backend-model",
                false,
                true,
            );
        });
        assert!(out.contains("completion request"), "got: {out}");
    }

    #[test]
    fn prompt_stays_out_of_logs_when_disabled() {
        // Privacy guard: with `log_prompts = false` (the default) the event
        // (and with it, the user content) must not be emitted at ANY log
        // level — the config flag is the only opt-in, not `RUST_LOG`.
        let out = captured_log(tracing::Level::TRACE, || {
            log_completion_request(
                &route_resolution(
                    Some(ModelTier::Fast),
                    "user-content-that-must-not-leak",
                    Some("window-content-that-must-not-leak"),
                ),
                "backend-model",
                false,
                false,
            );
        });
        assert!(
            !out.contains("user-content-that-must-not-leak"),
            "got: {out}"
        );
        assert!(
            !out.contains("window-content-that-must-not-leak"),
            "got: {out}"
        );
        assert!(!out.contains("completion request"), "got: {out}");
    }

    // ── upstream-usage reporting: see `crate::usage::tests` ────────────

    // ── classification skip reasons ─────────────────────────────
    #[test]
    fn trivial_only_text_routes_fast_and_logs_the_reason() {
        let body = json!({"messages": [{"role": "user", "content": "ok thanks"}]});
        let mut result = Classification::balanced_default();
        let out = captured_log(tracing::Level::DEBUG, || {
            result = classify_without_model(&body);
        });
        assert_eq!(result.complexity, ModelTier::Fast);
        assert!(!result.image_generation);
        assert!(
            out.contains("trivial filler; routing Fast without inference"),
            "got: {out}"
        );
    }

    #[test]
    fn no_user_text_defaults_balanced_and_logs_the_reason() {
        let body = json!({"messages": [{"role": "system", "content": "be terse"}]});
        let mut result = Classification {
            complexity: ModelTier::Fast,
            image_generation: true,
        };
        let out = captured_log(tracing::Level::DEBUG, || {
            result = classify_without_model(&body);
        });
        assert_eq!(result, Classification::balanced_default());
        assert!(
            out.contains("no user text to judge; using balanced default"),
            "got: {out}"
        );
    }

    // ── inferred image-output soft insertion ─────────────────────
    fn config_with(models: Vec<crate::config::ModelConfig>) -> RouterConfig {
        RouterConfig {
            server: Default::default(),
            classifier: Default::default(),
            logging: Default::default(),
            telemetry: None,
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
            // Effectively unbounded: these tests exercise the modality axis.
            context_window: std::num::NonZeroU64::new(u64::MAX).unwrap(),
        }
    }

    #[test]
    fn inferred_image_output_applied_when_covered() {
        let cfg = config_with(vec![
            model("text", &[Modality::Text]),
            model("image", &[Modality::Text, Modality::ImageOutput]),
        ]);
        let mut required: ModalitySet = [Modality::Text].into_iter().collect();
        assert!(try_require_image_output(&cfg, &mut required, 0));
        assert!(required.contains(Modality::ImageOutput));
    }

    #[test]
    fn inferred_image_output_dropped_when_uncovered() {
        // No image-capable backend: the inferred intent must degrade, keeping
        // the request routable, rather than turn into a 422.
        let cfg = config_with(vec![model("text", &[Modality::Text])]);
        let mut required: ModalitySet = [Modality::Text].into_iter().collect();
        assert!(!try_require_image_output(&cfg, &mut required, 0));
        assert!(!required.contains(Modality::ImageOutput));
    }

    #[test]
    fn inferred_image_output_dropped_when_no_image_model_fits() {
        // The only image-capable backend has an 8k window; a request estimated
        // at 100k tokens would be a guaranteed upstream failure there, so the
        // soft image intent degrades to a text route instead.
        let mut image = model("image", &[Modality::Text, Modality::ImageOutput]);
        image.context_window = std::num::NonZeroU64::new(8_000).unwrap();
        let cfg = config_with(vec![model("text", &[Modality::Text]), image]);

        let mut required: ModalitySet = [Modality::Text].into_iter().collect();
        assert!(!try_require_image_output(&cfg, &mut required, 100_000));
        assert!(!required.contains(Modality::ImageOutput));
        // The same request at a small size keeps the inferred intent.
        assert!(try_require_image_output(&cfg, &mut required, 500));
        assert!(required.contains(Modality::ImageOutput));
    }
}
