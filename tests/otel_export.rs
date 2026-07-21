//! OpenTelemetry export integration test.
//!
//! Spins up a **mock OTLP collector** (an in-process HTTP server accepting
//! protobuf `POST /v1/traces` and `/v1/metrics`, exactly what a local
//! collector sidecar exposes on :4318) plus a mock upstream backend, runs a
//! live router with a `[telemetry]` table pointing at the mock collector, and
//! asserts the whole telemetry contract over the wire:
//!
//! - the request exports the span tree (`chat_completions` → `classify`,
//!   `upstream_request`) with the routing decision as attributes — model
//!   selection, prompt categorization, prompt/window sizes, token usage;
//! - an inbound W3C `traceparent` is honored (exported spans join the
//!   caller's trace) and propagated to the upstream backend;
//! - metrics arrive with the documented `hyper_mcp_router.*` names;
//! - providers flush on shutdown.
//!
//! Everything lives in ONE test function: telemetry installs process globals
//! (tracing subscriber, meter provider, propagator), which parallel test
//! functions must not race on. The classifier is a fake (classification
//! correctness is `api_routing.rs`'s job; this test is about export).

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::StatusCode,
    response::Response,
    routing::post,
    Router,
};
use prost::Message;
use serde_json::json;

use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::any_value;
use opentelemetry_proto::tonic::trace::v1::Span;

use hyper_mcp_router::classifier::{Classification, ClassifierEngine, ModelTier};
use hyper_mcp_router::config;
use hyper_mcp_router::prompt::DEFAULT_TRIVIAL_MAX_WORDS;
use hyper_mcp_router::proxy::{build_router, AppState};
use hyper_mcp_router::telemetry;

// ───────────────────────────────────────────────────────────────────────────
// Mock OTLP collector: records raw protobuf export payloads per signal.
// ───────────────────────────────────────────────────────────────────────────

type Payloads = Arc<Mutex<Vec<Bytes>>>;

async fn spawn_mock_collector() -> (SocketAddr, Payloads, Payloads) {
    let traces: Payloads = Arc::new(Mutex::new(Vec::new()));
    let metrics: Payloads = Arc::new(Mutex::new(Vec::new()));

    async fn record(State(sink): State<Payloads>, body: Bytes) -> Response {
        sink.lock().unwrap().push(body);
        // An empty body decodes as the default Export*ServiceResponse, which
        // is exactly what a healthy collector returns.
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/x-protobuf")
            .body(Body::empty())
            .unwrap()
    }

    let app = Router::new()
        .route("/v1/traces", post(record).with_state(traces.clone()))
        .route("/v1/metrics", post(record).with_state(metrics.clone()));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock collector");
    let addr = listener.local_addr().expect("mock collector addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("collector serve");
    });
    (addr, traces, metrics)
}

// ───────────────────────────────────────────────────────────────────────────
// Mock upstream backend: records the traceparent it received, returns usage.
// ───────────────────────────────────────────────────────────────────────────

type Traceparents = Arc<Mutex<Vec<Option<String>>>>;

async fn spawn_mock_upstream() -> (SocketAddr, Traceparents) {
    let seen: Traceparents = Arc::new(Mutex::new(Vec::new()));

    async fn chat(State(seen): State<Traceparents>, headers: axum::http::HeaderMap) -> Response {
        seen.lock().unwrap().push(
            headers
                .get("traceparent")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned),
        );
        let body = json!({
            "id": "mock-cmpl",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48},
        });
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    let app = Router::new().route("/chat/completions", post(chat).with_state(seen.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock upstream");
    let addr = listener.local_addr().expect("mock upstream addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("upstream serve");
    });
    (addr, seen)
}

// ───────────────────────────────────────────────────────────────────────────
// Fake classifier: deterministic Frontier, no ONNX. Export is under test,
// not classification.
// ───────────────────────────────────────────────────────────────────────────

struct FakeFrontierEngine;

#[async_trait::async_trait]
impl ClassifierEngine for FakeFrontierEngine {
    fn name(&self) -> &'static str {
        "fake-frontier"
    }
    fn is_local(&self) -> bool {
        true
    }
    fn context_char_budget(&self) -> usize {
        4000
    }
    fn current_turn_char_budget(&self) -> usize {
        400
    }
    async fn classify(
        &self,
        _complexity_premise: &str,
        _image_premise: &str,
        _lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        Ok(Classification {
            complexity: ModelTier::Frontier,
            image_generation: false,
        })
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Config: three text tiers (so classification runs) + telemetry to the mock.
// ───────────────────────────────────────────────────────────────────────────

fn config_toml(upstream: SocketAddr, collector: SocketAddr) -> String {
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[telemetry]
otlp_endpoint = "http://{collector}"
service_name = "hyper-mcp-router-test"

[[models]]
name           = "fast-text"
base_url       = "http://{upstream}"
type           = "fast"
modalities     = ["text"]
context_window = 128000

[[models]]
name           = "balanced-text"
base_url       = "http://{upstream}"
type           = "balanced"
modalities     = ["text"]
context_window = 128000

[[models]]
name           = "frontier-text"
base_url       = "http://{upstream}"
type           = "frontier"
modalities     = ["text"]
context_window = 128000
"#
    )
}

// ───────────────────────────────────────────────────────────────────────────
// Protobuf helpers
// ───────────────────────────────────────────────────────────────────────────

fn decode_spans(payloads: &Payloads) -> Vec<Span> {
    payloads
        .lock()
        .unwrap()
        .iter()
        .map(|bytes| ExportTraceServiceRequest::decode(bytes.as_ref()).expect("decode traces"))
        .flat_map(|req| req.resource_spans)
        .flat_map(|rs| rs.scope_spans)
        .flat_map(|ss| ss.spans)
        .collect()
}

fn decode_metric_names(payloads: &Payloads) -> HashSet<String> {
    payloads
        .lock()
        .unwrap()
        .iter()
        .map(|bytes| ExportMetricsServiceRequest::decode(bytes.as_ref()).expect("decode metrics"))
        .flat_map(|req| req.resource_metrics)
        .flat_map(|rm| rm.scope_metrics)
        .flat_map(|sm| sm.metrics)
        .map(|m| m.name)
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The string value of a span attribute, if present with that type.
fn string_attr(span: &Span, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key == key)
        .and_then(|kv| kv.value.as_ref())
        .and_then(|v| v.value.as_ref())
        .and_then(|v| match v {
            any_value::Value::StringValue(s) => Some(s.clone()),
            _ => None,
        })
}

/// The integer value of a span attribute, if present with that type.
fn int_attr(span: &Span, key: &str) -> Option<i64> {
    span.attributes
        .iter()
        .find(|kv| kv.key == key)
        .and_then(|kv| kv.value.as_ref())
        .and_then(|v| v.value.as_ref())
        .and_then(|v| match v {
            any_value::Value::IntValue(i) => Some(*i),
            _ => None,
        })
}

// ───────────────────────────────────────────────────────────────────────────
// The test
// ───────────────────────────────────────────────────────────────────────────

/// One test function on purpose: telemetry installs process globals (the
/// tracing subscriber, the global meter provider, the propagator), so a
/// second concurrent test in this binary would race on them.
#[tokio::test(flavor = "multi_thread")]
async fn traces_and_metrics_export_to_a_local_otlp_collector() {
    let (collector_addr, trace_payloads, metric_payloads) = spawn_mock_collector().await;
    let (upstream_addr, upstream_traceparents) = spawn_mock_upstream().await;

    // Real config path: parse + validate, then telemetry init BEFORE the
    // AppState (instruments bind against the global meter provider), exactly
    // as `serve` orders it.
    let cfg = config::parse(&config_toml(upstream_addr, collector_addr)).expect("parse config");
    cfg.validate().expect("validate config");
    let (otel_layer, telemetry_handles) =
        telemetry::init(cfg.telemetry.as_ref()).expect("init telemetry");
    let otel_layer = otel_layer.expect("traces enabled");
    assert!(telemetry_handles.metrics_enabled());
    {
        use tracing_subscriber::layer::SubscriberExt;
        tracing::subscriber::set_global_default(
            tracing_subscriber::Registry::default().with(otel_layer),
        )
        .expect("install global subscriber");
    }

    let state = AppState::with_single_engine(
        Arc::new(FakeFrontierEngine),
        Arc::new(cfg),
        DEFAULT_TRIVIAL_MAX_WORDS,
    )
    .expect("build app state");
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind router");
    let router_addr = listener.local_addr().expect("router addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("router serve");
    });

    // A substantive prompt (not trivial filler), sent with a caller
    // traceparent so extraction is exercised end to end.
    let inbound_trace_id = "0af7651916cd43dd8448eb211c80319c";
    let prompt = "Prove that the halting problem is undecidable.";
    let response = reqwest::Client::new()
        .post(format!("http://{router_addr}/v1/chat/completions"))
        .header(
            "traceparent",
            format!("00-{inbound_trace_id}-b7ad6b7169203331-01"),
        )
        .json(&json!({
            "model": "hyper-mcp-router",
            "messages": [{"role": "user", "content": prompt}],
        }))
        .send()
        .await
        .expect("send chat request");
    assert_eq!(response.status(), 200);

    // ── Downstream propagation: the backend saw a traceparent carrying the
    //    caller's trace id (the router's span id, not the caller's).
    let forwarded = upstream_traceparents
        .lock()
        .unwrap()
        .first()
        .cloned()
        .flatten()
        .expect("upstream received a traceparent header");
    assert!(
        forwarded.contains(inbound_trace_id),
        "forwarded traceparent must carry the caller's trace id, got: {forwarded}"
    );
    assert!(
        !forwarded.contains("b7ad6b7169203331"),
        "forwarded parent span must be the router's span, not the caller's, got: {forwarded}"
    );

    // ── Flush-on-shutdown delivers the final batches (the same path `serve`
    //    runs after draining).
    telemetry_handles.shutdown();

    // ── Traces: the full span tree, joined to the caller's trace.
    let spans = decode_spans(&trace_payloads);
    let names: HashSet<&str> = spans.iter().map(|s| s.name.as_str()).collect();
    for expected in ["chat_completions", "classify", "upstream_request"] {
        assert!(
            names.contains(expected),
            "missing span {expected}: {names:?}"
        );
    }
    for span in &spans {
        assert_eq!(
            hex(&span.trace_id),
            inbound_trace_id,
            "span `{}` must join the caller's trace",
            span.name
        );
    }

    let request_span = spans
        .iter()
        .find(|s| s.name == "chat_completions")
        .expect("request span");
    // Model selection and prompt categorization…
    assert_eq!(
        string_attr(request_span, "model").as_deref(),
        Some("frontier-text")
    );
    assert_eq!(
        string_attr(request_span, "complexity").as_deref(),
        Some("Frontier")
    );
    assert_eq!(
        string_attr(request_span, "classifier_engine").as_deref(),
        Some("fake-frontier")
    );
    // …sizes (prompt under the fake's 400-char turn budget: exact length)…
    assert_eq!(
        int_attr(request_span, "prompt_chars"),
        Some(prompt.chars().count() as i64)
    );
    assert_eq!(
        int_attr(request_span, "window_chars"),
        Some(prompt.chars().count() as i64)
    );
    assert!(int_attr(request_span, "estimated_tokens").unwrap_or(0) > 0);
    // …outcome and the upstream's authoritative token usage.
    assert_eq!(int_attr(request_span, "upstream_status"), Some(200));
    assert_eq!(int_attr(request_span, "prompt_tokens"), Some(41));
    assert_eq!(int_attr(request_span, "completion_tokens"), Some(7));
    assert_eq!(int_attr(request_span, "total_tokens"), Some(48));

    // No prompt text in any exported attribute — spans carry sizes and
    // outcomes, never content.
    for span in &spans {
        for kv in &span.attributes {
            if let Some(any_value::Value::StringValue(s)) =
                kv.value.as_ref().and_then(|v| v.value.as_ref())
            {
                assert!(
                    !s.contains("halting problem"),
                    "span `{}` attribute `{}` leaked prompt text: {s}",
                    span.name,
                    kv.key
                );
            }
        }
    }

    let classify_span = spans
        .iter()
        .find(|s| s.name == "classify")
        .expect("classify span");
    assert_eq!(
        string_attr(classify_span, "engine").as_deref(),
        Some("fake-frontier")
    );

    // ── Metrics: the documented catalogue arrived under its documented names.
    let metric_names = decode_metric_names(&metric_payloads);
    for expected in [
        "hyper_mcp_router.requests",
        "hyper_mcp_router.request.duration",
        "hyper_mcp_router.classification.duration",
        "hyper_mcp_router.classified",
        "hyper_mcp_router.prompt.chars",
        "hyper_mcp_router.window.chars",
        "hyper_mcp_router.tokens.estimated",
        "hyper_mcp_router.tokens.prompt",
        "hyper_mcp_router.tokens.completion",
    ] {
        assert!(
            metric_names.contains(expected),
            "missing metric {expected}: {metric_names:?}"
        );
    }
}
