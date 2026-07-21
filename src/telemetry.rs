//! Optional OpenTelemetry export: traces and metrics over **OTLP/HTTP
//! (protobuf)** to a credential-less endpoint — a local collector sidecar
//! (`http://localhost:4318`) or an equally trusted network hop. The collector
//! owns vendor authentication; this module never handles credentials.
//!
//! Everything here is inert unless the config carries a `[telemetry]` table:
//! no providers are built, no background threads start, no sockets open, the
//! global meter stays the no-op default, and the propagation helpers become
//! no-ops. The rest of the codebase instruments unconditionally (spans via
//! `tracing`, metrics via [`Metrics`]) and pays only the disabled-layer cost
//! when telemetry is off.
//!
//! Design constraints (see README "Telemetry"):
//! - **Transport**: HTTP/protobuf only. The batch processors run on dedicated
//!   background threads and drive the exporter synchronously, so the exporter
//!   uses the *blocking* reqwest client (built off-thread by
//!   `opentelemetry-otlp` to stay panic-free under tokio).
//! - **Sampling**: independent-of-parent by default (`sample_ratio`), because
//!   platform ingress tracing (e.g. Cloud Run's ~0.1 req/s) would otherwise
//!   drop nearly every router span; `parent_based_sampling = true` restores
//!   the OTel-conventional behavior.
//! - **Failure isolation**: exporter errors drop batches with an internal
//!   warning and never touch the request path.

use std::time::Duration;

use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use opentelemetry_sdk::Resource;

use crate::config::TelemetryConfig;

/// Handles to the running providers, for flush-on-shutdown. Also carries the
/// answers to "is this signal on?" so instrumentation call sites can skip
/// work (e.g. parsing upstream bodies for usage) when nobody is listening.
pub struct Telemetry {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl Telemetry {
    /// Telemetry disabled: nothing to flush, all signals off.
    pub fn disabled() -> Self {
        Telemetry {
            tracer_provider: None,
            meter_provider: None,
        }
    }

    pub fn metrics_enabled(&self) -> bool {
        self.meter_provider.is_some()
    }

    /// Flush and shut down the providers (call after the server drains, so
    /// the final batches reach the collector before the process exits —
    /// container runtimes won't wait for a dropped batch). Errors are logged,
    /// not propagated: a failed final flush must not turn a clean shutdown
    /// into a non-zero exit.
    pub fn shutdown(&self) {
        if let Some(provider) = &self.tracer_provider {
            if let Err(e) = provider.shutdown() {
                tracing::warn!(error = %e, "tracer provider shutdown failed");
            }
        }
        if let Some(provider) = &self.meter_provider {
            if let Err(e) = provider.shutdown() {
                tracing::warn!(error = %e, "meter provider shutdown failed");
            }
        }
    }
}

/// Build the configured providers and install the process-global pieces (text
/// map propagator, meter provider). Returns the tracing-subscriber layer to
/// stack onto the registry (when traces are on) plus the shutdown handles.
///
/// `None` config is the fully-off path: returns `(None, Telemetry::disabled())`
/// and installs nothing global.
#[allow(clippy::type_complexity)]
pub fn init(
    config: Option<&TelemetryConfig>,
) -> anyhow::Result<(
    Option<
        tracing_opentelemetry::OpenTelemetryLayer<
            tracing_subscriber::Registry,
            opentelemetry_sdk::trace::SdkTracer,
        >,
    >,
    Telemetry,
)> {
    let Some(config) = config else {
        return Ok((None, Telemetry::disabled()));
    };

    // W3C `traceparent` propagation: extraction joins the caller's trace,
    // injection carries it to the routed backend. Installed whenever any
    // signal is on — even metrics-only deployments benefit from forwarding
    // trace context downstream.
    global::set_text_map_propagator(TraceContextPropagator::new());

    let endpoint = config.otlp_endpoint.as_str().trim_end_matches('/');
    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .build();

    let mut layer = None;
    let mut tracer_provider = None;
    if config.traces {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(format!("{endpoint}/v1/traces"))
            .build()?;
        let ratio_sampler = Sampler::TraceIdRatioBased(config.sample_ratio);
        let sampler = if config.parent_based_sampling {
            Sampler::ParentBased(Box::new(ratio_sampler))
        } else {
            ratio_sampler
        };
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_sampler(sampler)
            .with_resource(resource.clone())
            .build();
        use opentelemetry::trace::TracerProvider as _;
        layer = Some(tracing_opentelemetry::layer().with_tracer(provider.tracer(SCOPE)));
        tracer_provider = Some(provider);
    }

    let mut meter_provider = None;
    if config.metrics {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(format!("{endpoint}/v1/metrics"))
            .build()?;
        let reader = PeriodicReader::builder(exporter)
            .with_interval(Duration::from_secs(config.metrics_interval_secs))
            .build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(resource)
            .build();
        // Global, so `Metrics::new()` binds real instruments; when metrics
        // are off the global stays the built-in no-op meter provider.
        global::set_meter_provider(provider.clone());
        meter_provider = Some(provider);
    }

    Ok((
        layer,
        Telemetry {
            tracer_provider,
            meter_provider,
        },
    ))
}

/// Instrumentation scope for the router's tracer and meter.
const SCOPE: &str = "hyper-mcp-router";

// ───────────────────────────────────────────────────────────────────────────
// W3C trace-context propagation over http::HeaderMap
// ───────────────────────────────────────────────────────────────────────────
//
// Hand-rolled 20-line Extractor/Injector rather than the `opentelemetry-http`
// helpers — axum and reqwest share the same `http::HeaderMap` type in this
// dependency tree, so one pair covers both directions.

/// Read-only [`Extractor`] over incoming request headers.
pub struct HeaderExtractor<'a>(pub &'a http::HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }
    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(http::HeaderName::as_str).collect()
    }
}

/// Write-only [`Injector`] over outgoing upstream request headers.
pub struct HeaderInjector<'a>(pub &'a mut http::HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(value)) = (
            http::HeaderName::try_from(key),
            http::HeaderValue::try_from(value),
        ) {
            self.0.insert(name, value);
        }
    }
}

/// Extract the caller's trace context from request headers via the global
/// propagator. With telemetry off no propagator is installed, so this returns
/// an empty context (a no-op parent).
pub fn extract_context(headers: &http::HeaderMap) -> opentelemetry::Context {
    global::get_text_map_propagator(|propagator| propagator.extract(&HeaderExtractor(headers)))
}

/// Inject `span`'s trace context into upstream request headers
/// (`traceparent`), so the routed backend's telemetry joins the same trace.
/// No-op when telemetry is off or the span is disabled.
pub fn inject_context(span: &tracing::Span, headers: &mut http::HeaderMap) {
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    let context = span.context();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers))
    });
}

// ───────────────────────────────────────────────────────────────────────────
// Metrics catalogue
// ───────────────────────────────────────────────────────────────────────────

/// Every instrument the router records, created once at startup from the
/// global meter. With telemetry (or the metrics signal) off, the global meter
/// is the no-op provider and every record call is a cheap no-op — call sites
/// never branch.
///
/// Catalogue (all names under `hyper_mcp_router.`):
/// - `requests` (counter): completed requests, by `model`, `complexity`,
///   `streaming`, and `status` (upstream HTTP status, or a router error kind).
/// - `request.duration` (histogram, s): request wall time by `model`/`status`.
/// - `classification.duration` (histogram, s): classifier-engine inference
///   time by `engine`.
/// - `classified` (counter): prompt categorization outcomes, by `complexity`
///   and `engine`.
/// - `classification.skipped` (counter): model-free routing decisions, by
///   `reason` (`single_candidate` / `trivial_fast_path` / `no_user_text`).
/// - `prompt.chars` / `window.chars` (histograms): current-turn prompt size
///   (as classified) and compiled classification-window size.
/// - `tokens.estimated` (histogram): the router's context-fit estimate.
/// - `tokens.prompt` / `tokens.completion` (counters): the upstream's
///   authoritative `usage` counts, by `model` — comparing rates against
///   `tokens.estimated` calibrates the chars-per-token heuristic. Buffered
///   responses always report; streaming responses report only when the
///   upstream emits a trailing `usage` chunk, which OpenAI-compatible
///   backends do only if the client sent
///   `stream_options: {"include_usage": true}` (see `crate::usage`).
pub struct Metrics {
    pub requests: Counter<u64>,
    pub request_duration: Histogram<f64>,
    pub classification_duration: Histogram<f64>,
    pub classified: Counter<u64>,
    pub classification_skipped: Counter<u64>,
    pub prompt_chars: Histogram<u64>,
    pub window_chars: Histogram<u64>,
    pub estimated_tokens: Histogram<u64>,
    pub prompt_tokens: Counter<u64>,
    pub completion_tokens: Counter<u64>,
}

impl Metrics {
    /// Bind instruments against the global meter provider (real after
    /// [`init`] installed one; no-op otherwise).
    pub fn new() -> Self {
        let meter: Meter = global::meter(SCOPE);
        Metrics {
            requests: meter
                .u64_counter("hyper_mcp_router.requests")
                .with_description("Completed completion requests")
                .build(),
            request_duration: meter
                .f64_histogram("hyper_mcp_router.request.duration")
                .with_unit("s")
                .with_description("End-to-end request wall time")
                .build(),
            classification_duration: meter
                .f64_histogram("hyper_mcp_router.classification.duration")
                .with_unit("s")
                .with_description("Classifier-engine inference wall time")
                .build(),
            classified: meter
                .u64_counter("hyper_mcp_router.classified")
                .with_description("Prompt categorization outcomes by tier")
                .build(),
            classification_skipped: meter
                .u64_counter("hyper_mcp_router.classification.skipped")
                .with_description("Classifications skipped, by reason")
                .build(),
            prompt_chars: meter
                .u64_histogram("hyper_mcp_router.prompt.chars")
                .with_unit("{char}")
                .with_description("Current-turn prompt size as the classifier saw it")
                .build(),
            window_chars: meter
                .u64_histogram("hyper_mcp_router.window.chars")
                .with_unit("{char}")
                .with_description("Compiled classification-window size")
                .build(),
            estimated_tokens: meter
                .u64_histogram("hyper_mcp_router.tokens.estimated")
                .with_unit("{token}")
                .with_description("Estimated context-window occupancy per request")
                .build(),
            prompt_tokens: meter
                .u64_counter("hyper_mcp_router.tokens.prompt")
                .with_unit("{token}")
                .with_description("Upstream-reported prompt tokens")
                .build(),
            completion_tokens: meter
                .u64_counter("hyper_mcp_router.tokens.completion")
                .with_unit("{token}")
                .with_description("Upstream-reported completion tokens")
                .build(),
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics::new()
    }
}

/// Shorthand for a metric attribute.
pub fn attr(key: &'static str, value: impl Into<opentelemetry::Value>) -> KeyValue {
    KeyValue::new(key, value.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry::trace::TraceContextExt;

    // Serde tests live in `config`; these exercise init/propagation/metrics.
    fn config(endpoint: &str) -> TelemetryConfig {
        TelemetryConfig {
            otlp_endpoint: url::Url::parse(endpoint).expect("test endpoint"),
            service_name: "hyper-mcp-router".into(),
            traces: true,
            metrics: true,
            sample_ratio: 1.0,
            parent_based_sampling: false,
            metrics_interval_secs: 60,
        }
    }

    #[test]
    fn disabled_config_builds_nothing() {
        let (layer, telemetry) = init(None).expect("disabled init");
        assert!(layer.is_none());
        assert!(!telemetry.metrics_enabled());
        telemetry.shutdown(); // must be a harmless no-op
    }

    #[test]
    fn enabled_config_builds_both_providers_without_touching_the_network() {
        // Exporters are lazy: building providers opens no sockets, so this
        // is hermetic even though the endpoint is unreachable.
        let cfg = config("http://localhost:1");
        let (layer, telemetry) = init(Some(&cfg)).expect("enabled init");
        assert!(layer.is_some());
        assert!(telemetry.metrics_enabled());
        telemetry.shutdown();
    }

    #[test]
    fn signals_can_be_disabled_individually() {
        let cfg = config("http://localhost:1");
        let traces_only = TelemetryConfig {
            metrics: false,
            ..cfg.clone()
        };
        let (layer, telemetry) = init(Some(&traces_only)).expect("traces-only init");
        assert!(layer.is_some());
        assert!(!telemetry.metrics_enabled());
        telemetry.shutdown();

        let metrics_only = TelemetryConfig {
            traces: false,
            ..cfg
        };
        let (layer, telemetry) = init(Some(&metrics_only)).expect("metrics-only init");
        assert!(layer.is_none());
        assert!(telemetry.metrics_enabled());
        telemetry.shutdown();
    }

    #[test]
    fn header_extractor_roundtrips_w3c_traceparent() {
        // Extraction goes through the propagator directly (not the global,
        // which other tests may or may not have installed).
        let propagator = TraceContextPropagator::new();
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
                .parse()
                .unwrap(),
        );
        let context = propagator.extract(&HeaderExtractor(&headers));
        let span_context = context.span().span_context().clone();
        assert!(span_context.is_valid());
        assert_eq!(
            span_context.trace_id().to_string(),
            "0af7651916cd43dd8448eb211c80319c"
        );
        assert_eq!(span_context.span_id().to_string(), "b7ad6b7169203331");
        assert!(span_context.is_sampled());

        // And back out through the injector.
        let mut out = http::HeaderMap::new();
        propagator.inject_context(&context, &mut HeaderInjector(&mut out));
        let reinjected = out.get("traceparent").unwrap().to_str().unwrap();
        assert!(
            reinjected.contains("0af7651916cd43dd8448eb211c80319c"),
            "got: {reinjected}"
        );
    }

    #[test]
    fn extractor_ignores_garbage_traceparent() {
        let propagator = TraceContextPropagator::new();
        let mut headers = http::HeaderMap::new();
        headers.insert("traceparent", "not-a-traceparent".parse().unwrap());
        let context = propagator.extract(&HeaderExtractor(&headers));
        assert!(!context.span().span_context().is_valid());
    }

    #[test]
    fn metrics_record_safely_without_a_provider() {
        // Bound against the global meter (the no-op default, or whatever a
        // concurrently-running test installed): every instrument must accept
        // recordings without panicking either way.
        let metrics = Metrics::new();
        metrics.requests.add(1, &[attr("model", "m")]);
        metrics.request_duration.record(0.25, &[]);
        metrics.classification_duration.record(0.01, &[]);
        metrics.classified.add(1, &[attr("complexity", "Fast")]);
        metrics
            .classification_skipped
            .add(1, &[attr("reason", "single_candidate")]);
        metrics.prompt_chars.record(12, &[]);
        metrics.window_chars.record(120, &[]);
        metrics.estimated_tokens.record(34, &[]);
        metrics.prompt_tokens.add(100, &[attr("model", "m")]);
        metrics.completion_tokens.add(50, &[attr("model", "m")]);
    }
}
