//! Upstream token-usage accounting: parse the authoritative `usage` object
//! from a buffered upstream response and report estimated-vs-actual token
//! counts on every listening surface — the debug log, the request span, and
//! the OTel token counters. Extracted from `proxy` so the accounting (and its
//! gates) can be reasoned about and tested in isolation.

use serde_json::Value;

use crate::telemetry::{attr, Metrics};

/// The upstream's authoritative token accounting, as parsed from a response
/// body's `usage` object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UpstreamUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Best-effort `usage` extraction: `None` for non-JSON bodies or bodies
/// without a `usage` object (SSE chunks never reach here).
pub(crate) fn parse_usage(body: &[u8]) -> Option<UpstreamUsage> {
    let parsed: Value = serde_json::from_slice(body).ok()?;
    let usage = parsed.get("usage")?;
    let count = |key: &str| usage.get(key).and_then(Value::as_u64);
    Some(UpstreamUsage {
        prompt_tokens: count("prompt_tokens"),
        completion_tokens: count("completion_tokens"),
        total_tokens: count("total_tokens"),
    })
}

/// Estimated-vs-actual token accounting: the router's routing estimate
/// (message chars at ~4/token PLUS the requested completion budget — see
/// [`crate::prompt::estimate_request_tokens`]) next to the upstream's
/// authoritative `usage` object. Watching the two side by side calibrates
/// the context-fit heuristic.
///
/// Reported on three surfaces, each with its own gate so the body is parsed
/// only when someone is listening:
/// - a debug-level `"upstream token usage"` log event;
/// - the `prompt_tokens`/`completion_tokens`/`total_tokens` fields of the
///   request span (recorded whenever the parse ran);
/// - the token-usage counters, when `usage_metrics` carries the instruments
///   (i.e. `[telemetry]` metrics are on).
pub(crate) fn report_upstream_usage(
    model: &str,
    estimated_tokens: u64,
    body: &[u8],
    usage_metrics: Option<&Metrics>,
) {
    let debug = tracing::enabled!(tracing::Level::DEBUG);
    if !debug && usage_metrics.is_none() {
        return;
    }
    let Some(usage) = parse_usage(body) else {
        return;
    };
    if debug {
        tracing::debug!(
            model,
            estimated_tokens,
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            total_tokens = usage.total_tokens,
            "upstream token usage"
        );
    }
    // i64: `tracing-opentelemetry` maps i64 to a real OTel integer
    // attribute, while u64 would export as a debug-formatted STRING.
    let span = tracing::Span::current();
    if let Some(prompt) = usage.prompt_tokens {
        span.record("prompt_tokens", prompt as i64);
    }
    if let Some(completion) = usage.completion_tokens {
        span.record("completion_tokens", completion as i64);
    }
    if let Some(total) = usage.total_tokens {
        span.record("total_tokens", total as i64);
    }
    if let Some(metrics) = usage_metrics {
        let attrs = [attr("model", model.to_string())];
        if let Some(prompt) = usage.prompt_tokens {
            metrics.prompt_tokens.add(prompt, &attrs);
        }
        if let Some(completion) = usage.completion_tokens {
            metrics.completion_tokens.add(completion, &attrs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::captured_log;
    use serde_json::json;

    #[test]
    fn upstream_usage_logged_estimated_vs_actual_at_debug() {
        let body = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let out = captured_log(tracing::Level::DEBUG, || {
            report_upstream_usage("backend-model", 200, &bytes, None);
        });
        assert!(out.contains("upstream token usage"), "got: {out}");
        assert!(out.contains("estimated_tokens=200"), "got: {out}");
        assert!(out.contains("prompt_tokens=120"), "got: {out}");
        assert!(out.contains("completion_tokens=30"), "got: {out}");
        assert!(out.contains("total_tokens=150"), "got: {out}");
    }

    #[test]
    fn upstream_usage_skips_bodies_without_usage_or_non_json() {
        // A well-formed body without `usage`, and a non-JSON body: neither
        // may emit (or worse, error) — the report is best-effort only.
        let no_usage = serde_json::to_vec(&json!({"choices": []})).unwrap();
        let metrics = Metrics::new();
        let out = captured_log(tracing::Level::DEBUG, || {
            report_upstream_usage("backend-model", 200, &no_usage, Some(&metrics));
            report_upstream_usage("backend-model", 200, b"data: [DONE]\n\n", Some(&metrics));
        });
        assert!(out.is_empty(), "got: {out}");
    }

    #[test]
    fn upstream_usage_not_parsed_below_debug_without_metrics() {
        let body = serde_json::to_vec(&json!({"usage": {"total_tokens": 1}})).unwrap();
        let out = captured_log(tracing::Level::INFO, || {
            report_upstream_usage("backend-model", 200, &body, None);
        });
        assert!(out.is_empty(), "got: {out}");
    }

    #[test]
    fn upstream_usage_metrics_recorded_without_debug_logging() {
        // Metrics enabled but logging at info: the usage must still be parsed
        // and recorded (no-op instruments here — the assertion is that the
        // path runs without emitting logs or panicking).
        let body = serde_json::to_vec(&json!({"usage": {"prompt_tokens": 7}})).unwrap();
        let metrics = Metrics::new();
        let out = captured_log(tracing::Level::INFO, || {
            report_upstream_usage("backend-model", 200, &body, Some(&metrics));
        });
        assert!(out.is_empty(), "got: {out}");
    }

    #[test]
    fn parse_usage_extracts_counts_and_tolerates_partials() {
        let body = serde_json::to_vec(&json!({"usage": {"prompt_tokens": 12}})).unwrap();
        assert_eq!(
            parse_usage(&body),
            Some(UpstreamUsage {
                prompt_tokens: Some(12),
                completion_tokens: None,
                total_tokens: None,
            })
        );
        assert_eq!(parse_usage(b"not json"), None);
        let no_usage = serde_json::to_vec(&json!({"choices": []})).unwrap();
        assert_eq!(parse_usage(&no_usage), None);
    }
}
