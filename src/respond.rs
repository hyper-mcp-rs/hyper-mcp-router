//! Response construction: forwarding upstream responses to the client
//! (streaming and buffered passthrough, header hygiene), the router's own
//! error responses, and the per-request "upstream responded" event.
//! Extracted from `proxy` — everything here is about *what goes back to the
//! client and what that emits*, independent of how the route was chosen.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde_json::json;

use crate::modality::ModalitySet;
use crate::telemetry::Metrics;
use crate::usage::{report_upstream_usage, StreamUsageTap};

// ───────────────────────────────────────────────────────────────────────────
// Upstream-response logging
// ───────────────────────────────────────────────────────────────────────────

/// Cap on the logged upstream error body, in chars. Real error payloads
/// (OpenAI-shaped `{"error": ...}` JSON) fit comfortably; the cap exists so
/// a pathological upstream — an HTML error page from an intermediary —
/// cannot flood a log line.
pub(crate) const ERROR_BODY_LOG_MAX_CHARS: usize = 2048;

/// Render an upstream error body for logging: lossy UTF-8, truncated to
/// [`ERROR_BODY_LOG_MAX_CHARS`] with an explicit marker.
fn error_body_snippet(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    if text.chars().count() <= ERROR_BODY_LOG_MAX_CHARS {
        return text.into_owned();
    }
    let mut snippet: String = text.chars().take(ERROR_BODY_LOG_MAX_CHARS).collect();
    snippet.push_str("… [truncated]");
    snippet
}

/// The per-request "upstream responded" event. Success is **info**,
/// metadata-only; error statuses are **warn** — an upstream rejection is an
/// operational signal, not routine traffic — and additionally carry the
/// upstream's error body (bounded, see [`error_body_snippet`]) when the
/// prompt-logging policy allows content in logs. Upstream error messages
/// routinely echo request content back (invalid-parameter and
/// context-length errors quote the offending input), so the body is treated
/// as user content and gated exactly like the prompt itself (see
/// [`crate::logging::log_prompts_enabled`]).
///
/// `error_body` is `None` when the body is unavailable at emit time (the
/// streaming handoff) — never as a policy decision; the gate is applied
/// here, in one place.
pub(crate) fn log_upstream_response(
    model: &str,
    status: StatusCode,
    latency_ms: u64,
    streaming: bool,
    error_body: Option<&[u8]>,
) {
    let upstream_status = status.as_u16();
    if status.is_success() {
        tracing::info!(
            model,
            upstream_status,
            latency_ms,
            streaming,
            "upstream responded"
        );
        return;
    }
    let body = error_body
        .filter(|_| crate::logging::log_prompts_enabled())
        .map(error_body_snippet);
    match body {
        Some(body) => tracing::warn!(
            model,
            upstream_status,
            latency_ms,
            streaming,
            error_body = %body,
            "upstream responded"
        ),
        None => {
            tracing::warn!(
                model,
                upstream_status,
                latency_ms,
                streaming,
                "upstream responded"
            )
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Passthrough
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

/// Pipe raw upstream SSE bytes straight to the client. No buffering or
/// model-field rewriting; upstream end-to-end headers are preserved, with
/// SSE defaults filled in only when the upstream omitted them.
///
/// When someone is listening for token usage (debug logging on, or the
/// usage counters enabled), the bytes are forwarded through a
/// [`StreamUsageTap`]: still byte-for-byte untouched, but a bounded tail is
/// retained so the trailing `usage` chunk (sent by OpenAI-compatible
/// backends when the client requests `stream_options.include_usage`) is
/// reported at end-of-stream — the streaming counterpart of
/// [`buffered_passthrough`]'s usage peek. With nobody listening the stream
/// passes through untapped.
pub(crate) fn stream_passthrough(
    resp: reqwest::Response,
    model: &str,
    estimated_tokens: u64,
    usage_metrics: Option<Arc<Metrics>>,
) -> Response {
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
    let body = if tracing::enabled!(tracing::Level::DEBUG) || usage_metrics.is_some() {
        Body::from_stream(StreamUsageTap::new(
            resp.bytes_stream(),
            model,
            estimated_tokens,
            tracing::Span::current(),
            usage_metrics,
        ))
    } else {
        Body::from_stream(resp.bytes_stream())
    };
    builder.body(body).expect("valid streaming response")
}

/// Forward a non-streaming (or error) upstream response verbatim: same status,
/// same body, preserved end-to-end headers. Emits the "upstream responded"
/// event once the body is buffered (see [`log_upstream_response`] — an error
/// status may then carry the body under the prompt-logging gate); successful
/// responses additionally feed the usage report (see
/// [`report_upstream_usage`]) — the body is already fully buffered here, so
/// the peek is free.
pub(crate) async fn buffered_passthrough(
    resp: reqwest::Response,
    model: &str,
    estimated_tokens: u64,
    usage_metrics: Option<&Metrics>,
    latency_ms: u64,
    streaming: bool,
) -> Response {
    let status = resp.status();
    let upstream_headers = resp.headers().clone();

    match resp.bytes().await {
        Ok(bytes) => {
            log_upstream_response(
                model,
                status,
                latency_ms,
                streaming,
                (!status.is_success()).then_some(bytes.as_ref()),
            );
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
            // The "upstream responded" event is skipped for this (rare)
            // outcome; this warn carries the same identifiers instead.
            tracing::warn!(
                error = %e,
                model,
                upstream_status = status.as_u16(),
                "failed to read upstream response body"
            );
            upstream_error(StatusCode::BAD_GATEWAY, "upstream_body_error")
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Router-generated error responses
// ───────────────────────────────────────────────────────────────────────────

pub(crate) fn bad_request(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": { "message": message, "type": "invalid_request_error" } })),
    )
        .into_response()
}

/// 422 with a minimal JSON body naming the unsatisfiable modality set. (422
/// Unprocessable Content, not 415: the request *media type* is fine — it is
/// the required capability combination no configured backend can serve.)
pub(crate) fn unsupported_modality_error(required: &ModalitySet) -> Response {
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

pub(crate) fn upstream_error(code: StatusCode, kind: &str) -> Response {
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
    use crate::test_support::captured_log;

    // ── header passthrough ────────────────────────────────────────────────
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

    // ── "upstream responded" levels ───────────────────────────────────────
    #[test]
    fn upstream_response_logged_info_on_success_warn_on_error() {
        // Level semantics only — body handling is gate-dependent and lives
        // in `proxy::tests::prompt_logging_follows_the_global_policy_flag`
        // (the gate is a process global; one test owns it). Success: info.
        let out = captured_log(tracing::Level::INFO, || {
            log_upstream_response("backend-model", StatusCode::OK, 12, false, None);
        });
        assert!(out.contains("INFO"), "got: {out}");
        assert!(out.contains("upstream responded"), "got: {out}");
        assert!(out.contains("upstream_status=200"), "got: {out}");

        // Error statuses: warn — visible even under a warn-only filter.
        let out = captured_log(tracing::Level::WARN, || {
            log_upstream_response("backend-model", StatusCode::BAD_REQUEST, 12, true, None);
        });
        assert!(out.contains("WARN"), "got: {out}");
        assert!(out.contains("upstream responded"), "got: {out}");
        assert!(out.contains("upstream_status=400"), "got: {out}");

        // ...and a success is NOT a warn: nothing under a warn-only filter.
        let out = captured_log(tracing::Level::WARN, || {
            log_upstream_response("backend-model", StatusCode::OK, 12, false, None);
        });
        assert!(out.is_empty(), "got: {out}");
    }

    #[test]
    fn error_body_snippet_truncates_and_lossy_decodes() {
        assert_eq!(error_body_snippet(b"small body"), "small body");

        let big = "y".repeat(ERROR_BODY_LOG_MAX_CHARS + 100);
        let snippet = error_body_snippet(big.as_bytes());
        assert!(snippet.ends_with("… [truncated]"), "got: {snippet}");
        assert_eq!(
            snippet.chars().count(),
            ERROR_BODY_LOG_MAX_CHARS + "… [truncated]".chars().count()
        );

        // Invalid UTF-8 is replaced, never dropped or panicking.
        assert_eq!(error_body_snippet(&[0xff, 0xfe]), "\u{FFFD}\u{FFFD}");
    }
}
