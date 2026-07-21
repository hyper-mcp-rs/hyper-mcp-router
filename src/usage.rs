//! Upstream token-usage accounting: parse the authoritative `usage` object
//! from an upstream response and report estimated-vs-actual token counts on
//! every listening surface — the debug log, the request span, and the OTel
//! token counters. Extracted from `proxy` so the accounting (and its gates)
//! can be reasoned about and tested in isolation.
//!
//! Two response shapes are covered:
//! - **Buffered** bodies: [`report_upstream_usage`] peeks at the (already
//!   buffered) JSON body.
//! - **Streaming (SSE)** bodies: [`StreamUsageTap`] forwards bytes untouched
//!   while retaining a bounded tail, and reports the trailing `usage` chunk
//!   at end-of-stream. OpenAI-compatible backends send that chunk only when
//!   the client asks for it (`stream_options: {"include_usage": true}`), so
//!   streaming coverage is best-effort by design.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::Bytes;
use futures_core::Stream;
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

/// Best-effort `usage` extraction: `None` for non-JSON bodies and bodies
/// whose `usage` carries no counts — absent, `null` (what streaming chunks
/// before the final one carry under `include_usage`), or an empty object.
pub(crate) fn parse_usage(body: &[u8]) -> Option<UpstreamUsage> {
    let parsed: Value = serde_json::from_slice(body).ok()?;
    let usage = parsed.get("usage")?;
    let count = |key: &str| usage.get(key).and_then(Value::as_u64);
    let usage = UpstreamUsage {
        prompt_tokens: count("prompt_tokens"),
        completion_tokens: count("completion_tokens"),
        total_tokens: count("total_tokens"),
    };
    (usage.prompt_tokens.is_some()
        || usage.completion_tokens.is_some()
        || usage.total_tokens.is_some())
    .then_some(usage)
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
    report_usage(model, estimated_tokens, usage, usage_metrics);
}

/// Report an already-parsed usage object on every listening surface (see
/// [`report_upstream_usage`]). Span fields land on the **current** span, so
/// callers reporting outside the request context (the stream tap) must enter
/// the request span first.
fn report_usage(
    model: &str,
    estimated_tokens: u64,
    usage: UpstreamUsage,
    usage_metrics: Option<&Metrics>,
) {
    tracing::debug!(
        model,
        estimated_tokens,
        prompt_tokens = usage.prompt_tokens,
        completion_tokens = usage.completion_tokens,
        total_tokens = usage.total_tokens,
        "upstream token usage"
    );
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

/// Bounded tail of an SSE byte stream, kept to recover the trailing `usage`
/// chunk after the stream ends. Chunk boundaries are arbitrary (a `data:`
/// line may split across chunks), so raw bytes are accumulated and line
/// parsing is deferred to [`SseTail::usage`].
pub(crate) struct SseTail {
    buf: Vec<u8>,
}

impl SseTail {
    /// Comfortably covers the final usage chunk plus the `[DONE]` sentinel
    /// (a usage chunk is a few hundred bytes); memory per open stream stays
    /// bounded no matter how long the response runs.
    const CAP: usize = 16 * 1024;

    pub(crate) fn new() -> Self {
        SseTail { buf: Vec::new() }
    }

    /// Append a chunk, discarding the oldest bytes beyond [`Self::CAP`].
    pub(crate) fn extend(&mut self, chunk: &[u8]) {
        if chunk.len() >= Self::CAP {
            self.buf.clear();
            self.buf
                .extend_from_slice(&chunk[chunk.len() - Self::CAP..]);
            return;
        }
        let overflow = (self.buf.len() + chunk.len()).saturating_sub(Self::CAP);
        if overflow > 0 {
            self.buf.drain(..overflow);
        }
        self.buf.extend_from_slice(chunk);
    }

    /// The last `data:` event in the tail carrying a countable `usage`
    /// object. `[DONE]`, `usage: null` chunks, and lines truncated by the
    /// cap all parse to nothing and are skipped — best-effort only.
    pub(crate) fn usage(&self) -> Option<UpstreamUsage> {
        let text = String::from_utf8_lossy(&self.buf);
        let mut found = None;
        for line in text.lines() {
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            if let Some(usage) = parse_usage(payload.trim().as_bytes()) {
                found = Some(usage);
            }
        }
        found
    }
}

/// SSE passthrough tap: forwards upstream bytes to the client untouched
/// while feeding an [`SseTail`], and reports the stream's trailing `usage`
/// chunk (see the module docs) once the upstream closes the stream.
///
/// The request span handle is held for the stream's lifetime so the token
/// fields recorded at end-of-stream still reach the exported span — records
/// after span close are dropped. Deliberate consequence: while the tap is
/// active (someone is listening for usage), the exported request span covers
/// the full stream rather than time-to-response-headers.
pub(crate) struct StreamUsageTap<S> {
    inner: Pin<Box<S>>,
    tail: SseTail,
    model: String,
    estimated_tokens: u64,
    span: tracing::Span,
    metrics: Option<Arc<Metrics>>,
    reported: bool,
}

impl<S> StreamUsageTap<S> {
    pub(crate) fn new(
        inner: S,
        model: &str,
        estimated_tokens: u64,
        span: tracing::Span,
        metrics: Option<Arc<Metrics>>,
    ) -> Self {
        StreamUsageTap {
            inner: Box::pin(inner),
            tail: SseTail::new(),
            model: model.to_string(),
            estimated_tokens,
            span,
            metrics,
            reported: false,
        }
    }
}

impl<S, E> Stream for StreamUsageTap<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<Bytes, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let polled = this.inner.as_mut().poll_next(cx);
        match &polled {
            Poll::Ready(Some(Ok(chunk))) => this.tail.extend(chunk),
            Poll::Ready(None) if !this.reported => {
                this.reported = true;
                if let Some(usage) = this.tail.usage() {
                    // Enter the request span so the debug event and the
                    // span-field records land in the request's context.
                    let span = this.span.clone();
                    span.in_scope(|| {
                        report_usage(
                            &this.model,
                            this.estimated_tokens,
                            usage,
                            this.metrics.as_deref(),
                        );
                    });
                }
            }
            _ => {}
        }
        polled
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

    #[test]
    fn parse_usage_rejects_null_and_countless_usage() {
        // `usage: null` is what every pre-final streaming chunk carries under
        // `include_usage`; an empty object carries no counts. Neither may
        // produce a (pointless, all-None) report.
        assert_eq!(parse_usage(br#"{"usage": null}"#), None);
        assert_eq!(parse_usage(br#"{"usage": {}}"#), None);
    }

    #[test]
    fn sse_tail_finds_last_usage_chunk_skipping_null_and_done() {
        let mut tail = SseTail::new();
        tail.extend(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n");
        tail.extend(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\"total_tokens\":13}}\n\n",
        );
        tail.extend(b"data: [DONE]\n\n");
        assert_eq!(
            tail.usage(),
            Some(UpstreamUsage {
                prompt_tokens: Some(9),
                completion_tokens: Some(4),
                total_tokens: Some(13),
            })
        );
    }

    #[test]
    fn sse_tail_handles_chunk_boundaries_inside_a_data_line() {
        // Network chunking may split the usage event anywhere; the tail
        // accumulates raw bytes, so the reassembled line still parses.
        let mut tail = SseTail::new();
        tail.extend(b"data: {\"usage\":{\"tot");
        tail.extend(b"al_tokens\":7}}\n\ndata: [DONE]\n\n");
        assert_eq!(
            tail.usage(),
            Some(UpstreamUsage {
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: Some(7),
            })
        );
    }

    #[test]
    fn sse_tail_none_without_usage() {
        let mut tail = SseTail::new();
        tail.extend(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n");
        assert_eq!(tail.usage(), None);
        assert_eq!(SseTail::new().usage(), None);
    }

    #[test]
    fn sse_tail_stays_bounded_and_still_finds_the_trailing_usage() {
        let mut tail = SseTail::new();
        // Flood well past the cap (100 × ~440 bytes ≈ 44 KiB), then append
        // the real usage chunk: memory stays capped, the chunk is found, and
        // the cap-truncated leading line is skipped without error.
        let filler = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}],\"usage\":null}}\n\n",
            "x".repeat(400)
        );
        for _ in 0..100 {
            tail.extend(filler.as_bytes());
        }
        tail.extend(b"data: {\"usage\":{\"total_tokens\":42}}\n\ndata: [DONE]\n\n");
        assert!(tail.buf.len() <= SseTail::CAP, "got: {}", tail.buf.len());
        assert_eq!(
            tail.usage(),
            Some(UpstreamUsage {
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: Some(42),
            })
        );
    }

    /// Test stream yielding queued chunks, then end-of-stream.
    struct ChunkStream(std::collections::VecDeque<Bytes>);

    impl Stream for ChunkStream {
        type Item = Result<Bytes, std::convert::Infallible>;
        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.get_mut().0.pop_front().map(Ok))
        }
    }

    /// Drive `tap` to completion synchronously, returning the forwarded bytes.
    fn drain_tap(tap: &mut StreamUsageTap<ChunkStream>) -> Vec<u8> {
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut forwarded = Vec::new();
        loop {
            match Pin::new(&mut *tap).poll_next(&mut cx) {
                Poll::Ready(Some(Ok(chunk))) => forwarded.extend_from_slice(&chunk),
                Poll::Ready(None) => return forwarded,
                Poll::Ready(Some(Err(_))) | Poll::Pending => unreachable!(),
            }
        }
    }

    #[test]
    fn stream_tap_forwards_bytes_untouched_and_reports_trailing_usage() {
        let chunks: Vec<Bytes> = vec![
            Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}],\"usage\":null}\n\n",
            ),
            Bytes::from_static(
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"total_tokens\":16}}\n\ndata: [DONE]\n\n",
            ),
        ];
        let expected: Vec<u8> = chunks.iter().flat_map(|c| c.to_vec()).collect();
        let out = captured_log(tracing::Level::DEBUG, || {
            let mut tap = StreamUsageTap::new(
                ChunkStream(chunks.into()),
                "backend-model",
                200,
                tracing::Span::current(),
                None,
            );
            let forwarded = drain_tap(&mut tap);
            // Byte-for-byte passthrough: the tap only observes.
            assert_eq!(forwarded, expected);
        });
        assert!(out.contains("upstream token usage"), "got: {out}");
        assert!(out.contains("estimated_tokens=200"), "got: {out}");
        assert!(out.contains("prompt_tokens=11"), "got: {out}");
        assert!(out.contains("completion_tokens=5"), "got: {out}");
        assert!(out.contains("total_tokens=16"), "got: {out}");
    }

    #[test]
    fn stream_tap_without_usage_chunk_stays_silent() {
        // Upstreams only send the trailing usage chunk when the client asked
        // for it (`stream_options.include_usage`); without one the tap must
        // forward everything and report nothing.
        let chunks: Vec<Bytes> = vec![
            Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"),
            Bytes::from_static(b"data: [DONE]\n\n"),
        ];
        let expected: Vec<u8> = chunks.iter().flat_map(|c| c.to_vec()).collect();
        let out = captured_log(tracing::Level::DEBUG, || {
            let mut tap = StreamUsageTap::new(
                ChunkStream(chunks.into()),
                "backend-model",
                200,
                tracing::Span::current(),
                None,
            );
            assert_eq!(drain_tap(&mut tap), expected);
        });
        assert!(out.is_empty(), "got: {out}");
    }
}
