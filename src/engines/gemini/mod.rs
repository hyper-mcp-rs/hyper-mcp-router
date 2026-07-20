//! The Gemini Developer API engine family: shared transport here in `mod.rs`,
//! **one file per model** in this directory (`embedding_001.rs`,
//! `embedding_2.rs`), each declaring a
//! [`RemoteSpec`](crate::engines::embedding::RemoteSpec) and building the
//! provider-neutral engine over the shared [`GeminiTransport`].
//!
//! Note: `text-embedding-005` is **not** here — it is published only on
//! Vertex AI (a different API), so it lives in the `vertex/` family.
//!
//! The classification *method* (anchor prototypes, cosine scoring) and the
//! shared engine ([`RemoteEmbeddingEngine`]) are provider-neutral and live in
//! `crate::engines::embedding`; this file owns only what is Gemini-specific:
//! the `batchEmbedContents` wire format, the `x-goog-api-key` auth header,
//! and the endpoint layout.
//!
//! Anchor embedding happens once at startup and **fails fast** on any API
//! problem (bad key, unreachable endpoint) — after a bounded retry on
//! transient upstream failures — so misconfiguration is a clear boot error,
//! not a silent stream of balanced-default fallbacks.
//!
//! ## Privacy and failure notes
//!
//! Prompt text (the classification window and current turn) is sent to the
//! Gemini API. Per-request failures degrade to the balanced default via the
//! proxy, as with every engine. Error messages never echo user content.

pub mod embedding_001;
pub mod embedding_2;

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::config::RemoteEmbeddingConfig;
use crate::engines::embedding::{
    anchor_texts, is_transient_status, numeric_vector, transient_error, EmbedTexts,
    RemoteEmbeddingEngine, RemoteSpec, TransientUpstream,
};

/// Default public Gemini API endpoint.
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Task type sent with every embed request; anchors and premises must use the
/// same one for their similarities to be comparable.
const TASK_TYPE: &str = "SEMANTIC_SIMILARITY";

/// The Gemini Developer API transport: one `batchEmbedContents` call per
/// embed, authenticated via the `x-goog-api-key` header.
pub struct GeminiTransport {
    /// Model resource path baked into each request body.
    api_model: &'static str,
    http: reqwest::Client,
    /// Full `:batchEmbedContents` URL.
    url: String,
    /// Redacted; exposed only to build the `x-goog-api-key` header.
    api_key: SecretString,
    /// Bounds concurrent in-flight embedding requests — this engine's
    /// equivalent of the embedded engine's session pool.
    permits: Semaphore,
}

/// Build the family's engine: validate config (the API key is **required**,
/// exactly like a routed model's key but mandatory), construct the transport,
/// and hand it to [`RemoteEmbeddingEngine::connect`] — which embeds the class
/// anchors, failing fast on any API problem.
pub async fn connect(
    spec: &'static RemoteSpec,
    cfg: &RemoteEmbeddingConfig,
    image_gen_threshold: f32,
) -> anyhow::Result<RemoteEmbeddingEngine<GeminiTransport>> {
    let Some(api_key) = cfg.api_key.as_ref() else {
        anyhow::bail!(
            "classifier engine `{}` requires an API key: set `api_key` in \
             [classifier.{}] (plaintext, ${{ENV_VAR}}, or a keyring table)",
            spec.name,
            spec.name,
        );
    };
    let api_key = api_key.resolved()?.clone();

    let base = cfg
        .base_url
        .as_ref()
        .map(|u| u.as_str().trim_end_matches('/').to_string())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let url = format!("{base}/v1beta/{}:batchEmbedContents", spec.api_model);

    let timeout = cfg
        .request_timeout_secs
        .unwrap_or(spec.default_request_timeout_secs);
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(timeout))
        .build()?;

    let max_concurrency = cfg
        .max_concurrency
        .unwrap_or(spec.default_max_concurrency)
        .max(1);

    let transport = GeminiTransport {
        api_model: spec.api_model,
        http,
        url,
        api_key,
        permits: Semaphore::new(max_concurrency),
    };

    let engine = RemoteEmbeddingEngine::connect(spec, transport, image_gen_threshold).await?;

    tracing::info!(
        engine = spec.name,
        embedding_dims = engine.prototype_dims(),
        anchor_count = anchor_texts().len(),
        max_concurrency,
        request_timeout_secs = timeout,
        "gemini embedding engine ready"
    );
    Ok(engine)
}

#[async_trait]
impl EmbedTexts for GeminiTransport {
    /// Embed `texts` in one `batchEmbedContents` call, bounded by the
    /// concurrency semaphore. Never echoes the texts into errors. Transport
    /// failures and 429/5xx statuses carry the [`TransientUpstream`] marker
    /// so startup anchor embedding can retry them.
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("embedding concurrency semaphore closed"))?;

        let body = build_batch_request(self.api_model, texts);
        let resp = self
            .http
            .post(&self.url)
            .header("x-goog-api-key", self.api_key.expose_secret())
            .json(&body)
            .send()
            .await
            .context(TransientUpstream)
            .context("gemini embeddings request failed")?;

        let status = resp.status();
        if !status.is_success() {
            // The error body is the API's, not user content; truncate anyway.
            let detail: String = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(300)
                .collect();
            let message = format!("gemini embeddings returned {status}: {detail}");
            return Err(if is_transient_status(status) {
                transient_error(message)
            } else {
                anyhow::anyhow!(message)
            });
        }

        let value: Value = resp
            .json()
            .await
            .context("gemini embeddings response was not JSON")?;
        parse_embeddings(&value, texts.len())
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Wire format (network-free, unit-testable)
// ───────────────────────────────────────────────────────────────────────────

/// Build the `batchEmbedContents` request body.
fn build_batch_request(api_model: &str, texts: &[&str]) -> Value {
    let requests: Vec<Value> = texts
        .iter()
        .map(|t| {
            json!({
                "model": api_model,
                "content": { "parts": [{ "text": t }] },
                "taskType": TASK_TYPE,
            })
        })
        .collect();
    json!({ "requests": requests })
}

/// Extract `expected` embedding vectors from a `batchEmbedContents` response.
fn parse_embeddings(value: &Value, expected: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let embeddings = value
        .get("embeddings")
        .and_then(|e| e.as_array())
        .ok_or_else(|| anyhow::anyhow!("gemini response missing `embeddings` array"))?;
    if embeddings.len() != expected {
        anyhow::bail!(
            "gemini returned {} embeddings, expected {expected}",
            embeddings.len()
        );
    }
    embeddings
        .iter()
        .map(|e| {
            let values = e
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("gemini embedding missing `values`"))?;
            if values.is_empty() {
                anyhow::bail!("gemini returned an empty embedding vector");
            }
            numeric_vector(values).context("gemini embedding vector is malformed")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_request_shape() {
        let body = build_batch_request("models/gemini-embedding-001", &["a", "b"]);
        let reqs = body["requests"].as_array().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0]["model"], "models/gemini-embedding-001");
        assert_eq!(reqs[0]["content"]["parts"][0]["text"], "a");
        assert_eq!(reqs[1]["content"]["parts"][0]["text"], "b");
        assert_eq!(reqs[0]["taskType"], "SEMANTIC_SIMILARITY");
    }

    #[test]
    fn parse_embeddings_happy_path_and_errors() {
        let ok = serde_json::json!({"embeddings": [
            {"values": [1.0, 2.0]},
            {"values": [3.0, 4.0]},
        ]});
        let parsed = parse_embeddings(&ok, 2).unwrap();
        assert_eq!(parsed, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // Wrong count, missing array, and empty vectors are loud errors.
        assert!(parse_embeddings(&ok, 3).is_err());
        assert!(parse_embeddings(&serde_json::json!({}), 1).is_err());
        let empty = serde_json::json!({"embeddings": [{"values": []}]});
        assert!(parse_embeddings(&empty, 1).is_err());
    }

    #[test]
    fn parse_embeddings_rejects_non_numeric_elements() {
        // A string (or null) element must be a loud error, never a silent 0.0.
        let bad = serde_json::json!({"embeddings": [{"values": [1.0, "2.0"]}]});
        let err = parse_embeddings(&bad, 1).unwrap_err();
        assert!(format!("{err:#}").contains("not a number"), "got: {err:#}");
        let null = serde_json::json!({"embeddings": [{"values": [1.0, null]}]});
        assert!(parse_embeddings(&null, 1).is_err());
    }
}
