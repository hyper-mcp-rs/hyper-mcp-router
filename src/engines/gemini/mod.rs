//! The Gemini Developer API engine family: shared transport here in `mod.rs`,
//! **one file per model** in this directory (`embedding_001.rs`,
//! `embedding_2.rs`), each defining a [`GeminiSpec`] and delegating to the
//! shared [`GeminiEmbedding`] engine.
//!
//! Note: `text-embedding-005` is **not** here — it is published only on
//! Vertex AI (a different API), so it lives in the `vertex/` family.
//!
//! The classification *method* (anchor prototypes, cosine scoring) is
//! provider-neutral and lives in `crate::engines::embedding`; this file owns
//! only what is Gemini-specific: the `batchEmbedContents` wire format, the
//! `x-goog-api-key` auth header, and the endpoint layout.
//!
//! Anchor embedding happens once at startup and **fails fast** on any API
//! problem (bad key, unreachable endpoint), so misconfiguration is a clear
//! boot error, not a silent stream of balanced-default fallbacks.
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
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::classifier::{Classification, ClassifierEngine};
use crate::config::RemoteEmbeddingConfig;
use crate::engines::embedding::{anchor_texts, build_prototypes, combine_similarities, Prototypes};

/// Default public Gemini API endpoint.
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Task type sent with every embed request; anchors and premises must use the
/// same one for their similarities to be comparable.
const TASK_TYPE: &str = "SEMANTIC_SIMILARITY";

/// Everything that differs between Gemini embedding models. Each model file
/// declares one of these as a `const`.
pub struct GeminiSpec {
    /// Engine id (matches `ClassifierModel::as_str`).
    pub name: &'static str,
    /// Model resource path sent to the API, e.g. `models/gemini-embedding-001`.
    pub api_model: &'static str,
    /// Char budget for the complexity window (sized to the model's input
    /// token limit).
    pub context_char_budget: usize,
    /// Char budget for the current turn (image premise / lexical prefilter).
    pub current_turn_char_budget: usize,
    /// Default max concurrent embedding requests (the "session pool").
    pub default_max_concurrency: usize,
    /// Default per-call timeout, seconds.
    pub default_request_timeout_secs: u64,
}

/// A remote Gemini embedding engine (shared by every Gemini model file).
pub struct GeminiEmbedding {
    spec: &'static GeminiSpec,
    http: reqwest::Client,
    /// Full `:batchEmbedContents` URL.
    url: String,
    api_key: String,
    /// Bounds concurrent in-flight embedding requests — this engine's
    /// equivalent of the embedded engine's session pool.
    permits: Semaphore,
    /// Cosine-similarity floor for the image axis (see
    /// `crate::engines::embedding`).
    image_gen_threshold: f32,
    prototypes: Prototypes,
}

impl GeminiEmbedding {
    /// Build the engine: validate config (the API key is **required**, exactly
    /// like a routed model's key but mandatory), construct the HTTP client,
    /// and embed the class anchors — failing fast on any API problem.
    pub async fn connect(
        spec: &'static GeminiSpec,
        cfg: &RemoteEmbeddingConfig,
        image_gen_threshold: f32,
    ) -> anyhow::Result<Self> {
        let Some(api_key) = cfg.api_key.clone() else {
            anyhow::bail!(
                "classifier engine `{}` requires an API key: set `api_key` in \
                 [classifier.{}] (plaintext, ${{ENV_VAR}}, or a keyring table)",
                spec.name,
                spec.name,
            );
        };

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

        let mut engine = GeminiEmbedding {
            spec,
            http,
            url,
            api_key,
            permits: Semaphore::new(max_concurrency),
            image_gen_threshold,
            prototypes: Prototypes::default(),
        };

        // Embed every anchor in one batch and pool per class. Startup is the
        // right place to fail on a bad key or unreachable endpoint.
        let anchors = anchor_texts();
        let embeddings = engine
            .embed_batch(&anchors)
            .await
            .with_context(|| format!("embedding class anchors for `{}`", spec.name))?;
        engine.prototypes = build_prototypes(&embeddings)?;

        tracing::info!(
            engine = spec.name,
            embedding_dims = engine.prototypes.dims(),
            anchor_count = anchors.len(),
            max_concurrency,
            request_timeout_secs = timeout,
            "gemini embedding engine ready"
        );
        Ok(engine)
    }

    /// Embed `texts` in one `batchEmbedContents` call, bounded by the
    /// concurrency semaphore. Never echoes the texts into errors.
    async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("embedding concurrency semaphore closed"))?;

        let body = build_batch_request(self.spec.api_model, texts);
        let resp = self
            .http
            .post(&self.url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
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
            anyhow::bail!("gemini embeddings returned {status}: {detail}");
        }

        let value: Value = resp
            .json()
            .await
            .context("gemini embeddings response was not JSON")?;
        parse_embeddings(&value, texts.len())
    }
}

#[async_trait]
impl ClassifierEngine for GeminiEmbedding {
    fn name(&self) -> &'static str {
        self.spec.name
    }

    fn is_local(&self) -> bool {
        false // prompt text is sent to the Generative Language API
    }

    fn context_char_budget(&self) -> usize {
        self.spec.context_char_budget
    }

    fn current_turn_char_budget(&self) -> usize {
        self.spec.current_turn_char_budget
    }

    async fn classify(
        &self,
        complexity_premise: &str,
        image_premise: &str,
        lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        // An empty current turn cannot carry image intent; don't send an empty
        // text to the API (some models reject it).
        let image_text = image_premise.trim();
        let texts: Vec<&str> = if image_text.is_empty() {
            vec![complexity_premise]
        } else {
            vec![complexity_premise, image_text]
        };

        let embeddings = self.embed_batch(&texts).await?;
        let image_embedding = if image_text.is_empty() {
            None
        } else {
            Some(embeddings[1].as_slice())
        };

        Ok(combine_similarities(
            &embeddings[0],
            image_embedding,
            &self.prototypes,
            lexical_image_match,
            self.image_gen_threshold,
        ))
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
            Ok(values
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect())
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
}
