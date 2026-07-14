//! The OpenAI embedding engine family: shared transport here in `mod.rs`,
//! **one file per model** in this directory (`text_embedding_3_small.rs`,
//! `text_embedding_3_large.rs`), each defining an [`OpenAiSpec`] and
//! delegating to the shared [`OpenAiEmbedding`] engine.
//!
//! The classification *method* (anchor prototypes, cosine scoring) is
//! provider-neutral and lives in `crate::engines::embedding`; this file owns
//! only what is OpenAI-specific: the `/v1/embeddings` wire format (array
//! `input` for batching; `data[].embedding` keyed by `index` in the
//! response) and bearer-token auth.
//!
//! Anchor embedding happens once at startup and **fails fast** on any API
//! problem (bad key, unreachable endpoint), so misconfiguration is a clear
//! boot error, not a silent stream of balanced-default fallbacks.
//!
//! ## Privacy and failure notes
//!
//! Prompt text (the classification window and current turn) is sent to the
//! OpenAI API. Per-request failures degrade to the balanced default via the
//! proxy, as with every engine. Error messages never echo user content.

pub mod text_embedding_3_large;
pub mod text_embedding_3_small;

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::classifier::{Classification, ClassifierEngine};
use crate::config::RemoteEmbeddingConfig;
use crate::engines::embedding::{anchor_texts, build_prototypes, combine_similarities, Prototypes};

/// Default public OpenAI API endpoint. The engine appends `/v1/embeddings`,
/// so a `base_url` override must NOT include the `/v1` suffix (unlike routed
/// models' `base_url`, which does).
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Everything that differs between OpenAI embedding models. Each model file
/// declares one of these as a `const`.
pub struct OpenAiSpec {
    /// Engine id (matches `ClassifierModel::as_str`).
    pub name: &'static str,
    /// Model id sent to the API, e.g. `text-embedding-3-small`.
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

/// A remote OpenAI embedding engine (shared by every OpenAI model file).
pub struct OpenAiEmbedding {
    spec: &'static OpenAiSpec,
    http: reqwest::Client,
    /// Full `/v1/embeddings` URL.
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

impl OpenAiEmbedding {
    /// Build the engine: validate config (the API key is **required**, exactly
    /// like a routed model's key but mandatory), construct the HTTP client,
    /// and embed the class anchors — failing fast on any API problem.
    pub async fn connect(
        spec: &'static OpenAiSpec,
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
        let url = format!("{base}/v1/embeddings");

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

        let mut engine = OpenAiEmbedding {
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
            "openai embedding engine ready"
        );
        Ok(engine)
    }

    /// Embed `texts` in one `/v1/embeddings` call (array input), bounded by
    /// the concurrency semaphore. Never echoes the texts into errors.
    async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("embedding concurrency semaphore closed"))?;

        let body = build_embeddings_request(self.spec.api_model, texts);
        let resp = self
            .http
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("openai embeddings request failed")?;

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
            anyhow::bail!("openai embeddings returned {status}: {detail}");
        }

        let value: Value = resp
            .json()
            .await
            .context("openai embeddings response was not JSON")?;
        parse_embeddings(&value, texts.len())
    }
}

#[async_trait]
impl ClassifierEngine for OpenAiEmbedding {
    fn name(&self) -> &'static str {
        self.spec.name
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
        // text to the API (the embeddings endpoint rejects empty strings).
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

/// Build the `/v1/embeddings` request body (array input batches every text
/// into one call).
fn build_embeddings_request(api_model: &str, texts: &[&str]) -> Value {
    json!({
        "model": api_model,
        "input": texts,
    })
}

/// Extract `expected` embedding vectors from a `/v1/embeddings` response.
/// The API documents `data` in input order but keys each item by `index`;
/// vectors are placed by index so a permuted response cannot mis-assign
/// premises to prototypes.
fn parse_embeddings(value: &Value, expected: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let data = value
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow::anyhow!("openai response missing `data` array"))?;
    if data.len() != expected {
        anyhow::bail!(
            "openai returned {} embeddings, expected {expected}",
            data.len()
        );
    }

    let mut out: Vec<Option<Vec<f32>>> = vec![None; expected];
    for item in data {
        let index = item
            .get("index")
            .and_then(|i| i.as_u64())
            .ok_or_else(|| anyhow::anyhow!("openai embedding item missing `index`"))?
            as usize;
        let embedding = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| anyhow::anyhow!("openai embedding item missing `embedding`"))?;
        if embedding.is_empty() {
            anyhow::bail!("openai returned an empty embedding vector");
        }
        let slot = out
            .get_mut(index)
            .ok_or_else(|| anyhow::anyhow!("openai embedding index {index} out of range"))?;
        if slot.is_some() {
            anyhow::bail!("openai returned duplicate embedding index {index}");
        }
        *slot = Some(
            embedding
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect(),
        );
    }
    out.into_iter()
        .map(|v| {
            v.ok_or_else(|| anyhow::anyhow!("openai response left an embedding index unfilled"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_request_shape() {
        let body = build_embeddings_request("text-embedding-3-small", &["a", "b"]);
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"][0], "a");
        assert_eq!(body["input"][1], "b");
    }

    #[test]
    fn parse_embeddings_respects_index_order() {
        // Deliberately permuted: index must win over array position.
        let permuted = serde_json::json!({"data": [
            {"index": 1, "embedding": [3.0, 4.0]},
            {"index": 0, "embedding": [1.0, 2.0]},
        ]});
        let parsed = parse_embeddings(&permuted, 2).unwrap();
        assert_eq!(parsed, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn parse_embeddings_rejects_malformed_responses() {
        let ok = serde_json::json!({"data": [{"index": 0, "embedding": [1.0]}]});
        assert!(parse_embeddings(&ok, 1).is_ok());
        // Wrong count.
        assert!(parse_embeddings(&ok, 2).is_err());
        // Missing data array.
        assert!(parse_embeddings(&serde_json::json!({}), 1).is_err());
        // Empty vector.
        let empty = serde_json::json!({"data": [{"index": 0, "embedding": []}]});
        assert!(parse_embeddings(&empty, 1).is_err());
        // Out-of-range and duplicate indices.
        let oob = serde_json::json!({"data": [{"index": 5, "embedding": [1.0]}]});
        assert!(parse_embeddings(&oob, 1).is_err());
        let dup = serde_json::json!({"data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 0, "embedding": [2.0]},
        ]});
        assert!(parse_embeddings(&dup, 2).is_err());
    }
}
