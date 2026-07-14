//! The Gemini engine family: shared plumbing here in `mod.rs`, **one file per
//! model** in this directory (`embedding_001.rs`, `embedding_2.rs`,
//! `text_embedding_005.rs`), each defining a [`GeminiSpec`] and delegating to
//! the shared [`GeminiEmbedding`] engine.
//!
//! ## How embedding classification works
//!
//! Unlike the zero-shot NLI engine (which scores hypotheses directly), an
//! embedding model gives us vectors, so classification is done by
//! **anchor prototypes**: at startup, a curated set of exemplar texts per
//! class (fast / balanced / frontier / image-generation) is embedded in one
//! batch call, and each class's exemplar vectors are mean-pooled and
//! normalized into a single prototype. Per request, the complexity window and
//! the current turn are embedded (one batch call) and cosine-scored against
//! the prototypes:
//!
//! - **complexity** = argmax over the three tier prototypes;
//! - **image generation** = the image prototype is the *strict* argmax over
//!   all four prototypes for the current turn AND its similarity clears
//!   `image_generation_threshold` (interpreted as a cosine-similarity floor
//!   for these engines) — OR the lexical prefilter matched.
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
pub mod text_embedding_005;

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::classifier::{Classification, ClassifierEngine, ModelTier};
use crate::config::GeminiEmbeddingConfig;

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

/// Anchor exemplars per class. Deliberately short, unambiguous, and diverse;
/// these calibrate the prototypes, so edit with care and re-check the routing
/// distribution afterwards.
const FAST_ANCHORS: &[&str] = &[
    "What is the capital of France?",
    "How many days are in a leap year?",
    "What is 15% of 240?",
    "Define photosynthesis in one sentence.",
    "What time zone is Tokyo in?",
];
const BALANCED_ANCHORS: &[&str] = &[
    "Explain how a vaccine trains the immune system.",
    "Write a Python function to reverse a string and explain how it works.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "How do noise-cancelling headphones work?",
    "Draft a polite email declining a meeting invitation.",
];
const FRONTIER_ANCHORS: &[&str] = &[
    "Derive and rigorously prove the asymptotic time complexity of red-black tree rebalancing with a formal amortized analysis.",
    "Critically evaluate the philosophical arguments for and against compatibilism regarding free will.",
    "Design a distributed consensus protocol tolerant to Byzantine faults and prove its safety and liveness properties.",
    "Provide a rigorous derivation of the Black-Scholes option pricing formula from first principles.",
];
const IMAGE_ANCHORS: &[&str] = &[
    "Generate an image of a red bicycle on a beach.",
    "Draw a picture of a cat wearing a hat.",
    "Create a logo for a coffee shop.",
    "Paint a watercolor illustration of a mountain sunrise.",
];

/// Class prototype vectors (normalized), in a fixed order.
struct Prototypes {
    fast: Vec<f32>,
    balanced: Vec<f32>,
    frontier: Vec<f32>,
    image: Vec<f32>,
}

/// A remote Gemini embedding engine (shared by every Gemini model file).
pub struct GeminiEmbedding {
    spec: &'static GeminiSpec,
    http: reqwest::Client,
    /// Full `:batchEmbedContents` URL.
    url: String,
    api_key: String,
    /// Bounds concurrent in-flight embedding requests — this engine's
    /// equivalent of the zero-shot session pool.
    permits: Semaphore,
    /// Cosine-similarity floor for the image axis (see module docs).
    image_gen_threshold: f32,
    prototypes: Prototypes,
}

impl GeminiEmbedding {
    /// Build the engine: validate config (the API key is **required**, exactly
    /// like a routed model's key but mandatory), construct the HTTP client,
    /// and embed the class anchors — failing fast on any API problem.
    pub async fn connect(
        spec: &'static GeminiSpec,
        cfg: &GeminiEmbeddingConfig,
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
            prototypes: Prototypes {
                fast: Vec::new(),
                balanced: Vec::new(),
                frontier: Vec::new(),
                image: Vec::new(),
            },
        };

        // Embed every anchor in one batch and mean-pool per class. Startup is
        // the right place to fail on a bad key or unreachable endpoint.
        let anchors: Vec<&str> = FAST_ANCHORS
            .iter()
            .chain(BALANCED_ANCHORS)
            .chain(FRONTIER_ANCHORS)
            .chain(IMAGE_ANCHORS)
            .copied()
            .collect();
        let embeddings = engine
            .embed_batch(&anchors)
            .await
            .with_context(|| format!("embedding class anchors for `{}`", spec.name))?;

        let mut offset = 0;
        let mut take = |n: usize| {
            let slice = &embeddings[offset..offset + n];
            offset += n;
            mean_normalized(slice)
        };
        engine.prototypes = Prototypes {
            fast: take(FAST_ANCHORS.len()),
            balanced: take(BALANCED_ANCHORS.len()),
            frontier: take(FRONTIER_ANCHORS.len()),
            image: take(IMAGE_ANCHORS.len()),
        };

        tracing::info!(
            engine = spec.name,
            embedding_dims = engine.prototypes.fast.len(),
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
// Pure helpers (network-free, unit-testable)
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

/// Mean-pool a set of vectors and L2-normalize the result.
fn mean_normalized(vectors: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = vectors.first() else {
        return Vec::new();
    };
    let mut mean = vec![0.0f32; first.len()];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x;
        }
    }
    let n = vectors.len() as f32;
    for m in &mut mean {
        *m /= n;
    }
    let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for m in &mut mean {
            *m /= norm;
        }
    }
    mean
}

/// Cosine similarity; 0.0 for zero/mismatched vectors (never NaN).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Fold prototype similarities into a [`Classification`].
///
/// Complexity is the argmax over the tier prototypes (ties resolve to the
/// lower tier — cheaper on equal evidence). Image generation requires the
/// image prototype to be the **strict** argmax over all four prototypes for
/// the current turn AND to clear the threshold — or the lexical prefilter.
fn combine_similarities(
    complexity_embedding: &[f32],
    image_embedding: Option<&[f32]>,
    prototypes: &Prototypes,
    lexical_image_match: bool,
    image_gen_threshold: f32,
) -> Classification {
    let tiers = [
        (
            ModelTier::Fast,
            cosine(complexity_embedding, &prototypes.fast),
        ),
        (
            ModelTier::Balanced,
            cosine(complexity_embedding, &prototypes.balanced),
        ),
        (
            ModelTier::Frontier,
            cosine(complexity_embedding, &prototypes.frontier),
        ),
    ];
    let mut best = tiers[0];
    for &t in &tiers[1..] {
        if t.1 > best.1 {
            best = t;
        }
    }

    let image_generation = lexical_image_match
        || image_embedding.is_some_and(|emb| {
            let image_sim = cosine(emb, &prototypes.image);
            let max_tier_sim = [
                cosine(emb, &prototypes.fast),
                cosine(emb, &prototypes.balanced),
                cosine(emb, &prototypes.frontier),
            ]
            .into_iter()
            .fold(f32::NEG_INFINITY, f32::max);
            image_sim > max_tier_sim && image_sim >= image_gen_threshold
        });

    Classification {
        complexity: best.0,
        image_generation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protos() -> Prototypes {
        // Orthogonal unit prototypes: dims = [fast, balanced, frontier, image].
        Prototypes {
            fast: vec![1.0, 0.0, 0.0, 0.0],
            balanced: vec![0.0, 1.0, 0.0, 0.0],
            frontier: vec![0.0, 0.0, 1.0, 0.0],
            image: vec![0.0, 0.0, 0.0, 1.0],
        }
    }

    // ── vector math ─────────────────────────────────────────────────────────
    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        // Degenerate inputs never NaN.
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[0.0], &[0.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn mean_normalized_pools_and_normalizes() {
        let m = mean_normalized(&[vec![2.0, 0.0], vec![0.0, 2.0]]);
        // Mean is [1, 1]; normalized to 1/sqrt(2) each.
        assert!((m[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((m[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(mean_normalized(&[]).is_empty());
        // All-zero input stays zero (no NaN from 0/0).
        assert_eq!(mean_normalized(&[vec![0.0, 0.0]]), vec![0.0, 0.0]);
    }

    // ── request/response shapes ─────────────────────────────────────────────
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

    // ── classification combination ──────────────────────────────────────────
    #[test]
    fn complexity_is_argmax_over_tier_prototypes() {
        let p = protos();
        for (emb, want) in [
            (vec![0.9, 0.2, 0.1, 0.0], ModelTier::Fast),
            (vec![0.1, 0.9, 0.2, 0.0], ModelTier::Balanced),
            (vec![0.1, 0.2, 0.9, 0.0], ModelTier::Frontier),
        ] {
            let c = combine_similarities(&emb, None, &p, false, 0.5);
            assert_eq!(c.complexity, want);
        }
    }

    #[test]
    fn complexity_ties_resolve_to_the_lower_tier() {
        let p = protos();
        // Equidistant from fast and frontier: prefer the cheaper tier.
        let c = combine_similarities(&[0.5, 0.0, 0.5, 0.0], None, &p, false, 0.5);
        assert_eq!(c.complexity, ModelTier::Fast);
    }

    #[test]
    fn image_requires_strict_argmax_and_threshold() {
        let p = protos();
        // Image dominant and above threshold => image intent.
        let c = combine_similarities(
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.1, 0.0, 0.0, 0.9]),
            &p,
            false,
            0.5,
        );
        assert!(c.image_generation);
        // Image dominant (argmax) but below the configured threshold => no
        // image intent. Cosine is scale-invariant, so the vector must be
        // genuinely *angled away* from the prototype, not just small:
        // cos([0.5, 0, 0, 0.86], image) ≈ 0.86, under a 0.9 threshold.
        let c = combine_similarities(
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.5, 0.0, 0.0, 0.86]),
            &p,
            false,
            0.9,
        );
        assert!(!c.image_generation);
        // Image similar but NOT the argmax => no image intent.
        let c = combine_similarities(
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.9, 0.0, 0.0, 0.8]),
            &p,
            false,
            0.5,
        );
        assert!(!c.image_generation);
        // No current-turn embedding => image only via lexical.
        let c = combine_similarities(&[0.9, 0.0, 0.0, 0.0], None, &p, false, 0.5);
        assert!(!c.image_generation);
        let c = combine_similarities(&[0.9, 0.0, 0.0, 0.0], None, &p, true, 0.5);
        assert!(c.image_generation);
    }

    #[test]
    fn image_axis_never_affects_complexity() {
        let p = protos();
        let with_image = combine_similarities(
            &[0.0, 0.9, 0.1, 0.0],
            Some(&[0.0, 0.0, 0.0, 1.0]),
            &p,
            false,
            0.5,
        );
        let without = combine_similarities(&[0.0, 0.9, 0.1, 0.0], None, &p, false, 0.5);
        assert_eq!(with_image.complexity, without.complexity);
        assert!(with_image.image_generation && !without.image_generation);
    }

    #[test]
    fn anchor_classes_are_nonempty_and_distinct() {
        for class in [
            FAST_ANCHORS,
            BALANCED_ANCHORS,
            FRONTIER_ANCHORS,
            IMAGE_ANCHORS,
        ] {
            assert!(!class.is_empty());
        }
        // No anchor text may appear in two classes (it would blur prototypes).
        let all: Vec<&str> = FAST_ANCHORS
            .iter()
            .chain(BALANCED_ANCHORS)
            .chain(FRONTIER_ANCHORS)
            .chain(IMAGE_ANCHORS)
            .copied()
            .collect();
        let unique: std::collections::HashSet<&str> = all.iter().copied().collect();
        assert_eq!(all.len(), unique.len());
    }
}
