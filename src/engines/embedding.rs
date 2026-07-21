//! Anchor-prototype classification support shared by every embedding-based
//! engine, regardless of provider. **Not a model** — provider families
//! (`gemini/`, `vertex/`) own their transport; this module owns the method
//! and the provider-neutral engine built on top of it
//! ([`RemoteEmbeddingEngine`]).
//!
//! ## How embedding classification works
//!
//! Unlike the embedded NLI engine (which scores hypotheses directly), an
//! embedding model gives us vectors, so classification is done by **anchor
//! prototypes**: at startup, a curated set of exemplar texts per class
//! (fast / balanced / frontier / image-generation) is embedded in one batch
//! call, and each class's exemplar vectors are mean-pooled and normalized
//! into a single prototype. Per request, the complexity window and the
//! current turn are embedded and cosine-scored against the prototypes:
//!
//! - **complexity** = argmax over the three tier prototypes (ties resolve to
//!   the lower tier — cheaper on equal evidence);
//! - **image generation** = the image prototype is the *strict* argmax over
//!   all four prototypes for the current turn AND its similarity clears
//!   `image_generation_threshold` (interpreted as a cosine-similarity floor
//!   for embedding engines) — OR the lexical prefilter matched.
//!
//! Anchors and premises must be embedded by the **same model** for their
//! similarities to be comparable; prototypes are never shared across engines.
//!
//! ## How a provider family plugs in
//!
//! A family (`gemini/`, `vertex/`) implements [`EmbedTexts`] — "embed these
//! texts, return one vector per text, in order" — owning everything
//! provider-specific: wire format, auth, endpoint layout, and concurrency
//! bounds. Each model file declares a [`RemoteSpec`] `const`. The family's
//! build path hands both to [`RemoteEmbeddingEngine::connect`], which embeds
//! the anchors (with bounded retry on transient upstream failures), builds
//! the prototypes, and returns a ready engine — an engine therefore never
//! exists in a half-initialized, prototype-less state.

use std::future::Future;
use std::time::Duration;

use anyhow::Context as _;
use async_trait::async_trait;
use serde_json::Value;

use crate::classifier::{Classification, ClassifierEngine, ModelTier};

/// Anchor exemplars per class. Deliberately short, unambiguous, and diverse;
/// these calibrate the prototypes for every embedding engine, so edit with
/// care and re-check the routing distribution afterwards.
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

/// Every anchor text, in the fixed order expected by [`build_prototypes`].
/// Embed these (in order, in one batch) and hand the vectors back.
pub fn anchor_texts() -> Vec<&'static str> {
    FAST_ANCHORS
        .iter()
        .chain(BALANCED_ANCHORS)
        .chain(FRONTIER_ANCHORS)
        .chain(IMAGE_ANCHORS)
        .copied()
        .collect()
}

/// Class prototype vectors (normalized). Constructed only via
/// [`build_prototypes`], so a `Prototypes` value always carries real,
/// anchor-derived vectors.
pub struct Prototypes {
    fast: Vec<f32>,
    balanced: Vec<f32>,
    frontier: Vec<f32>,
    image: Vec<f32>,
}

impl Prototypes {
    /// Embedding dimensionality.
    pub fn dims(&self) -> usize {
        self.fast.len()
    }
}

/// Fold the anchor embeddings (in [`anchor_texts`] order) into per-class
/// prototypes. Errors if the count does not match the anchor list — a
/// provider returning the wrong number of vectors must be loud.
pub fn build_prototypes(embeddings: &[Vec<f32>]) -> anyhow::Result<Prototypes> {
    let expected = anchor_texts().len();
    if embeddings.len() != expected {
        anyhow::bail!(
            "expected {expected} anchor embeddings, got {}",
            embeddings.len()
        );
    }
    let mut offset = 0;
    let mut take = |n: usize| {
        let slice = &embeddings[offset..offset + n];
        offset += n;
        mean_normalized(slice)
    };
    Ok(Prototypes {
        fast: take(FAST_ANCHORS.len()),
        balanced: take(BALANCED_ANCHORS.len()),
        frontier: take(FRONTIER_ANCHORS.len()),
        image: take(IMAGE_ANCHORS.len()),
    })
}

/// Fold prototype similarities into a [`Classification`] (see module docs for
/// the exact rules). `image_embedding` is `None` when the current turn was
/// empty — image intent is then only reachable via the lexical prefilter.
///
/// `engine` names the calling engine in the debug-level similarity log — this
/// module is shared by every remote family (and several rungs of a ladder
/// may run it), so the event must say which one scored.
pub fn combine_similarities(
    engine: &str,
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

    // `(image similarity, max tier similarity)` of the current-turn
    // embedding; image intent requires the image prototype to win the argmax
    // AND clear the threshold.
    let image_axis = image_embedding.map(|emb| {
        let image_sim = cosine(emb, &prototypes.image);
        let max_tier_sim = [
            cosine(emb, &prototypes.fast),
            cosine(emb, &prototypes.balanced),
            cosine(emb, &prototypes.frontier),
        ]
        .into_iter()
        .fold(f32::NEG_INFINITY, f32::max);
        (image_sim, max_tier_sim)
    });
    let image_generation = lexical_image_match
        || image_axis.is_some_and(|(image_sim, max_tier_sim)| {
            image_sim > max_tier_sim && image_sim >= image_gen_threshold
        });

    // Debug-level score breakdown — the remote analogue of the embedded
    // engine's "NLI hypothesis scores" event: every tier's cosine similarity
    // plus both halves of the image decision (argmax opponent and
    // threshold), so a routing decision reads as "how close was the call".
    // Similarities only, never premise text. The `enabled!` guard skips the
    // rendering allocation on the hot path when debug is off.
    if tracing::enabled!(tracing::Level::DEBUG) {
        let similarities: Vec<(&'static str, f32)> = vec![
            ("fast", tiers[0].1),
            ("balanced", tiers[1].1),
            ("frontier", tiers[2].1),
        ];
        tracing::debug!(
            engine,
            similarities = ?similarities,
            complexity = ?best.0,
            complexity_similarity = best.1,
            image_similarity = image_axis.map(|(s, _)| s),
            image_max_tier_similarity = image_axis.map(|(_, m)| m),
            image_gen_threshold,
            lexical_image_match,
            image_generation,
            "embedding prototype similarities"
        );
    }

    Classification {
        complexity: best.0,
        image_generation,
    }
}

// ───────────────────────────────────────────────────────────────────────────
// The shared remote engine
// ───────────────────────────────────────────────────────────────────────────

/// Everything that differs between remote embedding models, regardless of
/// provider family. Each model file declares one of these as a `const`.
pub struct RemoteSpec {
    /// Engine id (matches `ClassifierModel::as_str`).
    pub name: &'static str,
    /// Model identifier sent to the API — a resource path for the Gemini
    /// surface (`models/gemini-embedding-001`), a bare publisher model id
    /// for Vertex (`text-embedding-005`).
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

/// A provider transport: embed a batch of texts, returning one vector per
/// text **in input order**. Implementations own everything provider-specific
/// — wire format, auth, endpoint layout, concurrency bounds — and must never
/// echo the texts into errors.
#[async_trait]
pub trait EmbedTexts: Send + Sync {
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// The provider-neutral remote embedding engine: a [`RemoteSpec`], a
/// transport, and the anchor prototypes built at [`connect`] time
/// (see [`Self::connect`]). One classify body serves every remote family.
pub struct RemoteEmbeddingEngine<T: EmbedTexts> {
    spec: &'static RemoteSpec,
    transport: T,
    /// Cosine-similarity floor for the image axis (see module docs).
    image_gen_threshold: f32,
    prototypes: Prototypes,
}

impl<T: EmbedTexts> RemoteEmbeddingEngine<T> {
    /// Embed the class anchors through `transport`, build the prototypes,
    /// and only then construct the engine — a `RemoteEmbeddingEngine` never
    /// exists with placeholder prototypes. Startup is the right place to
    /// fail on a bad credential or unreachable endpoint; **transient**
    /// upstream failures (429/5xx, send errors — see [`TransientUpstream`])
    /// are retried a bounded number of times so a single blip at boot does
    /// not kill the process.
    pub async fn connect(
        spec: &'static RemoteSpec,
        transport: T,
        image_gen_threshold: f32,
    ) -> anyhow::Result<Self> {
        let anchors = anchor_texts();
        let embeddings = retry_transient(spec.name, || transport.embed(&anchors))
            .await
            .with_context(|| format!("embedding class anchors for `{}`", spec.name))?;
        let prototypes = build_prototypes(&embeddings)?;
        Ok(RemoteEmbeddingEngine {
            spec,
            transport,
            image_gen_threshold,
            prototypes,
        })
    }

    /// Embedding dimensionality of the built prototypes, for the family
    /// "engine ready" startup logs.
    pub fn prototype_dims(&self) -> usize {
        self.prototypes.dims()
    }
}

#[async_trait]
impl<T: EmbedTexts> ClassifierEngine for RemoteEmbeddingEngine<T> {
    fn name(&self) -> &'static str {
        self.spec.name
    }

    fn is_local(&self) -> bool {
        false // every remote transport sends prompt text to a provider API
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
        // text to the API (some models reject empty content).
        let image_text = image_premise.trim();
        let texts: Vec<&str> = if image_text.is_empty() {
            vec![complexity_premise]
        } else {
            vec![complexity_premise, image_text]
        };

        // Deliberately NO retry here (retry exists only around startup anchor
        // embedding in `connect`): retrying per-request classification would
        // duplicate billable embedding work and blur the failure signal — the
        // proxy already degrades a failed classification to the balanced
        // default, which is the intended behavior under upstream trouble.
        // The span isolates the remote embed call's latency from the local
        // cosine scoring around it.
        let embed_span = tracing::info_span!(
            "embed",
            otel.kind = "client",
            engine = self.spec.name,
            texts = texts.len(),
        );
        let embeddings = {
            use tracing::Instrument;
            self.transport.embed(&texts).instrument(embed_span).await?
        };
        let image_embedding = if image_text.is_empty() {
            None
        } else {
            Some(embeddings[1].as_slice())
        };

        Ok(combine_similarities(
            self.spec.name,
            &embeddings[0],
            image_embedding,
            &self.prototypes,
            lexical_image_match,
            self.image_gen_threshold,
        ))
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Transient-failure marking and startup retry
// ───────────────────────────────────────────────────────────────────────────

/// Marker attached (via `anyhow` context) to transport errors that are worth
/// retrying at **startup**: request send/transport failures and HTTP 429 or
/// 5xx responses. Permanent failures (other 4xx, malformed JSON, shape
/// mismatches) never carry it. Checked by [`retry_transient`] through
/// `anyhow`'s downcast, which traverses context layers.
#[derive(Debug)]
pub(crate) struct TransientUpstream;

impl std::fmt::Display for TransientUpstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("transient upstream failure")
    }
}

impl std::error::Error for TransientUpstream {}

/// Whether an HTTP status indicates a transient upstream condition
/// (rate-limiting or a server-side failure) rather than a caller mistake.
pub(crate) fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

/// Build an error for a transient upstream failure: `message` stays the
/// outermost (displayed) layer, with the [`TransientUpstream`] marker as its
/// source so [`retry_transient`] can recognize it.
pub(crate) fn transient_error(message: String) -> anyhow::Error {
    anyhow::Error::new(TransientUpstream).context(message)
}

/// Whether any layer of the error chain carries the [`TransientUpstream`]
/// marker.
pub(crate) fn is_transient(err: &anyhow::Error) -> bool {
    err.downcast_ref::<TransientUpstream>().is_some()
}

/// Run `op` with bounded retry on [`TransientUpstream`]-marked failures:
/// up to 3 attempts total, backing off ~500ms then ~1s (each plus a small
/// `SystemTime`-derived jitter), so the worst-case added boot delay stays
/// under ~2s. Permanent errors return immediately.
///
/// Used **only** for startup anchor embedding — per-request classification
/// is deliberately never retried (see the note in
/// [`RemoteEmbeddingEngine::classify`](ClassifierEngine::classify)).
pub(crate) async fn retry_transient<T, F, Fut>(engine: &str, op: F) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    // ~500ms then ~1s: worst-case added boot delay stays under ~2s.
    retry_transient_with_backoff(engine, &[500, 1000], op).await
}

/// [`retry_transient`] with an injectable backoff schedule (whose length + 1
/// is the total attempt count) so unit tests need not sleep for real.
async fn retry_transient_with_backoff<T, F, Fut>(
    engine: &str,
    backoff_ms: &[u64],
    mut op: F,
) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    let max_attempts = backoff_ms.len() as u32 + 1;
    let mut attempt: u32 = 1;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt < max_attempts && is_transient(&err) => {
                let backoff = backoff_ms[attempt as usize - 1] + jitter_ms();
                tracing::warn!(
                    engine,
                    attempt,
                    backoff_ms = backoff,
                    error = format!("{err:#}"),
                    "transient failure embedding startup anchors; retrying"
                );
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                attempt += 1;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Small backoff jitter (0–99ms) derived from the system clock — enough to
/// de-synchronize engines retrying simultaneously, without a new dependency.
fn jitter_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()))
        .unwrap_or(0)
        % 100
}

// ───────────────────────────────────────────────────────────────────────────
// Shared JSON helpers
// ───────────────────────────────────────────────────────────────────────────

/// Convert a JSON array of embedding components into an `f32` vector,
/// erroring on any non-numeric element instead of silently coercing it to
/// zero (which would corrupt similarities undetectably). The message names
/// the offending index only — never the content.
pub(crate) fn numeric_vector(values: &[Value]) -> anyhow::Result<Vec<f32>> {
    values
        .iter()
        .enumerate()
        .map(|(index, x)| {
            x.as_f64().map(|f| f as f32).ok_or_else(|| {
                anyhow::anyhow!("embedding vector element at index {index} is not a number")
            })
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::Mutex;

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

    // ── prototype construction ──────────────────────────────────────────────
    #[test]
    fn build_prototypes_requires_exact_anchor_count() {
        let n = anchor_texts().len();
        let embeddings: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0, 0.0]).collect();
        let p = build_prototypes(&embeddings).unwrap();
        assert_eq!(p.dims(), 2);
        // Wrong count is a loud error.
        assert!(build_prototypes(&embeddings[..n - 1]).is_err());
        assert!(build_prototypes(&[]).is_err());
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
            let c = combine_similarities("test-engine", &emb, None, &p, false, 0.5);
            assert_eq!(c.complexity, want);
        }
    }

    #[test]
    fn complexity_ties_resolve_to_the_lower_tier() {
        let p = protos();
        // Equidistant from fast and frontier: prefer the cheaper tier.
        let c = combine_similarities("test-engine", &[0.5, 0.0, 0.5, 0.0], None, &p, false, 0.5);
        assert_eq!(c.complexity, ModelTier::Fast);
    }

    #[test]
    fn image_requires_strict_argmax_and_threshold() {
        let p = protos();
        // Image dominant and above threshold => image intent.
        let c = combine_similarities(
            "test-engine",
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
            "test-engine",
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.5, 0.0, 0.0, 0.86]),
            &p,
            false,
            0.9,
        );
        assert!(!c.image_generation);
        // Image similar but NOT the argmax => no image intent.
        let c = combine_similarities(
            "test-engine",
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.9, 0.0, 0.0, 0.8]),
            &p,
            false,
            0.5,
        );
        assert!(!c.image_generation);
        // No current-turn embedding => image only via lexical.
        let c = combine_similarities("test-engine", &[0.9, 0.0, 0.0, 0.0], None, &p, false, 0.5);
        assert!(!c.image_generation);
        let c = combine_similarities("test-engine", &[0.9, 0.0, 0.0, 0.0], None, &p, true, 0.5);
        assert!(c.image_generation);
    }

    #[test]
    fn image_axis_never_affects_complexity() {
        let p = protos();
        let with_image = combine_similarities(
            "test-engine",
            &[0.0, 0.9, 0.1, 0.0],
            Some(&[0.0, 0.0, 0.0, 1.0]),
            &p,
            false,
            0.5,
        );
        let without =
            combine_similarities("test-engine", &[0.0, 0.9, 0.1, 0.0], None, &p, false, 0.5);
        assert_eq!(with_image.complexity, without.complexity);
        assert!(with_image.image_generation && !without.image_generation);
    }

    #[test]
    fn similarities_logged_per_tier_at_debug() {
        let p = protos();
        // Orthogonal unit vectors give exact cosines — stable log rendering.
        let out = crate::test_support::captured_log(tracing::Level::DEBUG, || {
            combine_similarities(
                "test-engine",
                &[1.0, 0.0, 0.0, 0.0],
                Some(&[0.0, 0.0, 0.0, 1.0]),
                &p,
                false,
                0.5,
            );
        });
        assert!(
            out.contains("embedding prototype similarities"),
            "got: {out}"
        );
        // The engine is named — this module is shared across remote families.
        assert!(out.contains("test-engine"), "got: {out}");
        // Every tier's similarity…
        assert!(out.contains(r#"("fast", 1.0)"#), "got: {out}");
        assert!(out.contains(r#"("balanced", 0.0)"#), "got: {out}");
        assert!(out.contains(r#"("frontier", 0.0)"#), "got: {out}");
        // …and both halves of the image decision, plus the outcome.
        assert!(out.contains("image_similarity=1.0"), "got: {out}");
        assert!(out.contains("image_max_tier_similarity=0.0"), "got: {out}");
        assert!(out.contains("image_gen_threshold=0.5"), "got: {out}");
        assert!(out.contains("complexity=Fast"), "got: {out}");
        assert!(out.contains("image_generation=true"), "got: {out}");
    }

    #[test]
    fn similarities_stay_silent_below_debug() {
        let p = protos();
        let out = crate::test_support::captured_log(tracing::Level::INFO, || {
            combine_similarities("test-engine", &[1.0, 0.0, 0.0, 0.0], None, &p, false, 0.5);
        });
        assert!(out.is_empty(), "got: {out}");
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
        let all = anchor_texts();
        let unique: std::collections::HashSet<&str> = all.iter().copied().collect();
        assert_eq!(all.len(), unique.len());
    }

    // ── the generic remote engine ───────────────────────────────────────────

    static TEST_SPEC: RemoteSpec = RemoteSpec {
        name: "fake-embedding-engine",
        api_model: "models/fake",
        context_char_budget: 6000,
        current_turn_char_budget: 2000,
        default_max_concurrency: 4,
        default_request_timeout_secs: 10,
    };

    /// Premise texts with fixed fake embeddings (dims =
    /// [fast, balanced, frontier, image], matching the one-hot prototypes
    /// the anchor mapping below produces).
    const FRONTIER_PREMISE: &str = "premise: frontier-flavored";
    const IMAGE_TURN: &str = "turn: image-flavored";

    fn fake_embedding(text: &str) -> Vec<f32> {
        if FAST_ANCHORS.contains(&text) {
            return vec![1.0, 0.0, 0.0, 0.0];
        }
        if BALANCED_ANCHORS.contains(&text) {
            return vec![0.0, 1.0, 0.0, 0.0];
        }
        if FRONTIER_ANCHORS.contains(&text) {
            return vec![0.0, 0.0, 1.0, 0.0];
        }
        if IMAGE_ANCHORS.contains(&text) {
            return vec![0.0, 0.0, 0.0, 1.0];
        }
        match text {
            FRONTIER_PREMISE => vec![0.1, 0.2, 0.9, 0.0],
            IMAGE_TURN => vec![0.2, 0.0, 0.0, 0.9],
            _ => vec![0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Deterministic in-memory transport: records the text count of every
    /// embed call and can be switched into a failing mode.
    #[derive(Default)]
    struct FakeTransport {
        calls: Mutex<Vec<usize>>,
        fail: AtomicBool,
    }

    #[async_trait]
    impl EmbedTexts for FakeTransport {
        async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.len());
            if self.fail.load(Ordering::SeqCst) {
                anyhow::bail!("fake transport outage");
            }
            Ok(texts.iter().map(|t| fake_embedding(t)).collect())
        }
    }

    async fn connected_engine() -> RemoteEmbeddingEngine<FakeTransport> {
        RemoteEmbeddingEngine::connect(&TEST_SPEC, FakeTransport::default(), 0.5)
            .await
            .expect("connect against the fake transport")
    }

    #[tokio::test]
    async fn connect_embeds_anchors_in_one_call_and_builds_prototypes() {
        let engine = connected_engine().await;
        assert_eq!(engine.prototype_dims(), 4);
        assert_eq!(
            *engine.transport.calls.lock().unwrap(),
            vec![anchor_texts().len()],
            "anchors must go out as one batched embed"
        );
        // Trait plumbing comes straight from the spec.
        assert_eq!(engine.name(), "fake-embedding-engine");
        assert!(!engine.is_local());
        assert_eq!(engine.context_char_budget(), 6000);
        assert_eq!(engine.current_turn_char_budget(), 2000);
    }

    #[tokio::test]
    async fn classify_sends_two_texts_with_image_premise_one_without() {
        let engine = connected_engine().await;

        engine
            .classify(FRONTIER_PREMISE, IMAGE_TURN, false)
            .await
            .unwrap();
        engine.classify(FRONTIER_PREMISE, "", false).await.unwrap();
        engine
            .classify(FRONTIER_PREMISE, "   \n\t", false)
            .await
            .unwrap();

        let calls = engine.transport.calls.lock().unwrap();
        // calls[0] is the anchor batch from connect.
        assert_eq!(calls[1], 2, "non-empty image premise rides along");
        assert_eq!(calls[2], 1, "empty image premise must not be sent");
        assert_eq!(calls[3], 1, "whitespace image premise must not be sent");
    }

    #[tokio::test]
    async fn classify_matches_combine_similarities_expectations() {
        let engine = connected_engine().await;

        // Frontier-flavored premise with an image-flavored current turn:
        // complexity argmax = frontier, image strict-argmax above threshold.
        let c = engine
            .classify(FRONTIER_PREMISE, IMAGE_TURN, false)
            .await
            .unwrap();
        assert_eq!(c.complexity, ModelTier::Frontier);
        assert!(c.image_generation);

        // Same premise, no current turn: image only reachable lexically.
        let c = engine.classify(FRONTIER_PREMISE, "", false).await.unwrap();
        assert_eq!(c.complexity, ModelTier::Frontier);
        assert!(!c.image_generation);
        let c = engine.classify(FRONTIER_PREMISE, "", true).await.unwrap();
        assert!(c.image_generation);
    }

    #[tokio::test]
    async fn classify_propagates_transport_errors() {
        let engine = connected_engine().await;
        engine.transport.fail.store(true, Ordering::SeqCst);
        let err = engine
            .classify(FRONTIER_PREMISE, IMAGE_TURN, false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("fake transport outage"));
    }

    // ── transient marking and startup retry ─────────────────────────────────

    #[test]
    fn transient_marker_survives_added_context_layers() {
        let err = transient_error("upstream returned 503".into());
        assert!(is_transient(&err));
        assert_eq!(err.to_string(), "upstream returned 503");
        // anyhow downcast traverses context layers added on top.
        let wrapped = err.context("embedding class anchors for `x`");
        assert!(is_transient(&wrapped));
        // Unmarked errors are permanent.
        assert!(!is_transient(&anyhow::anyhow!("bad request")));
    }

    #[test]
    fn transient_statuses_are_429_and_5xx_only() {
        use reqwest::StatusCode;
        assert!(is_transient_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_transient_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_transient_status(StatusCode::SERVICE_UNAVAILABLE));
        assert!(!is_transient_status(StatusCode::BAD_REQUEST));
        assert!(!is_transient_status(StatusCode::UNAUTHORIZED));
        assert!(!is_transient_status(StatusCode::NOT_FOUND));
    }

    #[tokio::test]
    async fn retry_recovers_after_transient_failures() {
        // Zero backoff: tokio's paused clock needs a feature the crate does
        // not enable, so the schedule is injected instead of slept through.
        let attempts = AtomicU32::new(0);
        let value = retry_transient_with_backoff("test-engine", &[0, 0], || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if n < 2 {
                    Err(transient_error(format!("blip {n}")))
                } else {
                    Ok(42)
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(value, 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn permanent_errors_are_never_retried() {
        let attempts = AtomicU32::new(0);
        let err = retry_transient_with_backoff::<u32, _, _>("test-engine", &[0, 0], || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err(anyhow::anyhow!("bad request")) }
        })
        .await
        .unwrap_err();
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "exactly one attempt");
        assert!(err.to_string().contains("bad request"));
    }

    #[tokio::test]
    async fn persistent_transient_failure_gives_up_after_three_attempts() {
        let attempts = AtomicU32::new(0);
        let err = retry_transient_with_backoff::<u32, _, _>("test-engine", &[0, 0], || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err(transient_error("still down".into())) }
        })
        .await
        .unwrap_err();
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert!(is_transient(&err));
    }

    // Exercises the real production schedule through `connect` (~1.7s of
    // real sleep — accepted for this one end-to-end retry test).
    #[tokio::test]
    async fn connect_retries_transient_anchor_failures() {
        /// Fails the first two embeds with a transient error, then succeeds.
        struct FlakyTransport {
            attempts: AtomicU32,
        }

        #[async_trait]
        impl EmbedTexts for FlakyTransport {
            async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
                let n = self.attempts.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    return Err(transient_error("startup blip".into()));
                }
                Ok(texts.iter().map(|t| fake_embedding(t)).collect())
            }
        }

        let transport = FlakyTransport {
            attempts: AtomicU32::new(0),
        };
        let engine = RemoteEmbeddingEngine::connect(&TEST_SPEC, transport, 0.5)
            .await
            .expect("connect must survive two transient blips");
        assert_eq!(engine.transport.attempts.load(Ordering::SeqCst), 3);
        assert_eq!(engine.prototype_dims(), 4);
    }

    // ── shared JSON helpers ─────────────────────────────────────────────────

    #[test]
    fn numeric_vector_parses_numbers_and_rejects_anything_else() {
        let ok = [
            serde_json::json!(1.0),
            serde_json::json!(-2),
            serde_json::json!(0.5),
        ];
        assert_eq!(numeric_vector(&ok).unwrap(), vec![1.0, -2.0, 0.5]);

        for bad in [
            serde_json::json!("0.5"),
            serde_json::json!(null),
            serde_json::json!([1.0]),
            serde_json::json!({"v": 1.0}),
            serde_json::json!(true),
        ] {
            let values = [serde_json::json!(1.0), bad];
            let err = numeric_vector(&values).unwrap_err().to_string();
            assert!(
                err.contains("index 1") && err.contains("not a number"),
                "got: {err}"
            );
        }
    }
}
