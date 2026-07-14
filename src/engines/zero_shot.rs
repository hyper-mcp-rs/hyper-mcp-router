//! The embedded zero-shot NLI engine (the router's default — fully local,
//! nothing leaves the process).
//!
//! The model is `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, a
//! **binary** NLI model (`id2label = { 0: "entailment", 1: "not_entailment" }`,
//! `type_vocab_size = 0` — no `token_type_ids`), embedded into the binary at
//! build time. These facts are load-bearing for the inference code below.
//!
//! Model-specific concerns owned by this file: the ORT session pool and its
//! CPU/memory sizing (via `crate::planning`), the hypothesis set, the
//! NLI-score combination, and the 512-token context window.

use std::sync::{Arc, Condvar, Mutex};

use async_trait::async_trait;
use ort::session::{builder::GraphOptimizationLevel, Session};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use crate::classifier::{Classification, ClassifierEngine, ModelTier};
use crate::config::ClassifierConfig;
use crate::planning::{detect_memory_budget, overcommit_warnings, plan_inference};
use crate::{MODEL_BYTES, TOKENIZER_BYTES};

/// The model's hard token ceiling (`deberta-v3-xsmall`
/// `max_position_embeddings`). The tokenizer truncates premise+hypothesis pairs
/// to this so a long context window can never exceed what the model can encode.
const MODEL_MAX_TOKENS: usize = 512;

/// Character budget for the complexity-classification window. Deliberately
/// well under [`MODEL_MAX_TOKENS`] (~4 chars/token) so the packed context plus
/// a hypothesis stays inside the model even for dense/code text; tokenizer
/// truncation is the hard backstop.
const CLASSIFICATION_CHAR_BUDGET: usize = 1000;

/// Character budget for the current turn (the image premise / lexical
/// prefilter input). Tight for three model-specific reasons: (1) 400 chars
/// ≈ 100–130 tokens, comfortably inside the 512-token ceiling alongside a
/// hypothesis; (2) all rows of the NLI batch pad to the longest row, so a
/// short image row keeps the whole pass cheap; (3) image-generation intent
/// is a front-of-prompt property for prompts this small model can judge at
/// all. Engines with larger context windows are free to choose much more.
const CURRENT_TURN_CHAR_BUDGET: usize = 400;

/// Which axis (and which value on that axis) a given hypothesis string informs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HypothesisKind {
    Complexity(ModelTier),
    ImageGeneration,
}

// ───────────────────────────────────────────────────────────────────────────
// Session pool
// ───────────────────────────────────────────────────────────────────────────

/// A fixed pool of ORT sessions. `ort 2.0.0-rc.12`'s `Session::run` takes
/// `&mut self`, so concurrent inference is impossible on a single shared
/// session; the pool holds N independent sessions and hands out exclusive
/// access, allowing up to N inferences to run at once.
///
/// Checkout is **blocking** by design: inference already runs inside
/// `spawn_blocking` (see [`ZeroShot::classify`]), so parking that
/// blocking-pool thread until a session frees up is correct and needs no async
/// machinery.
struct SessionPool {
    idle: Mutex<Vec<Session>>,
    available: Condvar,
}

impl SessionPool {
    fn new(sessions: Vec<Session>) -> Self {
        SessionPool {
            idle: Mutex::new(sessions),
            available: Condvar::new(),
        }
    }

    /// Block until a session is free, returning an RAII guard that returns it to
    /// the pool on drop (including on panic/early-return).
    fn acquire(&self) -> anyhow::Result<PooledSession<'_>> {
        let mut idle = self
            .idle
            .lock()
            .map_err(|_| anyhow::anyhow!("session pool mutex poisoned"))?;
        while idle.is_empty() {
            idle = self
                .available
                .wait(idle)
                .map_err(|_| anyhow::anyhow!("session pool mutex poisoned"))?;
        }
        let session = idle.pop().expect("pool non-empty after wait");
        Ok(PooledSession {
            pool: self,
            session: Some(session),
        })
    }
}

/// Exclusive lease on a pooled [`Session`]; derefs to the session and returns it
/// to the pool when dropped.
struct PooledSession<'a> {
    pool: &'a SessionPool,
    session: Option<Session>,
}

impl Drop for PooledSession<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            if let Ok(mut idle) = self.pool.idle.lock() {
                idle.push(session);
                self.pool.available.notify_one();
            }
        }
    }
}

impl std::ops::Deref for PooledSession<'_> {
    type Target = Session;
    fn deref(&self) -> &Session {
        self.session.as_ref().expect("session present until drop")
    }
}

impl std::ops::DerefMut for PooledSession<'_> {
    fn deref_mut(&mut self) -> &mut Session {
        self.session.as_mut().expect("session present until drop")
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Engine
// ───────────────────────────────────────────────────────────────────────────

/// The zero-shot NLI engine.
///
/// All hypotheses are scored in a **single batched forward pass**: the request
/// prompt is paired with each hypothesis, the pairs are tokenised together
/// (padded to the batch's longest sequence), and one `run` over a `[N, seq]`
/// batch produces `[N, 2]` logits. Categorisation is therefore one ORT call
/// per request, independent of hypothesis count.
///
/// The innards live behind an internal `Arc` so [`ClassifierEngine::classify`]
/// can move a handle onto tokio's blocking pool (the pass is CPU-bound and
/// must not stall an async worker).
pub struct ZeroShot {
    inner: Arc<Inner>,
}

struct Inner {
    pool: SessionPool,
    tokenizer: Tokenizer,
    hypotheses: Vec<(HypothesisKind, String)>,
    /// Absolute P(entailment) floor for the image-generation axis.
    image_gen_threshold: f32,
}

impl ZeroShot {
    /// Construct from config, owning the model-specific sizing: detect cores
    /// and the memory budget, derive the CPU/memory plan (see
    /// [`crate::planning`]), apply the engine's own `[classifier.zero-shot]`
    /// settings, and warn — never clamp — on an overcommitted explicit
    /// configuration.
    pub fn from_config(cfg: &ClassifierConfig) -> anyhow::Result<Self> {
        let detected_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let memory_budget = detect_memory_budget();
        let plan = plan_inference(detected_cores, memory_budget);
        let pool_size = cfg.zero_shot.inference_pool_size.unwrap_or(plan.pool_size);
        let intra_op_threads = cfg
            .zero_shot
            .intra_op_threads
            .unwrap_or(plan.intra_op_threads);
        for warning in
            overcommit_warnings(pool_size, intra_op_threads, detected_cores, memory_budget)
        {
            tracing::warn!("{warning}");
        }
        let engine = Self::new(cfg.image_generation_threshold, pool_size, intra_op_threads)?;
        tracing::info!(
            detected_cores,
            memory_budget_mb = memory_budget.map(|b| b / (1024 * 1024)),
            pool_size,
            intra_op_threads,
            "zero-shot inference parallelism configured"
        );
        Ok(engine)
    }

    /// Load the embedded model and tokenizer and build the hypothesis list.
    ///
    /// `pool_size` independent sessions are created so up to that many inferences
    /// can run concurrently (see [`SessionPool`]); it is clamped to at least 1.
    /// `intra_op_threads` sets ORT intra-op parallelism per session (0 = runtime
    /// default). Size the two together so `pool_size * intra_op_threads` stays
    /// near the core count and the pool fits in memory (see
    /// [`crate::planning::plan_inference`]); otherwise sessions oversubscribe
    /// the CPU or risk an OOM kill.
    pub fn new(
        image_gen_threshold: f32,
        pool_size: usize,
        intra_op_threads: usize,
    ) -> anyhow::Result<Self> {
        // `ort::Error` is not `Send + Sync` for every generic parameter, so it
        // cannot flow through `?` into `anyhow::Error`; map each to a string.
        let pool_size = pool_size.max(1);
        let mut sessions = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let mut builder = Session::builder()
                .map_err(|e| anyhow::anyhow!("ort session builder: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("ort optimization level: {e}"))?;
            if intra_op_threads > 0 {
                builder = builder
                    .with_intra_threads(intra_op_threads)
                    .map_err(|e| anyhow::anyhow!("ort intra threads: {e}"))?;
            }
            let session = builder
                .commit_from_memory(MODEL_BYTES)
                .map_err(|e| anyhow::anyhow!("ort commit_from_memory: {e}"))?;
            sessions.push(session);
        }

        let mut tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;
        // Pad each batch to its longest sequence so the pairs stack into one
        // rectangular `[N, seq]` tensor; padded positions are masked out via
        // `attention_mask`, so the pad id is irrelevant to the result.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));
        // Hard backstop against the model's token ceiling. `LongestFirst` trims
        // the longer sequence (the premise, never the short hypothesis), and
        // `Left` drops from the front — i.e. the oldest context is shed first,
        // preserving the most recent turn. The char budget keeps this from
        // firing in the common case; this guarantees correctness in the tail.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MODEL_MAX_TOKENS,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Left,
                stride: 0,
            }))
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;

        let hypotheses = vec![
            (
                HypothesisKind::Complexity(ModelTier::Fast),
                "This is a simple task requiring a short, direct answer with no reasoning.".into(),
            ),
            (
                HypothesisKind::Complexity(ModelTier::Balanced),
                "This is a moderately complex task requiring explanation or multi-step reasoning."
                    .into(),
            ),
            (
                HypothesisKind::Complexity(ModelTier::Frontier),
                "This is a highly complex task requiring deep expertise, analysis, or long-form synthesis."
                    .into(),
            ),
            (
                HypothesisKind::ImageGeneration,
                "This is a request to generate, create, draw, paint, or edit an image or picture."
                    .into(),
            ),
        ];

        Ok(Self {
            inner: Arc::new(Inner {
                pool: SessionPool::new(sessions),
                tokenizer,
                hypotheses,
                image_gen_threshold,
            }),
        })
    }

    /// Synchronous classification (one batched forward pass). CPU-bound —
    /// callers on an async runtime should go through the
    /// [`ClassifierEngine::classify`] wrapper, which moves this onto the
    /// blocking pool. Exposed for tests and benchmarks.
    pub fn classify_sync(
        &self,
        complexity_premise: &str,
        image_premise: &str,
        lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        self.inner
            .classify_sync(complexity_premise, image_premise, lexical_image_match)
    }
}

#[async_trait]
impl ClassifierEngine for ZeroShot {
    fn name(&self) -> &'static str {
        "zero-shot"
    }

    fn context_char_budget(&self) -> usize {
        CLASSIFICATION_CHAR_BUDGET
    }

    fn current_turn_char_budget(&self) -> usize {
        CURRENT_TURN_CHAR_BUDGET
    }

    /// Categorise in a single batched forward pass, then combine the
    /// per-hypothesis scores.
    ///
    /// `complexity_premise` is the windowed recent user context;
    /// `image_premise` is the *current* turn, which alone decides
    /// image-generation intent — an old "draw a cat" turn in the context
    /// window must not trigger image routing for an unrelated request now.
    /// The two premises ride in one batch (different premise per row).
    ///
    /// The pass is CPU-bound, so it runs on tokio's blocking pool — never on
    /// an async worker.
    async fn classify(
        &self,
        complexity_premise: &str,
        image_premise: &str,
        lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        let inner = Arc::clone(&self.inner);
        let complexity_premise = complexity_premise.to_owned();
        let image_premise = image_premise.to_owned();
        tokio::task::spawn_blocking(move || {
            inner.classify_sync(&complexity_premise, &image_premise, lexical_image_match)
        })
        .await
        .map_err(|e| anyhow::anyhow!("classification task panicked: {e}"))?
    }
}

impl Inner {
    fn classify_sync(
        &self,
        complexity_premise: &str,
        image_premise: &str,
        lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        let scores = self.score_hypotheses(complexity_premise, image_premise)?;
        Ok(combine(
            &scores,
            lexical_image_match,
            self.image_gen_threshold,
        ))
    }

    /// Score every hypothesis in one batched NLI pass, returning
    /// `(kind, P(entailment))` in hypothesis order.
    ///
    /// Each hypothesis is paired with its premise (the complexity hypotheses use
    /// `complexity_premise`; the image-generation hypothesis uses `image_premise`)
    /// and packed into `input_ids` (the model consumes no `token_type_ids`). The
    /// pairs are tokenised together and padded to the longest, forming a
    /// `[N, seq]` batch fed through a single `run`; row `i` yields the entailment
    /// probability for hypothesis `i`. `Session::run` requires `&mut self`, so a
    /// session is leased from the pool for the pass; up to `pool_size` passes run
    /// concurrently.
    fn score_hypotheses(
        &self,
        complexity_premise: &str,
        image_premise: &str,
    ) -> anyhow::Result<Vec<(HypothesisKind, f32)>> {
        let pairs: Vec<(&str, &str)> = self
            .hypotheses
            .iter()
            .map(|(kind, hypothesis)| {
                let premise = match kind {
                    HypothesisKind::ImageGeneration => image_premise,
                    HypothesisKind::Complexity(_) => complexity_premise,
                };
                (premise, hypothesis.as_str())
            })
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode_batch: {e}"))?;

        let batch = encodings.len();
        // Padding makes every row equal-length; empty batch is impossible
        // (hypotheses is non-empty), but guard the index defensively.
        let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut ids = Vec::with_capacity(batch * seq_len);
        let mut mask = Vec::with_capacity(batch * seq_len);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
        }

        let input_ids = ndarray::Array2::from_shape_vec((batch, seq_len), ids)?;
        let attn_mask = ndarray::Array2::from_shape_vec((batch, seq_len), mask)?;

        let input_ids = ort::value::TensorRef::from_array_view(&input_ids)
            .map_err(|e| anyhow::anyhow!("ort input_ids tensor: {e}"))?;
        let attn_mask = ort::value::TensorRef::from_array_view(&attn_mask)
            .map_err(|e| anyhow::anyhow!("ort attention_mask tensor: {e}"))?;

        let mut session = self.pool.acquire()?;
        let outputs = session
            .run(ort::inputs![
                "input_ids"      => input_ids,
                "attention_mask" => attn_mask,
            ])
            .map_err(|e| anyhow::anyhow!("ort run: {e}"))?;

        // logits shape [batch, 2], row-major; label order [entailment, not].
        let (_shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("ort extract logits: {e}"))?;

        let scores = self
            .hypotheses
            .iter()
            .enumerate()
            .map(|(i, (kind, _))| (*kind, softmax2(logits[2 * i], logits[2 * i + 1])))
            .collect();
        Ok(scores)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Pure NLI math (session-free, unit-testable)
// ───────────────────────────────────────────────────────────────────────────

/// Fold a set of `(HypothesisKind, P(entailment))` scores plus the
/// lexical-prefilter result into a [`Classification`].
///
/// The complexity argmax is independent of the image-generation flag, and
/// `image_generation` is true iff the lexical prefilter matched OR the image
/// score reaches `image_gen_threshold`. The result is independent of the order
/// scores arrive in (strict-`>` argmax over distinct scores).
fn combine(
    scores: &[(HypothesisKind, f32)],
    lexical_image_match: bool,
    image_gen_threshold: f32,
) -> Classification {
    // Default tier is Balanced (matches the no-user-message default).
    let mut best_tier = ModelTier::Balanced;
    let mut best_score = f32::NEG_INFINITY;
    let mut image_gen_score = f32::NEG_INFINITY;

    for &(kind, p) in scores {
        match kind {
            HypothesisKind::Complexity(tier) => {
                if p > best_score {
                    best_score = p;
                    best_tier = tier;
                }
            }
            HypothesisKind::ImageGeneration => image_gen_score = p,
        }
    }

    Classification {
        complexity: best_tier,
        image_generation: lexical_image_match || image_gen_score >= image_gen_threshold,
    }
}

/// Numerically stable two-class softmax, returning P(entailment).
fn softmax2(entailment: f32, not_entailment: f32) -> f32 {
    let m = entailment.max(not_entailment);
    let e = (entailment - m).exp();
    let n = (not_entailment - m).exp();
    e / (e + n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::DEFAULT_IMAGE_GEN_THRESHOLD;

    // ── softmax2 ──────────────────────────────────────────────────────────
    #[test]
    fn softmax2_is_a_valid_probability() {
        let p = softmax2(2.0, -1.0);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn softmax2_is_monotonic_in_the_logit_gap() {
        let low = softmax2(0.0, 1.0);
        let mid = softmax2(1.0, 1.0);
        let high = softmax2(2.0, 1.0);
        assert!(low < mid && mid < high);
        assert!((mid - 0.5).abs() < 1e-6, "equal logits => 0.5");
    }

    #[test]
    fn softmax2_is_numerically_stable_for_large_inputs() {
        let p = softmax2(1000.0, -1000.0);
        assert!(p.is_finite());
        assert!((p - 1.0).abs() < 1e-6);
        let q = softmax2(-1000.0, 1000.0);
        assert!(q.is_finite());
        assert!(q.abs() < 1e-6);
    }

    // ── combine ───────────────────────────────────────────────────────────
    fn complexity_scores() -> Vec<(HypothesisKind, f32)> {
        vec![
            (HypothesisKind::Complexity(ModelTier::Fast), 0.2),
            (HypothesisKind::Complexity(ModelTier::Balanced), 0.3),
            (HypothesisKind::Complexity(ModelTier::Frontier), 0.9),
        ]
    }

    #[test]
    fn combine_argmax_is_independent_of_image_flag() {
        let mut scores = complexity_scores();
        let without = combine(&scores, false, 0.5);
        scores.push((HypothesisKind::ImageGeneration, 0.99));
        let with = combine(&scores, false, 0.5);
        assert_eq!(without.complexity, ModelTier::Frontier);
        assert_eq!(with.complexity, ModelTier::Frontier);
    }

    #[test]
    fn combine_is_order_independent() {
        let mut scores = complexity_scores();
        scores.push((HypothesisKind::ImageGeneration, 0.1));
        let forward = combine(&scores, false, 0.5);
        scores.reverse();
        let reversed = combine(&scores, false, 0.5);
        assert_eq!(forward, reversed);
    }

    #[test]
    fn combine_image_gen_via_threshold_or_lexical() {
        let mut scores = complexity_scores();
        // below threshold, no lexical match => false
        scores.push((HypothesisKind::ImageGeneration, 0.4));
        assert!(!combine(&scores, false, 0.5).image_generation);
        // lexical match forces true regardless of low score
        assert!(combine(&scores, true, 0.5).image_generation);
        // at/above threshold => true
        scores.pop();
        scores.push((HypothesisKind::ImageGeneration, 0.5));
        assert!(combine(&scores, false, 0.5).image_generation);
    }

    #[test]
    fn combine_threshold_boundary_and_lexical_override() {
        let below = vec![(HypothesisKind::ImageGeneration, 0.49)];
        assert!(!combine(&below, false, 0.5).image_generation);
        let at = vec![(HypothesisKind::ImageGeneration, 0.5)];
        assert!(combine(&at, false, 0.5).image_generation);
        // lexical match forces true even with a very low score
        let low = vec![(HypothesisKind::ImageGeneration, 0.01)];
        assert!(combine(&low, true, 0.5).image_generation);
    }

    // ── model-backed directionality guard (opt-in) ──────────────────────────
    // Loads the embedded ONNX model; run with `cargo test -- --ignored`.
    // Guards against label-order regressions (entailment must be index 0).
    #[test]
    #[ignore = "loads the embedded ONNX model"]
    fn routing_directionality_guard() {
        let clf = ZeroShot::new(DEFAULT_IMAGE_GEN_THRESHOLD, 1, 1).unwrap();
        let simple = clf.classify_sync("hi", "hi", false).unwrap();
        let complex_text = "Derive and rigorously prove the asymptotic time complexity of \
             red-black tree rebalancing across a sequence of insertions and deletions, \
             with a formal amortized analysis.";
        let complex = clf
            .classify_sync(complex_text, complex_text, false)
            .unwrap();
        assert!(
            simple.complexity <= complex.complexity,
            "trivial prompt ({:?}) should route no higher than a complex one ({:?})",
            simple.complexity,
            complex.complexity
        );
    }
}
