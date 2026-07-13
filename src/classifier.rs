//! Prompt classifier: zero-shot NLI session, hypothesis scoring, and the pure
//! metadata helpers used for modality-aware routing.
//!
//! The model is `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, a
//! **binary** NLI model (`id2label = { 0: "entailment", 1: "not_entailment" }`,
//! `type_vocab_size = 0` — no `token_type_ids`). These facts are load-bearing
//! for the inference code below.

use std::sync::{Condvar, LazyLock, Mutex};

use ort::session::{builder::GraphOptimizationLevel, Session};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::{MODEL_BYTES, TOKENIZER_BYTES};

/// Prompt-length guard, in **characters** (never bytes — see [`truncate_prompt`]).
const PROMPT_CHAR_LIMIT: usize = 400;

/// Default upper word count for the trivial fast-path (see [`looks_trivial`]).
/// Overridable via the `--trivial-max-words` CLI flag. Keeps the short-circuit
/// to genuinely terse turns; longer text always reaches the model. A value of 0
/// disables the fast path entirely.
pub const DEFAULT_TRIVIAL_MAX_WORDS: usize = 6;

/// Default absolute P(entailment) floor above which the image-generation
/// hypothesis alone is enough to route to the image modality.
pub const DEFAULT_IMAGE_GEN_THRESHOLD: f32 = 0.5;

// ───────────────────────────────────────────────────────────────────────────
// Type & modality axes
// ───────────────────────────────────────────────────────────────────────────

/// Complexity axis ("type"). `Ord` is intentional — it enforces complexity
/// escalation (see the routing escalation policy in `proxy.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    Fast,
    Balanced,
    Frontier,
}

/// Modality axis — the full Chat Completions v1 surface. A model declares the
/// set it supports; a request requires a (possibly multi-element) subset.
/// Direction is explicit for image/audio (asymmetric support); text is one
/// token. Deliberately **not** ordered: modality is a capability match, never
/// escalated against complexity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Modality {
    /// Text in/out (the baseline; every chat model has it).
    Text,
    /// `image_url` content part (vision / image analysis).
    ImageInput,
    /// `input_audio` content part (speech-to-text style).
    AudioInput,
    /// `file` content part (documents, e.g. PDFs).
    FileInput,
    /// Request `modalities` contains `"audio"` (text-to-speech style).
    AudioOutput,
    /// Image generation / creation (inferred; no native protocol field).
    ImageOutput,
    /// Tool / function calling: the request offers `tools`, so it must route to
    /// a model that can emit tool calls. A capability constraint only — it never
    /// affects the complexity tier.
    Tools,
}

impl Modality {
    /// Kebab-case wire name, used for logging and 415 error bodies.
    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::ImageInput => "image-input",
            Modality::AudioInput => "audio-input",
            Modality::FileInput => "file-input",
            Modality::AudioOutput => "audio-output",
            Modality::ImageOutput => "image-output",
            Modality::Tools => "tools",
        }
    }

    /// Single-bit mask for the [`ModalitySet`] bitset.
    fn bit(self) -> u8 {
        match self {
            Modality::Text => 1 << 0,
            Modality::ImageInput => 1 << 1,
            Modality::AudioInput => 1 << 2,
            Modality::FileInput => 1 << 3,
            Modality::AudioOutput => 1 << 4,
            Modality::ImageOutput => 1 << 5,
            Modality::Tools => 1 << 6,
        }
    }

    /// All modalities, in a stable order for iteration/logging.
    const ALL: [Modality; 7] = [
        Modality::Text,
        Modality::ImageInput,
        Modality::AudioInput,
        Modality::FileInput,
        Modality::AudioOutput,
        Modality::ImageOutput,
        Modality::Tools,
    ];
}

/// A small set of [`Modality`] values, backed by a bitset. Used for the
/// superset matching that drives model selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ModalitySet(u8);

impl ModalitySet {
    /// An empty set.
    pub fn new() -> Self {
        ModalitySet(0)
    }

    /// Add a modality to the set.
    pub fn insert(&mut self, m: Modality) {
        self.0 |= m.bit();
    }

    /// Whether the set contains `m`.
    pub fn contains(&self, m: Modality) -> bool {
        self.0 & m.bit() != 0
    }

    /// Whether `self` covers every modality in `required` (i.e. `required` is a
    /// subset of `self`).
    pub fn is_superset(&self, required: &ModalitySet) -> bool {
        self.0 & required.0 == required.0
    }

    /// Kebab-case names of the contained modalities, in stable order — for
    /// logging and 415 error bodies. Never includes user content.
    pub fn to_kebab_vec(&self) -> Vec<&'static str> {
        Modality::ALL
            .iter()
            .filter(|m| self.contains(**m))
            .map(|m| m.as_str())
            .collect()
    }
}

impl FromIterator<Modality> for ModalitySet {
    fn from_iter<I: IntoIterator<Item = Modality>>(iter: I) -> Self {
        let mut set = ModalitySet::new();
        for m in iter {
            set.insert(m);
        }
        set
    }
}

/// What the text classifier resolves to. `complexity` (the tier axis) is always
/// produced; `image_generation` (the modality axis) is an orthogonal intent
/// flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Classification {
    pub complexity: ModelTier,
    pub image_generation: bool,
}

impl Classification {
    /// The default used when there is no user message or classification fails:
    /// balanced complexity, no image generation.
    pub fn balanced_default() -> Self {
        Classification {
            complexity: ModelTier::Balanced,
            image_generation: false,
        }
    }
}

/// Which axis (and which value on that axis) a given hypothesis string informs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypothesisKind {
    Complexity(ModelTier),
    ImageGeneration,
}

// ───────────────────────────────────────────────────────────────────────────
// Inference parallelism planning
// ───────────────────────────────────────────────────────────────────────────

/// How to split the host's available cores between concurrent inference
/// sessions (`pool_size`) and intra-op threads per session (`intra_op_threads`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InferencePlan {
    /// Number of independent [`Session`]s (max concurrent inferences).
    pub pool_size: usize,
    /// ORT intra-op threads per session (0 = let the runtime decide).
    pub intra_op_threads: usize,
}

/// Derive an [`InferencePlan`] from the number of available cores. The embedded
/// NLI model is small and scales poorly per-inference, so we favor concurrency:
/// cap intra-op parallelism at 2 and give each session ~2 cores. Always yields
/// at least one (single-threaded) session, and never budgets more threads than
/// cores (`pool_size * intra_op_threads <= cores`).
pub fn plan_inference(available_cores: usize) -> InferencePlan {
    let cores = available_cores.max(1);
    let intra_op_threads = if cores >= 2 { 2 } else { 1 };
    let pool_size = (cores / intra_op_threads).max(1);
    InferencePlan {
        pool_size,
        intra_op_threads,
    }
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
/// `spawn_blocking`, so parking that blocking-pool thread until a session frees
/// up is correct and needs no async machinery.
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
// Classifier
// ───────────────────────────────────────────────────────────────────────────

/// The zero-shot NLI classifier.
///
/// All hypotheses are scored in a **single batched forward pass**: the request
/// prompt is paired with each hypothesis, the pairs are tokenised together
/// (padded to the batch's longest sequence), and one `run` over a `[N, seq]`
/// batch produces `[N, 2]` logits. Categorisation is therefore one ORT call
/// per request, independent of hypothesis count.
///
/// NOTE (deviation from the spec): the spec assumes `ort::Session::run` takes
/// `&self` (lock-free concurrent runs). In the pinned `ort = 2.0.0-rc.12`,
/// `run` takes `&mut self`, so a shared `&Session` cannot run inference. To
/// still run inferences concurrently, the classifier holds a [`SessionPool`] of
/// independent sessions and checks one out per request; only the batched `run`
/// is inside the lease. Tokenisation, request forwarding, and SSE streaming are
/// all outside it and remain fully concurrent.
pub struct Classifier {
    pool: SessionPool,
    tokenizer: Tokenizer,
    hypotheses: Vec<(HypothesisKind, String)>,
    /// Absolute P(entailment) floor for the image-generation axis.
    image_gen_threshold: f32,
    /// Word ceiling for the trivial fast-path (see [`looks_trivial`]).
    trivial_max_words: usize,
}

impl Classifier {
    /// Load the embedded model and tokenizer and build the hypothesis list.
    ///
    /// `pool_size` independent sessions are created so up to that many inferences
    /// can run concurrently (see [`SessionPool`]); it is clamped to at least 1.
    /// `intra_op_threads` sets ORT intra-op parallelism per session (0 = runtime
    /// default). Size the two together so `pool_size * intra_op_threads` stays
    /// near the core count (see [`plan_inference`]); otherwise sessions
    /// oversubscribe the CPU.
    pub fn new(
        image_gen_threshold: f32,
        trivial_max_words: usize,
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
            pool: SessionPool::new(sessions),
            tokenizer,
            hypotheses,
            image_gen_threshold,
            trivial_max_words,
        })
    }

    /// Model-free fast-path classification using this classifier's configured
    /// [`trivial_max_words`](Self::trivial_max_words) ceiling. Returns `Some`
    /// only for turns that can be routed without the NLI pass; see
    /// [`fast_path_classification`].
    pub fn fast_path(&self, prompt: &str) -> Option<Classification> {
        fast_path_classification(prompt, self.trivial_max_words)
    }

    /// Categorise the prompt in a single batched forward pass, then combine the
    /// per-hypothesis scores.
    ///
    /// CPU-bound: callers should invoke this on a blocking thread (the proxy
    /// does so via `spawn_blocking`) rather than on an async worker.
    pub fn classify(&self, prompt: &str) -> anyhow::Result<Classification> {
        let lexical_image_match = looks_like_image_generation(prompt);
        let scores = self.score_hypotheses(prompt)?;
        Ok(combine(
            &scores,
            lexical_image_match,
            self.image_gen_threshold,
        ))
    }

    /// Score every hypothesis against `prompt` in one batched NLI pass,
    /// returning `(kind, P(entailment))` in hypothesis order.
    ///
    /// Each hypothesis is paired with the prompt (packed into `input_ids`; the
    /// model consumes no `token_type_ids`). The pairs are tokenised together
    /// and padded to the longest, forming a `[N, seq]` batch fed through a
    /// single `run`; row `i` yields the entailment probability for hypothesis
    /// `i`. `Session::run` requires `&mut self`, so a session is leased from the
    /// pool for the pass; up to `pool_size` passes run concurrently.
    fn score_hypotheses(&self, prompt: &str) -> anyhow::Result<Vec<(HypothesisKind, f32)>> {
        let pairs: Vec<(&str, &str)> = self
            .hypotheses
            .iter()
            .map(|(_, hypothesis)| (prompt, hypothesis.as_str()))
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
// Pure helpers (session-free, unit-testable)
// ───────────────────────────────────────────────────────────────────────────

/// Fold a set of `(HypothesisKind, P(entailment))` scores plus the
/// lexical-prefilter result into a [`Classification`].
///
/// The complexity argmax is independent of the image-generation flag, and
/// `image_generation` is true iff the lexical prefilter matched OR the image
/// score reaches `image_gen_threshold`. The result is independent of the order
/// scores arrive in (strict-`>` argmax over distinct scores).
pub fn combine(
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

/// High-precision lexical prefilter for image *generation* intent. Requires an
/// image-creation verb within a short window of an image noun, or explicit
/// text-to-image phrasing / tool names. Deliberately conservative: because it
/// is OR-ed with the NLI signal, a false positive cross-routes a text request
/// to the image backend, so the pattern avoids weak matches like "draw a
/// conclusion", "create a plan", or "picture this".
pub fn looks_like_image_generation(prompt: &str) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?ix)
              \b(generate|create|draw|paint|render|design|illustrate|sketch|produce)\b
              .{0,40}?
              \b(image|images|picture|pictures|photo|photos|illustration|drawing|
                 painting|logo|icon|artwork|graphic|portrait|wallpaper|poster|avatar)\b
            | \btext[- ]to[- ]image\b
            | \b(dall[- ]?e|midjourney|stable[ -]diffusion|flux)\b
            ",
        )
        .expect("valid image-generation regex")
    });

    RE.is_match(prompt)
}

/// Cheap lexical/length guard for *trivially simple* turns — greetings and
/// acknowledgements like "hi", "ok", "thanks", "please continue". A match means
/// the turn can skip the (serialized, ~12 ms) NLI pass and route as [`ModelTier::Fast`].
///
/// Deliberately conservative on three axes, all of which must hold:
/// 1. **Short** — at most `max_words` words ("short ≠ simple", so a length cap
///    alone is unsafe; a terse "prove X" must not slip through). `max_words == 0`
///    disables the fast path (nothing is ever trivial).
/// 2. **No reasoning cues** — none of the complexity markers below (guards
///    against "ok, now derive the formula").
/// 3. **Looks like filler** — matches the acknowledgement/greeting set.
///
/// It only ever routes *down* to Fast; history escalation still applies on top,
/// so a terse turn on a deep/agentic thread is unaffected.
pub fn looks_trivial(prompt: &str, max_words: usize) -> bool {
    /// Reasoning cues that veto the fast path even on a short, filler-looking turn.
    static COMPLEXITY_MARKERS: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?ix)\b(
                 prove|proof|derive|derivation|analy[sz]e|analysis|evaluate|assess|
                 design|architect|optimi[sz]e|integrate|differentiate|refactor|debug|
                 implement|algorithm|complexity|theorem|rigorous|synthesi[sz]e|critique|
                 compare|contrast|explain|summari[sz]e|translate|calculate|solve
               )\b",
        )
        .expect("valid complexity-marker regex")
    });

    /// Acknowledgement / greeting / short-confirmation phrases, anchored at the
    /// start of the (trimmed) prompt.
    static ACK_PHRASES: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?ix)^\s*(?:
                 hi|hey|hello|yo|sup
               | ok(?:ay)?|k
               | ye(?:s|ah|p)|yup
               | no|nope|nah
               | thanks|thank\s+you|thx|ty|ta
               | sure|cool|great|nice|awesome|perfect|excellent
               | got\s+it|gotcha|understood|makes\s+sense|sounds\s+good|will\s+do
               | continue|go\s+on|carry\s+on|keep\s+going|proceed
               | please\s+continue
               | good\s+(?:morning|afternoon|evening|night)
               | bye|goodbye|see\s+you|cheers
               | no\s+problem|np
               | how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|whats\s+up
             )\b",
        )
        .expect("valid acknowledgement regex")
    });

    let trimmed = prompt.trim();
    let words = trimmed.split_whitespace().count();
    (1..=max_words).contains(&words)
        && !COMPLEXITY_MARKERS.is_match(trimmed)
        && ACK_PHRASES.is_match(trimmed)
}

/// Model-free routing decision for the cheap cases, returning `Some` only when a
/// classification can be made without the NLI pass:
/// - image-creation intent (lexical) is left to the full path (returns `None`),
///   so the NLI image-gen threshold still applies;
/// - otherwise a [`looks_trivial`] turn resolves to [`ModelTier::Fast`].
///
/// Callers that get `None` must run the classifier as usual. `max_words` bounds
/// the trivial-turn length (see [`looks_trivial`]); pass 0 to disable.
pub fn fast_path_classification(prompt: &str, max_words: usize) -> Option<Classification> {
    if looks_like_image_generation(prompt) {
        return None;
    }
    if looks_trivial(prompt, max_words) {
        return Some(Classification {
            complexity: ModelTier::Fast,
            image_generation: false,
        });
    }
    None
}

/// Numerically stable two-class softmax, returning P(entailment).
pub fn softmax2(entailment: f32, not_entailment: f32) -> f32 {
    let m = entailment.max(not_entailment);
    let e = (entailment - m).exp();
    let n = (not_entailment - m).exp();
    e / (e + n)
}

/// Extract the text to classify from the parsed request JSON: the `content` of
/// the last `role == "user"` message. Multi-part content concatenates the
/// `text` fields of its parts, ignoring non-text parts. Returns `None` when no
/// user message exists (the caller then uses the default classification).
pub fn extract_prompt(body: &serde_json::Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;

    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))?;

    let content = last_user.get("content")?;

    if let Some(text) = content.as_str() {
        return Some(text.to_owned());
    }

    if let Some(parts) = content.as_array() {
        let text: String = parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        return Some(text);
    }

    None
}

/// Truncate the prompt to [`PROMPT_CHAR_LIMIT`] **characters** (never byte
/// slicing, which would panic on a multi-byte UTF-8 boundary). A conservative
/// guard for the model's 512-token limit; the full request JSON is always
/// forwarded unchanged to the backend.
pub fn truncate_prompt(prompt: &str) -> String {
    prompt.chars().take(PROMPT_CHAR_LIMIT).collect()
}

/// Deterministic modalities required by a request, from content-part types
/// (input), the `modalities` request field (output), and the `tools`/`functions`
/// fields (tool calling). `ImageOutput` is **not** decided here — it has no
/// protocol field and is inferred by the classifier.
///
/// This function must not call the classifier: it is metadata-only.
pub fn detect_required_modalities(body: &serde_json::Value) -> ModalitySet {
    let mut set = ModalitySet::new();
    // Text I/O is always in play for chat/completions.
    set.insert(Modality::Text);

    // --- Input: scan message content parts ---
    for msg in body["messages"].as_array().into_iter().flatten() {
        for part in msg
            .get("content")
            .and_then(|c| c.as_array())
            .into_iter()
            .flatten()
        {
            match part.get("type").and_then(|t| t.as_str()) {
                // "input_image" tolerated as a newer alias for robustness.
                Some("image_url") | Some("input_image") => set.insert(Modality::ImageInput),
                Some("input_audio") => set.insert(Modality::AudioInput),
                Some("file") => set.insert(Modality::FileInput),
                _ => {}
            }
        }
    }

    // --- Output: the `modalities` request field (defaults to ["text"]) ---
    if let Some(mods) = body.get("modalities").and_then(|m| m.as_array()) {
        if mods.iter().any(|m| m.as_str() == Some("audio")) {
            set.insert(Modality::AudioOutput);
        }
    }

    // --- Tool calling: a request offering tools must route to a tool-capable
    // model. Both the current `tools` array and the deprecated `functions`
    // array count; an empty array does not.
    let offers = |field: &str| {
        body.get(field)
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty())
    };
    if offers("tools") || offers("functions") {
        set.insert(Modality::Tools);
    }

    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    // ── looks_like_image_generation ─────────────────────────────────────────
    #[test]
    fn lexical_positives_match() {
        for p in [
            "generate an image of a cat",
            "create a logo",
            "draw a picture of a house",
            "make it with midjourney",
            "text-to-image of a sunset",
            "please render an illustration of a dragon",
            "use stable diffusion to produce artwork",
            "dall-e a portrait",
        ] {
            assert!(looks_like_image_generation(p), "should match: {p:?}");
        }
    }

    #[test]
    fn lexical_negatives_do_not_match() {
        for p in [
            "draw a conclusion",
            "create a plan",
            "picture this scenario",
            "paint a grim outlook",
            "the big picture",
            "generate a report",
        ] {
            assert!(!looks_like_image_generation(p), "should NOT match: {p:?}");
        }
    }

    // ── plan_inference ──────────────────────────────────────────────────────
    #[test]
    fn inference_plan_stays_within_core_budget() {
        for cores in [1usize, 2, 3, 4, 8, 16, 18, 32, 64] {
            let p = plan_inference(cores);
            assert!(p.pool_size >= 1, "pool_size >= 1 for {cores}");
            assert!(p.intra_op_threads >= 1, "intra_op >= 1 for {cores}");
            assert!(
                p.pool_size * p.intra_op_threads <= cores.max(1),
                "budget exceeded for {cores}: {p:?}"
            );
        }
    }

    #[test]
    fn inference_plan_degenerate_cores() {
        // Zero/one core collapses to a single single-threaded session.
        let one = InferencePlan {
            pool_size: 1,
            intra_op_threads: 1,
        };
        assert_eq!(plan_inference(0), one);
        assert_eq!(plan_inference(1), one);
    }

    #[test]
    fn inference_plan_pool_grows_with_cores() {
        assert!(plan_inference(4).pool_size <= plan_inference(8).pool_size);
        assert!(plan_inference(8).pool_size <= plan_inference(18).pool_size);
        assert_eq!(plan_inference(18).pool_size, 9);
    }

    // ── looks_trivial / fast_path_classification ─────────────────────────────
    #[test]
    fn trivial_positives_match() {
        for p in [
            "hi",
            "hello there",
            "ok",
            "okay",
            "yes",
            "no",
            "thanks",
            "thank you",
            "thanks!",
            "sure",
            "cool, got it",
            "got it",
            "understood",
            "continue",
            "please continue",
            "go on",
            "good morning",
            "how are you?",
            "what's up",
            "sounds good",
            "bye",
        ] {
            assert!(
                looks_trivial(p, DEFAULT_TRIVIAL_MAX_WORDS),
                "should be trivial: {p:?}"
            );
        }
    }

    #[test]
    fn trivial_negatives_do_not_match() {
        for p in [
            // not filler at all
            "What is the capital of France?",
            "Tell me more about that.",
            // filler-prefixed but carries a reasoning cue (marker veto)
            "ok, now prove the theorem",
            "sure, please derive the formula",
            "yes, analyze these results",
            // too long
            "thanks so much for the detailed and very thorough breakdown you gave",
            // short but technical (short != simple)
            "Integrate sin(x).",
            "Solve for x.",
            "Prove P != NP.",
        ] {
            assert!(
                !looks_trivial(p, DEFAULT_TRIVIAL_MAX_WORDS),
                "should NOT be trivial: {p:?}"
            );
        }
    }

    #[test]
    fn trivial_max_words_zero_disables_fast_path() {
        // A ceiling of 0 makes nothing trivial, disabling the short-circuit.
        assert!(!looks_trivial("ok", 0));
        assert!(fast_path_classification("ok thanks", 0).is_none());
    }

    #[test]
    fn trivial_respects_word_ceiling() {
        // "ok sure thanks" is 3 words: trivial at ceiling 3, not at ceiling 2.
        assert!(looks_trivial("ok sure thanks", 3));
        assert!(!looks_trivial("ok sure thanks", 2));
    }

    #[test]
    fn fast_path_returns_fast_for_trivial() {
        let c = fast_path_classification("ok thanks", DEFAULT_TRIVIAL_MAX_WORDS)
            .expect("trivial => Some");
        assert_eq!(c.complexity, ModelTier::Fast);
        assert!(!c.image_generation);
    }

    #[test]
    fn fast_path_defers_image_generation_to_model() {
        // Short and image-y, but image intent must take the full (NLI) path.
        assert!(
            fast_path_classification("draw a picture of a cat", DEFAULT_TRIVIAL_MAX_WORDS)
                .is_none()
        );
        assert!(
            fast_path_classification("generate an image of a dog", DEFAULT_TRIVIAL_MAX_WORDS)
                .is_none()
        );
    }

    #[test]
    fn fast_path_defers_non_trivial_to_model() {
        assert!(fast_path_classification(
            "Explain how TLS handshakes work.",
            DEFAULT_TRIVIAL_MAX_WORDS
        )
        .is_none());
        assert!(fast_path_classification(
            "Prove that sqrt 2 is irrational.",
            DEFAULT_TRIVIAL_MAX_WORDS
        )
        .is_none());
    }

    // ── detect_required_modalities ──────────────────────────────────────────
    #[test]
    fn modality_text_always_present() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let set = detect_required_modalities(&body);
        assert!(set.contains(Modality::Text));
    }

    #[test]
    fn modality_absent_messages_still_text() {
        let body = json!({});
        let set = detect_required_modalities(&body);
        assert_eq!(set.to_kebab_vec(), vec!["text"]);
    }

    #[test]
    fn modality_content_part_types_map_correctly() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "input_audio", "input_audio": {}},
            {"type": "file", "file": {}},
            {"type": "text", "text": "describe"},
        ]}]});
        let set = detect_required_modalities(&body);
        assert!(set.contains(Modality::ImageInput));
        assert!(set.contains(Modality::AudioInput));
        assert!(set.contains(Modality::FileInput));
        assert!(set.contains(Modality::Text));
        assert!(!set.contains(Modality::AudioOutput));
    }

    #[test]
    fn modality_input_image_alias() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "input_image", "image_url": {"url": "x"}},
        ]}]});
        assert!(detect_required_modalities(&body).contains(Modality::ImageInput));
    }

    #[test]
    fn modality_audio_output_from_request_field() {
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}],
            "modalities": ["text", "audio"],
        });
        let set = detect_required_modalities(&body);
        assert!(set.contains(Modality::AudioInput));
        assert!(set.contains(Modality::AudioOutput));
    }

    #[test]
    fn modality_string_content_handled() {
        let body = json!({"messages": [{"role": "user", "content": "just text"}]});
        let set = detect_required_modalities(&body);
        assert_eq!(set.to_kebab_vec(), vec!["text"]);
    }

    #[test]
    fn modality_tools_kebab_name() {
        assert_eq!(Modality::Tools.as_str(), "tools");
    }

    #[test]
    fn modality_tools_detected_from_tools_array() {
        let body = json!({
            "messages": [{"role": "user", "content": "what's the weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        });
        assert!(detect_required_modalities(&body).contains(Modality::Tools));
    }

    #[test]
    fn modality_tools_detected_from_legacy_functions_array() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [{"name": "get_weather"}],
        });
        assert!(detect_required_modalities(&body).contains(Modality::Tools));
    }

    #[test]
    fn modality_tools_absent_or_empty_not_detected() {
        // No tools field at all.
        let none = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(!detect_required_modalities(&none).contains(Modality::Tools));
        // Empty tools array does not require the capability.
        let empty = json!({"messages": [{"role": "user", "content": "hi"}], "tools": []});
        assert!(!detect_required_modalities(&empty).contains(Modality::Tools));
    }

    // ── extract_prompt ──────────────────────────────────────────────────────
    #[test]
    fn extract_last_user_message_wins() {
        let body = json!({"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]});
        assert_eq!(extract_prompt(&body).as_deref(), Some("second"));
    }

    #[test]
    fn extract_multipart_content_concatenated() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "world"},
        ]}]});
        assert_eq!(extract_prompt(&body).as_deref(), Some("hello world"));
    }

    #[test]
    fn extract_missing_user_message_returns_none() {
        let body = json!({"messages": [{"role": "system", "content": "sys"}]});
        assert!(extract_prompt(&body).is_none());
    }

    // ── truncate_prompt ─────────────────────────────────────────────────────
    #[test]
    fn truncate_takes_400_chars() {
        let long = "a".repeat(1000);
        assert_eq!(truncate_prompt(&long).chars().count(), 400);
    }

    #[test]
    fn truncate_handles_multibyte_utf8_without_panicking() {
        // Each '😀' is 4 bytes; byte slicing at 400 would panic on a boundary.
        let s = "😀".repeat(500);
        let truncated = truncate_prompt(&s);
        assert_eq!(truncated.chars().count(), 400);
    }

    // ── model-backed directionality guard (opt-in) ──────────────────────────
    // Loads the embedded ONNX model; run with `cargo test -- --ignored`.
    // Guards against label-order regressions (entailment must be index 0).
    #[test]
    #[ignore = "loads the embedded ONNX model"]
    fn routing_directionality_guard() {
        let clf =
            Classifier::new(DEFAULT_IMAGE_GEN_THRESHOLD, DEFAULT_TRIVIAL_MAX_WORDS, 1, 1).unwrap();
        let simple = clf.classify("hi").unwrap();
        let complex = clf
            .classify(
                "Derive and rigorously prove the asymptotic time complexity of red-black \
                 tree rebalancing across a sequence of insertions and deletions, with a \
                 formal amortized analysis.",
            )
            .unwrap();
        assert!(
            simple.complexity <= complex.complexity,
            "trivial prompt ({:?}) should route no higher than a complex one ({:?})",
            simple.complexity,
            complex.complexity
        );
    }

    // ── ModalitySet ─────────────────────────────────────────────────────────
    #[test]
    fn modality_set_superset_logic() {
        let mut caps = ModalitySet::new();
        caps.insert(Modality::Text);
        caps.insert(Modality::AudioInput);
        caps.insert(Modality::AudioOutput);

        let mut req = ModalitySet::new();
        req.insert(Modality::Text);
        req.insert(Modality::AudioInput);
        assert!(caps.is_superset(&req));

        req.insert(Modality::ImageInput);
        assert!(!caps.is_superset(&req));
    }
}
