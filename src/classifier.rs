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
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use crate::{MODEL_BYTES, TOKENIZER_BYTES};

/// Prompt-length guard, in **characters** (never bytes — see [`truncate_prompt`]).
const PROMPT_CHAR_LIMIT: usize = 400;

/// The classifier model's hard token ceiling (`deberta-v3-xsmall`
/// `max_position_embeddings`). The tokenizer truncates premise+hypothesis pairs
/// to this so a long context window can never exceed what the model can encode.
const MODEL_MAX_TOKENS: usize = 512;

/// Character budget for the complexity-classification window
/// ([`build_classification_window`]). Deliberately well under [`MODEL_MAX_TOKENS`]
/// (~4 chars/token) so the packed context plus a hypothesis stays inside the
/// model even for dense/code text; tokenizer truncation is the hard backstop.
pub const CLASSIFICATION_CHAR_BUDGET: usize = 1000;

/// Default upper word count for the trivial fast-path (see [`looks_trivial`]).
/// Overridable via the `--trivial-max-words` CLI flag. Keeps the short-circuit
/// to genuinely terse turns; longer text always reaches the model. A value of 0
/// disables the fast path entirely.
pub const DEFAULT_TRIVIAL_MAX_WORDS: usize = 6;

/// Default absolute P(entailment) floor above which the image-generation
/// hypothesis alone is enough to route to the image modality.
pub const DEFAULT_IMAGE_GEN_THRESHOLD: f32 = 0.5;

// ───────────────────────────────────────────────────────────────────────
// Complexity axis (the modality axis lives in `crate::modality`)
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

// ───────────────────────────────────────────────────────────────────────
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
    /// near the core count and the pool fits in memory (see
    /// [`crate::planning::plan_inference`]); otherwise sessions oversubscribe
    /// the CPU or risk an OOM kill.
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
            pool: SessionPool::new(sessions),
            tokenizer,
            hypotheses,
            image_gen_threshold,
            trivial_max_words,
        })
    }

    /// The configured trivial fast-path word ceiling (see [`looks_trivial`]).
    pub fn trivial_max_words(&self) -> usize {
        self.trivial_max_words
    }

    /// Categorise in a single batched forward pass, then combine the per-hypothesis
    /// scores.
    ///
    /// `complexity_premise` is the windowed recent user context
    /// ([`build_classification_window`]); `image_premise` is the *current* turn,
    /// which alone decides image-generation intent — an old "draw a cat" turn in
    /// the context window must not trigger image routing for an unrelated request
    /// now. The two premises ride in one batch (different premise per row).
    ///
    /// CPU-bound: callers should invoke this on a blocking thread (the proxy
    /// does so via `spawn_blocking`) rather than on an async worker.
    ///
    /// `lexical_image_match` is the (precomputed) result of
    /// [`looks_like_image_generation`] on `image_premise` — the proxy already
    /// needs it before deciding whether to classify at all, so it is passed in
    /// rather than recomputed here.
    pub fn classify(
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

    /// Score every hypothesis against `prompt` in one batched NLI pass,
    /// returning `(kind, P(entailment))` in hypothesis order.
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
/// 3. **Is filler, entirely** — the whole (trimmed) turn must consist of
///    acknowledgement/greeting phrases and punctuation. A matching *prefix* is
///    not enough: "ok tell me about quantum computing" starts with an ack but
///    carries a substantive request, so it must reach the model.
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

    /// Acknowledgement / greeting / short-confirmation phrases. The **entire**
    /// trimmed prompt must be a punctuation-separated sequence of these — an
    /// end anchor, not a prefix match — so an ack-prefixed substantive turn
    /// ("ok tell me about X") is never mistaken for filler.
    static ACK_PHRASES: LazyLock<Regex> = LazyLock::new(|| {
        /// One filler phrase.
        const PHRASE: &str = r"(?:
                 (?:hi|hey|hello)(?:\s+there)?|yo|sup
               | ok(?:ay)?|k
               | ye(?:s|ah|p)|yup
               | no|nope|nah
               | thanks|thank\s+you|thx|ty|ta
               | please
               | sure|cool|great|nice|awesome|perfect|excellent
               | got\s+it|gotcha|understood|makes\s+sense|sounds\s+good|will\s+do
               | continue|go\s+on|carry\s+on|keep\s+going|proceed
               | good\s+(?:morning|afternoon|evening|night)
               | bye|goodbye|see\s+you|cheers
               | no\s+problem|np
               | how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|whats\s+up
             )";
        Regex::new(&format!(
            r#"(?ix)^\s*{PHRASE}(?:[\s,.;:!?'"()-]+{PHRASE})*[\s,.;:!?'"()-]*$"#
        ))
        .expect("valid acknowledgement regex")
    });

    let trimmed = prompt.trim();
    let words = trimmed.split_whitespace().count();
    (1..=max_words).contains(&words)
        && !COMPLEXITY_MARKERS.is_match(trimmed)
        && ACK_PHRASES.is_match(trimmed)
}

/// Numerically stable two-class softmax, returning P(entailment).
pub fn softmax2(entailment: f32, not_entailment: f32) -> f32 {
    let m = entailment.max(not_entailment);
    let e = (entailment - m).exp();
    let n = (not_entailment - m).exp();
    e / (e + n)
}

/// The text content of a single message: a string `content` verbatim, or the
/// concatenated `text` fields of a multi-part `content` (non-text parts ignored).
fn message_text(msg: &serde_json::Value) -> Option<String> {
    let content = msg.get("content")?;
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

/// The current turn's text: the `content` of the last `role == "user"` message.
/// Used for the image-generation axis (a per-current-turn intent) and logging.
/// Returns `None` when no user message exists.
pub fn extract_prompt(body: &serde_json::Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))?;
    message_text(last_user)
}

/// Whether any user message carries non-empty text. Distinguishes "the user
/// actually said something (however trivial)" from "there is no usable user
/// text at all" (no user messages, or only empty/attachment-only content) —
/// the former can honestly route as chit-chat, the latter falls back to the
/// balanced default.
pub fn has_nonempty_user_text(body: &serde_json::Value) -> bool {
    let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
        return false;
    };
    messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .filter_map(message_text)
        .any(|t| !t.trim().is_empty())
}

/// Build the complexity-classification premise by walking the conversation's
/// **user** turns newest→oldest, skipping trivially-simple ones ([`looks_trivial`]),
/// and accumulating substantive turns until the conversation start or
/// `char_budget` is reached. Surviving turns are returned in chronological order,
/// newline-joined. Returns `None` when no substantive user text remains (e.g.
/// pure chit-chat), which the caller routes as the baseline tier without the model.
///
/// This is what lets a terse follow-up inherit the difficulty of its recent
/// context: "ok, continue" is pruned as trivial, and the walk-back reaches the
/// substantive turns behind it. Only *user* turns are considered — assistant
/// responses (usually the longest messages) are skipped, so the budget stretches
/// across many turns of actual intent. Filler is pruned, so the window ages by
/// *substantive* turns, not by chit-chat.
///
/// **System messages are deliberately excluded.** They are usually static
/// deployment boilerplate ("You are a helpful assistant…") that would consume
/// budget and skew every conversation toward the same tier; the user's own
/// turns are the signal for how hard *this* request is.
pub fn build_classification_window(
    body: &serde_json::Value,
    trivial_max_words: usize,
    char_budget: usize,
) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    let mut collected: Vec<String> = Vec::new(); // newest-first
    let mut used = 0usize;

    for msg in messages.iter().rev() {
        if used >= char_budget {
            break;
        }
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let Some(text) = message_text(msg) else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() || looks_trivial(text, trivial_max_words) {
            continue;
        }
        // Truncate by characters (never bytes) to what remains of the budget.
        let piece: String = text.chars().take(char_budget - used).collect();
        used += piece.chars().count();
        collected.push(piece);
    }

    if collected.is_empty() {
        return None;
    }
    collected.reverse(); // chronological: oldest context first, current turn last
    Some(collected.join("\n"))
}

/// Truncate the prompt to [`PROMPT_CHAR_LIMIT`] **characters** (never byte
/// slicing, which would panic on a multi-byte UTF-8 boundary). A conservative
/// guard for the model's 512-token limit; the full request JSON is always
/// forwarded unchanged to the backend.
pub fn truncate_prompt(prompt: &str) -> String {
    prompt.chars().take(PROMPT_CHAR_LIMIT).collect()
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

    // ── looks_trivial ─────────────────────────────────────────────────────────
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
            // filler-prefixed, short, no reasoning cue — must still reach the
            // model (the ack match is whole-string, not prefix)
            "ok tell me about quantum computing",
            "no rewrite it in Rust",
            "yes but why is the sky blue",
            "thanks, what about France?",
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
    fn trivial_max_words_zero_disables_pruning() {
        // A ceiling of 0 makes nothing trivial, so no turn is pruned as filler.
        assert!(!looks_trivial("ok", 0));
        let body = json!({"messages": [{"role": "user", "content": "ok"}]});
        // With pruning disabled, even "ok" survives into the window.
        assert_eq!(
            build_classification_window(&body, 0, WIN_BUDGET).as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn trivial_respects_word_ceiling() {
        // "ok sure thanks" is 3 words: trivial at ceiling 3, not at ceiling 2.
        assert!(looks_trivial("ok sure thanks", 3));
        assert!(!looks_trivial("ok sure thanks", 2));
    }

    #[test]
    fn trivial_accepts_multi_phrase_filler_with_punctuation() {
        for p in [
            "ok, thanks!",
            "yes please",
            "great, sounds good!!",
            "ok... continue",
        ] {
            assert!(
                looks_trivial(p, DEFAULT_TRIVIAL_MAX_WORDS),
                "should be trivial: {p:?}"
            );
        }
    }

    // ── build_classification_window ───────────────────────────────────────────
    const WIN_BUDGET: usize = 1000;

    #[test]
    fn window_none_when_no_user_messages() {
        let body = json!({"messages": [{"role": "system", "content": "sys"}]});
        assert!(
            build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET).is_none()
        );
    }

    #[test]
    fn window_none_when_all_turns_trivial() {
        // Pure chit-chat prunes to nothing → caller routes baseline Fast.
        let body = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "thanks"},
            {"role": "user", "content": "ok"},
        ]});
        assert!(
            build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET).is_none()
        );
    }

    #[test]
    fn window_skips_assistant_and_trivial_turns_keeps_substantive() {
        // A terse follow-up inherits the substantive context behind it.
        let body = json!({"messages": [
            {"role": "user", "content": "Prove that sqrt 2 is irrational."},
            {"role": "assistant", "content": "A very long proof the window must ignore..."},
            {"role": "user", "content": "ok, continue"},
        ]});
        let window = build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET)
            .expect("substantive turn present");
        assert!(window.contains("sqrt 2 is irrational"));
        assert!(
            !window.contains("long proof"),
            "assistant text must be excluded"
        );
        assert!(
            !window.contains("ok, continue"),
            "trivial turn must be pruned"
        );
    }

    #[test]
    fn window_orders_chronologically_current_turn_last() {
        let body = json!({"messages": [
            {"role": "user", "content": "first substantive question about topology"},
            {"role": "user", "content": "second substantive question about homology"},
        ]});
        let window =
            build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET).unwrap();
        let first = window.find("topology").unwrap();
        let second = window.find("homology").unwrap();
        assert!(
            first < second,
            "older context should precede the current turn"
        );
    }

    #[test]
    fn window_respects_char_budget() {
        let long_a = "a".repeat(80);
        let long_b = "b".repeat(80);
        let body = json!({"messages": [
            {"role": "user", "content": long_a},
            {"role": "user", "content": long_b},
        ]});
        // Budget (90) fits the most recent turn (80) fully and only a sliver of
        // the older one; total collected content stays within budget.
        let window = build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, 90).unwrap();
        assert!(
            window.contains(long_b.as_str()),
            "most recent turn kept in full"
        );
        let a_count = window.chars().filter(|&c| c == 'a').count();
        assert!(
            a_count > 0 && a_count < 80,
            "older turn should be truncated to the remaining budget, got {a_count}"
        );
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
    fn user_text_presence_detected() {
        // Non-empty user text → true.
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(has_nonempty_user_text(&body));
        // Empty string content → false.
        let body = json!({"messages": [{"role": "user", "content": ""}]});
        assert!(!has_nonempty_user_text(&body));
        // Attachment-only multi-part content (no text parts) → false.
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}]});
        assert!(!has_nonempty_user_text(&body));
        // No user messages at all → false.
        let body = json!({"messages": [{"role": "system", "content": "sys"}]});
        assert!(!has_nonempty_user_text(&body));
        // An earlier user turn with text counts even if the last is empty.
        let body = json!({"messages": [
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": ""},
        ]});
        assert!(has_nonempty_user_text(&body));
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
        let simple = clf.classify("hi", "hi", false).unwrap();
        let complex_text = "Derive and rigorously prove the asymptotic time complexity of \
             red-black tree rebalancing across a sequence of insertions and deletions, \
             with a formal amortized analysis.";
        let complex = clf.classify(complex_text, complex_text, false).unwrap();
        assert!(
            simple.complexity <= complex.complexity,
            "trivial prompt ({:?}) should route no higher than a complex one ({:?})",
            simple.complexity,
            complex.complexity
        );
    }
}
