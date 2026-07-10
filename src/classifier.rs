//! Prompt classifier: zero-shot NLI session, hypothesis scoring, and the pure
//! metadata helpers used for modality-aware routing.
//!
//! The model is `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, a
//! **binary** NLI model (`id2label = { 0: "entailment", 1: "not_entailment" }`,
//! `type_vocab_size = 0` — no `token_type_ids`). These facts are load-bearing
//! for the inference code below.

use std::sync::{LazyLock, Mutex};

use ort::session::{builder::GraphOptimizationLevel, Session};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::{MODEL_BYTES, TOKENIZER_BYTES};

/// Prompt-length guard, in **characters** (never bytes — see [`truncate_prompt`]).
const PROMPT_CHAR_LIMIT: usize = 400;

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
        }
    }

    /// All modalities, in a stable order for iteration/logging.
    const ALL: [Modality; 6] = [
        Modality::Text,
        Modality::ImageInput,
        Modality::AudioInput,
        Modality::FileInput,
        Modality::AudioOutput,
        Modality::ImageOutput,
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
/// `run` takes `&mut self`, so a shared `&Session` cannot run inference; the
/// session is held behind a `Mutex` for interior mutability. Because each
/// request now takes exactly **one** batched pass, the lock is acquired once
/// per request rather than once per hypothesis. Request forwarding and SSE
/// streaming are outside this lock and remain fully concurrent.
pub struct Classifier {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    hypotheses: Vec<(HypothesisKind, String)>,
    /// Absolute P(entailment) floor for the image-generation axis.
    image_gen_threshold: f32,
}

impl Classifier {
    /// Load the embedded model and tokenizer and build the hypothesis list.
    ///
    /// Intra-op threading is left at the ORT default so the single batched pass
    /// can use the available cores; there is no longer a many-concurrent-passes
    /// design that would oversubscribe the thread pool.
    pub fn new(image_gen_threshold: f32) -> anyhow::Result<Self> {
        // `ort::Error` is not `Send + Sync` for every generic parameter, so it
        // cannot flow through `?` into `anyhow::Error`; map each to a string.
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort optimization level: {e}"))?
            .commit_from_memory(MODEL_BYTES)
            .map_err(|e| anyhow::anyhow!("ort commit_from_memory: {e}"))?;

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
            session: Mutex::new(session),
            tokenizer,
            hypotheses,
            image_gen_threshold,
        })
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
    /// `i`. `Session::run` requires `&mut self`, so the pass is taken under the
    /// session `Mutex` — once per request.
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

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("classifier session mutex poisoned"))?;
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
/// (input) and the `modalities` request field (output). `ImageOutput` is **not**
/// decided here — it has no protocol field and is inferred by the classifier.
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
        let clf = Classifier::new(DEFAULT_IMAGE_GEN_THRESHOLD).unwrap();
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
