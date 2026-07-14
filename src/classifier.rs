//! The classification domain: the complexity axis, the classification result,
//! the [`ClassifierEngine`] trait every engine implements, and the
//! [`ClassifierModel`] selector.
//!
//! Concrete engines live in `crate::engines` — **one file per model** — and
//! are the only code that knows how a given model is invoked, sized, or
//! windowed. The rest of the router (the proxy in particular) programs
//! against this trait alone.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Default score floor above which an engine's image-generation signal alone
/// is enough to route to the image modality. The score's *scale* is
/// engine-specific (P(entailment) for the zero-shot NLI engine); each engine
/// documents how it interprets the threshold.
pub const DEFAULT_IMAGE_GEN_THRESHOLD: f32 = 0.5;

// ───────────────────────────────────────────────────────────────────────────
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

/// What a classifier engine resolves to. `complexity` (the tier axis) is
/// always produced; `image_generation` (the modality axis) is an orthogonal
/// intent flag.
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

// ───────────────────────────────────────────────────────────────────────────
// Engine selection
// ───────────────────────────────────────────────────────────────────────────

/// Which classification model the router runs. **Exactly one is active per
/// process**, selected by the `[classifier] model` config setting —
/// config-only, no CLI override: each model brings its own configuration
/// (typically a whole `[classifier.<model>]` table), so different models mean
/// different config files. The kebab-case variant name is the config value.
/// Each variant maps to one file in `crate::engines` (see `engines::build`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClassifierModel {
    /// The embedded `deberta-v3-xsmall-zeroshot` NLI model — fully local, no
    /// data leaves the process. The default. (The id names the *model*, not
    /// the technique: other engines can also classify zero-shot.)
    #[default]
    #[serde(rename = "deberta-v3-xsmall-zeroshot")]
    DebertaV3XsmallZeroshot,
    /// Google `gemini-embedding-001` — remote anchor-prototype embedding
    /// classification. Requires an API key; prompt text is sent to the
    /// Gemini API for classification.
    #[serde(rename = "gemini-embedding-001")]
    GeminiEmbedding001,
    /// Google `gemini-embedding-2` — as above, with a larger context window.
    #[serde(rename = "gemini-embedding-2")]
    GeminiEmbedding2,
    /// Google `text-embedding-005` — remote anchor-prototype embedding
    /// classification on the smaller/cheaper text-embedding model. Requires
    /// an API key; prompt text is sent to the Gemini API.
    #[serde(rename = "text-embedding-005")]
    TextEmbedding005,
}

impl ClassifierModel {
    /// Kebab-case wire name (the `[classifier] model` config value).
    pub fn as_str(self) -> &'static str {
        match self {
            ClassifierModel::DebertaV3XsmallZeroshot => "deberta-v3-xsmall-zeroshot",
            ClassifierModel::GeminiEmbedding001 => "gemini-embedding-001",
            ClassifierModel::GeminiEmbedding2 => "gemini-embedding-2",
            ClassifierModel::TextEmbedding005 => "text-embedding-005",
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// The engine trait
// ───────────────────────────────────────────────────────────────────────────

/// A classification engine: everything model-specific behind one interface.
///
/// Implementations own their method of interaction (local inference, remote
/// API), their concurrency model ("sessions" — ORT sessions, in-flight
/// requests, …) and its sizing, and their context-window budget. What they do
/// **not** own is routing policy: window construction and filler pruning
/// (`crate::prompt`), the lexical image prefilter, the classification-skip
/// optimisation, and the failure fallback to the balanced default all live in
/// the proxy and apply uniformly to every engine.
#[async_trait]
pub trait ClassifierEngine: Send + Sync {
    /// Stable engine id for logs and the startup banner (kebab-case,
    /// matching [`ClassifierModel::as_str`]).
    fn name(&self) -> &'static str;

    /// Character budget for the complexity-classification window, derived
    /// from the model's context window. The proxy passes this to
    /// `prompt::build_classification_window`.
    fn context_char_budget(&self) -> usize;

    /// Character budget for the **current turn** as the classifier sees it.
    /// The truncated turn serves as the image premise passed to
    /// [`classify`](Self::classify) and as the input to the lexical image
    /// prefilter. Model-specific: a small local model keeps this tight (the
    /// zero-shot engine uses 400, sized to its 512-token ceiling), while an
    /// engine backed by a large-context embedding model may choose far more —
    /// image intent expressed deep in a long prompt is only visible within
    /// this budget. Bounds classifier input only; the forwarded request is
    /// never truncated.
    fn current_turn_char_budget(&self) -> usize;

    /// Resolve complexity and image-generation intent.
    ///
    /// `complexity_premise` is the windowed recent user context;
    /// `image_premise` is the *current* turn only (an old "draw a cat" turn
    /// in the window must not trigger image routing now).
    /// `lexical_image_match` is the precomputed lexical prefilter result for
    /// `image_premise` — the proxy already needs it before deciding whether
    /// to classify at all, so it is passed in rather than recomputed.
    ///
    /// CPU-bound engines must move their work onto a blocking thread
    /// internally; this method is called directly on async workers. Errors
    /// are mapped to [`Classification::balanced_default`] by the proxy.
    async fn classify(
        &self,
        complexity_premise: &str,
        image_premise: &str,
        lexical_image_match: bool,
    ) -> anyhow::Result<Classification>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_model_default_is_the_embedded_deberta() {
        assert_eq!(
            ClassifierModel::default(),
            ClassifierModel::DebertaV3XsmallZeroshot
        );
        assert_eq!(
            ClassifierModel::default().as_str(),
            "deberta-v3-xsmall-zeroshot"
        );
    }

    #[test]
    fn classifier_model_deserializes_from_kebab_case() {
        for (wire, expected) in [
            (
                "\"deberta-v3-xsmall-zeroshot\"",
                ClassifierModel::DebertaV3XsmallZeroshot,
            ),
            (
                "\"gemini-embedding-001\"",
                ClassifierModel::GeminiEmbedding001,
            ),
            ("\"gemini-embedding-2\"", ClassifierModel::GeminiEmbedding2),
            ("\"text-embedding-005\"", ClassifierModel::TextEmbedding005),
        ] {
            let m: ClassifierModel = serde_json::from_str(wire).unwrap();
            assert_eq!(m, expected);
            // The wire name and as_str must always agree.
            assert_eq!(format!("\"{}\"", m.as_str()), wire);
        }
        // Unknown model ids must fail loudly, not fall back silently.
        assert!(serde_json::from_str::<ClassifierModel>("\"not-a-model\"").is_err());
    }
}
