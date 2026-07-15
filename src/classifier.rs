//! The classification domain: the complexity axis, the classification result,
//! the [`ClassifierEngine`] trait every engine implements, the
//! [`ClassifierModel`] selector, and the [`EngineRoster`] capacity ladder.
//!
//! Concrete engines live in `crate::engines` — **one file per model** — and
//! are the only code that knows how a given model is invoked, sized, or
//! windowed. The rest of the router (the proxy in particular) programs
//! against this trait alone.

use std::sync::Arc;

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

/// A classification model the router can run. The `[classifier] model`
/// config setting names one — or a **list** of several, forming a capacity
/// ladder (see [`EngineRoster`]) — config-only, no CLI override: each model
/// brings its own configuration (typically a whole `[classifier.<model>]`
/// table), so different models mean different config files. The kebab-case
/// variant name is the config value. Each variant maps to one file in
/// `crate::engines` (see `engines::build`).
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
    /// classification on **Vertex AI** (this model is not published on the
    /// Gemini Developer API). Requires a GCP `project` and OAuth
    /// `access_token`; prompt text is sent to the Vertex AI API.
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

    /// Whether classification runs entirely in-process. `false` means prompt
    /// text (the classification window and current turn) is sent to a remote
    /// provider's API. Surfaced in the startup capacity-ladder log so a mixed
    /// local/remote roster is a visible, deliberate choice — with such a
    /// roster, *prompt length* decides whether text leaves the process.
    fn is_local(&self) -> bool;

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

// ───────────────────────────────────────────────────────────────────────────
// The engine roster (capacity ladder)
// ───────────────────────────────────────────────────────────────────────────

/// The configured classifier engines as a **capacity ladder**: sorted
/// ascending by [`ClassifierEngine::context_char_budget`], which must be
/// unique per engine (equal budgets are a startup error — the ladder needs a
/// total order). Per request the proxy builds the classification window at
/// the *top* engine's budget and hands it to the **smallest engine whose
/// budget covers it** ([`select`](Self::select)); only a window that exceeds
/// even the top budget is truncated. A single-engine roster degenerates to
/// exactly the previous one-engine behaviour.
///
/// Not called "tiers": [`ModelTier`] already names the fast/balanced/frontier
/// complexity axis of the *routed backends*, which is unrelated.
#[derive(Clone)]
pub struct EngineRoster {
    /// Ascending by `context_char_budget`; construction-validated, never
    /// empty. `Arc<[..]>` so the per-request `AppState` clone stays cheap.
    engines: Arc<[Arc<dyn ClassifierEngine>]>,
}

impl std::fmt::Debug for EngineRoster {
    /// Trait objects aren't `Debug`; show the ladder as `name(budget)` rungs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rungs: Vec<String> = self
            .engines
            .iter()
            .map(|e| format!("{}({})", e.name(), e.context_char_budget()))
            .collect();
        f.debug_tuple("EngineRoster").field(&rungs).finish()
    }
}

impl EngineRoster {
    /// Sort and validate the roster. Fails when:
    /// - it is empty;
    /// - two engines share a `context_char_budget` (the ladder must be a
    ///   total order — this also catches the same engine configured twice);
    /// - `current_turn_char_budget` is not monotone in ladder order (a
    ///   higher-capacity engine seeing *less* of the current turn would make
    ///   escalation lose image-intent signal — for today's engines this holds
    ///   naturally, but only by convention, so it is enforced here).
    pub fn new(mut engines: Vec<Arc<dyn ClassifierEngine>>) -> anyhow::Result<Self> {
        if engines.is_empty() {
            anyhow::bail!("no classifier engines configured");
        }
        engines.sort_by_key(|e| e.context_char_budget());
        for pair in engines.windows(2) {
            let (lower, upper) = (&pair[0], &pair[1]);
            if lower.context_char_budget() == upper.context_char_budget() {
                anyhow::bail!(
                    "classifier engines `{}` and `{}` share the same context_char_budget ({}); \
                     the capacity ladder requires distinct budgets",
                    lower.name(),
                    upper.name(),
                    lower.context_char_budget(),
                );
            }
            if lower.current_turn_char_budget() > upper.current_turn_char_budget() {
                anyhow::bail!(
                    "classifier engine `{}` (current_turn_char_budget {}) sees more of the \
                     current turn than the higher-capacity `{}` ({}); current-turn budgets \
                     must be monotone in capacity order",
                    lower.name(),
                    lower.current_turn_char_budget(),
                    upper.name(),
                    upper.current_turn_char_budget(),
                );
            }
        }
        Ok(EngineRoster {
            engines: engines.into(),
        })
    }

    /// The smallest engine whose window budget covers `window_chars`. A
    /// window can never exceed the top budget (the proxy builds it at exactly
    /// that budget), so the top engine covers everything by construction.
    pub fn select(&self, window_chars: usize) -> &Arc<dyn ClassifierEngine> {
        self.engines
            .iter()
            .find(|e| e.context_char_budget() >= window_chars)
            .unwrap_or_else(|| self.top())
    }

    /// The highest-capacity engine (the top of the ladder).
    pub fn top(&self) -> &Arc<dyn ClassifierEngine> {
        self.engines.last().expect("roster is never empty")
    }

    /// The window budget the proxy builds classification windows at: the top
    /// engine's. Anything longer is truncated — the "cut off only at the
    /// highest rung" rule.
    pub fn max_context_char_budget(&self) -> usize {
        self.top().context_char_budget()
    }

    /// Engines in ladder order (ascending capacity).
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn ClassifierEngine>> {
        self.engines.iter()
    }

    pub fn len(&self) -> usize {
        self.engines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trait-level fake: budgets only, never classified in these tests.
    struct FakeEngine {
        name: &'static str,
        window: usize,
        turn: usize,
    }

    #[async_trait]
    impl ClassifierEngine for FakeEngine {
        fn name(&self) -> &'static str {
            self.name
        }
        fn is_local(&self) -> bool {
            true
        }
        fn context_char_budget(&self) -> usize {
            self.window
        }
        fn current_turn_char_budget(&self) -> usize {
            self.turn
        }
        async fn classify(&self, _: &str, _: &str, _: bool) -> anyhow::Result<Classification> {
            unreachable!("roster tests never classify")
        }
    }

    fn fake(name: &'static str, window: usize, turn: usize) -> Arc<dyn ClassifierEngine> {
        Arc::new(FakeEngine { name, window, turn })
    }

    // ── EngineRoster ─────────────────────────────────────────────────────

    #[test]
    fn roster_sorts_by_window_budget_and_selects_smallest_adequate() {
        // Deliberately unsorted input: order must be derived, not declared.
        let roster = EngineRoster::new(vec![
            fake("big", 6000, 2000),
            fake("small", 1000, 400),
            fake("mid", 3000, 1000),
        ])
        .unwrap();

        assert_eq!(
            roster.iter().map(|e| e.name()).collect::<Vec<_>>(),
            ["small", "mid", "big"]
        );
        assert_eq!(roster.max_context_char_budget(), 6000);

        // Boundary semantics: a budget covers a window of exactly its size.
        assert_eq!(roster.select(0).name(), "small");
        assert_eq!(roster.select(1000).name(), "small");
        assert_eq!(roster.select(1001).name(), "mid");
        assert_eq!(roster.select(3000).name(), "mid");
        assert_eq!(roster.select(3001).name(), "big");
        assert_eq!(roster.select(6000).name(), "big");
        // The proxy never produces a longer window (it builds at the top
        // budget), but selection must still be total.
        assert_eq!(roster.select(usize::MAX).name(), "big");
    }

    #[test]
    fn roster_single_engine_always_selects_it() {
        let roster = EngineRoster::new(vec![fake("only", 1000, 400)]).unwrap();
        assert_eq!(roster.len(), 1);
        assert_eq!(roster.select(0).name(), "only");
        assert_eq!(roster.select(999_999).name(), "only");
        assert_eq!(roster.max_context_char_budget(), 1000);
    }

    #[test]
    fn roster_rejects_empty() {
        let err = EngineRoster::new(vec![]).unwrap_err();
        assert!(err.to_string().contains("no classifier engines"));
    }

    #[test]
    fn roster_rejects_duplicate_window_budgets_naming_both_engines() {
        let err = EngineRoster::new(vec![fake("a", 1000, 400), fake("b", 1000, 500)]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("`a`") && msg.contains("`b`"), "got: {msg}");
        assert!(msg.contains("1000"), "got: {msg}");
    }

    #[test]
    fn roster_rejects_non_monotone_current_turn_budgets() {
        // The bigger-window engine sees LESS of the current turn: escalation
        // would lose image-intent signal, so this must fail at startup.
        let err =
            EngineRoster::new(vec![fake("small", 1000, 800), fake("big", 6000, 400)]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("monotone"), "got: {msg}");
        assert!(
            msg.contains("`small`") && msg.contains("`big`"),
            "got: {msg}"
        );
    }

    #[test]
    fn roster_allows_equal_current_turn_budgets() {
        // Only the window budget needs to be unique; equal turn budgets are
        // fine (monotone non-decreasing).
        assert!(EngineRoster::new(vec![fake("a", 1000, 400), fake("b", 6000, 400)]).is_ok());
    }

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
