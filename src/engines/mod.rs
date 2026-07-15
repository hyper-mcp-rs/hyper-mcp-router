//! Classifier engines — **one file per model**, grouped by provider family.
//!
//! One or more engines are active per process, selected by the
//! `[classifier] model` config setting (a single id or a list; config-only:
//! each model brings its own configuration, so the config file is the single
//! source of truth). Several engines form a **capacity ladder**
//! ([`crate::classifier::EngineRoster`]): sorted by context budget, each
//! request is classified by the smallest engine whose budget covers it.
//! Everything model-specific lives inside the engine's own file: how it is
//! invoked, how many concurrent "sessions" it supports and how they are
//! sized, and how large its context window is. Families with several models
//! (e.g. `gemini/`) keep their shared plumbing in the family's `mod.rs` and
//! one file per model beside it. Model-specific *settings* live in a
//! per-engine config table (e.g. `[classifier.deberta-v3-xsmall-zeroshot]`).
//! The rest of the router only sees the [`ClassifierEngine`] trait.
//!
//! Engine ids name the **model**, never the technique — "zero-shot" would be
//! ambiguous the moment a second engine can also classify zero-shot.
//!
//! ## Adding a new engine
//!
//! 1. Add the model file implementing [`ClassifierEngine`] — inside an
//!    existing family directory (e.g. `gemini/<model>.rs`, delegating to the
//!    family's shared engine), or as a new top-level file / family directory
//!    (construct it from [`crate::config::ClassifierConfig`], owning any
//!    model-specific sizing/warnings).
//! 2. If it needs settings, add a `[classifier.<model>]` table struct in
//!    `crate::config` (see `DebertaV3XsmallZeroshotConfig`).
//! 3. Add a variant to [`ClassifierModel`] in `crate::classifier` (its
//!    kebab-case name is the config value).
//! 4. Add the `match` arm in [`build`] below.
//!
//! Nothing else in the router changes. (Roster note: an engine's
//! `context_char_budget` must differ from every other engine's, and its
//! `current_turn_char_budget` must not be smaller than that of any
//! lower-capacity engine — `EngineRoster::new` enforces both at startup.)

pub mod deberta_v3_xsmall_zeroshot;
pub mod embedding;
pub mod gemini;
pub mod vertex;

use std::sync::Arc;

use crate::classifier::{ClassifierEngine, ClassifierModel, EngineRoster};
use crate::config::{ClassifierConfig, GoogleApi, GoogleEmbeddingConfig};

/// Construct every engine named by `cfg.models` and assemble the capacity
/// ladder. Engine construction failures and roster-shape violations
/// (duplicate context budgets, non-monotone current-turn budgets — see
/// [`EngineRoster::new`]) all fail here, at boot.
///
/// The pure, offline checks ([`validate_config`]) run **first**, so a config
/// mistake in any rung fails before *any* engine does expensive startup work
/// (inference-session allocation, credential discovery, remote anchor
/// embedding). This also makes the `validate` subcommand's guarantee
/// structural: a config it rejects is rejected here identically.
pub async fn build_roster(cfg: &ClassifierConfig) -> anyhow::Result<EngineRoster> {
    validate_config(cfg)?;

    let mut engines = Vec::with_capacity(cfg.models.len());
    for model in &cfg.models {
        let engine = build(*model, cfg).await.map_err(|e| {
            e.context(format!(
                "initialising classifier engine `{}`",
                model.as_str()
            ))
        })?;
        engines.push(engine);
    }
    EngineRoster::new(engines)
}

/// Construct one engine. This is the **only** place that maps a
/// [`ClassifierModel`] to a concrete engine type. Async because remote
/// engines do startup work over the network (anchor embedding);
/// misconfiguration — including missing or ambiguous credentials — fails
/// here, at boot.
///
/// The gemini-embedding models are published on **two** Google API surfaces;
/// the auth fields of the engine's config table pick the concrete engine
/// (`api_key` ⇒ `gemini/`, `project` ⇒ `vertex/` — see
/// [`crate::config::GoogleEmbeddingConfig::surface`]).
pub async fn build(
    model: ClassifierModel,
    cfg: &ClassifierConfig,
) -> anyhow::Result<Arc<dyn ClassifierEngine>> {
    match model {
        ClassifierModel::DebertaV3XsmallZeroshot => Ok(Arc::new(
            deberta_v3_xsmall_zeroshot::DebertaV3XsmallZeroshot::from_config(cfg)?,
        )),
        ClassifierModel::GeminiEmbedding001 => {
            match cfg.gemini_embedding_001.surface("gemini-embedding-001")? {
                GoogleApi::GenerativeLanguage => {
                    Ok(Arc::new(gemini::embedding_001::build(cfg).await?))
                }
                GoogleApi::Vertex => Ok(Arc::new(vertex::gemini_embedding_001::build(cfg).await?)),
            }
        }
        ClassifierModel::GeminiEmbedding2 => {
            match cfg.gemini_embedding_2.surface("gemini-embedding-2")? {
                GoogleApi::GenerativeLanguage => {
                    Ok(Arc::new(gemini::embedding_2::build(cfg).await?))
                }
                GoogleApi::Vertex => Ok(Arc::new(vertex::gemini_embedding_2::build(cfg).await?)),
            }
        }
        ClassifierModel::TextEmbedding005 => {
            Ok(Arc::new(vertex::text_embedding_005::build(cfg).await?))
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// Offline validation (the `validate` subcommand)
// ───────────────────────────────────────────────────────────────────────

/// The context-window character budget an engine would report via
/// [`ClassifierEngine::context_char_budget`], available **without**
/// constructing it (pure; no I/O). For the dual-surface gemini models the
/// budget is identical on both surfaces (same model, same input limit).
pub fn context_char_budget(model: ClassifierModel) -> usize {
    match model {
        ClassifierModel::DebertaV3XsmallZeroshot => {
            deberta_v3_xsmall_zeroshot::CLASSIFICATION_CHAR_BUDGET
        }
        ClassifierModel::GeminiEmbedding001 => gemini::embedding_001::SPEC.context_char_budget,
        ClassifierModel::GeminiEmbedding2 => gemini::embedding_2::SPEC.context_char_budget,
        ClassifierModel::TextEmbedding005 => vertex::text_embedding_005::SPEC.context_char_budget,
    }
}

/// Whether an engine runs fully locally (no prompt text leaves the process),
/// without constructing it. Mirrors [`ClassifierEngine::is_local`].
pub fn is_local(model: ClassifierModel) -> bool {
    matches!(model, ClassifierModel::DebertaV3XsmallZeroshot)
}

/// Offline validation of the classifier engine configuration: everything
/// [`build_roster`] would reject that can be checked **without** constructing
/// engines — no network, no credential (ADC) resolution, no inference-session
/// allocation. Checked here:
///
/// - each dual-surface gemini engine names exactly one API surface
///   ([`GoogleEmbeddingConfig::surface`]);
/// - every Vertex-surface engine has the required `project` and `location`;
/// - the capacity ladder's context budgets are pairwise distinct (the shape
///   rule of [`EngineRoster::new`] that real configs can actually violate —
///   e.g. `gemini-embedding-001` and `text-embedding-005` share a budget).
///
/// Used by the `validate` CLI subcommand, and run first by [`build_roster`]
/// at boot — so the two can never drift apart on what they accept.
/// [`build_roster`] remains the deeper check (it additionally exercises
/// credentials and remote startup work).
pub fn validate_config(cfg: &ClassifierConfig) -> anyhow::Result<()> {
    for model in &cfg.models {
        match model {
            // Embedded local engine: nothing remote to misconfigure.
            ClassifierModel::DebertaV3XsmallZeroshot => {}
            ClassifierModel::GeminiEmbedding001 => {
                validate_google_surface(&cfg.gemini_embedding_001, "gemini-embedding-001")?
            }
            ClassifierModel::GeminiEmbedding2 => {
                validate_google_surface(&cfg.gemini_embedding_2, "gemini-embedding-2")?
            }
            ClassifierModel::TextEmbedding005 => cfg
                .text_embedding_005
                .project_and_location("text-embedding-005")
                .map(|_| ())?,
        }
    }

    for (i, a) in cfg.models.iter().enumerate() {
        for b in &cfg.models[i + 1..] {
            if context_char_budget(*a) == context_char_budget(*b) {
                anyhow::bail!(
                    "classifier engines `{}` and `{}` share the same context_char_budget ({}); \
                     the capacity ladder requires distinct budgets",
                    a.as_str(),
                    b.as_str(),
                    context_char_budget(*a),
                );
            }
        }
    }
    Ok(())
}

/// The dual-surface slice of [`validate_config`]: the auth fields must name
/// exactly one surface, and the Vertex surface additionally needs `project`
/// and `location`.
fn validate_google_surface(cfg: &GoogleEmbeddingConfig, engine: &str) -> anyhow::Result<()> {
    match cfg.surface(engine)? {
        GoogleApi::GenerativeLanguage => Ok(()),
        GoogleApi::Vertex => cfg.to_vertex().project_and_location(engine).map(|_| ()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(models: Vec<ClassifierModel>) -> ClassifierConfig {
        ClassifierConfig {
            models,
            ..ClassifierConfig::default()
        }
    }

    #[test]
    fn static_budgets_match_the_engine_specs() {
        // Drift guard: the pure mapping must report exactly what a built
        // engine would. The deberta value is the shared crate const; the
        // dual-surface gemini models must agree across both surfaces.
        assert_eq!(
            context_char_budget(ClassifierModel::DebertaV3XsmallZeroshot),
            deberta_v3_xsmall_zeroshot::CLASSIFICATION_CHAR_BUDGET
        );
        assert_eq!(
            context_char_budget(ClassifierModel::GeminiEmbedding001),
            vertex::gemini_embedding_001::SPEC.context_char_budget
        );
        assert_eq!(
            context_char_budget(ClassifierModel::GeminiEmbedding2),
            vertex::gemini_embedding_2::SPEC.context_char_budget
        );
        assert_eq!(
            context_char_budget(ClassifierModel::TextEmbedding005),
            vertex::text_embedding_005::SPEC.context_char_budget
        );
    }

    #[test]
    fn only_the_embedded_engine_is_local() {
        assert!(is_local(ClassifierModel::DebertaV3XsmallZeroshot));
        assert!(!is_local(ClassifierModel::GeminiEmbedding001));
        assert!(!is_local(ClassifierModel::GeminiEmbedding2));
        assert!(!is_local(ClassifierModel::TextEmbedding005));
    }

    #[test]
    fn default_config_validates() {
        validate_config(&ClassifierConfig::default()).unwrap();
    }

    #[test]
    fn full_ladder_with_vertex_auth_validates() {
        let mut cfg = cfg_with(vec![
            ClassifierModel::DebertaV3XsmallZeroshot,
            ClassifierModel::TextEmbedding005,
            ClassifierModel::GeminiEmbedding2,
        ]);
        cfg.text_embedding_005.project = Some("proj".into());
        cfg.text_embedding_005.location = Some("us-central1".into());
        cfg.gemini_embedding_2.project = Some("proj".into());
        cfg.gemini_embedding_2.location = Some("global".into());
        validate_config(&cfg).unwrap();
    }

    #[test]
    fn text_embedding_005_requires_project_and_location() {
        let mut cfg = cfg_with(vec![ClassifierModel::TextEmbedding005]);
        let err = validate_config(&cfg).unwrap_err().to_string();
        assert!(err.contains("requires a GCP project"), "got: {err}");

        cfg.text_embedding_005.project = Some("proj".into());
        let err = validate_config(&cfg).unwrap_err().to_string();
        assert!(err.contains("requires a `location`"), "got: {err}");
    }

    #[test]
    fn gemini_engine_rejects_ambiguous_or_missing_surface() {
        // Neither auth field: the surface is undecidable.
        let cfg = cfg_with(vec![ClassifierModel::GeminiEmbedding2]);
        let err = validate_config(&cfg).unwrap_err().to_string();
        assert!(err.contains("requires either `api_key`"), "got: {err}");

        // Both auth fields: ambiguous.
        let mut cfg = cfg_with(vec![ClassifierModel::GeminiEmbedding2]);
        cfg.gemini_embedding_2.api_key = Some("k".into());
        cfg.gemini_embedding_2.project = Some("proj".into());
        let err = validate_config(&cfg).unwrap_err().to_string();
        assert!(err.contains("sets both `api_key"), "got: {err}");
    }

    #[test]
    fn gemini_vertex_surface_requires_location() {
        let mut cfg = cfg_with(vec![ClassifierModel::GeminiEmbedding2]);
        cfg.gemini_embedding_2.project = Some("proj".into());
        let err = validate_config(&cfg).unwrap_err().to_string();
        assert!(err.contains("requires a `location`"), "got: {err}");
    }

    #[test]
    fn gemini_generative_language_surface_needs_no_location() {
        let mut cfg = cfg_with(vec![ClassifierModel::GeminiEmbedding2]);
        cfg.gemini_embedding_2.api_key = Some("k".into());
        validate_config(&cfg).unwrap();
    }

    #[tokio::test]
    async fn build_roster_runs_the_offline_checks_before_building_any_engine() {
        // A ladder whose only problem is the duplicate budget between
        // gemini-embedding-001 and text-embedding-005. Without the up-front
        // validate_config call, build_roster would first BUILD the gemini
        // engine — a live network call (anchor embedding) with this fake key
        // — before EngineRoster::new could object. The pre-check makes this
        // fail instantly and offline; this test hanging or erroring on I/O
        // means the fail-fast ordering regressed.
        let mut cfg = cfg_with(vec![
            ClassifierModel::GeminiEmbedding001,
            ClassifierModel::TextEmbedding005,
        ]);
        cfg.gemini_embedding_001.api_key = Some("fake-key-never-sent".into());
        cfg.text_embedding_005.project = Some("proj".into());
        cfg.text_embedding_005.location = Some("us-central1".into());
        let err = build_roster(&cfg).await.unwrap_err().to_string();
        assert!(
            err.contains("share the same context_char_budget"),
            "got: {err}"
        );
    }

    #[test]
    fn duplicate_context_budgets_are_rejected() {
        // gemini-embedding-001 and text-embedding-005 genuinely share a
        // budget — the one ladder-shape violation a real config can hit.
        let mut cfg = cfg_with(vec![
            ClassifierModel::GeminiEmbedding001,
            ClassifierModel::TextEmbedding005,
        ]);
        cfg.gemini_embedding_001.api_key = Some("k".into());
        cfg.text_embedding_005.project = Some("proj".into());
        cfg.text_embedding_005.location = Some("us-central1".into());
        let err = validate_config(&cfg).unwrap_err().to_string();
        assert!(
            err.contains("share the same context_char_budget"),
            "got: {err}"
        );
    }
}
