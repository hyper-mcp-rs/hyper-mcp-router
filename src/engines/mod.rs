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
use crate::config::{ClassifierConfig, GoogleApi};

/// Construct every engine named by `cfg.models` and assemble the capacity
/// ladder. Engine construction failures and roster-shape violations
/// (duplicate context budgets, non-monotone current-turn budgets — see
/// [`EngineRoster::new`]) all fail here, at boot.
pub async fn build_roster(cfg: &ClassifierConfig) -> anyhow::Result<EngineRoster> {
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
