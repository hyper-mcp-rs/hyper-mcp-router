//! Classifier engines — **one file per model**, grouped by provider family.
//!
//! Exactly one engine is active per process, selected by the
//! `[classifier] model` config setting (config-only: each model brings its
//! own configuration, so the config file is the single source of truth).
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
//! Nothing else in the router changes.

pub mod deberta_v3_xsmall_zeroshot;
pub mod embedding;
pub mod gemini;
pub mod openai;

use std::sync::Arc;

use crate::classifier::{ClassifierEngine, ClassifierModel};
use crate::config::ClassifierConfig;

/// Construct the engine selected by `cfg.model`. This is the **only** place
/// that maps a [`ClassifierModel`] to a concrete engine type. Async because
/// remote engines do startup work over the network (anchor embedding);
/// misconfiguration — including a missing required API key — fails here,
/// at boot.
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<Arc<dyn ClassifierEngine>> {
    match cfg.model {
        ClassifierModel::DebertaV3XsmallZeroshot => Ok(Arc::new(
            deberta_v3_xsmall_zeroshot::DebertaV3XsmallZeroshot::from_config(cfg)?,
        )),
        ClassifierModel::GeminiEmbedding001 => {
            Ok(Arc::new(gemini::embedding_001::build(cfg).await?))
        }
        ClassifierModel::GeminiEmbedding2 => Ok(Arc::new(gemini::embedding_2::build(cfg).await?)),
        ClassifierModel::TextEmbedding005 => {
            Ok(Arc::new(gemini::text_embedding_005::build(cfg).await?))
        }
        ClassifierModel::TextEmbedding3Small => {
            Ok(Arc::new(openai::text_embedding_3_small::build(cfg).await?))
        }
        ClassifierModel::TextEmbedding3Large => {
            Ok(Arc::new(openai::text_embedding_3_large::build(cfg).await?))
        }
    }
}
