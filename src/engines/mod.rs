//! Classifier engines — **one file per model**.
//!
//! Exactly one engine is active per process, selected by the
//! `[classifier] model` config setting (config-only: each model brings its
//! own configuration, so the config file is the single source of truth).
//! Everything model-specific lives inside the engine's own file: how it is
//! invoked, how many concurrent "sessions" it supports and how they are
//! sized, and how large its context window is. Model-specific *settings*
//! live in a per-engine config table (e.g. `[classifier.zero-shot]`). The
//! rest of the router only sees the [`ClassifierEngine`] trait.
//!
//! ## Adding a new engine
//!
//! 1. Add `src/engines/<model>.rs` implementing [`ClassifierEngine`]
//!    (construct it from [`crate::config::ClassifierConfig`], owning any
//!    model-specific sizing/warnings).
//! 2. If it needs settings, add a `[classifier.<model>]` table struct in
//!    `crate::config` (see `ZeroShotConfig`).
//! 3. Add a variant to [`ClassifierModel`] in `crate::classifier` (its
//!    kebab-case name is the config value).
//! 4. Add the `match` arm in [`build`] below.
//!
//! Nothing else in the router changes.

pub mod zero_shot;

use std::sync::Arc;

use crate::classifier::{ClassifierEngine, ClassifierModel};
use crate::config::ClassifierConfig;

/// Construct the engine selected by `cfg.model`. This is the **only** place
/// that maps a [`ClassifierModel`] to a concrete engine type.
pub fn build(cfg: &ClassifierConfig) -> anyhow::Result<Arc<dyn ClassifierEngine>> {
    match cfg.model {
        ClassifierModel::ZeroShot => Ok(Arc::new(zero_shot::ZeroShot::from_config(cfg)?)),
    }
}
