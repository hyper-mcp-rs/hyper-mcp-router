//! Classifier engines — **one file per model**.
//!
//! Exactly one engine is active per process, selected by
//! `--classifier-model` (which overrides the `[classifier] model` config
//! setting). Everything model-specific lives inside the engine's own file:
//! how it is invoked, how many concurrent "sessions" it supports and how they
//! are sized, and how large its context window is. The rest of the router
//! only sees the [`ClassifierEngine`] trait.
//!
//! ## Adding a new engine
//!
//! 1. Add `src/engines/<model>.rs` implementing [`ClassifierEngine`]
//!    (construct it from [`crate::config::ClassifierConfig`] +
//!    [`EngineOverrides`], owning any model-specific sizing/warnings).
//! 2. Add a variant to [`ClassifierModel`] in `crate::classifier` (its
//!    kebab-case name becomes both the CLI value and the config value).
//! 3. Add the `match` arm in [`build`] below.
//!
//! Nothing else in the router changes.

pub mod zero_shot;

use std::sync::Arc;

use crate::classifier::{ClassifierEngine, ClassifierModel};
use crate::config::ClassifierConfig;

/// Operator-provided sizing overrides, already merged by precedence
/// (CLI flag over config setting) before reaching an engine. `None` means
/// "auto": each engine decides its own defaults — for the zero-shot engine
/// that is the CPU/memory plan; a remote engine would pick a concurrency
/// default instead. Meaning is engine-specific by design.
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineOverrides {
    /// Concurrent inference "sessions" (local ORT sessions, or in-flight
    /// requests for a remote engine).
    pub inference_pool_size: Option<usize>,
    /// ORT intra-op threads per session; meaningless for remote engines.
    pub intra_op_threads: Option<usize>,
}

/// Construct the selected engine. This is the **only** place that maps a
/// [`ClassifierModel`] to a concrete engine type.
pub fn build(
    model: ClassifierModel,
    cfg: &ClassifierConfig,
    overrides: &EngineOverrides,
) -> anyhow::Result<Arc<dyn ClassifierEngine>> {
    match model {
        ClassifierModel::ZeroShot => {
            Ok(Arc::new(zero_shot::ZeroShot::from_config(cfg, overrides)?))
        }
    }
}
