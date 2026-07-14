//! The `gemini-embedding-2` engine **on Vertex AI** — a completely separate
//! engine from `engines/gemini/embedding_2.rs`, which talks to the same model
//! on the Generative Language API (see `gemini_embedding_001.rs` in this
//! directory for the surface-split rationale). `project` in
//! `[classifier.gemini-embedding-2]` selects this engine.
//!
//! NOTE: unlike its `-001` sibling, this engine could not be live-verified at
//! the time of writing — the model returned NOT_FOUND in the test project
//! (`us-central1` and `global`), likely a rollout/allowlist gap. The wire
//! format is the shared, live-verified Vertex `:predict` contract; treat the
//! first production selection of this engine as the availability check (it
//! fails fast at startup if the model is absent).
//!
//! Privacy note: selecting this engine sends prompt text (the classification
//! window and current turn) to the Vertex AI API.

use crate::config::ClassifierConfig;

use super::{VertexEmbedding, VertexSpec};

/// Model-specific parameters for `gemini-embedding-2` on Vertex AI.
pub const SPEC: VertexSpec = VertexSpec {
    name: "gemini-embedding-2",
    api_model: "gemini-embedding-2",
    // 8192-token input limit; ~4 chars/token with headroom (same model as the
    // Generative-Language twin, so the budgets match).
    context_char_budget: 24000,
    current_turn_char_budget: 8000,
    default_max_concurrency: 32,
    default_request_timeout_secs: 10,
};

// Compile-time spec coherence: current-turn budget within the window budget;
// window under the model's 8192-token input limit (~4 chars/token);
// concurrency at least 1. (The `gemini/` twin declares the same budgets —
// same underlying model — but deliberately without a code-level tie: the two
// engines are fully independent.)
const _: () = {
    assert!(SPEC.current_turn_char_budget <= SPEC.context_char_budget);
    assert!(SPEC.context_char_budget <= 8192 * 4);
    assert!(SPEC.default_max_concurrency >= 1);
};

/// Build the engine from the Vertex slice of its
/// `[classifier.gemini-embedding-2]` table.
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<VertexEmbedding> {
    VertexEmbedding::connect(
        &SPEC,
        &cfg.gemini_embedding_2.to_vertex(),
        cfg.image_generation_threshold,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::ClassifierModel;

    #[test]
    fn spec_name_matches_classifier_model_wire_name() {
        assert_eq!(SPEC.name, ClassifierModel::GeminiEmbedding2.as_str());
    }
}
