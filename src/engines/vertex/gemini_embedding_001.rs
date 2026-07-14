//! The `gemini-embedding-001` engine **on Vertex AI** — a completely separate
//! engine from `engines/gemini/embedding_001.rs`, which talks to the same
//! model on the Generative Language API. They share a model name and context
//! budgets, nothing else: different endpoint layout, wire format
//! (`instances`/`predictions` vs `batchEmbedContents`), and auth (ADC Bearer
//! vs API key). Which one runs is chosen by the auth fields of
//! `[classifier.gemini-embedding-001]` — `project` selects this engine (see
//! `config::GoogleEmbeddingConfig::surface`).
//!
//! Wire compatibility verified live: multi-instance `:predict` batches return
//! one 3072-dim prediction per instance, like `text-embedding-005`.
//!
//! Privacy note: selecting this engine sends prompt text (the classification
//! window and current turn) to the Vertex AI API.

use crate::config::ClassifierConfig;

use super::{VertexEmbedding, VertexSpec};

/// Model-specific parameters for `gemini-embedding-001` on Vertex AI.
pub const SPEC: VertexSpec = VertexSpec {
    name: "gemini-embedding-001",
    api_model: "gemini-embedding-001",
    // 2048-token input limit; ~4 chars/token with headroom (the model is the
    // same on both surfaces, so the budgets match the gemini/ twin).
    context_char_budget: 6000,
    current_turn_char_budget: 2000,
    default_max_concurrency: 32,
    default_request_timeout_secs: 10,
};

// Compile-time spec coherence: current-turn budget within the window budget;
// window under the model's 2048-token input limit (~4 chars/token);
// concurrency at least 1. (The `gemini/` twin declares the same budgets —
// same underlying model — but deliberately without a code-level tie: the two
// engines are fully independent.)
const _: () = {
    assert!(SPEC.current_turn_char_budget <= SPEC.context_char_budget);
    assert!(SPEC.context_char_budget <= 2048 * 4);
    assert!(SPEC.default_max_concurrency >= 1);
};

/// Build the engine from the Vertex slice of its
/// `[classifier.gemini-embedding-001]` table.
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<VertexEmbedding> {
    VertexEmbedding::connect(
        &SPEC,
        &cfg.gemini_embedding_001.to_vertex(),
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
        assert_eq!(SPEC.name, ClassifierModel::GeminiEmbedding001.as_str());
    }
}
