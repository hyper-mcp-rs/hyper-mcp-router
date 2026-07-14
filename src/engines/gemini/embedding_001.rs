//! The `gemini-embedding-001` engine: remote anchor-prototype embedding
//! classification against Google's `gemini-embedding-001` model (see this
//! directory's `mod.rs` for the shared method).
//!
//! Model-specific facts owned by this file: the API model path, the context
//! budgets (the model accepts 2048 input tokens, so the window budget stays
//! conservatively under that at ~4 chars/token), and the concurrency/timeout
//! defaults. Requires `[classifier.gemini-embedding-001] api_key`.
//!
//! Privacy note: selecting this engine sends prompt text (the classification
//! window and current turn) to the Gemini API.

use crate::config::ClassifierConfig;

use super::{GeminiEmbedding, GeminiSpec};

/// Model-specific parameters for `gemini-embedding-001`.
pub const SPEC: GeminiSpec = GeminiSpec {
    name: "gemini-embedding-001",
    api_model: "models/gemini-embedding-001",
    // 2048-token input limit; ~4 chars/token with headroom.
    context_char_budget: 6000,
    current_turn_char_budget: 2000,
    default_max_concurrency: 32,
    default_request_timeout_secs: 10,
};

// Compile-time spec coherence: the current turn is always part of the
// window's conversation, so its budget must not exceed the window budget;
// the window must stay under the model's 2048-token input limit
// (~4 chars/token); concurrency must be at least 1.
const _: () = {
    assert!(SPEC.current_turn_char_budget <= SPEC.context_char_budget);
    assert!(SPEC.context_char_budget <= 2048 * 4);
    assert!(SPEC.default_max_concurrency >= 1);
};

/// Build the engine from its `[classifier.gemini-embedding-001]` table.
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<GeminiEmbedding> {
    GeminiEmbedding::connect(
        &SPEC,
        &cfg.gemini_embedding_001,
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
