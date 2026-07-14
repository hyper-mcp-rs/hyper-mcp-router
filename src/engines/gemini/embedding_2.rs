//! The `gemini-embedding-2` engine: remote anchor-prototype embedding
//! classification against Google's `gemini-embedding-2` model (see this
//! directory's `mod.rs` for the shared method).
//!
//! Model-specific facts owned by this file: the API model path and the
//! substantially larger context budgets — the model accepts 8192 input
//! tokens, so both the window and the current turn can be much larger than
//! the zero-shot or `gemini-embedding-001` engines allow. That larger
//! current-turn budget is what lets image-generation intent expressed deep in
//! a long prompt stay visible. Requires
//! `[classifier.gemini-embedding-2] api_key`.
//!
//! Privacy note: selecting this engine sends prompt text (the classification
//! window and current turn) to the Gemini API.

use crate::config::ClassifierConfig;

use super::{GeminiEmbedding, GeminiSpec};

/// Model-specific parameters for `gemini-embedding-2`.
pub const SPEC: GeminiSpec = GeminiSpec {
    name: "gemini-embedding-2",
    api_model: "models/gemini-embedding-2",
    // 8192-token input limit; ~4 chars/token with headroom.
    context_char_budget: 24000,
    current_turn_char_budget: 8000,
    default_max_concurrency: 32,
    default_request_timeout_secs: 10,
};

// Compile-time spec coherence: current-turn budget within the window budget;
// window under the model's 8192-token input limit (~4 chars/token); a window
// strictly larger than gemini-embedding-001's (the point of this model);
// concurrency at least 1.
const _: () = {
    assert!(SPEC.current_turn_char_budget <= SPEC.context_char_budget);
    assert!(SPEC.context_char_budget <= 8192 * 4);
    assert!(SPEC.context_char_budget > super::embedding_001::SPEC.context_char_budget);
    assert!(SPEC.default_max_concurrency >= 1);
};

/// Build the engine from its `[classifier.gemini-embedding-2]` table.
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<GeminiEmbedding> {
    GeminiEmbedding::connect(
        &SPEC,
        &cfg.gemini_embedding_2,
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
