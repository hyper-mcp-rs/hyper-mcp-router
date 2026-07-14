//! The `text-embedding-005` engine: remote anchor-prototype embedding
//! classification against Google's `text-embedding-005` model (see this
//! directory's `mod.rs` for the shared method).
//!
//! Model-specific facts owned by this file: the API model path and the
//! context budgets — like `gemini-embedding-001` it accepts 2048 input
//! tokens, so the budgets match; it is a smaller/cheaper embedding model,
//! which makes it attractive for high-throughput deployments. Requires
//! `[classifier.text-embedding-005] api_key`.
//!
//! Privacy note: selecting this engine sends prompt text (the classification
//! window and current turn) to the Gemini API.

use crate::config::ClassifierConfig;

use super::{GeminiEmbedding, GeminiSpec};

/// Model-specific parameters for `text-embedding-005`.
pub const SPEC: GeminiSpec = GeminiSpec {
    name: "text-embedding-005",
    api_model: "models/text-embedding-005",
    // 2048-token input limit; ~4 chars/token with headroom.
    context_char_budget: 6000,
    current_turn_char_budget: 2000,
    default_max_concurrency: 32,
    default_request_timeout_secs: 10,
};

// Compile-time spec coherence: current-turn budget within the window budget;
// window under the model's 2048-token input limit (~4 chars/token);
// concurrency at least 1.
const _: () = {
    assert!(SPEC.current_turn_char_budget <= SPEC.context_char_budget);
    assert!(SPEC.context_char_budget <= 2048 * 4);
    assert!(SPEC.default_max_concurrency >= 1);
};

/// Build the engine from its `[classifier.text-embedding-005]` table.
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<GeminiEmbedding> {
    GeminiEmbedding::connect(
        &SPEC,
        &cfg.text_embedding_005,
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
        assert_eq!(SPEC.name, ClassifierModel::TextEmbedding005.as_str());
    }
}
