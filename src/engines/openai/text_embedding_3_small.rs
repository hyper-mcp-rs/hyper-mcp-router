//! The `text-embedding-3-small` engine: remote anchor-prototype embedding
//! classification against OpenAI's `text-embedding-3-small` model (see this
//! directory's `mod.rs` for the transport, `engines/embedding.rs` for the
//! method).
//!
//! Model-specific facts owned by this file: the API model id and the context
//! budgets — the model accepts 8191 input tokens, so both budgets are large
//! and image-generation intent deep in a long prompt stays visible. The
//! small variant is the cheaper, lower-dimensional option. Requires
//! `[classifier.text-embedding-3-small] api_key`.
//!
//! Privacy note: selecting this engine sends prompt text (the classification
//! window and current turn) to the OpenAI API.

use crate::config::ClassifierConfig;

use super::{OpenAiEmbedding, OpenAiSpec};

/// Model-specific parameters for `text-embedding-3-small`.
pub const SPEC: OpenAiSpec = OpenAiSpec {
    name: "text-embedding-3-small",
    api_model: "text-embedding-3-small",
    // 8191-token input limit; ~4 chars/token with headroom.
    context_char_budget: 24000,
    current_turn_char_budget: 8000,
    default_max_concurrency: 32,
    default_request_timeout_secs: 10,
};

// Compile-time spec coherence: current-turn budget within the window budget;
// window under the model's 8191-token input limit (~4 chars/token);
// concurrency at least 1.
const _: () = {
    assert!(SPEC.current_turn_char_budget <= SPEC.context_char_budget);
    assert!(SPEC.context_char_budget <= 8191 * 4);
    assert!(SPEC.default_max_concurrency >= 1);
};

/// Build the engine from its `[classifier.text-embedding-3-small]` table.
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<OpenAiEmbedding> {
    OpenAiEmbedding::connect(
        &SPEC,
        &cfg.text_embedding_3_small,
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
        assert_eq!(SPEC.name, ClassifierModel::TextEmbedding3Small.as_str());
    }
}
