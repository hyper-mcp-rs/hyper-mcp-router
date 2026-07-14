//! The `text-embedding-005` engine: remote anchor-prototype embedding
//! classification against Google's `text-embedding-005` model on **Vertex AI**
//! (see this directory's `mod.rs` for the transport, `engines/embedding.rs`
//! for the method).
//!
//! `text-embedding-005` is published only on Vertex AI, not the Gemini
//! Developer API — hence the Vertex family rather than the `gemini/` family.
//!
//! Model-specific facts owned by this file: the publisher model id and the
//! context budgets — the model accepts 2048 input tokens (and emits up to
//! 768-dim vectors specialised for English and code), so the budgets match
//! the 2048-token siblings. Requires `[classifier.text-embedding-005]`
//! `project` and `location` (no default — a region, or `global`); auth is
//! Application Default Credentials by default, or a static `access_token`
//! override (see this directory's `mod.rs`).
//!
//! Privacy note: selecting this engine sends prompt text (the classification
//! window and current turn) to the Vertex AI API.

use crate::config::ClassifierConfig;

use super::{VertexEmbedding, VertexSpec};

/// Model-specific parameters for `text-embedding-005`.
pub const SPEC: VertexSpec = VertexSpec {
    name: "text-embedding-005",
    api_model: "text-embedding-005",
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
pub async fn build(cfg: &ClassifierConfig) -> anyhow::Result<VertexEmbedding> {
    VertexEmbedding::connect(
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
