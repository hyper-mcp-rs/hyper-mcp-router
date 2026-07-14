//! The modality axis: what a request *requires* and what a model *declares*.
//!
//! Modalities are a capability match (superset selection), never escalated
//! against complexity. Everything here is deterministic, metadata-only, and
//! session-free — the classifier is never consulted.

use serde::{Deserialize, Serialize};

/// Modality axis — the full Chat Completions v1 surface. A model declares the
/// set it supports; a request requires a (possibly multi-element) subset.
/// Direction is explicit for image/audio (asymmetric support); text is one
/// token. Deliberately **not** ordered: modality is a capability match, never
/// escalated against complexity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Modality {
    /// Text in/out (the baseline; every chat model has it).
    Text,
    /// `image_url` content part (vision / image analysis).
    ImageInput,
    /// `input_audio` content part (speech-to-text style).
    AudioInput,
    /// `file` content part (documents, e.g. PDFs).
    FileInput,
    /// Request `modalities` contains `"audio"` (text-to-speech style).
    AudioOutput,
    /// Image generation / creation. Required explicitly via the request's
    /// `modalities: ["image"]` field, or inferred (lexical/NLI) as a soft
    /// preference when absent.
    ImageOutput,
    /// Tool / function calling: the request offers `tools` (or its transcript
    /// carries tool artifacts), so it must route to a model that can emit tool
    /// calls. A capability constraint only — it never affects the complexity
    /// tier.
    Tools,
}

impl Modality {
    /// Kebab-case wire name, used for logging and 422 error bodies.
    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::ImageInput => "image-input",
            Modality::AudioInput => "audio-input",
            Modality::FileInput => "file-input",
            Modality::AudioOutput => "audio-output",
            Modality::ImageOutput => "image-output",
            Modality::Tools => "tools",
        }
    }

    /// Single-bit mask for the [`ModalitySet`] bitset.
    fn bit(self) -> u8 {
        match self {
            Modality::Text => 1 << 0,
            Modality::ImageInput => 1 << 1,
            Modality::AudioInput => 1 << 2,
            Modality::FileInput => 1 << 3,
            Modality::AudioOutput => 1 << 4,
            Modality::ImageOutput => 1 << 5,
            Modality::Tools => 1 << 6,
        }
    }

    /// All modalities, in a stable order for iteration/logging.
    const ALL: [Modality; 7] = [
        Modality::Text,
        Modality::ImageInput,
        Modality::AudioInput,
        Modality::FileInput,
        Modality::AudioOutput,
        Modality::ImageOutput,
        Modality::Tools,
    ];
}

/// A small set of [`Modality`] values, backed by a bitset. Used for the
/// superset matching that drives model selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ModalitySet(u8);

impl ModalitySet {
    /// An empty set.
    pub fn new() -> Self {
        ModalitySet(0)
    }

    /// Add a modality to the set.
    pub fn insert(&mut self, m: Modality) {
        self.0 |= m.bit();
    }

    /// Whether the set contains `m`.
    pub fn contains(&self, m: Modality) -> bool {
        self.0 & m.bit() != 0
    }

    /// Whether `self` covers every modality in `required` (i.e. `required` is a
    /// subset of `self`).
    pub fn is_superset(&self, required: &ModalitySet) -> bool {
        self.0 & required.0 == required.0
    }

    /// Kebab-case names of the contained modalities, in stable order — for
    /// logging and 422 error bodies. Never includes user content.
    pub fn to_kebab_vec(&self) -> Vec<&'static str> {
        Modality::ALL
            .iter()
            .filter(|m| self.contains(**m))
            .map(|m| m.as_str())
            .collect()
    }
}

impl FromIterator<Modality> for ModalitySet {
    fn from_iter<I: IntoIterator<Item = Modality>>(iter: I) -> Self {
        let mut set = ModalitySet::new();
        for m in iter {
            set.insert(m);
        }
        set
    }
}

/// Deterministic modalities required by a request, from content-part types
/// (input), the `modalities` request field (output), the `tools`/`functions`
/// fields, and tool artifacts in the transcript (tool calling). An **explicit**
/// `modalities: ["image"]` request field (used by some OpenAI-compatible image
/// backends) requires `ImageOutput` here; otherwise `ImageOutput` is inferred
/// by the classifier's lexical/NLI signal as a soft preference.
///
/// This function must not call the classifier: it is metadata-only.
pub fn detect_required_modalities(body: &serde_json::Value) -> ModalitySet {
    let mut set = ModalitySet::new();
    // Text I/O is always in play for chat/completions.
    set.insert(Modality::Text);

    // --- Input: scan message content parts. A transcript carrying tool
    // artifacts (`role: "tool"` results, assistant `tool_calls`, or a legacy
    // `function_call`) also requires a tool-capable backend even when the
    // follow-up request omits the `tools` field — other models may reject or
    // mishandle those messages.
    for msg in body["messages"].as_array().into_iter().flatten() {
        if msg.get("role").and_then(|r| r.as_str()) == Some("tool")
            || msg
                .get("tool_calls")
                .and_then(|t| t.as_array())
                .is_some_and(|a| !a.is_empty())
            || msg.get("function_call").is_some_and(|f| f.is_object())
        {
            set.insert(Modality::Tools);
        }
        for part in msg
            .get("content")
            .and_then(|c| c.as_array())
            .into_iter()
            .flatten()
        {
            match part.get("type").and_then(|t| t.as_str()) {
                // "input_image" tolerated as a newer alias for robustness.
                Some("image_url") | Some("input_image") => set.insert(Modality::ImageInput),
                Some("input_audio") => set.insert(Modality::AudioInput),
                Some("file") => set.insert(Modality::FileInput),
                _ => {}
            }
        }
    }

    // --- Output: the `modalities` request field (defaults to ["text"]).
    // `"image"` is honored as an explicit, hard image-output requirement.
    if let Some(mods) = body.get("modalities").and_then(|m| m.as_array()) {
        if mods.iter().any(|m| m.as_str() == Some("audio")) {
            set.insert(Modality::AudioOutput);
        }
        if mods.iter().any(|m| m.as_str() == Some("image")) {
            set.insert(Modality::ImageOutput);
        }
    }

    // --- Tool calling: a request offering tools must route to a tool-capable
    // model. Both the current `tools` array and the deprecated `functions`
    // array count; an empty array does not.
    let offers = |field: &str| {
        body.get(field)
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty())
    };
    if offers("tools") || offers("functions") {
        set.insert(Modality::Tools);
    }

    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── ModalitySet ─────────────────────────────────────────────────────────
    #[test]
    fn modality_set_superset_logic() {
        let mut model = ModalitySet::new();
        model.insert(Modality::Text);
        model.insert(Modality::ImageInput);

        let mut req_ok = ModalitySet::new();
        req_ok.insert(Modality::Text);
        assert!(model.is_superset(&req_ok));

        let mut req_missing = ModalitySet::new();
        req_missing.insert(Modality::Text);
        req_missing.insert(Modality::AudioOutput);
        assert!(!model.is_superset(&req_missing));

        // Every set covers the empty set.
        assert!(model.is_superset(&ModalitySet::new()));
    }

    // ── detect_required_modalities ──────────────────────────────────────────
    #[test]
    fn modality_text_always_present() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let set = detect_required_modalities(&body);
        assert!(set.contains(Modality::Text));
    }

    #[test]
    fn modality_absent_messages_still_text() {
        let body = json!({});
        let set = detect_required_modalities(&body);
        assert_eq!(set.to_kebab_vec(), vec!["text"]);
    }

    #[test]
    fn modality_content_part_types_map_correctly() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "input_audio", "input_audio": {}},
            {"type": "file", "file": {}},
            {"type": "text", "text": "describe"},
        ]}]});
        let set = detect_required_modalities(&body);
        assert!(set.contains(Modality::ImageInput));
        assert!(set.contains(Modality::AudioInput));
        assert!(set.contains(Modality::FileInput));
        assert!(set.contains(Modality::Text));
        assert!(!set.contains(Modality::AudioOutput));
    }

    #[test]
    fn modality_input_image_alias() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "input_image", "image_url": {"url": "x"}},
        ]}]});
        assert!(detect_required_modalities(&body).contains(Modality::ImageInput));
    }

    #[test]
    fn modality_explicit_image_output_from_request_field() {
        let body = json!({
            "messages": [{"role": "user", "content": "a cat, please"}],
            "modalities": ["text", "image"],
        });
        let set = detect_required_modalities(&body);
        assert!(set.contains(Modality::ImageOutput));
    }

    #[test]
    fn modality_tools_detected_from_tool_role_message() {
        let body = json!({"messages": [
            {"role": "user", "content": "look this up"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "user", "content": "and now?"},
        ]});
        // No `tools` field on the follow-up request — the transcript alone
        // must require a tool-capable backend.
        assert!(detect_required_modalities(&body).contains(Modality::Tools));
    }

    #[test]
    fn modality_tools_detected_from_legacy_function_call_message() {
        let body = json!({"messages": [
            {"role": "user", "content": "look this up"},
            {"role": "assistant", "content": null, "function_call": {"name": "f", "arguments": "{}"}},
        ]});
        assert!(detect_required_modalities(&body).contains(Modality::Tools));
    }

    #[test]
    fn modality_audio_output_from_request_field() {
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}],
            "modalities": ["text", "audio"],
        });
        let set = detect_required_modalities(&body);
        assert!(set.contains(Modality::AudioInput));
        assert!(set.contains(Modality::AudioOutput));
    }

    #[test]
    fn modality_string_content_handled() {
        let body = json!({"messages": [{"role": "user", "content": "just text"}]});
        let set = detect_required_modalities(&body);
        assert_eq!(set.to_kebab_vec(), vec!["text"]);
    }

    #[test]
    fn modality_tools_kebab_name() {
        assert_eq!(Modality::Tools.as_str(), "tools");
    }

    #[test]
    fn modality_tools_detected_from_tools_array() {
        let body = json!({
            "messages": [{"role": "user", "content": "what's the weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        });
        assert!(detect_required_modalities(&body).contains(Modality::Tools));
    }

    #[test]
    fn modality_tools_detected_from_legacy_functions_array() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [{"name": "get_weather"}],
        });
        assert!(detect_required_modalities(&body).contains(Modality::Tools));
    }

    #[test]
    fn modality_tools_absent_or_empty_not_detected() {
        // No tools field at all.
        let none = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(!detect_required_modalities(&none).contains(Modality::Tools));
        // Empty tools array does not require the capability.
        let empty = json!({"messages": [{"role": "user", "content": "hi"}], "tools": []});
        assert!(!detect_required_modalities(&empty).contains(Modality::Tools));
    }
}
