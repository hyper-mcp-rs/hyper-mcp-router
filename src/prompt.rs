//! Engine-independent prompt handling: extracting user text from a request,
//! the trivial-filler fast path, the lexical image-generation prefilter, and
//! the complexity-classification window.
//!
//! Nothing here touches a model. Every classifier engine (see
//! `crate::engines`) consumes the same window/prompt text produced by these
//! helpers; only the scoring differs per engine.

use std::sync::LazyLock;

use regex::Regex;

/// Prompt-length guard, in **characters** (never bytes — see [`truncate_prompt`]).
const PROMPT_CHAR_LIMIT: usize = 400;

/// Default upper word count for the trivial fast-path (see [`looks_trivial`]).
/// Overridable via the `--trivial-max-words` CLI flag or the
/// `[classifier] trivial_max_words` config setting. Keeps the short-circuit
/// to genuinely terse turns; longer text always reaches the model. A value of 0
/// disables the fast path entirely.
pub const DEFAULT_TRIVIAL_MAX_WORDS: usize = 6;

/// High-precision lexical prefilter for image *generation* intent. Requires an
/// image-creation verb within a short window of an image noun, or explicit
/// text-to-image phrasing / tool names. Deliberately conservative: because it
/// is OR-ed with the engine's model signal, a false positive cross-routes a
/// text request to the image backend, so the pattern avoids weak matches like
/// "draw a conclusion", "create a plan", or "picture this".
pub fn looks_like_image_generation(prompt: &str) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?ix)
              \b(generate|create|draw|paint|render|design|illustrate|sketch|produce)\b
              .{0,40}?
              \b(image|images|picture|pictures|photo|photos|illustration|drawing|
                 painting|logo|icon|artwork|graphic|portrait|wallpaper|poster|avatar)\b
            | \btext[- ]to[- ]image\b
            | \b(dall[- ]?e|midjourney|stable[ -]diffusion|flux)\b
            ",
        )
        .expect("valid image-generation regex")
    });

    RE.is_match(prompt)
}

/// Cheap lexical/length guard for *trivially simple* turns — greetings and
/// acknowledgements like "hi", "ok", "thanks", "please continue". A match means
/// the turn can skip the model pass entirely and route as the Fast tier.
///
/// Deliberately conservative on three axes, all of which must hold:
/// 1. **Short** — at most `max_words` words ("short ≠ simple", so a length cap
///    alone is unsafe; a terse "prove X" must not slip through). `max_words == 0`
///    disables the fast path (nothing is ever trivial).
/// 2. **No reasoning cues** — none of the complexity markers below (guards
///    against "ok, now derive the formula").
/// 3. **Is filler, entirely** — the whole (trimmed) turn must consist of
///    acknowledgement/greeting phrases and punctuation. A matching *prefix* is
///    not enough: "ok tell me about quantum computing" starts with an ack but
///    carries a substantive request, so it must reach the model.
///
/// It only ever routes *down* to Fast; history escalation still applies on top,
/// so a terse turn on a deep/agentic thread is unaffected.
pub fn looks_trivial(prompt: &str, max_words: usize) -> bool {
    /// Reasoning cues that veto the fast path even on a short, filler-looking turn.
    static COMPLEXITY_MARKERS: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?ix)\b(
                 prove|proof|derive|derivation|analy[sz]e|analysis|evaluate|assess|
                 design|architect|optimi[sz]e|integrate|differentiate|refactor|debug|
                 implement|algorithm|complexity|theorem|rigorous|synthesi[sz]e|critique|
                 compare|contrast|explain|summari[sz]e|translate|calculate|solve
               )\b",
        )
        .expect("valid complexity-marker regex")
    });

    /// Acknowledgement / greeting / short-confirmation phrases. The **entire**
    /// trimmed prompt must be a punctuation-separated sequence of these — an
    /// end anchor, not a prefix match — so an ack-prefixed substantive turn
    /// ("ok tell me about X") is never mistaken for filler.
    static ACK_PHRASES: LazyLock<Regex> = LazyLock::new(|| {
        /// One filler phrase.
        const PHRASE: &str = r"(?:
                 (?:hi|hey|hello)(?:\s+there)?|yo|sup
               | ok(?:ay)?|k
               | ye(?:s|ah|p)|yup
               | no|nope|nah
               | thanks|thank\s+you|thx|ty|ta
               | please
               | sure|cool|great|nice|awesome|perfect|excellent
               | got\s+it|gotcha|understood|makes\s+sense|sounds\s+good|will\s+do
               | continue|go\s+on|carry\s+on|keep\s+going|proceed
               | good\s+(?:morning|afternoon|evening|night)
               | bye|goodbye|see\s+you|cheers
               | no\s+problem|np
               | how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|whats\s+up
             )";
        Regex::new(&format!(
            r#"(?ix)^\s*{PHRASE}(?:[\s,.;:!?'"()-]+{PHRASE})*[\s,.;:!?'"()-]*$"#
        ))
        .expect("valid acknowledgement regex")
    });

    let trimmed = prompt.trim();
    let words = trimmed.split_whitespace().count();
    (1..=max_words).contains(&words)
        && !COMPLEXITY_MARKERS.is_match(trimmed)
        && ACK_PHRASES.is_match(trimmed)
}

/// The text content of a single message: a string `content` verbatim, or the
/// concatenated `text` fields of a multi-part `content` (non-text parts ignored).
fn message_text(msg: &serde_json::Value) -> Option<String> {
    let content = msg.get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text.to_owned());
    }
    if let Some(parts) = content.as_array() {
        let text: String = parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        return Some(text);
    }
    None
}

/// The current turn's text: the `content` of the last `role == "user"` message.
/// Used for the image-generation axis (a per-current-turn intent) and logging.
/// Returns `None` when no user message exists.
pub fn extract_prompt(body: &serde_json::Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))?;
    message_text(last_user)
}

/// Whether any user message carries non-empty text. Distinguishes "the user
/// actually said something (however trivial)" from "there is no usable user
/// text at all" (no user messages, or only empty/attachment-only content) —
/// the former can honestly route as chit-chat, the latter falls back to the
/// balanced default.
pub fn has_nonempty_user_text(body: &serde_json::Value) -> bool {
    let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
        return false;
    };
    messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .filter_map(message_text)
        .any(|t| !t.trim().is_empty())
}

/// Build the complexity-classification premise by walking the conversation's
/// **user** turns newest→oldest, skipping trivially-simple ones ([`looks_trivial`]),
/// and accumulating substantive turns until the conversation start or
/// `char_budget` is reached. Surviving turns are returned in chronological order,
/// newline-joined. Returns `None` when no substantive user text remains (e.g.
/// pure chit-chat), which the caller routes as the baseline tier without the model.
///
/// `char_budget` is engine-specific (each model has its own context window —
/// see `ClassifierEngine::context_char_budget`).
///
/// This is what lets a terse follow-up inherit the difficulty of its recent
/// context: "ok, continue" is pruned as trivial, and the walk-back reaches the
/// substantive turns behind it. Only *user* turns are considered — assistant
/// responses (usually the longest messages) are skipped, so the budget stretches
/// across many turns of actual intent. Filler is pruned, so the window ages by
/// *substantive* turns, not by chit-chat.
///
/// **System messages are deliberately excluded.** They are usually static
/// deployment boilerplate ("You are a helpful assistant…") that would consume
/// budget and skew every conversation toward the same tier; the user's own
/// turns are the signal for how hard *this* request is.
pub fn build_classification_window(
    body: &serde_json::Value,
    trivial_max_words: usize,
    char_budget: usize,
) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    let mut collected: Vec<String> = Vec::new(); // newest-first
    let mut used = 0usize;

    for msg in messages.iter().rev() {
        if used >= char_budget {
            break;
        }
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let Some(text) = message_text(msg) else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() || looks_trivial(text, trivial_max_words) {
            continue;
        }
        // Truncate by characters (never bytes) to what remains of the budget.
        let piece: String = text.chars().take(char_budget - used).collect();
        used += piece.chars().count();
        collected.push(piece);
    }

    if collected.is_empty() {
        return None;
    }
    collected.reverse(); // chronological: oldest context first, current turn last
    Some(collected.join("\n"))
}

/// Truncate the prompt to [`PROMPT_CHAR_LIMIT`] **characters** (never byte
/// slicing, which would panic on a multi-byte UTF-8 boundary). A conservative
/// guard for small classifier context windows; the full request JSON is always
/// forwarded unchanged to the backend.
pub fn truncate_prompt(prompt: &str) -> String {
    prompt.chars().take(PROMPT_CHAR_LIMIT).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── looks_like_image_generation ─────────────────────────────────────────
    #[test]
    fn lexical_positives_match() {
        for p in [
            "generate an image of a cat",
            "create a logo",
            "draw a picture of a house",
            "make it with midjourney",
            "text-to-image of a sunset",
            "please render an illustration of a dragon",
            "use stable diffusion to produce artwork",
            "dall-e a portrait",
        ] {
            assert!(looks_like_image_generation(p), "should match: {p:?}");
        }
    }

    #[test]
    fn lexical_negatives_do_not_match() {
        for p in [
            "draw a conclusion",
            "create a plan",
            "picture this scenario",
            "paint a grim outlook",
            "the big picture",
            "generate a report",
        ] {
            assert!(!looks_like_image_generation(p), "should NOT match: {p:?}");
        }
    }

    // ── looks_trivial ─────────────────────────────────────────────────────────
    #[test]
    fn trivial_positives_match() {
        for p in [
            "hi",
            "hello there",
            "ok",
            "okay",
            "yes",
            "no",
            "thanks",
            "thank you",
            "thanks!",
            "sure",
            "cool, got it",
            "got it",
            "understood",
            "continue",
            "please continue",
            "go on",
            "good morning",
            "how are you?",
            "what's up",
            "sounds good",
            "bye",
        ] {
            assert!(
                looks_trivial(p, DEFAULT_TRIVIAL_MAX_WORDS),
                "should be trivial: {p:?}"
            );
        }
    }

    #[test]
    fn trivial_negatives_do_not_match() {
        for p in [
            // not filler at all
            "What is the capital of France?",
            "Tell me more about that.",
            // filler-prefixed but carries a reasoning cue (marker veto)
            "ok, now prove the theorem",
            "sure, please derive the formula",
            "yes, analyze these results",
            // filler-prefixed, short, no reasoning cue — must still reach the
            // model (the ack match is whole-string, not prefix)
            "ok tell me about quantum computing",
            "no rewrite it in Rust",
            "yes but why is the sky blue",
            "thanks, what about France?",
            // too long
            "thanks so much for the detailed and very thorough breakdown you gave",
            // short but technical (short != simple)
            "Integrate sin(x).",
            "Solve for x.",
            "Prove P != NP.",
        ] {
            assert!(
                !looks_trivial(p, DEFAULT_TRIVIAL_MAX_WORDS),
                "should NOT be trivial: {p:?}"
            );
        }
    }

    #[test]
    fn trivial_max_words_zero_disables_pruning() {
        // A ceiling of 0 makes nothing trivial, so no turn is pruned as filler.
        assert!(!looks_trivial("ok", 0));
        let body = json!({"messages": [{"role": "user", "content": "ok"}]});
        // With pruning disabled, even "ok" survives into the window.
        assert_eq!(
            build_classification_window(&body, 0, WIN_BUDGET).as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn trivial_respects_word_ceiling() {
        // "ok sure thanks" is 3 words: trivial at ceiling 3, not at ceiling 2.
        assert!(looks_trivial("ok sure thanks", 3));
        assert!(!looks_trivial("ok sure thanks", 2));
    }

    #[test]
    fn trivial_accepts_multi_phrase_filler_with_punctuation() {
        for p in [
            "ok, thanks!",
            "yes please",
            "great, sounds good!!",
            "ok... continue",
        ] {
            assert!(
                looks_trivial(p, DEFAULT_TRIVIAL_MAX_WORDS),
                "should be trivial: {p:?}"
            );
        }
    }

    // ── build_classification_window ───────────────────────────────────────────
    const WIN_BUDGET: usize = 1000;

    #[test]
    fn window_none_when_no_user_messages() {
        let body = json!({"messages": [{"role": "system", "content": "sys"}]});
        assert!(
            build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET).is_none()
        );
    }

    #[test]
    fn window_none_when_all_turns_trivial() {
        // Pure chit-chat prunes to nothing → caller routes baseline Fast.
        let body = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "thanks"},
            {"role": "user", "content": "ok"},
        ]});
        assert!(
            build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET).is_none()
        );
    }

    #[test]
    fn window_skips_assistant_and_trivial_turns_keeps_substantive() {
        // A terse follow-up inherits the substantive context behind it.
        let body = json!({"messages": [
            {"role": "user", "content": "Prove that sqrt 2 is irrational."},
            {"role": "assistant", "content": "A very long proof the window must ignore..."},
            {"role": "user", "content": "ok, continue"},
        ]});
        let window = build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET)
            .expect("substantive turn present");
        assert!(window.contains("sqrt 2 is irrational"));
        assert!(
            !window.contains("long proof"),
            "assistant text must be excluded"
        );
        assert!(
            !window.contains("ok, continue"),
            "trivial turn must be pruned"
        );
    }

    #[test]
    fn window_orders_chronologically_current_turn_last() {
        let body = json!({"messages": [
            {"role": "user", "content": "first substantive question about topology"},
            {"role": "user", "content": "second substantive question about homology"},
        ]});
        let window =
            build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, WIN_BUDGET).unwrap();
        let first = window.find("topology").unwrap();
        let second = window.find("homology").unwrap();
        assert!(
            first < second,
            "older context should precede the current turn"
        );
    }

    #[test]
    fn window_respects_char_budget() {
        let long_a = "a".repeat(80);
        let long_b = "b".repeat(80);
        let body = json!({"messages": [
            {"role": "user", "content": long_a},
            {"role": "user", "content": long_b},
        ]});
        // Budget (90) fits the most recent turn (80) fully and only a sliver of
        // the older one; total collected content stays within budget.
        let window = build_classification_window(&body, DEFAULT_TRIVIAL_MAX_WORDS, 90).unwrap();
        assert!(
            window.contains(long_b.as_str()),
            "most recent turn kept in full"
        );
        let a_count = window.chars().filter(|&c| c == 'a').count();
        assert!(
            a_count > 0 && a_count < 80,
            "older turn should be truncated to the remaining budget, got {a_count}"
        );
    }

    // ── extract_prompt ──────────────────────────────────────────────────────
    #[test]
    fn extract_last_user_message_wins() {
        let body = json!({"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]});
        assert_eq!(extract_prompt(&body).as_deref(), Some("second"));
    }

    #[test]
    fn extract_multipart_content_concatenated() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "world"},
        ]}]});
        assert_eq!(extract_prompt(&body).as_deref(), Some("hello world"));
    }

    #[test]
    fn user_text_presence_detected() {
        // Non-empty user text → true.
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(has_nonempty_user_text(&body));
        // Empty string content → false.
        let body = json!({"messages": [{"role": "user", "content": ""}]});
        assert!(!has_nonempty_user_text(&body));
        // Attachment-only multi-part content (no text parts) → false.
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}]});
        assert!(!has_nonempty_user_text(&body));
        // No user messages at all → false.
        let body = json!({"messages": [{"role": "system", "content": "sys"}]});
        assert!(!has_nonempty_user_text(&body));
        // An earlier user turn with text counts even if the last is empty.
        let body = json!({"messages": [
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": ""},
        ]});
        assert!(has_nonempty_user_text(&body));
    }

    #[test]
    fn extract_missing_user_message_returns_none() {
        let body = json!({"messages": [{"role": "system", "content": "sys"}]});
        assert!(extract_prompt(&body).is_none());
    }

    // ── truncate_prompt ─────────────────────────────────────────────────────
    #[test]
    fn truncate_takes_400_chars() {
        let long = "a".repeat(1000);
        assert_eq!(truncate_prompt(&long).chars().count(), 400);
    }

    #[test]
    fn truncate_handles_multibyte_utf8_without_panicking() {
        // Each '😀' is 4 bytes; byte slicing at 400 would panic on a boundary.
        let s = "😀".repeat(500);
        let truncated = truncate_prompt(&s);
        assert_eq!(truncated.chars().count(), 400);
    }
}
