//! Model-selection policy: which configured backend serves a request, given
//! its required modality set, classified complexity tier, and estimated
//! context occupancy.
//!
//! Split out of `config` deliberately: `config` owns the *schema* (what a
//! valid file looks like), this module owns the *ranking policy* over the
//! parsed catalogue. The functions are generic over any collection of
//! model-bearing items so the proxy can rank its runtime catalogue (config
//! paired with resolved auth) with the same single implementation.

use std::cmp::Reverse;

use crate::classifier::ModelTier;
use crate::config::ModelConfig;
use crate::modality::ModalitySet;

/// Pick the item whose model's declared modalities are a **superset** of
/// `required` and whose context window fits `estimated_tokens`, preferring
/// `complexity`. Returns `None` when no single model covers the whole
/// modality set (the proxy then returns 422). `model_of` projects an item to
/// its [`ModelConfig`].
///
/// 1. Filter by capability (superset), preserving declaration order.
/// 2. Keep candidates whose context window fits `estimated_tokens` — a
///    "fast" model with a small window must never receive a request that
///    cannot fit in it, regardless of the complexity verdict.
/// 3. Rank fitting survivors: exact type → nearest higher (escalation) →
///    highest lower (fallback); `min_by_key` returns the first minimum, so a
///    tie resolves toward the earlier-declared item.
/// 4. When NO covering candidate fits, fall back to the largest declared
///    window (best tier rank, then declaration order, break ties). The size
///    estimate is a chars-per-token heuristic, so a hard local rejection
///    could refuse requests the backend would actually accept — forwarding
///    to the most capacious backend lets the upstream be the judge.
pub fn select_candidate<'a, T: 'a>(
    items: impl IntoIterator<Item = &'a T>,
    model_of: impl Fn(&T) -> &ModelConfig,
    required: &ModalitySet,
    complexity: ModelTier,
    estimated_tokens: u64,
) -> Option<&'a T> {
    let covering: Vec<&'a T> = items
        .into_iter()
        .filter(|item| model_of(item).modality_set().is_superset(required))
        .collect();
    covering
        .iter()
        .copied()
        .filter(|item| model_of(item).fits_context(estimated_tokens))
        .min_by_key(|item| tier_rank(model_of(item).tier, complexity))
        .or_else(|| {
            covering.into_iter().min_by_key(|item| {
                let m = model_of(item);
                (
                    Reverse(m.context_window.get()),
                    tier_rank(m.tier, complexity),
                )
            })
        })
}

/// Companion to [`select_candidate`]: how many items could serve `required`
/// within their context window. When this is `<= 1` the complexity tier is
/// irrelevant (nothing to rank), so the proxy can skip classification
/// entirely and route directly.
pub fn count_candidates<'a, T: 'a>(
    items: impl IntoIterator<Item = &'a T>,
    model_of: impl Fn(&T) -> &ModelConfig,
    required: &ModalitySet,
    estimated_tokens: u64,
) -> usize {
    items
        .into_iter()
        .filter(|item| {
            let m = model_of(item);
            m.modality_set().is_superset(required) && m.fits_context(estimated_tokens)
        })
        .count()
}

/// A tier's position on the complexity axis. An **explicit** mapping, never
/// `ModelTier as i32`: ranking must not silently change if the enum's
/// variants are ever reordered or extended.
fn tier_level(tier: ModelTier) -> i32 {
    match tier {
        ModelTier::Fast => 0,
        ModelTier::Balanced => 1,
        ModelTier::Frontier => 2,
    }
}

/// Rank offset for a candidate above the wanted tier (escalation).
const ESCALATION_BASE: i32 = 10;
/// Rank offset for a candidate below the wanted tier (fallback). Must leave
/// room for every possible escalation distance underneath it.
const FALLBACK_BASE: i32 = 100;

// The banding only holds while the largest tier distance fits inside each
// band; enforce it so adding tiers can never silently corrupt the ranking.
const MAX_TIER_DISTANCE: i32 = 2; // Frontier - Fast
const _: () = {
    assert!(MAX_TIER_DISTANCE < ESCALATION_BASE);
    assert!(ESCALATION_BASE + MAX_TIER_DISTANCE < FALLBACK_BASE);
};

/// Distance ranking for model selection. Lower is better:
/// exact type (0) < escalation (nearest higher) < fallback (highest lower).
fn tier_rank(tier: ModelTier, want: ModelTier) -> i32 {
    let t = tier_level(tier);
    let w = tier_level(want);
    if t == w {
        0
    } else if t > w {
        ESCALATION_BASE + (t - w) // escalate: prefer the nearest higher type
    } else {
        FALLBACK_BASE + (w - t) // fallback: prefer the highest lower type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    use secrecy::SecretString;
    use url::Url;

    use crate::config::{ModelApiKey, ModelConfig};
    use crate::modality::Modality;

    /// A model with an effectively unbounded window, for tests that exercise
    /// the modality/tier axes without a capacity constraint.
    fn model(name: &str, tier: ModelTier, mods: &[Modality]) -> ModelConfig {
        model_ctx(name, tier, mods, u64::MAX)
    }

    /// [`model`] with a specific context window (tokens).
    fn model_ctx(name: &str, tier: ModelTier, mods: &[Modality], window: u64) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            base_url: Url::parse("http://x").unwrap(),
            api_key: Some(ModelApiKey::Static(SecretString::from("k"))),
            tier,
            modalities: mods.to_vec(),
            context_window: NonZeroU64::new(window).expect("nonzero window"),
        }
    }

    fn req(mods: &[Modality]) -> ModalitySet {
        mods.iter().copied().collect()
    }

    fn select<'a>(
        models: &'a [ModelConfig],
        required: &ModalitySet,
        complexity: ModelTier,
        estimated_tokens: u64,
    ) -> Option<&'a ModelConfig> {
        select_candidate(models.iter(), |m| m, required, complexity, estimated_tokens)
    }

    fn count(models: &[ModelConfig], required: &ModalitySet, estimated_tokens: u64) -> usize {
        count_candidates(models.iter(), |m| m, required, estimated_tokens)
    }

    // ── tier ranking ──────────────────────────────────────────────────────
    #[test]
    fn tier_level_orders_fast_balanced_frontier() {
        assert!(tier_level(ModelTier::Fast) < tier_level(ModelTier::Balanced));
        assert!(tier_level(ModelTier::Balanced) < tier_level(ModelTier::Frontier));
    }

    #[test]
    fn tier_rank_prefers_exact_then_escalation_then_fallback() {
        // Want Balanced: exact beats escalation beats fallback.
        let exact = tier_rank(ModelTier::Balanced, ModelTier::Balanced);
        let escalate = tier_rank(ModelTier::Frontier, ModelTier::Balanced);
        let fallback = tier_rank(ModelTier::Fast, ModelTier::Balanced);
        assert!(exact < escalate && escalate < fallback);
        // Want Fast: the NEAREST higher tier wins among escalations.
        assert!(
            tier_rank(ModelTier::Balanced, ModelTier::Fast)
                < tier_rank(ModelTier::Frontier, ModelTier::Fast)
        );
        // Want Frontier: the HIGHEST lower tier wins among fallbacks.
        assert!(
            tier_rank(ModelTier::Balanced, ModelTier::Frontier)
                < tier_rank(ModelTier::Fast, ModelTier::Frontier)
        );
    }

    // ── capability (modality superset) axis ───────────────────────────────
    #[test]
    fn select_superset_excludes_missing_modality() {
        let models = vec![
            model("text-only", ModelTier::Balanced, &[Modality::Text]),
            model(
                "vision",
                ModelTier::Balanced,
                &[Modality::Text, Modality::ImageInput],
            ),
        ];
        let chosen = select(
            &models,
            &req(&[Modality::Text, Modality::ImageInput]),
            ModelTier::Balanced,
            0,
        )
        .unwrap();
        assert_eq!(chosen.name, "vision");
    }

    #[test]
    fn select_exact_type_wins() {
        let models = vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model("balanced", ModelTier::Balanced, &[Modality::Text]),
            model("frontier", ModelTier::Frontier, &[Modality::Text]),
        ];
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Balanced, 0).unwrap();
        assert_eq!(chosen.name, "balanced");
    }

    #[test]
    fn select_escalates_to_nearest_higher() {
        let models = vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model("frontier", ModelTier::Frontier, &[Modality::Text]),
        ];
        // want Balanced, none exact => nearest higher is Frontier.
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Balanced, 0).unwrap();
        assert_eq!(chosen.name, "frontier");
    }

    #[test]
    fn select_falls_back_to_highest_lower() {
        let models = vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model("balanced", ModelTier::Balanced, &[Modality::Text]),
        ];
        // want Frontier, nothing at/above => highest lower is Balanced.
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Frontier, 0).unwrap();
        assert_eq!(chosen.name, "balanced");
    }

    #[test]
    fn select_first_declared_wins_on_tie() {
        let models = vec![
            model("first", ModelTier::Balanced, &[Modality::Text]),
            model("second", ModelTier::Balanced, &[Modality::Text]),
        ];
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Balanced, 0).unwrap();
        assert_eq!(chosen.name, "first");
    }

    #[test]
    fn select_covers_combination() {
        let models = vec![model(
            "voice",
            ModelTier::Balanced,
            &[Modality::Text, Modality::AudioInput, Modality::AudioOutput],
        )];
        let chosen = select(
            &models,
            &req(&[Modality::AudioInput, Modality::AudioOutput]),
            ModelTier::Balanced,
            0,
        )
        .unwrap();
        assert_eq!(chosen.name, "voice");
    }

    #[test]
    fn select_uncovered_combination_returns_none() {
        let models = vec![
            model(
                "audio-in",
                ModelTier::Balanced,
                &[Modality::Text, Modality::AudioInput],
            ),
            model(
                "audio-out",
                ModelTier::Balanced,
                &[Modality::Text, Modality::AudioOutput],
            ),
        ];
        // No single model covers both directions.
        assert!(select(
            &models,
            &req(&[Modality::AudioInput, Modality::AudioOutput]),
            ModelTier::Balanced,
            0
        )
        .is_none());
    }

    #[test]
    fn select_tools_requires_tool_capable_model() {
        let models = vec![
            model("plain", ModelTier::Balanced, &[Modality::Text]),
            model(
                "agent",
                ModelTier::Frontier,
                &[Modality::Text, Modality::Tools],
            ),
        ];
        // A tools request skips the non-tool model even though `plain` is the
        // closer tier match; capability is a hard constraint.
        let chosen = select(
            &models,
            &req(&[Modality::Text, Modality::Tools]),
            ModelTier::Balanced,
            0,
        )
        .unwrap();
        assert_eq!(chosen.name, "agent");
        // Without the tools requirement, tier preference picks `plain`.
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Balanced, 0).unwrap();
        assert_eq!(chosen.name, "plain");
    }

    #[test]
    fn candidate_count_reflects_superset_matches() {
        let models = vec![
            model("a", ModelTier::Fast, &[Modality::Text]),
            model("b", ModelTier::Balanced, &[Modality::Text]),
            model(
                "vision",
                ModelTier::Balanced,
                &[Modality::Text, Modality::ImageInput],
            ),
        ];
        // Three text models can serve plain text.
        assert_eq!(count(&models, &req(&[Modality::Text]), 0), 3);
        // Only the vision model can serve image input.
        assert_eq!(
            count(&models, &req(&[Modality::Text, Modality::ImageInput]), 0),
            1
        );
        // Nothing serves audio output.
        assert_eq!(count(&models, &req(&[Modality::AudioOutput]), 0), 0);
    }

    // ── context-window fit ────────────────────────────────────────────────
    #[test]
    fn select_skips_models_whose_window_cannot_fit_the_request() {
        // The fast model's tier matches, but its 8k window cannot hold a
        // 100k-token request: capacity beats tier preference.
        let models = vec![
            model_ctx("fast-small", ModelTier::Fast, &[Modality::Text], 8_000),
            model_ctx(
                "frontier-big",
                ModelTier::Frontier,
                &[Modality::Text],
                200_000,
            ),
        ];
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Fast, 100_000).unwrap();
        assert_eq!(chosen.name, "frontier-big");
        // A small request still routes by tier preference.
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Fast, 500).unwrap();
        assert_eq!(chosen.name, "fast-small");
    }

    #[test]
    fn select_falls_back_to_largest_window_when_nothing_fits() {
        // The estimate is a heuristic, so an oversized request is forwarded
        // to the most capacious covering model (best effort) rather than
        // rejected locally.
        let models = vec![
            model_ctx("small", ModelTier::Fast, &[Modality::Text], 8_000),
            model_ctx("medium", ModelTier::Balanced, &[Modality::Text], 32_000),
            model_ctx("large", ModelTier::Frontier, &[Modality::Text], 128_000),
        ];
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Fast, 1_000_000).unwrap();
        assert_eq!(chosen.name, "large");
    }

    #[test]
    fn select_fallback_still_respects_modalities() {
        // Best-effort capacity fallback never overrides a capability
        // constraint: an uncovered modality set stays a 422, whatever the size.
        let models = vec![model_ctx(
            "small",
            ModelTier::Fast,
            &[Modality::Text],
            8_000,
        )];
        assert!(select(
            &models,
            &req(&[Modality::AudioOutput]),
            ModelTier::Fast,
            1_000_000
        )
        .is_none());
    }

    #[test]
    fn select_fallback_breaks_window_ties_by_tier_then_declaration() {
        let models = vec![
            model_ctx("fast-a", ModelTier::Fast, &[Modality::Text], 8_000),
            model_ctx("balanced", ModelTier::Balanced, &[Modality::Text], 8_000),
            model_ctx("fast-b", ModelTier::Fast, &[Modality::Text], 8_000),
        ];
        // Nothing fits 100k; all windows tie at 8k → the wanted tier wins,
        // then declaration order.
        let chosen = select(&models, &req(&[Modality::Text]), ModelTier::Fast, 100_000).unwrap();
        assert_eq!(chosen.name, "fast-a");
    }

    #[test]
    fn candidate_count_reflects_context_fit() {
        let models = vec![
            model_ctx("small", ModelTier::Fast, &[Modality::Text], 8_000),
            model_ctx("large", ModelTier::Frontier, &[Modality::Text], 128_000),
        ];
        assert_eq!(count(&models, &req(&[Modality::Text]), 500), 2);
        // Only the large model fits: a single candidate, so the proxy can
        // skip classification — the tier cannot change the outcome.
        assert_eq!(count(&models, &req(&[Modality::Text]), 100_000), 1);
        // Nothing fits: zero FITTING candidates (selection then falls back).
        assert_eq!(count(&models, &req(&[Modality::Text]), 1_000_000), 0);
    }
}
