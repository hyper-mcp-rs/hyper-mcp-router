//! Anchor-prototype classification support shared by every embedding-based
//! engine, regardless of provider. **Not a model** — provider families
//! (`gemini/`, `vertex/`) own their transport; this module owns the method.
//!
//! ## How embedding classification works
//!
//! Unlike the embedded NLI engine (which scores hypotheses directly), an
//! embedding model gives us vectors, so classification is done by **anchor
//! prototypes**: at startup, a curated set of exemplar texts per class
//! (fast / balanced / frontier / image-generation) is embedded in one batch
//! call, and each class's exemplar vectors are mean-pooled and normalized
//! into a single prototype. Per request, the complexity window and the
//! current turn are embedded and cosine-scored against the prototypes:
//!
//! - **complexity** = argmax over the three tier prototypes (ties resolve to
//!   the lower tier — cheaper on equal evidence);
//! - **image generation** = the image prototype is the *strict* argmax over
//!   all four prototypes for the current turn AND its similarity clears
//!   `image_generation_threshold` (interpreted as a cosine-similarity floor
//!   for embedding engines) — OR the lexical prefilter matched.
//!
//! Anchors and premises must be embedded by the **same model** for their
//! similarities to be comparable; prototypes are never shared across engines.

use crate::classifier::{Classification, ModelTier};

/// Anchor exemplars per class. Deliberately short, unambiguous, and diverse;
/// these calibrate the prototypes for every embedding engine, so edit with
/// care and re-check the routing distribution afterwards.
const FAST_ANCHORS: &[&str] = &[
    "What is the capital of France?",
    "How many days are in a leap year?",
    "What is 15% of 240?",
    "Define photosynthesis in one sentence.",
    "What time zone is Tokyo in?",
];
const BALANCED_ANCHORS: &[&str] = &[
    "Explain how a vaccine trains the immune system.",
    "Write a Python function to reverse a string and explain how it works.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "How do noise-cancelling headphones work?",
    "Draft a polite email declining a meeting invitation.",
];
const FRONTIER_ANCHORS: &[&str] = &[
    "Derive and rigorously prove the asymptotic time complexity of red-black tree rebalancing with a formal amortized analysis.",
    "Critically evaluate the philosophical arguments for and against compatibilism regarding free will.",
    "Design a distributed consensus protocol tolerant to Byzantine faults and prove its safety and liveness properties.",
    "Provide a rigorous derivation of the Black-Scholes option pricing formula from first principles.",
];
const IMAGE_ANCHORS: &[&str] = &[
    "Generate an image of a red bicycle on a beach.",
    "Draw a picture of a cat wearing a hat.",
    "Create a logo for a coffee shop.",
    "Paint a watercolor illustration of a mountain sunrise.",
];

/// Every anchor text, in the fixed order expected by [`build_prototypes`].
/// Embed these (in order, in one batch) and hand the vectors back.
pub fn anchor_texts() -> Vec<&'static str> {
    FAST_ANCHORS
        .iter()
        .chain(BALANCED_ANCHORS)
        .chain(FRONTIER_ANCHORS)
        .chain(IMAGE_ANCHORS)
        .copied()
        .collect()
}

/// Class prototype vectors (normalized). Constructed via [`build_prototypes`];
/// the [`Default`] value is an empty placeholder used only during engine
/// construction, before the anchors have been embedded.
#[derive(Default)]
pub struct Prototypes {
    fast: Vec<f32>,
    balanced: Vec<f32>,
    frontier: Vec<f32>,
    image: Vec<f32>,
}

impl Prototypes {
    /// Embedding dimensionality (0 for the unbuilt placeholder).
    pub fn dims(&self) -> usize {
        self.fast.len()
    }
}

/// Fold the anchor embeddings (in [`anchor_texts`] order) into per-class
/// prototypes. Errors if the count does not match the anchor list — a
/// provider returning the wrong number of vectors must be loud.
pub fn build_prototypes(embeddings: &[Vec<f32>]) -> anyhow::Result<Prototypes> {
    let expected = anchor_texts().len();
    if embeddings.len() != expected {
        anyhow::bail!(
            "expected {expected} anchor embeddings, got {}",
            embeddings.len()
        );
    }
    let mut offset = 0;
    let mut take = |n: usize| {
        let slice = &embeddings[offset..offset + n];
        offset += n;
        mean_normalized(slice)
    };
    Ok(Prototypes {
        fast: take(FAST_ANCHORS.len()),
        balanced: take(BALANCED_ANCHORS.len()),
        frontier: take(FRONTIER_ANCHORS.len()),
        image: take(IMAGE_ANCHORS.len()),
    })
}

/// Fold prototype similarities into a [`Classification`] (see module docs for
/// the exact rules). `image_embedding` is `None` when the current turn was
/// empty — image intent is then only reachable via the lexical prefilter.
pub fn combine_similarities(
    complexity_embedding: &[f32],
    image_embedding: Option<&[f32]>,
    prototypes: &Prototypes,
    lexical_image_match: bool,
    image_gen_threshold: f32,
) -> Classification {
    let tiers = [
        (
            ModelTier::Fast,
            cosine(complexity_embedding, &prototypes.fast),
        ),
        (
            ModelTier::Balanced,
            cosine(complexity_embedding, &prototypes.balanced),
        ),
        (
            ModelTier::Frontier,
            cosine(complexity_embedding, &prototypes.frontier),
        ),
    ];
    let mut best = tiers[0];
    for &t in &tiers[1..] {
        if t.1 > best.1 {
            best = t;
        }
    }

    let image_generation = lexical_image_match
        || image_embedding.is_some_and(|emb| {
            let image_sim = cosine(emb, &prototypes.image);
            let max_tier_sim = [
                cosine(emb, &prototypes.fast),
                cosine(emb, &prototypes.balanced),
                cosine(emb, &prototypes.frontier),
            ]
            .into_iter()
            .fold(f32::NEG_INFINITY, f32::max);
            image_sim > max_tier_sim && image_sim >= image_gen_threshold
        });

    Classification {
        complexity: best.0,
        image_generation,
    }
}

/// Mean-pool a set of vectors and L2-normalize the result.
fn mean_normalized(vectors: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = vectors.first() else {
        return Vec::new();
    };
    let mut mean = vec![0.0f32; first.len()];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x;
        }
    }
    let n = vectors.len() as f32;
    for m in &mut mean {
        *m /= n;
    }
    let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for m in &mut mean {
            *m /= norm;
        }
    }
    mean
}

/// Cosine similarity; 0.0 for zero/mismatched vectors (never NaN).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protos() -> Prototypes {
        // Orthogonal unit prototypes: dims = [fast, balanced, frontier, image].
        Prototypes {
            fast: vec![1.0, 0.0, 0.0, 0.0],
            balanced: vec![0.0, 1.0, 0.0, 0.0],
            frontier: vec![0.0, 0.0, 1.0, 0.0],
            image: vec![0.0, 0.0, 0.0, 1.0],
        }
    }

    // ── vector math ─────────────────────────────────────────────────────────
    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        // Degenerate inputs never NaN.
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[0.0], &[0.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn mean_normalized_pools_and_normalizes() {
        let m = mean_normalized(&[vec![2.0, 0.0], vec![0.0, 2.0]]);
        // Mean is [1, 1]; normalized to 1/sqrt(2) each.
        assert!((m[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((m[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(mean_normalized(&[]).is_empty());
        // All-zero input stays zero (no NaN from 0/0).
        assert_eq!(mean_normalized(&[vec![0.0, 0.0]]), vec![0.0, 0.0]);
    }

    // ── prototype construction ──────────────────────────────────────────────
    #[test]
    fn build_prototypes_requires_exact_anchor_count() {
        let n = anchor_texts().len();
        let embeddings: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0, 0.0]).collect();
        let p = build_prototypes(&embeddings).unwrap();
        assert_eq!(p.dims(), 2);
        // Wrong count is a loud error.
        assert!(build_prototypes(&embeddings[..n - 1]).is_err());
        assert!(build_prototypes(&[]).is_err());
    }

    #[test]
    fn default_prototypes_are_an_empty_placeholder() {
        assert_eq!(Prototypes::default().dims(), 0);
    }

    // ── classification combination ──────────────────────────────────────────
    #[test]
    fn complexity_is_argmax_over_tier_prototypes() {
        let p = protos();
        for (emb, want) in [
            (vec![0.9, 0.2, 0.1, 0.0], ModelTier::Fast),
            (vec![0.1, 0.9, 0.2, 0.0], ModelTier::Balanced),
            (vec![0.1, 0.2, 0.9, 0.0], ModelTier::Frontier),
        ] {
            let c = combine_similarities(&emb, None, &p, false, 0.5);
            assert_eq!(c.complexity, want);
        }
    }

    #[test]
    fn complexity_ties_resolve_to_the_lower_tier() {
        let p = protos();
        // Equidistant from fast and frontier: prefer the cheaper tier.
        let c = combine_similarities(&[0.5, 0.0, 0.5, 0.0], None, &p, false, 0.5);
        assert_eq!(c.complexity, ModelTier::Fast);
    }

    #[test]
    fn image_requires_strict_argmax_and_threshold() {
        let p = protos();
        // Image dominant and above threshold => image intent.
        let c = combine_similarities(
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.1, 0.0, 0.0, 0.9]),
            &p,
            false,
            0.5,
        );
        assert!(c.image_generation);
        // Image dominant (argmax) but below the configured threshold => no
        // image intent. Cosine is scale-invariant, so the vector must be
        // genuinely *angled away* from the prototype, not just small:
        // cos([0.5, 0, 0, 0.86], image) ≈ 0.86, under a 0.9 threshold.
        let c = combine_similarities(
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.5, 0.0, 0.0, 0.86]),
            &p,
            false,
            0.9,
        );
        assert!(!c.image_generation);
        // Image similar but NOT the argmax => no image intent.
        let c = combine_similarities(
            &[0.9, 0.0, 0.0, 0.0],
            Some(&[0.9, 0.0, 0.0, 0.8]),
            &p,
            false,
            0.5,
        );
        assert!(!c.image_generation);
        // No current-turn embedding => image only via lexical.
        let c = combine_similarities(&[0.9, 0.0, 0.0, 0.0], None, &p, false, 0.5);
        assert!(!c.image_generation);
        let c = combine_similarities(&[0.9, 0.0, 0.0, 0.0], None, &p, true, 0.5);
        assert!(c.image_generation);
    }

    #[test]
    fn image_axis_never_affects_complexity() {
        let p = protos();
        let with_image = combine_similarities(
            &[0.0, 0.9, 0.1, 0.0],
            Some(&[0.0, 0.0, 0.0, 1.0]),
            &p,
            false,
            0.5,
        );
        let without = combine_similarities(&[0.0, 0.9, 0.1, 0.0], None, &p, false, 0.5);
        assert_eq!(with_image.complexity, without.complexity);
        assert!(with_image.image_generation && !without.image_generation);
    }

    #[test]
    fn anchor_classes_are_nonempty_and_distinct() {
        for class in [
            FAST_ANCHORS,
            BALANCED_ANCHORS,
            FRONTIER_ANCHORS,
            IMAGE_ANCHORS,
        ] {
            assert!(!class.is_empty());
        }
        // No anchor text may appear in two classes (it would blur prototypes).
        let all = anchor_texts();
        let unique: std::collections::HashSet<&str> = all.iter().copied().collect();
        assert_eq!(all.len(), unique.len());
    }
}
