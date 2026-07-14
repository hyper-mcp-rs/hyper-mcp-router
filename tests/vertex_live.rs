//! Live end-to-end check of the real Vertex `text-embedding-005` engine code
//! path (NOT a wire-format mock): builds the engine via `engines::build`
//! (which embeds the class anchors over the network and builds prototypes),
//! then runs `classify` on a few premises through the actual reqwest transport
//! and `parse_embeddings`/cosine logic.
//!
//! Ignored by default (needs credentials + network). Run with:
//!   TE005_PROJECT=<proj> cargo test --test vertex_live -- --ignored --nocapture
//! Auth: Application Default Credentials by default (`gcloud auth
//! application-default login`); set GOOGLE_ACCESS_TOKEN to pin a static
//! token instead.

use hyper_mcp_router::classifier::{ClassifierModel, ModelTier};
use hyper_mcp_router::config::{ClassifierConfig, VertexEmbeddingConfig};

#[tokio::test]
#[ignore = "hits the live Vertex AI API; needs TE005_PROJECT (+ ADC or GOOGLE_ACCESS_TOKEN)"]
async fn vertex_text_embedding_005_classifies_live() {
    let Ok(project) = std::env::var("TE005_PROJECT") else {
        panic!("set TE005_PROJECT to run this test");
    };
    // Optional static override; omitted means Application Default Credentials.
    let access_token = std::env::var("GOOGLE_ACCESS_TOKEN").ok();
    // Optional quota/billing project (sent as x-goog-user-project).
    let quota_project = std::env::var("TE005_QUOTA_PROJECT").ok();

    let cfg = ClassifierConfig {
        model: ClassifierModel::TextEmbedding005,
        text_embedding_005: VertexEmbeddingConfig {
            project: Some(project),
            quota_project,
            access_token,
            ..Default::default()
        },
        ..Default::default()
    };

    // Exercises connect(): 18 anchors embedded in one real :predict call, then
    // build_prototypes(). Fails fast on any auth/endpoint/parse problem.
    let engine = hyper_mcp_router::engines::build(&cfg)
        .await
        .expect("engine build (live anchor embedding) should succeed");
    assert_eq!(engine.name(), "text-embedding-005");

    // Each classify() is a real per-request embed + cosine scoring.
    let simple = engine
        .classify("What is the capital of France?", "", false)
        .await
        .expect("simple classify");
    let hard = engine
        .classify(
            "Derive and rigorously prove the asymptotic time complexity of \
             red-black tree rebalancing with a formal amortized analysis.",
            "",
            false,
        )
        .await
        .expect("hard classify");
    let image = engine
        .classify(
            "here is my request",
            "Draw a picture of a cat wearing a hat.",
            false,
        )
        .await
        .expect("image classify");

    println!("simple -> {simple:?}");
    println!("hard   -> {hard:?}");
    println!("image  -> {image:?}");

    // Sanity, not a strict oracle (prototypes are cosine-scored, not exact):
    // the hard prompt should never rank below the simple one, and the image
    // premise should trip the image axis.
    assert!(
        hard.complexity >= simple.complexity,
        "hard should not be cheaper than simple"
    );
    assert!(
        hard.complexity >= ModelTier::Balanced,
        "hard prompt should be balanced+"
    );
    assert!(
        image.image_generation,
        "image premise should set image_generation"
    );
}
