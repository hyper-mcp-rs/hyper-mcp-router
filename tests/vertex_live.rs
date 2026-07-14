//! Live end-to-end check of the real Vertex `text-embedding-005` engine code
//! path (NOT a wire-format mock): builds the engine via `engines::build`
//! (which embeds the class anchors over the network and builds prototypes),
//! then runs `classify` on a few premises through the actual reqwest transport
//! and `parse_embeddings`/cosine logic.
//!
//! Ignored by default (needs credentials + network). Run with:
//!   VERTEX_TEST_PROJECT=<proj> cargo test --test vertex_live -- --ignored --nocapture
//! Auth: Application Default Credentials by default (`gcloud auth
//! application-default login`); set GOOGLE_ACCESS_TOKEN to pin a static
//! token instead.

use hyper_mcp_router::classifier::{ClassifierModel, ModelTier};
use hyper_mcp_router::config::{ClassifierConfig, GoogleEmbeddingConfig, VertexEmbeddingConfig};

/// Build the engine for `cfg`, classify the standard premises at `location`,
/// and apply the shared sanity assertions (cosine scoring is not an exact
/// oracle; see comments).
async fn classify_and_assert(cfg: ClassifierConfig, engine_name: &str, location: &str) {
    let engine = hyper_mcp_router::engines::build(&cfg)
        .await
        .unwrap_or_else(|e| panic!("build {engine_name} at `{location}`: {e:#}"));
    assert_eq!(engine.name(), engine_name);

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

    println!("[{engine_name} @ {location}] simple -> {simple:?}");
    println!("[{engine_name} @ {location}] hard   -> {hard:?}");
    println!("[{engine_name} @ {location}] image  -> {image:?}");

    // Sanity, not a strict oracle: the hard prompt should never rank below
    // the simple one, and the image premise should trip the image axis.
    assert!(
        hard.complexity >= simple.complexity,
        "[{engine_name} @ {location}] hard should not be cheaper than simple"
    );
    assert!(
        hard.complexity >= ModelTier::Balanced,
        "[{engine_name} @ {location}] hard prompt should be balanced+"
    );
    assert!(
        image.image_generation,
        "[{engine_name} @ {location}] image premise should set image_generation"
    );
}

/// text-embedding-005: build and classify live at the `us-central1` region
/// AND the `us` multi-region (both live-verified to serve the model).
#[tokio::test]
#[ignore = "hits the live Vertex AI API; needs VERTEX_TEST_PROJECT (+ ADC or GOOGLE_ACCESS_TOKEN)"]
async fn vertex_text_embedding_005_classifies_live_at_region_and_multiregion() {
    let Ok(project) = std::env::var("VERTEX_TEST_PROJECT") else {
        panic!("set VERTEX_TEST_PROJECT to run this test");
    };
    // Optional static override; omitted means Application Default Credentials.
    let access_token = std::env::var("GOOGLE_ACCESS_TOKEN").ok();
    // Optional quota/billing project (sent as x-goog-user-project).
    let quota_project = std::env::var("VERTEX_TEST_QUOTA_PROJECT").ok();

    for location in ["us-central1", "us"] {
        let cfg = ClassifierConfig {
            model: ClassifierModel::TextEmbedding005,
            text_embedding_005: VertexEmbeddingConfig {
                project: Some(project.clone()),
                location: Some(location.into()),
                quota_project: quota_project.clone(),
                access_token: access_token.clone(),
                ..Default::default()
            },
            ..Default::default()
        };
        classify_and_assert(cfg, "text-embedding-005", location).await;
    }
}

/// gemini-embedding-2 on the **Vertex** surface: served only via
/// `:embedContent` at the `us` multi-region and `global` locations — build
/// and classify live at BOTH (anchor fan-out, per-request embeds, ADC).
#[tokio::test]
#[ignore = "hits the live Vertex AI API; needs VERTEX_TEST_PROJECT (+ ADC or GOOGLE_ACCESS_TOKEN)"]
async fn gemini_embedding_2_on_vertex_classifies_live_at_us_and_global() {
    let Ok(project) = std::env::var("VERTEX_TEST_PROJECT") else {
        panic!("set VERTEX_TEST_PROJECT to run this test");
    };
    let access_token = std::env::var("GOOGLE_ACCESS_TOKEN").ok();

    for location in ["us", "global"] {
        let cfg = ClassifierConfig {
            model: ClassifierModel::GeminiEmbedding2,
            gemini_embedding_2: GoogleEmbeddingConfig {
                project: Some(project.clone()),
                location: Some(location.into()),
                access_token: access_token.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        classify_and_assert(cfg, "gemini-embedding-2", location).await;
    }
}

/// gemini-embedding-001 on the **Vertex** surface (project set, no api_key):
/// the auth-driven dispatch must build the vertex twin and classify live at
/// the `us-central1` region AND the `us` multi-region.
#[tokio::test]
#[ignore = "hits the live Vertex AI API; needs VERTEX_TEST_PROJECT (+ ADC or GOOGLE_ACCESS_TOKEN)"]
async fn gemini_embedding_001_on_vertex_classifies_live_at_region_and_multiregion() {
    let Ok(project) = std::env::var("VERTEX_TEST_PROJECT") else {
        panic!("set VERTEX_TEST_PROJECT to run this test");
    };
    let access_token = std::env::var("GOOGLE_ACCESS_TOKEN").ok();

    for location in ["us-central1", "us"] {
        let cfg = ClassifierConfig {
            model: ClassifierModel::GeminiEmbedding001,
            gemini_embedding_001: GoogleEmbeddingConfig {
                project: Some(project.clone()),
                location: Some(location.into()),
                access_token: access_token.clone(),
                ..Default::default()
            },
            ..Default::default()
        };
        classify_and_assert(cfg, "gemini-embedding-001", location).await;
    }
}
