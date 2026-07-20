//! Live end-to-end checks of the Generative Language API engines (the
//! `gemini/` family, NOT the Vertex twins): builds each engine via
//! `engines::build` — exercising the auth-driven surface dispatch's
//! `api_key` branch — embeds the class anchors over the network, and runs
//! `classify` on the standard premises.
//!
//! Ignored by default (needs credentials + network). Run with:
//!   GEMINI_API_KEY=<key> cargo test --test gemini_live -- --ignored --nocapture
//!
//! The key is read from the environment only; it is never written to disk.

use hyper_mcp_router::classifier::{ClassifierModel, ModelTier};
use hyper_mcp_router::config::{ClassifierConfig, GoogleEmbeddingConfig};

/// Build the engine for `cfg`, classify the standard premises, and apply the
/// shared sanity assertions (cosine scoring is not an exact oracle).
async fn classify_and_assert(cfg: ClassifierConfig, engine_name: &str) {
    let engine = hyper_mcp_router::engines::build(cfg.models[0], &cfg)
        .await
        .unwrap_or_else(|e| panic!("build {engine_name} on the Generative Language API: {e:#}"));
    assert_eq!(engine.name(), engine_name);

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

    println!("[{engine_name}] simple -> {simple:?}");
    println!("[{engine_name}] hard   -> {hard:?}");
    println!("[{engine_name}] image  -> {image:?}");

    assert!(
        hard.complexity >= simple.complexity,
        "[{engine_name}] hard should not be cheaper than simple"
    );
    assert!(
        hard.complexity >= ModelTier::Balanced,
        "[{engine_name}] hard prompt should be balanced+"
    );
    assert!(
        image.image_generation,
        "[{engine_name}] image premise should set image_generation"
    );
}

fn api_key() -> String {
    std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY to run this test")
}

/// gemini-embedding-001 on the Generative Language API (`api_key` selects
/// this surface; `batchEmbedContents` + `x-goog-api-key`).
#[tokio::test]
#[ignore = "hits the live Generative Language API; needs GEMINI_API_KEY"]
async fn gemini_embedding_001_on_generative_language_classifies_live() {
    let cfg = ClassifierConfig {
        models: vec![ClassifierModel::GeminiEmbedding001],
        gemini_embedding_001: GoogleEmbeddingConfig {
            api_key: Some(api_key().into()),
            ..Default::default()
        },
        ..Default::default()
    };
    classify_and_assert(cfg, "gemini-embedding-001").await;
}

/// gemini-embedding-2 on the Generative Language API.
#[tokio::test]
#[ignore = "hits the live Generative Language API; needs GEMINI_API_KEY"]
async fn gemini_embedding_2_on_generative_language_classifies_live() {
    let cfg = ClassifierConfig {
        models: vec![ClassifierModel::GeminiEmbedding2],
        gemini_embedding_2: GoogleEmbeddingConfig {
            api_key: Some(api_key().into()),
            ..Default::default()
        },
        ..Default::default()
    };
    classify_and_assert(cfg, "gemini-embedding-2").await;
}
