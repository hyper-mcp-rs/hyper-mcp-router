//! Live end-to-end check of a `google-adc` routed model (NOT a mock): builds
//! the full router from a config whose only backend is the Vertex AI
//! OpenAI-compatible endpoint with `api_key = { source = "google-adc" }`,
//! then drives a real chat completion through it. Exercises startup ADC
//! discovery (`AppState::new`), the per-request token fetch in the proxy, and
//! the actual Vertex endpoint contract.
//!
//! Ignored by default (needs credentials + network + a GCP project with
//! Vertex AI enabled). Run with:
//!   VERTEX_TEST_PROJECT=<proj> cargo test --test google_adc_live -- --ignored --nocapture
//! Auth: Application Default Credentials (`gcloud auth application-default
//! login`).

use std::sync::Arc;

use serde_json::{json, Value};

use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};
use hyper_mcp_router::{config, engines};

#[tokio::test]
#[ignore = "hits the live Vertex AI OpenAI-compatible endpoint; needs VERTEX_TEST_PROJECT + ADC"]
async fn google_adc_routed_model_completes_live() {
    let Ok(project) = std::env::var("VERTEX_TEST_PROJECT") else {
        panic!("set VERTEX_TEST_PROJECT to run this test");
    };

    let toml = format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[[models]]
name = "google/gemini-2.5-flash"
base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/us-central1/endpoints/openapi"
api_key = {{ source = "google-adc" }}
type = "balanced"
modalities = ["text"]
context_window = 128000
"#
    );
    let cfg = config::parse(&toml).expect("parse config");
    cfg.validate().expect("validate config");
    assert_eq!(
        cfg.models[0].api_key,
        Some(config::ModelApiKey::GoogleAdc),
        "the marker must survive parsing"
    );

    // Default (embedded) classifier; with a single-model catalogue the proxy
    // skips classification anyway — this test is about auth, not routing.
    let engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
        .await
        .expect("build engine");
    let trivial_max_words = cfg.classifier.trivial_max_words;
    // ADC discovery happens HERE, at startup, because a google-adc model exists.
    let state = AppState::with_single_engine(engine, Arc::new(cfg), trivial_max_words)
        .expect("AppState::new must discover Application Default Credentials");
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": ADVERTISED_MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        }))
        .send()
        .await
        .expect("send chat request");

    let status = resp.status();
    let body: Value = resp.json().await.expect("JSON response body");
    println!("routed completion: {body}");
    assert_eq!(status, reqwest::StatusCode::OK, "body: {body}");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "google/gemini-2.5-flash");
}
