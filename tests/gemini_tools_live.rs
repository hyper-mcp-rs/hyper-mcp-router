//! Live end-to-end tool-loop check against the Generative Language API's
//! OpenAI-compatible endpoint (NOT a mock): builds the full router with a
//! single Gemini backend, elicits a real `tool_calls` response, then answers
//! it with the tool result — echoing the assistant message VERBATIM,
//! including any vendor extensions (`extra_content` carrying
//! `google.thought_signature`).
//!
//! This is the round trip that breaks when anything on the path drops
//! `extra_content`: Gemini 3.x requires the thought signature it attached to
//! a tool call to be echoed back with the tool result, and 400s without it
//! (Gemini 2.5 merely tolerated the omission). Turn 2 returning 200 through
//! the router is the live proof that the router's passthrough preserves
//! everything the model needs.
//!
//! Ignored by default (needs credentials + network). Run with:
//!   GEMINI_API_KEY=<key> cargo test --test gemini_tools_live -- --ignored --nocapture
//! Model defaults to `gemini-3.5-flash-lite`; override with GEMINI_LIVE_MODEL.
//!
//! The key is read from the environment only; it is never written to disk.

use std::sync::Arc;

use serde_json::{json, Value};

use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};
use hyper_mcp_router::{config, engines};

fn api_key() -> String {
    std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY to run this test")
}

fn live_model() -> String {
    std::env::var("GEMINI_LIVE_MODEL").unwrap_or_else(|_| "gemini-3.5-flash-lite".to_string())
}

/// Start the router with a single live Gemini backend (so classification is
/// skipped — this test is about the tool-loop contract, not routing) and
/// return its base URL.
async fn start_router(model: &str) -> String {
    let toml = format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[[models]]
name = "{model}"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key = "{key}"
type = "balanced"
modalities = ["text", "tools"]
context_window = 128000
"#,
        key = api_key(),
    );
    let cfg = config::parse(&toml).expect("parse config");
    cfg.validate().expect("validate config");

    let engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
        .await
        .expect("build engine");
    let trivial_max_words = cfg.classifier.trivial_max_words;
    let state =
        AppState::with_single_engine(engine, Arc::new(cfg), trivial_max_words).expect("app state");
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn weather_tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }])
}

#[tokio::test]
#[ignore = "hits the live Generative Language API; needs GEMINI_API_KEY"]
async fn tool_loop_round_trips_live_with_thought_signatures_preserved() {
    let model = live_model();
    let base = start_router(&model).await;
    let client = reqwest::Client::new();

    // ── Turn 1: elicit a real tool call. ────────────────────────────────
    let turn1 = json!({
        "model": ADVERTISED_MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris right now? \
              Use the get_weather tool; do not answer from memory."},
        ],
        "tools": weather_tools(),
        "tool_choice": "required",
    });
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&turn1)
        .send()
        .await
        .expect("send turn 1");
    let status = resp.status();
    let body: Value = resp.json().await.expect("turn 1 JSON body");
    println!("turn 1 ({model}): {body:#}");
    assert_eq!(status, reqwest::StatusCode::OK, "turn 1 body: {body}");

    // The assistant message, kept as raw JSON: whatever vendor extensions
    // the model attached (`extra_content.google.thought_signature` on
    // Gemini 3.x) ride along untouched when we echo it below.
    let assistant = body["choices"][0]["message"].clone();
    let tool_calls = assistant["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("turn 1 must return tool_calls; body: {body:#}"));
    assert!(!tool_calls.is_empty(), "empty tool_calls; body: {body:#}");
    let call_id = tool_calls[0]["id"].as_str().expect("tool call id");
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");

    let signature = &tool_calls[0]["extra_content"]["google"]["thought_signature"];
    println!(
        "thought signature on the tool call: {}",
        if signature.is_string() {
            "present (Gemini 3.x behavior — turn 2 would 400 if it were dropped)"
        } else {
            "absent (model did not attach one; turn 2 still validates the loop)"
        }
    );

    // ── Turn 2: answer the tool call, echoing the assistant turn verbatim.
    let turn2 = json!({
        "model": ADVERTISED_MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris right now? \
              Use the get_weather tool; do not answer from memory."},
            assistant,
            {"role": "tool", "tool_call_id": call_id,
             "content": "{\"city\": \"Paris\", \"temp_c\": 21, \"conditions\": \"partly cloudy\"}"},
        ],
        "tools": weather_tools(),
    });
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&turn2)
        .send()
        .await
        .expect("send turn 2");
    let status = resp.status();
    let body: Value = resp.json().await.expect("turn 2 JSON body");
    println!("turn 2 ({model}): {body:#}");
    // THE assertion: the tool-result turn is accepted. If the router (or
    // anything it does to the body) dropped `extra_content`, Gemini 3.x
    // rejects this request with a 400 naming the missing thought signature.
    assert_eq!(status, reqwest::StatusCode::OK, "turn 2 body: {body}");
    let answer = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(!answer.is_empty(), "turn 2 must answer; body: {body:#}");

    // ── Negative control (diagnostic only, model-behavior dependent): the
    // same turn with `extra_content` stripped from the echoed tool call.
    // On models that enforce thought signatures this 400s — direct evidence
    // that a signature-dropping hop (NOT this router) causes the failures.
    if signature.is_string() {
        let mut stripped = turn2.clone();
        stripped["messages"][1]["tool_calls"][0]
            .as_object_mut()
            .expect("tool call object")
            .remove("extra_content");
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&stripped)
            .send()
            .await
            .expect("send stripped control");
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        println!("negative control (extra_content stripped): {status} — {body}",);
    }
}
