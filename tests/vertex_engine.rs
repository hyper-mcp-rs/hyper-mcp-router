//! Integration tests for the Vertex AI embedding engine (`text-embedding-005`).
//!
//! A mock Vertex `:predict` API returns deterministic 4-dimensional
//! "embeddings" derived from keyword features (dims = [fast, balanced,
//! frontier, image]), so the engine's anchor prototypes collapse to near-unit
//! class vectors and routing through a real router instance is fully
//! predictable — no network, no real model. A mock chat backend records the
//! forwarded `model` field as the routing witness.
//!
//! Mirrors `gemini_engine.rs`, but for the Vertex wire format: the
//! `publishers/google/models/<model>:predict` endpoint, the
//! `instances`/`predictions` payload, and `Authorization: Bearer` auth.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode, Uri},
    response::Response,
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};

use hyper_mcp_router::config;
use hyper_mcp_router::engines;
use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};

// ───────────────────────────────────────────────────────────────────────────
// Mock Vertex AI :predict API
// ───────────────────────────────────────────────────────────────────────────

/// One recorded embed call: request path, the Bearer token from the
/// `Authorization` header, and how many texts were embedded.
#[derive(Clone, Debug)]
struct EmbedCall {
    path: String,
    bearer: Option<String>,
    text_count: usize,
}

#[derive(Clone, Default)]
struct EmbedState {
    calls: Arc<Mutex<Vec<EmbedCall>>>,
    /// When set, every embed call returns 500 (per-request failure mode).
    fail: Arc<AtomicBool>,
}

/// Deterministic keyword-feature embedding: [fast, balanced, frontier, image].
fn mock_embedding(text: &str) -> [f32; 4] {
    let t = text.to_lowercase();
    let count = |words: &[&str]| words.iter().filter(|w| t.contains(**w)).count() as f32;
    let mut v = [
        count(&[
            "capital",
            "days",
            "15%",
            "photosynthesis",
            "time zone",
            "what is",
        ]),
        count(&[
            "explain",
            "summarize",
            "email",
            "python",
            "vaccine",
            "headphones",
            "paragraph",
            "compound interest",
        ]),
        count(&[
            "prove",
            "rigorous",
            "amortized",
            "byzantine",
            "consensus",
            "compatibilism",
            "black-scholes",
            "undecidable",
        ]),
        count(&[
            "image",
            "draw",
            "picture",
            "logo",
            "watercolor",
            "paint",
            "illustration",
        ]),
    ];
    if v.iter().all(|&x| x == 0.0) {
        // Neutral text: mildly balanced so prototypes are never poisoned.
        v[1] = 0.1;
    }
    v
}

async fn mock_embed(
    State(state): State<EmbedState>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let texts: Vec<String> = body["instances"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|i| i["content"].as_str())
        .map(str::to_owned)
        .collect();

    state.calls.lock().unwrap().push(EmbedCall {
        path: uri.path().to_string(),
        bearer: headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(str::to_owned),
        text_count: texts.len(),
    });

    if state.fail.load(Ordering::SeqCst) {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(r#"{"error":{"message":"mock outage"}}"#))
            .unwrap();
    }

    let predictions: Vec<Value> = texts
        .iter()
        .map(|t| json!({"embeddings": {"values": mock_embedding(t)}}))
        .collect();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"predictions": predictions})).unwrap(),
        ))
        .unwrap()
}

async fn spawn_mock_vertex() -> (SocketAddr, EmbedState) {
    let state = EmbedState::default();
    // Fallback route matches any path, so the full `:predict` URL shape is
    // recorded rather than hardcoded here.
    let app = Router::new()
        .fallback(post(mock_embed))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, state)
}

// ───────────────────────────────────────────────────────────────────────────
// Mock chat backend (routing witness)
// ───────────────────────────────────────────────────────────────────────────

type ChatCalls = Arc<Mutex<Vec<Value>>>;

async fn mock_chat(State(calls): State<ChatCalls>, Json(body): Json<Value>) -> Json<Value> {
    calls.lock().unwrap().push(body);
    Json(json!({
        "id": "mock-cmpl",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    }))
}

async fn spawn_mock_chat() -> (SocketAddr, ChatCalls) {
    let calls: ChatCalls = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/chat/completions", post(mock_chat))
        .with_state(calls.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, calls)
}

// ───────────────────────────────────────────────────────────────────────────
// Harness
// ───────────────────────────────────────────────────────────────────────────

fn vertex_config_toml(embed_addr: SocketAddr, chat_addr: SocketAddr) -> String {
    let chat = format!("http://{chat_addr}");
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[classifier]
model = "text-embedding-005"

[classifier.text-embedding-005]
project = "test-project"
location = "us-central1"
access_token = "test-vertex-token"
base_url = "http://{embed_addr}"

[[models]]
name = "fast-text"
base_url = "{chat}"
type = "fast"
modalities = ["text"]

[[models]]
name = "balanced-text"
base_url = "{chat}"
type = "balanced"
modalities = ["text"]

[[models]]
name = "frontier-text"
base_url = "{chat}"
type = "frontier"
modalities = ["text"]

[[models]]
name = "image-gen"
base_url = "{chat}"
type = "balanced"
modalities = ["text", "image-output"]
"#
    )
}

struct Harness {
    base: String,
    chat_calls: ChatCalls,
    embed: EmbedState,
    client: reqwest::Client,
}

impl Harness {
    /// Start a full router whose classifier is the Vertex engine wired to the
    /// mock `:predict` API. Anchor embedding happens here, at engine build.
    async fn start() -> Harness {
        let (embed_addr, embed) = spawn_mock_vertex().await;
        let (chat_addr, chat_calls) = spawn_mock_chat().await;

        let cfg = config::parse(&vertex_config_toml(embed_addr, chat_addr)).expect("parse config");
        cfg.validate().expect("validate config");

        let engine = engines::build(&cfg.classifier)
            .await
            .expect("build vertex engine against mock");
        let trivial_max_words = cfg.classifier.trivial_max_words;
        let state =
            AppState::new(engine, Arc::new(cfg), trivial_max_words).expect("build app state");
        let app = build_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        Harness {
            base: format!("http://{addr}"),
            chat_calls,
            embed,
            client: reqwest::Client::new(),
        }
    }

    async fn chat(&self, prompt: &str) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&json!({
                "model": ADVERTISED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }))
            .send()
            .await
            .expect("send chat request")
    }

    fn routed_model(&self) -> String {
        self.chat_calls
            .lock()
            .unwrap()
            .last()
            .expect("at least one upstream call")["model"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn embed_calls(&self) -> Vec<EmbedCall> {
        self.embed.calls.lock().unwrap().clone()
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

/// Startup embeds the class anchors in one batch, hitting the regional
/// `:predict` endpoint with the configured OAuth token as a Bearer header.
#[tokio::test]
async fn startup_embeds_anchors_with_bearer_token() {
    let h = Harness::start().await;
    let calls = h.embed_calls();
    assert_eq!(calls.len(), 1, "anchors must be one batched call");
    assert!(
        calls[0]
            .path
            .contains("publishers/google/models/text-embedding-005:predict"),
        "unexpected endpoint path: {}",
        calls[0].path
    );
    assert!(
        calls[0]
            .path
            .contains("/projects/test-project/locations/us-central1/"),
        "unexpected endpoint path: {}",
        calls[0].path
    );
    assert_eq!(calls[0].bearer.as_deref(), Some("test-vertex-token"));
    assert!(calls[0].text_count >= 12, "all anchor classes embedded");
}

/// Engine identity and the 2048-token-class budgets are model-specific.
#[tokio::test]
async fn engine_name_and_budgets() {
    let (embed_addr, _embed) = spawn_mock_vertex().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let cfg = config::parse(&vertex_config_toml(embed_addr, chat_addr)).expect("parse config");
    let engine = engines::build(&cfg.classifier).await.expect("build engine");
    assert_eq!(engine.name(), "text-embedding-005");
    assert_eq!(engine.context_char_budget(), 6_000);
    assert_eq!(engine.current_turn_char_budget(), 2_000);
}

/// Complexity routing through embedding prototypes: balanced-flavored text to
/// the balanced tier, frontier-flavored text to the frontier tier.
#[tokio::test]
async fn embedding_classification_routes_by_tier() {
    let h = Harness::start().await;

    let resp = h
        .chat("Explain how compound interest works over several years.")
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.routed_model(), "balanced-text");

    let resp = h
        .chat("Prove that the halting problem is undecidable with a rigorous argument.")
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.routed_model(), "frontier-text");
}

/// Image intent detected by the *embedding* signal alone (deliberately phrased
/// to evade the lexical prefilter) routes to the image backend.
#[tokio::test]
async fn embedding_image_signal_routes_to_image_backend() {
    let h = Harness::start().await;
    let resp = h
        .chat("A watercolor of a mountain sunrise would be lovely please.")
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.routed_model(), "image-gen");
}

/// Per-request API failures degrade to the balanced default (the request is
/// still served), exactly like any engine failure.
#[tokio::test]
async fn embed_failure_degrades_to_balanced_default() {
    let h = Harness::start().await;
    h.embed.fail.store(true, Ordering::SeqCst);
    let resp = h
        .chat("Prove that the halting problem is undecidable with a rigorous argument.")
        .await;
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "request still served"
    );
    assert_eq!(
        h.routed_model(),
        "balanced-text",
        "engine failure must fall back to the balanced default"
    );
}

/// The trivial fast-path stays engine-agnostic: pure chit-chat routes to Fast
/// without a single embedding call beyond the startup anchors.
#[tokio::test]
async fn chit_chat_routes_fast_with_zero_embed_calls() {
    let h = Harness::start().await;
    let anchors_only = h.embed_calls().len();
    let resp = h.chat("thanks, ok!").await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.routed_model(), "fast-text");
    assert_eq!(
        h.embed_calls().len(),
        anchors_only,
        "trivial turns must never reach the embeddings API"
    );
}

/// The GCP project is mandatory: a selected Vertex engine without one must
/// fail at startup with an actionable message (never limp along).
#[tokio::test]
async fn missing_project_fails_engine_build() {
    let (embed_addr, _embed) = spawn_mock_vertex().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml =
        vertex_config_toml(embed_addr, chat_addr).replace("project = \"test-project\"\n", "");
    let cfg = config::parse(&toml).expect("config parses without the project");
    let msg = match engines::build(&cfg.classifier).await {
        Ok(_) => panic!("engine build must fail without a project"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("requires a GCP project") && msg.contains("text-embedding-005"),
        "unhelpful error: {msg}"
    );
}

/// The access token is mandatory too.
#[tokio::test]
async fn missing_access_token_fails_engine_build() {
    let (embed_addr, _embed) = spawn_mock_vertex().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml = vertex_config_toml(embed_addr, chat_addr)
        .replace("access_token = \"test-vertex-token\"\n", "");
    let cfg = config::parse(&toml).expect("config parses without the token");
    let msg = match engines::build(&cfg.classifier).await {
        Ok(_) => panic!("engine build must fail without an access token"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("requires an OAuth access token") && msg.contains("text-embedding-005"),
        "unhelpful error: {msg}"
    );
}
