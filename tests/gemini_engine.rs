//! Integration tests for the Gemini embedding engines.
//!
//! A mock Gemini API returns deterministic 4-dimensional "embeddings" derived
//! from keyword features (dims = [fast, balanced, frontier, image]), so the
//! engine's anchor prototypes collapse to near-unit class vectors and routing
//! through a real router instance is fully predictable — no network, no real
//! model. A mock chat backend records the forwarded `model` field as the
//! routing witness, exactly like `api_routing.rs`.

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
// Mock Gemini embeddings API
// ───────────────────────────────────────────────────────────────────────────

/// One recorded embed call: request path, the `x-goog-api-key` header, and
/// how many texts were embedded.
#[derive(Clone, Debug)]
struct EmbedCall {
    path: String,
    api_key: Option<String>,
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
    let texts: Vec<String> = body["requests"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|r| r["content"]["parts"][0]["text"].as_str())
        .map(str::to_owned)
        .collect();

    state.calls.lock().unwrap().push(EmbedCall {
        path: uri.path().to_string(),
        api_key: headers
            .get("x-goog-api-key")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned),
        text_count: texts.len(),
    });

    if state.fail.load(Ordering::SeqCst) {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(r#"{"error":{"message":"mock outage"}}"#))
            .unwrap();
    }

    let embeddings: Vec<Value> = texts
        .iter()
        .map(|t| json!({"values": mock_embedding(t)}))
        .collect();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"embeddings": embeddings})).unwrap(),
        ))
        .unwrap()
}

async fn spawn_mock_gemini() -> (SocketAddr, EmbedState) {
    let state = EmbedState::default();
    // Fallback route: matches any path, so the `:batchEmbedContents` URL
    // shape is recorded rather than hardcoded here.
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

fn gemini_config_toml(model: &str, embed_addr: SocketAddr, chat_addr: SocketAddr) -> String {
    let chat = format!("http://{chat_addr}");
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[classifier]
model = "{model}"

[classifier.{model}]
api_key = "test-gemini-key"
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
    /// Start a full router whose classifier is a Gemini engine wired to the
    /// mock embeddings API. Anchor embedding happens here, at engine build.
    async fn start(model: &str) -> Harness {
        let (embed_addr, embed) = spawn_mock_gemini().await;
        let (chat_addr, chat_calls) = spawn_mock_chat().await;

        let cfg =
            config::parse(&gemini_config_toml(model, embed_addr, chat_addr)).expect("parse config");
        cfg.validate().expect("validate config");

        let engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
            .await
            .expect("build gemini engine against mock");
        let trivial_max_words = cfg.classifier.trivial_max_words;
        let state = AppState::with_single_engine(engine, Arc::new(cfg), trivial_max_words)
            .expect("build app state");
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

/// Startup embeds the class anchors in one batch, hitting the model-specific
/// endpoint with the configured API key sent as a header (never in the URL).
#[tokio::test]
async fn startup_embeds_anchors_with_api_key_header() {
    let h = Harness::start("gemini-embedding-001").await;
    let calls = h.embed_calls();
    assert_eq!(calls.len(), 1, "anchors must be one batched call");
    assert!(
        calls[0]
            .path
            .contains("models/gemini-embedding-001:batchEmbedContents"),
        "unexpected endpoint path: {}",
        calls[0].path
    );
    assert_eq!(calls[0].api_key.as_deref(), Some("test-gemini-key"));
    assert!(calls[0].text_count >= 12, "all anchor classes embedded");
}

/// Complexity routing through embedding prototypes: balanced-flavored text to
/// the balanced tier, frontier-flavored text to the frontier tier.
#[tokio::test]
async fn embedding_classification_routes_by_tier() {
    let h = Harness::start("gemini-embedding-001").await;

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
    let h = Harness::start("gemini-embedding-001").await;
    // No lexical verb+noun pair; the mock scores "watercolor" on the image dim.
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
    let h = Harness::start("gemini-embedding-001").await;
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
    let h = Harness::start("gemini-embedding-001").await;
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

/// gemini-embedding-2 is a distinct engine: its own endpoint path and its
/// larger, model-specific context budgets.
#[tokio::test]
async fn gemini_embedding_2_uses_own_endpoint_and_budgets() {
    let h = Harness::start("gemini-embedding-2").await;
    let calls = h.embed_calls();
    assert!(
        calls[0]
            .path
            .contains("models/gemini-embedding-2:batchEmbedContents"),
        "unexpected endpoint path: {}",
        calls[0].path
    );

    // Budgets are exposed via the trait; check through a directly-built engine.
    let (embed_addr, _embed) = spawn_mock_gemini().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let cfg = config::parse(&gemini_config_toml(
        "gemini-embedding-2",
        embed_addr,
        chat_addr,
    ))
    .expect("parse config");
    let engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
        .await
        .expect("build engine");
    assert_eq!(engine.name(), "gemini-embedding-2");
    assert_eq!(engine.context_char_budget(), 24_000);
    assert_eq!(engine.current_turn_char_budget(), 8_000);

    // And it still routes.
    let resp = h.chat("Explain how compound interest works today.").await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.routed_model(), "balanced-text");
}

/// Credentials are mandatory and they pick the API surface: a selected
/// gemini-embedding engine with neither `api_key` (Generative Language) nor
/// `project` (Vertex) must fail at startup naming both options.
#[tokio::test]
async fn missing_credentials_fail_engine_build_naming_both_surfaces() {
    let (embed_addr, _embed) = spawn_mock_gemini().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml = gemini_config_toml("gemini-embedding-001", embed_addr, chat_addr)
        .replace("api_key = \"test-gemini-key\"\n", "");
    let cfg = config::parse(&toml).expect("config parses without the key");
    let msg = match engines::build(cfg.classifier.models[0], &cfg.classifier).await {
        Ok(_) => panic!("engine build must fail without credentials"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("requires either `api_key`")
            && msg.contains("`project`")
            && msg.contains("gemini-embedding-001"),
        "unhelpful error: {msg}"
    );
}

/// Setting BOTH `api_key` and `project` is ambiguous — the surface must be
/// chosen explicitly, never guessed.
#[tokio::test]
async fn both_surfaces_configured_fails_engine_build() {
    let (embed_addr, _embed) = spawn_mock_gemini().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml = gemini_config_toml("gemini-embedding-001", embed_addr, chat_addr).replace(
        "api_key = \"test-gemini-key\"\n",
        "api_key = \"test-gemini-key\"\nproject = \"some-project\"\n",
    );
    let cfg = config::parse(&toml).expect("config parses with both fields");
    let msg = match engines::build(cfg.classifier.models[0], &cfg.classifier).await {
        Ok(_) => panic!("engine build must fail when both surfaces are configured"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("sets both `api_key`") && msg.contains("exactly one"),
        "unhelpful error: {msg}"
    );
}
