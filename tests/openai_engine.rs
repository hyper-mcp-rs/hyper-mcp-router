//! Integration tests for the OpenAI embedding engines.
//!
//! A mock `/v1/embeddings` API returns deterministic 4-dimensional
//! "embeddings" derived from keyword features (dims = [fast, balanced,
//! frontier, image]) — deliberately in *reverse index order* to exercise the
//! index-keyed response parsing — so routing through a real router instance
//! is fully predictable. A mock chat backend records the forwarded `model`
//! field as the routing witness, exactly like `gemini_engine.rs`.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};

use hyper_mcp_router::config;
use hyper_mcp_router::engines;
use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};

// ───────────────────────────────────────────────────────────────────────────
// Mock OpenAI embeddings API
// ───────────────────────────────────────────────────────────────────────────

/// One recorded embed call: the requested model, the `Authorization` header,
/// and how many inputs were embedded.
#[derive(Clone, Debug)]
struct EmbedCall {
    model: String,
    authorization: Option<String>,
    input_count: usize,
}

type EmbedCalls = Arc<Mutex<Vec<EmbedCall>>>;

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
        v[1] = 0.1;
    }
    v
}

async fn mock_embeddings(
    State(calls): State<EmbedCalls>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let inputs: Vec<String> = body["input"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|t| t.as_str())
        .map(str::to_owned)
        .collect();

    calls.lock().unwrap().push(EmbedCall {
        model: body["model"].as_str().unwrap_or_default().to_string(),
        authorization: headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned),
        input_count: inputs.len(),
    });

    // Return items in REVERSE index order: correct parsing must key by
    // `index`, not array position.
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .rev()
        .map(|(i, t)| json!({"object": "embedding", "index": i, "embedding": mock_embedding(t)}))
        .collect();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"object": "list", "data": data})).unwrap(),
        ))
        .unwrap()
}

async fn spawn_mock_openai() -> (SocketAddr, EmbedCalls) {
    let calls: EmbedCalls = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(mock_embeddings))
        .with_state(calls.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, calls)
}

// ───────────────────────────────────────────────────────────────────────────
// Mock chat backend (routing witness) + harness
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

fn openai_config_toml(model: &str, embed_addr: SocketAddr, chat_addr: SocketAddr) -> String {
    let chat = format!("http://{chat_addr}");
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[classifier]
model = "{model}"

[classifier.{model}]
api_key = "sk-test-openai"
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
    embed_calls: EmbedCalls,
    client: reqwest::Client,
}

impl Harness {
    async fn start(model: &str) -> Harness {
        let (embed_addr, embed_calls) = spawn_mock_openai().await;
        let (chat_addr, chat_calls) = spawn_mock_chat().await;

        let cfg =
            config::parse(&openai_config_toml(model, embed_addr, chat_addr)).expect("parse config");
        cfg.validate().expect("validate config");

        let engine = engines::build(&cfg.classifier)
            .await
            .expect("build openai engine against mock");
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
            embed_calls,
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
        self.embed_calls.lock().unwrap().clone()
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

/// Startup embeds the class anchors in one batched array-input call with the
/// key as a bearer token, naming the model in the body.
#[tokio::test]
async fn startup_embeds_anchors_with_bearer_auth() {
    let h = Harness::start("text-embedding-3-small").await;
    let calls = h.embed_calls();
    assert_eq!(calls.len(), 1, "anchors must be one batched call");
    assert_eq!(calls[0].model, "text-embedding-3-small");
    assert_eq!(
        calls[0].authorization.as_deref(),
        Some("Bearer sk-test-openai")
    );
    assert!(calls[0].input_count >= 12, "all anchor classes embedded");
}

/// End-to-end routing through the OpenAI engine — with the mock returning
/// embeddings in reverse index order, so this also proves the index-keyed
/// response parsing assigns premises correctly.
#[tokio::test]
async fn openai_classification_routes_by_tier_despite_permuted_responses() {
    let h = Harness::start("text-embedding-3-small").await;

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

/// Image intent via the embedding signal alone (lexical-evading phrasing).
#[tokio::test]
async fn openai_embedding_image_signal_routes_to_image_backend() {
    let h = Harness::start("text-embedding-3-small").await;
    let resp = h
        .chat("A watercolor of a mountain sunrise would be lovely please.")
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.routed_model(), "image-gen");
}

/// text-embedding-3-large is a distinct engine with its own model id and the
/// same large budgets.
#[tokio::test]
async fn text_embedding_3_large_uses_own_model_id_and_budgets() {
    let h = Harness::start("text-embedding-3-large").await;
    assert_eq!(h.embed_calls()[0].model, "text-embedding-3-large");

    let (embed_addr, _embed) = spawn_mock_openai().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let cfg = config::parse(&openai_config_toml(
        "text-embedding-3-large",
        embed_addr,
        chat_addr,
    ))
    .expect("parse config");
    let engine = engines::build(&cfg.classifier).await.expect("build engine");
    assert_eq!(engine.name(), "text-embedding-3-large");
    assert_eq!(engine.context_char_budget(), 24_000);
    assert_eq!(engine.current_turn_char_budget(), 8_000);

    let resp = h.chat("Explain how compound interest works today.").await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.routed_model(), "balanced-text");
}

/// The API key is mandatory: a selected OpenAI engine without one must fail
/// at startup with an actionable message.
#[tokio::test]
async fn missing_api_key_fails_engine_build() {
    let (embed_addr, _embed) = spawn_mock_openai().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml = openai_config_toml("text-embedding-3-small", embed_addr, chat_addr)
        .replace("api_key = \"sk-test-openai\"\n", "");
    let cfg = config::parse(&toml).expect("config parses without the key");
    let msg = match engines::build(&cfg.classifier).await {
        Ok(_) => panic!("engine build must fail without an API key"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("requires an API key") && msg.contains("text-embedding-3-small"),
        "unhelpful error: {msg}"
    );
}
