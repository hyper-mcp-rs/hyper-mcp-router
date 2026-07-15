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
/// `Authorization` header, the `x-goog-user-project` quota header, and how
/// many texts were embedded.
#[derive(Clone, Debug)]
struct EmbedCall {
    path: String,
    bearer: Option<String>,
    quota_project: Option<String>,
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
    // Two wire flavors share this mock: `:predict` batches texts in
    // `instances[].content`; `:embedContent` carries exactly one text in
    // `content.parts[0].text`.
    let texts: Vec<String> = if let Some(instances) = body["instances"].as_array() {
        instances
            .iter()
            .filter_map(|i| i["content"].as_str())
            .map(str::to_owned)
            .collect()
    } else {
        body["content"]["parts"][0]["text"]
            .as_str()
            .map(str::to_owned)
            .into_iter()
            .collect()
    };
    let is_embed_content = body.get("instances").is_none();

    state.calls.lock().unwrap().push(EmbedCall {
        path: uri.path().to_string(),
        bearer: headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(str::to_owned),
        quota_project: headers
            .get("x-goog-user-project")
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

    let payload = if is_embed_content {
        json!({"embedding": {"values": mock_embedding(&texts[0])}})
    } else {
        let predictions: Vec<Value> = texts
            .iter()
            .map(|t| json!({"embeddings": {"values": mock_embedding(t)}}))
            .collect();
        json!({"predictions": predictions})
    };
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
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
quota_project = "test-quota-project"
access_token = "test-vertex-token"
base_url = "http://{embed_addr}"

[[models]]
name = "fast-text"
base_url = "{chat}"
type = "fast"
modalities = ["text"]
context_window = 128000

[[models]]
name = "balanced-text"
base_url = "{chat}"
type = "balanced"
modalities = ["text"]
context_window = 128000

[[models]]
name = "frontier-text"
base_url = "{chat}"
type = "frontier"
modalities = ["text"]
context_window = 128000

[[models]]
name = "image-gen"
base_url = "{chat}"
type = "balanced"
modalities = ["text", "image-output"]
context_window = 128000
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

        let engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
            .await
            .expect("build vertex engine against mock");
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
    assert_eq!(
        calls[0].quota_project.as_deref(),
        Some("test-quota-project"),
        "configured quota project must ride along as x-goog-user-project"
    );
    assert!(calls[0].text_count >= 12, "all anchor classes embedded");
}

/// The quota header is strictly opt-in: without `quota_project` configured,
/// no `x-goog-user-project` header is sent.
#[tokio::test]
async fn quota_project_header_omitted_when_not_configured() {
    let (embed_addr, embed) = spawn_mock_vertex().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml = vertex_config_toml(embed_addr, chat_addr)
        .replace("quota_project = \"test-quota-project\"\n", "");
    let cfg = config::parse(&toml).expect("parse config");
    let _engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
        .await
        .expect("build engine");
    let calls = embed.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1, "anchor embedding call");
    assert_eq!(calls[0].quota_project, None);
}

/// Engine identity and the 2048-token-class budgets are model-specific.
#[tokio::test]
async fn engine_name_and_budgets() {
    let (embed_addr, _embed) = spawn_mock_vertex().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let cfg = config::parse(&vertex_config_toml(embed_addr, chat_addr)).expect("parse config");
    let engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
        .await
        .expect("build engine");
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

/// Config for a gemini-embedding model on the **Vertex** surface: `project`
/// instead of `api_key`, so the auth-driven dispatch picks the Vertex twin.
fn gemini_twin_config_toml(
    model: &str,
    location: &str,
    embed_addr: SocketAddr,
    chat_addr: SocketAddr,
) -> String {
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[classifier]
model = "{model}"

[classifier.{model}]
project = "test-project"
location = "{location}"
access_token = "test-vertex-token"
base_url = "http://{embed_addr}"

[[models]]
name = "m"
base_url = "http://{chat_addr}"
type = "fast"
modalities = ["text"]
context_window = 128000
"#
    )
}

/// Build a gemini-embedding twin against the mock and return (engine, calls).
async fn build_gemini_twin(
    model: &str,
    location: &str,
) -> (
    std::sync::Arc<dyn hyper_mcp_router::classifier::ClassifierEngine>,
    Vec<EmbedCall>,
) {
    let (embed_addr, embed) = spawn_mock_vertex().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml = gemini_twin_config_toml(model, location, embed_addr, chat_addr);
    let cfg = config::parse(&toml).expect("parse config");
    let engine = engines::build(cfg.classifier.models[0], &cfg.classifier)
        .await
        .expect("build engine");
    let calls = embed.calls.lock().unwrap().clone();
    (engine, calls)
}

/// gemini-embedding-001 with a Vertex-shaped table (`project` instead of
/// `api_key`) builds the SEPARATE Vertex engine — same model name as the
/// `gemini/` twin, but this endpoint layout, wire format, and auth.
#[tokio::test]
async fn gemini_embedding_001_on_vertex_uses_predict_endpoint() {
    let (engine, calls) = build_gemini_twin("gemini-embedding-001", "us-central1").await;
    assert_eq!(engine.name(), "gemini-embedding-001");
    assert_eq!(engine.context_char_budget(), 6_000);
    assert_eq!(engine.current_turn_char_budget(), 2_000);

    assert_eq!(calls.len(), 1, "anchor embedding call");
    assert!(
        calls[0]
            .path
            .contains("publishers/google/models/gemini-embedding-001:predict"),
        "unexpected endpoint path: {}",
        calls[0].path
    );
    assert_eq!(calls[0].bearer.as_deref(), Some("test-vertex-token"));
}

/// Shared assertions for gemini-embedding-2's Vertex twin at a location: the
/// model has no `:predict` or batch surface, so anchors must FAN OUT as
/// concurrent single-text `:embedContent` calls, each carrying auth.
async fn assert_gemini_2_fanout_at(location: &str) {
    let (engine, calls) = build_gemini_twin("gemini-embedding-2", location).await;
    assert_eq!(engine.name(), "gemini-embedding-2");
    assert_eq!(engine.context_char_budget(), 24_000);
    assert_eq!(engine.current_turn_char_budget(), 8_000);

    assert!(
        calls.len() >= 12,
        "anchors must fan out as one call per text, got {}",
        calls.len()
    );
    let expected_path =
        format!("/locations/{location}/publishers/google/models/gemini-embedding-2:embedContent");
    for call in &calls {
        assert!(
            call.path.contains(&expected_path),
            "unexpected endpoint path: {}",
            call.path
        );
        assert_eq!(
            call.text_count, 1,
            "embedContent carries exactly one text per request"
        );
        assert_eq!(call.bearer.as_deref(), Some("test-vertex-token"));
    }
}

/// gemini-embedding-2's Vertex twin at the `us` multi-region (the location
/// where the model is live-verified to be served).
#[tokio::test]
async fn gemini_embedding_2_on_vertex_us_fans_out_embed_content() {
    assert_gemini_2_fanout_at("us").await;
}

/// gemini-embedding-2's Vertex twin at `global` (also live-verified).
#[tokio::test]
async fn gemini_embedding_2_on_vertex_global_fans_out_embed_content() {
    assert_gemini_2_fanout_at("global").await;
}

/// The location is mandatory and deliberately has NO default: it determines
/// model availability, data residency, and the endpoint host, so a selected
/// Vertex engine without one must fail at startup naming the options.
#[tokio::test]
async fn missing_location_fails_engine_build() {
    let (embed_addr, _embed) = spawn_mock_vertex().await;
    let (chat_addr, _chat) = spawn_mock_chat().await;
    let toml =
        vertex_config_toml(embed_addr, chat_addr).replace("location = \"us-central1\"\n", "");
    let cfg = config::parse(&toml).expect("config parses without the location");
    let msg = match engines::build(cfg.classifier.models[0], &cfg.classifier).await {
        Ok(_) => panic!("engine build must fail without a location"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("requires a `location`")
            && msg.contains("us-central1")
            && msg.contains("global"),
        "unhelpful error: {msg}"
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
    let msg = match engines::build(cfg.classifier.models[0], &cfg.classifier).await {
        Ok(_) => panic!("engine build must fail without a project"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("requires a GCP project") && msg.contains("text-embedding-005"),
        "unhelpful error: {msg}"
    );
}

// NOTE: there is deliberately no `missing_access_token_fails_engine_build`
// test. An omitted `access_token` now means "authenticate via Application
// Default Credentials", and ADC discovery reads the *host environment*
// (GOOGLE_APPLICATION_CREDENTIALS, gcloud user credentials, metadata server),
// so any assertion about that path would be environment-dependent, not
// hermetic. The ADC path is covered by the opt-in live test
// (`tests/vertex_live.rs`); every mock test here pins a static token.
