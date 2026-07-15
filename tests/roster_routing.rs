//! Multi-engine capacity-ladder integration tests.
//!
//! A live router is configured with an [`EngineRoster`] of **scripted**
//! classifier engines (fixed budgets, fixed verdicts, call recording) and one
//! recording mock backend per complexity tier. Which backend receives the
//! forwarded request — the router rewrites `body["model"]` — is a precise
//! witness of which engine classified: each scripted engine returns a
//! different tier. The real classifier is exercised by `api_routing.rs`;
//! *these* tests are about ladder selection, so the engines are mocks.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::StatusCode,
    response::Response,
    routing::post,
    Router,
};
use serde_json::{json, Value};

use hyper_mcp_router::classifier::{Classification, ClassifierEngine, EngineRoster, ModelTier};
use hyper_mcp_router::config;
use hyper_mcp_router::prompt::DEFAULT_TRIVIAL_MAX_WORDS;
use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};

// ───────────────────────────────────────────────────────────────────────────
// Scripted classifier engine
// ───────────────────────────────────────────────────────────────────────────

/// Each entry is one classify call: (complexity window, image premise).
type ClassifyLog = Arc<Mutex<Vec<(String, String)>>>;

/// A scripted engine for capacity-ladder tests: fixed budgets, a fixed
/// complexity verdict, and a record of every (window, image-premise) pair it
/// was asked to classify. The verdict doubles as a witness of *which* engine
/// classified — the routed backend's tier names the engine that ran.
struct ScriptedClassifier {
    name: &'static str,
    window_budget: usize,
    turn_budget: usize,
    verdict: ModelTier,
    calls: ClassifyLog,
}

impl ScriptedClassifier {
    fn new(
        name: &'static str,
        window_budget: usize,
        turn_budget: usize,
        verdict: ModelTier,
    ) -> (Arc<Self>, ClassifyLog) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = Arc::new(ScriptedClassifier {
            name,
            window_budget,
            turn_budget,
            verdict,
            calls: calls.clone(),
        });
        (engine, calls)
    }
}

#[async_trait::async_trait]
impl ClassifierEngine for ScriptedClassifier {
    fn name(&self) -> &'static str {
        self.name
    }
    fn is_local(&self) -> bool {
        true
    }
    fn context_char_budget(&self) -> usize {
        self.window_budget
    }
    fn current_turn_char_budget(&self) -> usize {
        self.turn_budget
    }
    async fn classify(
        &self,
        complexity_premise: &str,
        image_premise: &str,
        _lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        self.calls
            .lock()
            .unwrap()
            .push((complexity_premise.to_string(), image_premise.to_string()));
        Ok(Classification {
            complexity: self.verdict,
            image_generation: false,
        })
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Mock backend + harness (compact; see api_routing.rs for the full-size one)
// ───────────────────────────────────────────────────────────────────────────

type Calls = Arc<Mutex<Vec<Value>>>;

async fn mock_chat(State(calls): State<Calls>, body: Bytes) -> Response {
    let forwarded: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    calls.lock().unwrap().push(forwarded);
    let resp = json!({
        "id": "mock-cmpl",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&resp).unwrap()))
        .unwrap()
}

/// One text backend per complexity tier, all pointing at one recording mock.
fn tiered_config_toml(addr: SocketAddr) -> String {
    let base = format!("http://{addr}");
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[[models]]
name = "fast-text"
base_url = "{base}"
type = "fast"
modalities = ["text"]

[[models]]
name = "balanced-text"
base_url = "{base}"
type = "balanced"
modalities = ["text"]

[[models]]
name = "frontier-text"
base_url = "{base}"
type = "frontier"
modalities = ["text"]
"#
    )
}

struct Harness {
    base: String,
    calls: Calls,
    client: reqwest::Client,
}

impl Harness {
    async fn start(roster: EngineRoster) -> Harness {
        let calls: Calls = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/chat/completions", post(mock_chat))
            .with_state(calls.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock backend");
        let mock_addr = listener.local_addr().expect("mock backend addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("mock serve");
        });

        // Build the config through the real parse + validate path.
        let cfg = config::parse(&tiered_config_toml(mock_addr)).expect("parse config");
        cfg.validate().expect("validate config");
        let state = AppState::new(roster, Arc::new(cfg), DEFAULT_TRIVIAL_MAX_WORDS)
            .expect("build app state");
        let app = build_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind router");
        let addr = listener.local_addr().expect("router addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("router serve");
        });

        Harness {
            base: format!("http://{addr}"),
            calls,
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

    fn last_call(&self) -> Value {
        self.calls
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("at least one recorded upstream call")
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

/// The capacity ladder routes each request to the smallest engine whose
/// window budget covers the classification window; only a window exceeding
/// even the top budget is truncated (at the top budget).
#[tokio::test]
async fn roster_selects_engine_by_window_size() {
    let (small, small_calls) = ScriptedClassifier::new("small", 120, 100, ModelTier::Fast);
    let (big, big_calls) = ScriptedClassifier::new("big", 2000, 1500, ModelTier::Frontier);
    let roster = EngineRoster::new(vec![small, big]).expect("valid ladder");
    let h = Harness::start(roster).await;

    // Short (but non-trivial) prompt: fits the small engine's 120-char budget.
    let short = "Compare quicksort with mergesort on nearly sorted input arrays.";
    assert!(short.len() <= 120 && short.split_whitespace().count() > 6);
    let resp = h.chat(short).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        h.last_call()["model"],
        "fast-text",
        "the small engine's verdict (fast) must have routed this"
    );
    assert_eq!(small_calls.lock().unwrap().len(), 1);
    assert_eq!(big_calls.lock().unwrap().len(), 0);

    // Long prompt: exceeds 120 chars, fits the big engine's 2000.
    let long = "Prove the master theorem case boundaries rigorously. ".repeat(10);
    assert!(long.len() > 120 && long.len() <= 2000);
    let resp = h.chat(&long).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        h.last_call()["model"],
        "frontier-text",
        "the big engine's verdict (frontier) must have routed this"
    );
    assert_eq!(
        small_calls.lock().unwrap().len(),
        1,
        "small engine must not run again"
    );
    assert_eq!(big_calls.lock().unwrap().len(), 1);

    // Huge prompt: exceeds even the top budget — classified by the top
    // engine, window truncated at ITS budget (never rejected).
    let huge = "Derive the amortized complexity of a Fibonacci heap in detail. ".repeat(100);
    assert!(huge.len() > 2000);
    let resp = h.chat(&huge).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "frontier-text");
    let big_seen = big_calls.lock().unwrap();
    assert_eq!(big_seen.len(), 2);
    let (window, premise) = &big_seen[1];
    assert!(
        window.chars().count() <= 2000,
        "window must be cut off at the top engine's budget, got {}",
        window.chars().count()
    );
    // The current turn (image premise) is truncated to the SELECTED engine's
    // own turn budget.
    assert!(
        premise.chars().count() <= 1500,
        "image premise must respect the selected engine's turn budget, got {}",
        premise.chars().count()
    );
    // The full request body is still forwarded untruncated.
    assert_eq!(
        h.last_call()["messages"][0]["content"].as_str().unwrap(),
        huge,
        "classification budgets must never truncate the forwarded request"
    );
}

/// A duplicate window budget is a startup error — the ladder needs a total
/// order. (Config validation already rejects listing the same MODEL twice;
/// this guards the budget collision between two different engines.)
#[test]
fn roster_with_duplicate_budgets_fails_startup() {
    let (a, _) = ScriptedClassifier::new("a", 1000, 400, ModelTier::Fast);
    let (b, _) = ScriptedClassifier::new("b", 1000, 400, ModelTier::Frontier);
    let err = EngineRoster::new(vec![a, b]).unwrap_err();
    assert!(
        err.to_string().contains("same context_char_budget"),
        "got: {err}"
    );
}
