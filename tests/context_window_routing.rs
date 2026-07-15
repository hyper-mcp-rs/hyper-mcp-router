//! Context-window routing integration tests.
//!
//! A live router is configured with backends whose (required) `context_window`
//! declarations differ, plus a **scripted** classifier with a fixed verdict.
//! Which backend receives the forwarded request — the router rewrites
//! `body["model"]` — witnesses the capacity decision: a request that cannot
//! fit a "fast" model's small window must never be sent there, whatever the
//! complexity verdict says. The classifier call log also witnesses the
//! classification-skip optimisation: with a single *fitting* candidate there
//! is nothing to rank, so no inference runs.

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

use hyper_mcp_router::classifier::{Classification, ClassifierEngine, ModelTier};
use hyper_mcp_router::config;
use hyper_mcp_router::prompt::DEFAULT_TRIVIAL_MAX_WORDS;
use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};

// ───────────────────────────────────────────────────────────────────────────
// Scripted classifier: fixed verdict, records how often it ran.
// ───────────────────────────────────────────────────────────────────────────

struct FixedVerdict {
    verdict: ModelTier,
    calls: Arc<Mutex<usize>>,
}

impl FixedVerdict {
    fn new(verdict: ModelTier) -> (Arc<Self>, Arc<Mutex<usize>>) {
        let calls = Arc::new(Mutex::new(0));
        let engine = Arc::new(FixedVerdict {
            verdict,
            calls: calls.clone(),
        });
        (engine, calls)
    }
}

#[async_trait::async_trait]
impl ClassifierEngine for FixedVerdict {
    fn name(&self) -> &'static str {
        "fixed-verdict"
    }
    fn is_local(&self) -> bool {
        true
    }
    fn context_char_budget(&self) -> usize {
        100_000
    }
    fn current_turn_char_budget(&self) -> usize {
        100_000
    }
    async fn classify(
        &self,
        _complexity_premise: &str,
        _image_premise: &str,
        _lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        *self.calls.lock().unwrap() += 1;
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

/// A small-window fast model and a large-window frontier model. Window sizes
/// are in tokens; the router estimates ~4 chars per token, so the fast
/// model's 1,000-token window fits ~4,000 chars of message text.
fn windowed_config_toml(addr: SocketAddr) -> String {
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
context_window = 1000

[[models]]
name = "frontier-text"
base_url = "{base}"
type = "frontier"
modalities = ["text"]
context_window = 1000000
"#
    )
}

/// EVERY window is too small for a big request: fast is the tiniest,
/// balanced the largest. Exercises the best-effort fallback.
fn all_small_config_toml(addr: SocketAddr) -> String {
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
context_window = 500

[[models]]
name = "balanced-text"
base_url = "{base}"
type = "balanced"
modalities = ["text"]
context_window = 1000
"#
    )
}

struct Harness {
    base: String,
    calls: Calls,
    client: reqwest::Client,
}

impl Harness {
    async fn start(
        classifier: Arc<dyn ClassifierEngine>,
        config_of: impl Fn(SocketAddr) -> String,
    ) -> Harness {
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
        let cfg = config::parse(&config_of(mock_addr)).expect("parse config");
        cfg.validate().expect("validate config");
        let state =
            AppState::with_single_engine(classifier, Arc::new(cfg), DEFAULT_TRIVIAL_MAX_WORDS)
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

    async fn chat(&self, body: &Value) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/chat/completions", self.base))
            .json(body)
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

fn user_body(prompt: &str) -> Value {
    json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    })
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

/// The complexity verdict never overrides capacity: a request that cannot fit
/// the fast model's window routes to the large-window backend even though the
/// classifier says "fast" — and, with a single fitting candidate, no
/// classification runs at all.
#[tokio::test]
async fn oversized_request_escalates_past_the_small_window() {
    let (engine, classify_calls) = FixedVerdict::new(ModelTier::Fast);
    let h = Harness::start(engine, windowed_config_toml).await;

    // Small prompt: both windows fit, the verdict (fast) decides.
    let small = "Compare quicksort with mergesort on nearly sorted input arrays.";
    let resp = h.chat(&user_body(small)).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "fast-text");
    assert_eq!(*classify_calls.lock().unwrap(), 1);

    // Huge prompt: ~8,000 chars ≈ 2,000 tokens overflows the fast model's
    // 1,000-token window. Only the frontier backend fits, so the router
    // routes there directly — skipping classification (still 1 call).
    let huge = "Summarize the following server log line by line please. ".repeat(140);
    assert!(huge.chars().count() > 4 * 1000);
    let resp = h.chat(&user_body(&huge)).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        h.last_call()["model"],
        "frontier-text",
        "a request the fast window cannot hold must escalate"
    );
    assert_eq!(
        *classify_calls.lock().unwrap(),
        1,
        "one fitting candidate leaves nothing to rank; classification must be skipped"
    );
    // The forwarded body is never truncated to make a request "fit".
    assert_eq!(h.last_call()["messages"][0]["content"], huge.as_str());
}

/// The requested completion budget counts against the window: a tiny prompt
/// with a huge `max_tokens` cannot go to the small-window model either.
#[tokio::test]
async fn max_tokens_counts_toward_the_context_estimate() {
    let (engine, classify_calls) = FixedVerdict::new(ModelTier::Fast);
    let h = Harness::start(engine, windowed_config_toml).await;

    let mut body = user_body("Compare quicksort with mergesort on nearly sorted input arrays.");
    body["max_tokens"] = json!(500_000);
    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        h.last_call()["model"],
        "frontier-text",
        "prompt + max_tokens must fit the window, not just the prompt"
    );
    assert_eq!(*classify_calls.lock().unwrap(), 0);
}

/// When NO window fits, the request is still forwarded — to the
/// largest-window candidate — because the size estimate is a heuristic and
/// the upstream is the authority. Never a local rejection.
#[tokio::test]
async fn nothing_fits_falls_back_to_the_largest_window() {
    let (engine, _) = FixedVerdict::new(ModelTier::Fast);
    let h = Harness::start(engine, all_small_config_toml).await;

    let huge = "Summarize the following server log line by line please. ".repeat(200);
    let resp = h.chat(&user_body(&huge)).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        h.last_call()["model"],
        "balanced-text",
        "best-effort fallback must pick the most capacious backend"
    );
}
