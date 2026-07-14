//! API-level integration tests.
//!
//! These exercise the whole router over real HTTP: a client sends OpenAI
//! Chat Completions requests to a live `hyper-mcp-router` instance, which
//! classifies each prompt with the real embedded NLI model and forwards it to a
//! mocked backend. Every backend points at one in-process mock server that
//! *records* the forwarded request. Because the router rewrites `body["model"]`
//! to the selected backend's configured name, the recorded `model` field is a
//! precise witness of the routing decision — that is how we "track the actual
//! calls".
//!
//! The classifier is intentionally **not** mocked: it is the router's core
//! logic. Only the upstream models are mocked. The embedded ONNX model is loaded
//! once and shared across every test via `CLASSIFIER`.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::StatusCode,
    response::Response,
    routing::post,
    Router,
};
use serde_json::{json, Value};

use hyper_mcp_router::classifier::{ClassifierEngine, DEFAULT_IMAGE_GEN_THRESHOLD};
use hyper_mcp_router::config;
use hyper_mcp_router::engines::zero_shot::ZeroShot;
use hyper_mcp_router::prompt::DEFAULT_TRIVIAL_MAX_WORDS;
use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};

// ───────────────────────────────────────────────────────────────────────────
// Shared classifier (embedded ONNX model) — loaded once for the whole binary.
// ───────────────────────────────────────────────────────────────────────────

static CLASSIFIER: LazyLock<Arc<ZeroShot>> = LazyLock::new(|| {
    // Pool of 2 (default ORT intra-op threads) so the pooling path is exercised
    // by the correctness suite without a heavy N-session startup cost.
    Arc::new(ZeroShot::new(DEFAULT_IMAGE_GEN_THRESHOLD, 2, 0).expect("load embedded classifier"))
});

/// Every backend name declared by [`mock_config_toml`].
const BACKENDS: [&str; 8] = [
    "fast-text",
    "balanced-text",
    "frontier-text",
    "vision",
    "audio",
    "files",
    "image-gen",
    "agent",
];

/// The text-only tier backends, one per complexity type.
const TEXT_BACKENDS: [&str; 3] = ["fast-text", "balanced-text", "frontier-text"];

// ───────────────────────────────────────────────────────────────────────────
// Mock backend: records every forwarded request, returns a trivial completion.
// ───────────────────────────────────────────────────────────────────────────

/// A single recorded upstream call: the forwarded body plus the `Authorization`
/// header value the router sent (if any).
#[derive(Clone)]
struct Recorded {
    body: Value,
    authorization: Option<String>,
}

/// Thread-safe log of every request the router forwarded upstream.
type Calls = Arc<Mutex<Vec<Recorded>>>;

async fn spawn_mock_backend() -> (SocketAddr, Calls) {
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/chat/completions", post(mock_chat))
        // The router accepts large multimodal bodies; the mock must too.
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(calls.clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock backend");
    let addr = listener.local_addr().expect("mock backend addr");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("mock backend serve");
    });
    (addr, calls)
}

async fn mock_chat(
    State(calls): State<Calls>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let forwarded: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let authorization = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let streaming = forwarded
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    calls.lock().unwrap().push(Recorded {
        body: forwarded,
        authorization,
    });

    if streaming {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("x-mock-request-id", "mock-123")
            .body(Body::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
            ))
            .unwrap()
    } else {
        let resp = json!({
            "id": "mock-cmpl",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        });
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .header("x-mock-request-id", "mock-123")
            .body(Body::from(serde_json::to_vec(&resp).unwrap()))
            .unwrap()
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Router harness: a live router wired to the mock backend.
// ───────────────────────────────────────────────────────────────────────────

struct Harness {
    base: String,
    calls: Calls,
    client: reqwest::Client,
}

impl Harness {
    async fn start() -> Harness {
        Harness::start_with_classifier(CLASSIFIER.clone()).await
    }

    async fn start_with_classifier(classifier: Arc<dyn ClassifierEngine>) -> Harness {
        Harness::start_with(classifier, mock_config_toml).await
    }

    async fn start_with(
        classifier: Arc<dyn ClassifierEngine>,
        config_of: impl Fn(SocketAddr) -> String,
    ) -> Harness {
        let (mock_addr, calls) = spawn_mock_backend().await;

        // Build the config through the real parse + validate path so coverage
        // validation is exercised too.
        let cfg = config::parse(&config_of(mock_addr)).expect("parse config");
        cfg.validate().expect("validate config");

        let state = AppState::new(classifier, Arc::new(cfg), DEFAULT_TRIVIAL_MAX_WORDS)
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

    async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}{path}", self.base))
            .send()
            .await
            .expect("send get request")
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    fn last_call(&self) -> Value {
        self.calls
            .lock()
            .unwrap()
            .last()
            .map(|r| r.body.clone())
            .expect("at least one recorded upstream call")
    }

    /// The `Authorization` header the router sent on the last forwarded call, if
    /// any. `None` means no header was sent (keyless backend).
    fn last_authorization(&self) -> Option<String> {
        self.calls
            .lock()
            .unwrap()
            .last()
            .expect("at least one recorded upstream call")
            .authorization
            .clone()
    }
}

/// A minimal keyless catalogue (no `api_key` on any model) that still satisfies
/// startup coverage validation: text for every tier, all extra modalities on
/// the frontier model. Used to verify that keyless backends get no auth header.
fn keyless_config_toml(addr: SocketAddr) -> String {
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
name = "frontier-all"
base_url = "{base}"
type = "frontier"
modalities = ["text", "image-input", "audio-input", "file-input", "audio-output", "image-output", "tools"]
"#
    )
}

/// A single model covering every modality. Every request then has exactly one
/// candidate, so the router always skips classification (see
/// `single_model_deployment_routes_everything`).
fn single_model_config_toml(addr: SocketAddr) -> String {
    let base = format!("http://{addr}");
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[[models]]
name = "only"
base_url = "{base}"
api_key = "test-key"
type = "balanced"
modalities = ["text", "image-input", "audio-input", "file-input", "audio-output", "image-output", "tools"]
"#
    )
}

/// Text-only tiers: no image-output (or any other extra modality) anywhere.
/// Used to verify that *inferred* image intent degrades instead of 422-ing.
fn text_only_config_toml(addr: SocketAddr) -> String {
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

/// A full-coverage catalogue. Every backend points at the same mock server, so
/// the recorded `model` field alone identifies the routing decision. Names
/// encode (tier, modality) for legible assertions.
fn mock_config_toml(addr: SocketAddr) -> String {
    let base = format!("http://{addr}");
    format!(
        r#"
[server]
host = "127.0.0.1"
port = 0

[[models]]
name = "fast-text"
base_url = "{base}"
api_key = "test-key"
type = "fast"
modalities = ["text"]

[[models]]
name = "balanced-text"
base_url = "{base}"
api_key = "test-key"
type = "balanced"
modalities = ["text"]

[[models]]
name = "frontier-text"
base_url = "{base}"
api_key = "test-key"
type = "frontier"
modalities = ["text"]

[[models]]
name = "vision"
base_url = "{base}"
api_key = "test-key"
type = "balanced"
modalities = ["text", "image-input"]

[[models]]
name = "audio"
base_url = "{base}"
api_key = "test-key"
type = "balanced"
modalities = ["text", "audio-input", "audio-output"]

[[models]]
name = "files"
base_url = "{base}"
api_key = "test-key"
type = "balanced"
modalities = ["text", "file-input"]

[[models]]
name = "image-gen"
base_url = "{base}"
api_key = "test-key"
type = "balanced"
modalities = ["text", "image-output"]

[[models]]
name = "agent"
base_url = "{base}"
api_key = "test-key"
type = "balanced"
modalities = ["text", "tools"]
"#
    )
}

fn text_request(prompt: &str) -> Value {
    json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    })
}

// ───────────────────────────────────────────────────────────────────────────
// The 100+ prompt corpus.
// ───────────────────────────────────────────────────────────────────────────

/// A deliberately diverse corpus spanning trivial chatter, factual lookups,
/// math, code, explanations, deep analysis, creative writing, and planning.
fn corpus() -> Vec<&'static str> {
    vec![
        // ── trivial / chit-chat ──
        "hi",
        "hello there",
        "thanks!",
        "ok",
        "yes",
        "good morning",
        "how are you?",
        "what's up",
        "cool, got it",
        "bye",
        // ── short factual ──
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "How many continents are there?",
        "What year did World War II end?",
        "What is the boiling point of water in Celsius?",
        "Name the largest planet in our solar system.",
        "What is the chemical symbol for gold?",
        "How many days are in a leap year?",
        "What language is spoken in Brazil?",
        "What is the tallest mountain on Earth?",
        // ── arithmetic / math ──
        "What is 2 + 2?",
        "Calculate 15% of 240.",
        "What is the square root of 144?",
        "Convert 100 kilometers to miles.",
        "What is 7 factorial?",
        "Solve for x: 3x + 5 = 20.",
        "What is the derivative of x^2?",
        "Integrate sin(x) with respect to x.",
        "Find the greatest common divisor of 48 and 180.",
        "Prove that the square root of 2 is irrational.",
        // ── coding ──
        "Write a function to reverse a string in Python.",
        "How do I center a div with CSS?",
        "Explain the difference between let and const in JavaScript.",
        "Write a SQL query to find duplicate rows in a table.",
        "What is a race condition and how do I avoid it?",
        "Implement binary search in Rust.",
        "How does garbage collection work in the JVM?",
        "Refactor this nested loop into something more idiomatic.",
        "Explain the borrow checker in Rust to a beginner.",
        "Design a rate limiter for a REST API.",
        "Write unit tests for a function that parses ISO 8601 dates.",
        "What is the time complexity of quicksort in the worst case?",
        "How would you shard a Postgres database for horizontal scaling?",
        "Debug a segfault in a C program that uses malloc and free.",
        "Explain how TLS handshakes establish a secure connection.",
        // ── moderate explanations ──
        "Explain how a bill becomes a law in the United States.",
        "What causes the seasons to change?",
        "Summarize the plot of Romeo and Juliet.",
        "Explain the difference between weather and climate.",
        "How does a vaccine train the immune system?",
        "What is compound interest and how is it calculated?",
        "Describe how photosynthesis works.",
        "Explain the water cycle to a ten year old.",
        "What is the greenhouse effect?",
        "How do noise-cancelling headphones work?",
        "What is the difference between RAM and an SSD?",
        "Explain how GPS determines your location.",
        "How does inflation affect purchasing power?",
        "What is the placebo effect?",
        "Explain the theory of supply and demand.",
        // ── deep reasoning / analysis ──
        "Derive and rigorously prove the asymptotic time complexity of red-black tree rebalancing across a sequence of insertions and deletions, with a formal amortized analysis.",
        "Critically evaluate the philosophical arguments for and against compatibilism regarding free will.",
        "Analyze the macroeconomic tradeoffs of quantitative easing versus fiscal stimulus during a liquidity trap.",
        "Compare and contrast the epistemological foundations of Bayesian and frequentist statistics.",
        "Design a distributed consensus protocol tolerant to Byzantine faults and prove its safety and liveness properties.",
        "Assess the long-term geopolitical consequences of large-scale desalination on water-scarce regions.",
        "Provide a rigorous derivation of the Black-Scholes option pricing formula from first principles.",
        "Evaluate the ethical implications of deploying autonomous lethal weapons under international humanitarian law.",
        "Explain the measurement problem in quantum mechanics and compare the major interpretations.",
        "Analyze how transformer attention mechanisms scale and propose techniques to reduce their quadratic cost.",
        "Construct a formal proof of the halting problem's undecidability using diagonalization.",
        "Discuss the tradeoffs between CAP theorem guarantees in a globally distributed database.",
        "Synthesize the evidence for and against the efficient market hypothesis across market regimes.",
        "Derive the Euler-Lagrange equation and apply it to the brachistochrone problem.",
        "Critically analyze the reproducibility crisis in the social sciences and propose methodological reforms.",
        // ── creative writing ──
        "Write a haiku about autumn rain.",
        "Compose a short bedtime story about a brave little lighthouse.",
        "Write a limerick about a cat who loves jazz.",
        "Draft a heartfelt thank-you note to a mentor.",
        "Write the opening paragraph of a noir detective novel.",
        "Compose a motivational speech for a losing sports team at halftime.",
        "Write a product description for an eco-friendly water bottle.",
        "Invent a myth explaining why the moon changes shape.",
        "Write a dialogue between the sun and the sea.",
        "Compose a four-line poem about the passage of time.",
        // ── planning / multi-step ──
        "Plan a seven-day itinerary for a first trip to Japan.",
        "Create a weekly meal plan for a vegetarian on a budget.",
        "Outline a study schedule to prepare for a calculus final in two weeks.",
        "Draft a project plan for migrating a monolith to microservices.",
        "Design a beginner's twelve-week half-marathon training plan.",
        "Plan a surprise birthday party for twenty guests on a tight budget.",
        "Create an onboarding checklist for a new software engineer.",
        "Outline a go-to-market strategy for a new mobile app.",
        "Draft an incident response runbook for a database outage.",
        "Plan a zero-waste kitchen transition over three months.",
        // ── miscellaneous / edge ──
        "Translate 'good evening' into Spanish, French, and Japanese.",
        "Summarize this sentence in five words: the quick brown fox jumps over the lazy dog.",
        "List five synonyms for 'happy'.",
        "Correct the grammar: 'Me and him goes to the store yesterday.'",
        "Give me three tips for better sleep.",
        "Recommend a book similar to Dune.",
        "What are the pros and cons of remote work?",
        "Explain the joke: why did the scarecrow win an award?",
        "Give me a mnemonic to remember the order of operations.",
        "What should I ask when interviewing a plumber?",
        "Suggest a name for a friendly robot vacuum.",
        "How do I politely decline a meeting invitation?",
        "What's a good icebreaker for a team offsite?",
        "Explain 'opportunity cost' with a simple example.",
        "Turn this into a polite email: send me the report now.",
    ]
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

/// The headline test: 100+ distinct prompts, each exercised end-to-end over
/// HTTP. Asserts the core forwarding invariants for every prompt and reports the
/// resulting routing distribution.
#[tokio::test]
async fn routes_over_100_prompts_with_one_upstream_call_each() {
    let prompts = corpus();
    assert!(
        prompts.len() >= 100,
        "corpus must contain at least 100 prompts, has {}",
        prompts.len()
    );

    let h = Harness::start().await;
    let mut distribution: std::collections::BTreeMap<String, usize> = Default::default();

    for (i, prompt) in prompts.iter().enumerate() {
        let resp = h.chat(&text_request(prompt)).await;
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::OK,
            "prompt #{i} should route successfully: {prompt:?}"
        );

        let body: Value = resp.json().await.expect("json response body");
        assert_eq!(
            body["choices"][0]["message"]["content"], "ok",
            "prompt #{i} should receive the mock completion"
        );

        // Exactly one upstream call was made for this request (no retries, no
        // duplicate billable generations).
        assert_eq!(
            h.call_count(),
            i + 1,
            "prompt #{i} must produce exactly one upstream call"
        );

        let forwarded = h.last_call();
        let model = forwarded["model"].as_str().expect("forwarded model");

        // The model was rewritten away from the virtual id to a real backend.
        assert_ne!(
            model, ADVERTISED_MODEL,
            "the virtual model id must never be forwarded upstream"
        );
        assert!(
            BACKENDS.contains(&model),
            "prompt #{i} routed to an unknown backend {model:?}"
        );

        // The prompt itself was forwarded verbatim.
        let forwarded_prompt = forwarded["messages"]
            .as_array()
            .and_then(|m| m.last())
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str);
        assert_eq!(
            forwarded_prompt,
            Some(*prompt),
            "prompt #{i} content must pass through unchanged"
        );

        *distribution.entry(model.to_string()).or_default() += 1;
    }

    assert_eq!(
        h.call_count(),
        prompts.len(),
        "total upstream calls must equal the number of prompts"
    );

    // The router must actually differentiate: a corpus this varied cannot all
    // collapse onto a single backend.
    eprintln!(
        "routing distribution across {} prompts: {distribution:?}",
        prompts.len()
    );
    assert!(
        distribution.len() >= 2,
        "expected the router to use at least two backends, saw {distribution:?}"
    );

    // Pure-text prompts can only land on a text tier or the image-gen backend
    // (if an image-generation intent is detected); never on the audio/vision/
    // files backends, which require input modalities not present here.
    for model in distribution.keys() {
        assert!(
            TEXT_BACKENDS.contains(&model.as_str()) || model == "image-gen",
            "text prompt unexpectedly routed to {model:?}"
        );
    }
}

/// Image-analysis requests (an `image_url` content part) must resolve to the
/// only vision-capable backend, regardless of the complexity classification.
#[tokio::test]
async fn image_input_routes_to_vision_backend() {
    let h = Harness::start().await;
    // The text is deliberately analytical (not generative): the image-analysis
    // request must not be misread as an image-*generation* request, which would
    // add `image-output` and make the set unroutable.
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "How many people are shown here?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "vision");
}

/// Audio input + requested audio output must resolve to the audio backend.
#[tokio::test]
async fn audio_in_and_out_routes_to_audio_backend() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "modalities": ["text", "audio"],
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ]}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "audio");
}

/// A `file` content part must resolve to the file-capable backend.
#[tokio::test]
async fn file_input_routes_to_files_backend() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Summarize this document."},
            {"type": "file", "file": {"file_id": "doc-1"}},
        ]}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "files");
}

/// A lexical image-generation request must resolve to the image-output backend.
/// The lexical prefilter is deterministic, so this assertion is stable.
#[tokio::test]
async fn image_generation_routes_to_image_gen_backend() {
    let h = Harness::start().await;
    let resp = h
        .chat(&text_request(
            "Please generate an image of a red bicycle on a beach.",
        ))
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "image-gen");
}

/// The classifier sees at most `current_turn_char_budget()` chars of the
/// current turn (engine-specific; 400 for zero-shot): an image request within
/// the budget routes to the image backend, while the same phrase buried past
/// it is invisible to the image axis and the request degrades to a text route.
#[tokio::test]
async fn image_intent_visibility_follows_engine_current_turn_budget() {
    let h = Harness::start().await;
    // Neutral filler: no image verbs/nouns, no complexity markers.
    let filler = "The meeting notes from last week are attached below for reference. ";
    let phrase = "Please draw a picture of a cat.";

    // ~268 chars of filler + phrase => fully within the 400-char budget.
    let within = format!("{}{phrase}", filler.repeat(4));
    let resp = h.chat(&text_request(&within)).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        h.last_call()["model"],
        "image-gen",
        "image phrase within the premise budget must be seen"
    );

    // ~469 chars of filler first => the phrase starts beyond the budget.
    let beyond = format!("{}{phrase}", filler.repeat(7));
    let resp = h.chat(&text_request(&beyond)).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_ne!(
        h.last_call()["model"],
        "image-gen",
        "image phrase beyond the premise budget is invisible to the image axis"
    );
}

/// An explicit `modalities: ["image"]` request field is a deterministic,
/// hard image-output requirement — no lexical/NLI inference needed.
#[tokio::test]
async fn explicit_image_modalities_field_routes_to_image_backend() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "modalities": ["text", "image"],
        // Deliberately phrased to evade the lexical image signal.
        "messages": [{"role": "user", "content": "A cozy cabin in the woods at dusk."}],
    });
    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "image-gen");
}

/// *Inferred* image-generation intent is a soft preference: when no backend
/// covers image-output, the request degrades to a text route instead of 422.
#[tokio::test]
async fn inferred_image_intent_degrades_gracefully_when_uncovered() {
    let h = Harness::start_with(CLASSIFIER.clone(), text_only_config_toml).await;
    let resp = h.chat(&text_request("draw a picture of a cat")).await;
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "an inferred (probabilistic) modality must never make a request unroutable"
    );
    let model = h.last_call()["model"].as_str().unwrap().to_string();
    assert!(
        TEXT_BACKENDS.contains(&model.as_str()),
        "expected a text-tier backend, got {model:?}"
    );
}

/// A request that offers `tools` must route to a tool-capable backend, whatever
/// the complexity tier resolves to.
#[tokio::test]
async fn tools_request_routes_to_tool_capable_backend() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    // Only `agent` declares the `tools` modality in the mock catalogue.
    assert_eq!(h.last_call()["model"], "agent");
}

/// A tool-loop continuation (transcript carrying `tool_calls` and a
/// `role: "tool"` result) must route to a tool-capable backend even when the
/// follow-up request omits the `tools` field.
#[tokio::test]
async fn tool_role_continuation_routes_to_tool_capable_backend() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21}"},
        ],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "agent");
}

/// A large multimodal body (over axum's 2 MB default limit, under the router's
/// configured ceiling) is accepted and routed — base64 attachments are exactly
/// the requests the modality router exists to serve.
#[tokio::test]
async fn large_multimodal_body_is_accepted() {
    let h = Harness::start().await;
    // ~3 MB of base64 payload.
    let data_url = format!("data:image/png;base64,{}", "A".repeat(3 * 1024 * 1024));
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "vision");
}

/// Upstream end-to-end response headers (request ids, rate-limit metadata)
/// reach the client.
#[tokio::test]
async fn upstream_headers_pass_through_to_client() {
    let h = Harness::start().await;
    let resp = h.chat(&text_request("Say hello.")).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("x-mock-request-id")
            .and_then(|v| v.to_str().ok()),
        Some("mock-123"),
        "upstream response headers must pass through"
    );
}

/// With a single model covering every modality, every request has exactly one
/// candidate, so classification is skipped and everything routes to it —
/// regardless of complexity or (lexical) image-generation intent.
#[tokio::test]
async fn single_model_deployment_routes_everything() {
    let h = Harness::start_with(CLASSIFIER.clone(), single_model_config_toml).await;
    for prompt in [
        "hi",
        "Derive and rigorously prove a hard theorem with a formal amortized analysis.",
        "draw a picture of a cat",
    ] {
        let resp = h.chat(&text_request(prompt)).await;
        assert_eq!(resp.status(), reqwest::StatusCode::OK, "prompt {prompt:?}");
        assert_eq!(h.last_call()["model"], "only", "prompt {prompt:?}");
    }
}

/// A request whose combined modalities no single backend covers must yield 422
/// and must never reach a backend.
#[tokio::test]
async fn uncovered_modality_combination_returns_422_and_makes_no_call() {
    let h = Harness::start().await;
    // Needs image-input AND audio-input together — no single backend has both.
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ]}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        h.call_count(),
        0,
        "no upstream call for an unroutable request"
    );
}

/// Passthrough fidelity: the model is rewritten, and every other field
/// (including unknown ones, and an innocuous `n = 1`) is forwarded untouched.
#[tokio::test]
async fn request_fields_pass_through_except_model() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "n": 1,
        "temperature": 0.7,
        "top_logprobs": 5,
        "custom_unknown_key": {"nested": [1, 2, 3]},
        "messages": [{"role": "user", "content": "Say hello."}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let forwarded = h.last_call();
    assert_eq!(forwarded["n"], 1, "`n` = 1 passes through untouched");
    assert_ne!(
        forwarded["model"], ADVERTISED_MODEL,
        "model must be rewritten"
    );
    assert!(BACKENDS.contains(&forwarded["model"].as_str().unwrap()));
    assert_eq!(forwarded["temperature"], 0.7);
    assert_eq!(forwarded["top_logprobs"], 5);
    assert_eq!(forwarded["custom_unknown_key"]["nested"][2], 3);
}

/// A multi-choice request (`n > 1`) is rejected with 400 rather than silently
/// altered — the router serves exactly one completion per request.
#[tokio::test]
async fn multi_choice_request_is_rejected_with_400() {
    let h = Harness::start().await;
    let mut body = text_request("Say hello.");
    body["n"] = json!(4);
    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(
        h.call_count(),
        0,
        "a rejected request must not be forwarded"
    );
}

/// A configured `api_key` is forwarded upstream as a bearer token.
#[tokio::test]
async fn configured_api_key_is_forwarded_as_bearer() {
    let h = Harness::start().await;
    let resp = h.chat(&text_request("Say hello.")).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    // The mock catalogue sets api_key = "test-key" on every model.
    assert_eq!(h.last_authorization().as_deref(), Some("Bearer test-key"));
}

/// A keyless backend (no `api_key`) receives no `Authorization` header.
#[tokio::test]
async fn keyless_backend_sends_no_authorization_header() {
    let h = Harness::start_with(CLASSIFIER.clone(), keyless_config_toml).await;
    let resp = h.chat(&text_request("Say hello.")).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.call_count(), 1);
    assert_eq!(
        h.last_authorization(),
        None,
        "keyless backend must get no auth header"
    );
}

/// Streaming requests get a text/event-stream passthrough of the upstream SSE.
#[tokio::test]
async fn streaming_request_is_passed_through_as_sse() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "stream": true,
        "messages": [{"role": "user", "content": "Stream a short greeting."}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.contains("text/event-stream"),
        "expected SSE content type, got {content_type:?}"
    );

    // Upstream headers survive the streaming path too.
    assert_eq!(
        resp.headers()
            .get("x-mock-request-id")
            .and_then(|v| v.to_str().ok()),
        Some("mock-123"),
        "upstream headers must pass through on the streaming path"
    );

    let text = resp.text().await.expect("stream body");
    assert!(
        text.contains("data:"),
        "SSE body should contain data frames"
    );
    assert_eq!(h.last_call()["stream"], true);
}

/// A JSON body that is not an object is rejected with 400 before any routing.
#[tokio::test]
async fn non_object_body_is_rejected_with_400() {
    let h = Harness::start().await;
    let resp = h.chat(&json!("this is a bare string, not an object")).await;
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(h.call_count(), 0);
}

/// `/v1/models` advertises only the single virtual model, never the backends.
#[tokio::test]
async fn models_endpoint_advertises_only_the_virtual_model() {
    let h = Harness::start().await;
    let resp = h.get("/v1/models").await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], ADVERTISED_MODEL);
    for backend in BACKENDS {
        assert_ne!(
            data[0]["id"], backend,
            "backend names must never be advertised"
        );
    }
}

/// `/health` is a backend-free liveness probe.
#[tokio::test]
async fn health_endpoint_is_ok() {
    let h = Harness::start().await;
    let resp = h.get("/health").await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(h.call_count(), 0);
}

// ───────────────────────────────────────────────────────────────────────────
// Contextual complexity (windowed user turns)
// ───────────────────────────────────────────────────────────────────────────
//
// Complexity is classified from a window of recent *substantive* user turns
// (trivial greetings/acks pruned). There is no `infer_from_history` heuristic:
// a terse turn inherits the difficulty of the substantive context behind it,
// and a conversation of pure filler routes to Fast without the model.

/// Map a text-tier backend name to its rank (Fast < Balanced < Frontier). Panics
/// on any non-text-tier backend, so an unexpected reroute fails loudly.
fn text_tier_rank(model: &str) -> u8 {
    match model {
        "fast-text" => 0,
        "balanced-text" => 1,
        "frontier-text" => 2,
        other => panic!("expected a text-tier backend, prompt routed to {other:?}"),
    }
}

/// A conversation of only trivial turns prunes to an empty window and routes to
/// the Fast tier *without* invoking the model.
#[tokio::test]
async fn pure_chit_chat_routes_to_fast() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "thanks, ok"},
        ],
    });
    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "fast-text");
}

/// A terse follow-up inherits the difficulty of the recent context: the same
/// "ok, continue" routes to Fast on its own but escalates when it follows a hard
/// question. This is the behavior the old turn-count/tool heuristic faked.
#[tokio::test]
async fn terse_followup_inherits_hard_context() {
    let h = Harness::start().await;
    let hard = "Derive and rigorously prove the asymptotic time complexity of red-black \
                tree rebalancing across a sequence of insertions and deletions, with a \
                formal amortized analysis.";

    // Fresh terse turn, no context → empty window → Fast, no model call.
    h.chat(&json!({
        "model": ADVERTISED_MODEL,
        "messages": [{"role": "user", "content": "ok, continue"}],
    }))
    .await;
    let fresh = h.last_call()["model"].as_str().unwrap().to_string();

    // Same terse turn behind a hard question → window reaches the hard turn.
    h.chat(&json!({
        "model": ADVERTISED_MODEL,
        "messages": [
            {"role": "user", "content": hard},
            {"role": "assistant", "content": "(a long proof the window ignores)"},
            {"role": "user", "content": "ok, continue"},
        ],
    }))
    .await;
    let with_context = h.last_call()["model"].as_str().unwrap().to_string();

    assert_eq!(
        fresh, "fast-text",
        "a terse turn with no context routes to Fast"
    );
    assert!(
        text_tier_rank(&with_context) > text_tier_rank(&fresh),
        "a terse follow-up on a hard thread must escalate above Fast, got {with_context:?}"
    );
}

/// A growing multi-turn session forwards the full transcript each turn and makes
/// exactly one upstream call per turn (transport invariants, tier aside).
#[tokio::test]
async fn multi_turn_session_forwards_full_transcript_one_call_per_turn() {
    let h = Harness::start().await;
    let mut history: Vec<Value> = Vec::new();
    for (turn, prompt) in ["What is a monad?", "ok", "give an example", "thanks"]
        .iter()
        .enumerate()
    {
        history.push(json!({"role": "user", "content": prompt}));
        let resp = h
            .chat(&json!({"model": ADVERTISED_MODEL, "messages": history}))
            .await;
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        assert_eq!(
            h.call_count(),
            turn + 1,
            "exactly one upstream call per turn"
        );
        assert_eq!(
            h.last_call()["messages"].as_array().map(Vec::len),
            Some(history.len()),
            "the full accumulated transcript is forwarded"
        );
        history.push(json!({"role": "assistant", "content": "ok"}));
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Load test: progressive concurrency with latency/throughput measurement
// ───────────────────────────────────────────────────────────────────────────
//
// Opt-in (slow, timing-dependent). Run with:
//   cargo test --test api_routing -- --ignored --nocapture load
// Tune the per-level request count with the LOAD_REQUESTS env var (default 120).
//
// The classifier runs one batched ONNX pass under a `Mutex<Session>`, so
// inference is serialized across concurrent requests: expect throughput to
// plateau and per-request latency to grow with concurrency as requests queue on
// that lock. This test *measures and reports* that behaviour; it asserts only
// that every request succeeds, not any timing threshold (those are environment
// dependent).

/// Latency summary in milliseconds.
struct LatencyStats {
    n: usize,
    min: f64,
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_ms.len() - 1) as f64).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

fn summarize(latencies: &[Duration]) -> LatencyStats {
    let mut ms: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ms.len();
    let sum: f64 = ms.iter().sum();
    LatencyStats {
        n,
        min: *ms.first().unwrap_or(&0.0),
        mean: if n > 0 { sum / n as f64 } else { 0.0 },
        p50: percentile(&ms, 50.0),
        p95: percentile(&ms, 95.0),
        p99: percentile(&ms, 99.0),
        max: *ms.last().unwrap_or(&0.0),
    }
}

/// Fire `total` requests against the router with exactly `concurrency` in
/// flight, returning the per-request end-to-end latencies (send → full body).
/// When `fixed_prompt` is `Some`, every request uses it verbatim (set it to a
/// trivial phrase like "ok" to measure the lexical fast path); otherwise each
/// request gets a unique non-trivial prompt that exercises the NLI model.
/// Build a request body with `turns` user turns. The final turn is `content`;
/// each earlier turn is a distinct *substantive* user message (interleaved with
/// an assistant reply the classifier's window ignores), so a multi-turn body
/// exercises the windowed classifier over real accumulated context.
fn load_request_body(content: &str, turns: usize) -> Value {
    if turns <= 1 {
        return json!({
            "model": ADVERTISED_MODEL,
            "messages": [{"role": "user", "content": content}],
        });
    }
    let mut messages = Vec::with_capacity(turns * 2);
    for t in 0..(turns - 1) {
        messages.push(json!({
            "role": "user",
            "content": format!(
                "Explain in detail, with rigorous reasoning, aspect {t} of the \
                 distributed system design under discussion."
            ),
        }));
        messages.push(json!({
            "role": "assistant",
            "content": "(a long assistant response the classification window ignores)",
        }));
    }
    messages.push(json!({"role": "user", "content": content}));
    json!({"model": ADVERTISED_MODEL, "messages": messages})
}

async fn run_load(
    h: &Harness,
    concurrency: usize,
    total: usize,
    fixed_prompt: Arc<Option<String>>,
    turns: usize,
) -> Vec<Duration> {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::with_capacity(concurrency);

    for _ in 0..concurrency {
        let client = h.client.clone();
        let url = format!("{}/v1/chat/completions", h.base);
        let counter = Arc::clone(&counter);
        let fixed_prompt = Arc::clone(&fixed_prompt);
        workers.push(tokio::spawn(async move {
            let mut lats = Vec::new();
            loop {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    break;
                }
                let content = match fixed_prompt.as_ref() {
                    Some(p) => p.clone(),
                    None => format!("load request {i}"),
                };
                let body = load_request_body(&content, turns);
                let start = Instant::now();
                let resp = client.post(&url).json(&body).send().await.expect("send");
                assert!(
                    resp.status().is_success(),
                    "request {i} failed: {}",
                    resp.status()
                );
                let _ = resp.bytes().await.expect("drain body");
                lats.push(start.elapsed());
            }
            lats
        }));
    }

    let mut all = Vec::with_capacity(total);
    for w in workers {
        all.extend(w.await.expect("worker joined"));
    }
    all
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "load test; run explicitly with --ignored --nocapture"]
async fn load_test_progressive_concurrency() {
    let total: usize = std::env::var("LOAD_REQUESTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let levels = [1usize, 2, 4, 8, 16, 32, 64];
    // `LOAD_PROMPT=ok` (or any trivial phrase) measures the lexical fast path;
    // unset measures the full NLI model path with unique prompts.
    let fixed_prompt = Arc::new(std::env::var("LOAD_PROMPT").ok());
    // `LOAD_TURNS=N` builds N-user-turn conversations to measure how the
    // windowed classifier's cost scales with conversation depth (bounded by the
    // character budget); unset = single-turn.
    let turns: usize = std::env::var("LOAD_TURNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);

    // `LOAD_POOL_SIZE=N` (optionally `LOAD_INTRA_OP=T`) builds a dedicated
    // classifier with that pool size to sweep inference concurrency; unset uses
    // the shared 2-session classifier.
    let pool_override = std::env::var("LOAD_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let h = if let Some(pool) = pool_override {
        let intra_op = std::env::var("LOAD_INTRA_OP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        println!("(dedicated classifier: pool_size={pool}, intra_op_threads={intra_op})");
        let clf = Arc::new(
            ZeroShot::new(DEFAULT_IMAGE_GEN_THRESHOLD, pool, intra_op).expect("build classifier"),
        );
        Harness::start_with_classifier(clf).await
    } else {
        Harness::start().await
    };

    // Warm up: force the shared classifier's lazy init and the ORT session's
    // first pass so it doesn't skew the first measured level.
    let warm = h.chat(&text_request("warmup for the load test")).await;
    assert!(warm.status().is_success());

    let path = if fixed_prompt.is_some() {
        "lexical fast path"
    } else {
        "full NLI model path"
    };
    println!("\nload test — {total} requests per concurrency level ({path}, {turns} turn(s))");
    println!(
        "{:>5}  {:>8}  {:>9}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "conc", "wall_s", "req/s", "min_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"
    );

    for &conc in &levels {
        let started = Instant::now();
        let latencies = run_load(&h, conc, total, Arc::clone(&fixed_prompt), turns).await;
        let wall = started.elapsed().as_secs_f64();

        assert_eq!(latencies.len(), total, "every request must complete");
        let s = summarize(&latencies);
        let throughput = if wall > 0.0 { s.n as f64 / wall } else { 0.0 };

        println!(
            "{:>5}  {:>8.3}  {:>9.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}",
            conc, wall, throughput, s.min, s.mean, s.p50, s.p95, s.p99, s.max
        );
    }
    println!();
}
