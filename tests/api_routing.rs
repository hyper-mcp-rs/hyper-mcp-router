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

use hyper_mcp_router::classifier::{
    Classifier, DEFAULT_IMAGE_GEN_THRESHOLD, DEFAULT_TRIVIAL_MAX_WORDS,
};
use hyper_mcp_router::config;
use hyper_mcp_router::proxy::{build_router, AppState, ADVERTISED_MODEL};

// ───────────────────────────────────────────────────────────────────────────
// Shared classifier (embedded ONNX model) — loaded once for the whole binary.
// ───────────────────────────────────────────────────────────────────────────

static CLASSIFIER: LazyLock<Arc<Classifier>> = LazyLock::new(|| {
    // Pool of 2 (default ORT intra-op threads) so the pooling path is exercised
    // by the correctness suite without a heavy N-session startup cost.
    Arc::new(
        Classifier::new(DEFAULT_IMAGE_GEN_THRESHOLD, DEFAULT_TRIVIAL_MAX_WORDS, 2, 0)
            .expect("load embedded classifier"),
    )
});

/// Every backend name declared by [`mock_config_toml`].
const BACKENDS: [&str; 7] = [
    "fast-text",
    "balanced-text",
    "frontier-text",
    "vision",
    "audio",
    "files",
    "image-gen",
];

/// The text-only tier backends, one per complexity type.
const TEXT_BACKENDS: [&str; 3] = ["fast-text", "balanced-text", "frontier-text"];

// ───────────────────────────────────────────────────────────────────────────
// Mock backend: records every forwarded request, returns a trivial completion.
// ───────────────────────────────────────────────────────────────────────────

/// Thread-safe log of every request body the router forwarded upstream.
type Calls = Arc<Mutex<Vec<Value>>>;

async fn spawn_mock_backend() -> (SocketAddr, Calls) {
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/chat/completions", post(mock_chat))
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

async fn mock_chat(State(calls): State<Calls>, body: Bytes) -> Response {
    let forwarded: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let streaming = forwarded
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    calls.lock().unwrap().push(forwarded);

    if streaming {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
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

    async fn start_with_classifier(classifier: Arc<Classifier>) -> Harness {
        let (mock_addr, calls) = spawn_mock_backend().await;

        // Build the config through the real parse + validate path so coverage
        // validation is exercised too.
        let cfg = config::parse(&mock_config_toml(mock_addr)).expect("parse config");
        cfg.validate().expect("validate config");

        let state = AppState::new(classifier, Arc::new(cfg)).expect("build app state");
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
            .cloned()
            .expect("at least one recorded upstream call")
    }
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

/// A request whose combined modalities no single backend covers must yield 415
/// and must never reach a backend.
#[tokio::test]
async fn uncovered_modality_combination_returns_415_and_makes_no_call() {
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
    assert_eq!(resp.status(), reqwest::StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(
        h.call_count(),
        0,
        "no upstream call for an unroutable request"
    );
}

/// Passthrough fidelity: `n` is stripped, the model is rewritten, and every
/// other field (including unknown ones) is forwarded untouched.
#[tokio::test]
async fn request_fields_pass_through_except_n_and_model() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "n": 4,
        "temperature": 0.7,
        "top_logprobs": 5,
        "custom_unknown_key": {"nested": [1, 2, 3]},
        "messages": [{"role": "user", "content": "Say hello."}],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let forwarded = h.last_call();
    assert!(forwarded.get("n").is_none(), "`n` must be stripped");
    assert_ne!(
        forwarded["model"], ADVERTISED_MODEL,
        "model must be rewritten"
    );
    assert!(BACKENDS.contains(&forwarded["model"].as_str().unwrap()));
    assert_eq!(forwarded["temperature"], 0.7);
    assert_eq!(forwarded["top_logprobs"], 5);
    assert_eq!(forwarded["custom_unknown_key"]["nested"][2], 3);
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
// Conversation depth / history escalation (`infer_from_history`)
// ───────────────────────────────────────────────────────────────────────────
//
// The effective complexity is `max(classifier, infer_from_history)`, so history
// can only *escalate* the tier, never lower it. `infer_from_history` maps
// history metadata to a tier: assistant `tool_calls` → Frontier, >8 user turns →
// Frontier, >3 → Balanced, else Fast. Because Frontier is the ceiling, the
// `tool_calls` and deep-history cases are deterministic regardless of what the
// (non-deterministic) classifier returns for the final user turn; the mid-depth
// and monotonicity assertions hold because the classifier sees the *same* final
// user turn across every depth variant.

/// Map a text-tier backend name to its rank (Fast < Balanced < Frontier). Panics
/// on any non-text-tier backend, which makes an unexpected reroute a loud
/// failure rather than a silent skip.
fn text_tier_rank(model: &str) -> u8 {
    match model {
        "fast-text" => 0,
        "balanced-text" => 1,
        "frontier-text" => 2,
        other => panic!("expected a text-tier backend, prompt routed to {other:?}"),
    }
}

/// A conversation with exactly `user_turns` user messages and no `tool_calls`.
/// Every earlier user turn is filler; the final user turn carries `last_prompt`
/// (the only message the classifier sees).
fn with_user_turns(last_prompt: &str, user_turns: usize) -> Value {
    assert!(user_turns >= 1);
    let mut messages = Vec::new();
    for i in 0..(user_turns - 1) {
        messages.push(json!({"role": "user", "content": format!("Follow-up context {i}.")}));
        messages.push(json!({"role": "assistant", "content": format!("Understood ({i}).")}));
    }
    messages.push(json!({"role": "user", "content": last_prompt}));
    json!({"model": ADVERTISED_MODEL, "messages": messages})
}

/// A short conversation containing an assistant `tool_calls` turn, ending with
/// `last_prompt` as the final user message.
fn with_tool_calls(last_prompt: &str) -> Value {
    json!({
        "model": ADVERTISED_MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C and sunny"},
            {"role": "user", "content": last_prompt},
        ],
    })
}

/// Terse continuation prompts, all verified not to trip the image-generation
/// axis, so every one routes to a text-tier backend and `text_tier_rank` applies.
fn continuation_prompts() -> Vec<&'static str> {
    vec![
        "Please continue.",
        "Can you elaborate on that?",
        "Tell me more.",
        "Why is that the case?",
        "Summarize the discussion so far.",
        "What are the next steps?",
        "Give me an example.",
        "How does that compare to the alternative?",
        "What are the tradeoffs?",
        "Explain that in simpler terms.",
        "What could go wrong?",
        "Is there a better approach?",
        "Walk me through the reasoning.",
        "What assumptions are we making?",
        "How would you test this?",
        "What is the time complexity?",
        "Can you refactor that?",
        "What are the edge cases?",
        "How should we handle errors here?",
        "What would you recommend?",
    ]
}

/// An assistant `tool_calls` turn escalates to Frontier regardless of how the
/// final (terse) user turn classifies.
#[tokio::test]
async fn history_tool_calls_forces_frontier() {
    let h = Harness::start().await;
    let resp = h.chat(&with_tool_calls("Thanks, please continue.")).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "frontier-text");
}

/// A deep conversation (>8 user turns) escalates to Frontier regardless of how
/// the final (terse) user turn classifies.
#[tokio::test]
async fn history_deep_user_turns_forces_frontier() {
    let h = Harness::start().await;
    let resp = h.chat(&with_user_turns("Please continue.", 9)).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(h.last_call()["model"], "frontier-text");
}

/// Mid-depth history (4–8 user turns) guarantees at least the Balanced tier: the
/// selected tier is never Fast, whatever the classifier decides for the final
/// turn.
#[tokio::test]
async fn history_mid_depth_is_at_least_balanced() {
    let h = Harness::start().await;
    for turns in [4usize, 6, 8] {
        let resp = h.chat(&with_user_turns("Please continue.", turns)).await;
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        let model = h.last_call()["model"].as_str().unwrap().to_string();
        assert!(
            text_tier_rank(&model) >= 1,
            "{turns} user turns must escalate to at least Balanced, got {model:?}"
        );
    }
}

/// Escalation is monotonic in conversation depth: for a fixed final user turn,
/// deeper history never selects a lower tier, and both the deep-turn and
/// `tool_calls` variants reach the Frontier ceiling. Exercised across many
/// prompts at several depths so history is stressed broadly, not just once.
#[tokio::test]
async fn history_escalation_is_monotonic_across_depths() {
    let h = Harness::start().await;

    for prompt in continuation_prompts() {
        // depth 1 (Fast baseline), depth 4 (Balanced floor), depth 9 (Frontier).
        let r1 = {
            h.chat(&with_user_turns(prompt, 1)).await;
            h.last_call()["model"].as_str().unwrap().to_string()
        };
        let r4 = {
            h.chat(&with_user_turns(prompt, 4)).await;
            h.last_call()["model"].as_str().unwrap().to_string()
        };
        let r9 = {
            h.chat(&with_user_turns(prompt, 9)).await;
            h.last_call()["model"].as_str().unwrap().to_string()
        };
        let rt = {
            h.chat(&with_tool_calls(prompt)).await;
            h.last_call()["model"].as_str().unwrap().to_string()
        };

        let (t1, t4, t9) = (
            text_tier_rank(&r1),
            text_tier_rank(&r4),
            text_tier_rank(&r9),
        );
        // `tool_calls` must also sit at the ceiling.
        assert_eq!(
            text_tier_rank(&rt),
            2,
            "tool_calls must reach Frontier for {prompt:?}"
        );

        assert!(
            t1 <= t4 && t4 <= t9,
            "tier must not decrease with depth for {prompt:?}: d1={r1}, d4={r4}, d9={r9}"
        );
        assert!(
            t4 >= 1,
            "4 user turns must be at least Balanced for {prompt:?}, got {r4}"
        );
        assert_eq!(
            r9, "frontier-text",
            ">8 user turns must reach Frontier for {prompt:?}"
        );
        assert_eq!(
            rt, "frontier-text",
            "tool_calls must reach Frontier for {prompt:?}"
        );
    }
}

/// History escalation changes the *tier* but must never change the required
/// modality set: an image-analysis final turn behind a `tool_calls` history
/// still resolves to the vision backend (the only image-input-capable model),
/// even though the tier is escalated to Frontier.
#[tokio::test]
async fn history_escalation_preserves_modality_routing() {
    let h = Harness::start().await;
    let body = json!({
        "model": ADVERTISED_MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            {"role": "user", "content": [
                {"type": "text", "text": "How many people are shown here?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ]},
        ],
    });

    let resp = h.chat(&body).await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    // Frontier escalation cannot invent a frontier vision model; the only
    // image-input-capable backend is `vision`.
    assert_eq!(h.last_call()["model"], "vision");
}

// ───────────────────────────────────────────────────────────────────────────
// Extended multi-turn chat sessions (real corpus prompts, growing transcript)
// ───────────────────────────────────────────────────────────────────────────
//
// These simulate a real client: each turn appends the new user message to the
// running transcript, POSTs the *entire* accumulated history, then appends the
// (mocked) assistant reply before the next turn. This exercises
// `infer_from_history` the way it actually runs in production — the user-turn
// count grows with the session — rather than via synthetic one-shot histories.
//
// The classifier sees only the latest user turn, so per-turn complexity varies
// and the session tier is *not* globally monotonic. What is guaranteed is the
// history *floor*: effective >= infer_from_history(depth). We assert that floor.

/// Sessions drawn from the main corpus, verified to stay on the text path so
/// `text_tier_rank` applies to every turn. Lengths vary to cross all three
/// history floors (Fast ≤3, Balanced 4–8, Frontier >8).
fn corpus_sessions() -> Vec<Vec<&'static str>> {
    vec![
        // Short (5 turns): stays within the Fast/Balanced floors.
        vec![
            "hi",
            "What is the capital of France?",
            "What is 2 + 2?",
            "List five synonyms for 'happy'.",
            "What are the pros and cons of remote work?",
        ],
        // Mid (9 turns): crosses into the Balanced floor and reaches the
        // Frontier floor on the last turn.
        vec![
            "How are you?",
            "What is the boiling point of water in Celsius?",
            "Convert 100 kilometers to miles.",
            "Write a function to reverse a string in Python.",
            "What is the time complexity of quicksort in the worst case?",
            "Explain the difference between let and const in JavaScript.",
            "How does garbage collection work in the JVM?",
            "Explain how a bill becomes a law in the United States.",
            "What causes the seasons to change?",
        ],
        // Long (12 turns): spends several turns in the Frontier floor.
        vec![
            "Good morning",
            "Who wrote Hamlet?",
            "Solve for x: 3x + 5 = 20.",
            "What is the derivative of x^2?",
            "Implement binary search in Rust.",
            "Explain the borrow checker in Rust to a beginner.",
            "Design a rate limiter for a REST API.",
            "How would you shard a Postgres database for horizontal scaling?",
            "Derive and rigorously prove the asymptotic time complexity of red-black \
             tree rebalancing across a sequence of insertions and deletions, with a \
             formal amortized analysis.",
            "Compare and contrast the epistemological foundations of Bayesian and \
             frequentist statistics.",
            "Discuss the tradeoffs between CAP theorem guarantees in a globally \
             distributed database.",
            "What are the next steps?",
        ],
    ]
}

/// Drive an extended chat session, returning the backend chosen for each turn.
/// Asserts the per-turn transport invariants: exactly one upstream call per
/// turn, and the full growing transcript forwarded each time.
async fn run_session(h: &Harness, prompts: &[&str]) -> Vec<String> {
    let mut history: Vec<Value> = Vec::new();
    let mut models = Vec::new();
    let base = h.call_count();

    for (turn, prompt) in prompts.iter().enumerate() {
        history.push(json!({"role": "user", "content": prompt}));
        let resp = h
            .chat(&json!({"model": ADVERTISED_MODEL, "messages": history}))
            .await;
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::OK,
            "turn {} should route successfully: {prompt:?}",
            turn + 1
        );

        // One upstream call per turn — no duplicate billable generations.
        assert_eq!(
            h.call_count(),
            base + turn + 1,
            "turn {} must make exactly one upstream call",
            turn + 1
        );

        // The entire accumulated transcript is forwarded, not just the last turn.
        let forwarded = h.last_call();
        assert_eq!(
            forwarded["messages"].as_array().map(Vec::len),
            Some(history.len()),
            "turn {} must forward the full transcript",
            turn + 1
        );

        models.push(forwarded["model"].as_str().unwrap().to_string());

        // The client appends the assistant's reply before the next turn.
        history.push(json!({"role": "assistant", "content": "ok"}));
    }

    models
}

/// Corpus-driven extended sessions of varying depth must honour the
/// `infer_from_history` floor on every turn: turns 4–8 route at least Balanced,
/// turns beyond 8 route to the Frontier ceiling.
#[tokio::test]
async fn extended_sessions_honour_history_floor_per_turn() {
    let h = Harness::start().await;
    let sessions = corpus_sessions();
    assert!(
        sessions.iter().any(|s| s.len() > 8),
        "at least one session must exceed 8 turns to exercise the Frontier floor"
    );

    for (s, session) in sessions.iter().enumerate() {
        let models = run_session(&h, session).await;
        for (turn_idx, model) in models.iter().enumerate() {
            let user_turns = turn_idx + 1;
            let rank = text_tier_rank(model);
            if user_turns > 8 {
                assert_eq!(
                    model, "frontier-text",
                    "session {s} turn {user_turns} (>8 turns) must reach the Frontier floor"
                );
            } else if user_turns > 3 {
                assert!(
                    rank >= 1,
                    "session {s} turn {user_turns} (4–8 turns) must be at least Balanced, got {model:?}"
                );
            }
        }
    }
}

/// An agentic session: partway through, the assistant issues a tool call. From
/// then on the transcript carries a `tool_calls` message, so every subsequent
/// turn — even a trivial "thanks" — escalates to the Frontier tier.
#[tokio::test]
async fn agentic_session_tool_call_pins_frontier_thereafter() {
    let h = Harness::start().await;
    let mut history: Vec<Value> = Vec::new();

    // Turn 1: an ordinary request; the mocked assistant answers a tool call.
    history.push(json!({"role": "user", "content": "Book me a flight to Tokyo next Friday."}));
    let resp = h
        .chat(&json!({"model": ADVERTISED_MODEL, "messages": history}))
        .await;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // The client records the model's tool call and the tool's result.
    history.push(json!({"role": "assistant", "tool_calls": [{
        "id": "call_flight",
        "type": "function",
        "function": {"name": "search_flights", "arguments": "{\"dest\":\"HND\"}"},
    }]}));
    history
        .push(json!({"role": "tool", "tool_call_id": "call_flight", "content": "3 flights found"}));

    // Subsequent turns — even trivial ones — are pinned to Frontier by the
    // tool_calls turn now living in the history.
    for follow_up in ["Thanks!", "Which is cheapest?", "ok"] {
        history.push(json!({"role": "user", "content": follow_up}));
        let resp = h
            .chat(&json!({"model": ADVERTISED_MODEL, "messages": history}))
            .await;
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        assert_eq!(
            h.last_call()["model"],
            "frontier-text",
            "a tool_calls history must pin {follow_up:?} to Frontier"
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
async fn run_load(
    h: &Harness,
    concurrency: usize,
    total: usize,
    fixed_prompt: Arc<Option<String>>,
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
                let body = json!({
                    "model": ADVERTISED_MODEL,
                    "messages": [{"role": "user", "content": content}],
                });
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
            Classifier::new(
                DEFAULT_IMAGE_GEN_THRESHOLD,
                DEFAULT_TRIVIAL_MAX_WORDS,
                pool,
                intra_op,
            )
            .expect("build classifier"),
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
    println!("\nload test — {total} requests per concurrency level ({path})");
    println!(
        "{:>5}  {:>8}  {:>9}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "conc", "wall_s", "req/s", "min_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"
    );

    for &conc in &levels {
        let started = Instant::now();
        let latencies = run_load(&h, conc, total, Arc::clone(&fixed_prompt)).await;
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
