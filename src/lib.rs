//! Hyper MCP Router — an adaptive, OpenAI Chat Completions compatible LLM router.
//!
//! A single self-contained binary presents one virtual model to any client,
//! classifies each incoming prompt with an embedded zero-shot NLI model, and
//! transparently forwards the request to the appropriate backend model tier.
//!
//! This library crate holds all of the router's logic — the classifier, the
//! proxy, and configuration handling — so it is reachable from the crate's unit
//! tests. `main.rs` is a thin binary wrapper over it.

pub mod classifier;
pub mod cli;
pub mod config;
pub mod logging;
pub mod modality;
pub mod planning;
pub mod proxy;

/// The quantized ONNX NLI model, embedded at build time (see `build.rs`).
pub(crate) static MODEL_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/router_model.onnx"));

/// The model's tokenizer, embedded at build time (see `build.rs`).
pub(crate) static TOKENIZER_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/tokenizer.json"));
