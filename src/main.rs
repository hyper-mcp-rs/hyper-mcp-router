//! Binary entry point: parse the CLI, initialise logging, run startup, and bind
//! the axum server. A thin wrapper over the `hyper_mcp_router` library crate.

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;

use hyper_mcp_router::classifier::{plan_inference, Classifier};
use hyper_mcp_router::cli::{Cli, Command, ServeArgs};
use hyper_mcp_router::config;
use hyper_mcp_router::logging;
use hyper_mcp_router::proxy::{build_router, AppState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve(args).await,
    }
}

async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    // 1. Logging first, so every subsequent startup step (including failures)
    //    is captured. Hold the guard for the process lifetime.
    let _guard = logging::init(args.log_stdout)?;

    // 2. Resolve, expand, parse, and validate config (fatal on any problem).
    let config_path = config::resolve_config_path(args.config)?;
    let cfg = config::load(&config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    // (config::load runs field + startup coverage validation internally.)

    // 4. Initialise the classifier from the embedded model bytes. Size the
    //    inference pool + intra-op threads from the detected core count
    //    (container-aware since Rust 1.64), overridable via CLI flags.
    let detected_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let plan = plan_inference(detected_cores);
    let pool_size = args.inference_pool_size.unwrap_or(plan.pool_size);
    let intra_op_threads = args.intra_op_threads.unwrap_or(plan.intra_op_threads);
    let classifier = Classifier::new(
        cfg.classifier.image_generation_threshold,
        args.trivial_max_words,
        pool_size,
        intra_op_threads,
    )
    .context("initialising classifier")?;
    tracing::info!(
        detected_cores,
        pool_size,
        intra_op_threads,
        "inference parallelism configured"
    );

    // 5. Log the resolved configuration — names, types, modalities, base URLs.
    //    Never API keys.
    tracing::info!(
        config_path = %config_path.display(),
        advertised_model = hyper_mcp_router::proxy::ADVERTISED_MODEL,
        model_count = cfg.models.len(),
        image_generation_threshold = cfg.classifier.image_generation_threshold,
        trivial_max_words = args.trivial_max_words,
        "configuration loaded"
    );
    for m in &cfg.models {
        tracing::info!(
            model = %m.name,
            tier = ?m.tier,
            base_url = %m.base_url,
            modalities = ?m.modality_set().to_kebab_vec(),
            "configured backend"
        );
    }

    // 6. Bind the server and begin accepting requests.
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let state = AppState::new(Arc::new(classifier), Arc::new(cfg))?;
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding to {addr}"))?;
    tracing::info!(%addr, "hyper-mcp-router listening");
    axum::serve(listener, app).await.context("server error")?;

    Ok(())
}
