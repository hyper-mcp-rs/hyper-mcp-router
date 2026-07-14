//! Binary entry point: parse the CLI, initialise logging, run startup, and bind
//! the axum server. A thin wrapper over the `hyper_mcp_router` library crate.

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;

use hyper_mcp_router::cli::{Cli, Command, ServeArgs};
use hyper_mcp_router::config;
use hyper_mcp_router::engines::{self, EngineOverrides};
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

    // 3. Build the selected classifier engine (exactly one per process; CLI
    //    flag overrides the config setting). All model-specific sizing —
    //    session pools, thread counts, memory planning, overcommit warnings —
    //    is owned by the engine itself (see `engines/`); the operator
    //    overrides are merged by precedence (CLI over config) here and passed
    //    through.
    let model = args.classifier_model.unwrap_or(cfg.classifier.model);
    let overrides = EngineOverrides {
        inference_pool_size: args
            .inference_pool_size
            .or(cfg.classifier.inference_pool_size),
        intra_op_threads: args.intra_op_threads.or(cfg.classifier.intra_op_threads),
    };
    let classifier = engines::build(model, &cfg.classifier, &overrides)
        .with_context(|| format!("initialising classifier engine `{}`", model.as_str()))?;
    let trivial_max_words = args
        .trivial_max_words
        .unwrap_or(cfg.classifier.trivial_max_words);
    tracing::info!(
        engine = classifier.name(),
        context_char_budget = classifier.context_char_budget(),
        current_turn_char_budget = classifier.current_turn_char_budget(),
        "classifier engine ready"
    );

    // 4. Log the resolved configuration — names, types, modalities, base URLs.
    //    Never API keys.
    tracing::info!(
        config_path = %config_path.display(),
        advertised_model = hyper_mcp_router::proxy::ADVERTISED_MODEL,
        model_count = cfg.models.len(),
        image_generation_threshold = cfg.classifier.image_generation_threshold,
        trivial_max_words,
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

    // 5. Bind the server and accept requests until a shutdown signal arrives;
    //    then stop accepting and drain in-flight requests (including open SSE
    //    streams) instead of severing them.
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let state = AppState::new(classifier, Arc::new(cfg), trivial_max_words)?;
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding to {addr}"))?;
    tracing::info!(%addr, "hyper-mcp-router listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    Ok(())
}

/// Resolve when the process receives a shutdown signal: Ctrl+C (SIGINT)
/// anywhere, or SIGTERM on Unix (what container orchestrators send first).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received; draining in-flight requests");
}
