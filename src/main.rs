//! Binary entry point: parse the CLI, initialise logging, run startup, and bind
//! the axum server. A thin wrapper over the `hyper_mcp_router` library crate.

use std::sync::Arc;

use anyhow::Context;
use clap::Parser;

use hyper_mcp_router::cli::{Cli, Command, ServeArgs};
use hyper_mcp_router::config;
use hyper_mcp_router::engines;
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

    // 3. Build the classifier engine(s) selected by `[classifier] model`
    //    (config-only; one, or several forming a capacity ladder). All
    //    model-specific sizing — session pools, thread counts, memory
    //    planning, overcommit warnings — is owned by each engine itself,
    //    configured via its own `[classifier.<model>]` table (see `engines/`).
    let classifiers = engines::build_roster(&cfg.classifier)
        .await
        .context("initialising classifier engines")?;
    let trivial_max_words = cfg.classifier.trivial_max_words;
    // The capacity ladder, one line per rung, smallest first. `local` makes a
    // mixed local/remote roster visible: with one, prompt LENGTH decides
    // whether prompt text is sent to a remote provider for classification.
    for (rung, engine) in classifiers.iter().enumerate() {
        tracing::info!(
            rung,
            engine = engine.name(),
            local = engine.is_local(),
            context_char_budget = engine.context_char_budget(),
            current_turn_char_budget = engine.current_turn_char_budget(),
            "classifier engine ready"
        );
    }
    if !classifiers.iter().all(|e| e.is_local()) {
        tracing::info!(
            "classification sends prompt text to a remote provider (see the ladder above); \
             only fully-local rosters keep all prompt text in-process"
        );
    }

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
    let state = AppState::new(classifiers, Arc::new(cfg), trivial_max_words)?;
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
