//! Command-line interface (clap derive). The binary exposes a single `serve`
//! subcommand.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "hyper-mcp-router", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the router HTTP server.
    Serve(ServeArgs),
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Explicit config path. If provided it is used verbatim; a missing or
    /// unparseable file is fatal (no fallback to well-known locations).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Write structured JSON logs to stdout instead of the well-known rolling
    /// file location (use this for Cloud Run and other container deployments).
    #[arg(long)]
    pub log_stdout: bool,

    /// Word ceiling for the trivial-prompt fast path. A user turn at or below
    /// this length that looks like an acknowledgement/greeting (and carries no
    /// reasoning cues) skips the classifier and routes as the Fast tier. Set to
    /// 0 to disable the fast path entirely. Default mirrors
    /// `classifier::DEFAULT_TRIVIAL_MAX_WORDS`.
    #[arg(long, default_value_t = 6)]
    pub trivial_max_words: usize,

    /// Number of concurrent classifier inference sessions (pool size). Each is
    /// an independent copy of the embedded model. Omit to auto-size from the
    /// detected core count (see `classifier::plan_inference`).
    #[arg(long)]
    pub inference_pool_size: Option<usize>,

    /// ONNX Runtime intra-op threads per inference session (0 = runtime
    /// default). Omit to auto-size from the detected core count. Keep
    /// `pool_size * intra_op_threads` near the core count to avoid
    /// oversubscription.
    #[arg(long)]
    pub intra_op_threads: Option<usize>,
}
