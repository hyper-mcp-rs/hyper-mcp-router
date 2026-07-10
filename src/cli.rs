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
}
