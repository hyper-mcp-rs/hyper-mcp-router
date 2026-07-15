//! Command-line interface (clap derive). The binary exposes a `serve`
//! subcommand and an offline `validate` subcommand.
//!
//! Deliberately minimal: the CLI selects **where the config lives and where
//! logs go** — nothing else. All routing and classifier behaviour (including
//! which classifier model runs, and its model-specific tuning) lives in the
//! config file: different classifier models mean different configurations, so
//! the config file is the single source of truth.

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
    /// Load and validate a config file, then exit — without starting the
    /// server, initialising classifier engines, or touching the network.
    Validate(ValidateArgs),
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

#[derive(Debug, Args)]
pub struct ValidateArgs {
    /// Explicit config path. If provided it is used verbatim; otherwise the
    /// well-known locations are probed, exactly as `serve` would.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_subcommand_parses_with_and_without_config() {
        let cli = Cli::try_parse_from(["hyper-mcp-router", "validate"]).unwrap();
        let Command::Validate(args) = cli.command else {
            panic!("expected the validate subcommand");
        };
        assert_eq!(args.config, None);

        let cli =
            Cli::try_parse_from(["hyper-mcp-router", "validate", "--config", "c.toml"]).unwrap();
        let Command::Validate(args) = cli.command else {
            panic!("expected the validate subcommand");
        };
        assert_eq!(args.config, Some(PathBuf::from("c.toml")));
    }

    #[test]
    fn validate_subcommand_rejects_serve_only_flags() {
        // `--log-stdout` selects where logs go; validate never initialises
        // logging, so the flag must not silently parse there.
        let err = Cli::try_parse_from(["hyper-mcp-router", "validate", "--log-stdout"]);
        assert!(err.is_err());
    }
}
