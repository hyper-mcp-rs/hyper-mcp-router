//! Command-line interface (clap derive). The binary exposes a `serve`
//! subcommand and an offline `validate` subcommand.
//!
//! Deliberately minimal: the CLI selects **where the config lives** — nothing
//! else. Logs always go to stdout (redirect with the shell); all routing and
//! classifier behaviour (including which classifier model runs, and its
//! model-specific tuning) lives in the config file: different classifier
//! models mean different configurations, so the config file is the single
//! source of truth.

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
    fn serve_subcommand_parses_with_and_without_config() {
        let cli = Cli::try_parse_from(["hyper-mcp-router", "serve"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected the serve subcommand");
        };
        assert_eq!(args.config, None);

        let cli = Cli::try_parse_from(["hyper-mcp-router", "serve", "--config", "c.toml"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected the serve subcommand");
        };
        assert_eq!(args.config, Some(PathBuf::from("c.toml")));
    }

    #[test]
    fn removed_log_stdout_flag_is_rejected() {
        // Logs always go to stdout now; the old flag must fail loudly rather
        // than silently parse, so stale deployment scripts surface at once.
        let err = Cli::try_parse_from(["hyper-mcp-router", "serve", "--log-stdout"]);
        assert!(err.is_err());
    }
}
