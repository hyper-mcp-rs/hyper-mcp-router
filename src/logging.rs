//! Centralised structured logging: JSON (NDJSON) output to a rolling daily file
//! (default) or stdout (`--log-stdout`), an `EnvFilter` defaulting to `info`,
//! and a panic hook routing panics into the log stream.

use std::path::{Path, PathBuf};

use directories::ProjectDirs;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

const APP_NAME: &str = "hyper-mcp-router";

/// Where structured logs are written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogSink {
    /// JSON lines to stdout.
    Stdout,
    /// Rolling daily `router.log` inside this directory.
    File(PathBuf),
}

/// Pure resolution of `(--log-stdout, ROUTER_LOG_PATH, config dir)` to a sink:
///
/// - `--log-stdout` always selects stdout.
/// - otherwise `ROUTER_LOG_PATH` overrides the default directory.
/// - otherwise the default is `{config dir}/logs`.
///
/// Split out (and taking its inputs as arguments) so it is unit-testable
/// without touching the real environment or filesystem.
pub fn resolve_log_sink(
    log_stdout: bool,
    env_path: Option<PathBuf>,
    config_dir: Option<&Path>,
) -> anyhow::Result<LogSink> {
    if log_stdout {
        return Ok(LogSink::Stdout);
    }
    if let Some(dir) = env_path {
        return Ok(LogSink::File(dir));
    }
    if let Some(dir) = config_dir {
        return Ok(LogSink::File(dir.join("logs")));
    }
    anyhow::bail!("could not determine a log directory; set ROUTER_LOG_PATH or use --log-stdout")
}

/// Initialise structured logging and install the panic hook. The returned
/// [`WorkerGuard`] **must** be held for the process lifetime so buffered lines
/// flush on exit.
pub fn init(log_stdout: bool) -> anyhow::Result<WorkerGuard> {
    let env_path = std::env::var_os("ROUTER_LOG_PATH").map(PathBuf::from);
    let config_dir = ProjectDirs::from("", "", APP_NAME).map(|p| p.config_dir().to_path_buf());
    let sink = resolve_log_sink(log_stdout, env_path, config_dir.as_deref())?;

    // Default to `info`, but quiet ONNX Runtime's very verbose per-session
    // graph-transform logging. `RUST_LOG`, when set, overrides this entirely.
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));

    let (writer, guard) = match sink {
        LogSink::Stdout => tracing_appender::non_blocking(std::io::stdout()),
        LogSink::File(dir) => {
            std::fs::create_dir_all(&dir).map_err(|e| {
                anyhow::anyhow!("failed to create log directory `{}`: {e}", dir.display())
            })?;
            let appender = tracing_appender::rolling::daily(&dir, "router.log");
            tracing_appender::non_blocking(appender)
        }
    };

    tracing_subscriber::fmt()
        .json()
        .with_env_filter(env_filter)
        .with_writer(writer)
        .with_ansi(false)
        .with_target(true)
        .with_line_number(true)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    install_panic_hook();
    Ok(guard)
}

/// Route panics into the structured log stream, then chain the default hook.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown".to_string());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };
        tracing::error!(panic.payload = %payload, panic.location = %location, "panic");
        default_hook(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_selects_stdout() {
        let sink = resolve_log_sink(
            true,
            Some(PathBuf::from("/ignored")),
            Some(Path::new("/config")),
        )
        .unwrap();
        assert_eq!(sink, LogSink::Stdout);
    }

    #[test]
    fn env_path_overrides_default() {
        let sink = resolve_log_sink(
            false,
            Some(PathBuf::from("/var/log/router")),
            Some(Path::new("/config")),
        )
        .unwrap();
        assert_eq!(sink, LogSink::File(PathBuf::from("/var/log/router")));
    }

    #[test]
    fn default_resolves_under_config_dir() {
        let sink = resolve_log_sink(false, None, Some(Path::new("/config/app"))).unwrap();
        assert_eq!(sink, LogSink::File(PathBuf::from("/config/app/logs")));
    }

    #[test]
    fn no_dir_available_is_error() {
        assert!(resolve_log_sink(false, None, None).is_err());
    }
}
