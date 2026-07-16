//! Centralised structured logging: JSON (NDJSON) on **stdout**, always, with
//! an `EnvFilter` defaulting to `info` and a panic hook routing panics into
//! the log stream.
//!
//! The router is a standalone process, so it never manages log files itself —
//! the operator redirects stdout with the shell
//! (`hyper-mcp-router serve > router.log`), a service manager (systemd's
//! journal), or a container runtime, all of which do rotation and shipping
//! better than we could.

use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

/// Default filter directives when `RUST_LOG` is unset: `info`, with ONNX
/// Runtime's very verbose per-session graph-transform logging quieted. A set
/// `RUST_LOG` overrides this entirely.
const DEFAULT_DIRECTIVES: &str = "info,ort=warn";

/// Initialise structured JSON logging on stdout and install the panic hook.
/// Call once at startup, before anything logs.
pub fn init() {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_DIRECTIVES));

    tracing_subscriber::fmt()
        .json()
        .with_env_filter(env_filter)
        .with_writer(std::io::stdout)
        .with_ansi(false)
        .with_target(true)
        .with_line_number(true)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    install_panic_hook();
}

/// Route panics into the structured log stream, then chain the default hook.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown".to_string());
        let payload = payload_string(info.payload());
        tracing::error!(panic.payload = %payload, panic.location = %location, "panic");
        default_hook(info);
    }));
}

/// Best-effort text of a panic payload: panics carry `&str` (literal messages)
/// or `String` (formatted messages); anything else gets a placeholder rather
/// than being dropped from the log.
fn payload_string(payload: &dyn std::any::Any) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_directives_quiet_ort_at_info() {
        // The default must parse as valid EnvFilter directives (a typo here
        // would silently fall back to the subscriber's own default) and keep
        // the two intentional decisions: info baseline, ort quieted.
        let filter: EnvFilter = DEFAULT_DIRECTIVES.parse().expect("directives parse");
        let rendered = filter.to_string();
        assert!(rendered.contains("info"), "got: {rendered}");
        assert!(rendered.contains("ort=warn"), "got: {rendered}");
    }

    #[test]
    fn panic_payload_str_and_string_extracted() {
        let s: &dyn std::any::Any = &"literal message";
        assert_eq!(payload_string(s), "literal message");

        let owned: &dyn std::any::Any = &String::from("formatted message");
        assert_eq!(payload_string(owned), "formatted message");
    }

    #[test]
    fn panic_payload_other_types_get_placeholder() {
        let n: &dyn std::any::Any = &42u32;
        assert_eq!(payload_string(n), "non-string panic payload");
    }
}
