//! Shared Google Cloud authentication: Application Default Credentials (ADC)
//! discovery and per-call Bearer-token resolution, used by every part of the
//! router that talks to a Google Cloud API — the `engines/vertex/` classifier
//! family and `google-adc`-authenticated routed models (`[[models]] api_key =
//! { source = "google-adc" }`).
//!
//! ADC discovery (a service-account key file via
//! `GOOGLE_APPLICATION_CREDENTIALS`, `gcloud auth application-default login`
//! user credentials, or the GCE/Cloud Run metadata server) happens once, at
//! construction; tokens are then fetched per call with caching and refresh
//! handled entirely by `google-cloud-auth`, so long-running processes never
//! hold an expiring token.
//!
//! Testing note: everything here reads the *host environment* by design.
//! Hermetic unit tests below cover the discovery *failure* paths by pointing
//! `GOOGLE_APPLICATION_CREDENTIALS` (which takes precedence in ADC discovery)
//! at broken inputs — deterministic on any host, no network. The happy-path
//! token fetch is not unit-testable hermetically and remains covered by the
//! callers — the static-token paths in the mock integration tests, and the
//! ADC path in the opt-in live test (`tests/vertex_live.rs`).

use anyhow::Context;
use google_cloud_auth::credentials::Builder as AdcBuilder;

pub use google_cloud_auth::credentials::AccessTokenCredentials;

/// OAuth scope requested for ADC tokens; the standard scope covering Google
/// Cloud APIs (including Vertex AI).
pub const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Discover Application Default Credentials, scoped for Google Cloud APIs.
/// Discovery reads the environment only (no token is fetched yet); a broken
/// or missing credential setup fails here with an actionable message, so
/// callers running at startup fail fast.
pub fn adc_credentials() -> anyhow::Result<AccessTokenCredentials> {
    AdcBuilder::default()
        .with_scopes([CLOUD_PLATFORM_SCOPE])
        .build_access_token_credentials()
        .context(
            "Application Default Credentials could not be loaded (set \
             GOOGLE_APPLICATION_CREDENTIALS to a service-account key file, or run \
             `gcloud auth application-default login`)",
        )
}

/// Fetch a current Bearer token from `credentials`. Cheap when the cached
/// token is fresh; the auth library refreshes transparently near expiry.
pub async fn bearer(credentials: &AccessTokenCredentials) -> anyhow::Result<String> {
    Ok(credentials
        .access_token()
        .await
        .context("fetching a Google Cloud access token")?
        .token)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    use super::*;

    /// Env-var mutation is process-global and `cargo test` runs tests in
    /// parallel; every test that touches the environment must hold this lock
    /// for its whole body.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that sets an env var and restores the prior state (previous
    /// value, or unset) on drop — including on panic — so tests can never
    /// leak `GOOGLE_APPLICATION_CREDENTIALS` into each other or into a
    /// developer's real ADC setup on the host.
    struct EnvVarGuard {
        key: &'static str,
        prior: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let prior = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prior }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prior {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    /// RAII guard for a temp file: removed on drop, even on panic.
    struct TempFile(PathBuf);

    impl TempFile {
        fn write(name: &str, contents: &str) -> Self {
            // Unique per test process so parallel `cargo test` runs of the
            // same suite can't collide.
            let path =
                std::env::temp_dir().join(format!("gcp-auth-test-{}-{name}", std::process::id()));
            std::fs::write(&path, contents).expect("write temp credentials file");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Full error chain as a string (`{:#}` renders every context layer).
    fn chain_text(err: &anyhow::Error) -> String {
        format!("{err:#}")
    }

    fn assert_actionable(err: &anyhow::Error) {
        let text = chain_text(err);
        assert!(
            text.contains("GOOGLE_APPLICATION_CREDENTIALS"),
            "error should mention GOOGLE_APPLICATION_CREDENTIALS, got: {text}"
        );
        assert!(
            text.contains("gcloud auth application-default login"),
            "error should mention the gcloud fallback, got: {text}"
        );
    }

    #[test]
    fn cloud_platform_scope_is_pinned() {
        assert_eq!(
            CLOUD_PLATFORM_SCOPE,
            "https://www.googleapis.com/auth/cloud-platform"
        );
    }

    #[test]
    fn nonexistent_credentials_file_fails_with_actionable_context() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let _env = EnvVarGuard::set(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "/nonexistent/hyper-mcp-router-gcp-auth-test.json",
        );

        let err = adc_credentials().expect_err("discovery must fail for a missing key file");
        assert_actionable(&err);
    }

    #[test]
    fn malformed_credentials_file_fails_with_actionable_context() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let file = TempFile::write("malformed.json", "this is not json{");
        let _env = EnvVarGuard::set("GOOGLE_APPLICATION_CREDENTIALS", file.path());

        let err = adc_credentials().expect_err("discovery must fail for a malformed key file");
        assert_actionable(&err);
    }
}
