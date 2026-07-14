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
//! Testing note: everything here reads the *host environment* by design, so
//! there are no hermetic unit tests in this module. Coverage comes from the
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
