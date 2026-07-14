//! The Vertex AI embedding engine family: shared transport here in `mod.rs`,
//! **one file per model** in this directory (`text_embedding_005.rs`), each
//! defining a [`VertexSpec`] and delegating to the shared [`VertexEmbedding`]
//! engine.
//!
//! Vertex AI is a **separate API from the Gemini Developer API** used by the
//! `gemini/` family: some embedding models (notably `text-embedding-005`) are
//! published only on Vertex. This file owns what is Vertex-specific: the
//! regional `publishers/google/models/<model>:predict` endpoint layout, the
//! `instances`/`predictions` wire format, and OAuth Bearer auth.
//!
//! The classification *method* (anchor prototypes, cosine scoring) is
//! provider-neutral and lives in `crate::engines::embedding`.
//!
//! Anchor embedding happens once at startup and **fails fast** on any API
//! problem (bad token, wrong project, unreachable endpoint), so
//! misconfiguration is a clear boot error, not a silent stream of
//! balanced-default fallbacks.
//!
//! ## Auth
//!
//! By default the engine authenticates via [Application Default Credentials]
//! (ADC), resolved by `google-cloud-auth`: a service-account key file
//! (`GOOGLE_APPLICATION_CREDENTIALS`), `gcloud auth application-default
//! login` user credentials, or the GCE/Cloud Run metadata server — with
//! token caching and refresh handled by the library. Setting `access_token`
//! in the engine's config table overrides ADC with a single **static** token
//! (useful for quick tests; such tokens expire in ~1h and are never
//! refreshed). See [`TokenSource`].
//!
//! An optional `quota_project` attributes API-call quota/billing to a chosen
//! project via the `x-goog-user-project` header — sent with every request in
//! **both** auth modes. Mostly relevant with user-credential ADC (user
//! credentials carry no project) or deliberate cross-project billing; the
//! principal needs `serviceusage.services.use` on that project.
//!
//! [Application Default Credentials]:
//!     https://cloud.google.com/docs/authentication/application-default-credentials
//!
//! ## Privacy and failure notes
//!
//! Prompt text (the classification window and current turn) is sent to the
//! Vertex AI API. Per-request failures degrade to the balanced default via the
//! proxy, as with every engine. Error messages never echo user content, and
//! the access token is never logged.

pub mod text_embedding_005;

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use reqwest::header::HeaderValue;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::classifier::{Classification, ClassifierEngine};
use crate::config::VertexEmbeddingConfig;
use crate::engines::embedding::{anchor_texts, build_prototypes, combine_similarities, Prototypes};
use crate::gcp_auth::{self, AccessTokenCredentials};

/// Task type sent with every embed request; anchors and premises must use the
/// same one for their similarities to be comparable.
const TASK_TYPE: &str = "SEMANTIC_SIMILARITY";

/// Header carrying the [quota project](https://cloud.google.com/docs/quotas/quota-project):
/// which project's API quota is consumed and billed for the call.
const QUOTA_PROJECT_HEADER: &str = "x-goog-user-project";

/// How the engine obtains a Bearer token for the Vertex AI API: a static,
/// operator-supplied token (the `access_token` config override), or
/// Application Default Credentials (the default), whose token caching and
/// refresh are handled by `google-cloud-auth`.
enum TokenSource {
    /// A single operator-supplied token; used verbatim, never refreshed.
    Static(String),
    /// Application Default Credentials; a current token is resolved per call
    /// (cached and auto-refreshed by the auth library).
    Adc(AccessTokenCredentials),
}

impl TokenSource {
    /// Build the ADC-backed source (used when no static `access_token` is
    /// configured), via the shared [`crate::gcp_auth`] support. Credential
    /// *discovery* problems surface here; a broken or expired credential
    /// surfaces on the first token fetch — either way, startup anchor
    /// embedding fails fast with an actionable error.
    fn adc(engine: &str) -> anyhow::Result<Self> {
        let credentials = gcp_auth::adc_credentials().with_context(|| {
            format!(
                "classifier engine `{engine}`: no static `access_token` is set in \
                 [classifier.{engine}], so Application Default Credentials are required \
                 (or configure `access_token`)"
            )
        })?;
        Ok(TokenSource::Adc(credentials))
    }

    /// Resolve the Bearer token for one request. The ADC arm awaits the auth
    /// library, which serves a cached token until it nears expiry.
    async fn bearer(&self) -> anyhow::Result<String> {
        match self {
            TokenSource::Static(token) => Ok(token.clone()),
            TokenSource::Adc(credentials) => gcp_auth::bearer(credentials).await,
        }
    }

    /// Auth mode label for the startup log (never the token itself).
    fn mode(&self) -> &'static str {
        match self {
            TokenSource::Static(_) => "static-token",
            TokenSource::Adc(_) => "adc",
        }
    }
}

/// Everything that differs between Vertex embedding models. Each model file
/// declares one of these as a `const`.
pub struct VertexSpec {
    /// Engine id (matches `ClassifierModel::as_str`).
    pub name: &'static str,
    /// Publisher model id sent in the endpoint path, e.g. `text-embedding-005`
    /// (no `models/` prefix — that is part of the URL template).
    pub api_model: &'static str,
    /// Char budget for the complexity window (sized to the model's input
    /// token limit).
    pub context_char_budget: usize,
    /// Char budget for the current turn (image premise / lexical prefilter).
    pub current_turn_char_budget: usize,
    /// Default max concurrent embedding requests (the "session pool").
    pub default_max_concurrency: usize,
    /// Default per-call timeout, seconds.
    pub default_request_timeout_secs: u64,
}

/// A remote Vertex AI embedding engine (shared by every Vertex model file).
pub struct VertexEmbedding {
    spec: &'static VertexSpec,
    http: reqwest::Client,
    /// Full regional `:predict` URL.
    url: String,
    token: TokenSource,
    /// Pre-validated `x-goog-user-project` header value, when configured.
    quota_project: Option<HeaderValue>,
    /// Bounds concurrent in-flight embedding requests — this engine's
    /// equivalent of the embedded engine's session pool.
    permits: Semaphore,
    /// Cosine-similarity floor for the image axis (see
    /// `crate::engines::embedding`).
    image_gen_threshold: f32,
    prototypes: Prototypes,
}

impl VertexEmbedding {
    /// Build the engine: validate config (`project` is **required** when the
    /// engine is selected; auth is ADC unless a static `access_token` is
    /// set), construct the HTTP client, and embed the class anchors —
    /// failing fast on any API problem.
    pub async fn connect(
        spec: &'static VertexSpec,
        cfg: &VertexEmbeddingConfig,
        image_gen_threshold: f32,
    ) -> anyhow::Result<Self> {
        let project = cfg
            .project
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "classifier engine `{}` requires a GCP project: set `project` in \
                     [classifier.{}]",
                    spec.name,
                    spec.name,
                )
            })?;

        // Static token if configured, otherwise Application Default
        // Credentials — built before the HTTP client so a credential
        // discovery problem is the first thing to fail.
        let token = match cfg.access_token.clone() {
            Some(token) => TokenSource::Static(token),
            None => TokenSource::adc(spec.name)?,
        };

        let quota_project = parse_quota_project(spec.name, cfg.quota_project.as_deref())?;

        let location = cfg.location.trim();
        if location.is_empty() {
            anyhow::bail!(
                "classifier engine `{}` has an empty `location` in [classifier.{}]",
                spec.name,
                spec.name,
            );
        }

        let base = cfg
            .base_url
            .as_ref()
            .map(|u| u.as_str().trim_end_matches('/').to_string())
            .unwrap_or_else(|| format!("https://{location}-aiplatform.googleapis.com"));
        let url = build_predict_url(&base, project, location, spec.api_model);

        let timeout = cfg
            .request_timeout_secs
            .unwrap_or(spec.default_request_timeout_secs);
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(timeout))
            .build()?;

        let max_concurrency = cfg
            .max_concurrency
            .unwrap_or(spec.default_max_concurrency)
            .max(1);

        let mut engine = VertexEmbedding {
            spec,
            http,
            url,
            token,
            quota_project,
            permits: Semaphore::new(max_concurrency),
            image_gen_threshold,
            prototypes: Prototypes::default(),
        };

        // Embed every anchor in one batch and pool per class. Startup is the
        // right place to fail on a bad token or unreachable endpoint.
        let anchors = anchor_texts();
        let embeddings = engine
            .embed_batch(&anchors)
            .await
            .with_context(|| format!("embedding class anchors for `{}`", spec.name))?;
        engine.prototypes = build_prototypes(&embeddings)?;

        tracing::info!(
            engine = spec.name,
            project,
            location,
            auth = engine.token.mode(),
            quota_project = engine
                .quota_project
                .as_ref()
                .and_then(|v| v.to_str().ok())
                .unwrap_or("(unset)"),
            embedding_dims = engine.prototypes.dims(),
            anchor_count = anchors.len(),
            max_concurrency,
            request_timeout_secs = timeout,
            "vertex embedding engine ready"
        );
        Ok(engine)
    }

    /// Embed `texts` in one `:predict` call (an `instances` array batches every
    /// text into one request), bounded by the concurrency semaphore. Never
    /// echoes the texts into errors.
    async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("embedding concurrency semaphore closed"))?;

        let token = self.token.bearer().await?;
        let body = build_predict_request(texts);
        let mut request = self.http.post(&self.url).bearer_auth(token).json(&body);
        if let Some(quota_project) = &self.quota_project {
            request = request.header(QUOTA_PROJECT_HEADER, quota_project.clone());
        }
        let resp = request
            .send()
            .await
            .context("vertex embeddings request failed")?;

        let status = resp.status();
        if !status.is_success() {
            // The error body is the API's, not user content; truncate anyway.
            let detail: String = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(300)
                .collect();
            anyhow::bail!("vertex embeddings returned {status}: {detail}");
        }

        let value: Value = resp
            .json()
            .await
            .context("vertex embeddings response was not JSON")?;
        parse_embeddings(&value, texts.len())
    }
}

#[async_trait]
impl ClassifierEngine for VertexEmbedding {
    fn name(&self) -> &'static str {
        self.spec.name
    }

    fn context_char_budget(&self) -> usize {
        self.spec.context_char_budget
    }

    fn current_turn_char_budget(&self) -> usize {
        self.spec.current_turn_char_budget
    }

    async fn classify(
        &self,
        complexity_premise: &str,
        image_premise: &str,
        lexical_image_match: bool,
    ) -> anyhow::Result<Classification> {
        // An empty current turn cannot carry image intent; don't send an empty
        // text to the API (the embeddings endpoint rejects empty content).
        let image_text = image_premise.trim();
        let texts: Vec<&str> = if image_text.is_empty() {
            vec![complexity_premise]
        } else {
            vec![complexity_premise, image_text]
        };

        let embeddings = self.embed_batch(&texts).await?;
        let image_embedding = if image_text.is_empty() {
            None
        } else {
            Some(embeddings[1].as_slice())
        };

        Ok(combine_similarities(
            &embeddings[0],
            image_embedding,
            &self.prototypes,
            lexical_image_match,
            self.image_gen_threshold,
        ))
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Wire format (network-free, unit-testable)
// ───────────────────────────────────────────────────────────────────────────

/// Validate the optional quota project into a ready-to-send header value:
/// whitespace is trimmed, empty counts as absent, and a value that cannot be
/// an HTTP header fails here at startup — never at request time.
fn parse_quota_project(
    engine: &str,
    quota_project: Option<&str>,
) -> anyhow::Result<Option<HeaderValue>> {
    quota_project
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .map(|q| {
            HeaderValue::from_str(q).map_err(|_| {
                anyhow::anyhow!(
                    "classifier engine `{engine}` has an invalid `quota_project` in \
                     [classifier.{engine}]: must be a plain ASCII project id"
                )
            })
        })
        .transpose()
}

/// Assemble the regional `:predict` endpoint URL. `base` is already trimmed of
/// any trailing slash.
fn build_predict_url(base: &str, project: &str, location: &str, api_model: &str) -> String {
    format!(
        "{base}/v1/projects/{project}/locations/{location}/publishers/google/models/{api_model}:predict"
    )
}

/// Build the `:predict` request body. An `instances` array batches every text
/// into one call; `autoTruncate` guards against an over-long input silently
/// erroring instead of being clipped to the model's token limit.
fn build_predict_request(texts: &[&str]) -> Value {
    let instances: Vec<Value> = texts
        .iter()
        .map(|t| {
            json!({
                "content": t,
                "task_type": TASK_TYPE,
            })
        })
        .collect();
    json!({
        "instances": instances,
        "parameters": { "autoTruncate": true },
    })
}

/// Extract `expected` embedding vectors from a `:predict` response. Vertex
/// returns one prediction per instance, in input order, each carrying
/// `embeddings.values`.
fn parse_embeddings(value: &Value, expected: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let predictions = value
        .get("predictions")
        .and_then(|p| p.as_array())
        .ok_or_else(|| anyhow::anyhow!("vertex response missing `predictions` array"))?;
    if predictions.len() != expected {
        anyhow::bail!(
            "vertex returned {} predictions, expected {expected}",
            predictions.len()
        );
    }
    predictions
        .iter()
        .map(|p| {
            let values = p
                .get("embeddings")
                .and_then(|e| e.get("values"))
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("vertex prediction missing `embeddings.values`"))?;
            if values.is_empty() {
                anyhow::bail!("vertex returned an empty embedding vector");
            }
            Ok(values
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn static_token_source_returns_the_token() {
        let source = TokenSource::Static("tok-123".to_string());
        assert_eq!(source.bearer().await.unwrap(), "tok-123");
        assert_eq!(source.mode(), "static-token");
    }

    #[test]
    fn quota_project_parses_trims_and_rejects() {
        // Absent and empty (incl. whitespace-only) mean "unset".
        assert_eq!(parse_quota_project("e", None).unwrap(), None);
        assert_eq!(parse_quota_project("e", Some("")).unwrap(), None);
        assert_eq!(parse_quota_project("e", Some("   ")).unwrap(), None);
        // Valid ids pass through trimmed.
        let v = parse_quota_project("e", Some("  my-billing-proj ")).unwrap();
        assert_eq!(v.unwrap().to_str().unwrap(), "my-billing-proj");
        // A value that cannot be an HTTP header fails fast with the engine id.
        let err = parse_quota_project("text-embedding-005", Some("bad\nvalue"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("quota_project") && err.contains("text-embedding-005"));
    }

    #[test]
    fn predict_url_is_the_regional_publisher_path() {
        let url = build_predict_url(
            "https://us-central1-aiplatform.googleapis.com",
            "my-proj",
            "us-central1",
            "text-embedding-005",
        );
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/\
             us-central1/publishers/google/models/text-embedding-005:predict"
        );
    }

    #[test]
    fn predict_request_shape() {
        let body = build_predict_request(&["a", "b"]);
        let instances = body["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0]["content"], "a");
        assert_eq!(instances[1]["content"], "b");
        assert_eq!(instances[0]["task_type"], "SEMANTIC_SIMILARITY");
        assert_eq!(body["parameters"]["autoTruncate"], true);
    }

    #[test]
    fn parse_embeddings_happy_path_and_errors() {
        let ok = serde_json::json!({"predictions": [
            {"embeddings": {"values": [1.0, 2.0], "statistics": {"token_count": 2}}},
            {"embeddings": {"values": [3.0, 4.0]}},
        ]});
        let parsed = parse_embeddings(&ok, 2).unwrap();
        assert_eq!(parsed, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // Wrong count, missing array, missing values, and empty vectors are
        // all loud errors.
        assert!(parse_embeddings(&ok, 3).is_err());
        assert!(parse_embeddings(&serde_json::json!({}), 1).is_err());
        let no_values = serde_json::json!({"predictions": [{"embeddings": {}}]});
        assert!(parse_embeddings(&no_values, 1).is_err());
        let empty = serde_json::json!({"predictions": [{"embeddings": {"values": []}}]});
        assert!(parse_embeddings(&empty, 1).is_err());
    }
}
