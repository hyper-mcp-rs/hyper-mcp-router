//! The `classifier.<model>` engine settings (`[classifier.<model>]` in TOML):
//! per-engine settings structs for every classifier engine. Split out of the
//! config root module — this file owns the *shapes* of those settings
//! (including the auth-driven Google API surface choice); parsing, env
//! expansion, key resolution, and the model catalogue stay in `config`.

use anyhow::Context;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use url::Url;

use super::{deserialize_opt_http_url, deserialize_static_secret, keyring_secret};

/// A classifier engine table's static secret (`api_key` / `access_token`):
/// either already **resolved** (plaintext or `${ENV_VAR}`-expanded, held
/// redacted as a [`SecretString`]) or an **unresolved keyring reference**.
/// Deserialization performs no I/O; `config::load` resolves the tables of
/// engines actually selected by `[classifier] model` (see
/// [`super::RouterConfig::resolve_secrets`]). Never logged — `Debug` output
/// redacts the value.
#[derive(Debug, Clone)]
pub enum StaticSecret {
    /// A resolved secret, ready to use.
    Resolved(SecretString),
    /// An OS-keyring reference, not yet looked up.
    Keyring {
        /// Keyring service name.
        service: String,
        /// Keyring user/account name.
        user: String,
    },
}

impl StaticSecret {
    /// The resolved secret value. Errs on an unresolved keyring reference —
    /// engines only ever see configs whose selected tables were resolved at
    /// load, so hitting this means the config skipped `resolve_secrets`
    /// (e.g. a caller wired an engine up manually).
    pub fn resolved(&self) -> anyhow::Result<&SecretString> {
        match self {
            StaticSecret::Resolved(secret) => Ok(secret),
            StaticSecret::Keyring { service, user } => Err(anyhow::anyhow!(
                "keyring secret (service={service}, user={user}) was never resolved; \
                 config::load resolves the tables of selected engines — call \
                 RouterConfig::resolve_secrets after parsing"
            )),
        }
    }

    /// Resolve a slot in place: a keyring reference is looked up (empty ⇒
    /// the slot becomes `None`, matching empty-plaintext behavior); resolved
    /// values pass through. `engine` names the table in errors.
    pub(crate) fn resolve_slot(
        slot: &mut Option<StaticSecret>,
        engine: &str,
    ) -> anyhow::Result<()> {
        if let Some(StaticSecret::Keyring { service, user }) = slot {
            let secret = keyring_secret(service, user)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("resolving a secret for [classifier.{engine}]"))?;
            *slot =
                (!secret.is_empty()).then(|| StaticSecret::Resolved(SecretString::from(secret)));
        }
        Ok(())
    }
}

/// Equality compares exposed values — config equality for tests and drift
/// guards, never an authentication check.
impl PartialEq for StaticSecret {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (StaticSecret::Resolved(a), StaticSecret::Resolved(b)) => {
                a.expose_secret() == b.expose_secret()
            }
            (
                StaticSecret::Keyring { service, user },
                StaticSecret::Keyring {
                    service: s2,
                    user: u2,
                },
            ) => service == s2 && user == u2,
            _ => false,
        }
    }
}

impl Eq for StaticSecret {}

impl From<&str> for StaticSecret {
    fn from(s: &str) -> Self {
        StaticSecret::Resolved(SecretString::from(s))
    }
}

impl From<String> for StaticSecret {
    fn from(s: String) -> Self {
        StaticSecret::Resolved(SecretString::from(s))
    }
}

/// Settings for a remote embedding engine on the Generative Language API
/// (the transport slice consumed by `engines/gemini`). Remote engines have no
/// local session pool; their "sessions" are concurrent in-flight API
/// requests.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteEmbeddingConfig {
    /// API key for the provider. **Required when the engine is selected**
    /// (engine construction fails at startup without it). Resolves exactly
    /// like a routed model's `api_key`: a plaintext/env-expanded string or a
    /// `{ source = "keyring", service, user }` table; an empty resolved value
    /// counts as absent. Never logged (redacted [`StaticSecret`]).
    #[serde(default, deserialize_with = "deserialize_static_secret")]
    pub api_key: Option<StaticSecret>,
    /// Endpoint override (e.g. a proxy/gateway, or a mock in tests). Must be
    /// http/https. Defaults to the provider's public endpoint.
    #[serde(default, deserialize_with = "deserialize_opt_http_url")]
    pub base_url: Option<Url>,
    /// Maximum concurrent embedding requests in flight (this engine's
    /// "session pool"). Omit for the model's default.
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    /// Per-call total timeout, seconds. Omit for the model's default.
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
}

/// Settings for the `text-embedding-005` engine
/// (`[classifier.text-embedding-005]`), which targets **Vertex AI** rather
/// than the Gemini Developer API: `text-embedding-005` is published only on
/// Vertex, so it needs a GCP `project`, a `location`, and an OAuth Bearer
/// `access_token` instead of a plain `api_key`. Also the transport slice
/// consumed by `engines/vertex` for every Vertex engine.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VertexEmbeddingConfig {
    /// GCP project id. **Required when the engine is selected** (engine
    /// construction fails at startup without it).
    #[serde(default)]
    pub project: Option<String>,
    /// Vertex AI location. **Required when the engine is selected —
    /// deliberately no default**: the location determines model availability
    /// (some of the newest models are served only at multi-region `us`/`eu`
    /// or `global`), data residency, and the endpoint host, so the operator
    /// must choose it. A hyphenated region such as `us-central1` selects the
    /// regional host (`https://{location}-aiplatform.googleapis.com`);
    /// multi-region (`us`, `eu`) and `global` locations select the bare
    /// `https://aiplatform.googleapis.com`. Ignored when `base_url` is
    /// overridden (but still required).
    #[serde(default)]
    pub location: Option<String>,
    /// Optional [quota project](https://cloud.google.com/docs/quotas/quota-project):
    /// which project's API quota is consumed and billed for the embed calls,
    /// sent as the `x-goog-user-project` header on every request (both auth
    /// modes). Mostly relevant when authenticating as a *user* via
    /// Application Default Credentials (user credentials carry no project of
    /// their own) or when deliberately charging API usage to a different
    /// project than `project`. The authenticating principal needs
    /// `serviceusage.services.use` on this project. Omit to let Google
    /// attribute quota by its defaults.
    #[serde(default)]
    pub quota_project: Option<String>,
    /// Static OAuth 2.0 Bearer access token override for the Vertex AI API.
    /// **Optional**: when omitted, the engine authenticates via Application
    /// Default Credentials (a service-account key file via
    /// `GOOGLE_APPLICATION_CREDENTIALS`, `gcloud auth application-default
    /// login` user credentials, or the GCE/Cloud Run metadata server) with
    /// token caching and refresh handled by `google-cloud-auth`. When set,
    /// this exact token is used verbatim and **never refreshed** — handy for
    /// quick tests (`gcloud auth print-access-token`), but such tokens expire
    /// in ~1h. Resolves exactly like a routed model's `api_key` (a
    /// plaintext/env-expanded string or a keyring table); an empty resolved
    /// value counts as absent. Never logged (redacted [`StaticSecret`]).
    #[serde(default, deserialize_with = "deserialize_static_secret")]
    pub access_token: Option<StaticSecret>,
    /// Endpoint override (e.g. a proxy/gateway, or a mock in tests). Must be
    /// http/https. Defaults to the regional Vertex host derived from
    /// `location`.
    #[serde(default, deserialize_with = "deserialize_opt_http_url")]
    pub base_url: Option<Url>,
    /// Maximum concurrent embedding requests in flight (this engine's
    /// "session pool"). Omit for the model's default.
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    /// Per-call total timeout, seconds. Omit for the model's default.
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
    /// Per-engine score floor for the image-generation axis (embedding
    /// similarity scale). Omit to inherit the top-level `[classifier]
    /// image_generation_threshold`. With several engines configured, prefer
    /// this per-engine key — threshold scales are engine-specific.
    #[serde(default)]
    pub image_generation_threshold: Option<f32>,
}

impl VertexEmbeddingConfig {
    /// The required-when-selected `project` and `location`, trimmed. Pure —
    /// shared by engine construction (`engines/vertex`) and the offline
    /// `validate` subcommand so the error messages stay single-sourced.
    pub fn project_and_location(&self, engine: &str) -> anyhow::Result<(&str, &str)> {
        let project = self
            .project
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "classifier engine `{engine}` requires a GCP project: set `project` in \
                     [classifier.{engine}]"
                )
            })?;
        // Deliberately no default: the location determines model availability
        // (some models are multi-region/global-only), data residency, and the
        // endpoint host, so the operator must choose it consciously.
        let location = self
            .location
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "classifier engine `{engine}` requires a `location` in [classifier.{engine}] \
                     (a Vertex region such as \"us-central1\", a multi-region such as \"us\", \
                     or \"global\")"
                )
            })?;
        Ok((project, location))
    }
}

/// Which Google API surface an engine talks to. The gemini-embedding models
/// are published on **both**; from the router's perspective these are two
/// completely different engines (different endpoint layout, wire format, and
/// auth) that happen to share a model name — see `engines/gemini/` and
/// `engines/vertex/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoogleApi {
    /// The Gemini Developer API (`generativelanguage.googleapis.com`).
    GenerativeLanguage,
    /// Vertex AI (`{location}-aiplatform.googleapis.com`).
    Vertex,
}

/// Settings for an engine whose model is published on **both** Google API
/// surfaces (`gemini-embedding-001`, `gemini-embedding-2`): the union of the
/// Generative-Language shape ([`RemoteEmbeddingConfig`]) and the Vertex shape
/// ([`VertexEmbeddingConfig`]). **The auth fields choose the surface** — see
/// [`Self::surface`]: `api_key` means Generative Language; `project` (with
/// ADC or `access_token`) means Vertex. Setting both is a startup error.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoogleEmbeddingConfig {
    /// Generative-Language credential. Setting it selects that surface.
    /// Resolves like a routed model's static key; empty counts as absent.
    /// Never logged (redacted [`StaticSecret`]).
    #[serde(default, deserialize_with = "deserialize_static_secret")]
    pub api_key: Option<StaticSecret>,
    /// GCP project id. Setting it selects the **Vertex AI** surface.
    #[serde(default)]
    pub project: Option<String>,
    /// Vertex AI location (see [`VertexEmbeddingConfig::location`]):
    /// **required on the Vertex surface, deliberately no default**. Ignored
    /// on the Generative-Language surface.
    #[serde(default)]
    pub location: Option<String>,
    /// Optional Vertex quota project (see
    /// [`VertexEmbeddingConfig::quota_project`]). Ignored on the
    /// Generative-Language surface.
    #[serde(default)]
    pub quota_project: Option<String>,
    /// Optional static Vertex token override of ADC (see
    /// [`VertexEmbeddingConfig::access_token`]). Ignored on the
    /// Generative-Language surface.
    #[serde(default, deserialize_with = "deserialize_static_secret")]
    pub access_token: Option<StaticSecret>,
    /// Endpoint override for whichever surface is selected.
    #[serde(default, deserialize_with = "deserialize_opt_http_url")]
    pub base_url: Option<Url>,
    /// Maximum concurrent embedding requests in flight.
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    /// Per-call total timeout, seconds.
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
    /// Per-engine score floor for the image-generation axis (embedding
    /// similarity scale for this engine family). Omit to inherit the
    /// top-level `[classifier] image_generation_threshold`. With several
    /// engines configured, prefer this per-engine key — threshold scales are
    /// engine-specific, so one global number cannot fit all engines.
    #[serde(default)]
    pub image_generation_threshold: Option<f32>,
}

impl GoogleEmbeddingConfig {
    /// Which API surface this table's auth selects: `api_key` ⇒ Generative
    /// Language, `project` ⇒ Vertex AI. Both or neither is a loud startup
    /// error — the surface must be unambiguous, never guessed.
    pub fn surface(&self, engine: &str) -> anyhow::Result<GoogleApi> {
        let project = self
            .project
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty());
        match (&self.api_key, project) {
            (Some(_), None) => Ok(GoogleApi::GenerativeLanguage),
            (None, Some(_)) => Ok(GoogleApi::Vertex),
            (Some(_), Some(_)) => anyhow::bail!(
                "classifier engine `{engine}`: [classifier.{engine}] sets both `api_key` \
                 (Generative Language API) and `project` (Vertex AI); set exactly one to \
                 choose the API surface"
            ),
            (None, None) => anyhow::bail!(
                "classifier engine `{engine}` requires either `api_key` (Generative \
                 Language API) or `project` (Vertex AI; auth via ADC or `access_token`) \
                 in [classifier.{engine}]"
            ),
        }
    }

    /// The Generative-Language slice of this table, for
    /// `engines/gemini`'s transport.
    pub fn to_generative_language(&self) -> RemoteEmbeddingConfig {
        RemoteEmbeddingConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            max_concurrency: self.max_concurrency,
            request_timeout_secs: self.request_timeout_secs,
        }
    }

    /// The Vertex slice of this table, for `engines/vertex`'s transport.
    pub fn to_vertex(&self) -> VertexEmbeddingConfig {
        VertexEmbeddingConfig {
            project: self.project.clone(),
            location: self.location.clone(),
            quota_project: self.quota_project.clone(),
            access_token: self.access_token.clone(),
            base_url: self.base_url.clone(),
            max_concurrency: self.max_concurrency,
            request_timeout_secs: self.request_timeout_secs,
            image_generation_threshold: self.image_generation_threshold,
        }
    }
}

/// `[classifier.deberta-v3-xsmall-zeroshot]`: settings that only make sense
/// for the embedded NLI engine (local ORT sessions). Other engines have
/// their own concurrency models and their own tables.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DebertaV3XsmallZeroshotConfig {
    /// Concurrent ORT inference sessions. Omit for auto-sizing from the
    /// detected core count and memory budget (see
    /// `planning::plan_inference`). An explicit value larger than the host
    /// can handle is honored but logs a warning.
    #[serde(default)]
    pub inference_pool_size: Option<usize>,
    /// ONNX Runtime intra-op threads per session (`0` = runtime default).
    /// Omit for auto-sizing.
    #[serde(default)]
    pub intra_op_threads: Option<usize>,
    /// Per-engine score floor for the image-generation axis (P(entailment)
    /// scale for this engine). Omit to inherit the top-level `[classifier]
    /// image_generation_threshold`. With several engines configured, prefer
    /// this per-engine key — threshold scales are engine-specific.
    #[serde(default)]
    pub image_generation_threshold: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::ClassifierModel;
    use crate::config::parse;

    #[test]
    fn gemini_engine_settings_parse_with_api_key_resolution() {
        std::env::set_var("ROUTER_TEST_GEMINI_KEY", "g-key");
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=\"gemini-embedding-001\"\n\
             [classifier.gemini-embedding-001]\napi_key=\"${ROUTER_TEST_GEMINI_KEY}\"\n\
             base_url=\"http://localhost:9\"\nmax_concurrency=8\nrequest_timeout_secs=5\n\
             [classifier.gemini-embedding-2]\napi_key=\"plain-key\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.models, [ClassifierModel::GeminiEmbedding001]);
        let g1 = &cfg.classifier.gemini_embedding_001;
        // Env-expanded, exactly like a routed model's key.
        assert_eq!(g1.api_key, Some("g-key".into()));
        assert_eq!(
            g1.base_url.as_ref().unwrap().as_str(),
            "http://localhost:9/"
        );
        assert_eq!(g1.max_concurrency, Some(8));
        assert_eq!(g1.request_timeout_secs, Some(5));
        assert_eq!(
            cfg.classifier.gemini_embedding_2.api_key,
            Some("plain-key".into())
        );
    }

    #[test]
    fn text_embedding_005_table_parses() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=\"text-embedding-005\"\n\
             [classifier.text-embedding-005]\nproject=\"my-proj\"\nlocation=\"us-east1\"\n\
             quota_project=\"billing-proj\"\naccess_token=\"te5-token\"\nmax_concurrency=16\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.models, [ClassifierModel::TextEmbedding005]);
        let te5 = &cfg.classifier.text_embedding_005;
        assert_eq!(te5.project.as_deref(), Some("my-proj"));
        assert_eq!(te5.location.as_deref(), Some("us-east1"));
        assert_eq!(te5.quota_project.as_deref(), Some("billing-proj"));
        assert_eq!(te5.access_token, Some("te5-token".into()));
        assert_eq!(te5.max_concurrency, Some(16));
    }

    #[test]
    fn text_embedding_005_omitted_fields_stay_absent() {
        // Omitted table: project/token absent, and — deliberately — NO
        // location default (the engine requires an explicit choice at build);
        // an empty access_token counts as absent.
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier.text-embedding-005]\naccess_token=\"\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        let te5 = &cfg.classifier.text_embedding_005;
        assert_eq!(te5.location, None, "location must never default");
        assert_eq!(te5.project, None);
        assert_eq!(te5.quota_project, None);
        assert_eq!(te5.access_token, None);
        assert_eq!(te5.base_url, None);
    }

    #[test]
    fn project_and_location_trims_and_requires_both() {
        let mut v = VertexEmbeddingConfig::default();

        let err = v.project_and_location("text-embedding-005").unwrap_err();
        assert!(
            err.to_string().contains("requires a GCP project"),
            "got: {err}"
        );

        // Blank-after-trim counts as absent, for both fields.
        v.project = Some("  ".into());
        let err = v.project_and_location("text-embedding-005").unwrap_err();
        assert!(
            err.to_string().contains("requires a GCP project"),
            "got: {err}"
        );

        v.project = Some(" my-proj ".into());
        let err = v.project_and_location("text-embedding-005").unwrap_err();
        assert!(
            err.to_string().contains("requires a `location`"),
            "got: {err}"
        );

        v.location = Some(" us-central1 ".into());
        let (project, location) = v.project_and_location("text-embedding-005").unwrap();
        assert_eq!((project, location), ("my-proj", "us-central1"));
    }

    #[test]
    fn google_embedding_surface_is_chosen_by_auth_fields() {
        let mut cfg = GoogleEmbeddingConfig::default();
        // Neither credential: loud error naming both options.
        let err = cfg.surface("g").unwrap_err().to_string();
        assert!(
            err.contains("either `api_key`") && err.contains("`project`"),
            "got: {err}"
        );
        // api_key alone => Generative Language.
        cfg.api_key = Some("k".into());
        assert_eq!(cfg.surface("g").unwrap(), GoogleApi::GenerativeLanguage);
        // Both => ambiguous, refused.
        cfg.project = Some("p".into());
        let err = cfg.surface("g").unwrap_err().to_string();
        assert!(
            err.contains("sets both") && err.contains("exactly one"),
            "got: {err}"
        );
        // project alone => Vertex.
        cfg.api_key = None;
        assert_eq!(cfg.surface("g").unwrap(), GoogleApi::Vertex);
        // Whitespace-only project counts as absent.
        cfg.project = Some("   ".into());
        assert!(cfg.surface("g").is_err());
    }

    #[test]
    fn gemini_table_vertex_shape_parses_and_slices() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=\"gemini-embedding-001\"\n\
             [classifier.gemini-embedding-001]\nproject=\"p\"\nlocation=\"europe-west4\"\n\
             quota_project=\"q\"\naccess_token=\"tok\"\nmax_concurrency=4\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        let g = &cfg.classifier.gemini_embedding_001;
        assert_eq!(
            g.surface("gemini-embedding-001").unwrap(),
            GoogleApi::Vertex
        );
        // The vertex slice carries everything the vertex transport needs.
        let v = g.to_vertex();
        assert_eq!(v.project.as_deref(), Some("p"));
        assert_eq!(v.location.as_deref(), Some("europe-west4"));
        assert_eq!(v.quota_project.as_deref(), Some("q"));
        assert_eq!(v.access_token, Some("tok".into()));
        assert_eq!(v.max_concurrency, Some(4));
        // And the GL slice carries the GL fields (api_key absent here).
        assert_eq!(g.to_generative_language().api_key, None);
    }

    #[test]
    fn gemini_table_defaults_and_empty_key_is_none() {
        // Omitted table: all defaults; empty api_key string counts as absent
        // (the engine then fails at startup with a clear message).
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier.gemini-embedding-001]\napi_key=\"\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.gemini_embedding_001.api_key, None);
        assert_eq!(cfg.classifier.gemini_embedding_001.base_url, None);
        assert_eq!(cfg.classifier.gemini_embedding_2.max_concurrency, None);
    }

    #[test]
    fn gemini_base_url_rejects_non_http_scheme() {
        let err = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier.gemini-embedding-001]\nbase_url=\"ftp://x\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        );
        assert!(err.is_err());
    }

    #[test]
    fn engine_settings_reject_google_adc() {
        // The engines operate differently (Gemini takes real API keys; the
        // vertex engine already defaults to ADC), so the marker is
        // routed-model-only.
        let err = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier.gemini-embedding-001]\napi_key={ source = \"google-adc\" }\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("only supported on routed models"),
            "got: {err}"
        );
    }
}
