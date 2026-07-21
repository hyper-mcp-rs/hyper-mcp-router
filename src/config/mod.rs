//! Configuration: the config **schema** — typed structs, custom
//! deserializers, API-key resolution (plaintext / env / keyring), the model
//! catalogue, and startup coverage validation. Loading — path discovery,
//! TOML/YAML/JSON format selection, env expansion, parsing — lives in the
//! `load` submodule; the model-selection *policy* over the parsed catalogue
//! lives in `crate::selection`.
//!
//! Requests and responses elsewhere are handled as raw JSON; this module is the
//! only place typed structs are used, and only for the operator's config file.
//!
//! ## Secrets
//!
//! Every resolved secret is held as a [`secrecy::SecretString`]: `Debug`
//! prints `[REDACTED]`, so a stray `tracing::debug!(?cfg)` can never leak a
//! credential, and every real access is an explicit, greppable
//! `expose_secret()` call. Keyring references are parsed as **unresolved
//! markers** and looked up only by [`RouterConfig::resolve_secrets`] after
//! parsing — deserialization performs no I/O, and engine tables are resolved
//! only for engines actually selected by `[classifier] model`.

use std::fmt;
use std::num::NonZeroU64;

use anyhow::Context;
use secrecy::{ExposeSecret, SecretString};
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::Deserialize;
use url::Url;

use crate::classifier::{ClassifierModel, ModelTier, DEFAULT_IMAGE_GEN_THRESHOLD};
use crate::modality::{Modality, ModalitySet};
use crate::prompt::DEFAULT_TRIVIAL_MAX_WORDS;

mod engines;
pub use engines::{
    DebertaV3XsmallZeroshotConfig, GoogleApi, GoogleEmbeddingConfig, RemoteEmbeddingConfig,
    StaticSecret, VertexEmbeddingConfig,
};

mod load;
pub use load::{expand_env, load, parse, parse_with_format, resolve_config_path};

/// Application identifier used for OS config/log directory discovery.
const APP_NAME: &str = "hyper-mcp-router";

// ───────────────────────────────────────────────────────────────────────────
// Config structs
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterConfig {
    pub server: ServerConfig,
    #[serde(default)]
    pub classifier: ClassifierConfig,
    /// Logging behavior beyond the `RUST_LOG` level — see [`LoggingConfig`].
    #[serde(default)]
    pub logging: LoggingConfig,
    /// OpenTelemetry export — **optional**: absent means telemetry is fully
    /// off (no exporters, no background tasks, no sockets). See
    /// [`TelemetryConfig`].
    #[serde(default)]
    pub telemetry: Option<TelemetryConfig>,
    pub models: Vec<ModelConfig>,
}

/// `[logging]` — content controls for the log stream. Verbosity is `RUST_LOG`
/// (an environment concern); whether **user content** may appear in logs is a
/// deployment policy, so it lives in the config file instead of being coupled
/// to a log level.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Emit the info-level `"completion request"` event carrying each
    /// request's ENTIRE current-turn prompt and compiled classification
    /// window, alongside the routing decision. **Default `false`** — this is
    /// the only path by which user content reaches the logs, and it is
    /// independent of `RUST_LOG`: a deployment can log every prompt without
    /// enabling debug noise, or run full debug diagnostics without ever
    /// logging a prompt. Treat logs produced under this flag as customer
    /// data.
    #[serde(default)]
    pub log_prompts: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Upstream connect timeout, seconds.
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,
    /// Upstream total request timeout for **non-streaming** requests, seconds.
    /// Streaming responses have no total deadline (an SSE stream may
    /// legitimately outlive any fixed budget); they are guarded by
    /// `stream_idle_timeout_secs` instead.
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
    /// Upstream idle (per-read) timeout, seconds: abort when no bytes arrive
    /// for this long. This is what bounds a stalled streaming response.
    /// `0` disables the idle guard.
    #[serde(default = "default_stream_idle_timeout")]
    pub stream_idle_timeout_secs: u64,
    /// Maximum accepted request body size, bytes. Base64-encoded image/audio/
    /// file payloads are large; the default (32 MiB) comfortably covers them.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: default_host(),
            port: default_port(),
            connect_timeout_secs: default_connect_timeout(),
            request_timeout_secs: default_request_timeout(),
            stream_idle_timeout_secs: default_stream_idle_timeout(),
            max_body_bytes: default_max_body_bytes(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}

fn default_connect_timeout() -> u64 {
    10
}
fn default_request_timeout() -> u64 {
    600
}
fn default_stream_idle_timeout() -> u64 {
    300
}
fn default_max_body_bytes() -> usize {
    32 * 1024 * 1024
}

/// `[telemetry]` — OTLP trace/metric export over **HTTP/protobuf**. The
/// endpoint carries **no credentials**: it is meant for a local collector
/// sidecar (`http://localhost:4318`) or an equally trusted network hop; the
/// collector owns vendor authentication (ADC on Cloud Run, task role on ECS).
/// Header-based auth for direct-to-vendor export is deliberately out of scope
/// for now.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    /// OTLP/HTTP base endpoint. Signal paths (`/v1/traces`, `/v1/metrics`)
    /// are appended automatically — configure the root, not a signal URL.
    #[serde(deserialize_with = "deserialize_http_url")]
    pub otlp_endpoint: Url,
    /// `service.name` resource attribute on every span and metric.
    #[serde(default = "default_service_name")]
    pub service_name: String,
    /// Export spans. `false` keeps metrics-only telemetry.
    #[serde(default = "default_true")]
    pub traces: bool,
    /// Export metrics. `false` keeps trace-only telemetry.
    #[serde(default = "default_true")]
    pub metrics: bool,
    /// Head-sampling ratio in `[0.0, 1.0]`, applied **independently of the
    /// caller's sampling decision** by default — platform ingress tracing
    /// (e.g. Cloud Run's ~0.1 req/s) would otherwise silently drop nearly
    /// every router span. Trace IDs are still inherited, so sampled router
    /// spans join the caller's trace whenever both are kept.
    #[serde(default = "default_sample_ratio")]
    pub sample_ratio: f64,
    /// Respect the incoming `traceparent` sampled flag instead of sampling
    /// independently (the OTel-conventional `ParentBased` sampler). Leave
    /// `false` on platforms whose ingress samples aggressively.
    #[serde(default)]
    pub parent_based_sampling: bool,
    /// Metric export interval, seconds.
    #[serde(default = "default_metrics_interval")]
    pub metrics_interval_secs: u64,
}

impl TelemetryConfig {
    /// Field-level validation, called from [`RouterConfig::validate`].
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&self.sample_ratio) {
            anyhow::bail!(
                "[telemetry] sample_ratio must be within 0.0..=1.0, got {}",
                self.sample_ratio
            );
        }
        if !self.traces && !self.metrics {
            anyhow::bail!("[telemetry] disables both traces and metrics; remove the table instead");
        }
        if self.metrics_interval_secs == 0 {
            anyhow::bail!("[telemetry] metrics_interval_secs must be at least 1");
        }
        Ok(())
    }
}

fn default_service_name() -> String {
    APP_NAME.to_string()
}
fn default_true() -> bool {
    true
}
fn default_sample_ratio() -> f64 {
    1.0
}
fn default_metrics_interval() -> u64 {
    60
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifierConfig {
    /// Which classification model(s) to run — a single id, or a **list**
    /// forming a capacity ladder (see `classifier::EngineRoster`): engines
    /// are ordered by their context budget and each request is classified by
    /// the smallest engine whose budget covers its classification window.
    /// **Config-only** — there is no CLI override, because each model brings
    /// its own configuration; different models mean different config files.
    /// Defaults to the embedded `deberta-v3-xsmall-zeroshot` model.
    #[serde(
        default = "default_classifier_models",
        rename = "model",
        deserialize_with = "one_or_many_models"
    )]
    pub models: Vec<ClassifierModel>,
    /// Score floor for the image-generation axis, used by every engine that
    /// does not set its own `image_generation_threshold` in its
    /// `[classifier.<model>]` table. The scale is **engine-specific**
    /// (P(entailment) for the zero-shot engine, embedding similarity for the
    /// remote engines) — with several engines configured, prefer the
    /// per-engine keys; one global number cannot mean the same thing to all
    /// of them.
    #[serde(default = "default_image_gen_threshold")]
    pub image_generation_threshold: f32,
    /// Word ceiling for the trivial fast-path (`0` disables).
    #[serde(default = "default_trivial_max_words")]
    pub trivial_max_words: usize,
    /// Settings specific to the `deberta-v3-xsmall-zeroshot` engine
    /// (`[classifier.deberta-v3-xsmall-zeroshot]`). Ignored unless that model
    /// is selected. Engine-specific settings live in per-engine tables; a new
    /// engine adds its own table here.
    #[serde(default, rename = "deberta-v3-xsmall-zeroshot")]
    pub deberta_v3_xsmall_zeroshot: DebertaV3XsmallZeroshotConfig,
    /// Settings for the `gemini-embedding-001` engine
    /// (`[classifier.gemini-embedding-001]`). Ignored unless selected. The
    /// auth fields choose the API surface — see [`GoogleEmbeddingConfig`].
    #[serde(default, rename = "gemini-embedding-001")]
    pub gemini_embedding_001: GoogleEmbeddingConfig,
    /// Settings for the `gemini-embedding-2` engine
    /// (`[classifier.gemini-embedding-2]`). Ignored unless selected. The
    /// auth fields choose the API surface — see [`GoogleEmbeddingConfig`].
    #[serde(default, rename = "gemini-embedding-2")]
    pub gemini_embedding_2: GoogleEmbeddingConfig,
    /// Settings for the `text-embedding-005` engine
    /// (`[classifier.text-embedding-005]`). Ignored unless selected. Uses the
    /// Vertex-specific shape (this model is Vertex-AI-only), not the shared
    /// [`RemoteEmbeddingConfig`] the Gemini family uses.
    #[serde(default, rename = "text-embedding-005")]
    pub text_embedding_005: VertexEmbeddingConfig,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        ClassifierConfig {
            models: default_classifier_models(),
            image_generation_threshold: default_image_gen_threshold(),
            trivial_max_words: default_trivial_max_words(),
            deberta_v3_xsmall_zeroshot: DebertaV3XsmallZeroshotConfig::default(),
            gemini_embedding_001: GoogleEmbeddingConfig::default(),
            gemini_embedding_2: GoogleEmbeddingConfig::default(),
            text_embedding_005: VertexEmbeddingConfig::default(),
        }
    }
}

fn default_image_gen_threshold() -> f32 {
    DEFAULT_IMAGE_GEN_THRESHOLD
}
fn default_trivial_max_words() -> usize {
    DEFAULT_TRIVIAL_MAX_WORDS
}
fn default_classifier_models() -> Vec<ClassifierModel> {
    vec![ClassifierModel::default()]
}

/// Deserialize `[classifier] model` as either a single model id or a list of
/// them. A visitor (rather than an untagged enum) so an unknown id keeps its
/// loud, specific error message instead of "did not match any variant".
fn one_or_many_models<'de, D>(deserializer: D) -> Result<Vec<ClassifierModel>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OneOrManyVisitor;

    impl<'de> Visitor<'de> for OneOrManyVisitor {
        type Value = Vec<ClassifierModel>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a classifier model id, or a list of them")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            let model = ClassifierModel::deserialize(de::value::StrDeserializer::<E>::new(v))?;
            Ok(vec![model])
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut models = Vec::new();
            while let Some(model) = seq.next_element::<ClassifierModel>()? {
                models.push(model);
            }
            Ok(models)
        }
    }

    deserializer.deserialize_any(OneOrManyVisitor)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Model string sent upstream as `body["model"]`.
    pub name: String,
    /// Backend base URL. Parsed and constrained to `http`/`https` at config
    /// load, so a malformed or non-HTTP URL is a startup error rather than a
    /// per-request failure. `{base_url}/chat/completions` is the forward target.
    #[serde(deserialize_with = "deserialize_http_url")]
    pub base_url: Url,
    /// Authentication material, or `None` for a keyless backend. A plaintext
    /// / env-expanded string parses straight to a redacted static secret; a
    /// keyring table parses to an **unresolved marker** looked up by
    /// [`RouterConfig::resolve_secrets`] after parsing; `{ source =
    /// "google-adc" }` marks the model for per-request Google OAuth tokens
    /// (see [`ModelApiKey`]). An omitted field, an empty string, or a keyring
    /// value that resolves empty all mean "no auth": no `Authorization`
    /// header is sent. Never logged — the secret is a [`SecretString`], so
    /// even `Debug` output is redacted.
    #[serde(default, deserialize_with = "deserialize_model_api_key")]
    pub api_key: Option<ModelApiKey>,
    /// Deserialised from `type` (a Rust reserved word).
    #[serde(rename = "type")]
    pub tier: ModelTier,
    /// Non-empty; matched as a set for superset checks.
    pub modalities: Vec<Modality>,
    /// The model's context window in **tokens** (prompt + completion), e.g.
    /// `context_window = 128000`. REQUIRED: routing avoids models whose
    /// window cannot fit the request's estimated size — "fast" models
    /// typically have much smaller windows than "frontier" ones, and an
    /// oversized request sent to a small-window backend is a guaranteed
    /// upstream failure — so every model must declare its capacity (an
    /// omitted or zero value is a startup error, not a silent "fits
    /// everything").
    pub context_window: NonZeroU64,
}

impl ModelConfig {
    /// The declared modalities as a set, for superset matching.
    pub fn modality_set(&self) -> ModalitySet {
        self.modalities.iter().copied().collect()
    }

    /// Whether a request estimated at `estimated_tokens` fits this model's
    /// declared context window.
    pub fn fits_context(&self, estimated_tokens: u64) -> bool {
        estimated_tokens <= self.context_window.get()
    }
}

/// Deserialize a `base_url` string into a [`Url`], rejecting anything that is
/// not a well-formed `http`/`https` URL. Deserializing via `String` first keeps
/// this robust across the `config` crate's value model; the scheme check makes
/// `file://`, `ftp://`, missing-scheme, and other footguns fatal at load.
fn deserialize_http_url<'de, D>(deserializer: D) -> Result<Url, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    parse_http_url(&raw).map_err(de::Error::custom)
}

/// Optional variant of [`deserialize_http_url`] for engine `base_url`
/// overrides: absent stays `None`; present must be a valid http/https URL.
fn deserialize_opt_http_url<'de, D>(deserializer: D) -> Result<Option<Url>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Option<String> = Option::deserialize(deserializer)?;
    raw.map(|s| parse_http_url(&s).map_err(de::Error::custom))
        .transpose()
}

/// Parse and scheme-check an http/https URL, with a readable error.
fn parse_http_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw).map_err(|e| format!("invalid base_url `{raw}`: {e}"))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => Err(format!(
            "base_url `{raw}` must be an http or https URL, got scheme `{other}`"
        )),
    }
}

// ───────────────────────────────────────────────────────────────────────
// API key resolution
// ───────────────────────────────────────────────────────────────────────

/// A routed model's authentication material, as it stands after config load.
#[derive(Debug, Clone)]
pub enum ModelApiKey {
    /// A static secret — plaintext, `${ENV_VAR}`-expanded, or resolved from
    /// the OS keyring — sent verbatim as `Authorization: Bearer`. Held as a
    /// [`SecretString`], so `Debug` output is redacted.
    Static(SecretString),
    /// `api_key = { source = "keyring", service, user }`: an **unresolved**
    /// OS-keyring reference. Deserialization performs no I/O; the lookup
    /// happens in [`RouterConfig::resolve_secrets`] (called by
    /// `config::load`), which replaces this with [`Self::Static`] — or `None`
    /// when the entry resolves empty.
    Keyring {
        /// Keyring service name.
        service: String,
        /// Keyring user/account name.
        user: String,
    },
    /// `api_key = { source = "google-adc" }`: authenticate with a Google
    /// OAuth 2.0 access token resolved **per request** via Application
    /// Default Credentials (for backends hosted on Vertex AI). Deliberately a
    /// *marker*: at startup it is resolved into a credential-owning runtime
    /// handle (see `proxy::ModelAuth`), not at config load — parsing stays
    /// hermetic and tokens are cached/refreshed by `google-cloud-auth`,
    /// never baked in.
    GoogleAdc,
}

/// Equality compares exposed secret values. Acceptable here: this is config
/// equality (tests, drift guards), never an authentication check, so a
/// non-constant-time comparison is fine.
impl PartialEq for ModelApiKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ModelApiKey::Static(a), ModelApiKey::Static(b)) => {
                a.expose_secret() == b.expose_secret()
            }
            (
                ModelApiKey::Keyring { service, user },
                ModelApiKey::Keyring {
                    service: s2,
                    user: u2,
                },
            ) => service == s2 && user == u2,
            (ModelApiKey::GoogleAdc, ModelApiKey::GoogleAdc) => true,
            _ => false,
        }
    }
}

impl Eq for ModelApiKey {}

/// Custom deserializer for a routed model's `api_key`. A TOML string yields
/// the plaintext/env-expanded key verbatim (as a redacted [`SecretString`]);
/// `{ source = "keyring", service, user }` records an **unresolved** keyring
/// reference (no I/O here — see [`RouterConfig::resolve_secrets`]);
/// `{ source = "google-adc" }` records the per-request Google-token marker.
/// An empty string becomes `None` (keyless backend), the same as omitting
/// the field.
fn deserialize_model_api_key<'de, D>(deserializer: D) -> Result<Option<ModelApiKey>, D::Error>
where
    D: Deserializer<'de>,
{
    /// Empty string => no auth.
    fn non_empty(s: String) -> Option<ModelApiKey> {
        (!s.is_empty()).then(|| ModelApiKey::Static(SecretString::from(s)))
    }

    struct ApiKeyVisitor;

    impl<'de> Visitor<'de> for ApiKeyVisitor {
        type Value = Option<ModelApiKey>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(
                "a plaintext API key string, a keyring lookup table, or \
                 { source = \"google-adc\" }",
            )
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<ModelApiKey>, E> {
            Ok(non_empty(v.to_owned()))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Option<ModelApiKey>, E> {
            Ok(non_empty(v))
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Option<ModelApiKey>, A::Error> {
            let mut source: Option<String> = None;
            let mut service: Option<String> = None;
            let mut user: Option<String> = None;

            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "source" => source = Some(map.next_value()?),
                    "service" => service = Some(map.next_value()?),
                    "user" => user = Some(map.next_value()?),
                    other => {
                        return Err(de::Error::custom(format!(
                            "unknown api_key table field `{other}`"
                        )))
                    }
                }
            }

            api_key_from_table(source, service, user).map_err(de::Error::custom)
        }
    }

    deserializer.deserialize_any(ApiKeyVisitor)
}

/// Dispatch a parsed `api_key` table (`source` + optional fields) to its
/// marker: an unresolved keyring reference, or `google-adc`. **No I/O** —
/// keyring lookups happen later in [`RouterConfig::resolve_secrets`]. String
/// errors are wrapped by the caller into serde errors.
fn api_key_from_table(
    source: Option<String>,
    service: Option<String>,
    user: Option<String>,
) -> Result<Option<ModelApiKey>, String> {
    let source = source.ok_or("missing field `source`")?;
    match source.as_str() {
        "keyring" => {
            let service = service.ok_or("missing field `service`")?;
            let user = user.ok_or("missing field `user`")?;
            Ok(Some(ModelApiKey::Keyring { service, user }))
        }
        "google-adc" if service.is_some() || user.is_some() => {
            Err("api_key source `google-adc` takes no `service`/`user` fields".into())
        }
        "google-adc" => Ok(Some(ModelApiKey::GoogleAdc)),
        other => Err(format!(
            "unsupported api_key source `{other}`; expected `keyring` or `google-adc`"
        )),
    }
}

/// Look up a secret in the OS keyring, with readable failure messages.
fn keyring_secret(service: &str, user: &str) -> Result<String, String> {
    let entry = keyring::Entry::new(service, user)
        .map_err(|e| format!("keyring entry (service={service}, user={user}) unavailable: {e}"))?;
    entry
        .get_password()
        .map_err(|e| format!("keyring lookup failed (service={service}, user={user}): {e}"))
}

/// Custom deserializer for the classifier engine tables' static secrets
/// (`api_key` / `access_token`): the same surface as a routed model's
/// `api_key` — plaintext / `${ENV_VAR}` / keyring — **except** `google-adc`,
/// which only makes sense for routed models (the engines own their auth:
/// Gemini takes real API keys, and the vertex engine already defaults to
/// ADC). Keyring references stay unresolved until
/// [`RouterConfig::resolve_secrets`], which only resolves the tables of
/// **selected** engines.
fn deserialize_static_secret<'de, D>(deserializer: D) -> Result<Option<StaticSecret>, D::Error>
where
    D: Deserializer<'de>,
{
    match deserialize_model_api_key(deserializer)? {
        None => Ok(None),
        Some(ModelApiKey::Static(secret)) => Ok(Some(StaticSecret::Resolved(secret))),
        Some(ModelApiKey::Keyring { service, user }) => {
            Ok(Some(StaticSecret::Keyring { service, user }))
        }
        Some(ModelApiKey::GoogleAdc) => Err(de::Error::custom(
            "api_key source `google-adc` is only supported on routed models \
             (`[[models]] api_key`); this field takes a static secret \
             (plaintext, ${ENV_VAR}, or a keyring table)",
        )),
    }
}

impl RouterConfig {
    /// Resolve every **unresolved keyring reference** in place: the `api_key`
    /// of every routed model, and the `api_key`/`access_token` of the engine
    /// tables of engines actually **selected** by `[classifier] model` —
    /// unselected tables are deliberately never touched, so a keyring entry
    /// referenced by an inactive table can neither block loading (an OS
    /// keyring lookup can prompt or fail) nor be read needlessly.
    ///
    /// Called by `config::load` after parse + [`validate`](Self::validate);
    /// deserialization itself performs no I/O. A keyring entry that resolves
    /// to an empty string becomes `None` ("no auth"), matching the empty
    /// plaintext-string behavior.
    pub fn resolve_secrets(&mut self) -> anyhow::Result<()> {
        for model in &mut self.models {
            if let Some(ModelApiKey::Keyring { service, user }) = &model.api_key {
                let secret = keyring_secret(service, user)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| format!("resolving api_key for model `{}`", model.name))?;
                model.api_key =
                    (!secret.is_empty()).then(|| ModelApiKey::Static(SecretString::from(secret)));
            }
        }

        let selected = self.classifier.models.clone();
        for engine in selected {
            match engine {
                // Embedded local engine: no secrets to resolve.
                ClassifierModel::DebertaV3XsmallZeroshot => {}
                ClassifierModel::GeminiEmbedding001 => {
                    let table = &mut self.classifier.gemini_embedding_001;
                    StaticSecret::resolve_slot(&mut table.api_key, "gemini-embedding-001")?;
                    StaticSecret::resolve_slot(&mut table.access_token, "gemini-embedding-001")?;
                }
                ClassifierModel::GeminiEmbedding2 => {
                    let table = &mut self.classifier.gemini_embedding_2;
                    StaticSecret::resolve_slot(&mut table.api_key, "gemini-embedding-2")?;
                    StaticSecret::resolve_slot(&mut table.access_token, "gemini-embedding-2")?;
                }
                ClassifierModel::TextEmbedding005 => {
                    StaticSecret::resolve_slot(
                        &mut self.classifier.text_embedding_005.access_token,
                        "text-embedding-005",
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Schema-level validation: a non-empty model catalogue, a well-formed
    /// classifier ladder list, per-model modality declarations, and text
    /// coverage ([`validate_coverage`](Self::validate_coverage)). Fails fast
    /// with a clear message.
    ///
    /// Deliberately **not** the whole story: per-engine table completeness
    /// (API surface choice, required Vertex `project`/`location`, distinct
    /// ladder budgets) is validated by `engines::validate_config` — those
    /// rules are engine knowledge, and `config` cannot depend on `engines`.
    /// Both `serve` (via `engines::build_roster`) and the `validate`
    /// subcommand run that check; a config passing here alone may still be
    /// rejected there.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.models.is_empty() {
            anyhow::bail!("no [[models]] configured");
        }
        if let Some(telemetry) = &self.telemetry {
            telemetry.validate()?;
        }
        if self.classifier.models.is_empty() {
            anyhow::bail!("[classifier] model lists no models; name at least one engine");
        }
        for (i, model) in self.classifier.models.iter().enumerate() {
            if self.classifier.models[..i].contains(model) {
                anyhow::bail!(
                    "[classifier] model lists `{}` more than once",
                    model.as_str()
                );
            }
        }
        for m in &self.models {
            if m.modalities.is_empty() {
                anyhow::bail!("model `{}` declares no modalities", m.name);
            }
            // `api_key` is optional: an omitted/empty value means a keyless
            // backend (no `Authorization` header sent).
        }
        self.validate_coverage()
    }

    /// Startup coverage validation. Text is the fallback baseline: a request
    /// with no special modality is plain text, so the **only** requirement is
    /// that at least one model (in any tier) is text-capable. Every other
    /// modality is best-effort — a request that requires an uncovered modality
    /// is answered with a `422` at request time, not rejected at startup.
    pub fn validate_coverage(&self) -> anyhow::Result<()> {
        let text_covered = self
            .models
            .iter()
            .any(|m| m.modality_set().contains(Modality::Text));
        if !text_covered {
            anyhow::bail!(
                "no model declares the `text` modality; text is the fallback baseline and must be served by at least one model"
            );
        }
        Ok(())
    }
}

// Model-selection policy (`select_candidate`, `count_candidates`, tier
// ranking) lives in `crate::selection` — it is routing policy over the
// parsed catalogue, not schema.

#[cfg(test)]
mod tests {
    use super::*;

    // ── env expansion, formats, loading: see `load::tests` ───────────

    // ── server / classifier tuning fields ─────────────────────────────
    #[test]
    fn server_tuning_fields_default_when_omitted() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        assert_eq!(cfg.server.stream_idle_timeout_secs, 300);
        assert_eq!(cfg.server.max_body_bytes, 32 * 1024 * 1024);
        assert_eq!(
            cfg.classifier.models,
            [ClassifierModel::DebertaV3XsmallZeroshot]
        );
        assert_eq!(cfg.classifier.trivial_max_words, DEFAULT_TRIVIAL_MAX_WORDS);
        assert_eq!(
            cfg.classifier
                .deberta_v3_xsmall_zeroshot
                .inference_pool_size,
            None
        );
        assert_eq!(
            cfg.classifier.deberta_v3_xsmall_zeroshot.intra_op_threads,
            None
        );
    }

    // ── [telemetry] ──────────────────────────────────────────────
    const MODELS_STANZA: &str = "[server]\nhost=\"0.0.0.0\"\nport=1\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n";

    #[test]
    fn telemetry_absent_means_off() {
        let cfg = parse(MODELS_STANZA).unwrap();
        assert!(cfg.telemetry.is_none());
    }

    // ── [logging] ────────────────────────────────────────────────
    #[test]
    fn log_prompts_defaults_off_and_parses() {
        // Absent table and absent field both mean "never log prompts".
        let cfg = parse(MODELS_STANZA).unwrap();
        assert!(!cfg.logging.log_prompts);
        let cfg = parse(&format!("{MODELS_STANZA}[logging]\n")).unwrap();
        assert!(!cfg.logging.log_prompts);
        let cfg = parse(&format!("{MODELS_STANZA}[logging]\nlog_prompts=true\n")).unwrap();
        assert!(cfg.logging.log_prompts);
        // No silent typos.
        assert!(parse(&format!("{MODELS_STANZA}[logging]\nlog_prompt=true\n")).is_err());
    }

    #[test]
    fn telemetry_parses_with_defaults() {
        let toml = format!("{MODELS_STANZA}[telemetry]\notlp_endpoint=\"http://localhost:4318\"\n");
        let cfg = parse(&toml).unwrap();
        let t = cfg.telemetry.as_ref().expect("telemetry table parses");
        assert_eq!(t.otlp_endpoint.as_str(), "http://localhost:4318/");
        assert_eq!(t.service_name, "hyper-mcp-router");
        assert!(t.traces);
        assert!(t.metrics);
        assert_eq!(t.sample_ratio, 1.0);
        assert!(!t.parent_based_sampling);
        assert_eq!(t.metrics_interval_secs, 60);
        cfg.validate().expect("defaults validate");
    }

    #[test]
    fn telemetry_endpoint_is_required_and_scheme_checked() {
        // Missing endpoint: the table cannot silently mean "off".
        let missing = format!("{MODELS_STANZA}[telemetry]\nservice_name=\"x\"\n");
        assert!(parse(&missing).is_err());
        // Non-HTTP scheme is a load error, not a runtime export failure.
        let grpcish =
            format!("{MODELS_STANZA}[telemetry]\notlp_endpoint=\"grpc://localhost:4317\"\n");
        let err = parse(&grpcish).unwrap_err().to_string();
        assert!(err.contains("http"), "got: {err}");
    }

    #[test]
    fn telemetry_validation_rejects_bad_values() {
        let base = format!("{MODELS_STANZA}[telemetry]\notlp_endpoint=\"http://localhost:4318\"\n");
        // sample_ratio outside [0, 1]
        let cfg = parse(&format!("{base}sample_ratio=1.5\n")).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("sample_ratio"), "got: {err}");
        // both signals off: remove the table instead
        let cfg = parse(&format!("{base}traces=false\nmetrics=false\n")).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("traces and metrics"), "got: {err}");
        // zero interval
        let cfg = parse(&format!("{base}metrics_interval_secs=0\n")).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("metrics_interval_secs"), "got: {err}");
    }

    #[test]
    fn telemetry_rejects_unknown_fields() {
        // No silent typos — and specifically, no `headers` table: credentialed
        // endpoints are out of scope (local collectors only; see docs).
        let toml = format!(
            "{MODELS_STANZA}[telemetry]\notlp_endpoint=\"http://localhost:4318\"\n[telemetry.headers]\nx-api-key=\"k\"\n"
        );
        assert!(parse(&toml).is_err());
    }

    #[test]
    fn classifier_model_parses_and_rejects_unknown() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[classifier]\nmodel=\"deberta-v3-xsmall-zeroshot\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        assert_eq!(
            cfg.classifier.models,
            [ClassifierModel::DebertaV3XsmallZeroshot]
        );

        // An unknown model id must be a loud config error, never a silent default.
        let err = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[classifier]\nmodel=\"not-a-model\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        );
        assert!(err.is_err());
    }

    #[test]
    fn classifier_model_accepts_a_list() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=[\"gemini-embedding-2\", \"deberta-v3-xsmall-zeroshot\"]\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        // Config order is preserved verbatim — the capacity ladder is derived
        // from engine budgets at startup, never from list order.
        assert_eq!(
            cfg.classifier.models,
            [
                ClassifierModel::GeminiEmbedding2,
                ClassifierModel::DebertaV3XsmallZeroshot,
            ]
        );
        cfg.validate().expect("a distinct list must validate");
    }

    #[test]
    fn classifier_model_list_rejects_unknown_ids_loudly() {
        let err = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=[\"deberta-v3-xsmall-zeroshot\", \"not-a-model\"]\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        );
        assert!(err.is_err());
    }

    #[test]
    fn classifier_model_list_rejects_duplicates_at_validation() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=[\"deberta-v3-xsmall-zeroshot\", \"deberta-v3-xsmall-zeroshot\"]\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");
    }

    #[test]
    fn classifier_model_empty_list_fails_validation() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[classifier]\nmodel=[]\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("lists no models"), "got: {err}");
    }

    #[test]
    fn per_engine_image_generation_threshold_overrides_global() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nimage_generation_threshold=0.7\n\
             [classifier.deberta-v3-xsmall-zeroshot]\nimage_generation_threshold=0.9\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.image_generation_threshold, 0.7);
        assert_eq!(
            cfg.classifier
                .deberta_v3_xsmall_zeroshot
                .image_generation_threshold,
            Some(0.9)
        );
        // Engines without their own key inherit the global at build time.
        assert_eq!(
            cfg.classifier.gemini_embedding_2.image_generation_threshold,
            None
        );
        assert_eq!(
            cfg.classifier.text_embedding_005.image_generation_threshold,
            None
        );
    }

    #[test]
    fn server_and_classifier_tuning_fields_parse() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\nstream_idle_timeout_secs=42\nmax_body_bytes=1024\n\
             [classifier]\ntrivial_max_words=3\n\
             [classifier.deberta-v3-xsmall-zeroshot]\ninference_pool_size=4\nintra_op_threads=1\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n",
        )
        .unwrap();
        assert_eq!(cfg.server.stream_idle_timeout_secs, 42);
        assert_eq!(cfg.server.max_body_bytes, 1024);
        assert_eq!(cfg.classifier.trivial_max_words, 3);
        assert_eq!(
            cfg.classifier
                .deberta_v3_xsmall_zeroshot
                .inference_pool_size,
            Some(4)
        );
        assert_eq!(
            cfg.classifier.deberta_v3_xsmall_zeroshot.intra_op_threads,
            Some(1)
        );
    }

    // ── engine settings: see `engines::tests` ───────────────────────

    // ── ApiKey resolution ─────────────────────────────────────────────────────────
    fn parse_single_model(toml: &str) -> RouterConfig {
        parse(toml).expect("config should parse")
    }

    const BASE: &str = r#"
[server]
host = "0.0.0.0"
port = 8080
"#;

    #[test]
    fn api_key_plaintext() {
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key=\"sk-plain\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Static("sk-plain".into()))
        );
    }

    #[test]
    fn api_key_absent_is_none() {
        // Field omitted entirely: keyless backend.
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(cfg.models[0].api_key, None);
    }

    #[test]
    fn api_key_empty_string_is_none() {
        // Explicit empty string is treated the same as omitting the field.
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key=\"\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(cfg.models[0].api_key, None);
    }

    #[test]
    fn keyless_model_passes_validation() {
        // A keyless single-tier catalogue must still validate (coverage aside).
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"fast\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n[[models]]\nname=\"bal\"\nbase_url=\"http://u\"\ntype=\"balanced\"\nmodalities=[\"text\"]\ncontext_window=128000\n[[models]]\nname=\"front\"\nbase_url=\"http://u\"\ntype=\"frontier\"\nmodalities=[\"text\", \"image-input\", \"audio-input\", \"file-input\", \"audio-output\", \"image-output\", \"tools\"]\ncontext_window=128000\n"
        ));
        assert!(cfg.validate().is_ok());
        assert!(cfg.models.iter().all(|m| m.api_key.is_none()));
    }

    #[test]
    fn api_key_env_expanded() {
        std::env::set_var("ROUTER_TEST_KEY", "sk-from-env");
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key=\"${{ROUTER_TEST_KEY}}\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Static("sk-from-env".into()))
        );
    }

    #[test]
    fn api_key_keyring_parses_hermetically_without_io() {
        // A keyring reference must parse to an UNRESOLVED marker with no OS
        // keyring I/O — the entry deliberately does not exist, and parsing
        // must still succeed. Resolution happens later, in resolve_secrets.
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"keyring\", service = \"hyper-mcp-router-no-such-service\", user = \"nobody\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Keyring {
                service: "hyper-mcp-router-no-such-service".into(),
                user: "nobody".into(),
            })
        );
    }

    #[test]
    fn resolve_secrets_resolves_model_keyring_when_store_available() {
        // Gate on store availability, as hyper-mcp does.
        let service = "hyper-mcp-router-test";
        let user = "keyring-probe";
        let probe = keyring::Entry::new(service, user)
            .and_then(|e| e.set_password("sk-keyring-secret").map(|_| e));
        let Ok(entry) = probe else {
            eprintln!("keyring store unavailable; skipping");
            return;
        };
        let mut cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"keyring\", service = \"{service}\", user = \"{user}\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        cfg.resolve_secrets().expect("resolution should succeed");
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Static("sk-keyring-secret".into()))
        );
        let _ = entry.delete_credential();
    }

    #[test]
    fn resolve_secrets_names_the_model_on_keyring_failure() {
        // A nonexistent keyring entry on a ROUTED model is a resolution
        // error naming the model (whatever the platform's store situation,
        // looking up a missing entry cannot succeed).
        let mut cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"frontier\"\nbase_url=\"http://u\"\napi_key={{ source = \"keyring\", service = \"hyper-mcp-router-no-such-service\", user = \"nobody\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        let err = format!("{:#}", cfg.resolve_secrets().unwrap_err());
        assert!(err.contains("frontier"), "got: {err}");
    }

    #[test]
    fn resolve_secrets_skips_unselected_engine_tables() {
        // The gemini table references a keyring entry that does not exist,
        // but the engine is NOT selected ([classifier] model defaults to the
        // embedded deberta), so resolution must neither fail nor touch the
        // keyring — the reference stays unresolved.
        let mut cfg = parse(&format!(
            "{BASE}\n[classifier.gemini-embedding-2]\napi_key={{ source = \"keyring\", service = \"hyper-mcp-router-no-such-service\", user = \"nobody\" }}\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ))
        .unwrap();
        cfg.resolve_secrets()
            .expect("unselected engine tables must not be resolved");
        assert!(matches!(
            cfg.classifier.gemini_embedding_2.api_key,
            Some(StaticSecret::Keyring { .. })
        ));
    }

    // ── secret redaction (Debug must never print a credential) ────────
    #[test]
    fn debug_output_redacts_all_secrets() {
        let cfg = parse(&format!(
            "{BASE}\n[classifier.gemini-embedding-2]\napi_key=\"sk-engine-secret\"\n\
             [classifier.text-embedding-005]\nproject=\"p\"\nlocation=\"us\"\naccess_token=\"ya29-token-secret\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key=\"sk-model-secret\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ))
        .unwrap();
        let debug = format!("{cfg:?}");
        for secret in ["sk-model-secret", "sk-engine-secret", "ya29-token-secret"] {
            assert!(
                !debug.contains(secret),
                "Debug output leaked `{secret}`: {debug}"
            );
        }
        assert!(debug.contains("REDACTED"), "got: {debug}");
    }

    // ── unknown-field rejection (typos must be loud) ──────────────────
    #[test]
    fn unknown_fields_are_rejected() {
        let model_block = "[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n";
        // Top-level, [server], [classifier] typo, [[models]] typo, and an
        // unknown engine table must all fail to parse rather than be dropped.
        let cases = [
            format!("{BASE}\n[not_a_section]\nx=1\n{model_block}"),
            format!("[server]\nhost=\"0.0.0.0\"\nport=1\nmax_body_byte=1\n{model_block}"),
            format!("{BASE}\n[classifier]\ntrivial_max_word=3\n{model_block}"),
            format!("{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\ncontext_windows=1\n"),
            format!("{BASE}\n[classifier.gemini-embeding-2]\napi_key=\"k\"\n{model_block}"),
        ];
        for case in cases {
            assert!(parse(&case).is_err(), "should reject: {case}");
        }
    }

    #[test]
    fn api_key_google_adc_parses_to_marker() {
        // Parsing must record the marker WITHOUT touching ADC (hermetic):
        // credential discovery happens at startup, not at config load.
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"google-adc\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(cfg.models[0].api_key, Some(ModelApiKey::GoogleAdc));
    }

    #[test]
    fn api_key_google_adc_rejects_extra_fields() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"google-adc\", service = \"x\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("takes no `service`/`user`"),
            "got: {err}"
        );
    }

    #[test]
    fn api_key_unknown_source_lists_the_valid_ones() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"vault\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ))
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("expected `keyring` or `google-adc`"),
            "got: {err}"
        );
    }

    #[test]
    fn base_url_parses_and_is_retained() {
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"https://api.example.com/v1\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(
            cfg.models[0].base_url.as_str(),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn base_url_rejects_malformed() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"not a url\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ))
        .unwrap_err();
        assert!(err.to_string().contains("base_url"), "got: {err}");
    }

    #[test]
    fn base_url_rejects_non_http_scheme() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"ftp://files.example.com\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ))
        .unwrap_err();
        assert!(err.to_string().contains("http or https"), "got: {err}");
    }

    // ── selection policy: see `crate::selection::tests` ────────────────

    /// A model with an effectively unbounded window, for the coverage tests.
    fn model(name: &str, tier: ModelTier, mods: &[Modality]) -> ModelConfig {
        model_ctx(name, tier, mods, u64::MAX)
    }

    /// [`model`] with a specific context window (tokens).
    fn model_ctx(name: &str, tier: ModelTier, mods: &[Modality], window: u64) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            base_url: Url::parse("http://x").unwrap(),
            api_key: Some(ModelApiKey::Static(SecretString::from("k"))),
            tier,
            modalities: mods.to_vec(),
            context_window: NonZeroU64::new(window).expect("nonzero window"),
        }
    }

    fn catalogue(models: Vec<ModelConfig>) -> RouterConfig {
        RouterConfig {
            server: ServerConfig::default(),
            classifier: ClassifierConfig::default(),
            logging: LoggingConfig::default(),
            telemetry: None,
            models,
        }
    }

    // ── coverage validation ─────────────────────────────────────────────────
    fn full_catalogue() -> RouterConfig {
        catalogue(vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model(
                "balanced",
                ModelTier::Balanced,
                &[
                    Modality::Text,
                    Modality::ImageInput,
                    Modality::FileInput,
                    Modality::ImageOutput,
                ],
            ),
            model("frontier", ModelTier::Frontier, &[Modality::Text]),
            model(
                "voice",
                ModelTier::Balanced,
                &[Modality::Text, Modality::AudioInput, Modality::AudioOutput],
            ),
            model(
                "agent",
                ModelTier::Balanced,
                &[Modality::Text, Modality::Tools],
            ),
        ])
    }

    #[test]
    fn coverage_passes_for_full_catalogue() {
        assert!(full_catalogue().validate_coverage().is_ok());
    }

    #[test]
    fn coverage_passes_with_text_in_a_single_tier() {
        // Text is the only mandatory coverage, and it need exist in just one
        // tier. Complexity requests fall back to it via selection policy.
        let cfg = catalogue(vec![model("only", ModelTier::Balanced, &[Modality::Text])]);
        assert!(cfg.validate_coverage().is_ok());
    }

    #[test]
    fn coverage_allows_uncovered_non_text_modalities() {
        // No audio/image/file/tools model anywhere: still valid. Such requests
        // get a 422 at request time (see `select_uncovered_combination_...`).
        let cfg = catalogue(vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model("frontier", ModelTier::Frontier, &[Modality::Text]),
        ]);
        assert!(cfg.validate_coverage().is_ok());
    }

    #[test]
    fn coverage_fails_when_no_text_model() {
        // Every model is special-modality-only; nothing can serve plain text.
        let cfg = catalogue(vec![
            model("vision", ModelTier::Balanced, &[Modality::ImageInput]),
            model("voice", ModelTier::Balanced, &[Modality::AudioInput]),
        ]);
        let err = cfg.validate_coverage().unwrap_err();
        assert!(err.to_string().contains("text"), "got: {err}");
    }

    // ── context-window fit ────────────────────────────────────────
    #[test]
    fn context_window_boundary_is_inclusive() {
        let m = model_ctx("m", ModelTier::Fast, &[Modality::Text], 8_000);
        assert!(m.fits_context(8_000));
        assert!(!m.fits_context(8_001));
    }

    #[test]
    fn context_window_parses() {
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=128000\n"
        ));
        assert_eq!(
            cfg.models[0].context_window,
            NonZeroU64::new(128_000).unwrap()
        );
    }

    #[test]
    fn context_window_is_required() {
        // Capacity is a routing axis, so every model must declare it — an
        // omitted window is a load-time error, not a silent "fits everything".
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ))
        .unwrap_err();
        assert!(err.to_string().contains("context_window"), "got: {err}");
    }

    #[test]
    fn context_window_rejects_zero() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\ncontext_window=0\n"
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("context_window") || err.to_string().contains("nonzero"),
            "got: {err}"
        );
    }
}
