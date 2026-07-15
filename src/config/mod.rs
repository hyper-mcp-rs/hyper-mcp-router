//! Configuration: the config **schema** — typed structs, custom
//! deserializers, API-key resolution (plaintext / env / keyring), the model
//! catalogue, model selection, and startup coverage validation. Loading —
//! path discovery, TOML/YAML/JSON format selection, env expansion, parsing —
//! lives in the `load` submodule.
//!
//! Requests and responses elsewhere are handled as raw JSON; this module is the
//! only place typed structs are used, and only for the operator's config file.

use std::fmt;

use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::Deserialize;
use url::Url;

use crate::classifier::{ClassifierModel, ModelTier, DEFAULT_IMAGE_GEN_THRESHOLD};
use crate::modality::{Modality, ModalitySet};
use crate::prompt::DEFAULT_TRIVIAL_MAX_WORDS;

mod engines;
pub use engines::{
    DebertaV3XsmallZeroshotConfig, GoogleApi, GoogleEmbeddingConfig, RemoteEmbeddingConfig,
    VertexEmbeddingConfig,
};

mod load;
pub use load::{expand_env, load, parse, parse_with_format, resolve_config_path};

/// Application identifier used for OS config/log directory discovery.
const APP_NAME: &str = "hyper-mcp-router";

// ───────────────────────────────────────────────────────────────────────────
// Config structs
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct RouterConfig {
    pub server: ServerConfig,
    #[serde(default)]
    pub classifier: ClassifierConfig,
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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
pub struct ModelConfig {
    /// Model string sent upstream as `body["model"]`.
    pub name: String,
    /// Backend base URL. Parsed and constrained to `http`/`https` at config
    /// load, so a malformed or non-HTTP URL is a startup error rather than a
    /// per-request failure. `{base_url}/chat/completions` is the forward target.
    #[serde(deserialize_with = "deserialize_http_url")]
    pub base_url: Url,
    /// Authentication material, or `None` for a keyless backend. A static
    /// secret (plaintext / env / keyring) is resolved at load; `{ source =
    /// "google-adc" }` marks the model for per-request Google OAuth tokens
    /// (see [`ModelApiKey`]). An omitted field, an empty string, or a keyring
    /// value that resolves empty all mean "no auth": no `Authorization`
    /// header is sent. Never logged.
    #[serde(default, deserialize_with = "resolve_model_api_key")]
    pub api_key: Option<ModelApiKey>,
    /// Deserialised from `type` (a Rust reserved word).
    #[serde(rename = "type")]
    pub tier: ModelTier,
    /// Non-empty; matched as a set for superset checks.
    pub modalities: Vec<Modality>,
}

impl ModelConfig {
    /// The declared modalities as a set, for superset matching.
    pub fn modality_set(&self) -> ModalitySet {
        self.modalities.iter().copied().collect()
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelApiKey {
    /// A static secret — plaintext, `${ENV_VAR}`-expanded, or looked up from
    /// the OS keyring — fully resolved at config load and sent verbatim as
    /// `Authorization: Bearer`.
    Static(String),
    /// `api_key = { source = "google-adc" }`: authenticate with a Google
    /// OAuth 2.0 access token resolved **per request** via Application
    /// Default Credentials (for backends hosted on Vertex AI). Deliberately a
    /// *marker*: at startup it is resolved into a credential-owning runtime
    /// handle (see `proxy::ModelAuth`), not at config load — parsing stays
    /// hermetic and tokens are cached/refreshed by `google-cloud-auth`,
    /// never baked in.
    GoogleAdc,
}

/// Custom deserializer for a routed model's `api_key`. A TOML string yields
/// the plaintext/env-expanded key verbatim; `{ source = "keyring", service,
/// user }` triggers an OS keyring lookup at load time (only the resolved
/// secret is retained); `{ source = "google-adc" }` records the per-request
/// Google-token marker. An empty resolved value becomes `None` (keyless
/// backend), so `""` is treated the same as omitting the field.
fn resolve_model_api_key<'de, D>(deserializer: D) -> Result<Option<ModelApiKey>, D::Error>
where
    D: Deserializer<'de>,
{
    /// Empty string => no auth.
    fn non_empty(s: String) -> Option<ModelApiKey> {
        (!s.is_empty()).then_some(ModelApiKey::Static(s))
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
/// resolution: keyring lookup, or the `google-adc` marker. String errors are
/// wrapped by the caller into serde errors.
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
            let secret = keyring_secret(&service, &user)?;
            Ok((!secret.is_empty()).then_some(ModelApiKey::Static(secret)))
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
/// ADC).
fn resolve_api_key<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    match resolve_model_api_key(deserializer)? {
        None => Ok(None),
        Some(ModelApiKey::Static(secret)) => Ok(Some(secret)),
        Some(ModelApiKey::GoogleAdc) => Err(de::Error::custom(
            "api_key source `google-adc` is only supported on routed models \
             (`[[models]] api_key`); this field takes a static secret \
             (plaintext, ${ENV_VAR}, or a keyring table)",
        )),
    }
}

impl RouterConfig {
    /// Field-level and startup coverage validation. Fails fast with a clear
    /// message on any incomplete configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.models.is_empty() {
            anyhow::bail!("no [[models]] configured");
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

    /// Pick the model whose declared modalities are a **superset** of
    /// `required`, preferring `complexity`. Returns `None` when no single model
    /// covers the whole set (the proxy then returns 422).
    ///
    /// Ranking among survivors: exact type match → nearest higher type
    /// (escalation) → highest lower type. Ties break toward the first-declared
    /// model in config.
    pub fn select_model(
        &self,
        required: &ModalitySet,
        complexity: ModelTier,
    ) -> Option<&ModelConfig> {
        select_candidate(self.models.iter(), |m| m, required, complexity)
    }

    /// How many models can serve `required` (declare a superset of it). When this
    /// is `<= 1` the complexity tier is irrelevant — there is nothing to rank —
    /// so the proxy can skip classification entirely and route directly.
    pub fn candidate_count(&self, required: &ModalitySet) -> usize {
        count_candidates(self.models.iter(), |m| m, required)
    }
}

/// The model-selection policy, generic over any collection of model-bearing
/// items — the one implementation behind [`RouterConfig::select_model`]
/// (pure config) and the proxy's runtime catalogue (config paired with
/// resolved auth). `model_of` projects an item to its [`ModelConfig`].
///
/// 1. Filter by capability (superset), preserving declaration order.
/// 2. Rank survivors: exact type → nearest higher (escalation) → highest
///    lower (fallback); `min_by_key` returns the first minimum, so a tie
///    resolves toward the earlier-declared item.
pub(crate) fn select_candidate<'a, T: 'a>(
    items: impl IntoIterator<Item = &'a T>,
    model_of: impl Fn(&T) -> &ModelConfig,
    required: &ModalitySet,
    complexity: ModelTier,
) -> Option<&'a T> {
    items
        .into_iter()
        .filter(|item| model_of(item).modality_set().is_superset(required))
        .min_by_key(|item| tier_rank(model_of(item).tier, complexity))
}

/// Companion to [`select_candidate`]: how many items could serve `required`.
/// When this is `<= 1` the complexity tier is irrelevant (nothing to rank).
pub(crate) fn count_candidates<'a, T: 'a>(
    items: impl IntoIterator<Item = &'a T>,
    model_of: impl Fn(&T) -> &ModelConfig,
    required: &ModalitySet,
) -> usize {
    items
        .into_iter()
        .filter(|item| model_of(item).modality_set().is_superset(required))
        .count()
}

/// Distance ranking for model selection. Lower is better:
/// exact type (0) < escalation (nearest higher) < fallback (highest lower).
fn tier_rank(tier: ModelTier, want: ModelTier) -> i32 {
    let t = tier as i32;
    let w = want as i32;
    if t == w {
        0
    } else if t > w {
        10 + (t - w) // escalate: prefer the nearest higher type
    } else {
        100 + (w - t) // fallback: prefer the highest lower type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── env expansion, formats, loading: see `load::tests` ───────────

    // ── server / classifier tuning fields ─────────────────────────────
    #[test]
    fn server_tuning_fields_default_when_omitted() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
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

    #[test]
    fn classifier_model_parses_and_rejects_unknown() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[classifier]\nmodel=\"deberta-v3-xsmall-zeroshot\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        )
        .unwrap();
        assert_eq!(
            cfg.classifier.models,
            [ClassifierModel::DebertaV3XsmallZeroshot]
        );

        // An unknown model id must be a loud config error, never a silent default.
        let err = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[classifier]\nmodel=\"not-a-model\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        );
        assert!(err.is_err());
    }

    #[test]
    fn classifier_model_accepts_a_list() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=[\"gemini-embedding-2\", \"deberta-v3-xsmall-zeroshot\"]\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
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
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        );
        assert!(err.is_err());
    }

    #[test]
    fn classifier_model_list_rejects_duplicates_at_validation() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=[\"deberta-v3-xsmall-zeroshot\", \"deberta-v3-xsmall-zeroshot\"]\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        )
        .unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");
    }

    #[test]
    fn classifier_model_empty_list_fails_validation() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[classifier]\nmodel=[]\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
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
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
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
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
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
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key=\"sk-plain\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
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
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ));
        assert_eq!(cfg.models[0].api_key, None);
    }

    #[test]
    fn api_key_empty_string_is_none() {
        // Explicit empty string is treated the same as omitting the field.
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key=\"\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ));
        assert_eq!(cfg.models[0].api_key, None);
    }

    #[test]
    fn keyless_model_passes_validation() {
        // A keyless single-tier catalogue must still validate (coverage aside).
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"fast\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n[[models]]\nname=\"bal\"\nbase_url=\"http://u\"\ntype=\"balanced\"\nmodalities=[\"text\"]\n[[models]]\nname=\"front\"\nbase_url=\"http://u\"\ntype=\"frontier\"\nmodalities=[\"text\", \"image-input\", \"audio-input\", \"file-input\", \"audio-output\", \"image-output\", \"tools\"]\n"
        ));
        assert!(cfg.validate().is_ok());
        assert!(cfg.models.iter().all(|m| m.api_key.is_none()));
    }

    #[test]
    fn api_key_env_expanded() {
        std::env::set_var("ROUTER_TEST_KEY", "sk-from-env");
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key=\"${{ROUTER_TEST_KEY}}\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ));
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Static("sk-from-env".into()))
        );
    }

    #[test]
    fn api_key_keyring_when_store_available() {
        // Gate on store availability, as hyper-mcp does.
        let service = "hyper-mcp-router-test";
        let user = "keyring-probe";
        let probe = keyring::Entry::new(service, user)
            .and_then(|e| e.set_password("sk-keyring-secret").map(|_| e));
        let Ok(entry) = probe else {
            eprintln!("keyring store unavailable; skipping");
            return;
        };
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"keyring\", service = \"{service}\", user = \"{user}\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ));
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Static("sk-keyring-secret".into()))
        );
        let _ = entry.delete_credential();
    }

    #[test]
    fn api_key_google_adc_parses_to_marker() {
        // Parsing must record the marker WITHOUT touching ADC (hermetic):
        // credential discovery happens at startup, not at config load.
        let cfg = parse_single_model(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"google-adc\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ));
        assert_eq!(cfg.models[0].api_key, Some(ModelApiKey::GoogleAdc));
    }

    #[test]
    fn api_key_google_adc_rejects_extra_fields() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"google-adc\", service = \"x\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\n"
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
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"http://u\"\napi_key={{ source = \"vault\" }}\ntype=\"fast\"\nmodalities=[\"text\"]\n"
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
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"https://api.example.com/v1\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ));
        assert_eq!(
            cfg.models[0].base_url.as_str(),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn base_url_rejects_malformed() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"not a url\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ))
        .unwrap_err();
        assert!(err.to_string().contains("base_url"), "got: {err}");
    }

    #[test]
    fn base_url_rejects_non_http_scheme() {
        let err = parse(&format!(
            "{BASE}\n[[models]]\nname=\"m\"\nbase_url=\"ftp://files.example.com\"\ntype=\"fast\"\nmodalities=[\"text\"]\n"
        ))
        .unwrap_err();
        assert!(err.to_string().contains("http or https"), "got: {err}");
    }

    // ── select_model ──────────────────────────────────────────────────────────
    fn model(name: &str, tier: ModelTier, mods: &[Modality]) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            base_url: Url::parse("http://x").unwrap(),
            api_key: Some(ModelApiKey::Static("k".to_string())),
            tier,
            modalities: mods.to_vec(),
        }
    }

    fn catalogue(models: Vec<ModelConfig>) -> RouterConfig {
        RouterConfig {
            server: ServerConfig::default(),
            classifier: ClassifierConfig::default(),
            models,
        }
    }

    fn req(mods: &[Modality]) -> ModalitySet {
        mods.iter().copied().collect()
    }

    #[test]
    fn select_superset_excludes_missing_modality() {
        let cfg = catalogue(vec![
            model("text-only", ModelTier::Balanced, &[Modality::Text]),
            model(
                "vision",
                ModelTier::Balanced,
                &[Modality::Text, Modality::ImageInput],
            ),
        ]);
        let chosen = cfg
            .select_model(
                &req(&[Modality::Text, Modality::ImageInput]),
                ModelTier::Balanced,
            )
            .unwrap();
        assert_eq!(chosen.name, "vision");
    }

    #[test]
    fn select_exact_type_wins() {
        let cfg = catalogue(vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model("balanced", ModelTier::Balanced, &[Modality::Text]),
            model("frontier", ModelTier::Frontier, &[Modality::Text]),
        ]);
        let chosen = cfg
            .select_model(&req(&[Modality::Text]), ModelTier::Balanced)
            .unwrap();
        assert_eq!(chosen.name, "balanced");
    }

    #[test]
    fn select_escalates_to_nearest_higher() {
        let cfg = catalogue(vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model("frontier", ModelTier::Frontier, &[Modality::Text]),
        ]);
        // want Balanced, none exact => nearest higher is Frontier.
        let chosen = cfg
            .select_model(&req(&[Modality::Text]), ModelTier::Balanced)
            .unwrap();
        assert_eq!(chosen.name, "frontier");
    }

    #[test]
    fn select_falls_back_to_highest_lower() {
        let cfg = catalogue(vec![
            model("fast", ModelTier::Fast, &[Modality::Text]),
            model("balanced", ModelTier::Balanced, &[Modality::Text]),
        ]);
        // want Frontier, nothing at/above => highest lower is Balanced.
        let chosen = cfg
            .select_model(&req(&[Modality::Text]), ModelTier::Frontier)
            .unwrap();
        assert_eq!(chosen.name, "balanced");
    }

    #[test]
    fn select_first_declared_wins_on_tie() {
        let cfg = catalogue(vec![
            model("first", ModelTier::Balanced, &[Modality::Text]),
            model("second", ModelTier::Balanced, &[Modality::Text]),
        ]);
        let chosen = cfg
            .select_model(&req(&[Modality::Text]), ModelTier::Balanced)
            .unwrap();
        assert_eq!(chosen.name, "first");
    }

    #[test]
    fn select_covers_combination() {
        let cfg = catalogue(vec![model(
            "voice",
            ModelTier::Balanced,
            &[Modality::Text, Modality::AudioInput, Modality::AudioOutput],
        )]);
        let chosen = cfg
            .select_model(
                &req(&[Modality::AudioInput, Modality::AudioOutput]),
                ModelTier::Balanced,
            )
            .unwrap();
        assert_eq!(chosen.name, "voice");
    }

    #[test]
    fn select_uncovered_combination_returns_none() {
        let cfg = catalogue(vec![
            model(
                "audio-in",
                ModelTier::Balanced,
                &[Modality::Text, Modality::AudioInput],
            ),
            model(
                "audio-out",
                ModelTier::Balanced,
                &[Modality::Text, Modality::AudioOutput],
            ),
        ]);
        // No single model covers both directions.
        assert!(cfg
            .select_model(
                &req(&[Modality::AudioInput, Modality::AudioOutput]),
                ModelTier::Balanced
            )
            .is_none());
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
        // tier. Complexity requests fall back to it via `select_model`.
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

    #[test]
    fn candidate_count_reflects_superset_matches() {
        let cfg = catalogue(vec![
            model("a", ModelTier::Fast, &[Modality::Text]),
            model("b", ModelTier::Balanced, &[Modality::Text]),
            model(
                "vision",
                ModelTier::Balanced,
                &[Modality::Text, Modality::ImageInput],
            ),
        ]);
        // Three text models can serve plain text.
        assert_eq!(cfg.candidate_count(&req(&[Modality::Text])), 3);
        // Only the vision model can serve image input.
        assert_eq!(
            cfg.candidate_count(&req(&[Modality::Text, Modality::ImageInput])),
            1
        );
        // Nothing serves audio output.
        assert_eq!(cfg.candidate_count(&req(&[Modality::AudioOutput])), 0);
    }

    #[test]
    fn select_tools_requires_tool_capable_model() {
        let cfg = catalogue(vec![
            model("plain", ModelTier::Balanced, &[Modality::Text]),
            model(
                "agent",
                ModelTier::Frontier,
                &[Modality::Text, Modality::Tools],
            ),
        ]);
        // A tools request skips the non-tool model even though `plain` is the
        // closer tier match; capability is a hard constraint.
        let chosen = cfg
            .select_model(
                &req(&[Modality::Text, Modality::Tools]),
                ModelTier::Balanced,
            )
            .unwrap();
        assert_eq!(chosen.name, "agent");
        // Without the tools requirement, tier preference picks `plain`.
        let chosen = cfg
            .select_model(&req(&[Modality::Text]), ModelTier::Balanced)
            .unwrap();
        assert_eq!(chosen.name, "plain");
    }
}
