//! Configuration: TOML/YAML/JSON deserialization, the model catalogue,
//! API-key resolution (plaintext / env / keyring), model selection, and
//! startup coverage validation.
//!
//! Requests and responses elsewhere are handled as raw JSON; this module is the
//! only place typed structs are used, and only for the operator's config file.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use directories::ProjectDirs;
use regex::Regex;
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
    /// Which classification model to run (exactly one per process; see
    /// `classifier::ClassifierModel`). **Config-only** — there is no CLI
    /// override, because each model brings its own configuration; different
    /// models mean different config files. Defaults to the embedded
    /// `deberta-v3-xsmall-zeroshot` model.
    #[serde(default)]
    pub model: ClassifierModel,
    /// Score floor for the image-generation axis (scale is engine-specific;
    /// P(entailment) for the zero-shot engine).
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
            model: ClassifierModel::default(),
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

// ───────────────────────────────────────────────────────────────────────────
// Environment-variable expansion
// ───────────────────────────────────────────────────────────────────────────

/// Expand `${VAR}` / `${VAR:-default}` references in the raw config text before
/// TOML parsing. `$${VAR}` is an escape hatch for a literal `${VAR}`, and bare
/// `$VAR` is left untouched. All missing (no-default) variables are collected
/// and reported together in a single error.
pub fn expand_env(input: &str) -> anyhow::Result<String> {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")
            .expect("valid env-expansion regex")
    });

    let mut out = String::with_capacity(input.len());
    let mut last = 0;
    let mut missing: Vec<String> = Vec::new();

    for caps in RE.captures_iter(input) {
        let m = caps.get(0).unwrap();
        out.push_str(&input[last..m.start()]);
        last = m.end();

        let escaped = caps.get(1).is_some_and(|g| !g.as_str().is_empty());
        let var = &caps[2];
        let default = caps.get(4).map(|g| g.as_str());

        if escaped {
            // `$${VAR[:-default]}` → literal `${VAR[:-default]}`.
            out.push_str("${");
            out.push_str(var);
            if let Some(d) = default {
                out.push_str(":-");
                out.push_str(d);
            }
            out.push('}');
            continue;
        }

        match std::env::var(var) {
            Ok(v) if !v.is_empty() => out.push_str(&v),
            _ => match default {
                Some(d) => out.push_str(d),
                None => missing.push(var.to_string()),
            },
        }
    }
    out.push_str(&input[last..]);

    if !missing.is_empty() {
        anyhow::bail!(
            "unset environment variable(s) with no default: {}",
            missing.join(", ")
        );
    }
    Ok(out)
}

// ───────────────────────────────────────────────────────────────────────────
// Loading, path resolution, validation
// ───────────────────────────────────────────────────────────────────────────

/// Resolve the config path: an explicit `--config` is used verbatim (a missing
/// or unparseable file is then fatal — no fallback); otherwise the first
/// existing well-known OS location is used. Returns an error listing every path
/// searched when none exists.
pub fn resolve_config_path(explicit: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }

    let candidates = well_known_config_paths();
    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    let searched = candidates
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n  ");
    anyhow::bail!(
        "no config file found. Pass --config <path> or create one at a well-known location. Searched:\n  {searched}"
    );
}

/// Config file names probed in each well-known directory, in priority order.
const CONFIG_FILE_NAMES: [&str; 4] = ["config.toml", "config.yaml", "config.yml", "config.json"];

/// Platform-appropriate config locations, user-scoped before system-wide.
/// Within a directory, `config.toml` wins over `config.yaml`/`config.yml`,
/// which win over `config.json`.
fn well_known_config_paths() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dirs) = ProjectDirs::from("", "", APP_NAME) {
        for name in CONFIG_FILE_NAMES {
            candidates.push(dirs.config_dir().join(name));
        }
    }
    #[cfg(unix)]
    for name in CONFIG_FILE_NAMES {
        candidates.push(PathBuf::from(format!("/etc/{APP_NAME}/{name}")));
    }
    candidates
}

/// Map a config path to its parse format by file extension
/// (case-insensitive). Anything else is a loud error rather than a silent
/// guess at the wrong format.
fn format_for_path(path: &Path) -> anyhow::Result<config::FileFormat> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("toml") => Ok(config::FileFormat::Toml),
        Some("yaml") | Some("yml") => Ok(config::FileFormat::Yaml),
        Some("json") => Ok(config::FileFormat::Json),
        _ => anyhow::bail!(
            "unsupported config file extension for `{}`: expected `.toml`, `.yaml`, `.yml`, or `.json`",
            path.display()
        ),
    }
}

/// Load, env-expand, parse, and validate the config at `path`. The format is
/// chosen by file extension (`.toml`, `.yaml`/`.yml`, or `.json`).
pub fn load(path: &Path) -> anyhow::Result<RouterConfig> {
    let format = format_for_path(path)?;
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read config `{}`: {e}", path.display()))?;
    let cfg = parse_with_format(&raw, format)?;
    cfg.validate()?;
    Ok(cfg)
}

/// Env-expand then TOML-parse config text. Split out for unit testing.
pub fn parse(raw: &str) -> anyhow::Result<RouterConfig> {
    parse_with_format(raw, config::FileFormat::Toml)
}

/// Env-expand then parse config text in the given format. Env expansion runs
/// on the raw text, so `${VAR}` works identically in TOML, YAML, and JSON.
pub fn parse_with_format(raw: &str, format: config::FileFormat) -> anyhow::Result<RouterConfig> {
    let expanded = expand_env(raw)?;
    let cfg: RouterConfig = config::Config::builder()
        .add_source(config::File::from_str(&expanded, format))
        .build()?
        .try_deserialize()
        .map_err(|e| anyhow::anyhow!("failed to parse config: {e}"))?;
    Ok(cfg)
}

impl RouterConfig {
    /// Field-level and startup coverage validation. Fails fast with a clear
    /// message on any incomplete configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.models.is_empty() {
            anyhow::bail!("no [[models]] configured");
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

    // ── env expansion ───────────────────────────────────────────────────────
    #[test]
    fn expand_simple_var() {
        std::env::set_var("ROUTER_TEST_A", "value-a");
        assert_eq!(expand_env("x=${ROUTER_TEST_A}").unwrap(), "x=value-a");
    }

    #[test]
    fn expand_default_used_when_unset() {
        std::env::remove_var("ROUTER_TEST_UNSET_1");
        assert_eq!(
            expand_env("x=${ROUTER_TEST_UNSET_1:-fallback}").unwrap(),
            "x=fallback"
        );
    }

    #[test]
    fn expand_default_ignored_when_set() {
        std::env::set_var("ROUTER_TEST_B", "real");
        assert_eq!(
            expand_env("x=${ROUTER_TEST_B:-fallback}").unwrap(),
            "x=real"
        );
    }

    #[test]
    fn expand_escape_hatch_literal() {
        assert_eq!(expand_env("x=$${VAR}").unwrap(), "x=${VAR}");
    }

    #[test]
    fn expand_bare_var_passthrough() {
        assert_eq!(expand_env("x=$VAR").unwrap(), "x=$VAR");
    }

    #[test]
    fn expand_collects_all_missing() {
        std::env::remove_var("ROUTER_TEST_MISS_1");
        std::env::remove_var("ROUTER_TEST_MISS_2");
        let err = expand_env("${ROUTER_TEST_MISS_1} ${ROUTER_TEST_MISS_2}").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ROUTER_TEST_MISS_1"));
        assert!(msg.contains("ROUTER_TEST_MISS_2"));
    }

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
            cfg.classifier.model,
            ClassifierModel::DebertaV3XsmallZeroshot
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
            cfg.classifier.model,
            ClassifierModel::DebertaV3XsmallZeroshot
        );

        // An unknown model id must be a loud config error, never a silent default.
        let err = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n[classifier]\nmodel=\"not-a-model\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        );
        assert!(err.is_err());
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

    // ── config file formats ───────────────────────────────────────────
    #[test]
    fn format_for_path_maps_known_extensions() {
        use config::FileFormat;
        let cases = [
            ("config.toml", FileFormat::Toml),
            ("config.yaml", FileFormat::Yaml),
            ("config.yml", FileFormat::Yaml),
            ("config.json", FileFormat::Json),
            // Extension matching must be case-insensitive.
            ("CONFIG.TOML", FileFormat::Toml),
            ("config.YAML", FileFormat::Yaml),
        ];
        for (name, want) in cases {
            let got = format_for_path(Path::new(name)).expect(name);
            // `FileFormat` isn't PartialEq; compare debug representations.
            assert_eq!(format!("{got:?}"), format!("{want:?}"), "path {name}");
        }
    }

    #[test]
    fn format_for_path_rejects_unknown_or_missing_extension() {
        for name in ["config.ini", "config", "config.tml"] {
            let err = format_for_path(Path::new(name)).unwrap_err();
            assert!(
                err.to_string()
                    .contains("unsupported config file extension"),
                "path {name}: got {err}"
            );
        }
    }

    #[test]
    fn yaml_config_parses_like_toml() {
        std::env::set_var("ROUTER_TEST_YAML_KEY", "yaml-key");
        let cfg = parse_with_format(
            "server:\n  host: \"0.0.0.0\"\n  port: 1\nmodels:\n  - name: m\n    base_url: \"http://u\"\n    api_key: \"${ROUTER_TEST_YAML_KEY}\"\n    type: fast\n    modalities: [text]\n  - name: adc\n    base_url: \"http://u\"\n    api_key: { source: google-adc }\n    type: frontier\n    modalities: [text]\n",
            config::FileFormat::Yaml,
        )
        .expect("YAML config should parse");
        cfg.validate().expect("YAML config should validate");
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.models.len(), 2);
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Static("yaml-key".to_string()))
        );
        // A map-shaped api_key (YAML mapping instead of a TOML inline table)
        // must hit the same custom deserializer path.
        assert_eq!(cfg.models[1].api_key, Some(ModelApiKey::GoogleAdc));
    }

    #[test]
    fn json_config_parses_like_toml() {
        std::env::set_var("ROUTER_TEST_JSON_KEY", "json-key");
        let cfg = parse_with_format(
            r#"{
                "server": { "host": "0.0.0.0", "port": 1 },
                "models": [
                    {
                        "name": "m",
                        "base_url": "http://u",
                        "api_key": "${ROUTER_TEST_JSON_KEY}",
                        "type": "fast",
                        "modalities": ["text"]
                    }
                ]
            }"#,
            config::FileFormat::Json,
        )
        .expect("JSON config should parse");
        cfg.validate().expect("JSON config should validate");
        assert_eq!(cfg.models.len(), 1);
        assert_eq!(
            cfg.models[0].api_key,
            Some(ModelApiKey::Static("json-key".to_string()))
        );
    }

    #[test]
    fn well_known_paths_probe_all_formats_toml_first() {
        let paths = well_known_config_paths();
        assert!(!paths.is_empty());
        // Every candidate is one of the supported file names, and within each
        // directory TOML is probed before YAML before JSON.
        let names: Vec<&str> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        for chunk in names.chunks(CONFIG_FILE_NAMES.len()) {
            assert_eq!(chunk, CONFIG_FILE_NAMES, "candidates {names:?}");
        }
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

    // ── shipped example configs (drift guard) ─────────────────────────

    /// Parse + validate a checked-in example. Secret *resolution* (env vars,
    /// OS keyring) is covered by dedicated tests, so neutralize those two
    /// constructs (in their format-specific spelling) to a plaintext key here
    /// rather than depend on external state.
    fn parse_example(
        label: &str,
        raw: &str,
        keyring_table: &str,
        format: config::FileFormat,
    ) -> RouterConfig {
        let neutralized = raw
            .replace(keyring_table, "\"sk-example\"")
            .replace("${OPENAI_API_KEY}", "sk-example");
        let cfg = parse_with_format(&neutralized, format)
            .unwrap_or_else(|e| panic!("{label} must parse under the current schema: {e}"));
        cfg.validate()
            .unwrap_or_else(|e| panic!("{label} must pass startup validation: {e}"));
        cfg
    }

    fn toml_example() -> RouterConfig {
        parse_example(
            "config.example.toml",
            include_str!("../../config.example.toml"),
            "{ source = \"keyring\", service = \"hyper-mcp-router\", user = \"openai-frontier\" }",
            config::FileFormat::Toml,
        )
    }

    #[test]
    fn example_config_matches_current_schema() {
        // The checked-in examples are documentation only (nothing parses them
        // at build or run time), so they can silently drift out of sync with
        // the schema. This guards that the canonical TOML example still parses
        // and passes startup validation.
        let cfg = toml_example();

        // Sanity: the example still demonstrates its documented features.
        assert!(
            cfg.models.iter().any(|m| m.api_key.is_none()),
            "example should include a keyless backend"
        );
        assert!(
            cfg.models
                .iter()
                .any(|m| m.modality_set().contains(Modality::AudioOutput)),
            "example should cover audio-output"
        );
    }

    #[test]
    fn yaml_example_config_is_equivalent_to_toml() {
        let cfg = parse_example(
            "config.example.yaml",
            include_str!("../../config.example.yaml"),
            "{ source: \"keyring\", service: \"hyper-mcp-router\", user: \"openai-frontier\" }",
            config::FileFormat::Yaml,
        );
        // The YAML example is documented as an exact translation of the TOML
        // one; comparing the fully-parsed configs guards against drift in
        // either direction.
        assert_eq!(format!("{cfg:?}"), format!("{:?}", toml_example()));
    }

    #[test]
    fn json_example_config_is_equivalent_to_toml() {
        let cfg = parse_example(
            "config.example.json",
            include_str!("../../config.example.json"),
            "{ \"source\": \"keyring\", \"service\": \"hyper-mcp-router\", \"user\": \"openai-frontier\" }",
            config::FileFormat::Json,
        );
        assert_eq!(format!("{cfg:?}"), format!("{:?}", toml_example()));
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
