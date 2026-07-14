//! Configuration: TOML deserialization, the model catalogue, API-key
//! resolution (plaintext / env / keyring), model selection, and startup
//! coverage validation.
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
    /// (`[classifier.gemini-embedding-001]`). Ignored unless selected.
    #[serde(default, rename = "gemini-embedding-001")]
    pub gemini_embedding_001: RemoteEmbeddingConfig,
    /// Settings for the `gemini-embedding-2` engine
    /// (`[classifier.gemini-embedding-2]`). Ignored unless selected.
    #[serde(default, rename = "gemini-embedding-2")]
    pub gemini_embedding_2: RemoteEmbeddingConfig,
    /// Settings for the `text-embedding-005` engine
    /// (`[classifier.text-embedding-005]`). Ignored unless selected. Uses the
    /// Vertex-specific shape (this model is Vertex-AI-only), not the shared
    /// [`RemoteEmbeddingConfig`] the Gemini/OpenAI families use.
    #[serde(default, rename = "text-embedding-005")]
    pub text_embedding_005: VertexEmbeddingConfig,
    /// Settings for the `text-embedding-3-small` engine
    /// (`[classifier.text-embedding-3-small]`). Ignored unless selected.
    #[serde(default, rename = "text-embedding-3-small")]
    pub text_embedding_3_small: RemoteEmbeddingConfig,
    /// Settings for the `text-embedding-3-large` engine
    /// (`[classifier.text-embedding-3-large]`). Ignored unless selected.
    #[serde(default, rename = "text-embedding-3-large")]
    pub text_embedding_3_large: RemoteEmbeddingConfig,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        ClassifierConfig {
            model: ClassifierModel::default(),
            image_generation_threshold: default_image_gen_threshold(),
            trivial_max_words: default_trivial_max_words(),
            deberta_v3_xsmall_zeroshot: DebertaV3XsmallZeroshotConfig::default(),
            gemini_embedding_001: RemoteEmbeddingConfig::default(),
            gemini_embedding_2: RemoteEmbeddingConfig::default(),
            text_embedding_005: VertexEmbeddingConfig::default(),
            text_embedding_3_small: RemoteEmbeddingConfig::default(),
            text_embedding_3_large: RemoteEmbeddingConfig::default(),
        }
    }
}

/// Settings for a remote embedding engine (any provider — the
/// `[classifier.<model>]` tables of the Gemini and OpenAI families all share
/// this shape). Remote engines have no local session pool; their "sessions"
/// are concurrent in-flight API requests.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RemoteEmbeddingConfig {
    /// API key for the provider. **Required when the engine is selected**
    /// (engine construction fails at startup without it). Resolves exactly
    /// like a routed model's `api_key`: a plaintext/env-expanded string or a
    /// `{ source = "keyring", service, user }` table; an empty resolved value
    /// counts as absent. Never logged.
    #[serde(default, deserialize_with = "resolve_api_key")]
    pub api_key: Option<String>,
    /// Endpoint override (e.g. a proxy/gateway, or a mock in tests). Must be
    /// http/https. Defaults to the provider's public endpoint. NOTE: the
    /// OpenAI engines append `/v1/embeddings`, so their override must not
    /// include a `/v1` suffix.
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
/// `access_token` instead of a plain `api_key`.
#[derive(Debug, Clone, Deserialize)]
pub struct VertexEmbeddingConfig {
    /// GCP project id. **Required when the engine is selected** (engine
    /// construction fails at startup without it).
    #[serde(default)]
    pub project: Option<String>,
    /// Vertex AI region, e.g. `us-central1`. Also selects the default regional
    /// endpoint host (`https://{location}-aiplatform.googleapis.com`) when
    /// `base_url` is not overridden.
    #[serde(default = "default_vertex_location")]
    pub location: String,
    /// OAuth 2.0 Bearer access token for the Vertex AI API. **Required when the
    /// engine is selected.** Resolves exactly like a routed model's `api_key`
    /// (a plaintext/env-expanded string or a keyring table); an empty resolved
    /// value counts as absent. Never logged.
    ///
    /// NOTE (Option 1): this is a *static* token. `gcloud`-printed tokens
    /// expire (~1h), so a long-running process needs an externally refreshed
    /// token; auto-refreshing Application Default Credentials is a planned
    /// follow-up.
    #[serde(default, deserialize_with = "resolve_api_key")]
    pub access_token: Option<String>,
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
}

impl Default for VertexEmbeddingConfig {
    fn default() -> Self {
        VertexEmbeddingConfig {
            project: None,
            location: default_vertex_location(),
            access_token: None,
            base_url: None,
            max_concurrency: None,
            request_timeout_secs: None,
        }
    }
}

fn default_vertex_location() -> String {
    "us-central1".to_string()
}

/// `[classifier.deberta-v3-xsmall-zeroshot]`: settings that only make sense
/// for the embedded NLI engine (local ORT sessions). Other engines have
/// their own concurrency models and their own tables.
#[derive(Debug, Clone, Default, Deserialize)]
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
    /// Resolved secret (from plaintext / env / keyring), or `None` for a keyless
    /// backend. An omitted field, an empty string, or a keyring value that
    /// resolves empty all mean "no auth": no `Authorization` header is sent.
    /// Never logged.
    #[serde(default, deserialize_with = "resolve_api_key")]
    pub api_key: Option<String>,
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

// ───────────────────────────────────────────────────────────────────────────
// API key resolution
// ───────────────────────────────────────────────────────────────────────────

/// Custom deserializer for a model's `api_key`. A TOML string yields the
/// plaintext/env-expanded key verbatim; an inline table
/// `{ source = "keyring", service = "...", user = "..." }` triggers an OS
/// keyring lookup at load time. Only the resolved secret is retained. An empty
/// resolved value becomes `None` (keyless backend), so `""` is treated the same
/// as omitting the field.
fn resolve_api_key<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    /// Empty string => no auth.
    fn non_empty(s: String) -> Option<String> {
        (!s.is_empty()).then_some(s)
    }

    struct ApiKeyVisitor;

    impl<'de> Visitor<'de> for ApiKeyVisitor {
        type Value = Option<String>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a plaintext API key string or a keyring lookup table")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<String>, E> {
            Ok(non_empty(v.to_owned()))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Option<String>, E> {
            Ok(non_empty(v))
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Option<String>, A::Error> {
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

            let source = source.ok_or_else(|| de::Error::missing_field("source"))?;
            if source != "keyring" {
                return Err(de::Error::custom(format!(
                    "unsupported api_key source `{source}`; expected `keyring`"
                )));
            }
            let service = service.ok_or_else(|| de::Error::missing_field("service"))?;
            let user = user.ok_or_else(|| de::Error::missing_field("user"))?;

            let entry = keyring::Entry::new(&service, &user).map_err(|e| {
                de::Error::custom(format!(
                    "keyring entry (service={service}, user={user}) unavailable: {e}"
                ))
            })?;
            let password = entry.get_password().map_err(|e| {
                de::Error::custom(format!(
                    "keyring lookup failed (service={service}, user={user}): {e}"
                ))
            })?;
            Ok(non_empty(password))
        }
    }

    deserializer.deserialize_any(ApiKeyVisitor)
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

/// Platform-appropriate config locations, user-scoped before system-wide.
fn well_known_config_paths() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dirs) = ProjectDirs::from("", "", APP_NAME) {
        candidates.push(dirs.config_dir().join("config.toml"));
    }
    #[cfg(unix)]
    candidates.push(PathBuf::from(format!("/etc/{APP_NAME}/config.toml")));
    candidates
}

/// Load, env-expand, parse, and validate the config at `path`.
pub fn load(path: &Path) -> anyhow::Result<RouterConfig> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read config `{}`: {e}", path.display()))?;
    let cfg = parse(&raw)?;
    cfg.validate()?;
    Ok(cfg)
}

/// Env-expand then TOML-parse config text. Split out for unit testing.
pub fn parse(raw: &str) -> anyhow::Result<RouterConfig> {
    let expanded = expand_env(raw)?;
    let cfg: RouterConfig = config::Config::builder()
        .add_source(config::File::from_str(&expanded, config::FileFormat::Toml))
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
        // 1. Filter by capability (superset), preserving config order.
        // 2 & 3–6. Rank survivors; `min_by_key` returns the first minimum, so a
        //          tie resolves toward the earlier-declared model.
        self.models
            .iter()
            .filter(|m| m.modality_set().is_superset(required))
            .min_by_key(|m| tier_rank(m.tier, complexity))
    }

    /// How many models can serve `required` (declare a superset of it). When this
    /// is `<= 1` the complexity tier is irrelevant — there is nothing to rank —
    /// so the proxy can skip classification entirely and route directly.
    pub fn candidate_count(&self, required: &ModalitySet) -> usize {
        self.models
            .iter()
            .filter(|m| m.modality_set().is_superset(required))
            .count()
    }
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

    // ── gemini engine tables ─────────────────────────────────────────
    #[test]
    fn gemini_engine_tables_parse_with_api_key_resolution() {
        std::env::set_var("ROUTER_TEST_GEMINI_KEY", "g-key");
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=\"gemini-embedding-001\"\n\
             [classifier.gemini-embedding-001]\napi_key=\"${ROUTER_TEST_GEMINI_KEY}\"\n\
             base_url=\"http://localhost:9\"\nmax_concurrency=8\nrequest_timeout_secs=5\n\
             [classifier.gemini-embedding-2]\napi_key=\"plain-key\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.model, ClassifierModel::GeminiEmbedding001);
        let g1 = &cfg.classifier.gemini_embedding_001;
        // Env-expanded, exactly like a routed model's key.
        assert_eq!(g1.api_key.as_deref(), Some("g-key"));
        assert_eq!(
            g1.base_url.as_ref().unwrap().as_str(),
            "http://localhost:9/"
        );
        assert_eq!(g1.max_concurrency, Some(8));
        assert_eq!(g1.request_timeout_secs, Some(5));
        assert_eq!(
            cfg.classifier.gemini_embedding_2.api_key.as_deref(),
            Some("plain-key")
        );
    }

    #[test]
    fn text_embedding_005_table_parses() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=\"text-embedding-005\"\n\
             [classifier.text-embedding-005]\nproject=\"my-proj\"\nlocation=\"us-east1\"\n\
             access_token=\"te5-token\"\nmax_concurrency=16\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.model, ClassifierModel::TextEmbedding005);
        let te5 = &cfg.classifier.text_embedding_005;
        assert_eq!(te5.project.as_deref(), Some("my-proj"));
        assert_eq!(te5.location, "us-east1");
        assert_eq!(te5.access_token.as_deref(), Some("te5-token"));
        assert_eq!(te5.max_concurrency, Some(16));
    }

    #[test]
    fn text_embedding_005_defaults_location_and_empty_token_is_none() {
        // Omitted table: location defaults, project/token absent, and an empty
        // access_token counts as absent (the engine then fails fast).
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier.text-embedding-005]\naccess_token=\"\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        )
        .unwrap();
        let te5 = &cfg.classifier.text_embedding_005;
        assert_eq!(te5.location, "us-central1");
        assert_eq!(te5.project, None);
        assert_eq!(te5.access_token, None);
        assert_eq!(te5.base_url, None);
    }

    #[test]
    fn gemini_table_defaults_and_empty_key_is_none() {
        // Omitted table: all defaults; empty api_key string counts as absent
        // (the engine then fails at startup with a clear message).
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier.gemini-embedding-001]\napi_key=\"\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.gemini_embedding_001.api_key, None);
        assert_eq!(cfg.classifier.gemini_embedding_001.base_url, None);
        assert_eq!(cfg.classifier.gemini_embedding_2.max_concurrency, None);
    }

    #[test]
    fn openai_engine_tables_parse() {
        let cfg = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier]\nmodel=\"text-embedding-3-small\"\n\
             [classifier.text-embedding-3-small]\napi_key=\"sk-small\"\nmax_concurrency=8\n\
             [classifier.text-embedding-3-large]\napi_key=\"sk-large\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        )
        .unwrap();
        assert_eq!(cfg.classifier.model, ClassifierModel::TextEmbedding3Small);
        assert_eq!(
            cfg.classifier.text_embedding_3_small.api_key.as_deref(),
            Some("sk-small")
        );
        assert_eq!(
            cfg.classifier.text_embedding_3_small.max_concurrency,
            Some(8)
        );
        assert_eq!(
            cfg.classifier.text_embedding_3_large.api_key.as_deref(),
            Some("sk-large")
        );
        // Omitted fields default.
        assert_eq!(cfg.classifier.text_embedding_3_large.base_url, None);
    }

    #[test]
    fn gemini_base_url_rejects_non_http_scheme() {
        let err = parse(
            "[server]\nhost=\"0.0.0.0\"\nport=1\n\
             [classifier.gemini-embedding-001]\nbase_url=\"ftp://x\"\n\
             [[models]]\nname=\"m\"\nbase_url=\"http://u\"\ntype=\"fast\"\nmodalities=[\"text\"]\n",
        );
        assert!(err.is_err());
    }

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
        assert_eq!(cfg.models[0].api_key.as_deref(), Some("sk-plain"));
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
        assert_eq!(cfg.models[0].api_key.as_deref(), Some("sk-from-env"));
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
        assert_eq!(cfg.models[0].api_key.as_deref(), Some("sk-keyring-secret"));
        let _ = entry.delete_credential();
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

    // ── shipped example config (drift guard) ──────────────────────────────────
    #[test]
    fn example_config_matches_current_schema() {
        // The checked-in example is documentation only (nothing parses it at
        // build or run time), so it can silently drift out of sync with the
        // schema. This guards that it still parses and passes startup
        // validation. Secret *resolution* (env vars, OS keyring) is covered by
        // dedicated tests, so neutralize those two constructs to a plaintext key
        // here rather than depend on external state.
        let raw = include_str!("../config.example.toml");
        let neutralized = raw
            .replace(
                "{ source = \"keyring\", service = \"hyper-mcp-router\", user = \"openai-frontier\" }",
                "\"sk-example\"",
            )
            .replace("${OPENAI_API_KEY}", "sk-example");

        let cfg =
            parse(&neutralized).expect("config.example.toml must parse under the current schema");
        cfg.validate()
            .expect("config.example.toml must pass startup validation");

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

    // ── select_model ──────────────────────────────────────────────────────────
    fn model(name: &str, tier: ModelTier, mods: &[Modality]) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            base_url: Url::parse("http://x").unwrap(),
            api_key: Some("k".to_string()),
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
