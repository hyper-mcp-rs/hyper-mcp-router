//! Config **loading**: well-known path discovery, extension-driven format
//! selection (TOML / YAML / JSON), environment-variable expansion, and
//! parsing into [`RouterConfig`]. The schema itself (structs, custom
//! deserializers, validation, model selection) lives in the parent module;
//! this file owns everything between "a path or raw text" and "a parsed
//! config".

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use directories::ProjectDirs;
use regex::Regex;

use super::{RouterConfig, APP_NAME};

// ───────────────────────────────────────────────────────────────────────────
// Environment-variable expansion
// ───────────────────────────────────────────────────────────────────────────

/// Expand `${VAR}` / `${VAR:-default}` references in the raw config text
/// before parsing (any format — expansion runs on the raw text). `$${VAR}` is
/// an escape hatch for a literal `${VAR}`, and bare `$VAR` is left untouched.
/// All missing (no-default) variables are collected and reported together in
/// a single error.
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
// Path resolution and format selection
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

// ───────────────────────────────────────────────────────────────────────────
// Loading and parsing
// ───────────────────────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelApiKey;
    use crate::modality::Modality;

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
            "server:\n  host: \"0.0.0.0\"\n  port: 1\nmodels:\n  - name: m\n    base_url: \"http://u\"\n    api_key: \"${ROUTER_TEST_YAML_KEY}\"\n    type: fast\n    modalities: [text]\n    context_window: 128000\n  - name: adc\n    base_url: \"http://u\"\n    api_key: { source: google-adc }\n    type: frontier\n    modalities: [text]\n    context_window: 128000\n",
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
                        "modalities": ["text"],
                        "context_window": 128000
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
}
