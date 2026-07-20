//! Build-time model embedding.
//!
//! Fetches the quantized ONNX NLI model and its tokenizer from HuggingFace into
//! `OUT_DIR` once, so `lib.rs` can `include_bytes!` them into the final binary.
//! Users never download anything at runtime.
//!
//! Integrity: artifacts are fetched from a **pinned revision** (never a moving
//! branch) and verified against pinned SHA-256 digests — both after download
//! and for cached copies — so a partial download, a corrupted cache, or an
//! upstream change can never be silently embedded. Files are written via a
//! temp-file + rename so the final path never holds a partial file.
//!
//! Offline / air-gapped builds: set `HYPER_MCP_ROUTER_ARTIFACT_DIR` to a
//! directory containing pre-fetched copies of the artifacts under their
//! destination names (`router_model.onnx`, `tokenizer.json`) and they are
//! read from there instead of downloaded. Vendored files go through the
//! same SHA-256 verification — a mismatching file fails the build.
//!
//! Note on a second download channel: this script only pins the *model*
//! artifacts. The `ort` crate's `download-binaries` feature separately
//! fetches the ONNX Runtime shared library at build time, with its own
//! checksum verification, outside this script's pinning regime.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

/// HuggingFace repository holding the model.
const REPO: &str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33";

/// Pinned repository revision (commit hash). Never a branch name: `main` can
/// move underneath us, silently changing what gets embedded.
const REVISION: &str = "262ae02f29173eec1c250f90804dc7edc677dcff";

/// (destination file name in `OUT_DIR`, path within the repo, expected SHA-256)
const ARTIFACTS: [(&str, &str, &str); 2] = [
    (
        "router_model.onnx",
        "onnx/model_quantized.onnx",
        "1fbf78872ceab4e8003eaa0696896008c019d5a239be734c5d6230f312ad480e",
    ),
    (
        "tokenizer.json",
        "tokenizer.json",
        "05402ffae6dd382a8491b1d29bfc139bec5d332662e86a026f433ce54c25c202",
    ),
];

fn main() {
    // Only re-run when the build script itself changes; the pinned artifacts
    // are verified by digest on every build, so they cannot drift.
    println!("cargo:rerun-if-changed=build.rs");
    // …and when toggling a docs.rs-style build, so stubs and real artifacts
    // can never be mistaken for each other across builds (the digest check
    // below re-downloads over a stub regardless).
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    // …and when the vendored-artifact directory changes, so switching between
    // offline and online sourcing is picked up.
    println!("cargo:rerun-if-env-changed=HYPER_MCP_ROUTER_ARTIFACT_DIR");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");

    // docs.rs builds run with NO network access, so the downloads below can
    // never succeed there. Rustdoc never executes the model, so empty stubs
    // keep `include_bytes!` compiling. (`ort-sys` handles DOCS_RS the same
    // way and skips its own binary download.)
    if std::env::var_os("DOCS_RS").is_some() {
        for (name, _, _) in ARTIFACTS {
            std::fs::write(Path::new(&out_dir).join(name), [])
                .unwrap_or_else(|e| panic!("failed to write docs.rs stub {name}: {e}"));
        }
        return;
    }

    // Vendored-artifact escape hatch for air-gapped/offline builds: read the
    // artifacts from this directory instead of downloading. Verification
    // below still applies — vendoring bypasses the network, not the pinning.
    let vendor_dir = std::env::var_os("HYPER_MCP_ROUTER_ARTIFACT_DIR").map(PathBuf::from);

    for (name, repo_path, expected_sha256) in ARTIFACTS {
        let dest = Path::new(&out_dir).join(name);

        // Trust the cache only if its digest matches; a stale or truncated
        // file is re-fetched rather than embedded forever.
        if dest.exists() {
            let cached = std::fs::read(&dest)
                .unwrap_or_else(|e| panic!("failed to read cached {}: {e}", dest.display()));
            if sha256_hex(&cached) == expected_sha256 {
                continue;
            }
            println!("cargo:warning={name}: cached copy failed checksum; re-fetching");
        }

        let (bytes, source) = match &vendor_dir {
            Some(dir) => read_vendored(dir, name),
            None => download(name, repo_path),
        };

        let actual = sha256_hex(&bytes);
        assert_eq!(
            actual, expected_sha256,
            "{name} from {source} failed SHA-256 verification \
             (expected {expected_sha256}, got {actual})"
        );

        // Temp-file + rename: the final path is only ever complete + verified.
        let tmp = dest.with_extension("part");
        std::fs::write(&tmp, &bytes)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", tmp.display()));
        std::fs::rename(&tmp, &dest)
            .unwrap_or_else(|e| panic!("failed to move {} into place: {e}", dest.display()));
    }
}

/// Read a vendored artifact from `HYPER_MCP_ROUTER_ARTIFACT_DIR`. Returns the
/// bytes and a source description for error messages. The caller still
/// verifies the digest — a wrong or stale vendored file fails the build.
fn read_vendored(dir: &Path, name: &str) -> (Vec<u8>, String) {
    let path = dir.join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "HYPER_MCP_ROUTER_ARTIFACT_DIR is set but {} could not be read: {e}",
            path.display()
        )
    });
    (bytes, path.display().to_string())
}

/// Download an artifact from the pinned HuggingFace revision. Returns the
/// bytes and the URL for error messages. The caller verifies the digest.
fn download(name: &str, repo_path: &str) -> (Vec<u8>, String) {
    let url = format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{repo_path}");

    // A global timeout so a stalled connection can never hang the build
    // indefinitely; 10 minutes is generous for the ~87 MB model on a slow
    // link (~1.2 Mbit/s sustained).
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .into();

    let mut response = agent
        .get(&url)
        .call()
        .unwrap_or_else(|e| panic!("failed to download {name} from {url}: {e}"));

    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("failed to read {name} response body: {e}"));

    (bytes, url)
}

/// Lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
