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

use std::io::Read;
use std::path::Path;

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

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");

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
            println!("cargo:warning={name}: cached copy failed checksum; re-downloading");
        }

        let url = format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{repo_path}");
        let mut response = ureq::get(&url)
            .call()
            .unwrap_or_else(|e| panic!("failed to download {name} from {url}: {e}"));

        let mut bytes = Vec::new();
        response
            .body_mut()
            .as_reader()
            .read_to_end(&mut bytes)
            .unwrap_or_else(|e| panic!("failed to read {name} response body: {e}"));

        let actual = sha256_hex(&bytes);
        assert_eq!(
            actual, expected_sha256,
            "{name} downloaded from {url} failed SHA-256 verification \
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

/// Lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
