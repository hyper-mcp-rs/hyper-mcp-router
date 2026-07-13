//! Build-time model embedding.
//!
//! Fetches the quantized ONNX NLI model and its tokenizer from HuggingFace into
//! `OUT_DIR` once, so `lib.rs` can `include_bytes!` them into the final binary.
//! Users never download anything at runtime. Downloads are skipped if the files
//! already exist in `OUT_DIR`, so repeated builds do not re-fetch.

use std::io::Read;
use std::path::Path;

/// (destination file name in `OUT_DIR`, source URL)
const ARTIFACTS: [(&str, &str); 2] = [
    (
        "router_model.onnx",
        "https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33/resolve/main/onnx/model_quantized.onnx",
    ),
    (
        "tokenizer.json",
        "https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33/resolve/main/tokenizer.json",
    ),
];

fn main() {
    // Only re-run when the build script itself changes; the downloaded model is
    // content-addressed by presence in OUT_DIR and never changes underneath us.
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");

    for (name, url) in ARTIFACTS {
        let dest = Path::new(&out_dir).join(name);
        if dest.exists() {
            continue;
        }

        let mut response = ureq::get(url)
            .call()
            .unwrap_or_else(|e| panic!("failed to download {name} from {url}: {e}"));

        let mut bytes = Vec::new();
        response
            .body_mut()
            .as_reader()
            .read_to_end(&mut bytes)
            .unwrap_or_else(|e| panic!("failed to read {name} response body: {e}"));

        std::fs::write(&dest, &bytes)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", dest.display()));
    }
}
