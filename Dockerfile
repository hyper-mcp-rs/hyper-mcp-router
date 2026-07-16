# syntax=docker/dockerfile:1.7
#
# Multi-stage build producing the SMALLEST possible image: a `scratch`
# container holding a fully statically linked (musl) executable plus CA
# certificates — nothing else. Supports linux/amd64 and linux/arm64.
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t hyper-mcp-router .
#
# How the static link works:
#   * pyke's prebuilt ONNX Runtime binaries (the `download-binaries` feature)
#     only cover *-linux-gnu, so stage 1 compiles ONNX Runtime from source as
#     STATIC libraries against musl.
#   * ort-sys checks ORT_LIB_PATH *before* its download-binaries path, so the
#     Cargo.toml feature stays (it still serves native dev builds); setting
#     the env var here overrides it.
#   * Rust's *-unknown-linux-musl targets default to +crt-static; we assert
#     it explicitly because the official rust:alpine images set
#     RUSTFLAGS="-C target-feature=-crt-static", which would silently produce
#     a dynamic binary.
#
# Build-time notes:
#   * The ONNX Runtime compile is the expensive step: ~30-60 min on native
#     hardware, HOURS under QEMU emulation. For multi-arch, prefer one native
#     builder per architecture (docker buildx with remote/native nodes, or
#     per-arch CI runners + `docker manifest`), not QEMU.
#   * Stage 2's `cargo build` downloads the embedded model (~87 MB, pinned
#     revision + SHA-256, see build.rs) — network access is required once;
#     the BuildKit cache mount keeps it across rebuilds.
#   * Both stages use the same base image so ONNX Runtime's static archives
#     and the final link see identical musl/gcc versions.

ARG RUST_IMAGE=rust:1.97-alpine

# ───────────────────────────────────────────────────────────────────────────
# Stage 1: ONNX Runtime, compiled from source as static musl libraries.
# CPU-only execution providers — matches what the router uses.
# ───────────────────────────────────────────────────────────────────────────
FROM ${RUST_IMAGE} AS onnxruntime

RUN apk add --no-cache \
    build-base \
    cmake \
    git \
    linux-headers \
    python3 \
    py3-packaging \
    py3-psutil

# Keep in lockstep with the ort crate: ort-sys 2.0.0-rc.12 tracks upstream
# ONNX Runtime 1.24.2 (see ort-sys build/download/dist.txt: "ms@1.24.2").
# Deliberately NOT Renovate-managed — never bump this on its own; it moves
# together with the `ort` dependency in Cargo.toml.
ARG ONNXRUNTIME_VERSION=v1.24.2

RUN git clone --depth 1 --shallow-submodules --recursive \
    --branch ${ONNXRUNTIME_VERSION} \
    https://github.com/microsoft/onnxruntime /src/onnxruntime

WORKDIR /src/onnxruntime

# musl patch: <execinfo.h> (glibc's backtrace API) is included unconditionally
# on Linux, but every USE of it sits behind `#if !defined(NDEBUG)` — and
# MinSizeRel defines NDEBUG — so guarding the include is sufficient and
# correct. musl has no execinfo.h; __has_include makes this build on both.
RUN sed -i 's|#include <execinfo.h>|#if __has_include(<execinfo.h>)\n#include <execinfo.h>\n#endif|' \
    onnxruntime/core/platform/posix/stacktrace.cc && \
    grep -A2 '__has_include' onnxruntime/core/platform/posix/stacktrace.cc

# No --build_shared_lib => static archives. MinSizeRel favours binary size.
# CMake fetches ONNX Runtime's own third-party deps (abseil, protobuf, …)
# during configure, so this step needs network too.
RUN python3 tools/ci_build/build.py \
    --build_dir build \
    --config MinSizeRel \
    --parallel \
    --skip_tests \
    --skip_submodule_sync \
    --allow_running_as_root \
    --compile_no_warning_as_error \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF && \
    # ort-sys links libre2.a unconditionally, but with unit tests disabled
    # nothing in ORT itself depends on re2 (it's EXCLUDE_FROM_ALL), so the
    # archive never gets built — build that one target explicitly.
    cmake --build build/MinSizeRel --target re2 --parallel && \
    test -f build/MinSizeRel/_deps/re2-build/libre2.a

# ───────────────────────────────────────────────────────────────────────────
# Stage 2: the router, statically linked against stage 1's ONNX Runtime.
# ───────────────────────────────────────────────────────────────────────────
FROM ${RUST_IMAGE} AS builder

# build-base/linux-headers: C/C++ build deps in the graph (oniguruma, esaxx).
# openssl/zlib dev + static: build.rs's model download uses ureq, which pulls
# native-tls on Linux — a BUILD-time dependency only; the router's own
# runtime TLS is rustls. Host build units on musl link statically by
# default, hence the static libs.
# ca-certificates: the bundle copied into the scratch image (and used by the
# build-time model download).
RUN apk add --no-cache \
    build-base \
    linux-headers \
    pkgconfig \
    openssl-dev \
    openssl-libs-static \
    zlib-dev \
    zlib-static \
    ca-certificates

WORKDIR /app

# ONNX Runtime static archives from stage 1. ort-sys walks this directory
# (profile dir + _deps/) and links every archive statically. ORT_LIB_PATH
# takes precedence over the crate's `download-binaries` feature (which stays
# in Cargo.toml for native dev builds); ORT_SKIP_DOWNLOAD makes that
# non-negotiable — if the lib path were ever not honored, the build fails
# loudly instead of falling back to a fetch.
COPY --from=onnxruntime /src/onnxruntime/build/MinSizeRel /opt/onnxruntime
ENV ORT_LIB_PATH=/opt/onnxruntime \
    ORT_SKIP_DOWNLOAD=1

# +crt-static: fully static executable (overrides the rust:alpine default of
# -crt-static). strip=symbols: smallest binary. link-arg=-lgcc: ONNX
# Runtime's x86 cpuid_info.cc uses GCC's __builtin_cpu_supports, whose
# runtime symbols (__cpu_model / __cpu_features2) live in libgcc's cpuinfo.o
# — rustc links its own compiler-builtins instead of libgcc on static musl
# targets and lacks them, so the amd64 link fails without this (aarch64 has
# no such init path; the extra arg is harmless there). These flags apply
# only to --target units — host build scripts / proc-macros are unaffected.
ENV RUSTFLAGS="-C target-feature=+crt-static -C strip=symbols -C link-arg=-lgcc"

COPY rust-toolchain.toml Cargo.toml Cargo.lock build.rs ./
COPY src ./src

# TARGETARCH is provided by buildx (amd64|arm64). Building natively per
# platform means host triple == target triple; passing --target explicitly
# keeps RUSTFLAGS off the host units regardless.
ARG TARGETARCH
RUN case "${TARGETARCH}" in \
      amd64) echo x86_64-unknown-linux-musl  > /rust-target ;; \
      arm64) echo aarch64-unknown-linux-musl > /rust-target ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked --target "$(cat /rust-target)" && \
    cp "target/$(cat /rust-target)/release/hyper-mcp-router" /hyper-mcp-router

# Fail the build if the executable is not fully static: a static(-PIE)
# binary has no DT_NEEDED entries — any shared-object requirement would make
# the scratch image unrunnable. Then smoke-test that it actually executes.
RUN if readelf -d /hyper-mcp-router | grep -q 'NEEDED'; then \
      echo 'ERROR: binary is dynamically linked:' >&2; \
      readelf -d /hyper-mcp-router | grep 'NEEDED' >&2; \
      exit 1; \
    fi && \
    /hyper-mcp-router --version

# ───────────────────────────────────────────────────────────────────────────
# Stage 3: scratch — the executable and CA certificates. Nothing else.
# ───────────────────────────────────────────────────────────────────────────
FROM scratch

COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /hyper-mcp-router /hyper-mcp-router

# Mount the config at one of the probed well-known paths, e.g.:
#   docker run -v ./config.toml:/etc/hyper-mcp-router/config.toml:ro \
#     -p 8080:8080 hyper-mcp-router
# (`server.host` must be "0.0.0.0" to be reachable from outside the
# container.) Logs go to stdout — there is no writable filesystem here, and
# no shell: use `serve --config /path` as arguments if you mount elsewhere.
#
# For `api_key = { source = "google-adc" }` backends (and the Vertex AI
# classifier engines), mount Application Default Credentials and point the
# standard env var at them — there is no gcloud in this image:
#   docker run -v ./config.toml:/etc/hyper-mcp-router/config.toml:ro \
#     -v ~/.config/gcloud/application_default_credentials.json:/gcloud/adc.json:ro \
#     -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud/adc.json \
#     -p 8080:8080 hyper-mcp-router
# (On GCE/Cloud Run/GKE the metadata server provides ADC; no mount needed.)
#
# NOTE: `api_key = { source = "keyring" }` requires an OS secret store and
# will not work in this image; use env-expanded or plaintext keys instead.
USER 65532:65532
EXPOSE 8080

ENTRYPOINT ["/hyper-mcp-router"]
CMD ["serve", "--log-stdout"]
