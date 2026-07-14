//! Inference parallelism planning: how many classifier sessions the host can
//! support, constrained by **both** CPU and memory.
//!
//! CPU comes from `std::thread::available_parallelism()` (container-aware).
//! Memory comes from [`detect_memory_budget`], which prefers the **cgroup
//! limit** over the host total: inside containers (Cloud Run, Kubernetes,
//! Docker) `/proc/meminfo` — and therefore `sysinfo`'s plain `total_memory()`
//! — reports the *host/VM* memory, while the enforced limit lives in the
//! cgroup files. Sizing against the host total would overshoot the limit and
//! get the instance OOM-killed.
//!
//! The per-session and baseline costs are **measured**, not guessed — see
//! `scripts/measure_session_memory.sh` (weights ~105 MB/session at startup,
//! ~190 MB/session under sustained max-length load, ~420 MB fixed baseline).

/// Measured steady-state memory cost per inference session under sustained
/// max-length load (weights + retained activation arena), rounded up.
pub const SESSION_MEMORY_BYTES: u64 = 200 * 1024 * 1024;

/// Measured fixed process baseline (binary, embedded model bytes, tokenizer,
/// tokio runtime) independent of pool size.
pub const BASELINE_MEMORY_BYTES: u64 = 420 * 1024 * 1024;

/// How to split the host's available resources between concurrent inference
/// sessions (`pool_size`) and intra-op threads per session (`intra_op_threads`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InferencePlan {
    /// Number of independent sessions (max concurrent inferences).
    pub pool_size: usize,
    /// ORT intra-op threads per session (0 = let the runtime decide).
    pub intra_op_threads: usize,
}

/// The fraction of the memory budget the plan is allowed to consume (90%);
/// the rest is headroom for request buffers, TLS, allocator slack, and the
/// roughness of the measured constants.
fn usable_memory(budget: u64) -> u64 {
    budget / 10 * 9
}

/// How many sessions fit in `memory_budget` bytes after the fixed baseline,
/// at 90% utilisation. Can be 0 (the caller decides whether to clamp; the
/// router always needs at least one session to classify at all).
pub fn max_sessions_for_memory(memory_budget: u64) -> usize {
    (usable_memory(memory_budget).saturating_sub(BASELINE_MEMORY_BYTES) / SESSION_MEMORY_BYTES)
        as usize
}

/// Derive an [`InferencePlan`] from the available cores and (when known) the
/// memory budget. The embedded NLI model is small and scales poorly
/// per-inference, so we favor concurrency: cap intra-op parallelism at 2 and
/// give each session ~2 cores — then take the **minimum** of the core-based
/// and memory-based pool sizes. Always yields at least one (single-threaded)
/// session, even when the memory budget says zero fit: the router cannot
/// classify with no sessions, and the operator is warned instead (see
/// [`overcommit_warnings`]). Never budgets more threads than cores
/// (`pool_size * intra_op_threads <= cores`).
pub fn plan_inference(available_cores: usize, memory_budget: Option<u64>) -> InferencePlan {
    let cores = available_cores.max(1);
    let intra_op_threads = if cores >= 2 { 2 } else { 1 };
    let core_pool = (cores / intra_op_threads).max(1);
    let pool_size = match memory_budget {
        Some(budget) => core_pool.min(max_sessions_for_memory(budget)).max(1),
        None => core_pool,
    };
    InferencePlan {
        pool_size,
        intra_op_threads,
    }
}

/// Detect the memory budget available to this process, in bytes.
///
/// Prefers the **cgroup limit** (Linux containers — what Cloud Run gen2,
/// Kubernetes, and Docker actually enforce) and falls back to the host/VM
/// total elsewhere (macOS, bare Linux, or gVisor sandboxes like Cloud Run
/// gen1, where `/proc/meminfo` is itself virtualised to the instance limit).
/// Returns `None` when nothing sensible can be read; the plan is then
/// CPU-bound only.
pub fn detect_memory_budget() -> Option<u64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    if let Some(limits) = sys.cgroup_limits() {
        if limits.total_memory > 0 {
            return Some(limits.total_memory);
        }
    }
    let total = sys.total_memory();
    (total > 0).then_some(total)
}

/// Warnings for a **final** (possibly operator-overridden) pool configuration
/// that exceeds what the host can handle. Pure and returned as strings so it
/// is unit-testable; the caller logs them. Explicit settings are respected —
/// never clamped — but never silently.
pub fn overcommit_warnings(
    pool_size: usize,
    intra_op_threads: usize,
    available_cores: usize,
    memory_budget: Option<u64>,
) -> Vec<String> {
    let mut warnings = Vec::new();

    // CPU: `0` intra-op threads means "runtime default"; treat it as 1 for
    // the estimate (conservative — the real default may be higher).
    let threads = pool_size * intra_op_threads.max(1);
    let cores = available_cores.max(1);
    if threads > cores {
        warnings.push(format!(
            "inference pool oversubscribes the CPU: pool_size {pool_size} x \
             intra_op_threads {} = {threads} threads > {cores} detected cores; \
             expect degraded latency",
            intra_op_threads.max(1),
        ));
    }

    if let Some(budget) = memory_budget {
        let estimated = BASELINE_MEMORY_BYTES + pool_size as u64 * SESSION_MEMORY_BYTES;
        if estimated > budget {
            warnings.push(format!(
                "inference pool may exceed available memory: estimated ~{} MiB \
                 (baseline ~{} MiB + {pool_size} sessions x ~{} MiB, measured under \
                 max-length load) > detected budget {} MiB; risk of OOM kill — \
                 consider lowering --inference-pool-size",
                estimated / (1024 * 1024),
                BASELINE_MEMORY_BYTES / (1024 * 1024),
                SESSION_MEMORY_BYTES / (1024 * 1024),
                budget / (1024 * 1024),
            ));
        }
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    // ── plan_inference: CPU axis (memory unconstrained) ─────────────────────
    #[test]
    fn inference_plan_stays_within_core_budget() {
        for cores in [1usize, 2, 3, 4, 8, 16, 18, 32, 64] {
            let p = plan_inference(cores, None);
            assert!(p.pool_size >= 1, "pool_size >= 1 for {cores}");
            assert!(p.intra_op_threads >= 1, "intra_op >= 1 for {cores}");
            assert!(
                p.pool_size * p.intra_op_threads <= cores.max(1),
                "budget exceeded for {cores}: {p:?}"
            );
        }
    }

    #[test]
    fn inference_plan_degenerate_cores() {
        // Zero/one core collapses to a single single-threaded session.
        let one = InferencePlan {
            pool_size: 1,
            intra_op_threads: 1,
        };
        assert_eq!(plan_inference(0, None), one);
        assert_eq!(plan_inference(1, None), one);
    }

    #[test]
    fn inference_plan_pool_grows_with_cores() {
        assert!(plan_inference(4, None).pool_size <= plan_inference(8, None).pool_size);
        assert!(plan_inference(8, None).pool_size <= plan_inference(18, None).pool_size);
        assert_eq!(plan_inference(18, None).pool_size, 9);
    }

    // ── plan_inference: memory axis ─────────────────────────────────────────
    #[test]
    fn memory_constrains_pool_below_core_plan() {
        // 18 cores would give pool 9, but 1 GiB fits only 2 sessions:
        // 90% of 1024 MiB - 420 MiB baseline = ~501 MiB / 200 MiB = 2.
        assert_eq!(plan_inference(18, Some(GIB)).pool_size, 2);
        // Plenty of memory: the core plan stands.
        assert_eq!(plan_inference(18, Some(64 * GIB)).pool_size, 9);
    }

    #[test]
    fn memory_floor_is_one_session() {
        // A budget too small for even one session still plans one — the
        // router cannot run with zero; the operator gets a warning instead.
        let p = plan_inference(8, Some(256 * 1024 * 1024));
        assert_eq!(p.pool_size, 1);
    }

    #[test]
    fn max_sessions_for_memory_is_monotonic() {
        let mut last = 0;
        for gib in 1..=16 {
            let n = max_sessions_for_memory(gib * GIB);
            assert!(n >= last, "sessions must not shrink as memory grows");
            last = n;
        }
        // Spot values from the measured model.
        assert_eq!(max_sessions_for_memory(GIB), 2);
        assert_eq!(max_sessions_for_memory(2 * GIB), 7);
    }

    #[test]
    fn unknown_memory_budget_falls_back_to_core_plan() {
        assert_eq!(plan_inference(8, None), plan_inference(8, Some(64 * GIB)));
    }

    // ── overcommit_warnings ─────────────────────────────────────────────────
    #[test]
    fn no_warnings_when_configuration_fits() {
        assert!(overcommit_warnings(2, 2, 8, Some(8 * GIB)).is_empty());
    }

    #[test]
    fn warns_on_cpu_oversubscription() {
        let w = overcommit_warnings(8, 2, 8, Some(64 * GIB));
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("oversubscribes the CPU"), "got: {}", w[0]);
    }

    #[test]
    fn warns_on_memory_overcommit() {
        // 8 sessions ≈ 420 + 1600 MiB > 1 GiB budget.
        let w = overcommit_warnings(8, 1, 8, Some(GIB));
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("exceed available memory"), "got: {}", w[0]);
        assert!(w[0].contains("OOM"), "got: {}", w[0]);
    }

    #[test]
    fn warns_on_both_axes_independently() {
        let w = overcommit_warnings(16, 2, 8, Some(GIB));
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn intra_op_zero_treated_as_one_for_cpu_estimate() {
        // pool 4 with runtime-default threads on 8 cores: 4x1 <= 8, no warning.
        assert!(overcommit_warnings(4, 0, 8, None).is_empty());
        // pool 16 with runtime-default threads on 8 cores: 16x1 > 8, warns.
        assert_eq!(overcommit_warnings(16, 0, 8, None).len(), 1);
    }

    // ── detect_memory_budget ────────────────────────────────────────────────
    #[test]
    fn detects_a_nonzero_budget_on_a_real_host() {
        // Environment-dependent by nature, but every host this test runs on
        // has *some* memory; `None` would mean detection silently broke.
        let budget = detect_memory_budget();
        assert!(budget.is_some_and(|b| b > 0), "got: {budget:?}");
    }
}
