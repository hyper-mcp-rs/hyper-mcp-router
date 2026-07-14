#!/usr/bin/env bash
# Measure per-inference-session memory empirically.
#
# Starts the router with different --inference-pool-size values, records RSS
# after startup (weights materialized) and again after a burst of max-length
# classification requests (activation arenas grown), then divides the deltas
# by the pool-size difference. Fixed costs (binary, embedded model bytes,
# tokio runtime) cancel out in the subtraction.
#
# The generated config declares TWO text tiers so complexity classification
# actually runs (with <= 1 candidate the router skips inference entirely and
# the sessions would never be touched). Backends point at a dead port —
# classification happens before forwarding, so the resulting 502s are expected
# and harmless.
#
# Usage:
#   scripts/measure_session_memory.sh
#
# Environment overrides:
#   PROFILE=release|debug   cargo profile to build/measure (default: debug)
#   POOLS="1 4"             pool sizes to sample, ascending (default: "1 4")
#   REQUESTS=200            requests per load burst (default: 200)
#   PORT=8199               router port (default: 8199)

set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE=${PROFILE:-debug}
POOLS_STR=${POOLS:-"1 4"}
REQUESTS=${REQUESTS:-200}
PORT=${PORT:-8199}

read -r -a POOLS <<<"$POOLS_STR"
if [ "${#POOLS[@]}" -lt 2 ]; then
  echo "error: POOLS needs at least two ascending values (got: $POOLS_STR)" >&2
  exit 1
fi

BIN="target/$PROFILE/hyper-mcp-router"
if [ ! -x "$BIN" ]; then
  echo "building $PROFILE binary..."
  if [ "$PROFILE" = release ]; then cargo build --release --quiet; else cargo build --quiet; fi
fi

TMP=$(mktemp -d)
ROUTER_PID=""
cleanup() {
  [ -n "$ROUTER_PID" ] && kill "$ROUTER_PID" 2>/dev/null || true
  [ -n "$ROUTER_PID" ] && wait "$ROUTER_PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

# Two text tiers => classification runs. Dead upstream => fast refusal after
# the (already-measured) inference.
cat >"$TMP/config.toml" <<EOF
[server]
host = "127.0.0.1"
port = $PORT
connect_timeout_secs = 1
request_timeout_secs = 5

[[models]]
name = "fast"
base_url = "http://127.0.0.1:1"
type = "fast"
modalities = ["text"]

[[models]]
name = "frontier"
base_url = "http://127.0.0.1:1"
type = "frontier"
modalities = ["text"]
EOF

# A dense prompt well past the 1000-char window budget, so every request
# pushes the tokenizer to the model's 512-token ceiling (worst-case arena).
SENTENCE="Derive and rigorously prove the asymptotic time complexity of red-black tree rebalancing across insertions and deletions with a formal amortized analysis of the potential function. "
PROMPT=""
while [ ${#PROMPT} -lt 1600 ]; do PROMPT="$PROMPT$SENTENCE"; done
printf '{"model":"hyper-mcp-router","messages":[{"role":"user","content":"%s"}]}' "$PROMPT" >"$TMP/body.json"

rss_kb() { ps -o rss= -p "$1" | tr -d ' '; }

wait_healthy() {
  for _ in $(seq 1 100); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  echo "error: router did not become healthy on port $PORT" >&2
  return 1
}

STARTUP_KB=()
LOADED_KB=()

for pool in "${POOLS[@]}"; do
  echo "── pool_size=$pool ──────────────────────────────" >&2
  "$BIN" serve --config "$TMP/config.toml" --log-stdout \
    --inference-pool-size "$pool" --intra-op-threads 2 >/dev/null 2>&1 &
  ROUTER_PID=$!
  wait_healthy
  sleep 1 # let allocations settle
  startup=$(rss_kb "$ROUTER_PID")
  echo "   startup RSS: $((startup / 1024)) MB" >&2

  # Burst with concurrency > pool so every session gets leased and its arena
  # sees a max-length batch. 502s from the dead upstream are expected.
  seq "$REQUESTS" | xargs -P $((pool * 2)) -I{} \
    curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' --data-binary @"$TMP/body.json"
  sleep 1
  loaded=$(rss_kb "$ROUTER_PID")
  echo "   after-load RSS: $((loaded / 1024)) MB" >&2

  kill "$ROUTER_PID"
  wait "$ROUTER_PID" 2>/dev/null || true
  ROUTER_PID=""

  STARTUP_KB+=("$startup")
  LOADED_KB+=("$loaded")
done

echo
echo "profile=$PROFILE requests=$REQUESTS intra_op_threads=2"
printf '%-10s %-16s %-16s\n' "pool" "startup_rss_mb" "after_load_rss_mb"
for i in "${!POOLS[@]}"; do
  printf '%-10s %-16s %-16s\n' "${POOLS[$i]}" \
    "$((STARTUP_KB[i] / 1024))" "$((LOADED_KB[i] / 1024))"
done

echo
last=$((${#POOLS[@]} - 1))
dpool=$((POOLS[last] - POOLS[0]))
per_startup=$(((STARTUP_KB[last] - STARTUP_KB[0]) / dpool / 1024))
per_loaded=$(((LOADED_KB[last] - LOADED_KB[0]) / dpool / 1024))
echo "per-session (startup, weights only):        ~${per_startup} MB"
echo "per-session (after load, weights + arena):  ~${per_loaded} MB"
