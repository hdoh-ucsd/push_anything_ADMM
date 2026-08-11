#!/usr/bin/env bash
# Reference-stack harness: run the dairlib sampling_c3 REFERENCE simulation
# (franka_sim + franka_osc_controller + franka_sampling_c3_controller) without
# procman, mirroring franka_sim_t.pmd script:start_experiment_no_logs
# (sim, osc, wait 50 ms, sampling_c3).
#
# Usage:
#   scripts/reference/run_reference.sh [push_t|jacktoy] [MAX_SEC] [TAG]
#
# Defaults: push_t, 180 s, timestamp tag. Logs + provenance land in
# results/reference/<demo>_<tag>/ (sim.log, osc.log, c3controller.log,
# ref_commit.txt, ref_patches.diff). Ends with a summary of the
# c3controller.log via summarize_reference_log.py.
#
# The reference repo carries local [EXPERIMENT]/[EXPERIMENT-PATCH] edits
# (strict-loader yaml keys, verbose diagnostics) — captured per-run in
# ref_patches.diff.
set -euo pipefail

DEMO="${1:-push_t}"
MAX_SEC="${2:-180}"
TAG="${3:-$(date +%Y%m%d_%H%M%S)}"

REF_ROOT=/root/reference_repos/dairlib_sampling_c3
BIN="$REF_ROOT/bazel-bin/examples/sampling_c3"
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="/root/push_anything_ADMM/results/reference/${DEMO}_${TAG}"

case "$DEMO" in
  push_t|jacktoy) ;;
  *) echo "error: demo must be push_t or jacktoy (got '$DEMO')" >&2; exit 2 ;;
esac

for b in franka_sim franka_osc_controller franka_sampling_c3_controller; do
  if [[ ! -x "$BIN/$b" ]]; then
    echo "error: missing $BIN/$b — build in $REF_ROOT first" >&2
    exit 2
  fi
done

# One reference stack at a time: concurrent runs share the LCM multicast
# group and would cross-talk.
if pgrep -f "$BIN/franka_sim" > /dev/null 2>&1; then
  echo "error: a franka_sim is already running — stop it first (pgrep -af franka_sim)" >&2
  exit 2
fi

# LCM over loopback. The multicast route on lo does not survive WSL restarts.
export LCM_DEFAULT_URL="udpm://239.255.76.67:7667?ttl=0"
ip route show | grep -q "^224.0.0.0/4 dev lo" || ip route add 224.0.0.0/4 dev lo

mkdir -p "$OUT_DIR"
git -C "$REF_ROOT" rev-parse HEAD > "$OUT_DIR/ref_commit.txt"
git -C "$REF_ROOT" diff > "$OUT_DIR/ref_patches.diff"

# Binaries resolve parameter paths relative to the repo root.
cd "$REF_ROOT"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2> /dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2> /dev/null || true; done
}
trap cleanup EXIT INT TERM

"$BIN/franka_sim" --demo_name="$DEMO" > "$OUT_DIR/sim.log" 2>&1 &
PIDS+=($!)
"$BIN/franka_osc_controller" --is_simulation=true --demo_name="$DEMO" \
  > "$OUT_DIR/osc.log" 2>&1 &
PIDS+=($!)
sleep 0.05  # procman waits 50 ms between franka_osc and sampling_c3
"$BIN/franka_sampling_c3_controller" --is_simulation=true --demo_name="$DEMO" \
  > "$OUT_DIR/c3controller.log" 2>&1 &
PIDS+=($!)

echo "[run_reference] demo=$DEMO max_time=${MAX_SEC}s out=$OUT_DIR"
echo "[run_reference] pids: sim=${PIDS[0]} osc=${PIDS[1]} c3=${PIDS[2]}"

EARLY_EXIT=0
for ((t = 0; t < MAX_SEC; t++)); do
  sleep 1
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2> /dev/null; then
      echo "[run_reference] pid $pid exited at t=${t}s — stopping run" >&2
      EARLY_EXIT=1
      break 2
    fi
  done
done

cleanup
trap - EXIT INT TERM

python3 "$HARNESS_DIR/summarize_reference_log.py" "$OUT_DIR" || true

if [[ "$EARLY_EXIT" == 1 ]]; then
  echo "[run_reference] FAILED (early process exit) — logs in $OUT_DIR"
  exit 1
fi
echo "[run_reference] done — logs in $OUT_DIR"
