#!/usr/bin/env bash
# N=2 SERIAL 16s pair at HEAD 7479985 + working-tree Ipopt determinism pins.
#
# Purpose: close out possibility 2 (serial bit-identity confirmed through
# step 251 but not yet at t=16s). Bar: BYTE-IDENTICAL at t=16s — goal_dist
# all decimals + full [STEP] line series match between the paired runs.
#
# Protocol matches scripts/_nondet_seed0_518bcfa.sh in everything but:
#   --max-time stays 16.0s
#   parallelism = 1 (SERIAL — the load-bearing variable)
#   N = 2 paired runs (minimum to verify identity; not a statistic)
#
# Each run writes run.log; the reducer (manual diff at end) verifies
# byte-identity on [STEP] lines and [RESULT] line.
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/nondet_seed0_serial_16s_pair
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
YAML=/tmp/sampling_c3_kik_518bcfa.yaml

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$YAML" ] || { echo "ABORT: YAML missing at $YAML"; exit 1; }

mkdir -p "$OUTBASE"
cp "$YAML" "$OUTBASE/effective_config.yaml"
git -C /root/push_anything_ADMM rev-parse HEAD > "$OUTBASE/main_tree_HEAD.txt"
git -C /root/push_anything_ADMM diff HEAD > "$OUTBASE/working_tree_diff.patch"

run_one() {
  local I="$1"
  local OUT="$OUTBASE/run${I}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== run $I start $START seed=0 16s serial parallelism=1 ===" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 16.0 \
      --no-record \
      --name "serial_16s_run${I}_seed0" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  echo "DONE run=$I ${ELAPSED}s" >> "$OUT/run.log"
}

for I in 1 2; do
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "ABORT mid-sweep: /tmp at ${TMP_PCT}% before run ${I}"
    exit 1
  fi
  run_one "$I"
done

for I in 1 2; do
  tail -1 "$OUTBASE/run${I}/run.log" 2>/dev/null
done
