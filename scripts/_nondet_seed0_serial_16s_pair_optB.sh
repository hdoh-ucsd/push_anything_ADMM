#!/usr/bin/env bash
# N=2 SERIAL 16s VERIFY pair for Option B (retry-on-IK-fail with fixed eps).
#
# Purpose: verify that the IK retry at reposition_ik.py:1100-1130 with FIXED
# eps=1e-4 produces BYTE-IDENTICAL trajectories across paired seed=0 serial
# runs through t=16s. Bar: every [STEP] line matches between run1/run2,
# final goal_dist agrees to all decimals.
#
# Protocol matches scripts/_nondet_seed0_serial_16s_pair.sh — only the
# working-tree code differs (now includes Option B retry).
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/nondet_seed0_serial_16s_pair_optB
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
  echo "=== run $I start $START seed=0 16s serial OptB(retry-fixed-eps) ===" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 16.0 \
      --no-record \
      --name "serial_16s_optB_run${I}_seed0" \
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
