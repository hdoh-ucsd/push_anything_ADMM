#!/usr/bin/env bash
# Combined-conformance early-window run: w_align=0 + traj_type=kPiecewiseLinear
# at HEAD 7479985, seed=0, SERIAL, <=1.5s sim, --no-record.
#
# This is the headline conformance configuration — the fully-conformed
# early-window behavior with both components ON together.
# Comparisons (post-hoc):
#   - vs baseline_kIK  (current: kIK + w30k)  → aggregate delta
#   - vs reference_kPWL (kPWL + w30k)         → additivity check (~=0 expected
#     since seed=0 w_align ablation was null)
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/early_window_pair_combined_wAlign0_kPWL
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$OUTBASE/combined_wAlign0_kPWL.yaml" ] || { echo "ABORT: combined YAML missing"; exit 1; }

run_one() {
  local LABEL="$1"
  local YAML="$2"
  local OUT="$OUTBASE/${LABEL}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== ${LABEL} start $START seed=0 1.5s serial yaml=$YAML ===" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 1.5 \
      --no-record \
      --name "early_window_${LABEL}" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  echo "DONE ${LABEL} ${ELAPSED}s" >> "$OUT/run.log"
}

run_one "combined_wAlign0_kPWL" "$OUTBASE/combined_wAlign0_kPWL.yaml"

tail -2 "$OUTBASE/combined_wAlign0_kPWL/run.log" 2>/dev/null
