#!/usr/bin/env bash
# Early-window ablation pair: w_align=30000.0 (baseline) vs w_align=0.0
# at HEAD 7479985, seed=0, SERIAL, <=1.5s sim, --no-record.
#
# Same protocol as _early_window_pair_kIK_vs_kPWL.sh. Both legs use kIK so
# only the goal-bias term differs. Predicted sample-winner flip at step 1
# (goal-aligned strat_0 bonus tips the K-best selection).
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/early_window_pair_wAlign_ablation
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$OUTBASE/wAlign30k.yaml" ] || { echo "ABORT: wAlign30k YAML missing"; exit 1; }
[ -r "$OUTBASE/wAlign0.yaml" ] || { echo "ABORT: wAlign0 YAML missing"; exit 1; }

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
      --name "early_window_${LABEL}_seed0" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  echo "DONE ${LABEL} ${ELAPSED}s" >> "$OUT/run.log"
}

run_one "wAlign30k" "$OUTBASE/wAlign30k.yaml"

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT mid-sweep: /tmp at ${TMP_PCT}% before wAlign0"
  exit 1
fi

run_one "wAlign0" "$OUTBASE/wAlign0.yaml"

for L in wAlign30k wAlign0; do
  tail -2 "$OUTBASE/${L}/run.log" 2>/dev/null
done
