#!/usr/bin/env bash
# Early-window w_align ablation pair: w_align=30000.0 (baseline) vs w_align=0.0
# at HEAD 7479985, seed=1, SERIAL, <=1.5s sim, --no-record.
#
# Companion to _early_window_pair_wAlign_ablation.sh (seed=0). The seed=0 leg
# found a clean NULL — sample-winner identical at every tick. This seed=1
# leg probes whether the null is global (goal-bias inert) or per-seed
# (goal-bias conditionally active under different sample geometries).
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/early_window_pair_wAlign_ablation_seed1
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$OUTBASE/wAlign30k_seed1.yaml" ] || { echo "ABORT: wAlign30k_seed1 YAML missing"; exit 1; }
[ -r "$OUTBASE/wAlign0_seed1.yaml" ] || { echo "ABORT: wAlign0_seed1 YAML missing"; exit 1; }

run_one() {
  local LABEL="$1"
  local YAML="$2"
  local OUT="$OUTBASE/${LABEL}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== ${LABEL} start $START seed=1 1.5s serial yaml=$YAML ===" \
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
      --seed 1 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  echo "DONE ${LABEL} ${ELAPSED}s" >> "$OUT/run.log"
}

run_one "wAlign30k_seed1" "$OUTBASE/wAlign30k_seed1.yaml"

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT mid-sweep: /tmp at ${TMP_PCT}% before wAlign0_seed1"
  exit 1
fi

run_one "wAlign0_seed1" "$OUTBASE/wAlign0_seed1.yaml"

for L in wAlign30k_seed1 wAlign0_seed1; do
  tail -2 "$OUTBASE/${L}/run.log" 2>/dev/null
done
