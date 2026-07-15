#!/usr/bin/env bash
# Early-window ablation pair: kIK (current) vs kPiecewiseLinear (reference)
# at HEAD 7479985, seed=0, SERIAL, <=1.5s sim, --no-record.
#
# Measure delta INSIDE the bit-identical 1-130 step window confirmed by the
# nondet_seed0_serial_16s_optA_N3 runs (all 3 runs identical obj_xy through
# step 132 -- per-call FP divergence emerges at step 133, visible-scale at
# ~150). The conformance candidate (kIK -> kPiecewiseLinear) bites from
# step 1 (tracker drives EE during free mode), so a single-pair N=1
# comparison yields a clean zero-noise delta in [0, 1.0s] / steps 1-100.
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/early_window_pair_kIK_vs_kPWL
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$OUTBASE/baseline_kIK.yaml" ] || { echo "ABORT: baseline YAML missing"; exit 1; }
[ -r "$OUTBASE/reference_kPWL.yaml" ] || { echo "ABORT: reference YAML missing"; exit 1; }

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

run_one "baseline_kIK" "$OUTBASE/baseline_kIK.yaml"

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT mid-sweep: /tmp at ${TMP_PCT}% before kPWL"
  exit 1
fi

run_one "reference_kPWL" "$OUTBASE/reference_kPWL.yaml"

for L in baseline_kIK reference_kPWL; do
  tail -2 "$OUTBASE/${L}/run.log" 2>/dev/null
done
