#!/usr/bin/env bash
# §7.31 — Live push sim with the all-at-once reconciliation flag
# (REF_RECONCILE_APPROACH=1) atomic with always-on (LCS_ALWAYS_ON_EE_BOX=1).
# Clamp OFF (LCS_NORMAL_PHI_CLAMP UNSET) for clean attribution — isolate
# whether reconciling the approach path to the reference dissolves the
# +2 mm equilibrium.
set -eo pipefail
OUT=${OUT:-reconcile_approach_live}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/HEAD.txt"
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}%" | tee -a "$OUT/launch.log"
  exit 1
fi
unset LCS_NORMAL_PHI_CLAMP
LCS_ALWAYS_ON_EE_BOX=1 \
REF_RECONCILE_APPROACH=1 \
LCS_NORMAL_VELOCITY_LEVEL=0 \
LCS_NORMAL_COMPLIANCE_K=0.0 \
timeout 1800 "$PY" -u main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 6 \
    --name reconcile_approach_live \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
