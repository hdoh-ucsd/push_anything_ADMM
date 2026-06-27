#!/usr/bin/env bash
# §7.35 — Live push sim with VALIDATED CLAMP RE-ENABLED on the §7.33 working
# config: LCS_NORMAL_PHI_CLAMP=0.034 (§7.27 E-PASSES validated value, NOT
# re-fit) + LCS_ALWAYS_ON_EE_BOX=1 + REF_RECONCILE_APPROACH=1. Gated test
# run only — clamp default stays OFF. NO new build (the clamp has been
# built + validated; this enables it on the working state).
set -eo pipefail
OUT=${OUT:-clamp_reenable_live}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/HEAD.txt"
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}%" | tee -a "$OUT/launch.log"
  exit 1
fi
LCS_NORMAL_PHI_CLAMP=0.034 \
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
    --name clamp_reenable_live \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
