#!/usr/bin/env bash
# §7.27 — live flip of LCS_NORMAL_PHI_CLAMP=0.034 (the v_cap that
# Candidate E validated E-PASSES at, anchors 3/3 + held-out IN-BAND).
# Runs the full push sim ONCE on the ee-space path (where the clamp
# lives — R^7 path was NOT mirrored, so the clamp would be inert in
# the R^7 default).
#
# This is a record of the exact command used for the c5314ae →
# §7.27-live-result commit.
set -eo pipefail
OUT=${OUT:-clamp_live_v034}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/HEAD.txt"
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}%" | tee -a "$OUT/launch.log"
  exit 1
fi
LCS_NORMAL_PHI_CLAMP=0.034 \
LCS_NORMAL_VELOCITY_LEVEL=0 \
LCS_NORMAL_COMPLIANCE_K=0.0 \
timeout 900 "$PY" -u main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 6 \
    --name clamp_live_v034 \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
