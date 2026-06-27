#!/usr/bin/env bash
# §7.30 — Live push sim with LCS_ALWAYS_ON_EE_BOX=1 (always-on EE-BOX
# admission, mirrors the reference's pre-specified contact_geoms / no-
# distance-threshold LCS construction). Clamp OFF (LCS_NORMAL_PHI_CLAMP
# UNSET) for clean attribution — isolate whether admission visibility
# alone unblocks contact establishment.
set -eo pipefail
OUT=${OUT:-always_on_live}
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
LCS_NORMAL_VELOCITY_LEVEL=0 \
LCS_NORMAL_COMPLIANCE_K=0.0 \
timeout 1800 "$PY" -u main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 6 \
    --name always_on_live \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
