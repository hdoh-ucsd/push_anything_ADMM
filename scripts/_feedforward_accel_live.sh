#!/usr/bin/env bash
# §7.34 — Live push sim with FAITHFUL-DESIRED-STATE FEEDFORWARD-ACCEL:
# REF_RECONCILE_APPROACH=1 (atomic with always-on). Adds the yddot_des
# feedforward leg to the OSC PD law so the port matches the reference's
# `yddot_command = yddot_des + Kp·error_y + Kd·error_ydot`. KEEPS all of
# c893af3 (position+velocity desired-state, proxy-off, position-OSC,
# always-on). NO Kp change (400 HELD as deferred hyperparameter per §7.33).
# Clamp OFF for clean attribution.
set -eo pipefail
OUT=${OUT:-feedforward_accel_live}
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
    --name feedforward_accel_live \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
