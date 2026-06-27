#!/usr/bin/env bash
# §7.32 — Live push sim with FAITHFUL-DESIRED-STATE: REF_RECONCILE_APPROACH=1
# (atomic with always-on). The static surface-point override (§7.31 'a') is
# DROPPED; _p_ee_des stays as the planner's first-knot prediction, _v_ee_des
# carries the planner's predicted EE velocity (alpha=1.0, v_max-clipped) —
# matching the reference OSC's y_des = traj.value(t) + ydot_des = traj.Eval-
# Derivative(t,1). Clamp OFF for clean attribution.
set -eo pipefail
OUT=${OUT:-faithful_desired_state_live}
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
    --name faithful_desired_state_live \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
