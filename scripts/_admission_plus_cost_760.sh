#!/usr/bin/env bash
# §7.60 — admission unit (§7.59) + cost stages 1+2 (PUSHA_COST_OBJ_ONLY=1,
# PUSHA_COST_SIM_LCS=1). Sequential seeds 0,1,2,3. Stop on first EIO.
#
# Configuration (delta from §7.59):
#   + PUSHA_COST_OBJ_ONLY=1   — Stage 1: object-only Q variant in belief cost
#   + PUSHA_COST_SIM_LCS=1    — Stage 2: replace belief x_seq with LCP rollout
# Everything else identical to §7.59 admission unit. Cost stages now meaningful
# because the planner reasons about contact (always-on lifts J_n_ee≠0).
#
# Gotcha: do NOT set PUSHA_REF_OSC_ALIGN=1 (triggers §7.43 W_track→1 / Kp→200,
# the §7.47 handoff break). Set the four bundle pieces explicitly; W_force=1
# comes from yaml; W_track=100 / Kp_cart=400 stay at yaml defaults.
set -uo pipefail

OUT=${OUT:-admission_plus_cost_760}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
SEEDS=${SEEDS:-"0 1 2 3"}

mkdir -p "$OUT"

preflight_disk() {
  local TMP_PCT
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "ABORT: /tmp at ${TMP_PCT}%" | tee -a "$OUT/launch.log"
    exit 1
  fi
}

run_seed() {
  local SEED="$1"
  local LOG="$OUT/seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] >>> launching seed=$SEED -> $LOG" | tee -a "$OUT/launch.log"
  preflight_disk
  unset LCS_NORMAL_PHI_CLAMP
  unset PUSHA_REF_OSC_ALIGN
  PUSHA_FORCE_ROUTING=u_sol \
  PUSHA_STAGE5_U_HORIZONTAL=10 \
  PUSHA_STAGE5_U_VERTICAL=3 \
  PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
  PUSHA_EE_APPROACH_FACE_TARGET=1 \
  PUSHA_DISABLE_C3_OVERRIDE=1 \
  LCS_ALWAYS_ON_EE_BOX=1 \
  REF_RECONCILE_APPROACH=1 \
  LCS_NORMAL_VELOCITY_LEVEL=0 \
  LCS_NORMAL_COMPLIANCE_K=0.0 \
  PUSHA_COST_OBJ_ONLY=1 \
  PUSHA_COST_SIM_LCS=1 \
  timeout 1800 "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time 6 \
      --name "admission_plus_cost_760_seed${SEED}" \
      --seed "$SEED" \
      --no-record \
      > "$LOG" 2>&1
  local RC=$?
  echo "[$(date +%H:%M:%S)] <<< seed=$SEED rc=$RC" | tee -a "$OUT/launch.log"
  if grep -qE "Input/output error|OSError:\s*\[Errno 5\]|EIO" "$LOG"; then
    echo "[$(date +%H:%M:%S)] HALT: EIO detected in seed=$SEED log; stopping sweep" | tee -a "$OUT/launch.log"
    return 99
  fi
  if [ $RC -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] WARN: seed=$SEED rc=$RC (non-zero, non-EIO) — continuing" | tee -a "$OUT/launch.log"
  fi
  if ! grep -q "Simulation complete" "$LOG"; then
    echo "[$(date +%H:%M:%S)] WARN: seed=$SEED log has no 'Simulation complete' marker" | tee -a "$OUT/launch.log"
  fi
  return 0
}

echo "=== admission_plus_cost_760 sweep start $(date) ===" | tee -a "$OUT/launch.log"
echo "HEAD=$(git rev-parse HEAD)" | tee -a "$OUT/launch.log"

for s in $SEEDS; do
  run_seed "$s"
  rc=$?
  if [ $rc -eq 99 ]; then
    exit 99
  fi
done

echo "=== admission_plus_cost_760 sweep done $(date) ===" | tee -a "$OUT/launch.log"
