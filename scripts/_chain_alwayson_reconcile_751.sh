#!/usr/bin/env bash
# §7.51 chain + always-on + reconcile sweep, sequential, stop-on-EIO.
#
#   chain   : PUSHA_REF_OSC_ALIGN + PUSHA_EE_APPROACH_FACE_TARGET +
#             PUSHA_DISABLE_C3_OVERRIDE          (the §7.51 chain)
#   always-on: LCS_ALWAYS_ON_EE_BOX=1            (§7.30)
#   reconcile: REF_RECONCILE_APPROACH=1          (§7.31, faithful-desired-state)
#   cost stages OFF: PUSHA_COST_OBJ_ONLY / PUSHA_COST_SIM_LCS unset
#   feedforward-accel OFF (default, §7.35 sub-gate)
#
# Seeds 0,1,2,3 in order; halts the whole sweep on first EIO/IOError.
# Never runs seeds in parallel (the EIO trigger).
set -uo pipefail

OUT=${OUT:-chain_alwayson_reconcile_751}
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
  # The §7.51 chain — bundle pieces set EXPLICITLY (NOT via PUSHA_REF_OSC_ALIGN,
  # which would trigger the §7.43 OSC override W_track→1 / Kp→200 — that's
  # §7.47's known IK→c3 handoff break. We keep W_track=100, Kp=400 from
  # config/osc_franka.yaml; W_force=1.0 comes from config/sampling_c3_kik.yaml.
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
  timeout 1800 "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time 6 \
      --name "chain_751_alwayson_recon_seed${SEED}" \
      --seed "$SEED" \
      --no-record \
      > "$LOG" 2>&1
  local RC=$?
  echo "[$(date +%H:%M:%S)] <<< seed=$SEED rc=$RC" | tee -a "$OUT/launch.log"
  # EIO detection — look in the seed log for either errno-5 form.
  if grep -qE "Input/output error|OSError:\s*\[Errno 5\]|EIO" "$LOG"; then
    echo "[$(date +%H:%M:%S)] HALT: EIO detected in seed=$SEED log; stopping sweep (no retry)" | tee -a "$OUT/launch.log"
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

echo "=== chain_alwayson_reconcile_751 sweep start $(date) ===" | tee -a "$OUT/launch.log"
echo "HEAD=$(git rev-parse HEAD)" | tee -a "$OUT/launch.log"

for s in $SEEDS; do
  run_seed "$s"
  rc=$?
  if [ $rc -eq 99 ]; then
    exit 99
  fi
done

echo "=== chain_alwayson_reconcile_751 sweep done $(date) ===" | tee -a "$OUT/launch.log"
