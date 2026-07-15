#!/usr/bin/env bash
# =============================================================================
# T-Push Completion Plan — Phase 2 sweep
# Reference gate: goal_dist < 0.02 m AND orient_err < 0.1 rad. Pass ≥ 3/4.
# Seeds {0, 1, 2, 4} sequential + checkpointed + halt-on-EIO.
#
# Substrate discipline (§8):
#   - pre-batch: wsl --shutdown + health-gate (already run externally)
#   - per seed: strictly sequential
#   - per seed: checkpoint to disk on completion
#   - EIO: STOP on first hit, do not push through
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/_phase2_completion_${STAMP}"
CKPT="$OUT_DIR/checkpoints.tsv"
SUMMARY="$OUT_DIR/summary.txt"
mkdir -p "$OUT_DIR"

echo "[PHASE2] out_dir=$OUT_DIR"
echo "[PHASE2] HEAD=$(git rev-parse HEAD)"
echo "[PHASE2] tree_dirty=$(git diff --stat HEAD | tail -1)"

# Checkpoint header
printf "seed\tstatus\tgoal_dist_m\torient_err_rad\tref_gate\tfinal_obj_xy\twall_s\n" > "$CKPT"

SEEDS=(0 1 2 4)

for SEED in "${SEEDS[@]}"; do
  LOG="$OUT_DIR/seed${SEED}.log"
  echo
  echo "[PHASE2] === seed=$SEED start $(date +%H:%M:%S) ==="

  T0=$(date +%s)

  # Canonical T-push invocation (matches _t1c_effect_check.sh / _p2_tripwire.sh).
  PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
  PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
  PUSHA_STAGE5_U_HORIZONTAL=50 \
  PUSHA_STAGE5_U_VERTICAL=3 \
  PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
  LCS_ALWAYS_ON_EE_BOX=1 \
  PUSHA_FORCE_ROUTING=u_sol \
  PUSHA_EE_APPROACH_FACE_TARGET=1 \
  PUSHA_DISABLE_C3_OVERRIDE=1 \
  python main.py push_t \
      --solver c3plus --c3plus-projection lcp \
      --ee-space \
      --sampling-c3 config/sampling_c3_kik_t.yaml \
      --admm-iter 25 \
      --max-time 8 \
      --seed "$SEED" \
      --no-record \
      --math-diag > "$LOG" 2>&1
  RC=$?

  T1=$(date +%s)
  WALL=$((T1 - T0))

  # Inline EIO detection — HALT on hit, do not push through.
  if grep -qE "Input/output error|OSError:\s*\[Errno 5\]|EIO|Bus error|SIGBUS" "$LOG"; then
    echo "[PHASE2] HALT: EIO/SIGBUS detected in seed=$SEED log; stopping sweep (no retry)"
    printf "%d\tEIO_HALT\t\t\t\t\t%d\n" "$SEED" "$WALL" >> "$CKPT"
    echo "[PHASE2] see: $LOG" | tee -a "$SUMMARY"
    exit 2
  fi

  # Extract [RESULT] fields from the log.
  RESULT_LINE="$(grep -E '^\[RESULT\]' "$LOG" | tail -1 || true)"
  if [[ -z "$RESULT_LINE" ]]; then
    echo "[PHASE2] WARN: no [RESULT] line in seed=$SEED log (rc=$RC, wall=${WALL}s)"
    printf "%d\tNO_RESULT_LINE\t\t\t\t\t%d\n" "$SEED" "$WALL" >> "$CKPT"
    continue
  fi

  # Parse fields: goal_dist=X.YYYYm, orient_err=X.YYYYrad, ref_gate=PASS|FAIL,
  # final_obj_xy=(X, Y).
  GD=$(echo "$RESULT_LINE" | grep -oE 'goal_dist=[0-9.]+m' | tr -dc '0-9.')
  OE=$(echo "$RESULT_LINE" | grep -oE 'orient_err=[0-9.]+rad' | tr -dc '0-9.')
  GATE=$(echo "$RESULT_LINE" | grep -oE 'ref_gate=(PASS|FAIL)' | cut -d= -f2)
  XY=$(echo "$RESULT_LINE" | grep -oE 'final_obj_xy=\([^)]+\)' | tr -d ' ')

  printf "%d\tOK\t%s\t%s\t%s\t%s\t%d\n" "$SEED" "$GD" "$OE" "$GATE" "$XY" "$WALL" >> "$CKPT"
  echo "[PHASE2] seed=$SEED  goal_dist=${GD}m  orient_err=${OE}rad  ref_gate=$GATE  wall=${WALL}s  rc=$RC"
done

# ---- Aggregate ----
echo
echo "[PHASE2] === SUMMARY ==="
{
  echo "T-Push Completion Plan Phase 2 — sweep summary"
  echo "out_dir: $OUT_DIR"
  echo "HEAD:    $(git rev-parse HEAD)"
  echo
  cat "$CKPT" | column -t -s $'\t'
  echo
  N_PASS=$(awk -F'\t' 'NR>1 && $5=="PASS" {n++} END {print n+0}' "$CKPT")
  N_TOTAL=$(awk -F'\t' 'NR>1 {n++} END {print n+0}' "$CKPT")
  echo "PASS: $N_PASS / $N_TOTAL  (reference gate: ≥3/4 required)"
  if (( N_PASS >= 3 )); then
    echo "PHASE_2_VERDICT: PASS (T translates on ≥3/4 seeds — phase-close candidate)"
  else
    echo "PHASE_2_VERDICT: FAIL (T does not translate on the aligned port — Phase 3 decision)"
  fi
} | tee "$SUMMARY"
