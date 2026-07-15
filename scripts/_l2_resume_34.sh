#!/usr/bin/env bash
# Resume Stage 2 L2 sweep for seeds 3 and 4 only.
# Seed 3 wedged at sim t~4.2s on the first attempt (process died ~22:00 PDT);
# seed 4 never launched. Mirrors run_stage2_L2_sweep.sh body and APPENDS to
# the same sweep.summary so the verdict parse sees one continuous file.
# STRICTLY SERIAL — NO Monitor/loop/wakeup. Launch detached and walk away.
set -eo pipefail
OUTDIR=${OUTDIR:-stage2_L2_sweep}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
echo "=== RESUME 3,4 start $(date +%s) ===" | tee -a "$OUTDIR/sweep.summary"
for SEED in 3 4; do
  echo "=== seed=$SEED start $(date +%s) ===" | tee -a "$OUTDIR/sweep.summary"
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "ABORT: /tmp at ${TMP_PCT}% before seed $SEED — STOP, do NOT rm -rf /tmp/claude-0/*" \
      | tee -a "$OUTDIR/sweep.summary"
    exit 1
  fi
  START=$(date +%s)
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus \
      --c3plus-projection lcp \
      --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time 8 \
      --no-record \
      --seed "$SEED" \
      > "$OUTDIR/seed${SEED}_stage2_L2.log" 2>&1 || true
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  LOG="$OUTDIR/seed${SEED}_stage2_L2.log"
  N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$LOG" || true)
  N_GATE_L1=$(grep -c '^\[GATE-ALIGN\]' "$LOG" || true)
  N_GATE_L2=$(grep -c '^\[GATE-COMMIT-FACE\]' "$LOG" || true)
  N_TAG_NM=$(grep -c '^\[GATE-COMMIT-FACE\].* tag=near-miss-cone' "$LOG" || true)
  N_TAG_PP=$(grep -c '^\[GATE-COMMIT-FACE\].* tag=perpendicular' "$LOG" || true)
  N_TAG_AG=$(grep -c '^\[GATE-COMMIT-FACE\].* tag=anti-goal' "$LOG" || true)
  RESULT_LINE=$(grep '^\[RESULT\]' "$LOG" | tail -1 || true)
  echo "seed=$SEED rc=$RC ${ELAPSED}s n_lcs=$N_LCS gate_L1=$N_GATE_L1 gate_L2=$N_GATE_L2 (nm=$N_TAG_NM perp=$N_TAG_PP anti=$N_TAG_AG) | $RESULT_LINE" \
    | tee -a "$OUTDIR/sweep.summary"
done
echo "RESUME 3,4 DONE $(date +%s); see $OUTDIR/sweep.summary" | tee -a "$OUTDIR/sweep.summary"
