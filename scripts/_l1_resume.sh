#!/usr/bin/env bash
set -eo pipefail
cd /root/push_anything_ADMM
OUTDIR=stage2_L1_sweep
PY=/root/miniconda3/envs/push_anything_ADMM/bin/python
: > "$OUTDIR/sweep.resume.summary"
for SEED in 1 2 3 4; do
  echo "=== seed=$SEED start $(date +%s) ===" >> "$OUTDIR/sweep.resume.summary"
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
      > "$OUTDIR/seed${SEED}_stage2_L1.log" 2>&1 || true
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  N_LCS=$(grep -c "\[CONTACT-RUN\] step=.* contact_type=EE-BOX" "$OUTDIR/seed${SEED}_stage2_L1.log" || true)
  N_AEE1=$(grep -c "A_is_ee=1" "$OUTDIR/seed${SEED}_stage2_L1.log" || true)
  N_GATE=$(grep -c "\[GATE-ALIGN\]" "$OUTDIR/seed${SEED}_stage2_L1.log" || true)
  RESULT_LINE=$(grep "^\[RESULT\]" "$OUTDIR/seed${SEED}_stage2_L1.log" | tail -1 || true)
  echo "seed=$SEED rc=$RC ${ELAPSED}s n_lcs=$N_LCS n_aee1=$N_AEE1 n_gate=$N_GATE | $RESULT_LINE" \
    >> "$OUTDIR/sweep.resume.summary"
done
echo "DONE $(date +%s)" >> "$OUTDIR/sweep.resume.summary"
touch "$OUTDIR/.resume_done"
