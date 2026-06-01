#!/usr/bin/env bash
# Stage 1 sweep for the wrong-face race-fix (2026-06-01 plan).
# Runs the descent-gate alone on 5 seeds, no Stage 2 changes. Read against
# pre-registered SC' / SC-goal / SC-noregress / deadlock criterion before
# any Stage 2 work.
#
# Usage:
#   ./scripts/run_altitude_hold_sweep.sh                   # default OUTDIR
#   OUTDIR=altitude_hold_sweep_v2 ./scripts/run_altitude_hold_sweep.sh
set -eo pipefail
OUTDIR=${OUTDIR:-altitude_hold_sweep}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTDIR"
git rev-parse HEAD > "$OUTDIR/HEAD.txt"
: > "$OUTDIR/sweep.summary"
for SEED in 0 1 2 3 4; do
  echo "=== seed=$SEED ==="
  START=$(date +%s)
  "$PY" -u main.py pushing \
      --seed "$SEED" --max-time 8 \
      --admm-iter 25 --use-osc \
      --no-record \
      --sampling-c3 config/sampling_c3_kik.yaml \
      > "$OUTDIR/seed${SEED}_altitude_hold.log" 2>&1 || true
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$OUTDIR/seed${SEED}_altitude_hold.log" || true)
  N_AEE1=$(grep -c 'A_is_ee=1' "$OUTDIR/seed${SEED}_altitude_hold.log" || true)
  N_TGTCHG=$(grep -c '\[TGT-CHANGE\]' "$OUTDIR/seed${SEED}_altitude_hold.log" || true)
  RESULT_LINE=$(grep '^\[RESULT\]' "$OUTDIR/seed${SEED}_altitude_hold.log" | tail -1 || true)
  echo "seed=$SEED rc=$RC ${ELAPSED}s n_lcs=$N_LCS n_aee1=$N_AEE1 n_tgtchg=$N_TGTCHG | $RESULT_LINE" \
    | tee -a "$OUTDIR/sweep.summary"
done
echo "DONE; see $OUTDIR/sweep.summary"
