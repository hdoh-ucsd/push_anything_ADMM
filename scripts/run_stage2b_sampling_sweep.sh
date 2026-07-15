#!/usr/bin/env bash
# Stage 2B sampler-bias sweep — pre-registered 5-seed sweep at face_bias_strength=2.0
# Read against SC-modeB-convert (seeds 1,2 -> +x face/-x motion),
# SC-noregress-working (seeds 0,3,4 still +x face/-x motion at Stage-1 levels),
# SC-noregress-lambda (lcp_res_max stays ~1e-7 — instrumented by d316462).
# Goal-overshoot on converted seeds 1/2 is EXPECTED (goal-stop's job next stage).
set -eo pipefail
OUTDIR=${OUTDIR:-stage2b_sampling_sweep}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTDIR"
git rev-parse HEAD > "$OUTDIR/HEAD.txt"
: > "$OUTDIR/sweep.summary"
for SEED in 0 1 2 3 4; do
  echo "=== seed=$SEED ==="
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
      > "$OUTDIR/seed${SEED}_stage2b_sampling.log" 2>&1 || true
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$OUTDIR/seed${SEED}_stage2b_sampling.log" || true)
  N_AEE1=$(grep -c 'A_is_ee=1' "$OUTDIR/seed${SEED}_stage2b_sampling.log" || true)
  N_TGTCHG=$(grep -c '\[TGT-CHANGE\]' "$OUTDIR/seed${SEED}_stage2b_sampling.log" || true)
  RESULT_LINE=$(grep '^\[RESULT\]' "$OUTDIR/seed${SEED}_stage2b_sampling.log" | tail -1 || true)
  echo "seed=$SEED rc=$RC ${ELAPSED}s n_lcs=$N_LCS n_aee1=$N_AEE1 n_tgtchg=$N_TGTCHG | $RESULT_LINE" \
    | tee -a "$OUTDIR/sweep.summary"
done
echo "DONE; see $OUTDIR/sweep.summary"
