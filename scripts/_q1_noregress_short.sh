#!/usr/bin/env bash
# Q1 SC-Q1-noregress-working: seeds 0, 2, 4 short runs (6s each).
# CRITICAL: --save-video OFF. Sequential (parallelism <=2 not needed for 3 seeds).
# Plan: docs/superpowers/plans/2026-06-03-q1-dispatcher-reentry-guard.md
set -eo pipefail
OUTBASE=${OUTBASE:-q1_noregress}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTBASE"
git rev-parse HEAD > "$OUTBASE/HEAD.txt"
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% — STOP, do NOT rm -rf /tmp/claude-0/*" \
    | tee -a "$OUTBASE/launch.log"
  exit 1
fi
> "$OUTBASE/summary.txt"
for SEED in 0 2 4; do
  OUT="$OUTBASE/seed${SEED}"
  mkdir -p "$OUT"
  echo "=== seed $SEED start $(date +%s) ===" | tee -a "$OUT/run.log"
  START=$(date +%s)
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time 6 \
      --name "seed${SEED}_noregress" \
      --seed "$SEED" \
      >> "$OUT/run.log" 2>&1 || true
  ELAPSED=$(( $(date +%s) - START ))
  N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$OUT/run.log" || true)
  FIRST_C3_STEP=$(grep -m1 -oE '^\[GS\] step=[0-9]+ mode=c3' "$OUT/run.log" | grep -oE '[0-9]+' || echo "-1")
  GOAL_DIST_T6=$(grep -m1 -oE '^\[STEP\] step=600 .* goal_dist=[0-9.]+' "$OUT/run.log" | grep -oE 'goal_dist=[0-9.]+' | grep -oE '[0-9.]+' || echo "NaN")
  echo "DONE seed=$SEED ${ELAPSED}s n_lcs=$N_LCS first_c3=$FIRST_C3_STEP goal_dist_t6=$GOAL_DIST_T6" \
    | tee -a "$OUT/run.log" | tee -a "$OUTBASE/summary.txt"
done
echo "FINISHED $(date +%s)" | tee -a "$OUTBASE/summary.txt"
