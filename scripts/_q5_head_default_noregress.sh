#!/usr/bin/env bash
# Task 5 — HEAD-default noregress, seeds 0/2/4.
# Carries: SC-noregress-working + grounded-SC-legit-approach-works (the SC's
# real home — F=5 only ever had LCS-phantom pushes per ee_outward_attribution)
# + 0/2/4-seed-stability (gate barely fires here per addition-1).
# Compare vs q1_noregress/baseline.txt: goal_dist_t6 ±10%, first_c3 ±20 ticks,
# trajectory_xy ±15mm. NOT n_lcs (feedback_nlcs_not_noregress).
# Parallelism <= 2: seed0+seed2 concurrent, then seed4. Video off.
set -eo pipefail
OUTBASE=${OUTBASE:-q5_head_default_noregress}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTBASE"
git rev-parse HEAD > "$OUTBASE/HEAD.txt"

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% — STOP, do NOT rm -rf /tmp/claude-0/*" \
    | tee -a "$OUTBASE/launch.log"
  exit 1
fi

run_seed() {
  local SEED="$1"
  local OUT="$OUTBASE/seed${SEED}"
  mkdir -p "$OUT"
  local START=$(date +%s)
  echo "=== seed $SEED start $START ===" >> "$OUT/run.log"
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time 6 \
      --name "seed${SEED}_q5_head" \
      --seed "$SEED" \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  # Extract per-seed metrics
  local N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$OUT/run.log" || true)
  local FIRST_C3_STEP=$(grep -m1 -oE '^\[GS\] step=[0-9]+ mode=c3' "$OUT/run.log" | grep -oE '[0-9]+' | head -1 || echo "-1")
  local GOAL_DIST_T6=$(grep -m1 -oE '^\[STEP\] step=600 .* goal_dist=[0-9.]+' "$OUT/run.log" | grep -oE 'goal_dist=[0-9.]+' | grep -oE '[0-9.]+' || echo "NaN")
  local GATE_CAP_FIRED=$(grep -c 'gate_cap=1' "$OUT/run.log" || true)
  local ADMIT_GUARD_TICKS=$(grep -c '\[ADMIT-GUARD\] step=' "$OUT/run.log" || true)
  echo "DONE seed=$SEED ${ELAPSED}s n_lcs=$N_LCS first_c3=$FIRST_C3_STEP goal_dist_t6=$GOAL_DIST_T6 gate_cap1_count=$GATE_CAP_FIRED admit_guard_ticks=$ADMIT_GUARD_TICKS" \
    | tee -a "$OUT/run.log" | tee -a "$OUTBASE/summary.txt"
}

> "$OUTBASE/summary.txt"

# Echo baselines for reference (q1_noregress, HEAD=fa8db2b — pre jitter, pre gate)
{
  echo "BASELINE_SOURCE=q1_noregress (HEAD=fa8db2b, pre-jitter, pre-gate)"
  echo "BASELINE seed=0 first_c3=109 goal_dist_t6=0.263"
  echo "BASELINE seed=2 first_c3=164 goal_dist_t6=0.179"
  echo "BASELINE seed=4 first_c3=159 goal_dist_t6=0.247"
  echo "HEAD_UNDER_TEST=$(cat "$OUTBASE/HEAD.txt") (jitter=0 + Q2c gate ee_z<0.090)"
} | tee -a "$OUTBASE/summary.txt"

# Batch 1: seed0 + seed2 in parallel
run_seed 0 &
PID0=$!
run_seed 2 &
PID2=$!
wait "$PID0" || true
wait "$PID2" || true

# Batch 2: seed4 alone
run_seed 4

echo "FINISHED $(date +%s)" | tee -a "$OUTBASE/summary.txt"
