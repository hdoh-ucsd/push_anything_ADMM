#!/usr/bin/env bash
# Re-run baseline for seeds 2/3/4 at pre-change HEAD (ae071d1).
# Seed 0 already clean from the original baseline launch (started before
# the Q2c gate code was written into the working tree).
# Working-tree files restored via:
#   git restore --source=ae071d1 -- control/sampling_c3/reposition_ik.py \
#                                    control/sampling_c3/wrapper.py
set -euo pipefail
OUT=audit_output/admit_guard_gate/baseline
mkdir -p "$OUT"

for SEED in 2 3 4; do
  LOG="$OUT/seed${SEED}.log"
  echo "[baseline-rerun] seed=$SEED → $LOG"
  timeout 360 python -u main.py pushing \
    --task-id 4 \
    --max-time 8.0 \
    --admm-iter 3 \
    --solver c3plus \
    --ee-space \
    --seed "$SEED" \
    --no-record \
    --sampling-c3 config/sampling_c3_kik.yaml \
    > "$LOG" 2>&1 || echo "[baseline-rerun] seed=$SEED exit=$?"
done
echo "[baseline-rerun] complete"
