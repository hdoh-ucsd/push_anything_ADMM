#!/usr/bin/env bash
# Baseline at pre-change HEAD for Q2c admit-guard EE_z gate.
# Protocol matches scripts/_q2_force_sweep_seed3.sh:51 (the run that documented
# the F=5 bump): --admm-iter 3 --ee-space, HEAD-default push forces (nominal=5.0,
# min=2.0 — D2 cap means effective force = nominal = 5.0 per
# project_d2_cap_neutralizes_min_push.md memory).
set -euo pipefail
OUT=audit_output/admit_guard_gate/baseline
mkdir -p "$OUT"

for SEED in 0 2 3 4; do
  LOG="$OUT/seed${SEED}.log"
  echo "[baseline] seed=$SEED → $LOG"
  timeout 360 python -u main.py pushing \
    --task-id 4 \
    --max-time 8.0 \
    --admm-iter 3 \
    --solver c3plus \
    --ee-space \
    --seed "$SEED" \
    --no-record \
    --sampling-c3 config/sampling_c3_kik.yaml \
    > "$LOG" 2>&1 || echo "[baseline] seed=$SEED exit=$?"
done
echo "[baseline] all four seeds complete"
