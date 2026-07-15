#!/usr/bin/env bash
# Per-ablation-commit sweep helper.
# Usage: ./scripts/_conformance_sweep.sh <OUTDIR_NAME>
# Runs seeds 0+2 in parallel, then seed 4 alone (parallelism <= 2).
# Single-retry on wedge (no [RESULT] line). --no-record (video off).
set -uo pipefail
OUTBASE="${1:?usage: $0 OUTBASE}"
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}

# /tmp abort guard FIRST.
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% — STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi

mkdir -p "$OUTBASE"
git rev-parse HEAD > "$OUTBASE/HEAD.txt"
git log -1 --format='%s' > "$OUTBASE/HEAD_subject.txt"

run_seed() {
  local SEED="$1"
  local OUT="$OUTBASE/seed${SEED}"
  mkdir -p "$OUT"
  local TRY=1 RESULT=""
  while [ $TRY -le 2 ]; do
    local START
    START=$(date +%s)
    echo "=== seed $SEED try=$TRY start $START HEAD=$(cat "$OUTBASE/HEAD.txt") ===" \
      >> "$OUT/run.log"
    "$PY" -u main.py pushing \
        --task-id 4 \
        --solver c3plus --c3plus-projection lcp --ee-space \
        --sampling-c3 config/sampling_c3_kik.yaml \
        --admm-iter 25 \
        --max-time 16 \
        --no-record \
        --name "seed${SEED}_conformance" \
        --seed "$SEED" \
        >> "$OUT/run.log" 2>&1 || true
    RESULT=$(grep -E '^\[RESULT\]' "$OUT/run.log" | tail -1)
    if [ -n "$RESULT" ]; then break; fi
    echo "WEDGE detected on seed=$SEED try=$TRY — retrying once" >> "$OUT/run.log"
    TRY=$((TRY+1))
  done
  local ELAPSED=$(( $(date +%s) - START ))
  echo "DONE seed=$SEED ${ELAPSED}s | $RESULT" >> "$OUT/run.log"
}

run_seed 0 &
PID0=$!
run_seed 2 &
PID2=$!
wait "$PID0" "$PID2"
run_seed 4

# Summary parse.
{
  echo "=== $OUTBASE — HEAD=$(cat "$OUTBASE/HEAD.txt") ==="
  echo "    subject: $(cat "$OUTBASE/HEAD_subject.txt")"
  echo ""
  echo "seed  final_obj_xy                 goal_dist  oy_drift_signed   c3_entries  gate_pre  gate_post"
  echo "----  ----------                   ---------  ----------------  ----------  --------  ---------"
  for SEED in 0 2 4; do
    LOG="$OUTBASE/seed${SEED}/run.log"
    [ -f "$LOG" ] || { echo "$SEED  MISSING"; continue; }
    R=$(grep -E '^\[RESULT\]' "$LOG" | tail -1)
    OXY=$(echo "$R" | grep -oE 'final_obj_xy=\([^)]+\)' | tr -d '()' | sed 's/final_obj_xy=//')
    GD=$(echo "$R" | grep -oE 'goal_dist=[-0-9.]+m' | tr -d 'm' | sed 's/goal_dist=//')
    OY=$(echo "$OXY" | awk -F, '{print $2}' | tr -d ' ')
    NCE=$(grep -c 'Switching to C3' "$LOG" 2>/dev/null || echo 0)
    NGP=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$LOG" 2>/dev/null || echo 0)
    NGPOST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$LOG" 2>/dev/null || echo 0)
    printf '%-4s  %-26s  %-9s  %-16s  %-10s  %-8s  %-9s\n' \
      "$SEED" "$OXY" "$GD" "$OY" "$NCE" "$NGP" "$NGPOST"
  done
} | tee "$OUTBASE/SUMMARY.txt"
echo "FINISHED $(date +%s)" >> "$OUTBASE/launch.log"
