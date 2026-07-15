#!/usr/bin/env bash
# Polls for seed4 baseline DONE, writes baseline SUMMARY, then chains 3 commits.
set -uo pipefail
LOG=/root/push_anything_ADMM/conformance_orchestrator.log
exec >> "$LOG" 2>&1

BASELINE_LOG=/root/push_anything_ADMM/conformance_baseline/seed4/run.log

echo "[$(date)] orchestrator starting, polling for seed4 baseline DONE"
# Poll every 60s for [RESULT] line (sim finished) OR exit= marker.
while :; do
  if grep -qE '^\[RESULT\]' "$BASELINE_LOG" 2>/dev/null; then
    echo "[$(date)] seed4 [RESULT] detected"
    break
  fi
  if grep -qE '^exit=' "$BASELINE_LOG" 2>/dev/null; then
    # Process exited (maybe crashed).
    if grep -qE '^\[RESULT\]' "$BASELINE_LOG"; then
      echo "[$(date)] seed4 finished cleanly"
    else
      echo "[$(date)] seed4 process exited WITHOUT [RESULT] — aborting orchestrator"
      exit 1
    fi
    break
  fi
  sleep 60
done

# Append DONE marker so the baseline summary tool sees it.
LAST_RESULT=$(grep -E '^\[RESULT\]' "$BASELINE_LOG" | tail -1)
if ! grep -q '^DONE seed=4' "$BASELINE_LOG"; then
  echo "DONE seed=4 (rerun) | $LAST_RESULT" >> "$BASELINE_LOG"
fi

# Write baseline SUMMARY.txt.
echo "[$(date)] writing baseline SUMMARY.txt"
OUTBASE=/root/push_anything_ADMM/conformance_baseline
{
  echo "=== conformance_baseline — HEAD=$(cat "$OUTBASE/HEAD.txt") ==="
  echo "    subject: $(cat "$OUTBASE/HEAD_subject.txt")"
  echo ""
  echo "seed  final_obj_xy                 goal_dist  oy_drift_signed   c3_entries  gate_pre  gate_post"
  echo "----  ----------                   ---------  ----------------  ----------  --------  ---------"
  for SEED in 0 2 4; do
    LOGF="$OUTBASE/seed${SEED}/run.log"
    [ -f "$LOGF" ] || { echo "$SEED  MISSING"; continue; }
    R=$(grep -E '^\[RESULT\]' "$LOGF" | tail -1)
    OXY=$(echo "$R" | grep -oE 'final_obj_xy=\([^)]+\)' | tr -d '()' | sed 's/final_obj_xy=//')
    GD=$(echo "$R" | grep -oE 'goal_dist=[-0-9.]+m' | tr -d 'm' | sed 's/goal_dist=//')
    OY=$(echo "$OXY" | awk -F, '{print $2}' | tr -d ' ')
    NCE=$(grep -c 'Switching to C3' "$LOGF" 2>/dev/null || echo 0)
    NGP=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$LOGF" 2>/dev/null || echo 0)
    NGPOST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$LOGF" 2>/dev/null || echo 0)
    printf '%-4s  %-26s  %-9s  %-16s  %-10s  %-8s  %-9s\n' \
      "$SEED" "$OXY" "$GD" "$OY" "$NCE" "$NGP" "$NGPOST"
  done
} | tee "$OUTBASE/SUMMARY.txt"

echo "[$(date)] baseline done, launching commit chain"
bash /root/push_anything_ADMM/scripts/_conformance_chain.sh
echo "[$(date)] orchestrator done"
