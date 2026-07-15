#!/usr/bin/env bash
# Resume of nondet_seed0_518bcfa: runs 3..6 only (1,2 already complete).
# Same protocol as the original. Aggregates all 6 at the end.
set -uo pipefail
OUTBASE=/root/push_anything_ADMM/nondet_seed0_518bcfa
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
YAML=/tmp/sampling_c3_kik_518bcfa.yaml

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$YAML" ] || { echo "ABORT: YAML missing at $YAML"; exit 1; }

# Move EIO-truncated logs aside so they can be inspected if needed.
for I in 3 4; do
  L="$OUTBASE/run${I}/run.log"
  if [ -f "$L" ]; then
    mv "$L" "$OUTBASE/run${I}/run.log.eio_truncated"
  fi
done

run_one() {
  local I="$1"
  local OUT="$OUTBASE/run${I}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== run $I start $START effective_HEAD=518bcfa seed=0 (resume) ===" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 16 \
      --no-record \
      --name "nondet_run${I}_seed0" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  local N_GATE_PRE N_GATE_POST N_DRAKE_PAIRS RESULT
  N_GATE_PRE=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$OUT/run.log" 2>/dev/null || true)
  N_GATE_POST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$OUT/run.log" 2>/dev/null || true)
  N_DRAKE_PAIRS=$(grep -c '^\[DRAKE-CONTACT\] ' "$OUT/run.log" 2>/dev/null || true)
  RESULT=$(grep '^\[RESULT\]' "$OUT/run.log" | tail -1 || true)
  echo "DONE run=$I ${ELAPSED}s gate_pre=$N_GATE_PRE gate_post=$N_GATE_POST drake_pairs_lines=$N_DRAKE_PAIRS | $RESULT" \
    >> "$OUT/run.log"
}

# Batch 2: runs 3+4 (parallelism 2).
run_one 3 & P3=$!
run_one 4 & P4=$!
wait "$P3" "$P4"

# Batch 3: runs 5+6 (parallelism 2).
run_one 5 & P5=$!
run_one 6 & P6=$!
wait "$P5" "$P6"

# Final aggregate over all 6 runs.
{
  echo "=== nondet_seed0_518bcfa N=6 (final, post-resume) ==="
  echo "main_tree_HEAD=$(cat $OUTBASE/main_tree_HEAD.txt 2>/dev/null)"
  echo "effective_HEAD=518bcfa (YAML: $YAML)"
  echo ""
  echo "run  final_obj_xy                 goal_dist  oy_drift_signed   gate_pre  gate_post  drake_pairs_emits"
  echo "---  ----------                   ---------  ----------------  --------  ---------  -----------------"
  for I in 1 2 3 4 5 6; do
    LOG="$OUTBASE/run${I}/run.log"
    [ -f "$LOG" ] || { echo "$I  MISSING"; continue; }
    R=$(grep -E '^\[RESULT\]' "$LOG" | tail -1)
    if [ -z "$R" ]; then
      printf '%-3s  TRUNCATED (no [RESULT])\n' "$I"
      continue
    fi
    OXY=$(echo "$R" | grep -oE 'final_obj_xy=\([^)]+\)' | tr -d '()' | sed 's/final_obj_xy=//')
    GD=$(echo "$R" | grep -oE 'goal_dist=[-0-9.]+m' | tr -d 'm' | sed 's/goal_dist=//')
    OY=$(echo "$OXY" | awk -F, '{print $2}' | tr -d ' ')
    NGP=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$LOG" 2>/dev/null || echo 0)
    NGPOST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$LOG" 2>/dev/null || echo 0)
    NDR=$(grep -c '^\[DRAKE-CONTACT\] ' "$LOG" 2>/dev/null || echo 0)
    printf '%-3s  %-26s  %-9s  %-16s  %-8s  %-9s  %-17s\n' \
      "$I" "$OXY" "$GD" "$OY" "$NGP" "$NGPOST" "$NDR"
  done
} | tee "$OUTBASE/SUMMARY.txt"
echo "RESUME FINISHED $(date +%s)" >> "$OUTBASE/launch.log"
