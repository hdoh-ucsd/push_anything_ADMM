#!/usr/bin/env bash
# Q7: wrong-face re-engagement guard sweep.
# Plan: docs/superpowers/plans/2026-06-10-wrong-face-reengage-guard.md
#
# Multi-seed (0, 2, 4) at max-time=16s, --no-record, detached, parallelism<=2.
# Parses per-seed: gate_pre / gate_post counts, unique_refused_steps (dedup),
# n_watchdog, n_c3_entries, ping_pong_cycles, box_y_drift.
# SC routing matrix at SUMMARY.txt completion.
set -eo pipefail
OUTBASE=${OUTBASE:-q7_wrong_face_reengage}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTBASE"
git rev-parse HEAD > "$OUTBASE/HEAD.txt"

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*" \
    | tee -a "$OUTBASE/launch.log"
  exit 1
fi

run_seed() {
  local SEED="$1"
  local OUT="$OUTBASE/seed${SEED}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== seed $SEED start $START HEAD=$(cat "$OUTBASE/HEAD.txt") ===" \
    | tee -a "$OUT/run.log"
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time 16 \
      --no-record \
      --name "seed${SEED}_q7_reengage" \
      --seed "$SEED" \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  local N_GATE_PRE
  N_GATE_PRE=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$OUT/run.log" || true)
  local N_GATE_POST
  N_GATE_POST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$OUT/run.log" || true)
  local N_WATCHDOG
  N_WATCHDOG=$(grep -c '^\[GS-watchdog\] ' "$OUT/run.log" || true)
  local RESULT
  RESULT=$(grep '^\[RESULT\]' "$OUT/run.log" | tail -1 || true)
  echo "DONE seed=$SEED ${ELAPSED}s gate_pre=$N_GATE_PRE gate_post=$N_GATE_POST watchdog=$N_WATCHDOG | $RESULT" \
    | tee -a "$OUT/run.log"
}

# Parallelism <= 2: seeds 0 + 2 concurrent, then seed 4 alone.
run_seed 0 &
PID0=$!
run_seed 2 &
PID2=$!
wait "$PID0" "$PID2"
run_seed 4

"$PY" -u scripts/parse_q7_wrong_face_reengage.py "$OUTBASE" \
  | tee "$OUTBASE/SUMMARY.txt"

echo "FINISHED $(date +%s)" | tee -a "$OUTBASE/launch.log"
