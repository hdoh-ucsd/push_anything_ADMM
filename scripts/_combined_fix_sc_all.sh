#!/usr/bin/env bash
# Combined-fix SCs — Tasks 6+7 sequentially.
# Phase A (SC-runaway-stopped): seeds 0+4, 16s, video off, parallel (2 procs).
# Phase B (SC-noregress-working): seeds 0+2 parallel, then seed 4 alone (≤2 procs).
# Total wall: ~30 min Phase A + ~25 min Phase B ≈ 55-60 min.
set -eo pipefail

OUTBASE=${OUTBASE:-q6_combined_fix}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTBASE/sc_runaway" "$OUTBASE/sc_noregress"

git rev-parse HEAD > "$OUTBASE/HEAD.txt"
git diff > "$OUTBASE/WORKING_DIFF.patch"

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% — do NOT rm -rf /tmp/claude-0/*" \
    | tee -a "$OUTBASE/launch.log"
  exit 1
fi

echo "=== START $(date +%s) HEAD=$(git rev-parse HEAD) ===" | tee "$OUTBASE/launch.log"

run_seed() {
  local SEED="$1"; local MAXT="$2"; local TAG="$3"; local OUT="$4"
  mkdir -p "$OUT"
  date +%s > "$OUT/START_EPOCH"
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time "$MAXT" \
      --no-record \
      --name "seed${SEED}_${TAG}" \
      --seed "$SEED" \
      >> "$OUT/run.log" 2>&1 || true
  date +%s > "$OUT/END_EPOCH"
}

# Phase A — SC-runaway-stopped (seeds 0+4 parallel, 16s)
echo "=== PHASE A SC-runaway-stopped $(date +%s) ===" | tee -a "$OUTBASE/launch.log"
run_seed 0 16 combined_runaway "$OUTBASE/sc_runaway/seed0" &
PID0=$!
run_seed 4 16 combined_runaway "$OUTBASE/sc_runaway/seed4" &
PID4=$!
wait "$PID0" || true
wait "$PID4" || true
echo "=== PHASE A DONE $(date +%s) ===" | tee -a "$OUTBASE/launch.log"

# Phase B — SC-noregress-working (seeds 0/2/4, 6s; 0+2 parallel then 4 alone)
echo "=== PHASE B SC-noregress-working $(date +%s) ===" | tee -a "$OUTBASE/launch.log"
run_seed 0 6 combined_noregress "$OUTBASE/sc_noregress/seed0" &
PID0=$!
run_seed 2 6 combined_noregress "$OUTBASE/sc_noregress/seed2" &
PID2=$!
wait "$PID0" || true
wait "$PID2" || true
run_seed 4 6 combined_noregress "$OUTBASE/sc_noregress/seed4"
echo "=== PHASE B DONE $(date +%s) ===" | tee -a "$OUTBASE/launch.log"

echo "=== ALL FINISHED $(date +%s) ===" | tee -a "$OUTBASE/launch.log"
