#!/usr/bin/env bash
# Fig 8 replication campaign (user directive 2026-08-15): time-to-goal
# distributions for single-object manipulation, canonical tasks (box +
# mesh-T), 10 trials each (seeds 0-9), canonical 180 s protocol.
#
# Idempotent: a trial whose log already contains [RESULT] is skipped, so
# the campaign resumes cleanly after WSL crashes — just re-run this
# script. Trials interleave box/T per seed so partial data covers both
# objects evenly.
#
# Time-to-goal metric = sim time of the first [ACHIEVED-FIXED-GOAL]
# latch (geodesic object_on_target); runs that never latch are timeout
# outliers (>180 s) for the figure's shaded region.
set -u
cd "$(dirname "$0")/.."
OUT=results/fig8_replication
mkdir -p "$OUT"

run_one() {
  local task=$1 tid_args=$2 yaml=$3 seed=$4 name=$5
  local log="$OUT/${name}_seed${seed}.log"
  if grep -aq "\[RESULT\]" "$log" 2>/dev/null; then
    echo "[FIG8] SKIP (done) $log"
    return 0
  fi
  echo "[FIG8] RUN $log  ($(date +%H:%M:%S))"
  # shellcheck disable=SC2086
  python3 main.py "$task" $tid_args --max-time 180 \
      --sampling-c3 "$yaml" --seed "$seed" \
      --name "fig8_${name}_s${seed}" > "$log" 2>&1
  local rc=$?
  local res
  res=$(grep -a "\[RESULT\]" "$log" | head -1)
  local latch
  latch=$(grep -am1 "ACHIEVED-FIXED-GOAL\]" "$log" | grep -o "step=[0-9]*" | head -1)
  echo "[FIG8] DONE rc=$rc $log first_latch=${latch:-none}"
  echo "[FIG8]   $res"
}

for seed in 0 1 2 3 4 5 6 7 8 9; do
  run_one pushing "--task-id 4" config/sampling_c3_kik.yaml "$seed" box
  run_one push_t_mesh "" config/sampling_c3_kik_t.yaml "$seed" meshT
done
echo "[FIG8] CAMPAIGN COMPLETE"
