#!/usr/bin/env bash
# Fig 8 campaign — rerun of the 20 objects that failed under the buggy
# face-table sampler (user directive 2026-08-15), now on the
# geometry-generic projection sampler (commit 5c82410). Old logs are
# archived as *_seed0.oldsampler.log; these runs write the canonical
# <task>_seed0.log names that scripts/plot_fig8.py collects.
# Idempotent: trials whose log already contains [RESULT] are skipped.
set -u
cd "$(dirname "$0")/.."
OUT=results/fig8_objects
mkdir -p "$OUT"

run_one() {
  local task=$1
  local log="$OUT/${task}_seed0.log"
  if grep -aq "\[RESULT\]" "$log" 2>/dev/null; then
    echo "[FIG8-RERUN] SKIP (done) $log"
    return 0
  fi
  echo "[FIG8-RERUN] RUN $task  ($(date +%H:%M:%S))"
  python3 main.py "$task" --max-time 180 \
      --sampling-c3 config/sampling_c3_kik_t.yaml --seed 0 \
      --name "fig8r_${task}" > "$log" 2>&1
  local res
  res=$(grep -a "\[RESULT\]" "$log" | head -1)
  local latch
  latch=$(grep -am1 "ACHIEVED-FIXED-GOAL\]" "$log" | grep -o "step=[0-9]*" | head -1)
  echo "[FIG8-RERUN] DONE $task first_latch=${latch:-none}"
  echo "[FIG8-RERUN]   $res"
}

for task in I_shape_texture C_shape_texture R_shape_texture A_shape_video \
            Y_shape_video G_shape_video B_shape_video 3_shape_video \
            H_shape_texture E_shape_video expo_box lotion wood_block tape \
            eraser egg_carton book baby_toy gallon_milk xbox; do
  run_one "$task"
done
echo "[FIG8-RERUN] ALL DONE"
