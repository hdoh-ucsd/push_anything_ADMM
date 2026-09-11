#!/usr/bin/env bash
# Assemble results/c3plus_baseline_t01/<scene>/seed_N/ from frozen-protocol trials,
# post-process metrics, render videos (seed_0 only), and plot errors.
set -u
ROOT=/root/push_anything_ADMM
OUT=$ROOT/results/c3plus_baseline_t01
PP=$ROOT/scripts/c3ab/postprocess_baseline.py
RD=$ROOT/scripts/c3ab/render_c3plus_run.py
SDF=examples/sampling_c3/urdf/push_t_oimscale_m01.sdf
PY=/root/miniconda3/envs/push_anything_ADMM/bin/python3
declare -A GOAL BOXES DEMO
GOAL[open_table]="0.5 -0.3 3.14159265"; GOAL[single_obstacle]="0.5 -0.3 3.14159265"
GOAL[ycb_clutter]="0.5 -0.3 3.14159265"; GOAL[box_clutter]="0.5 -0.3 3.14159265"
GOAL[icra_sign]="0.5 -0.4 1.57079633"
BOXES[open_table]=""; BOXES[icra_sign]=""
BOXES[single_obstacle]="0.5,0.0,0.021,0.1,0.1,0.1"
BOXES[ycb_clutter]="0.5,0.0,0.021,0.1,0.1,0.1;0.37,0.2,0.001,0.175,0.095,0.06"
BOXES[box_clutter]="0.418,0.089,0.0008,0.108,0.089,0.0596;0.329,-0.070,0.0008,0.108,0.089,0.0596;0.621,-0.070,0.0008,0.108,0.089,0.0596"
DEMO[open_table]=push_t_oimT_m01_scaledQ; DEMO[single_obstacle]=push_t_oimT_m01_obst_scaledQ
DEMO[ycb_clutter]=push_t_oimT_m01_ycb_scaledQ; DEMO[icra_sign]=push_t_oimT_m01_icra_scaledQ
DEMO[box_clutter]=push_t_oimT_m01_boxclutter_scaledQ
WT=$ROOT/external/oim_c++_anything/.claude/worktrees/c3plus-ablation
for scene in open_table single_obstacle ycb_clutter box_clutter icra_sign; do
  for i in 1 2 3; do
    src=$ROOT/results/c3ab_frozen_t01_${scene}_trial${i}
    [ -d "$src" ] || continue
    seed=$((i-1)); dst=$OUT/$scene/seed_$seed
    mkdir -p "$dst/config_snapshot"
    cp $src/state_trace.jsonl $src/*.log "$dst/" 2>/dev/null
    cp -r $WT/examples/sampling_c3/${DEMO[$scene]}/parameters "$dst/config_snapshot/"
    $PY $PP "$dst" $scene ${GOAL[$scene]} > /dev/null 2>&1 || echo "PP-FAIL $scene $i"
    if [ $i = 1 ]; then
      cd $WT && $PY $RD --run-dir "$dst" --object-sdf $SDF \
        --title "C3+ frozen baseline | $scene | 0.1kg T" \
        ${BOXES[$scene]:+--boxes "${BOXES[$scene]}"} > "$dst/render.log" 2>&1 \
        && mv "$dst"/*.mp4 "$dst/rollout.mp4" 2>/dev/null || echo "RENDER-FAIL $scene"
      cd - > /dev/null
    fi
  done
done
# error plots (all scenes, seed_0)
$PY - <<'EOF'
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path("/root/push_anything_ADMM/results/c3plus_baseline_t01")
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for scene in ["open_table","single_obstacle","ycb_clutter","box_clutter","icra_sign"]:
    f = out/scene/"seed_0"/"trajectory.csv"
    if not f.exists(): continue
    rows = list(csv.DictReader(open(f)))
    t = [float(r["t"]) for r in rows]
    axes[0].plot(t, [float(r["pos_err"]) for r in rows], label=scene)
    axes[1].plot(t, [float(r["ang_err"]) for r in rows], label=scene)
axes[0].set_ylabel("position error [m]"); axes[0].axhline(0.02, ls="--", c="k", lw=0.7)
axes[1].set_ylabel("yaw error [rad]"); axes[1].axhline(0.10, ls="--", c="k", lw=0.7)
axes[1].set_xlabel("sim time [s]")
for a in axes: a.legend(fontsize=8); a.grid(alpha=0.3)
fig.suptitle("C3+ frozen baseline, 0.1 kg T — seed_0 error trajectories")
fig.savefig(out/"error_trajectories_seed0.png", dpi=130, bbox_inches="tight")
print("plots done")
EOF
echo ASSEMBLE_DONE
