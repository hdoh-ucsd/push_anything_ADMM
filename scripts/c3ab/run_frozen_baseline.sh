#!/usr/bin/env bash
# Frozen-baseline protocol: 5 scenes x 3 trials, max 2 concurrent lanes.
set -u
R=/root/push_anything_ADMM/scripts/c3ab/run_c3plus_trials.sh
run() { bash "$R" "$@"; }
# pair 1
run c3ab_frozen_t01_open_table      push_t_oimT_m01_scaledQ            0.5 -0.3 3.14159265 3 900 7991 &
run c3ab_frozen_t01_single_obstacle push_t_oimT_m01_obst_scaledQ       0.5 -0.3 3.14159265 3 900 7992 &
wait
# pair 2
run c3ab_frozen_t01_ycb_clutter     push_t_oimT_m01_ycb_scaledQ        0.5 -0.3 3.14159265 3 900 7993 &
run c3ab_frozen_t01_icra_sign       push_t_oimT_m01_icra_scaledQ       0.5 -0.4 1.57079633 3 900 7994 &
wait
# solo
run c3ab_frozen_t01_box_clutter     push_t_oimT_m01_boxclutter_scaledQ 0.5 -0.3 3.14159265 3 900 7995
echo ALL_DONE
