#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Baseline run — box pushing (by direction), reference-aligned defaults.
#
# Canonical settings ported from experiments/§7.75c_repro/run_canonical.sh
# (72% closure on seed 0, goal_dist 0.30 m -> 0.085 m). Not a guarantee of
# goal-reach; seed sensitivity is real. Video ON by default.
#
# Usage:
#   ./run_box.sh <DIR> [NAME]     # DIR = N|E|S|W (or 1|2|3|4)
#
# Direction map (matches main.py --task-id):
#   1 = N (north)   2 = E (east)   3 = S (south)   4 = W (west)
#
# Outputs (all under results/):
#   <stem>.mp4      Encoded Drake-frames MP4
#   <stem>.html     Meshcat HTML replay
#   <stem>.txt      Full stdout log
#   <stem>_frames/  Per-step Drake VTK PNG frames (retained for re-encode)
# -----------------------------------------------------------------------------
set -euo pipefail
mkdir -p results

usage() {
    sed -n '2,20p' "$0" >&2
    exit 1
}

DIR="${1:-}"
NAME="${2:-}"
[[ -z "$DIR" ]] && { echo "run_box.sh requires a direction (N|E|S|W)" >&2; usage; }

case "$DIR" in
    N|n|1) TASK_ID=1 ;;
    E|e|2) TASK_ID=2 ;;
    S|s|3) TASK_ID=3 ;;
    W|w|4) TASK_ID=4 ;;
    -h|--help|help) usage ;;
    *) echo "invalid direction: $DIR (expected N|E|S|W or 1..4)" >&2; exit 1 ;;
esac

STEM="${NAME:-pushing_$(date +%Y%m%d_%H%M%S)}"

# §7.73 four production env flags + §7.75c support bundle. All default-OFF in
# main.py; must be set explicitly here (PUSHA_REF_OSC_ALIGN is a trap — re-
# triggers the §7.47 W_track=1 / Kp=200 handoff break).
# REF_RECONCILE_APPROACH retired 2026-07-10 — its =1 behaviour is now the
# default (see control/sampling_c3/sampling_based_c3_controller.py and
# control/task_costs.py). PUSHA_DECOUPLE_RECONCILE_FORCE_TRACKING is now a
# no-op guard-workaround; setting it is harmless but no longer required.
PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_STAGE5_U_HORIZONTAL=10 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.01,0.01,1 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
python main.py pushing \
    --task-id "$TASK_ID" \
    --solver c3plus \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 3 \
    --max-time 30 \
    --name "$STEM" \
    --drake-frames-dir "results/${STEM}_frames" \
    --force-save-video \
    --math-diag
