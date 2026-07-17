#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Baseline run — T-shape pushing (push_t task, yaw target).
#
# Reference-aligned defaults: same env bundle as run_box.sh, YAML swapped to
# config/sampling_c3_kik_t.yaml (per §9 reference-faithful push_t work).
# No proven-good invocation exists yet for push_t on this branch — treat this
# as best-guess, not a validated setting. Video ON by default.
#
# Usage:
#   ./run_T.sh [NAME]
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
    sed -n '2,17p' "$0" >&2
    exit 1
}

NAME="${1:-}"
case "$NAME" in
    -h|--help|help) usage ;;
esac

STEM="${NAME:-push_t_$(date +%Y%m%d_%H%M%S)}"

# §7.73 four production env flags + §7.75c support bundle (see run_box.sh
# for provenance). Ported unchanged to push_t; the T-shape task hasn't been
# swept against these knobs yet, so the bundle may need re-tuning.
# REF_RECONCILE_APPROACH retired 2026-07-10 — its =1 behaviour is the default.
# PUSHA_DECOUPLE_RECONCILE_FORCE_TRACKING no longer needed (was a workaround
# for the retired REF_RECONCILE_APPROACH=1 silent force-tracking-off).
PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=50 \
PUSHA_STAGE5_R_VECTOR=0.01,0.01,0.01 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
python main.py push_t \
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --admm-iter 3 \
    --max-time 8 \
    --name "$STEM" \
    --drake-frames-dir "results/${STEM}_frames" \
    --force-save-video \
    --math-diag
