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

# 2026-07-26: env bundle pruned to reference-conformant flags + reference-
# matching VALUES set via port-specific mechanisms.
#
# KEPT — pure reference-conformance opt-ins:
#   REFCONF_OSC_C3_MODE_GAINS=1 — swaps port's c3-mode OSC gains to
#       reference values (§7.70).
#   PORT_DISABLE_C3_OVERRIDE=1        — disables port-only APPROACH-OVERRIDE
#       block (§7.51). Removing this workaround moves TOWARD reference.
#
# KEPT — port mechanism, reference-matching VALUE:
#   PORT_U_HORIZONTAL/VERTICAL=50 — per-axis EE-force cap. Mechanism
#       is port-only, but the 50 N value matches reference push_t's operating
#       force regime. Without this flag, main.py falls back to a 30 N scalar
#       cap — a 40% reduction that hurts metrics (p97 rot regressed +46%
#       vs p90 with this dropped alongside other flags).
#   PORT_EE_APPROACH_FACE_TARGET=1 — re-targets `w_ee_approach`'s x_ref to
#       the actual box-face contact point (matches reference q_vector on
#       EE position pointing at contact face).
#
# REMOVED — port-only, no reference analog:
#   PORT_G_WEIGHT_EE_BOX_FINAL=1  — Bui §IV-B.2 final-iter G-weighting;
#       port-only ST-layout hack.
#   PORT_R_VECTOR          — R override; identical to w_torque default.
#   PORT_LCS_ALWAYS_ON_EE_BOX=1         — phantom EE-BOX injection; reference
#       uses the 2mm signed-distance threshold. Explicitly set to 0 here.
#   PORT_FORCE_ROUTING=u_sol      — port-only force-routing hack.
PORT_LCS_ALWAYS_ON_EE_BOX=0 \
REFCONF_OSC_C3_MODE_GAINS=1 \
PORT_DISABLE_C3_OVERRIDE=1 \
PORT_U_HORIZONTAL=50 \
PORT_U_VERTICAL=50 \
PORT_EE_APPROACH_FACE_TARGET=1 \
python main.py push_t \
    --solver c3plus \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --admm-iter 3 \
    --max-time 30 \
    --name "$STEM" \
    --drake-frames-dir "results/${STEM}_frames" \
    --force-save-video \
    --math-diag
