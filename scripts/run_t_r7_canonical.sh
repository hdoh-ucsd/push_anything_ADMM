#!/usr/bin/env bash
# Canonical T-push run — reference-conformance DEFAULTS (2026-07-28 flip).
#
# As of the defaults flip, every reference-conformance behavior that was
# previously behind a REFCONF_* env gate is hard-wired ON in code, and the
# executor architecture is the reference one (OSC executes ALL modes; the
# R^7 direct-torque hybrid is removed from dispatch). This script is
# therefore a plain launch — no env gates.
#
# What is now default (formerly env-gated):
#   G-on ADMM + rho=1.0 under G      (REFCONF_USE_G_MATRIX / _ADMM_RHO_UNDER_G)
#   LCS scaling, arm-Cart inertia,   (REFCONF_SCALE_LCS / _ARM_CART_INERTIA /
#     E-block split                   _E_BLOCK_SPLIT)
#   reference OSQP options           (REFCONF_OSQP_OPTS)
#   surrogate x_pred clamp           (REFCONF_C3_SURROGATE_XPRED_CLAMP)
#   c_C3_raw mode-switch signal      (REFCONF_MODE_SWITCH_USE_C3_RAW)
#   object-slot-only sample ranking  (REFCONF_SAMPLE_RANK_OBJ_ONLY)
#   gravity-centered R^7 u-box       (REFCONF_R7_U_GRAVITY_CENTERED)
#   ONE reference OSC gain set incl. (REFCONF_OSC_ALIGN / _C3_MODE_GAINS /
#     rot task W=10/Kp=800/Kd=40      _FREE_MODE_GAINS / _EE_ROT_TASK)
#   OSC executes c3 mode             (REFCONF_R7_DIRECT_TORQUE removed)
#   u_sol force target               (PORT_FORCE_ROUTING default u_sol)
#   LTD approach-override skipped    (PORT_DISABLE_C3_OVERRIDE hard-wired)
#
# Historical: the pre-flip hybrid recipe (p118/p123 goal passes) lived at
# git 50d9e2f..c6992e8 — see memory r7-full-package-p126-p129 for why the
# reference-default architecture measures differently on the port frame.
#
# Usage: scripts/run_t_r7_canonical.sh [STEM] [MAX_TIME]
set -euo pipefail
cd "$(dirname "$0")/.."

STEM="${1:-tight_goal_ref_default_$(date +%Y%m%d_%H%M%S)}"
MAX_TIME="${2:-60}"

python main.py push_t --solver c3plus --admm-iter 3 \
    --max-time "${MAX_TIME}" \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --name "${STEM}" \
    2>&1 | tee "results/${STEM}.launcher.log"

grep "\[RESULT\]" "results/${STEM}.txt" || true
