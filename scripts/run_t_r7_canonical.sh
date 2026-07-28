#!/usr/bin/env bash
# Canonical T-push R^7 run — the p118 PASSING recipe (2026-07-27).
#
# p118 (60 s): tight_goal PASS (latched) + loose_goal PASS —
# trans 0.0116 m, rot 0.1023 rad, 17 push cycles. First goal pass on the
# reference-conformant stack. See memory r7-p118-first-goal-pass and
# results/tight_goal_p118_r7_phantom_gate.*.
#
# Env gates (why each is here):
#   REFCONF_USE_G_MATRIX=1            arc-2 G-on ADMM; D1 override -> rho=1.0.
#                                     OMITTING THIS silently runs the G-off
#                                     rho=100 regime (contaminated p106-p111).
#   REFCONF_OSC_C3_MODE_GAINS=1       §7.70 reference c3-mode OSC gains.
#   PORT_DISABLE_C3_OVERRIDE=1        §7.51 skip LTD approach-override in c3.
#   REFCONF_SAMPLE_RANK_OBJ_ONLY=1    object-slot-only sample ranking
#                                     (a194280; removes arm-posture noise).
#   PORT_EE_Z_HOLD=1                  §7.76 R^7 z-hold (0786abb; breaks the
#                                     sphere-climb twist amplifier).
#   PORT_FREE_STALL_JOINT_RECOVERY=1  Cartesian-trap escape backstop
#                                     (d533feb/ffb7816; rarely fires).
#   PORT_DISABLE_CONTACT_LOSS_GATE=0  ARMS the phantom-aware contact-loss
#                                     watchdog (b6e1c22) — INVERTED SENSE:
#                                     =0 enables. Aborts phantom retreats in
#                                     ~0.5 s (the p118 17-cycle mechanism).
#   PORT_ACHIEVED_VERTICAL_RETRACT=1  vertical-first retraction after the
#                                     goal latch (e46afdb; prevents the p121
#                                     corner-cut shove that pushed the T
#                                     +90 mm past goal post-achievement).
#
# Usage: scripts/run_t_r7_canonical.sh [STEM] [MAX_TIME]
set -euo pipefail
cd "$(dirname "$0")/.."

STEM="${1:-tight_goal_r7_canonical_$(date +%Y%m%d_%H%M%S)}"
MAX_TIME="${2:-60}"

REFCONF_USE_G_MATRIX=1 \
REFCONF_OSC_C3_MODE_GAINS=1 \
PORT_DISABLE_C3_OVERRIDE=1 \
REFCONF_SAMPLE_RANK_OBJ_ONLY=1 \
PORT_EE_Z_HOLD=1 \
PORT_FREE_STALL_JOINT_RECOVERY=1 \
PORT_DISABLE_CONTACT_LOSS_GATE=0 \
PORT_ACHIEVED_VERTICAL_RETRACT=1 \
python main.py push_t --solver c3plus --admm-iter 3 \
    --max-time "${MAX_TIME}" \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --name "${STEM}" \
    2>&1 | tee "results/${STEM}.launcher.log"

grep "\[RESULT\]" "results/${STEM}.txt" || true
