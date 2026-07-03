#!/usr/bin/env bash
# §7.75c CANONICAL INVOCATION — reproduces §7.72 (goal_dist~0.085m = 72% closure, seed 0)
#
# Reproducibility recovery: the §7.72 result was lost because the invocation was
# NEVER RECORDED. main.py defaults to `--solver c3` (Anitescu) + `--c3plus-projection
# componentwise`; §7.72 needs `--solver c3plus --c3plus-projection lcp`. Without
# these two CLI flags the LCP residual is NaN, dispatcher flaps ~120×, box doesn't
# move. Tree is not corrupted; the invocation was incomplete.
#
# Four production flags (§7.73 bank, all default-OFF, all required):
#   PUSHA_G_WEIGHT_EE_BOX_FINAL=1         (§7.67 B1-A v4 plan-side G-weighting)
#   PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1   (§7.70 executor-side ref gains in c3 mode)
#   PUSHA_DECOUPLE_RECONCILE_FORCE_TRACKING=1  (§7.63 force channel)
#   PUSHA_STAGE5_U_HORIZONTAL=50          (matches reference u-bound)
#
# Support env flags (default-OFF from earlier bands, required for the mode):
#   REF_RECONCILE_APPROACH=1              (§7.63)
#   LCS_ALWAYS_ON_EE_BOX=1                (§7.31 always-on)
#   PUSHA_FORCE_ROUTING=u_sol             (§7.42 planner u_sol → OSC λ_des)
#   PUSHA_EE_APPROACH_FACE_TARGET=1       (§7.46 face-target proxy bypass)
#   PUSHA_DISABLE_C3_OVERRIDE=1           (§7.51 skip LTD APPROACH-OVERRIDE in c3)
#   PUSHA_STAGE5_U_VERTICAL=3             (matches reference u-bound)
#   PUSHA_STAGE5_R_VECTOR=0.1,0.1,10      (reference R cost)
#
# Diagnostic env flags (this session's tracing):
#   PUSHA_SETPOINT_TRACE=1                (emits x_seq[1][7:10] = p_ee_des per tick)
#
# CLI (all required):
#   pushing --task-id 4                   (West push)
#   --solver c3plus --c3plus-projection lcp  (§7.72 needs LCP projection, NOT componentwise)
#   --ee-space                            (paper-aligned n_x=19, n_u=3 EE-force planner)
#   --sampling-c3 config/sampling_c3_kik.yaml
#   --admm-iter 25                        ("ST/25")
#   --max-time 6
#   --seed 0
#   --pitch-probe                         ([PITCH-PROBE] emit box tip + ee_z per tick)
#
# Note: `wsl --shutdown` from PowerShell before invocation is recommended for
# clean WSL state. Cannot be executed from inside WSL.

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root

PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_DECOUPLE_RECONCILE_FORCE_TRACKING=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
REF_RECONCILE_APPROACH=1 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
PUSHA_SETPOINT_TRACE=1 \
python main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 6 \
    --seed 0 \
    --pitch-probe \
    --no-record 2>&1 | tee "§7.75c_repro/run.log"
