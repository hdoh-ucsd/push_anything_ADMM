# Phase 0 baseline — reproduce-dairlib

**Branch:** `reproduce-dairlib` **HEAD:** `dacba48` **Date:** 2026-07-13
**Scope doc:** `docs/superpowers/plans/2026-07-13-reproduce-dairlib-scope.md` §3 Phase 0

Pinned "before" reference for the three-subsystem rebuild. Every subsequent phase
(Executor swap → Reposition swap → Admission/LCS conformance) is measured against
these seed-0 runs on the same protocol.

---

## Box, seed=0 (task-id 4 = West)

- **Invocation:** §7.51 chain bundle
  `PUSHA_FORCE_ROUTING=u_sol PUSHA_STAGE5_U_HORIZONTAL=10 PUSHA_STAGE5_U_VERTICAL=3 PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 PUSHA_EE_APPROACH_FACE_TARGET=1 PUSHA_DISABLE_C3_OVERRIDE=1 LCS_ALWAYS_ON_EE_BOX=1 REF_RECONCILE_APPROACH=1 LCS_NORMAL_VELOCITY_LEVEL=0 LCS_NORMAL_COMPLIANCE_K=0.0`
  `python main.py pushing --task-id 4 --solver c3plus --c3plus-projection lcp --ee-space --sampling-c3 config/sampling_c3_kik.yaml --admm-iter 25 --max-time 6 --seed 0 --no-record`
- **Run:** 601 steps complete (6.01s sim).
- **RESULT:** `final_obj_xy=(-0.4195, -0.0483)  goal_dist=0.1289m  orient_err=1.6759rad  success=NO  ref_gate=FAIL`
- Closure vs 0.3m init: **57%**. Overshoots the West goal by 12 cm (final x = -0.42 vs goal x = -0.30) and drifts y = -0.048.
- Tumble at final: **96° orientation error**. Consistent with the alignment-finding observation that "the box tumbles under the reference's own controller."
- Mode structure: 2 mode switches, mostly `kStayInRepos`. OSC 0 QP failures, 6% saturation. ADMM iters=25/25 (non-converged, pr=4.7 dr=25.6, tol=1e-3).
- Files: `box_seed0/run.log`, `box_seed0/result.txt`, `box_seed0/manifest.txt`.

## T (push_t), seed=0 — PARTIAL

- **Invocation:** phase-2-completion-sweep bundle
  `PUSHA_G_WEIGHT_EE_BOX_FINAL=1 PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 PUSHA_STAGE5_U_HORIZONTAL=50 PUSHA_STAGE5_U_VERTICAL=3 PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 LCS_ALWAYS_ON_EE_BOX=1 PUSHA_FORCE_ROUTING=u_sol PUSHA_EE_APPROACH_FACE_TARGET=1 PUSHA_DISABLE_C3_OVERRIDE=1`
  `python main.py push_t --solver c3plus --c3plus-projection lcp --ee-space --sampling-c3 config/sampling_c3_kik_t.yaml --admm-iter 25 --max-time 8 --seed 0 --no-record --math-diag`
- **Run:** 469 of 800 steps (4.69s of 8s sim). Killed by internal `timeout 1800` (30 min). No `[RESULT]` line. Rerun to full 8s at this stack takes ~53 min wall (see prior Phase-2 sweep on HEAD `490a7ca`, `logs/_phase2_completion_sweep_20260712_182324.log`). Partial state is unambiguous and sufficient for a "before" reference; full completion deferred.
- **Last observed state (step 469, t=4.69s):** `ee=(-0.055, -0.039, +0.078)  obj=(-0.001, +0.005, +0.020)  goal_dist=0.183m`. Init goal_dist = 0.187m → **T never translated** (obj drift < 6 mm from init through 4.69s).
- Contact briefly formed around step 232 (`[DRAKE-CONTACT] ee_box_normal=306N`) then lost by step 234 (down to 98N).
- Mode structure: 9 c3-mode steps out of 469 (1.9%); many `kToBetterRepos` retargets; EE_z stuck ~78mm (above 44mm ceiling for most of the run).
- **Characterization matches the alignment finding**: the T can transiently reach contact under the aligned stack but does not translate to goal.
- Files: `t_seed0/run.log`, `t_seed0/last_step.txt`, `t_seed0/manifest.txt`.

---

## What Phase-1 (Cartesian-force OSC) must beat vs this baseline

- **Box:** currently 57% closure with 96° tumble → Phase 1 must at minimum reproduce that closure (executor-only swap; §3 Phase-1 go/no-go). A degraded box after executor swap = STOP and reassess.
- **T:** currently no translation, only transient contact → Phase 1's expectation is "executor behavior reference-shaped" (the T's reposition/admission still diverge, so partial). The T's translation gate (position < 0.02m AND orient < 0.1 rad) is a Phase-3 target, not Phase-1.
