# Paper alignment plan

Tracks the project's progress toward faithful implementation of Venkatesh et al. 2025 ("Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity", arXiv:2505.13350, IEEE RA-L).

Source: step 9 structural findings (`docs/step9_findings.md`), 9.3.2 wrapper audit, 9.4 kIK isolation test, and 9.4.1 torque-breakdown diagnostic.

## Item inventory (8 paper-alignment surfaces + 1 newly-surfaced binding constraint)

### Category 1 — Paper-alignment fixes shipped

Items addressing paper deviations or step-9 structural findings. Both shipped today; both mechanistically verified; neither sufficient for verdict-A success.

**Item 1.1 — Mechanism α (surrogate ADMM iteration count)**
Status: SHIPPED (commit 3e58cc6)
Audit reference: 9.3.2 item 3.3 (SIMPLIFIED)
Paper reference: §IV-C Algorithm 1 line 5
Location: `config/sampling_c3_params.yaml` (`surrogate_admm_iters: 3`)

Paper specifies surrogate C3 evaluation per sample without specifying a reduced-iteration approximation. Project's prior value (1 iteration) produced systematically pessimistic costs for the contact-seeking proxy sample.

Verification (9.3.3 data): c_C3 cost gap inverted on both paths. Path A +75k → −10.6k at step 800. Path D +69k → −181.6k. Mode switches collapsed (A: 10 → 4, D: 78 → 6). Empty-LCS fraction collapsed (A: 94.9% → 50.1%, D: 95.0% → 2.75%). Object moved 33.6mm (A) and 47.4mm (D). Direction: NW (away from goal at +x). PARTIALLY LOAD-BEARING.

**Item 1.2 — Finding C (staging geometry coefficient)**
Status: SHIPPED (commit 8f0b738)
Step 9 finding: Finding C (geometric mismatch with contact threshold)
Paper reference: §IV-A (LCS-region locking)
Location: `control/task_costs.py:247-248` (`pre_approach_3d = obj − 0.16·g_hat`)

Stage-1 staging target was 5mm outside Drake's 100mm contact-activation threshold. Coefficient change moves it 15mm inside.

Verification (9.3.4 data): change applied correctly, verified via `[proxy]` log on Path D at step 800. NO-CHANGE in behavior on either path. Path A: EE never enters stage-1 regime (stuck at ee_to_box ≈ 0.17m). Path D: EE overshoots new staging target by ~10cm, invariant to coefficient.

Implication: Finding C as framed was either not load-bearing for verdict-A, or its locus is downstream of the binding constraint (see 9.4 / 9.4.1).

### Category 2 — Newly-surfaced binding constraint

**Item 2.1 — kIK guide-path self-cancelling target (Effect A) AND executor cannot hold home pose (Effect B)**
Status: DIAGNOSED, not yet addressed
Source: 9.4 kIK isolation → 9.4.1 torque breakdown → 9.4.2 integrator characterization → 9.4.3 clamped-integrator probe → 9.4.4 guide-path trace → 9.4.5-A baseline → 9.4.5-A.1 hold-home-pose
Paper reference: outside paper scope (kIK trajectory generator is a project extension; paper does not specify a per-step Cartesian guide policy or a joint-space PD executor)
Location: **Effect A** at `control/sampling_c3/reposition_ik.py` `_build_guide_path` (lines 873-906) + per-loop reset (lines 1092-1099). **Effect B** at the joint-PD-with-grav-comp law (lines 1173-1202) + config gains (Kp_q=60, Ki_q=8, Kd_q=8, I_max=2.0, torque_limit=30 Nm in `config/sampling_c3_kik.yaml`)

The binding constraint factors into **two coupled effects**, both surfaced today:

- **Effect A (Hypothesis F per-stride self-cancelling target)** — documented in `docs/reposition_ik.md:148-156` from step 8 work as Hypothesis F, with measurement TS4↔TS3 = 11.1 mm = one full per-stride distance. Re-derived 9.4 → 9.4.3 in the standalone kIK probe and promoted from second-order (deferred behind α-pessimism) to binding (under post-α / post-C-fix configuration). Planner-side mechanism.
- **Effect B (executor cannot hold home pose)** — surfaced 9.4.5-A baseline (commit `8827917`): all 7 standalone tests at tier 1-3 settle at the same EE fixed point ~(-0.016, -0.084, +0.025) within 14 control steps, regardless of target. Confirmed independent of kIK at 9.4.5-A.1 (commit `1102939`): bypassing the kIK and driving PD directly with `q_target = q_home` for 30 simulated seconds produces 197 mm EE displacement, with 3 integrators clamped at I_max=2.0 (joint 1 q_err = -0.405 rad) and zero torque saturation. Executor-side mechanism. **Step 8 did not test hold-home-pose**, so this is a genuinely-new diagnostic finding (the executor-tuning catalog at `docs/reposition_ik.md` §"Step 8 executor-tuning catalog (synthesis)" makes the gap explicit).

Both effects must be addressed for the contact-free stage to work. **Planner-only fixes (G2/G4) handle Effect A only.** A planner fix that produced a perfect `q_target = q_home` (or any other fixed-pose command) would still see the arm fall to the same (-0.016, -0.084, +0.025) fixed point under the production PD law. Step 8's executor-tuning catalog (linked above) is the essential context for the executor-side fix decision: it shows what was tried (Kp doubling ran into the √6·30 Nm saturation ceiling; I_max raise from 0.5 to 2.0 was the shipped fix; grav-comp at q_target was structurally chosen over q_now for tracking control), what was deferred (anti-windup, velocity-reference D-term, feedforward acceleration, OSC architecture), and what is genuinely new territory (hold-home-pose verification was not in step 8's scope).

Diagnostic chain that re-derived the mechanism:

- **9.4 (commit 0b0ee69)** — Standalone kIK probe drove the tracker toward verdict-A's W1 and W2 targets with no wrapper, no C3, no box. Both converged to essentially the same EE position (~(-0.016, -0.084, 0.025)) regardless of commanded target. IK reports feasible (knot-0 IK lands within √3·1mm of the guide knot per `docs/reposition_ik.md:150`).
- **9.4.1 (commit 61f064c)** — Per-joint torque breakdown ruled OUT saturation as the cause: 6 saturated joint-steps out of 5,607 (joint 5 transient only). Steady-state demand sub-clamp on every joint. Reframed as PD steady-state equilibrium issue.
- **9.4.2 (commit dd49e59)** — 30s integrator probe ruled OUT anti-windup, leak, reset-events, and different-error hypotheses (H1/H3/H4/H5). Confirmed H2 (slow update) — integrator winds linearly toward clamp.
- **9.4.3 (commit d87f386)** — 90s clamped-integrator probe + analytical Path B sanity check. Joint 1 clamps at t≈60s with residual q_err -0.033 rad. **EE-to-target distance is invariant from t=1s through t=90s at 0.159m**, indistinguishable from verdict-A's 16cm miss. Ki sufficiency and gravity-model error both ruled out (Path A and Path B agree on closed-loop force balance to within rounding). The closed-loop fixed point IS the binding constraint.
- **9.4.4** — `_build_guide_path` trace surfaced the structural cause: the IK targets `p_guide[:, 0] = next_waypoint(ee_now, p_target, z_safe=0.20, ds=0.01m)`, recomputed from current `ee_now` each control step. With `num_full_ik_knots=1` (default), only knot 0 is sent to IK. The PD reaches the per-loop "1 cm ahead of where I am" target; the guide rebuilds with new `ee_now` ≈ old `ee_now`; the cycle is stable at a fixed point.

**This was step 8 Hypothesis F**, documented at `docs/reposition_ik.md:148-156` with the same `TS4 - TS3 = 11.1 mm` empirical signature. Step 8 correctly identified Hypothesis F and correctly deferred it: the deferral rationale (§Step 8 closure at `docs/reposition_ik.md:182-186`) was that the wrapper at that time wasn't commanding contact-seeking targets — surrogate-C3 evaluation (mechanism α) was pessimistic about the contact-seeking proxy, so even a perfect Hypothesis F fix wouldn't have moved the box. With α (3e58cc6) and C-fix (8f0b738) shipped, that rationale no longer holds. Hypothesis F is **promoted from second-order to binding constraint** as a consequence of the upstream configuration change, not as a step 8 oversight.

Implication for paper alignment: this is **kIK-internal**, not a low-level controller issue and not a paper-deviation in the audit sense. The kIK is structurally working as designed — the design assumes a moving target with cumulative integration, but the verdict-A scenario produces a stationary fixed point because per-loop guide rebuild uses `ee_now` as anchor. Tuning Kp / Ki / I_max / torque_limit / grav-comp model cannot break this fixed point; the structural change must touch how `p_guide[:, 0]` is built relative to the IK target.

Open question (next investigation, 9.4.5 fix attempt): which G-fix preserves the design properties while breaking the fixed point. Fix surfaces, with design-constraint trade-offs:

- **G1 (direct target — IK targets `p_target` instead of `p_guide[:, 0]`)** — Removes the lift-traverse-descend collision avoidance the guide enforces. The `tests/test_reposition.py:test_full_path_clears_box_bounding_box` invariant ("the EE must never enter the box's xy footprint at z below box top — the key safety property the PWL design buys") protects this; G1 breaks it unless `fk_min_distance > 0` is opted into. The opt-in path has its own documented cost: ~19 extra `ComputeSignedDistancePairwiseClosestPoints` calls per loop (`docs/reposition_ik.md:49-58`), which on V-8 measurement borderline overshoots the 8 ms IK budget. So G1 is "small diff" only nominally — in practice it's small plus a known-expensive collision-checking budget.
- **G2 (horizon-aware static guide — build once on target change, consume next knot per step)** — Preserves the box-clearance safety property because the PWL shape is intact, just constructed once from initial `ee_now` to `p_target`. **But introduces planner-executor asynchrony**: the guide advances on a planning clock, the arm advances on dynamics, and they need to stay in sync. If the arm falls behind (disturbance, transient, integrator-still-winding), the guide is ahead of where the arm actually is and the PD may not catch up. Mitigation patterns to revisit at fix-design time: re-anchor if `ee_now` drifts more than X from expected position; slow anchor advance when `q_err` is large; hybrid (persistent anchor for direction, `ee_now` for progress check). These are design considerations within G2, not separate G-fixes.
- **G3 (adaptive ds — larger stride when far from `p_target`)** — Larger `ds` doesn't just affect PD trackability; it also affects box-clearance because the lift-traverse-descend discretization assumes per-step `ds`. Doubling `ds` during far-from-target phases means the lift phase might overshoot z_safe in a single step. The `test_full_path_clears_box_bounding_box` regression would need to verify clearance still holds with variable `ds`.
- **G4 (decoupled recomputation policy — rebuild from a persistent anchor, not from `ee_now`)** — Same trade-off as G2: preserves PWL shape but introduces planner-executor asynchrony. Requires anchor-advance discipline (when does anchor advance? per planning step? per achieved-knot?). Mitigation patterns same as G2.

The current design's **self-consistency property** — the guide is always valid for where the arm actually is, no matter what just happened — is what we want to preserve when breaking the fixed point. G1 and G3 sacrifice the box-clearance safety property; G2 and G4 sacrifice the self-consistency property in exchange for forward progress, with mitigation patterns to revisit.

### Category 3 — Polish items (paper deviations, not yet load-bearing)

Same items as the prior plan, reordered now that the binding constraint is identified.

**Item 3.1 — Asymmetric hysteresis**
Status: NOT STARTED
Audit reference: 9.3.2 item 4.2 (DIFFERENT)
Paper reference: §IV-D
Location: `config/sampling_c3_params.yaml`

Paper recommends asymmetric hysteresis (h_rich-to-free larger). Project uses symmetric 0.30. With α now fixed, the H1/H2 question (is symmetric hysteresis the problem, or is α-pessimism overcoming any reasonable hysteresis) could be tested via raising `hyst_repos_to_c3_frac` alone.

**Item 3.2 — Empty-LCS sample handling**
Status: NOT STARTED
Audit reference: 9.3.2 item 3.5 (DIFFERENT)
Paper reference: silent
Location: `control/sampling_c3/inner_solve.py`

Project lets empty-LCS samples compete on `c_C3_raw` without `align_bonus`. Subordinates contact-seeking proxy samples. Code change required.

**Item 3.3 — Wrapper class rename**
Status: NOT STARTED
Audit reference: 9.3.2 Block 6
Paper reference: §IV-B Eq. 5 (bilevel optimization)

Rename `SamplingC3MPC → SamplingBilevelController`, `wrapper.py → sampling_bilevel_controller.py`. Mechanical.

**Item 3.4 — Sample strategy gap**
Status: NOT STARTED
Audit reference: 9.3.2 item 3.1 (SIMPLIFIED)
Paper reference: §IV-C
Location: `control/sampling_c3/sampling.py`

3 of 7 strategy enums wired. Proxy injection is project-specific. Audit classified MEDIUM impact.

### Category 4 — Conditional path

**Item 4.1 — Finding B (goal-aware contact selection)**
Status: CONDITIONAL, now DEMOTED
Audit reference: not in audit (LCS-formulator-level)
Step 9 finding: Finding B
Location: `control/lcs_formulator.py:199, 206-213`

Demoted because 9.4 / 9.4.1 showed the binding constraint is the executor, not contact-pair selection. Finding B may still be a real paper-deviation but isn't relevant until the executor reaches commanded targets.

### Category 5 — Deferred or perf

**Item 5.1 — Parallel sample evaluation**
Status: DEFERRED
Audit reference: 9.3.2 item 3.6 (SIMPLIFIED)
Performance-only, not correctness.

## Status summary

| Item | Status | Category |
|---|---|---|
| 1.1 Mechanism α | SHIPPED | Paper-alignment shipped |
| 1.2 Finding C | SHIPPED | Paper-alignment shipped |
| 2.1 kIK guide-path (Effect A, step 8 Hypothesis F promoted) + executor cannot hold home pose (Effect B, new) | DIAGNOSED, not addressed | Binding constraint — two coupled effects |
| 3.1 Asymmetric hysteresis | NOT STARTED | Polish |
| 3.2 Empty-LCS handling | NOT STARTED | Polish |
| 3.3 Wrapper rename | NOT STARTED | Polish |
| 3.4 Sample strategy | NOT STARTED | Polish |
| 4.1 Finding B | DEMOTED | Conditional |
| 5.1 Parallelization | DEFERRED | Perf |

Count: 2 shipped, 1 diagnosed (binding constraint), 4 polish, 1 demoted, 1 deferred.

## Critical takeaway from today's investigation

Step 9's Findings A, B, C were correct as structural observations but were not the binding constraint for verdict-A. The binding constraint is the kIK guide-path's self-cancelling target — **the same Hypothesis F that step 8 documented at `docs/reposition_ik.md:148-156` and correctly deferred** because, at that time, the wrapper wasn't commanding contact-seeking targets (mechanism α was suppressing them). With α (3e58cc6) and C-fix (8f0b738) shipped, the wrapper now commands contact-seeking targets and Hypothesis F is the binding constraint.

The day's load-bearing methodological lesson is **prior-art rediscovery, not novel diagnosis**. The 9.4 → 9.4.3 chain re-derived a mechanism the project had already characterized empirically (TS4↔TS3 = 11.1 mm = one full per-stride distance). The chain was rigorous but inefficient — searching `docs/reposition_ik.md` for "guide" or "stride" or "Hypothesis F" earlier in the chain would have surfaced the prior characterization. The institutional discipline note added to step 9's backlog: when investigating a mechanism, search project docs early for prior-art entries, especially docs that catalog hypotheses or defects (e.g., `docs/reposition_ik.md` Bug catalog, Operational notes, Step 8 closure sections).

What is genuinely new is the **promotion**: prior to α + C-fix shipping, Hypothesis F was second-order behind "wrapper picks wrong targets." After α + C-fix, the wrapper picks contact-seeking targets and Hypothesis F becomes binding. Future paper-alignment work on Items 3.1-3.4 (polish) should remain gated behind the Hypothesis F fix — the same gating step 8 identified, now with a promoted urgency.
