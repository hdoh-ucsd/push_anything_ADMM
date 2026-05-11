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

**Item 2.1 — Joint-PD-with-grav-comp steady-state equilibrium**
Status: DIAGNOSED, not yet addressed
Source: 9.4 kIK isolation + 9.4.1 torque breakdown
Paper reference: outside paper scope (paper uses an operational-space controller at 1 kHz; project uses joint-space PD with grav-comp)
Location: `control/sampling_c3/reposition_ik.py` PD law (lines 1174-1202), config gains (Kp=60, Kd=8, Ki=8, I_max=2.0, torque_limit=30 Nm)

The standalone kIK probe (9.4) drove the tracker toward verdict-A's exact failed targets with no wrapper, no C3, and no box. Both targets converged to essentially the same EE position (~(-0.016, -0.084, 0.025)) regardless of where commanded. IK reports feasible at every step; the offset comes from the joint-PD-with-grav-comp executor.

9.4.1 torque breakdown (commit 61f064c) characterized the mechanism:

- **Saturation is NOT the binding constraint.** Only 6 saturated joint-steps total (joint 5: 5 steps in the transient at t<0.5s; joint 2: 1 step) out of 5,607 total joint-steps (= 801 steps × 7 joints).
- **The dominant saturating component is tau_P on joint 5** during the initial transient: `q_err = -0.86 rad × Kp=60` produces -51.7 Nm vs the 30 Nm clip. Joint 5 saturation pattern is identical on W1 and W2 (different targets, same initial joint-5 error).
- **Steady state is sub-clamp on every joint.** Max steady-state demand: joint 1 at 27.0 Nm, under the 30 Nm limit. `tau_grav = +35.8 Nm` partially cancelled by `tau_P = -3.3 Nm` and `tau_I = -5.6 Nm`.
- **Integrator is NOT clamped.** Integral magnitudes range 0.03–0.70 vs I_max=2.0. No joint hits the integral clamp.

The persistent EE offset comes from a steady-state joint error pattern (`q_err ≈ 0.01-0.05 rad` per joint) that accumulates through forward kinematics into 7-16 cm of EE miss. Raising torque_limit alone would speed up the initial transient but would not move the steady-state q_err — that requires either higher Kp (to shrink the q_err that balances the residual), higher Ki (to let the integrator absorb more), or a grav-comp model correction.

Open question (next investigation): the integrator at step 700 sits at 0.66-0.70 (35% of I_max=2.0) despite a sustained q_err of -0.054 rad on joint 1. A standard PI with no anti-windup, fed sustained error, would accumulate to clamp. Something prevents further accumulation — possibly an anti-windup mechanism, possibly a per-step rate-limit, possibly a leak term. Characterizing this is the 9.4.2 follow-up (constant-error probe).

Implication for paper alignment: this is outside the Venkatesh paper's scope. The paper specifies an operational-space controller (OSC) tracking task-space commands at 1 kHz. Project uses joint-space PD with grav-comp. If verdict-A is unblocked by tuning the executor's parameters, no paper-alignment work is added by this category. If verdict-A requires an OSC replacement, that becomes a major architectural item.

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
| 2.1 PD steady-state equilibrium | DIAGNOSED, not addressed | Binding constraint (new) |
| 3.1 Asymmetric hysteresis | NOT STARTED | Polish |
| 3.2 Empty-LCS handling | NOT STARTED | Polish |
| 3.3 Wrapper rename | NOT STARTED | Polish |
| 3.4 Sample strategy | NOT STARTED | Polish |
| 4.1 Finding B | DEMOTED | Conditional |
| 5.1 Parallelization | DEFERRED | Perf |

Count: 2 shipped, 1 diagnosed (binding constraint), 4 polish, 1 demoted, 1 deferred.

## Critical takeaway from today's investigation

Step 9's Findings A, B, C were correct as structural observations but were not the binding constraint for verdict-A. The binding constraint is the joint-PD-with-grav-comp executor settling at a steady-state equilibrium that doesn't reach commanded EE targets.

This was discovered by isolating the kIK from the wrapper and the C3 solver (9.4), showing that even with no target-switching, no contact dynamics, and no box, the EE cannot reach verdict-A's failed targets. The 9.4.1 breakdown then showed the mechanism is NOT torque saturation (which is rare and transient) but a sub-clamp steady-state where the integrator at ~0.7 vs I_max=2.0 plus Kp times small joint errors balances gravity at the wrong q.

Both α and C-fix produced their intended mechanistic effects upstream of the executor but couldn't help because the executor was the bottleneck.

Future paper-alignment work should be informed by 9.4.2 (integrator characterization). The right fix for the steady-state equilibrium depends on whether the integrator can wind further than 0.7 if given more time or a different gain — and that determines whether more paper-alignment work (Items 3.1 through 3.4) becomes meaningful or remains blocked behind the executor.
