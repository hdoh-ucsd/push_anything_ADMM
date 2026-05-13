# Paper alignment plan

Tracks the project's progress toward faithful implementation of Venkatesh et al. 2025 ("Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity", arXiv:2505.13350, IEEE RA-L).

Source: step 9 structural findings (`docs/step9_findings.md`), 9.3.2 wrapper audit, 9.4 kIK isolation test, and 9.4.1 torque-breakdown diagnostic. Round-by-round history lives in `docs/journal.md`; cross-round disciplines in `docs/lessons.md`; verbatim archived prose in `docs/_archive/paper_alignment_plan_2026-05-13.md`.

## Item inventory (8 paper-alignment surfaces + 1 newly-surfaced binding constraint)

### Category 1 — Paper-alignment fixes shipped

Items addressing paper deviations or step-9 structural findings. Both shipped; both mechanistically verified; neither sufficient for verdict-A success.

**Item 1.1 — Mechanism α (surrogate ADMM iteration count)**
Status: SHIPPED (commit `3e58cc6`) — see journal 2026-05-11. PARTIALLY LOAD-BEARING.
Audit reference: 9.3.2 item 3.3 (SIMPLIFIED)
Paper reference: §IV-C Algorithm 1 line 5
Location: `config/sampling_c3_params.yaml` (`surrogate_admm_iters: 3`)

Paper specifies surrogate C3 evaluation per sample without specifying a reduced-iteration approximation. Project's prior value (1 iteration) produced systematically pessimistic costs for the contact-seeking proxy sample.

**Item 1.2 — Finding C (staging geometry coefficient)**
Status: SHIPPED (commit `8f0b738`) — see journal 2026-05-11. NO-CHANGE in observed behaviour.
Step 9 finding: Finding C (geometric mismatch with contact threshold)
Paper reference: §IV-A (LCS-region locking)
Location: `control/task_costs.py:247-248` (`pre_approach_3d = obj − 0.16·g_hat`)

Stage-1 staging target was 5 mm outside Drake's 100 mm contact-activation threshold. Coefficient change moves it 15 mm inside. Finding C as framed was either not load-bearing for verdict-A or its locus is downstream of the binding constraint (see 9.4 / 9.4.1).

### Category 2 — Newly-surfaced binding constraint

**Item 2.1 — Contact-free stage triad: kIK guide-path (Effect A), executor home-hold (Effect B), wrapper-executor coupling (Effect C)**
Status: **REOPENED (wrapper-side, 9.4.7)** — Finding A RESOLVED by F2 (commit `761f9f8`, `sampling_radius` 0.18→0.13 m); Effect C now standalone binding constraint (persists under non-empty LCS regime); 1d's prior falsification reason directly invalidated, 1a/1b/1c candidates for re-evaluation, 1d re-tested and REFINED-FALSIFIED. Prior **PAUSED** state retained as institutional record in archive — see journal 2026-05-12 (architectural pause) and 2026-05-13 (F2 close + A/B/C execution).
Source: 9.4 kIK isolation → 9.4.1 torque breakdown → 9.4.2 integrator characterization → 9.4.3 clamped-integrator probe → 9.4.4 guide-path trace → 9.4.5-A baseline → 9.4.5-A.1 hold-home-pose → 9.4.5-B Attempt 1 (commit `161b7d9`) + sub-attempt 1.5 (reverted) → 9.4.5-C (read-only prior-art) → 9.4.5-D (1a falsified) → 9.4.5-E (1c falsified) → 9.4.5-F (1d falsified, reverted) → 9.4.6 (LCS-contents probe, commit `bd18003`) → 9.4.7 F2 (commit `761f9f8`) → 9.4.7 A/B/C (commit `e2f4099`).
Paper reference: outside paper scope (kIK trajectory generator is a project extension; paper does not specify a per-step Cartesian guide policy or a joint-space PD executor)
Location: **Effect A** at `control/sampling_c3/reposition_ik.py` `_build_guide_path` (lines 873-906) + per-loop reset (lines 1092-1099). **Effect B** at the joint-PD-with-grav-comp law (lines 1173-1202) + config gains (Kp_q=60, Ki_q=8, Kd_q=8, I_max=4.0 after commit `161b7d9`, torque_limit=30 Nm in `config/sampling_c3_kik.yaml`). **Effect C** at the wrapper's `prev_repos` selection logic (likely `control/sampling_c3/wrapper.py`; specific lines TBD by 9.4.5-C investigation).

The binding constraint factors into **three compounding effects**:

- **Effect A (Hypothesis F per-stride self-cancelling target)** — documented in `docs/reposition_ik.md:148-156` from step 8 work as Hypothesis F, with measurement TS4↔TS3 = 11.1 mm = one full per-stride distance. Re-derived in the standalone kIK probe and promoted from second-order (deferred behind α-pessimism) to binding (under post-α / post-C-fix configuration) — see journal 2026-05-11. Planner-side mechanism.
- **Effect B (executor cannot hold home pose)** — surfaced by 9.4.5-A baseline: all 7 standalone tests at tier 1-3 settle at the same EE fixed point ~(-0.016, -0.084, +0.025) within 14 control steps, regardless of target. Confirmed independent of kIK by the 9.4.5-A.1 hold-home-pose probe (driving PD directly with `q_target = q_home` falls to the same fixed point). Executor-side mechanism. **Step 8 did not test hold-home-pose**, so this is a genuinely-new diagnostic finding (the executor-tuning catalog at `docs/reposition_ik.md` §"Step 8 executor-tuning catalog (synthesis)" makes the gap explicit). See journal 2026-05-11.
- **Effect C (wrapper-executor coupling)** — surfaced by the 9.4.5-B sub-attempt 1.5 I_max sweep (Path D empty-LCS fraction monotonic in I_max, ruling out the measurement-artifact hypothesis from Attempt 1). Mechanism: the wrapper's sample selection produces `prev_repos` as the winner on the overwhelming majority of free-mode loops. The travel-cost discount + alignment-bonus calibration creates a positive feedback loop: as the executor drives EE toward `prev_repos`, `w_travel · ‖p_prev_repos − ee_now‖` drops, making the *same* target progressively cheaper than fresh strategy samples each loop. Under low executor authority (step 8 era) the EE never reached the chosen `prev_repos`, so the wrapper appeared to fail at *tracking*. Under high executor authority (9.4.5-B) the EE parks precisely at `prev_repos` — but those targets are themselves contact-inactive. **Same wrapper bug; inverted visible symptom.** Cross-ref: step 8 Candidate 2 (`docs/step8_sampling_c3_candidates.md:46-52`). Wrapper-side mechanism. Fix location: `control/sampling_c3/wrapper.py` `_build_samples` (lines 217-261) and the `c_sample` cost weights `w_align` / `w_travel` consumed in `control/sampling_c3/inner_solve.py`. See journal 2026-05-12.

**Step 8 candidate status (mapped to current binding state):**

| Step 8 candidate | Original hypothesis | Status |
|---|---|---|
| Candidate 1 (`docs/step8_sampling_c3_candidates.md:28-44`) | `prev_repos` wins via travel-cost discount; `surrogate_admm_iters=1` may produce systematically pessimistic estimates for fresh samples | RESOLVED by `3e58cc6` — see journal 2026-05-11. |
| Candidate 2 (`docs/step8_sampling_c3_candidates.md:46-52`) | `w_align` (30000) vs `w_travel` (200) calibration imbalance; alignment dominates by ~150×; once the wrapper commits to an alignment-positive target, no penalty dislodges it. Sub-options: (a) decay `w_align` over `steps_since_improve`, (b) raise `w_travel`, (c) replace cost-based lock-in with a "lock K loops then force re-sample" rule. | FALSIFIED (pre-F2) — see journal 2026-05-12. Sub-options 1a/1b/1c are candidates for re-evaluation post-F2 — see journal 2026-05-13. |
| Candidate 3 (`docs/step8_sampling_c3_candidates.md:54-69`) | `kToReposCost` / `kToC3Cost` thresholds (`hyst_repos_to_c3_frac=0.30`, position=0.50) require c3-mode cost to undercut repos-mode by 30–50% before switch fires; unreachable from a stuck state. Sub-option: (d) `steps_since_improve > N → force c3-mode-with-no-prev_repos` watchdog bypassing cost-comparison hysteresis. | FALSIFIED (pre-F2) — see journal 2026-05-12. Re-tested post-F2 (Option A) → REFINED-FALSIFIED — see journal 2026-05-13. |

All three effects must be addressed for the contact-free stage to work. **Planner-only fixes (G2/G4) handle Effect A only.** A planner fix that produced a perfect `q_target = q_home` (or any other fixed-pose command) would still see the arm fall to the same (-0.016, -0.084, +0.025) fixed point under the production PD law. Step 8's executor-tuning catalog (linked above) is the essential context for the executor-side fix decision: it shows what was tried (Kp doubling ran into the √6·30 Nm saturation ceiling; I_max raise from 0.5 to 2.0 was the shipped fix; grav-comp at q_target was structurally chosen over q_now for tracking control), what was deferred (anti-windup, velocity-reference D-term, feedforward acceleration, OSC architecture), and what is genuinely new territory (hold-home-pose verification was not in step 8's scope).

**Fix-direction implications.** Effect C is **wrapper-internal**, not executor-side — further executor tuning will not help and may worsen the symptom monotonically (the 9.4.5-B I_max sweep is the receipt). The two natural fix surfaces are step 8 OPEN items, promoted to binding by the upstream configuration change:

- **Candidate 2 sub-options (a/b/c)** touch the `c_sample` cost calibration consumed by `InnerSolver.evaluate_samples` in `control/sampling_c3/inner_solve.py`. Sub-option (a) — decay `w_align` over `steps_since_improve` — was the lightest-weight first surface; sub-options (b) raise `w_travel`, (c) replace cost-based lock-in with a K-loop hard rule. All three are now candidates for re-evaluation against the post-F2 cost landscape (see "Refined sub-option status" below).
- **Candidate 3 sub-option (d)** touches mode arbitration in `control/sampling_c3/wrapper.py` and/or `control/sampling_c3/mode_switch.py` — a `steps_since_improve > N → force c3-mode-with-no-prev_repos` watchdog bypassing the cost-comparison hysteresis. Watchdog code is now in tree (default off) — see `control/sampling_c3/mode_switch.py` `SwitchReason.kForceC3Watchdog` and `ProgressParams.watchdog_steps_since_improve_threshold`.

**Finding A is RESOLVED** by 9.4.7 F2 (commit `761f9f8`) — see journal 2026-05-13. Across 1594 `[C3]` dispatches between the two post-F2 verification runs, zero `n_λ=0` lines. The four upstream options enumerated during the 9.4.5-G architectural pause are reassessed:

- **U1 (Finding A directly):** RESOLVED by F2. The 5 mm geometric mismatch was the load-bearing mechanism.
- **U2 (sampler bias):** OPEN, deprioritized.
- **U3 (c_C3_raw mechanism):** OPEN.
- **G5 (OSC architectural pivot):** DEFERRED. The project's architecture demonstrably produces motion (Path A 106.5 mm post-F2) when geometry is correct — the problem is direction of motion, not capability. Architectural pivot becomes appropriate only if the sub-option re-evaluation chain also exhausts.

**Refined sub-option status (post-9.4.7 A/B/C, current).** Cost-landscape characterization in journal 2026-05-13 invalidates the original c_C3_raw arithmetic on which 1a/1b/1c rested:

- **1a (w_align decay)** — original c_C3_raw arithmetic rationale invalidated. Reducing alignment would now flip ranking toward "current EE" (bonus ~12k, vs ~30k on prev_repos/strat_*), entrenching c3-mode at current EE rather than pursuing fresh samples. Re-evaluation is open but requires redesign for the new landscape.
- **1b (raise w_travel)** — same revised status; prev_repos and strat_0 have nearly equal travel under F2, so travel-tuning alone does not promote fresh strat_*.
- **1c (K-loop lock-in)** — partially re-evaluable. Under F2, strat_0 wins WHEN generated. Pairing 1c with a workspace-filter relaxation could let fresh samples win the cleared slot. Run alone, 1c would still hit the "current/buffer wins instead" outcome.
- **1d (watchdog)** — re-tested, REFINED-FALSIFIED (failure mode shifted from empty-LCS to c3-mode non-persistence).

**New fix surface — F3 (workspace y-bound relaxation).** `workspace_xy_max[1] = 0.0` is the binding constraint on which fresh strat_* samples reach the candidate list. With the box near `obj_y = 0`, ~50% of the sampling circle is rejected. Under F2 the wrapper CAN find correct-direction samples — they just rarely survive the workspace filter. F3 has not been characterized for safety (the y-bound presumably exists for a reason). Cross-link: `control/sampling_c3/sampling.py:170-180`. See journal 2026-05-13.

**Implication for paper alignment** (Effect A planner-side fix): this is **kIK-internal**, not a low-level controller issue and not a paper-deviation in the audit sense. The kIK is structurally working as designed — the design assumes a moving target with cumulative integration, but the verdict-A scenario produces a stationary fixed point because per-loop guide rebuild uses `ee_now` as anchor. Tuning Kp / Ki / I_max / torque_limit / grav-comp model cannot break this fixed point; the structural change must touch how `p_guide[:, 0]` is built relative to the IK target.

Open question (next planner-side fix attempt): which G-fix preserves the design properties while breaking the fixed point. Fix surfaces, with design-constraint trade-offs:

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
| 1.1 Mechanism α | SHIPPED — journal 2026-05-11 | Paper-alignment shipped |
| 1.2 Finding C | SHIPPED — journal 2026-05-11 | Paper-alignment shipped |
| 2.1 kIK guide-path (Effect A) + executor home-hold (Effect B) + wrapper-executor coupling (Effect C) | **REOPENED (wrapper-side, 9.4.7)** — Finding A RESOLVED by F2 (commit `761f9f8`); Effect C now standalone binding constraint; 1d REFINED-FALSIFIED, 1a/1b/1c candidates for re-evaluation under new cost landscape; G5 deferred; F3 (workspace y-bound) enumerated. See journal 2026-05-12, 2026-05-13. | Binding constraint — geometric mechanism closed; wrapper arbitration is the binding surface |
| 3.1 Asymmetric hysteresis | NOT STARTED | Polish |
| 3.2 Empty-LCS handling | NOT STARTED | Polish |
| 3.3 Wrapper rename | NOT STARTED | Polish |
| 3.4 Sample strategy | NOT STARTED | Polish |
| 4.1 Finding B | DEMOTED | Conditional |
| 5.1 Parallelization | DEFERRED | Perf |

Count: 2 shipped, 1 diagnosed (binding constraint), 4 polish, 1 demoted, 1 deferred.

## Critical takeaway

Step 9's Findings A, B, C were structurally correct; Finding A was load-bearing under the post-α / post-C-fix configuration and has now been closed by F2 (commit `761f9f8`). Effect C (wrapper arbitration `prev_repos` lock-in) is the binding constraint going forward, with sub-options 1a/1b/1c candidates for re-evaluation under the new cost landscape and F3 (workspace y-bound) the most direct new surface. Future paper-alignment work on Items 3.1-3.4 (polish) should remain gated behind Effect A's planner-side fix and the Effect C re-evaluation chain.
