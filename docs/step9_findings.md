# Step 9 findings: structural causes of the empty-LCS failure mode

## Executive summary

Step 9 isolated three structural mechanisms in the C3 ADMM stack that jointly produce the verdict-A failure mode from step 8 (both kIK and PWL end with 0 mm of box displacement after 800 control loops, park the EE southwest of the box, and refuse to re-enter contact). The three are not independent — they compose into a single degenerate stable point that no individual mechanism creates on its own.

Finding A — *empty-LCS gradient decoupling*. When the contact set is empty (n_c = 0, true in 776 / 801 verdict-A loops), the contact-Jacobian column block D vanishes and the box subsystem decouples from u in the linearized plant. Every box-state cost term (object-xy tracking, perpendicular-velocity penalty, terminal cost) carries an identically-zero u-gradient over the horizon; the stage-1 horizon is mathematically equivalent to an arm-tracking problem with no goal awareness.

Finding B — *goal-blind contact-pair selection*. LCSFormulator retains whichever Drake proximity pairs fall inside its 100 mm threshold, in Drake's enumeration order, filtered only by manipuland↔pusher id-set membership. No goal, target, or normal-direction predicate enters selection. The verdict-A step-1 contact has nhat_onto_box = [0, 0, −1] (vertical); under g_hat = [+1, 0, 0], λ_n along this normal cannot move the box toward the goal, and no filter rejects it.

Finding C — *stage-2/3 geometric mismatch with the contact threshold*. The three-stage approach geometry places stages 1 and 2 at surface-to-surface distances of 105 mm and 125 mm — both outside Drake's 100 mm proximity threshold. Reaching stage 2 deactivates contact rather than maintaining it. Stage 3 is inside the threshold but at −25 mm signed distance (penetrating), which the QP cannot stabilize on as a stationary contact-making configuration.

Compound failure: the EE parks at stage-2 geometry, contact is lost, the LCS goes empty (B and C drive arrival, A makes the regime absorbing), and the remaining cost gradient is satisfied at the parked configuration. The fixed point is stable in the sense the step 8 data showed — see Combined failure model for the trajectory through the verdict-A run.

## Finding A: Empty-LCS gradient decoupling

### Summary

When n_c = 0, the contact-Jacobian column block D has shape (n_x, 0). The box subsystem is then completely decoupled from u in the linearized plant: ∂(box state in plan)/∂u = 0 over the entire horizon, every box-state cost term carries an identically-zero u-gradient, and the QP minimizes only the arm-side cost. The 776 stage-1 loops in the verdict-A run are mathematically equivalent to a 7-DOF arm-tracking problem with zero goal awareness.

### Derivation

Three structural facts in the linearized plant:

1. *B_ctrl is block-diagonal at the arm/box partition.* `control/lcs_formulator.py:405-408` builds B_ctrl from J_u = ∂f/∂u = M⁻¹·B at the linearization point. M is block-diagonal at the arm-box partition and B has zero rows in the unactuated box block, so J_u has zero rows on box velocities — and B_ctrl has zero rows on every box state component.
2. *A is block-diagonal at the arm/box partition.* `control/lcs_formulator.py:399-403` builds A from J_q, J_v, N_mat, all drawn from the unconstrained dynamics linearization. With no contact coupling in J_f, A carries no off-diagonal block that would move an arm-side perturbation into box state.
3. *D is the only u→box path, and it vanishes.* `control/lcs_formulator.py:411-418` populates D inside `if n_c > 0:`. When n_c = 0, D is shape (n_x, 0). With A block-diagonal and B_ctrl zero on the box block, every box trajectory in the predicted horizon is determined entirely by x_0; no choice of u alters it.

The reader can re-derive this in five minutes from the three references above.

### Evidence

The verdict-A diag-kik run logs n_c = 0 on 776 / 801 control loops (97%). Cost-weight inventory: `w_obj_xy = 1e5` (read at `control/task_costs.py:161`, configured per task in `config/tasks.yaml`) is the dominant box-cost weight; the Q_obj construction at `control/task_costs.py:188-200` places every box-cost entry as a block-diagonal weight on box-state coordinates. In the empty-LCS regime each one is multiplied by a zero u-gradient — structurally deadweight.

### Fix surface

- `control/lcs_formulator.py:181` — `distance_threshold = 0.10` gates whether n_c > 0 at all.
- `control/lcs_formulator.py:206-213` — id-set-only contact filter (goal-aware filter or virtual-contact fallback would live here).
- `control/task_costs.py` — auxiliary u-gradient signal independent of D·λ (would require a new cost term coupling box-state cost into arm DOFs directly).

## Finding B: Goal-blind contact-pair selection

### Summary

LCSFormulator reads whichever Drake proximity pairs sit inside a 100 mm signed-distance threshold, in Drake's enumeration order, with id-set membership as the only filter. The first contact established between pusher and box may have any nhat orientation — and the QP accepts it as the contact whose λ_n is the only u→box-state path.

### Mechanism

The query is `ComputeSignedDistancePairwiseClosestPoints(distance_threshold)` at `control/lcs_formulator.py:199` with `distance_threshold = 0.10` from `control/lcs_formulator.py:181`. The filter at `control/lcs_formulator.py:206-213` retains pairs only when one side is in `_manipuland_geom_ids` and the other in `_ee_geom_ids` — no other predicate. `LCSFormulator.__init__` at `control/lcs_formulator.py:54` takes no goal, g_hat, or task parameter; the formulator is structurally goal-blind. The tangent-basis ref vector at `control/lcs_formulator.py:272-274` is a numerical-stability fallback (`[1,0,0]` unless the normal is nearly parallel, then `[0,1,0]`), not a goal-direction preference.

### Evidence

The step-1 contact in the verdict-A log has nhat_onto_box = [0, 0, −1] (purely vertical; pusher contacting from above). With g_hat = [+1, 0, 0], nhat · g_hat = 0; λ_n along this normal produces a vertical force on the box and cannot drive eastward motion. No filter rejects this top-down pair, and it is among the 25 non-empty-LCS loops in the 800-loop run. See Combined failure model for the t=13 closest-pass trace.

### Fix surface

- `control/lcs_formulator.py:199` — query call site (re-rank or filter returned pairs by goal-alignment).
- `control/lcs_formulator.py:206-213` — id-only filter (add `nhat · g_hat` predicate or normal-tolerance band).
- `control/lcs_formulator.py:54` — `__init__` needs a goal/g_hat injection point for either of the above.

## Finding C: Stage-2/3 geometric mismatch with contact threshold

### Summary

The three-stage approach geometry encoded in `QuadraticManipulationCost.build` places stages 1 and 2 outside Drake's 100 mm proximity threshold. The cost function's intermediate target is therefore not inside the contact-detection range; tracking it activates Finding A (empty LCS) rather than maintaining a contact set. Stage 3 lies inside the threshold but at penetrating signed distance, which the QP cannot stabilize on as a stationary contact-making configuration.

### Mechanism

Proximity threshold = `distance_threshold = 0.10` (surface-to-surface) at `control/lcs_formulator.py:181`. The three-stage targets at `control/task_costs.py:246-267` are `pre_approach_3d = obj_xy − 0.18·g_hat`, `approach_3d = obj_xy − (d_push + 0.15)·g_hat = obj_xy − 0.20·g_hat` (with `d_push = 0.05`), `proxy_3d = obj_xy − 0.05·g_hat`. Pusher radius = `PUSHER_RADIUS = 0.025` at `sim/env_builder.py:32`; box half-extent along g_hat = 0.05 m (`tasks.yaml pushing`). Surface-to-surface signed distance = centre-to-centre − 0.075 m.

### Evidence

| Stage | Target along −g_hat | Centre-to-centre | Surface-to-surface | Inside 100 mm threshold? |
|---|---|---|---|---|
| 1 (pre_approach) | 0.18 m | 0.180 m | 0.105 m | NO (5 mm out) |
| 2 (approach)     | 0.20 m | 0.200 m | 0.125 m | NO (25 mm out) |
| 3 (proxy)        | 0.05 m | 0.050 m | −0.025 m | YES (penetrating) |

Reconciliation with 9.1's wrapper-side reframing: `kRandomOnCircle` with `sampling_radius = 0.18 m` and the wrapper-proxy at `obj_xy − r·g_hat` both fall in the stages-1/2 band — the 109–130 mm sample-distance range reported in 9.1 is the same geometric regime viewed from the wrapper.

### Fix surface

- `control/lcs_formulator.py:181` — `distance_threshold` value (raising brings stage 2 inside, trades phantom-contact risk).
- `control/task_costs.py:246-267` — three-stage target geometry (shortening pre_approach / approach brings them inside).
- `sim/env_builder.py:32` — `PUSHER_RADIUS` (changing it shifts the surface-to-surface gap symmetrically).

## Combined failure model: degenerate stable point

### What the stable point is

The EE parks at stage-2 geometry — surface-to-surface ~125 mm from the box along −g_hat, outside the 100 mm threshold. LCS is empty (n_c = 0, D shape (n_x, 0)). By Finding A, every box-state cost term carries zero u-gradient. The remaining gradient signal comes from the control penalty (R·u, minimized at u ≈ τ_g) and the EE-approach term pointing the arm at the stage-2 target. Both are satisfied at the parked configuration. The first-order condition is met; no u-perturbation is rewarded. The fixed point is therefore stable in the sense the step 8 data showed — EE swings to position, contact is lost, arm parks.

### Why A + B + C compound

Counterfactual: what would break the stable point if any single mechanism were absent.

- *Without A*: even with empty LCS, a non-zero u-gradient on box-state cost would push the optimizer to explore u-perturbations that move the box; no parking.
- *Without B*: a goal-aligned normal filter would have rejected the t=0..20 top-down (nhat = [0,0,−1]) pair, leaving the LCS empty until a useful (nhat·g_hat > 0) contact was found — and the cost function would have driven the EE toward the correct face rather than accepting the wrong-face λ.
- *Without C*: the stage-2 target inside the 100 mm threshold would activate the contact set on arrival; reaching stage 2 maintains rather than breaks contact; Finding A's box-decoupling regime would not engage.

No single mechanism alone reproduces the verdict-A failure. The combination is what is absorbing.

### Trajectory through the stable point in the verdict-A run

c3-mode side, from the 8.4.6 trace (200-loop diag-kik, four step 8 fixes in place, box at obj_xy ≈ (0, 0), goal at (+0.300, 0)):

| Phase | Loops | EE z range | EE xy range | Notes |
|---|---|---|---|---|
| Early c3 (descent) | t=0..20 | 96 → 200 mm | (0, 0) to (-76, -157) mm | Above box top throughout |
| Late c3 / free | t=21..199 | settles at ~25 mm | drifts to (-145, -315) mm | EE far outbound in xy |

Closest pass during c3-mode: t=13. EE = (-26, -70, 132). Closest point on box surface = (-26, -50, 100), the south-top edge. Centre-to-box distance = 38 mm; signed distance pusher-surface to box-surface = +13 mm. **Pusher never touches the box across all 24 c3-mode loops.** Min signed distance +12.95 mm; median +56 mm; max +256 mm. Re-framed by finding: t=0..20 is the c3-mode window with Finding B's top-down contact (vertical normal accepted by the filter); t=13 is the closest pass at +13 mm — outside the 100 mm threshold per Finding C, and the EE never penetrates; t=22+ enters Finding A's empty-LCS regime and parks.

free-mode side, from the 8.6.5 trace receipt: across 177 free-mode loops × 20 knots = 3540 knot evaluations, the minimum signed distance from any IK-trajectory knot to the box surface is **+102.2 mm** — over four pusher-radii of clearance. **0 of 177 free-mode IK trajectories have any knot within contact-distance of the box.** This is the same stable point viewed via the wrapper's sample geometry: `kRandomOnCircle` samples sit in Finding C's stages-1/2 band (109–130 mm in the 9.1 reframe), and the IK targets that geometry faithfully. The 225 mm-displacement reading from the original 8.6.5 framing was the basis of overturned mechanism β; that interpretation is removed. The +102 mm clearance finding stands as the free-mode-side evidence of the empty-LCS stable point.

## 9.3 / 9.4 fix-design and isolation outcomes

Step 9's three findings were validated as structural observations but were not the binding constraint for verdict-A. The binding constraint is the joint-PD-with-grav-comp executor; see `docs/paper_alignment_plan.md` item 2.1.

### α fix (9.3.3) — PARTIALLY LOAD-BEARING

Commit 3e58cc6 raised `surrogate_admm_iters` from 1 to 3 (audit item 3.3). Mechanism α was reframed in step 9 as "surrogate-C3 evaluation pessimism." The α fix experimentally confirmed this:

- c_C3 cost gap (proxy − current) at step 800 inverted on both wrapper-based paths
- Mode switches collapsed
- Empty-LCS fraction collapsed
- Object moved for the first time (33-47mm)

However, motion was NW (away from goal). Mechanism α is confirmed mechanistically real but not behaviorally sufficient. The mechanism α reframing in this doc now resolves to: **CONFIRMED MECHANISM, NOT SUFFICIENT.**

### C-fix (9.3.4) — NO-CHANGE

Commit 8f0b738 changed `pre_approach_3d` coefficient from 0.18 to 0.16, moving stage-1 staging target from 5mm outside Drake's contact threshold to 15mm inside.

- Change applied correctly (verified via `[proxy]` log at Path D step 800: effective shifted from `[-0.194, 0.07, 0.05]` to `[-0.171, 0.05, 0.05]`)
- NO behavioral change on either path
- Path A: EE never enters stage-1 regime (stuck upstream of C-fix's locus)
- Path D: EE overshoots new staging target by ~10cm, invariant to coefficient

Finding C as framed was either not load-bearing or its locus is downstream of the binding constraint. The Finding C reframing in this doc now resolves to: **STRUCTURAL OBSERVATION CORRECT, NOT LOAD-BEARING UNDER CURRENT CONFIGURATION.**

### kIK standalone isolation (9.4) — Scenario B + addendum

Commit 0b0ee69 ran `scripts/probe_9_4_kik_reachability.py` driving the standalone kIK toward verdict-A's exact failed targets with no wrapper, no C3, no box. Both targets converged to essentially the same EE position (~(-0.016, -0.084, 0.025)) regardless of where commanded. IK reports feasible.

The verdict-A reachability stall is now localized to the joint-PD-with-grav-comp executor, downstream of IK (which solves cleanly) and outside the wrapper (which the probe excluded).

Path D's "10cm overshoot west of staging target" reframes as: the EE reached its PD equilibrium first, then the wrapper began commanding different targets, and the integrator wound up further west across multiple targets. Not active overshoot; residual drift across changing commands.

### Torque breakdown (9.4.1) — steady-state PD equilibrium, not saturation

Commit 61f064c added per-joint, per-component torque instrumentation to the probe. The headline finding:

- **Saturation is rare and transient.** Only 6 saturated joint-steps total out of 5,607 (joint 5: 5 transient steps; joint 2: 1 step). At steady state every joint is unsaturated; max steady-state demand is joint 1 at 27 Nm vs torque_limit=30 Nm.
- **The dominant transient saturator is tau_P on joint 5** during the initial overshoot. `q_err = -0.86 rad × Kp=60` produces -51.7 Nm vs the 30 Nm clip. This is the initial-transient signature, identical on both targets.
- **Integrator is not clamped.** Integral magnitudes 0.03–0.70 vs I_max=2.0 across all joints. Why the integrator doesn't wind further despite sustained q_err of 0.054 rad on joint 1 is the open question for 9.4.2.

The persistent EE-to-target offset comes from a steady-state joint-error pattern (`q_err ≈ 0.01-0.05 rad` per joint) that accumulates through FK into 7-16 cm of EE miss. The mechanism is NOT torque saturation; raising torque_limit alone would not move the steady state.

### Implications for step 9 framing

Findings A, B, C remain structurally correct as characterizations of the C3 / LCS / staging-geometry problem. But under the current executor configuration, the executor's steady-state equilibrium binds before any of these findings becomes load-bearing. Step 9's "combined failure model" should be amended to note that the failure model assumed an executor capable of tracking the optimizer's solutions — an assumption now known to be invalid for the verdict-A scenario.

Future work on Findings A, B, C remains paper-relevant but is gated behind the executor fix.

### 9.4.3 / 9.4.4 — guide-path self-cancelling target (step 8 Hypothesis F, promoted)

After the 9.4 / 9.4.1 / 9.4.2 chain (rolled up in the four subsections above) localized the verdict-A reachability stall to the kIK PD steady-state, two further diagnostics resolved the binding mechanism — and surfaced that it had been documented in step 8 already.

**9.4.3 (commit d87f386)** extended the integrator probe to 90 simulated seconds and added analytical Path B sanity checking. Joint 1 reaches integrator clamp at t≈60s with residual q_err -0.033 rad. The closed-loop force balance equation `Kp·|q_err| + Ki·I_max = tau_g_drake(q_now) + tau_g_drake(q_target)` is satisfied to within rounding (Path A measured: 18.0 Nm; Path B predicted: 17.998 Nm). **Critically, the EE-to-target distance is invariant from t=1s through t=90s at 0.159m**, indistinguishable from verdict-A's W1 miss of 0.16m. Both Ki sufficiency and gravity-model error are ruled out: the EE doesn't reach target even at clamp, and the gravity model is internally consistent.

**9.4.4 (read-only trace)** examined `_build_guide_path` at `control/sampling_c3/reposition_ik.py:873-906`. The IK targets `p_guide[:, 0] = next_waypoint(ee_now, p_target, z_safe=0.20, ds=0.01m)`, recomputed from current `ee_now` each control step. With `num_full_ik_knots=1` (the default), only knot 0 is sent to IK. The PD reaches the per-loop "1 cm ahead of where I am" target; `ee_now` advances marginally; the guide rebuilds with new ≈ old `ee_now`; the cycle is stable at a fixed point. The lift-traverse-descend PWL shape is the load-bearing safety property the design buys (per `tests/test_reposition.py:test_full_path_clears_box_bounding_box`: "the EE must never enter the box's xy footprint at z below box top — the key safety property the PWL design buys"); any G-fix that breaks this property must compensate via per-knot signed-distance constraints (currently disabled with `fk_min_distance=0` per V-9 revert at `docs/reposition_ik.md:49-58`).

**Prior-art rediscovery — the day's load-bearing methodological lesson.** The mechanism the 9.4 → 9.4.3 chain re-derived is documented at `docs/reposition_ik.md:148-156` from step 8 work as **Hypothesis F: joint-PD failure to track the IK solution**, with the same `TS4↔TS3 = 11.1 mm = one full per-stride distance` empirical signature. Step 8 correctly identified Hypothesis F and correctly deferred it: the deferral rationale (`docs/reposition_ik.md:182-186`) was that the wrapper at that time wasn't commanding contact-seeking targets — surrogate-C3 evaluation (mechanism α) was pessimistic, so even a perfect Hypothesis F fix wouldn't have moved the box. Step 9's α (3e58cc6) and C-fix (8f0b738) commits removed that upstream confound; the wrapper now commands contact-seeking targets and Hypothesis F is the binding constraint. The mechanism is the same; the framing has been promoted from "second-order defect" to "binding constraint for verdict-A under post-α/C-fix configuration." This is a configuration-promotion story, not a step 8 oversight.

**Reframing the 9.4 / 9.4.1 / 9.4.2 outcomes.** All three were correct as observations but each misidentified the binding constraint at its own level (IK reachability at 9.4; PD saturation at 9.4.1; integrator dynamics at 9.4.2). The 9.4.3 / 9.4.4 resolution shows the chain converged through three layers of symptoms before reaching the actual mechanism. The G1/G2/G3/G4 fix surfaces from 9.4.4 are queued for 9.4.5 with design-constraint trade-offs documented in `docs/paper_alignment_plan.md` Item 2.1.

### 9.4.5-A / 9.4.5-A.1 — Effect B surfaced (executor cannot hold home pose)

The 9.4.5-A pre-G-fix baseline (commit `8827917`) extended the standalone kIK probe to 7 targets across three tiers (1cm/5cm/10cm horizontal; pure descent; traverse+descend; W2; W1). The expectation from 9.4.4's framing was that Hypothesis F's per-stride truncation would manifest in the targeted phase per test (Phase 2 traverse for tier 1, Phase 3 descent for T2.1, Phase 1 lift for T2.3). What actually happened: **all 7 tests settled at the same EE fixed point ~(-0.016, -0.084, +0.025) within 14 control steps, with `phase=phase1-lift` and `knot0_offset=(0, 0, +10mm)` invariant across targets**. Even the smallest commanded motion (T1.1, +1cm horizontal at z=0.20) saw the arm fall from z=0.20 to z=0.025 in 14 steps. The per-stride truncation fixed point is the same regardless of target; the arm cannot hold its initial z=0.20 pose under torque-limited grav-comp, so every test ends up in phase1-lift trying (and failing) to lift back to z_safe.

This produced a new diagnostic question: is the inability-to-hold caused by the kIK's q_target sequence (the IK is producing q_targets that destabilize the executor), or by the executor itself? **9.4.5-A.1 (commit `1102939`)** answered it by bypassing the kIK entirely — driving the joint-PD law directly with `q_target = INITIAL_ARM_Q` fixed for 30 simulated seconds, no IK solves, no guide-path construction. Result: **the arm still falls**. EE displacement at t=30s = 197 mm; three integrators clamp at I_max=2.0 (joints 1, 2, 3 — the gravity-loaded shoulder/elbow chain); joint 1 settles at q_err = -0.405 rad (-23.2°); zero joints are torque-saturated. The mechanism is `tau_grav(q_target) + Kp·q_err + Ki·I_max` balancing the live gravity load at q_now ≠ q_target, with the equilibrium far from q_target because the integrator's Ki·I_max=16 Nm of authority plus Kp·q_err is not enough to reach q_target under the production gain configuration.

**Effect B is independent of the kIK.** The executor itself cannot hold home pose under the production PD-with-grav-comp configuration. Planner-only G-fixes (G2 horizon-aware static guide / G4 persistent anchor) handle the per-stride truncation (Effect A) but cannot resolve verdict-A on their own — even an ideal planner producing q_target=q_home commands would see the arm fall to the same fixed point. The fix decision must include an executor-side change.

**Step 8 did not test hold-home-pose.** Step 8's TS4 work measured per-stride tracking error (TS4↔TS3 = 11.1 mm) and per-step IK feasibility (TS4↔TS2 = 1.7 mm), both relative to the kIK's moving guide knot. There is no step-8 measurement of "can the executor hold a fixed q_target for 30 seconds?" The 9.4.5-A.1 finding is genuinely-new diagnostic territory rather than a re-derivation. The step 8 executor-tuning catalog at `docs/reposition_ik.md` §"Step 8 executor-tuning catalog (synthesis)" makes this gap explicit alongside the parameter-tuning history that did happen (Kp_q doubling reverted, I_max raised 0.5→2.0 shipped, grav-comp at q_target structurally chosen).

The two effects compound: Effect B drives the arm to a fixed point unrelated to p_target; Effect A's per-stride guide rebuild then locks the arm there because every guide knot is "1 cm ahead of where I am" computed from the fixed-point ee_now. Verdict-A is the joint manifestation. The 9.4.5-B fix design must address both.

### 9.4.5-B Attempt 1 + sub-attempt 1.5 — premise falsified; Effect C surfaced

**Attempt 1 (commit `161b7d9`, I_max 2.0→4.0)** produced PARTIAL home-hold improvement: clamped integrators 3→1 (joints 2, 3 unclamped; joint 1 still clamped at I_max=4.0), EE displacement 197mm → 132mm at t=30s, joint 1 q_err -0.405 → -0.343 rad. Verdict-A regression check showed a Path D shift (empty-LCS 1.43% → 80.9%, obj motion 35.2mm → 48.7mm) but Path A clean; the Path D number was open as "possible regression vs measurement-convention artifact" because the cited 9.3.4 baseline numbers were not findable in repo state for direct verification.

**Sub-attempt 1.5 (I_max 4.0→7.0, reverted; not committed)** confirmed the executor-side problem is solvable by integrator authority alone. The closed-form sizing analysis (see "Closed-form executor sizing rule" backlog entry below) predicted joint 1's integrator must wind to -6.23 rad·s to hold q_err=0; measured value was **-6.12 rad·s, within 2% of prediction**. EE displacement at t=30s: 25.25mm. Joint 1 q_err: -0.0151 rad. All 7 integrators unclamped, 0/7 torque-saturated. The trajectory shows clear asymptotic convergence (206mm at t=10s recovering to 25mm at t=30s) — the system is mid-recovery rather than stuck at a fixed point, with the residual q_err bounded by integrator wind-up time constant, not authority.

**Verdict-A regression on Path D, however, was severe.** Box motion: 35.2mm (cited baseline) → 48.7mm (Attempt 1) → **0.0mm** (sub-attempt 1.5). Mode switches: 6 → 8 → **70** (wrapper thrashing). Empty-LCS fraction: 1.43% → 80.9% → **100%** (every C3+ inner solve sees no contact pairs within threshold). Path A remained clean across all three configurations. Per the regression discipline ("Any regression → revert and reassess"), sub-attempt 1.5 was reverted; the working tree returned to commit `161b7d9` state.

**Three monotonic data points reveal a wrapper-executor coupling** that the two-effects framing did not anticipate. The hypothesis: more authoritative executor parks the EE more precisely at the wrapper's `prev_repos` target, which itself is contact-inactive in the verdict-A scenario; the C3+ inner solver then has no contact pairs to plan against, and the wrapper alternates mode searching for a way out. This is now labeled **Effect C** and reframed in `docs/paper_alignment_plan.md` Item 2.1 as a third compounding effect distinct from Effect A (planner-side) and Effect B (executor-side).

**Premise falsification.** 9.4.5-B was designed under the premise "fix executor home-hold → unblock verdict-A." The data falsifies this directly: fixing Effect B reveals Effect C as the next binding constraint, not verdict-A success. Step 8's executor-tuning work is institutionally correct and the home-hold gap it correctly identified can be closed; but closing it does not unblock verdict-A on Path D. **9.4.5-C is queued to characterize Effect C directly** — likely starting at the wrapper's `prev_repos` selection logic (`control/sampling_c3/wrapper.py`). Step 8's `docs/step8_sampling_c3_candidates.md` candidates 1/2/3 are the natural prior-art surface for that investigation (the same `prev_repos` mechanism, observed but not yet localized).

### 9.4.5-C — Effect C mechanism confirmed via step 8 prior art

**Read-only investigation** (no source / config / docs changes during the investigation pass; this docs entry is the institutional record produced after).

**Inversion framing.** Step 8 S8.3.2 (`docs/reposition_ik.md:419-421`) framed the wrapper failure as "wrapper picks prev_repos but EE diverges from it" — direct measurement of 175/200 prev_repos-winning loops showed the EE-to-target gap *grew* monotonically from 75 mm to 216 mm at +0.86 mm/loop. The pivot from S8.3 (wrapper-side) to S8.4 (controller-side) followed from that observation: the wrapper's sample selection looked correct on cost; the executor wasn't tracking. 9.4.5-B sub-attempt 1.5 inverted this: with I_max=7.0 the executor *does* track precisely (joint 1 q_err -0.0151 rad, closed-form steady-state hold validated), and the EE *does* converge to prev_repos — but the target itself is contact-inactive, and the wrapper picks the same kind of target again next loop. **Same wrapper bug (prev_repos lock-in via cost calibration); inverted visible symptom (EE diverges → EE parks).**

**Prior-art findings.** `docs/step8_sampling_c3_candidates.md` catalogues three candidate explanations for wrapper failure, surfaced during step 7 closure:

- **Candidate 1** (`step8_sampling_c3_candidates.md:28-44`) — prev_repos travel-cost discount + surrogate-iters pessimism on fresh samples. **RESOLVED** by mechanism α (commit `3e58cc6`, `surrogate_admm_iters` 1→3).
- **Candidate 2** (`step8_sampling_c3_candidates.md:46-52`) — `w_align` (30000) vs `w_travel` (200) calibration imbalance creating prev_repos lock-in. **OPEN** in step 8 docs; no commit addresses. Now confirmed binding by 9.4.5-C Path D data.
- **Candidate 3** (`step8_sampling_c3_candidates.md:54-69`) — `kToReposCost` / `kToC3Cost` thresholds (`hyst_repos_to_c3_frac=0.30`, position=0.50) require c3-mode cost to undercut repos-mode by 30–50% before switch fires; unreachable from a stuck state. **OPEN** in step 8 docs; no commit addresses.

**9.4.5-C Path D evidence (Candidate 2 binding).** The best_src histograms across the full 801-step runs:

| Configuration | best_src=prev_repos | best_src=current | best_src=strat_* | best_src=buffer |
|---|---|---|---|---|
| I_max=4.0 (commit `161b7d9`) | **761 / 801 (95%)** | 36 | 0 | 4 |
| I_max=7.0 (sub-attempt 1.5, reverted) | **626 / 801 (78%)** | 71 | 82 (strat_0/1/2) | 18 |

At step 800-801 in both runs, `target_changed=N` (prev_repos slot is the same target as last loop); the wrapper is in a sustained selection loop where `prev_repos` wins via lower `c_C3` + alignment bonus while the single fresh `strat_0` (with `num_additional_samples_repos = 1`) has full travel-cost penalty. The mechanism is the positive feedback loop step 8 Candidate 2 hypothesized: as the executor drives EE toward prev_repos, `w_travel · ‖p_prev_repos − ee_now‖` drops monotonically; no selection mechanism penalizes "I picked this sample last loop and it hasn't produced contact."

**Fix surface (rediscovery + promotion to binding).** The natural 9.4.5-D fix axes are step 8 OPEN items, not novel investigation surfaces:

- Candidate 2 sub-options touch `c_sample` calibration in `control/sampling_c3/inner_solve.py` (the `align_bonus` / `travel_pen` weights and their combination with `c_C3_raw`). Sub-option (a) — decay `w_align` over `steps_since_improve` — is the lightest-weight first attempt.
- Candidate 3 sub-option touches mode arbitration in `control/sampling_c3/wrapper.py` and/or `control/sampling_c3/mode_switch.py` — a `steps_since_improve > N → force c3-mode-with-no-prev_repos` watchdog.

This is the third instance in 2026-05-11/12 of the prior-art rediscovery pattern: 9.4.4 found Hypothesis F was already documented in step 8; 9.4.5-A.2 found step 8's executor-tuning history was directly applicable to home-hold sizing; 9.4.5-C found step 8 Candidates 2/3 documented and OPEN. Each rediscovery shipped without first searching the existing diagnostic doc surface. See the "Wrapper-executor coupling pattern" backlog entry below for the institutional discipline note.

## Backlog

### C3Solver discards per-knot λ_k

Architectural verification gap surfaced in 9.2.1. `control/admm_solver.py:501-506` extracts only `u_seq` and `x_seq` from `z_sol`; per-knot λ_k is computed during ADMM iterations but never returned. Downstream consumers (controller, wrapper, instrumentation) cannot inspect the planner's contact-force trajectory or verify plan-vs-reality on the contact dimension. Flagged as noted-for-future-work, not a finding — the empty-LCS regime makes λ_k structurally absent in the cases studied here. See 9.2.1 inventory.

### Related material

`docs/g2_admm_iter_sweep.md` is an adjacent investigation of ADMM iteration count effects, including the `surrogate_admm_iters = 1` default at `control/sampling_c3/params.py:404` referenced in the 9.1 reframing of mechanism α. Not edited as part of step 9.5; available if 9.3 fix-design needs to revisit surrogate-C3 evaluation accuracy.

### α receipts in production logs

[GS-table] entries at any sampled step show c_C3 deltas between fresh samples (strat_*) and incumbent samples (current, prev_repos). The ~70-180k pessimism gap visible in `results/probe_9_3_0_baseline_kik.txt` is the production-side manifestation of mechanism α (`surrogate_admm_iters=1` inadequacy). No new instrumentation needed to characterize α quantitatively — the existing log carries it.

### D matrix shape evolution (C3+ feature)

Finding A's derivation cites D having shape (n_x, n_λ) with the structural property that D is empty when n_c=0. This claim is preserved across the C3+ feature commit, but the column composition changed: D previously had columns for (λ_n, λ_t) only (5·n_c columns when n_c>0); after the Phase 2 Stewart-Trinkle slack addition, D has columns for (γ, λ_n, λ_t) (6·n_c columns when n_c>0). The γ-columns are zero by construction (γ has no dynamics coupling), so the "only u→box path is D·λ when n_c>0" claim is preserved with γ-columns acting as structural zeros. Future Finding A re-derivations should note this shape detail.

### Prior-art search discipline (additive to existing diagnostic discipline)

The 9.4 → 9.4.3 chain re-derived a mechanism documented from step 8 at `docs/reposition_ik.md:148-156` (kIK and PWL trackers settle at different z equilibria) and `:152-156` (Joint-PD covers ~0% of per-stride distance per 10 ms control loop). Step 8 labeled this **Hypothesis F: joint-PD failure to track the IK solution** and correctly deferred it on the rationale documented at `docs/reposition_ik.md:182-186` — the wrapper at that time wasn't commanding contact-seeking targets (mechanism α was pessimistic). Step 9's α (3e58cc6) and C-fix (8f0b738) removed that upstream confound; Hypothesis F became the binding constraint as a consequence of the upstream change, not as a step 8 oversight.

The discipline note, additive to the existing 9.1 / 9.2.x narrative:

- **Search project docs early** when investigating a mechanism. The first stop should be docs that catalog hypotheses or defects (in this codebase: `docs/reposition_ik.md` Bug catalog, Operational notes, Refactor-protection notes; `docs/step8_sampling_c3_candidates.md` Candidate list; this file's Findings A/B/C and Backlog). A brief grep against the suspected mechanism's keywords ("guide," "stride," "Hypothesis F," "tracking failure," etc.) before the first probe would have surfaced the prior characterization.
- **A prior diagnostic's deferral rationale is precious context.** When a hypothesis was named-but-deferred in earlier work, the deferral itself encodes a constraint judgment that should be preserved or explicitly re-evaluated. In this case, step 8's deferral of Hypothesis F was correct-at-that-time and the rationale ("wrapper not commanding contact-seeking targets") was the precise hook for re-evaluation.
- **When configuration changes upstream, audit deferred mechanisms.** Each time a fix lands that touches the upstream conditions of an earlier deferral, walk through the deferred mechanisms and check whether the deferral's rationale still holds. This is a low-cost institutional practice that prevents downstream re-derivation chains.

This is additive to (not a replacement for) the existing diagnostic discipline narrative below — the 9.1 ↔ overturned-mechanism-β reframing and the 9.2.x derivations remain the canonical story for the C3 / LCS / staging-geometry findings; the prior-art search note is for executor-side investigations where step 8 produced extensive empirical work that future investigations should consult first.

### Closed-form executor sizing rule (institutional knowledge)

Validated 9.4.5-B sub-attempt 1.5. At steady-state hold (q_err → 0, v → 0), joint j's integrator must wind to **2·tau_g(q_target_j) / Ki**, not 1·tau_g/Ki. The factor of 2 arises from the PD law structure (`reposition_ik.py:1173-1202`): the gravity feedforward at q_target is applied additively to the integrator's contribution, so the integrator must both *cancel* the feedforward and *provide* the negative actuation torque needed for force balance at q_now = q_target. For joint 1's home-hold case (tau_g(q_target) = 24.93 Nm, Ki = 8), required integrator magnitude = 6.23 rad·s; measured value at I_max=7.0 was -6.12 rad·s, within 2% of prediction.

**Step 8's Fix 6 sized to 1×** (pushing-task load ~7.39 Nm → I_max=2.0), which was correct for the pushing task because the executor is not asked to reach a static fixed point — the integrator is winding in response to a moving target, and reaching 2× of the instantaneous tau_g is not required mid-trajectory. For *static* holds (verification probes, idle states, prepositioned starts), the 2× rule applies.

**Prospective application.** When tuning Ki·I_max for any new task or verification probe, size Ki·I_max to **2 · max(tau_g(q_target)) across the loaded joints**, not 1×. The discipline note generalizes beyond the specific I_max value: the rule applies to any (Ki, I_max) pair as long as their product reaches the 2× threshold. This is additive institutional knowledge that survives any specific 9.4.5-B fix decision — the rule is correct regardless of whether the home-hold scenario itself is the right target for executor tuning (which 9.4.5-B sub-attempt 1.5 shows it is *not*, because of Effect C).

### Wrapper-executor coupling pattern (institutional discipline)

Wrapper-executor coupling pattern: improving the executor exposes wrapper defects that were previously masked. Step 8's S8.3 → S8.4 pivot from wrapper-side to executor-side was correct at the time (executor was the visible bottleneck), but it made the wrapper-side Candidates 2 and 3 invisible. Future investigations should re-evaluate documented OPEN items when fixes upstream of them land. The three-round prior-art rediscovery pattern of 2026-05-11/12 (Hypothesis F at 9.4.4, step 8 executor-tuning catalog at 9.4.5-A.2, Candidates 2/3 at 9.4.5-C) makes this pattern institutional: **check OPEN items in prior step docs before any new fix attempt.**

The pattern is structurally distinct from the "search project docs early" discipline above (which is about avoiding re-derivation of mechanisms documented as findings or hypotheses): this pattern is about re-evaluating prior **deferral decisions** whose underlying constraints may have changed. A candidate deferred when the executor was the bottleneck may become binding once the executor is fixed; a candidate deferred behind α-pessimism may become binding once α ships. The discipline cost is low (one grep against the prior step's OPEN/deferred list, before designing a new fix); the failure mode it prevents is high (re-deriving a documented mechanism through an experimental chain).

## Diagnostic discipline (parallel to V-1→V-9 and S8.0→S8.4)

### Round-by-round narrative

- **9.1** — wrapper-side sample evaluation. The proxy is the contact-seeking sample by design, but receives `align_bonus = 0` from the surrogate-C3 evaluation. The 225 mm finding from the original step-9 framing turned out to be ee_now drift inherited from c3-mode pre-displacement, not solver-induced deflection.
- **9.2.1** — C3 trajectory storage location. C3Solver discards per-knot λ_k; only `u_seq` and `x_seq` survive past `control/admm_solver.py:501-506`. Architectural gap, not a finding.
- **9.2.2** — cost gradient ∂cost/∂u in the empty-LCS regime. Finding A derived from the B_ctrl + A block-diagonal structure in `control/lcs_formulator.py`. The decoupling is structural in the linearized plant, not a parameter choice.
- **9.2.3** — contact-pair selection logic. Inspection of `LCSFormulator.extract_lcs_contacts` surfaced Finding B (goal-blind pair selection) and Finding C (stage-2/3 mismatch with the 100 mm threshold). Both visible from `control/lcs_formulator.py:180-213` plus the cost-function geometry at `control/task_costs.py:246-267`.

### Overturned hypothesis: mechanism β

Original framing: "IK solution diverges from the proxy by ~225 mm southward — kinematic-redundancy null-space deflection." 9.1 showed the 225 mm displacement was an artifact of conflating `ee_now` (already pre-displaced by c3-mode steps 0-20) with IK-induced deflection. The kIK IK is clean; it solves correctly to a target ~10 mm above ee_now. β does not exist as a mechanism. Methodological lesson: distinguish trajectory-inherited displacement from solver-induced displacement when characterizing IK behavior in a closed-loop run.

### Reframed hypothesis: mechanism α

Original α: "proxy is at 130 mm behind box surface by design — not designed to be in contact-distance." 9.1 showed this is a correct description of the proxy but does not name the failure. The proxy receives `align_bonus = 0` because the surrogate-C3 evaluation with `surrogate_admm_iters = 1` is structurally pessimistic about the symmetric behind-box pose — not because the proxy distance is wrong. The fix surface is the wrapper's sample evaluation, not the proxy design.

### Prescient-but-deferred candidate

`docs/step8_sampling_c3_candidates.md` Candidate 1 (line 42) asked exactly the question 9.1 answered ("Is the cheap-solve ADMM iteration count `surrogate_admm_iters=1` producing systematically pessimistic estimates for fresh samples?"). The candidate was named but not investigated at step 8 — it sat as "investigation needed" rather than "deferred to step N pending [specific data]." Institutional lesson: when a candidate is named but not investigated, label it explicitly as "deferred to step N pending [trigger condition or required instrumentation]," not merely as "uninvestigated." Candidate 1 would have been a stronger artifact framed as "deferred to step 9 pending per-sample `c_C3_raw` and `align_bonus` instrumentation" than left as an open question.
