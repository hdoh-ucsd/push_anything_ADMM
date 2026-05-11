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

## Backlog

### C3Solver discards per-knot λ_k

Architectural verification gap surfaced in 9.2.1. `control/admm_solver.py:501-506` extracts only `u_seq` and `x_seq` from `z_sol`; per-knot λ_k is computed during ADMM iterations but never returned. Downstream consumers (controller, wrapper, instrumentation) cannot inspect the planner's contact-force trajectory or verify plan-vs-reality on the contact dimension. Flagged as noted-for-future-work, not a finding — the empty-LCS regime makes λ_k structurally absent in the cases studied here. See 9.2.1 inventory.

### Related material

`docs/g2_admm_iter_sweep.md` is an adjacent investigation of ADMM iteration count effects, including the `surrogate_admm_iters = 1` default at `control/sampling_c3/params.py:404` referenced in the 9.1 reframing of mechanism α. Not edited as part of step 9.5; available if 9.3 fix-design needs to revisit surrogate-C3 evaluation accuracy.

### α receipts in production logs

[GS-table] entries at any sampled step show c_C3 deltas between fresh samples (strat_*) and incumbent samples (current, prev_repos). The ~70-180k pessimism gap visible in `results/probe_9_3_0_baseline_kik.txt` is the production-side manifestation of mechanism α (`surrogate_admm_iters=1` inadequacy). No new instrumentation needed to characterize α quantitatively — the existing log carries it.

### D matrix shape evolution (C3+ feature)

Finding A's derivation cites D having shape (n_x, n_λ) with the structural property that D is empty when n_c=0. This claim is preserved across the C3+ feature commit, but the column composition changed: D previously had columns for (λ_n, λ_t) only (5·n_c columns when n_c>0); after the Phase 2 Stewart-Trinkle slack addition, D has columns for (γ, λ_n, λ_t) (6·n_c columns when n_c>0). The γ-columns are zero by construction (γ has no dynamics coupling), so the "only u→box path is D·λ when n_c>0" claim is preserved with γ-columns acting as structural zeros. Future Finding A re-derivations should note this shape detail.

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
