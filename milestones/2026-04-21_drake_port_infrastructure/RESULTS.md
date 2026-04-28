# Drake C3 Port — Infrastructure Milestone

**Date frozen**: 2026-04-21
**Scope**: Reimplementation of Consensus Complementarity Control (Aydinoglu & Posa, ICRA 2022) for 7-DOF Franka Panda planar pushing in Drake.
**Predecessor milestone**: `planar_sandbox/milestones/2026-04-20_vanilla_c3_working/` (Drake-free 2D reference).

## Status: Infrastructure verified, algorithmic limitations documented

The contact-model infrastructure works correctly. The ADMM optimization converges cleanly (primal residuals ~0.01–0.15) when given a clean contact configuration. Three layered issues prevent end-to-end goal-reaching:
(1) workspace-limited approach, (2) initial-penetration in prepositioned mode, (3) vanilla C3's greedy contact mode selection.

The third issue is the known limitation that motivates C3+ (Bui et al. 2025) — this is not a bug but a design gap in vanilla C3.

## What works

### 1. LCS extraction from Drake SceneGraph
- Signed-distance queries via `ComputeSignedDistancePairwiseClosestPoints` (threshold = 0.10m)
- Contact-pair filter restricts pairs to pusher ↔ box_link
- 4-edge polyhedral friction pyramid: tangent basis = `[+t1, −t1, +t2, −t2]` where `t1 = nhat × ref`, `t2 = nhat × t1`
- Translational-velocity Jacobians evaluated at witness points via `CalcJacobianTranslationalVelocity`
- Discrete-time LCS matrices `A, B, D, d` from `M, Cv, tau_g, B` with first-order linearization
- Soft complementarity penalty (`w_comp = 100.0 · max(phi, 0)` on λ_n per horizon step)

Verified via `diagnostic_logs/pusher_weld_diagnosis.txt` and the contact sanity block in each run's initial log.

### 2. Pusher is rigidly welded to panda_link8
- `WeldJoint` named `panda_link8_welds_to_pusher`
- `pusher.is_floating() == False`
- 1 collision geometry (sphere, radius 2.5cm, id=223)
- Pusher world position at default state: `(0.088, −0.600, 0.876)` — 82cm above box

Verified in `diagnostic_logs/pusher_weld_diagnosis.txt`.

### 3. ADMM optimization (once the projection was fixed)
- Consensus split: z = full-horizon trajectory (x, λ, u stacked), δ = projected copy
- z-update: stacked QP via OSQP with dynamics + bounding-box constraints
- δ-update: Lorentz projection per contact, polymorphic on tangent dimension
- ω-update: dual ascent ω = ω + z − δ
- Adaptive ρ (Boyd §3.4.1, every 10 iterations, clamped to [0.1, 1000]) — not used at 3-iter cap but still available
- Early termination when primal, dual < 1e-3

### 4. Polyhedral friction projection fix (k=4)
The prior implementation applied a raw Lorentz formula `‖λ_t_4D‖₂ ≤ μ·λ_n` to the 4-component polyhedral tangent vector. This is geometrically incorrect: each λ_t[i] represents a force magnitude along a pre-specified direction, not a Euclidean component. The corrected projection:
1. Converts polyhedral → Cartesian: `F_t = [λ_t[0]−λ_t[1], λ_t[2]−λ_t[3]]`
2. Applies Lorentz projection to `(λ_n, F_t) ∈ ℝ³`
3. Splits back: `λ_t*[i] = max(±F_t[k], 0)` per axis

Verified via inline sanity check: `non-negative: True, cone ok: True`. Cross-consistency with sandbox's scalar projection: 17/17 tests pass.

### 5. Sim-to-LCS alignment
- 3 ADMM iterations per control step (Push Anything Table I, single-object)
- Control dt = 0.05s, sim dt ≈ 0.01s (Drake adaptive)
- Horizon N = 20
- Force limit 30 Nm (Franka safe)

## What doesn't work

### 1. Approach heuristic unreachable from default pose
Run: `diagnostic_logs/run_3iter.txt`.

- `INITIAL_ARM_Q = [0.0, 0.4, 0.0, −2.8, 0.0, 3.2, 0.785]` places pusher at world `(0.088, −0.600, 0.876)` — 82cm above box
- Proxy target: `(−0.18, 0.0, 0.05)` (box position minus 0.18m along goal direction)
- Arm settles at `(−0.019, −0.286, 0.056)` after 6s, stalls for remaining 2s
- 802/802 steps report `n_c = 0` (no contact made)
- Applied torque: 18–35 Nm (not saturated) — arm is not constrained by force limit
- Box at goal distance 0.300m throughout; no motion

Interpretation: The approach target is outside the reachable subset of the workspace from the default starting pose. The arm descends partway and then cannot continue without self-collision or joint-limit violation.

### 2. Prepositioned IK produces 2.7mm initial penetration
Run: `diagnostic_logs/run_3iter_preposed.txt`. Supporting: `diagnostic_logs/prepositioned_ik.txt`.

- IK target: `(−0.075, 0, 0.05)` (pusher center at pusher radius + clearance from box west face)
- Best IK solution (seed 4) yields `err = 4.05mm` from target
- Actual pusher at t=0: `(−0.072, −0.003, 0.050)` — pusher surface east edge at −0.047
- Box west face at x = −0.050
- Signed distance at t=0: `−0.00270m` (2.7mm penetration)

Drake's contact solver resolves this penetration with a large repulsion impulse in the first sim step:
- By t=0.5s, box moved from `(0,0,0.050)` to `(0.076, −0.057, 0.130)` — lifted 8cm
- By t=1.0s, box at `(−0.012, −0.200, 0.150)` — dragged 20cm south, lifted 10cm
- Final position (t=8s): `(0.081, −0.452, 0.170)` — 45cm south of origin, 12cm off table

This cannot be attributed to C3. The impulse throws the box before the first control loop closes.

### 3. Vanilla C3 greedy contact-mode selection
Once the initial impulse displaces the box, the arm makes contact on the wrong face (south face or underside of box). Vanilla C3 optimizes for whatever contact mode is currently active — it does not search over contact-face alternatives. So it plans forces that push the south face east, which is physically correct for that contact but doesn't move the box toward the goal.

The Push Anything paper (Bui et al. 2025) solves this with C3+: low-dimensional sampling over end-effector positions, evaluating each via MPC cost. This requires:
- LCS formulator to expose `E, F, H, c` for the η-slack reformulation
- ADMM to use closed-form piecewise projection (Bui et al. eq. 12)
- A sampling layer that proposes candidate pusher positions and runs MPC for each

This is the next milestone. Implementation deferred to the sandbox first for validation.

## Key numerical observations

### ADMM residuals (3-iter, prepositioned, single contact)
```
primal: 0.0175 → 0.0068  mono=True
primal: 0.0781 → 0.0044
primal: 0.1261 → 0.0079
primal: 0.2172 → 0.0240
```
Primal drops within each solve. Dual starts high (~160) due to warm-start discontinuity, drops to ~6 within 3 iterations. No pathological behavior.

### ADMM residuals (80-iter, prepositioned, single contact — pre-projection-fix)
```
primal: 0.0057 → 0.0480   (grew 8×)
primal: 0.0059 → 0.0372
primal: 0.0064 → 0.0515
```
Diverging. Projection bug was dominant cause.

### ADMM residuals (80-iter, prepositioned, two contacts pusher+ground)
```
primal: 9.47 → 2981.11      (grew 315×)
dual:   1714.65 → 14335.39
max λ_n: 36,829 N
```
Scaling mismatch between contacts with different force magnitudes. Vanilla C3's single Lorentz projection cannot handle simultaneous contacts at different scales.

## Files in this snapshot

### `source_snapshot/`
- `admm_solver.py` — C3Solver with fixed polyhedral projection (k=4)
- `lcs_formulator.py` — LCS extraction with pusher-box contact filter
- `task_costs.py` — Cost functions including approach-phase proxy
- `main.py` — Control loop driver with `--prepositioned` flag
- `env_builder.py` — Drake MultibodyPlant construction, pusher weld
- `tasks.yaml` — Task config (admm_max_iters=3, dt=0.05, horizon=20)
- `find_prepositioned_q.py` — IK script for PREPOSITIONED_ARM_Q
- `diagnose_pusher_weld.py` — Topology / collision-pair verification

### `diagnostic_logs/`
Contents of `results/drake_port_smoke/` at time of freeze:
- Multiple run logs (run_3iter.txt, run_3iter_preposed.txt, etc.)
- Projection-diff analysis (projection_diff.md)
- IK output (prepositioned_ik.txt)
- Pusher weld diagnostic (pusher_weld_diagnosis.txt)

## Next milestone

**`milestones/????-??-??_c3plus_sandbox_validation/`** — implement C3+ in the Drake-free planar sandbox first. The sandbox's algorithmic isolation (no Drake, no URDF loading, deterministic timestepping) is the correct environment to validate C3+'s sampling strategy and closed-form projection before porting to Drake.

Specific deliverables for that milestone:
1. Sandbox LCS formulator exposing `E, F, H, c` matrices
2. ADMM with η-slack reformulation and Bui et al. eq. 12 projection
3. Sampling layer over end-effector positions
4. Demonstration on the existing `off_axis` scenario (currently 0% progress in vanilla C3) — success criterion: ≥ 50% progress with C3+

After sandbox validation, port to Drake using the infrastructure frozen in this milestone.
