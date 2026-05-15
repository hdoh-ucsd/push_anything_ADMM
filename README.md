# Push Anything ADMM

A **PyDrake** implementation of Contact-Implicit Model Predictive Control (CI-MPC) for non-prehensile pushing, ported from Bui et al., *"Push Anything"* (ICRA 2026). A 7-DOF Franka Emika Panda arm pushes objects toward a goal by linearising the Drake plant into a discrete Linear Complementarity System (LCS) at each control step and solving the N-step trajectory as one stacked QP with a custom ADMM solver.

Two algorithm variants are implemented:

- **C3** (Aydinoglu et al. 2024, *T-RO*) — ADMM with consensus set + Stewart-Trinkle LCP projection (Aydinoglu §V-B.3.b)
- **C3+** (Bui et al. 2026, *ICRA*) — slack-variable formulation with closed-form per-component projection (4–5 orders of magnitude faster on the projection step)

An outer **sampling-C3** wrapper (Venkatesh et al. 2025, *RA-L*) layers global sample-based contact-mode reasoning over the inner C3/C3+ loop, enabling reposition behavior when the local LCS linearisation gets stuck on the wrong contact face.

> **Scope note.** The reference architecture from Bui 2026 uses Cartesian end-effector force as the MPC decision variable and a 1 kHz Operational-Space Controller (OSC) underneath. **This port keeps the 7-DOF joint-torque formulation** — it does not implement OSC. The trade-offs of that choice are documented in [Joint-Space Trade-offs](#joint-space-trade-offs).

---

## Status

This is an **active research port**, not a finished product. The infrastructure works end-to-end on the standard pushing scenarios, and the algorithmic core matches the paper math. See [Workflow: What's Built and What's Missing](#workflow-whats-built-and-whats-missing) for current state.

If you read the accompanying progress deck, this README maps onto the same story: slides 3–4 describe the DAIR lab reference architecture vs this port, and slides 5–6 describe the joint-space-vs-Cartesian trade-offs and the path-sequencing direction this port is pursuing.

---

## Quick Start

```bash
# Standard pushing (light box, 0.2 kg)
python main.py pushing

# Heavy box (1.5 kg, high friction)
python main.py hard_pushing

# Sphere shepherding
python main.py shepherding

# Save MP4 (requires ffmpeg)
python main.py pushing --save-video results/pushing.mp4
```

Meshcat visualisation opens at `http://127.0.0.1:7000`. Install ffmpeg via `conda install -c conda-forge ffmpeg` to use `--save-video`.

See [Appendix: CLI Reference](#appendix-cli-reference) for `--task-id`, `--sampling-c3`, and other flags.

---

## Architecture Overview

The control stack is bilevel: a global sample-based dispatcher selects between contact-rich and contact-free modes per timestep, and an inner ADMM-based MPC solves the local trajectory optimization.

```
                         ┌─────────────────────────┐
                         │  Sampling dispatcher    │   (Venkatesh 2025)
                         │  φ < 0.10 m gate        │
                         └───────────┬─────────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
              ┌──────────────────┐        ┌────────────────────────┐
              │  C3 / C3+ MPC    │        │ PiecewiseLinearTracker │
              │  inner solver    │        │  joint-PD + gravity    │
              │  contact-rich    │        │  contact-free          │
              └────────┬─────────┘        └──────────┬─────────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      ▼
                              ┌───────────────┐
                              │ τ ∈ ℝ⁷ joint  │
                              │ torque        │
                              └───────┬───────┘
                                      ▼
                              ┌───────────────┐
                              │  Drake plant  │
                              └───────────────┘
```

Both branches emit joint torques directly to the Drake plant; there is no OSC layer underneath. The contact force on the object is whatever the joint torques and the manipulator Jacobian produce: `F_EE = J⁻ᵀ(q) τ`.

---

## Algorithm: C3 and C3+

Both variants share the same Linear Complementarity System abstraction:

```
x[t+1] = A x[t] + B u[t] + D λ[t] + d
0 ≤ λ ⊥ E x + F λ + H u + c ≥ 0
```

The Stewart-Trinkle formulation packs `λ = [γ; λ_n; λ_t]` of dimension `6·n_c`: friction-cone slack `γ`, normal force `λ_n ≥ 0`, and a 4-edge polyhedral tangent basis `λ_t ≥ 0`. The MathematicalProgram is built **once** per control step; only the linear cost term `q` is refreshed each ADMM iteration via `UpdateCoefficients`.

### C3 (Aydinoglu 2024)

Decision variable `z = (x, λ, u)`. ADMM splits over a consensus set with three updates per iteration:

| Step | Variable | Operation |
| --- | --- | --- |
| z-update | `z` | OSQP QP (dynamics + `λ ≥ 0` Stewart-Trinkle bound + torque limits) |
| δ-update | `δ` | LCP projection per timestep: `δ_λ = solve_lcp(F, E·δ_x + H·δ_u + c)` (Aydinoglu §V-B.3.b) |
| ω-update | `ω` | Dual ascent `ω += z − δ` |

The δ-update produces feasible `λ` by construction (LCP solution satisfies `λ ≥ 0`, `Fλ + q ≥ 0`, `λᵀ(Fλ + q) = 0`). `x` and `u` blocks pass through unprojected. The soft-complementarity linear penalty used in earlier phases is disabled in the current build (`w_comp = 0`) — the LCP projection enforces complementarity exactly.

A Lorentz cone projection routine (`_project_single_contact`, `project_lorentz`) is retained in `admm_solver.py` for unit tests and for sandbox/2D problems where it remains the natural δ-update. It is not used on the main Drake path.

### C3+ (Bui 2026)

Adds a slack variable η so the decision variable becomes `z = (x, λ, u, η)` with a hard equality `η = Ex + Fλ + Hu + c` baked into the QP. The δ-update becomes a closed-form per-component projection on each scalar pair `(λ°, η°)` (Bui eq 12, with `r = √(u_λ/u_η)`, default 1):

| Case | Condition | δ_λ | δ_η |
| --- | --- | --- | --- |
| 1 | `η° ≥ 0` and `η° ≥ r·λ°` (η wins) | 0 | η° |
| 2 | `λ° ≥ 0` and `η° < r·λ°` (λ wins) | λ° | 0 |
| 3 | otherwise | 0 | 0 |

This makes the projection step roughly **4–5 orders of magnitude faster** than C3's cone projection (Bui Table III: 0.007 ms vs 10.4 ms for the 1-object case). The QP gains one row block and a few decision variables; the projection becomes nearly free.

### Sampling-C3 outer wrapper

Per control step the wrapper:

1. Builds a sample set: current EE pose plus N random samples on a horizontal circle around the object
2. Evaluates each sample with a cheap one-step C3 cost + alignment cost + travel cost
3. Picks the winner k*
4. Decides mode (with absolute or relative hysteresis):
   - **`c3` (rich) mode** — delegate to baseline inner C3/C3+ MPC
   - **`free` (reposition) mode** — execute piecewise-linear lift→traverse→descend toward target sample, tracked with joint-PD + gravity comp
5. Executes the chosen action

This unblocks tasks where the local LCS linearisation cannot escape the wrong contact face (e.g. the WEST directional task, where the EE must reposition to the opposite side of the box).

Enable with `--sampling-c3 [PATH.yaml]`. Default config: `config/sampling_c3_params.yaml`. See `control/sampling_c3/` and `docs/c3_math.md` for derivations.

---

## Joint-Space Trade-offs

The reference architecture (Bui 2026 / Yang 2024) uses **Cartesian end-effector force** as the MPC decision variable (`u ∈ ℝ³`) and a downstream OSC running at 1 kHz to convert force commands to joint torques. This port commits to **joint torque** as the decision variable (`u_arm ∈ ℝ⁷`) and skips the OSC layer. The contact force on the object is therefore implicit through the manipulator Jacobian:

```
F_EE = J⁻ᵀ(q) u_arm
```

This is a deliberate scope choice. OSC is a large secondary implementation effort and we want to first understand how far the joint-torque formulation can go on its own. Three known properties of `J⁻ᵀ` shape the behavior:

| Property | What it means for this port |
| --- | --- |
| **Configuration-dependent** — `J = J(q)`; same `u_arm` at different poses yields different `F_EE` in magnitude **and direction** | The MPC's planned push direction depends on the arm's current pose. Mitigated by the EE-approach cost in `QuadraticManipulationCost`, which biases the arm toward poses where the Jacobian is well-aligned with the goal direction. |
| **Ill-conditioned near singularities** — `σ_min(J) → 0` means `J⁻ᵀ` amplifies torques, so modest contact forces can demand large joint torques | Joint torques saturate at the 30 N·m limit when the arm operates near singular configurations. The sampling-C3 wrapper reduces exposure by repositioning the EE to higher-manipulability poses for contact-free segments. |
| **Non-square for 7-DOF arm** — `J ∈ ℝ^{m×7}` with m ≤ 6, so the implicit pseudoinverse maps differ from the paper's Q-space tuning | The cost surface in joint-torque space is distorted relative to the paper's F-space tuning. Worked around with empirical re-tuning of Q/R weights in `config/tasks.yaml`. |

The active mitigation strategy is **path-sequencing** — keeping each per-step goal small so the controller's transient torque demand stays within budget. See [Workflow](#workflow-whats-built-and-whats-missing) for what's implemented and what's still being investigated.

---

## Workflow: What's Built and What's Missing

### Built

**Plant and simulation**

- Drake `DiagramBuilder` environment: table + Franka Panda + spherical pusher (welded to `panda_link8` at +5 cm Z) + manipulable object (box or sphere)
- Meshcat visualisation + top-down MP4 export via matplotlib + ffmpeg
- Logger that mirrors stdout to timestamped files in `results/`
- YAML-driven task configuration in `config/tasks.yaml`; adding a new task requires zero Python changes

**LCS extraction**

- `lcs_formulator.py` reads contact geometry from Drake at each control step
- `ComputeSignedDistancePairwiseClosestPoints` filters EE-to-object pairs only (avoids 32–59 phantom contacts from self-collision)
- Normal and tangential Jacobians (`J_n`, `J_t`) computed live from `CalcJacobianTranslationalVelocity`
- Sign convention enforced so `J_n^T λ_n` always pushes the box away from the EE regardless of Drake's body-ordering choice
- First-order autodiff dynamics linearisation (Aydinoglu eq. 8) and Stewart-Trinkle complementarity slack assembly `(E, F, H, c)` (Aydinoglu eq. 9), shared by both C3 and C3+

**Inner MPC — both variants**

- **C3**: full-horizon ADMM with z/δ/ω updates, adaptive ρ (Boyd §3.4.1) every 10 iterations, LCP projection in the δ-update (Aydinoglu §V-B.3.b) via `lcp_solver.solve_lcp`
- **C3+**: slack-variable formulation with closed-form per-component δ-update (Bui eq 12)
- Shared infrastructure: MathematicalProgram built once per control step, only the linear cost term refreshed each ADMM iteration via `UpdateCoefficients`
- Joint torque clamp at ±30 N·m enforced both inside the QP and as a post-solve clip

**Cost function**

- `QuadraticManipulationCost`: LQR-style Q/R/QN tracking with object XY/Z/quaternion penalties
- 3-stage approach waypoint (`pre_approach → approach_waypoint → proxy`) that drives the arm onto the push axis before contact
- Jacobian-linearised proxy target injected into `Q[0:n_u, 0:n_u]` so the QP has a cost gradient even before contact is established
- Perpendicular box velocity penalty (discourages sideways drift)
- Lateral alignment correction (shifts effective proxy when EE is close but laterally offset)

**Outer wrapper**

- Sampling-C3 (Venkatesh 2025): sample evaluation + mode switching (`c3` rich / `free` reposition) + piecewise-linear lift→traverse→descend trajectories
- Mode hysteresis prevents thrashing between rich and free modes

**Tasks**

- Three core tasks (`pushing` / `hard_pushing` / `shepherding`) and four directional variants (`--task-id 1..4`) covering north/east/south/west goal directions

**Tooling**

- Unit tests for the projection step and sampling-C3 sub-systems (`tests/`)
- Profiling with `cProfile` + `SectionTimer` (`profiling/profile_run.py`)

### Missing / In Progress

**Path-sequencing for the contact-rich mode**

- The inner C3/C3+ MPC currently receives the goal pose directly as its planning target. When the goal requires a large `Δq` from the current pose, the controller commands transient torques near the saturation limit.
- Path-sequencing would interpose intermediate waypoints between the current pose and the goal so each planner cycle handles a small step. The sampling-C3 wrapper does this for the **contact-free** mode (`lift → traverse → descend`); the contact-rich mode does not yet have an equivalent.
- Open question for this direction: is the bottleneck transient torque (which sequencing addresses) or static torque to hold the goal pose (which it doesn't)? Resolving this requires measuring `g(q_target)` at the test scenarios' settle poses and comparing to the 30 N·m joint limit.

**Multi-object scenarios**

- Bui 2026's headline result is single-and-multi-object pushing. The current port handles single-object only.
- The LCS formulator's contact-pair filter needs extension to multi-pair cases. The δ-update already handles per-contact projection so the ADMM core itself is multi-object-ready.

**Cost-function tuning under joint-torque parameterization**

- The paper tunes Q assuming the decision variable is Cartesian force F. With joint torque `u_arm` as the decision variable, the cost surface is reshaped through the implicit `τ → F` mapping in a configuration-dependent way.
- Current Q/R weights are tuned empirically; a more principled approach (e.g., curvature-based weight selection at the operating pose) is open.

**Quantitative validation**

- The sampling-C3 wrapper hasn't been validated across the full directional sweep (NORTH / SOUTH / WEST). Initial results on the WEST task are encouraging; systematic comparison vs the cost-bias state machine (`--cost-bias`) is pending.

**Not on the roadmap**

- **OSC (Operational-Space Control).** We are not implementing the 1 kHz Cartesian-force tracker from Bui 2026 / Yang 2024. The joint-torque formulation is the chosen direction; if it proves insufficient, OSC remains a fallback option.

---

## Tasks

All task parameters live in `config/tasks.yaml`.

| Task | Object | Mass | μ | Goal (XY) |
| --- | --- | --- | --- | --- |
| `pushing` | box 10×10×10 cm | 0.2 kg | 0.4 | [0.30, 0.00] |
| `hard_pushing` | box 10×10×10 cm | 1.5 kg | 0.8 | [0.30, 0.00] |
| `shepherding` | sphere r=6 cm | 0.15 kg | 0.2 | [0.30, 0.00] |

Four directional variants of `pushing` are available via `--task-id {1,2,3,4}`:

| ID | Direction | Goal | Notes |
| --- | --- | --- | --- |
| 1 | north | [0.0, 0.3] | Push orthogonal to default contact normal |
| 2 | east | [0.3, 0.0] | Baseline: aligned with default contact normal |
| 3 | south | [0.0, −0.3] | Push orthogonal (opposite north) |
| 4 | west | [−0.3, 0.0] | Anti-aligned — requires contact mode switch (sampling-C3 recommended) |

---

## Repository Structure

```
push_anything_ADMM/
├── main.py                       # Entry point (argparse + sim loop + _Tee logger)
├── run_pushing.sh                # One-shot: pushing + MP4
├── config/
│   ├── tasks.yaml                # Task params: mass, friction, goal, cost weights
│   ├── directional_tasks.json    # Four fixed-goal pushing variants
│   ├── sampling_c3_params.yaml   # Sampling-C3 outer-controller params
│   └── sampling_c3_kik.yaml      # Sampling-C3 variant with IK-based reposition
├── sim/
│   ├── env_builder.py            # Drake DiagramBuilder: table + Panda + pusher + object
│   └── video_recorder.py         # Top-down MP4 via matplotlib + ffmpeg
├── control/
│   ├── lcs_formulator.py         # Drake plant → A, B, D, d, J_n, J_t, φ, μ, E, F, H, c
│   ├── admm_solver.py            # C3 / C3+ ADMM core (z/δ/ω updates, adaptive ρ)
│   ├── lcp_solver.py             # Stewart-Trinkle LCP solve for the C3 δ-update
│   ├── ci_mpc_c3.py              # C3 inner controller
│   ├── ci_mpc_c3plus.py          # C3+ inner controller (slack-variable formulation)
│   ├── task_costs.py             # QuadraticManipulationCost (Q/R/QN, 3-stage waypoint)
│   └── sampling_c3/              # Outer sampling-C3 wrapper (Venkatesh 2025)
├── tests/                        # pytest: projection + sampling-C3 + LCS/jacobian tests
├── docs/c3_math.md               # C3 math derivation reference
├── profiling/
│   ├── section_timer.py          # Zero-overhead per-section timer
│   └── profile_run.py            # cProfile + SectionTimer wrapper
├── scripts/                      # Diagnostic/sweep helpers (check_pose.py, probe_*.py, …)
└── models/drake_models/          # Franka Panda URDF + meshes
```

---

## Key Design Decisions

**LCS extraction from Drake.** Contact Jacobians `J_n, J_t` are computed live from `ComputeSignedDistancePairwiseClosestPoints` + `CalcJacobianTranslationalVelocity`, filtered to EE-to-object pairs only (avoids 32–59 phantom contacts from arm-link self-collisions). Sign convention: `J_n = nhat_BA_W · (J_A − J_B)` so `J_n^T λ_n` always pushes the box away from the EE regardless of which body Drake assigns to A vs B.

**Spherical pusher** (`PUSHER_RADIUS = 0.025 m`) is welded to `panda_link8` at +5 cm Z. No new DOF — the arm stays 7-DOF. Single registered collision geometry; a strict assertion at startup prevents silent regression to phantom-contact behavior.

**Dedicated EE-approach cost.** When `D ≈ 0` (no contact) the QP would minimize `u^T R u → 0` and freeze the arm. The Jacobian-linearized proxy-target term in `QuadraticManipulationCost` shifts `x_ref[0:n_u]` toward a 3-stage waypoint behind the object, providing a cost gradient even before contact is established.

**LCP δ-update (C3, Aydinoglu §V-B.3.b).** At each ADMM iteration, the δ-update solves `solve_lcp(F, q_lcp)` per timestep with `q_lcp = E·δ_x + H·δ_u + c`. The LCP solution produces feasible `(γ, λ_n, λ_t)` by construction (`λ ≥ 0`, `Fλ + q ≥ 0`, complementarity slackness). This replaces the earlier Lorentz-cone projection path, which is retained in code (`project_lorentz`) for unit-test coverage of the standalone cone projector but is no longer on the main Drake path.

**Per-component δ-update (C3+).** Three closed-form cases per scalar `(λ°, η°)` pair (Bui eq 12) — no Cartesian projection needed. Roughly 4–5 orders of magnitude faster than a cone projection.

**YAML-driven tasks.** Every task-specific number lives in `config/tasks.yaml`. Adding a new task requires zero Python changes beyond the argparse choices list in `main.py`.

---

## Cost Function (`QuadraticManipulationCost`)

LQR-style quadratic tracking cost. All weights from `tasks.yaml`:

- **Q** — diagonal; penalises object XY position error (`w_obj_xy`), Z height, roll/pitch quaternion (qx, qy — keeps box upright; qz/yaw left free)
- **R** — `w_torque · I_nu` joint torque penalty
- **QN** — `w_terminal · Q` terminal weight
- **x_ref** — zero everywhere except object position set to `[goal_x, goal_y, z_ee_target]`

```
                 goal ★
                  ▲
                  │  g_hat (push direction)
          ┌───────┼───────┐
          │       ●       │   ← object
          └───────────────┘
                  │
              proxy y_ref     ← obj_pos − d_push · g_hat
                  │
                 EE           ← arm tracks proxy via Jacobian-linearised cost
```

**Linearised EE approach + 3-stage waypoint.** When `plant_ctx` is supplied to `build()`, a second cost term drives the arm toward the proxy contact point behind the object:

- Stage 1 (`> 0.25 m`): target `pre_approach = obj_pos − 0.16 · g_hat` — get on the push axis first
- Stage 2 (`0.10–0.25 m`): blend `pre_approach → approach_waypoint` (= `obj_pos − (d_push + 0.15) · g_hat`)
- Stage 3 (`< 0.10 m`): blend `approach_waypoint → proxy`

Augments `Q[0:n_u, 0:n_u] += 2·w_ee_approach · J_arm^T J_arm` and shifts `x_ref[0:n_u]` toward the effective proxy via damped pseudoinverse IK (λ = 0.001).

**Perpendicular box velocity penalty.** Adds `(v_box · g_perp)^2` to Q with weight `10 · w_obj_xy` (where `g_perp` = 90° CCW from `g_hat`). Discourages sideways drift during a push.

**Lateral alignment correction.** When EE is close (`< 0.15 m`) and laterally offset, shifts the effective proxy back onto the push axis weighted by perpendicular magnitude.

---

## MPC Parameters

| Parameter | Value | Notes |
| --- | --- | --- |
| `horizon` | 20 | 20 × 0.05 s = 1.0 s lookahead |
| `admm_iter` | 3 | Per control step (adaptive ρ rarely fires at this depth; override with `--admm-iter`) |
| `dt` | 0.05 s | Planning timestep |
| `rho` | 100.0 | Initial ADMM penalty (adaptive within solve) |
| `torque_limit` | ±30 N·m | Hard clamp (in QP + final clip) |
| `dt_ctrl` | 0.01 s | Real-sim control rate |
| `max_time` | 8.0 s | Simulation duration (override with `--max-time`) |

---

## Profiling

```bash
python profiling/profile_run.py pushing 5
# → profiling/results/pushing_cprofile.prof  (snakeviz-compatible)
# → profiling/results/pushing_sections.txt   (per-section wall time)
```

Section names: `lcs.extract_dynamics`, `lcs.geometry_query`, `lcs.calc_jacobians`, `admm.qp_build`, `admm.osqp_solve`, `admm.z_update`.

---

## Dependencies

| Package | Purpose |
| --- | --- |
| `pydrake` | Physics engine, solver, visualiser (install separately) |
| `numpy` | Numerics |
| `pyyaml` | Task + sampling-C3 config loading |
| `matplotlib` + `ffmpeg` | Video rendering (only with `--save-video`) |
| `pytest` | Unit tests in `tests/` (most tests touch pydrake; `test_progress.py`, `test_mode_switch.py`, `test_inner_solve.py`, `test_sample_buffer.py`, `test_sampling_c3_params.py`, `test_sampling_strategies.py`, `test_projection.py`, `test_reposition.py` exercise pure-numpy logic) |

---

## Citations

H. Bui et al., **"Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC,"** *arXiv:2510.19974v2*, 2025.

A. Aydinoglu, A. Wei, W.-C. Huang, and M. Posa, **"Consensus Complementarity Control for Multi-Contact MPC,"** *IEEE Transactions on Robotics*, vol. 40, pp. 3879–3896, 2024.

S. Venkatesh, B. Bianchini, A. Aydinoglu, W. Yang, and M. Posa, **"Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity,"** *RA-L 2025*, arXiv:2505.13350.

W. Heemels, J. M. Schumacher, and S. Weiland, **"Linear Complementarity Systems,"** *SIAM Journal on Applied Mathematics*, vol. 60, no. 4, pp. 1234–1269, 2000.

---

## Appendix: CLI Reference

### Extended CLI examples

```bash
python main.py hard_pushing                          # heavy box task, Meshcat only
python main.py shepherding --save-video              # save results/shepherding_<ts>.mp4
python main.py pushing --prepositioned               # start pusher already touching box
python main.py pushing --task-id 1                   # directional task: push north
python main.py pushing --video-path results/foo.html # save Meshcat HTML replay
python main.py pushing --max-time 4.0                # override sim duration
python main.py pushing --math-diag                   # verbose [MATH.*] solver diagnostics
python main.py pushing --cost-bias                   # face-transition cost bias (PUSH/LIFT/APPROACH)
python main.py pushing --task-id 4 --sampling-c3     # WEST task with sampling-C3 outer controller
python main.py pushing --solver c3plus               # use the C3+ slack-variable inner solver
python main.py pushing --admm-iter 10                # let adaptive ρ fire (every 10 iters)
python main.py pushing --name baseline_v3            # shared <stem> for .txt/.mp4/.html outputs
python main.py pushing --no-record                   # smoke test (no MP4, no HTML)
```

### CLI flags

| Flag | Description |
| --- | --- |
| `--save-video [OUT.mp4]` | Top-down MP4. No arg → `results/<stem>.mp4` |
| `--video-path [PATH.html]` | Meshcat HTML replay. No arg → `results/<stem>.html` |
| `--no-record` | Disable both MP4 and HTML output (overrides above) |
| `--name BASENAME` | Shared basename for `<stem>.txt/.mp4/.html` in `results/` (default: `<task>_<timestamp>`) |
| `--reset-every N` | Reset arm + object every N control steps |
| `--prepositioned` | Start arm touching box west face |
| `--task-id {1,2,3,4}` | Override `goal_xy` from `config/directional_tasks.json` |
| `--goal-xy X,Y` | In-memory override of `task_cfg['goal_xy']` (sweep helper) |
| `--max-time T` | Override sim duration in seconds (default 8.0) |
| `--math-diag` | Verbose `[MATH.*]` solver diagnostics |
| `--admm-iter N` | ADMM iterations per control step (default 3; ≥ 10 enables adaptive-ρ) |
| `--solver {c3,c3plus}` | Inner ADMM solver (default `c3`) |
| `--cost-bias` | Face-transition cost-bias state machine. Mutually exclusive with `--sampling-c3` |
| `--sampling-c3 [PATH.yaml]` | Wrap inner MPC with sampling outer controller. Default: `config/sampling_c3_params.yaml` |
| `--workspace-y-max YMAX` | F3 sweep override: in-memory `sampling_params.workspace_xy_max[1]` (only with `--sampling-c3`) |
| `--seed INT` | Sweep helper: seed the SamplingC3MPC rng for deterministic sampling angles |
