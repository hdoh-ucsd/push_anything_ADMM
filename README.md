# Push Anything ADMM

A **C3 Contact-Implicit Model Predictive Control (CI-MPC)** framework in PyDrake. A 7-DOF Franka Emika Panda arm manipulates objects via non-prehensile pushing by linearising the Drake plant into a discrete LCS at each control step and solving the full N-step trajectory as one stacked QP with ADMM, alternating between a QP subproblem (dynamics) and a Lorentz cone projection (contact constraints). An optional outer **sampling-C3** wrapper (Venkatesh 2025) layers global sample-based contact-mode reasoning on top of the local C3 inner loop.

---

## Quick Start

```bash
# Standard pushing (light box, 0.2 kg)
python main.py pushing

# Heavy box (1.5 kg, high friction)
python main.py hard_pushing

# Sphere shepherding
python main.py shepherding

# Any task with MP4 output
python main.py pushing --save-video results/pushing.mp4
```

Meshcat visualisation opens at `http://127.0.0.1:7000`.
Requires `ffmpeg` for `--save-video` (`conda install -c conda-forge ffmpeg`).

See [Appendix: Directional Tasks & CLI Reference](#appendix-directional-tasks--cli-reference) for `--task-id`, `--sampling-c3`, and other flags.

---

## Algorithm: C3

C3 (Consensus Complementarity Control; Aydinoglu et al. 2022/2024) solves the contact-implicit trajectory optimisation problem over an N-step horizon by linearising the Drake plant at the current state into a discrete Linear Complementarity System (LCS):

```
x[t+1] = A x[t] + B_ctrl u[t] + D λ[t] + d
λ_n ≥ 0
‖λ_t‖ ≤ μ λ_n          (friction cone)
```

The full-horizon problem is split via ADMM consensus over `z = [x₀, λ₀, u₀, x₁, λ₁, u₁, …, x_N]`:

| Step | Variable | Operation |
|---|---|---|
| z-update | `z` | OSQP QP (dynamics + `λ_n ≥ 0` + ±torque limits + soft complementarity) |
| δ-update | `δ` | Lorentz cone projection per contact, per horizon step |
| ω-update | `ω` | Dual ascent `ω += z − δ` |

The MathematicalProgram is built **once** per control step; only the linear cost term `q` is refreshed each ADMM iteration via `UpdateCoefficients`. Adaptive ρ (Boyd §3.4.1) every 10 iterations, early termination on `primal_res < 1e-3 ∧ dual_res < 1e-3`, and a soft-complementarity penalty (`w_comp = 100`) bias the QP away from planning contact forces when the gap is open.

The 4-edge polyhedral tangent basis `{t₁, −t₁, t₂, −t₂}` is converted to 2D Cartesian for projection and split back, correctly enforcing the **true** Lorentz friction cone (not just the polyhedral inner approximation).

### Sampling-C3 (optional outer wrapper)

Ports the §IV-D outer sampling controller from Venkatesh et al. 2025. A two-mode controller wraps `C3MPC`:

- **`c3` (rich) mode** — delegate to baseline `C3MPC`.
- **`free` (reposition) mode** — pick a target sample, run a piecewise-linear lift→traverse→descend trajectory, track with joint-PD + gravity comp.

Per control step: build sample set (current EE + random samples on a circle), evaluate each via cheap C3 + alignment + travel cost, pick winner k\*, decide mode (with absolute or relative hysteresis), execute. Helps on tasks where the local LCS linearisation cannot escape the wrong contact face (e.g. WEST directional task).

Enable with `--sampling-c3 [PATH.yaml]`. Default config: `config/sampling_c3_params.yaml`. See `control/sampling_c3/` and `docs/c3_math.md` for details.

---

## Tasks

All task parameters live in `config/tasks.yaml`.

| Task | Object | Mass | μ | Goal (XY) |
|---|---|---|---|---|
| `pushing` | box 10×10×10 cm | 0.2 kg | 0.4 | [0.30, 0.00] |
| `hard_pushing` | box 10×10×10 cm | 1.5 kg | 0.8 | [0.30, 0.00] |
| `shepherding` | sphere r=6 cm | 0.15 kg | 0.2 | [0.30, 0.00] |

---

## Repository Structure

```
push_anything_ADMM/
├── main.py                       # Entry point (argparse + sim loop + _Tee logger)
├── run_pushing.sh                # One-shot: pushing + MP4
├── config/
│   ├── tasks.yaml                # Task params: mass, friction, goal, cost weights
│   ├── directional_tasks.json    # Four fixed-goal pushing variants (--task-id 1..4)
│   └── sampling_c3_params.yaml   # Sampling-C3 outer-controller params
├── sim/
│   ├── env_builder.py            # Drake DiagramBuilder: table + Panda + pusher + object
│   └── video_recorder.py         # Top-down MP4 via matplotlib + ffmpeg
├── control/
│   ├── lcs_formulator.py         # Drake plant → A, B_ctrl, D, d, J_n, J_t, φ, μ
│   ├── admm_solver.py            # C3 ADMM core (z/δ/ω updates, adaptive ρ, soft comp)
│   ├── ci_mpc_c3.py              # Active inner controller (full-horizon C3)
│   ├── ci_mpc_c3plus.py          # Alternative MPPI-style C3+ (research alt, not wired)
│   ├── base_mpc.py               # Rollout engine for C3+ variants only
│   ├── task_costs.py             # QuadraticManipulationCost (Q/R/QN, 3-stage waypoint)
│   └── sampling_c3/              # Outer sampling-C3 wrapper (Venkatesh 2025)
├── tests/                        # pytest: projection + sampling-C3 sub-systems
├── docs/c3_math.md               # Auto-generated C3 math derivation
├── milestones/                   # Frozen Drake-port snapshots (RESULTS.md per milestone)
├── profiling/
│   ├── section_timer.py          # Zero-overhead per-section timer
│   └── profile_run.py            # cProfile + SectionTimer wrapper
├── scripts/check_pose.py         # FK diagnostic for arm-pose candidates
└── models/drake_models/          # Franka Panda URDF + meshes
```

---

## Key Design Decisions

**LCS extraction from Drake** — Contact Jacobians `J_n, J_t` are computed live from `ComputeSignedDistancePairwiseClosestPoints` + `CalcJacobianTranslationalVelocity`, filtered to EE-to-object pairs only (avoids 26–59 phantom contacts). Sign convention: `J_n = nhat_BA_W · (J_A − J_B)` so `J_n^T λ_n` always pushes the box away from the EE regardless of which body Drake assigns to A vs B.

**Spherical pusher** (`PUSHER_RADIUS = 0.025 m`) welded to `panda_link8` at +5 cm z. No new DOF — arm stays 7-DOF. Single registered collision geometry; a strict assertion at startup prevents silent regression to phantom-contact behaviour.

**Dedicated EE approach cost** — When `D ≈ 0` (no contact) the QP would minimise `u^T R u → 0` and freeze the arm. The Jacobian-linearised proxy-target term in `QuadraticManipulationCost` shifts `x_ref[0:n_u]` toward a 3-stage waypoint behind the object, providing a cost gradient even before contact is established.

**Lorentz cone δ-update** — Closed-form per contact: inside cone (return unchanged), polar cone (project to apex), surface (`s = (λ_n + μ‖F_t‖)/(1+μ²)`). Tangents stored as a 4-edge polyhedral pyramid but projected in 2D Cartesian — the true cone is enforced, not just its polyhedral inner approximation.

**YAML-driven tasks** — Every task-specific number lives in `config/tasks.yaml`. Adding a new task requires zero Python changes beyond the argparse choices list in `main.py`.

---

## MPC Parameters

| Parameter | Value | Notes |
|---|---|---|
| `horizon` | 20 | 20 × 0.05 s = 1.0 s lookahead |
| `admm_iter` | 3 | Per control step (adaptive ρ rarely fires at this depth) |
| `dt` | 0.05 s | Planning timestep |
| `rho` | 100.0 | Initial ADMM penalty (adaptive within solve) |
| `torque_limit` | ±30 Nm | Hard clamp (in QP + final clip) |
| `dt_ctrl` | 0.01 s | Real-sim control rate |
| `max_time` | 8.0 s | Simulation duration (override with `--max-time`) |

---

## Cost Function (`QuadraticManipulationCost`)

LQR-style quadratic tracking cost. All weights from `tasks.yaml`:

- **Q** — diagonal, penalises object XY position error (`w_obj_xy`), Z height, roll/pitch quaternion (qx, qy — keeps box upright; qz/yaw left free)
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

**Linearised EE approach + 3-stage waypoint** — When `plant_ctx` is supplied to `build()`, a second cost term drives the arm toward the proxy contact point behind the object:
- Stage 1 (`> 0.25 m`): target `pre_approach = obj_pos − 0.18 · g_hat` — get on the push axis first
- Stage 2 (`0.10–0.25 m`): blend `pre_approach → approach_waypoint` (= `obj_pos − (d_push + 0.15) · g_hat`)
- Stage 3 (`< 0.10 m`): blend `approach_waypoint → proxy`

Augments `Q[0:n_u, 0:n_u] += 2·w_ee · J_arm^T J_arm` and shifts `x_ref[0:n_u]` toward the effective proxy via damped pseudoinverse IK (λ = 0.001).

**Perpendicular box velocity penalty** — adds `(v_box · g_perp)^2` to Q with weight `10 · w_obj_xy` (where `g_perp` = 90° CCW from `g_hat`). Discourages sideways drift during a push.

**Lateral alignment correction** — when EE is close (`< 0.15 m`) and laterally offset, shifts the effective proxy back onto the push axis weighted by perpendicular magnitude.

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
|---|---|
| `pydrake` | Physics engine, solver, visualiser (install separately) |
| `numpy` | Numerics |
| `pyyaml` | Task + sampling-C3 config loading |
| `matplotlib` + `ffmpeg` | Video rendering (only with `--save-video`) |
| `pytest` | Unit tests in `tests/` (pure-numpy except `test_inner_solve.py`) |

---

## Status & Roadmap

**Done**
- Drake environment with box and sphere objects; spherical pusher welded to `panda_link8`
- LCS extraction via `ComputeSignedDistancePairwiseClosestPoints` with EE-to-object contact filtering
- Full-horizon C3 ADMM: stacked QP (built once, `q` refreshed each iter), Lorentz cone δ-update on 4-edge polyhedral tangents, adaptive ρ, soft complementarity
- `QuadraticManipulationCost` with 3-stage approach waypoint, lateral alignment correction, perpendicular-velocity penalty
- Three tasks (pushing / hard_pushing / shepherding) parameterised via YAML
- Four directional pushing variants (`config/directional_tasks.json`, `--task-id 1..4`)
- Sampling-C3 outer wrapper (Venkatesh 2025) for tasks where local C3 cannot escape wrong contact face
- Top-down MP4 + Meshcat HTML replay; `_Tee` logger mirrors stdout to `results/<task>_<ts>.txt`
- Two milestones: infrastructure-verified, directional characterisation (see `milestones/INDEX.md`)

**Next Steps**
- Real-time profiling — identify the dominant per-step bottleneck experimentally
- Validate sampling-C3 across the full directional sweep (NORTH / SOUTH / WEST)
- Multi-object scenarios

---

## Citations

H. Bui et al., **"Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC,"** *arXiv:2510.19974v2*, 2025.

A. Aydinoglu, A. Wei, W.-C. Huang, and M. Posa, **"Consensus Complementarity Control for Multi-Contact MPC,"** *IEEE Transactions on Robotics*, vol. 40, pp. 3879–3896, 2024.

S. Venkatesh, B. Bianchini, A. Aydinoglu, W. Yang, and M. Posa, **"Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity,"** *RA-L 2025*, arXiv:2505.13350.

W. Yang and W. Jin, **"ContactSDF: Signed Distance Functions as Multi-Contact Models for Dexterous Manipulation,"** *arXiv:2408.09612v2*, 2024.

W. Heemels, J. M. Schumacher, and S. Weiland, **"Linear Complementarity Systems,"** *SIAM Journal on Applied Mathematics*, vol. 60, no. 4, pp. 1234–1269, 2000.

---

## Appendix: Directional Tasks & CLI Reference

### Directional pushing variants

Four fixed-goal variants of `pushing`, selected via `--task-id {1,2,3,4}`. Override `goal_xy` at runtime; all other parameters from `tasks.yaml: pushing`.

| ID | Name | Goal | Description |
|---|---|---|---|
| 1 | north | [0.0, 0.3] | Push orthogonal to EE contact normal |
| 2 | east | [0.3, 0.0] | Baseline: contact normal aligned with goal |
| 3 | south | [0.0, −0.3] | Push orthogonal (opposite north) |
| 4 | west | [−0.3, 0.0] | Contact normal anti-aligned — requires contact mode switch |

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
python main.py pushing --no-record                   # smoke test (no MP4, no HTML)
```

### CLI flags

| Flag | Description |
|---|---|
| `--save-video [OUT.mp4]` | Top-down MP4. No arg → `results/<task>_<ts>.mp4` |
| `--video-path [PATH.html]` | Meshcat HTML replay. No arg → `results/<task>_<ts>.html` |
| `--no-record` | Disable both MP4 and HTML output (overrides above) |
| `--reset-every N` | Reset arm + object every N control steps |
| `--prepositioned` | Start arm touching box west face |
| `--task-id {1,2,3,4}` | Override `goal_xy` from `config/directional_tasks.json` |
| `--max-time T` | Override sim duration in seconds (default 8.0) |
| `--math-diag` | Verbose `[MATH.*]` solver diagnostics |
| `--cost-bias` | Face-transition cost-bias state machine. Mutually exclusive with `--sampling-c3` |
| `--sampling-c3 [PATH.yaml]` | Wrap `C3MPC` with sampling outer controller. Default: `config/sampling_c3_params.yaml` |
