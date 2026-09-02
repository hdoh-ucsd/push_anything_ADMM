# Push Anything / C3+ Reproduction

A Python/PyDrake reproduction and research port of *Push Anything*: sampling-based
contact-implicit model predictive control for non-prehensile manipulation with a
Franka Panda. The repository combines local Linear Complementarity System (LCS)
models, C3/C3+ trajectory optimization, candidate contact placement, and
operational-space execution in simulation.

> **Status:** active research code. The planning and simulation stack runs
> end-to-end, but this is not a claim of full paper-level reproduction. Stored
> benchmarks include successful and censored trials, and the current test
> baseline is not fully green.

![System architecture: sampling, local LCS construction, C3/C3+ MPC, OSC, and PyDrake simulation](docs/figures/system_architecture.svg)

## Overview

The implementation follows the central decomposition used by *Push Anything*
and sampling-C3:

- **PyDrake plant:** a Franka Panda with a spherical pusher interacts with
  configurable rigid objects and a table.
- **Candidate sampling:** the outer controller generates possible end-effector
  placements around the current object geometry.
- **Local contact models:** each relevant state/candidate is linearized into an
  LCS with dynamics and complementarity matrices.
- **C3+ / C3 MPC:** the default C3+ path solves a finite-horizon contact-implicit
  problem; C3 remains available as a comparison/falsification path.
- **Mode selection:** candidate objectives and progress logic select either a
  contact-rich MPC trajectory or a contact-free reposition trajectory.
- **Execution:** an operational-space controller tracks the selected trajectory
  at a faster inner cadence and applies torques to the Franka simulation.

The current default planner uses the repository's reduced end-effector-space
formulation (`x ∈ R^19`, `u ∈ R^3` Cartesian force). The older full-plant
joint-torque formulation remains behind `--r7` for historical falsification
runs; it is not the default architecture.

## Experiments

The measured experiment reports are organized into two tracks:

- **3D Jack Manipulation:** full SE(3) translation and support-tripod
  reorientation with the Jack object.
- **Single-Object Pushing:** fixed-goal planar pushing for imported objects and
  the T-shaped benchmark.

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for commands, measured outcomes,
artifacts, and reporting limitations.

## Current Research

![Research roadmap separating current implementation, active investigations, and planned work](docs/figures/research_roadmap.svg)

The maintained implementation covers the Push Anything reproduction stack,
planar single-object pushing, C3/C3+, imported object geometries, and an
experimental jack task with full orientation goals. Current research directions
include a **CRISP comparison study** and investigation of **continuous contact
location**. There is no CRISP implementation in this repository.

General 3D/SE(3) non-prehensile manipulation, broader cube-object studies, and
GPU acceleration are roadmap items, not completed capabilities or benchmark
claims. Existing experimental task/configuration branches should not be read as
validated general-purpose 3D manipulation.

## Method

![C3+ solver flow: local LCS, stacked QP, ADMM updates, and MPC action](docs/figures/c3plus_solver_flow.svg)

For each local candidate, `LCSFormulator` produces discrete dynamics

```text
x[t+1] = A x[t] + B u[t] + D λ[t] + d
```

and complementarity data based on the current contact geometry. C3+ introduces
the slack

```text
η[t] = E x[t] + F λ[t] + H u[t] + c,
0 ≤ λ[t] ⟂ η[t] ≥ 0.
```

`C3Solver` then alternates:

1. a global constrained-QP update;
2. the C3+ componentwise `(λ, η)` projection;
3. consensus/dual and penalty updates;
4. a final QP/trajectory extraction.

The candidate objective feeds the sampling-C3 dispatcher. As in receding-horizon
MPC, only the first execution interval is applied before the state and local
contact model are refreshed. See `control/admm_solver.py`,
`control/lcs_formulator.py`, `control/ci_mpc_c3plus.py`, and
`control/sampling_c3/` for the implementation.

## Single-Object Pushing

The corrected protocol uses one uninterrupted manipulation session per object:
the robot, object, controller state, and random-goal stream carry over across all
28 goals. Each goal has its own 600-second simulated-time limit. A session passes
only if it reaches all 28 goals consecutively; failed sessions are not replaced
with another seed.

The latest-model campaign reran 23 objects on August 28, 2026 and retained the
requested prior outcomes for Eraser and Gallon Milk. The combined result is
**17 passing sessions out of 25 objects**. Full current results and provenance
are in [`FIG8_CONSECUTIVE_28_LATEST_SESSIONS.csv`](FIG8_CONSECUTIVE_28_LATEST_SESSIONS.csv).

| Object | Consecutive goals | Outcome | Failure cause |
|---|---:|:---:|---|
| Letter I | 28/28 | Pass | — |
| Letter C | 28/28 | Pass | — |
| Letter R | 28/28 | Pass | — |
| Letter A | 28/28 | Pass | — |
| Letter Y | 4/28 | Fail | Near-success timeout: 0.0022 m position error and 0.1078 rad orientation error |
| Letter G | 28/28 | Pass | — |
| Letter B | 28/28 | Pass | — |
| Letter 3 | 28/28 | Pass | — |
| Letter H | 28/28 | Pass | — |
| Letter E | 4/28 | Fail | No-contact recovery loop; final errors 0.0263 m and 0.0861 rad |
| Letter S | 28/28 | Pass | — |
| Expo Box | 28/28 | Pass | — |
| Lotion | 28/28 | Pass | — |
| Wood Block | 24/28 | Fail | Planner/reposition stall; final errors 0.0840 m and 1.5641 rad |
| Tape | 2/28 | Fail | Near-success timeout: 0.0154 m and 0.1155 rad, followed by no-contact retries |
| Eraser | 0/28 | Fail | Persistent topple (retained) |
| Milk Bottle | 5/28 | Fail | No-contact recovery loop; final position error 0.1808 m |
| Clamp | 28/28 | Pass | — |
| Chicken Broth | 28/28 | Pass | — |
| Egg Carton | 3/28 | Fail | Inner workspace limit: EE radius 0.2787 m (minimum 0.280 m) |
| Book | 28/28 | Pass | — |
| Baby Toy | 28/28 | Pass | — |
| Gallon Milk | 0/28 | Fail | Outer workspace limit: EE radius 0.7543 m (maximum 0.750 m; retained) |
| Xbox | 28/28 | Pass | — |
| Push T | 28/28 | Pass | — |

The success gate remains reference-conformant: position error must be below
0.02 m and quaternion geodesic orientation error below 0.10 rad simultaneously.
The primary actionable defect is a no-contact recovery loop: after an
unproductive C3 segment, the dispatcher can reposition and select an equivalent
ineffective contact repeatedly until timeout. Candidate failures should be
invalidated, accepted reposition samples should predict contact closure and a
nontrivial force, and repeated retries should force a fresh global sample set.
Workspace targets and their tracked trajectories also need a 5–10 mm safety
margin. Eraser requires separate non-planar topple recovery.

## Object-Informed-Manipulation

The job is to fetch the five tabletop scenarios from the external OIM reference,
preserve their xArm model, scene assets, start/goal poses, and evaluation
protocol, then run each scenario with our C3+ controller and publish comparable
result artifacts. C3+ replans from the measured xArm and object state after each
executed control interval until the OIM goal gate passes or the run budget ends.

![OIM reference scenarios imported into the xArm C3+ evaluation pipeline](docs/figures/oim_c3plus_scenario_pipeline.svg)

### Native C++ integration: `oim_c++_anything`

The native DAIRLab integration is isolated from the modified reference checkout
in a clean Git worktree:

- worktree: `reference_repos/oim_c++_anything`
- branch: `oim_c++_anything`
- baseline/design commit: `854a8afc`
- canonical task configuration:
  `examples/sampling_c3/oim_t/parameters/oim_t.yaml`
- maintained process diagram and validation gates:
  `examples/sampling_c3/oim_t/ARCHITECTURE.md`

The intended native process topology is:

```text
oim_t.yaml
 ├── xarm6_sim
 ├── xarm6_osc_controller
 └── xarm6_sampling_c3_controller
```

#### DAIRLab/Drake-to-xArm6 file correspondence

"Reference" below means the DAIRLab Sampling-C3 Franka/Push-T implementation
built on Drake, not an upstream Drake example. A row marked **reused** calls the
same implementation from the xArm6 targets; **adapted** identifies the xArm6
counterpart; **consolidated** records several reference configuration inputs in
the single canonical OIM YAML. This is an interface correspondence, not a claim
that different robot models or physics engines are numerically identical.

| DAIRLab/Drake reference | OIM xArm6 counterpart | Relationship |
|---|---|---|
| `examples/sampling_c3/BUILD.bazel` | `examples/sampling_c3/oim_t/BUILD.bazel` | **Adapted:** defines the three canonical `xarm6_*` processes and compatibility aliases without changing the Franka targets. |
| `examples/sampling_c3/franka_sim.cc` | `examples/sampling_c3/oim_t/xarm6_sim.cc` | **Adapted:** Drake plant/simulator and LCM state-command loop for the six-joint xArm, table, pusher capsule, and OIM T. |
| `examples/sampling_c3/franka_osc_controller.cc` | `examples/sampling_c3/oim_t/xarm6_osc_controller.cc` | **Adapted:** the same DAIRLab inverse-dynamics OSC architecture, with xArm6 frames, six-joint posture, passive spring, gravity, and source velocity-servo semantics. |
| `examples/sampling_c3/franka_sampling_c3_controller.cc` | `examples/sampling_c3/oim_t/xarm6_sampling_c3_controller.cc` | **Adapted:** live-state Sampling-C3+ orchestration, contact acquisition, receding execution, physical-response conditioning, and acceptance gates. |
| `systems/controllers/sampling_based_c3_controller.{cc,h}` | `examples/sampling_c3/oim_t/xarm6_full_sampling_c3plus.{cc,h}` | **Specialized wrapper:** preserves the C3/C3+ trajectory contract while adding the 19-state OIM T model, sampled contacts, xArm6 execution receipts, and safety predicates. |
| `systems/controllers/osc/operational_space_control.{cc,h}` and its tracking-data classes | Same `systems/controllers/osc/*` files | **Reused:** xArm6 links DAIRLab's OSC QP directly; there is no copied xArm-specific OSC solver. |
| `examples/sampling_c3/sampling_c3_utils.{cc,h}` and `@c3//` | Same utility and C3 library targets | **Reused:** trajectory encoding, LCS/C3 interfaces, and consensus/projection implementation remain shared. |
| `examples/sampling_c3/parameter_headers/{franka_sim_params,goal_params,sampling_c3_controller_params}.h` | `examples/sampling_c3/parameter_headers/oim_t_params.h` | **Consolidated:** typed Drake YAML schema for robot, simulation, object, task, controller, full Sampling-C3+, and LCM sections. |
| `examples/sampling_c3/push_t/parameters/{sim_params,goal_params,sampling_c3_controller_params}.yaml` | `examples/sampling_c3/oim_t/parameters/oim_t.yaml` | **Consolidated:** one authoritative OIM scenario file replaces the three task-level YAML inputs. |
| `push_t/parameters/{sampling_c3_options,sampling_c3plus_options,sampling_params,reposition_params}.yaml` | `oim_t.yaml` sections `full_sampling_c3plus` and `controller` | **Consolidated with provenance:** structural solver, sampling, and reposition values are explicit; unchanged values remain identified as Push-T/Anything carryovers. |
| `@drake_models//:franka_description` | `examples/sampling_c3/urdf/oim_xarm6_tabletop/xarm6/xarm6.xml` plus `xarm6_lcs_pusher.urdf` | **Model replacement:** six xArm joints and the physical pusher capsule replace the seven-joint Franka description. |
| `examples/sampling_c3/urdf/push_t.sdf` | `examples/sampling_c3/urdf/oim_xarm6_tabletop/t_block.sdf` | **Object replacement:** the OIM two-box T has its own measured geometry, mass, inertia, and collision representation. |
| Reference executable smoke coverage | `oim_t_config_check.cc` and `test/{xarm6_open_table_test,xarm6_full_sampling_c3plus_test}.cc` | **New validation:** checks the consolidated contract, six-joint plant, spatial Sampling-C3+ invariants, and open-table model loading. |

`oim_t.yaml` replaces the legacy task-level composition through
`sim_params.yaml`, `goal_params.yaml`, and
`sampling_c3_controller_params.yaml`. It owns the xArm model contract, OIM T
start and goal in the unwarped `oim_world` frame, simulation timing, success
tolerances, and LCM routing. Algorithm-specific C3+, sampling, repositioning,
progress, and OSC parameter files remain separate until their numerical
provenance has been validated for xArm.

### Task: `open_table`

An xArm6 with a vertical pushing stick must push a planar T block across an
open table from its start pose to a goal pose. The goal variables are the
object's planar SE(2) pose in the `oim_world` frame
(`examples/sampling_c3/oim_t/parameters/oim_t.yaml` in the C++ worktree):

```text
g = (x_g, y_g, θ_g) = (0.381, -0.400, 3.1416)      # object.goal_pose
x^o = (x, y, θ)                                     # measured T pose (yaw from quaternion)
```

Success is a terminal tolerance check on both goal variables simultaneously
(`task.translation_tolerance`, `task.orientation_tolerance`):

```text
e_p = || (x, y) - (x_g, y_g) ||_2         <  0.05 m
e_θ = | wrap(θ - θ_g) |                   <  0.10 rad,   wrap(a) = atan2(sin a, cos a)
```

The yaw is extracted from the measured quaternion and wrapped exactly as in
`EvaluateXarmFullSamplingC3PlanarSettle`
(`xarm6_full_sampling_c3plus.cc:92-107`). The task therefore requires roughly
0.8 m of translation plus a ~π reorientation of the T.

### What we optimize

Each control cycle solves a finite-horizon contact-implicit MPC problem with
C3+ (ADMM over consensus copies) on a locally linearized Linear
Complementarity System. The state is `x ∈ R^19` (pusher position, object
quaternion, object position, then velocities), the input is `u ∈ R^3`
(Cartesian pusher force), contact forces are `λ ∈ R^20`, and the horizon is
`N = 5`:

```text
minimize    Σ_{t=0..N} (x_t - x_d)ᵀ Q (x_t - x_d)  +  Σ_{t=0..N-1} u_tᵀ R u_t

subject to  x_{t+1} = A x_t + B u_t + D λ_t + d          (LCS dynamics)
            0 ≤ λ_t ⊥ E x_t + F λ_t + H u_t + c ≥ 0     (complementarity)
```

The desired state `x_d` encodes the `open_table` goal variables directly: the
object-position slots hold `(x_g, y_g, resting_height)` and the
object-quaternion slots hold `q(θ_g)`, a pure yaw rotation built from
`goal_pose.z()` (`xarm6_full_sampling_c3plus.cc:1597-1605`). The cost
matrices are assembled in `RunSolveAtSampledPusher`
(`xarm6_full_sampling_c3plus.cc:1579-1596`):

```text
Q = state_cost_scale · diag(state_cost_diagonal)
  = 50 · diag(0.01, 0.01, 0.01,          # pusher position
              0.1, 0.1, 0.1, 0.1,        # object quaternion  → orientation goal
              200, 200, 120,             # object position    → translation goal
              5, 5, 5,  0.05 ×6)         # velocities
R = 1.0 · diag(0.01, 0.01, 0.01)
```

so translation error is weighted at an effective 10,000 per m² on object x/y
and orientation enters through the quaternion-error terms. ADMM additionally
carries consensus and projection penalties `G = 0.01·diag(g)` and
`U = 0.26·diag(u)` on the `λ`/`η` copies (weights 2/1 and 20/1); these enforce
the complementarity structure and are not part of the task objective.

Around that inner QP, three more objectives shape the behavior:

- **Sample ranking.** Candidate pusher placements are each solved and then
  scored by forward-simulating the plan through the LCS and accumulating the
  same quadratic error, `Σ eᵀ Q e` plus a terminal term
  (`dynamic_rollout_cost`, `xarm6_full_sampling_c3plus.cc:1740-1834`); the
  minimum-cost candidate is executed. A separate
  `object_yaw_cost_weight: 50.0` biases goal/sample selection toward yaw
  progress.
- **Acquisition IK.** Repositioning to a sampled contact solves an inverse
  kinematics problem with a position constraint on the stick tip and the
  restored source tilt objective `w_tilt · (1 - cos ψ)`, `w_tilt = 80`, which
  keeps the stick vertical inside the feasibility band
  (`xarm6_sampling_c3_controller.cc:120`, `:391-400`).
- **OSC tracking.** The 500 Hz operational-space controller tracks the
  selected trajectory with Cartesian gains `kp = 200`, `kd = 20` and a
  0.01-weight joint-posture regularizer.

In short: the *task* asks for `e_p < 0.05 m` and `e_θ < 0.10 rad` on the T's
planar pose; the *optimizer* minimizes a quadratic penalty on exactly those
two errors (plus small pusher/velocity/effort regularization) subject to
contact-implicit dynamics, and every sampled contact location competes on the
same objective.

### Status

Terminal `open_table` success has not yet been achieved. The chronological
gate records (gates 101–388: contact budgeting, admission gating,
dwell/release cycles, contact receipts, the GATE_380 direction-preserving
limiter + restored `w_tilt = 80` root-cause fix, recovery admission, and
neutral-retreat fallback) live in the C++ worktree with one ledger per gate
range under `examples/sampling_c3/oim_t/` and are summarized in that
worktree's `README.md`. The exact model provenance for the vendored T is in
`examples/sampling_c3/oim_t/OIM_T_PROVENANCE.md`.

## Quick Start

Create the audited environment and run from the repository root:

```bash
conda env create -f environment.yml
conda activate push_anything_admm

# Basic configured task
python main.py pushing

# Sampling-C3 with the default outer-controller configuration
python main.py pushing --sampling-c3 --seed 0 --name pushing_seed0

# Stored T-object workflow configuration
python main.py push_t --max-time 600 \
  --sampling-c3 config/sampling_c3_kik_t.yaml \
  --seed 0 --name push_t_seed0
```

Runs write `results/<name>.txt` and include Git/configuration metadata in the
log. Result media can be rendered separately:

```bash
scripts/make_run_video.sh push_t_seed0 --task push_t
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before comparing or reporting
experiments. It records the audited dependency versions, canonical run metadata,
result-storage policy, and current limitations.

## Repository Structure

```text
control/                 C3/C3+, ADMM, LCS/LCP, costs, OSC, sampling-C3
sim/                     PyDrake environment and object models
config/                  tasks, controller settings, experiment variants
scripts/                 launchers, diagnostics, analysis, plotting
tools/visualizer/        log parsing and result/video rendering
tests/                   unit, solver, model, integration, regression tests
docs/                    conformance notes, investigations, generated figures
results/                 ignored working outputs and stored local campaigns
main.py                  CLI, simulation loop, logging, orchestration
```

More detailed maps are in [REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md) and
the local indexes under `config/`, `scripts/`, and `tests/`.

## Tests

```bash
python -m pytest tests
```

The recorded cleanup baseline collected 344 tests: 292 passed, 37 failed, and
15 errored. Twenty-five non-passing nodes were blocked because the audit runner
forbids the localhost sockets that Meshcat requires; the remaining discrepancies
are classified without changing numerical expectations. See
[TEST_BASELINE.md](docs/TEST_BASELINE.md) and
[`tests/baseline_failures.yaml`](tests/baseline_failures.yaml).

Before changing C3/C3+, ADMM, complementarity projection, LCS construction, or
MPC behavior, establish a clean numerical baseline and preserve the reported
experiment configuration.

## Figure Reproduction

The README figures are regenerated with:

```bash
python docs/figures/generate_readme_figures.py
```

The three SVGs are conceptual diagrams derived from the current source
architecture. The PNG is copied from an existing measured result and is never
recomputed by the README generator.

## References

- H. Bui et al., “Push Anything: Single- and Multi-Object Pushing From First
  Sight with Contact-Implicit MPC,” arXiv:2510.19974, 2025.
- A. Aydinoglu, A. Wei, W.-C. Huang, and M. Posa, “Consensus Complementarity
  Control for Multi-Contact MPC,” *IEEE Transactions on Robotics*, 40,
  3879–3896, 2024.
- S. Venkatesh, B. Bianchini, A. Aydinoglu, W. Yang, and M. Posa,
  “Approximating Global Contact-Implicit MPC via Sampling and Local
  Complementarity,” arXiv:2505.13350, 2025.
- Y. Li, H. Han, S. Kang, J. Ma, and H. Yang, “On the Surprising Robustness of
  Sequential Convex Optimization for Contact-Implicit Motion Planning,”
  arXiv:2502.01055, 2025. Comparison work is a research direction only.
