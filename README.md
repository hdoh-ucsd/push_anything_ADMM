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

## Object-Informed-Manipulation

### Single-Object Pushing

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
python main.py push_t --max-time 180 \
  --sampling-c3 config/sampling_c3_kik_t.yaml \
  --seed 0 --name push_t_seed0
```

Runs write `results/<name>.txt` and include Git/configuration metadata in the
log. Result media can be rendered separately:

```bash
scripts/make_run_video.sh push_t_seed0 --task push_t
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before comparing or reporting
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

More detailed maps are in [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md),
[CLEANUP_INVENTORY.md](CLEANUP_INVENTORY.md), and the local indexes under
`config/`, `scripts/`, and `tests/`.

## Tests

```bash
python -m pytest tests
```

The recorded cleanup baseline collected 344 tests: 292 passed, 37 failed, and
15 errored. Twenty-five non-passing nodes were blocked because the audit runner
forbids the localhost sockets that Meshcat requires; the remaining discrepancies
are classified without changing numerical expectations. See
[TEST_BASELINE.md](TEST_BASELINE.md) and
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
