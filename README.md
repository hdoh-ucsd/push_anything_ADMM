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

## Figure 8 — 28 Consecutive SE(2) Goals

The corrected protocol uses one uninterrupted manipulation session per object:
the robot, object, controller state, and random-goal stream carry over across all
28 goals. Each goal has its own 600-second simulated-time limit. A session passes
only if it reaches all 28 goals consecutively; failed sessions are not replaced
with another seed.

The completed campaign contains **6 passing sessions out of 25 objects**. The
table below contains only those 28/28 sessions. Click an animated preview to open
the full MP4 with video controls. The complete per-session and per-goal results
are available in [`FIG8_CONSECUTIVE_28_SESSIONS.csv`](FIG8_CONSECUTIVE_28_SESSIONS.csv)
and [`FIG8_CONSECUTIVE_28_GOALS.csv`](FIG8_CONSECUTIVE_28_GOALS.csv).

| Object | Consecutive goals | Full session video |
|---|---:|---|
| Baby Toy | 28/28 | [![Baby Toy — 28 consecutive goals](results/fig8_consecutive_gallery/previews/baby-toy.gif)](results/fig8_consecutive_gallery/videos/baby-toy.mp4) |
| Chicken Broth | 28/28 | [![Chicken Broth — 28 consecutive goals](results/fig8_consecutive_gallery/previews/chicken-broth.gif)](results/fig8_consecutive_gallery/videos/chicken-broth.mp4) |
| Egg Carton | 28/28 | [![Egg Carton — 28 consecutive goals](results/fig8_consecutive_gallery/previews/egg-carton.gif)](results/fig8_consecutive_gallery/videos/egg-carton.mp4) |
| Expo Box | 28/28 | [![Expo Box — 28 consecutive goals](results/fig8_consecutive_gallery/previews/expo-box.gif)](results/fig8_consecutive_gallery/videos/expo-box.mp4) |
| Milk Bottle | 28/28 | [![Milk Bottle — 28 consecutive goals](results/fig8_consecutive_gallery/previews/milk-bottle.gif)](results/fig8_consecutive_gallery/videos/milk-bottle.mp4) |
| Xbox | 28/28 | [![Xbox — 28 consecutive goals](results/fig8_consecutive_gallery/previews/xbox.gif)](results/fig8_consecutive_gallery/videos/xbox.mp4) |

### Genuine failure videos

These 12 sessions produced a valid experiment failure. “Goals reached” is the
consecutive prefix completed before the terminal failed goal. Click a preview
to open the full failed-session MP4.

| Object | Goals reached | Failure reason | Full session video |
|---|---:|---|---|
| Letter A | 1/28 | Persistent topple | [![Letter A failure](results/fig8_consecutive_failures/previews/letter-a.gif)](results/fig8_consecutive_failures/videos/letter-a.mp4) |
| Letter Y | 7/28 | Persistent topple | [![Letter Y failure](results/fig8_consecutive_failures/previews/letter-y.gif)](results/fig8_consecutive_failures/videos/letter-y.mp4) |
| Letter H | 1/28 | Runaway displacement / safety limit | [![Letter H failure](results/fig8_consecutive_failures/previews/letter-h.gif)](results/fig8_consecutive_failures/videos/letter-h.mp4) |
| Letter E | 0/28 | Tight-tolerance timeout | [![Letter E failure](results/fig8_consecutive_failures/previews/letter-e.gif)](results/fig8_consecutive_failures/videos/letter-e.mp4) |
| Lotion | 7/28 | Persistent topple | [![Lotion failure](results/fig8_consecutive_failures/previews/lotion.gif)](results/fig8_consecutive_failures/videos/lotion.mp4) |
| Wood Block | 3/28 | Persistent topple | [![Wood Block failure](results/fig8_consecutive_failures/previews/wood-block.gif)](results/fig8_consecutive_failures/videos/wood-block.mp4) |
| Tape | 0/28 | Tight-tolerance timeout | [![Tape failure](results/fig8_consecutive_failures/previews/tape.gif)](results/fig8_consecutive_failures/videos/tape.mp4) |
| Eraser | 0/28 | Persistent topple | [![Eraser failure](results/fig8_consecutive_failures/previews/eraser.gif)](results/fig8_consecutive_failures/videos/eraser.mp4) |
| Clamp | 3/28 | Workspace limit | [![Clamp failure](results/fig8_consecutive_failures/previews/clamp.gif)](results/fig8_consecutive_failures/videos/clamp.mp4) |
| Book | 5/28 | Workspace limit | [![Book failure](results/fig8_consecutive_failures/previews/book.gif)](results/fig8_consecutive_failures/videos/book.mp4) |
| Gallon Milk | 0/28 | Workspace limit | [![Gallon Milk failure](results/fig8_consecutive_failures/previews/gallon-milk.gif)](results/fig8_consecutive_failures/videos/gallon-milk.mp4) |
| Push T | 4/28 | Goal timeout | [![Push T failure](results/fig8_consecutive_failures/previews/push-t.gif)](results/fig8_consecutive_failures/videos/push-t.mp4) |

### Protocol-invalid abort videos

These seven sessions were stopped by the erroneous displacement guard. They are
shown for diagnosis but are not counted as genuine controller failures.

| Object | Goals reached | Abort reason | Full session video |
|---|---:|---|---|
| Letter I | 2/28 | Displacement-guard abort | [![Letter I invalid session](results/fig8_consecutive_failures/previews/letter-i.gif)](results/fig8_consecutive_failures/videos/letter-i.mp4) |
| Letter C | 2/28 | Displacement-guard abort | [![Letter C invalid session](results/fig8_consecutive_failures/previews/letter-c.gif)](results/fig8_consecutive_failures/videos/letter-c.mp4) |
| Letter R | 2/28 | Displacement-guard abort | [![Letter R invalid session](results/fig8_consecutive_failures/previews/letter-r.gif)](results/fig8_consecutive_failures/videos/letter-r.mp4) |
| Letter G | 2/28 | Displacement-guard abort | [![Letter G invalid session](results/fig8_consecutive_failures/previews/letter-g.gif)](results/fig8_consecutive_failures/videos/letter-g.mp4) |
| Letter B | 2/28 | Displacement-guard abort | [![Letter B invalid session](results/fig8_consecutive_failures/previews/letter-b.gif)](results/fig8_consecutive_failures/videos/letter-b.mp4) |
| Letter 3 | 2/28 | Displacement-guard abort | [![Letter 3 invalid session](results/fig8_consecutive_failures/previews/letter-3.gif)](results/fig8_consecutive_failures/videos/letter-3.mp4) |
| Letter S | 2/28 | Displacement-guard abort | [![Letter S invalid session](results/fig8_consecutive_failures/previews/letter-s.gif)](results/fig8_consecutive_failures/videos/letter-s.mp4) |

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
