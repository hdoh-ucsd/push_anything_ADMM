# Copilot Instructions for push_anything_ADMM

- This repo is a Drake-based pushing simulator. The main runtime path is `main.py`, which builds a Franka + box environment, starts Meshcat, and runs a simulation loop.
- The key architecture is: environment builder -> Drake plant formulator -> ADMM contact solver -> MPC controller -> simulation loop.

## Key files and components

- `main.py`: primary entry point. It constructs the physics `Diagram`, initializes `LCSFormulator`, `ADMMSolver`, and `C3PlusMPC`, then advances the Drake simulator.
- `control/lcs_formulator.py`: extracts dynamics matrices (`M`, `Cv`, `tau_g`, `B`) and contact Jacobians (`J_n`, `J_t`) from a Drake `plant` and context.
- `control/admm_solver.py`: implements the ADMM loop with a QP x-update and a proximal z-update. This is the physics/contact solver used by MPC rollouts.
- `control/base_mpc.py`: contains shared rollout logic and cost evaluation. It is critical that rollouts update the Drake plant state in-place before querying dynamics.
- `control/ci_mpc_c3.py` and `control/ci_mpc_c3plus.py`: follow two different MPC styles. `C3MPC` is finite-difference gradient-based; `C3PlusMPC` uses sampled trajectory weighting.
- `sim/plant_builder.py`: alternate world builder that loads local Drake model packages and adds visualization.

## Important conventions

- Rollout state is updated directly on the Drake plant with `SetPositions` and `SetVelocities` in `BaseMPC.rollout_trajectory`.
- Contact structure is represented as separate normal and tangent Jacobians, then stacked into `J_c = np.vstack([J_n, J_t])`.
- Cost evaluation is currently hard-coded for the target box and Franka end-effector:
  - box state indices `11:14`
  - end-effector frame `panda_link8`
- `control/controller.py` is currently an unused stub file; do not assume it contains active logic.
- The repo uses Meshcat for visualization and expects `ad.StartMeshcat()` to be available.

## Developer workflow notes

- There is no root test suite or CI config in the repo root.
- `setup.sh` is a scaffold script that creates the expected directories and placeholder files but does not install dependencies.
- Runtime command is `python main.py`.
- The repo contains placeholder config files (`config/sim_settings.yaml`, `config/admm_params.yaml`) and an empty `environment.yml`; they are not currently read by the code.

## What to avoid

- Do not introduce a new runtime path that bypasses the existing `formulator -> admm_solver -> mpc` pipeline unless there is a clear reason.
- Avoid changing hard-coded indices or frame names without verifying the Drake model state ordering, since `BaseMPC.evaluate_cost` uses fixed indices.
- Do not assume there is a package-managed build system or tests unless explicit files are added.

## Suggested focus for changes

- Keep interaction logic localized to `control/*` and `main.py`; these are the active execution points.
- Prefer minimal edits in `sim/plant_builder.py` only when modifying environment loading or Drake package mapping.
- If adding new controller variants, implement them as subclasses of `BaseMPC` and preserve the existing `compute_control(current_context, current_q, current_v, target_q)` interface.
