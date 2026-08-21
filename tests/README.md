# Test Suite Guide

Run the intended suite from the repository root:

```bash
python -m pytest tests
```

The current baseline and all non-passing nodes are documented in
`../TEST_BASELINE.md` and `baseline_failures.yaml`. Do not change tolerances,
expected values, configs, or solver behavior merely to make the baseline green.

## Current categories

### Solver and complementarity correctness

- `test_c3plus_projection_eq12.py`
- `test_c3plus_vs_c3_smoke.py`
- `test_lcp_projection.py`
- `test_projection.py`
- `test_solver_scale_plumbing.py`

### LCS and model construction

- `test_lcs_ee_space_ulin_consistency.py`
- `test_lcs_efhc.py`
- `test_lcs_jacobian.py`
- `test_t_mesh_witness_conformance.py`

The first three groups currently require Drake environment construction and are
blocked on runners that cannot create Meshcat's localhost socket.

### MPC and OSC

- `test_inner_solve.py`
- `test_osc_default_gains.py`
- `test_osc_force_tracking.py`
- `test_osc_joint2_posture.py`
- `test_osc_rot_task.py`
- `test_osc_tracking.py`
- `test_osc_trajectory_interface.py`
- `test_osc_unit.py`
- `test_planner_workspace_bounds.py`
- `test_task_costs_lateral.py`

### Sampling-C3 policy and candidate handling

- `control/sampling_c3/test_altitude_hold_gate.py`
- `test_buffer_travel_normalization.py`
- `test_commit_face_gate.py`
- `test_commit_face_gate_post_decide.py`
- `test_flip_goal_mode.py`
- `test_jack_goal_generator.py`
- `test_mesh_normal_sampling.py`
- `test_mode_switch.py`
- `test_perimeter_sample_clearance.py`
- `test_progress.py`
- `test_repos_target_collision_gate.py`
- `test_reposition.py`
- `test_reposition_trajectory.py`
- `test_reposition_trajectory_degenerate.py`
- `test_sample_buffer.py`
- `test_sampling_c3_params.py`
- `test_sampling_strategies.py`
- `test_slerp_lookahead.py`
- `test_topple_roll_plan.py`
- `test_update_buffer_no_reappend.py`
- `test_workspace_radial_gate.py`

### Compatibility and configuration regression

- `test_tick_to_simtime_yaml_compat.py`
- configuration assertions in `test_sampling_c3_params.py`
- default-gain assertions in `test_osc_default_gains.py`

## Proposed markers

Markers are documented but not yet implemented:

- `unit`: pure Python/NumPy, no Drake plant or networking;
- `drake`: requires PyDrake but must run headlessly;
- `meshcat`: explicitly requires localhost networking;
- `integration`: multiple controller/model components;
- `numerical_regression`: approved fixed numerical behavior;
- `slow`: long simulation or benchmark.

Do not move test files into category subdirectories until imports, collection,
fixtures, and the approved baseline can be preserved.
