# Drake C3 Port — Directional Characterization Milestone

**Date frozen**: 2026-04-21
**Scope**: Reimplementation of Consensus Complementarity Control (Aydinoglu & Posa, ICRA 2022) for 7-DOF Franka Panda planar pushing in Drake. Empirical characterization of vanilla C3's capabilities and limits.
**Predecessor**: `planar_sandbox/milestones/2026-04-20_vanilla_c3_working/` (Drake-free 2D reference).
**Successor** (planned): `milestones/????-??-??_c3plus_sandbox_validation/` — C3+ sampling in the sandbox.

## Status: Infrastructure verified; vanilla C3 characterized across 4 goal directions

The contact-model infrastructure and ADMM optimization work correctly. With the arm prepositioned on the box's west face (7mm clearance, no initial penetration), vanilla C3 successfully pushes the box 35% toward an east goal. When the goal is orthogonal or anti-aligned to the active contact normal, vanilla C3 fails in structured, interpretable ways. These failures empirically demonstrate the greedy-contact-mode limitation that motivates C3+ (Bui et al. 2025).

## Verified infrastructure

### 1. LCS extraction from Drake SceneGraph
- `ComputeSignedDistancePairwiseClosestPoints` with 10cm threshold
- Contact pair filter: pusher ↔ box_link only
- 4-edge polyhedral friction pyramid: `[+t1, -t1, +t2, -t2]` with `t1 = nhat × ref, t2 = nhat × t1`
- Translational-velocity Jacobians at witness points via `CalcJacobianTranslationalVelocity`
- Discrete-time LCS matrices `A, B, D, d` from `M, Cv, tau_g, B` with first-order linearization
- Soft complementarity penalty: `w_comp = 100.0 · max(phi, 0)` on λ_n per horizon step

Verified via `diagnostic_logs/pusher_weld_diagnosis.txt`.

### 2. Pusher rigidly welded to panda_link8
- WeldJoint `panda_link8_welds_to_pusher`
- `is_floating() == False`
- 1 collision geometry (sphere r=2.5cm, GeometryId 223)
- Welded 5cm along link8 `+z` (per SDF)

### 3. ADMM with fixed polyhedral projection
- 3 iterations per control step (Push Anything Table I, single-object)
- Consensus split: `z = [x, λ, u]` stacked over horizon; `δ = proj_C(z + ω)` per-contact
- OSQP for the z-update QP
- Lorentz projection for the δ-update, polymorphic over tangent dim:
  - k=1 (sandbox scalar): direct Lorentz projection
  - k=4 (polyhedral): convert to 2D Cartesian F_t, project on `(λ_n, F_t) ∈ ℝ³`, split back into non-negative components

Projection verified to produce non-negative polyhedral components with cone constraint satisfied exactly.

### 4. Prepositioned initial condition (no penetration)
- `PREPOSITIONED_ARM_Q = [0.602050, 1.368574, 1.119371, -1.896220, 1.722135, 2.991599, 0.785000]`
- IK target: pusher center at `(−0.085, 0, 0.05)`, achieved `(−0.082, -0.003, 0.051)` with 4.04mm error
- Initial contact distance at t=0: **+0.007m** (7mm clearance, no penetration)
- No spurious impulse at t=0

## 4-task directional sweep (headline result)

All four tasks use identical starting conditions. Only the goal direction differs.

| Task | Direction | Goal     | End box pos              | Goal dist (start → end) | Progress |
|------|-----------|----------|--------------------------|-------------------------|----------|
| 2    | East (8s) | (0.3, 0) | (0.105, -0.003, 0.050)   | 0.300 → 0.195 m         | **35%**  |
| 2    | East (30s)| (0.3, 0) | (0.103, -0.016, 0.050)   | 0.300 → 0.197 m         | 34% (stalled) |
| 1    | North     | (0, 0.3) | (0.215, -0.221, 0.050)   | 0.300 → 0.564 m         | -88%     |
| 3    | South     | (0, -0.3)| (0.152, 0.004, 0.052)    | 0.300 → 0.340 m         | -13%     |
| 4    | West      | (-0.3,0) | (0.169, 0.025, 0.050)    | 0.300 → 0.469 m         | -56%     |

### Task 2 (east) — partial success with stall

**C3 works when the goal is aligned with the contact normal, up to a stiction-limited fixed point.**

At the 8-second scope:
- Box moved 10.5cm east with only 3mm y-drift and no z-drift (stayed on table)
- 801 ADMM solves with contact throughout
- 35% progress toward goal

Extending to 30 seconds revealed the stall:
- Between t=5s and t=30s, the box moved 2mm backward (0.105 → 0.103) and 15mm south
- Goal distance bottomed at 0.195m and plateaued at 0.197m
- Applied torque remained ~20 Nm throughout the stall (controller was actively trying)
- 3000 ADMM solves over 30s, contact maintained 95% of steps

**Interpretation**: The LCS controller does not model box-ground friction, so it plans forces as if the box slides freely. When the planned force exceeds static friction (μ · m · g ≈ 0.78 N for this box), the box slips briefly; when it slips, MPC observes the motion and immediately reduces the planned force; below stiction, the box stops. The net result is a stiction-limited fixed point: the arm reaches a pose where its deliverable force exactly balances static friction, and both the controller and the simulator settle there.

This is the concrete manifestation of the LCS-simulator mismatch noted in the Known Limitations. Vanilla C3 cannot resolve it without either:
(a) adding ground contact to the LCS — which requires C3+'s multi-contact projection to handle the scaling mismatch between pusher-box and box-ground forces,
(b) applying a velocity-damping approximation to the box in the LCS — a hack that partially compensates.

### Task 1 (north) — fails structurally

**C3 loses contact when commanded to push orthogonally.**

- Final box position: (0.215, -0.221, 0.050) — box moved SOUTH 22cm and east 21cm
- Only 106 ADMM solves (vs 801 for others) — contact lost for 87% of steps
- The arm's attempted north-pushing tangential friction forces displaced the box sideways; box escaped the pusher's reach; arm could not re-approach

### Task 3 (south) — fails similarly to north

**Same orthogonal-goal failure mode as task 1, opposite direction.**

- Final box position: (0.152, 0.004, 0.052) — box moved mostly east despite south goal
- 801 ADMM solves (maintained contact, unlike task 1)
- Arm pushed east through the west-face contact; C3 could not apply a south force sufficient to overcome ground static friction

### Task 4 (west) — catastrophic failure (expected)

**C3 cannot push through the box.**

- Final box position: (0.169, 0.025, 0.050) — box moved 17cm EAST toward the exact opposite of the goal
- Pusher is on the west face; pushing west would require the pusher to phase through the box
- C3 has no mechanism to reposition the end-effector to the east face
- Box motion is driven by whatever lateral forces C3 can apply through friction

## Interpretation

The four tasks form a controlled experiment. The independent variable is goal direction. The dependent variables are box final position and ADMM behavior. Results:

- **1/4 task success rate** (25%)
- **Success correlates with goal-contact alignment**: the one successful task (east) has its goal aligned with the contact normal (pushing on west face pushes box east); the three failures have goals that are orthogonal or anti-aligned.
- **Failure modes are structured, not random**: orthogonal goals produce off-axis drift; anti-aligned goals produce motion opposite to goal.
- **ADMM is internally consistent across all tasks** (3 iterations, bounded residuals, no divergence). The controller is not broken; it is solving the LCS correctly, but the LCS itself cannot represent contact-mode switching.

These results empirically demonstrate the greedy-contact-mode limitation identified analytically by Aydinoglu & Posa (2022) and addressed by C3+ sampling in Bui et al. (2025). The next step is C3+.

## Why C3+ would solve these failures

C3+ adds a sampling layer that proposes candidate end-effector positions and runs MPC for each. For each of the 4 tasks, C3+ would:

- **Task 1 (north)**: sample positions on the box's south face; prefer those where MPC projects north-push is feasible.
- **Task 3 (south)**: sample positions on the north face.
- **Task 4 (west)**: sample positions on the east face.
- **Task 2 (east)**: current west-face position is already optimal; sampling would confirm it.

## Controller settings (as of freeze)

| Parameter | Value | Source |
|---|---|---|
| Horizon (N) | 20 | tasks.yaml |
| Control dt | 0.05 s | tasks.yaml |
| ADMM max_iters | 3 | tasks.yaml (Push Anything Table I) |
| ρ_init | 100.0 | admm_solver.py |
| Force limit | 30 Nm | tasks.yaml |
| w_obj_xy | 100000.0 | tasks.yaml |
| w_ee_approach | 8000.0 | tasks.yaml |
| Friction μ | 0.4 | tasks.yaml |
| Box mass | 0.2 kg | tasks.yaml |
| Contact filter | pusher ↔ box_link only | lcs_formulator.py |

## Known limitations

1. **Greedy contact mode**: the primary failure mode demonstrated above. Requires C3+ sampling to resolve.
2. **No ground contact in LCS — stall at stiction limit**: the LCS controller does not model box-ground friction. Drake's sim applies it, so there is a model-sim mismatch. The 30-second east run shows this mismatch causes a hard stall at ~35% progress. The controller is not broken; it reaches an equilibrium where planned force equals stiction and cannot plan forces large enough to slip the box further. Resolving this requires multi-contact LCS — which vanilla C3's single Lorentz projection cannot handle (see Drake port attempt logs in the project history). C3+ is required.
3. **Default arm pose cannot reach box**: `INITIAL_ARM_Q = [0.0, 0.4, 0.0, -2.8, 0.0, 3.2, 0.785]` leaves the pusher 82cm above the box, and the approach heuristic's workspace-boundary target cannot be reached. The prepositioned pose is required for C3 evaluation.

## Files in this snapshot

### source_snapshot/

| File | Purpose |
|---|---|
| admm_solver.py | C3Solver with polyhedral-aware projection (k=4) |
| lcs_formulator.py | Pusher-box contact filter, polyhedral friction pyramid |
| task_costs.py | Approach-phase proxy cost, C3 tracking cost |
| main.py | CLI with `--prepositioned` and `--task-id {1,2,3,4}` flags |
| env_builder.py | Drake MultibodyPlant, pusher weld, `PREPOSITIONED_ARM_Q` |
| tasks.yaml | Base task config (admm_max_iters=3) |
| directional_tasks.json | Goal-direction lookup (1=north, 2=east, 3=south, 4=west) |
| find_prepositioned_q.py | IK script for PREPOSITIONED_ARM_Q |
| diagnose_pusher_weld.py | Plant topology + collision filter diagnostic |

### diagnostic_logs/

| File | Purpose |
|---|---|
| run_3iter_clearance_task1.txt | North goal — original 8s run (legacy reference) |
| run_3iter_clearance_task3.txt | South goal — original 8s run (legacy reference) |
| run_3iter_clearance_task4.txt | West goal — original 8s run (legacy reference) |
| task1_north_8s.txt | North goal — 8s run with video (authoritative) |
| task2_east_30s.txt | East goal — 30s extended run; documents the stall (authoritative) |
| task3_south_8s.txt | South goal — 8s run with video (authoritative) |
| task4_west_8s.txt | West goal — 8s run with video (authoritative) |
| pusher_weld_diagnosis.txt | Plant topology, weld verification, collision geom inventory |
| prepositioned_ik_clearance.txt | IK solution for PREPOSITIONED_ARM_Q |

## Video replays

HTML files in `video_replays/` are self-contained Meshcat playbacks. Open any in a browser to view the robot motion. Each preserves the full 3D scene and allows pausing, scrubbing, and rotating the view.

| File | Task | Duration |
|---|---|---|
| `task1_north_8s.html` | North goal | 8s |
| `task2_east_30s.html` | East goal (extended) | 30s |
| `task3_south_8s.html` | South goal | 8s |
| `task4_west_8s.html` | West goal | 8s |

The task 2 video shows the stall clearly: rapid progress in the first 5 seconds, then the arm visibly hovering and making micro-movements without moving the box for 25 more seconds.

## Reproducing the results

Prerequisites: `push_anything_ADMM` conda env active, in `/d/projects/ERL/push_anything_ADMM`.

```bash
# Tasks 1, 3, 4 — 8 seconds
for id in 1 3 4; do
    python main.py pushing --prepositioned --task-id $id \
        --video-path results/task${id}.html 2>&1 | tee run_task${id}.txt
done

# Task 2 — 30 seconds (to observe the stall)
python main.py pushing --prepositioned --task-id 2 --max-time 30.0 \
    --video-path results/task2_east_30s.html 2>&1 | tee run_task2.txt
```

Tasks 1/3/4 take ~1 minute wall-clock each. Task 2 takes ~3 minutes.
