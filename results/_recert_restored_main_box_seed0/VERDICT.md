# Restored-main box re-cert verdict

**HEAD:** `main @ 05046c9` — post two reverts (ec7fc4a contamination, 18498c1 gain flip) + test-update commit.
**Protocol:** identical to `_recert_clean_box_v2` (which hit 75.3% at 687b8a6).
**Expected:** ~75.3% closure (matching the clean baseline).

## Measured

`[RESULT] method=sampling-c3  final_obj_xy=(-0.2304, 0.0908)  goal_dist=0.1144m  orient_err=2.4322rad  success=NO  ref_gate=FAIL`

`[OSC-SUMMARY] calls=801  qp_failures=0 (0.00%)  saturation=10 (1.25%)  avg_solve_ms=0.32`

Closure = (0.300 − 0.1144) / 0.300 = **61.9 %**. **Regression vs clean baseline: −13.4 pp.** Orient err = **139°**.

## Diagnosis

The kept Phase-1 additions (joint-2 posture, trajectory-shaped interface, readout fix) were assumed harmless. This re-cert falsifies "harmless" for the aggregate.

Trajectory snapshots (per-100-step):

| step | t | EE (m) | obj_xy (m) | goal_dist | mode |
|---|---|---|---|---|---|
| 100 | 1.0 | (+0.07, −0.006, +0.134) | (+0.00, +0.00) | 0.300 | free |
| 200 | 2.0 | (−0.075, −0.024, +0.210) | (−0.125, +0.003) | 0.175 | free |
| 300 | 3.0 | (−0.183, −0.052, +0.095) | (−0.196, +0.036) | 0.110 | **c3** |
| 400 | 4.0 | (−0.350, −0.038, **+0.711**) | (−0.230, +0.091) | 0.114 | free |
| 500 | 5.0 | (−0.150, +0.046, +0.630) | (−0.230, +0.091) | 0.114 | free |
| 600 | 6.0 | (−0.178, +0.029, +0.327) | (−0.230, +0.091) | 0.114 | free |
| 700 | 7.0 | (−0.178, +0.029, +0.104) | (−0.230, +0.091) | 0.114 | free |
| 801 | 8.0 | (−0.059, +0.072, +0.140) | (−0.230, +0.091) | 0.114 | c3 |

Between step 300 and 400: **EE rockets vertically from z=95 mm to z=711 mm in 1 s** (~62 m/s² vertical accel). Box stops moving at step 300 (obj drift < 1 mm from step 300 to step 801). The EE is thrown upward, exits contact zone, box halts.

## Flagged root cause

**Joint-2 posture pull** (commits `33d8208` + `91cc587`, kept from Phase 1):

- YAML config: `Kp_joint2=200`, `Kd_joint2=10`, `W_joint2=1`, `target=1.1 rad`, `joint2_idx=1` (Franka joint 2).
- Cost term: `W_joint2 · ‖v̇[j2] − (Kp·(1.1 − q_j2) + Kd·(−v_j2))‖²`.
- During c3-mode contact, joint-2 tends to be low (elbow flexed for pusher-near-box configuration). If `q_j2 ≈ 0.5 rad`, `a_j2 ≈ 200·(1.1 − 0.5) = 120 rad/s²`. The QP satisfies this by preferring vdot solutions that yank joint 2 back toward 1.1, which in Franka kinematics unavoidably raises the EE.

## Byte-identity of the other kept additions

- **Trajectory-shaped interface** (491ad1a wrapper import + aa42789 c3 dispatch): unit test `test_trajectory_interface_delegates_to_compute_torque` verifies the single-knot ZOH PP passes the same `p_ee_desired` byte-identically to `compute_torque`. No behavior change.
- **Readout fix** (dacba48): renormalizes final box quaternion before `ad.RotationMatrix()` — post-sim math only, cannot affect the control loop.

## Recommendation (not action; awaiting your call)

**Option A** — turn joint-2 pull OFF via YAML (set `W_joint2: 0.0` in `config/osc_franka.yaml`) but keep the code paths (default-OFF, easily flipped when re-scoping the coupling). Zero code churn, byte-identical to pre-Phase-1 for control, keeps the harmless code structure.

**Option B** — revert both joint-2 commits (`91cc587` YAML + `33d8208` + `a8e77c6` code). Full removal.

Option A is smaller-scope and lets the joint-2 code stay ready for reference-aligned OSC gain experiments later (where the dairlib joint-2 pull IS load-bearing). Awaiting your call.

## Also un-verified (deferred to your call)

Even after joint-2 turn-off, we haven't re-verified the box returns to 75.3%. If Option A/B lands, a third re-cert box run confirms.
