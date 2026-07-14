# REPOSITION TIER-2 EVIDENCE (2026-07-14)

Diagnostic commit: `64ffdee` — `PUSHA_REPOS_T2_DIAG=1`, env-gated, default-OFF byte-identical.

## Runs

- `run_default.log`: `PUSHA_REPOS_T2_DIAG=1` (default `PUSHA_REPOSITION_PWL=0` → legacy path)
- `run_pwl.log`: `PUSHA_REPOS_T2_DIAG=1 PUSHA_REPOSITION_PWL=1` (Stage-A PWL trajectory path)

Both: `pushing --task-id 4 --max-time 1.0 --seed 0 --admm-iter 25 --solver c3plus --sampling-c3 config/sampling_c3_kik.yaml`.

## Reference-side deep reads (Tier-2 (a), reference is READ-ONLY)

- **Reposition mechanism**: `examples/sampling_c3/reposition.cc` — `Reposition(...)` builds knots for one of 5 traj types (`kSpline`, `kSpherical`, `kCircular`, `kPiecewiseLinear`, {kIK — port-only, no reference analog}). Called once per wrapper tick.
- **Delivery**: `sampling_based_c3_controller.cc:1848-1879` wraps knots in `LcmTrajectory::Trajectory` as `end_effector_position_target` (3-row datapoints, per-knot timestamps `t_context + filtered_solve_time_ + i·dt`), publishes on `repos_traj_execute_actor_port_`.
- **Receiver**: `LcmTrajectoryReceiver::OutputTrajectory` (`lcm_trajectory_systems.cc:46-75`) reconstructs as `PiecewisePolynomial<double>::FirstOrderHold(times, datapoints)` — a PWL PP that DOES carry a piecewise-constant first derivative.
- **OSC consumption**: `osc_tracking_data.cc:88-108` reads `y_des=traj.value(t)`, `ydot_des=traj.EvalDerivative(t,1)`, `yddot_des=traj.EvalDerivative(t,2)=0` from the FirstOrderHold PP.
- **Orientation trajectory**: `sampling_based_c3_controller.cc:1896-1917` builds a per-knot tilted-quaternion trajectory based on distance from workspace center × `max_tilt_angle`. But `EndEffectorOrientationTrajectoryGenerator::CalcTraj` (`end_effector_orientation.cc:33-57`) OVERRIDES the trajectory with a constant identity-quaternion PWL when `track_orientation_=false` (per shared YAML `track_end_effector_orientation: false`). The OSC then **still** actively tracks the identity quaternion with `EndEffectorRotW=diag(10)`, `EndEffectorRotKp=diag(800)`, `EndEffectorRotKd=diag(40)` → compound rotational authority **8000 per axis**. The tilt-orientation code is DEAD; the identity-quaternion tracking is LIVE.
- **filtered_solve_time offset**: `sampling_based_c3_controller.cc:1390-1391` maintains a low-pass-filtered solve-latency; the published knot timestamps start `filtered_solve_time_` seconds INTO THE FUTURE. This compensates for planner-solve delay so the OSC's `t=now` samples the trajectory at the point corresponding to when this knot will actually be executed.
- **IK-based tracker**: `grep InverseKinematics examples/sampling_c3/ systems/controllers/` → **0 hits**. The reference has NO IK-based per-knot tracker.
- **admit-latch / descent-gate**: `grep admit_active ADMIT_LATCH TARGET_STABLE` → **0 hits** in the reference. Port-only mechanisms.
- **Reference reposition_params**:
  ```
  examples/sampling_c3/anything/parameters/reposition_params.yaml:
    traj_type: 3 (kPiecewiseLinear)
    speed: 0.18 m/s
    pwl_waypoint_height: 0.07738005 (generated per-run)
    use_straight_line_traj_under_piecewise_linear: 0.008
    max_tilt_angle: 20°
  examples/sampling_c3/push_t/parameters/reposition_params.yaml:
    speed: 0.18 m/s
    pwl_waypoint_height: 0.06
  ```

## Port-side runtime capture (Tier-2 (b), PUSHA_REPOS_T2_DIAG=1)

```
[REPOS-T2] tracker=PiecewiseLinearTracker traj_type=kPiecewiseLinear
           use_pwl_traj=False env_PUSHA_REPOSITION_PWL=0
[REPOS-T2] params: speed=0.4 pwl_speed=0.18 pwl_waypoint_height=0.15
           use_straight_line_traj_under_piecewise_linear=0.008
[REPOS-T2] reference: traj_type=kPiecewiseLinear speed=0.18
           pwl_waypoint_height=0.06-0.077 straight_line_thresh=0.008
[REPOS-T2] free_mode_v_ee_desired_wired=False
           (legacy path passes v_ee_desired=None; PWL path passes v_des)
```

Default path (used by canonical box runs):
- Tracker: `PiecewiseLinearTracker` (per-tick setpoint march via `next_waypoint`, IK to q_des, joint-PD torque **discarded** — wrapper overrides with OSC's u).
- `speed=0.4` m/s → **2.22× reference 0.18 m/s** (only affects the legacy tracker's per-tick setpoint stride; the tracker's `u` is dead code so this only affects `p_des` handed to the OSC).
- `pwl_waypoint_height=0.15` m → **~2× reference 0.06-0.077 m** (5-8 cm above vs port 15 cm above; higher lift = longer traversal time and more time out of contact).
- `v_ee_desired=None` handed to OSC → OSC uses `v_err = -v_ee_now` (no velocity feedforward). **Confirms 1.a/1.b executor coupling: the port's default reposition drops the derivative that the reference's FirstOrderHold PP carries.**
- `[ADMIT-GUARD] admit_active=0 latch=0/0` throughout — the kIK-only admit-latch mechanism is **INERT under kPWL** (per source comment: "consumed-and-ignored here … NOT an implementation of the debounced admit-active latch"). Same for `[ALT-GATE] target_stable=0/0 allow_descent=1` (descent-gate stability counter).

Stage-A path (`PUSHA_REPOSITION_PWL=1`, env-gated):
```
[STAGE-A-PWL] PUSHA_REPOSITION_PWL=1 → using RepositionTrajectory + OSC position-tracking
[REPOS-T2] tracker=PiecewiseLinearTracker traj_type=kPiecewiseLinear
           use_pwl_traj=True env_PUSHA_REPOSITION_PWL=1
[REPOS-T2] free_mode_v_ee_desired_wired=True
[STAGE-A-PWL] step=1 build p_start=(+0.0001,-0.0005,+0.2001)
              p_target=(-0.0230,-0.0800,+0.0300) K=4 t_end=1.415
[STAGE-A-PWL] step=61 build p_start=(-0.0160,-0.0561,+0.1501)
              p_target=(+0.0800,+0.0000,+0.0300) K=3 t_end=1.894
```

- Stage-A `RepositionTrajectory` builds a 3-leg PWL with K=4 knots (start / lift-end / traverse-end / target), matching the reference's kPiecewiseLinear leg structure.
- Uses `pwl_speed=0.18` m/s (matches reference `speed=0.18` m/s).
- Uses `pwl_waypoint_height=0.15` m (still 2× reference).
- Emits `(p_des, v_des, done)` per tick; wrapper passes `v_des` as `v_ee_desired` to OSC → **v_ee_desired IS wired in this path**, matches reference's FirstOrderHold-derivative semantics.
- Rebuild triggers correctly on target-change > 5 mm (`step=61`).
