# DAIR-lib OSC executor — port re-implementation target

**Source:** `dairlib_sampling_c3` @ `push_anything_dev` `257e3ed` on disk at
`/root/reference_repos/dairlib_sampling_c3/`.
**Purpose:** Phase-1 (Cartesian-force OSC) implementation target per
Originally captured for the completed reproduce-dairlib arc. Retained as a
technical reference; current implementation status lives in
`docs/conformance-map.md`.
**Compiled:** 2026-07-13 via Explore agent.

---

## 1. Top-level executor: `franka_osc_controller.cc`

Path: `examples/sampling_c3/franka_osc_controller.cc`.

**Inputs (Drake input ports on the OSC leaf-system):**
- Robot state (`OutputVector<double>`, positions + velocities + applied inputs)
  → `osc->get_input_port_robot_output()` (lines 237-238).
- Abstract `Trajectory<double>` ports fed via LCM subscribers (lines 194-261):
  - `"end_effector_target"` ← `end_effector_position_receiver` (position target)
  - `"end_effector_orientation_target"` ← `end_effector_orientation_receiver`
  - `"end_effector_force"` ← `end_effector_force_receiver`
  (LCM message type: `lcmt_timestamped_saved_traj`)

**Output (line 234):**
`osc->get_output_port_osc_command()` → `TimestampedVector<double>` of 7 joint
torques → LCM `osc_command_sender` → Franka driver.

**Gravity handling (lines 202-222):** optional
`GravityCompensationRemover` when `cancel_gravity_compensation=true`.

**Key architectural finding:** the OSC does NOT know whether an input
trajectory comes from `sampling_based_c3_controller` or from
`reposition.cc`. Both publish to the same LCM channels; the OSC just
tracks whatever `Trajectory<double>` is on `end_effector_target`. The port
must mirror this — the executor should not encode c3-vs-reposition mode
knowledge.

---

## 2. TrackingData set for push_anything

Constructed in `franka_osc_controller.cc` lines 149-188, weights from
`examples/sampling_c3/shared_parameters/osc_params.yaml`:

| # | Class | Name | Kp | Kd | Weight | Notes |
|---|---|---|---|---|---|---|
| 1 | `TransTaskSpaceTrackingData` | `"end_effector_target"` | [200,200,200] | [20,20,20] | [1,1,1] | tracks EE position |
| 2 | `RotTaskSpaceTrackingData` | `"end_effector_orientation_target"` | [800,800,800] | [40,40,40] | [10,10,10] | default OFF (flag `track_end_effector_orientation`) |
| 3 | `ExternalForceTrackingData` | `"end_effector_force"` | — | — | `W_ee_lambda` (identity per YAML :74-77) | applied at `panda_hand`, zero offset |
| 4 | `JointSpaceTrackingData` | `"panda_joint2_target"` | 200 | 10 | 1 | fixes joint2=1.1 rad (elbow reg) |

`AddForceTrackingData` call at line 188.

---

## 3. OSC QP structure

Files: `systems/controllers/osc/inverse_dynamics_qp.h` + `.cc`, and
`operational_space_control.cc`.

**Decision variables** (`inverse_dynamics_qp.cc:78-83`):
- `dv_` — generalized accelerations (n_v)
- `u_` — joint torques (n_u)
- `lambda_h_` — holonomic constraint forces (n_h)
- `lambda_c_` — contact constraint forces (n_c)
- `lambda_e_` — **external forces from ExternalForceTrackingData** (n_e = 3 for the push_anything EE force)
- `epsilon_` — soft-contact slacks

**Equality/inequality constraints:**
- Dynamics (line 85-90): `M·dv − B·u − Jh'·λ_h − Jc'·λ_c − Je'·λ_e = −bias`
- Holonomic (line 92-95): `Jh·dv + Jh_dot·v = 0`
- Contact (line 97-101): `Jc_active·dv + ε = −Jc_active_dot·v`
- Friction cone (line 103-115): 5-facet linearized per contact
- Optional input effort bounds (line 117-126)
- Optional accel bounds (line 128-144)

**Costs (assembled in `operational_space_control.cc:469-481` for each
TrackingData plus base regs):**
- Task tracking (per TrackingData): standard `‖J·dv + J_dot·v − a_des‖²_W`
  where `a_des = Kp·(x_des − x) + Kd·(v_des − v)`
- Force tracking (ExternalForceTrackingData): `‖λ_e − λ_des‖²_W`
  → this is the key term for c3-λ handoff
- Joint accel reg: `W_joint_accel · ‖dv‖²`
- Torque reg: `W_input_regularization · ‖u‖²`
- Input smoothing: `W_input_smoothing · ‖u − u_prev‖²`
- Soft-constraint slack: `w_soft_constraint · ‖ε‖²`

**Solver:** OSQP via `solvers/fast_osqp_solver.h`.

---

## 4. Wiring — planner → OSC

- `sampling_based_c3_controller` outputs trajectories (actor + object) →
  publishes to LCM → OSC subscribes via `end_effector_position_receiver`
  and `end_effector_force_receiver`.
- `reposition.cc:104` returns a `PiecewisePolynomial<double>::FirstOrderHold(times, points)`
  from an N-knot `knots` matrix → also published to LCM →
  `end_effector_position_receiver` consumes it.
- **Both planner and reposition feed the same LCM channels.** The
  executor is source-agnostic; the c3/reposition mode-switch happens
  UPSTREAM of the OSC.

---

## 5. Surprises + port re-implementation constraints

- **All wiring is LCM-based** (lines 2, 28, 32-35). The Python port has
  no LCM — direct in-process function calls replace LCM messages, but the
  *interface contract* is unchanged: the executor consumes a
  `Trajectory<double>`-equivalent, not a per-tick point.
- The **abstract `Trajectory<double>` port** is the key contract: to
  match reference semantics, the port executor must consume a **full
  trajectory (start→end knots + interpolation)**, not a single setpoint.
  The current port passes `p_ee_desired` as a single R³ point — that is
  the main upstream divergence, and it's what changes in Phase 2.
- For Phase 1 (executor only, holding the port's existing reposition +
  admission), the trajectory input can be a degenerate single-knot
  PiecewisePolynomial or interpolated to match the port's setpoint
  cadence — the interface must still be trajectory-shaped so Phase 2's
  full-PWL swap is a signature-compatible upgrade.
- **`RotTaskSpaceTrackingData` is default-OFF** in push_anything.
  Match: the port's OSC should not track EE orientation unless the flag
  flips.
- **Joint-2 elbow regularization** at 1.1 rad is a load-bearing posture
  term. The port's `q_nominal` currently plays this role in the impedance
  controller; the new OSC's `JointSpaceTrackingData` should target the
  same joint-2 value.
- **Gravity compensation**: the reference removes it optionally via
  `GravityCompensationRemover`. The port's OSC currently gravity-comps
  via `bias_term − gravity_forces`. Match: keep gravity comp on for the
  Franka. (URDF gravity in the port already matches.)

---

## 6. Phase-1 implementation checklist (target)

- [ ] Introduce a `TrackingData`-shaped interface in the port OSC
  (position, force, joint-space; no orientation for push_anything).
- [ ] Accept a `Trajectory`-shaped input (start with a degenerate
  single-knot PiecewisePolynomial; Phase 2 flips it to full-PWL).
- [ ] QP decision vars: `[dv, u, lambda_e]` (drop the port's ad-hoc
  `lambda_ext` naming; adopt reference `lambda_e`; keep n_e = 3).
- [ ] Costs: match reference weight set + Kp/Kd from `osc_params.yaml`
  (Kp_pos = 200·I, Kd_pos = 20·I, W_pos = I, W_ee_lambda = I,
  W_joint_accel, W_input_reg, W_input_smoothing, w_soft_constraint).
- [ ] Gravity comp ON (`GravityCompensationRemover` equivalent).
- [ ] Joint-2 posture at 1.1 rad via JointSpaceTrackingData equivalent.
- [ ] Solver: keep OSQP.

Validation gate for Phase 1: swap in this executor with the port's
existing reposition + admission held. If the box push does not
reproduce, STOP per §3 Phase-1 go/no-go.
