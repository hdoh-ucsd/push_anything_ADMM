# EXECUTOR TIER-2 EVIDENCE (2026-07-14)

Diagnostic commit: `bde8d64` (+ `ce29c9f`) — `PUSHA_EXEC_T2_DIAG=1`, log-only, env-gated, default-OFF byte-identical.

Run: `python main.py pushing --task-id 4 --max-time 4.0 --seed 0 --admm-iter 25 --solver c3plus --no-record --sampling-c3 config/sampling_c3_kik.yaml` (terminated by 480 s wall timeout at sim step ~250).

Full raw log: `audit_output/exec_tier2/run_default_full.log`
Filtered T2 lines: `audit_output/exec_tier2/run_default.log`

## Reference-side (Tier-2 (a), read-only in /root/reference_repos/dairlib_sampling_c3 @ push_anything_dev 257e3ed)

- **1.c / 1.g — n_c, n_h**: grep `AddContactPoint` / `AddKinematicConstraint` in `examples/sampling_c3/` returns **0 hits**. `SetContactFriction(osc_params.mu)` is called (`franka_osc_controller.cc:197`) but registers no contact point. **Reference OSC for push_anything has n_c = 0, n_h = 0.** Port's missing `λ_h, λ_c, ε` variables + friction-cone constraints are true no-ops. **UNKNOWN → KNOWN-INERT.**
- **1.o — FSM**: `franka_osc_controller.cc:138`: `builder.AddSystem<OperationalSpaceControl>(plant, plant_context.get(), false)` — third arg `used_with_finite_state_machine = false`. Impact-invariant projection block (`operational_space_control.cc:394-397, 411-423, 589-685`) is gated on `used_with_finite_state_machine_` and never executes for push_anything. Port omission is inert. **UNKNOWN → KNOWN-INERT.**
- **1.f — reference frame identity**: `franka_osc_controller.cc:170` (ExternalForceTrackingData): `kEndEffectorName, Vector3d::Zero()`. `sampling_c3_utils.h:18`: `kEndEffectorName = "end_effector_tip"`. URDF: `end_effector_full.urdf:57` defines a rigid `end_effector_tip` link with a sphere-radius-0.0195 m collision at zero-offset from the link origin, welded downstream of `end_effector_flange`. **Reference target: body `end_effector_tip`, offset `Vector3d::Zero()`.** (Draft claim "panda_hand + Zero()" was incorrect — corrected here.)

## Port-side (Tier-2 (b), instrumented run)

### 1.f — port ee_frame identity

```
[EXEC-T2] ee_frame body='pusher' offset=[0.0, 0.0, 0.0]
          (reference target: 'end_effector_tip' + Vector3d::Zero())
```

Port's `ee_frame` = `plant.GetFrameByName("pusher")`. The "pusher" body is a programmatic rigid body (`sim/env_builder.py:276`) welded to `panda_link8` via `RigidTransform([0, 0, 0.05])` with a sphere-radius-0.025 m collision at identity offset. **Port target: body `pusher`, offset `[0, 0, 0]`.**

**Verdict**: Both agents target a zero-offset custom rigid tip body welded off the last actuated link. Body identity differs (`pusher` vs `end_effector_tip`), but the OSC-level Jacobian STRUCTURE (translational Jacobian of a body origin) is equivalent. Kp acts on `p_body − p_des` in both. Sphere-radius mismatch (port 0.025 m default vs reference 0.0195 m) is a separate 3-mm geometric divergence in the contact geometry — belongs in the (5) sim/env or (3) admission subsystem, not executor. **1.f (Jacobian identity) → CONFIRMED STRUCTURALLY-CONFORMANT; body-name divergence is cosmetic; sphere-radius divergence is out-of-scope for executor.**

### 1.e / 1.f — effective gains + compound authority (runtime)

```
[OSC-INIT]   Kp_cart=[400.0, 400.0, 400.0]  Kd_cart=[40.0, 40.0, 40.0]
[OSC-INIT]   W_track=100.0  W_posture=1.0  W_torque=0.001  W_acc=0.001
[OSC-INIT]   use_force_tracking=True  W_force=1.0
[EXEC-T2] compound_authority: pos=40000.0 force=1.0 ratio(pos:force)=40000.0:1
[EXEC-T2] c3_ref_gains_flag=False env_PUSHA_REF_OSC_ALIGN=0
          env_PUSHA_OSC_C3_MODE_REFERENCE_GAINS=0
```

**Verdict**: Compound position authority at runtime is exactly `W_track · Kp_cart[0] = 100 · 400 = 40000`; force authority = `W_force = 1.0`. Runtime ratio **40000:1**, matching the static-read prediction. No overrides active (`PUSHA_REF_OSC_ALIGN=0`, `PUSHA_OSC_C3_MODE_REFERENCE_GAINS=0`). Reference gains from `osc_params.yaml`: `W_end_effector=I_3, Kp_end_effector=diag(200), W_ee_lambda=I_3` → compound ratio **200:1**. **Port over-drives position by 200× vs reference. 1.e/1.f LOAD-BEARING → CONFIRMED at runtime.**

### 1.d — over-cap torque (runtime)

```
[EXEC-T2] step=60 t=0.60s
  u_task     = [ -3.36, -70.24,  34.04,  -1.71, -2.23,  1.91, -0.04]
  tau_g      = [  0.00, -27.20,  12.19,   7.86, -0.20,  1.16, -0.01]
  total_plant= [ -3.36, -97.44,  46.23,   6.15, -2.43,  3.07, -0.05]
  cap        = [ 87.00,  87.00,  87.00,  87.00, 12.00, 12.00, 12.00]
  task_over=0  plant_over=1  plant_over_joints=[1]  worst_j=1  worst_headroom_Nm=-10.44
```

**Verdict**: At sim step 60 (t=0.60 s), the port's QP produces `u_task[1] = -70.24 Nm` — task_over = 0 (inside cap 87 Nm). But `main.py:816` adds `tau_g[1] = -27.20 Nm`, and the plant's actuation port receives `total_plant[1] = -97.44 Nm`, **exceeding the URDF cap 87 Nm by 10.44 Nm**. The QP's per-joint bound `|u| ≤ tau_max` (`qp_builder.py:150-151`) is respected by the QP's own decision variable, but the mechanism divergence 1.d predicts EXACTLY this: **the plant sees `tau_g + u_clamped`, not just `u_clamped`**. The reference clamps a `u` that already includes gravity comp against the SAME cap (`inverse_dynamics_qp.cc:117-126` with `EnableGravityCompensation`), so the reference plant is guaranteed `|τ_plant| ≤ cap`. **1.d LOAD-BEARING → CONFIRMED at runtime.**

Frequency in the 4 s / step 0–250 sample window: 1 recorded over-cap event out of the ~250 QP-produced ticks, at joint 1 (shoulder — the largest gravity-load joint). Sparse but real. The port's `_saturation_events` counter (`operational_space_controller.py:307-311`) does NOT count this event because it only fires when `|u_opt| = tau_max`, not when `|tau_g + u_opt| > tau_max`. **This means the port's OSC-SUMMARY saturation percentage under-reports actual plant-side cap breaches.**

## Coupling reconfirmed (1.d ↔ 1.h; 1.e ↔ 1.f)

- **1.d ↔ 1.h**: The gravity-comp reintegration (1.d) turns the URDF-effort-limit constraint (1.h) into a `task-only` clamp — the plant can see torque exceeding the cap, and the port's own saturation counter can't detect this. RUNTIME-CONFIRMED with a 10.44 Nm over-cap breach at joint 1.
- **1.e ↔ 1.f**: The 40000:1 compound authority pushes the QP toward large `v̇` residual → large `u` → shoulder-joint overshoot when combined with `tau_g` post-clamp. RUNTIME-CONFIRMED that the static-read compound authority IS the runtime authority (no envvar override active in default config).

## Open scope-adjacent divergences noted for other subsystems
- Pusher sphere radius port 0.025 m vs reference 0.0195 m (out of executor scope; belongs to (5) sim/env). Memory `project_S9_leaked_to_box_stage_e_blocked.md` covers the prior globalization/regression story.
- Force-tracking `use_force_tracking=True` is a port-side add on top of reference's simpler position-only tracking — the reference DOES have an `ExternalForceTrackingData`, so this is aligned, but the port's `λ_des = magnitude · (−g_hat)` derivation (`sampling_based_c3_controller.py:365` per CLAUDE.md) is a port-only reduction that belongs in (2) sampling-c3 wrapper.
