# push_anything_ADMM — Conformance map (port vs reference)

**Purpose:** A subsystem-by-subsystem cold-read map of the divergences between the port (`push_anything_ADMM @ main dd2294d`) and the reference (`dairlib_sampling_c3 @ push_anything_dev 257e3ed`), each entry tagged for its dataflow-derived load-bearing verdict and confidence.

**Format:**
- **Tier 1 — Static read.** File-line evidence + dataflow reasoning. Every entry classified as `LOAD-BEARING`, `COSMETIC`, or `UNKNOWN`, with a confidence tag.
- **Tier 2 — Runtime confirmation.** Every Tier-1 `UNKNOWN` gets a reference-side deeper-read verdict (READ, no run) OR a port-side log-only instrumented-run verdict. Every Tier-1 `LOAD-BEARING` gets a runtime CONFIRMED / REFUTED tag using the actual value.
- **Coupling observed.** Consolidated cross-entry graph — the decision-critical bucket. Kept as a bucket (not merged into per-entry notes) so the coupling graph is visible at a glance.

Evidence is separate from verdict — instrumented run outputs live under `audit_output/<subsystem>_tier2/`, never in this doc.

**Status:** Subsystem (1) Executor Tier-1 approved 2026-07-14; Tier-2 landed 2026-07-14. Subsystems (2)–(5) pending.

---

# Subsystem (1) — Executor / torque path

## Sources read

- **Reference** (`dairlib_sampling_c3 @ push_anything_dev 257e3ed`):
  `systems/controllers/osc/{inverse_dynamics_qp.cc, operational_space_control.cc, external_force_tracking_data.cc, end_effector_position.cc, end_effector_force.cc, osc_tracking_data.cc, osc_gains.h}`;
  `examples/sampling_c3/{franka_osc_controller.cc, urdf/end_effector_full.urdf, sampling_c3_utils.h, shared_parameters/osc_params.yaml}`.
- **Port** (`push_anything_ADMM @ main dd2294d`):
  `control/osc/{qp_builder.py, operational_space_controller.py, dynamics_helpers.py}`;
  `main.py:772-816`; `config/osc_franka.yaml`; `sim/env_builder.py:270-303`.

**Executor scope:** what happens between "here is the EE-pos trajectory + λ_des command" and "here is τ to send to the arm." Excludes upstream planner and downstream sim.

## Index table

| # | Divergence | Tier-1 tag | Tier-2 |
|---|---|---|---|
| 1.a | Input trajectory contract | LOAD-BEARING (for traj-derivative fields) | (deferred — not in current-cold-read Tier-2 target set) |
| 1.b | Position PD command formula | COSMETIC in shape; LOAD-BEARING if `a_ff` differs | (couples to 1.a) |
| 1.c | Decision variables | LOAD-BEARING iff `n_h>0 ∨ n_c>0` | KNOWN-INERT (n_c=n_h=0 reference-side, grep-confirmed) |
| 1.d | Dynamics equality (gravity-comp reintegration) | LOAD-BEARING at input-limit boundary | CONFIRMED at runtime (plant sees 97.44 Nm at joint 1, cap=87 Nm) |
| 1.e | Position tracking cost (W·Kp) | LOAD-BEARING | CONFIRMED at runtime (compound authority 40000:1 vs ref 200:1) |
| 1.f | External-force tracking + variable coupling / frame identity | LOAD-BEARING (weight); UNKNOWN (frame) | Frame KNOWN — port `pusher`+Zero vs ref `end_effector_tip`+Zero (both zero-offset body Jacobians, structurally equivalent); weight/authority couples 1.e |
| 1.g | Contact & holonomic constraints (friction cone) | LOAD-BEARING iff n_c > 0 | KNOWN-INERT (n_c=0, grep-confirmed) |
| 1.h | Input effort limits | LOAD-BEARING per-joint | Couples to 1.d — CONFIRMED (plant-over-cap event mediated through 1.h clamp scope) |
| 1.i | Acceleration limits | COSMETIC (both disabled for push_anything) | — |
| 1.j | Acceleration regularization | LOAD-BEARING for conditioning; likely COSMETIC for behavior | (deferred) |
| 1.k | Input regularization + smoothing | LOAD-BEARING iff ref values ≠ 0 | (ref values are 0; port adds 1e-3 diagonal; COSMETIC by dataflow) |
| 1.l | Soft-constraint slack cost | COSMETIC (both inert) | — |
| 1.m | Joint-space tracking data (joint-2 pin) | LOAD-BEARING per gain values | Port `W_joint2=0.0` at runtime (deliberate default per memory `project_reproduce_dairlib_main_honest_option_a.md`) — CONFIRMED-INERT-BY-CONFIG |
| 1.n | Warm start / OSQP + failure handling | LOAD-BEARING on failure; COSMETIC on success | (QP-failure frequency = 0 in nominal runs per OSC-SUMMARY historical) |
| 1.o | Impact-invariant projection / state-switch handling | UNKNOWN → depends on FSM flag | KNOWN-INERT (`used_with_finite_state_machine=false` at construction) |
| 1.p | Extra output diagnostics | COSMETIC | — |

## 1.a — Input trajectory contract

- **Reference:** `franka_osc_controller.cc:149-188` + `operational_space_control.cc:183-201` — three abstract input ports declared as `drake::trajectories::Trajectory<double>`. Position, orientation (default OFF per YAML `track_end_effector_orientation=false`), and force trajectories arrive as pydrake trajectory objects; `osc_tracking_data.cc:88-108` calls `traj.value(t)`, `traj.EvalDerivative(t,1)`, `traj.EvalDerivative(t,2)` to extract `y_des`, `ydot_des`, `yddot_des`.
- **Port:** `operational_space_controller.py:366-408` — `compute_torque_from_trajectory(traj, t_sim, ...)` calls `np.asarray(traj.value(t_sim)).reshape(3)` and delegates to `compute_torque(p_ee_desired=...)`. **Never evaluates `ydot_des` or `yddot_des`** from the traj object.
- **Tag:** LOAD-BEARING for the traj-derivative fields.
- **Dataflow:** `ydot_des` feeds `error_ydot = ydot_des − ydot_actual` at `osc_tracking_data.cc:65` via `UpdateYdotError()`, which then enters `yddot_command = yddot_des + Kp·error_y + Kd·error_ydot` (line 115). `yddot_des` feeds the same equation directly as the feedforward-accel term. In the port, `v_ee_desired` may be passed as an EXPLICIT kwarg by the wrapper but is NOT extracted from the traj; wrapper-level derivative info is present only if the caller supplies it. **Divergence at the interface contract**: a caller passing a real N-knot PWL to the port's `compute_torque_from_trajectory` loses `ydot_des` and `yddot_des` even though the traj object carries them, because the port's method drops those calls.
- **Confidence:** high.
- **Tier 2:** deferred — not in the current instrument set. Belongs in the Stage-A PWL-trajectory reposition subsystem (2) Tier-2, where the derivative-carrying PWL is actually fed.

## 1.b — Position PD command formula

- **Reference:** `osc_tracking_data.cc:113-116` — `yddot_command = yddot_des_converted + Kp·error_y + Kd·error_ydot`.
- **Port:** `qp_builder.py:159-163` — `a_des = a_ff + Kp_cart·p_err + Kd_cart·v_err` (with `a_ff = 0` if caller omits it).
- **Tag:** COSMETIC in symbolic form (same PD structure) but LOAD-BEARING if `a_ff` differs.
- **Dataflow:** Both feed the tracking residual `J·v̇ + J̇·v − a_des`. If the port's caller never supplies `a_ff` (see 1.a — the trajectory-shaped path drops derivative extraction), then `a_ff=0` while reference's `yddot_des` from a real trajectory would be non-zero. Under the current port wrapper wiring (single-knot ZOH trajectory), `traj.EvalDerivative(t,2)` would return zero anyway, so this divergence is inert for a constant traj but becomes real once a PWL traj is used.
- **Confidence:** high.
- **Tier 2:** couples to 1.a; deferred to subsystem (2) Tier-2.

## 1.c — Decision variables

- **Reference:** `inverse_dynamics_qp.cc:78-83` — `[dv (n_v), u (n_u), λ_h (n_h), λ_c (n_c), λ_e (n_e), ε (n_c_active)]`. Six variable blocks.
- **Port:** `qp_builder.py:120-141` — `[vdot (n_v), u (n_u)]` always; `λ_ext (3)` added only when `use_force_tracking=True`. Two or three variable blocks. No `λ_h`, no `λ_c`, no `ε`.
- **Tag:** LOAD-BEARING **only if** `n_h > 0` or `n_c > 0` in the push_anything OSC configuration; otherwise the missing variables carry zero information and cost the same amount to omit.
- **Dataflow:** `franka_osc_controller.cc:149-200` shows the constructor does NOT call `osc->AddContactPoint(...)` or `osc->AddKinematicConstraint(...)`. Tier-2 grep pass required to confirm no other call site adds contact/holonomic constraints.
- **Confidence (Tier 1):** medium.
- **Tier 2 — reference-side grep, 2026-07-14:** `grep -rn "AddContactPoint\|AddKinematicConstraint" examples/sampling_c3/` returns **0 hits** in the entire example directory. `SetContactFriction(osc_params.mu)` is called at `franka_osc_controller.cc:197` but sets μ globally without registering a contact point. **Reference OSC for push_anything: n_c = 0, n_h = 0.** Missing port variables (`λ_h, λ_c, ε`) are zero-size and their cost terms + friction-cone constraints are strict no-ops. **1.c → KNOWN-INERT.** UNKNOWN → RESOLVED.

## 1.d — Dynamics equality constraint

- **Reference:** `inverse_dynamics_qp.cc:213-225` — `M·dv − B·u − Jh^T·λ_h − Jc^T·λ_c − Je^T·λ_e = −bias`; if `with_gravity_compensation_=true`, `bias = C·v − grav` (subtracts gravity so gravity comp is inside the QP-solved u). Sim path per YAML `cancel_gravity_compensation: false` → `EnableGravityCompensation` → gravity comp IS inside the QP-solved u.
- **Port:** `qp_builder.py:132-141` — `M·v̇ − B·u − Jv^T·λ_ext = F_ff_external − bias`; `operational_space_controller.py:224-229` sets `bias = Cv` (no `−grav` subtraction — QP-solved u is TASK-ONLY). `main.py:774, 815` computes `tau_g = −CalcGravityGeneralizedForces(...)` and applies `total_torque = tau_g[:n_u] + u_opt` on the actuation port.
- **Tag:** LOAD-BEARING at the input-effort-limit boundary; COSMETIC elsewhere.
- **Dataflow:** In the reference sim path, the `u_min ≤ u ≤ u_max` constraint (`inverse_dynamics_qp.cc:117-126`) clamps a u that INCLUDES gravity comp — so gravity-comp-torque counts against the URDF cap. In the port, the QP clamps a TASK-ONLY u; `main.py:815` adds gravity comp on top of the clamped u, so the plant sees `tau_g[:n_u] + u_clamped`, which can exceed the URDF cap. Also affects the interpretation of the `saturated` diagnostic (`operational_space_controller.py:307-311`) — port saturation counter fires when task torque hits cap, while reference saturation would fire when task+gravity hits cap. These are DIFFERENT thresholds.
- **Confidence (Tier 1):** high.
- **Tier 2 — port-side instrumented run, 2026-07-14:** `PUSHA_EXEC_T2_DIAG=1` capture at step 60 (t=0.60 s, pushing task-id 4, seed 0, c3plus, admm-iter 25, `config/sampling_c3_kik.yaml`):
  ```
  u_task[1]      = -70.24 Nm   (task_over = 0; within cap 87 Nm)
  tau_g[1]       = -27.20 Nm
  total_plant[1] = -97.44 Nm   → exceeds cap by 10.44 Nm
  plant_over_joints = [1]  worst_headroom_Nm = -10.44
  ```
  **1.d → CONFIRMED at runtime.** Joint 1 (shoulder) sees over-cap torque 10.44 Nm above URDF spec because the port's task-only QP clamp is added to gravity-comp AFTER the clamp. Reference-side would clamp `u_effective` (with gravity comp) at the same 87 Nm cap. Frequency in the 4 s / 250-tick window: 1 recorded event at joint 1. Rare but real; port's own `_saturation_events` counter is BLIND to this event because it only tests `|u_opt| == tau_max`, not `|tau_g + u_opt| > tau_max`. Evidence: `audit_output/exec_tier2/SUMMARY.md`.

## 1.e — Position tracking cost

- **Reference:** `operational_space_control.cc:454-461` — `2·J^T·W·J` and `2·J^T·W·(JdotV − ydd_cmd)`, added as a QuadraticCost on `dv`. `W = W_end_effector = diag(1,1,1)` per `osc_params.yaml:47-50`. `Kp = diag(200,200,200)`, `Kd = diag(20,20,20)`.
- **Port:** `qp_builder.py:167-170` — `2·W_track·(J^T·J)` and `2·W_track·J^T·(JdotV − a_des)`, added as QuadraticCost on `vdot`. `W_track` is a scalar (multiplies identity). Under current YAML `W_track=100.0`, `Kp_cart=[400,400,400]`, `Kd_cart=[40,40,40]`.
- **Tag:** LOAD-BEARING.
- **Dataflow:** Both feed the QP Hessian on `dv`/`vdot` — a scalar-`W` port × Kp differs from a diagonal-`W` × Kp reference only if W is not proportional to I, which it isn't (both are I·scalar in practice). What matters is the **scalar product `W·Kp`** — the compound position authority. Port `100 × 400 = 40 000` vs reference `1 × 200 = 200`. The compound authority sets how hard the QP pushes vdot to close p_err.
- **Confidence (Tier 1):** high.
- **Tier 2 — port-side instrumented run, 2026-07-14:**
  ```
  [OSC-INIT]   Kp_cart=[400.0, 400.0, 400.0]  Kd_cart=[40.0, 40.0, 40.0]
  [OSC-INIT]   W_track=100.0
  [EXEC-T2] compound_authority: pos=40000.0 force=1.0 ratio(pos:force)=40000.0:1
  [EXEC-T2] c3_ref_gains_flag=False env_PUSHA_REF_OSC_ALIGN=0
            env_PUSHA_OSC_C3_MODE_REFERENCE_GAINS=0
  ```
  **1.e → CONFIRMED at runtime.** Compound authority = `100 × 400 = 40000`, matching the static-read prediction exactly. No envvar overrides active in the default config. Reference compound authority = `1 × 200 = 200`. **Port over-drives position by 200× vs reference.** The `PUSHA_REF_OSC_ALIGN` and `PUSHA_OSC_C3_MODE_REFERENCE_GAINS` flags (`operational_space_controller.py:113, 147`) are the two levers that would collapse this ratio; neither is default-on and neither activated in this run.

## 1.f — External-force tracking cost + variable coupling

- **Reference:** `external_force_tracking_data.cc:33-41` extracts `λ_des = traj.value(t)` (3-D). `operational_space_control.cc:469-484` adds cost `2·W·‖λ_e − λ_des‖²` on the `λ_e` variable. `W = W_ee_lambda = I_3` per `osc_params.yaml:74-77`. The `Je^T·λ_e` term appears in the dynamics equality (1.d), so the QP couples `dv, u, λ_e` through dynamics.
- **Port:** `qp_builder.py:198-206` — `2·W_force·I_3` and `−2·W_force·λ_des`, added as QuadraticCost on `lam_ext`. `W_force=1.0` per current YAML `config/sampling_c3_kik.yaml`. `Jv^T·λ_ext` appears in dynamics (1.d). Cost term active only when `use_force_tracking=True`.
- **Tag:** LOAD-BEARING (cost weight and dynamics coupling).
- **Dataflow:** Same mechanism symbolically. The reference's Je is built from `ExternalForceTrackingData(...)` targeting `kEndEffectorName` at `Vector3d::Zero()` offset (`franka_osc_controller.cc:170`). The port uses `J_v = ee_jacobian_translational(plant, plant_ctx, self.ee_frame)` (`operational_space_controller.py:219-220`) — Drake's translational Jacobian evaluated at whatever `ee_frame` was passed at construction.
- **Confidence (Tier 1):** medium (interface behavior verified; frame identity not cross-checked in the cold read).
- **Tier 2 — reference-side deep read + port-side instrumented run, 2026-07-14:**
  - **Reference frame:** `sampling_c3_utils.h:18`: `kEndEffectorName = "end_effector_tip"`. `end_effector_full.urdf:57` defines `end_effector_tip` as a rigid link with a sphere-radius-0.0195 m collision at zero-offset. Reference `ExternalForceTrackingData` at `franka_osc_controller.cc:170` targets `("end_effector_tip", Vector3d::Zero())`. **Reference target: body `end_effector_tip`, offset `Vector3d::Zero()`.** (The Tier-1 draft's "panda_hand + Zero()" assumption is corrected here.)
  - **Port frame:**
    ```
    [EXEC-T2] ee_frame body='pusher' offset=[0.0, 0.0, 0.0]
    ```
    `main.py:611` uses `plant.GetFrameByName("pusher")`. The `"pusher"` body is a programmatic rigid body (`sim/env_builder.py:276`) welded to `panda_link8` via `RigidTransform([0, 0, 0.05])`, with a sphere-radius-0.025 m collision at identity offset. **Port target: body `pusher`, offset `[0, 0, 0]`.**
  - **Verdict:** Both agents target a **zero-offset** rigid tip body welded downstream of the last actuated arm link. The Jacobian STRUCTURE (Drake's translational Jacobian of a body origin) is equivalent under both. Body identity differs (`pusher` vs `end_effector_tip`) but the OSC-level meaning is preserved for the position tracking + force injection paths. **1.f Jacobian identity → CONFIRMED STRUCTURALLY-CONFORMANT.** The sphere-radius mismatch (port 0.025 m default vs reference 0.0195 m) is a separate 3-mm contact-geometry divergence that belongs to (5) sim/env or (3) admission subsystem — flagged for those Tier-1 passes.
  - **W_force at runtime:** `W_force=1.0` (matches reference `W_ee_lambda=I_3` scalar). The compound-authority divergence stays in the position side (1.e); the force side is aligned by number.

## 1.g — Contact & holonomic constraints (variables + friction cone)

- **Reference:** `inverse_dynamics_qp.cc:35-73, 97-115` — `λ_h` enforced by `Jh·dv + Jh·v = 0`; `λ_c` per contact constrained by a 5-facet linearized friction cone with friction `μ` (`SetContactFriction(mu)`, `franka_osc_controller.cc:197`); `Jc_active·dv + ε = −Jc_active·v` for the tangential-direction equality-with-slack.
- **Port:** absent.
- **Tag:** LOAD-BEARING **iff** the reference OSC has any active contact/holonomic constraints for push_anything.
- **Confidence (Tier 1):** medium.
- **Tier 2 — see 1.c:** same grep-pass verdict. n_c = n_h = 0 for push_anything; the reference's friction-cone / holonomic-equality constraints are zero-count under this configuration. `SetContactFriction` sets μ globally without registering a contact point. **Port omission is a strict no-op. 1.g → KNOWN-INERT.**

## 1.h — Input effort limits

- **Reference:** `inverse_dynamics_qp.cc:117-126` — `u_min[i] = -plant.get_joint_actuator(i).effort_limit()`, symmetric bounds sourced from Drake URDF-parsed limits. `with_input_constraints_=true` by default.
- **Port:** `qp_builder.py:150-151, 138` — `AddBoundingBoxConstraint(-limits.tau_max, limits.tau_max, u)`; `limits.tau_max` sourced from `config/osc_franka.yaml:45` (`[87,87,87,87,12,12,12]`) or URDF (`operational_space_controller.py:117-128` precedence: override > yaml > URDF).
- **Tag:** LOAD-BEARING per-joint (interacts with the gravity-comp mechanism divergence in 1.d).
- **Dataflow:** Both clamp `u`; VALUES appear to match the Franka URDF numeric spec (87/12 Nm). Effective clamping differs due to the gravity-comp reintegration difference (1.d).
- **Confidence (Tier 1):** high.
- **Tier 2 — port-side instrumented run, 2026-07-14:**
  ```
  [OSC-INIT]   tau_max=[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
  ```
  Port cap matches URDF/reference numeric spec exactly. **1.h → CONFIRMED (values conformant); however the CLAMP SCOPE differs (1.d) — see 1.d Tier-2 for the runtime consequence (plant torque 10.44 Nm over the cap at joint 1).**

## 1.i — Acceleration limits

- **Reference:** `inverse_dynamics_qp.cc:128-144` — optional; when `enforce_acceleration_constraints=true`, per-joint `[ddq_min, ddq_max]` from `plant.get_joint(i).acceleration_lower_limits()`. Per YAML `enforce_acceleration_constraints: false` → not enforced.
- **Port:** absent.
- **Tag:** COSMETIC (both effectively disabled for push_anything).

## 1.j — Acceleration regularization

- **Reference:** `operational_space_control.cc:292-296` — cost `W_joint_accel·‖dv‖²`, weight matrix built via `osc_gains.h:58` as `W_acceleration = w_accel · diag(W_accel)`. YAML: `w_accel=1e-5`, `W_accel=[0.01]×7` → **effective weight per axis = 1e-7**.
- **Port:** `qp_builder.py:187-190` — cost `W_acc·‖v̇‖²` on all `n_v` variables. YAML `W_acc = 0.001`.
- **Tag:** LOAD-BEARING for numerical conditioning; likely COSMETIC for behavior at this magnitude ratio.
- **Dataflow:** Port's `W_acc=1e-3` is **10 000× stronger** than reference's `1e-7` effective. Both are small compared to tracking weights so behavior effect is likely minor, but a 10 000× gap is worth noting; larger reg pushes the QP toward `v̇=0` when tracking is under-determined.
- **Confidence:** high (magnitude); medium (whether the difference is behaviorally observable — needs a run).

## 1.k — Input regularization + input smoothing

- **Reference:** `operational_space_control.cc:287-290, 297-300, 540-544` — `input_cost = W_input·‖u‖²` and `input_smoothing_cost = W_input_smoothing·‖u − u_prev‖²`. Both weight matrices built from `osc_gains.h:59-60` as `W_input_reg = w_input · diag(W_input_reg)`, `W_input_smoothing = w_input_reg · diag(W_input_reg)`. YAML: `w_input=0`, `w_input_reg=0`, `W_input_reg=[1,1,1,1,1,1,10]` → **both matrices = 0** for push_anything.
- **Port:** `qp_builder.py:182-185` — `W_torque·‖u‖²`. YAML `W_torque = 0.001`. No smoothing cost.
- **Tag:** LOAD-BEARING only if the reference values weren't zero. Given they ARE zero for push_anything, the port's `W_torque=1e-3` is a **superset** — it adds a tiny reg the reference doesn't have.
- **Dataflow:** Reference u has no regularization for push_anything; port has a tiny (`1e-3`) diagonal reg. Adds a mild bias against large u when tracking is under-determined. Likely COSMETIC; UNKNOWN whether it changes anything measurable.
- **Confidence:** high (magnitudes); low (behavioral significance).

## 1.l — Soft-constraint slack cost

- **Reference:** `operational_space_control.cc:316-321` — `w_soft_constraint·‖ε‖²` when `w_soft_constraint > 0`. YAML `w_soft_constraint: 0.0` → term inert.
- **Port:** absent (no `ε` variable — see 1.c).
- **Tag:** COSMETIC (both inert for push_anything).

## 1.m — Joint-space tracking data (joint-2 pin)

- **Reference:** `franka_osc_controller.cc:159-165, 185-186` — `AddConstTrackingData(JointSpaceTrackingData("panda_joint2_target"), 1.1·VectorXd::Ones(1))`. Gains from YAML: `elbow_kp=200, elbow_kd=10, w_elbow=1`. The tracking-data cost mechanism (`operational_space_control.cc:459-461`) gives `2·J^T·W·J·dv − 2·J^T·W·(JdotV − ydd_cmd)`; for a joint-space tracking data on joint 2, `J = e_j2^T` (unit selector) and the cost reduces to `W·‖v̇[j2] − a_j2‖²` with `a_j2 = 1·200·(1.1 − q_j2) + 1·10·(−v_j2)`.
- **Port:** `qp_builder.py:215-226` — same cost structure via joint-2 fields (`Kp_joint2, Kd_joint2, W_joint2, joint2_target_rad, joint2_idx`). Currently `W_joint2=0.0` per `config/osc_franka.yaml:34` → cost term inert.
- **Tag:** LOAD-BEARING per gain values.
- **Dataflow:** Both are quadratic on `dv[j2]`. Reference is always-on for push_anything; port is presently OFF via `W=0`. This is a DELIBERATE port state — see memory `project_reproduce_dairlib_main_honest_option_a.md` (`W_joint2 → 0.0` via Option A recert 2026-07-14). Reintroducing the reference's `w_elbow=1, Kp=200` is what the "characterize-and-write vs coupled-re-tune" scope decision (blocked on user + Atanasov) will answer.
- **Confidence:** high.

## 1.n — Warm start / OSQP + failure handling

- **Reference:** `operational_space_control.cc:556-576` — `FastOsqpSolver` with configured solver options; on solve failure, sets `u_prev_ = 0.99·u_sol + random(n_u)` and disables warm start (retry next tick). Warm start also disabled on FSM state transitions (line 386-389).
- **Port:** `operational_space_controller.py:175, 285-303` — Drake's stock `OsqpSolver` reused across ticks; on failure, `u_opt = zeros(n_u)`, `_qp_failures += 1` diagnostic incremented. No warm-start disable, no perturbation retry.
- **Tag:** LOAD-BEARING on failure; COSMETIC on success.
- **Dataflow:** On QP failure, reference commands a slightly-perturbed prior-tick solution; port commands ZERO torque. Different behavior at failure. Frequency of failure per current runs = 0 (from OSC-SUMMARY lines observed at run-time; not from a fresh audit run) — so the divergence is inert in nominal operation.
- **Confidence:** high (structure); medium (behavioral impact frequency — inert observed but not guaranteed).

## 1.o — Impact-invariant projection / state-switch handling

- **Reference:** `operational_space_control.cc:394-397, 411-423, 589-685` — near FSM impacts (`alpha != 0`), computes an invariant projection `v_proj = alpha·M⁻¹Jᵀ·λ_ii` and injects it into tracking-data updates.
- **Port:** absent. No FSM, no impact projection.
- **Tag:** UNKNOWN — depends whether the sampling-c3 diagram wires the OSC with `used_with_finite_state_machine=true` for push_anything.
- **Confidence (Tier 1):** low.
- **Tier 2 — reference-side grep, 2026-07-14:** `franka_osc_controller.cc:138`: `builder.AddSystem<OperationalSpaceControl>(plant, plant_context.get(), false)`. Third argument `used_with_finite_state_machine = false`. The impact-projection block at `operational_space_control.cc:394-397, 411-423, 589-685` is guarded by `used_with_finite_state_machine_` (verified at `operational_space_control.cc:70, 693, 812` and header `operational_space_control.h:47, 101-103, 399`) and never executes. **Port omission is inert reference-side. 1.o → KNOWN-INERT.** UNKNOWN → RESOLVED.

## 1.p — Extra output diagnostics

- **Reference:** `operational_space_control.cc:687-795, 92-99` — publishes an `lcmt_osc_debug` abstract output port + a `failure_signal` binary output.
- **Port:** `operational_space_controller.py:343-362, 411-421` — returns a Python diag `dict` + prints `[OSC-SUMMARY]` line at end.
- **Tag:** COSMETIC.

---

## Coupling observed (from code + Tier-2 evidence)

- **1.a ↔ 1.b** — the interface contract (traj-shaped input) and the PD command formula are coupled: the port drops `ydot_des` and `yddot_des` extraction at 1.a, which then propagates as `a_ff = 0` at 1.b. Fixing one without the other breaks the contract. Not currently observable (single-knot ZOH traj), but becomes real the moment a PWL traj is fed — belongs in the Stage-A PWL trajectory reposition subsystem review.
- **1.d ↔ 1.h** — the gravity-comp reintegration in the wrapper (1.d) and the input effort clamp (1.h) couple: port's clamp acts on task-only τ, then gravity is added AFTER the clamp, so the plant can see τ that exceeds the URDF cap. **Tier-2 CONFIRMED at runtime**: at step 60 of a default `pushing --task-id 4 --seed 0` run, joint 1 sees `total_plant = -97.44 Nm` (cap = 87 Nm), a −10.44 Nm headroom breach. The port's own `_saturation_events` counter cannot detect this because it tests `|u_opt| == tau_max` (task-only), not `|tau_g + u_opt| > tau_max` (plant-side). Reference-side clamp includes gravity so `|τ_plant| ≤ cap` by construction.
- **1.e ↔ 1.f** — position tracking `W·Kp` (1.e) and force tracking `W_force` (1.f) compete on `dv` and `λ_e` respectively, both feeding the dynamics equality. Their RATIO is the "compound position:force authority" quantity. **Tier-2 CONFIRMED at runtime**: port ratio 40000:1 with default gains vs reference 200:1. The 200× position over-drive is the standing hypothesis for why the OSC hammers joint 1 into over-cap territory (couples 1.e/1.f → 1.d).
- **1.c ↔ 1.g ↔ 1.o** — the three "missing infrastructure" divergences (contact variables, friction-cone constraints, impact projection) collapse together: **all three are KNOWN-INERT** because push_anything's reference OSC has `n_c = n_h = 0` and `used_with_finite_state_machine = false`. Any future push_anything reference variant that turns these on would flip all three back to LOAD-BEARING in lockstep.

**Not observed as coupled (cold read):**
- 1.m (joint-2 pin) has no code-visible coupling to 1.e or 1.f beyond both being cost terms on `dv`. Whether the pin fights position tracking at a given q_j2 depends on the physical posture — not visible from cost-term structure alone; would need a coupled-re-tune run to observe. (Deferred pending user + Atanasov scope call — see memory `project_reproduce_dairlib_main_honest_option_a.md`.)

## Deferred / out-of-executor-scope items surfaced

- Pusher sphere radius (port 0.025 m default vs reference 0.0195 m) — belongs to (5) sim/env. Memory `project_S9_leaked_to_box_stage_e_blocked.md` covers the prior globalization-and-revert incident.
- Force-tracking `λ_des = magnitude · (−g_hat)` derivation (`sampling_based_c3_controller.py:365`) — belongs to (2) sampling-c3 wrapper.
- Stage-A PWL reposition trajectory feeding (`PUSHA_REPOSITION_PWL=1`) — the path that would exercise the 1.a/1.b coupling; belongs to (2) reposition.

## Executor Tier-2 evidence artefacts

- Diagnostic commits: `ce29c9f` (initial), `bde8d64` (always-log-on-over-cap).
- Instrumentation guard: `PUSHA_EXEC_T2_DIAG=1` — default OFF, byte-identical to `dd2294d` baseline.
- Full run stdout: `audit_output/exec_tier2/run_default_full.log`
- Filtered T2/OSC-INIT lines: `audit_output/exec_tier2/run_default.log`
- Human-readable summary: `audit_output/exec_tier2/SUMMARY.md`

## Executor Tier-2 verdict roll-up

| # | Divergence | Tier-1 | Tier-2 (this pass) |
|---|---|---|---|
| 1.c | Contact/holonomic variables | UNKNOWN | **KNOWN-INERT** (n_c=n_h=0 grep-confirmed) |
| 1.d | Gravity-comp reintegration | LOAD-BEARING | **CONFIRMED** (plant +10.44 Nm over cap at joint 1, step 60) |
| 1.e | W·Kp compound authority | LOAD-BEARING | **CONFIRMED** (runtime 40000:1 vs reference 200:1) |
| 1.f | Frame identity + force weight | UNKNOWN (frame) | **STRUCTURALLY CONFORMANT** (both zero-offset body Jacobians; body names differ; force W aligned at 1.0) |
| 1.g | Friction cone / kin constraints | UNKNOWN | **KNOWN-INERT** (n_c=0 grep-confirmed) |
| 1.o | Impact projection | UNKNOWN | **KNOWN-INERT** (`used_with_finite_state_machine=false`) |

Six UNKNOWNs/LOAD-BEARINGs → six CONFIRMED verdicts. Zero remaining executor-subsystem gaps in the current cold-read set.

---

# Subsystem (2) — Reposition

## Sources read

- **Reference** (`dairlib_sampling_c3 @ push_anything_dev 257e3ed`):
  `examples/sampling_c3/reposition.{cc,h}`;
  `examples/sampling_c3/parameter_headers/reposition_params.h`;
  `examples/sampling_c3/anything/parameters/reposition_params.yaml`;
  `examples/sampling_c3/push_t/parameters/reposition_params.yaml`;
  `systems/controllers/sampling_based_c3_controller.cc:1838-1928` (wrapper's `UpdateRepositioningExecutionTrajectory`);
  `systems/trajectory_optimization/lcm_trajectory_systems.cc:29-75` (`LcmTrajectoryReceiver`);
  `systems/controllers/osc/end_effector_orientation.cc` (`SetTrackOrientation` semantics);
  `examples/sampling_c3/franka_osc_controller.cc:125-135` (orientation-generator wiring).
- **Port** (`push_anything_ADMM @ main dd2294d + ce29c9f/bde8d64/94774fe/64ffdee`):
  `control/sampling_c3/reposition.py` (`PiecewiseLinearTracker`, `next_waypoint`);
  `control/sampling_c3/reposition_ik.py` (`RepositionIKTracker`, `_solve_single_knot_ik`);
  `control/sampling_c3/reposition_trajectory.py` (`RepositionTrajectory`);
  `control/sampling_c3/params.py:81-93, 410-490` (`RepositioningTrajectoryType`, `RepositionParams`);
  `control/sampling_c3/sampling_based_c3_controller.py:155-197` (tracker dispatch),
  `:266-322` (Stage-A PWL wiring), `:2200-2270` (per-tick tracker call),
  `:3000-3163` (free-mode OSC dispatch);
  `config/sampling_c3_kik.yaml` (runtime traj_type=kPiecewiseLinear).

**Reposition scope:** what happens between "planner has emitted a reposition target for this tick" and "the OSC receives its per-tick `(p_ee_desired [, v_ee_desired])`." Excludes the OSC's per-tick QP (subsystem 1) and the mode-switch decision (subsystem 4's dispatcher).

## Index table

| # | Divergence | Tier-1 tag | Tier-2 |
|---|---|---|---|
| 2.a | Reposition tracker dispatch | LOAD-BEARING (structure) | CONFIRMED — port default = `PiecewiseLinearTracker` (kPiecewiseLinear); wrapper overrides its `u` with OSC output → tracker exists only to compute `p_des` |
| 2.b | Knot construction (analytic Reposition() vs port setpoint march) | LOAD-BEARING | CONFIRMED — reference builds all N knots analytically in one call; port default marches ONE setpoint per tick |
| 2.c | Trajectory → OSC interface (LCM PP with FirstOrderHold vs Python (p_des,v_des)) | LOAD-BEARING (derivative) | CONFIRMED — default port drops v_des; Stage-A port wires v_des; reference FirstOrderHold carries piecewise-constant first derivative |
| 2.d | `filtered_solve_time_` future-offset | LOAD-BEARING for cadence | UNKNOWN → REFERENCE-CONFIRMED (offset is real, `sampling_based_c3_controller.cc:1390-1391, 1718`); port has no analog |
| 2.e | `RepositioningTrajectoryType.kIK` (port-only) | LOAD-BEARING iff kIK selected | CONFIRMED — kik.yaml selects `kPiecewiseLinear`, so kIK code path INERT for default box run |
| 2.f | admit_active latch (`ADMIT_LATCH_TICKS=8`, port-only) | LOAD-BEARING (kIK) | CONFIRMED-INERT — `PiecewiseLinearTracker.compute_torque` consumes-and-ignores `admit_active`; runtime `latch=0/0` |
| 2.g | descent-gate stability counter (`TARGET_STABLE_TICKS=5`, `TARGET_STABLE_TOL=5mm`, port-only) | LOAD-BEARING (kIK) | CONFIRMED-INERT (kPWL path); `target_stable=0/0 allow_descent=1` at runtime |
| 2.h | `speed` field mismatch | LOAD-BEARING | CONFIRMED — port `speed=0.4` m/s vs reference `speed=0.18` m/s (2.22× over); PWL-path `pwl_speed=0.18` matches reference |
| 2.i | `pwl_waypoint_height` mismatch | LOAD-BEARING | CONFIRMED — port `0.15` m vs reference `0.06-0.077` m (~2×) |
| 2.j | Reposition-mode `v_ee_desired` handshake to OSC | LOAD-BEARING (couples exec 1.a) | CONFIRMED — default legacy path passes `None` (matches exec-1.a divergence); Stage-A path passes `v_des` (aligned) |
| 2.k | Rotation tracking cost (reference identity-quaternion hold vs port none) | LOAD-BEARING (surfaced by S2 cold read; corrects executor 1.p) | REFERENCE-CONFIRMED — reference OSC applies `W_rot·Kp_rot = 10·800 = 8000` compound authority tracking identity quaternion during ALL modes; port QP has NO rotation cost |
| 2.l | `finished_reposition_flag` semantics | LOAD-BEARING | REFERENCE-KNOWN (reference sets it when knot 1 reaches the goal in a single-step arrival; port uses euclidean tol OR PWL time-elapsed + tol) |
| 2.m | `is_doing_c3` boolean threading into Reposition() | LOAD-BEARING semantics | REFERENCE-KNOWN (reference passes it as arg; port has no analog — reposition planning is called only from `free` mode branch) |
| 2.n | max_tilt_angle / tilted-quaternion trajectory generation | UNKNOWN | REFERENCE-CONFIRMED-INERT — reference wrapper builds a tilted-quaternion trajectory (`sampling_based_c3_controller.cc:1896-1917`) but `EndEffectorOrientationTrajectoryGenerator` OVERRIDES with identity when `track_orientation_=false` (per shared YAML). Tilt code is dead reference-side; port omission is inert. |

## 2.a — Reposition tracker dispatch

- **Reference:** No "tracker" concept. `sampling_based_c3_controller.cc:1848` calls `Reposition(...)` once per wrapper tick (planner rate), produces `Eigen::MatrixXd knots (n_x × N_)`, writes into the LCM `end_effector_position_target` trajectory. The OSC (subsystem 1) is the sole per-tick torque generator for both c3 and reposition modes.
- **Port:** `sampling_based_c3_controller.py:155-197` dispatches on `params.reposition_params.traj_type`:
  - `kIK` → `RepositionIKTracker` (per-knot IK, joint-PD torque, admit-latch, descent-gate, target-stability counter). Port-only.
  - Everything else (`kSpline`, `kSpherical`, `kCircular`, `kPiecewiseLinear`) → `PiecewiseLinearTracker` (per-tick setpoint march via `next_waypoint`, IK to q_des, joint-PD torque). Only `kPiecewiseLinear` is actually implemented in the wrapper's dispatch; other enum values fall through to PWL.
  In both cases the tracker's torque output is **discarded** at wrapper.py:3126,3154 — the wrapper takes `free_diag.get("p_des")` (or `RepositionTrajectory.eval(sim_t).p_des`) and calls `self.executor.compute_torque(p_ee_desired=p_des, ...)`. So the tracker exists **only to compute p_des**.
- **Tag:** LOAD-BEARING (structural).
- **Dataflow:** The reference has ONE torque generator (OSC) and analytic knot construction. The port has TWO parallel construction paths (per-tick tracker `p_des` OR Stage-A PWL trajectory `(p_des, v_des)`) that feed a single OSC. When `PUSHA_REPOSITION_PWL=0` (default), the OSC receives per-tick `p_des` snapshots with no velocity. When `=1`, the OSC receives `(p_des, v_des)` from a persisted PWL trajectory.
- **Confidence:** high.
- **Tier 2 — port-side runtime capture:**
  ```
  [REPOS-T2] tracker=PiecewiseLinearTracker traj_type=kPiecewiseLinear
             use_pwl_traj=False env_PUSHA_REPOSITION_PWL=0
  ```
  **Default box-run path**: `PiecewiseLinearTracker`, per-tick setpoint march, no PWL trajectory. **2.a → CONFIRMED.** The "which tracker" dispatch and "which reposition path" are structural divergences from the reference's single-generator model, but the RESULT for the OSC is a single Cartesian target — which is the reference-equivalent interface. Load-bearing HOW the target is computed, but not WHAT the OSC receives (subject to 2.c).

## 2.b — Knot construction

- **Reference:** `reposition.cc:13-80` — `Reposition(...)` returns `Eigen::MatrixXd knots (n_x × N_)` in ONE analytic call per wrapper tick, containing ALL N future knots. The 5 sub-functions (`RepositionStraightLine`, `RepositionSpline`, `RepositionSpherical`, `RepositionCircular`, `RepositionPiecewiseLinear`) each fill the whole knot matrix based on their trajectory type. Knot i corresponds to sim time `t_context + filtered_solve_time_ + i·dt`.
- **Port (legacy PWL tracker path)**: `reposition.py:33-132` — `next_waypoint(...)` computes ONE Cartesian setpoint each control tick, based on the tracker's remembered `self._setpoint_pos`. No horizon-scale trajectory built in advance; the setpoint marches at `speed·dt` per tick along the 3-phase (lift/traverse/descend) path.
- **Port (Stage-A PWL path)**: `reposition_trajectory.py:22-140` — `RepositionTrajectory(p_start, p_target, z_safe, speed, t_start)` builds a full 3-leg PWL with K∈{2,3,4} knots ONCE per rebuild trigger (target change > 5 mm or c3→free transition). Then `eval(sim_t) → (p_des, v_des, done)` samples this trajectory each control tick. **This is the reference-analog.**
- **Tag:** LOAD-BEARING (per-tick vs pre-computed; ONE-generator vs shared).
- **Dataflow:** Reference's knot matrix is fed to the LCM traj, which the OSC's TransTaskSpaceTrackingData consumes at time `t=now`, extracting `y_des, ydot_des` at the current tick's sim time. Port's legacy path bypasses the trajectory: the tracker directly hands the current-tick `p_des` to the OSC, with no way to derive `ydot_des`. Port's Stage-A path builds the trajectory (like reference) and derives velocity from it (like reference).
- **Confidence:** high.
- **Tier 2:** Structural divergence CONFIRMED by cold read. Runtime behavior confirmed by `[STAGE-A-PWL] step=1 build ... K=4 t_end=1.415` (Stage-A) vs `[STAGE-A-TRACE] step=1 ... mode=free` with no `build` line (default path) — see `audit_output/repos_tier2/SUMMARY.md`.

## 2.c — Trajectory → OSC interface (derivative-carrying)

- **Reference:** `lcm_trajectory_systems.cc:60-64` — the `LcmTrajectoryReceiver` reconstructs the LCM-carried knot matrix as `PiecewisePolynomial<double>::FirstOrderHold(times, datapoints)`. This IS a derivative-carrying PP: `traj.value(t)` = linear-interpolated position, `traj.EvalDerivative(t, 1)` = piecewise-constant velocity, `traj.EvalDerivative(t, 2)` = 0. `TransTaskSpaceTrackingData::UpdateY / UpdateYdot / UpdateYddotDes` calls all three per tick.
- **Port (legacy PWL path):** wrapper.py:3154-3163 → `self.executor.compute_torque(p_ee_desired=_p_ee_des, v_ee_desired=None, ...)`. **`v_ee_desired` is None** → OSC uses `v_err = -v_ee_now` (no velocity feedforward). Executor 1.a divergence directly manifests here.
- **Port (Stage-A PWL path):** wrapper.py:3126-3136 → `self.executor.compute_torque(p_ee_desired=_p_des, v_ee_desired=_v_des, ...)`. `_v_des` extracted from `RepositionTrajectory.eval(sim_t)`. Matches reference semantics.
- **Tag:** LOAD-BEARING (derivative field, coupled to executor 1.a/1.b).
- **Confidence:** high.
- **Tier 2 — port-side runtime capture:**
  ```
  # default:
  [REPOS-T2] free_mode_v_ee_desired_wired=False
  # PWL:
  [REPOS-T2] free_mode_v_ee_desired_wired=True
  ```
  **2.c → CONFIRMED.** Default path drops `v_ee_desired` (executor 1.a manifests at runtime); Stage-A path threads it correctly. Whether this is behaviorally observable depends on: (a) the OSC's Kd_cart × ‖v_ee_now‖ magnitude vs Kp_cart × ‖p_err‖, and (b) whether c3-mode uses the same handshake (it does: c3-mode passes `v_ee_desired = _v_ee_des` from `_velocity_feedforward_from_xseq` at wrapper.py:2931).

## 2.d — filtered_solve_time future-offset

- **Reference:** `sampling_based_c3_controller.cc:1390-1391` maintains `filtered_solve_time_ = (1-α)·solve_time + α·prev`, a low-pass-filtered planner solve latency. `:1718, 1870`: published knot timestamps start `t_context + filtered_solve_time_` seconds into the FUTURE. This compensates for solve delay so `t=now` samples the trajectory at the correct point.
- **Port:** No analog. Published-target semantics: wrapper computes `_sim_t = self._step * self._dt_ctrl` and passes it as the OSC's evaluation time — no offset for solve latency.
- **Tag:** LOAD-BEARING for cadence alignment; likely COSMETIC in the port's single-process synchronous loop (planner and executor share the same tick, so solve latency doesn't produce a phase shift the OSC sees).
- **Confidence:** medium.
- **Tier 2:** REFERENCE-KNOWN (grep-confirmed the offset exists). Port omission is likely OK because the port runs planner + OSC in one Python loop where the "solve time" is real wall-clock delay but there's no LCM boundary that would let OSC sample at a stale trajectory point — every OSC call is preceded by a fresh planner call. The reference's multi-process LCM architecture is what makes this offset load-bearing (planner publishes at t=T, OSC subscribes and receives at t=T+latency, wants to sample the traj at the CORRECT future point). Port bypasses this by co-locating planner and OSC.

## 2.e — `RepositioningTrajectoryType.kIK` (port-only)

- **Reference:** No IK-based per-knot tracker. `grep InverseKinematics examples/sampling_c3/ systems/controllers/` → 0 hits (only the trajectory-optimization InverseKinematics classes for LCS/C3, not reposition).
- **Port:** `params.py:92` defines `kIK = 4` as a fifth `RepositioningTrajectoryType` (no reference analog). Selects `RepositionIKTracker` (`reposition_ik.py:548`) which solves per-knot IK via Ipopt, applies joint-PD-plus-integrator torque (which is then discarded by wrapper — see 2.a).
- **Tag:** LOAD-BEARING IFF kIK is selected. Structural add-on with no reference analog.
- **Confidence:** high.
- **Tier 2:** Runtime disposition CONFIRMED: `kik.yaml` sets `traj_type: kPiecewiseLinear`, wrapper selects `PiecewiseLinearTracker`, kIK code path is INERT for default box runs. **The `_kik` config-name suffix is misleading — the config selects kPiecewiseLinear, not kIK.** The RepositionIKTracker + Ipopt determinism memory (`project_ik_ipopt_nondeterminism_pinned.md`) covers a code path that is not exercised under the current default config.

## 2.f — admit_active latch (port-only)

- **Reference:** `grep admit_active ADMIT_LATCH consecutive streak no_contact disengage` in reference dispatcher → 0 hits. Same finding banked in memory `feedback_baseline_substrate.md` linkage.
- **Port:** `reposition_ik.py:693` — `ADMIT_LATCH_TICKS = 8`; when LCS admits an EE-BOX pair, the tracker suspends Phase 1 (lift) and holds current altitude for 8 control ticks. Debounces admit toggle to keep the EE pressed against the box face during contact formation.
- **Tag:** LOAD-BEARING IFF kIK selected; otherwise INERT (PWL tracker consumes-and-ignores `admit_active` per source comment `reposition.py:212`).
- **Confidence:** high.
- **Tier 2 — port-side runtime capture:**
  ```
  [ADMIT-GUARD] step=1 admit_active=0 latch=0/0 ee_z=0.200 gate_cap=0
  [ADMIT-GUARD] step=2 admit_active=0 latch=0/0 ee_z=0.200 gate_cap=0
  ...
  ```
  Latch stays at `0/0` throughout the run — **CONFIRMED-INERT under kPWL**. Would flip to LOAD-BEARING under kIK.

## 2.g — descent-gate stability counter (port-only)

- **Reference:** No analog. Grep-confirmed at 2.f.
- **Port:** `reposition_ik.py:709-710` — `TARGET_STABLE_TICKS = 5`, `TARGET_STABLE_TOL = 5e-3` (5 mm). Delays descent (Phase 3) until the dispatcher's `p_target` has been stable (within 5 mm) for 5 consecutive control ticks. Prevents dropping onto a wrong face during dispatcher oscillation.
- **Tag:** LOAD-BEARING IFF kIK selected; INERT under kPWL.
- **Confidence:** high.
- **Tier 2 — port-side runtime capture:**
  ```
  [ALT-GATE] step=1 target_stable=0/0 allow_descent=1
  [ALT-GATE] step=2 target_stable=0/0 allow_descent=1
  ...
  ```
  Descent-gate stays at `0/0 allow_descent=1` — **CONFIRMED-INERT under kPWL** (the `_target_stable_ticks` field is set to 0 by `RepositionIKTracker` but never touched by `PiecewiseLinearTracker`, so the print reflects a zero default not an active mechanism).

## 2.h — `speed` field mismatch

- **Reference:** `examples/sampling_c3/{anything,push_t}/parameters/reposition_params.yaml`: `speed: 0.18 m/s`. Used across all traj types in `reposition.cc` for both the per-leg step size (`step_size = speed * dt`) and the total travel time (`total_travel_time = travel_distance / speed`).
- **Port:** `config/sampling_c3_kik.yaml`: `speed: 0.40 m/s` (2.22× reference) + `pwl_speed: 0.18 m/s` (matches reference).
- **Tag:** LOAD-BEARING (setpoint-march stride in legacy path; unused in Stage-A which uses `pwl_speed`).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[REPOS-T2] params: speed=0.4 pwl_speed=0.18`. Port default legacy path marches the setpoint at 0.40 m/s → OSC receives a p_des that advances 2.22× faster than reference expects. Combined with executor 1.e's compound-authority (40000:1), this pushes the QP harder on `p_err` → part of the mechanism behind executor 1.d's over-cap events. Stage-A path uses `pwl_speed=0.18` which matches reference.

## 2.i — `pwl_waypoint_height` mismatch

- **Reference:** `anything/reposition_params.yaml: pwl_waypoint_height: 0.07738005` (auto-generated per run). `push_t/reposition_params.yaml: pwl_waypoint_height: 0.06`. Range ~6-8 cm.
- **Port:** `config/sampling_c3_kik.yaml: pwl_waypoint_height: 0.15 m` (~2× reference).
- **Tag:** LOAD-BEARING for lift-time / traversal-safety trade.
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[REPOS-T2] params: pwl_waypoint_height=0.15`. Port lifts the EE **7-9 cm higher than reference**. Consequences: (a) longer traversal-away-from-target, (b) more time out of contact, (c) larger EE→box distance at descent start (potentially longer descent phase). Reference-conformant value would need to be ≥ `box_top(0.10) + pusher_radius(0.025) + safety_margin` — port's 0.15 = box_top + pusher + 25 mm safety; reference's 0.06-0.077 would be BELOW box top, which is only viable if pusher enters near the object edge from outside (reference has different manipulanda geometry — need (5) sim/env cross-check).

## 2.j — Reposition-mode `v_ee_desired` handshake to OSC

- **Reference:** OSC always extracts `ydot_des = traj.EvalDerivative(t, 1)` from the FirstOrderHold PP (`osc_tracking_data.cc:88-108`). Non-zero for real PWL trajectories.
- **Port (legacy):** `wrapper.py:3154-3163` — `v_ee_desired=None`. OSC uses `v_err = -v_ee_now`.
- **Port (Stage-A):** `wrapper.py:3126-3136` — `v_ee_desired=_v_des` extracted from `RepositionTrajectory.eval(sim_t)`.
- **Tag:** LOAD-BEARING (couples exec 1.a/1.b).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED (see 2.c). Default path drops `v_ee_desired` → the executor 1.a divergence "PWL traj derivative dropped" fires on every default reposition tick. Reduces the OSC's damping-toward-target behavior; likely part of why the port's OSC needed higher Kp_cart/W_track (over-drive) to close the tracking loop without a velocity feedforward.

## 2.k — Rotation tracking cost (surfaced by S2 cold read; corrects executor Tier-1)

- **Reference:** `osc_params.yaml:65-79`:
  ```
  EndEffectorRotW  = diag(10)
  EndEffectorRotKp = diag(800)
  EndEffectorRotKd = diag(40)
  ```
  Compound rotational authority = `W_rot · Kp_rot = 10 · 800 = 8000 per axis`. Even with `track_end_effector_orientation: false`, `EndEffectorOrientationTrajectoryGenerator::CalcTraj` (`end_effector_orientation.cc:49-54`) OVERRIDES the input orientation trajectory with a constant identity-quaternion PWL: `PiecewiseQuaternionSlerp([0, 1], [Q_identity, Q_identity])`. The OSC's `RotTaskSpaceTrackingData` then actively tracks identity-quaternion with the 8000-authority above → non-trivial rotational torque applied every tick in BOTH c3 and reposition modes.
- **Port:** `qp_builder.py` — 0 hits for "rot", "orientation", "angular", "J_rot". Port QP has NO rotation cost, no rotational Jacobian, no rotational task. The OSC applies zero orientation-holding torque.
- **Tag:** LOAD-BEARING. (Surfaced by S2 cold read; **retroactively corrects executor Tier-1** which noted `track_orientation=false` as "orientation OFF" — that was WRONG. The generator emits identity, but the tracking data still applies 8000-authority tracking to identity.)
- **Confidence:** high (magnitudes verified from `osc_params.yaml` + `end_effector_orientation.cc`).
- **Tier 2:** REFERENCE-CONFIRMED by static read of `osc_params.yaml:65-79` + `end_effector_orientation.cc:33-57`. Port confirmed missing by grep. **Not runtime-instrumented** — the deciding value is a static parameter, not a runtime state. Consequence: reference OSC keeps the EE from twisting during ALL modes (identity-hold); port OSC lets the arm's null-space redundancy resolve rotation freely (guided only by `Kp_null·q_arm_err` posture cost, which is 10× smaller in authority). Downstream: port EE orientation may drift during reposition + push; reference EE holds identity. This may be the mechanism behind some of the "box tumble" observations if the port's pusher tip is at a different angle when it makes contact. Belongs also in a possible S(?)-executor supplement.

## 2.l — `finished_reposition_flag` semantics

- **Reference:** `reposition.cc:112, 148, 279, 386, 464` — set to `true` when the PWL/spline/etc. reaches its target in ONE step (i.e., `t_line >= total_travel_time` at knot 1 in `RepositionStraightLine`, or `t_spline == 1` at knot 1 in `RepositionSpline`, etc.). This flag signals "the traj step-1 is at the goal" — a completion criterion at the KNOT LEVEL, not a Cartesian-tolerance test.
- **Port (PWL tracker):** `reposition.py:314` — `finished = abs(ee_now[2] - p_target[2]) <= 0.005` (5 mm z-tolerance). Cartesian-tolerance, not knot-level.
- **Port (IK tracker):** `reposition_ik.py:1298-1299` — `finished = ||p_target - ee_now|| ≤ 0.02` (2 cm 3D-euclidean tolerance).
- **Port (RepositionTrajectory Stage-A):** `reposition_trajectory.py:129-139` — `is_finished(sim_t, ee_now, tol) = (sim_t >= t_end AND ||p_target - ee_now|| <= tol)`. Time-elapsed AND Cartesian-tolerance.
- **Tag:** LOAD-BEARING (dispatcher consumes this to decide free→c3 transition; different semantics = different transition timing).
- **Confidence:** high.
- **Tier 2:** REFERENCE-KNOWN. Port has THREE different finish-criteria across the three code paths — none of which match reference's knot-level single-step arrival test. Whether this is behaviorally load-bearing depends on how the wrapper consumes `finished` — the port's `self._last_repos_finished` is consumed by `mode_switch.decide_mode` (subsystem 4), which uses it to trigger the `kToC3ReachedReposTarget` transition. Divergent finish-criteria → divergent free→c3 transition timing.

## 2.m — `is_doing_c3` flag threading

- **Reference:** `Reposition(...)` takes `const bool& is_doing_c3` — gates the `finished_reposition_flag` setting (line 111, 146, 277, 385, 463: `if (i == 1 && ... && !is_doing_c3) finished_reposition_flag = true;`). Meaning: only mark reposition as finished when NOT currently in c3 mode. This prevents a stale finished-flag from firing during a c3 push that happens to be near the reposition target.
- **Port:** No analog. The port's tracker `compute_torque(...)` doesn't take an `is_doing_c3` argument; `finished` is computed unconditionally from the current EE-target distance.
- **Tag:** LOAD-BEARING for mode-switch decoupling.
- **Confidence:** high (structural).
- **Tier 2:** REFERENCE-KNOWN. Port omission may allow stale `finished_repos=True` to fire during c3 push if the EE happens to be within the finished-tolerance of the last reposition target. Whether this misfires in practice depends on: (a) the free→c3 transition typically zeros the target, (b) the wrapper's guarding at `wrapper.py:644-674` (contact-proximity entry gate per CLAUDE.md executor doc) may catch this. Runtime confirmation would require a mode-transition trace + finished-flag audit.

## 2.n — max_tilt_angle / tilted-quaternion trajectory

- **Reference:** `sampling_based_c3_controller.cc:1896-1917` — computes per-knot `q_rotated` tilted quaternion based on distance from workspace center × `reposition_params_.max_tilt_angle` (20° in `anything`). Wraps in an `ee_orientations` LCM trajectory published as `end_effector_orientation_target`. THIS TRAJECTORY IS DEAD REFERENCE-SIDE: `EndEffectorOrientationTrajectoryGenerator::CalcTraj` overrides with identity-quaternion when `track_orientation_=false`.
- **Port:** No orientation trajectory generation; no max_tilt_angle parameter.
- **Tag:** UNKNOWN (before Tier-2), COSMETIC (after Tier-2 confirmation that reference tilt code is dead).
- **Confidence:** high (post-Tier-2).
- **Tier 2:** REFERENCE-CONFIRMED-INERT (see 2.k for the identity-quaternion mechanism). Port omission of the tilted-quaternion computation is aligned with reference's INERT tilt code. Port's 2.k rotation-cost omission is the LIVE divergence, not this trajectory-generator divergence.

## Coupling observed (from code + Tier-2 evidence)

- **2.a ↔ 2.b ↔ 2.c** — the dispatch (2.a), knot construction (2.b), and OSC handshake (2.c) form the reposition pipeline. The port has TWO pipelines: (legacy) tracker per-tick march → p_des only; (Stage-A) `RepositionTrajectory` → `(p_des, v_des)`. Only the Stage-A pipeline exercises the reference-conformant derivative handshake. Default box runs use the legacy pipeline (`env_PUSHA_REPOSITION_PWL=0`) — so exec 1.a manifests every reposition tick. Flipping Stage-A on requires ALL of: `PUSHA_REPOSITION_PWL=1` + Q correction (the executor Kp_cart/W_track compound authority stays 40000:1 unless separately reduced) — this is the COUPLED subset flagged in the consolidation directive.
- **2.h ↔ 2.i ↔ 1.e** — port `speed=0.4` m/s + `pwl_waypoint_height=0.15` m + executor `W·Kp=40000` compose into "the OSC chases a fast-moving high-flying target with 200× over-drive." Reducing any one of these individually MAY not fix the tracking-mode over-drive because the other two still contribute (this is the coupled-re-tune story from the memory `project_reproduce_dairlib_phase1_recert_false_positive.md`).
- **2.e ↔ 2.f ↔ 2.g** — the three port-only mechanisms (kIK tracker + admit-latch + descent-gate stability) form a set that all activate together IFF `traj_type=kIK`. Under the default `kPiecewiseLinear`, all three are INERT (confirmed by [ADMIT-GUARD]/[ALT-GATE] runtime prints). Any transition to kIK re-activates all three in lockstep. **Consolidation note**: these three are "COUPLED + INDEPENDENT of reference" — they can be REMOVED wholesale (revert kIK support entirely) without touching any reference-conformant path.
- **2.k ↔ 1.e/1.f** — the reference's 8000-authority rotation-hold to identity supports keeping the EE from twisting; port lacks this. Downstream: port EE may reach contact at a different orientation than reference expects, potentially affecting box/T tumble outcomes. This may explain part of the "box tumble" observations that survived the joint-2 pin (memory `project_reproduce_dairlib_main_honest_option_a.md`) — the pin holds a specific joint but not the EE end orientation directly.
- **2.l ↔ 2.m ↔ subsystem (4)** — finished-flag semantics + is_doing_c3 gating both feed the dispatcher's mode-switch decision, which is subsystem 4's territory. Belongs in the (4) cold read: verify the port's mode-switch consumes `finished_repos` and `is_doing_c3` semantics correctly.
- **2.d (filtered_solve_time) ↔ port's single-process loop** — the reference's future-offset compensates for LCM-delivery latency in a multi-process architecture. Port collapses to single-process → offset likely inert, but coupled to the (F) multi-process potential future work.

**Not observed as coupled (cold read):**
- 2.h `speed=0.4` vs `pwl_speed=0.18` is a within-port divergence (two speed parameters serving different code paths). Does not directly couple to reference-side timing (reference has one `speed` field).

## Reposition Tier-2 verdict roll-up

| # | Divergence | Tier-1 | Tier-2 (this pass) |
|---|---|---|---|
| 2.a | Tracker dispatch structure | LOAD-BEARING | **CONFIRMED** — default = PiecewiseLinearTracker, torque discarded, tracker computes p_des only |
| 2.b | Knot construction (analytic N-knot vs per-tick march) | LOAD-BEARING | **CONFIRMED** — port default marches one setpoint per tick; Stage-A builds K∈{2,3,4} knots per rebuild |
| 2.c | Derivative-carrying trajectory to OSC | LOAD-BEARING | **CONFIRMED** — default drops v_des; Stage-A passes v_des |
| 2.d | filtered_solve_time future-offset | LOAD-BEARING (cadence) | **REFERENCE-CONFIRMED**; port omission likely inert in single-process loop |
| 2.e | kIK RepositioningTrajectoryType | LOAD-BEARING iff selected | **INERT** — kik.yaml selects kPiecewiseLinear |
| 2.f | admit_active latch (ADMIT_LATCH_TICKS=8) | LOAD-BEARING (kIK) | **INERT** — PWL tracker consumes-and-ignores, runtime latch=0/0 |
| 2.g | descent-gate stability counter | LOAD-BEARING (kIK) | **INERT** — same reason as 2.f |
| 2.h | speed (0.18 ref vs 0.40 port) | LOAD-BEARING | **CONFIRMED** — runtime speed=0.4 (2.22× reference); pwl_speed=0.18 aligned |
| 2.i | pwl_waypoint_height (0.06-0.077 ref vs 0.15 port) | LOAD-BEARING | **CONFIRMED** — runtime 0.15 (~2× reference) |
| 2.j | v_ee_desired handshake | LOAD-BEARING | **CONFIRMED** — free_mode_v_ee_desired_wired=False in default path |
| 2.k | Rotation tracking cost (identity-quaternion hold) | LOAD-BEARING (NEW — surfaced by S2 cold read) | **REFERENCE-CONFIRMED** — reference W_rot·Kp_rot=8000/axis; port qp_builder has zero rotation cost |
| 2.l | finished_reposition_flag semantics | LOAD-BEARING | **REFERENCE-KNOWN** — reference: knot-level single-step arrival; port: 3 different Cartesian-tolerance predicates |
| 2.m | is_doing_c3 flag threading | LOAD-BEARING | **REFERENCE-KNOWN** — reference gates finished-flag on `!is_doing_c3`; port has no analog |
| 2.n | max_tilt_angle / tilted quaternion trajectory | UNKNOWN | **REFERENCE-CONFIRMED-INERT** — reference generator overrides tilt with identity when `track_orientation=false` |

14 entries → 6 CONFIRMED at runtime, 3 REFERENCE-CONFIRMED-INERT, 2 CONFIRMED-INERT (kIK path), 3 REFERENCE-KNOWN (static-verified, not runtime-tested). Zero remaining UNKNOWNs. **NEW Tier-1 correction to executor**: 2.k retroactively corrects executor 1.p's "orientation is inert" tag — orientation tracking is LIVE with 8000-authority identity-hold; port lacks the cost entirely.

## Reposition Tier-2 evidence artefacts

- Diagnostic commit: `64ffdee` (`diag(repos-tier2): PUSHA_REPOS_T2_DIAG log-only reposition disclosure`).
- Instrumentation guard: `PUSHA_REPOS_T2_DIAG=1` — default OFF, byte-identical to `dd2294d` baseline.
- Run 1 (default): `audit_output/repos_tier2/run_default.log`
- Run 2 (Stage-A PWL): `audit_output/repos_tier2/run_pwl.log`
- Summary: `audit_output/repos_tier2/SUMMARY.md`

---

# Subsystems (3)–(5)

Pending user re-authorization after Reposition Tier-2 review. Order:

- (3) LCS admission / contact pair filter
- (4) Planner / ADMM C3+ solver + mode-switch dispatcher (consumes 2.l/2.m)
- (5) Sim / env-builder / URDF geometry (holds the 3-mm sphere-radius divergence)

Same two-tier discipline. Consolidation directive registered: apply AFTER the full map + coupling graph is complete, per-flag with box-tripwire, on the COSMETIC + INDEPENDENT subset only; the COUPLED subset (2.a/2.b/2.c/2.h/2.i/2.j/2.k/1.e/1.f and any (3)/(4)/(5) coupling neighbors) stays as-is pending a coupled-re-tune decision.
