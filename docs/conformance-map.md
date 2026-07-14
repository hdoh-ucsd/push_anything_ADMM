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

# Subsystems (2)–(5)

Pending user re-authorization after executor Tier-2 review. Order:

- (2) Reposition / IK tracker + PWL trajectory feed (exercises 1.a/1.b coupling)
- (3) LCS admission / contact pair filter
- (4) Planner / ADMM C3+ solver
- (5) Sim / env-builder / URDF geometry (holds the 3-mm sphere-radius divergence)

Same two-tier discipline: cold-read Tier-1 (per-entry depth + Coupling observed bucket + index table), then Tier-2 (reference deep-reads for UNKNOWNs + port instrumented log-only runs for LOAD-BEARINGs, EIO-gated).
