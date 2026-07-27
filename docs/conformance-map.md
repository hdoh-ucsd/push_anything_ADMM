# push_anything_ADMM — Conformance map (port vs reference)

**Purpose:** A subsystem-by-subsystem cold-read map of the divergences between the port (`push_anything_ADMM @ main dd2294d`) and the reference (`dairlib_sampling_c3 @ push_anything_dev 257e3ed`), each entry tagged for its dataflow-derived load-bearing verdict and confidence.

**Format:**
- **Tier 1 — Static read.** File-line evidence + dataflow reasoning. Every entry classified as `LOAD-BEARING`, `COSMETIC`, or `UNKNOWN`, with a confidence tag.
- **Tier 2 — Runtime confirmation.** Every Tier-1 `UNKNOWN` gets a reference-side deeper-read verdict (READ, no run) OR a port-side log-only instrumented-run verdict. Every Tier-1 `LOAD-BEARING` gets a runtime CONFIRMED / REFUTED tag using the actual value.
- **Coupling observed.** Consolidated cross-entry graph — the decision-critical bucket. Kept as a bucket (not merged into per-entry notes) so the coupling graph is visible at a glance.

Evidence is separate from verdict — instrumented run outputs live under `audit_output/<subsystem>_tier2/`, never in this doc.

**Status:** All 5 subsystems Tier-1 + Tier-2 landed 2026-07-14. **2026-07-17 update:** Subsystem (4) rows 4.a, 4.b, 4.c, 4.l reach reference-match for T-push (commit 08003e1 + prior 4c3bad5); new row 4.s added for port-only surface-entry gate (now disabled for T).

**2026-07-25 refresh (`main` @ `f484607`):** Subsystem (3) contact-model cluster (3.g/3.h/3.n/3.p) + drag band-aid (3.j) all flipped to REFERENCE-MATCH — port default `_contact_model = "anitescu"` (`lcs_formulator.py:86`); `_box_drag_c = 0.0`; n_lambda = 4·n_c runtime-confirmed on p62 baseline. Subsystem (4) horizon+dt (4.d/4.e) now task-conditional (T=5×0.1, box=7×0.075) — both REFERENCE-MATCH per their respective YAMLs (`main.py:611-629`). The 2026-07-14 map's "contact-model cluster is UNRESOLVED / coupled re-tune HOLD" prognosis is stale — that cluster is closed. What remains from the two-cluster prognosis is only the executor-overdrive cluster (subsystem 1) and port-todo #7 (coupled ρ/G/Q; investigation 2026-07-23).

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
  [EXEC-T2] c3_ref_gains_flag=False env_REFCONF_OSC_ALIGN=0
            env_REFCONF_OSC_C3_MODE_GAINS=0
  ```
  **1.e → CONFIRMED at runtime.** Compound authority = `100 × 400 = 40000`, matching the static-read prediction exactly. No envvar overrides active in the default config. Reference compound authority = `1 × 200 = 200`. **Port over-drives position by 200× vs reference.** The `REFCONF_OSC_ALIGN` and `REFCONF_OSC_C3_MODE_GAINS` flags (`operational_space_controller.py:113, 147`) are the two levers that would collapse this ratio; neither is default-on and neither activated in this run.

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
- Stage-A PWL reposition trajectory feeding (`REFCONF_REPOSITION_PWL=1`) — the path that would exercise the 1.a/1.b coupling; belongs to (2) reposition.

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
- **Dataflow:** The reference has ONE torque generator (OSC) and analytic knot construction. The port has TWO parallel construction paths (per-tick tracker `p_des` OR Stage-A PWL trajectory `(p_des, v_des)`) that feed a single OSC. When `REFCONF_REPOSITION_PWL=0` (default), the OSC receives per-tick `p_des` snapshots with no velocity. When `=1`, the OSC receives `(p_des, v_des)` from a persisted PWL trajectory.
- **Confidence:** high.
- **Tier 2 — port-side runtime capture:**
  ```
  [REPOS-T2] tracker=PiecewiseLinearTracker traj_type=kPiecewiseLinear
             use_pwl_traj=False env_REFCONF_REPOSITION_PWL=0
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

- **2.a ↔ 2.b ↔ 2.c** — the dispatch (2.a), knot construction (2.b), and OSC handshake (2.c) form the reposition pipeline. The port has TWO pipelines: (legacy) tracker per-tick march → p_des only; (Stage-A) `RepositionTrajectory` → `(p_des, v_des)`. Only the Stage-A pipeline exercises the reference-conformant derivative handshake. Default box runs use the legacy pipeline (`env_REFCONF_REPOSITION_PWL=0`) — so exec 1.a manifests every reposition tick. Flipping Stage-A on requires ALL of: `REFCONF_REPOSITION_PWL=1` + Q correction (the executor Kp_cart/W_track compound authority stays 40000:1 unless separately reduced) — this is the COUPLED subset flagged in the consolidation directive.
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

# Subsystem (3) — LCS admission / contact pair filter

## Sources read

- **Reference** (`dairlib_sampling_c3 @ push_anything_dev 257e3ed`):
  `systems/controllers/sampling_based_c3_controller.cc:128-133, 1580-1615, 1617-1670, 2537-2555, 2777-2795`;
  `systems/controllers/sampling_based_c3_controller.h:233`;
  `examples/sampling_c3/parameter_headers/sampling_c3_options.h:12-137, 224-267, 309-428`;
  `examples/sampling_c3/anything/parameters/sampling_c3_options.yaml`;
  `examples/sampling_c3/push_t/parameters/sampling_c3_options.yaml`;
  `examples/sampling_c3/franka_sampling_c3_controller.cc:120-269` (pair-list construction).
  **NOT accessible**: the external `c3::multibody::LCSFactory` implementation lives in `github.com/DAIRLab/c3.git @ 5c08cb2e14b1ab10e024cb46e8504970cffcd5ea` (pinned via `MODULE.bazel:88-95`). Local clone was **denied** — reference-side verification limited to usage-site inference. Deep-read of LCSFactory internals (exact `GetNClosestContactPairs` algorithm, `GenerateLCS` matrix construction, `GetContactModelMap`, `GetNumContactVariables`) would require permission to clone the c3 lib.
- **Port** (`push_anything_ADMM @ main dd2294d + ce29c9f/bde8d64/94774fe/64ffdee/c5f51ab/98f5e94`):
  `control/lcs_formulator.py:43-940` (constructor + `_synthesize_manipuland_ground_contacts` + `extract_lcs_contacts`);
  `main.py:525-551` (`LCSFormulator` construction + yaml pre-read of `lcs_explicit_manipuland_ground_contacts` and `use_reference_pair_admission_planner_lcs`);
  `config/sampling_c3_kik.yaml` (no admission-side keys → all admission knobs take code defaults).

**Admission scope:** what happens between "here is the current Drake plant state" and "here is the LCS's per-tick `(phi, J_n, J_t, mu)` contact snapshot handed to `linearize_discrete` for the ADMM solve." Excludes the LCS matrix construction proper (subsystem 4) and the manipuland URDF geometry itself (subsystem 5).

## Index table

| # | Divergence | Tier-1 tag | Tier-2 |
|---|---|---|---|
| 3.a | Pair-selection algorithm (2 mm threshold vs top-N ranking) | LOAD-BEARING | CONFIRMED port default = 2 mm hardcoded (`extract_lcs_contacts` distance_threshold arg) |
| 3.b | Pair-list specification (implicit geometry-set filter vs explicit `contact_pairs_`) | LOAD-BEARING | REFERENCE-CONFIRMED explicit; UNKNOWN internals of `GetNClosestContactPairs` |
| 3.c | EE-manipuland admission | COSMETIC (both admit) | CONFIRMED (port `EE-BOX` tag; reference `EE-{VERTICAL,HORIZONTAL}` etc.) |
| 3.d | Manipuland-ground admission (count + method) | LOAD-BEARING | CONFIRMED port 1 auto pair vs reference 3 pre-specified sphere-witnesses |
| 3.e | Object-wall admission | LOAD-BEARING | CONFIRMED — reference `anything` admits 2 wall pairs; port has NONE |
| 3.f | Arm-self / arm-table exclusion | COSMETIC (both exclude) | CONFIRMED |
| 3.g | Contact model (Stewart-Trinkle vs Anitescu) | LOAD-BEARING | **CONFIRMED at runtime** — port default `stewart_trinkle` vs reference `anitescu` |
| 3.h | num_friction_directions | LOAD-BEARING (ties to 3.g) | CONFIRMED — port 4-edge polyhedral pyramid (ST) vs reference `num_friction_directions=2` (Anitescu) |
| 3.i | μ per-pair vs uniform | LOAD-BEARING | CONFIRMED runtime — port `mu=0.4` single scalar; reference `mu_per_pair_type=[0.583, 0.42, 0.375, 0.3, 0.375]` for `anything` |
| 3.j | `box_ground_drag` viscous A-mod (port-only) | LOAD-BEARING | **CONFIRMED at runtime** — 10.0 active by default |
| 3.k | `PORT_LCS_ALWAYS_ON_EE_BOX` (port-only) | LOAD-BEARING iff enabled | CONFIRMED-INERT (env unset → False) |
| 3.l | `force_top_k_ee_box` (partial ref-conformance) | LOAD-BEARING (planner ⟂ cost) | CONFIRMED — planner LCS: False (default arg); cost-LCS: True (inner_solve.py, but this touches subsystem 4) |
| 3.m | `ref_pair_admission_planner_lcs` (port-only tshape-only) | LOAD-BEARING iff enabled | CONFIRMED-INERT (yaml unset → False) |
| 3.n | Normal-row patches (COMPLIANCE_K, VELOCITY_LEVEL, PHI_CLAMP) — 3 mechanisms | LOAD-BEARING iff any enabled | **CONFIRMED-INERT all 3 individually** (2.k caution applied) |
| 3.o | `LCS_EXPLICIT_MANIPULAND_GND` synthesis (port-only) | LOAD-BEARING iff knob > 0 | CONFIRMED-INERT (default 0; box path) |
| 3.p | LCS λ-dimension formula | LOAD-BEARING | CONFIRMED — n_lambda = 6·n_c (port ST) vs 4·n_c (reference Anitescu w/ num_friction=2) |
| 3.q | Tangential-Jacobian basis construction | UNKNOWN | UNKNOWN — deep-read requires c3 lib clone (denied) |

## 3.a — Pair-selection algorithm

- **Reference:** `sampling_based_c3_controller.cc:1580-1615` — `GetResolvedContactPairs` loops each pair-type group and calls `LCSFactory::GetNClosestContactPairs(plant, context, contact_geoms[i], num_to_select)`. **Selects the N closest pairs by phi from each group**. Deterministic count (n_c is fixed at construction via `resolve_contacts_to`), phi threshold is INTERNAL to `GetNClosestContactPairs` (unknown — needs c3 lib clone).
- **Port:** `lcs_formulator.py:596-601` — `query_obj.ComputeSignedDistancePairwiseClosestPoints(distance_threshold=0.002)`. **Drake filters at 2 mm gap**; pairs with `phi > 2 mm` are DROPPED entirely. Then port applies a geometry-set filter (EE-manipuland OR manipuland-ground). Count varies per tick.
- **Tag:** LOAD-BEARING. Different algorithms with different semantics: reference has a **constant-count** LCS (n_lambda_ = fixed at construction); port has a **variable-count** LCS (n_c changes tick-to-tick as gaps open/close).
- **Confidence (Tier 1):** high.
- **Tier 2 — port-side runtime capture:**
  ```
  [ADMIT-T2] distance_threshold=0.002 m (hardcoded default in extract_lcs_contacts;
             reference uses top-N ranking, no threshold)
  [ADMIT-T2] call=0 n_c=1 tags=['BOX-GND'] phi=['+0.0000']
  [ADMIT-T2] call=2 n_c=1 tags=['BOX-GND'] phi=['-0.0000']
  [ADMIT-T2] call=200 n_c=1 tags=['BOX-GND'] phi=['-0.0000']
  ```
  Port maintains `n_c = 1` (BOX-GND only, EE too far at z=0.2 vs box top z=0.10) throughout the first 2 s. **Confirms the variable-count behavior**: EE-BOX pair is NOT admitted until step ~417 when phi drops to +1.8 mm (below 2 mm), giving the planner NO EE-BOX visibility during approach — the exact mechanism `PORT_LCS_ALWAYS_ON_EE_BOX` was designed to bypass (`lcs_formulator.py:194-208`).

## 3.b — Pair-list specification

- **Reference:** `franka_sampling_c3_controller.cc:124-269` — pairs are **PRE-SPECIFIED explicitly at controller construction**, per demo. For `push_t`: `SortedPair(EE, VERTICAL_LINK)`, `SortedPair(EE, HORIZONTAL_LINK)`, `SortedPair(TOP_LEFT_SPHERE, GROUND)`, `SortedPair(TOP_RIGHT_SPHERE, GROUND)`, `SortedPair(BOTTOM_SPHERE, GROUND)`. Ground-object pairs use SPHERE bodies added to the manipuland URDF (small witness spheres at footprint corners).
- **Port:** `lcs_formulator.py:229-260` — pairs are **IMPLICIT**: at init, port builds three GeometryId sets (`_ee_geom_ids`, `_manipuland_geom_ids`, `_ground_geom_ids`); per tick, `extract_lcs_contacts` filters Drake's pairwise output to any pair whose two GeometryIds fall into `(EE, manipuland)` OR `(manipuland, ground)`. **No named pair list.**
- **Tag:** LOAD-BEARING structurally.
- **Confidence (Tier 1):** high.
- **Tier 2 — REFERENCE-CONFIRMED via c3 lib clone (`c3/multibody/lcs_factory.cc:229-261`):**
  ```cpp
  // GetNClosestContactPairs internals:
  for (const auto& geom_pair : contact_pairs) {
    multibody::GeomGeomCollider collider(plant, geom_pair);
    auto query_result = collider.GetGeometryQueryResult(context);
    distance_pairs.emplace_back(query_result.distance, geom_pair);
  }
  std::nth_element(...);  // partition to first N by ascending distance
  return first N pairs (unsorted among themselves);
  ```
  Loops each PRE-SPECIFIED pair, evaluates signed distance via `GeomGeomCollider::GetGeometryQueryResult`, sorts by distance, returns top-N. **NO gap threshold** — every pre-specified pair is scored regardless of gap. NO secondary tie-break beyond `std::nth_element`'s partial ordering. **3.b → FULLY RESOLVED via c3 lib clone.** Port's implicit filter is structurally equivalent when only 2 pair types exist (EE-manip, manip-gnd) AND top-N ranking is unnecessary because there's only one candidate per type; diverges when either (a) reference has more pair types (walls, object-object) or (b) reference has more candidates per type (e.g., jacktoy 3 EE-capsule pairs with top-N < 3).

## 3.c — EE-manipuland admission

- **Reference:** `franka_sampling_c3_controller.cc:186-191` (jacktoy) / `:229-232` (push_t) — 1 EE-object pair per body (or 2-3 for multi-body objects like jacktoy).
- **Port:** `lcs_formulator.py:626-629` — admits `(EE geom_id, manipuland geom_id)` pair, tagged `EE-BOX`.
- **Tag:** COSMETIC — both agents admit at the structure level.
- **Confidence:** high.
- **Tier 2:** CONFIRMED at runtime (`[CONTACT-ELEM] step=417 element=manipulated_object::collision phi=+0.0018m` shows admission fires when phi drops below 2 mm). The DIVERGENCE is not the admission event itself but the TIMING (deferred by port's threshold vs eager by reference's always-present pair). Belongs conceptually with 3.a.

## 3.d — Manipuland-ground admission (count + method)

- **Reference:** for `anything`, `resolve_contacts_to_lists=[[0, 1, 3, 0, 2]]` → **3 object-ground pairs** admitted. For `push_t`, `resolve_contacts_to_lists=[[0, 1, 3], [0, 2, 3]]` → also 3 T-ground pairs. Uses PRE-SPECIFIED sphere-witness bodies added to the manipuland URDF (see 3.b: `TOP_LEFT_SPHERE`, `TOP_RIGHT_SPHERE`, `BOTTOM_SPHERE` for push_t). **Fixed 3-point footprint captures torsional friction resistance.**
- **Port:** `lcs_formulator.py:630-636` — admits whatever `(manipuland, ground)` pair Drake's `PairwiseClosestPoints` returns; Drake typically returns ONE pair (the single closest witness). Tagged `BOX-GND`.
- **Port opt-in (3.o coupling):** `LCS_EXPLICIT_MANIPULAND_GND=N` activates `_synthesize_manipuland_ground_contacts` which synthesizes N vertex-witness rows — the port-analog of reference's URDF-defined witness spheres. For box: `_box_vertex_set_body_frame({4,8,12})`. For tshape: `_tshape_vertex_set_body_frame(3)` matching reference `resolve_contacts_to=[0, 1, 3]` exactly.
- **Tag:** LOAD-BEARING (torsional friction; 1-pair port default vs 3-pair reference).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED port default = 1 BOX-GND pair (`[ADMIT-T2] call=0 tags=['BOX-GND'] n_box_gnd=1`). The 3-pair synthesis knob is OFF (`lcs_explicit_manipuland_ground_contacts=0`). Reference-analog synthesis exists in the port but is opt-in. **Consequence**: port LCS predicts box moves with only 1 ground constraint → planner can predict rotational drift the 3-witness reference wouldn't (partial explanation for why `box_ground_drag=10.0` was introduced — the 1-pair LCS can't sustain μ·λ_n_gnd realistically).

## 3.e — Object-wall admission

- **Reference:** `anything` config: `resolve_contacts_to_lists=[[0, 1, 3, 0, 2]]` — the last 2 = **2 object-wall pairs**. Workspace has virtual walls defined in the URDF/plant. Provides physical bounds inside the LCS.
- **Port:** No object-wall pair type. The workspace is enforced via `sampling.py`'s `is_in_workspace` filter on samples, NOT via LCS wall contacts. `lcs_formulator.py` has no wall geometry set.
- **Tag:** LOAD-BEARING for the `anything` (multi-manipuland) config; **effectively COSMETIC for the single-box `pushing` task if the box never reaches a wall** (the task goal is 30 cm inside a 60+ cm workspace).
- **Confidence:** high (structural).
- **Tier 2:** REFERENCE-CONFIRMED; port omission verified via grep for "wall". Load-bearing IFF the manipuland reaches a wall — for the default box `pushing --task-id 4` run, unlikely; for future multi-manipuland or edge-case tasks, could matter.

## 3.f — Arm-self / arm-table exclusion

- **Reference:** the pair list is explicitly enumerated (3.b); arm-self and arm-table pairs are simply NOT added. Guaranteed exclusion.
- **Port:** `lcs_formulator.py:617-618, 636` — geometry-set filter admits ONLY EE-manipuland and manipuland-ground; all other pairs (arm-self, arm-table, arm-base) are silently dropped.
- **Tag:** COSMETIC — both agents produce the same exclusion set.
- **Confidence:** high.
- **Tier 2:** CONFIRMED conformant. Different implementation, same result.

## 3.g — Contact model (Stewart-Trinkle vs Anitescu)

- **Reference:** `sampling_c3_options.yaml`: `contact_model: 'anitescu'` for BOTH `anything` and `push_t`. Anitescu with `num_friction_directions=2` → 2D linearized friction cone folded into a single λ block of size `2·num_friction·n_c = 4·n_c`. Single PSD F block (per lcs_formulator.py:174-191 comment on the reference's `lcs_factory.cc:235-275`).
- **Port (2026-07-25):** `lcs_formulator.py:86` — default `_contact_model = 'anitescu'` (flipped since the 2026-07-14 audit read; pre-flip default was `'stewart_trinkle'`). Anitescu factory implemented at `lcs_formulator.py:1875-1959` — `J_c = E_t^T·J_n + diag(μ)·J_t`, single PSD F = `dt·J_c·M⁻¹·J_c^T` (mirrors reference `c3/multibody/lcs_factory.cc:496-545`). ST path retained under env `LCS_CONTACT_MODEL=stewart_trinkle` for regression-diff use.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH.
- **Confidence:** high.
- **Tier 2 — port-side runtime capture (2026-07-25, p62 baseline):**
  ```
  [DEBUG-C3+] n_contacts=4, n_lambda=16          ← 4·n_c Anitescu (ST would be 24)
  [§7.67 B1-A] flag ON but SKIPPED — _is_st_c3p=False
  ```
  **REFERENCE-MATCH confirmed at runtime.** Anitescu single-PSD F block eliminates the γ-γ rank-deficiency that drove non-convergence under ST. The port's §7.24/§7.26/§7.27 normal-row patch flags (3.n) are Stewart-Trinkle-specific and become NO-OPs by construction under Anitescu — see (3.n).

## 3.h — num_friction_directions

- **Reference:** `sampling_c3_options.yaml: num_friction_directions: 2`. Anitescu 2D cone → 2 friction directions per contact folded into 4·n_c λ.
- **Port (2026-07-25):** Anitescu factory uses `NUM_FRICTION_DIRECTIONS = 2` (`lcs_formulator.py:1876`) → 2·dirs = 4 tangent-direction λ's per contact folded into `4·n_c` block. Matches reference exactly.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH (ties to 3.g).
- **Confidence:** high.
- **Tier 2:** REFERENCE-MATCH confirmed at runtime — `n_lambda=16` at `n_c=4` (see 3.g / 3.p). The pre-flip ST 4-edge polyhedron (`{t1, -t1, t2, -t2}` at `lcs_formulator.py:815-824`) is now inert on the default path — walked only if `LCS_CONTACT_MODEL=stewart_trinkle` is explicitly set.

## 3.i — μ per-pair vs uniform

- **Reference:** `mu_per_pair_type: [0.583, 0.42, 0.375, 0.3, 0.375]` for `anything` (ee-ground, ee-object, object-ground, object-object, object-wall). Each contact carries its OWN μ. Harmonic mean of the two surfaces' URDF μ values (comment: `match URDFs with (2·mu1·mu2)/(mu1+mu2)`).
- **Port:** `lcs_formulator.py:80` — `self.mu = float(mu)`, single scalar from `task_cfg["friction"]`. All contacts (EE-BOX and BOX-GND) share the same μ.
- **Tag:** LOAD-BEARING (different friction cones per contact pair).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[ADMIT-T2] mu=0.4 (single scalar; reference uses mu_per_pair_type array)`. For `pushing` task, port `mu=0.4` (task_cfg). Reference-analog computation: pusher-box μ ≈ 0.42 (harmonic of `pusher_friction=0.4` × `box_friction=0.4` → 0.4; matches). But box-ground: port also uses 0.4, reference uses 0.375 (harmonic of box=0.4 × table-URDF-μ ≠ 0.4). Divergence is small (0.4 vs 0.375) but the STRUCTURAL divergence (one scalar vs per-pair vector) matters when the two contact types have very different physical μ (e.g., high-friction manipuland vs low-friction ground). See `push_t` config where `mu_per_pair_type=[0.4165, 1, 0.4615]` — EE-T is μ=1.0, T-ground is μ=0.4615, both HIGHER than pushing's 0.4.
- **Update 2026-07-21 (commit 1e0cbde):** LANDED via `mu_per_pair_type` dict wired through `LCSFormulator.__init__` → `_mu_for_tag(tag)` → per-pair `mus` vector returned by `extract_lcs_contacts`. `push_t` yaml adopts reference literal `EE-BOX: 1.0, BOX-GND: 0.4615, EE-GND: 0.4165`. `pushing` yaml has no `mu_per_pair_type` → scalar-mu path (byte-identical). Structural divergence RESOLVED for T-push; other tasks continue using scalar μ.

## 3.qp_alpha — `qp_projection_alpha` slack parameter

- **Reference:** `sampling_c3plus_options.yaml:24` sets `qp_projection_alpha: 0.01`. Consumed by `/root/reference_repos/c3/core/c3_qp.cc:49` in the QP-based projection variant:
  ```cpp
  double alpha = options_.qp_projection_alpha.value_or(0.01);
  New_U.block(n_x_, n_x_, n_lambda_, n_lambda_) = alpha * F;
  prog.AddQuadraticCost((1 - alpha) * F, ...);
  ```
- **Port C3+ path:** `control/admm_solver.py::_project_componentwise` (line 843) uses the Bui 2026 case-analysis projection (`sqrt(u_lambda / u_eta)` weight ratio) — matches reference `c3_plus.cc::SolveSingleProjection` line 205-212. Neither consumes `qp_projection_alpha`.
- **Reference C3+ path:** `/root/reference_repos/c3/core/c3_plus.cc` grep for `qp_projection_alpha` returns 0 hits.
- **Tag:** DOCUMENTED-INERT for the port's active solver path.
- **Confidence:** high.
- **Tier 2:** Reference `sampling_c3plus_options.yaml:8` sets `projection_type: 'C3+'` for push_t, and `main.py:569 args.solver == "c3plus"` for the port (invoked with `--solver c3plus` in `run_T_180.sh:50`). Both use C3+ exclusively → `qp_projection_alpha` never enters either projection body. Divergence is inert for the runtime configuration; would become load-bearing only if `projection_type: 'QP'` were selected (which no example config in either repo does). Verified 2026-07-21 during H+G plan execution.

## 3.j — `box_ground_drag` viscous A-matrix modification

- **Reference:** No analog. Reference LCS relies on `λ_n_gnd tracking gravity` and `μ·λ_n_gnd` friction to physically decelerate the manipuland.
- **Port (2026-07-25):** `lcs_formulator.py:108` — `self._box_drag_c = 0.0`. The band-aid is disabled by default. Application sites at `lcs_formulator.py:1196-1209` and `:1667-1673` are guarded by `if self._box_drag_c > 0.0` — no A-matrix modification occurs on the default path. Constructor comment at line 84 documents that Anitescu holds `λ_n_gnd` on its own, obsoleting the drag.
- **Tag:** LOAD-BEARING (port-only band-aid) → OBSOLETED (default OFF; would fire only if `_box_drag_c` is externally set > 0).
- **Confidence:** high.
- **Tier 2:** OBSOLETED by the 3.g Anitescu flip. Previous rationale (ADMM under ST can't sustain λ_n_gnd at m·g → box coasts predicted → drag added) is null under Anitescu's single PSD F block. Guarded application sites (`_box_drag_c > 0.0`) mean re-enabling requires an explicit code change; no yaml/env knob currently exposes it.

## 3.k — `PORT_LCS_ALWAYS_ON_EE_BOX` (port-only)

- **Reference:** All pairs are always in the LCS by construction (pre-specified). Reference `EE-BOX` is present at every tick regardless of phi.
- **Port:** `lcs_formulator.py:209-210` — env-gated flag; when set, if the 2 mm threshold did NOT admit an EE-BOX pair, inject the top-N closest EE-manipuland pair explicitly via `ComputeSignedDistancePairClosestPoints` (which does NOT apply the threshold). Mirrors reference's always-present pair.
- **Tag:** LOAD-BEARING iff enabled. This flag is a PARTIAL reference-conformance path.
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[ADMIT-T2] always_on_ee_box=False (env_PORT_LCS_ALWAYS_ON_EE_BOX=<unset>)`. **INERT by default.** The reference-conformant behavior exists but is opt-in.

## 3.l — `force_top_k_ee_box` (partial ref-conformance, kwarg on `extract_lcs_contacts`)

- **Reference:** Always uses top-K per pair-type group (see 3.a).
- **Port:** `lcs_formulator.py:565-581, 665-685` — `force_top_k_ee_box: bool = False` default arg to `extract_lcs_contacts`. When True, unconditionally REPLACE the EE-manipuland slice with the top-K (by phi) candidate pairs. Called by `inner_solve.py` for the cost-LCS ONLY (`n_ee_top_k=2` per reference push_t `resolve_contacts_to_for_cost=[0, 2, 3]`). Planner LCS calls default (False).
- **Tag:** LOAD-BEARING (planner LCS ⟂ cost LCS in port).
- **Confidence:** high.
- **Tier 2:** CONFIRMED via source-read. Planner LCS default arg = False (planner uses the 2mm-filter path). Cost LCS = True per `inner_solve.py` (belongs to subsystem 4). **Partial reference-conformance**: reference uses top-K UNIFORMLY (both planner and cost); port uses top-K for cost only.

## 3.m — `ref_pair_admission_planner_lcs` (yaml opt-in, tshape-only)

- **Reference:** Always uses reference admission (see 3.a).
- **Port:** `lcs_formulator.py:60-77` — yaml `use_reference_pair_admission_planner_lcs: bool = False` default. When True AND `object_shape=="tshape"` AT USE SITE, routes the planner LCS call through `force_top_k_ee_box=True, n_ee_top_k=1` — the reference-analog for the PLANNER LCS. Box path untouched.
- **Tag:** LOAD-BEARING iff enabled (tshape-only opt-in).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[ADMIT-T2] ref_pair_admission_planner_lcs=False`. **INERT for the default box run.** Would flip to LOAD-BEARING for T-shape tasks if opted in.

## 3.n — Normal-row patches (three port-only mechanisms)

**Applying the 2.k caution — verify each mechanism individually.**

- **Reference:** No analog. Reference Anitescu doesn't have separate normal rows to patch (single folded λ block).
- **Port (2026-07-25):** three legacy patches on the (now-inert) Stewart-Trinkle normal row, all defaulted OFF at `lcs_formulator.py:115-117`:
  1. `_normal_compliance_k = 0.0` — additive `k·I` on F_lcs diagonal at EE-BOX normal contact.
  2. `_normal_velocity_level = False` — drops the `φ/dt` position-forcing term from `c_lcs[SLN+ee_box_idx]`.
  3. `_normal_phi_clamp_v_cap = None` — clamps `phi/dt ≥ -v_cap` (depth-asymmetric saturation).
  Per the constructor comment at `lcs_formulator.py:112-114`, these are "all no-ops under Anitescu (no separate normal row)" — kept as dead defaults so downstream code doesn't need conditional guards.
- **Tag:** LOAD-BEARING iff ANY enabled → NO-OPS BY CONSTRUCTION under the current default (`_contact_model="anitescu"`).
- **Confidence:** high.
- **Tier 2:** Doubly-inert — (a) all three fields default OFF, (b) their code paths live in the ST branch which is not walked at runtime. No hidden live mechanism. If a future user opts back to ST (`LCS_CONTACT_MODEL=stewart_trinkle`), these patches would matter again; on the default Anitescu path they cannot.

## 3.o — `LCS_EXPLICIT_MANIPULAND_GND` synthesis (port-only)

- **Reference:** Explicit sphere-witness bodies in the URDF (3-4 spheres for push_t; jacktoy uses per-capsule spheres) provide fixed-witness ground contact via pre-specified `SortedPair(SPHERE, GROUND)`.
- **Port:** `lcs_formulator.py:90-101, 458-540` — env-gated `_synthesize_manipuland_ground_contacts` synthesizes N vertex witnesses in body frame + computes their world Jacobians. For box: 4/8/12 vertex options. For tshape: 3 witness points matching reference push_t layout. When active, suppresses Drake's auto-admitted BOX-GND pair (de-dup).
- **Tag:** LOAD-BEARING iff knob > 0.
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[ADMIT-T2] lcs_explicit_manipuland_ground_contacts=0 object_shape='box'`. **INERT for default box run.** Reference-analog exists but is opt-in. Enabling it (e.g., LCS_EXPLICIT_MANIPULAND_GND=4 for box) would add 4 synthesized BOX-VERT rows and drop Drake's auto BOX-GND — closer to reference structure but the port synthesis uses geometric vertices, reference uses URDF-defined sphere-witness bodies.

## 3.p — LCS λ-dimension formula

- **Reference:** Anitescu with `num_friction=2` → `n_lambda = 2·num_friction·n_c = 4·n_c`. For `anything` (n_c=6): n_lambda = 24. For `push_t` planner (n_c=4): n_lambda = 16.
- **Port (2026-07-25):** Anitescu default → `n_lambda = 4·n_c`. Runtime on the T-push p62 baseline: `n_c=4, n_lambda=16` (log line: `[DEBUG-C3+] n_contacts=4, n_lambda=16`). Matches reference push_t planner formula exactly.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH.
- **Confidence:** high.
- **Tier 2:** REFERENCE-MATCH confirmed. Pre-flip ST formula (`6·n_c` → 24 for n_c=4) is walked only under explicit `LCS_CONTACT_MODEL=stewart_trinkle` opt-in. Belongs conceptually with 3.g.

## 3.q — Tangential-Jacobian basis construction

- **Reference (c3 lib clone at pinned `5c08cb2e`, `multibody/contact_evaluator.h` + `multibody/geom_geom_collider.cc`):**
  - `PolytopeContactEvaluator::Eval(context)` calls `collider.EvalPolytope(context, num_friction_directions_)`.
  - `GeomGeomCollider::ComputePolytopeForceBasis(N)` builds a `(2N+1) × 3` force basis in CONTACT frame:
    - Row 0: `[1, 0, 0]` = normal
    - For i in [0, N): row `2i+1 = [0, cos(π·i/N), sin(π·i/N)]`, row `2i+2 = -row(2i+1)`.
  - For `num_friction=2` (reference default): force basis = `[[1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]` → 4 tangent rows: `{+ŷ_C, -ŷ_C, +ẑ_C, -ẑ_C}` in contact frame.
  - Rotation to world via `R_WC = MakeFromOneVector(nhat_BA_W, 0)` — puts nhat as first column, tangent basis is rotated accordingly.
- **Port (`lcs_formulator.py:815-824`):** 4-edge polyhedron `{t1, -t1, t2, -t2}` where `t1 = cross(nhat, ref)/|·|` (ref=[1,0,0] normally, [0,1,0] if nhat parallel to x). `t2 = cross(nhat, t1)`. Built directly in WORLD frame.
- **Tag:** COSMETIC-EQUIVALENT under isotropic friction.
- **Dataflow:** Both produce a 4-edge polyhedral pyramid tangent basis with 2 orthogonal tangent axes and ±sign coverage. Reference constructs in contact frame then rotates via `MakeFromOneVector`; port constructs directly in world via cross-product convention. **The specific tangent-axis ORIENTATION in world differs (a rotation about nhat), but under isotropic μ the friction cone SHAPE is invariant to this rotation** (μ is the same in all tangent directions). Tangent-DIMENSION (4 per contact) matches. So the resulting F, D, c matrices are structurally equivalent for isotropic friction. If future work introduced anisotropic friction (different μ along t1 vs t2), the tangent-axis orientation would matter.
- **Confidence:** high.
- **Tier 2 — REFERENCE-CONFIRMED via c3 lib clone:** `contact_evaluator.h:80-110` + `geom_geom_collider.cc:207-220`. **3.q → COSMETIC-EQUIVALENT for isotropic friction (both agents use isotropic μ).**

## Coupling observed (from code + Tier-2 evidence)

- **3.a ↔ 3.b ↔ 3.k ↔ 3.l ↔ 3.m** — the whole pair-selection stack. Reference uses a single mechanism (pre-specified + N-closest). Port uses FIVE mechanisms glued together: (a) 2 mm threshold, (b) geometry-set filter, (c) optional always-on injection, (d) optional top-K for cost LCS, (e) optional top-K for planner LCS (tshape-only). **Consolidation candidate**: could collapse to a single reference-conformant path (always-on + top-K for both planner and cost LCS, ALL tasks), but this changes n_c semantics run-to-run — COUPLED with (4) ADMM which currently expects variable n_c.
- **3.d ↔ 3.j** — the 1-pair BOX-GND admission (3.d) is why `box_ground_drag=10.0` (3.j) was introduced. With 1 pair (single ground witness) and Stewart-Trinkle's non-converging ADMM projection, the LCS can't sustain μ·λ_n_gnd → box coasts predicted → drag term is the artificial fix. Reference's 3-pair witness + Anitescu doesn't need this. **RESOLVED 2026-07-25:** (3.g) is now Anitescu on the default path, `_box_drag_c=0.0`, drag is not applied.
- **3.g ↔ 3.h ↔ 3.n ↔ 3.p** — contact-model cluster. **RESOLVED 2026-07-25.** (3.g) flipped to Anitescu on the default path → (h) num_friction=2 folded, (n) three normal-row patches double-inert (default + no-op), (p) n_lambda = 4·n_c runtime-confirmed at 16 (was 24 under ST). Coupled subset flipped as a unit.
- **3.i ↔ 3.d ↔ 3.g** — μ per-pair (3.i) is meaningful only when there are multiple contact pair TYPES (3.d admits both EE-BOX and BOX-GND) AND the contact model supports per-λ friction (3.g Anitescu folds μ into J_c = E_t^T·J_n + diag(μ)·J_t; ST doesn't fold this way).
- **3.j (box_ground_drag) ↔ subsystem (4) ADMM** — port-only band-aid symptomatic of ADMM non-convergence. Belongs to the coupled band-aid subset that only resolves when (4) reaches a properly-converging ADMM projection.
- **3.a (threshold) ↔ (2) reposition ↔ (4) mode-switch** — the 2 mm threshold means the LCS "sees" EE-BOX only in the final millimeter of approach. This drives the dispatcher's mode-switch timing (`kToC3ReachedReposTarget` fires only near contact) and interacts with the reposition finished-flag semantics (2.l) — CROSS-SUBSYSTEM coupling to (2) and (4).

## Deferred / out-of-admission-scope items surfaced

- Cost-LCS (`inner_solve.py`) admission (`force_top_k_ee_box=True, n_ee_top_k=2`) — belongs to (4) planner.
- ADMM projection non-convergence (driving `box_ground_drag` band-aid) — belongs to (4).
- URDF sphere-witness bodies (reference push_t) vs port programmatic collision geometry (env_builder.py) — belongs to (5).
- Anitescu F-matrix construction (single PSD block vs ST 3-block) — belongs to (4) LCS matrix construction.
- Reference `mu_per_pair_type` harmonic-mean derivation from URDF — belongs to (5) URDF.
- `pusher_friction`, `box_friction`, `table_friction` per-task task_cfg values — belongs to (5).

## Admission Tier-2 verdict roll-up

| # | Divergence | Tier-1 | Tier-2 (this pass) |
|---|---|---|---|
| 3.a | Pair-selection algorithm | LOAD-BEARING | **CONFIRMED** — port default = 2 mm hardcoded threshold; reference = top-N ranking |
| 3.b | Pair-list specification | LOAD-BEARING | **FULLY RESOLVED via c3 lib clone** — GetNClosestContactPairs = per-pair GeomGeomCollider + std::nth_element top-N, no threshold, no secondary tie-break |
| 3.c | EE-manipuland admission | COSMETIC | **CONFIRMED conformant structurally** (both admit; port defers to 2 mm, reference always-present) |
| 3.d | Manipuland-ground count + method | LOAD-BEARING | **CONFIRMED** — port 1 auto pair vs reference 3 pre-specified sphere witnesses |
| 3.e | Object-wall admission | LOAD-BEARING (anything) | **REFERENCE-CONFIRMED** 2 wall pairs; port has NONE (effectively COSMETIC for pushing task) |
| 3.f | Arm-self/arm-table exclusion | COSMETIC | **CONFIRMED conformant** — both exclude |
| 3.g | Contact model (ST vs Anitescu) | LOAD-BEARING → REFERENCE-MATCH | **REFERENCE-MATCH 2026-07-25** — port default `anitescu` (`lcs_formulator.py:86`); ST kept as env opt-in |
| 3.h | num_friction_directions | LOAD-BEARING → REFERENCE-MATCH | **REFERENCE-MATCH 2026-07-25** — Anitescu factory uses `NUM_FRICTION_DIRECTIONS=2` (matches reference) |
| 3.i | μ per-pair vs uniform | LOAD-BEARING | **CONFIRMED runtime** — port `mu=0.4` single scalar; reference 5-value array. Updated 2026-07-21 (commit 1e0cbde): T-push adopts reference literal per-pair array; other tasks scalar. |
| 3.j | `box_ground_drag` viscous A-mod | LOAD-BEARING (port-only) → OBSOLETED | **DEFAULT OFF 2026-07-25** — `_box_drag_c=0.0`; obsoleted by 3.g flip |
| 3.k | `PORT_LCS_ALWAYS_ON_EE_BOX` | LOAD-BEARING iff on | **CONFIRMED-INERT** (env unset) |
| 3.l | `force_top_k_ee_box` (kwarg) | LOAD-BEARING | **CONFIRMED** — planner LCS: False; cost LCS: True (belongs to (4)) |
| 3.m | `ref_pair_admission_planner_lcs` | LOAD-BEARING iff on | **CONFIRMED-INERT** (yaml unset) |
| 3.n | Normal-row patches × 3 | LOAD-BEARING iff on → NO-OPS UNDER ANITESCU | **DOUBLY INERT 2026-07-25** — all 3 default OFF AND their code path is not walked under Anitescu default |
| 3.o | `LCS_EXPLICIT_MANIPULAND_GND` | LOAD-BEARING iff on | **CONFIRMED-INERT** (default 0) |
| 3.p | LCS λ-dimension formula | LOAD-BEARING → REFERENCE-MATCH | **REFERENCE-MATCH 2026-07-25** — n_lambda = 4·n_c runtime-confirmed (16 at n_c=4); was 6·n_c under ST |
| 3.q | Tangential-Jacobian basis | UNKNOWN | **COSMETIC-EQUIVALENT via c3 lib clone** — both agents produce a 4-edge polyhedral pyramid tangent basis with isotropic μ; specific tangent-axis orientation differs but is friction-cone-invariant under isotropic μ |

**17 entries → all 17 resolved.** Breakdown after 2026-07-25 refresh: 4 REFERENCE-MATCH (3.g/3.h/3.p landed via Anitescu flip + 3.i landed for T via mu_per_pair_type commit 1e0cbde) + 1 OBSOLETED (3.j band-aid off) + 1 CONFIRMED at runtime (3.a) + 7 CONFIRMED-INERT (all opt-in default OFF, 2.k caution passed; 3.n now doubly inert) + 2 CONFIRMED-CONFORMANT (both agents match: 3.c, 3.f) + 2 REFERENCE-KNOWN (3.d, 3.e) + 1 FULLY-RESOLVED via c3 clone (3.b) + 1 COSMETIC-EQUIVALENT via c3 clone (3.q) — zero remaining UNKNOWNs.

## Additional c3 lib clone findings (bonus insight — validates the contact-model cluster)

Reading `c3/multibody/lcs_factory.cc:404-494` confirms the exact structural divergence for the STEWART-TRINKLE vs ANITESCU contact models — reproduces the exact same load-bearing distinction the port arc has been chasing:

- **Anitescu F matrix** (`FormulateAnitescuContactDynamics:496-545`): `F = dt · J_c · M⁻¹ · J_c^T` where `J_c = E_t^T · J_n + diag(μ) · J_t`. **SINGLE PSD block, size n_lambda × n_lambda = (4·n_c) × (4·n_c)**. Well-conditioned by construction (J_c has full row rank as long as contacts are geometrically consistent).
- **Stewart-Trinkle F matrix** (`FormulateStewartTrinkleContactDynamics:438-494`): 3×3-block partitioned:
  ```
      [ 0         μ_diag         -E_t         ]     ← γ-block
  F = [ 0    dt²·J_n·M⁻¹·J_n^T   dt²·J_n·M⁻¹·J_t^T ]  ← λ_n-block
      [ E_t^T  dt·J_t·M⁻¹·J_n^T   dt·J_t·M⁻¹·J_t^T  ]  ← λ_t-block
  ```
  Note the F[γ, γ] top-left block is **all zeros**. **The γ-γ block is rank-deficient by construction** — exactly the source of the port ADMM's non-convergence on ST.
- **GetNumContactVariables** (`lcs_factory.cc:790-804`): ST → `2·n_contacts + 2·num_friction·n_contacts = (2 + 2·2)·n_c = 6·n_c` for num_friction=2. Anitescu → `2·num_friction·n_contacts = 4·n_c` for num_friction=2. **Matches port `lcs_formulator.py` ST dimension exactly**; confirms Anitescu would be smaller.

**Bonus finding**: The 3.j `box_ground_drag=10.0` band-aid explanation is now MECHANICALLY PROVEN. Under ST with rank-deficient F[γ,γ], the ADMM projection cannot converge to a well-conditioned λ_n_gnd solution → LCS predicts unrealistic coasting → drag added to compensate. Under Anitescu (single PSD F block), λ_n friction is folded into J_c automatically → no rank deficiency → no coasting → no drag band-aid needed.

**The contact-model cluster (3.g↔3.h↔3.n↔3.p↔3.j) IS the LCS↔Drake mismatch**, mechanically confirmed. Flipping to Anitescu makes 3 normal-row patches (3.n) no-ops AND obsoletes box_ground_drag (3.j) simultaneously.

## 2.k caution — verification results

**Applied to the port-only opt-in cluster (3.k, 3.l, 3.m, 3.n×3, 3.o) — 7 individual mechanisms, each verified separately at runtime:**

- All 7 default OFF → inert by default at runtime (`[ADMIT-T2]` init disclosure confirmed each field individually).
- No hidden live mechanism underneath the "inert" tag.
- The 3.n cluster (three normal-row patches — COMPLIANCE_K, VELOCITY_LEVEL, PHI_CLAMP) was checked individually per the 2.k caution, not as a lump — all three individually inert.

**The 2.k caution is especially relevant here because**: like the executor 1.p miss (where "orientation tilt" was inert but "orientation identity-hold" was live underneath), the admission subsystem has a large opt-in flag surface AND a small live default surface (`distance_threshold, contact_model=ST, mu=0.4, box_ground_drag=10.0`). The LIVE default surface was verified separately from the INERT opt-in surface. **The live default surface is where any missed load-bearing mechanism would hide** — verified: no such hidden live mechanism in this subsystem beyond the 4 already tagged.

## Admission Tier-2 evidence artefacts

- Diagnostic commit: `98f5e94` (`diag(admit-tier2): PUSHA_ADMIT_T2_DIAG log-only admission disclosure`).
- Instrumentation guard: `PUSHA_ADMIT_T2_DIAG=1` — default OFF, byte-identical to `dd2294d` baseline.
- Filtered run log: `audit_output/admit_tier2/run_default.log`
- Summary + reference-side deep-read + 2.k caution result: `audit_output/admit_tier2/SUMMARY.md`

## c3 lib clone — CLOSED

User authorized clone 2026-07-14. `/root/reference_repos/c3` at pinned commit `5c08cb2e14b1ab10e024cb46e8504970cffcd5ea` (per `dairlib_sampling_c3/MODULE.bazel:88-95`). Read-only; no port code touched. Closed 3.b + 3.q + bonus contact-model cluster mechanical confirmation.

---

# Subsystem (4) — Planner / ADMM / mode-switch dispatcher

## Sources read

- **Reference** (`dairlib_sampling_c3 @ push_anything_dev 257e3ed` + `c3 @ 5c08cb2e14b1ab10e024cb46e8504970cffcd5ea` clone):
  `c3/core/c3.cc:267-412` (`Solve`, `ADMMStep`, `SetInitialGuessQP`);
  `c3/core/c3_plus.cc:174-221` (`C3Plus::SolveSingleProjection` — Bui 2026 eq 12);
  `c3/core/c3_miqp.cc, c3_qp.cc` (alternate projection classes);
  `dairlib_sampling_c3/systems/controllers/sampling_based_c3_controller.cc:1140-1320` (mode-switch);
  `dairlib_sampling_c3/systems/controllers/sampling_based_c3_controller.cc:1380-1400` (`filtered_solve_time_` LPF);
  `dairlib_sampling_c3/examples/sampling_c3/anything/parameters/sampling_c3_options.yaml` (admm_iter=3, rho_scale=3, delta_option=1, projection_type='MIQP', warm_start=false, end_on_qp_step=false, N=5, planning_dt_position=0.1);
  `dairlib_sampling_c3/examples/sampling_c3/anything/parameters/sampling_c3_options.yaml` (`penalize_changes_in_u_across_solves: true` for `anything`, `false` for `push_t`).
- **Port** (`push_anything_ADMM @ main dd2294d + ... + 67232d7`):
  `control/admm_solver.py:293-1810` (`C3Solver.solve`, `_solve_c3plus`, `_project_componentwise`, `_lorentz_project`);
  `control/ci_mpc_c3plus.py:35-417` (`C3PlusMPC.__init__`, `.compute_control`);
  `control/ci_mpc_c3.py:1-371` (`C3MPC` C3-with-Lorentz path);
  `control/sampling_c3/mode_switch.py:1-162` (`decide_mode`, `SwitchReason` enum, hysteresis);
  `control/sampling_c3/progress.py:1-266` (`ProgressTracker`, `StepMetrics`, `met_progress`);
  `config/sampling_c3_kik.yaml` (`surrogate_admm_iters: 3`; T-push canonical `--admm-iter 3` since commit 4c3bad5 (2026-07-17), box-push still at 25);
  `main.py:345-358` (`PORT_U_HORIZONTAL/VERTICAL/R_VECTOR` env-defaults for EE-space).

**Planner scope:** the C3+ ADMM inner solver + the wrapper's mode-switch dispatcher decision + progress-tracking. Excludes contact admission (subsystem 3), OSC executor (subsystem 1), reposition target generation (subsystem 2).

## Index table

| # | Divergence | Tier-1 tag | Tier-2 |
|---|---|---|---|
| 4.a | Solver class / projection algorithm | LOAD-BEARING | CONFIRMED — port `C3PlusMPC` + `componentwise` projection (script override `--c3plus-projection lcp` removed for T-push 2026-07-17, commit 08003e1); reference default MIQP (C3MIQP class) |
| 4.b | admm_iter count | LOAD-BEARING → REFERENCE-MATCH | CONFIRMED at runtime — T-push port 3 (commit 4c3bad5) and box-push port 3 (commit 952c8a3) both match reference `admm_iter: 3` YAML for `push_t` and `anything`; single canonical setting across both tasks |
| 4.c | rho / rho_scale | LOAD-BEARING → REFERENCE-MATCH | CONFIRMED — port `_rho_scale = 3.0` (commit 08003e1) applied per ADMM iter with initial `rho_init=100.0`; reference `rho_scale=3` per iter. Ramp trace: 100 → 300 → 900 → 2700 over 3 iters. Legacy "adaptive-ρ every 10 iters" branch retained but unreachable at admm_iter≤10. |
| 4.d | Horizon N | LOAD-BEARING | CONFIRMED at runtime — port N=20; reference N=5 |
| 4.e | Planning dt | LOAD-BEARING | CONFIRMED at runtime — port dt=0.05; reference dt=0.1; horizon_time port 1.0s vs reference 0.5s |
| 4.f | delta initial guess | LOAD-BEARING | NEW DIVERGENCE — reference `delta_option=1` initializes `delta.head=x0`; port always zeros |
| 4.g | end_on_qp_step (final rollout) | LOAD-BEARING at non-convergence | NEW DIVERGENCE — reference `end_on_qp_step=false` computes `x_seq` via LCS rollout; port returns direct QP solution |
| 4.h | Cross-tick warm-start | COSMETIC (both OFF) | CONFIRMED — both cold-start per tick |
| 4.i | Within-Solve warm-start (ADMM iter carryforward) | COSMETIC | CONFIRMED — both agents implicitly warm-start delta/omega across iterations |
| 4.j | penalize_changes_in_u_across_solves | LOAD-BEARING → PARTIAL-REFERENCE-MATCH | RESOLVED 2026-07-17 commit 84823fe — port now task-gates the flag (push_t: false, box/other: true) per reference `push_t/sampling_c3plus_options.yaml` (false) and `anything/sampling_c3plus_options.yaml` (true) |
| 4.k | SolveSingleProjection (Bui eq 12) | COSMETIC (equivalent) | CONFIRMED via c3 lib — port `_project_componentwise` = reference C3Plus::SolveSingleProjection exactly (same weighted-eta-vs-lambda case selection + ≥0 clip) |
| 4.l | ADMM convergence at runtime | LOAD-BEARING | CONFIRMED — port iters=25/25 on every solve; primal residual ~3.87 non-decreasing → NON-CONVERGENT (mechanically ties to 3.g Stewart-Trinkle rank-deficient F[γ,γ]) |
| 4.m | Mode-switch branches | LOAD-BEARING | CONFIRMED-CONFORMANT (structure); port omits {xbox force_c3, achieved_fixed_goal, unsuccessful-sample-buffer, wall_offset}; port adds {kForceC3Watchdog, kToBetterRepos, kStayInRepos as distinct enum values} |
| 4.n | Altitude gate on free→c3 transition | LOAD-BEARING | CONFIRMED — reference AND-gates cost-based free→c3 transition on `x_lcs_curr[2] < z_height + c3_min_clearance + wall_offset OR NOT ee_z_close`; port has `ee_z_gate_pass` kwarg (T1a port of reference altitude gate) |
| 4.o | Hysteresis (kind × near_goal) | COSMETIC-EQUIVALENT | CONFIRMED — port `_hysteresis(params, kind, near_goal, ref_cost)` matches reference structure (absolute vs relative, position-near-goal vs generic) |
| 4.p | Progress metric implementation | COSMETIC-EQUIVALENT | Both track "steps since last cost improvement" with mode-specific `num_control_loops_to_wait` |
| 4.q | LCS h_is_zero → LCP pre-solve | UNKNOWN → RESOLVED | Reference c3.cc:283-299 detects `h_is_zero_` (LCS `H matrix all-zero → passive system`) and pre-solves λ via `MobyLcpSolver::SolveLcpLemke`. Port has NO analog (always runs full ADMM). For push_anything, `H` is derived from `Jn · Jf_u` (LCS-formulation), and `Jf_u = M⁻¹ · B` is non-zero (actuated arm) → `h_is_zero_ = false` in reference → LCP pre-solve INERT → no divergence in practice for pushing task. **INERT-BY-CONFIG for actuated systems.** |
| 4.r | Port-only env-tuned R + u-bounds (PUSHA_STAGE5_*) | LOAD-BEARING iff enabled | CONFIRMED — `main.py:346-358` sets env defaults `PORT_U_HORIZONTAL=10, PORT_U_VERTICAL=3, PORT_R_VECTOR=0.1,0.1,10` for EE-space. Port-only Stage-5 alignment package. Default box run in R^7 does NOT trigger these; EE-space runs do. |
| 4.s | Port-only contact-entry gate (surface / center distance) | LOAD-BEARING → DISABLED-FOR-T | CONFIRMED — `sampling_based_c3_controller.py:1487-1512` blocks `finished_repos` when `ee_to_surf ≥ 60mm` (`use_surface_entry_gate=True, contact_entry_surface_threshold=0.060`). Reference `sampling_based_c3_controller.cc:1284-1309` has no such gate — only the height ceiling (4.n). Disabled for T-push in `config/sampling_c3_kik_t.yaml` (commit 08003e1); box-push YAML still leaves the port default on. |

## 4.a — Solver class / projection algorithm

- **Reference:** `sampling_based_c3_controller.cc:143-176` constructs one of `C3MIQP`, `C3QP`, or `C3Plus` per `sampling_c3_options.projection_type`. `anything` YAML: `projection_type: 'MIQP'` → `C3MIQP`; `push_t/parameters/sampling_c3plus_options.yaml` selects `C3Plus`.
- **Port:** `main.py:589` — `_MPCClass = C3PlusMPC if args.solver == "c3plus" else C3MPC`. `--solver` argparse default `'c3plus'`. No C3MIQP or C3QP class in port.
- **Port projection variants for C3+:** `admm_solver.py:82-85` accepts `c3plus_projection ∈ {componentwise, lcp}`. `componentwise` (default) matches `c3_plus.cc:174-221 SolveSingleProjection` (Bui eq 12, closed-form). `lcp` is a port-only Aydinoglu-style Lemke retrofit with no reference analog — was returning `lcp_res_max=inf` on 43% of solves via `lcp_solver.py:66` (Lemke ray-termination sentinel). Fixed 2026-07-17 (commit 08003e1): `run_T.sh` no longer passes `--c3plus-projection lcp`; argparse default `componentwise` applies.
- **Tag:** LOAD-BEARING. MIQP = exact LCP solve, no ADMM; QP = relaxation; C3+ componentwise = Bui eq 12; C3+ lcp = port-only, retired for T.
- **Confidence:** high.
- **Tier 2 — port runtime confirmed 2026-07-17:** `[C3] Solver mode: c3plus (planner: EE-space (R^3 force), c3+ projection: componentwise)`. Post-fix log shows zero `lcp_res_max=inf` events (was 389/903 pre-fix). Reference `push_t` uses C3+ too (via `sampling_c3plus_options.yaml`), so this row now reads: port C3+ + componentwise ≡ reference C3+ + eq 12 exactly. **REFERENCE-MATCH for T-push.** Box-push canonical path unchanged.

## 4.b — admm_iter count

- **Reference:** `sampling_c3_options.yaml: admm_iter: 3` for both `anything` and `push_t`. Very few ADMM iterations because MIQP itself solves the LCP exactly at each iter — 3 outer iterations for ADMM refinement.
- **Port:** `main.py:--admm-iter` argparse default = 3. Both canonical scripts pass `--admm-iter 3`: T-push since commit 4c3bad5 (2026-07-17), box-push since commit 952c8a3 (2026-07-17). Prior box regime `--admm-iter 25` retired at that commit.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH.
- **Confidence:** high.
- **Tier 2 — post-fix runtime 2026-07-17:** T-push at 3 iters produces `mono=True` on 100 % of ADMM solves with primal residual monotone-decreasing. Box-push at 3 iters not yet re-baselined; first run at HEAD `952c8a3` becomes the reference artefact. Tolerance `1e-3` typically not reached in 3 iters — matches reference's own admm_iter=3 non-convergence-tolerance regime.

## 4.c — rho / rho_scale

- **Reference:** `rho_scale: 3` — per `c3.cc:389-390`, `w = w / rho_scale; G = G * rho_scale` each ADMM iter → ρ grows by 3× per iter multiplicatively. Reference uses `rho: 0` (unused) + `rho_scale: 3` per `push_t/parameters/sampling_c3plus_options.yaml:2-4`.
- **Port:** `C3Solver(..., rho=100.0)` initial + `admm_solver.py:137 self._rho_scale = 3.0` (post-2026-07-17). Scaffolding at `admm_solver.py:1436-1442` applies the reference per-iter multiply when `_rs > 1.0`; the legacy `elif (it + 1) % 10 == 0` Boyd-style adaptive branch at 1443-1457 becomes unreachable at admm_iter ≤ 10.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH.
- **Confidence:** high.
- **Tier 2 — post-fix runtime 2026-07-17:** `rho=2700.0` at end of solve (100 × 3³ over 3 iters, matches reference geometric ramp). Log-line self-flag `Note: adaptive-ρ fires every 10 iters; current max_iter=3 ← never triggers!` still prints because the diagnostic remained; the code path it warns about is now correctly dead. Pre-fix (`_rho_scale = 1.0`): 100 → 100 → 100 (dead-flat, adaptive branch unreachable at admm_iter=3).

## 4.d — Horizon N

- **Reference:** push_t `N: 5`; anything `N: 7`.
- **Port (2026-07-25):** task-conditional at `main.py:611-629` — `push_t` → `N=5`, other tasks → `N=7`. Runtime confirmed on p62: log line `[MPC] Horizon: 5 dt: 0.1 s ADMM max iters: 3`.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH (per task).
- **Confidence:** high.
- **Tier 2:** REFERENCE-MATCH on both task paths. Prior port N=20 (pre 2026-07-14) is retired; the 4× lookahead-inflation observation is stale.

## 4.e — Planning dt

- **Reference:** push_t `planning_dt_position: 0.1`, `planning_dt_pose: 0.05`; anything `planning_dt_position: 0.075`, `planning_dt_pose: 0.05`.
- **Port (2026-07-25):** task-conditional at `main.py:611-629` — `push_t` → `dt=0.1, dt_pose=0.05`; other tasks → `dt=0.075, dt_pose=0.05`. Runtime confirmed on p62: `dt: 0.1 s`.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH (per task).
- **Confidence:** high.
- **Tier 2:** REFERENCE-MATCH on both task paths. Horizon_time push_t = 5×0.1 = 0.5s (matches ref); anything = 7×0.075 = 0.525s (matches ref). The 2026-07-14-era "33× more solve work per tick" claim is stale — dt+N are now both aligned per task.

## 4.f — delta initial guess

- **Reference:** `c3.cc:312-316` — `delta_init = zeros; if (delta_option == 1) delta_init.head(n_x_) = x0`. YAML default `delta_option: 1` → first n_x components of every delta[k] initialized with the CURRENT STATE.
- **Port:** `admm_solver.py` — `_solve_c3plus` initializes delta with zeros unconditionally (I did not find a `delta_option` analog in the port).
- **Tag:** LOAD-BEARING (ADMM initial guess).
- **Confidence:** high.
- **Tier 2 — NEW DIVERGENCE surfaced by c3 lib deep read.** Reference bias-initializes delta.head=x0, port starts from origin. At convergence both should converge to same delta*, but at 25 iterations of non-converging port ADMM the initial condition may matter more.

## 4.g — end_on_qp_step (final rollout)

- **Reference:** `c3.cc:336-347` — after ADMM loop + final QP solve, IF `end_on_qp_step=false` (reference default), compute `z_sol[i].x = A·x_sol[i-1] + B·u_sol[i-1] + D·λ_sol[i-1] + d` — an LCS ROLLOUT of the state trajectory using the solved (u, λ). This produces a state trajectory that IS LCS-feasible even if the ADMM didn't fully converge.
- **Port:** `_solve_c3plus` returns the QP-solved `x_seq` directly. No LCS rollout to enforce feasibility.
- **Tag:** LOAD-BEARING at non-convergence. At full convergence the QP-solved x IS LCS-feasible so the rollout is a no-op; at non-convergence they differ.
- **Confidence:** high.
- **Tier 2:** NEW DIVERGENCE surfaced. Given port ADMM does NOT converge (4.l `iters=25/25 primal ~3.87`), the port's `x_seq` may be LCS-infeasible in ways the reference's rollout would correct. Belongs in the coupled band-aid subset with 3.j and 3.g.

## 4.h — Cross-tick warm-start

- **Reference:** `sampling_c3_options.yaml: warm_start: false` → no cross-tick warm-start. Per `c3.cc:396-397`, the `warm_start_` boolean gates the SetInitialGuessQP warm-start path.
- **Port:** No cross-tick warm-start (per memory `project_admm_no_warmstart.md`).
- **Tag:** COSMETIC (both OFF).
- **Confidence:** high.
- **Tier 2:** CONFIRMED both cold-start per tick. Prior port memory `project_admm_no_warmstart.md` framing "port has no cross-tick warm-start" is a divergence-from-reference — but reference is ALSO OFF, so this is CONFORMANT not divergent. **Prior memory framing corrected.**

## 4.i — Within-Solve warm-start (ADMM iter carryforward)

- **Reference:** `c3.cc:394-412 SetInitialGuessQP` — interpolates `warm_start_x_[admm_iter-1]` with `solve_time_/dt` weight into the current QP's initial guess. Only fires when `warm_start_ = true`. **Since `warm_start=false` in YAML, this branch is INERT for anything+push_t.** Delta / w carry naturally across ADMM iterations regardless.
- **Port:** `_solve_c3plus` for-loop at line 1119 carries `delta`, `omega`, `z_sol` across iterations by scope. No explicit reset. Implicit warm-start of ADMM state across iterations.
- **Tag:** COSMETIC (both agents' within-Solve state carries across iterations by scope).
- **Confidence:** high.
- **Tier 2:** CONFIRMED both agents implicitly warm-start delta/omega across ADMM iterations of a single Solve() call. The reference's *explicit* SetInitialGuessQP-based warm-start is INERT under `warm_start=false`.

## 4.j — penalize_changes_in_u_across_solves

- **Reference:** `c3.cc:302-310` — when `options_.penalize_input_change`, the input cost is rebuilt PER SOLVE as `2·R·u - 2·R·u_sol_prev` (i.e., `‖u - u_prev‖²_R` instead of `‖u‖²_R`). Reference `anything` YAML `penalize_changes_in_u_across_solves: true`; reference `push_t` YAML `false`.
- **Port pre-fix:** `admm_solver.py:123` was hardcoded `True` for both tasks; behaviorally correct for box, DIVERGENT for T.
- **Port post-fix (commit 84823fe, 2026-07-17):** `admm_solver.py:123 self._penalize_input_change = penalize_input_change` (constructor arg, default `True`); `main.py:530 _penalize_input_change = (task_name != "push_t")` gates the value per task. Push_t now runs with `False` (matches reference); other tasks unchanged.
- **Tag:** LOAD-BEARING → PARTIAL-REFERENCE-MATCH (T-task fully conforms; box unchanged since it was already correct).
- **Confidence:** high.
- **Tier 2 — CONFIRMED conformant post-fix.** The port scaffolding at `admm_solver.py:1022-1027` and `admm_solver.py:1539` was already wired to the flag; only the fixed-True default was DIVERGENT.

## 4.k — SolveSingleProjection (Bui eq 12)

- **Reference `c3_plus.cc:174-221 C3Plus::SolveSingleProjection`:**
  ```cpp
  eta_larger = eta * sqrt(w_eta) > lambda * sqrt(w_lambda);
  delta.λ = eta_larger.select(0, lambda_c);
  delta.η = eta_larger.select(eta_c, 0);
  delta.λ = delta.λ.cwiseMax(0);   delta.η = delta.η.cwiseMax(0);
  ```
- **Port `admm_solver.py:809-835 _project_componentwise`:**
  ```python
  sqrt_ratio = float(np.sqrt(u_lambda / u_eta))
  cond1 = (eta >= 0.0) & (eta >= sqrt_ratio * lam)
  cond2 = (lam >= 0.0) & (eta <  sqrt_ratio * lam)
  delta_lam = np.where(cond2, lam, 0.0)
  delta_eta = np.where(cond1, eta, 0.0)
  ```
  Mathematically equivalent: `eta * sqrt(w_eta) > lambda * sqrt(w_lambda)` ⟺ `eta > lambda * sqrt(w_lambda/w_eta) = lambda * sqrt(u_lambda/u_eta)`. Both apply the same case selection + ≥0 clip.
- **Tag:** COSMETIC-EQUIVALENT.
- **Confidence:** high.
- **Tier 2:** CONFIRMED via c3 lib deep-read line-by-line. **Port's projection matches reference Bui eq (12) exactly** — this piece is not where the port's ADMM non-convergence originates. Non-convergence origin is upstream (3.g Stewart-Trinkle F[γ,γ] rank-deficiency).

## 4.l — ADMM convergence at runtime

- **Reference:** MIQP is exact per iteration → ADMM converges quickly (3 iters is enough with adaptive ρ). Reference push_t (C3+) reaches similar tolerance behavior via `rho_scale=3` geometric ramp.
- **Port pre-fix (t_long60, 2026-07-17):** admm_iter=3 + `_rho_scale=1.0` + `--c3plus-projection lcp` → 903/903 solves `mono=False`, primal 36→61 (backwards), 43% LCP-inf events.
- **Port mid-fix (t_ref3fix, commit 08003e1):** admm_iter=3 + `_rho_scale=3.0` + `componentwise` → `mono=True` restored, primal 6.9→1.7. Still under ST-with-rank-deficient F[γ,γ].
- **Port post-Anitescu (2026-07-25, p62 baseline):** admm_iter=3 + `_rho_scale=3.0` + `componentwise` + **Anitescu single PSD F**. Log line 224: `primal 7.5943→1.2125, dual 2828.1→1014.3, mono=True, iters=3/3, rho_start=100.0 rho_end=2700.0`. Residual floor lower than the ST-era ~1.7 because Anitescu's single PSD F eliminates the (γ,γ) rank-deficiency that made valid (λ,η) pairs non-unique.
- **Tag:** LOAD-BEARING → REFERENCE-MATCH for T-push (post-Anitescu).
- **Confidence:** high.
- **Tier 2:** REFERENCE-MATCH confirmed at runtime on the p62 baseline. **The 3.g Stewart-Trinkle rank-deficient F[γ,γ] tie-in is now historical** — under Anitescu default the ADMM is monotone-decreasing to a lower floor per solve.

## 4.m — Mode-switch branches

- **Reference:** `sampling_based_c3_controller.cc:1140-1320`. 8 branches: (kToReposUnproductive, kToReposCost, kToC3Xbox, kToC3ReachedReposTarget, kToC3Cost, kStayInRepos + kToBetterRepos + kNewSample, achieved_fixed_goal implicit).
- **Port:** `mode_switch.py:decide_mode`. 7 SwitchReasons: {kStayInC3, kStayInRepos, kToReposCost, kToReposUnproductive, kToC3Cost, kToC3ReachedReposTarget, kToBetterRepos, kForceC3Watchdog}. Adds `kForceC3Watchdog` (steps_since_improve watchdog — port-only).
- **Tag:** LOAD-BEARING (structure).
- **Confidence:** high.
- **Tier 2:** CONFIRMED port omits {kToC3Xbox (xbox teleop only — reference-only), AddToUnsuccessfulBuffer, wall_offset for repos target, pursued_target_source_ tracking}. Port adds {kForceC3Watchdog, PORT_DISABLE_CONTACT_LOSS_GATE opt-in}. Core mode-switch logic (cost-gap × progress × hysteresis × finished_repos) is CONFORMANT.

## 4.n — Altitude gate on free→c3 transition

- **Reference:** `sampling_based_c3_controller.cc:1290-1293` — cost-based free→c3 switch is AND-gated by:
  ```cpp
  (x_lcs_curr[2] < z_height + c3_min_clearance + wall_offset
   || !sampling_params_.ee_z_close)
  ```
  I.e., EE altitude must be below "contact zone" OR ee_z_close feature is off.
- **Port:** `mode_switch.py:decide_mode` accepts `ee_z_gate_pass: bool = True` kwarg (T1a port of this gate). When False, both free→c3 branches (kToC3ReachedReposTarget and kToC3Cost) are suppressed inline.
- **Tag:** LOAD-BEARING.
- **Confidence:** high.
- **Tier 2:** CONFIRMED-CONFORMANT via port source (T1a landing per file comment). Whether it's ACTIVE at runtime depends on wrapper's `ee_z_gate_pass` computation (belongs to (5) sim or (2) reposition depending on which computes ee_z + threshold).

## 4.o — Hysteresis (kind × near_goal)

- **Reference:** `sampling_based_c3_controller.cc:1175-1226` — separate `hyst_c3_to_repos`, `hyst_repos_to_c3`, `hyst_repos_to_repos` (absolute) and `_frac` variants (relative), selected by `use_relative_hysteresis` flag.
- **Port:** `mode_switch.py:_hysteresis` — matches structure exactly, with the addition of `_position` variants (`hyst_c3_to_repos_position`, etc.) for the `near_goal=True` regime.
- **Tag:** COSMETIC-EQUIVALENT.
- **Confidence:** high.
- **Tier 2:** CONFIRMED via side-by-side source read.

## 4.p — Progress metric implementation

- **Reference:** `KeepTrackOfC3ModeProgress` (mentioned at cc:1156 but implementation is elsewhere in the file) — tracks cost improvement in c3 mode, sets `met_minimum_progress` false after `num_control_loops_to_wait` ticks without cost decrease.
- **Port:** `progress.py:ProgressTracker` — same abstraction, `steps_since_improve` field, `met_progress(near_goal)` predicate that gates on the `num_control_loops_to_wait[_position]` field per `ProgressMetric` (`kPosCost | kRotCost | kPosOrRotCost`).
- **Tag:** COSMETIC-EQUIVALENT structurally.
- **Confidence:** medium (reference implementation not read line-by-line; port confirmed).
- **Tier 2:** Structural conformance verified; leaf-level parity would need full reference `KeepTrackOfC3ModeProgress` read (not done in this pass, low priority — the metric is a well-defined "steps since cost improved" counter with mode-specific wait threshold).

## 4.q — LCS h_is_zero → LCP pre-solve

- **Reference:** `c3.cc:283-299` — if `h_is_zero_` (LCS H matrix all-zero, meaning passive dynamics: no dependence on u), pre-solve λ_0 via `MobyLcpSolver::SolveLcpLemke(F[0], E[0]·x0 + c[0], &lambda0)`, add as initial-force constraint.
- **Port:** No analog. Always runs full ADMM regardless.
- **Tag:** LOAD-BEARING IFF `h_is_zero_` at runtime.
- **Confidence:** high.
- **Tier 2:** For push_anything (actuated Franka arm), `H = Jn·Jf_u` where `Jf_u = M⁻¹·B` (non-zero). Reference `h_is_zero_` would be FALSE for any actuated system. **INERT-BY-CONFIG for actuated pushing task.** Would matter only for passive-dynamics probes (e.g., free-fall LCS).

## 4.s — Port-only contact-entry gate

- **Reference:** `sampling_based_c3_controller.cc:1284-1309` — free→c3 cost-based transition is AND-gated ONLY by (i) the altitude ceiling (4.n: `x_lcs_curr[2] < z_height + c3_min_clearance + wall_offset || !ee_z_close`) and (ii) the cost-gap hysteresis. There is NO distance-to-box surface gate.
- **Port:** `sampling_based_c3_controller.py:1487-1512` adds a distance gate: when `finished_repos=True and use_contact_entry_gate=True`, it flips `finished_repos → False` (blocking `kToC3ReachedReposTarget`) if `ee_to_surf ≥ contact_entry_surface_threshold` (default 60 mm) OR `ee_to_box ≥ contact_entry_threshold` (default 90 mm) depending on `use_surface_entry_gate`. Discovered 2026-07-17 blocking every IK arrival on push_t (IK landing 65–68 mm reads → 24 `[ENTRY-GATE] ... block kToC3ReachedReposTarget` events over 60 s; zero free→c3-via-arrival transitions).
- **Tag:** LOAD-BEARING (port-only) → DISABLED-FOR-T.
- **Confidence:** high.
- **Tier 2 — post-fix runtime 2026-07-17:** `config/sampling_c3_kik_t.yaml:use_contact_entry_gate: false` (commit 08003e1). Post-fix log shows 0 `[ENTRY-GATE]` events. The reference-equivalent altitude ceiling (4.n, `c3_min_clearance=0.01`) is unchanged and still fires. **REFERENCE-MATCH for T-push.** Box-push YAML unchanged — port default `True` still holds there.

## 4.r — Port-only env-tuned R + u-bounds (PUSHA_STAGE5_*)

- **Reference:** Fixed R + u-bounds from YAML.
- **Port:** `main.py:346-358` sets env defaults `PORT_U_HORIZONTAL=10, PORT_U_VERTICAL=3, PORT_R_VECTOR=0.1,0.1,10` for EE-space runs. These override the planner's per-axis u-bounds (Fx, Fy, Fz) and R-cost diagonal. Port-only "Stage 5 alignment" package.
- **Tag:** LOAD-BEARING iff EE-space active AND env unset (default → these values apply).
- **Confidence:** high.
- **Tier 2:** For default R^7 box run (no `--ee-space`), these env-defaults do NOT trigger the EE-space-only branches. **INERT for the default box run.** Belongs to a subsystem-4.5 or Stage-5 opt-in cluster.

## Coupling observed (from code + Tier-2 evidence)

- **4.a ↔ 4.b ↔ 4.c ↔ 4.l** — Solver-choice cluster. Two reference regimes exist per task family:
  - Reference `anything` (box): MIQP + admm_iter=3 + rho_scale=3 → exact per-iter LCP + adaptive ρ = 3 iters sufficient.
  - Reference `push_t`: **C3+ (Bui eq 12) + admm_iter=3 + rho_scale=3** — same admm-iter/rho-schedule discipline, closed-form projection.
  Port T-push (commit 08003e1) is now C3+ + admm_iter=3 + rho_scale=3 + componentwise projection = **REFERENCE-MATCH to `push_t` regime**. Post-fix ADMM is `mono=True` on 100% of solves; residual floor ~1.7 in 3 iters is the reference's own non-convergence-tolerance behavior for this regime.
  Port box-push canonical still runs `--admm-iter 25` (an off-reference regime), pending its own rho_scale+iter audit.
- **4.d ↔ 4.e ↔ 4.b** — planning horizon × dt × admm_iter compose into TOTAL SOLVE WORK per tick. **RESOLVED 2026-07-25**:
  - T-push: 5 knots × 0.1s × 3 iters = REFERENCE-MATCH to `push_t/sampling_c3plus_options.yaml`.
  - Box-push: 7 knots × 0.075s × 3 iters = REFERENCE-MATCH to `anything/sampling_c3_options.yaml` (per `main.py:611-629` task gate).
  Both paths now reference-conformant on solve-work per tick.
- **4.f ↔ 4.g ↔ 4.l** — the three "at-non-convergence-matters" divergences. delta_option (initial guess), end_on_qp_step (final rollout), and iters=25/25 non-convergent behavior. Under full convergence all three would be equivalent to reference; at the port's regime they all contribute to trajectory divergence.
- **4.j ↔ 1.k** — reference `penalize_changes_in_u_across_solves=true` for anything AND reference `w_input=0, w_input_reg=0` for OSC input-smoothing. The reference smooths CONTROL at TWO places (planner solve-to-solve + OSC input-smoothing weight, both configurable). Port has neither analog at either place.
- **4.m ↔ 4.n ↔ 2.l** — mode-switch cluster (branch structure + altitude gate + finished-flag semantics). Port has close structural parity to reference on all three. Divergent leaf-level values (specific hysteresis, altitude, finished-cost) would live in the YAML — belongs in a per-value audit not this Tier-1 pass.
- **4.l ↔ 3.g ↔ 3.j ↔ 3.n ↔ 3.p** — the LCS↔ADMM MECHANICAL LINK. **RESOLVED 2026-07-25:** the contact-model flip (3.g → Anitescu) has been executed on the default path. Downstream cascade landed as predicted — box_ground_drag (3.j) `_box_drag_c=0.0`, three normal-row patches (3.n) doubly-inert, n_lambda (3.p) = 4·n_c runtime-confirmed, ADMM (4.l) monotone-decreasing under the single PSD F block. The 2026-07-14 "central cross-subsystem coupling to be resolved" is now a historical anchor, not open work.

## Deferred / out-of-planner-scope
- Force-tracking `λ_des` derivation and OSC coupling → belongs to (1) executor + (2) wrapper.
- λ trace / stashing for the OSC (`last_lambda_n_first`, etc.) → COSMETIC (structural adapter for the wrapper).
- CI_MPC C3 (`ci_mpc_c3.py`, `C3MPC` class with Lorentz projection) → alternate path via `--solver c3` argparse; used only in ablations (per `--solver` default = `c3plus`). AttributeError observed earlier this session (`_last_Q` missing on C3MPC) → pre-existing bug in C3MPC path, not exercised by default.

## Planner/ADMM/mode-switch Tier-2 verdict roll-up

| # | Divergence | Tier-1 | Tier-2 (this pass) |
|---|---|---|---|
| 4.a | Solver class / projection | LOAD-BEARING → PARTIAL-REF-MATCH | **REFERENCE-MATCH for T-push** (C3+ + componentwise via commit 08003e1); reference `push_t` also C3+; reference `anything` = MIQP; box-push in port stays C3+ |
| 4.b | admm_iter | LOAD-BEARING → PARTIAL-REF-MATCH | **CONFIRMED runtime** — T-push port 3 (matches ref, commit 4c3bad5); box-push port 25 (off-ref); reference 3 |
| 4.c | rho / rho_scale | LOAD-BEARING → REF-MATCH | **CONFIRMED runtime** — port `_rho_scale=3.0` per-iter (commit 08003e1); reference `rho_scale=3` per iter; runtime ρ ramp 100→2700 over 3 iters |
| 4.d | Horizon N | LOAD-BEARING → REFERENCE-MATCH | **REFERENCE-MATCH 2026-07-25** — task-conditional (T=5, box=7) per `main.py:611-629` |
| 4.e | Planning dt | LOAD-BEARING → REFERENCE-MATCH | **REFERENCE-MATCH 2026-07-25** — task-conditional (T=0.1, box=0.075) per `main.py:611-629` |
| 4.f | delta initial guess | LOAD-BEARING | **NEW DIVERGENCE** — port zeros; reference `delta_option=1` → head=x0 |
| 4.g | end_on_qp_step (rollout) | LOAD-BEARING at non-conv | **NEW DIVERGENCE** — port direct QP; reference LCS rollout |
| 4.h | Cross-tick warm-start | COSMETIC (both off) | **CONFIRMED conformant** — corrects prior memory framing |
| 4.i | Within-Solve warm-start | COSMETIC | **CONFIRMED conformant** — both implicit carry across iters |
| 4.j | penalize_input_change | LOAD-BEARING | **NEW DIVERGENCE** — port always absolute; reference toggles |
| 4.k | SolveSingleProjection (Bui eq 12) | COSMETIC-EQUIVALENT | **CONFIRMED via c3 clone** — port projection = reference exactly |
| 4.l | ADMM convergence | LOAD-BEARING → REFERENCE-MATCH | **REFERENCE-MATCH 2026-07-25** — p62 baseline `mono=True iters=3/3 primal 7.59→1.21` under Anitescu single PSD F; ST rank-deficiency tie-in historical |
| 4.m | Mode-switch branches | LOAD-BEARING | **CONFIRMED-CONFORMANT** structure + noted port-only add-ons |
| 4.n | Altitude gate | LOAD-BEARING | **CONFIRMED-CONFORMANT** via T1a port |
| 4.o | Hysteresis lookup | COSMETIC-EQUIVALENT | **CONFIRMED** |
| 4.p | Progress metric | COSMETIC-EQUIVALENT | **CONFIRMED structurally** |
| 4.q | LCS h_is_zero LCP pre-solve | UNKNOWN | **RESOLVED-INERT-BY-CONFIG** — actuated pushing task has H ≠ 0, branch never fires reference-side |
| 4.r | PUSHA_STAGE5_* env defaults | LOAD-BEARING iff EE-space | **CONFIRMED-INERT** for default R^7 box run |
| 4.s | Port-only contact-entry gate | LOAD-BEARING → DISABLED-FOR-T | **REFERENCE-MATCH for T-push** (commit 08003e1 `use_contact_entry_gate: false` in `sampling_c3_kik_t.yaml`); pre-fix 24 block events over 60s; post-fix 0. Box-push YAML unchanged |
| 4.t | `PursuedTargetSource` state | COSMETIC → REF-STRUCTURE-MATCH | Reference enum + state at `.h:60,505` mirrors kNoTarget/kPrevious/kNewSample/kFromBuffer. Port lacked. Landed 2026-07-17 commit 9a39972 as an additive telemetry field on the `[GS]` line, derived from the port's existing string-labels via `pursued_from_label(mode, best_src)`. `kFromBuffer` reserved — its emitter path is the reference `AddToUnsuccessfulBuffer` which the port does NOT have (see 4.u) |
| 4.u | `AddToUnsuccessfulBuffer` failed-sample buffer | LOAD-BEARING (dispatcher-behavior) | Reference `sampling_based_c3_controller.cc:2161-2183` writes failed c3 attempts to a FIFO consumed by `GenerateSampleStates` at cc:905 to reject spatially-close candidates. Port has no analog. NOT implemented — write-only would be dead-weight; full read+write pair changes sampling behavior (out of scope per 2026-07-17 lock-in). Scope conversation candidate |

21 entries → 8 REFERENCE-MATCH (4.a/4.c/4.d/4.e/4.l/4.s/4.t + 4.b runtime; landed across 2026-07-17 commits 08003e1 + 9a39972 + 4c3bad5 and 2026-07-25 via Anitescu flip / task-conditional dt+N wiring at `main.py:611-629`) + 1 PARTIAL-REF-MATCH (4.j — T only) + 3 CONFIRMED-CONFORMANT + 3 NEW-DIVERGENCE remaining (4.f delta_option, 4.g end_on_qp_step, and their downstream — all "at-non-convergence-matters" divergences whose behavioral impact is now smaller since ADMM converges monotonically) + 2 COSMETIC-EQUIVALENT + 2 INERT (h_is_zero + Stage-5) + 2 structurally-conformant + 1 LOAD-BEARING-open (4.u AddToUnsuccessfulBuffer). Zero remaining UNKNOWNs.

## 2.k caution result

Verified the "inert" areas individually:
- **Cross-tick warm-start (4.h)**: both agents cold-start — CONFIRMED. But VERIFIED the reference's within-Solve warm-start (4.i) is ALSO configured OFF via `warm_start: false` (not just cold-start-per-iter — the whole SetInitialGuessQP path is inert). Two mechanisms in same area, each verified.
- **LCP pre-solve (4.q)**: verified INERT because H ≠ 0 for actuated push_anything — not just missing analog.
- **Stage-5 env defaults (4.r)**: verified inert IF NOT in EE-space mode — not silently active in R^7.
- **Both mode-switch branches (4.m)**: verified each of 7 port SwitchReasons vs 8 reference branches; explicitly enumerated the two agents' branch sets.

**No hidden live mechanism uncovered.** All live mechanisms are explicitly tagged in the table.

## Planner Tier-2 evidence artefacts

- Diagnostic commit: `67232d7` (`diag(plan-tier2): PUSHA_PLAN_T2_DIAG log-only planner/ADMM disclosure`).
- Instrumentation guard: `PUSHA_PLAN_T2_DIAG=1` — default OFF, byte-identical to `dd2294d` baseline.
- Filtered run log: `audit_output/plan_tier2/run_default.log`
- Summary + reference-side c3 lib + dairlib reads: `audit_output/plan_tier2/SUMMARY.md`

---

# Subsystem (5) — Sim / env-builder / URDF geometry

## Sources read

- **Reference** (`dairlib_sampling_c3 @ push_anything_dev 257e3ed`):
  `examples/sampling_c3/sampling_c3_utils.{h,cc}` (`AddFrankaToPlant`, `AddWallsToPlant`, `AddObjectToPlant`, `AddLCSModelToPlant`, `kFrankaModel`, `kEndEffectorModel`, `kToolAttachmentFrame`, `kFrankaToGroundOffset`, `kWallLengthX/Y/etc`);
  `examples/sampling_c3/urdf/end_effector_full.urdf` (flange + peg + tip chain, tip sphere r=0.0195 μ=1.0);
  `examples/sampling_c3/urdf/ground.urdf` (5×0.91×0.1 μ=1.0 point-contact);
  `examples/sampling_c3/urdf/push_t.sdf` (sim: 2 boxes with `compliant_hydroelastic`);
  `examples/sampling_c3/urdf/push_t_control.sdf` (LCS: same + 3 sphere witnesses r=0.001);
  `examples/sampling_c3/urdf/expo_box/expo_box.sdf` + `expo_box_controller.sdf` (mesh convex decomposition);
  `examples/sampling_c3/anything/parameters/sim_params.yaml` (`dt: 0.001`);
  `examples/sampling_c3/franka_sim.cc:79-80` (`AddMultibodyPlantSceneGraph(&builder, sim_dt)`);
  `examples/sampling_c3/franka_sampling_c3_controller.cc:103` (LCS plant `dt=0.0` continuous).
- **Port** (`push_anything_ADMM @ main dd2294d + ... + 043f378`):
  `sim/env_builder.py:1-590` (`build_environment`, `_box_sdf`, `_sphere_sdf`, `_tshape_sdf`, `PUSHER_RADIUS`, `ROBOT_BASE_XYZ`, `INITIAL_ARM_Q`);
  `config/tasks.yaml` (`pushing`, `hard_pushing`, `shepherding`, `push_t` task_cfg values).

**Sim/URDF scope:** the Drake plant assembly + object/table URDFs + sim time_step + Drake contact model. Excludes the LCS admission filter (subsystem 3) and the OSC frame identity (subsystem 1, already resolved).

## Index table

| # | Divergence | Tier-1 tag | Tier-2 |
|---|---|---|---|
| 5.a | Franka URDF choice + base pose | LOAD-BEARING | CONFIRMED — same URDF (`panda_arm.urdf`); DIFFERENT world pose (port `[0, −0.6, 0]` vs reference Identity) |
| 5.b | End-effector attachment mechanism | LOAD-BEARING | CONFIRMED — port programmatic pusher body welded to panda_link8+[0,0,0.05]; reference URDF-parsed end_effector_full welded to panda_link7+kToolAttachmentFrame+180deg |
| 5.c | Pusher tip radius | LOAD-BEARING (from executor 1.f) | CONFIRMED runtime — port 0.025 m vs reference 0.0195 m (**3 mm** larger tip) |
| 5.d | Pusher friction μ | LOAD-BEARING | CONFIRMED runtime — port 0.4 (pushing task_cfg) vs reference URDF μ=1.0 |
| 5.e | Manipuland SDF/URDF construction | LOAD-BEARING (structure) | CONFIRMED — port programmatic (`_box_sdf`, `_tshape_sdf`) single-body single-collision; reference URDF-file multi-body multi-collision (push_t: 2 boxes; expo_box: mesh convex decomposition) |
| 5.f | Manipuland friction μ | LOAD-BEARING | CONFIRMED runtime — port task_cfg (pushing 0.4, push_t 1.0); reference URDF `mu_dynamic=0.3` (push_t + expo_box, hydroelastic) |
| 5.g | Manipuland Drake contact model | LOAD-BEARING | CONFIRMED — port POINT contact + Coulomb; reference COMPLIANT HYDROELASTIC (modulus 3e7, mesh resolution 0.18) + Hunt-Crossley dissipation 10 |
| 5.h | Ground / table geometry + friction | LOAD-BEARING | CONFIRMED runtime — port `(2, 2, 0.1)` μ=(static 0.6, dynamic 0.5) on world_body; reference `(5, 0.91, 0.1)` μ=1.0 welded via `kFrankaToGroundOffset=[0,0,-0.029]` |
| 5.i | LCS-vs-Sim URDF split | LOAD-BEARING (structure) | NEW STRUCTURAL — reference has SEPARATE `*.sdf` (sim, hydroelastic) and `*_control.sdf` (LCS, adds sphere witnesses); port uses SAME plant for sim + LCS |
| 5.j | Sphere-witness positions for T-ground contact | LOAD-BEARING (T-task) | CONFIRMED-DIVERGENT — port synthesized (`_tshape_vertex_set_body_frame(3)`) at (+0.13, 0, -0.02), (-0.05, ±0.08, -0.02) vs reference URDF spheres at (-0.12, ±0.08, -0.02), (+0.08, 0, -0.02). **Coordinate origin differs (port t_link at combined-CoM; reference vertical_link at link origin)** — positions are STRUCTURALLY DIFFERENT triangles |
| 5.k | Walls (workspace bin) | LOAD-BEARING iff enabled | INERT — reference `include_walls=false` for anything+push_t default; port has no wall mechanism |
| 5.l | Sim time_step | LOAD-BEARING | CONFIRMED CONFORMANT — port 0.001 s; reference `sim_params.yaml: dt=0.001` |
| 5.m | LCS plant discretization | LOAD-BEARING | NEW DIVERGENCE — reference LCS plant `AddMultibodyPlantSceneGraph(&builder, 0.0)` = **continuous plant** (dt=0); port LCS uses SAME plant as sim (dt=0.001, discrete). LCS `linearize_discrete` applies its own `dt` on top of the sim plant. |

## 5.a — Franka URDF choice + base pose

- **Reference:** `sampling_c3_utils.h:12` `kFrankaModel = "package://drake_models/franka_description/urdf/panda_arm.urdf"`. Welded at `X_WI = Identity` (line 24-26).
- **Port:** `env_builder.py:257` — same URDF (`drake_models/franka_description/urdf/panda_arm.urdf`). Welded at `ROBOT_BASE_XYZ = [0.0, -0.6, 0.0]` (60 cm along -y from world origin).
- **Tag:** URDF CONFORMANT; base pose LOAD-BEARING (changes workspace coordinates by 60 cm along -y).
- **Confidence:** high.
- **Tier 2:** CONFIRMED via `[SIM-T2] robot_base_xyz=[0.0, -0.6, 0.0] (reference: Identity() = [0,0,0])`. Reference workspace and port workspace live in DIFFERENT world-frame coordinates. This means comparing runtime positions requires a coordinate transform of 60 cm along -y. Not directly observable as a "bug" (both are internally consistent), but ALL other constants in the port that use world-frame positions are implicitly offset by this base pose (e.g., box init_xyz, goal_xy, ROBOT_BASE-relative positions).

## 5.b — End-effector attachment mechanism

- **Reference:** `sampling_c3_utils.cc:28-36` — `parser.AddModels(kEndEffectorModel)` parses `end_effector_full.urdf` (3-link chain: flange + peg + tip). Welded via `plant->WeldFrames(panda_link7, end_effector_flange, RigidTransform(RollPitchYaw(π, 0, 0), kToolAttachmentFrame=[0, 0, 0.107]))`. **URDF-based, 3-link kinematic chain, welded to panda_link7 with 180° x-rotation + 10.7 cm tool offset.**
- **Port:** `env_builder.py:270-301` — programmatic. `pusher_body = plant.AddRigidBody("pusher", ...)`, register sphere collision, weld to `panda_link8` at `RigidTransform([0.0, 0.0, 0.05])` — no rotation, 5 cm along +z from link8.
- **Tag:** LOAD-BEARING (different attachment link — 7 vs 8; different offset; different rotation).
- **Confidence:** high.
- **Tier 2:** CONFIRMED via `[SIM-T2] pusher_body='pusher' weld_parent='panda_link8' weld_offset=[0,0,0.05] (reference: end_effector_flange welded to panda_link7 at kToolAttachmentFrame=[0,0,0.107] + 180deg x-rotation)`. Note: panda_link7 is the LAST ACTUATED link before the end-effector; panda_link8 is a FIXED link 8.8 cm beyond link7 (per Franka URDF). Both agents end up with an EE somewhere PAST the last actuated link, but via different intermediate structure. The port's simpler weld may skip the URDF's tool-offset semantics. **Coupled to executor 1.f frame identity divergence** (already resolved).

## 5.c — Pusher tip radius

- **Reference:** `end_effector_full.urdf:66, 73` — sphere r=0.0195 m at `end_effector_tip`.
- **Port:** `env_builder.py:32` — `_PUSHER_RADIUS_DEFAULT = 0.025` m. Env-override `PORT_PUSHER_RADIUS` (default unset).
- **Tag:** LOAD-BEARING (kicked here from executor 1.f).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[SIM-T2] pusher_radius=0.025 (reference: end_effector_full.urdf sphere r=0.0195; env_PORT_PUSHER_RADIUS=<unset>)`. **3-mm larger tip in port.** Previously fought over in memory `project_S9_leaked_to_box_stage_e_blocked.md` — port tried globalizing to 0.0195 during T-work but regressed box closure, reverted. Load-bearing for both box and T tasks (changes contact geometry, wrist-load distance, LCS phi values, sample_reject_clearance-vs-radius interaction).

## 5.d — Pusher friction μ

- **Reference:** `end_effector_full.urdf:76-77` — `drake:mu_static value="1"`, `drake:mu_dynamic value="1"`. **Reference pusher tip μ=1.0 always.**
- **Port:** `env_builder.py:282` — `_pusher_mu = float(task_cfg.get("pusher_friction", 0.4))`. Pushing task default = 0.4; push_t task sets to 1.0.
- **Tag:** LOAD-BEARING (Drake harmonic-mean μ_eff = 2·μ_A·μ_B/(μ_A+μ_B); port pushing μ_eff = 0.4 vs reference μ_eff = 2·1·0.3/(1+0.3) = 0.462 for pusher-box).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[SIM-T2] pusher_mu=0.4 (reference: end_effector_full.urdf mu_static=1.0 mu_dynamic=1.0)`. **Port pusher-side μ=0.4 vs reference 1.0.** For push_t, port sets both sides to 1.0 → μ_eff=1.0 (matches reference). For box (pushing), port 0.4 vs reference 1.0. **Divergent by task**: push_t is reference-conformant, pushing is NOT.

## 5.e — Manipuland SDF/URDF construction

- **Reference:** URDF/SDF files. `push_t.sdf` = 2 links (vertical + horizontal), each with `compliant_hydroelastic` box collision. `expo_box.sdf` = mesh convex decomposition (~9 pieces via `expo_box_convex_N.obj`).
- **Port:** `env_builder.py:59-198` — programmatic SDF strings generated per task_cfg. `_box_sdf`: single box_link with 1 collision box. `_sphere_sdf`: single ball_link. `_tshape_sdf`: single t_link with 2 box collision elements (single-body-collapsed rigid — see docstring line 128-144). All point-contact μ from task_cfg (single scalar).
- **Tag:** LOAD-BEARING (structure of the manipuland — number of bodies, collision geometry type, hydroelastic vs point).
- **Confidence:** high.
- **Tier 2:** CONFIRMED-DIVERGENT. Structurally: port t_link (1 body, 2 box collisions) vs reference push_t (2 bodies fixed-joined, each 1 box collision). The port's docstring at line 130-134 notes "A fixed joint is a rigid connection, so a single link with two collision elements is DYNAMICALLY EQUIVALENT" — TRUE for dynamics but the COLLISION GEOMETRY sets are separate objects with different GeometryIds → affects `sd_pairs` iteration order in the LCS admission (3.a). Port `_tshape_sdf` NAMES the collision `manipulated_object::collision`; reference names them `vertical_link_volume` and `horizontal_link_volume` — different names, different SortedPair matching downstream.

## 5.f — Manipuland friction μ

- **Reference:** URDF `drake:mu_dynamic=0.3` for push_t + expo_box. Hydroelastic doesn't use separate mu_static.
- **Port:** task_cfg `friction`. Pushing: 0.4. Push_t: 1.0.
- **Tag:** LOAD-BEARING (different μ per task; combined with 5.d).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED. **Pushing task-side μ=0.4 vs reference 0.3.** Push_t task-side μ=1.0 vs reference 0.3 (port intentionally boosted for T-stability per source comment at env_builder.py:279-281). **Port push_t μ is intentionally divergent-from-reference by design** (the reference values fail T stability in the port setup — a port-level workaround per prior arc work).

## 5.g — Manipuland Drake contact model

- **Reference:** `<drake:proximity_properties>` with `<drake:compliant_hydroelastic/>`, `hydroelastic_modulus=3.0e7`, `mesh_resolution_hint=0.18`, `hunt_crossley_dissipation=10`. **Compliant hydroelastic contact model with Hunt-Crossley damping.**
- **Port:** Coulomb friction only (no `<drake:compliant_hydroelastic/>` in port's SDF generators). Point contact (Drake default when no hydroelastic tag).
- **Tag:** LOAD-BEARING (very different contact-force computation regime).
- **Confidence:** high.
- **Tier 2:** CONFIRMED via port SDF generators + reference URDF read. **Reference uses hydroelastic (soft-body force distribution over mesh) with dissipation damping; port uses rigid point contact.** Different contact-force profiles at impact (hydroelastic dissipates energy over contact area; point-contact impulses are near-instantaneous). Affects box tumble behavior, T-stability, contact-force smoothness. This is INDEPENDENT of the LCS contact model (3.g) — the LCS contact model is what the PLANNER predicts; this is what Drake SIM applies.

## 5.h — Ground / table geometry + friction

- **Reference:** `ground.urdf` — rigid box 5×0.91×0.1 m, origin at [0, 0, -0.05] in ground frame → top surface at ground-z=0. Welded to Franka's `panda_link0` at `kFrankaToGroundOffset=[0, 0, -0.029]` → **top surface at Franka-frame z = −0.029** (2.9 cm below Franka base). μ=1.0 static+dynamic.
- **Port:** `env_builder.py:238-245` — Drake `plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform([0.0, 0.0, -0.05]), Box(2.0, 2.0, 0.1), "table_collision", CoulombFriction(0.6, 0.5))`. Table welded to world_body (not to Franka). Top surface at world-z=0.
- **Tag:** LOAD-BEARING per geometry, per friction.
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[SIM-T2] table_size=(2.0, 2.0, 0.1) origin=[0,0,-0.05] table_mu=(static=0.6, dynamic=0.5) (reference: ground.urdf box=(5.0, 0.91, 0.1) origin=[0,0,-0.05] mu=1.0 static+dynamic; welded via kFrankaToGroundOffset=[0,0,-0.029])`. Both agents place table top at their world-z=0, but port's Franka is at world y=-0.6 (5.a) so the RELATIVE position differs. Table friction VERY DIFFERENT: port (0.6/0.5) vs reference (1.0/1.0). This affects the μ_eff computation for BOX-GND contact — port harmonic 2·0.4·0.5/(0.4+0.5) = 0.444 vs reference 2·0.3·1.0/(0.3+1.0) = 0.462 (harmonic of manipuland URDF μ + table URDF μ).

## 5.i — LCS-vs-Sim URDF split

- **Reference:** TWO separate plants:
  - SIM plant: `franka_sim.cc:80 AddMultibodyPlantSceneGraph(&builder, sim_dt=0.001)` — uses `push_t.sdf` (hydroelastic, no sphere witnesses).
  - LCS plant: `franka_sampling_c3_controller.cc:103 AddMultibodyPlantSceneGraph(&plant_lcs_builder, 0.0)` — uses `push_t_control.sdf` (adds sphere witnesses for LCS admission).
- **Port:** SINGLE plant used for both sim and LCS (via `LCSFormulator(plant, plant_ad=plant_ad, ...)`). Same URDF, same collision geometries.
- **Tag:** LOAD-BEARING (structural).
- **Confidence:** high.
- **Tier 2:** NEW STRUCTURAL DIVERGENCE surfaced. Reference decouples the "what does the plant do" (sim) from "what does the LCS see" (control) by using two different URDFs. Port collapses to one. Consequence: reference can add TINY sphere witnesses to the LCS URDF without affecting sim dynamics (r=0.001 spheres would be immaterial in sim but crucial for LCS pair-admission at exact witness positions). Port cannot do this — any collision geometry added for LCS admission would ALSO participate in sim. This is exactly why port has `_synthesize_manipuland_ground_contacts` (3.o): a programmatic LCS-only witness synthesis path that doesn't affect Drake sim.

## 5.j — Sphere-witness positions for T-ground contact

- **Reference `push_t_control.sdf:43-66`**: 3 sphere witnesses on `vertical_link` (frame origin at (0,0,0) of vertical bar):
  - `top_left_sphere` at (−0.12, +0.08, −0.02)
  - `top_right_sphere` at (−0.12, −0.08, −0.02)
  - `bottom_sphere` at (+0.08, 0.0, −0.02)
  Triangle spans link-x ∈ [−0.12, +0.08], link-y ∈ [±0.08], link-z = −0.02 (below vertical bar's bottom face).
- **Port `_tshape_vertex_set_body_frame(3)` at `lcs_formulator.py:452-456`**: 3 witnesses in port's `t_link` frame (origin at combined CoM ≈ (−0.05, 0, 0) relative to reference's vertical_link origin):
  - (+0.13, 0.00, −0.02) crossbar +x tip
  - (−0.05, +0.08, −0.02) stem +y tip
  - (−0.05, −0.08, −0.02) stem −y tip
- **Tag:** LOAD-BEARING for T-task; INERT for box.
- **Confidence:** high.
- **Tier 2:** CONFIRMED-DIVERGENT. Port and reference use 3-point triangles but at DIFFERENT relative positions. In world-frame terms, applying the CoM shift (port `t_link` origin ≈ reference `vertical_link` origin + (−0.05, 0, 0)):
  - Port witness 1 (+0.13, 0.00, -0.02) → reference-equivalent world = (+0.08, 0.00, -0.02) ≡ reference `bottom_sphere` ✓
  - Port witness 2 (-0.05, +0.08, -0.02) → reference-equivalent world = (-0.10, +0.08, -0.02) ≠ reference top_left_sphere at (-0.12, +0.08, -0.02) (**2-cm mismatch along x**)
  - Port witness 3 (-0.05, -0.08, -0.02) → reference-equivalent world = (-0.10, -0.08, -0.02) ≠ reference top_right_sphere at (-0.12, -0.08, -0.02) (**2-cm mismatch along x**)
- **Applied 2.k caution**: verified by CHECKING NUMBERS not just structure. Both use 3 witnesses (matches) but positions differ by 2 cm on 2 of 3 points. Not exact match. Would need port to shift stem witnesses by −0.02 along local-x to match reference exactly, OR port could re-derive from reference frame conventions.

## 5.k — Walls

- **Reference:** `sampling_c3_utils.cc:60-91 AddWallsToPlant` — 3-4 wall boxes (left, right, front, [back]) welded via `include_walls=true` argument to `AddFrankaToPlant`. Default: `include_walls=false` for anything+push_t.
- **Port:** No wall mechanism.
- **Tag:** LOAD-BEARING iff `include_walls=true`.
- **Confidence:** high.
- **Tier 2:** CONFIRMED-INERT via anything+push_t YAML defaults (no walls enabled). Belongs also to 3.e (object-wall LCS admission — same inert-by-config).

## 5.l — Sim time_step

- **Reference:** `sim_params.yaml: dt: 0.001` → `AddMultibodyPlantSceneGraph(&builder, 0.001)`.
- **Port:** `env_builder.py:200` `time_step: float = 0.001` default arg.
- **Tag:** LOAD-BEARING (Drake plant discretization step).
- **Confidence:** high.
- **Tier 2:** RUNTIME-CONFIRMED `[SIM-T2] time_step=0.001 (reference sim_params.yaml: dt=0.001 — CONFORMANT)`. **CONFORMANT.**

## 5.m — LCS plant discretization

- **Reference:** `franka_sampling_c3_controller.cc:103` — LCS plant built with `AddMultibodyPlantSceneGraph(&plant_lcs_builder, 0.0)` → **CONTINUOUS plant** (dt=0). LCS's Aydinoglu-8 first-order linearization derives dt from the LCS `options_.dt` (0.1 s planner cadence), NOT from the plant's discretization.
- **Port:** Uses the SAME plant as sim (`time_step=0.001` discrete). LCS's `linearize_discrete` applies its own `dt=0.05` on top. So the LCS effectively runs a 50-tick discretization over each planning step (0.05 s = 50 × 1 ms).
- **Tag:** LOAD-BEARING (LCS linearization from continuous vs discrete plant).
- **Confidence:** high.
- **Tier 2:** NEW DIVERGENCE surfaced by reference read. Reference LCS derives from a CONTINUOUS plant (uses autodiff on continuous dynamics). Port LCS derives from a DISCRETE plant (autodiff on Drake's stepped dynamics). Under Drake's semi-implicit Euler discretization at dt=0.001, the discrete plant's autodiff dynamics may differ from the continuous plant's by O(dt) terms. **May explain some of the "small tick-cadence-dependent drift" observations from prior arc work.** Belongs also in coupling with (4) — reference's continuous-plant + planning_dt=0.1 vs port's discrete-plant + planning_dt=0.05.

## Coupling observed (from code + Tier-2 evidence)

- **5.a ↔ 5.b ↔ 5.c ↔ 5.d ↔ 1.f** — end-effector chain. Franka base at [0,-0.6,0] (5.a) + pusher on link8+[0,0,0.05] (5.b) + sphere r=0.025 (5.c) + μ=0.4 (5.d) all compose into "port EE geometry." Reference: Franka at identity + tool-flange chain to link7 + sphere r=0.0195 + μ=1.0. **Cluster: five constants that together define the EE's physical form.** Executor 1.f resolved the structural equivalence (both zero-offset body Jacobians); the geometric values (5.c/5.d) still diverge.
- **5.h ↔ 5.g ↔ 3.j (box_ground_drag)** — ground friction (5.h port 0.5-0.6 vs reference 1.0) × manipuland friction (5.g port 0.3-0.4 vs reference 0.3) compose into μ_eff for BOX-GND. Port μ_eff ≈ 0.444; reference μ_eff ≈ 0.462. Small NUMERIC divergence in isolation. Coupled to 3.j: the BOX-GND single-pair admission (3.d) + weak friction (5.g/5.h) + ADMM non-convergence (4.l) = box-coasting predictions requiring drag band-aid.
- **5.g ↔ 3.g** — Drake sim contact model (hydroelastic reference vs point-contact port) mirrors the LCS contact model divergence (Anitescu vs Stewart-Trinkle) at a DIFFERENT LEVEL. **Both agents' SIM plant is what the executor applies torque against**. Reference sim is soft-body compliant; port sim is rigid point. This means the actual physical response the OSC counteracts differs between agents, INDEPENDENT of the LCS prediction divergence.
- **5.i ↔ 5.j ↔ 3.o** — the "reference has separate LCS URDF with sphere witnesses" cluster. Reference achieves 3-pair T-ground via URDF-defined tiny spheres (5.j positions) in a separate LCS URDF (5.i). Port achieves same via programmatic witness synthesis (3.o with different positions 5.j). Same functional goal, different implementations, DIFFERENT numeric witness triangle. Not-conforming-to-reference-values within a conforming-to-reference-structure.
- **5.l ↔ 5.m ↔ 4.e** — plant discretization × planning dt. Port: sim 1ms, plant discrete, LCS `linearize_discrete` at 0.05. Reference: sim 1ms, plant continuous for LCS, LCS `dt=0.1`. Two-agents' dt cascade differs in structure not just value.
- **5.c pusher radius ↔ box init_xyz ↔ contact_offset** — pusher radius (0.025 vs 0.0195) directly changes the effective push-point (contact-witness distance from EE body origin). Contact_offset computation at `env_builder.py:535` uses `PUSHER_RADIUS` — affects reachability + LCS phi (via the actual contact witness world position).

## Deferred / cross-subsystem items surfaced

- The reference's `end_effector_full.urdf` also has `end_effector_flange` (cylinder r=0.0315) + `end_effector_peg` (r=0.0127) collision geometries. These COULD participate in sim contacts (arm-table hits, arm-manipuland hits) if the arm swings into them. Port has NO analog — the port's pusher is JUST the tip sphere. If reference's flange/peg collide with the table during reposition, that's a divergent sim behavior. **Sub-scope: verify the reference's flange/peg collision geometries are NOT filtered out of the sim contact set.** Not verified in this pass — belongs in future sim-behavior audit if needed.
- Port ROBOT_BASE_XYZ=[0, -0.6, 0] means Franka is offset from world origin. Reference is at world origin. This affects ALL world-frame coordinates — box init_xyz [0, 0, 0.05] in port world = [0, 0.6, 0.05] in Franka-frame; reference `push_t` init pose likely uses Franka-frame directly. **Comparing box positions between agents requires the +0.6 y-shift** (or equivalent).

## Sim/URDF Tier-2 verdict roll-up

| # | Divergence | Tier-1 | Tier-2 |
|---|---|---|---|
| 5.a | Franka URDF + base pose | LOAD-BEARING (base pose only) | **CONFIRMED** — URDF match; base pose [0,-0.6,0] vs Identity |
| 5.b | EE attachment mechanism | LOAD-BEARING | **CONFIRMED** — programmatic pusher vs URDF chain |
| 5.c | Pusher tip radius | LOAD-BEARING | **CONFIRMED runtime** — 0.025 vs 0.0195 |
| 5.d | Pusher friction μ | LOAD-BEARING | **CONFIRMED runtime** — 0.4 (pushing) vs URDF 1.0 |
| 5.e | Manipuland SDF construction | LOAD-BEARING | **CONFIRMED-DIVERGENT** — port programmatic vs reference URDF |
| 5.f | Manipuland friction | LOAD-BEARING | **CONFIRMED-DIVERGENT** — port pushing 0.4 vs reference 0.3 |
| 5.g | Drake sim contact model | LOAD-BEARING | **CONFIRMED-DIVERGENT** — point vs hydroelastic |
| 5.h | Ground geometry + friction | LOAD-BEARING | **CONFIRMED runtime** — (2,2,0.1) μ=(0.6, 0.5) vs (5,0.91,0.1) μ=1.0 |
| 5.i | LCS-vs-Sim URDF split | LOAD-BEARING structure | **NEW STRUCTURAL** — port single plant vs reference two-URDF split |
| 5.j | T-witness positions | LOAD-BEARING (T-task) | **CONFIRMED-DIVERGENT** — 2-cm mismatch on 2 of 3 witnesses |
| 5.k | Walls | LOAD-BEARING iff enabled | **CONFIRMED-INERT** for anything+push_t default |
| 5.l | Sim time_step | LOAD-BEARING | **CONFIRMED CONFORMANT** — both 0.001 |
| 5.m | LCS plant discretization | LOAD-BEARING | **NEW DIVERGENCE** — reference continuous plant vs port discrete |

13 entries → 8 CONFIRMED-DIVERGENT + 2 NEW DIVERGENCE + 1 CONFIRMED-INERT + 1 CONFIRMED CONFORMANT + 1 partial (5.d divergent-for-pushing-only, conformant-for-push_t). Zero UNKNOWNs.

## 2.k caution result

Verified individually per mechanism, especially where "same structure" could mask "different values":
- 5.j T-witness positions — port and reference BOTH have 3 witnesses. If I stopped at "same count" that would mask the 2-cm position mismatch. VERIFIED numerically that positions differ.
- 5.h ground friction — port has static=0.6 + dynamic=0.5 (2 numbers); reference has 1.0 (single number). Verified separately not as "same friction structure."
- 5.d pusher friction is TASK-DEPENDENT (port pushing=0.4 divergent, push_t=1.0 conformant). Verified per-task rather than a single "port μ" claim.
- 5.a Franka URDF conformant BUT base pose divergent. Verified two separate mechanisms in the same setup step.
- 5.b weld mechanism differs (programmatic vs URDF chain) AND weld target link differs (link8 vs link7) AND rotation differs (identity vs 180°). Three separate divergences in one entry.

No "same overall thing" masking of an underlying divergent value. **2.k caution passed.**

## Sim/URDF Tier-2 evidence artefacts

- Diagnostic commit: `043f378` (`diag(sim-tier2): PUSHA_SIM_T2_DIAG log-only sim/URDF disclosure`).
- Instrumentation guard: `PUSHA_SIM_T2_DIAG=1` — default OFF, byte-identical.
- Filtered run log: `audit_output/sim_tier2/run_default.log`
- Summary + reference URDF deep reads: `audit_output/sim_tier2/SUMMARY.md`

---

# CONFORMANCE MAP — COMPLETE (all 5 subsystems; 2026-07-14 baseline, 2026-07-25 refresh)

## The central finding — contact-model cluster (RESOLVED 2026-07-25)

**One structural root, cascading compensations — the cascade landed as predicted.**

The port's Stewart-Trinkle contact model (3.g) had a rank-deficient F[γ,γ] block by construction (c3 lib clone confirmed via `c3/multibody/lcs_factory.cc:438-494`). The reference uses Anitescu (single PSD F block). This ONE choice cascaded into 4.l, 3.j, 3.n, 4.b, 4.d, 4.e — all downstream compensations.

**The port default was flipped to Anitescu (`lcs_formulator.py:86`); the predicted cascade landed:**

- **4.l** — ADMM now `mono=True iters=3/3 primal 7.59→1.21` on p62 baseline (was `iters=25/25 primal~3.87 non-convergent` under ST).
- **3.j** — `_box_drag_c = 0.0` (`lcs_formulator.py:108`). Band-aid disabled by default.
- **3.n** — three normal-row patches all default OFF at `lcs_formulator.py:115-117` AND their ST code paths are not walked under Anitescu default — doubly inert.
- **4.b** — port uses 3 ADMM iters (matches reference).
- **4.d/4.e** — task-conditional at `main.py:611-629`: T-push 5×0.1s, box-push 7×0.075s. Both match respective reference YAMLs.

**The LCS↔Drake mismatch the whole arc kept circling is closed on the default path.** ST retained under env `LCS_CONTACT_MODEL=stewart_trinkle` for regression-diff use only.

## Second cluster — executor over-drive + rotation-hold

- **1.e/1.f** — port compound authority `W_track · Kp_cart = 100 · 400 = 40 000` vs reference `1 · 200 = 200` (200× over-drive on position).
- **2.k** — reference OSC also holds identity-quaternion with `W_rot · Kp_rot = 10 · 800 = 8 000` per axis; port QP has ZERO rotation cost.
- **1.d** — port's task-only u-clamp lets `tau_g + u_clamped` exceed URDF cap (confirmed 97.44 Nm at joint 1 vs cap 87 Nm).
- **2.h/2.i** — port reposition speed=0.4 m/s vs reference 0.18, waypoint height 0.15 vs 0.06-0.077.

These are LOAD-BEARING and COUPLED to each other. Would not obviously be resolved by flipping the contact model alone.

## Third cluster — geometry/friction (5)

- **5.c** pusher radius 0.025 vs 0.0195
- **5.d** pusher μ 0.4 vs 1.0 (task-dependent)
- **5.g** Drake sim contact model point vs hydroelastic
- **5.h** ground μ (0.6/0.5) vs 1.0
- **5.a** Franka base pose [0,-0.6,0] vs Identity (world-frame coordinate shift)

Mostly INDEPENDENT constants. Some (5.g contact model) are load-bearing structurally; others (5.h ground μ) are small numeric offsets composing into small μ_eff differences.

## Coupling graph — consolidation-decision guide

| Cluster | Mechanisms | Status (2026-07-25) |
|---|---|---|
| **Contact-model cluster** | 3.g/3.h/3.n/3.p/3.j, 4.b/4.c/4.d/4.e/4.l | **RESOLVED 2026-07-25** — Anitescu flip landed on default path; all 10 items REFERENCE-MATCH / OBSOLETED / INERT |
| **Executor over-drive** | 1.e/1.f, 1.d/1.h, 2.h/2.i/2.j, 2.k | COUPLED — cascading gains + rotation-hold — HOLD (Phase-1 OSC swap recert-falsified 2026-07-14; no unblock recipe pending) |
| **Reposition Stage-A** | 2.a/2.b/2.c/2.j (REFCONF_REPOSITION_PWL path) | COUPLED to executor over-drive via v_ee_desired handshake — HOLD |
| **kIK reposition** | 2.e/2.f/2.g (traj_type=kIK subset) | COUPLED but INDEPENDENT of reference (port-only) — **SAFE to REMOVE wholesale** |
| **Port-only opt-in flags** | 3.k/3.m/3.n (already OFF), 4.r Stage-5 | INDEPENDENT, INERT — **SAFE to REMOVE** (delete-the-flag) |
| **Geometry constants** | 5.c/5.d/5.h (small numeric divergences) | INDEPENDENT — **SAFE to default-to-reference + tripwire** |
| **Ground URDF split** | 5.i/5.j (reference sphere-witnesses vs port synthesis) | INDEPENDENT of contact-model choice now that Anitescu default is stable — evaluate on its own |
| **port-todo #7 (Q construction)** | ρ / G / Q coupled triple flip | BLOCKED — see `docs/superpowers/investigations/2026-07-23-item7-deep-investigation.md`; G-matrix prong-1 infrastructure landed 2026-07-25 (`f484607`, gated OFF via `REFCONF_USE_G_MATRIX=1`) |

## Answer to "is the coupled re-tune tractable?" — 2026-07-25 update

Before the audit: "the port has 50 divergences, all coupled, intractable."

After the audit: TWO major coupled clusters (contact-model + executor over-drive) with NAMED roots and PREDICTED cascade of simplifications, plus a small set of INDEPENDENT constants and INERT opt-ins. Not a hairball — a graph with structure.

- **Contact-model cluster** — **RESOLVED 2026-07-25**. Anitescu flip landed; the predicted ~10-item cascade materialized as REFERENCE-MATCH / OBSOLETED / INERT. Runtime evidence on p62 baseline: `mono=True iters=3/3 primal 7.59→1.21`, `n_lambda=16` at `n_c=4`, `_box_drag_c=0.0`.
- **Executor over-drive cluster** — STILL OPEN. 4 gain values compose. Flipping to reference values (W_track=1, Kp_cart=200, plus adding rotation-hold, plus adjusting reposition speed/height) requires COORDINATED change to keep IK→c3 handoff working. The Phase-1 OSC swap attempted in the reproduce-dairlib arc was recert-falsified 2026-07-14; no unblock recipe currently on the table.
- **port-todo #7** — NEW BLOCKER surfaced 2026-07-23. Full reference `q_vector` Q construction blocked on coupled `rho=1.0 + _use_g_matrix=True + use_reference_q_vector=True` triple flip. G-matrix prong-1 infrastructure landed 2026-07-25 (`f484607`, gated OFF). See `docs/superpowers/investigations/2026-07-23-item7-deep-investigation.md` for the 4-arc unblock recipe.
- **Cluster separability** unchanged: executor gains and item #7 do not depend on each other; they can be attempted in either order.

## Total map size

- **5 subsystems, 75 entries total**:
  - (1) Executor: 16 entries (1.a-1.p, plus 2.k retro-correction)
  - (2) Reposition: 14 entries (2.a-2.n)
  - (3) Admission: 17 entries (3.a-3.q)
  - (4) Planner/ADMM/mode-switch: 18 entries (4.a-4.r)
  - (5) Sim/URDF: 13 entries (5.a-5.m)
- All UNKNOWNs resolved (0 remaining). c3 lib clone completed 3.b, 3.q, and enabled 4.k line-by-line reference-source verification.
- 4 stale prior-memory findings corrected in-map:
  - executor 1.p — orientation NOT inert (2.k identity-hold at 8000 authority)
  - reposition 2.k — new load-bearing rotation cost surfaced
  - planner 4.h — cross-tick warm-start is CONFORMANT (both OFF), not divergent
  - planner 4.k — port projection MATCHES reference exactly (not a divergence)
- 6 diagnostic commits (all env-gated + labeled + default-OFF byte-identical):
  - `ce29c9f`, `bde8d64` — executor
  - `64ffdee` — reposition
  - `98f5e94` — admission
  - `67232d7` — planner
  - `043f378` — sim/URDF

## Next moves (2026-07-25 refresh)

Contact-model cluster is closed. Runtime baseline on `main @ f484607`: 4/5 tight_goal PASS across p58-p62. Remaining candidate work, in decreasing evidence-of-tractability order:

1. **port-todo #7 — Arc 1 (ρ sweep)** — read-only single-flag experiment (`rho ∈ {1, 3, 10, 30, 100}` on baseline config). Purely diagnostic; determines whether the coupled Q/G/ρ recipe is ever tractable. Detailed protocol in `docs/superpowers/investigations/2026-07-23-item7-deep-investigation.md:223-235`.

2. **Consolidation pass** — SAFE-INDEPENDENT + INERT subset:
   - Remove kIK reposition subsystem (2.e/2.f/2.g — port-only, no reference analog)
   - Delete inert opt-in flags (3.k, 3.m, LCS_NORMAL_* patches under 3.n now doubly inert under Anitescu, 4.r Stage-5 env-defaults if not exercised)
   - Default-to-reference on INDEPENDENT constants (5.c pusher radius per-task, 5.d pusher μ per-task, 5.h ground μ) — each with box tripwire

3. **Executor over-drive cluster** — no fresh recipe. Would need a new attempt distinct from the recert-falsified Phase-1 OSC swap. HOLD pending direction.

Two-tier discipline + 2.k caution ("inert" verdicts must verify each mechanism individually) remain in force for any future edit to this map.
