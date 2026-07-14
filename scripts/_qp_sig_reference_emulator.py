"""Reference-OSC emulator: same-input signature diff at executor boundary.

Consumes a QP-signature dump written by
`OperationalSpaceController._write_qp_sig_dump` (env-gated
`PUSHA_QP_SIG_DUMP=1`) and rebuilds the QP the reference (dairlib
`sampling_based_c3 @ push_anything_dev 257e3ed`) would solve for the
SAME inputs — same M, Cv, gravity, J_v, Jdot_v_v, p_err, v_err, q_arm,
v_arm — but with reference weights + gravity-comp-inside-bias per
`inverse_dynamics_qp.cc:213-225 + operational_space_control.cc:454-461,
469-484, 292-296` and `examples/sampling_c3/shared_parameters/osc_params.yaml`.

Reports the plant-side τ diff:
  τ_plant_port = u_opt + tau_g[:n_arm]      (main.py:732)
  τ_plant_ref  = u_ref                       (gravity comp inside QP)

Usage:
  python scripts/_qp_sig_reference_emulator.py audit_output/exec_qp_sig/dump_call60.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from control.osc.qp_builder import ...` when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pydrake.solvers import MathematicalProgram, OsqpSolver


# ---------------------------------------------------------------- reference gains
# All values sourced from
# dairlib_sampling_c3/examples/sampling_c3/shared_parameters/osc_params.yaml
# @ push_anything_dev 257e3ed. Cited inline.

REF_W_EE          = np.eye(3)                             # EndEffectorW  (line 47-50)
REF_KP_EE         = 200.0 * np.eye(3)                     # EndEffectorKp (line 51-54)
REF_KD_EE         = 20.0  * np.eye(3)                     # EndEffectorKd (line 55-58)
REF_W_EE_LAMBDA   = np.eye(3)                             # LambdaEndEffectorW (line 74-77)
REF_W_ELBOW       = 1.0                                   # w_elbow  (line 42)
REF_ELBOW_KP      = 200.0                                 # elbow_kp (line 43)
REF_ELBOW_KD      = 10.0                                  # elbow_kd (line 44)
REF_JOINT2_TGT    = 1.1                                   # AddConstTrackingData(...) franka_osc_controller.cc:161
REF_JOINT2_IDX    = 1                                     # franka_osc_controller.cc:164 "panda_joint2"
REF_W_ACCEL_SCAL  = 1.0e-5                                # w_accel (line 5)
REF_W_ACCEL_DIAG  = np.array([0.01]*7)                    # W_accel (line 12), arm-only 7 entries
REF_W_INPUT       = 0.0                                   # w_input (line 3), w_input_reg (line 4)
REF_W_SOFT        = 0.0                                   # w_soft_constraint (line 6)
REF_GRAVITY_COMP  = True                                  # sampling_c3 example enables EnableGravityCompensation()
# Reference OSC has track_end_effector_orientation=false → rotation tracking data
# is NOT added, so no rotation cost fires. Port has W_rot=10 with rot data — a
# port-only cost. Setting REF_W_ROT=0 makes the reference emulator match.
REF_W_ROT         = 0.0                                   # per track_end_effector_orientation (line 37) = false
REF_KP_ROT        = np.zeros(3)
REF_KD_ROT        = np.zeros(3)


def build_and_solve_qp_generic(
    inp: dict,
    *,
    # QP structural knobs — flip to isolate divergence contributions.
    gravity_comp_in_bias: bool,          # 1.d: True=inside (ref), False=outside (port)
    include_full_arm_posture: bool,      # port-only cost (W_posture, Kp_null, Kd_null)
    kp_ee_scale: np.ndarray,             # (3,3) — Kp_ee matrix
    kd_ee_scale: np.ndarray,             # (3,3) — Kd_ee matrix
    w_ee_matrix: np.ndarray,             # (3,3) — position tracking W (Hessian factor)
    w_force_scalar: float,               # λ_e tracking weight scalar (W_ee_lambda diagonal)
    w_torque_scalar: float,              # u regularization weight
    w_acc_scalar: float,                 # dv regularization weight (uniform diagonal)
    include_joint2: bool,                # W_joint2 · (dv[j2] − ddy_j2)²
    kp_j2: float, kd_j2: float, w_j2: float, j2_target: float, j2_idx: int,
    include_lam_reg_when_no_target: bool = True,  # ref always adds λ tracking; port only if present
    # Rotation tracking (port-only in current YAML): W_rot·‖J_w·dv + Jdot_w_v − a_rot‖²,
    # a_rot = Kp_rot·w_err + Kd_rot·(-w_ee_now). Skipped if w_rot_scalar == 0.
    w_rot_scalar: float = 0.0,
    kp_rot: np.ndarray = None,
    kd_rot: np.ndarray = None,
    J_w: np.ndarray = None,
    Jdot_w_v: np.ndarray = None,
    w_err: np.ndarray = None,
    w_ee_now: np.ndarray = None,
    label: str = "ref",
) -> dict:
    """Solve a generic OSC QP with knobs matching either port or reference."""
    n_arm = int(inp["n_arm"])
    n_v   = int(inp["n_v"])
    n_u   = n_arm

    M        = inp["M"]
    Cv       = inp["Cv"]
    gravity  = inp["gravity"]
    B        = inp["B"]
    J_v      = inp["J_v"]
    JdotV    = inp["Jdot_v_v"]
    p_err    = inp["p_err"]
    v_err    = inp["v_err"]
    q_arm    = inp["q_arm"]
    v_arm    = inp["v_arm"]
    q_arm_err= inp["q_arm_err"]
    v_arm_err= inp["v_arm_err"]
    tau_max  = inp["tau_max"]
    a_ff     = inp["a_ee_desired"]
    lam_des  = inp["lambda_des"]
    lam_des_present = bool(inp["lambda_des_present"])
    F_ff_for_qp = inp["F_ff_for_qp"]
    use_force_tracking = bool(inp["use_force_tracking"])
    Kp_null  = inp["Kp_null"]
    Kd_null  = inp["Kd_null"]
    W_posture= float(inp["W_posture"])

    prog = MathematicalProgram()
    dv     = prog.NewContinuousVariables(n_v, "dv")
    u      = prog.NewContinuousVariables(n_u, "u")
    lam_e  = prog.NewContinuousVariables(3,   "lambda_e")

    # ----- Dynamics equality
    bias_eff = Cv - gravity if gravity_comp_in_bias else Cv
    # RHS: reference has -bias + 0; port has -bias + F_ff_for_qp (when force-tracking
    # is ON, F_ff_for_qp=0 so equivalent).
    A_eq = np.hstack([M, -B, -J_v.T])
    b_eq = F_ff_for_qp - bias_eff
    prog.AddLinearEqualityConstraint(A_eq, b_eq,
                                     np.concatenate([dv, u, lam_e]))

    # ----- Input bounds
    prog.AddBoundingBoxConstraint(-tau_max, tau_max, u)

    # ----- Position tracking cost
    ddy_cmd = a_ff + kp_ee_scale @ p_err + kd_ee_scale @ v_err
    Q_pos = 2.0 * (J_v.T @ w_ee_matrix @ J_v)
    Q_pos = 0.5 * (Q_pos + Q_pos.T) + 1e-12 * np.eye(n_v)
    b_pos = 2.0 * (J_v.T @ w_ee_matrix @ (JdotV - ddy_cmd))
    prog.AddQuadraticCost(Q_pos, b_pos, dv, is_convex=True)

    # ----- Joint-2 tracking cost (unit selector)
    if include_joint2 and w_j2 > 0.0:
        e_j2 = np.zeros(n_v)
        e_j2[j2_idx] = 1.0
        q_j2v = float(q_arm[j2_idx])
        v_j2v = float(v_arm[j2_idx])
        ddy_j2 = kp_j2 * (j2_target - q_j2v) + kd_j2 * (-v_j2v)
        Q_j2 = 2.0 * w_j2 * np.outer(e_j2, e_j2)
        b_j2 = 2.0 * w_j2 * (-ddy_j2) * e_j2
        prog.AddQuadraticCost(Q_j2, b_j2, dv, is_convex=True)

    # ----- Full-arm posture cost (port-only)
    # port: qp_builder.py:229 — cost W_posture · ‖dv[:n_arm] − a_posture‖²
    # where a_posture = Kp_null·q_arm_err + Kd_null·v_arm_err.
    if include_full_arm_posture and W_posture > 0.0:
        a_posture = Kp_null * q_arm_err + Kd_null * v_arm_err
        # dv[:n_arm] is a selector; build the cost as
        #   Q_p = 2·W_posture · [I_{n_arm} 0; 0 0], b_p = -2·W_posture·[a_posture; 0]
        Q_p = np.zeros((n_v, n_v))
        Q_p[:n_arm, :n_arm] = 2.0 * W_posture * np.eye(n_arm)
        b_p = np.zeros(n_v)
        b_p[:n_arm] = -2.0 * W_posture * a_posture
        prog.AddQuadraticCost(Q_p, b_p, dv, is_convex=True)

    # ----- External force tracking cost
    # Port qp_builder.py:212-220 — added unconditionally when use_force_tracking
    # AND W_force>0 (uses zero target when lambda_des is None). Reference always
    # iterates force_tracking_data_vec_ per tick. `include_lam_reg_when_no_target`
    # is kept for the reference path where we want cost unconditionally.
    _fire_lam = (
        (use_force_tracking and w_force_scalar > 0.0)
        or lam_des_present
        or include_lam_reg_when_no_target
    )
    if _fire_lam and w_force_scalar > 0.0:
        W_lam = w_force_scalar * np.eye(3)
        Q_lam = 2.0 * W_lam
        b_lam = -2.0 * (W_lam @ lam_des) if lam_des_present else np.zeros(3)
        prog.AddQuadraticCost(Q_lam, b_lam, lam_e, is_convex=True)

    # ----- Acceleration reg — port passes 2·W_acc·I (qp_builder.py:203).
    if w_acc_scalar > 0.0:
        Q_acc = 2.0 * w_acc_scalar * np.eye(n_v)
        prog.AddQuadraticCost(Q_acc, np.zeros(n_v), dv, is_convex=True)

    # ----- Input reg — port passes 2·W_torque·I (qp_builder.py:198).
    if w_torque_scalar > 0.0:
        Q_u = 2.0 * w_torque_scalar * np.eye(n_u)
        prog.AddQuadraticCost(Q_u, np.zeros(n_u), u, is_convex=True)

    # ----- Rotation tracking (port-only in current setup, per YAML W_rot=10)
    # Port qp_builder.py:242-258. Same structure as position tracking but with
    # angular Jacobian, orientation-error small-angle vector, and W_rot·Kp_rot
    # authority.
    if (w_rot_scalar > 0.0
            and J_w is not None and Jdot_w_v is not None
            and w_err is not None and w_ee_now is not None
            and kp_rot is not None and kd_rot is not None):
        a_rot = kp_rot * w_err + kd_rot * (-w_ee_now)
        Q_rot = 2.0 * w_rot_scalar * (J_w.T @ J_w)
        Q_rot = 0.5 * (Q_rot + Q_rot.T) + 1e-12 * np.eye(n_v)
        b_rot = 2.0 * w_rot_scalar * (J_w.T @ (Jdot_w_v - a_rot))
        prog.AddQuadraticCost(Q_rot, b_rot, dv, is_convex=True)

    solver = OsqpSolver()
    result = solver.Solve(prog)
    return dict(
        u_sol=result.GetSolution(u),
        vdot_sol=result.GetSolution(dv),
        lam_e_sol=result.GetSolution(lam_e),
        qp_success=result.is_success(),
        qp_result=str(result.get_solution_result()),
        label=label,
    )


def build_and_solve_reference_qp(inp: dict) -> dict:
    """Rebuild + solve the reference-formula OSC QP for the dumped input.

    Formulation source (line refs in dairlib_sampling_c3 @ push_anything_dev 257e3ed):
      - Variables & dynamics eq: inverse_dynamics_qp.cc:78-83, 213-225
      - Position tracking cost: operational_space_control.cc:454-461 + osc_tracking_data.cc:113-116
      - External force tracking cost: operational_space_control.cc:469-484
      - Joint-2 tracking cost: same pattern as position tracking with J = e_{j2}
      - Acceleration reg cost: operational_space_control.cc:292-296
      - Input cost / smoothing cost: operational_space_control.cc:287-290, 297-300 (both 0)
      - Input bounds: inverse_dynamics_qp.cc:117-126
    """
    n_arm = int(inp["n_arm"])
    n_v   = int(inp["n_v"])
    n_u   = n_arm                                          # arm-only actuation

    M        = inp["M"]                                     # (n_v, n_v)
    Cv       = inp["Cv"]                                    # (n_v,)
    gravity  = inp["gravity"]                               # (n_v,)  = plant.CalcGravityGeneralizedForces
    B        = inp["B"]                                     # (n_v, n_u)
    J_v      = inp["J_v"]                                   # (3, n_v)
    JdotV    = inp["Jdot_v_v"]                              # (3,)
    p_err    = inp["p_err"]                                 # (3,)
    v_err    = inp["v_err"]                                 # (3,)
    q_arm    = inp["q_arm"]                                 # (n_arm,)
    v_arm    = inp["v_arm"]                                 # (n_arm,)
    tau_max  = inp["tau_max"]                               # (n_u,)
    a_ff     = inp["a_ee_desired"]                          # (3,) = yddot_des
    v_ff     = inp["v_ee_desired"]                          # (3,) = ydot_des (already folded into v_err by port; kept for structure)
    v_ff_present = bool(inp["v_ee_desired_present"])
    lam_des  = inp["lambda_des"]                            # (3,)
    lam_des_present = bool(inp["lambda_des_present"])

    prog = MathematicalProgram()
    dv     = prog.NewContinuousVariables(n_v, "dv")
    u      = prog.NewContinuousVariables(n_u, "u")
    lam_e  = prog.NewContinuousVariables(3,   "lambda_e")

    # ----- Dynamics equality: M·dv − B·u − Je^T·λ_e = -(Cv − grav)
    # inverse_dynamics_qp.cc:213-225 (n_h=n_c=0 for push_anything: 1.c/1.g KNOWN-INERT).
    bias_ref = Cv - gravity if REF_GRAVITY_COMP else Cv
    A_eq = np.hstack([M, -B, -J_v.T])                       # (n_v, n_v + n_u + 3)
    b_eq = -bias_ref                                        # (n_v,)
    prog.AddLinearEqualityConstraint(A_eq, b_eq,
                                     np.concatenate([dv, u, lam_e]))

    # ----- Input bounds: -tau_max ≤ u ≤ +tau_max
    # inverse_dynamics_qp.cc:117-126 (with_input_constraints_=true).
    prog.AddBoundingBoxConstraint(-tau_max, tau_max, u)

    # ----- Position tracking cost: 2·J^T·W·J on dv, 2·J^T·W·(JdotV − ddy_cmd)
    # operational_space_control.cc:454-461. Drake AddQuadraticCost convention:
    # cost = 0.5·x^T·Q·x + b^T·x, so passing Q=2·J^T·W·J yields ‖J·dv‖²_W.
    # ddy_cmd = ydd_des + Kp·(y_des − y_now) + Kd·(ydot_des − ydot_now)
    #         = a_ff + Kp·p_err + Kd·v_err
    ddy_cmd = a_ff + REF_KP_EE @ p_err + REF_KD_EE @ v_err
    Q_pos = 2.0 * (J_v.T @ REF_W_EE @ J_v)
    Q_pos = 0.5 * (Q_pos + Q_pos.T) + 1e-12 * np.eye(n_v)   # PSD snap (numerical)
    b_pos = 2.0 * (J_v.T @ REF_W_EE @ (JdotV - ddy_cmd))
    prog.AddQuadraticCost(Q_pos, b_pos, dv, is_convex=True)

    # ----- Joint-2 tracking cost: J = e_{j2} unit selector, W = w_elbow
    # AddConstTrackingData(JointSpaceTrackingData(...)) at franka_osc_controller.cc:160.
    e_j2 = np.zeros(n_v)
    e_j2[REF_JOINT2_IDX] = 1.0
    q_j2 = float(q_arm[REF_JOINT2_IDX])
    v_j2 = float(v_arm[REF_JOINT2_IDX])
    ddy_j2 = REF_ELBOW_KP * (REF_JOINT2_TGT - q_j2) + REF_ELBOW_KD * (-v_j2)
    # JdotV for a unit selector J = e_j2^T is 0 (J does not depend on q).
    Q_j2 = 2.0 * REF_W_ELBOW * np.outer(e_j2, e_j2)
    b_j2 = 2.0 * REF_W_ELBOW * (-ddy_j2) * e_j2
    prog.AddQuadraticCost(Q_j2, b_j2, dv)

    # ----- External force tracking cost: 2·W·λ_e^2 − 2·W·λ_des·λ_e
    # operational_space_control.cc:469-484. Only added if λ_des is present.
    if lam_des_present:
        Q_lam = 2.0 * REF_W_EE_LAMBDA
        b_lam = -2.0 * (REF_W_EE_LAMBDA @ lam_des)
        prog.AddQuadraticCost(Q_lam, b_lam, lam_e)
    else:
        # Reference always adds the cost (with lambda_des from trajectory); if the
        # port didn't have one, we mirror by adding a zero-target regularizer.
        Q_lam = 2.0 * REF_W_EE_LAMBDA
        b_lam = np.zeros(3)
        prog.AddQuadraticCost(Q_lam, b_lam, lam_e)

    # ----- Acceleration regularization: W_joint_accel · dv^2 (matrix Q, no b)
    # operational_space_control.cc:292-296. Drake convention: 0.5·dv^T·Q·dv, so
    # passing Q = W_joint_accel yields 0.5·‖dv‖²_{W_joint_accel}. Reference plant
    # is arm-only (n_v=7); the port plant is n_v=13 (arm + manipuland). Extend
    # reference reg diagonally across all n_v so the Hessian remains PD on the
    # manipuland slice (matches the port's own `W_acc · I_{n_v}` pattern).
    Q_acc = REF_W_ACCEL_SCAL * REF_W_ACCEL_DIAG[0] * np.eye(n_v)
    b_acc = np.zeros(n_v)
    prog.AddQuadraticCost(Q_acc, b_acc, dv)

    # ----- Input cost and smoothing: both zero-weight (w_input=0, w_input_reg=0).
    # Skipped — no cost term added.

    # ----- Soft constraint slack: w_soft_constraint=0. n_c=0 anyway.
    # Skipped — no ε variable, no cost.

    # ----- Solve.
    solver = OsqpSolver()
    result = solver.Solve(prog)
    dv_sol   = result.GetSolution(dv)
    u_sol    = result.GetSolution(u)
    lam_sol  = result.GetSolution(lam_e)
    success  = result.is_success()
    solver_result = str(result.get_solution_result())

    return dict(
        u_ref=u_sol, vdot_ref=dv_sol, lam_e_ref=lam_sol,
        qp_success_ref=success, qp_result_ref=solver_result,
        A_eq=A_eq, b_eq=b_eq,
        # For downstream diagnostics
        ddy_cmd_pos=ddy_cmd, ddy_cmd_j2=ddy_j2, bias_ref=bias_ref,
    )


def _fmt_vec(v, n=4):
    return "[" + ", ".join(f"{x:+.{n}f}" for x in np.asarray(v).ravel()) + "]"


def report(dump_path: str) -> None:
    d = dict(np.load(dump_path, allow_pickle=True))
    # Reconstruct dicts from the flat npz keys.
    inp = {k[3:]: v for k, v in d.items() if k.startswith("in_")}
    out = {k[4:]: v for k, v in d.items() if k.startswith("out_")}

    # Cast scalars stored as 0-d arrays.
    for k, v in list(inp.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            inp[k] = v.item()
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            out[k] = v.item()

    n_arm = int(inp["n_arm"])
    n_v   = int(inp["n_v"])

    print("=" * 78)
    print(f"QP signature diff — dump {dump_path}")
    print("=" * 78)
    print(f"  compute_torque call idx = {inp['n_calls_idx']}")
    print(f"  mode                    = {inp['mode']}")
    print(f"  use_force_tracking      = {inp['use_force_tracking']}")
    print(f"  c3_ref_gains_active     = {inp['c3_ref_gains_active']}")
    print(f"  n_arm={n_arm}  n_v={n_v}")
    print()

    print("--- Port-side (dumped from OperationalSpaceController) ---")
    print(f"  Kp_cart = {_fmt_vec(inp['Kp_cart'])}")
    print(f"  Kd_cart = {_fmt_vec(inp['Kd_cart'])}")
    print(f"  W_track={inp['W_track']}  W_posture={inp['W_posture']}  "
          f"W_torque={inp['W_torque']}  W_acc={inp['W_acc']}  W_force={inp['W_force']}")
    print(f"  W_joint2={inp['W_joint2']}  Kp_j2={inp['Kp_joint2']}  Kd_j2={inp['Kd_joint2']}  "
          f"target={inp['joint2_target_rad']}  idx={inp['joint2_idx']}")
    print(f"  p_err   = {_fmt_vec(inp['p_err'])}   |p_err|={np.linalg.norm(inp['p_err']):.4f}m")
    print(f"  v_err   = {_fmt_vec(inp['v_err'])}")
    print(f"  lambda_des = {_fmt_vec(inp['lambda_des'])} (present={inp['lambda_des_present']})")
    print()
    print(f"  u_opt (port)      = {_fmt_vec(out['u_opt'])}   |u|={np.linalg.norm(out['u_opt']):.3f} Nm")
    print(f"  tau_g[:n_arm]     = {_fmt_vec(-inp['gravity'][:n_arm])}")
    tau_plant_port = out['u_opt'] + (-inp['gravity'][:n_arm])
    print(f"  τ_plant (port)    = {_fmt_vec(tau_plant_port)}   |τ|={np.linalg.norm(tau_plant_port):.3f} Nm")
    print(f"  τ_plant vs cap    = {['OVER' if abs(t)>c else 'ok' for t,c in zip(tau_plant_port, inp['tau_max'])]}")
    print(f"  vdot_opt[:n_arm]  = {_fmt_vec(out['vdot_opt'][:n_arm])}")
    print(f"  lam_ext_opt       = {_fmt_vec(out['lam_ext_opt'])}")
    print()

    # --- Sanity 1: call the port's own build_and_solve_qp directly.
    # If this doesn't match u_opt, the dumped inputs are missing some field
    # the port's QP uses. If it matches, the dump is faithful and any
    # emulator disagreement is my implementation, not missing state.
    from control.osc.qp_builder import build_and_solve_qp, OscGains, OscLimits
    _kp_rot = inp.get("Kp_rot", np.zeros(3))
    _kd_rot = inp.get("Kd_rot", np.zeros(3))
    _g = OscGains(
        Kp_cart=inp["Kp_cart"], Kd_cart=inp["Kd_cart"],
        Kp_null=inp["Kp_null"], Kd_null=inp["Kd_null"],
        W_track=float(inp["W_track"]), W_posture=float(inp["W_posture"]),
        W_torque=float(inp["W_torque"]), W_acc=float(inp["W_acc"]),
        W_force=float(inp["W_force"]),
        Kp_joint2=float(inp["Kp_joint2"]),
        Kd_joint2=float(inp["Kd_joint2"]),
        W_joint2=float(inp["W_joint2"]),
        joint2_target_rad=float(inp["joint2_target_rad"]),
        joint2_idx=int(inp["joint2_idx"]),
        Kp_rot=(_kp_rot if np.any(_kp_rot != 0) else None),
        Kd_rot=(_kd_rot if np.any(_kd_rot != 0) else None),
        W_rot=float(inp.get("W_rot", 0.0)),
    )
    _l = OscLimits(tau_max=inp["tau_max"])
    _rot_active = bool(inp.get("rot_active", False))
    _u_port_replay, _vdot_port_replay, _succ, _res, _lam_port_replay = build_and_solve_qp(
        M=inp["M"], bias=inp["bias"], B=inp["B"], n_arm=int(inp["n_arm"]),
        J_v=inp["J_v"], Jdot_v_v=inp["Jdot_v_v"],
        p_err=inp["p_err"], v_err=inp["v_err"],
        q_arm_err=inp["q_arm_err"], v_arm_err=inp["v_arm_err"],
        gains=_g, limits=_l,
        F_ff_external=inp["F_ff_for_qp"],
        solver=OsqpSolver(),
        use_force_tracking=bool(inp["use_force_tracking"]),
        lambda_des=(inp["lambda_des"] if inp["lambda_des_present"] else None),
        a_ff=(inp["a_ee_desired"] if inp["a_ee_desired_present"] else None),
        q_arm=inp["q_arm"], v_arm=inp["v_arm"],
        J_w=(inp.get("J_w") if _rot_active else None),
        Jdot_w_v=(inp.get("Jdot_w_v") if _rot_active else None),
        w_err=(inp.get("w_err") if _rot_active else None),
        w_ee_now=(inp.get("w_ee_now") if _rot_active else None),
    )
    print("--- Sanity 1: port's own build_and_solve_qp on the dumped inputs ---")
    print(f"  qp_success={_succ}  ({_res})")
    print(f"  u_replay          = {_fmt_vec(_u_port_replay)}")
    print(f"  u_port (dumped)   = {_fmt_vec(out['u_opt'])}")
    _dr = _u_port_replay - out['u_opt']
    print(f"  Δu (replay − port)= {_fmt_vec(_dr)}   ‖Δ‖={np.linalg.norm(_dr):.6f}")
    print(f"  (zero Δ ⇒ dump captures everything build_and_solve_qp needs)")
    print()

    # --- Self-check: rebuild PORT QP using the generic solver with port knobs.
    # If the emulator is faithful, this should reproduce u_opt exactly. If not,
    # the difference tells us the emulator is missing structure — draw no
    # port-vs-reference conclusions until this reproduces.
    print("--- Emulator self-check (port gains + port structure via generic solver) ---")
    Kp_ee_port = np.diag(inp["Kp_cart"])
    Kd_ee_port = np.diag(inp["Kd_cart"])
    W_ee_port  = float(inp["W_track"]) * np.eye(3)
    self_check = build_and_solve_qp_generic(
        inp,
        gravity_comp_in_bias=False,          # port: bias = Cv
        include_full_arm_posture=True,       # port has this
        kp_ee_scale=Kp_ee_port,
        kd_ee_scale=Kd_ee_port,
        w_ee_matrix=W_ee_port,
        w_force_scalar=float(inp["W_force"]),
        w_torque_scalar=float(inp["W_torque"]),
        w_acc_scalar=float(inp["W_acc"]),
        include_joint2=True,
        kp_j2=float(inp["Kp_joint2"]),
        kd_j2=float(inp["Kd_joint2"]),
        w_j2=float(inp["W_joint2"]),
        j2_target=float(inp["joint2_target_rad"]),
        j2_idx=int(inp["joint2_idx"]),
        include_lam_reg_when_no_target=False,  # port only adds if λ_des present
        # Rotation (port-side W_rot=10 by default; forwarded from the dump)
        w_rot_scalar=float(inp.get("W_rot", 0.0)),
        kp_rot=inp.get("Kp_rot", None),
        kd_rot=inp.get("Kd_rot", None),
        J_w=inp.get("J_w", None),
        Jdot_w_v=inp.get("Jdot_w_v", None),
        w_err=inp.get("w_err", None),
        w_ee_now=inp.get("w_ee_now", None),
        label="self-check port-eq",
    )
    print(f"  qp_success        = {self_check['qp_success']}  ({self_check['qp_result']})")
    print(f"  u_selfcheck       = {_fmt_vec(self_check['u_sol'])}")
    print(f"  u_port (dumped)   = {_fmt_vec(out['u_opt'])}")
    dsc = self_check['u_sol'] - out['u_opt']
    print(f"  Δu (selfcheck − port) = {_fmt_vec(dsc)}   ‖Δ‖={np.linalg.norm(dsc):.4f}")
    print(f"  (small Δ ⇒ emulator faithful; large Δ ⇒ emulator missing structure)")
    print()

    # Rebuild reference QP on the same inputs.
    print("--- Reference-formula emulator (dairlib push_anything_dev 257e3ed) ---")
    print(f"  W_ee=I_3  Kp_ee={REF_KP_EE[0,0]}  Kd_ee={REF_KD_EE[0,0]}")
    print(f"  W_ee_lambda=I_3  W_elbow={REF_W_ELBOW}  Kp_j2={REF_ELBOW_KP}  "
          f"Kd_j2={REF_ELBOW_KD}  target={REF_JOINT2_TGT}")
    print(f"  W_joint_accel diag = {REF_W_ACCEL_SCAL * REF_W_ACCEL_DIAG[0]:.2e} (arm-slice)")
    print(f"  w_input=0  w_input_smoothing=0  w_soft=0")
    print(f"  bias = Cv − grav (gravity comp INSIDE QP, matches reference)")

    ref = build_and_solve_reference_qp(inp)
    print(f"  qp_success (ref)  = {ref['qp_success_ref']}  ({ref['qp_result_ref']})")
    print(f"  u_ref             = {_fmt_vec(ref['u_ref'])}   |u_ref|={np.linalg.norm(ref['u_ref']):.3f} Nm")
    tau_plant_ref = ref['u_ref']    # gravity already inside QP → u_ref IS the plant torque
    print(f"  τ_plant (ref)     = {_fmt_vec(tau_plant_ref)}   |τ|={np.linalg.norm(tau_plant_ref):.3f} Nm")
    print(f"  vdot_ref[:n_arm]  = {_fmt_vec(ref['vdot_ref'][:n_arm])}")
    print(f"  lam_e_ref         = {_fmt_vec(ref['lam_e_ref'])}")
    print()

    # ----- Diff.
    print("--- Δ (port − ref), plant-side τ ---")
    d_tau = tau_plant_port - tau_plant_ref
    print(f"  Δτ_plant          = {_fmt_vec(d_tau)}")
    print(f"  ‖Δτ_plant‖        = {np.linalg.norm(d_tau):.3f} Nm  "
          f"(port |τ|={np.linalg.norm(tau_plant_port):.3f}, ref |τ|={np.linalg.norm(tau_plant_ref):.3f})")
    print(f"  worst joint       = j{int(np.argmax(np.abs(d_tau)))}  "
          f"Δ={d_tau[int(np.argmax(np.abs(d_tau)))]:+.3f} Nm")
    print()
    d_vdot = out['vdot_opt'][:n_arm] - ref['vdot_ref'][:n_arm]
    print(f"  Δv̇[:n_arm]        = {_fmt_vec(d_vdot)}")
    print(f"  ‖Δv̇[:n_arm]‖      = {np.linalg.norm(d_vdot):.4f}")
    d_lam = out['lam_ext_opt'] - ref['lam_e_ref']
    print(f"  Δλ_e              = {_fmt_vec(d_lam)}   ‖Δ‖={np.linalg.norm(d_lam):.4f}")
    print()

    # ----- Where the divergence comes from (structural readout).
    print("--- Divergence structural readout ---")
    print(f"  port bias    = Cv (task-only; main.py adds tau_g post-QP)     — 1.d")
    print(f"  ref  bias    = Cv − grav (gravity comp INSIDE QP)             — 1.d")
    print(f"  Δ bias[:n_arm] = {_fmt_vec(inp['bias'][:n_arm] - (inp['Cv'][:n_arm] - inp['gravity'][:n_arm]))}")
    print(f"  port W_rot           = {inp.get('W_rot', 0.0)}  Kp_rot={inp.get('Kp_rot', np.zeros(3)).tolist()}")
    print(f"  ref  W_rot           = 0.0  (track_end_effector_orientation=false in osc_params.yaml:37)")
    print(f"                        → PORT-ONLY rotation-hold cost with compound authority "
          f"{inp.get('W_rot', 0.0) * (inp.get('Kp_rot', np.zeros(3))[0] if len(inp.get('Kp_rot', np.zeros(3)))>0 else 0):.1f}")
    ref_W_Kp = REF_KP_EE[0,0] * 1.0                     # W_ee=1
    port_W_Kp = inp['Kp_cart'][0] * inp['W_track']
    print(f"  port W_track·Kp_cart = {port_W_Kp:.1f}   (1.e compound authority)")
    print(f"  ref  W_ee·Kp_ee      = {ref_W_Kp:.1f}")
    print(f"  ratio (port/ref)     = {port_W_Kp / ref_W_Kp:.2f}")
    ref_W_Kd = REF_KD_EE[0,0] * 1.0
    port_W_Kd = inp['Kd_cart'][0] * inp['W_track']
    print(f"  port W_track·Kd_cart = {port_W_Kd:.1f}")
    print(f"  ref  W_ee·Kd_ee      = {ref_W_Kd:.1f}")
    ref_W_Wlam = 1.0
    port_W_Wlam = inp['W_force']
    print(f"  port W_force         = {port_W_Wlam}  vs ref W_ee_lambda={ref_W_Wlam}")
    print(f"  port W_posture       = {inp['W_posture']}  (port-only; reference has no full-arm posture)")
    print(f"  port W_torque        = {inp['W_torque']}  vs ref w_input={REF_W_INPUT}")
    print(f"  port W_acc           = {inp['W_acc']}     vs ref effective={REF_W_ACCEL_SCAL*REF_W_ACCEL_DIAG[0]:.2e}")
    print()


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/dump_callN.npz>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 1
    report(str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
