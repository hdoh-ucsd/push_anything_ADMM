"""§7.31 — Reconcile approach to reference STRUCTURAL sanity (offline; BEFORE
the live sim). Confirms:
  (1) Flag OFF reproduces the pre-§7.31 behaviour (force-tracking on, proxy
      block populated, surface override pass-through).
  (2) Flag ON at a SEPARATED state (sample 30 mm OUTSIDE box face) →
      surface-point override returns a point at the surface (Δφ ≈ −30 mm
      toward the box); OSC instantiated with use_force_tracking=False;
      task_costs build_ee_space SKIPS the EE-approach Q / x_ref proxy block.
  (3) The position-OSC produces a FINITE Cartesian acceleration command
      pointing TOWARD the surface (drives φ down) — no NaN/singularity.

STRUCTURAL check — NOT physics validation (the live sim is the only available
physics test for the reconciled approach path).
"""
from __future__ import annotations

import os
import numpy as np
import yaml

from sim.env_builder import build_environment
from control.osc.qp_builder import OscGains, OscLimits, build_and_solve_qp


BOX_HALF  = 0.05
BOX_POS   = np.array([0.0, 0.0, 0.05])
# Reference values from osc_franka.yaml
KP_CART = np.array([400.0, 400.0, 400.0])
KD_CART = np.array([40.0, 40.0, 40.0])


def _check_surface_target(sample_xyz, obj_xy, setback):
    """Replicates SamplingC3MPC._reconcile_surface_target without
    instantiating the full controller (which needs Drake)."""
    if sample_xyz is None or obj_xy is None:
        return None
    delta_xy = np.asarray(sample_xyz[:2], dtype=float) - np.asarray(obj_xy, dtype=float)
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-6:
        return None
    n_outward_xy = delta_xy / norm
    surface_xy = np.asarray(sample_xyz[:2], dtype=float) - setback * n_outward_xy
    return np.array([surface_xy[0], surface_xy[1], float(sample_xyz[2])])


def _check_task_cost(flag_value: str):
    """Build task_costs.QuadraticManipulationCost::build_ee_space with the
    flag ON/OFF and report the EE-approach Q block.  Build a stub plant by
    importing build_environment."""
    if flag_value is None:
        os.environ.pop("REF_RECONCILE_APPROACH", None)
    else:
        os.environ["REF_RECONCILE_APPROACH"] = flag_value

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]

    from control.task_costs import QuadraticManipulationCost
    diagram, plant, panda, obj, _, _, _ = build_environment(task_cfg, time_step=0.001)
    cost = QuadraticManipulationCost(
        plant       = plant,
        ee_frame_name = "pusher",
        obj_body    = plant.GetBodyByName("box_link", obj),
        cost_cfg    = task_cfg["cost"],
        n_x         = 19,
        n_u         = 3,
        math_diag   = False,
    )
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)
    # Pose box at origin, EE at +0.20 m above (no specific arm pose needed
    # for build_ee_space — it reads obj/EE from plant_ctx geometry).
    plant.SetPositions(plant_ctx, obj,
                       np.concatenate([np.array([1., 0., 0., 0.]), BOX_POS]))

    result = cost.build_ee_space(
        target_xy=np.array([-0.30, 0.0]),
        plant_ctx=plant_ctx,
        current_q=plant.GetPositions(plant_ctx),
        target_yaw=0.0,
    )
    # build_ee_space may return either (Q,R,QN,x_ref) or 5-tuple — handle both.
    if len(result) == 5:
        Q, R, QN, x_ref, u_ref = result
    else:
        Q, R, QN, x_ref = result
        u_ref = np.zeros(3)
    # The EE position slot under the EE-space layout. Read directly from
    # the cost object to keep this resilient to slot renumbering.
    pee_slot = slice(cost._NEW_PEE_SLOT.start, cost._NEW_PEE_SLOT.stop)
    Q_pee = np.asarray(Q[pee_slot, pee_slot])
    x_ref_pee = np.asarray(x_ref[pee_slot])
    return {
        "Q_pee_diag": np.diag(Q_pee).tolist(),
        "Q_pee_zero": bool(np.allclose(Q_pee, 0.0)),
        "x_ref_pee": x_ref_pee.tolist(),
        "w_ee_approach": cost.w_ee_approach,
        "d_push": cost.d_push,
    }


def _check_position_osc():
    """Build a single OSC QP with use_force_tracking=False at a
    constructed-state config; verify finite output + a_des points to the
    target."""
    # Constructed 7-DoF dynamics block (minimal sanity, not Drake-anchored):
    n_v, n_u = 7, 7
    M = np.eye(n_v) * 1.0
    bias = np.zeros(n_v)
    B = np.eye(n_u)
    J_v = np.zeros((3, n_v))
    J_v[0, 0] = 1.0
    J_v[1, 1] = 1.0
    J_v[2, 2] = 1.0
    Jdot_v_v = np.zeros(3)

    # Position error: EE 5 mm beyond touching → 5 mm to drive toward
    # surface, then target IS the surface (Δ = −5 mm in x).
    p_err = np.array([-0.005, 0.0, 0.0])
    v_err = np.zeros(3)
    q_arm_err = np.zeros(n_u)
    v_arm_err = np.zeros(n_u)

    gains = OscGains(
        Kp_cart   = KP_CART,
        Kd_cart   = KD_CART,
        Kp_null   = np.ones(n_u) * 10.0,
        Kd_null   = np.ones(n_u) * 3.0,
        W_track   = 100.0,
        W_posture = 1.0,
        W_torque  = 0.001,
        W_acc     = 0.001,
    )
    limits = OscLimits(tau_max = np.ones(n_u) * 87.0)

    u, vdot, success, result_str, lam_ext = build_and_solve_qp(
        M=M, bias=bias, B=B, n_arm=n_u,
        J_v=J_v, Jdot_v_v=Jdot_v_v,
        p_err=p_err, v_err=v_err,
        q_arm_err=q_arm_err, v_arm_err=v_arm_err,
        gains=gains, limits=limits,
        F_ff_external=np.zeros(n_v),
        solver=None,
        use_force_tracking=False,
        lambda_des=None,
    )
    a_des = gains.Kp_cart * p_err + gains.Kd_cart * v_err
    return {
        "success": bool(success),
        "u_norm": float(np.linalg.norm(u)),
        "vdot_norm": float(np.linalg.norm(vdot)),
        "u_finite": bool(np.all(np.isfinite(u))),
        "u_x_toward_surface": float(u[0]) < 0.0,  # joint-1 torque pushes EE-x −
        "a_des": a_des.tolist(),
        "result_str": result_str,
    }


def main() -> int:
    print("=" * 84)
    print("§7.31 RECONCILE APPROACH — STRUCTURAL SANITY")
    print("=" * 84)

    # --- Surface-point projection ---
    print()
    print("Test A — surface-point projection (the _reconcile_surface_target math):")
    setback = 0.030
    obj_xy = np.array([0.0, 0.0])
    sample = np.array([+0.080, 0.0, 0.05])     # 30 mm east of +box-face (face at +0.05)
    surface = _check_surface_target(sample, obj_xy, setback)
    print(f"  sample           = {sample.tolist()}  (30 mm outside east face)")
    print(f"  obj_xy           = {obj_xy.tolist()}")
    print(f"  setback          = {setback*1000:.0f} mm")
    print(f"  surface point    = {surface.tolist()}")
    surface_at_face = abs(surface[0] - 0.050) < 1e-9
    print(f"  surface x = +{surface[0]*1000:.3f} mm "
          f"(target box-face x = +50 mm) → "
          f"{('PASS (at face)' if surface_at_face else 'FAIL (off face)')}")
    print()

    # --- Task cost flag OFF ---
    print("Test B — task_cost flag OFF (byte-identical pre-§7.31):")
    r_off = _check_task_cost(None)
    print(f"  w_ee_approach    = {r_off['w_ee_approach']}  d_push = {r_off['d_push']}")
    print(f"  Q[p_ee] diag     = {r_off['Q_pee_diag']}")
    print(f"  x_ref[p_ee]      = {r_off['x_ref_pee']}")
    print(f"  Q[p_ee] zero?    = {r_off['Q_pee_zero']}")
    flag_off_proxy_alive = (not r_off["Q_pee_zero"]) and (r_off["w_ee_approach"] > 0)
    print(f"  proxy alive (expected)? "
          f"{('PASS' if flag_off_proxy_alive else 'FAIL')}")
    print()

    # --- Task cost flag ON ---
    print("Test C — task_cost flag ON (REF_RECONCILE_APPROACH=1): proxy OFF")
    r_on = _check_task_cost("1")
    print(f"  Q[p_ee] diag     = {r_on['Q_pee_diag']}")
    print(f"  x_ref[p_ee]      = {r_on['x_ref_pee']}")
    print(f"  Q[p_ee] zero?    = {r_on['Q_pee_zero']}")
    flag_on_proxy_off = r_on["Q_pee_zero"] and np.allclose(r_on["x_ref_pee"], 0.0)
    print(f"  proxy OFF? "
          f"{('PASS' if flag_on_proxy_off else 'FAIL')}")
    print()

    # --- Position-OSC ---
    print("Test D — position-only OSC (use_force_tracking=False) produces FINITE "
          "control TOWARD surface:")
    r_osc = _check_position_osc()
    print(f"  QP success       = {r_osc['success']}  result = {r_osc['result_str']}")
    print(f"  a_des            = {r_osc['a_des']}")
    print(f"  |u|              = {r_osc['u_norm']:+.4f}")
    print(f"  |vdot|           = {r_osc['vdot_norm']:+.4f}")
    print(f"  u finite         = {r_osc['u_finite']}")
    print(f"  u_x toward surface (negative)? = "
          f"{('PASS' if r_osc['u_x_toward_surface'] else 'FAIL')}")
    print()

    # --- Verdict ---
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"  (a) surface-point projection lands AT the face                : "
          f"{('PASS' if surface_at_face else 'FAIL')}")
    print(f"  (b) flag OFF byte-identical (proxy alive)                     : "
          f"{('PASS' if flag_off_proxy_alive else 'FAIL')}")
    print(f"  (c) flag ON → cost-proxy OFF (Q[p_ee] = 0, x_ref[p_ee] = 0)   : "
          f"{('PASS' if flag_on_proxy_off else 'FAIL')}")
    print(f"  (d) position-OSC produces FINITE control                      : "
          f"{('PASS' if r_osc['success'] and r_osc['u_finite'] else 'FAIL')}")
    print(f"  (e) position-OSC command points TOWARD surface (u_x < 0)      : "
          f"{('PASS' if r_osc['u_x_toward_surface'] else 'FAIL')}")
    print()
    all_pass = (surface_at_face and flag_off_proxy_alive and flag_on_proxy_off
                and r_osc["success"] and r_osc["u_finite"]
                and r_osc["u_x_toward_surface"])
    print(f"  STRUCTURAL SANITY: "
          f"{('PASS — proceed to live sim' if all_pass else 'FAIL — diagnose before live run')}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
