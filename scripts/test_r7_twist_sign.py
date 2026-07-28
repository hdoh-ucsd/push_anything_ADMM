"""R^7 LCS twist-direction sign test (p113/p114 rot diagnosis).

The p114 qz probe showed the planner predicting CW box rotation
(qz_pred → −0.30, toward goal_yaw=−0.738) while the real push twists the
T CCW — a sign-inverted rotational coupling somewhere in the R^7 LCS.
This test splits the two suspect stages:

  Stage 1 (contact Jacobian): apply a unit Anitescu λ on the EE-BOX
      contact's 4 folded columns and read the predicted box ω_z from the
      velocity rows of D (= dt·M⁻¹·J_cᵀ). Compare against the analytic
      rigid-body torque sign τ_z = (r × F)_z at the witness point.
  Stage 2 (quaternion N-map): read the predicted Δqz from the position
      rows of D (= dt²·N_mat·M⁻¹·J_cᵀ) and compare against
      0.5·dt·Δω_z (exact for the identity quaternion).

Two probes with opposite lever arms make the discrimination unambiguous:
  probe A: EE on crossbar south face, EAST of CoM  → push north → CCW (+)
  probe B: EE on stem-bottom face,   WEST of CoM  → push north → CW  (−)

Run:  python scripts/test_r7_twist_sign.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import load_task                                  # noqa: E402
from sim.env_builder import build_environment                # noqa: E402
from control.lcs_formulator import LCSFormulator             # noqa: E402
from control.sampling_c3.ik import solve_ik_to_ee_pos        # noqa: E402

DT = 0.1


def main():
    task = load_task("push_t")
    out = build_environment(task, time_step=0.001)
    diagram, plant, panda_model = out[0], out[1], out[2]
    plant_ad, context_ad = out[5], out[6]

    dctx = diagram.CreateDefaultContext()
    ctx = plant.GetMyContextFromRoot(dctx)
    from sim.env_builder import EE_BODY_NAME
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)   # "pusher" tip sphere
    obj_body = plant.GetBodyByName("t_link")

    # μ exactly as main.py:539-551 derives it.
    mu_manip = float(task.get("friction", 0.3))
    mu_pusher = float(task.get("pusher_friction", mu_manip))
    mu_lcs = 2.0 * mu_manip * mu_pusher / (mu_manip + mu_pusher)
    formulator = LCSFormulator(
        plant, mu=mu_lcs, obj_body=obj_body,
        plant_ad=plant_ad, context_ad=context_ad,
        object_shape="tshape",
        mu_per_pair_type=task.get("mu_per_pair_type", None),
    )

    n_q, n_v, n_u = plant.num_positions(), plant.num_velocities(), 7
    ps = obj_body.floating_positions_start()
    vs = obj_body.floating_velocities_start_in_v()
    QZ_X = ps + 3                 # x-index of box quat qz
    WZ_X = n_q + vs + 2           # x-index of box ω_z
    WZ_V = vs + 2                 # v-index of box ω_z

    box_xy = np.array([0.5, 0.0])
    box_z = 0.02

    # (name, ee_target_world, witness_body_xy, expected τ_z sign)
    # Force on box from a south-side push is +y; τ_z = r_x·F_y.
    probes = [
        ("A east-lever crossbar-south",
         np.array([box_xy[0] + 0.03, box_xy[1] - 0.040, 0.019]),
         np.array([+0.03, -0.02]), +1),
        ("B west-lever stem-bottom",
         np.array([box_xy[0] - 0.05, box_xy[1] - 0.100, 0.019]),
         np.array([-0.05, -0.08]), -1),
    ]

    failures = 0
    for name, p_ee, r_body, tau_sign in probes:
        # Pose the box.
        q = plant.GetPositions(ctx).copy()
        q[ps:ps + 4] = [1.0, 0.0, 0.0, 0.0]
        q[ps + 4:ps + 7] = [box_xy[0], box_xy[1], box_z]
        plant.SetPositions(ctx, q)
        # IK the arm EE to the probe point.
        q_sol, ik_err, ik_iters = solve_ik_to_ee_pos(
            plant, ee_frame, p_target=p_ee, q_init=q.copy(),
            plant_ctx=ctx, n_arm_dofs=n_u)
        q_sol[ps:ps + 4] = [1.0, 0.0, 0.0, 0.0]
        q_sol[ps + 4:ps + 7] = [box_xy[0], box_xy[1], box_z]
        plant.SetPositions(ctx, q_sol)
        plant.SetVelocities(ctx, np.zeros(n_v))

        tau_g = plant.CalcGravityGeneralizedForces(ctx)
        u_lin = -np.asarray(tau_g[:n_u])

        (A, B, D, d, E, F, H, c,
         J_n, J_t, phi, mu_out) = formulator.linearize_discrete(
            ctx, DT, u_lin=u_lin)

        cinfo = list(getattr(formulator, "_last_contact_info", []))
        tags = [ci.get("tag", "?") for ci in cinfo]
        try:
            i_ee = tags.index("EE-BOX")
        except ValueError:
            print(f"[{name}] FAIL — no EE-BOX pair admitted (tags={tags}, "
                  f"ik_err={ik_err:.4f})")
            failures += 1
            continue
        n_c = len(cinfo)
        n_lam = D.shape[1]
        assert n_lam == 4 * n_c, f"n_lam={n_lam} != 4·n_c={4*n_c}"

        lam = np.zeros(n_lam)
        lam[4 * i_ee: 4 * i_ee + 4] = 1.0     # unit push on all 4 folds
        dx = D @ lam

        d_wz = float(dx[WZ_X])
        d_qz = float(dx[QZ_X])
        nmap_pred = 0.5 * DT * d_wz           # identity-quat q̇_qz = ½ω_z
        # Stage-2 consistency: D_q = dt²·N·M⁻¹Jᵀ vs dt·(½·dt·ω) — the
        # dt² and the ½ live inside N_mat·dt²; compare SIGN and ratio.
        jac_ok = (np.sign(d_wz) == tau_sign)
        nmap_ok = (np.sign(d_qz) == np.sign(nmap_pred)) or d_qz == nmap_pred == 0.0

        nh = cinfo[i_ee].get("nhat_onto_box", cinfo[i_ee].get("nhat_BA_W"))
        print(f"[{name}] ik_err={ik_err:.4f} tags={tags}")
        print(f"    A={cinfo[i_ee]['body_A']}/{cinfo[i_ee]['elem_A']} "
              f"B={cinfo[i_ee]['body_B']}/{cinfo[i_ee]['elem_B']} "
              f"a_is_box={cinfo[i_ee]['a_is_box']}")
        print(f"    p_ACa={np.round(cinfo[i_ee]['p_ACa'], 4)} "
              f"p_BCb={np.round(cinfo[i_ee]['p_BCb'], 4)} "
              f"(box-side witness in BODY frame should be "
              f"({r_body[0]:+.2f},{r_body[1]:+.2f},·))")
        print(f"    nhat_onto_box={np.round(np.asarray(nh, float), 3)} "
              f"phi={cinfo[i_ee].get('distance', float('nan')):+.4f}")
        print(f"    expected τ_z sign: {'+' if tau_sign > 0 else '-'}  "
              f"(r_body={r_body}, F=+y on box)")
        print(f"    Δω_z (D vel rows)  = {d_wz:+.6f}   "
              f"→ Jacobian stage {'OK' if jac_ok else '*** SIGN FLIP ***'}")
        print(f"    Δqz  (D pos rows)  = {d_qz:+.6f}   "
              f"vs ½·dt·Δω_z = {nmap_pred:+.6f} "
              f"→ N-map stage {'OK' if nmap_ok else '*** SIGN FLIP ***'}")
        if not (jac_ok and nmap_ok):
            failures += 1

    print(f"\n{'FAIL' if failures else 'PASS'} — {failures} probe(s) with "
          f"sign inversion")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
