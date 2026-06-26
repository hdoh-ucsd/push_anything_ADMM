"""Stage C friction audit (offline).

Question (§7.13 routed): of the 1.43× residual at Δt=0.005 (sub-stepped
path), how much is friction-mechanism (Stewart-Trinkle's instantaneous
λ_t resolution vs Drake's compliant friction)?

By elimination at this state, count=4: Δt resolved as dominant (§7.13),
drag resolved as tiny (§7.12), μ matches (0.4). The only remaining
mechanism difference is friction enforcement.

Method (approach (a) — direct decomposition; cleanest isolation):

  1. Run the LCS sub-stepped (Δt=0.005, 10× re-extract) path. At each
     sub-step, decompose the LCS-predicted Δbox_x change into:
       • EE-NORMAL push  : D[box_x, λ_n_EE_idx] · λ[λ_n_EE_idx]
       • FLOOR-NORMAL    : D[box_x, λ_n_floor_idx] · λ[λ_n_floor_idx]
                            (should be ~0; floor normal is +ẑ)
       • EE-TANGENT      : D[box_x, λ_t_EE_idx] · λ[λ_t_EE_idx]
                            (~0; tangent at east face is in y/z)
       • FLOOR-TANGENT   : D[box_x, λ_t_floor_idx] · λ[λ_t_floor_idx]
                            (the floor-friction resistance to push)

  2. Run Drake forward 0.05 s in 1 ms ticks. At each tick, read
     ContactResults, decompose each pair's contact_force into normal +
     tangent components in world frame, identify EE-BOX vs BOX-GND
     pairs, project onto +x. Trapezoidal-integrate to get impulses over
     the 0.05 s window, decomposed:
       • EE-BOX normal impulse on box_x  (push driver)
       • BOX-GND tangent impulse on box_x (friction resister)

  3. Compare LCS impulses to Drake impulses:
       • LCS EE-push vs Drake EE-push   (if mismatched → normal mechanism;
         entangled, NOT clean friction isolation)
       • LCS floor-friction vs Drake floor-friction
         (THE friction-mechanism test)

Isolation flag: if the LCS EE-push impulse already matches Drake's →
clean isolation; the residual is friction-mechanism. If LCS EE-push <
Drake's EE-push → friction is entangled with normal-mechanism, the
audit does NOT cleanly isolate friction.

Pre-registered three-outcome (on the friction explanation of the 1.43×
residual):
  FRICTION-CLOSES        — friction-mechanism ≈ residual; LCS matches
                           Drake to ~1× → MODEL-FIXED-REAL.
  FRICTION-PARTIAL-FLOOR — friction explains part; sub-10% residual is
                           the IRREDUCIBLE Stewart-Trinkle-vs-compliant
                           floor → ACCEPT, a legitimate stopping point.
  FRICTION-BARELY        — friction barely moves it; residual is some-
                           thing else → anitescu scoping becomes live.
"""
from __future__ import annotations

import os
import importlib
from dataclasses import dataclass

import numpy as np
import yaml

from pydrake.systems.analysis import Simulator
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import Solve

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

DT_BIG    = 0.05
DT_SUB    = 0.005
N_SUB     = int(round(DT_BIG / DT_SUB))    # 10
DT_DRAKE  = 0.001
N_DRAKE   = int(round(DT_BIG / DT_DRAKE))  # 50

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_PEN_M  = 0.0001
EE_POS    = np.array([BOX_HALF + EE_RADIUS - EE_PEN_M, 0.0, 0.05])
EE_VEL_X  = -0.05

BOX_POS_X_IDX  = 4    # box position x in EE-space layout (after quat[0:4])
BOX_VEL_X_IDX  = 13   # box velocity x (after box pos[4:7] + ee pos[7:10] + box ω[10:13])
N_CONTACTS_EXPECTED = 5   # 1 EE-BOX + 4 BOX-VERT under count=4


# --------------- contact-state setup (copy of dt-factorial helper) ---------

def setup_contact_state(diagram, plant, panda_model, object_model, ee_frame):
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_ctx, object_model, np.concatenate([BOX_QUAT, BOX_POS]))
    plant.SetPositions(plant_ctx, panda_model,
                       np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785]))
    ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    p_tol = 1e-4
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3), plant.world_frame(),
        EE_POS - p_tol, EE_POS + p_tol,
    )
    ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
    res = Solve(ik.prog())
    if not res.is_success():
        return None, None, None
    plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
    q_arm_init = plant.GetPositions(plant_ctx, panda_model).copy()
    plant.SetVelocities(plant_ctx, object_model, np.zeros(6))
    J = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
        plant.world_frame(), plant.world_frame())
    s = plant.GetJointByName("panda_joint1").velocity_start()
    q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7], np.array([EE_VEL_X, 0., 0.]), rcond=None)
    plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
    return context, plant_ctx, q_arm_init


def write_state_to_plant(plant, plant_ctx, x, panda_model, object_model,
                         ee_frame, q_arm_warm):
    plant.SetPositions(plant_ctx, object_model, x[0:7])
    plant.SetVelocities(plant_ctx, object_model, x[10:16])
    plant.SetPositions(plant_ctx, panda_model, q_arm_warm)
    ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3), plant.world_frame(),
        x[7:10] - 1e-5, x[7:10] + 1e-5,
    )
    ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
    res = Solve(ik.prog())
    if not res.is_success():
        return None
    plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
    q_arm_new = plant.GetPositions(plant_ctx, panda_model).copy()
    J = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
        plant.world_frame(), plant.world_frame())
    s = plant.GetJointByName("panda_joint1").velocity_start()
    q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7], x[16:19], rcond=None)
    plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
    return q_arm_new


def build_x0(plant, plant_ctx, ee_frame):
    ee = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    return np.concatenate([
        BOX_QUAT, BOX_POS, ee, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])


# ----- LCS path with per-sub-step force decomposition ---------------------

@dataclass
class LCSSubStep:
    t: float
    box_x_pred: float
    lam_n_ee: float
    lam_n_floor_sum: float
    delta_box_x_from_ee_normal: float
    delta_box_x_from_floor_normal: float
    delta_box_x_from_ee_tangent: float
    delta_box_x_from_floor_tangent: float
    delta_box_x_total: float


def lcs_substep_path_decomposed(plant, plant_ctx, ee_frame, obj_body,
                                plant_ad, ctx_ad, mu,
                                panda_model, object_model, q_arm_init,
                                box_drag=10.0):
    from control.lcs_formulator import LCSFormulator
    x_curr = build_x0(plant, plant_ctx, ee_frame)
    box_x_0 = float(x_curr[BOX_POS_X_IDX])
    q_arm = q_arm_init.copy()
    rows = []
    fail_steps = 0
    for k in range(N_SUB):
        f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                          plant_ad=plant_ad, context_ad=ctx_ad,
                          box_ground_drag=box_drag)
        A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
            f.linearize_discrete_ee_space(plant_ctx, DT_SUB, np.zeros(3))
        n_lam = D.shape[1]
        n_c = n_lam // 6
        if n_c != N_CONTACTS_EXPECTED:
            print(f"  [WARN] sub-step {k}: n_c={n_c} (expected {N_CONTACTS_EXPECTED})")

        # Index groups: λ = [γ (n_c), λ_n (n_c), λ_t (4·n_c)]
        # contact 0 is the EE-BOX pair (first admitted); 1..4 are BOX-VERT.
        # Verify via _last_contact_info.
        contacts = getattr(f, '_last_contact_info', [])
        ee_box_idx = None
        floor_idx = []
        for ci, info in enumerate(contacts):
            tag = info.get('tag', '')
            if tag == 'EE-BOX':
                ee_box_idx = ci
            else:
                floor_idx.append(ci)
        if ee_box_idx is None or len(floor_idx) != 4:
            print(f"  [WARN] sub-step {k}: contact tagging unexpected "
                  f"(EE-BOX={ee_box_idx}, floor={floor_idx})")

        lam_n_offset = n_c
        lam_t_offset = 2 * n_c

        ee_box_lam_n = lam_n_offset + ee_box_idx
        floor_lam_n  = [lam_n_offset + i for i in floor_idx]
        ee_box_lam_t = list(range(lam_t_offset + 4*ee_box_idx,
                                  lam_t_offset + 4*ee_box_idx + 4))
        floor_lam_t  = []
        for i in floor_idx:
            floor_lam_t.extend(range(lam_t_offset + 4*i,
                                     lam_t_offset + 4*i + 4))

        # Solve LCP
        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        x_next = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const

        # Decompose Δbox_x = D[BOX_POS_X_IDX, :] @ λ (relative to x_curr terms
        # in A and d which are NOT λ-dependent). We isolate the λ-DRIVEN
        # change-of-state at box_x — what each contact-group contributes.
        D_box_x = D[BOX_POS_X_IDX, :]
        delta_box_x_from_ee_normal     = D_box_x[ee_box_lam_n] * lam[ee_box_lam_n]
        delta_box_x_from_floor_normal  = float(D_box_x[floor_lam_n] @ lam[floor_lam_n])
        delta_box_x_from_ee_tangent    = float(D_box_x[ee_box_lam_t] @ lam[ee_box_lam_t])
        delta_box_x_from_floor_tangent = float(D_box_x[floor_lam_t] @ lam[floor_lam_t])
        delta_box_x_lambda_total       = float(D_box_x @ lam)

        rows.append(LCSSubStep(
            t=(k+1) * DT_SUB,
            box_x_pred=float(x_next[BOX_POS_X_IDX]),
            lam_n_ee=float(lam[ee_box_lam_n]),
            lam_n_floor_sum=float(np.sum(lam[floor_lam_n])),
            delta_box_x_from_ee_normal=float(delta_box_x_from_ee_normal),
            delta_box_x_from_floor_normal=delta_box_x_from_floor_normal,
            delta_box_x_from_ee_tangent=delta_box_x_from_ee_tangent,
            delta_box_x_from_floor_tangent=delta_box_x_from_floor_tangent,
            delta_box_x_total=delta_box_x_lambda_total,
        ))

        if k < N_SUB - 1:
            new_q_arm = write_state_to_plant(
                plant, plant_ctx, x_next, panda_model, object_model,
                ee_frame, q_arm)
            if new_q_arm is None:
                fail_steps += 1
                break
            q_arm = new_q_arm
        x_curr = x_next
    return rows, float(x_curr[BOX_POS_X_IDX]) - box_x_0, fail_steps


# ----- Drake reference path with per-tick contact-force decomposition ----

@dataclass
class DrakeTick:
    t: float
    box_x: float
    box_v_x: float
    F_pusher_box_x: float     # x-force on box from pusher↔box (the LEGITIMATE EE pair)
    F_arm_box_x: float        # x-force on box from panda_link*↔box (ARTIFACT, LCS-excluded)
    F_floor_normal_x: float   # x-force on box from floor NORMAL (~0)
    F_floor_tangent_x: float  # x-force on box from floor TANGENT (the friction)


def drake_reference_path(diagram, plant, ctx, ctx_plant, obj_body, ee_body):
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    ticks = []
    diag_printed = False
    for k in range(N_DRAKE + 1):
        t = k * DT_DRAKE
        if k > 0:
            sim.AdvanceTo(t)
        plant_ctx_now = plant.GetMyContextFromRoot(ctx)
        box_xyz = plant.EvalBodyPoseInWorld(plant_ctx_now, obj_body).translation()
        box_v   = plant.EvalBodySpatialVelocityInWorld(plant_ctx_now,
                                                       obj_body).translational()
        # ContactResults
        try:
            cr = plant.get_contact_results_output_port().Eval(plant_ctx_now)
        except Exception as e:
            print(f"[drake] contact_results eval failed at t={t}: {e}")
            ticks.append(DrakeTick(t, float(box_xyz[0]), float(box_v[0]),
                                   0.0, 0.0, 0.0))
            continue
        F_pusher_box_x = 0.0
        F_arm_box_x = 0.0
        F_floor_normal_x = 0.0
        F_floor_tangent_x = 0.0
        npp = cr.num_point_pair_contacts()
        for i in range(npp):
            info = cr.point_pair_contact_info(i)
            f_on_C2_W = info.contact_force()  # Drake: force on bodyB applied at contact
            nA = info.bodyA_index()
            nB = info.bodyB_index()
            bodyA = plant.get_body(nA)
            bodyB = plant.get_body(nB)
            # Drake convention: contact_force() is the force applied to body B
            # at the contact point (per pydrake docs of PointPairContactInfo).
            # If body B is the box → F_box = +f_on_C2_W. Else (B != box) F_box = -f_on_C2_W.
            if bodyB.index() == obj_body.index():
                F_box = f_on_C2_W
                other = bodyA
            elif bodyA.index() == obj_body.index():
                F_box = -f_on_C2_W
                other = bodyB
            else:
                continue  # contact doesn't involve the box

            # Normal direction (point pair): use point_pair.nhat_BA_W convention
            pp = info.point_pair()
            nhat = pp.nhat_BA_W  # unit vector from B to A
            # Force normal component: project F_box onto the normal direction
            # The contact normal points along nhat; "normal force on A from B"
            # ≈ projection magnitude on nhat. Decompose F_box into:
            #   F_normal = (F_box · nhat) * nhat
            #   F_tangent = F_box - F_normal
            f_norm_mag = float(F_box @ nhat)
            F_norm = f_norm_mag * nhat
            F_tang = F_box - F_norm

            other_name = other.name().lower()
            is_ground = (other.index() == plant.world_body().index()
                         or "world" in other_name
                         or "ground" in other_name)
            is_pusher = "pusher" in other_name
            is_arm = ("panda" in other_name) and not is_pusher
            if k == 5 and not diag_printed:
                print(f"    [diag t={t:.4f}] pair {i}: other='{other_name}' "
                      f"pusher={is_pusher} arm={is_arm} ground={is_ground} "
                      f"F_box=({F_box[0]:+.3f},{F_box[1]:+.3f},{F_box[2]:+.3f}) N "
                      f"nhat_BA=({nhat[0]:+.3f},{nhat[1]:+.3f},{nhat[2]:+.3f})")
            if is_pusher:
                F_pusher_box_x += float(F_box[0])
            elif is_arm:
                F_arm_box_x += float(F_box[0])
            elif is_ground:
                F_floor_normal_x  += float(F_norm[0])
                F_floor_tangent_x += float(F_tang[0])
            else:
                pass
        if k == 5:
            diag_printed = True
        ticks.append(DrakeTick(t, float(box_xyz[0]), float(box_v[0]),
                               F_pusher_box_x, F_arm_box_x,
                               F_floor_normal_x, F_floor_tangent_x))
    return ticks


def trapz_impulse(ts, fs):
    """Trapezoidal integral of f(t) over the time series."""
    s = 0.0
    for i in range(1, len(ts)):
        s += 0.5 * (fs[i] + fs[i-1]) * (ts[i] - ts[i-1])
    return s


# ----- main --------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("STAGE C  FRICTION AUDIT  (offline, sub-stepped path, count=4)")
    print("=" * 72)

    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]
    print(f"  μ = {mu} (matches Drake compliant friction coefficient)")

    # ---- LCS path (drag=10.0; drag=0 gives essentially same result per §7.13) ----
    diagram, plant, panda, object_, _, plant_ad, ctx_ad = \
        build_environment(task_cfg, time_step=0.001)
    obj_body = plant.GetBodyByName("box_link", object_)
    ee_frame = plant.GetFrameByName("pusher")
    ctx, pctx, q_arm_init = setup_contact_state(
        diagram, plant, panda, object_, ee_frame)
    rows, total_dx_lcs, fail_steps = lcs_substep_path_decomposed(
        plant, pctx, ee_frame, obj_body, plant_ad, ctx_ad, mu,
        panda, object_, q_arm_init, box_drag=10.0)
    print(f"\n  LCS sub-stepped (Δt={DT_SUB}, N={N_SUB}, drag=10.0): "
          f"total Δbox_x = {total_dx_lcs*1000:+.3f} mm  "
          f"(IK fails: {fail_steps})")

    # Per-sub-step decomposition table
    print("\n  Per-sub-step Δbox_x decomposition (mm):")
    print(f"    {'t':>6} {'λn_EE':>8} {'λn_flr':>8} "
          f"{'ΔEE_n':>8} {'Δfl_n':>8} {'ΔEE_t':>8} {'Δfl_t':>8} {'Σλ':>8}")
    for r in rows:
        print(f"    {r.t:6.3f} {r.lam_n_ee:8.3f} {r.lam_n_floor_sum:8.3f} "
              f"{r.delta_box_x_from_ee_normal*1000:+8.3f} "
              f"{r.delta_box_x_from_floor_normal*1000:+8.3f} "
              f"{r.delta_box_x_from_ee_tangent*1000:+8.3f} "
              f"{r.delta_box_x_from_floor_tangent*1000:+8.3f} "
              f"{r.delta_box_x_total*1000:+8.3f}")

    # ---- Drake reference path ----
    diagram_g, plant_g, panda_g, object_g, _, plant_ad_g, ctx_ad_g = \
        build_environment(task_cfg, time_step=0.001)
    obj_body_g = plant_g.GetBodyByName("box_link", object_g)
    ee_frame_g = plant_g.GetFrameByName("pusher")
    ee_body_g = plant_g.GetBodyByName("pusher")  # pusher body
    ctx_g, pctx_g, _ = setup_contact_state(
        diagram_g, plant_g, panda_g, object_g, ee_frame_g)
    print(f"\n  Drake reference (Δt={DT_DRAKE}, N={N_DRAKE}):")
    ticks = drake_reference_path(diagram_g, plant_g, ctx_g, pctx_g,
                                 obj_body_g, ee_body_g)

    total_dx_drake = ticks[-1].box_x - ticks[0].box_x
    print(f"    total Δbox_x = {total_dx_drake*1000:+.3f} mm")

    # ---- impulse decomposition (Drake) — separate pusher↔box from arm↔box artifact ----
    ts = [t.t for t in ticks]
    F_pusher  = [t.F_pusher_box_x  for t in ticks]
    F_arm     = [t.F_arm_box_x     for t in ticks]
    F_floor_n = [t.F_floor_normal_x for t in ticks]
    F_floor_t = [t.F_floor_tangent_x for t in ticks]
    J_pusher_drake = trapz_impulse(ts, F_pusher)
    J_arm_drake    = trapz_impulse(ts, F_arm)
    J_floor_n_drake = trapz_impulse(ts, F_floor_n)
    J_floor_t_drake = trapz_impulse(ts, F_floor_t)
    print(f"\n  Drake impulse on box_x over 0.05s (N·s):")
    print(f"    pusher↔box  (LEGIT EE pair) : {J_pusher_drake:+.6f}")
    print(f"    arm↔box     (ARTIFACT, LCS-excluded) : {J_arm_drake:+.6f}")
    print(f"    floor normal (≈0 expected)  : {J_floor_n_drake:+.6f}")
    print(f"    floor tangent (friction)    : {J_floor_t_drake:+.6f}")
    sum_all = J_pusher_drake + J_arm_drake + J_floor_n_drake + J_floor_t_drake
    print(f"    sum                          : {sum_all:+.6f}")
    if abs(J_arm_drake) > 0.1 * abs(J_pusher_drake):
        print(f"    [FLAG] arm↔box artifact ({J_arm_drake:+.4f} N·s) is "
              f"comparable to pusher↔box ({J_pusher_drake:+.4f} N·s).")
        print(f"           The IK-set arm pose puts panda_link* into the box.")
        print(f"           Drake's reference is contaminated by this contact;")
        print(f"           the LCS does NOT see it (filtered to pusher only).")
        print(f"           Apples-to-apples comparison: subtract arm impulse")
        print(f"           from Drake's reference Δbox_x estimate.")

    # ---- impulse decomposition (LCS) ----
    # The LCS impulse-equivalent at each sub-step is the λ × Δt scalar
    # (Stewart-Trinkle scaling: λ has units of N). But the friction-vs-push
    # comparison we want is at the level of "force on box_x" — equivalent to
    # Δbox_x_from_X / (DT_SUB * inv_inertial_term). Since we cannot easily
    # de-invert M^-1 from D, we compare in box-displacement units (mm)
    # instead — same units across the row.
    delta_ee_n     = sum(r.delta_box_x_from_ee_normal     for r in rows) * 1000
    delta_floor_n  = sum(r.delta_box_x_from_floor_normal  for r in rows) * 1000
    delta_ee_t     = sum(r.delta_box_x_from_ee_tangent    for r in rows) * 1000
    delta_floor_t  = sum(r.delta_box_x_from_floor_tangent for r in rows) * 1000
    print(f"\n  LCS cumulative Δbox_x decomposition (mm, λ-driven only):")
    print(f"    from EE  normal : {delta_ee_n:+.4f}")
    print(f"    from EE  tangent: {delta_ee_t:+.4f}")
    print(f"    from floor normal: {delta_floor_n:+.4f}")
    print(f"    from floor tangent (friction): {delta_floor_t:+.4f}")
    print(f"    SUM (λ-driven) : "
          f"{(delta_ee_n + delta_ee_t + delta_floor_n + delta_floor_t):+.4f}")

    # ---- trajectory comparison: LCS at sub-step times vs Drake at same t ----
    print(f"\n  Trajectory comparison at sub-step times (box_x in mm):")
    print(f"    {'t':>6} {'LCS':>9} {'Drake':>9} {'gap':>9}")
    box_x0 = ticks[0].box_x
    for r in rows:
        i_drake = int(round(r.t / DT_DRAKE))
        if i_drake >= len(ticks):
            i_drake = len(ticks) - 1
        drake_x = ticks[i_drake].box_x
        lcs_x = r.box_x_pred
        gap_mm = (drake_x - lcs_x) * 1000
        # box_x relative to start to keep numbers small
        print(f"    {r.t:6.3f} {(lcs_x - box_x0)*1000:+9.4f} "
              f"{(drake_x - box_x0)*1000:+9.4f} {gap_mm:+9.4f}")

    # ---- friction-isolation read ----
    print()
    print("=" * 72)
    print("FRICTION-ISOLATION READ")
    print("=" * 72)
    # The cleanest read: LCS EE-driven push (EE-normal + EE-tangent) vs
    # Drake's EE-on-box impulse. If they match → friction is isolated.
    lcs_ee_dx_mm = delta_ee_n + delta_ee_t
    drake_total_dx_mm = total_dx_drake * 1000

    # Drake EE-on-box impulse → effective box displacement contribution:
    # an impulse J [N·s] on a mass m_box ~ 0.2 kg gives velocity change
    # Δv = J/m. Over 0.05s the position contribution is harder to compare
    # without integrating; instead we compare the *force balance ratio*.
    drake_pusher_to_friction_ratio = (J_pusher_drake / J_floor_t_drake
                                      if abs(J_floor_t_drake) > 1e-9 else float('inf'))
    drake_arm_to_friction_ratio = (J_arm_drake / J_floor_t_drake
                                   if abs(J_floor_t_drake) > 1e-9 else float('inf'))
    lcs_ee_to_friction_ratio = (lcs_ee_dx_mm / delta_floor_t
                                if abs(delta_floor_t) > 1e-9 else float('inf'))
    print(f"  Drake pusher-impulse / floor-friction-impulse : "
          f"{drake_pusher_to_friction_ratio:+.4f}  (the LEGIT EE/friction balance)")
    print(f"  Drake arm-impulse    / floor-friction-impulse : "
          f"{drake_arm_to_friction_ratio:+.4f}  (the ARTIFACT/friction balance)")
    print(f"  LCS   EE-Δbox        / floor-friction-Δbox    : "
          f"{lcs_ee_to_friction_ratio:+.4f}")
    factor_remaining = (abs(drake_total_dx_mm / (total_dx_lcs * 1000))
                        if abs(total_dx_lcs) > 1e-9 else float('inf'))
    print(f"\n  Residual under-prediction factor (§7.13): {factor_remaining:.2f}×")

    # Apples-to-apples: estimate Drake-without-arm-artifact Δbox_x by
    # subtracting the arm-impulse's velocity contribution (rough; assumes
    # constant box mass and linear superposition over the 50ms — order-of-
    # magnitude only).
    if abs(J_arm_drake) > 0.01 * abs(J_pusher_drake):
        # Rough estimate: J_arm transfers momentum to box; mean velocity gain
        # over 0.05 s is J_arm/m, contributing Δx ≈ 0.5 · (J_arm/m) · T (if
        # the impulse is mid-window). Use box mass guess m_box ≈ 0.4 kg.
        m_box_guess = 0.4
        approx_dx_from_arm = 0.5 * (J_arm_drake / m_box_guess) * DT_BIG * 1000  # mm
        drake_dx_corrected = drake_total_dx_mm - approx_dx_from_arm
        factor_corrected = (abs(drake_dx_corrected / (total_dx_lcs * 1000))
                            if abs(total_dx_lcs) > 1e-9 else float('inf'))
        print(f"  Drake Δbox_x  RAW (with arm artifact): {drake_total_dx_mm:+.3f} mm")
        print(f"  Drake Δbox_x  ARM-CORRECTED (rough)  : {drake_dx_corrected:+.3f} mm  "
              f"(subtracted ~{approx_dx_from_arm:+.3f} mm est. from arm impulse)")
        print(f"  Residual factor ARM-CORRECTED         : {factor_corrected:.2f}×")

    print()
    print("=" * 72)
    print("OUTCOME ROUTING")
    print("=" * 72)
    # The friction component of the residual:
    # If LCS floor-friction Δbox > 0 (resists push) and is LARGER in
    # magnitude than Drake's equivalent → LCS over-friction → friction is
    # part of the residual. If smaller or zero → friction is NOT the
    # residual driver → BARELY → anitescu.
    # The arm-artifact magnitude overrides the friction-three-outcome:
    arm_dominance = (abs(J_arm_drake) / max(abs(J_pusher_drake), 1e-9))
    if arm_dominance > 5.0:
        outcome = "AUDIT-SETUP-BROKEN — phantom arm↔box contact dominates Drake reference"
        print(f"  ► OUTCOME: {outcome}")
        print(f"    arm impulse magnitude is {arm_dominance:.1f}× the legit pusher impulse;")
        print(f"    Drake's reference push is driven by panda_link*↔box, not pusher↔box.")
        print(f"    The §7.11→§7.13 chain's quantitative numbers (3.73×, 1.43×) compare")
        print(f"    LCS-pusher-only-model to Drake-with-arm-artifact-contact; apples-to-")
        print(f"    oranges. The audit cannot cleanly isolate friction-mechanism on this")
        print(f"    setup. Required: re-pose the constructed contact state so panda_link*")
        print(f"    does not contact the box, then redo §7.11/§7.12/§7.13/§7.14 quantitatively.")
    elif factor_remaining < 1.10:
        outcome = "FRICTION-CLOSES"
        print(f"  ► OUTCOME: {outcome}")
    elif factor_remaining < 1.30:
        outcome = "FRICTION-PARTIAL-FLOOR (sub-10% irreducible)"
        print(f"  ► OUTCOME: {outcome}")
    else:
        outcome = "FRICTION-BARELY (residual uncharacterized)"
        print(f"  ► OUTCOME: {outcome}")
        print(f"    (residual factor {factor_remaining:.2f}× at sub-stepped Δt=0.005)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
