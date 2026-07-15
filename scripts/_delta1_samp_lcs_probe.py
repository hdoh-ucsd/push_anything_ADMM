"""Delta-1 audit probe — discriminator for plan-vs-cost LCS sharing.

Builds the InnerSolver stack (LCSFormulator + C3Solver + QuadraticManipulationCost +
SamplingC3Params + InnerSolver), poses the EE FAR from the box, and calls
evaluate_samples ONCE with two contrived samples:

  sample 0 (is_current_ee=True): current EE pose (far from box, no contact)
  sample 1 (is_current_ee=False): box face (touching), where a per-sample LCS
                                  rebuild would admit the EE-BOX contact pair

PUSHA_SAMP_LCS_DUMP=1 makes evaluate_samples emit `[SAMP-LCS]` lines per sample.
Discriminator on the captured output:

  IDENTICAL (n_c, phi_min, J_n_hash) across samples + same ee_pos_resolved
    → port shares ONE LCS at current EE pose. Delta-1 gap CONFIRMED.

  DIFFERENT (n_c, phi_min, or J_n_hash) across samples
    → port rebuilds per-sample. Delta-1 gap CLOSED.

Short sample-eval pass, NO full push sim. No EIO monitor required (the user's
spec: "A micro-dump run that's a short sample-eval pass is fine without the
monitor").
"""
from __future__ import annotations

import os
import sys
import numpy as np
import yaml

# Force the dump ON for this probe
os.environ["PUSHA_SAMP_LCS_DUMP"] = "1"
# Make sure no other gates perturb the baseline
for k in ("LCS_CONTACT_MODEL", "LCS_ALWAYS_ON_EE_BOX",
          "LCS_NORMAL_PHI_CLAMP", "LCS_NORMAL_COMPLIANCE_K",
          "LCS_NORMAL_VELOCITY_LEVEL", "LCS_EXPLICIT_BOX_GND",
          "REF_RECONCILE_APPROACH", "REF_RECONCILE_FEEDFORWARD_ACCEL"):
    if k in os.environ:
        del os.environ[k]

from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import Solve
from pydrake.geometry import Role
from pydrake.math import RotationMatrix

from sim.env_builder import build_environment
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver
from control.task_costs import QuadraticManipulationCost
from control.sampling_c3 import SamplingC3Params
from control.sampling_c3.inner_solve import InnerSolver

EE_BODY_NAME = "pusher"
DT_PLANNER   = 0.05
HORIZON      = 8

# Box geometry (matches the always-on sanity, matches config/tasks.yaml pushing)
BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])

# Pose EE FAR from box (well outside the 2 mm Drake admission filter)
# Choose a comfortable pose: ~30 cm east of box center, ~20 cm above table
EE_POS_FAR = np.array([0.30, 0.0, 0.20])

# Sample positions
SAMPLE_CURRENT     = EE_POS_FAR.copy()             # = current EE
SAMPLE_BOX_FACE    = np.array([                    # EE touching +x face of box
    BOX_POS[0] + BOX_HALF + EE_RADIUS,             # = +0.075
    0.0,
    BOX_POS[2]                                       # = +0.05
])

POSTURE_NOMINAL = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785])
POSTURE_WEIGHT  = 5.0
IK_SEEDS = [
    np.array([0.0,  0.0,  0.0, -1.5,  0.0,  1.5,  0.785]),
    np.array([0.0, -0.3,  0.0, -1.7,  0.0,  1.4,  0.785]),
    np.array([0.0,  0.3,  0.0, -1.3,  0.0,  1.6,  0.785]),
]


def pose_ee_at(diagram, plant, panda, obj, ee_frame, ee_target):
    world = plant.world_frame()
    box_body = plant.GetBodyByName("box_link", obj)
    p_tol = 1e-4
    for seed in IK_SEEDS:
        context = diagram.CreateDefaultContext()
        plant_ctx = plant.GetMyContextFromRoot(context)
        plant.SetPositions(plant_ctx, obj,
                           np.concatenate([BOX_QUAT, BOX_POS]))
        plant.SetPositions(plant_ctx, panda, seed)
        ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
        box_frame_pin = box_body.body_frame()
        ik.AddPositionConstraint(box_frame_pin, np.zeros(3), world,
                                 BOX_POS - 1e-5, BOX_POS + 1e-5)
        ik.AddOrientationConstraint(world, RotationMatrix(),
                                    box_frame_pin, RotationMatrix(), 0.001)
        ik.AddPositionConstraint(ee_frame, np.zeros(3), world,
                                 ee_target - p_tol, ee_target + p_tol)
        q_dec = ik.q()
        j1 = plant.GetJointByName("panda_joint1").position_start()
        for k in range(7):
            ik.get_mutable_prog().AddQuadraticErrorCost(
                np.array([[POSTURE_WEIGHT]]),
                np.array([POSTURE_NOMINAL[k]]),
                np.array([q_dec[j1 + k]]))
        ik.get_mutable_prog().SetInitialGuess(ik.q(),
                                              plant.GetPositions(plant_ctx))
        res = Solve(ik.prog())
        if not res.is_success():
            continue
        plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
        p_ee_actual = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world).flatten()
        if np.linalg.norm(p_ee_actual - ee_target) > 5e-3:
            continue
        plant.SetVelocities(plant_ctx, obj, np.zeros(6))
        plant.SetVelocities(plant_ctx, panda, np.zeros(7))
        return plant_ctx, p_ee_actual
    return None, None


def main() -> int:
    print("=" * 90)
    print("Delta-1 audit — [SAMP-LCS] discriminator probe (no full sim, no monitor)")
    print("=" * 90)
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    print(f"  task: pushing, μ={task_cfg['friction']}, dt={DT_PLANNER}s")
    print(f"  EE target: {EE_POS_FAR} m (FAR from box — out of 2 mm filter)")
    print(f"  sample 0 = current EE pose: {SAMPLE_CURRENT}")
    print(f"  sample 1 = box +x face touching: {SAMPLE_BOX_FACE}")
    print()

    diagram, plant, panda, obj, _, plant_ad, ctx_ad = build_environment(
        task_cfg, time_step=0.001)
    obj_body = plant.GetBodyByName("box_link", obj)
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    plant_ctx, p_ee_actual = pose_ee_at(
        diagram, plant, panda, obj, ee_frame, EE_POS_FAR)
    if plant_ctx is None:
        print("IK FAILED — cannot pose EE at the far target")
        return 1
    print(f"  posed EE at: {p_ee_actual}")
    print(f"  distance from box: "
          f"{np.linalg.norm(p_ee_actual - BOX_POS)*1000:.1f} mm  "
          f"(>> 2 mm filter)")
    print()

    # ------------------------------------------------------------------
    # Build the InnerSolver stack (mirrors main.py)
    # ------------------------------------------------------------------
    formulator = LCSFormulator(plant, mu=task_cfg["friction"],
                                obj_body=obj_body, plant_ad=plant_ad,
                                context_ad=ctx_ad, box_ground_drag=0.0)
    solver = C3Solver(n_x=19, n_u=3, rho=100.0, mode="c3plus",
                      math_diag=False, c3plus_projection="lcp")

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    quad_cost = QuadraticManipulationCost(
        plant, EE_BODY_NAME, obj_body, task_cfg["cost"], n_x=n_q + n_v,
        n_u=n_u, math_diag=False)

    sc3_params = SamplingC3Params.from_yaml("config/sampling_c3_kik.yaml")
    inner = InnerSolver(plant=plant, ee_frame=ee_frame, obj_body=obj_body,
                        formulator=formulator, solver=solver,
                        quad_cost=quad_cost, horizon=HORIZON, dt=DT_PLANNER,
                        torque_limit=30.0, base_admm_iter=25,
                        params=sc3_params)

    current_q = plant.GetPositions(plant_ctx)
    current_v = plant.GetVelocities(plant_ctx)
    target_xy = np.array(task_cfg["goal_xy"], dtype=float)
    g_hat_3d  = np.array([-1.0, 0.0, 0.0])   # west push (matches task 4)

    # ------------------------------------------------------------------
    # Invoke evaluate_samples ONCE — the dump fires per sample.
    # ------------------------------------------------------------------
    samples = [SAMPLE_CURRENT, SAMPLE_BOX_FACE]
    print("=== [SAMP-LCS] dump from inner.evaluate_samples (n_samples=2): ===")
    results = inner.evaluate_samples(
        samples       = samples,
        current_q     = current_q,
        current_v     = current_v,
        plant_ctx     = plant_ctx,
        target_xy     = target_xy,
        ee_pos_now    = p_ee_actual,
        g_hat_3d      = g_hat_3d,
        threading     = False,
        target_yaw    = 0.0,
    )
    print()

    # ------------------------------------------------------------------
    # Programmatic verdict
    # ------------------------------------------------------------------
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    if len(results) != 2:
        print(f"  unexpected results count {len(results)}")
        return 1
    r0, r1 = results
    if r0.J_n is None or r1.J_n is None:
        print(f"  one or both samples produced no LCS (J_n None):"
              f" r0.J_n is None={r0.J_n is None}, r1.J_n is None={r1.J_n is None}")
        return 1

    import hashlib
    n_c0 = int(r0.J_n.shape[0])
    n_c1 = int(r1.J_n.shape[0])
    phi0 = (float(np.min(r0.phi)) if r0.phi is not None and r0.phi.size > 0
            else float("nan"))
    phi1 = (float(np.min(r1.phi)) if r1.phi is not None and r1.phi.size > 0
            else float("nan"))
    h0 = hashlib.sha1(r0.J_n.tobytes()).hexdigest()
    h1 = hashlib.sha1(r1.J_n.tobytes()).hexdigest()
    er0 = r0.ee_pos_resolved
    er1 = r1.ee_pos_resolved

    same_n_c    = (n_c0 == n_c1)
    same_phi    = (np.isfinite(phi0) and np.isfinite(phi1)
                   and abs(phi0 - phi1) < 1e-12)
    same_J_n    = (h0 == h1)
    same_er     = (np.allclose(er0, er1, atol=1e-12))
    all_identical = same_n_c and same_phi and same_J_n and same_er

    print(f"  sample 0 (sample_pos far): n_c={n_c0}  phi_min={phi0:+.5f}  "
          f"J_n_hash={h0[:8]}  ee_pos_resolved={np.round(er0, 4)}")
    print(f"  sample 1 (sample_pos box-face): n_c={n_c1}  phi_min={phi1:+.5f}  "
          f"J_n_hash={h1[:8]}  ee_pos_resolved={np.round(er1, 4)}")
    print()
    print(f"  same n_c          : {same_n_c}")
    print(f"  same phi_min      : {same_phi}")
    print(f"  same J_n hash     : {same_J_n}")
    print(f"  same ee_pos_resolved: {same_er}")
    print()
    if all_identical:
        print("  → DELTA-1 GAP CONFIRMED: port builds ONE LCS at current EE")
        print("    pose and reuses it for every sample within the dispatch.")
        print("    Sample sample_pos differs but the linearization input does not.")
    else:
        print("  → DELTA-1 GAP CLOSED: port rebuilds the LCS per sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
