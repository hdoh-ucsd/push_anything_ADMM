"""Smoke test for RepositionIKTracker after the obstacle-collision-filter
wiring (steps 4h-4j).

Run:
    python scripts/probe_reposition_ik_smoke.py

Expected (PASS):
    feasible[0] == True for an in-workspace target above the table.
    The (table, manipuland) resting-contact pair is filtered, so the
    global AddMinimumDistanceLowerBoundConstraint no longer reports the
    near-zero gap between table and box and IK can succeed.

Pre-fix behaviour (FAIL):
    feasible[0] == False because the global min-distance constraint sees
    the resting (table, box) contact at gap ≈ 0 < d_min and rejects every
    IK candidate regardless of arm pose.

Scope: this is the gating smoke test before wiring RepositionIKTracker
into wrapper.py. It should be the only thing standing between the
collision-filter change and step 5 of the larger plan.
"""
from __future__ import annotations

import numpy as np
import pydrake.all as ad

from sim.env_builder import (
    build_environment, EE_BODY_NAME, INITIAL_ARM_Q, ROBOT_BASE_XYZ,
)
from control.sampling_c3.params import (
    RepositionParams, RepositionIKParams, RepositioningTrajectoryType,
)
from control.sampling_c3.reposition_ik import RepositionIKTracker


def _set_state(plant, plant_ctx, panda_model, obj_body, *, arm_q, obj_xy, obj_z):
    """Mirror main.py's startup state — arm at INITIAL_ARM_Q, box upright
    on the table at (obj_xy, obj_z)."""
    n_q = plant.num_positions()
    q   = np.zeros(n_q)
    # Arm joints (first 7 positions in our layout — env_builder welds the
    # base, so panda_link0..panda_link7's joints fill positions 0..6).
    q[:7] = arm_q
    # Floating-base box: identity quaternion, xy-z translation.
    s = obj_body.floating_positions_start()
    q[s + 0] = 1.0    # qw
    q[s + 1] = 0.0    # qx
    q[s + 2] = 0.0    # qy
    q[s + 3] = 0.0    # qz
    q[s + 4] = obj_xy[0]
    q[s + 5] = obj_xy[1]
    q[s + 6] = obj_z
    plant.SetPositions(plant_ctx, q)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
    return q


def main() -> int:
    # ---- Build the environment exactly as main.py does for `pushing` ----
    task_cfg = {
        "object_type": "box",
        "size": [0.1, 0.1, 0.1],
        "mass": 0.2,
        "friction": 0.4,
        "color_rgba": [0.6, 0.3, 0.1, 1.0],
    }
    diagram, plant, panda_model, object_model, _meshcat, _plant_ad, _ctx_ad = \
        build_environment(task_cfg)

    # Resolve scene_graph by walking the diagram. AddMultibodyPlantSceneGraph
    # gives us a SceneGraph instance in env_builder.py, but doesn't return it
    # — we recover it from the diagram's subsystems.
    scene_graph = None
    for sys in diagram.GetSystems():
        if isinstance(sys, ad.SceneGraph):
            scene_graph = sys
            break
    assert scene_graph is not None, "could not find SceneGraph in diagram"

    obj_body = plant.GetBodyByName("box_link", object_model)
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    # ---- Live sim's diagram context (the one main.py uses) ----
    diag_ctx_live  = diagram.CreateDefaultContext()
    plant_ctx_live = plant.GetMyMutableContextFromRoot(diag_ctx_live)

    # Box at (0, 0) on table top (z = 0.05 = box half-extent above z=0 table top).
    initial_q = _set_state(
        plant, plant_ctx_live, panda_model, obj_body,
        arm_q=INITIAL_ARM_Q, obj_xy=(0.0, 0.0), obj_z=0.05,
    )

    # ---- Build the tracker ----
    # The IPOPT warm-up at the end of __init__ pre-pays the cold-start
    # cost so the very first compute_torque() call hits the warm path
    # within the production 8 ms cap. Use defaults — no relaxed timeout.
    repos_params = RepositionParams(traj_type=RepositioningTrajectoryType.kIK)
    ik_params    = RepositionIKParams()

    tracker = RepositionIKTracker(
        plant=plant,
        ee_frame=ee_frame,
        obj_body=obj_body,
        n_arm_dofs=7,
        horizon=20,
        dt=0.05,
        repos_params=repos_params,
        ik_params=ik_params,
        diagram=diagram,
        scene_graph=scene_graph,
        # table_body=None — defaults to plant.world_body() (env_builder.py
        # registers the table on the world body).
    )
    print("[smoke] tracker constructed (filter assertion passed)")

    # Sanity diagnostic: confirm the live sim's plant_ctx still sees the
    # (table, box) pair (filter is context-local — should NOT bleed over).
    insp = scene_graph.model_inspector()
    table_frame_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    box_frame_id   = plant.GetBodyFrameIdOrThrow(obj_body.index())
    table_names = {insp.GetName(g)
                   for g in insp.GetGeometries(table_frame_id, ad.Role.kProximity)}
    box_names   = {insp.GetName(g)
                   for g in insp.GetGeometries(box_frame_id, ad.Role.kProximity)}
    print(f"[smoke] table geom names: {sorted(table_names)}")
    print(f"[smoke] box   geom names: {sorted(box_names)}")

    qo_live = plant.get_geometry_query_input_port().Eval(plant_ctx_live)
    pairs_live = qo_live.ComputeSignedDistancePairwiseClosestPoints(1.0)
    table_box_seen_live = False
    for p in pairs_live:
        a = insp.GetName(p.id_A)
        b = insp.GetName(p.id_B)
        if (a in table_names and b in box_names) or (a in box_names and b in table_names):
            table_box_seen_live = True
            print(f"[smoke] live sim sees (table, box) pair at gap = {p.distance:.4f}m  "
                  "(expected — filter is context-local)")
            break
    assert table_box_seen_live, "live sim should still see (table, box) pair"

    # ---- Pick an in-workspace target with no obstacle in the arm path ----
    # 30 cm above the table, 30 cm in front of the panda base, well clear
    # of the box at (0, 0, 0.05). Comfortably inside the panda's reach.
    p_target = np.array([0.0, -0.30, 0.30])
    print(f"[smoke] p_target = {p_target}")

    # ---- Call compute_torque ----
    n_v = plant.num_velocities()
    u, diag = tracker.compute_torque(
        current_q=initial_q,
        current_v=np.zeros(n_v),
        plant_ctx=plant_ctx_live,
        p_target=p_target,
        dt_ctrl=0.01,
    )

    # ---- Report ----
    print()
    print("[smoke] --- diag dict ---")
    print(f"  feasible[0]        = {diag['feasible'][0]}")
    print(f"  feasible[:5]       = {diag['feasible'][:5]}")
    print(f"  any_infeasible     = {diag['any_infeasible']}")
    print(f"  knot0_feasible     = {diag['knot0_feasible']}")
    print(f"  knot0_solve_ms     = {diag['knot0_solve_ms']:.2f}")
    print(f"  knot0_overshoot_ms = {diag['knot0_overshoot_ms']:.2f}")
    print(f"  ee_now             = {diag['ee_now']}")
    print(f"  p_des              = {diag['p_des']}")
    print(f"  ik_err             = {diag['ik_err']:.4e}")
    print(f"  ik_iters           = {diag['ik_iters']}")
    print(f"  finished           = {diag['finished']}")
    print(f"  ||u||              = {float(np.linalg.norm(u)):.3f}")
    print()

    if bool(diag["feasible"][0]):
        print("[SMOKE] PASS — knot-0 IK feasible after collision-filter wiring.")
        return 0
    else:
        print("[SMOKE] FAIL — knot-0 IK still infeasible. The filter is the "
              "wrong thing to fix, or it is not actually applying.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
