"""
Find a better INITIAL_ARM_Q for the Franka Panda + spherical pusher setup.

Criteria:
  1. Pusher centre z ≈ 0.05 m  (within ±0.02 m)
  2. Pusher centre x < -0.10 m  (west of box, which sits at (0,0,0.05))
  3. All arm links ≥ 5 cm from each other  (no self-collision)
  4. All arm links ≥ 10 cm from the box    (box clearance)

Approach: random seed → Drake InverseKinematics with collision constraints.
Reports every solution that passes all four checks, sorted by box clearance.

Usage (from project root):
    python scripts/find_initial_q.py [--n-seeds N]
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
import pydrake.all as ad
from sim.env_builder import build_environment, EE_BODY_NAME, PUSHER_RADIUS

# ---- Joint limits (Franka Panda, radians) -----------------------------------
Q_LO = np.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.0175, -2.897])
Q_HI = np.array([ 2.897,  1.763,  2.897, -0.0698,  2.897,  3.752,   2.897])

# ---- Targets ----------------------------------------------------------------
PUSHER_Z_TARGET  = 0.05    # m — mid-box height
PUSHER_Z_TOL     = 0.02    # m — acceptable band: [0.03, 0.07]
PUSHER_X_MAX     = -0.10   # m — must be west of this (x < -0.10)
MIN_SELF_DIST    = 0.05    # m — arm links must stay this far apart
MIN_BOX_DIST     = 0.10    # m — arm links (not pusher) must be this far from box


def load_task_cfg() -> dict:
    with open("config/tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]["pushing"]


def check_pose(plant, plant_ctx, scene_graph_ctx,
               diagram_ctx, q_arm: np.ndarray,
               box_q: np.ndarray, n_q: int,
               scene_graph,
               pusher_body, pusher_geom_ids: set,
               box_geom_ids: set) -> dict | None:
    """
    Set q_arm in plant_ctx, then evaluate FK and collision.
    Returns a result dict if all criteria pass, else None.
    """
    q_full = np.concatenate([q_arm, box_q])
    if len(q_full) != n_q:
        return None
    plant.SetPositions(plant_ctx, q_full)

    # FK: pusher centre
    pusher_pos = plant.EvalBodyPoseInWorld(plant_ctx, pusher_body).translation()
    pz = float(pusher_pos[2])
    px = float(pusher_pos[0])

    if not (PUSHER_Z_TARGET - PUSHER_Z_TOL <= pz <= PUSHER_Z_TARGET + PUSHER_Z_TOL):
        return None
    if px >= PUSHER_X_MAX:
        return None

    # Collision queries require geometry query port (diagram context)
    query_obj = scene_graph.get_query_output_port().Eval(diagram_ctx)
    pairs = query_obj.ComputeSignedDistancePairwiseClosestPoints(0.30)

    min_self_dist = float('inf')
    min_box_dist  = float('inf')

    for pair in pairs:
        idA = pair.id_A
        idB = pair.id_B
        d   = float(pair.distance)

        is_pusher = (idA in pusher_geom_ids or idB in pusher_geom_ids)
        is_box    = (idA in box_geom_ids    or idB in box_geom_ids)

        if not is_box and not is_pusher:
            # arm link vs arm link → self-collision check
            min_self_dist = min(min_self_dist, d)
        elif is_box and not is_pusher:
            # arm link (not pusher) vs box → box clearance check
            min_box_dist = min(min_box_dist, d)

    if min_self_dist < MIN_SELF_DIST:
        return None
    if min_box_dist < MIN_BOX_DIST:
        return None

    return {
        "q":             q_arm.tolist(),
        "pusher_xyz":    pusher_pos.round(4).tolist(),
        "min_self_dist": round(min_self_dist, 4),
        "min_box_dist":  round(min_box_dist, 4),
    }


def run_ik_seed(plant, diagram, scene_graph,
                pusher_body, q_seed: np.ndarray,
                target_xyz: np.ndarray) -> np.ndarray | None:
    """Run Drake IK from q_seed targeting pusher at target_xyz; return q or None."""
    ik = ad.InverseKinematics(plant, with_joint_limits=True)

    # Pusher position constraint: target with ±0.015 m box
    ik.AddPositionConstraint(
        pusher_body.body_frame(),
        np.zeros(3),
        plant.world_frame(),
        target_xyz - 0.015,
        target_xyz + 0.015,
    )

    # Minimum distance between all collision geometries (self-col + box)
    ik.AddMinimumDistanceLowerBoundConstraint(MIN_SELF_DIST, 0.01)

    prog = ik.prog()
    q_var = ik.q()
    prog.SetInitialGuess(q_var, q_seed)

    result = ad.Solve(prog)
    if not result.is_success():
        return None
    q_sol = result.GetSolution(q_var)
    return q_sol[:7]   # arm joints only


def main():
    parser = argparse.ArgumentParser(description="Find good INITIAL_ARM_Q candidates")
    parser.add_argument("--n-seeds", type=int, default=200,
                        help="Number of random seeds to try (default 200)")
    args = parser.parse_args()

    task_cfg = load_task_cfg()
    diagram, plant, panda_model, obj_model, _ = build_environment(task_cfg)
    ctx       = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(ctx)

    # Find SceneGraph from the diagram
    scene_graph = None
    for sys in diagram.GetSystems():
        if isinstance(sys, ad.SceneGraph):
            scene_graph = sys
            break
    if scene_graph is None:
        # Fallback: get via port inspection
        print("[WARN] Could not find SceneGraph by type scan. Collision checks disabled.")

    sg_ctx = scene_graph.GetMyContextFromRoot(ctx) if scene_graph else None

    n_q = plant.num_positions()
    pusher_body = plant.GetBodyByName(EE_BODY_NAME)
    box_body    = plant.GetBodyByName(task_cfg["link_name"])

    pusher_geom_ids = set(plant.GetCollisionGeometriesForBody(pusher_body))
    box_geom_ids    = set(plant.GetCollisionGeometriesForBody(box_body))

    # Box q: identity quaternion + init_xyz
    box_q = np.array([1.0, 0.0, 0.0, 0.0,
                      task_cfg["init_xyz"][0],
                      task_cfg["init_xyz"][1],
                      task_cfg["init_xyz"][2]])

    # IK target positions to try: several x offsets, fixed y=0, z=0.05
    target_positions = [
        np.array([x, 0.0, 0.05])
        for x in [-0.12, -0.15, -0.18, -0.20, -0.25]
    ]

    rng = np.random.default_rng(42)
    results = []

    print(f"Searching {args.n_seeds} random seeds × {len(target_positions)} targets "
          f"= {args.n_seeds * len(target_positions)} IK attempts ...")
    print(f"Criteria: pusher z∈[{PUSHER_Z_TARGET-PUSHER_Z_TOL:.2f}, "
          f"{PUSHER_Z_TARGET+PUSHER_Z_TOL:.2f}]  "
          f"x<{PUSHER_X_MAX}  self≥{MIN_SELF_DIST}m  box≥{MIN_BOX_DIST}m\n")

    for i in range(args.n_seeds):
        q_seed_arm = rng.uniform(Q_LO, Q_HI)
        q_seed_full = np.concatenate([q_seed_arm, box_q])
        plant.SetPositions(plant_ctx, q_seed_full)

        for target_xyz in target_positions:
            q_arm = run_ik_seed(plant, diagram, scene_graph,
                                pusher_body, q_seed_full, target_xyz)
            if q_arm is None:
                continue

            res = check_pose(plant, plant_ctx, sg_ctx, ctx,
                             q_arm, box_q, n_q, scene_graph,
                             pusher_body, pusher_geom_ids, box_geom_ids)
            if res is not None:
                # Deduplicate: skip if very close to an already-found solution
                duplicate = any(
                    np.linalg.norm(np.array(r["q"]) - np.array(res["q"])) < 0.05
                    for r in results
                )
                if not duplicate:
                    results.append(res)
                    print(f"  [found {len(results)}] "
                          f"pusher={res['pusher_xyz']}  "
                          f"self_dist={res['min_self_dist']}  "
                          f"box_dist={res['min_box_dist']}")

    print(f"\n{'='*70}")
    if not results:
        print("No valid poses found. Try increasing --n-seeds or relaxing thresholds.")
        return

    # Sort by box clearance (largest first = most conservative)
    results.sort(key=lambda r: r["min_box_dist"], reverse=True)

    print(f"\n{len(results)} valid pose(s) found. Best candidates:\n")
    for k, r in enumerate(results[:5]):
        q_str = ", ".join(f"{v:.4f}" for v in r["q"])
        print(f"Rank {k+1}:")
        print(f"  INITIAL_ARM_Q = np.array([{q_str}])")
        print(f"  pusher xyz    = {r['pusher_xyz']}")
        print(f"  self clearance = {r['min_self_dist']} m  "
              f"(need ≥{MIN_SELF_DIST})")
        print(f"  box  clearance = {r['min_box_dist']} m  "
              f"(need ≥{MIN_BOX_DIST})")
        print()

    print("Paste the Rank 1 line into sim/env_builder.py as INITIAL_ARM_Q.")
    print("Then run:  python scripts/check_pose.py  to verify the FK.")


if __name__ == "__main__":
    main()
