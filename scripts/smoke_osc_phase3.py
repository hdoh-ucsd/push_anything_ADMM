"""Phase 3 STEP B smoke test for OperationalSpaceTracker.

Builds the diagram (no sim step), instantiates the OSC tracker, calls
compute_torque on a synthetic state where p_target == current EE.
Expectation: zero-error case → τ holds the arm against gravity, so
|τ| ≈ |τ_g[arm]| (not literally zero).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sim.env_builder import build_environment, EE_BODY_NAME, INITIAL_ARM_Q
from control.osc import OperationalSpaceTracker


def main() -> int:
    with open(REPO / "config" / "tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]

    print("[smoke] build_environment(pushing) ...")
    diagram, plant, panda_model, object_model, meshcat, plant_ad, ctx_ad = \
        build_environment(task_cfg)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_arm = 7
    print(f"[smoke] plant: n_q={n_q} n_v={n_v} n_arm={n_arm}")

    ee_frame = plant.GetFrameByName(EE_BODY_NAME)
    print(f"[smoke] ee_frame={EE_BODY_NAME} OK")

    with open(REPO / "config" / "osc_franka.yaml") as f:
        gains_cfg = yaml.safe_load(f)
    gains_cfg["q_home_arm"] = list(INITIAL_ARM_Q)
    print(f"[smoke] gains_cfg loaded; q_home_arm shape={len(gains_cfg['q_home_arm'])}")

    tracker = OperationalSpaceTracker(
        plant=plant, ee_frame=ee_frame,
        n_arm_dofs=n_arm, gains_cfg=gains_cfg,
    )
    print("[smoke] OperationalSpaceTracker instantiated OK")

    diag_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diag_ctx)

    q = np.zeros(n_q)
    q[:n_arm] = INITIAL_ARM_Q
    obj_floating_qstart = plant.GetBodyByName(task_cfg["link_name"], object_model) \
        .floating_positions_start()
    q[obj_floating_qstart + 0] = 1.0
    q[obj_floating_qstart + 4] = task_cfg["init_xyz"][0]
    q[obj_floating_qstart + 5] = task_cfg["init_xyz"][1]
    q[obj_floating_qstart + 6] = task_cfg["init_xyz"][2]
    v = np.zeros(n_v)
    plant.SetPositions(plant_ctx, q)
    plant.SetVelocities(plant_ctx, v)

    ee_now = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()
    ).flatten()
    print(f"[smoke] ee_now @ INITIAL_ARM_Q = ({ee_now[0]:+.3f},"
          f"{ee_now[1]:+.3f},{ee_now[2]:+.3f})")

    tau, diag = tracker.compute_torque(
        current_q=q, current_v=v, plant_ctx=plant_ctx,
        p_target=ee_now.copy(),
        dt_ctrl=0.01,
    )

    print(f"[smoke] tau.shape={tau.shape}  |tau|={np.linalg.norm(tau):.3f}")
    print(f"[smoke] tau per-joint = {tau}")
    print(f"[smoke] diag = {diag}")

    from control.osc.dynamics_helpers import get_gravity
    tau_g = get_gravity(plant, plant_ctx)
    print(f"[smoke] |tau_g[arm]| = {np.linalg.norm(tau_g[:n_arm]):.3f}")
    print(f"[smoke] tau_g[arm] = {tau_g[:n_arm]}")

    ok = (
        tau.shape == (n_arm,)
        and np.all(np.isfinite(tau))
        and diag.get("finished") is True
    )
    print(f"[smoke] tau shape OK={tau.shape == (n_arm,)}  "
          f"tau finite={bool(np.all(np.isfinite(tau)))}  "
          f"finished={diag.get('finished')}")
    print(f"[smoke] VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
