"""Stage C model-plant consistency probe (offline, single-instance).

Asks: does the LCS-with-oracle predict box motion that Drake renders?

Method:
  1. Load the captured ADMM instance (stage_c/admm_dump/seed0_full50.npz).
  2. LCS prediction under the brute-force oracle λ:
        x_next = A·x0 + B·0 + D·λ_oracle + d_const
  3. Drake-side step from the same x0 (box state set from x0; arm at home
     config; integrate 0.05 s with default contact resolution).
  4. Compare box position delta.

Pinning:
  - oracle λ = [γ=0.146, λ_n=0.5839, λ_t={0,0.117,0,0.117}] (brute-force
    enumeration; max|λ·w| = 1.15e-7, complementarity to FP precision).

Findings (committed):
  LCS predicts box falls 17.2 mm in z over 0.05 s; Drake renders the box
  stationary. 1.7e7× mismatch. The captured LCS has n_lambda = 6 = one
  EE-BOX contact only — NO box-ground contact modeled. The brute-force
  oracle is a feasible solution to a LCS that omits the floor — solving
  ANY LCS-consistent λ on this instance, including the oracle, produces
  a fictional next-state.

Run: `python scripts/_stage_c_model_plant_consistency.py` from repo root.
"""
from __future__ import annotations

import numpy as np
import yaml

from pydrake.systems.analysis import Simulator

from sim.env_builder import build_environment

ORACLE_LAMBDA = np.array([0.146119, 0.583936, 0.0, 0.116787, 0.0, 0.116787])
DT_PLANNER = 0.05  # base_mpc.dt — the planner timestep the LCS A,B,D,d are discretized over


def main() -> int:
    print("=" * 64)
    print("STAGE C MODEL-PLANT CONSISTENCY PROBE")
    print("=" * 64)

    d = np.load("stage_c/admm_dump/seed0_full50.npz", allow_pickle=True)
    x0 = d["x0"]
    A = d["A"]; B_ctrl = d["B_ctrl"]; D = d["D"]; d_const = d["d"]
    E = d["E"]
    n_normals = int(d["J_n"].shape[0])
    n_t = int(d["J_t"].shape[0])
    n_lambda = E.shape[0]

    print(f"\nLCS layout:  n_lambda = {n_lambda}")
    print(f"  γ slack: 1 slot   λ_n: {n_normals} slot(s)   λ_t: {n_t} slots")
    print(f"  → ONE contact pair admitted (the EE-BOX pair). No box-ground.")

    # LCS prediction under oracle
    x_next = A @ x0 + B_ctrl @ np.zeros(3) + D @ ORACLE_LAMBDA + d_const
    lcs_delta = x_next[4:7] - x0[4:7]
    lcs_v = x_next[13:16]

    print(f"\nLCS-PREDICTED next-state under oracle (Δt = {DT_PLANNER}s, u = 0):")
    print(f"  Δ box xyz = ({lcs_delta[0]*1000:+.3f}, {lcs_delta[1]*1000:+.3f}, "
          f"{lcs_delta[2]*1000:+.3f}) mm")
    print(f"  box lin v after = ({lcs_v[0]:+.5f}, {lcs_v[1]:+.5f}, {lcs_v[2]:+.5f}) m/s")

    # Drake side: set box state, step 0.05s
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    diagram, plant, panda_model, object_model, *_ = build_environment(
        task_cfg, time_step=0.001)

    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)

    # Arm home (doesn't affect box-floor test; EE is far from box)
    q_arm_home = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    plant.SetPositions(plant_ctx, panda_model, q_arm_home)
    plant.SetVelocities(plant_ctx, panda_model, np.zeros(7))

    # Box from x0
    box_q = np.concatenate([x0[0:4], x0[4:7]])
    plant.SetPositions(plant_ctx, object_model, box_q)
    box_v = np.concatenate([x0[10:13], x0[13:16]])
    plant.SetVelocities(plant_ctx, object_model, box_v)

    box_body = plant.GetBodyByName("box_link", object_model)
    xyz_before = plant.EvalBodyPoseInWorld(plant_ctx, box_body).translation()

    sim = Simulator(diagram, context)
    sim.Initialize()
    sim.AdvanceTo(DT_PLANNER)

    plant_ctx_after = plant.GetMyContextFromRoot(context)
    xyz_after = plant.EvalBodyPoseInWorld(plant_ctx_after, box_body).translation()
    v_after = plant.EvalBodySpatialVelocityInWorld(
        plant_ctx_after, box_body).translational()
    drake_delta = xyz_after - xyz_before

    print(f"\nDRAKE-RENDERED next-state (same x0, Δt = {DT_PLANNER}s):")
    print(f"  Δ box xyz = ({drake_delta[0]*1000:+.3f}, {drake_delta[1]*1000:+.3f}, "
          f"{drake_delta[2]*1000:+.3f}) mm")
    print(f"  box lin v after = ({v_after[0]:+.5f}, {v_after[1]:+.5f}, {v_after[2]:+.5f}) m/s")

    print(f"\n{'='*64}")
    print("VERDICT")
    print('='*64)
    print(f"  z-drop      LCS:  {lcs_delta[2]*1000:+.3f} mm")
    print(f"            Drake:  {drake_delta[2]*1000:+.3f} mm")
    ratio = abs(lcs_delta[2]) / max(abs(drake_delta[2]), 1e-9)
    print(f"  ratio:            {ratio:.0e}×")
    print()
    if ratio > 100:
        print("  → MODEL-BROKEN. The LCS predicts box motion Drake does NOT render.")
        print("    The brute-force oracle is a feasible solution to a LCS that does")
        print("    NOT model the box-ground contact (n_lambda=6 = EE-BOX only).")
        print("    Solving any LCS-consistent λ on this instance, including the")
        print("    oracle, predicts a fictional next-state.")
        print()
        print("  Next gate: contact-model alignment (the LCS_EXPLICIT_BOX_GND knob)")
        print("    + the reference's anitescu reformulation. Reference-settings")
        print("    precondition is MOOT until the LCS matches the plant.")
        return 0
    print("  → MODEL-OK. LCS-with-oracle predicts what Drake renders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
