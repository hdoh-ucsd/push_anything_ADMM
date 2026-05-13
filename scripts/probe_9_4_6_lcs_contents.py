"""9.4.6 — LCS-contents probe (Finding A direct investigation).

Yesterday's 1d watchdog (9.4.5-F, reverted) observed n_λ=0 in every
forced c3-mode dispatch on Path D. This probe dumps the LCS contents
at each ``LCSFormulator.extract_lcs_contacts`` call during a short
WEST-task run, classifying the empty-LCS mechanism as one of:

  Class A — Drake finds no pairs at all at the wrapper-commanded EE
            position (the configured 0.10 m signed-distance threshold
            is too tight given how far the EE sits from the box).
  Class B — Drake returns pairs but the project filter (pusher⇄box only,
            see ``LCSFormulator._manipuland_geom_ids`` /
            ``_ee_geom_ids``) excludes all of them.
  Class C — Mixed.

Methodology
-----------
Read-only probe. Monkey-patches
``LCSFormulator.extract_lcs_contacts`` to record one CSV row per call
with:

  step, ctx_label, q_arm[:7], ee_pos, obj_xy,
  ee_to_box_dist,
  n_raw_at_0p10, n_raw_at_0p20, n_raw_at_0p50,
  n_filtered_at_0p10  (the n_c the QP actually sees),
  n_ee_box_pairs_at_0p50, min_ee_box_dist_at_0p50

``ctx_label`` is "k=0", "k=1", ... when the call originates from
``InnerSolver.evaluate_sample`` (set by a second monkey-patch), and
"c3mpc" when ``C3MPC.compute_control`` is the caller (rich-mode
delegation from the wrapper). All other intermediate state on the
formulator is untouched.

Output: ``results/probe_9_4_6_lcs_contents_west.csv``.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pydrake.all as ad
import yaml

from control.admm_solver import C3Solver
from control.ci_mpc_c3 import C3MPC
from control.lcs_formulator import LCSFormulator
from control.sampling_c3 import SamplingC3MPC, SamplingC3Params
from control.sampling_c3.inner_solve import InnerSolver
from control.task_costs import QuadraticManipulationCost
from sim.env_builder import EE_BODY_NAME, INITIAL_ARM_Q, build_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Run-config (matches `python main.py pushing --task-id 4 --sampling-c3`)
# ---------------------------------------------------------------------------

TASK_NAME       = "pushing"
DIRECTIONAL_ID  = 4              # WEST — Path D
SAMPLING_YAML   = "config/sampling_c3_params.yaml"
MAX_STEPS       = 100            # ~1 s of sim (dt_ctrl=0.01)
DT_CTRL         = 0.01
CSV_OUT         = PROJECT_ROOT / "results" / "probe_9_4_6_lcs_contents_west.csv"


# ---------------------------------------------------------------------------
# Globals — set by the monkey-patches
# ---------------------------------------------------------------------------

_step_counter:    int            = 0
_current_ctx:     str            = "init"
_csv_rows:        list           = []


def _classify_pair(inspector, sdp, ee_ids: set, box_ids: set) -> str:
    a_ee  = sdp.id_A in ee_ids
    a_box = sdp.id_A in box_ids
    b_ee  = sdp.id_B in ee_ids
    b_box = sdp.id_B in box_ids
    if (a_ee and b_box) or (a_box and b_ee):
        return "ee_box"
    if a_ee or b_ee:
        return "ee_other"
    if a_box or b_box:
        return "box_other"
    return "other_other"


def _patched_extract_lcs_contacts(self, context, distance_threshold: float = 0.10):
    """Wrapping replacement for ``LCSFormulator.extract_lcs_contacts``.

    Calls the original logic verbatim (no behavior change for the live
    controller). Additionally:
      - runs the Drake query at 0.20 m and 0.50 m to characterize what
        the threshold-extension envelope would expose
      - logs one row to ``_csv_rows``
    """
    plant = self.plant
    # Geometry summary at the call's plant_ctx
    q   = plant.GetPositions(context)
    arm_q = q[:plant.num_actuators()]
    # Box position — we know obj_body is welded with floating base; use
    # the geom-id set to pull the box body and compute its world frame.
    # Simpler: locate via the manipuland geom id set.
    query_obj = plant.get_geometry_query_input_port().Eval(context)
    inspector = query_obj.inspector()

    # EE world position
    ee_body  = plant.GetBodyByName("pusher")
    ee_pose  = plant.EvalBodyPoseInWorld(context, ee_body)
    ee_pos   = np.array(ee_pose.translation())

    # Box body — pull from the geom-id set (one body for the manipuland).
    box_body = None
    for gid in self._manipuland_geom_ids:
        fid = inspector.GetFrameId(gid)
        box_body = plant.GetBodyFromFrameId(fid)
        break
    if box_body is not None:
        box_pose = plant.EvalBodyPoseInWorld(context, box_body)
        obj_xy   = np.array(box_pose.translation())[:2]
    else:
        obj_xy   = np.array([np.nan, np.nan])

    ee_to_box = float(np.linalg.norm(ee_pos[:2] - obj_xy))

    # Multi-threshold Drake sweep (READ ONLY — does not affect the call).
    raw_counts:         dict = {}
    ee_box_counts:      dict = {}
    min_ee_box_dists:   dict = {}
    for thr in (0.10, 0.20, 0.50):
        pairs = query_obj.ComputeSignedDistancePairwiseClosestPoints(thr)
        raw_counts[thr] = len(pairs)
        ee_box = [p for p in pairs
                  if _classify_pair(inspector, p,
                                     self._ee_geom_ids,
                                     self._manipuland_geom_ids) == "ee_box"]
        ee_box_counts[thr]    = len(ee_box)
        min_ee_box_dists[thr] = (min(p.distance for p in ee_box)
                                  if ee_box else float("nan"))

    # Filtered count at the live threshold (0.10) — replicate the
    # project's filter rule to compute what the QP will see.
    raw_010 = query_obj.ComputeSignedDistancePairwiseClosestPoints(0.10)
    pair_classes = [
        _classify_pair(inspector, p,
                        self._ee_geom_ids, self._manipuland_geom_ids)
        for p in raw_010
    ]
    n_filtered_010 = sum(1 for c in pair_classes if c == "ee_box")

    # Class label for THIS call:
    #   A  — Drake returns 0 pairs at any threshold ≤ 0.10
    #   B  — Drake returns pairs at 0.10 but none are ee_box
    #   ok — at least one ee_box pair at 0.10
    if raw_counts[0.10] == 0:
        class_label = "A"
    elif n_filtered_010 == 0:
        class_label = "B"
    else:
        class_label = "ok"

    _csv_rows.append({
        "step":                    _step_counter,
        "ctx":                     _current_ctx,
        "ee_x":                    ee_pos[0],
        "ee_y":                    ee_pos[1],
        "ee_z":                    ee_pos[2],
        "obj_x":                   obj_xy[0],
        "obj_y":                   obj_xy[1],
        "ee_to_box_xy":            ee_to_box,
        "n_raw_t0p10":             raw_counts[0.10],
        "n_raw_t0p20":             raw_counts[0.20],
        "n_raw_t0p50":             raw_counts[0.50],
        "n_filtered_t0p10":        n_filtered_010,
        "n_ee_box_pairs_t0p20":    ee_box_counts[0.20],
        "n_ee_box_pairs_t0p50":    ee_box_counts[0.50],
        "min_ee_box_dist_t0p50":   min_ee_box_dists[0.50],
        "pair_class_counts_t0p10": json.dumps({
            c: pair_classes.count(c)
            for c in set(pair_classes)
        }),
        "class":                   class_label,
        "arm_q":                   json.dumps(np.round(arm_q, 4).tolist()),
    })

    # Fall through to original behavior — this is the read-only part of
    # the contract. We rebuild the original logic locally rather than
    # binding the unwrapped method, to keep the patch self-contained.
    return _orig_extract_lcs_contacts(self, context, distance_threshold)


# Bound at install time (see ``main()``).
_orig_extract_lcs_contacts = None


def _patched_evaluate_sample(self, *args, **kwargs):
    """Wraps ``InnerSolver.evaluate_sample`` to label the call context.

    The sample index isn't exposed in this method's signature; we
    derive it from ``is_current_ee``. The wrapper's evaluate_samples
    loop sets ``is_current_ee=(k==0)``, so any non-current call is
    k>=1. We collapse those to ``"k>=1"`` because their EE positions
    are recorded individually via sample_pos anyway.
    """
    global _current_ctx
    prior = _current_ctx
    is_current_ee = kwargs.get("is_current_ee", False)
    if not is_current_ee and len(args) >= 7:
        # Positional fallback isn't used by the wrapper, but be defensive.
        is_current_ee = bool(args[6])
    _current_ctx = "k=0" if is_current_ee else "k>=1"
    try:
        return _orig_evaluate_sample(self, *args, **kwargs)
    finally:
        _current_ctx = prior


_orig_evaluate_sample = None


def _patched_compute_control(self, *args, **kwargs):
    """Wraps ``C3MPC.compute_control`` so c3-mode dispatches are labeled."""
    global _current_ctx
    prior = _current_ctx
    _current_ctx = "c3mpc"
    try:
        return _orig_compute_control(self, *args, **kwargs)
    finally:
        _current_ctx = prior


_orig_compute_control = None


# ---------------------------------------------------------------------------
# Probe driver
# ---------------------------------------------------------------------------

def _load_task_cfg() -> dict:
    with open(PROJECT_ROOT / "config/tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"][TASK_NAME]


def _install_patches() -> None:
    global _orig_extract_lcs_contacts
    global _orig_evaluate_sample
    global _orig_compute_control
    _orig_extract_lcs_contacts = LCSFormulator.extract_lcs_contacts
    _orig_evaluate_sample      = InnerSolver.evaluate_sample
    _orig_compute_control      = C3MPC.compute_control
    LCSFormulator.extract_lcs_contacts = _patched_extract_lcs_contacts
    InnerSolver.evaluate_sample         = _patched_evaluate_sample
    C3MPC.compute_control               = _patched_compute_control


def _build_controller(task_cfg):
    """Replicates the controller pipeline from main.py — pushing+task-id 4+
    sampling-C3."""
    diagram, plant, panda_model, _obj_model, _meshcat, plant_ad, ctx_ad = \
        build_environment(task_cfg)

    obj_body  = plant.GetBodyByName(task_cfg["link_name"])
    ee_frame  = plant.GetFrameByName(EE_BODY_NAME)

    simulator = ad.Simulator(diagram)
    sim_ctx   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    # Set initial state — default pose (no --prepositioned).
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"])
    )
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    n_x = n_q + n_v

    formulator = LCSFormulator(plant, mu=task_cfg["friction"], obj_body=obj_body,
                                plant_ad=plant_ad, context_ad=ctx_ad)
    solver     = C3Solver(n_x=n_x, n_u=n_u, rho=100.0,
                           math_diag=False, mode="c3")
    quad_cost  = QuadraticManipulationCost(
        plant, EE_BODY_NAME, obj_body, task_cfg["cost"], n_x, n_u,
        math_diag=False, cost_bias=False,
    )
    base_mpc = C3MPC(
        formulator=formulator, solver=solver, quadratic_cost=quad_cost,
        horizon=20, dt=0.05, torque_limit=30.0,
        admm_iter=3, math_diag=False,
    )

    sc3_params = SamplingC3Params.from_yaml(str(PROJECT_ROOT / SAMPLING_YAML))
    wrapper = SamplingC3MPC(
        base_mpc=base_mpc,
        plant=plant,
        ee_frame=ee_frame,
        obj_body=obj_body,
        params=sc3_params,
        log_diag=False,             # quiet — we have CSV
        start_in_c3_mode=False,
        diagram=diagram,
    )

    return diagram, plant, plant_ctx, simulator, wrapper, obj_body, n_u


def _step_classification_summary(rows: list[dict]) -> dict:
    """Per-call class tally."""
    tally = {"A": 0, "B": 0, "ok": 0}
    for r in rows:
        tally[r["class"]] = tally.get(r["class"], 0) + 1
    return tally


def main() -> int:
    print(f"[PROBE 9.4.6] WEST-task LCS-contents probe")
    print(f"[PROBE 9.4.6] yaml={SAMPLING_YAML}  max_steps={MAX_STEPS}")

    task_cfg = _load_task_cfg()
    # Apply directional task-id 4 override (WEST). Mirrors main.py:298-305.
    dir_path = PROJECT_ROOT / "config" / "directional_tasks.json"
    with open(dir_path) as f:
        dir_cfg = json.load(f)
    task_cfg["goal_xy"] = dir_cfg["tasks"][str(DIRECTIONAL_ID)]["goal"]
    print(f"[PROBE 9.4.6] task=WEST goal={task_cfg['goal_xy']}")

    _install_patches()

    diagram, plant, plant_ctx, simulator, wrapper, obj_body, n_u = \
        _build_controller(task_cfg)
    target_xy = np.array(task_cfg["goal_xy"], dtype=float)

    global _step_counter
    sim_time = 0.0
    for step in range(MAX_STEPS):
        _step_counter = step
        current_q = plant.GetPositions(plant_ctx)
        current_v = plant.GetVelocities(plant_ctx)

        tau_g = plant.CalcGravityGeneralizedForces(plant_ctx)
        u_opt = wrapper.compute_control(current_q, current_v, plant_ctx, target_xy)

        if wrapper.last_mode == "free":
            plant.get_actuation_input_port().FixValue(plant_ctx, u_opt)
        else:
            total = tau_g[:n_u] + u_opt
            plant.get_actuation_input_port().FixValue(plant_ctx, total)

        sim_time += DT_CTRL
        simulator.AdvanceTo(sim_time)

        if step % 20 == 0:
            print(f"  step={step:3d}  mode={wrapper.last_mode}  "
                  f"sim_t={sim_time:.2f}s")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(_csv_rows[0].keys()) if _csv_rows else []
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in _csv_rows:
            w.writerow(row)
    print(f"[PROBE 9.4.6] wrote {len(_csv_rows)} rows → {CSV_OUT}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    tally = _step_classification_summary(_csv_rows)
    print(f"[PROBE 9.4.6] call-class tally (over all extract_lcs_contacts "
          f"invocations): {tally}")

    # Per-context tally
    ctx_class: dict = {}
    for r in _csv_rows:
        key = (r["ctx"], r["class"])
        ctx_class[key] = ctx_class.get(key, 0) + 1
    print(f"[PROBE 9.4.6] per-context class tally:")
    for (ctx, cls), n in sorted(ctx_class.items()):
        print(f"  ctx={ctx:7s} class={cls:2s}  n={n}")

    # Min EE-to-box distance witnessed and Drake-threshold relationship
    ee_to_box = np.array([r["ee_to_box_xy"] for r in _csv_rows])
    print(f"[PROBE 9.4.6] EE-to-box xy distance (m): "
          f"min={ee_to_box.min():.3f}  median={np.median(ee_to_box):.3f}  "
          f"max={ee_to_box.max():.3f}")

    n_raw_010 = np.array([r["n_raw_t0p10"] for r in _csv_rows])
    n_raw_020 = np.array([r["n_raw_t0p20"] for r in _csv_rows])
    n_raw_050 = np.array([r["n_raw_t0p50"] for r in _csv_rows])
    print(f"[PROBE 9.4.6] n_pairs at threshold 0.10 — "
          f"min={n_raw_010.min()} max={n_raw_010.max()} "
          f"frac_zero={(n_raw_010 == 0).mean():.3f}")
    print(f"[PROBE 9.4.6] n_pairs at threshold 0.20 — "
          f"min={n_raw_020.min()} max={n_raw_020.max()} "
          f"frac_zero={(n_raw_020 == 0).mean():.3f}")
    print(f"[PROBE 9.4.6] n_pairs at threshold 0.50 — "
          f"min={n_raw_050.min()} max={n_raw_050.max()} "
          f"frac_zero={(n_raw_050 == 0).mean():.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
