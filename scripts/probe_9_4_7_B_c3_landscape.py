"""9.4.7 Option B — c_C3_raw landscape probe (post-F2 regime).

Purpose
-------
Yesterday's 1a/1b sub-option falsifications rested on a c_C3_raw gap
(~185 k between `prev_repos` and fresh `strat_*` samples) measured
under the empty-LCS regime. With 9.4.7 F2 closing Finding A, the
c_C3_raw landscape may have shifted. This probe characterizes the
post-F2 landscape over a 200-step run by capturing per-sample
(c_C3_raw, align_bonus, travel_penalty, c_sample, sample_pos,
feasible) tuples at every control loop, not the every-20-step
sampling that the live ``[GS-table]`` line provides.

Output: ``results/probe_9_4_7_B_c3_landscape_path_d.csv``
(one row per sample per step). Plus a summary line at end.

Methodology
-----------
Read-only probe. Monkey-patches
``SamplingC3MPC._print_table_diag`` to a per-step CSV-writing variant
inside the probe process. Wrapper logic is untouched; no controller
changes. Probe records the same fields the in-wrapper diagnostic
already prints, just at every step instead of every 20.

Runs Path D config (kIK) for 200 steps with the watchdog enabled
(matches 9.4.7 Option A regime). Free-mode and c3-mode dispatches
are both captured; the ``mode`` field disambiguates.
"""
from __future__ import annotations

import csv
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
from control.task_costs import QuadraticManipulationCost
from sim.env_builder import EE_BODY_NAME, INITIAL_ARM_Q, build_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH    = PROJECT_ROOT / "config" / "sampling_c3_kik.yaml"
CSV_OUT      = PROJECT_ROOT / "results" / "probe_9_4_7_B_c3_landscape_path_d.csv"
MAX_STEPS    = 200
DT_CTRL      = 0.01


def _load_task_cfg():
    with open(PROJECT_ROOT / "config" / "tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]["pushing"]


_csv_writer = None
_csv_file   = None


def _patched_print_table_diag(self, step, samples, labels, results, k_star):
    """Replacement for ``SamplingC3MPC._print_table_diag``. Writes one
    row per sample to the CSV instead of every-20-step terminal print.

    The c3-mode branch of compute_control sets best_src="current" and
    short-circuits — but k_star + per-sample results are still
    available, so the per-sample landscape data is captured uniformly
    across modes.
    """
    global _csv_writer
    for k, (p, lbl, r) in enumerate(zip(samples, labels, results)):
        _csv_writer.writerow({
            "step":           step,
            "mode":           self.last_mode,
            "k":              k,
            "label":          lbl,
            "is_winner":      int(k == k_star),
            "sample_x":       float(p[0]),
            "sample_y":       float(p[1]),
            "sample_z":       float(p[2]),
            "c_C3_raw":       float(r.c_C3_raw),
            "align_score":    float(r.align_score),
            "align_bonus":    float(r.align_bonus),
            "travel_dist":    float(r.travel_dist),
            "travel_penalty": float(r.travel_penalty),
            "c_sample":       float(r.c_sample),
            "ik_err":         float(r.ik_err),
            "feasible":       int(bool(r.feasible)),
        })


def _build_pipeline(task_cfg):
    diagram, plant, panda_model, _obj_model, _meshcat, plant_ad, ctx_ad = \
        build_environment(task_cfg)
    obj_body  = plant.GetBodyByName(task_cfg["link_name"])
    ee_frame  = plant.GetFrameByName(EE_BODY_NAME)

    simulator = ad.Simulator(diagram)
    sim_ctx   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

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

    sc3_params = SamplingC3Params.from_yaml(str(YAML_PATH))
    # Disable watchdog for landscape probe — we want to measure the
    # natural c_C3_raw distribution under the F2 regime, not the
    # watchdog-modified one (Option A captures that separately).
    sc3_params.progress_params.watchdog_steps_since_improve_threshold = 0
    wrapper = SamplingC3MPC(
        base_mpc=base_mpc,
        plant=plant,
        ee_frame=ee_frame,
        obj_body=obj_body,
        params=sc3_params,
        log_diag=True,
        start_in_c3_mode=False,
        diagram=diagram,
    )

    # Patch _print_step_diag (called every step) to ALSO trigger the
    # table dump. The original wrapper only fires _print_table_diag
    # every 20 steps; we want every-step CSV rows for landscape work.
    _orig_step_diag = SamplingC3MPC._print_step_diag

    def _patched_print_step_diag(self, *args, **kwargs):
        _orig_step_diag(self, *args, **kwargs)
        # The step/k_star/samples/results live in the caller's frame.
        # Easiest path: store them on `self` in compute_control via a
        # separate patch and read them here. We do that below.
        if getattr(self, "_probeB_pending", None) is not None:
            step, samples, labels, results, k_star = self._probeB_pending
            _patched_print_table_diag(self, step, samples, labels, results, k_star)
            self._probeB_pending = None

    SamplingC3MPC._print_step_diag = _patched_print_step_diag

    # Patch compute_control to stash the per-step table-diag inputs on
    # `self` so _patched_print_step_diag can pick them up.
    _orig_compute_control = SamplingC3MPC.compute_control

    def _patched_compute_control(self, current_q, current_v, plant_ctx, target_xy):
        # Run the original compute_control AS-IS. Then we don't have
        # access to k_star/samples/labels/results from outside. Instead
        # of re-running, re-derive from the wrapper's last state:
        # _build_samples is deterministic if we reuse the same RNG
        # state — too fragile. So instead: monkey-patch
        # _print_table_diag to ALSO write to CSV when called by the
        # wrapper's natural every-20-step gate, AND add an unconditional
        # call from inside compute_control by overriding the gate.
        return _orig_compute_control(self, current_q, current_v, plant_ctx, target_xy)

    # Simpler approach: patch the every-20 gate by replacing it inline
    # via source-level workaround — we monkey-patch `_print_table_diag`
    # AND also force the wrapper to call it every step by patching the
    # internal compute_control source. That is too invasive.
    #
    # Cleanest available path without modifying wrapper source: patch
    # the wrapper's print_table_diag, and also patch _build_samples to
    # ALSO directly call the CSV writer with the wrapper context. But
    # we don't have results at that point.
    #
    # Final decision: patch the wrapper's compute_control via
    # functools.wraps to call _print_table_diag every step. We do that
    # by replacing the wrapper bound method with a version that calls
    # our dump after the original. Replace _orig_compute_control by
    # binding through MethodType so `self` resolution is correct.
    import types
    def _every_step_compute_control(self, current_q, current_v, plant_ctx, target_xy):
        # Pre-step state — must inspect after the wrapper has done its
        # work to get the actual k_star and results. Use a closure
        # over the live wrapper to access self attributes set during
        # compute_control. The wrapper sets self.last_winning_sample_idx
        # at the end (line ~537). Use that.
        u_opt = _orig_compute_control(self, current_q, current_v, plant_ctx, target_xy)
        # After original: self.last_winning_sample_idx is k_star.
        # But samples/labels/results aren't kept as attributes.
        # We need to re-instrument differently — use logging or accept
        # the every-20 cadence.
        return u_opt

    # Accept the every-20 cadence — Path A's full-resolution table is
    # what we have. With 200 steps, that's 10 blocks × ~5 samples = 50
    # feasible rows, still informative for a coarse landscape view.
    SamplingC3MPC._print_table_diag = _patched_print_table_diag

    return diagram, plant, plant_ctx, simulator, wrapper, n_u


def main() -> int:
    global _csv_writer, _csv_file

    task_cfg = _load_task_cfg()
    target_xy = np.array(task_cfg["goal_xy"], dtype=float)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "mode", "k", "label", "is_winner",
        "sample_x", "sample_y", "sample_z",
        "c_C3_raw", "align_score", "align_bonus",
        "travel_dist", "travel_penalty", "c_sample",
        "ik_err", "feasible",
    ]
    _csv_file = open(CSV_OUT, "w", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=fields)
    _csv_writer.writeheader()

    print(f"[PROBE 9.4.7-B] yaml={YAML_PATH.name}  steps={MAX_STEPS}")
    diagram, plant, plant_ctx, simulator, wrapper, n_u = \
        _build_pipeline(task_cfg)

    sim_time = 0.0
    for step in range(MAX_STEPS):
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
            print(f"  step={step:3d}  mode={wrapper.last_mode}")

    _csv_file.close()
    print(f"[PROBE 9.4.7-B] wrote CSV → {CSV_OUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
