"""
Inner-solve adapter — evaluates the C3 trajectory cost for one sampled
end-effector location.

This is the (LCS, x_init, q_seed) → (plan, scalar_cost) interface
referenced in the prompt, factored out of the legacy
GlobalSamplingC3MPC.compute_control() loop (lines 419–516 of the old
wrapper) so it can be reused by the new mode-switch dispatcher and
unit-tested independently.

Inputs: an existing C3MPC's components (LCSFormulator, C3Solver,
QuadraticManipulationCost) — none are modified.

Per sample the adapter:
  1. (k=0 only) uses current_q directly; (k>0) seeds q via iterated DLS IK.
  2. Sets plant_ctx to (q_seed, current_v); linearises the LCS.
  3. Captures contact normals (formulator._last_nhats) for the alignment bonus.
  4. Builds Q, R, QN, x_ref via QuadraticManipulationCost.build(...).
  5. Calls C3Solver.solve(...) — `surrogate_admm_iters` for k>0, full iters for k=0.
  6. Computes c_C3_raw = Σ x^T Q x + Σ u^T R u + terminal (lifted from
     legacy `_traj_cost` verbatim).
  7. Computes align_score = max(0, n_hat_i · g_hat_3d) over contacts (i=0
     when there are no contacts).
  8. Returns SampleResult with c_sample = c_C3_raw - w_align*align - w_travel*0
     plus + w_travel * Cartesian travel distance from the current EE.
  9. RESTORES plant_ctx to (current_q, current_v) before returning, so
     downstream consumers (the next sample, FK calls in the wrapper) see
     the original state.

stdout from linearize_discrete + solve is suppressed for k>0 so the
hypothetical evaluations don't pollute the diagnostic stream. The k=0
sample emits its normal diagnostics (which become the visible "rich
plan" output when the wrapper delegates to base_mpc).
"""
from __future__ import annotations

import copy
import io
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pydrake.all as ad

from control.admm_solver import C3Solver
from control.lcs_formulator import LCSFormulator
from control.sampling_c3.ik import ik_seed_one_step, solve_ik_to_ee_pos
from control.sampling_c3.params import SamplingC3Params
from control.solver_api import CandidateSemantics


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Drake-convention quaternion (w,x,y,z) → yaw about world z, in [-π, π]."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def _wrap_to_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s*(qy*qy + qz*qz), s*(qx*qy - qw*qz),       s*(qx*qz + qw*qy)      ],
        [s*(qx*qy + qw*qz),       1.0 - s*(qx*qx + qz*qz), s*(qy*qz - qw*qx)      ],
        [s*(qx*qz - qw*qy),       s*(qy*qz + qw*qx),       1.0 - s*(qx*qx + qy*qy)],
    ])


def _predicted_box_contact(sample_pos:    np.ndarray,
                           p_box_w:       np.ndarray,
                           box_quat_wxyz: np.ndarray,
                           box_half:      float):
    """Predict the (p_contact_w, nhat_onto_box_w) for a sample whose EE
    would press against a cube of side 2*box_half centered at p_box_w
    with orientation box_quat_wxyz=(w,x,y,z).

    Geometry-only (no LCS admission needed). Sample → box body frame,
    find dominant side face (±x_b or ±y_b), clamp to face surface in
    body frame, transform back to world. Inward normal = -outward face
    normal. Returns None if the sample is below the box (top/bottom face
    dominant) — irrelevant for side-pushing.
    """
    R = _quat_to_rot(float(box_quat_wxyz[0]), float(box_quat_wxyz[1]),
                     float(box_quat_wxyz[2]), float(box_quat_wxyz[3]))
    p_rel_w = np.asarray(sample_pos, dtype=float) - np.asarray(p_box_w, dtype=float)
    p_b     = R.T @ p_rel_w
    # Only side faces (±x_b, ±y_b) matter for pushing; ignore top/bottom.
    if abs(p_b[0]) < 1e-9 and abs(p_b[1]) < 1e-9:
        return None
    if abs(p_b[0]) >= abs(p_b[1]):
        i = 0
    else:
        i = 1
    sign = 1.0 if p_b[i] >= 0.0 else -1.0
    # Body-frame contact: clamp dominant axis to ±h, tangential coord
    # to [-h, h], z to [-h, h].
    p_c_b = np.array([
        max(-box_half, min(box_half, p_b[0])),
        max(-box_half, min(box_half, p_b[1])),
        max(-box_half, min(box_half, p_b[2])),
    ])
    p_c_b[i] = sign * box_half
    n_outward_b = np.zeros(3)
    n_outward_b[i] = sign
    p_c_w = R @ p_c_b + np.asarray(p_box_w, dtype=float)
    n_onto_box_w = -(R @ n_outward_b)
    return p_c_w, n_onto_box_w


# ---------------------------------------------------------------------------
# Per-sample evaluation result
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    """Everything the wrapper needs about one evaluated sample."""

    # Inputs
    sample_pos:       np.ndarray              # (3,) Cartesian EE target
    is_current_ee:    bool

    # IK
    q_seed:           np.ndarray              # (n_q,) seeded joint config
    ee_pos_resolved:  np.ndarray              # (3,) FK at q_seed
    ik_err:           float
    ik_iters:         int

    # Inner C3 solve
    feasible:         bool
    c_C3_raw:         float                   # Σ x^T Q x + Σ u^T R u + terminal
    align_score:      float                   # max(0, n_hat · g_hat_3d)
    align_bonus:      float                   # w_align * align_score
    travel_dist:      float                   # ||sample_pos - ee_pos_now||
    travel_penalty:   float                   # w_travel * travel_dist
    # Rotation-aware bonus (layer 2): rewards off-center contacts whose moment
    # turns the box toward goal_yaw. Inert when w_rot=0 or task has w_yaw=0.
    rot_score:        float                   # max(0, max_i M_z_i * yaw_sign)
    rot_bonus:        float                   # w_rot * rot_score
    c_sample:         float                   # ranked cost (lower is better)

    # C3 plan output (None when infeasible)
    u_seq:            Optional[np.ndarray]    # (N, n_u)
    x_seq:            Optional[np.ndarray]    # (N+1, n_x)

    # LCS matrices — kept for an optional re-solve at full ADMM iters
    # after the winner is selected (legacy wrapper does this on lines 649-677)
    A:                Optional[np.ndarray]    = None
    B:                Optional[np.ndarray]    = None
    D:                Optional[np.ndarray]    = None
    d:                Optional[np.ndarray]    = None
    J_n:              Optional[np.ndarray]    = None
    J_t:              Optional[np.ndarray]    = None
    phi:              Optional[np.ndarray]    = None
    mu:               Optional[float]         = None

    # Cost-breakdown components for [GS-table] diagnostic
    Q:                Optional[np.ndarray]    = None
    R:                Optional[np.ndarray]    = None
    QN:               Optional[np.ndarray]    = None
    x_ref:            Optional[np.ndarray]    = None
    x0:               Optional[np.ndarray]    = None

    # Contact normals captured at this sample's IK-resolved config
    nhats:            list                    = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trajectory cost (lifted verbatim from legacy global_sampling_c3.py:287-303)
# ---------------------------------------------------------------------------

def traj_cost(x_seq:  np.ndarray,
              u_seq:  np.ndarray,
              Q:      np.ndarray,
              R:      np.ndarray,
              QN:     np.ndarray,
              x_ref:  np.ndarray) -> float:
    """Σ_{t=0}^{N-1} (x_t - x_ref)^T Q (x_t - x_ref) + u_t^T R u_t
       + (x_N - x_ref)^T QN (x_N - x_ref)
    """
    N = len(u_seq)
    J = 0.0
    for t in range(N):
        e  = x_seq[t] - x_ref
        J += float(e @ Q @ e + u_seq[t] @ R @ u_seq[t])
    e_N = x_seq[N] - x_ref
    J  += float(e_N @ QN @ e_N)
    return J


# Shapes whose sample ranking uses the reference cost-LCS forward-sim path.
# Audited 2026-08-18: "box" is DELIBERATELY excluded even though
# sampling_c3_kik.yaml sets use_cost_lcs_ranking=true — the box banked its
# 72% closure on the w_ee_approach ranking and regressed to ~39% under the
# object-only path (§9-leak memory), so the shape gate, not the yaml key, is
# what protects it. "jack" added 2026-08-18: reference jacktoy ranks with
# CalcCost forward-sim like every non-box demo (cost_type 3, richer
# resolve_contacts_to_for_cost [0,3,6]); before this the jack's yaml request
# was silently ignored and its samples were ranked on the planner's own plan.
# The canonical Block-T declares tshape via sampling_c3_kik_t.yaml and takes
# this path; H declares hshape.
_COST_LCS_RANKING_SHAPES = ("tshape", "hshape", "jack")


def _object_only_cost_matrices_ee_space(Q, QN, R):
    """Return copies of (Q, QN, R) with robot pos/vel/torque entries zeroed —
    reference's C3CostComputationType::kSimImpedanceObjectCostOnly semantics
    (sampling_based_c3_controller.cc:601-609). For the port's EE-space layout
    x = [box_q (7), p_ee (3), box_v (6), v_ee (3)]:
      - Q[7:10, 7:10]   = 0  (p_ee — was w_ee_approach)
      - Q[16:19, 16:19] = 0  (v_ee)
      - R                = 0
    QN mirrors Q's zeroing. Object dims (box_q, box_v) retain their weights.
    """
    Q_obj  = Q.copy()
    QN_obj = QN.copy()
    R_obj  = np.zeros_like(R)
    Q_obj[7:10, 7:10] = 0.0
    Q_obj[16:19, 16:19] = 0.0
    QN_obj[7:10, 7:10] = 0.0
    QN_obj[16:19, 16:19] = 0.0
    return Q_obj, QN_obj, R_obj


def traj_cost_breakdown(x_seq, u_seq, Q, R, QN, x_ref,
                        n_arm_dofs: int,
                        obj_x_idx:  int,
                        obj_y_idx:  int,
                        obj_z_idx:  int,
                        obj_ps:     int) -> dict:
    """Per-term breakdown for the [GS-table] diagnostic line. Lifted from
    legacy _traj_cost_breakdown."""
    N = len(u_seq)
    obj_xy  = 0.0
    obj_z   = 0.0
    box_rp  = 0.0
    ee_app  = 0.0
    torque  = 0.0
    for t in range(N):
        e = x_seq[t] - x_ref
        obj_xy += Q[obj_x_idx, obj_x_idx] * e[obj_x_idx] ** 2 \
                + Q[obj_y_idx, obj_y_idx] * e[obj_y_idx] ** 2
        obj_z  += Q[obj_z_idx, obj_z_idx] * e[obj_z_idx] ** 2
        box_rp += Q[obj_ps + 1, obj_ps + 1] * e[obj_ps + 1] ** 2 \
                + Q[obj_ps + 2, obj_ps + 2] * e[obj_ps + 2] ** 2
        e_arm   = e[:n_arm_dofs]
        ee_app += float(e_arm @ Q[:n_arm_dofs, :n_arm_dofs] @ e_arm)
        torque += float(u_seq[t] @ R @ u_seq[t])
    e_N      = x_seq[N] - x_ref
    terminal = float(e_N @ QN @ e_N)
    return dict(obj_xy_term=obj_xy, obj_z_term=obj_z, box_rp_term=box_rp,
                ee_approach=ee_app, torque=torque, terminal=terminal)


# ---------------------------------------------------------------------------
# Worker plant contexts for parallel sample evaluation
# ---------------------------------------------------------------------------

def _assert_worker_clone_covers_ctor_args(parent, child, synced_attrs,
                                          per_worker=()):
    """Fail loudly if a worker clone could silently diverge from its parent.

    Worker kits rebuild the LCSFormulator and C3Solver from a HAND-WRITTEN
    argument list, then patch an allowlist of attributes on top. Both lists
    are easy to forget to update. In 2026-08-21 four LCSFormulator ctor args
    (`controller_object_mass`, `controller_inertia`, `tshape_mesh_witnesses`,
    `mesh_ground_witnesses_body`) were never forwarded, so workers evaluated
    samples against a different physics model than the parent -- identical
    contact geometry, different mass/inertia, ~2.3x different costs, and no
    error anywhere.

    This checks the property that actually matters: for every constructor
    parameter, the child ends up with the same value as the parent. It runs
    once per pool build, not per sample, so the cost is irrelevant.
    """
    import inspect

    import numpy as _np

    mismatched = []
    for name in inspect.signature(type(parent).__init__).parameters:
        if name == "self" or name in per_worker:
            # `per_worker` args are SUPPOSED to differ -- each worker owns
            # its own autodiff context, which is the point of the pool.
            continue
        attr = "_" + name
        if not hasattr(parent, attr):
            attr = name
            if not hasattr(parent, attr):
                continue            # ctor arg isn't stored under either name
        p_val, c_val = getattr(parent, attr), getattr(child, attr, None)
        try:
            same = bool(_np.all(p_val == c_val)) if not isinstance(
                p_val, dict) else all(
                    _np.all(p_val[k] == c_val[k]) for k in p_val
                ) if isinstance(c_val, dict) and set(p_val) == set(c_val) else False
        except Exception:
            same = p_val is c_val
        if not same:
            mismatched.append((name, attr))
    if mismatched:
        raise RuntimeError(
            f"worker clone of {type(parent).__name__} diverges from its "
            f"parent on constructor-derived state: "
            f"{[m[0] for m in mismatched]}. Forward it in "
            f"_lazy_init_worker_kits or add the attribute to the sync list "
            f"({sorted(synced_attrs)}).")


def make_worker_plant_context(plant, diagram):
    """Return a plant Context whose `geometry_query` port is CONNECTED.

    `plant.CreateDefaultContext()` returns a STANDALONE LeafContext. The
    plant lives inside a Diagram next to a SceneGraph, and every collision
    or signed-distance query the LCS build performs reads the plant's
    `geometry_query` input port -- which is only wired up for a plant
    subcontext extracted from a DIAGRAM context. A standalone context
    therefore raises

        InputPort::Eval(): required InputPort[0] (geometry_query) of
        System ::_::plant (MultibodyPlant<double>) is not connected

    on the FIRST geometry query. That was the 2026-08-21 parallel-path
    defect: every worker raised this, `evaluate_sample`'s bare
    `except Exception: pass` swallowed it, the worker returned a
    SampleResult with J_n=None and a NaN cost, and the dispatcher saw
    best_other=nan on every tick -- so it never repositioned, never
    entered c3, and the run failed while looking 2x faster because it did
    no work. See tests/test_parallel_worker_context.py.

    Note a plant subcontext cannot simply be cloned: Drake rejects
    `Context::Clone()` on a non-root context. The root has to be created
    (or cloned) and the subcontext re-extracted from it.
    """
    if diagram is None:
        raise RuntimeError(
            "parallel sample evaluation needs the diagram to build worker "
            "plant contexts (geometry_query must be connected); got "
            "diagram=None. Pass diagram= to InnerSolver, or run serially.")
    return plant.GetMyContextFromRoot(diagram.CreateDefaultContext())


# ---------------------------------------------------------------------------
# InnerSolver
# ---------------------------------------------------------------------------

class InnerSolver:
    """Per-sample C3 evaluation, wrapping the existing C3MPC components.

    Construction takes references to the inner C3 stack — they're only
    READ from, never modified, matching the prompt's "C3MPC class is kept
    unchanged" constraint.
    """

    def __init__(self,
                 plant,
                 ee_frame,
                 obj_body,
                 formulator,
                 solver,
                 quad_cost,
                 horizon:        int,
                 dt:             float,
                 torque_limit:   float,
                 base_admm_iter: int,
                 params:         SamplingC3Params,
                 dt_pose:        Optional[float] = None,
                 diagram=None):
        # Needed ONLY for parallel sample evaluation: worker plant contexts
        # must be extracted from a diagram context so `geometry_query` is
        # connected. See make_worker_plant_context. Serial evaluation uses
        # the caller's plant_ctx and does not touch this.
        self._diagram    = diagram
        self.plant       = plant
        self.world_frame = plant.world_frame()
        self.ee_frame    = ee_frame
        self.obj_body    = obj_body
        self.formulator  = formulator
        self.solver      = solver
        self.quad_cost   = quad_cost
        self.horizon       = int(horizon)
        self.dt            = float(dt)
        # Reference-conformant dt swap near-goal. When
        # `quad_cost._crossed_switching_threshold` is True, per-sample cost-LCS
        # builds use dt_pose (finer resolution, matches reference
        # `GetLCSFactoryOptions(crossed_)` semantics at
        # sampling_based_c3_controller.cc:2537-2553). Wrapper syncs the flag
        # onto quad_cost each tick (sampling_based_c3_controller.py:1581).
        # Defaults to `dt` (no swap) for backward compat.
        self.dt_pose       = float(dt_pose) if dt_pose is not None else float(dt)
        self.torque_limit  = float(torque_limit)
        self.base_admm_iter   = int(base_admm_iter)
        self.surrogate_iter   = int(params.surrogate_admm_iters)
        self.w_align          = float(params.w_align)

        # 2026-07-26 arc-2 E2 surrogate-side x_pred clamp (partial fix).
        # ci_mpc_c3plus.py:381-397 already clamps x0's EE pos/vel to
        # (x_pred ± nominal_ee_accel·dt²) for the PRIMARY planner solve.
        # This mirror applies the same clamp to the k=0 surrogate solve's
        # `p_ee_for_x0` so cost-signal for mode-switch matches the primary
        # solve's linearization point. DEFAULT since the 2026-07-28
        # defaults flip (was REFCONF_C3_SURROGATE_XPRED_CLAMP env gate).
        # Set each tick by SamplingC3Controller.compute_control() via
        # `set_ee_pos_clamp()`.
        self._x_pred_ee_pos: Optional[np.ndarray] = None
        self._x_pred_ee_delta_pos: float = 0.0
        self._surrogate_xpred_clamp_enabled = True

        # Diag 2 fix — per-axis force bounds for EE-space surrogate solves.
        # ci_mpc_c3plus.py:310-321 reads these env vars for the MAIN planner's
        # solve() call so the QP installs a per-axis [±U_H, ±U_H, ±U_V] box
        # instead of the scalar torque_limit uniform cap. But InnerSolver's
        # per-sample surrogate solves at self.solver.solve(...) (lines below)
        # bypass ci_mpc_c3plus.py and were passing u_lower=u_upper=None →
        # admm_solver.py:1056 falls back to np.full(n_u, ±torque_limit) →
        # uniform scalar cap (30 N in all 3 axes). Mirror the same read here
        # so surrogate solves see the same per-axis box the main solve does.
        import os as _os_env
        self._u_lo = None
        self._u_hi = None
        _uh_s = _os_env.environ.get("PORT_U_HORIZONTAL", "")
        _uv_s = _os_env.environ.get("PORT_U_VERTICAL", "")
        if _uh_s and _uv_s:
            try:
                _uh = float(_uh_s)
                _uv = float(_uv_s)
                self._u_lo = np.array([-_uh, -_uh, -_uv])
                self._u_hi = np.array([+_uh, +_uh, +_uv])
            except ValueError:
                pass
        self.w_travel         = float(params.w_travel)
        self.w_rot            = float(getattr(params, "w_rot", 0.0))
        # §9 Option B (Stage 2) — cost-LCS forward-sim ranking
        self._use_cost_lcs_ranking = bool(getattr(
            params, "use_cost_lcs_ranking", False))
        # Cost-LCS contact resolution (reference resolve_contacts_to_for_cost,
        # [n_EE_ground, n_EE_object, n_object_ground]). The reference builds
        # the cost LCS with a DIFFERENT and generally RICHER contact set than
        # the planner's -- for jacktoy, plan [0,1,3] vs cost [0,3,6], i.e. the
        # ranking rollout carries all six jack tips and all three capsules
        # while the planner keeps only the closest of each. That is how the
        # reference tolerates a planner that cannot see which tripod is
        # active: the model that RANKS samples does not select at all.
        #
        # None => historical port behaviour (2 EE-object, planner's ground
        # count), which is exactly push_t's [0,2,3].
        _rc = getattr(params, "resolve_contacts_to_for_cost", None)
        if _rc is not None and len(_rc) >= 3:
            self._cost_n_ee  = max(1, int(_rc[1]))
            self._cost_n_gnd = int(_rc[2])
            if int(_rc[0]) != 0:
                print(f"[COST-LCS] WARNING resolve_contacts_to_for_cost[0]="
                      f"{int(_rc[0])} (EE-ground) is not modelled by the port; "
                      f"ignored.", flush=True)
        else:
            self._cost_n_ee, self._cost_n_gnd = 2, None
        print(f"[COST-LCS] contact resolution: {self._cost_n_ee} EE-object + "
              f"{self._cost_n_gnd if self._cost_n_gnd is not None else 'planner'}"
              f" object-ground", flush=True)
        # Scalar or per-axis [x, y, z]; normalized to a (3,) array so the
        # PD rollout applies per-axis gains (anything-N1: Kp [100,100,50]).
        self._Kp_ee_pd_rollout = np.broadcast_to(np.asarray(getattr(
            params, "Kp_for_ee_pd_rollout", 100.0), dtype=float), (3,)).copy()
        self._Kd_ee_pd_rollout = np.broadcast_to(np.asarray(getattr(
            params, "Kd_for_ee_pd_rollout", 0.5), dtype=float), (3,)).copy()
        # Reference builds the cost-LCS at N·res knots and dt/res
        # (sampling_based_c3_controller.cc:1658-1659, push_t
        # sampling_c3plus_options.yaml `lcs_dt_resolution: 4`), then ZOHs the
        # coarse plan onto that fine grid, rolls out, and downsamples. The
        # refinement is load-bearing: at the coarse planning dt the PD rollout
        # has rho(A_cl) = 2.61 (dt=0.05) / 16.41 (dt=0.1) and diverges; at
        # dt/4 it is 0.94 / 0.88 and is stable.
        self._lcs_dt_resolution = max(1, int(getattr(
            params, "lcs_dt_resolution", 4)))
        self._pgs_max_iter = int(getattr(params, "cost_lcs_pgs_max_iter", 50))
        self._pgs_tol      = float(getattr(params, "cost_lcs_pgs_tol", 1.0e-6))
        self._pgs_reg      = float(getattr(params, "cost_lcs_pgs_reg", 1.0e-8))
        # Reference progress_params `cost_type` (C3CostComputationType) —
        # which weights score the ranking rollout. 5 (object-only) is the
        # push_t/anything/H reference value and the historical port
        # behaviour; jacktoy uses 3 (full Q/R). See the selection site below.
        self._cost_lcs_cost_type = int(getattr(params, "cost_type", 5))
        if self._cost_lcs_cost_type not in (3, 5):
            print(f"[COST-LCS] WARNING cost_type={self._cost_lcs_cost_type} "
                  f"unsupported (only 3=kSimImpedance, 5=ObjectCostOnly); "
                  f"using 5.", flush=True)
            self._cost_lcs_cost_type = 5
        print(f"[COST-LCS] ranking cost_type={self._cost_lcs_cost_type} "
              f"({'kSimImpedance/full-QR' if self._cost_lcs_cost_type == 3 else 'kSimImpedanceObjectCostOnly'})",
              flush=True)
        # §9-leak gate: object-only ranking cost + cost-LCS path apply ONLY to
        # tshape (reference-faithful for push_t). Box path keeps the pre-§9
        # w_ee_approach-weighted ranking (72 % closure banked at b23fa82;
        # HEAD without this gate regressed to ~39 %).
        self._object_shape = str(getattr(
            params.sampling_params, "object_shape", "box"))
        self._box_half_extent = float(params.sampling_params.box_half_extent)

        self.n_u = plant.num_actuators()
        self.n_q = plant.num_positions()
        self.n_v = plant.num_velocities()

        # Object position indices (Drake floating-body: qw,qx,qy,qz,x,y,z)
        ps = obj_body.floating_positions_start()
        self._obj_x_idx = ps + 4
        self._obj_y_idx = ps + 5
        self._obj_z_idx = ps + 6
        self._obj_ps    = ps

        # Solve-count perf counters
        self.full_solves:  int = 0
        self.cheap_solves: int = 0

        # ---- Parallel sample eval (port-todo #1) -------------------------
        # Reference `num_outer_threads` (sampling_c3plus_options.yaml:6).
        # `_num_threads_to_use` = requested pool size; 1 = serial. Env var
        # PORT_NUM_THREADS_TO_USE overrides the YAML for A/B smoke tests
        # without reloading configs.
        _env_nt = os.environ.get("PORT_NUM_THREADS_TO_USE", "").strip()
        if _env_nt:
            try:
                self._num_threads_to_use = max(1, int(_env_nt))
            except ValueError:
                self._num_threads_to_use = int(
                    getattr(params, "num_threads_to_use", 1))
        else:
            self._num_threads_to_use = int(
                getattr(params, "num_threads_to_use", 1))
        # Pool state: `_worker_kits` = list of (InnerSolver-clone, plant_ctx);
        # `_worker_queue` = Queue passing kits between threads.
        # Lazily populated on first parallel dispatch to avoid paying
        # per-worker LCSFormulator init cost when the caller is single-thread.
        self._worker_kits:  list = []
        self._worker_queue: Optional[queue.Queue] = None
        self._worker_lock:  threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # E2 surrogate-side clamp setter — called each tick before
    # evaluate_samples() by SamplingC3Controller so the k=0 surrogate solve
    # can clamp its x0.p_ee to (x_pred_ee ± delta_pos). Pass None to disable
    # for a single tick.
    # ------------------------------------------------------------------

    def set_ee_pos_clamp(self,
                         x_pred_ee_pos: Optional[np.ndarray],
                         delta_pos:     float) -> None:
        self._x_pred_ee_pos = (
            np.asarray(x_pred_ee_pos, dtype=float).copy()
            if x_pred_ee_pos is not None else None)
        self._x_pred_ee_delta_pos = float(delta_pos)

    # ------------------------------------------------------------------
    # Single-sample evaluation
    # ------------------------------------------------------------------

    def evaluate_sample(self,
                        sample_pos:    np.ndarray,
                        current_q:     np.ndarray,
                        current_v:     np.ndarray,
                        plant_ctx,
                        target_xy:     np.ndarray,
                        ee_pos_now:    np.ndarray,
                        g_hat_3d:      np.ndarray,
                        is_current_ee: bool = False,
                        full_iters:    bool = False,
                        suppress_io:   bool = True,
                        target_yaw:    float = 0.0) -> SampleResult:
        """Evaluate one sample. Restores plant_ctx to (current_q, current_v)
        before returning.

        EE-space dispatch (when solver is constructed at n_x=19): the
        "sample" is just an EE position, so no IK is needed for non-current
        samples — x0[7:10] is set to sample_pos directly. The LCS is
        linearized at the current Drake state (plant context unchanged),
        which means contact admission reflects the CURRENT EE position
        rather than the hypothetical sample. This is the simplest fix that
        gets c_samples finite; a follow-up could compute sample-specific
        contact admission analytically (sphere-vs-box SDF) or via IK.
        """
        # Dispatch indicator: EE-space planner has n_x=19, n_u=3.
        # When that's the active solver, route through the EE-space path.
        _ee_space = (self.solver.n_x == 19 and self.solver.n_u == 3)

        # --- IK / state setup --------------------------------------------
        if is_current_ee:
            q_seed   = current_q.copy()
            ik_err   = 0.0
            ik_iters = 0
            self.plant.SetPositions(plant_ctx, q_seed)
            self.plant.SetVelocities(plant_ctx, current_v)
        elif _ee_space:
            # Per-sample LCS linearization: port of reference behavior at
            # sampling_based_c3_controller.cc:1628-1644 (CreateLCSObjectsForSamples).
            # Reference calls UpdateContext(plant, candidate_states[i]) BEFORE
            # GenerateLCS() for each sample so each LCS reflects the hypothetical
            # EE position's contact geometry (phi, J_n). For repos samples at
            # sampling_setback=0.030m, phi > 0.002m → no EE-BOX pair admitted →
            # higher C3 cost → curr_cost < best_other → kToC3Cost fires.
            #
            # Implementation: solve IK to place the pusher sphere at sample_pos,
            # then set plant context to that arm config before linearize_discrete_ee_space.
            # On IK failure, fall back to current_q (degrades to previous behavior
            # for that sample). Note: x0 still uses sample_pos directly (line ~444)
            # regardless of IK accuracy; only the contact geometry (phi, J_n) changes.
            q_warm = ik_seed_one_step(self.plant, self.ee_frame,
                                       current_q, sample_pos, plant_ctx,
                                       n_arm_dofs=self.n_u)
            try:
                q_seed, ik_err, ik_iters = solve_ik_to_ee_pos(
                    self.plant, self.ee_frame,
                    p_target=sample_pos, q_init=q_warm,
                    plant_ctx=plant_ctx, n_arm_dofs=self.n_u,
                )
            except Exception:
                # IK failed — fall back to current arm config so that
                # this sample's LCS uses current contact geometry (same
                # as the previous behavior). A fallback is safe since the
                # cost for this sample will be inflated by the shared-LCS
                # bias only; the entry-gate will still fire on later ticks
                # once IK succeeds for the arriving EE position.
                q_seed   = current_q.copy()
                ik_err   = float("inf")
                ik_iters = 0
            self.plant.SetPositions(plant_ctx, q_seed)
            self.plant.SetVelocities(plant_ctx, current_v)
        else:
            q_warm = ik_seed_one_step(self.plant, self.ee_frame,
                                       current_q, sample_pos, plant_ctx,
                                       n_arm_dofs=self.n_u)
            q_seed, ik_err, ik_iters = solve_ik_to_ee_pos(
                self.plant, self.ee_frame,
                p_target=sample_pos, q_init=q_warm,
                plant_ctx=plant_ctx, n_arm_dofs=self.n_u,
            )
            self.plant.SetVelocities(plant_ctx, current_v)

        # FK current EE at the IK-resolved (or current) config — used as a
        # "where did the planner think we are?" reference downstream.
        ee_pos_resolved = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten().copy()

        # Reference cc:971-1085 runs every sample at the same admm_iter
        # inside the OpenMP parallel loop. 2026-07-26 arc-2 E1 fix — port
        # was giving k=0 the full 25-iter budget and k≥1 samples only
        # 1 iter, so under G-on the k=0 solve accumulated leakage across
        # 25 iterations while surrogates stayed near their init, biasing
        # the argmin over c_samples.
        admm_iter_k = self.base_admm_iter

        feasible = False
        nhats: list = []
        ee_box_contacts: list = []   # (p_W, n_W) tuples, EE-BOX only
        c_C3_raw = float("inf")
        u_seq = x_seq = None
        A = B = D = d = J_n = J_t = phi = mu = None
        Q = R = QN = x_ref = x0 = None
        _cost_lcs_probe = None       # populated only when cost-LCS ranking active

        # Reference-conformant dt swap: `GetLCSFactoryOptions(crossed_)`
        # at cc:2537-2553 returns `planning_dt_pose` when the flag is True.
        # quad_cost._crossed_switching_threshold is set each tick by
        # wrapper.py:1581. When flag is False (position regime or attribute
        # absent), self.dt is used — bit-identical to prior behavior.
        _dt_effective = (self.dt_pose
                         if getattr(self.quad_cost,
                                    "_crossed_switching_threshold", False)
                         else self.dt)

        _buf = io.StringIO()
        ctx = redirect_stdout(_buf) if suppress_io else _NullContext()
        try:
            with ctx:
                if _ee_space:
                    # EE-space dispatch — paper-aligned (Push-Anything §IV-A).
                    (A, B, D, d,
                     E_lcs, F_lcs, H_lcs, c_lcs,
                     J_n, J_t, phi, mu) = \
                        self.formulator.linearize_discrete_ee_space(
                            plant_ctx, _dt_effective)
                    nhats = list(self.formulator._last_nhats)
                    ee_box_contacts = list(
                        getattr(self.formulator, "_last_ee_box_contacts", [])
                    )
                    Q, R, QN, x_ref = self.quad_cost.build_ee_space(
                        target_xy, plant_ctx=plant_ctx,
                        current_q=q_seed, target_yaw=target_yaw,
                    )
                    # x0 = [box_q (7), p_ee_sample (3), box_v (6), v_ee=0 (3)]
                    BOX_Q_START = self.obj_body.floating_positions_start()
                    BOX_V_START = self.obj_body.floating_velocities_start_in_v()
                    box_q = current_q[BOX_Q_START : BOX_Q_START + 7]
                    box_v = current_v[BOX_V_START : BOX_V_START + 6]
                    # For is_current_ee: use FK on current arm config (matches
                    # base_mpc's x0 construction). For non-current: use sample_pos.
                    if is_current_ee:
                        p_ee_for_x0 = ee_pos_resolved
                        # E2 surrogate-side clamp: mirror primary planner
                        # (ci_mpc_c3plus.py:381-397). Bounds the k=0 surrogate's
                        # x0 EE-pos to `x_pred_ee ± delta_pos` so cost-signal
                        # driving mode-switch matches the primary linearization.
                        # Set by SamplingC3Controller via `set_ee_pos_clamp()`.
                        if (self._surrogate_xpred_clamp_enabled
                                and self._x_pred_ee_pos is not None
                                and self._x_pred_ee_delta_pos > 0.0):
                            _dpos = float(self._x_pred_ee_delta_pos)
                            p_ee_for_x0 = np.clip(
                                np.asarray(self._x_pred_ee_pos, dtype=float),
                                p_ee_for_x0 - _dpos,
                                p_ee_for_x0 + _dpos,
                            )
                    else:
                        p_ee_for_x0 = np.asarray(sample_pos, dtype=float).reshape(3)
                    v_ee_for_x0 = np.zeros(3)   # hypothetical: EE starts at rest
                    x0 = np.concatenate([box_q, p_ee_for_x0, box_v, v_ee_for_x0])
                else:
                    # R^7 path (legacy).
                    (A, B, D, d,
                     E_lcs, F_lcs, H_lcs, c_lcs,
                     J_n, J_t, phi, mu) = \
                        self.formulator.linearize_discrete(plant_ctx, _dt_effective)
                    nhats = list(self.formulator._last_nhats)
                    ee_box_contacts = list(
                        getattr(self.formulator, "_last_ee_box_contacts", [])
                    )
                    Q, R, QN, x_ref = self.quad_cost.build(
                        target_xy, plant_ctx=plant_ctx, current_q=q_seed,
                        target_yaw=target_yaw,
                    )
                    x0 = np.concatenate([q_seed, current_v])

                # Diag 2: EE-space per-axis bounds (installed only for R^3
                # planner with n_u=3; None for R^7 arm-torque). Mirrors
                # ci_mpc_c3plus.py:310-321 for the surrogate-solve path.
                # F2 fix (2026-07-28b deep report): the former tshape-only
                # gate (§9-leak protection for the pre-frame-migration 72%
                # box baseline, now retired) left BOX surrogates with NO
                # per-axis u-box — rollouts applied up to 49 N fictitious
                # force and predicted full goal attainment from ANY sample
                # face, flattening the ranking. The reference installs
                # u_horizontal/vertical_limits on EVERY per-sample solve
                # (sampling_based_c3_controller.cc:1040-1053); do the same
                # for all EE-space surrogates.
                _ee_space = (self.solver.n_u == 3)
                # Polygonal (non-box) manipulands share the cost-LCS ranking
                # path: both T and H are concave outlines whose ranking needs
                # the object-only forward-sim, unlike the convex box.
                _tshape_gate = _ee_space and \
                    self._object_shape in _COST_LCS_RANKING_SHAPES
                _u_lo = self._u_lo if _ee_space else None
                _u_hi = self._u_hi if _ee_space else None
                if not _ee_space:
                    # R^7: gravity-centered u-box at the SAMPLE's arm config
                    # (plant_ctx currently holds q_seed). Same computation as
                    # ci_mpc_c3plus's main-solve box — without it, surrogate
                    # solves fall back to the symmetric ±torque_limit box
                    # that cannot even contain the gravity-holding torque
                    # (−34 Nm on joint 2 vs ±30).
                    _u_lo, _u_hi = self._r7_gravity_centered_ubox(plant_ctx)
                # §7.67 — plumb _ee_box_pair_idx per-surrogate. Without this,
                # the shared C3Solver instance uses whatever index the main
                # planner set at the previous tick (or None on tick 0), so
                # B1-A's final-iter G-weighting either skips or lands on the
                # wrong pair — surrogate's ADMM under-solves the EE-BOX λ_n
                # for its OWN LCS, making c_C3_raw non-informative for
                # ranking. Same scan as ci_mpc_c3plus.py:328-335.
                if _tshape_gate:
                    _ee_box_idx = None
                    _cinfo_s = getattr(self.formulator, "_last_contact_info", None)
                    if _cinfo_s:
                        for _i_s, _info_s in enumerate(_cinfo_s):
                            if _info_s.get("tag", "") == "EE-BOX":
                                _ee_box_idx = _i_s
                                break
                    self.solver._ee_box_pair_idx = _ee_box_idx
                u_seq, x_seq = self.solver.solve(
                    x0, A, B, D, d, J_n, J_t, mu,
                    Q, R, QN, x_ref,
                    N=self.horizon,
                    admm_iter=admm_iter_k,
                    torque_limit=self.torque_limit,
                    phi=phi,
                    E=E_lcs, F=F_lcs, H=H_lcs, c_lcs=c_lcs,
                    u_lower=_u_lo, u_upper=_u_hi,
                )
            # §9 Option B (Stage 1): reference-faithful ranking cost.
            # Reference (sampling_based_c3_controller.cc:601-609, cost_type=5
            # kSimImpedanceObjectCostOnly) evaluates the sample cost with
            # robot pos/vel/torque entries of Q and R zeroed — only object
            # tracking errors count. This decouples ranking from the port's
            # w_ee_approach (which favors samples parked at the setback
            # target with the arm in position, biasing dispatcher toward
            # reposition every tick even when the c3 sample can actually
            # push).
            #
            # STAGE 1 caveat: we use the planner's own x_seq (kUseC3Plan
            # variant) rather than SimulatePDControlWithLCS(...) on a
            # separate 5-pair cost-LCS. The full forward-sim (Stage 2) is
            # left for a follow-up if Stage 1 doesn't unblock c3 dispatch.
            if _ee_space and self._object_shape in _COST_LCS_RANKING_SHAPES:
                # Reference cost_type selects the weights on the rolled-out
                # trajectory (progress_params C3CostComputationType):
                #   5 = kSimImpedanceObjectCostOnly — robot pos/vel Q-blocks
                #       and R zeroed (push_t/anything/H: cost_type 5 in BOTH
                #       regimes, progress_params_c3plus.yaml:18-19).
                #   3 = kSimImpedance — FULL Q and R on the simulated states
                #       and controls (jacktoy pose regime,
                #       jacktoy/parameters/progress_params_c3plus.yaml:18).
                # The jacktoy position-regime variant (cost_type_position=2,
                # kSimLCSReplaceC3EEPlan open-loop rollout) is not
                # implemented; jack goals sit inside the 0.5 m cost-switching
                # threshold from step 1, so the pose regime is the live one.
                if self._cost_lcs_cost_type == 3:
                    Q_obj, QN_obj, R_obj = Q, QN, R
                else:
                    Q_obj, QN_obj, R_obj = _object_only_cost_matrices_ee_space(
                        Q, QN, R)
                # §9 Option B (Stage 2): forward-simulate the plan on the LCS
                # via PD-with-feedforward + PGS LCP per knot, then score the
                # SIMULATED trajectory (kSimImpedanceObjectCostOnly).
                # Reference: sampling_based_c3_controller.cc:571-590 + 601-609.
                # When use_cost_lcs_ranking=False, fall back to Stage-1
                # (planner's own x_seq).
                if self._use_cost_lcs_ranking:
                    from control.sampling_c3.lcs_simulator import (
                        simulate_pd_control_with_lcs)
                    # §9 Option B (faithful 5-pair cost-LCS): build a
                    # SEPARATE LCS with top-2 EE-manipuland pairs (reference
                    # push_t resolve_contacts_to_for_cost=[0,2,3] → 0
                    # EE-ground + 2 EE-T + 3 T-GND = 5 pairs). With 1 EE-T
                    # the forward-sim couldn't distinguish productive-face
                    # (east) from dead-face (north) samples on the T; the
                    # 2nd EE-T gives the sim two contact modes to resolve
                    # between via LCP. force_top_k_ee_box=True forces the
                    # top-K EE-manipuland injection unconditionally (bypasses
                    # the 2 mm auto-admit / always-on gates) — mirrors the
                    # reference's GetResolvedContactPairs.
                    #
                    # Delta-1 gap fix (per-sample plant-context update): the
                    # reference's UpdateContext(plant, current_v,
                    # candidate_states[i]) — sampling_based_c3_controller.cc:
                    # 1628-1631 — places the plant at each sample's EE state
                    # BEFORE the cost-LCS build. The port previously skipped
                    # this in the EE-space path (line 371-378 above), so the
                    # cost-LCS was linearized at CURRENT arm config and its
                    # contact geometry was extrapolated ~15 cm to reach an
                    # east-face sample — the wrong contact model. v10 evidence
                    # (pathT_smoke_v10_5pair/CRUX_ANALYSIS.md) showed east
                    # sample's forward-sim moved T 0.018 m in the WRONG
                    # direction, scoring worse than north's near-zero motion.
                    # This block solves per-sample IK, temporarily sets the
                    # plant to the sample's arm config, builds the cost-LCS
                    # at that state, and restores the plant afterward.
                    _q_saved_for_cost_lcs = np.array(
                        self.plant.GetPositions(plant_ctx), copy=True)
                    _v_saved_for_cost_lcs = np.array(
                        self.plant.GetVelocities(plant_ctx), copy=True)
                    _cost_lcs_ik_err   = 0.0
                    _cost_lcs_ik_iters = 0
                    if not is_current_ee:
                        try:
                            _q_warm_cost = ik_seed_one_step(
                                self.plant, self.ee_frame,
                                _q_saved_for_cost_lcs, sample_pos, plant_ctx,
                                n_arm_dofs=self.n_u,
                            )
                            _q_sample_arm, _cost_lcs_ik_err, \
                                _cost_lcs_ik_iters = solve_ik_to_ee_pos(
                                    self.plant, self.ee_frame,
                                    p_target=sample_pos,
                                    q_init=_q_warm_cost,
                                    plant_ctx=plant_ctx,
                                    n_arm_dofs=self.n_u,
                                )
                            # Keep box q/v from current — only arm moves.
                            _q_for_cost_lcs = _q_saved_for_cost_lcs.copy()
                            _q_for_cost_lcs[:self.n_u] = \
                                _q_sample_arm[:self.n_u]
                            self.plant.SetPositions(
                                plant_ctx, _q_for_cost_lcs)
                            self.plant.SetVelocities(
                                plant_ctx, _v_saved_for_cost_lcs)
                        except Exception:
                            # If per-sample IK fails, fall back to current
                            # arm config (matches v10 behavior).
                            self.plant.SetPositions(
                                plant_ctx, _q_saved_for_cost_lcs)
                            self.plant.SetVelocities(
                                plant_ctx, _v_saved_for_cost_lcs)
                    # The cost LCS may use a different ground-contact count
                    # than the planner (reference resolve_contacts_to_for_cost).
                    # The formulator holds that count as state, so swap it for
                    # the build and restore in the finally below -- leaking the
                    # cost value into the planner would silently change the
                    # planning LCS dimension.
                    _gnd_saved = self.formulator\
                        .lcs_explicit_manipuland_ground_contacts
                    try:
                        if self._cost_n_gnd is not None:
                            self.formulator\
                                .lcs_explicit_manipuland_ground_contacts = \
                                self._cost_n_gnd
                        (A_c, B_c, D_c, d_c,
                         E_c, F_c, H_c, c_lcs_c,
                         J_n_c, J_t_c, phi_c, mu_c) = \
                            self.formulator.linearize_discrete_ee_space(
                                plant_ctx,
                                _dt_effective / self._lcs_dt_resolution,
                                n_ee_top_k=self._cost_n_ee,
                                force_top_k_ee_box=True)
                        # Cost-LCS admission audit for the crux instrumentation
                        # (task 2): number of EE-manipuland rows in the actual
                        # cost-LCS, and each EE-row's phi at build time.
                        _cost_cinfo = list(getattr(
                            self.formulator, "_last_contact_info", []))
                        _n_ee_t_cost = sum(
                            1 for info in _cost_cinfo
                            if info.get("tag", "") == "EE-BOX")
                        _ee_t_phi_cost = [
                            float(phi_c[i])
                            for i, info in enumerate(_cost_cinfo)
                            if info.get("tag", "") == "EE-BOX"]
                        # Fine cost-LCS built at dt/res → ZOH the coarse plan
                        # onto it and downsample the rollout.
                        _cost_lcs_rate = self._lcs_dt_resolution
                    except Exception:
                        # Fall back to planner LCS if the top-2 build fails.
                        # That LCS is at the COARSE dt, so no upsampling.
                        A_c, B_c, D_c, d_c = A, B, D, d
                        E_c, F_c, H_c, c_lcs_c = E_lcs, F_lcs, H_lcs, c_lcs
                        _n_ee_t_cost = -1        # signals cost-LCS build failed
                        _ee_t_phi_cost = []
                        _cost_lcs_rate = 1
                    finally:
                        self.formulator\
                            .lcs_explicit_manipuland_ground_contacts = _gnd_saved
                    # Restore plant to the state expected by downstream
                    # callers (matches the pre-existing R^7 path's convention
                    # of leaving plant_ctx at current_q/current_v).
                    self.plant.SetPositions(
                        plant_ctx, _q_saved_for_cost_lcs)
                    self.plant.SetVelocities(
                        plant_ctx, _v_saved_for_cost_lcs)
                    XX_sim, UU_sim = simulate_pd_control_with_lcs(
                        x_plan=x_seq, u_plan=u_seq,
                        A=A_c, B=B_c, D=D_c, d=d_c,
                        E=E_c, F=F_c, H=H_c, c_lcs=c_lcs_c,
                        Kp_ee=self._Kp_ee_pd_rollout,
                        Kd_ee=self._Kd_ee_pd_rollout,
                        x0_override=x0,
                        lcp_max_iter=self._pgs_max_iter,
                        lcp_tol=self._pgs_tol,
                        lcp_reg=self._pgs_reg,
                        upsample_rate=_cost_lcs_rate,
                    )
                    c_C3_raw = traj_cost(XX_sim, UU_sim,
                                         Q_obj, R_obj, QN_obj, x_ref)
                    # Stash sim-side motion for the [COST-LCS] trace
                    # (printed after align_score is computed, below).
                    # T motion direction: signed 2-vec end-to-end so we can
                    # verify per-sample linearization redirects east's T
                    # motion toward the goal (west/+y) vs v10's wrong dir.
                    _dT_vec_xy = XX_sim[-1, 4:6] - XX_sim[0, 4:6]
                    _cost_lcs_probe = {
                        "n_ee_t":       _n_ee_t_cost,
                        "ee_t_phi":     _ee_t_phi_cost,
                        "dT_xy":        float(np.linalg.norm(_dT_vec_xy)),
                        "dT_dx":        float(_dT_vec_xy[0]),   # signed +x
                        "dT_dy":        float(_dT_vec_xy[1]),   # signed +y
                        "dEE_xy":       float(np.linalg.norm(
                                            XX_sim[-1, 7:9] - XX_sim[0, 7:9])),
                        "box_v_peak":   float(np.max(np.linalg.norm(
                                            XX_sim[:, 13:16], axis=1))),
                        "ik_err":       float(_cost_lcs_ik_err),
                        "ik_iters":     int(_cost_lcs_ik_iters),
                        # Phantom-split probe (2026-08-11): the PLAN's own
                        # end-to-end EE and object xy displacement. Plan
                        # moving the object while the rollout dT_xy=0 =
                        # phantom-lambda progress exposed by the real LCP.
                        "plan_dEE_xy":  float(np.linalg.norm(
                                            np.asarray(x_seq)[-1, 7:9]
                                            - np.asarray(x_seq)[0, 7:9])),
                        "plan_dT_xy":   float(np.linalg.norm(
                                            np.asarray(x_seq)[-1, 4:6]
                                            - np.asarray(x_seq)[0, 4:6])),
                    }
                else:
                    c_C3_raw = traj_cost(x_seq, u_seq,
                                         Q_obj, R_obj, QN_obj, x_ref)
            else:
                # REFCONF_SAMPLE_RANK_OBJ_ONLY=1 — rank samples by the
                # OBJECT-slot cost only (obj quat/pos q-slots + obj ω/v
                # v-slots; u-term zeroed). p109 diagnosis: under R^7 the
                # raw cost's arm-dependent terms (R·u² over ~45 Nm
                # gravity-holding joint torques, J-mapped ee_pos/ee_vel
                # blocks evaluated at per-sample IK arm configs) inject
                # posture-dependent noise on the same order (~250) as the
                # productive-vs-wrong-face margin (~5% of ~5000), letting a
                # west-face sample outrank the goal-aligned south face.
                # The reference's ranking is object-dominated by
                # construction: its u is a ~5 N EE force (R·u² ≈ 0.25) and
                # its EE slots carry no posture ambiguity. Masking to the
                # object block recovers that property for R^7. The FULL
                # cost still drives the committed solve — this affects
                # ranking only.
                # 2026-07-28 defaults flip: object-slot-only ranking is
                # unconditional (was REFCONF_SAMPLE_RANK_OBJ_ONLY,
                # canonical since a194280).
                #
                # F1 fix (2026-07-28b deep report): the slot arithmetic
                # below assumes the R^7 layout [q_arm(n_u), q_box, v_arm,
                # v_box]. The EE-space layout is [box_q(0:7), p_ee(7:10),
                # box_v(10:16), v_ee(16:19)] — applying the R^7 indices
                # there keeps the EE slots (error ≈ 0 vs their own sample
                # ref) and drops most box weights, flattening every
                # sample's rank to ~0.000 (COST-BD: c_C3_raw=0.000 vs true
                # object cost 2411). EE-space paths use the layout-correct
                # helper instead.
                if _ee_space:
                    Q_obj_r, QN_obj_r, R_obj_r = \
                        _object_only_cost_matrices_ee_space(Q, QN, R)
                    c_C3_raw = traj_cost(
                        x_seq, u_seq, Q_obj_r, R_obj_r, QN_obj_r, x_ref)
                else:
                    _n_x_r = Q.shape[0]
                    _obj_mask = np.zeros(_n_x_r, dtype=bool)
                    _obj_mask[self.n_u:self.n_q] = True          # obj quat+pos
                    _obj_mask[self.n_q + self.n_u:_n_x_r] = True  # obj ω+v
                    _M_r = np.outer(_obj_mask, _obj_mask)
                    c_C3_raw = traj_cost(
                        x_seq, u_seq,
                        Q * _M_r, np.zeros_like(R), QN * _M_r, x_ref)
            feasible = True
            if admm_iter_k >= self.base_admm_iter:
                self.full_solves += 1
            else:
                self.cheap_solves += 1
        except Exception as _evexc:
            # The control flow (swallow -> infeasible sample) is deliberate
            # and unchanged: a single sample that fails to build must not
            # take the run down. What IS new (2026-08-21) is that the
            # failure is no longer SILENT.
            #
            # This swallow hid a 100%-failure bug for an entire run: with
            # threads>1 every worker raised "geometry_query is not
            # connected", every sample came back NaN, and the dispatcher
            # quietly degraded to never repositioning. It had previously
            # hidden a dimension mismatch too (noted in the original
            # comment). Measured cost of speaking up: 0 occurrences over a
            # healthy serial run, 164 over a broken parallel one -- so this
            # is quiet unless something is actually wrong.
            #
            # stderr on purpose: the parallel dispatch wraps its pool in
            # redirect_stdout, which would swallow a stdout print.
            import sys as _sys_ev
            InnerSolver._eval_exc_count = getattr(
                InnerSolver, "_eval_exc_count", 0) + 1
            _n_exc = InnerSolver._eval_exc_count
            if _n_exc == 1 or _n_exc % 500 == 0:
                print(f"[EVAL-EXC] sample evaluation failed "
                      f"(count={_n_exc}, is_current={is_current_ee}): "
                      f"{type(_evexc).__name__}: {_evexc}",
                      file=_sys_ev.stderr, flush=True)
                if os.environ.get("DIAG_EVAL_EXC", ""):
                    import traceback as _tb_ev
                    _tb_ev.print_exc(file=_sys_ev.stderr)
        finally:
            # Exception safety: guarantee plant_ctx is restored to (current_q,
            # current_v) regardless of solver outcome. Required because the
            # per-sample LCS fix (elif _ee_space branch above) now temporarily
            # mutates plant_ctx to the IK-solved config — without this finally,
            # a solver exception would leave plant_ctx in the sample's arm-config
            # state and corrupt all downstream ticks. Matches the reference's
            # "after loop: reset context to x_lcs_curr" (cc:1673-1674).
            self.plant.SetPositions(plant_ctx, current_q)
            self.plant.SetVelocities(plant_ctx, current_v)

        # Geometric align_score: bonus for samples whose contact would push
        # the box in the goal direction. Replaces the prior LCS-admitted-
        # nhats version which was always 0 at the 30 mm setback position
        # (samples sit outside the 2 mm LCS admission threshold, so
        # _last_nhats=[] and align_score=0 dead-by-construction).
        #
        # The contact force on the box from a sample-side approach is along
        # n_onto_box (the inward normal of the contacted face). Aligning that
        # with g_hat (which points from box toward goal) gives the bonus:
        #     align = max(0, n_onto_box · g_hat)
        #
        # Worked check, west goal g_hat=(-1,0,0):
        #   east-face sample  → n_onto_box=(-1, 0, 0) → align = +1.0  ✓ favored
        #   south-face sample → n_onto_box=( 0,+1, 0) → align =  0.0  ✓ unfavored
        #   west-face sample  → n_onto_box=(+1, 0, 0) → align = -1→0  ✓ unfavored
        #
        # Reuses _predicted_box_contact (line 71) for full box-rotation
        # handling. Returns None when sample is above/below the box (top/
        # bottom face dominant) — irrelevant for side-pushing, score 0.
        align_score = 0.0
        _ps = self._obj_ps
        _p_box_w = np.array([
            float(current_q[self._obj_x_idx]),
            float(current_q[self._obj_y_idx]),
            float(current_q[self._obj_z_idx]),
        ])
        _box_quat = np.array([
            float(current_q[_ps + 0]),
            float(current_q[_ps + 1]),
            float(current_q[_ps + 2]),
            float(current_q[_ps + 3]),
        ])
        _pc = _predicted_box_contact(
            sample_pos, _p_box_w, _box_quat, self._box_half_extent,
        )
        if _pc is not None:
            _, _n_onto_box_w = _pc
            align_score = max(0.0, float(np.dot(_n_onto_box_w, g_hat_3d)))
        align_bonus    = self.w_align  * align_score
        travel_dist    = float(np.linalg.norm(sample_pos - ee_pos_now))
        travel_penalty = self.w_travel * travel_dist

        # Task 2 instrumentation — the productive-face distinction test.
        # Emit one line per sample when the cost-LCS forward-sim ran, so
        # the log records what the 5-pair cost-LCS actually saw at build
        # time (n EE-manipuland admissions + their phi) and what the sim
        # produced (dT_xy, box_v_peak). Gated on DIAG_COST_LCS_TRACE=1
        # to preserve the existing sample table when regression-checking.
        if _cost_lcs_probe is not None and \
                os.environ.get("DIAG_COST_LCS_TRACE", ""):
            _phi_str = ",".join(f"{p:+.4f}" for p in
                                _cost_lcs_probe["ee_t_phi"])
            print(f"[COST-LCS] sample_pos="
                  f"({float(sample_pos[0]):+.4f},"
                  f"{float(sample_pos[1]):+.4f},"
                  f"{float(sample_pos[2]):+.4f}) "
                  f"is_current={int(is_current_ee)} "
                  f"n_ee_t={_cost_lcs_probe['n_ee_t']} "
                  f"ee_t_phi=[{_phi_str}] "
                  f"dT_xy={_cost_lcs_probe['dT_xy']:.4f}m "
                  f"dT=(dx={_cost_lcs_probe['dT_dx']:+.4f},"
                  f"dy={_cost_lcs_probe['dT_dy']:+.4f}) "
                  f"box_v_peak={_cost_lcs_probe['box_v_peak']:.4f}m/s "
                  f"dEE_xy={_cost_lcs_probe['dEE_xy']:.4f}m "
                  f"plan_dEE_xy={_cost_lcs_probe.get('plan_dEE_xy', -1):.4f}m "
                  f"plan_dT_xy={_cost_lcs_probe.get('plan_dT_xy', -1):.4f}m "
                  f"align={align_score:.4f} "
                  f"ik_err={_cost_lcs_probe['ik_err']:.4f}m "
                  f"ik_iters={_cost_lcs_probe['ik_iters']} "
                  f"c_C3_sim={c_C3_raw:.2f}",
                  flush=True)

        # --- Layer 2.5: rotation-aware sample bonus (intent-based) ---
        # Reward samples whose PREDICTED EE-BOX contact would produce a
        # moment M_z = (r × n̂_onto_box)·ẑ that turns the box toward
        # goal_yaw. Predicted contact comes from sample EE position +
        # box pose (RECONSTRUCT case): project sample onto nearest face,
        # take that face's inward normal. Decouples the bonus from LCS
        # admission (which never fires for strategy samples sitting
        # ~32mm out at sampling_setback). Moment-arm formula and sign
        # convention are UNCHANGED from layer 2 (27e0727); only the
        # source of (p_contact_w, nhat_onto_box) changed.
        # Gated on quad_cost.w_yaw > 0 so translation-only tasks see no effect.
        rot_score = 0.0
        rot_bonus = 0.0
        if self.w_rot > 0.0 and getattr(self.quad_cost, "w_yaw", 0.0) > 0.0:
            ps = self._obj_ps
            qw = float(current_q[ps + 0]); qx = float(current_q[ps + 1])
            qy = float(current_q[ps + 2]); qz = float(current_q[ps + 3])
            psi_now = _yaw_from_quat(qw, qx, qy, qz)
            yaw_err = _wrap_to_pi(float(target_yaw) - psi_now)
            if abs(yaw_err) > 1e-3:
                yaw_sign = 1.0 if yaw_err > 0.0 else -1.0
                p_box_x = float(current_q[self._obj_x_idx])
                p_box_y = float(current_q[self._obj_y_idx])
                p_box_z = float(current_q[self._obj_z_idx])
                p_box_w = np.array([p_box_x, p_box_y, p_box_z])
                box_quat = np.array([qw, qx, qy, qz])
                pc = _predicted_box_contact(
                    sample_pos, p_box_w, box_quat, self._box_half_extent,
                )
                if pc is not None:
                    p_c_W, n_W = pc
                    rx = float(p_c_W[0]) - p_box_x
                    ry = float(p_c_W[1]) - p_box_y
                    nx = float(n_W[0]); ny = float(n_W[1])
                    m_z = rx * ny - ry * nx          # (r × n̂)·ẑ
                    rot_score = m_z * yaw_sign
                    # Layer 2.6: scale by |yaw_err| so the bonus magnitude
                    # tracks the rotational urgency. L2.5 measured the off-
                    # center sample's c_C3 ~26k vs current ~12k (gap ~14k);
                    # at yaw_err=π/4 and m_z~0.025, w_rot~8e5 yields bonus
                    # ~16k — enough to flip the selection in favor of the
                    # off-center torque-producing sample. As the box rotates
                    # toward goal, |yaw_err| → 0 and the bonus self-attenuates.
                    rot_bonus = self.w_rot * abs(yaw_err) * max(0.0, rot_score)

        c_sample       = c_C3_raw - align_bonus - rot_bonus + travel_penalty

        # Restore plant_ctx to current state for downstream consumers
        self.plant.SetPositions(plant_ctx, current_q)
        self.plant.SetVelocities(plant_ctx, current_v)

        return SampleResult(
            sample_pos      = sample_pos,
            is_current_ee   = is_current_ee,
            q_seed          = q_seed,
            ee_pos_resolved = ee_pos_resolved,
            ik_err          = ik_err,
            ik_iters        = ik_iters,
            feasible        = feasible,
            c_C3_raw        = c_C3_raw,
            align_score     = align_score,
            align_bonus     = align_bonus,
            travel_dist     = travel_dist,
            travel_penalty  = travel_penalty,
            rot_score       = rot_score,
            rot_bonus       = rot_bonus,
            c_sample        = c_sample,
            u_seq           = u_seq,
            x_seq           = x_seq,
            A=A, B=B, D=D, d=d, J_n=J_n, J_t=J_t, phi=phi, mu=mu,
            Q=Q, R=R, QN=QN, x_ref=x_ref, x0=x0,
            nhats=nhats,
        )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def _lazy_init_worker_kits(self, n_workers: int) -> None:
        """Idempotent pool init.  Builds `n_workers` (InnerSolver-clone,
        plant_ctx) kits and (re)populates `_worker_queue`.  Caller must
        hold `_worker_lock`.

        Each kit owns its own `plant_ctx`, `LCSFormulator` (with its own
        autodiff context + `_last_contact_info` cache), and `C3Solver`
        (with its own `_u_prev_solve` + `_last_lambda_*`).  Kits share the
        (thread-safe read-only) `plant`, `plant_ad`, `params`, and (for
        the mutation-idempotent case) the `quad_cost` — reference C++
        similarly shares the `Q_/R_/G_/U_` cost matrices across the
        `#pragma omp parallel for` loop at cc:971.
        """
        if (len(self._worker_kits) == n_workers
                and self._worker_queue is not None):
            # Drain any already-checked-out kits back into the queue for
            # this dispatch.
            self._worker_queue = queue.Queue()
            for kit in self._worker_kits:
                self._worker_queue.put(kit)
            return

        self._worker_kits = []
        self._worker_queue = queue.Queue()

        _formulator_attrs_to_sync = (
            "_always_on_ee_box", "_scale_lcs", "_contact_model",
            "_ref_pair_admission_planner_lcs", "_box_drag_c",
            "_normal_compliance_k", "_normal_velocity_level",
            "_normal_phi_clamp_v_cap", "_ground_z", "_box_half_extents",
            "lcs_explicit_manipuland_ground_contacts",
            # 2026-08-21: these four are LCSFormulator CONSTRUCTOR args that
            # the clone below does not forward, so without syncing them a
            # worker built the LCS with a DIFFERENT PHYSICS MODEL than the
            # parent -- same contact geometry (J_n hashes matched) but
            # different mass/inertia, hence different A/B/D and ~2.3x
            # different sample costs. All four are load-bearing:
            # `_controller_inertia` is one of the three legs of the
            # 2026-08-15 hover root cause, and the witness attrs carry the
            # T-mesh witness-triangle conformance.
            # Synced as PROCESSED attributes, not re-passed to the
            # constructor: `_mesh_ground_witnesses_body` is stored as
            # `.reshape(3,3).T`, so round-tripping it would transpose twice.
            "_controller_object_mass", "_controller_inertia",
            "_tshape_mesh_witnesses", "_mesh_ground_witnesses_body",
        )
        _solver_attrs_to_sync = (
            "_u_lambda", "_u_eta", "_end_on_qp_step", "_rho_scale",
            "_use_g_matrix", "_w_G", "_g_lambda", "_g_eta", "_g_x", "_g_u",
            "_g_x_vector",
            "_w_G_ee_contact",
            # 2026-08-21: the C3+ final-solve contact boost
            # (c3_plus.cc:131-145, 1000 on the box lineage). The controller
            # sets it on base_mpc.solver and its comment claims worker
            # clones "inherit via _solver_attrs_to_sync" -- but it was never
            # actually listed here, so workers ran without the boost.
            "_final_aug_contact_scaling",
            # Workspace position constraint rows (main.py sets this on the
            # base solver for tasks that define a planner workspace). Absent
            # on clones, their QPs were missing those constraints entirely.
            "state_position_bounds",
        )

        for _ in range(n_workers):
            # MUST come from a diagram context: a standalone plant context
            # has `geometry_query` unconnected and every LCS build in the
            # worker raises. See make_worker_plant_context for the full
            # story (this was the 2026-08-21 parallel-path defect).
            ctx_i    = make_worker_plant_context(self.plant, self._diagram)
            # The AUTODIFF context is deliberately NOT given the same
            # treatment: sim/env_builder.py:932-933 builds the shared one
            # with plant_ad.CreateDefaultContext() too, and the serial path
            # works with it, so standalone is correct here.
            ctx_ad_i = self.formulator.plant_ad.CreateDefaultContext()

            formulator_i = LCSFormulator(
                plant           = self.plant,
                mu              = self.formulator.mu,
                obj_body        = self.formulator._obj_body,
                plant_ad        = self.formulator.plant_ad,
                context_ad      = ctx_ad_i,
                object_shape    = self.formulator._object_shape,
                mu_per_pair_type= self.formulator._mu_per_pair_type,
            )
            for _attr in _formulator_attrs_to_sync:
                if hasattr(self.formulator, _attr):
                    setattr(formulator_i, _attr,
                            getattr(self.formulator, _attr))

            solver_i = C3Solver(
                n_x                    = self.solver.n_x,
                n_u                    = self.solver.n_u,
                rho                    = self.solver.rho,
                mode                   = self.solver.mode,
                math_diag              = False,
                penalize_input_change  = self.solver._penalize_input_change,
            )
            for _attr in _solver_attrs_to_sync:
                if hasattr(self.solver, _attr):
                    setattr(solver_i, _attr, getattr(self.solver, _attr))

            # Shallow InnerSolver clone — shares plant/params/quad_cost;
            # swaps in per-worker formulator+solver.  Clones MUST NOT
            # recurse into their own pool (num_threads_to_use = 1) or
            # nested dispatches would deadlock the queue.
            clone = copy.copy(self)
            clone.formulator            = formulator_i
            clone.solver                = solver_i
            clone._num_threads_to_use   = 1
            clone._worker_kits          = []
            clone._worker_queue         = None
            clone._worker_lock          = threading.Lock()

            # Guard against the defect class re-appearing: every
            # LCSFormulator/C3Solver constructor arg must either be
            # forwarded above or synced by name. A new ctor arg that is
            # neither would silently give workers a different model, which
            # is exactly how `controller_inertia` and friends diverged.
            _assert_worker_clone_covers_ctor_args(
                self.formulator, formulator_i, _formulator_attrs_to_sync,
                per_worker=("context_ad",))
            _assert_worker_clone_covers_ctor_args(
                self.solver, solver_i, _solver_attrs_to_sync)

            self._worker_kits.append((clone, ctx_i))
            self._worker_queue.put((clone, ctx_i))

    def evaluate_samples(self,
                         samples:       List[np.ndarray],
                         current_q:     np.ndarray,
                         current_v:     np.ndarray,
                         plant_ctx,
                         target_xy:     np.ndarray,
                         ee_pos_now:    np.ndarray,
                         g_hat_3d:      np.ndarray,
                         use_threading: Optional[bool] = None,
                         target_yaw:    float = 0.0) -> List[SampleResult]:
        """Evaluate every sample.

        Dispatch:
        - Serial when `use_threading` is False, when the pool size is 1,
          or when there is only one sample (nothing to parallelise).
        - Parallel when `self._num_threads_to_use > 1` (or explicit
          `use_threading=True`): k=0 runs serially FIRST (so its full
          diagnostic stream stays visible), then k>=1 dispatch through
          a `ThreadPoolExecutor` fed from `_worker_queue`, matching
          reference `sampling_based_c3_controller.cc:971`
          (`#pragma omp parallel for num_threads(num_threads_to_use_)`).

        The parallel path suppresses stdout globally around the pool
        because `contextlib.redirect_stdout` swaps `sys.stdout`
        process-wide, so per-worker suppression via `suppress_io` would
        race across threads.
        """
        _resolved_threading = (use_threading
                               if use_threading is not None
                               else self._num_threads_to_use > 1)

        # Parallel evaluation needs the diagram to build worker contexts with
        # `geometry_query` connected. Without it every worker LCS build
        # raises, evaluate_sample swallows it, and the samples come back with
        # NaN costs -- which the dispatcher silently treats as "no better
        # sample", failing the whole run while appearing fast. Refuse to run
        # parallel in that state; fall back to serial and say so ONCE.
        if _resolved_threading and self._diagram is None:
            if not getattr(self, "_no_diagram_warned", False):
                self._no_diagram_warned = True
                print("[SAMP-PARALLEL] WARNING: threading requested "
                      f"(num_threads_to_use={self._num_threads_to_use}) but "
                      "InnerSolver has no diagram; worker plant contexts "
                      "would have geometry_query unconnected. Falling back "
                      "to SERIAL evaluation.", flush=True)
            _resolved_threading = False

        # Delta-1 audit (read-only, default-OFF): when DIAG_SAMP_LCS_DUMP=1,
        # emit a [SAMP-LCS] line per sample with sample_idx, sample_pos, the
        # actually-resolved EE pose the LCS was linearized at (ee_pos_resolved),
        # n_c, phi_min, and a J_n hash. Discriminator: if (n_c, phi_min,
        # J_n_hash) is IDENTICAL across samples within one dispatch and
        # ee_pos_resolved is also identical, the port shares ONE LCS at the
        # current EE pose across all samples — the Delta-1 gap. If they
        # differ, the port rebuilds per-sample. No behavior change otherwise.
        import os as _os_d1
        import hashlib as _hl_d1
        _samp_lcs_dump = bool(_os_d1.environ.get("DIAG_SAMP_LCS_DUMP", ""))

        # --- Parallel dispatch -------------------------------------------
        if _resolved_threading and len(samples) > 1:
            results: list[Optional[SampleResult]] = [None] * len(samples)
            # k=0 runs serially and keeps its diagnostic stream.
            results[0] = self.evaluate_sample(
                sample_pos    = samples[0],
                current_q     = current_q,
                current_v     = current_v,
                plant_ctx     = plant_ctx,
                target_xy     = target_xy,
                ee_pos_now    = ee_pos_now,
                g_hat_3d      = g_hat_3d,
                is_current_ee = True,
                full_iters    = True,
                suppress_io   = False,
                target_yaw    = target_yaw,
            )

            n_workers = min(self._num_threads_to_use, len(samples) - 1)
            with self._worker_lock:
                self._lazy_init_worker_kits(n_workers)

            def _run_one(k_idx: int, p_sample):
                clone, ctx = self._worker_queue.get()
                try:
                    self.plant.SetPositions(ctx, current_q)
                    self.plant.SetVelocities(ctx, current_v)
                    r = clone.evaluate_sample(
                        sample_pos    = p_sample,
                        current_q     = current_q,
                        current_v     = current_v,
                        plant_ctx     = ctx,
                        target_xy     = target_xy,
                        ee_pos_now    = ee_pos_now,
                        g_hat_3d      = g_hat_3d,
                        is_current_ee = False,
                        full_iters    = False,
                        suppress_io   = True,
                        target_yaw    = target_yaw,
                    )
                    return k_idx, r
                finally:
                    self._worker_queue.put((clone, ctx))

            _buf = io.StringIO()
            with redirect_stdout(_buf):
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = [pool.submit(_run_one, k, p)
                               for k, p in enumerate(samples[1:], start=1)]
                    for f in as_completed(futures):
                        k_idx, r = f.result()
                        results[k_idx] = r

            if _samp_lcs_dump:
                for k, r in enumerate(results):
                    if r is None:
                        continue
                    if r.J_n is not None:
                        _n_c   = int(r.J_n.shape[0])
                        _phi_m = (float(np.min(r.phi))
                                  if r.phi is not None and r.phi.size > 0
                                  else float("nan"))
                        _jh    = _hl_d1.sha1(r.J_n.tobytes()).hexdigest()[:8]
                        _er    = r.ee_pos_resolved
                        _p     = samples[k]
                        print(f"[SAMP-LCS] sample_idx={k} "
                              f"is_current={int(r.is_current_ee)} "
                              f"sample_pos=({_p[0]:+.4f},{_p[1]:+.4f},"
                              f"{_p[2]:+.4f}) "
                              f"ee_pos_resolved=({_er[0]:+.4f},{_er[1]:+.4f},"
                              f"{_er[2]:+.4f}) "
                              f"n_c={_n_c} phi_min={_phi_m:+.5f} "
                              f"J_n_hash={_jh}", flush=True)
                    else:
                        _p = samples[k]
                        print(f"[SAMP-LCS] sample_idx={k} "
                              f"sample_pos=({_p[0]:+.4f},{_p[1]:+.4f},"
                              f"{_p[2]:+.4f}) "
                              f"n_c=NULL (LCS not built)", flush=True)
            # Arc-2 cost-breakdown dump (parallel path).
            self._eval_call_count = getattr(self, "_eval_call_count", 0) + 1
            import os as _os_bd_p
            _bd_at_p = int(_os_bd_p.environ.get("DIAG_COST_BD_AT_TICK", "0") or "0")
            if _bd_at_p > 0 and self._eval_call_count == _bd_at_p:
                self._dump_cost_breakdown(results, labels=None)
            return results  # type: ignore[return-value]

        # --- Serial path (bit-identical to prior behavior) ---------------
        #
        # CANDIDATE WARM-START SEMANTICS (measurement gate, 2026-08-21).
        # `_u_prev_solve` is written at the end of EVERY C3+ solve and read
        # by the next one via `q_ref[u] += -2*R@u_prev`, so in this serial
        # loop candidate k warm-starts candidate k+1. The loop is therefore
        # ORDER-DEPENDENT, and a fully parallel GPU batch cannot reproduce
        # it by construction. PORT_CANDIDATE_WARMSTART selects:
        #   ordered      (default) -- current behaviour, k warm-starts k+1
        #   independent  -- every candidate sees the tick's ENTRY u_prev
        #   reset        -- every candidate starts from u_prev = None
        # Unset => "ordered" => byte-identical. Measurement only.
        _sem = CandidateSemantics.coerce(
            os.environ.get("PORT_CANDIDATE_WARMSTART", "legacy_ordered"))
        _slv = getattr(self, "solver", None)
        # Captured ONCE, before any candidate is solved. Under
        # INDEPENDENT_BATCH every candidate sees exactly this value, so no
        # candidate can influence another's initialization.
        _u_prev_at_entry = getattr(_slv, "_u_prev_solve", None) \
            if _slv is not None else None
        if (_sem is not CandidateSemantics.LEGACY_ORDERED
                and not getattr(self, "_ws_banner", False)):
            self._ws_banner = True
            _note = ("reproduces the C++ reference (fresh C3 per candidate, "
                     "u_sol_=zeros)"
                     if _sem is CandidateSemantics.REFERENCE_RESET else
                     "one tick-entry u_prev broadcast to every candidate")
            print(f"[CAND-SEMANTICS] mode={_sem.value} — {_note}; "
                  f"candidate-to-candidate propagation is suppressed "
                  f"(default is legacy_ordered)", flush=True)

        # Candidate ORDER sweep (measurement only): PORT_CANDIDATE_ORDER
        # permutes which candidate is solved when, WITHOUT changing which
        # candidates exist or how results are indexed -- results are written
        # back into original positions, so the controller sees an unchanged
        # list. Only the warm-start chain order changes. "as-is" = default.
        _ord_mode = os.environ.get("PORT_CANDIDATE_ORDER",
                                   "as-is").strip().lower()
        _order = list(range(len(samples)))
        if _ord_mode == "reversed":
            _order = _order[::-1]
        elif _ord_mode.startswith("rot"):
            _sh = int(_ord_mode[3:] or 1) % max(len(_order), 1)
            _order = _order[_sh:] + _order[:_sh]
        elif _ord_mode != "as-is":
            raise ValueError(f"PORT_CANDIDATE_ORDER={_ord_mode!r} not in "
                             f"(as-is, reversed, rotN)")

        results_by_k: dict = {}
        for k in _order:
            p = samples[k]
            if _slv is not None:
                if _sem is CandidateSemantics.INDEPENDENT_BATCH:
                    _slv._u_prev_solve = _u_prev_at_entry
                elif _sem is CandidateSemantics.REFERENCE_RESET:
                    _slv._u_prev_solve = None
            r = self.evaluate_sample(
                sample_pos    = p,
                current_q     = current_q,
                current_v     = current_v,
                plant_ctx     = plant_ctx,
                target_xy     = target_xy,
                ee_pos_now    = ee_pos_now,
                g_hat_3d      = g_hat_3d,
                is_current_ee = (k == 0),
                full_iters    = (k == 0),
                suppress_io   = (k != 0),   # k=0 is the "real" diagnostic stream
                target_yaw    = target_yaw,
            )
            if _samp_lcs_dump:
                if r.J_n is not None:
                    _n_c   = int(r.J_n.shape[0])
                    _phi_m = (float(np.min(r.phi))
                              if r.phi is not None and r.phi.size > 0
                              else float("nan"))
                    _jh    = _hl_d1.sha1(r.J_n.tobytes()).hexdigest()[:8]
                    _er    = r.ee_pos_resolved
                    print(f"[SAMP-LCS] sample_idx={k} "
                          f"is_current={int(r.is_current_ee)} "
                          f"sample_pos=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
                          f"ee_pos_resolved=({_er[0]:+.4f},{_er[1]:+.4f},"
                          f"{_er[2]:+.4f}) "
                          f"n_c={_n_c} phi_min={_phi_m:+.5f} "
                          f"J_n_hash={_jh}", flush=True)
                else:
                    print(f"[SAMP-LCS] sample_idx={k} "
                          f"sample_pos=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
                          f"n_c=NULL (LCS not built)", flush=True)
            results_by_k[k] = r

        # Restore ORIGINAL candidate order regardless of solve order, so the
        # controller's positional indexing (mode switch, prev-repos
        # inflation) is unaffected by the PORT_CANDIDATE_ORDER sweep.
        results: list[SampleResult] = [results_by_k[k]
                                       for k in range(len(samples))]

        # 2026-07-26 arc-2 diagnostic: per-sample cost-breakdown dump.
        # Purpose: understand why surrogate c_C3_raw for the current sample
        # (in-contact hypothesis) inverts against prev_repos (no-contact
        # hypothesis) under G-on + arm-Cartesian LCS. Env-gated one-shot at
        # a specific evaluate_samples call.
        # Usage: DIAG_COST_BD_AT_TICK=<N> — dumps on N-th call, then never
        # again. Set N to a tick number well past reposition-entry.
        self._eval_call_count = getattr(self, "_eval_call_count", 0) + 1
        import os as _os_bd
        _bd_at = int(_os_bd.environ.get("DIAG_COST_BD_AT_TICK", "0") or "0")
        if _bd_at > 0 and self._eval_call_count == _bd_at:
            self._dump_cost_breakdown(results, labels=None)
        return results

    def _dump_cost_breakdown(self, results, labels=None):
        """Per-sample cost breakdown for arc-2 diagnostic. Called once by
        evaluate_samples when env-gated. Prints per-knot state cost, terminal,
        control cost, and Q-block contributions (obj xy, obj z, obj vel,
        ee pos, ee vel) so we can see which term drives the inversion."""
        print(f"[COST-BD] === per-sample breakdown at eval_call={self._eval_call_count} ===",
              flush=True)
        # EE-space layout: x = [box_q(7), p_ee(3), box_v(6), v_ee(3)]
        _BOX_Q = slice(0, 7)
        _P_EE  = slice(7, 10)
        _BOX_V = slice(10, 16)
        _V_EE  = slice(16, 19)
        for k, r in enumerate(results):
            if r.x_seq is None or r.u_seq is None or r.Q is None:
                print(f"[COST-BD] k={k} infeasible/no-plan; c_C3_raw={r.c_C3_raw:.3f}",
                      flush=True)
                continue
            x_seq, u_seq, Q, R, QN, x_ref = r.x_seq, r.u_seq, r.Q, r.R, r.QN, r.x_ref
            N = len(u_seq)
            # Per-knot state cost split by block
            box_q_cost = box_v_cost = p_ee_cost = v_ee_cost = 0.0
            for t in range(N):
                e = x_seq[t] - x_ref
                box_q_cost += float(e[_BOX_Q] @ Q[_BOX_Q, _BOX_Q] @ e[_BOX_Q])
                box_v_cost += float(e[_BOX_V] @ Q[_BOX_V, _BOX_V] @ e[_BOX_V])
                p_ee_cost  += float(e[_P_EE]  @ Q[_P_EE,  _P_EE]  @ e[_P_EE])
                v_ee_cost  += float(e[_V_EE]  @ Q[_V_EE,  _V_EE]  @ e[_V_EE])
            # Terminal
            eN = x_seq[N] - x_ref
            xN_box_q = float(eN[_BOX_Q] @ QN[_BOX_Q, _BOX_Q] @ eN[_BOX_Q])
            xN_box_v = float(eN[_BOX_V] @ QN[_BOX_V, _BOX_V] @ eN[_BOX_V])
            xN_p_ee  = float(eN[_P_EE]  @ QN[_P_EE,  _P_EE]  @ eN[_P_EE])
            xN_v_ee  = float(eN[_V_EE]  @ QN[_V_EE,  _V_EE]  @ eN[_V_EE])
            xN_total = xN_box_q + xN_box_v + xN_p_ee + xN_v_ee
            # Control cost
            u_cost = float(sum(u_seq[t] @ R @ u_seq[t] for t in range(N)))
            # Total (sanity check vs c_C3_raw)
            state_total = box_q_cost + box_v_cost + p_ee_cost + v_ee_cost + xN_total
            total = state_total + u_cost
            # Contact info
            lam_first = float(np.max(np.abs(r.J_n[0])) if r.J_n is not None
                              and r.J_n.size > 0 else 0.0)
            print(f"[COST-BD] k={k} sample_pos=({r.sample_pos[0]:+.4f},"
                  f"{r.sample_pos[1]:+.4f},{r.sample_pos[2]:+.4f}) "
                  f"x0_p_ee=({r.x0[7]:+.4f},{r.x0[8]:+.4f},{r.x0[9]:+.4f}) "
                  f"c_C3_raw={r.c_C3_raw:.3f} sum_check={total:.3f}",
                  flush=True)
            print(f"[COST-BD]   state total={state_total:.3f}: "
                  f"box_q={box_q_cost:.3f} box_v={box_v_cost:.3f} "
                  f"p_ee={p_ee_cost:.3f} v_ee={v_ee_cost:.3f}",
                  flush=True)
            print(f"[COST-BD]   terminal={xN_total:.3f}: "
                  f"box_q={xN_box_q:.3f} box_v={xN_box_v:.3f} "
                  f"p_ee={xN_p_ee:.3f} v_ee={xN_v_ee:.3f}",
                  flush=True)
            print(f"[COST-BD]   u_cost={u_cost:.3f}  "
                  f"|u|_max_knot={np.max(np.linalg.norm(u_seq, axis=1)):.3f}N",
                  flush=True)
            # Trajectory-level box position drift (to see if this sample's
            # plan predicts box moving toward goal — key for understanding
            # why one sample "wins" over another under Q_obj_pos weight)
            box_p_x0 = x_seq[0, 4]
            box_p_xN = x_seq[N, 4]
            box_p_y0 = x_seq[0, 5]
            box_p_yN = x_seq[N, 5]
            print(f"[COST-BD]   box_p drift x={box_p_x0:+.4f}->{box_p_xN:+.4f} "
                  f"({(box_p_xN-box_p_x0)*1000:+.2f}mm)  "
                  f"y={box_p_y0:+.4f}->{box_p_yN:+.4f} "
                  f"({(box_p_yN-box_p_y0)*1000:+.2f}mm)",
                  flush=True)
        print("[COST-BD] === end breakdown ===", flush=True)

    # ------------------------------------------------------------------
    # Re-solve a winning sample with full ADMM iters
    # ------------------------------------------------------------------

    def _r7_gravity_centered_ubox(self, plant_ctx=None):
        """R^7 gravity-centered u-box — mirror of the main-solve box in
        ci_mpc_c3plus (REFCONF_R7_U_GRAVITY_CENTERED, default ON):

            u ∈ [u_hold − Δ, u_hold + Δ],  u_hold = −τ_g_arm(q),
            Δ_j = (|J_arm|ᵀ·F_ref)_j,  F_ref from PORT_U_HORIZONTAL/VERTICAL
            (default 50 N — reference push_t sampling_c3plus_options.yaml),
            floored at 1 Nm/joint, clipped to ±87 Nm.

        Evaluated at whatever arm config plant_ctx currently holds (the
        sample's IK'd config during evaluate_sample). With plant_ctx=None
        (resolve_at_full_iters has no ctx) returns the last computed box —
        τ_g varies by only a few Nm across sample postures, small against
        the ~35-50 Nm half-widths. Returns (None, None) when gated off,
        letting the solver fall back to the scalar torque_limit."""
        if plant_ctx is None:
            return getattr(self, "_last_r7_ubox", (None, None))
        _n_arm = int(self.n_u)
        _tau_g = self.plant.CalcGravityGeneralizedForces(plant_ctx)
        _u_hold = -np.asarray(_tau_g[:_n_arm], dtype=float)
        _J = self.plant.CalcJacobianTranslationalVelocity(
            plant_ctx, ad.JacobianWrtVariable.kV,
            self.ee_frame, np.zeros(3),
            self.plant.world_frame(), self.plant.world_frame(),
        )[:, :_n_arm]
        _F_ref = np.array([
            float(__import__("os").environ.get("PORT_U_HORIZONTAL", "50.0")),
            float(__import__("os").environ.get("PORT_U_HORIZONTAL", "50.0")),
            float(__import__("os").environ.get("PORT_U_VERTICAL",   "50.0")),
        ])
        _delta = np.maximum(np.abs(_J).T @ _F_ref, 1.0)
        _lo = np.maximum(_u_hold - _delta, -87.0)
        _hi = np.minimum(_u_hold + _delta, +87.0)
        self._last_r7_ubox = (_lo, _hi)
        return _lo, _hi

    def resolve_at_full_iters(self,
                              r: SampleResult,
                              suppress_io: bool = True) -> SampleResult:
        """Re-run a sample at full ADMM iters using its captured LCS
        matrices. Matches legacy lines 649-677 — used when entering rich
        mode with k* != 0."""
        if not r.feasible or r.x0 is None:
            return r
        _buf = io.StringIO()
        ctx = redirect_stdout(_buf) if suppress_io else _NullContext()
        try:
            with ctx:
                # Diag 2: same per-axis-bounds plumbing as the surrogate path.
                # F2 fix (2026-07-28b): all EE-space, not tshape-only —
                # see evaluate_sample site.
                _ee_space = (self.solver.n_u == 3)
                _u_lo = self._u_lo if _ee_space else None
                _u_hi = self._u_hi if _ee_space else None
                if not _ee_space:
                    # R^7 gravity-centered u-box (see surrogate site above).
                    # No plant_ctx in this method — uses the box cached at
                    # this sample's evaluate_sample pass.
                    _u_lo, _u_hi = self._r7_gravity_centered_ubox(None)
                u_seq, x_seq = self.solver.solve(
                    r.x0, r.A, r.B, r.D, r.d, r.J_n, r.J_t, r.mu,
                    r.Q, r.R, r.QN, r.x_ref,
                    N=self.horizon,
                    admm_iter=self.base_admm_iter,
                    torque_limit=self.torque_limit,
                    u_lower=_u_lo, u_upper=_u_hi,
                    phi=r.phi,
                )
            # F1 fix (2026-07-28b): score with the SAME object-only mask the
            # ranking path uses, else this full-resolve value (full Q/R,
            # ~1.5× larger) is scale-inconsistent with the c_samples it is
            # compared against in the dispatcher.
            if _ee_space:
                _Qm, _QNm, _Rm = _object_only_cost_matrices_ee_space(
                    r.Q, r.QN, r.R)
                c_C3_raw = traj_cost(x_seq, u_seq, _Qm, _Rm, _QNm, r.x_ref)
            else:
                _n_x_r = r.Q.shape[0]
                _obj_mask = np.zeros(_n_x_r, dtype=bool)
                _obj_mask[self.n_u:self.n_q] = True
                _obj_mask[self.n_q + self.n_u:_n_x_r] = True
                _M_r = np.outer(_obj_mask, _obj_mask)
                c_C3_raw = traj_cost(
                    x_seq, u_seq,
                    r.Q * _M_r, np.zeros_like(r.R), r.QN * _M_r, r.x_ref)
            self.full_solves += 1
            r.u_seq    = u_seq
            r.x_seq    = x_seq
            r.c_C3_raw = c_C3_raw
            r.c_sample = c_C3_raw - r.align_bonus - r.rot_bonus + r.travel_penalty
        except Exception:
            pass
        return r


class _NullContext:
    """No-op context manager (used when suppress_io=False)."""
    def __enter__(self):  return None
    def __exit__(self, *args): return False
