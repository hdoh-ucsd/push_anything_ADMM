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

import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pydrake.all as ad

from control.sampling_c3.ik import ik_seed_one_step, solve_ik_to_ee_pos
from control.sampling_c3.params import SamplingC3Params


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
                 params:         SamplingC3Params):
        self.plant       = plant
        self.world_frame = plant.world_frame()
        self.ee_frame    = ee_frame
        self.obj_body    = obj_body
        self.formulator  = formulator
        self.solver      = solver
        self.quad_cost   = quad_cost
        self.horizon       = int(horizon)
        self.dt            = float(dt)
        self.torque_limit  = float(torque_limit)
        self.base_admm_iter   = int(base_admm_iter)
        self.surrogate_iter   = int(params.surrogate_admm_iters)
        self.w_align          = float(params.w_align)
        self.w_travel         = float(params.w_travel)

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
                        suppress_io:   bool = True) -> SampleResult:
        """Evaluate one sample. Restores plant_ctx to (current_q, current_v)
        before returning."""
        if is_current_ee:
            q_seed   = current_q.copy()
            ik_err   = 0.0
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

        # FK current EE at the IK-resolved config
        ee_pos_resolved = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten().copy()

        admm_iter_k = self.base_admm_iter if (is_current_ee or full_iters) \
                      else self.surrogate_iter

        feasible = False
        nhats: list = []
        c_C3_raw = float("inf")
        u_seq = x_seq = None
        A = B = D = d = J_n = J_t = phi = mu = None
        Q = R = QN = x_ref = x0 = None

        _buf = io.StringIO()
        ctx = redirect_stdout(_buf) if suppress_io else _NullContext()
        try:
            with ctx:
                A, B, D, d, J_n, J_t, phi, mu = \
                    self.formulator.linearize_discrete(plant_ctx, self.dt)
                # Capture immediately — _last_nhats is overwritten on the
                # next linearize_discrete call.
                nhats = list(self.formulator._last_nhats)
                Q, R, QN, x_ref = self.quad_cost.build(
                    target_xy, plant_ctx=plant_ctx, current_q=q_seed,
                )
                x0 = np.concatenate([q_seed, current_v])
                u_seq, x_seq = self.solver.solve(
                    x0, A, B, D, d, J_n, J_t, mu,
                    Q, R, QN, x_ref,
                    N=self.horizon,
                    admm_iter=admm_iter_k,
                    torque_limit=self.torque_limit,
                    phi=phi,
                )
            c_C3_raw = traj_cost(x_seq, u_seq, Q, R, QN, x_ref)
            feasible = True
            if admm_iter_k >= self.base_admm_iter:
                self.full_solves += 1
            else:
                self.cheap_solves += 1
        except Exception:
            pass

        # Alignment bonus over contact normals
        if nhats:
            alignments  = [max(0.0, float(np.dot(n, g_hat_3d))) for n in nhats]
            align_score = max(alignments)
        else:
            align_score = 0.0
        align_bonus    = self.w_align  * align_score
        travel_dist    = float(np.linalg.norm(sample_pos - ee_pos_now))
        travel_penalty = self.w_travel * travel_dist
        c_sample       = c_C3_raw - align_bonus + travel_penalty

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

    def evaluate_samples(self,
                         samples:       List[np.ndarray],
                         current_q:     np.ndarray,
                         current_v:     np.ndarray,
                         plant_ctx,
                         target_xy:     np.ndarray,
                         ee_pos_now:    np.ndarray,
                         g_hat_3d:      np.ndarray,
                         threading:     bool = False) -> List[SampleResult]:
        """Sequential by default. `threading=True` is reserved for future
        parallelisation; raises NotImplementedError until profiled."""
        if threading:
            raise NotImplementedError(
                "Parallel sample evaluation not yet implemented. "
                "Profile the sequential K=4 baseline first.")

        results: list[SampleResult] = []
        for k, p in enumerate(samples):
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
            )
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Re-solve a winning sample with full ADMM iters
    # ------------------------------------------------------------------

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
                u_seq, x_seq = self.solver.solve(
                    r.x0, r.A, r.B, r.D, r.d, r.J_n, r.J_t, r.mu,
                    r.Q, r.R, r.QN, r.x_ref,
                    N=self.horizon,
                    admm_iter=self.base_admm_iter,
                    torque_limit=self.torque_limit,
                    phi=r.phi,
                )
            c_C3_raw = traj_cost(x_seq, u_seq, r.Q, r.R, r.QN, r.x_ref)
            self.full_solves += 1
            r.u_seq    = u_seq
            r.x_seq    = x_seq
            r.c_C3_raw = c_C3_raw
            r.c_sample = c_C3_raw - r.align_bonus + r.travel_penalty
        except Exception:
            pass
        return r


class _NullContext:
    """No-op context manager (used when suppress_io=False)."""
    def __enter__(self):  return None
    def __exit__(self, *args): return False
