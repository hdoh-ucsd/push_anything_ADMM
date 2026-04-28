"""
BaseMPC: shared rollout engine for C3+ trajectory optimization.

Key design decisions:
 - Owns a *private* diagram context for rollouts so the real simulator
   context is never mutated during candidate trajectory evaluation.
 - Accepts a task-specific cost callable (ManipulationCost) injected at
   construction time — no hardcoded box indices or cost weights.
 - Renormalises floating-body quaternions after each Euler integration
   step to prevent norm drift corrupting contact detection.
"""
import numpy as np

try:
    from profiling.section_timer import timed
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def timed(_name):   # noqa: E306
        yield


def _renormalize_quaternions(q: np.ndarray, plant) -> None:
    """In-place unit-normalisation of all floating-body quaternions in q."""
    for body_index in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_index)
        ps   = body.floating_positions_start()
        norm = np.linalg.norm(q[ps:ps + 4])
        if norm > 1e-9:
            q[ps:ps + 4] /= norm


class BaseMPC:
    """
    Shared rollout and nominal-plan storage for all MPC variants.

    Parameters
    ----------
    n_u, n_v, n_q : system dimensions (actuators, velocities, positions)
    formulator    : LCSFormulator  — extracts M, Cv, tau_g, B, J_n, J_t
    admm_solver   : ADMMSolver    — returns contact-consistent v_dot
    diagram       : Drake Diagram — used to create an isolated rollout context
    cost_fn       : callable(plant_ctx, q_sim, target_xy) -> float
    horizon       : int   planning horizon (number of timesteps)
    dt            : float integration timestep (seconds)
    """

    def __init__(self, n_u, n_v, n_q, formulator, admm_solver,
                 diagram, cost_fn, horizon: int = 15, dt: float = 0.03):
        self.n_u          = n_u
        self.n_v          = n_v
        self.n_q          = n_q
        self.formulator   = formulator
        self.admm_solver  = admm_solver
        self.cost_fn      = cost_fn
        self.horizon      = horizon
        self.dt           = dt

        # Nominal (warm-started) torque plan: shape (horizon, n_u)
        self.U_nominal = np.zeros((horizon, n_u))

        # Private rollout context — never exposed outside BaseMPC.
        # Must be a *diagram* context so the geometry query port is connected.
        self._rollout_ctx = diagram.CreateDefaultContext()
        self._rollout_plant_ctx = formulator.plant.GetMyContextFromRoot(
            self._rollout_ctx
        )

    # ------------------------------------------------------------------
    def rollout_trajectory(self,
                           current_q: np.ndarray,
                           current_v: np.ndarray,
                           U_sequence: np.ndarray,
                           target_xy: np.ndarray) -> float:
        """
        Simulate the plant forward for `horizon` steps under U_sequence
        and accumulate the running + terminal cost.

        Parameters
        ----------
        current_q  : (n_q,)         real plant positions (read-only)
        current_v  : (n_v,)         real plant velocities (read-only)
        U_sequence : (horizon, n_u) torque commands to evaluate
        target_xy  : (2,)           2D goal [x_goal, y_goal] in world frame

        Returns
        -------
        total_cost : float
        """
        plant = self.formulator.plant
        ctx   = self._rollout_plant_ctx

        # Seed the private rollout context from the real state
        q_sim = current_q.copy()
        v_sim = current_v.copy()
        plant.SetPositions(ctx, q_sim)
        plant.SetVelocities(ctx, v_sim)

        total_cost = 0.0

        for t in range(self.horizon):
            # Running cost at current imagined state
            with timed("rollout.cost_fn"):
                total_cost += self.cost_fn(ctx, q_sim, target_xy)

            # Extract LCS matrices
            M, Cv, tau_g, B = self.formulator.extract_dynamics(ctx)
            _, J_n, J_t, mu = self.formulator.extract_lcs_contacts(ctx)

            # Total generalised force = gravity + actuation
            tau_total = tau_g + B @ U_sequence[t]

            # Contact-consistent acceleration via ADMM
            with timed("rollout.admm_loop"):
                v_dot, _ = self.admm_solver.run_admm_loop(
                    M, Cv, tau_total, B, J_n, J_t, mu=mu, max_iters=3
                )

            # Euler integration
            with timed("rollout.integration"):
                v_sim = v_sim + v_dot * self.dt
                q_dot = plant.MapVelocityToQDot(ctx, v_sim)
                q_sim = q_sim + q_dot * self.dt
                _renormalize_quaternions(q_sim, plant)

            # Advance rollout context to imagined next state
            plant.SetPositions(ctx, q_sim)
            plant.SetVelocities(ctx, v_sim)

        # Terminal cost at the end of the horizon
        total_cost += self.cost_fn(ctx, q_sim, target_xy)

        return total_cost
