"""
==============================================================================
EXPERIMENTAL — NOT WIRED INTO main.py
==============================================================================

Status: experimental research alternative. Not wired into main.py and WILL
NOT run as-is. Two known interface mismatches block execution:

  1. rollout_trajectory() calls self.admm_solver.run_admm_loop(...), but the
     active C3Solver (control/admm_solver.py) implements solve(), not
     run_admm_loop(). Reviving this code path requires either restoring an
     older ADMMSolver or rewriting the rollout to use C3Solver.solve().

  2. QuadraticManipulationCost is not compatible with the
     cost_fn(plant_ctx, q_sim, target_xy) -> float signature this file
     expects. A cost-function adapter would also be needed.

Kept around as a reference for the MPPI-style sampled-trajectory variant.
The previous BaseMPC parent class has been merged into this file so the
experimental controller is self-contained.
==============================================================================

C3+ MPC: MPPI-style sampled trajectory optimizer using C3+ contact physics.

Algorithm per control step:
  1. Sample K Gaussian-noise perturbations around the nominal plan.
  2. Roll out each sample through Drake plant + ADMM (in private context).
  3. Weight samples via the MPPI information-theoretic rule.
  4. Update nominal plan as the weighted noise correction.
  5. Return u[0] as the next torque command and shift plan forward.
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


class C3PlusMPC:
    """
    MPPI-style sampled trajectory optimizer with private rollout context.

    Owns a private diagram context for rollouts so the real simulator
    context is never mutated during candidate trajectory evaluation.
    Renormalises floating-body quaternions after each Euler integration
    step to prevent norm drift corrupting contact detection.

    Parameters
    ----------
    n_u, n_v, n_q : system dimensions (actuators, velocities, positions)
    formulator    : LCSFormulator  — extracts M, Cv, tau_g, B, J_n, J_t
    admm_solver   : ADMM solver    — must implement run_admm_loop(...)
                                     (see WARNING in module docstring)
    diagram       : Drake Diagram  — used to create an isolated rollout context
    cost_fn       : callable(plant_ctx, q_sim, target_xy) -> float
    horizon       : int   planning horizon (number of timesteps)
    dt            : float integration timestep (seconds)
    num_samples   : int   number of parallel candidate trajectories
    noise_std     : float torque perturbation standard deviation (Nm)
    torque_limit  : float hard torque clamp applied to all samples (Nm)
    temperature   : float MPPI temperature lambda (lower = greedier weighting)
    """

    def __init__(self, n_u, n_v, n_q, formulator, admm_solver,
                 diagram, cost_fn,
                 horizon: int = 8, dt: float = 0.03,
                 num_samples: int = 10, noise_std: float = 2.0,
                 torque_limit: float = 30.0, temperature: float = 1.0):
        self.n_u          = n_u
        self.n_v          = n_v
        self.n_q          = n_q
        self.formulator   = formulator
        self.admm_solver  = admm_solver
        self.cost_fn      = cost_fn
        self.horizon      = horizon
        self.dt           = dt
        self.num_samples  = num_samples
        self.noise_std    = noise_std
        self.torque_limit = torque_limit
        self.temperature  = temperature

        # Nominal (warm-started) torque plan: shape (horizon, n_u)
        self.U_nominal = np.zeros((horizon, n_u))

        # Private rollout context — never exposed externally.
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

    # ------------------------------------------------------------------
    def compute_control(self, current_q: np.ndarray,
                        current_v: np.ndarray,
                        target_xy: np.ndarray) -> np.ndarray:
        """
        Compute one MPPI-weighted torque command.

        Parameters
        ----------
        current_q : (n_q,) real plant positions
        current_v : (n_v,) real plant velocities
        target_xy : (2,)   goal [x_goal, y_goal] in world frame

        Returns
        -------
        u_opt : (n_u,) torque command for this timestep
        """
        # 1. Sample torque perturbations around nominal plan
        eps = np.random.normal(
            0.0, self.noise_std,
            (self.num_samples, self.horizon, self.n_u)
        )
        U_samples = np.clip(
            self.U_nominal + eps, -self.torque_limit, self.torque_limit
        )

        # 2. Roll out each sample (private rollout context — real context untouched)
        costs = np.array([
            self.rollout_trajectory(current_q, current_v, U_samples[k], target_xy)
            for k in range(self.num_samples)
        ])

        # 3. MPPI information-theoretic weighting
        beta       = costs.min()
        raw_w      = np.exp(-(costs - beta) / self.temperature)
        weight_sum = raw_w.sum()

        if weight_sum < 1e-12:
            weights = np.ones(self.num_samples) / self.num_samples
        else:
            weights = raw_w / weight_sum

        # 4. Update nominal plan: U_nominal += weighted average of noise
        weighted_noise = np.einsum("k,kth->th", weights, eps)
        self.U_nominal = np.clip(
            self.U_nominal + weighted_noise,
            -self.torque_limit, self.torque_limit
        )

        # 5. Extract first action and shift plan forward (receding horizon)
        u_opt = self.U_nominal[0].copy()
        self.U_nominal[:-1] = self.U_nominal[1:]
        self.U_nominal[-1]  = np.zeros(self.n_u)

        return u_opt
