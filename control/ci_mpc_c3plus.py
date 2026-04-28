"""
C3+ MPC: MPPI-style sampled trajectory optimizer using C3+ contact physics.

Algorithm per control step:
  1. Sample K Gaussian-noise perturbations around the nominal plan.
  2. Roll out each sample through Drake plant + ADMM (in private context).
  3. Weight samples via the MPPI information-theoretic rule.
  4. Update nominal plan as the weighted noise correction.
  5. Return u[0] as the next torque command and shift plan forward.
"""
import numpy as np
from .base_mpc import BaseMPC


class C3PlusMPC(BaseMPC):
    """
    Parameters (beyond BaseMPC)
    ---------------------------
    num_samples  : int   Number of parallel candidate trajectories.
    noise_std    : float Torque perturbation standard deviation (Nm).
    torque_limit : float Hard torque clamp applied to all samples (Nm).
    temperature  : float MPPI temperature lambda (lower = greedier weighting).
    """

    def __init__(self, n_u, n_v, n_q, formulator, admm_solver,
                 diagram, cost_fn,
                 horizon: int = 8, dt: float = 0.03,
                 num_samples: int = 10, noise_std: float = 2.0,
                 torque_limit: float = 30.0, temperature: float = 1.0):
        super().__init__(n_u, n_v, n_q, formulator, admm_solver,
                         diagram, cost_fn, horizon, dt)
        self.num_samples  = num_samples
        self.noise_std    = noise_std
        self.torque_limit = torque_limit
        self.temperature  = temperature

    def compute_control(self, current_q: np.ndarray,
                        current_v: np.ndarray,
                        target_xy: np.ndarray) -> np.ndarray:
        """
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

        # 2. Roll out each sample (uses private rollout context — real context untouched)
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
