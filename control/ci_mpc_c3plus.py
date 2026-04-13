import numpy as np
from .base_mpc import BaseMPC

class C3PlusMPC(BaseMPC):
    """
    Sampled C3+.
    Uses MPPI to evaluate parallel trajectories.
    Easily bypasses local minima by exploring multiple contact modes.
    """
    def __init__(self, n_u, n_v, n_q, formulator, admm_solver, horizon=10, dt=0.01, num_samples=50):
        super().__init__(n_u, n_v, n_q, formulator, admm_solver, horizon, dt)
        self.num_samples = num_samples
        
        # Reduced noise to prevent physical instability ("Exploding Robot" bug)
        self.noise_std = 2.0 
        
        # Franka physical torque limits (approximate safe bounds)
        self.torque_limit = 30.0

    def compute_control(self, current_context, current_q, current_v, target_q):
        # 1. Generate random sample sequences
        noise = np.random.normal(0, self.noise_std, (self.num_samples, self.horizon, self.n_u))
        U_samples = self.U_nominal + noise
        
        # Clip samples to physical limits so we don't simulate impossible physics
        U_samples = np.clip(U_samples, -self.torque_limit, self.torque_limit)
        
        sample_costs = np.zeros(self.num_samples)

        # 2. Rollout all samples
        for k in range(self.num_samples):
            sample_costs[k] = self.rollout_trajectory(
                current_context, current_q, current_v, U_samples[k], target_q)

        self.formulator.plant.SetPositions(current_context, current_q)
        self.formulator.plant.SetVelocities(current_context, current_v)

        # 3. Information Theoretic Weighting
        temperature = 1.0
        beta = np.min(sample_costs)
        
        # Prevent division by zero
        if np.all(sample_costs == beta):
            weights = np.ones(self.num_samples) / self.num_samples
        else:
            weights = np.exp(-1.0 / temperature * (sample_costs - beta))
            weights = weights / np.sum(weights)

        # 4. Update nominal trajectory using weighted sum of successful noise
        weighted_noise = np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)
        self.U_nominal = self.U_nominal + weighted_noise
        
        # Clip nominal trajectory to keep the baseline safe
        self.U_nominal = np.clip(self.U_nominal, -self.torque_limit, self.torque_limit)
        
        # 5. Shift plan forward
        optimal_u_now = np.copy(self.U_nominal[0])
        self.U_nominal[:-1] = self.U_nominal[1:]
        self.U_nominal[-1] = np.zeros(self.n_u)

        return optimal_u_now