import numpy as np
from .base_mpc import BaseMPC

class C3MPC(BaseMPC):
    """
    Deterministic C3.
    Uses finite-difference gradients on a SINGLE trajectory.
    Highly prone to getting stuck in local minima during contact switches.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = 0.01
        self.epsilon = 1e-3  # Perturbation for finite difference

    def compute_control(self, current_context, current_q, current_v, target_q):
        # 1. Evaluate current plan
        base_cost = self.rollout_trajectory(current_context, current_q, current_v, self.U_nominal, target_q)
        
        grad = np.zeros_like(self.U_nominal)
        
        # 2. Compute Gradient via Finite Difference
        for t in range(self.horizon):
            for i in range(self.n_u):
                U_perturbed = np.copy(self.U_nominal)
                U_perturbed[t, i] += self.epsilon
                
                cost_perturbed = self.rollout_trajectory(
                    current_context, current_q, current_v, U_perturbed, target_q)
                
                grad[t, i] = (cost_perturbed - base_cost) / self.epsilon
                
        # 3. Gradient Descent Step
        self.U_nominal = self.U_nominal - self.learning_rate * grad
        
        # 4. Shift plan forward
        optimal_u_now = np.copy(self.U_nominal[0])
        self.U_nominal[:-1] = self.U_nominal[1:]
        self.U_nominal[-1] = np.zeros(self.n_u)
        
        return optimal_u_now