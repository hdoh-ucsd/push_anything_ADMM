import numpy as np
import copy

class BaseMPC:
    """Shared configuration and physics rollout for both MPC approaches."""
    def __init__(self, n_u, n_v, n_q, formulator, admm_solver, horizon=10, dt=0.01):
        self.n_u = n_u
        self.n_v = n_v
        self.n_q = n_q
        self.formulator = formulator
        self.admm_solver = admm_solver
        self.horizon = horizon
        self.dt = dt
        self.U_nominal = np.zeros((self.horizon, self.n_u))

    def evaluate_cost(self, q_sim, target_q, current_context):
        """
        Cost function: Distance of box to target + Distance of Robot EE to box
        This tells the arm to reach for the box even if it hasn't touched it yet!
        """
        box_x_idx, box_y_idx = 11, 12
        box_pos = q_sim[11:14] # Approximate box translation
        
        # 1. Box distance to target (Heavy weight)
        box_dist = np.linalg.norm(q_sim[box_x_idx:box_y_idx+1] - target_q[box_x_idx:box_y_idx+1])
        
        # 2. Arm distance to box (To guide the blind arm)
        ee_frame = self.formulator.plant.GetFrameByName("panda_link8")
        ee_pos = self.formulator.plant.CalcPointsPositions(
            current_context, ee_frame, [0,0,0], self.formulator.plant.world_frame()
        ).flatten()
        
        reach_dist = np.linalg.norm(ee_pos - box_pos)
        
        return (box_dist * 10.0) + reach_dist

    def rollout_trajectory(self, current_context, current_q, current_v, U_sequence, target_q):
        """Simulates the physics forward for a given control sequence."""
        q_sim = np.copy(current_q)
        v_sim = np.copy(current_v)
        
        for t in range(self.horizon):
            # ---> CRITICAL FIX: Update the physics engine with our imagined state <---
            self.formulator.plant.SetPositions(current_context, q_sim)
            self.formulator.plant.SetVelocities(current_context, v_sim)
            
            u_t = U_sequence[t]
            
            # Now the matrices correctly register if the arm hits the box in the future!
            M, Cv, tau_g, B = self.formulator.extract_dynamics(current_context)
            phi, J_n, J_t = self.formulator.extract_lcs_contacts(current_context)
            
            tau_total = tau_g + B @ u_t
            
            # Solve ADMM Physics
            v_dot, _ = self.admm_solver.run_admm_loop(M, Cv, tau_total, B, J_n, J_t)
            
            # Euler Integration
            v_sim = v_sim + v_dot * self.dt
            q_dot = self.formulator.plant.MapVelocityToQDot(current_context, v_sim)
            q_sim = q_sim + q_dot * self.dt
            
        return self.evaluate_cost(q_sim, target_q, current_context)