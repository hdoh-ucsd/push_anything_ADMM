import numpy as np
import pydrake.all as ad

class ADMMSolver:
    def __init__(self, n_v):
        """
        Initializes the ADMM Solver.
        n_v: Number of generalized velocities in the plant.
        """
        self.n_v = n_v
        
        # ADMM Penalty Parameter (rho)
        # Determines how aggressively the solver forces consensus between physics and contacts
        self.rho = 1e-3 
        
        # Initialize the Drake OSQP Solver (highly optimized for QPs)
        self.solver = ad.OsqpSolver()

    def solve_x_update_qp(self, M, Cv, tau_g, B, J_n, J_t, z_prev, u_prev):
        """
        The QP Block: Solves for the smooth robot dynamics while trying to stay 
        close to the contact constraints dictated by the z-update.
        """
        # Create a fresh mathematical program for this iteration
        prog = ad.MathematicalProgram()
        
        # Determine total number of contact constraints (normals + tangents)
        n_c_total = J_n.shape[0] + J_t.shape[0]
        
        # 1. Decision Variables
        v_dot = prog.NewContinuousVariables(self.n_v, "v_dot")
        lambda_c = prog.NewContinuousVariables(n_c_total, "lambda_c")
        
        # Stack Jacobians to match lambda_c
        if n_c_total > 0:
            J_c = np.vstack([J_n, J_t])
        else:
            J_c = np.zeros((0, self.n_v))

        # 2. The Dynamics Constraint
        # M * v_dot - J_c^T * lambda_c = tau_g - C*v
        A_dyn = np.hstack([M, -J_c.T])
        vars_dyn = np.concatenate([v_dot, lambda_c])
        rhs_dyn = tau_g - Cv
        
        prog.AddLinearEqualityConstraint(A_dyn, rhs_dyn, vars_dyn)

        # 3. The ADMM Consensus Cost
        # Min || J_c * v_dot - (z_prev - u_prev) ||^2_2
        if n_c_total > 0:
            target = z_prev - u_prev
            
            Q = 2.0 * self.rho * (J_c.T @ J_c)
            b = -2.0 * self.rho * (target.T @ J_c)
            
            # 1. Force perfect mathematical symmetry to cancel floating-point errors
            Q = 0.5 * (Q + Q.T)
            # 2. Add tiny regularization to the diagonal to guarantee strict convexity
            Q += np.eye(Q.shape[0]) * 1e-4
            
            prog.AddQuadraticCost(Q, b, v_dot)

        # 4. Solve the QP
        result = self.solver.Solve(prog)

        # 4. Solve the QP
        result = self.solver.Solve(prog)
        
        if result.is_success():
            return result.GetSolution(v_dot), result.GetSolution(lambda_c)
        else:
            print("QP Solver Failed!")
            return np.zeros(self.n_v), np.zeros(n_c_total)

    def solve_z_update(self, v_dot_opt, J_c, u_prev, num_normals, mu=0.5):
        """
        The Proximal Block: Projects the proposed velocities onto the valid contact constraints.
        """
        # Proposed velocities in contact space
        v_c = J_c @ v_dot_opt
        proposal = v_c + u_prev
        z_new = np.zeros_like(proposal)
        
        # 1. Normal Projection (Objects cannot penetrate)
        if num_normals > 0:
            normals_proposal = proposal[:num_normals]
            z_new[:num_normals] = np.maximum(0, normals_proposal)
            
        # 2. Friction Projection (Sticking vs Sliding)
        if proposal.shape[0] > num_normals:
            tangents_proposal = proposal[num_normals:]
            
            # Simple thresholding: if moving very slowly, snap to sticking (0.0)
            sticking_threshold = 1e-4 
            z_new[num_normals:] = np.where(np.abs(tangents_proposal) < sticking_threshold, 
                                           0.0, 
                                           tangents_proposal)
            
        return z_new

    def run_admm_loop(self, M, Cv, tau_g, B, J_n, J_t, max_iters=25):
        """
        The complete Alternating Direction Method of Multipliers loop.
        Alternates between physics (QP) and contact rules (Proximal) to find a consensus.
        """
        num_normals = J_n.shape[0]
        n_c_total = num_normals + J_t.shape[0]
        
        if n_c_total > 0:
            J_c = np.vstack([J_n, J_t])
        else:
            J_c = np.zeros((0, self.n_v))
            
        # Initialize ADMM variables (z and u)
        z = np.zeros(n_c_total)
        u = np.zeros(n_c_total)
        
        v_dot_opt = np.zeros(self.n_v)
        lambda_opt = np.zeros(n_c_total)
        
        for i in range(max_iters):
            # Step 1: x-update (Smooth Dynamics QP)
            v_dot_opt, lambda_opt = self.solve_x_update_qp(M, Cv, tau_g, B, J_n, J_t, z, u)
            
            if n_c_total == 0:
                break # If there are no contacts, the pure physics QP is the exact answer
            
            # Step 2: z-update (Non-smooth Contact Proximal)
            z_new = self.solve_z_update(v_dot_opt, J_c, u, num_normals)
            
            # Step 3: u-update (Dual Variable Accumulation)
            u = u + (J_c @ v_dot_opt - z_new)
            
            # Update z for the next loop
            z = z_new
            
        return v_dot_opt, lambda_opt