import numpy as np
import pydrake.all as ad

class LCSFormulator:
    def __init__(self, plant):
        self.plant = plant
        
        # System dimensions
        self.n_q = self.plant.num_positions()
        self.n_v = self.plant.num_velocities()
        self.n_u = self.plant.num_actuators()

    def extract_dynamics(self, context):
        """
        Extracts: M(q)v_dot + C(q,v)v = tau_g(q) + B*u
        """
        M = self.plant.CalcMassMatrixViaInverseDynamics(context)
        Cv = self.plant.CalcBiasTerm(context)
        tau_g = self.plant.CalcGravityGeneralizedForces(context)
        B = self.plant.MakeActuationMatrix()
        
        return M, Cv, tau_g, B

    def extract_lcs_contacts(self, context, distance_threshold=0.05):
        """
        Finds objects close to collision and calculates the Gap (phi), 
        Normal Contact Jacobian (J_n), and Polyhedral Friction Jacobian (J_t).
        """
        query_object = self.plant.get_geometry_query_input_port().Eval(context)
        inspector = query_object.inspector()
        sd_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(distance_threshold)
        
        phis = []
        J_n_list = []
        J_t_list = [] # Tangential (Friction) Jacobians

        W = self.plant.world_frame()

        for sdp in sd_pairs:
            phis.append(sdp.distance)
            
            body_A = self.plant.GetBodyFromFrameId(inspector.GetFrameId(sdp.id_A))
            body_B = self.plant.GetBodyFromFrameId(inspector.GetFrameId(sdp.id_B))

            # Jacobians for the specific contact points
            J_v_A = self.plant.CalcJacobianTranslationalVelocity(
                context, ad.JacobianWrtVariable.kV, body_A.body_frame(), sdp.p_ACa, W, W)
            
            J_v_B = self.plant.CalcJacobianTranslationalVelocity(
                context, ad.JacobianWrtVariable.kV, body_B.body_frame(), sdp.p_BCb, W, W)

            J_rel = J_v_A - J_v_B
            nhat = sdp.nhat_BA_W

            # --- 1. Normal Jacobian ---
            J_n_list.append(nhat.T @ J_rel)

            # --- 2. Polyhedral Friction (Quadhedron) ---
            # Create a robust tangent plane (t1, t2) orthogonal to the normal
            arbitrary_vec = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(nhat, arbitrary_vec)) > 0.99:
                arbitrary_vec = np.array([0.0, 1.0, 0.0])
            
            t1 = np.cross(nhat, arbitrary_vec)
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(nhat, t1)

            # The 4 edges of the quadhedron pyramid
            D = [t1, -t1, t2, -t2]
            
            # Extract how joints move along these 4 friction lines
            for d in D:
                J_t_list.append(d.T @ J_rel)

        if len(phis) > 0:
            return np.array(phis), np.vstack(J_n_list), np.vstack(J_t_list)
        else:
            return np.array([]), np.zeros((0, self.n_v)), np.zeros((0, self.n_v))