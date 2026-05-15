"""Clean-room sign test for Drake's CalcGravityGeneralizedForces.

Builds a one-link pendulum from primitives (no URDF, no env_builder),
queries the API at a configuration where the analytic answer is known,
and reports whether Drake returns the gravitational generalized force
(positive = drives joint in +theta) or the compensation torque
(positive = what an actuator must apply to hold position).

Geometry: revolute joint at world origin, axis = +y. Point mass m = 1 kg
attached at p_BoCm_B = (L, 0, 0) with L = 1 m. Gravity = (0, 0, -9.81).

At theta = 0 the COM is at (1, 0, 0): horizontal lever, gravity pulling
straight down. Analytically:
  r x F = (L,0,0) x (0,0,-mg) = (0, +mgL, 0)   ->   tau_grav.y = +mgL
The compensation torque the actuator must command is -mgL.
"""
import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import SpatialInertia, RotationalInertia, RevoluteJoint


def main():
    m = 1.0
    L = 1.0
    g = 9.81

    plant = MultibodyPlant(time_step=0.0)

    # Near-point mass: COM at (L, 0, 0) in body frame, small isotropic
    # central inertia so the SpatialInertia is physically valid.
    I_cm = 1e-4
    inertia = SpatialInertia.MakeFromCentralInertia(
        mass=m,
        p_PScm_E=np.array([L, 0.0, 0.0]),
        I_SScm_E=RotationalInertia(I_cm, I_cm, I_cm),
    )
    body = plant.AddRigidBody("link", inertia)

    joint = plant.AddJoint(
        RevoluteJoint(
            name="hinge",
            frame_on_parent=plant.world_frame(),
            frame_on_child=body.body_frame(),
            axis=[0.0, 1.0, 0.0],
        )
    )

    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -g])
    plant.Finalize()

    # -------- STEP 2: plant sanity --------
    print("---- plant sanity ----")
    print(f"num_positions  = {plant.num_positions()}")
    print(f"num_velocities = {plant.num_velocities()}")
    print(f"gravity_vector = {plant.gravity_field().gravity_vector()}")

    context = plant.CreateDefaultContext()

    joint.set_angle(context, 0.0)
    X_WB = plant.EvalBodyPoseInWorld(context, body)
    p_WCm = X_WB @ np.array([L, 0.0, 0.0])
    print(f"theta = 0     : COM in world = ({p_WCm[0]:+.4f}, {p_WCm[1]:+.4f}, {p_WCm[2]:+.4f})  (expect (+1, 0, 0))")

    joint.set_angle(context, np.pi / 2.0)
    X_WB = plant.EvalBodyPoseInWorld(context, body)
    p_WCm = X_WB @ np.array([L, 0.0, 0.0])
    print(f"theta = pi/2  : COM in world = ({p_WCm[0]:+.4f}, {p_WCm[1]:+.4f}, {p_WCm[2]:+.4f})  (expect (0, 0, -1))")
    print()

    # -------- STEP 3: gravity-force query --------
    joint.set_angle(context, 0.0)
    g_drake = plant.CalcGravityGeneralizedForces(context)

    print("---- gravity-force query ----")
    print(f"theta = 0, analytic gravity force on joint = +{m*g*L:.4f} N*m")
    print(f"theta = 0, analytic compensation torque    = -{m*g*L:.4f} N*m")
    print(f"theta = 0, Drake CalcGravityGeneralizedForces = {g_drake[0]:+.4f} N*m")
    print()

    if abs(g_drake[0] - m * g * L) < 0.01:
        print("RESULT: Drake returns generalized gravity force.")
        print("        Production code (u = tau_g + u_pd) is SIGN-WRONG.")
    elif abs(g_drake[0] + m * g * L) < 0.01:
        print("RESULT: Drake returns compensation torque.")
        print("        Production code sign is correct; probe failure is elsewhere.")
    else:
        print(f"RESULT: AMBIGUOUS. g_drake[0] = {g_drake[0]:+.4f} doesn't match either prediction.")
        print("        Toy plant likely misconstructed.")


if __name__ == "__main__":
    main()
