"""
Generic Drake environment builder for all three manipulation tasks.

Builds: table (static) + Franka Panda arm + task-specific manipulable object.
Object geometry (box vs sphere) and physical properties come from tasks.yaml.
"""
import numpy as np
import pydrake.all as ad

# Panda base weld: arm sits 0.6 m behind table centre along -Y
ROBOT_BASE_XYZ = [0.0, -0.6, 0.0]

# Confirmed working pose: EE at (0.347, -0.600, 0.097) world at t=0,
# no self-collisions. Forward-lean places EE near table height.
INITIAL_ARM_Q = np.array([
    +0.552150, +0.675037, +0.976275, -2.246164, -0.188979, +3.044706, +0.785000,
])
# FK: EE at (0.000, 0.000, 0.200) — 20 cm directly above the box centre


# Legacy pre-positioned start pose: pusher already touching box's west face.
# Hand-tuned for the EAST push only — does not generalise to other directions.
# Kept as a fallback seed for `compute_prepositioned_arm_q` when the IK cascade
# from `INITIAL_ARM_Q` fails to converge for a particular push direction.
_LEGACY_PREPOSITIONED_ARM_Q = np.array([
    +0.602050, +1.368574, +1.119371, -1.896220, +1.722135, +2.991599, +0.785000,
])

# Dedicated pusher body — spherical puck rigidly welded to panda_link8.
# This is the single authoritative name for EE body and contact filter.
EE_BODY_NAME  = "pusher"
PUSHER_RADIUS = 0.025   # m — matches Dairlab C3 planar-pushing benchmark


# ---------------------------------------------------------------------------
# SDF generators (runtime-parameterised so all properties come from config)
# ---------------------------------------------------------------------------

def _box_sdf(cfg: dict) -> str:
    sx, sy, sz = cfg["size"]
    m  = cfg["mass"]
    mu = cfg["friction"]
    r, g, b, a = cfg["color_rgba"]
    # Solid-box principal inertia
    ixx = m / 12.0 * (sy**2 + sz**2)
    iyy = m / 12.0 * (sx**2 + sz**2)
    izz = m / 12.0 * (sx**2 + sy**2)
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="manipulated_object">
    <link name="box_link">
      <inertial>
        <mass>{m}</mass>
        <inertia>
          <ixx>{ixx:.6f}</ixx><iyy>{iyy:.6f}</iyy><izz>{izz:.6f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <surface>
          <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def _sphere_sdf(cfg: dict) -> str:
    rad = cfg["radius"]
    m   = cfg["mass"]
    mu  = cfg["friction"]
    cr, cg, cb, ca = cfg["color_rgba"]
    # Solid-sphere inertia: I = 2/5 m r^2
    I = 2.0 / 5.0 * m * rad**2
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="manipulated_object">
    <link name="ball_link">
      <inertial>
        <mass>{m}</mass>
        <inertia>
          <ixx>{I:.6f}</ixx><iyy>{I:.6f}</iyy><izz>{I:.6f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><sphere><radius>{rad}</radius></sphere></geometry>
        <surface>
          <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><sphere><radius>{rad}</radius></sphere></geometry>
        <material><diffuse>{cr} {cg} {cb} {ca}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_environment(task_cfg: dict, time_step: float = 0.001):
    """
    Build a Drake diagram for a Franka Panda arm + table + task object.

    Parameters
    ----------
    task_cfg  : dict   Task configuration from config/tasks.yaml.
    time_step : float  Drake simulation timestep (seconds).

    Returns
    -------
    diagram      : Drake Diagram
    plant        : MultibodyPlant
    panda_model  : ModelInstanceIndex for the arm
    object_model : ModelInstanceIndex for the manipulated object
    meshcat      : Meshcat instance (visualiser at http://127.0.0.1:7000)
    """
    builder = ad.DiagramBuilder()
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = ad.Parser(plant)

    # ------------------------------------------------------------------
    # Table — a thin static box providing the collision ground plane
    # ------------------------------------------------------------------
    table_friction = ad.CoulombFriction(static_friction=0.6, dynamic_friction=0.5)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        ad.RigidTransform([0.0, 0.0, -0.05]),
        ad.Box(2.0, 2.0, 0.1),
        "table_collision",
        table_friction,
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        ad.RigidTransform([0.0, 0.0, -0.05]),
        ad.Box(2.0, 2.0, 0.1),
        "table_visual",
        [0.85, 0.80, 0.65, 1.0],
    )

    # ------------------------------------------------------------------
    # Franka Panda arm (7 revolute joints, welded base)
    # ------------------------------------------------------------------
    panda_file = "package://drake_models/franka_description/urdf/panda_arm.urdf"
    panda_model = parser.AddModelsFromUrl(panda_file)[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", panda_model),
        ad.RigidTransform(ROBOT_BASE_XYZ),
    )

    # ------------------------------------------------------------------
    # Spherical pusher — rigidly welded to panda_link8, 5 cm along +z.
    # Gives clean point contact with well-defined horizontal normals when
    # approaching the box from the side (Dairlab C3 benchmark geometry).
    # Fixed joint: no new DOF — q_arm stays 7-dim.
    # ------------------------------------------------------------------
    _pusher_inertia = ad.SpatialInertia(
        mass=0.05,
        p_PScm_E=np.zeros(3),
        G_SP_E=ad.UnitInertia.SolidSphere(PUSHER_RADIUS),
    )
    pusher_body = plant.AddRigidBody(EE_BODY_NAME, panda_model, _pusher_inertia)
    plant.RegisterCollisionGeometry(
        pusher_body,
        ad.RigidTransform(),
        ad.Sphere(PUSHER_RADIUS),
        "pusher_collision",
        ad.CoulombFriction(static_friction=0.4, dynamic_friction=0.4),
    )
    plant.RegisterVisualGeometry(
        pusher_body,
        ad.RigidTransform(),
        ad.Sphere(PUSHER_RADIUS),
        "pusher_visual",
        [0.2, 0.5, 1.0, 1.0],
    )
    plant.WeldFrames(
        plant.GetFrameByName("panda_link8", panda_model),
        pusher_body.body_frame(),
        ad.RigidTransform([0.0, 0.0, 0.05]),   # 5 cm past link8 along +z
    )

    # ------------------------------------------------------------------
    # Manipulated object — generated from task config at runtime
    # ------------------------------------------------------------------
    obj_type = task_cfg["object_type"]
    if obj_type == "box":
        sdf_str = _box_sdf(task_cfg)
    elif obj_type == "sphere":
        sdf_str = _sphere_sdf(task_cfg)
    else:
        raise ValueError(
            f"Unknown object_type '{obj_type}' in task config. Use 'box' or 'sphere'."
        )

    object_model = parser.AddModelsFromString(sdf_str, "sdf")[0]

    plant.Finalize()

    # ------------------------------------------------------------------
    # Meshcat visualiser
    # ------------------------------------------------------------------
    meshcat = ad.StartMeshcat()
    ad.MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    return diagram, plant, panda_model, object_model, meshcat


# ---------------------------------------------------------------------------
# Prepositioned-pose IK (push-direction-aware)
# ---------------------------------------------------------------------------

def compute_prepositioned_arm_q(plant,
                                plant_ctx,
                                panda_model,
                                ee_frame,
                                obj_body,
                                task_cfg: dict,
                                *,
                                contact_clearance: float = +0.001,
                                intermediate_z:    float = 0.25,
                                seed_arm_q:        np.ndarray = None,
                                verbose:           bool = True) -> np.ndarray:
    """Solve IK so the pusher rests on the object face opposite the goal.

    Why this exists
    ---------------
    The default `contact_clearance` of +1 mm seats the pusher just outside
    the object (1 mm gap, no interpenetration). The LCS contact filter's
    `distance_threshold=0.10 m` still returns this pair, so `_last_nhats`
    is non-empty and the alignment bonus on `c_sample[k=0]` still fires —
    `decide_mode` takes the `kToC3Cost` branch on step 0 just as it did
    with the previous -2 mm seed. The change avoids the contact-separation
    impulse the previous -2 mm penetration produced at t=0, which (coupled
    with an off-limit IK solution at joint 6) launched the box off the
    table within 0.5 s.

    A two-stage IK cascade (lifted waypoint, then descend) is used because
    `INITIAL_ARM_Q` parks the EE ~0.7 m from the WEST/NORTH targets and a
    single DLS pass gets stuck in local minima for those directions.

    Parameters
    ----------
    contact_clearance : float
        Signed gap between the pusher surface and the object face along
        `g_hat`. Positive (default +1 mm) places the pusher just outside
        contact; the LCS filter still captures the pair so the
        alignment-bonus path through `decide_mode` is unchanged. The
        previous default of -2 mm caused a contact-separation impulse on
        the very first integration step.
    """
    # Local import keeps env_builder.py free of control/* deps when this
    # function isn't called.
    from control.sampling_c3.ik import solve_ik_to_ee_pos

    init_xyz = np.asarray(task_cfg["init_xyz"], dtype=float)
    goal_xy  = np.asarray(task_cfg["goal_xy"],  dtype=float)
    obj_xy   = init_xyz[:2]
    delta    = goal_xy - obj_xy
    norm     = float(np.linalg.norm(delta))
    if norm < 1e-9:
        raise ValueError(
            "compute_prepositioned_arm_q: goal coincides with object init "
            "position — push direction undefined."
        )
    g_hat = delta / norm

    obj_type = task_cfg["object_type"]
    if obj_type == "box":
        sx, sy, _sz = task_cfg["size"]
        half_extent = abs(g_hat[0]) * sx / 2.0 + abs(g_hat[1]) * sy / 2.0
    elif obj_type == "sphere":
        half_extent = float(task_cfg["radius"])
    else:
        raise ValueError(
            f"compute_prepositioned_arm_q: unknown object_type '{obj_type}' "
            "(expected 'box' or 'sphere')."
        )

    contact_offset = half_extent + PUSHER_RADIUS + contact_clearance
    p_target_xy = obj_xy - contact_offset * g_hat
    p_target    = np.array([p_target_xy[0], p_target_xy[1], init_xyz[2]])

    seed = INITIAL_ARM_Q if seed_arm_q is None else np.asarray(seed_arm_q, float)
    plant.SetPositions(plant_ctx, panda_model, seed)
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), init_xyz.tolist()),
    )

    n_arm_dofs = plant.num_actuators()
    q_full     = plant.GetPositions(plant_ctx).copy()

    # Slice to n_arm_dofs: the floating-base object DOFs (xyz + quat) carry
    # ±inf limits in Drake, which we don't want to propagate as a clip
    # target inside the arm IK.
    q_lo_arm = plant.GetPositionLowerLimits()[:n_arm_dofs]
    q_hi_arm = plant.GetPositionUpperLimits()[:n_arm_dofs]

    # Stage 1: lifted waypoint to escape any pose-induced local minima.
    p_waypoint = np.array([p_target[0], p_target[1], intermediate_z])
    q1, err1, it1 = solve_ik_to_ee_pos(
        plant, ee_frame, p_waypoint, q_full, plant_ctx,
        n_arm_dofs=n_arm_dofs, max_iter=80, damping=0.05,
        q_lo=q_lo_arm, q_hi=q_hi_arm,
    )

    # Stage 2: descend onto the contact target.
    q2, err2, it2 = solve_ik_to_ee_pos(
        plant, ee_frame, p_target, q1, plant_ctx,
        n_arm_dofs=n_arm_dofs, max_iter=80, damping=0.02,
        q_lo=q_lo_arm, q_hi=q_hi_arm,
    )

    # Read EE position at the final iterate for the diagnostic line.
    ee_after = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame(),
    ).flatten()

    if verbose:
        print(
            f"[ENV]  --prepositioned: g_hat={g_hat.round(3).tolist()} "
            f"target={p_target.round(4).tolist()} "
            f"ee_after_ik={ee_after.round(4).tolist()} "
            f"ik_err=(stage1={err1*1000:.2f}mm/{it1}it, "
            f"stage2={err2*1000:.2f}mm/{it2}it)"
        )
        if err2 > 5e-3:
            print(
                f"[ENV]  WARN stage-2 IK error {err2*1000:.2f}mm > 5mm — "
                "contact may not be captured at t=0. Try "
                "seed_arm_q=_LEGACY_PREPOSITIONED_ARM_Q or raise intermediate_z."
            )

    return q2[:n_arm_dofs]
