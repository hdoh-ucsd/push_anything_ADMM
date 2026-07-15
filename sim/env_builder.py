"""
Generic Drake environment builder for all three manipulation tasks.

Builds: table (static) + Franka Panda arm + task-specific manipulable object.
Object geometry (box vs sphere) and physical properties come from tasks.yaml.
"""
import numpy as np
import pydrake.all as ad

# Panda base weld: arm sits 0.6 m behind table centre along -Y
ROBOT_BASE_XYZ = [0.0, -0.6, 0.0]

# IK seed for compute_safe_init_arm_q. Not used directly as the start
# pose — `compute_safe_init_arm_q` runs IK from this seed to place the EE
# at a task-specific safe-offset position (opposite goal direction, above
# object top). Retained for continuity with prior compute_prepositioned
# IK cascades.
_INITIAL_ARM_Q_SEED = np.array([
    +0.552150, +0.675037, +0.976275, -2.246164, -0.188979, +3.044706, +0.785000,
])
# FK of the seed: EE at (0.000, 0.000, 0.200) — this is the OLD default
# that placed EE directly over box CoM and caused the descend-through-box
# bug. Kept only as an IK warm-start seed; runtime uses IK-derived pose.

# Dedicated pusher body — spherical puck rigidly welded to panda_link8.
# This is the single authoritative name for EE body and contact filter.
EE_BODY_NAME  = "pusher"
_PUSHER_RADIUS_DEFAULT = 0.0195  # m — matches reference end_effector_full.urdf sphere radius (push_anything_dev@257e3ed).
# NOTE: §9 attempted to globally shrink this to 0.0195 to match DAIR push_t's
# end_effector_full.urdf, but that regressed the box's 72 % banked closure to
# ~39 % (contact geometry / setback / sampling_setback all reference this
# constant). If push_t needs 0.0195, plumb it as task-configurable via
# task_cfg["pusher_radius"] rather than mutating the global.
#
# Runtime override — set env PUSHA_PUSHER_RADIUS=<meters> to change the pusher
# tip radius for a single run without editing the source. Used by the STEP 3
# reconcile-run in the d_push-fix arc (2026-07-09) to test whether the
# 19.5 mm-sphere regression is a d_push-penetration interaction. Default-OFF.
def _effective_pusher_radius() -> float:
    import os as _os
    v = _os.environ.get("PUSHA_PUSHER_RADIUS", "").strip()
    if not v:
        return _PUSHER_RADIUS_DEFAULT
    try:
        return float(v)
    except ValueError:
        return _PUSHER_RADIUS_DEFAULT
PUSHER_RADIUS = _effective_pusher_radius()


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
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>3.0e7</drake:hydroelastic_modulus>
          <drake:mesh_resolution_hint>0.18</drake:mesh_resolution_hint>
          <drake:hunt_crossley_dissipation>10</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{mu}</drake:mu_dynamic>
        </drake:proximity_properties>
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
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>3.0e7</drake:hydroelastic_modulus>
          <drake:mesh_resolution_hint>0.18</drake:mesh_resolution_hint>
          <drake:hunt_crossley_dissipation>10</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{mu}</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <geometry><sphere><radius>{rad}</radius></sphere></geometry>
        <material><diffuse>{cr} {cg} {cb} {ca}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def _tshape_sdf(cfg: dict) -> str:
    """Reference push_t.sdf ported as a single-body-collapsed rigid.

    The reference (examples/sampling_c3/urdf/push_t.sdf) uses two 0.16×0.04×0.04
    box links (vertical_link + horizontal_link) joined by a fixed joint. A fixed
    joint is a rigid connection, so a single link with two collision elements is
    DYNAMICALLY EQUIVALENT to the two-body construction. This preserves the
    port's single-body-obj assumption without any fidelity loss.

    Link origin placed at the T's combined CoM (both links 0.5 kg, one at (0,0,0),
    one at (-0.10,0,0) in the reference frame → combined CoM at (-0.05, 0, 0)).
    In this LINK frame:
      - vertical bar (crossbar) collision at pose (+0.05, 0, 0)
      - horizontal bar (stem) collision at pose (-0.05, 0, 0, 0, 0, π/2)
    Both bars 0.16×0.04×0.04. Total mass 1.0 kg. Principal inertias computed
    about the combined CoM with parallel axis for each half (horizontal_link's
    principal axes swap ixx↔iyy in the T frame after its 90° z-rotation):
      ixx = 1.267e-3, iyy = 3.767e-3, izz = 4.767e-3 kg·m²
    """
    m  = cfg["mass"]
    mu = cfg["friction"]
    r, g, b, a = cfg["color_rgba"]
    # Precomputed for the reference T geometry with mass 1.0 kg split 0.5/0.5.
    # For arbitrary mass m, scale linearly (uniform density scales inertia ∝ m).
    scale = m / 1.0
    ixx = 1.267e-3 * scale
    iyy = 3.767e-3 * scale
    izz = 4.767e-3 * scale
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="manipulated_object">
    <link name="t_link">
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>{m}</mass>
        <inertia>
          <ixx>{ixx:.6f}</ixx><iyy>{iyy:.6f}</iyy><izz>{izz:.6f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="vertical_bar">
        <pose>0.05 0 0 0 0 0</pose>
        <geometry><box><size>0.16 0.04 0.04</size></box></geometry>
        <surface>
          <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>
        </surface>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>3.0e7</drake:hydroelastic_modulus>
          <drake:mesh_resolution_hint>0.18</drake:mesh_resolution_hint>
          <drake:hunt_crossley_dissipation>10</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{mu}</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <collision name="horizontal_bar">
        <pose>-0.05 0 0 0 0 1.5708</pose>
        <geometry><box><size>0.16 0.04 0.04</size></box></geometry>
        <surface>
          <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>
        </surface>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>3.0e7</drake:hydroelastic_modulus>
          <drake:mesh_resolution_hint>0.18</drake:mesh_resolution_hint>
          <drake:hunt_crossley_dissipation>10</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{mu}</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="vertical_bar_visual">
        <pose>0.05 0 0 0 0 0</pose>
        <geometry><box><size>0.16 0.04 0.04</size></box></geometry>
        <material><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
      <visual name="horizontal_bar_visual">
        <pose>-0.05 0 0 0 0 1.5708</pose>
        <geometry><box><size>0.16 0.04 0.04</size></box></geometry>
        <material><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_environment(task_cfg: dict, time_step: float = 0.001,
                      *, add_camera: bool = False,
                      camera_xyz=(-0.10, -0.05, 1.05),
                      camera_width: int = 1280,
                      camera_height: int = 720,
                      camera_fov_y_deg: float = 55.0,
                      goal_ghost_rgba=(0.10, 0.90, 0.10, 0.45)):
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
    plant_ad     : AutoDiffXd structural copy of `plant`. Used by
                   LCSFormulator.extract_dynamics_with_jacobian to compute
                   J_f = ∂f/∂(q,v,u) per Aydinoglu 2024 eq. (8). Built
                   once at startup; one ToAutoDiffXd() call costs ~50ms,
                   amortised over thousands of MPC steps.
    context_ad   : default context for plant_ad. Reused across linearize
                   calls (positions/velocities reset per call).
    """
    builder = ad.DiagramBuilder()
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(builder, time_step=time_step)

    parser = ad.Parser(plant)

    # ------------------------------------------------------------------
    # Table — a thin static box providing the collision ground plane
    # ------------------------------------------------------------------
    # Reference ground.urdf: box 5×0.91×0.1 m, μ_static=μ_dynamic=1.0.
    table_friction = ad.CoulombFriction(static_friction=1.0, dynamic_friction=1.0)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        ad.RigidTransform([0.0, 0.0, -0.05]),
        ad.Box(5.0, 0.91, 0.1),
        "table_collision",
        table_friction,
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        ad.RigidTransform([0.0, 0.0, -0.05]),
        ad.Box(5.0, 0.91, 0.1),
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
    # Pusher-surface friction is task-configurable. Drake combines per-surface
    # μ via harmonic mean: μ_eff = 2·μ_A·μ_B / (μ_A + μ_B). Reference
    # end_effector_full.urdf: pusher μ=1.0. Reference push_t.sdf: T μ=0.3;
    # combined EE-T μ_eff = 2·1.0·0.3/(1.3) = 0.462 (reproduces reference
    # EE-T pair exactly). Box tasks keep box μ=0.3 → box-EE μ_eff=0.462 too.
    _pusher_mu = float(task_cfg.get("pusher_friction", 1.0))
    plant.RegisterCollisionGeometry(
        pusher_body,
        ad.RigidTransform(),
        ad.Sphere(PUSHER_RADIUS),
        "pusher_collision",
        ad.CoulombFriction(static_friction=_pusher_mu, dynamic_friction=_pusher_mu),
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
    elif obj_type == "tshape":
        sdf_str = _tshape_sdf(task_cfg)
    else:
        raise ValueError(
            f"Unknown object_type '{obj_type}' in task config. "
            "Use 'box', 'sphere', or 'tshape'."
        )

    object_model = parser.AddModelsFromString(sdf_str, "sdf")[0]

    # Goal ghost (illustration-only translucent box at goal pose).  Anchored
    # to world_body so it shows in the VTK render alongside the opaque box.
    # Only registered when add_camera=True; non-render runs keep the scene clean.
    if add_camera:
        _goal_xy = task_cfg.get("goal_xy", [0.3, 0.0])
        _init_z  = task_cfg["init_xyz"][2]
        if task_cfg["object_type"] == "box":
            _sx, _sy, _sz = task_cfg["size"]
            _ghost_shape = ad.Box(_sx, _sy, _sz)
            plant.RegisterVisualGeometry(
                plant.world_body(),
                ad.RigidTransform([float(_goal_xy[0]), float(_goal_xy[1]), float(_init_z)]),
                _ghost_shape,
                "goal_ghost",
                list(goal_ghost_rgba),
            )
        elif task_cfg["object_type"] == "tshape":
            # Ghost T: two boxes at the T's collision poses. Yaw the goal ghost
            # by task_cfg.get("goal_yaw", 0.0) so the operator sees the target
            # orientation, not just position.
            _goal_yaw = float(task_cfg.get("goal_yaw", 0.0))
            _R_goal = ad.RotationMatrix.MakeZRotation(_goal_yaw)
            _T_goal = ad.RigidTransform(_R_goal,
                [float(_goal_xy[0]), float(_goal_xy[1]), float(_init_z)])
            for _local_x, _local_yaw, _tag in (
                (+0.05, 0.0,    "goal_ghost_vbar"),
                (-0.05, 1.5708, "goal_ghost_hbar"),
            ):
                _R_local = ad.RotationMatrix.MakeZRotation(_local_yaw)
                _T_local = ad.RigidTransform(_R_local, [_local_x, 0.0, 0.0])
                plant.RegisterVisualGeometry(
                    plant.world_body(),
                    _T_goal.multiply(_T_local),
                    ad.Box(0.16, 0.04, 0.04),
                    _tag,
                    list(goal_ghost_rgba),
                )
        else:
            _ghost_shape = ad.Sphere(task_cfg["radius"])
            plant.RegisterVisualGeometry(
                plant.world_body(),
                ad.RigidTransform([float(_goal_xy[0]), float(_goal_xy[1]), float(_init_z)]),
                _ghost_shape,
                "goal_ghost",
                list(goal_ghost_rgba),
            )

    plant.Finalize()

    # ------------------------------------------------------------------
    # Meshcat visualiser
    # ------------------------------------------------------------------
    meshcat = ad.StartMeshcat()
    ad.MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # OOM-safe Drake VTK render camera (top-down).  Frames pulled per-tick
    # in main.py and written to disk one-at-a-time -> ffmpeg encode offline.
    # Real Drake 3D scene (mesh-level) — distinct from any matplotlib
    # 2D render path; renders scene_graph, so /goal_ghost (registered above)
    # appears in every captured PNG.
    if add_camera:
        from pydrake.geometry import (
            MakeRenderEngineVtk, RenderEngineVtkParams,
            ClippingRange, DepthRange,
            RenderCameraCore, ColorRenderCamera, DepthRenderCamera,
        )
        from pydrake.systems.sensors import CameraInfo
        scene_graph.AddRenderer("drake_render_vtk",
                                MakeRenderEngineVtk(RenderEngineVtkParams()))
        intrinsics = CameraInfo(width=camera_width, height=camera_height,
                                fov_y=np.radians(camera_fov_y_deg))
        core = RenderCameraCore("drake_render_vtk", intrinsics,
                                ClippingRange(0.05, 5.0), ad.RigidTransform())
        color_cam = ColorRenderCamera(core, show_window=False)
        depth_cam = DepthRenderCamera(core, DepthRange(0.1, 5.0))
        # Top-down: camera at camera_xyz, looking along world -Z.
        # Rotation: 180 deg about X — world +X -> image right, world +Y -> image up.
        #
        # §7.73d — PUSHA_CAMERA_PERSPECTIVE=1 replaces the top-down pose
        # with an oblique perspective (elevated SE, looking at mid-scene)
        # so the box's out-of-plane pitch reads. Default-OFF preserves
        # the top-down capture used for the deck's original take-1.
        import os as _os_cam
        if _os_cam.environ.get("PUSHA_CAMERA_PERSPECTIVE", "0") == "1":
            def _parse_triple_env(var, default):
                v = _os_cam.environ.get(var)
                if not v:
                    return np.asarray(default, dtype=float)
                try:
                    return np.asarray([float(x) for x in v.split(",")],
                                      dtype=float)
                except Exception:
                    return np.asarray(default, dtype=float)
            _cp     = _parse_triple_env("PUSHA_CAM_EYE",    [0.30, -0.55, 0.45])
            _target = _parse_triple_env("PUSHA_CAM_TARGET", [-0.15, 0.0, 0.05])
            _fwd = _target - _cp; _fwd = _fwd / np.linalg.norm(_fwd)
            _world_up = np.array([0.0, 0.0, 1.0])
            # Right-hand rule: right = fwd × world_up (viewer facing fwd with
            # head towards world_up → right hand points to world +right).
            # The old (world_up × fwd) inverted _right AND propagated to
            # _down, flipping world +Z onto image DOWN — the §7.73 clip's
            # upside-down artifact.  Drake camera axes are +X right, +Y down,
            # +Z forward, so _R = [right | down | fwd].
            _right = np.cross(_fwd, _world_up)
            _right = _right / np.linalg.norm(_right)
            _down = np.cross(_fwd, _right)
            _R = np.column_stack([_right, _down, _fwd])
            X_PB = ad.RigidTransform(ad.RotationMatrix(_R), _cp.tolist())
            print(f"[C3] PUSHA_CAMERA_PERSPECTIVE=1  cam@{_cp.tolist()}  "
                  f"target@{_target.tolist()}", flush=True)
        else:
            X_PB = ad.RigidTransform(ad.RotationMatrix.MakeXRotation(np.pi),
                                     list(camera_xyz))
        rgbd = builder.AddSystem(
            ad.RgbdSensor(parent_id=scene_graph.world_frame_id(),
                          X_PB=X_PB,
                          color_camera=color_cam, depth_camera=depth_cam))
        rgbd.set_name("drake_render_camera")
        builder.Connect(scene_graph.get_query_output_port(),
                        rgbd.query_object_input_port())

    diagram = builder.Build()

    # ------------------------------------------------------------------
    # AutoDiffXd structural copy of the plant — Phase 1 (Aydinoglu eq. 8).
    # Used by LCSFormulator to compute J_f = ∂f/∂(q,v,u) at each MPC step.
    # ToAutoDiffXd preserves all geometry/contact registration; the
    # context is reused across linearize calls.
    # ------------------------------------------------------------------
    plant_ad   = plant.ToAutoDiffXd()
    context_ad = plant_ad.CreateDefaultContext()

    return diagram, plant, panda_model, object_model, meshcat, plant_ad, context_ad


# ---------------------------------------------------------------------------
# Prepositioned-pose IK (push-direction-aware)
# ---------------------------------------------------------------------------

def compute_safe_init_arm_q(plant,
                            plant_ctx,
                            panda_model,
                            ee_frame,
                            obj_body,
                            task_cfg: dict,
                            *,
                            safe_xy_offset:  float = 0.15,
                            safe_z_margin:   float = 0.05,
                            intermediate_z:  float = 0.30,
                            seed_arm_q:      np.ndarray = None,
                            verbose:         bool = True) -> np.ndarray:
    """Solve IK to place the pusher OPPOSITE the goal at safe altitude.

    Places EE at:
      xy = obj_xy - g_hat * (object_half_extent + safe_xy_offset)
      z  = object_top + safe_z_margin

    This avoids the descend-through-box bug (prior INITIAL_ARM_Q's
    EE-at-(0,0,0.2) put the sphere directly over the box; PWL Phase-1
    straight-down descent passed through box top since pwl_waypoint_height
    < box_top). By starting OFFSET in xy on the goal-opposite side and
    ABOVE box top, the first PWL lift/descend has clear space.
    """
    from control.sampling_c3.ik import solve_ik_to_ee_pos

    init_xyz = np.asarray(task_cfg["init_xyz"], dtype=float)
    goal_xy  = np.asarray(task_cfg["goal_xy"],  dtype=float)
    obj_xy   = init_xyz[:2]
    delta    = goal_xy - obj_xy
    norm     = float(np.linalg.norm(delta))
    if norm < 1e-9:
        raise ValueError(
            "compute_safe_init_arm_q: goal coincides with object init "
            "position — push direction undefined."
        )
    g_hat = delta / norm

    obj_type = task_cfg["object_type"]
    if obj_type == "box":
        sx, sy, sz = task_cfg["size"]
        half_extent = abs(g_hat[0]) * sx / 2.0 + abs(g_hat[1]) * sy / 2.0
        obj_top_z   = init_xyz[2] + sz / 2.0
    elif obj_type == "sphere":
        r = float(task_cfg["radius"])
        half_extent = r
        obj_top_z   = init_xyz[2] + r
    elif obj_type == "tshape":
        # T occupies x∈[-0.07,+0.13], y∈[-0.08,+0.08], z half-extent 0.02.
        half_extent = abs(g_hat[0]) * 0.13 + abs(g_hat[1]) * 0.08
        obj_top_z   = init_xyz[2] + 0.02
    else:
        raise ValueError(
            f"compute_safe_init_arm_q: unknown object_type '{obj_type}' "
            "(expected 'box', 'sphere', or 'tshape')."
        )

    # SAFE-OFFSET target: xy offset opposite goal direction by
    # (object half-extent + safe_xy_offset), z above object top by margin.
    safe_offset = half_extent + PUSHER_RADIUS + safe_xy_offset
    p_target_xy = obj_xy - safe_offset * g_hat
    p_target_z  = obj_top_z + safe_z_margin
    p_target    = np.array([p_target_xy[0], p_target_xy[1], p_target_z])

    seed = _INITIAL_ARM_Q_SEED if seed_arm_q is None else np.asarray(seed_arm_q, float)
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
            f"[ENV]  safe-init pose: g_hat={g_hat.round(3).tolist()} "
            f"target={p_target.round(4).tolist()} "
            f"ee_after_ik={ee_after.round(4).tolist()} "
            f"ik_err=(stage1={err1*1000:.2f}mm/{it1}it, "
            f"stage2={err2*1000:.2f}mm/{it2}it)"
        )
        if err2 > 5e-3:
            print(
                f"[ENV]  WARN stage-2 IK error {err2*1000:.2f}mm > 5mm — "
                "safe init pose may be off. Raise intermediate_z or "
                "check task_cfg."
            )

    return q2[:n_arm_dofs]
