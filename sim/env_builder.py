"""
Generic Drake environment builder for all three manipulation tasks.

Builds: table (static) + Franka Panda arm + task-specific manipulable object.
Object geometry (box vs sphere) and physical properties come from tasks.yaml.
"""
import numpy as np
import pydrake.all as ad

# Panda base weld: arm sits 0.6 m behind table centre along -Y and 29 mm
# above ground/table top (matches reference kFrankaToGroundOffset = (0, 0,
# -0.029) at sampling_c3_utils.h:32 — reference has Franka mounted 29 mm
# above the ground origin). Prior port had z=0.0, mounting the arm flush
# on the table top. The 29 mm elevation gives the arm the vertical range
# reference relies on when tracking sampling_height / z_height at 0.002-0.005
# (targets that would otherwise be below the port's tip-below-base reach
# in the R^7 → R^3 EE-space projection).
# 2026-07-21: aligned to reference sampling_c3_utils.cc:24-26
# `plant->WeldFrames(world, panda_link0, X_WI=Identity)` — Franka base
# at world origin. Reference operates with T-init at (0.5, 0, -0.009)
# (0.5 m directly +x from arm) and goal at (0.482, 0.187, -0.009).
# The 29 mm z offset is retained (was port-only elevation for tip-
# below-base reach; reference has ground at z=-0.029 and arm at z=0,
# equivalent 29mm arm-above-ground separation).
# Prior port value (0, -0.6, 0.029) placed arm 0.6 m in -y direction
# from origin, forcing T-push operations to happen in the arm's rear
# workspace where +y-reach is kinematically limited to ~0.30 m
# (workspace_xy_max_y=0.30 = the dominant failure envelope).
ROBOT_BASE_XYZ = [0.0, 0.0, 0.029]

# IK seed for compute_safe_init_arm_q. Not used directly as the start
# pose — `compute_safe_init_arm_q` runs IK from this seed to place the EE
# at a task-specific safe-offset position (opposite goal direction, above
# object top). Retained for continuity with prior compute_prepositioned
# IK cascades.
# Joint 2 (index 1) at 1.1 rad matches reference q_init_franka[1] = 1.1
# (osc_params.yaml:44 elbow_kp target). Reference has zero posture error at
# init; port previously had 0.675 giving a 0.43 rad error at t=0 which
# joint-2 tracking (W=1, Kp=200) drove upward continuously → arm-fly-up
# (ee_z reached 0.809 m in box_bugfix_100749, 0.624 m in box_frame3d_101625).
# Aligning the IK seed pose with the joint-2 target zeros the posture error
# at start so the OSC has no forced upward pull.
_INITIAL_ARM_Q_SEED = np.array([
    +0.552150, +1.100000, +0.976275, -2.246164, -0.188979, +3.044706, +0.785000,
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
# Runtime override — set env PORT_PUSHER_RADIUS=<meters> to change the pusher
# tip radius for a single run without editing the source. Used by the STEP 3
# reconcile-run in the d_push-fix arc (2026-07-09) to test whether the
# 19.5 mm-sphere regression is a d_push-penetration interaction. Default-OFF.
def _effective_pusher_radius() -> float:
    import os as _os
    v = _os.environ.get("PORT_PUSHER_RADIUS", "").strip()
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
        <!-- 2026-07-22: <surface><friction><ode><mu*> tags removed.
             They were duplicating the friction data already declared in
             <drake:proximity_properties><drake:mu_dynamic>, and Drake
             silently ignored them ("...tags are ignored" warning per
             collision element on every load). Only drake:mu_dynamic is
             consumed for LCS friction; the ODE mu tags contributed
             nothing except log noise. -->
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
        <!-- 2026-07-22: <surface><friction><ode><mu*> tags removed.
             They were duplicating the friction data already declared in
             <drake:proximity_properties><drake:mu_dynamic>, and Drake
             silently ignored them ("...tags are ignored" warning per
             collision element on every load). Only drake:mu_dynamic is
             consumed for LCS friction; the ODE mu tags contributed
             nothing except log noise. -->
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
        <!-- 2026-07-22: <surface><friction><ode><mu*> tags removed.
             They were duplicating the friction data already declared in
             <drake:proximity_properties><drake:mu_dynamic>, and Drake
             silently ignored them ("...tags are ignored" warning per
             collision element on every load). Only drake:mu_dynamic is
             consumed for LCS friction; the ODE mu tags contributed
             nothing except log noise. -->
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
        <!-- 2026-07-22: <surface><friction><ode><mu*> tags removed.
             They were duplicating the friction data already declared in
             <drake:proximity_properties><drake:mu_dynamic>, and Drake
             silently ignored them ("...tags are ignored" warning per
             collision element on every load). Only drake:mu_dynamic is
             consumed for LCS friction; the ODE mu tags contributed
             nothing except log noise. -->
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


def _hshape_sdf(cfg: dict) -> str:
    """Low-CoM H, dimensionally conformant to the reference H_shape_texture.

    Reference (examples/sampling_c3/urdf/H_shape_texture/) ships the H as four
    convex MESH pieces. Their combined bounding box is 0.111 x 0.128 x 0.032 m,
    with mass 1.0 kg and drake:mu_dynamic 0.3. The port has no mesh pipeline for
    the manipuland (the sampler face tables and the LCS ground witnesses are
    both analytic), so this reproduces the same envelope as three boxes, exactly
    the way _tshape_sdf collapses the reference's two-link T into one body:

        left  bar  x[-0.056,-0.032]  y[-0.064,+0.064]   0.024 x 0.128 x 0.032
        right bar  x[+0.032,+0.056]  y[-0.064,+0.064]   0.024 x 0.128 x 0.032
        crossbar   x[-0.032,+0.032]  y[-0.012,+0.012]   0.064 x 0.024 x 0.032

    Conformant to the reference H: total mass 1.0 kg, mu_dynamic 0.3, overall
    envelope, and the identical hydroelastic block used by both push_t.sdf and
    H_shape_texture.sdf (modulus 3.0e7, resolution hint 0.18, dissipation 10).

    DELIBERATE DEVIATION -- the low centre of mass. The reference H puts its CoM
    at the geometric centre (<inertial><pose>0 0 0</pose>, i.e. 16 mm above the
    base). `com_height_frac` (default 0.25) places it at 8 mm instead. Rationale:
    the EE presses at sampling_height, which for a 32 mm object sits at or above
    mid-height, so a centred CoM gives the push a tipping moment about the
    leading bottom edge. Dropping the CoM increases the gravitational restoring
    moment and biases the contact into sliding rather than rocking.

    Inertia is computed for the actual 3-box geometry at uniform density about
    the geometric centre, then parallel-axis reduced to the lowered CoM. That
    reduction models "same envelope, mass biased toward the base" (a weighted
    base plate); it is an approximation, since a genuinely redistributed body
    would also change the in-plane terms slightly. izz is unaffected by a
    z-shift and is therefore exact.
    """
    m  = cfg["mass"]
    mu = cfg["friction"]
    r, g, b, a = cfg["color_rgba"]
    # Fraction of the object's height at which the CoM sits (0.5 = geometric
    # centre = reference behaviour; lower = bottom-weighted).
    com_frac = float(cfg.get("com_height_frac", 0.25))

    HX, HY, HZ = 0.112, 0.128, 0.032          # overall envelope
    ax, ay, az = 0.024, HY,    HZ             # side bar
    cx, cy, cz = 0.064, 0.024, HZ             # crossbar
    d_bar      = 0.044                        # side-bar |x| offset

    scale = m / 1.0
    # Uniform-density masses for the 3-box decomposition (0.4/0.4/0.2 at 1 kg).
    Vb, Vc = ax * ay * az, cx * cy * cz
    mb = m * Vb / (2.0 * Vb + Vc)
    mc = m * Vc / (2.0 * Vb + Vc)

    def _box_I(mm, u, v, w):
        return (mm / 12.0 * (v * v + w * w),
                mm / 12.0 * (u * u + w * w),
                mm / 12.0 * (u * u + v * v))

    bx, by, bz = _box_I(mb, ax, ay, az)
    Cx, Cy, Cz = _box_I(mc, cx, cy, cz)
    ixx_gc = 2.0 * bx + Cx
    iyy_gc = 2.0 * (by + mb * d_bar ** 2) + Cy
    izz_gc = 2.0 * (bz + mb * d_bar ** 2) + Cz

    # CoM offset below the geometric centre (negative z in link frame).
    z_com = (com_frac - 0.5) * HZ
    ixx = ixx_gc - m * z_com ** 2
    iyy = iyy_gc - m * z_com ** 2
    izz = izz_gc

    def _collision(name, pose, size):
        return f"""      <collision name="{name}">
        <pose>{pose}</pose>
        <geometry><box><size>{size}</size></box></geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>3.0e7</drake:hydroelastic_modulus>
          <drake:mesh_resolution_hint>0.18</drake:mesh_resolution_hint>
          <drake:hunt_crossley_dissipation>10</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{mu}</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="{name}_vis">
        <pose>{pose}</pose>
        <geometry><box><size>{size}</size></box></geometry>
        <material><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>"""

    parts = "\n".join([
        _collision("left_bar",  f"{-d_bar} 0 0 0 0 0", f"{ax} {ay} {az}"),
        _collision("right_bar", f"{+d_bar} 0 0 0 0 0", f"{ax} {ay} {az}"),
        _collision("crossbar",  "0 0 0 0 0 0",         f"{cx} {cy} {cz}"),
    ])
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="manipulated_object">
    <link name="h_link">
      <inertial>
        <pose>0 0 {z_com:.6f} 0 0 0</pose>
        <mass>{m}</mass>
        <inertia>
          <ixx>{ixx:.6e}</ixx><iyy>{iyy:.6e}</iyy><izz>{izz:.6e}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
{parts}
    </link>
  </model>
</sdf>"""



def _jack_sdf(cfg: dict) -> str:
    """Reference jack.sdf ported as a single-body-collapsed rigid.

    Reference (examples/sampling_c3/urdf/jack.sdf) is three capsule LINKS welded
    by two fixed joints. A fixed joint is a rigid connection, so one link with
    three collision elements is dynamically equivalent -- the same collapse
    _tshape_sdf applies to the reference's two-link T.

    Geometry, verbatim from the reference: three capsules, radius 0.025,
    length 0.125, mass 0.052 each (0.156 kg total), mutually orthogonal --

        capsule_1  pose 0 0 0  0      0      0   -> axis +z
        capsule_2  pose 0 0 0  0      1.5708 0   -> axis +x
        capsule_3  pose 0 0 0  1.5708 0      0   -> axis +y

    A caltrop: six tips at +/-(0.0625 + 0.025) = +/-0.0875 along each body axis.

    This object is qualitatively unlike the T and H. It has no flat resting
    face -- it balances on three tips and ROLLS between tripods, so its ground
    contact set changes discontinuously with pose. That is why jacktoy uses
    sampling_strategy 2 (kRandomOnSphere) rather than the perimeter sampler.

    Inertia: each capsule is modelled about its own axis and the three are
    summed about the shared origin. Because the capsules are orthogonal and
    concentric the result is isotropic, which the reference's own per-link
    values (ixx 7.072e-5, izz 5.72e-6 each) reproduce once summed.
    """
    m_total = float(cfg.get("mass", 0.156))
    mu = float(cfg["friction"])
    r, g, b, a = cfg["color_rgba"]
    m_cap = m_total / 3.0
    RAD, LEN = 0.025, 0.125

    # Reference per-capsule inertia at 0.052 kg, scaled linearly with mass.
    scale = m_cap / 0.052
    i_perp = 7.072e-05 * scale    # about the two axes perpendicular to the capsule
    i_axis = 5.72e-06 * scale     # about the capsule's own axis
    # Three orthogonal, concentric capsules -> isotropic sum.
    I = 2.0 * i_perp + i_axis

    # Per-capsule colours matching reference urdf/jack.sdf exactly:
    #   capsule_1 (body +z) blue, capsule_2 (body +x) red, capsule_3 (body +y) green.
    # These are not decoration -- the reference's eight nominal goal orientations
    # are NAMED by them (kQuatRedUp, kQuatGreenDown, ...), each name being the
    # sign triple of the (red, green, blue) axes, so the colour tells you at a
    # glance which of the 8 tripods the jack is resting on. A single-colour jack
    # makes a render unreadable: you cannot see which way it rolled.
    _CAP_RGBA = {
        "capsule_1": (0.0, 0.0, 1.0, 1.0),   # blue  — body +z
        "capsule_2": (1.0, 0.0, 0.0, 1.0),   # red   — body +x
        "capsule_3": (0.0, 1.0, 0.0, 1.0),   # green — body +y
    }

    def _cap(name, pose):
        r, g, b, a = _CAP_RGBA[name]
        return f"""      <collision name="{name}">
        <pose>{pose}</pose>
        <geometry><capsule><radius>{RAD}</radius><length>{LEN}</length></capsule></geometry>
        <!-- Reference urdf/jack.sdf declares ONLY mu_dynamic here: no
             compliant_hydroelastic, no modulus, no dissipation. Unlike
             push_t.sdf and H_shape_texture.sdf, which do declare hydroelastic
             3.0e7, the jack is deliberately left as point contact. Copying the
             box/T/H block here would be non-conformant (it is also inert,
             since the port's pusher and table are registered with plain
             CoulombFriction and Drake's kHydroelasticWithFallback needs BOTH
             sides hydroelastic -- but inert-and-wrong still misleads). -->
        <drake:proximity_properties>
          <drake:mu_dynamic>{mu}</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="{name}_vis">
        <pose>{pose}</pose>
        <geometry><capsule><radius>{RAD}</radius><length>{LEN}</length></capsule></geometry>
        <material><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>"""

    parts = "\n".join([
        _cap("capsule_1", "0 0 0 0 0 0"),            # +z
        _cap("capsule_2", "0 0 0 0 1.5708 0"),       # +x
        _cap("capsule_3", "0 0 0 1.5708 0 0"),       # +y
    ])
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="manipulated_object">
    <link name="jack_link">
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>{m_total}</mass>
        <inertia>
          <ixx>{I:.6e}</ixx><iyy>{I:.6e}</iyy><izz>{I:.6e}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
{parts}
    </link>
  </model>
</sdf>"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_environment(task_cfg: dict, time_step: float | None = None,
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
    # Sim timestep. Reference sets it PER TASK in sim_params.yaml:
    #   push_t 0.0001, jacktoy 0.0001, anything 0.001
    # The port historically hardcoded 0.001 for everything, i.e. 10x coarser
    # than the reference for push_t and the jack. With Drake's discrete contact
    # solver the penetration accumulated before the solver responds scales with
    # the step, so a coarse step turns a light object's contacts into impacts.
    # An explicit `time_step` argument still wins (callers that pass one).
    if time_step is None:
        time_step = float(task_cfg.get("sim_dt", 0.001))
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    print(f"[ENV]  sim time_step = {time_step:g} s", flush=True)

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

    # Platform — reference platform.urdf (0.3×0.6×0.029 slab, "the wooden
    # and aluminium sheet on which the franka is mounted"), welded to
    # panda_link0 at kFrankaToPlatformOffset (0,0,-0.0145) with the box
    # geometry offset (-0.07,0,0) (sampling_c3_utils.{h,cc}). Port frame:
    # arm base at z=+0.029, ground top at z=0 → platform center world z
    # = 0.029-0.0145 = +0.0145; slab spans z∈[0,0.029] (top flush with
    # the arm base plane), x∈[-0.22,+0.08]. Collision per reference
    # (object cannot slide under the base area). 2026-07-28 mount-question
    # close-out.
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        ad.RigidTransform([-0.07, 0.0, 0.0145]),
        ad.Box(0.3, 0.6, 0.029),
        "platform_collision",
        ad.CoulombFriction(static_friction=1.0, dynamic_friction=1.0),
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        ad.RigidTransform([-0.07, 0.0, 0.0145]),
        ad.Box(0.3, 0.6, 0.029),
        "platform_visual",
        [0.55, 0.45, 0.35, 1.0],
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
    # Reference EE chain (dairlib_sampling_c3 examples/sampling_c3/urdf/
    # end_effector_full.urdf @257e3ed): flange cylinder + peg cylinder +
    # tip sphere welded to panda_link7 via kToolAttachmentFrame=[0,0,0.107]
    # with 180° roll flip. Total ~23 cm of physical separation between
    # panda_link7 and the tip sphere — arm-link collision geometries
    # (Drake stock panda_arm.urdf: r=0.06 spheres per link) can't reach
    # manipulands during normal EE-contact configurations. Reference does
    # NOT use explicit collision-filter code (verified in franka_sim.cc);
    # the geometric separation IS the filter.
    #
    # Prior port: bare sphere welded 5 cm past panda_link8. Too close to
    # arm — box_nobs_044007 showed the arm links pushing the box directly
    # (n_pairs=2-5 events during freeze regions, ee_box_normal=0 while
    # box_p continued to drift).
    # ------------------------------------------------------------------
    _pusher_mu = float(task_cfg.get("pusher_friction", 1.0))

    # Flange (cylinder r=0.0315, L=0.0096, mass 0.0779 kg per reference URDF).
    # Inertia uses a spherical approximation — the reference thin-disc cylinder
    # yields a degenerate SpatialInertia that fails Drake's triangle-inequality
    # check. Bodies are welded → rotational inertia is nearly inert for dynamics.
    # Place body frame AT the CoM (p_PScm_E=0) — Drake's SpatialInertia
    # constructor requires G_SP_E to be about the BODY FRAME ORIGIN, not Bcm;
    # keeping origin=Bcm avoids the parallel-axis shift trap.
    flange_body = plant.AddRigidBody(
        "end_effector_flange", panda_model,
        ad.SpatialInertia(
            mass=0.0779312,
            p_PScm_E=np.zeros(3),
            G_SP_E=ad.UnitInertia.SolidSphere(0.02),
        ),
    )
    # Reference LCS-plant architecture (franka_sampling_c3_controller.cc:130-138)
    # uses ONLY the tip sphere for planner contact — reference LCS plant is a
    # floating sphere without any peg or flange. To mimic that on the port's
    # single-plant setup we skip RegisterCollisionGeometry for the flange and
    # peg bodies (visual kept so meshcat still shows them). Sim plant no
    # longer simulates peg-box or flange-box contact, matching reference
    # planner's mental model. Prior port had all three collision geometries;
    # box_j2track_093539 showed 172 n_pairs=2 events driven by peg contact
    # (per user observation: "the box touched the bar that connects the tip
    # and arm").
    plant.RegisterVisualGeometry(
        flange_body,
        ad.RigidTransform(np.array([0.0, 0.0, -0.0048])),
        ad.Cylinder(0.0315, 0.0096),
        "flange_visual",
        [0.3, 0.3, 0.3, 1.0],
    )
    plant.WeldFrames(
        plant.GetFrameByName("panda_link7", panda_model),
        flange_body.body_frame(),
        ad.RigidTransform(
            ad.RotationMatrix(ad.RollPitchYaw(3.1415, 0.0, 0.0)),
            np.array([0.0, 0.0, 0.107]),
        ),
    )

    # Peg (cylinder r=0.0127, L=0.1016, mass 0.134 kg per reference URDF).
    # Long thin cylinder — spherical approximation for the inertia is used to
    # keep the SpatialInertia validation happy; welded bodies won't feel it.
    peg_body = plant.AddRigidBody(
        "end_effector_peg", panda_model,
        ad.SpatialInertia(
            mass=0.1340688,
            p_PScm_E=np.zeros(3),
            G_SP_E=ad.UnitInertia.SolidSphere(0.03),
        ),
    )
    # (see flange note above) — visual only, no collision.
    plant.RegisterVisualGeometry(
        peg_body,
        ad.RigidTransform(np.array([0.0, 0.0, -0.0508])),
        ad.Cylinder(0.0127, 0.1016),
        "peg_visual",
        [0.3, 0.3, 0.3, 1.0],
    )
    plant.WeldFrames(
        flange_body.body_frame(),
        peg_body.body_frame(),
        ad.RigidTransform(np.array([0.0, 0.0, -0.0096])),
    )

    # Tip sphere (reference r=0.0195, mass 0.057 kg). Kept as "pusher" name
    # so downstream code (LCS filter, wrapper, OSC) works unchanged.
    pusher_body = plant.AddRigidBody(
        EE_BODY_NAME, panda_model,
        ad.SpatialInertia(
            mass=0.057,
            p_PScm_E=np.zeros(3),
            G_SP_E=ad.UnitInertia.SolidSphere(PUSHER_RADIUS),
        ),
    )
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
        peg_body.body_frame(),
        pusher_body.body_frame(),
        ad.RigidTransform(np.array([0.0, 0.0, -0.1169])),
    )

    # ------------------------------------------------------------------
    # Manipulated object — generated from task config at runtime
    # ------------------------------------------------------------------
    obj_type = task_cfg["object_type"]
    # 2026-08-11 mesh-T migration: a task may carry `object_sdf` (path to a
    # reference SDF, e.g. sim/models/T_shape_video/T_shape_video.sdf). The
    # file is loaded verbatim (mesh collision pieces, inertia, friction from
    # the reference asset); all shape-keyed behavior stays on object_type.
    _obj_sdf_path = task_cfg.get("object_sdf", None)
    if _obj_sdf_path:
        object_model = parser.AddModels(str(_obj_sdf_path))[0]
    else:
        if obj_type == "box":
            sdf_str = _box_sdf(task_cfg)
        elif obj_type == "sphere":
            sdf_str = _sphere_sdf(task_cfg)
        elif obj_type == "tshape":
            sdf_str = _tshape_sdf(task_cfg)
        elif obj_type == "hshape":
            sdf_str = _hshape_sdf(task_cfg)
        elif obj_type == "jack":
            sdf_str = _jack_sdf(task_cfg)
        else:
            raise ValueError(
                f"Unknown object_type '{obj_type}' in task config. "
                "Use 'box', 'sphere', 'tshape', 'hshape', or 'jack'."
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
        elif task_cfg["object_type"] == "hshape":
            # Ghost H: the same three boxes _hshape_sdf builds.
            _R_goal = ad.RotationMatrix.MakeZRotation(
                float(task_cfg.get("goal_yaw", 0.0)))
            _T_goal = ad.RigidTransform(_R_goal,
                [float(_goal_xy[0]), float(_goal_xy[1]), float(_init_z)])
            for _lx, _sz, _tag in (
                (-0.044, (0.024, 0.128, 0.032), "goal_ghost_lbar"),
                (+0.044, (0.024, 0.128, 0.032), "goal_ghost_rbar"),
                ( 0.000, (0.064, 0.024, 0.032), "goal_ghost_cbar"),
            ):
                plant.RegisterVisualGeometry(
                    plant.world_body(),
                    _T_goal.multiply(ad.RigidTransform([_lx, 0.0, 0.0])),
                    ad.Box(*_sz), _tag, list(goal_ghost_rgba),
                )
        elif task_cfg["object_type"] == "jack":
            # Ghost jack: three orthogonal capsules at the goal POSE. The jack's
            # goal is a full quaternion (it rolls onto a different tripod), so
            # unlike every other task the ghost cannot be built from a yaw.
            # Golden tint, NOT the default green: the jack itself has a green
            # capsule (per-capsule RGB), and at goal the two overlap into one
            # unreadable green blob (2026-08-16 video-legibility root cause).
            _jack_ghost_rgba = (1.0, 0.80, 0.10, 0.45)
            _gq = task_cfg.get("goal_quat", [1.0, 0.0, 0.0, 0.0])
            _gq = np.asarray(_gq, dtype=float)
            _gq = _gq / float(np.linalg.norm(_gq))
            _T_goal = ad.RigidTransform(
                ad.RotationMatrix(ad.Quaternion(_gq[0], _gq[1], _gq[2], _gq[3])),
                [float(_goal_xy[0]), float(_goal_xy[1]), float(_init_z)])
            for _rpy, _tag in (
                ((0.0, 0.0, 0.0),    "goal_ghost_cap1"),
                ((0.0, 1.5708, 0.0), "goal_ghost_cap2"),
                ((1.5708, 0.0, 0.0), "goal_ghost_cap3"),
            ):
                _T_local = ad.RigidTransform(
                    ad.RotationMatrix(ad.RollPitchYaw(*_rpy)), [0.0, 0.0, 0.0])
                plant.RegisterVisualGeometry(
                    plant.world_body(),
                    _T_goal.multiply(_T_local),
                    ad.Capsule(0.025, 0.125), _tag, list(_jack_ghost_rgba),
                )
        elif "radius" in task_cfg:
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
        # §7.73d — PORT_CAMERA_PERSPECTIVE=1 replaces the top-down pose
        # with an oblique perspective (elevated SE, looking at mid-scene)
        # so the box's out-of-plane pitch reads. Default-OFF preserves
        # the top-down capture used for the deck's original take-1.
        import os as _os_cam
        if _os_cam.environ.get("PORT_CAMERA_PERSPECTIVE", "0") == "1":
            def _parse_triple_env(var, default):
                v = _os_cam.environ.get(var)
                if not v:
                    return np.asarray(default, dtype=float)
                try:
                    return np.asarray([float(x) for x in v.split(",")],
                                      dtype=float)
                except Exception:
                    return np.asarray(default, dtype=float)
            _cp     = _parse_triple_env("PORT_CAM_EYE",    [0.30, -0.55, 0.45])
            _target = _parse_triple_env("PORT_CAM_TARGET", [-0.15, 0.0, 0.05])
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
            print(f"[C3] PORT_CAMERA_PERSPECTIVE=1  cam@{_cp.tolist()}  "
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

        # 2026-07-22: Reference-style video path (opt-in via
        # PORT_USE_DRAKE_VIDEO_WRITER=1). Mirrors reference
        # examples/sampling_c3/process_lcm_logs.py:457-461 which uses
        # pydrake.visualization.VideoWriter with backend="cv2". Drake
        # accumulates frames from the RgbdSensor at `fps` automatically;
        # main.py calls video_writer.Save() at end-of-sim. Alternative
        # to the port's default PNG-per-tick + ffmpeg pipeline (which
        # remains active when the env var is unset).
        if _os_cam.environ.get("PORT_USE_DRAKE_VIDEO_WRITER", "0") == "1":
            from pydrake.visualization import VideoWriter as _DrakeVideoWriter
            _vid_fps = float(_os_cam.environ.get(
                "PORT_DRAKE_VIDEO_FPS", "16.0"))
            # Reference uses backend="cv2" (process_lcm_logs.py:459).
            # Requires opencv-python installed (`pip install opencv-python`).
            # Fallback: PIL backend supports gif/apng/webp but NOT mp4.
            _vid_backend = _os_cam.environ.get(
                "PORT_DRAKE_VIDEO_BACKEND", "cv2")
            _vid_filename = _os_cam.environ.get(
                "PORT_DRAKE_VIDEO_FILENAME",
                "results/_drake_video_writer_output.mp4")
            _drake_video_writer = _DrakeVideoWriter(
                filename=_vid_filename, fps=_vid_fps, backend=_vid_backend)
            builder.AddSystem(_drake_video_writer)
            _drake_video_writer.ConnectRgbdSensor(
                builder=builder, sensor=rgbd)
            print(f"[C3] PORT_USE_DRAKE_VIDEO_WRITER=1 — "
                  f"pydrake.visualization.VideoWriter added "
                  f"(filename={_vid_filename} fps={_vid_fps} "
                  f"backend={_vid_backend})  ref: "
                  f"process_lcm_logs.py:457-461", flush=True)
        else:
            _drake_video_writer = None

    diagram = builder.Build()

    # ------------------------------------------------------------------
    # AutoDiffXd structural copy of the plant — Phase 1 (Aydinoglu eq. 8).
    # Used by LCSFormulator to compute J_f = ∂f/∂(q,v,u) at each MPC step.
    # ToAutoDiffXd preserves all geometry/contact registration; the
    # context is reused across linearize calls.
    # ------------------------------------------------------------------
    plant_ad   = plant.ToAutoDiffXd()
    context_ad = plant_ad.CreateDefaultContext()

    # 2026-07-22: 8th return element is the reference-style
    # pydrake.visualization.VideoWriter instance when
    # PORT_USE_DRAKE_VIDEO_WRITER=1 (else None).
    return (diagram, plant, panda_model, object_model, meshcat,
            plant_ad, context_ad,
            _drake_video_writer if add_camera else None)


# ---------------------------------------------------------------------------
# Prepositioned-pose IK (push-direction-aware)
# ---------------------------------------------------------------------------

def init_rotation(task_cfg: dict):
    """Initial object orientation from tasks.yaml `init_quat` [w, x, y, z].

    Flat-resting objects (box, T, H) spawn upright and omit the key, so this
    returns identity and they are unaffected. The jack has no flat face -- it
    balances on a tripod of tip spheres -- so it MUST be spawned at a specific
    orientation or it starts on a single tip and immediately falls over.

    Single source of truth for both stagers: main.py's initial pose AND
    compute_safe_init_arm_q, which re-stages the object before running IK.
    That second site used to hardcode `RotationMatrix()`, silently resetting a
    configured init_quat to identity while preserving the position -- invisible
    for every task whose init orientation IS identity, fatal for the jack.
    """
    q = task_cfg.get("init_quat", None)
    if q is None:
        return ad.RotationMatrix()
    q = np.asarray(q, dtype=float).flatten()
    if q.shape[0] != 4:
        raise ValueError(f"init_quat must have 4 entries [w,x,y,z]; got {q.shape[0]}")
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("init_quat has zero norm")
    q = q / n
    return ad.RotationMatrix(ad.Quaternion(q[0], q[1], q[2], q[3]))


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

    Reference override (2026-08-14 joint2 fix): when the task provides
    `q_init_franka`, return it verbatim — the reference sets the initial
    arm joints directly (sim_params.yaml q_init_franka) with NO IK
    preposition, and the joint2 pin (osc_franka.yaml W_joint2) needs the
    arm to START on the reference branch (joint2 = 1.1 rad) to be
    transient-free. Tasks without the key keep the legacy IK path.
    """
    _q_ref = task_cfg.get("q_init_franka")
    if _q_ref is not None:
        _q_ref = np.asarray(_q_ref, dtype=float)
        if verbose:
            print(f"[ENV]  init arm q: reference q_init_franka "
                  f"{np.round(_q_ref, 4).tolist()} (IK preposition skipped)")
        return _q_ref

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
    elif obj_type == "hshape":
        # H occupies x∈[-0.056,+0.056], y∈[-0.064,+0.064], z half-extent 0.016
        # (see _hshape_sdf). Envelope is symmetric, unlike the T's.
        half_extent = abs(g_hat[0]) * 0.056 + abs(g_hat[1]) * 0.064
        obj_top_z   = init_xyz[2] + 0.016
    elif obj_type == "jack":
        # The jack has no axis-aligned envelope: its extent depends on the
        # current tripod. Every tip is 0.0875 from the centre, so that radius
        # bounds the footprint in ANY direction and is orientation-independent
        # -- the right conservative choice for a body that rolls.
        half_extent = 0.0875
        # Highest point of a resting jack: the three UP tips sit one tripod
        # height above the CoM, symmetric with the three resting ones.
        obj_top_z   = init_xyz[2] + 0.061084
    else:
        raise ValueError(
            f"compute_safe_init_arm_q: unknown object_type '{obj_type}' "
            "(expected 'box', 'sphere', 'tshape', 'hshape', or 'jack')."
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
        # Honour the task's init_quat. Hardcoding identity here reset any
        # configured initial orientation (see init_rotation).
        ad.RigidTransform(init_rotation(task_cfg), init_xyz.tolist()),
    )

    n_arm_dofs = plant.num_actuators()
    q_full     = plant.GetPositions(plant_ctx).copy()

    # Slice to n_arm_dofs: the floating-base object DOFs (xyz + quat) carry
    # ±inf limits in Drake, which we don't want to propagate as a clip
    # target inside the arm IK.
    q_lo_arm = plant.GetPositionLowerLimits()[:n_arm_dofs]
    q_hi_arm = plant.GetPositionUpperLimits()[:n_arm_dofs]

    # 2026-07-28 lunge fix: both stages solve 6-DOF with a TOOL-VERTICAL
    # (world-identity) orientation target. The position-only spawn IK left
    # the tool up to 64.7 deg tilted (box W preposition); the OSC's
    # constant-identity rotation hold (Kp_rot=800) then wrenched the arm
    # through the box at >1.5 m/s on the first ticks (box p5, 78.8%
    # saturation). The reference's q_init_franka posture holds the tool
    # vertical — this makes the spawn conformant with that property.
    _R_vert = ad.RotationMatrix()

    # Stage 1: lifted waypoint to escape any pose-induced local minima.
    p_waypoint = np.array([p_target[0], p_target[1], intermediate_z])
    q1, err1, it1 = solve_ik_to_ee_pos(
        plant, ee_frame, p_waypoint, q_full, plant_ctx,
        n_arm_dofs=n_arm_dofs, max_iter=80, damping=0.05,
        q_lo=q_lo_arm, q_hi=q_hi_arm, R_target=_R_vert,
    )

    # Stage 2: descend onto the contact target.
    q2, err2, it2 = solve_ik_to_ee_pos(
        plant, ee_frame, p_target, q1, plant_ctx,
        n_arm_dofs=n_arm_dofs, max_iter=80, damping=0.02,
        q_lo=q_lo_arm, q_hi=q_hi_arm, R_target=_R_vert,
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
