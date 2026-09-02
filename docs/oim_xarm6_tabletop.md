# OIM xArm6 tabletop scenes in PyDrake

The port vendors these Object-Informed-Manipulation-MJX scenes:

- `open_table`
- `single_obstacle`
- `shelf_gap`
- `ycb_clutter`
- `icra_sign`

An OIM rollout is successful when translational error is at most `0.05 m`
and orientation error is at most `0.1 rad`.

The arm/controller and run settings are loaded from the unchanged vendored
`oim/configs/robots/xarm6.yaml`; the MJCF supplies the physical arm model.

Scenario tables use the OIM evaluator definitions:

- `SR`: fraction whose final pose meets both tolerances.
- `eps_d`: position error averaged over every executed trajectory state.
- `eps_d^s`: the same trajectory mean over successful trials only; blank
  when there are no successes.
- `theta`: orientation error averaged over the trajectory.
- `steps`: first step meeting both tolerances, censored at configured steps
  when never reached.
- `f (Hz)`: reciprocal of mean recorded planning `compute_time`.
- `T (s)`: `steps_run * dt`; failures receive the slowest simulated time
  among every run loaded into the comparison.

Import or refresh the assets after cloning upstream:

```bash
python scripts/import_oim_xarm6_tabletop.py
```

Build a scene and apply OIM's start configuration:

```python
import numpy as np
from sim.oim_xarm6_tabletop import (
    build_oim_tabletop_scene,
    set_oim_start_configuration,
    xarm_velocity_servo_torque,
)

scene = build_oim_tabletop_scene("shelf_gap")
context = scene.diagram.CreateDefaultContext()
set_oim_start_configuration(scene, context)
tau = xarm_velocity_servo_torque(scene, context, np.zeros(5))
scene.diagram.GetInputPort("xarm6_actuation").FixValue(context, tau)
```

The MJCF is loaded natively by Drake, preserving the xArm body transforms,
joint axes and limits, link masses, rigid pushing stick, table, pushed object,
and static obstacle geometry. OIM's five-joint home configuration is
`[49.2, 34.8, -80.6, 0, 45.9]` degrees. Joint 6 remains rigidly fixed.

## Translation notes

Drake's MJCF parser accepts OBJ but not STL, so the importer converts the
seven xArm meshes to OBJ and changes only those filename extensions. The
scene, YCB, and glyph OBJ files remain unchanged.

Drake currently ignores MuJoCo velocity actuators. The wrapper recreates five
joint actuators with OIM's effort limits and provides
`xarm_velocity_servo_torque`, which implements the source gains, ±0.5 rad/s
command bound, torque clipping, and joint-4 passive spring.
`xarm_configured_actuation` additionally applies the arm-only gravity
compensation after actuator clipping, matching MuJoCo's passive `gravcomp=1`
ordering. The default Drake plant timestep now comes from
`world3d.exec_timestep` (`0.002 s`) in the YAML.

The YAML is also resolved into `OimC3PlusConfiguration` so its native
experiment can be inspected without importing OIM/JAX. Those optimizer values
are not applied to the C3+ baseline.

## C3+ baseline architecture

`control/oim_c3plus_architecture.py` defines the integration boundary:

- OIM supplies the five scenes, xArm model, start and goal poses, 2 ms
  execution plant, run length, and 5 cm / 0.1 rad evaluation tolerances.
- The existing `control.admm_solver.C3Solver(mode="c3plus")` supplies ADMM
  consensus and the componentwise C3+ projection unchanged. The baseline
  keeps N=7, planning dt=75 ms, three ADMM iterations, rho=100, and a 30 N
  Cartesian force bound.
- OIM's sampler, costs, `n_admm`, `rho`, `rho_torque`, `gamma`, and consensus
  selection are not consumed by the baseline.

`OimXarmDrakeAdapter.observe()` projects the planar Drake plant into the
existing 19-state planner contract: object quaternion and xyz, stick-tip xyz,
object spatial velocity, and stick-tip velocity. Drake ignores the MJCF
`site`, so the wrapper reconstructs the tip at `[0, 0, 0.1794]` in the stick
body frame.

The execution seam is implemented by `OimXarmCartesianForceExecutor`, which
maps the C3+ Cartesian force through the live stick-tip Jacobian and adds
gravity compensation and joint damping. It therefore preserves the C3+
action's force units instead of passing it to OIM's velocity servo.

`scripts/run_oim_c3plus.py` runs the complete observe-plan-execute loop and
prints OIM metrics as JSON. For example:

```bash
python scripts/run_oim_c3plus.py --scene open_table --steps 10
```

The current planar LCS frontend uses one conservative circumscribed
manipuland contact. It is sufficient for an end-to-end executable baseline
smoke test, but it does not yet represent obstacle contacts or the exact T/C
mesh boundary. Results must not be reported as the final five-scenario
benchmark until those geometry-derived contact witnesses are added.

The following MuJoCo features remain engine approximations before a
physics-parity benchmark:

- `solref`/`solimp` versus Drake's contact compliance parameters;
- `condim` torsional/rolling friction (Drake's standard Coulomb model covers
  translational sliding, but not the identical MuJoCo cone);
- `implicitfast`, solver iterations, and line-search iterations;
- MuJoCo collision masks/exclusions (declare equivalent SceneGraph filters);
- mocap goals, sensors, keyframes, and camera declarations (the wrapper sets
  the start state directly; overlays and cameras should be added as Drake
  systems).

These limitations do not change scene geometry or arm kinematics, but they do
matter before comparing dynamic rollouts or controller performance.
