# OIM experiment source of truth

This file is the canonical index for OIM results. Do not infer a run's robot,
controller, success, or render validity from a filename. Check
[`results/campaigns/OIM_RUN_REGISTRY.json`](../results/campaigns/OIM_RUN_REGISTRY.json)
first.

## Required identity fields

Every reported experiment has four independent identities:

1. **Controller stack**: `dairlab_cpp_c3plus` or `python_port_c3plus`.
2. **Physics robot**: the robot actually simulated, currently `franka7` or
   `xarm6`.
3. **Outcome evidence**: the success marker and tolerances emitted or evaluated
   for that exact run.
4. **Render evidence**: whether joint telemetry, object telemetry, and obstacle
   geometry all came from that exact run.

Never substitute one identity for another. In particular:

- A DAIRLab C++ run is a Franka run unless its simulator was explicitly
  replaced with xArm and the registry says `physics_robot: "xarm6"`.
- Imported xArm visual geometry does not turn Franka telemetry into xArm
  telemetry.
- A successful object trajectory with missing joint telemetry is an
  `object_only` success video, not an authoritative robot video.
- Obstacles drawn only by the renderer do not validate obstacle collisions.
- A smoke render is not a successful experiment.

## Current authoritative status

| Run | Stack | Physics robot | Outcome | Robot render | Scenario physics |
|---|---|---|---|---|---|
| DAIRLab open-table five goals | C++ C3+ | Franka 7-DOF | 5/5 success | unavailable; no joint telemetry | valid for open table |
| Fresh DAIRLab goal-4 rerun | C++ C3+ | Franka 7-DOF | success marker present | unavailable; zero joint records | valid for open table |
| Legacy DAIRLab success MP4 | C++ C3+ | Franka 7-DOF | object success | invalid/parked arm | valid for open table |
| Python seed-0 8,000-step run | Python C3+ port | xArm6 | failure | authoritative xArm replay | valid for open table |
| Python integrated renderer smoke | Python C3+ port | xArm6 | not a benchmark | authoritative for two logged steps | valid for open table |

Therefore the currently supportable claims are:

- **DAIRLab C++ C3+ open-table success:** yes, using the native Franka stack.
- **Python-port xArm open-table success:** no.
- **DAIRLab C++ C3+ success with a properly recorded xArm:** no such run yet.

## Reporting gate

A video may be called an **authoritative successful xArm experiment** only when
one registry entry has all of the following:

```text
controller_stack = intended stack
physics_robot = xarm6
success = true
joint_telemetry = recorded
object_telemetry = recorded
obstacle_physics = native (when the scene has obstacles)
render_robot_source = recorded_joint_telemetry
```

If any field is missing, report exactly what is valid and explicitly name the
missing evidence. Do not combine logs, videos, or success markers across run
IDs.

## Naming convention for new runs

Use:

```text
<stack>__<robot>__<scene>__start-<key>__goal-<key>__<date>
```

Examples:

```text
dairlab-cpp-c3plus__franka7__open-table__start-1__goal-4__20260829
python-port-c3plus__xarm6__open-table__start-5__goal-4__20260829
```

Each new run directory should contain a `run_manifest.json` with the same
schema used by the central registry, plus logs and videos generated from only
that run.
