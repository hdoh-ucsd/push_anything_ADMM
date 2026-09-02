# xArm6 exact-8000 run: behavior after controller update 4000

## Scope and provenance

- Analysis date: 2026-09-01, America/Los_Angeles
- Run receipt: `results/xarm6_gates375_379_exact8000_final2_v8rOvB`
- Source implementation commit: `b27a22ede277a4c324140fa0953f0a40bba98437`
- Source implementation branch: `oim_c++_anything`
- Source worktree state: dirty; the run manifest records that prior-gate and
  unrelated work was preserved.
- Requested and completed controller updates: 8,000
- Planner mode: `full_sampling_c3plus`
- Configuration: `examples/sampling_c3/oim_t/parameters/oim_t.yaml`
- Configuration SHA-256:
  `d11cd65efbcf6ac7c814a0b690cc76f9135d27a9f607f6f10dc7d9b26051b990`

The word **step** below means a controller update. It does not mean a rendered
telemetry sample. The clipped video ends at telemetry sample 2,742 while its
controller counter reaches exactly 8,000.

## Finding

After controller update 4,000, the arm is not pushing the object. It is in a
high-clearance, collision-safe repositioning sequence. The subsequent local
lowering waypoint fails the tip/posture conformance check, after which the
planner repeatedly lifts, traverses, rejects the lower waypoint, invalidates
the candidate, and replans.

The principal failure is therefore candidate executability during lowering,
not sustained contact or contact-force saturation.

## Timeline

| Controller update | Observed behavior |
| ---: | --- |
| 3,866 | The fourth productive cycle finishes. Sample `mesh_crossbar_top_seeded_11` passes a 27-step dwell. Translation progress is `0.00940395 m`, yaw progress is `0.0396512 rad`, and lateral error is `0.000285472 m`. |
| 3,871 | The controller releases contact; `receding_release=PASS`. |
| 3,874 | It admits `mesh_crossbar_top_seeded_12` with 4,126 updates remaining. |
| 3,926 | The arm lifts away from the object. |
| 3,933–4,028 | The pusher capsule is verticalized at high clearance, around `z = 0.18 m`. Update 4,000 maps to approximately simulator time `127.58 s` and lies within this phase. There is no object contact. |
| 4,107 | The `progress_local_lower` posture step is rejected. Tip error would increase from `0.0100 m` to `0.0132073 m`. |
| 4,170–4,187 | Preview recovery lifts the arm back to a capsule-clear anchor, and the candidate is invalidated after completing only four of seven acquisition phases. |
| 4,190–5,618 | The controller repeats lift, verticalize, traverse, rejected lower, clearance recovery, invalidation, and replanning. The cumulative candidate rejection count reaches 11. No productive push occurs in this interval. |
| 5,621 | The planner switches to the component fallback `rotation_only_crossbar_right_seeded_2` and approaches another face. |
| 5,786 | A five-step dwell fails lateral-response acceptance. The measured terminal yaw disagrees with the prediction by approximately `-0.401203 rad`. |
| 6,049 | Corrective continuation fails. `cycle_lateral_recovery=FAIL` and `post_recovery_progress=FAIL`; live manipulation ends. |
| 6,050–8,000 | The controller publishes an explicit measured-state terminal hold. There is no new planning, manipulation, or claimed task progress, and commanded joint velocity is essentially zero. |

## Physical-contact evidence

The physical contact log reports that contact episode 10 ends at simulator time
`119.238 s`. Contact episode 11 does not begin until `226.158 s`. Because
controller update 4,000 maps to approximately `127.58 s`, it falls inside a
roughly 107-second contact-free interval.

Around update 4,000, the logged joint-speed magnitude reaches `0.5 rad/s`; the
largest nearby logged torque magnitude is approximately `43.704 N m` at
simulator time `129.564 s`. These values occur during verticalization and
traverse, not while applying force to the object. The visually sharp arm motion
is therefore a capped free-space repositioning motion and is not evidence of a
push at update 4,000.

## Late fallback and terminal result

The rotation-only fallback predicts terminal planar pose
`(0.375543, 0.355631, 0.625213)`, while the measured pose is
`(0.378157, 0.332536, 0.224010)`. Its yaw residual is approximately
`-0.401203 rad`, and the measured response is rejected laterally.

The run then enters terminal hold to satisfy the explicit exact-8,000-update
diagnostic contract without claiming false progress. The unchanged terminal
gate remains failed:

| Metric | Measured | Acceptance tolerance | Result |
| --- | ---: | ---: | --- |
| Translation error | `0.732329 m` | `0.05 m` | FAIL |
| Wrapped-yaw error | `2.92973 rad` | `0.10 rad` | FAIL |

## Diagnosis

The causal sequence after update 4,000 is:

1. A completed productive push is released.
2. The arm moves to high clearance for the next sampled face.
3. The proposed lower waypoint increases rather than decreases tip error.
4. The conformance/safety gate rejects the lower motion.
5. Candidate invalidation and replanning repeat without physical contact.
6. A late rotation-only fallback makes brief contact, but its measured yaw and
   lateral response disagree strongly with the prediction.
7. No admissible continuation remains, so live manipulation ends and the
   controller holds measured state through update 8,000.

The next implementation work should focus on executable local-lowering
generation and ranking after large measured pose changes, while retaining the
existing whole-capsule collision and tip-conformance protections. Relaxing the
safety gate or numerical tolerances is not supported by this evidence.

## Evidence files

- `results/xarm6_gates375_379_exact8000_final2_v8rOvB/MANIFEST.md`
- `results/xarm6_gates375_379_exact8000_final2_v8rOvB/planner.log`
- `results/xarm6_gates375_379_exact8000_final2_v8rOvB/sim.log`
- `results/xarm6_gates375_379_exact8000_final2_v8rOvB/osc.log`
- `results/xarm6_gates375_379_exact8000_final2_v8rOvB/xarm_control.csv`
- `results/xarm6_gates375_379_exact8000_final2_v8rOvB/contact_join_summary_exact8000.json`
- `results/xarm6_gates375_379_exact8000_final2_v8rOvB/xarm6_exact8000_sidepanel.mp4`

Windows artifact receipts recorded by the manifest:

- `D:\xarm6_exact8000_20260901_v8rOvB`
- `D:\xarm6_exact8000_sidepanel_20260901.mp4`
