# SIM / URDF TIER-2 EVIDENCE (2026-07-14)

Diagnostic commit: `043f378` — `PUSHA_SIM_T2_DIAG=1`, env-gated, default-OFF byte-identical.

## Runtime captures

```
[SIM-T2] pusher_radius=0.025
         (reference: end_effector_full.urdf sphere r=0.0195; env_PUSHA_PUSHER_RADIUS=<unset>)
[SIM-T2] pusher_mu=0.4
         (reference: end_effector_full.urdf mu_static=1.0 mu_dynamic=1.0)
[SIM-T2] pusher_body='pusher' weld_parent='panda_link8' weld_offset=[0,0,0.05]
         (reference: end_effector_flange welded to panda_link7 at
          kToolAttachmentFrame=[0,0,0.107] + 180deg x-rotation)
[SIM-T2] robot_base_xyz=[0.0, -0.6, 0.0]
         (reference: Identity() = [0,0,0])
[SIM-T2] table_size=(2.0, 2.0, 0.1) origin=[0,0,-0.05] table_mu=(static=0.6, dynamic=0.5)
         (reference: ground.urdf box=(5.0, 0.91, 0.1) origin=[0,0,-0.05]
          mu=1.0 static+dynamic; welded via kFrankaToGroundOffset=[0,0,-0.029])
[SIM-T2] time_step=0.001
         (reference sim_params.yaml: dt=0.001 — CONFORMANT)
```

## Reference-side reads (Tier-2 (a), read-only)

### Franka + end-effector chain
- `sampling_c3_utils.h:12-18`: `kFrankaModel = "package://drake_models/franka_description/urdf/panda_arm.urdf"`, `kEndEffectorModel = "examples/sampling_c3/urdf/end_effector_full.urdf"`, `kEndEffectorName = "end_effector_tip"`.
- `sampling_c3_utils.cc:23-36 AddFrankaToPlant`: Franka parsed and welded to world at `X_WI = Identity()`, then `end_effector_full.urdf` parsed and welded to `panda_link7` at `RigidTransform(RollPitchYaw(π, 0, 0), kToolAttachmentFrame=[0, 0, 0.107])` (180° x-rotation + 10.7 cm tool offset).
- `end_effector_full.urdf`:
  - `end_effector_flange`: cylinder r=0.0315 len=0.0096
  - `end_effector_peg`: cylinder r=0.0127 len=0.1016; welded via `fix_link_base` at xyz=[0,0,-0.0096]
  - `end_effector_tip`: sphere r=0.0195 μ=1.0; welded via `fix_tip_link` at xyz=[0,0,-0.1169]
  - Full chain: panda_link7 + 10.7 cm + 180° x-rotation + flange + 0.96 cm + peg + 11.69 cm + tip → tip is welded position + 22.65 cm along -z_link7 (roughly).

### Ground / table
- `ground.urdf`: rigid body "ground", collision box 5×0.91×0.1 at origin xyz=[0, 0, -0.05], **μ=1.0 static+dynamic** (drake:proximity_properties). Hydroelastic block COMMENTED OUT → point contact. Welded to Franka's `panda_link0` at `kFrankaToGroundOffset = [0, 0, -0.029]`. **Ground TOP surface at z = -0.029 in Franka frame** (i.e., 2.9 cm below Franka base).

### Object URDFs (reference has SEPARATE sim + LCS URDFs)
- **push_t.sdf** (SIM plant): 2 links (vertical_link + horizontal_link), boxes 0.16×0.04×0.04 each, `compliant_hydroelastic` collision, hydroelastic_modulus 3e7, hunt_crossley_dissipation 10, **mu_dynamic=0.3**.
- **push_t_control.sdf** (LCS plant): SAME 2 links + **3 tiny sphere witnesses r=0.001 m on vertical_link:**
  - `top_left_sphere` pose=(−0.12, +0.08, −0.02)
  - `top_right_sphere` pose=(−0.12, −0.08, −0.02)
  - `bottom_sphere` pose=(+0.08, 0.0, −0.02)
  These are the SPHERE WITNESS BODIES that participate in the LCS 3-pair T-ground admission (per 3.d).
- **expo_box.sdf** (SIM): mesh convex decomposition (`expo_box_convex_N.obj` × ~9 pieces), compliant_hydroelastic, mu_dynamic=0.3.
- **expo_box_controller.sdf** (LCS): mesh convex decomposition (same pieces).

### Sim / Drake plant
- `franka_sim.cc:79-80`: `sim_dt = sim_params.dt; auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, sim_dt)`. Sim plant is DISCRETE at sim_dt.
- `anything/parameters/sim_params.yaml: dt: 0.001` — sim time_step 1 ms. **Port matches.**
- LCS plant: `franka_sampling_c3_controller.cc:103`: `AddMultibodyPlantSceneGraph(&plant_lcs_builder, 0.0)` — **continuous plant for LCS** (dt=0). Port uses SAME plant for sim + LCS.

## Divergences summary (13 entries — full list in map)

Confirmed LIVE default divergences for pushing task:
- 5.a Franka BASE offset: port ROBOT_BASE_XYZ=[0,-0.6,0] vs reference [0,0,0] identity
- 5.b End-effector attachment: port programmatic pusher body on panda_link8+[0,0,0.05] vs reference URDF end_effector chain on panda_link7+[0,0,0.107] with 180° x-rotation
- 5.c Pusher radius: port 0.025 m vs reference 0.0195 m (3-mm mismatch, in-scope from executor 1.f)
- 5.d Pusher μ: port 0.4 (pushing task_cfg) vs reference 1.0 (URDF)
- 5.g Manipuland friction: port 0.4 (pushing task_cfg) vs reference 0.3 (expo_box URDF)
- 5.h Drake contact model: port POINT contact + Coulomb vs reference COMPLIANT HYDROELASTIC + Hunt-Crossley dissipation
- 5.i LCS-vs-Sim URDF split: port uses SAME plant for sim+LCS; reference uses SEPARATE URDFs (*.sdf sim vs *_control.sdf LCS with sphere witnesses)
- 5.k Ground: port 2×2×0.1 μ=(0.6, 0.5) on world_body vs reference ground.urdf 5×0.91×0.1 μ=1.0 welded via kFrankaToGroundOffset
- 5.j T-witness positions (T-task only): port synthesized (+0.13, 0, -0.02), (-0.05, ±0.08, -0.02) vs reference URDF spheres (-0.12, ±0.08, -0.02), (+0.08, 0, -0.02) — **positions DIFFER** (different origin conventions between port t_link at CoM vs reference vertical_link at link origin)

CONFORMANT:
- 5.n Sim time_step: both 0.001 s
- 5.a Franka URDF: both `panda_arm.urdf` from drake_models

Inert-by-config:
- 5.l Walls: port has none, reference `include_walls=false` for anything+push_t defaults

## 2.k caution result

Verified individually:
- table μ (5.k) split into static/dynamic vs reference single μ=1.0 — port has TWO separate friction values (0.6 static, 0.5 dynamic); reference has ONE (1.0 for both). Both agents' PROXIMITY-PROPERTIES sub-mechanism verified separately.
- contact model (5.h) — port point + Coulomb; reference hydroelastic + Hunt-Crossley + point-fallback. Reference has BOTH mechanisms active (some geoms hydroelastic, others point-contact via fallback). No hidden port-only live mechanism.
- pusher radius env override (5.c) — verified `env_PUSHA_PUSHER_RADIUS=<unset>` at runtime; not silently active.
- 5.j T-witness positions divergence caught by CHECKING VALUES not just structure — port has same NUMBER of witnesses (3) but DIFFERENT positions than reference. Passed 2.k by not stopping at "both have 3 witnesses".
