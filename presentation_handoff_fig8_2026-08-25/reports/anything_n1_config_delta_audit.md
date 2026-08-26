# anything-N1 config-delta audit (port vs authentic reference lineage)

**Date:** 2026-08-11. **Trigger:** the reference `anything` demo with native
`T_shape_video` reaches TIGHT (<0.02 m, <0.1 rad) in <60 s at the exact
head-to-head goal demand, while the bit-rotted `push_t` demo — the port's
conformance lineage — fails at 1800 s (`memory/project_anything_T_tight_pass_2026-08-11.md`).

**Authoritative reference column:** `dairlib_sampling_c3 @ 257e3ed`,
`examples/sampling_c3/anything/parameters/` AFTER running the authors' own
`multiyaml_rewrite.py` with `base_names: [T_shape_video]` (N=1). Snapshot:
`results/reference/anything_T_confirm1/ref_patches.diff`. The N=1 values in
that tool's tables are the authors' own single-object tuning.

**Port column:** `config/sampling_c3_kik_t.yaml`, `config/tasks.yaml` push_t
block, `main.py`, `control/admm_solver.py` at HEAD `655b5fd`.

**Stale column** (context): `push_t/parameters/` as-shipped+loader-fills —
the lineage the port conformed to.

Verdict per row: does the port match ANYTHING-N1 (authentic) or STALE?

---

## P1 — Mode-switch / progress cluster (directly implicated in the fail anatomy)

| param | anything-N1 | port (source) | stale push_t | port matches |
|---|---|---|---|---|
| cost_switching_threshold_distance | **0.05** | 0.50 (`sampling_c3_kik_t.yaml` progress_params) | 0.50 | STALE |
| progress_enforced_cost_drop | **0.5** | 0.01 (kik_t) | 0.01 | STALE |
| progress_enforced_over_n_loops | **35** | 180 (kik_t `progress_enforced_over_n_loops_ref`) | 180 | STALE |
| hyst_c3_to_repos_frac | **0.6** | 0.4 (kik_t) | 0.4 | STALE |
| hyst_c3_to_repos_frac_position | **0.7** | 0.6 (kik_t) | 0.6 | STALE |
| hyst_repos_to_repos_frac | **0.7** | 0.3 (kik_t) | 0.3 | STALE |
| hyst_repos_to_repos_frac_position | **0.7** | 0.1 (kik_t) | 0.1 | STALE |
| hyst_repos_to_c3_frac / _position | 0.9 / 0.5 | 0.9 / 0.5 (kik_t) | same | ✓ (unchanged by lineage) |
| num_control_loops_to_wait (+_position) | 5 / 5 | 5 / 5 | same | ✓ |
| finished_reposition_cost | 1e9 | 1e9 | same | ✓ |

Mechanism links: hyst_repos_to_repos 0.3→0.7 is the churn killer (214
aborted lift-cycles/9 min in the push_t-demo run); c3_to_repos 0.4→0.6
keeps pushing episodes alive longer; threshold 0.50→0.05 flips most of the
run into the position regime (different hysteresis constants + orientation
ignored until 5 cm out).

**HAZARD:** `config/tasks.yaml:272` comment already claims
`cost_switching_threshold_distance = 0.05` for the near-goal weight swap
while kik_t progress_params says 0.50 — check which value each consumer
actually reads before landing (regime swap in task_costs vs mode_switch).

## P2 — Planner cadence / authority

| param | anything-N1 | port (source) | stale push_t | port matches |
|---|---|---|---|---|
| N (horizon) | **10** | 5 (`main.py:614`) | 5 | STALE |
| planning_dt_position | **0.075** | 0.1 (`main.py:615`) | 0.1 | STALE |
| planning_dt_pose | **0.075** | 0.05 (`main.py:616`) | 0.05 | STALE |
| w_Q (pose regime) | **80** | 50 (`tasks.yaml:239 w_Q`) | 50 | STALE |
| w_R | **6** | 1-equivalent (r_torque path) | 1 | STALE |
| w_G (ADMM per-slot G scale) | **0.18** | 0.01 (`admm_solver.py:201` hardcoded) | 0.01 | STALE |
| w_U | **0.5** | 0.26-equivalent (U-weight block near `admm_solver.py:174`) | 0.26 | STALE |
| w_Q_position / w_R_position / w_G_position / w_U_position | 50 / 6 / 0.18 / 0.26 | 50 / 1 / 0.01 / 0.26 | 50 / 1 / 0.01 / 0.26 | mixed |
| q_vector ee_vel slots | **[15, 15, 10]** | [5, 5, 5] (`tasks.yaml:243`) | [15,15,10]? (port deviated deliberately, p108 ee_vel reg) | NEITHER |
| q_vector obj pos (pose) | [150, 150, 120] ×80 = **12000** | pose-regime ~0.80×12500=10000 (`tasks.yaml:252` + swap) | [150,150,120]×50=7500 | NEITHER (близко) |
| q_vector_position obj pos | [200, 200, 120] ×50 = **10000/10000/6000** | 12500/12500/12500 (`tasks.yaml:252,260`) | [250…]×50=12500 | STALE |
| Kp_for_ee_pd_rollout | **[100, 100, 50]** (z halved) | 100 scalar (kik_t) | [100,100,100] | STALE |
| admm_iter | 3 | 3 (`--admm-iter` default) | 3 | ✓ |
| rho_scale | 3 | 3 | 3 | ✓ |
| lcs_dt_resolution | 4 | 4 (kik_t) | 4 | ✓ |
| ee_velocity_limits / nominal_ee_accel | [-0.14,0.14] / 2 | same | same | ✓ |
| use_quaternion_dependent_cost / weight | true / 1000 | true / 1000 | same | ✓ |
| use_predicted_x0_reset_mechanism | false | n/a (port lacks; irrelevant for T since ref=false) | false | ✓-effective |
| num_outer_threads | 6 | serial (WSL determinism rule) | 4 | ENV — leave |

## P3 — Contact physics

| param | anything-N1 | port (source) | stale push_t | port matches |
|---|---|---|---|---|
| mu EE-object | **0.42** | 1.0 (`tasks.yaml:175`) | 1.0 | STALE |
| mu object-ground | **0.46** | 0.4615 (`tasks.yaml:176`) | 0.4615 | STALE (≈) |
| mu EE-ground | **0.823** | 0.4165 (`tasks.yaml:177`) | 0.4165 | STALE |
| mu object-wall | **0.375** (walls active) | no wall contacts in port LCS | n/a | GAP (pre-existing) |
| resolve_contacts_to_lists | [[0, 1, 3, 1]] (wall planar-resolved) | `resolve_contacts_to_for_cost: [0,2,3]` (kik_t; no wall) | [[0,1,3]] | STALE-shape |
| workspace_limits x | [0.23, 0.8] | [0.15, 0.9] (kik_t planner_workspace_x) | [0.15, 0.75] | NEITHER — see standing rule: workspace widening worsened overshoot; treat with care |

Note: μ_EE-obj 1.0→0.42 is a large physical change — the stale value gave
the planner 2.4× the tangential authority at the push contact. Interacts
with w_R (6× torque reg) and w_Q (1.6×): land as a cluster-aware sequence,
not blind single knobs, but still one-change-one-baseline.

## P4 — Sampler

| param | anything-N1 | port (source) | stale push_t | port matches |
|---|---|---|---|---|
| sampling_strategy | **7 kMeshNormalMultiObject** | kRandomOnPerimeter (kik_t) | 4 kRandomOnPerimeter | STALE |
| num_additional_samples_c3 | **5** | 2 (kik_t) | 2 | STALE |
| num_additional_samples_repos | **4** | 1 (kik_t) | 1 | STALE |
| N_sample_buffer | **200** | 100 (kik_t) | 100 | STALE |
| pos/ang_error_sample_retention | **0.004 / 0.05** | 0.003 / 0.0349 (kik_t) | 0.003 / 0.0349 | STALE |
| sample_projection_clearance | **0.027** | 0.02 (kik_t) | 0.02 | STALE |
| avoid_choosing_unsuccessful_samples | **true** (+N=10 buffer, retentions 0.006/0.05, radius 0.01) | WIRED with these exact values (`params.py:318-322` defaults; mission memory 2026-08-03 audit) | absent (loader-fill only) | ✓ ANYTHING — already conformant |
| z_height | 0.002 abs (= 31 mm above ground −0.029) | port-frame 0.034 (task sampling_height) | −0.004 (25 mm) / fills | NEITHER (31 vs 34 mm) |
| gen_planar_samples / sample_on_wall / c3_min_clearance / ee_z_close | true / true / 0.01 / true | ee_z_close true; others n/a-partial | fills from anything | partial |

kMeshNormal sampler: projects normals off mesh faces (sampling.py has
face-normal machinery + `use_mesh_normal_area_weighting` already — port
delta is strategy selection + multi-object variant details).

## P5 — Reposition

| param | anything-N1 | port (source) | stale push_t | port matches |
|---|---|---|---|---|
| pwl_waypoint_height | **0.0699 abs** (ground −0.029 ⇒ 98.9 mm above ground; computed −0.029+obj_height+0.05) | 0.075 above ground-0 (`tasks.yaml:213` override) | 0.06 abs (89 mm above ground) | NEITHER (75 vs 98.9 mm) |
| traj type / speeds | kPiecewiseLinear etc. | same family (kik_t) | same | ✓-family |

## No-delta confirmations (audited ✓)

goal_params: lookahead_step_size 0.15, lookahead_angle 2, angle_hysteresis
0.4, ee_target_z_offset_above_object 0.06, success thresholds 0.02/0.1 —
identical in stale and anything; port carries all (tasks.yaml:268, kik_t).
admm_iter 3, rho_scale 3, delta_option 1, warm_start false, end_on_qp_step
false, gamma 1.0, include_walls true(ref both; port GAP noted P3),
dt_cost 0, solve_time_filter_alpha 0.95.

## Proposed landing order (one change per seeded baseline, pre-change-HEAD baseline first)

1. **L1 progress/hysteresis cluster** (P1 rows; kik_t progress_params +
   the 0.50/0.05 consumer check). Highest mechanism confidence — directly
   the anatomy knobs. Single logical change (one param *set*, one commit).
2. **L2 horizon/dt**: N=10, dt 0.075/0.075 (`main.py:613-616`).
3. **L3 cost/ADMM scale set**: w_Q 80, w_R 6, w_G 0.18, w_U 0.5 (+position
   variants), q_vector ee_vel [15,15,10], position-regime obj_pos
   10000/10000/6000, Kp_rollout z 50.
4. **L4 friction set**: μ 0.42/0.46/0.823.
5. **L5 sampler counts/buffers**: 5/4 samples, buffer 200, retentions,
   clearance 0.027; then strategy 7 and the unsuccessful-sample buffer
   (new mechanism — own falsify-check).
6. **L6 reposition waypoint height** 0.0989 port-frame.

Baselines per `memory/feedback_baseline_provenance.md`: run pre-change HEAD
baseline with the same protocol first; goal-relative pos/rot + trajectory
shape are the metrics (NOT n_lcs). Contact-formation bar per
`memory/feedback_contact_formation_bar.md` where applicable.

## Landing hazards

- `params.py from_dict` hardcoded fallbacks override dataclass defaults —
  update BOTH sites per value (bit us at commit 3669368).
- tasks.yaml-vs-kik_t threshold inconsistency (P1 hazard note).
- `w_G` is hardcoded at `admm_solver.py:201`, not yaml-fed.
- Port ee_vel [5,5,5] was a DELIBERATE port deviation (p108 over-damping);
  landing [15,15,10] may resurface that — it interacts with L2/L3 scale
  changes, which is why it lands inside the L3 cluster, not alone.
- Workspace x-limits: do NOT widen along the way
  (`memory/feedback_workspace_widening_no_fix_overshoot.md`).
- T_shape_video ≠ push_t box-T geometry: reference-side re-validation of
  any specific knob should use the anything-T harness runs, but port
  baselines stay on the port's own T at the same goal demand.
