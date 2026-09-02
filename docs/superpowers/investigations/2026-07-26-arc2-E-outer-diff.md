# Arc-2 §E — Outer sampling-C3 pipeline structural diff (port ↔ reference)

**Date:** 2026-07-26
**Author:** Claude (Opus 4.7)
**Scope:** OUTER control loop only. Inner ADMM (`_solve_c3plus` vs `c3_plus.cc`) is being
covered by a sibling agent — see §6 handoff.

**Port baseline:** `main` HEAD (2026-07-25 sweep, `f484607` + local WIP on
`config/tasks.yaml`, `control/admm_solver.py`), file
`control/sampling_c3/sampling_based_c3_controller.py` (4726 lines).

**Reference baseline:** `dairlib_sampling_c3 @ push_anything_dev 257e3ede`, file
`systems/controllers/sampling_based_c3_controller.cc` (3319 lines) +
`systems/controllers/sampling_based_c3_controller.h` (509 lines) +
`examples/sampling_c3/{generate_samples,reposition,sampling_c3_utils}.cc` and the
five parameter headers in `examples/sampling_c3/parameter_headers/`.

**Conformance-map baseline (do not re-derive):** `docs/conformance-map.md`,
first landed 2026-07-14 with 5 subsystems / 75 entries, refreshed 2026-07-17
(§4 rows) and 2026-07-25 (§3 contact-model cluster closed). This report only
records **outer-loop deltas discovered after that map** — the map itself is
authoritative for subsystems 1 (Executor), 2 (Reposition), 3 (LCS/contact),
4 (Cost matrices), 5 (Progress) at their 2026-07-14 / 2026-07-25 states.

---

## 0 — TL;DR (top-3 outer-loop divergences most likely to amplify ADMM instability)

1. **Per-sample ADMM iteration budget is asymmetric in the port, symmetric in the reference.**
   Port: `k=0` gets full iters (`base_admm_iter=25`), `k≥1` get `surrogate_admm_iters=1`
   (`inner_solve.py:1120`, `:293`). Reference: every sample in the OpenMP loop runs the same
   `admm_iter=3` (`sampling_based_c3_controller.cc:971-1085`, `sampling_c3plus_options.yaml:2`).
   Under G-on, the surrogate solves *never converge* and produce noisy cost signals that
   compete with the full-iter k=0 cost — the port's argmin is comparing apples (25-iter
   winner) to oranges (1-iter surrogates). **Suspected amplifier.**

2. **Predicted-x0 clamping is present in the reference for BOTH modes, but only wired
   for repos mode in the port.** Reference `ResolvePredictedEEState` (`cc:1406-1454`) is
   called every tick and clamps `x_lcs_curr` to
   `x_pred_curr_plan ± nominal_ee_accel · dt²` when
   `use_predicted_x0_c3 && is_doing_c3`. Port implements the repos-mode side
   (`sampling_based_c3_controller.py:269`, `:3916`, `:4001-4043`) but has NO c3-mode
   `ResolvePredictedEEState` equivalent. Concretely: the LCS x0 fed into the ADMM
   inner solve is the raw Drake state instead of the smoothed predicted state.
   This is a first-order input to the ADMM subproblem — mis-linearizing at a
   noisier x0 can compound with G-diag instability.

3. **Progress-tracker cost feed is *raw C3 quadratic* in the reference, but
   `c_C3_raw` (bonus-stripped) in the port — while the mode-switch cost gap
   uses the *bonus-inflated* score.** Reference `KeepTrackOfC3ModeProgress`
   (`cc:2208-2305`) reads `all_sample_costs_[kCurrentLocation]` which for the
   reference is `c3_cost + travel_cost_per_meter · 0` — i.e., the same figure
   that drives the mode gate. Port ProgressTracker gets `results[0].c_C3_raw`
   (bonus-stripped, `sampling_based_c3_controller.py:1543-1547`), while the
   mode-switch cost gap uses `c_curr = c_samples[0]` (bonus-inflated). This
   two-metric split means "met_progress" and "cost-gap decision" live in
   different cost spaces — a mode transition can fire on the bonus-inflated
   gap while progress tracker still thinks no progress was made.

---

## 1 — Subsystem-level map

| Reference subsystem | Reference file/class | Port mirror | Status |
|---|---|---|---|
| `SamplingC3Controller` (LeafSystem, per-tick `ComputePlan`) | `sampling_based_c3_controller.cc:722`, `.h:69` | `SamplingC3Controller` class + `compute_control` / `_solve_plan` (`sampling_based_c3_controller.py:67`, `:1055`, `:1237`) | PRESENT (renamed 2026-07-10, docs stale in CLAUDE.md) |
| `CalcCost` (per-sample cost dispatcher over 6 `C3CostComputationType` variants) | `cc:493-720` | `InnerSolver.evaluate_sample` (`inner_solve.py:213`, `:380`) — single mode only, plus optional `use_cost_lcs_ranking` path for tshape | PARTIAL — 5 of the 6 cost types missing; see §3 |
| `UpdateCostMatrices` (Q, R, G, U rebuild + quaternion Hessian near-goal) | `cc:1498-1580` | `QuadraticManipulationCost.build` (`task_costs.py:205`) + G-diag WIP (`admm_solver.py` per-slot G) | PARTIAL — see conformance-map §4 |
| `CreateLCSObjectsForSamples` (parallel per-sample LCS build via `LCSFactory`) | `cc:1619-1698` | `LCSFormulator.linearize_discrete` invoked inside `InnerSolver.evaluate_sample` | PRESENT (mirrored in Python, no `_for_cost` LCS — see §3 & §4) |
| `GenerateSampleStates` (8 sampling strategies) | `examples/.../generate_samples.cc:24-165` | `sampling.py:90` `generate_samples` — 5 strategies (`kRandomOnCircle`, `kRadiallySymmetric`, `kFaceNormal`, `kRandomOnPerimeter`, `kFixed`) | PARTIAL — port's `kFaceNormal` and hardcoded T-face-table are LOCAL (see §2a) |
| `SampleIsAcceptable` (workspace + unsuccessful buffer filter) | `generate_samples.cc:167-210` | `sampling.py:809` `is_in_workspace` — no unsuccessful-buffer filter inside the sampler | Filter loop landed at wrapper level via `unsuccessful_buffer` (`sampling_based_c3_controller.py:1949-1976`), not in the sampler |
| `MaintainSampleBuffers` (prune, insert, sort-lowest-cost-to-end) | `cc:2002-2102` | `SampleBuffer.prune/append` + `_update_buffer` (`sample_buffer.py:95-125`, `sampling_based_c3_controller.py:988`) | PRESENT with divergent policy (see §2a) |
| `AugmentSamplesWithBuffer` (best-from-buffer added as extra sample in c3 mode) | `cc:2106-2158` | `sampling_based_c3_controller.py:1417-1442` (re-enabled 2026-07-19) | PRESENT |
| `AddToUnsuccessfulBuffer` (fires at free→c3 transition) | `cc:2161-2205` | `sampling_based_c3_controller.py:1935-1976` — fires on `kToC3Cost`, `kToC3ReachedReposTarget`, PLUS `kToBetterRepos` (port-only) | PORT-DIVERGENT (extra trigger) |
| `KeepTrackOfC3ModeProgress` (best-ever metric + steps-since-improve) | `cc:2208-2305` | `progress.py:118` `update` + `:171` `met_progress` | PRESENT (see §2c and §5 conformance-map row) |
| `ResetProgressMetrics` | `cc:2308-2318` | `progress.py:208` `reset` | PRESENT |
| `ResolvePredictedEEState` (c3 mode x0 clamp) | `cc:1406-1454` | **MISSING for c3 mode** (repos mode has a partial equivalent at `sampling_based_c3_controller.py:4001-4043` via `_x_pred_repos_plan`, and `ci_mpc_c3plus.py:139/:497` stores `_x_pred_curr_plan` for the c3 planner but doesn't clamp the tick's x0) | **MISSING in port (c3 side)** |
| `ClampEndEffectorAcceleration` | `cc:1457-1472` | Port only clamps repos-mode `p_start` via `_delta_pos = nominal_ee_accel · dt²` at `sampling_based_c3_controller.py:4014` | PARTIAL (repos only) |
| `CheckForWorkspaceLimitViolations` (DRAKE_DEMAND per tick — hard failure if EE outside workspace or radius shell) | `cc:1476-1494` | LOCAL: `sampling.py:809` `is_in_workspace` filters SAMPLES only, no per-tick invariant on the ACTUAL EE state. `params.robot_radius_limits` exists but no runtime check. | **MISSING in port** — item-#7 arc-2 memory notes p73 G-on ρ=1 "crashes (workspace violation)" — that's the port's `assert`-in-inner-loop version tripping; reference would `DRAKE_DEMAND`-abort here explicitly |
| `UpdateC3ExecutionTrajectory` (writes the N-knot EE-pos + EE-orientation + EE-force LCM trajectory, tilts EE, holds z constant) | `cc:1700-1836` | Port has NO N-knot output trajectory. Executor consumes `x_seq[1]` directly (`sampling_based_c3_controller.py:1105`, per `CLAUDE.md`). No EE-orientation/tilt logic. No wall_offset z-raise for c3-mode. | **MISSING in port** — architectural divergence: reference publishes LCM traj; port shortcuts to next-knot |
| `UpdateRepositioningExecutionTrajectory` | `cc:1839-1940` | `reposition_trajectory.py:211` + `PiecewiseLinearTracker`/`RepositionIKTracker` (`reposition.py`, `reposition_ik.py`) | PRESENT with divergent implementation (port uses IK per knot, reference uses `Reposition()` closed-form trajectory in `reposition.cc:13-79`) — see §5 conformance-map row 2.a |
| `Cost-switching threshold` (sticky flag `crossed_cost_switching_threshold_`, swaps c3_options + LCS factory options + resets sample buffers) | `cc:800-848` | `sampling_based_c3_controller.py:1588-1608` — sticky flag AND resets progress metrics AND swaps `quad_cost._crossed_switching_threshold` + `base_mpc._crossed_switching_threshold`. **Port DOES NOT clear the sample buffers on threshold cross.** | PARTIAL — sample-buffer wipe missing |
| `Goal-change detection` (resets sample buffers, resets `crossed_cost_switching_threshold_`, resets `detected_goal_changes_`) | `cc:787-818` | **MISSING in port** — no goal-change detection block. Port `main.py` treats the goal as constant; the lookahead sub-goal at `sampling_based_c3_controller.py:746` updates in the caller, not detected here. | **MISSING in port** |
| `achieved_fixed_goal` sticky flag + "get out of the way" sample pin | `cc:887-897, 1160-1164, 1279-1283` | `sampling_based_c3_controller.py:1559-1579, 1931-1933` — sticky flag + force `mode = "free"` when set. NO "sample pinned at (0.3, 0.4, 0.1)" get-out-of-way behavior. | PARTIAL |
| Per-tick output ports (23 of them: `c3_solution_curr_plan`, `is_c3_mode`, `all_sample_locations`, `dynamically_feasible_*`, `sample_buffer_*`, ...) | `cc:2321-3319` | Port emits **console diagnostics** and writes `audit_output/lambda_trace.csv` — no LCM output ports, no lcmt trajectories | PORT-LOCAL (log-only; downstream is Drake's diagram in the same process, not LCM subscribers) |
| Solve-time filter (`filtered_solve_time_ = α * filtered + (1-α) * solve_time`, feeds `x_pred_curr_plan_` interpolation index and `t_context`) | `cc:1387-1391`, `1718`, `1723-1732` | Port has `_step_times_ms` tally (`_run_osc` writes it) but no exponentially-filtered solve time feeding a lookup into `x_seq` | **MISSING in port** — related to §2 finding above; the reference's `x_pred_curr_plan_` is *interpolated* from the plan's knots at the filtered solve time, not just knot 1 |
| Radio input (Xbox: force-c3 mode, print-cost-breakdown, teleop) | `cc:728-729`, `:964`, `:974`, `:1118`, `:1155` | **MISSING in port** — no `radio_out` equivalent; `force_c3_mode`/`force_tracking_disabled` env-var overrides only | PORT-LOCAL (env vars replace) |

**Port-only outer components (no reference equivalent):**

| Port component | Location | Purpose |
|---|---|---|
| Contact-loss disengage streak (`_no_ee_box_streak`) | `sampling_based_c3_controller.py:2087-2141` | Forces c3→free after `contact_loss_threshold_*_s` consecutive c3 steps with no EE-BOX LCS pair admitted |
| Contact-proximity entry gate | `sampling_based_c3_controller.py:1724-1749` | Blocks `kToC3ReachedReposTarget` when `ee_to_box ≥ contact_entry_threshold` (default 0.09 m surface / 0.08 m center) |
| Commit-face gate L2 (pre + post decide) | `sampling_based_c3_controller.py:1800-1811`, `:2030-2048`, `commit_face_gate.py` | Blocks free→c3 when the pending contact face normal is anti-goal |
| Goal-align gate L1 | `sampling_based_c3_controller.py:1759-1785` | Blocks `kToC3ReachedReposTarget` when `nhat_onto_box · g_hat ≤ entry_align_threshold` |
| PHASE A / B / C approach-override state machine (`_approach_override_phase`) with independent stall/hard-cap gates | `sampling_based_c3_controller.py:2087-2185`, `params.py:contact_loss_threshold_phase{A_ltd,B_ltd,C}_s`, `phaseC_stall_threshold_s`, `phaseC_hard_cap_s` | State-driven approach lift→traverse→descend controller; feeds independent disengage timeouts |
| EE_z altitude gate (tshape-only) | `sampling_based_c3_controller.py:1866-1888` — passed into `decide_mode` as `ee_z_gate_pass` | Reference-conformant to `cc:1290-1293` but tshape-gated in the port because box regressed |
| Watchdog (steps-since-improve force-c3) | `sampling_based_c3_controller.py:2187-2203`, reason `kForceC3Watchdog` | Port-only escape hatch when free-mode fails to improve for a threshold count |
| `kToBetterRepos` reason + inline `AddToUnsuccessfulBuffer` fire | `mode_switch.py:44`, `sampling_based_c3_controller.py:1945-1976` | Reference only fires `AddToUnsuccessfulBuffer` at free→c3; port also fires at free→free re-target |
| Selection audit (`_sel_audit_*`) + `[ALL-SAMP]`, `[COST-DECOMP]`, `[SAMP-LCS]`, `[COST-LCS]`, `[X-SEQ-PROBE]`, `[PLAN-VS-EXEC]`, `[IK-LANDING]`, `[BODY-VS-CONTACT]`, `[CONTACT-CHECK]` diag emits | `sampling_based_c3_controller.py:1148-1236` and elsewhere | Env-gated diagnostics — outer-only, doesn't affect control decisions |
| `_use_pwl_traj` / `_pwl_traj` reposition-trajectory wrapper | `sampling_based_c3_controller.py:1657-1701` | Additional "did we arrive" euclidean-fallback predicate that OR's with the build-time flag |

---

## 2 — Per-tick control-flow diff

### Reference (`sampling_based_c3_controller.cc:722-1399`)

```
ComputePlan(context) [called at 100 Hz via DeclareForcedDiscreteUpdateEvent]
├── read radio, x_lcs_des, x_lcs_final_des, lcs_x_curr, teleop flag
├── ResolvePredictedEEState(is_teleop, x_lcs_curr)         # MUTATES x_lcs_curr
│     └── ClampEndEffectorAcceleration if !reset_condition # ± nominal_ee_accel·dt² box
├── CheckForWorkspaceLimitViolations(lcs_x_curr)          # DRAKE_DEMAND-abort if outside
├── compute current_position_error_, current_orientation_error_ (per-object AA angle)
├── goal-change detection → reset crossed_flag + reset all 4 buffers + reset dt_
├── if !crossed_ && pose_diff < threshold: set crossed_ + reset buffers + reset progress
├── build c3_options + lcs_factory_options via GetC3Options(crossed_flag)
├── UpdateCostMatrices(x_lcs_curr, x_lcs_des, c3_options)  # Q,R,G,U + optional quat Hessian
├── object_on_target check per object → all_reached flag
├── if achieved_fixed_goal_ OR (all_reached && kFixedGoal): fixed samples at (0.3,0.4,0.1)
├── else: candidate_states = GenerateSampleStates(strategy, num_samples, ..., unsuccessful_buffer)
├── check prev_repositioning_target_ in collision via ComputeSignedDistanceToPoint
├── if !is_doing_c3_ && !in_collision: prepend previous repos target as candidate_states[1]
├── prepend x_lcs_curr as candidate_states[0]
├── update all_sample_locations_ (per-tick clear)
├── CreateLCSObjectsForSamples(candidate_states, x_lcs_curr, lcs_factory_options)
│     └── For each sample: UpdateContext + GetResolvedContactPairs + build LCS + build LCS_for_cost
├── **#pragma omp parallel for** (over ALL samples, num_threads_to_use):
│     ├── build C3 / C3Plus object with (Q,R,G,U)
│     ├── AddLinearConstraint × N (workspace + object bounds + ee-vel + u_h/u_v)
│     ├── SetSolverOptions + Solve(test_state)                # SAME admm_iter for all
│     ├── CalcCost(cost_type or cost_type_position, ...)      # 6 possible reduction paths
│     └── all_sample_costs_[i] = c3_cost + travel_cost_per_meter · ‖xy‖
├── if i==kCurrentReposTarget && finished_reposition_flag_: costs[1] += finished_repos_cost; flag=false
├── MaintainSampleBuffers(x_lcs)
│     ├── prune unsuccessful buffer + prune main buffer by (pos, ang) delta
│     ├── overflow: shift oldest out
│     └── append new samples EXCEPT k=0 and (in repos + full size) k=1; store cost − travel
│     └── sort lowest-cost to end of buffer
├── AugmentSamplesWithBuffer(c3_objects)
│     └── (c3 mode only) if buffer best ≠ current best: add as extra sample
├── select hysteresis tier by crossed_flag (position vs pose variants)
├── best_other_cost = argmin over samples[1:]
├── **mode-switch decision:**
│     ├── c3-mode:
│     │     ├── achieved_fixed_goal → free (kNoSwitch)
│     │     ├── !met_progress → free (kToReposUnproductive)
│     │     └── curr_cost > best_other + hyst_c3_to_repos → free (kToReposCost)
│     └── free-mode:
│           ├── best_sample_index_ == kCurrentReposTarget && !in_collision → keep target (kPrevious)
│           ├── in_collision → switch to new sample (kNewSample)
│           ├── else: repos_target_cost < best_other + hyst_repos_to_repos → keep target (kPrevious)
│           ├── else: switch target + inflate best_other_cost += hyst_repos_to_repos
│           ├── wall_offset = 0.01 if near wall
│           ├── force_c3_mode → c3 (kToC3Xbox), add to unsuccessful buffer
│           ├── achieved_fixed_goal → keep free
│           └── best_other > curr + hyst_repos_to_c3 && ee_z < sampling_z+min_clearance+wall_offset
│               → c3 (kToC3ReachedReposTarget if repos_target_cost > finished_repos_cost, else kToC3Cost),
│                 add to unsuccessful buffer
├── c3_curr_plan_ = c3_objects[kCurrentLocation]; c3_best_plan_ = c3_objects[best_sample_index_]
├── UpdateC3ExecutionTrajectory(x_lcs, t)      # writes LCM traj (EE-pos + orientation + force)
├── UpdateRepositioningExecutionTrajectory(x_lcs, t)
├── update filtered_solve_time_
└── Succeeded()
```

### Port (`sampling_based_c3_controller.py:1055-2500+`)

```
compute_control(current_q, current_v, plant_ctx, target_xy, target_yaw, final_target_xy)
├── self._step += 1; N_plan=1 forced; should_solve=True per tick
├── if should_solve: plan_ctx = self._solve_plan(...)
├── return self._run_osc(current_q, current_v, plant_ctx, plan_ctx, elapsed)

_solve_plan(...):
├── SetPositions/Velocities(plant_ctx, current_q/v)               # NO x0 predict/clamp
├── (deferred [PLAN-VS-EXEC] diag dump if _pve_pending)
├── obj_xy, obj_quat, ee_pos_now = drake queries; compute g_hat, final_goal_dist
├── samples, labels = _build_samples(ee_pos_now, obj_xy, g_hat, _prev_mode, obj_quat, yaw_delta)
│     └── [current] + ([prev_repos] if repos-mode & valid) + strategy_samples + (optional buffer sample)
├── **serial** loop: results = inner_solver.evaluate_samples(samples, ...)
│     └── each sample: is_current_ee=(k==0), full_iters=(k==0)
│         └── if k==0: base_admm_iter (25); else: surrogate_iter (1)
├── c_samples = [r.c_sample for r in results]           # bonus-inflated
├── if _last_repos_finished && labels[1]=="prev_repos": c_samples[1] += finished_reposition_cost
├── if _prev_mode=="c3" && buffer.best exists: AugmentSamplesWithBuffer (append)
├── k_star = argmin(c_samples); _all_sample_costs, _best_sample_index bookkeeping
├── best_other_idx / best_other_cost = argmin over k>0
├── config_cost_now = w_obj_xy·final_goal_dist² + w_yaw·(sin(rot/2))²
├── **_prev_mode == "c3" only:** progress.update(StepMetrics(c3_cost=results[0].c_C3_raw, config, pos, rot))
├── if not _achieved_fixed_goal: check `_final_goal_dist < 0.02` + rot criterion → latch
├── crossed_switching_threshold latch on _final_goal_dist < cost_switch_thr (no buffer wipe!)
├── set near_goal = crossed_switching_threshold flag
├── sync quad_cost._crossed_switching_threshold, base_mpc._crossed_switching_threshold
├── finished_repos = _last_repos_finished (or PWL flag OR euclid tol)
├── **entry gates (port-only):**
│     ├── contact-proximity gate: if ee_to_surf ≥ 0.06 m → finished_repos=False, [ENTRY-GATE] log
│     ├── goal-align gate L1: if nhat_onto_box · g_hat ≤ entry_align_threshold → finished_repos=False
│     └── commit-face gate L2 (pre-decide): if face_align refused → finished_repos=False
├── met = progress.met_progress(near_goal); optional pos-regression override
├── ee_z altitude gate → ee_z_gate_pass (tshape-only)
├── _fresh_repos_cost = c_samples[1] if prev_repos slot, else fallback to _current_repos_cost
├── mode, reason = decide_mode(prev_mode, c_curr, best_other_cost, _fresh_repos_cost, met,
│                             near_goal, finished_repos, params, ee_z_gate_pass=ee_z_gate_pass)
├── stash _last_hyst_kind/ref/gap for [GS] diagnostic
├── if _achieved_fixed_goal && mode=="c3": force mode=free, kToReposUnproductive
├── if reason in {kToC3Cost, kToC3ReachedReposTarget, kToBetterRepos} && _avoid_unsuccessful:
│     └── unsuccessful_buffer.prune + append(BufferedSample(ee_pos_now, c_curr, obj_xy, obj_quat))
├── commit-face-post override: free→c3 with wrong face → mode=free, kStayInRepos
├── **contact-loss disengage:** if _no_ee_box_streak ≥ disengage_threshold (phase-dependent):
│     └── mode=free, kToReposUnproductive
├── on prev_mode=="free": reset _no_ee_box_streak + phaseC trackers
├── PHASE-C progress-gated exit
├── watchdog force-c3 override
├── update_buffer(results, obj_xy, obj_quat)
├── if mode=="c3":
│     ├── (kToC3ReachedReposTarget: emit [IK-LANDING], [IK-SOLVE], [BODY-VS-CONTACT] diagnostics)
│     ├── if prev_mode=="free": tracker.reset() (IK reset)
│     ├── u_opt = base_mpc.compute_control(current_q, current_v, plant_ctx, target_xy, target_yaw)
│     │     └── inside ci_mpc_c3plus: linearize_discrete at current x0 + solve
│     ├── log [CONTACT-RUN]; update _no_ee_box_streak
│     └── (many env-gated diagnostics)
└── returns plan_ctx dict; then _run_osc computes tau via OSC using _derive_force_command
```

### Side-by-side call-graph deltas

| Step | Reference | Port | Delta |
|---|---|---|---|
| 0 | `ResolvePredictedEEState` clamps x0 (c3 + repos) | Repos only via `_x_pred_repos_plan` (`:4001`); c3 uses raw `current_q, current_v` | **DELTA A** — see §5 finding 1 |
| 1 | `CheckForWorkspaceLimitViolations` DRAKE_DEMAND | Port `sampling.py:is_in_workspace` filters samples only; no per-tick invariant on actual EE | DELTA B — masks a symptom (workspace_violation crashes become sample-space filter no-ops) |
| 2 | Goal-change detection + buffer wipes (`cc:787-818`) | Missing — port assumes fixed goal per run | DELTA C — irrelevant for current benchmarks (fixed goal) |
| 3 | `crossed_cost_switching_threshold_` wipes buffers | Port latches the flag + resets progress + swaps dt/u-limits, but does NOT wipe sample buffers | DELTA D — the buffer holds pose-regime costs incompatibly with post-transition scoring |
| 4 | Sample list = `[current, prev_repos?, strategy_samples...]` (order at cc:930-938) | Port matches order (`_build_samples`) | Matches |
| 5 | **All samples get full admm_iter=3 via OpenMP parallel** | **k=0 gets 25 iters, k≥1 get 1 iter** | **DELTA E — top finding, see §5 finding 1** |
| 6 | Cost = `CalcCost(cost_type)` — 6 possible reductions; default `kSimLCS`/`kSimImpedanceObjectCostOnly` for T | Cost = `traj_cost(x_seq, u_seq, Q, R, QN, x_ref)` — SINGLE `kUseC3Plan`-equivalent path | DELTA F — cost surface differs; ranking basis differs |
| 7 | Cost has NO `w_align` / `w_travel` bonuses beyond `travel_cost_per_meter` | Port cost has `w_align`, `w_travel`, `w_rot`, `align_bonus`, `rot_bonus` deducted from `c_C3_raw` | **DELTA G — top finding, see §5 finding 3** |
| 8 | `AugmentSamplesWithBuffer` (c3 mode: add buffer's best if distinct) | Port matches (`sampling_based_c3_controller.py:1417-1442`) | Matches |
| 9 | Mode-switch dispatcher runs on raw `all_sample_costs_` | Port mode-switch runs on bonus-inflated `c_samples` | DELTA H — decision-space differs from cost-tracking space |
| 10 | Two-branch dispatcher (c3, free) — 4 c3-exit reasons, 5 free-exit reasons | Port has 4 c3-exit + 5 free-exit + 3 port-only overrides (contact-loss, commit-face-post, watchdog) + 3 entry gates | DELTA I — additional side gates before/after `decide_mode` |
| 11 | `UpdateC3ExecutionTrajectory` writes N-knot LCM traj (EE-pos + orientation + force), holds z constant, tilts EE | Port skips this: `_p_ee_des = FK(x_seq[1])`, orientation held constant by OSC posture, no wall-offset z-raise | DELTA J — one-knot lookahead vs N-knot spline; matters for OSC responsiveness |
| 12 | `filtered_solve_time_` = α-filtered; feeds `x_pred_curr_plan_` = interpolated knot | Port `_step_times_ms` for stats only; no predictor interpolation | DELTA K — coupled with DELTA A |
| 13 | Reposition via `Reposition()` closed-form knots (5 traj types) | Port via IK-per-knot (`RepositionIKTracker`) OR PWL 3-knot (`PiecewiseLinearTracker`) — different tolerance semantics | DELTA L — landed 2026-07-14 (map §2) |

---

## 3 — Per-tick control-flow diff detail

### 3a — Sample generation (which samples, filter, ordering, buffer usage)

**Reference (`cc:886-949`, `generate_samples.cc:24-165`):**
- `num_additional_samples_repos` or `_c3` chosen by mode
- Strategies: 8 (`kRadiallySymmetric`, `kRandomOnCircle`, `kRandomOnSphere`, `kFixed`, `kRandomOnPerimeter`, `kRandomOnShell`, `kMeshNormal`, `kMeshNormalMultiObject`)
- `SampleIsAcceptable`: workspace (bool `filter_samples_for_safety`) + `SampleAvoidsBadSpots` (via unsuccessful buffer)
- Fixed-goal override: pin all samples at `(0.3, 0.4, 0.1)`
- Repos mode: `candidate_states[0] = x_lcs_curr; candidate_states[1] = prev_repositioning_target_ (if not in collision)`; rest are new samples
- C3 mode: `candidate_states[0] = x_lcs_curr;` rest are new samples (buffer's best appended by `AugmentSamplesWithBuffer`)
- Ordering is authoritative — `SampleIndex::kCurrentLocation = 0`, `kCurrentReposTarget = 1` (enum in `.h:51-56`)

**Port (`sampling.py:90`, `sampling_based_c3_controller.py:898-988`):**
- 5 strategies (`kRandomOnCircle`, `kRadiallySymmetric`, `kFaceNormal`, `kRandomOnPerimeter`, `kFixed`); missing `kRandomOnSphere`, `kRandomOnShell`, `kMeshNormal`, `kMeshNormalMultiObject`
- Local `kFaceNormal` strategy (Push-Anything §IV-B1) — the port's canonical sampler for box/T
- `is_in_workspace` filters workspace only (`sampling.py:809`); the `SampleAvoidsBadSpots` equivalent is applied at the **wrapper** level in `_update_buffer` and `unsuccessful_buffer.prune`, not inside the sampler
- Ordering matches reference: `[current, prev_repos?, strategy_samples..., buffer_sample?]`
- Env `PORT_ALL_SAMP=1` dumps `[ALL-SAMP]` per tick

**Diff:**
- Port's `kFaceNormal` produces samples with `sampling_setback=0.030 m` outside the box surface. Reference has no direct equivalent; `kMeshNormal` uses face-normal projection with `sample_projection_clearance` (different name, same idea).
- Port's `unsuccessful_radius` default is 0.010 m (aggressive small window) vs reference `push_t` yaml's `unsuccessful_radius: 0.05` (checked at `sampling_params.yaml`, not reproduced here). Port already flags this in `params.py:288`.
- **Ordering + slot semantics are aligned.** Cost inflation at slot 1 (`finished_reposition_cost`) matches.

### 3b — Per-sample inner-solve dispatch

**Reference (`cc:969-1085`):**
- **All samples run in `#pragma omp parallel for num_threads(num_threads_to_use_)`**
- Each thread constructs its own `C3` / `C3MIQP` / `C3QP` / `C3Plus` object with `(Q_, R_, G_, U_)`
- Each solve calls `test_c3_object->Solve(test_state)` **with the same `admm_iter` regardless of sample index** — `admm_iter=3` for push_t (`sampling_c3plus_options.yaml:2`)
- Then `CalcCost(cost_type, lcs_for_cost, ...)` — a SEPARATE LCS `lcs_candidates_for_cost` built with `resolve_contacts_to_for_cost` (different contact resolution!) is used for the cost reduction path
- If `cost_type == kSimImpedance*`: rolls PD control forward via `TrajectoryEvaluator::SimulatePDControlWithLCS`
- Fixed `travel_cost_per_meter · ‖xy_travel‖` added
- Finished-repos: `all_sample_costs_[kCurrentReposTarget] += finished_reposition_cost` if flag set (flag then reset)

**Port (`inner_solve.py:333-1144`):**
- **Serial by default** (`_num_threads_to_use = 1`, `PORT_NUM_THREADS_TO_USE` env overrides); a parallel path exists but only lazily instantiated
- **Asymmetric iters**: `admm_iter_k = base_admm_iter if (is_current_ee or full_iters) else surrogate_iter` (`inner_solve.py:467`)
- **Only ONE LCS per sample**, no `lcs_for_cost` — the same LCS drives both `Solve` and `traj_cost`
- **Cost = `traj_cost(x_seq, u_seq, Q, R, QN, x_ref)`** — always the same reduction; no `SimulateLCSOverTrajectory`, no `SimulatePDControlWithLCS`, no `_object_only_cost_matrices_ee_space` (tshape-only; box path uses full `Q, R, QN`)
- **Bonuses (`c_sample = c_C3_raw − w_align·align_score − w_rot·rot_score + w_travel·travel`)** applied AFTER the cost is computed — no reference equivalent
- Optional `use_cost_lcs_ranking` path (`inner_solve.py:321-329`, tshape only) runs a PGS LCS rollout to compute `c_C3_raw`; still not the reference's `CalcCost` branches

**Diff (deltas from the map, since 2026-07-14):**
- **DELTA E (top-finding, new since 2026-07-14 map)** — asymmetric per-sample admm iters. This means the "k=0 vs the k≥1 competition" is not apples-to-apples. When G-on destabilizes ADMM (item-#7 arc-2), the k=0 25-iter cost is bounded by its worst iterate; k≥1 1-iter costs are essentially just projected initial conditions. Argmin under this asymmetry favours whichever slot has smaller magnitude noise — usually k≥1 (they never get the chance to accumulate 25 iterations of leakage).
- **DELTA F** — `CalcCost` cost-type variants are absent. Reference `push_t` uses `cost_type = kSimImpedanceObjectCostOnly` for the sample cost (`progress_params.yaml`), which zeros out robot-Q and R and rolls PD-controlled LCS forward. Port's `traj_cost` uses the raw C3 output (equivalent to `kUseC3Plan`). Cost surface for sample ranking differs by construction. This is not new since 2026-07-14 but the conformance map's §5.b/c row lists it as LOAD-BEARING for T (and confirmed inert for box under `use_cost_lcs_ranking=False`).
- **DELTA G (top-finding)** — bonuses on top of `c_C3_raw`. The `w_align=30000` bonus dominates cost values at typical operating points (box path `c_C3_raw ~ 20000`), so ranking is bonus-dominated. This is documented in `params.py:20-25` as "empirically required to overcome friction-cone discretization bias" — a compensation-for-a-compensation that would not be needed if the reference's `kSimImpedance*` cost-LCS path were ported.

### 3c — Mode-switch decision

**Reference (`cc:1146-1310`):**
- 6-value `ModeSwitchReason` enum (`.h:58-65`): `kNoSwitch, kToC3Cost, kToC3ReachedReposTarget, kToReposCost, kToReposUnproductive, kToC3Xbox`
- Hysteresis is TWO-tier only: `_position` variant (far from goal) vs unnamed (near goal), triggered by `crossed_cost_switching_threshold_`. Within each tier, `use_relative_hysteresis` selects fractional-of-cost-ref vs absolute — same field structure as port
- Order (c3 branch):
  1. `achieved_fixed_goal_` → free (kNoSwitch — because they're pinned)
  2. `!met_minimum_progress && !force_c3_mode && num_samples_c3 > 0` → free, `kToReposUnproductive`
  3. `curr_cost > best_other + hyst_c3_to_repos && !force_c3_mode` → free, `kToReposCost`
  4. else: stay in c3
- Order (free branch):
  1. Choose target: prev-repos (if not in-collision + cost close) OR new-sample (`kFromBuffer` if best_sample_index > num_c3+1). If new-sample, inflate best_other_cost by `hyst_repos_to_repos`
  2. Compute `wall_offset` (0.01 if near workspace wall, else 0)
  3. `force_c3_mode` → c3, `kToC3Xbox` + add to unsuccessful buffer
  4. `achieved_fixed_goal_` → stay in free
  5. `best_other > curr + hyst_repos_to_c3 && ee_z < sampling_z + c3_min_clearance + wall_offset` → c3, `kToC3ReachedReposTarget` if `repos_target_cost > finished_repos_cost`, else `kToC3Cost`; add to unsuccessful buffer
- No "contact loss" gate, no "entry proximity" gate, no watchdog, no phase-A/B/C state machine

**Port (`mode_switch.py:130-214` + `sampling_based_c3_controller.py:1900-2205`):**
- 8-value `SwitchReason` enum: `kStayInC3, kStayInRepos, kToReposCost, kToReposUnproductive, kToC3Cost, kToC3ReachedReposTarget, kToBetterRepos, kForceC3Watchdog`
- Additional port-only outer-controller states/gates:
  - `contact_entry_threshold` / `contact_entry_surface_threshold` (default 0.09 m / 0.06 m surface) — blocks `kToC3ReachedReposTarget` when EE too far from box
  - `entry_align_threshold` — blocks when contact normal anti-goal
  - `use_commit_face_gate` (L2 pre-decide + L2 post-decide) — blocks when face is anti-goal
  - `_no_ee_box_streak` disengage (phase-dependent thresholds `contact_loss_threshold_*_s`)
  - `_approach_override_phase` ∈ {A_lift_trav, B_descend, C_approach} with distinct disengage thresholds
  - `_phaseC_stall_streak` / `_phaseC_active_streak` — independent stall & hard-cap gates
  - `watchdog_steps_since_improve_threshold` → force c3 with `kForceC3Watchdog`
  - `_achieved_fixed_goal` override (no reference "pin sample at (0.3, 0.4, 0.1)" behavior)
  - EE_z altitude gate (tshape-only) at `_ee_z_gate_pass` passed into `decide_mode`
- `decide_mode` ordering (`mode_switch.py:158-214`):
  - c3: `!met_progress` → free `kToReposUnproductive`; `best_other + gap < c3_cost` → free `kToReposCost`; else stay
  - free: (1) `_repos_to_repos_would_fire` check (inflates `best_other_cost` in-place), (2) c3-entry gate `c3_cost + gap_back < best_other_cost && ee_z_gate_pass` → c3 (`kToC3ReachedReposTarget` if finished_repos else `kToC3Cost`), (3) inflated case → `kToBetterRepos`, (4) else `kStayInRepos`

**Diff:**
- The core cost-gap arithmetic is the same shape.
- Port has **10+ additional outer gates** that fire before/around `decide_mode`. Each is documented as chasing a specific empirical failure (contact-loss disengage, wrong-face entry, PHASE-C hard cap, altitude gate). Under the G-on ADMM instability regime, several of these gates can trigger cascading transitions:
  - `_no_ee_box_streak` counts consecutive c3 steps with no admitted EE-BOX pair. When ADMM produces noisy `λ_n` (item-#7 arc-2's 70-80% non-monotone iters), the streak accumulates faster because the planner never emits contact-forming EE positions.
  - `kToBetterRepos` also fires `AddToUnsuccessfulBuffer` (port-only), which prunes buffer samples within `unsuccessful_radius`. Rapid repos-target churn under ADMM instability erodes the buffer.
  - `_achieved_fixed_goal` is sticky and force-mode=free — noisy G-on ADMM may push the box near-goal transiently, latch the flag, and freeze the controller.
- **`met_progress` divergence:**
  - Reference uses `all_sample_costs_[kCurrentLocation]` (post-travel-cost, no bonuses) — same figure as the decision gate
  - Port uses `results[0].c_C3_raw` (pre-bonus quadratic), while `decide_mode` sees `c_curr = c_samples[0]` (bonus-inflated)
  - **When `w_align·align_score` swings, mode-switch cost gap and progress metric are in DIFFERENT COST SPACES.**

### 3d — Executor dispatch (target `p_ee_des`, force `λ_des`, sign conventions)

**Reference (`cc:1700-1836` UpdateC3ExecutionTrajectory + downstream OSC in `franka_osc_controller.cc`):**
- `p_ee_des` published as an N-knot LcmTrajectory `end_effector_position_target` with `dt = filtered_solve_time_ + i·dt`
- z-coord held constant at `sampling_params_.z_height + wall_offset` (raise to `+0.01` if near wall)
- EE-orientation quaternion via `direction · (rot_90_around_z)` + `max_tilt_angle · (‖direction‖ / max_dist)` — a tilt AWAY from workspace center (see `cc:1783-1808`). Publishes as `end_effector_orientation_target`
- EE-force target: `force_traj = u_sol` (planner's raw output). Published as `end_effector_force_target`
- Downstream OSC (external `franka_osc_controller.cc`) consumes these three LCM channels, drives OSC QP with position-tracking + force-tracking + posture cost — force-tracking authority per `osc_params.yaml`

**Port (`sampling_based_c3_controller.py:1055 → _run_osc:2934 → _derive_force_command:580`):**
- `p_ee_des = FK(x_seq[1])` — just the NEXT knot of the plan (per CLAUDE.md and `sampling_based_c3_controller.py:1105-1112` design)
- No EE orientation target; OSC posture cost pins q_arm_nominal
- No z hold at `sampling_height + wall_offset`; z comes from `x_seq[1]` planner output
- `λ_des = _derive_force_command(lambda_n, g_hat_3d)`:
  - Direction: `−g_hat` (recoil-on-arm convention)
  - Magnitude: `max(Σ|λ_n|, min_push_force)` if planner emitted EE-BOX pair; else `nominal_push_force`
  - Optional `PORT_FORCE_ROUTING=u_sol` recovers Cartesian force from planner's `u_seq[0]` (EE-space planner) or `pinv(J^T)·u` (R^7 planner); default OFF
- Direction convention documented at `sampling_based_c3_controller.py:594-596`: "box goes west ⇒ EE on east ⇒ recoil east = −g_hat for g_hat=[−1,0,0]"

**Diff:**
- **Reference publishes N-knot Cartesian traj + tilt + wall-offset z-raise;** port shortcuts to knot 1. Under noisy G-on ADMM, `x_seq[1]` is more volatile per tick than an α-filtered `filtered_solve_time_`-interpolated knot would be.
- **Reference `force_traj = u_sol` — the planner's raw u vector,** which in `sampling_c3plus_options.yaml:34-35` is bounded to `[-50, 50] N` horizontal and `[-50, 50] N` vertical. Port `_derive_force_command` **fabricates** `λ_des = mag · (−g_hat)` where `mag = max(Σ|λ_n|, min_push_force=2.0)` or `nominal_push_force=5.0`; the actual planner `u_seq` is only used via env-gate `PORT_FORCE_ROUTING=u_sol` (default off).
- **Sign convention identical** (recoil convention on arm side).
- Port `min_push_force=2.0` / `nominal_push_force=5.0` (`params.py:533, 538`) are drastically smaller than reference `u_horizontal_limits=[±50, ±50]` in `sampling_c3plus_options.yaml`. This is because the reference publishes `u_sol` as *target*, letting OSC modulate; port publishes the `−g_hat`-fabricated command as *target* directly with the planner's magnitude as a hint.

### 3e — Warm-start plumbing across ticks + between full-iter / surrogate solves

**Reference:**
- `sampling_c3plus_options.yaml:10` sets `warm_start: false` for push_t.
- `cc:1325` comment: `// TODO If doing warmstarting, will need to save z, delta, and w vectors.` — **feature not implemented on the outer side.**
- OSQP inner-loop warm-start inside a single ADMM iteration is a matter for the C3 solver (out of scope here).

**Port:**
- `admm_solver.py:112` sets OSQP `warm_starting=1` inside a single ADMM iter — consistent with reference default OSQP behavior.
- No cross-tick warm-start of ADMM state (ω, δ, ρ) either.

**Diff:** **None on outer-loop warm-start plumbing.** Both sides restart the ADMM from a fresh initial state each tick. This is a *symmetric* absence — worth noting because the coupled ρ/G issue may be sensitive to warm-start OFF (loss of ADMM history means every tick pays the full "walk toward the fixed point" cost, and G-diag augmentation shifts where the fixed point IS).

---

## 4 — Config / parameter deltas

Only outer-controller params are listed. Inner ADMM params (rho_scale, G, delta_option) are the sibling agent's scope.

| Reference param | Ref default (push_t) | Port default | Location & delta |
|---|---|---|---|
| `use_predicted_x0_c3` | `true` | N/A (c3 side missing) | `sampling_c3plus_options.yaml:38` — port has no c3-side wire |
| `use_predicted_x0_repos` | `true` | `true` (implicit, `_x_pred_repos_plan` populated when non-null) | `sampling_based_c3_controller.py:269, 4011` |
| `use_predicted_x0_reset_mechanism` | `false` | N/A | `sampling_c3plus_options.yaml:40` — port doesn't implement the reset heuristic |
| `nominal_ee_accel` | (yaml value, not shown in inspected snippet) | via `getattr(self.base_mpc, "nominal_ee_accel", 2.0)` — default 2.0 m/s² | `sampling_based_c3_controller.py:4009-4010` |
| `include_walls` | (yaml default) | N/A on port | `cc:996-1024` — port never adds workspace linear constraints inside C3 |
| `workspace_limits` | 3-row array `[x, y, z, lb, ub]` | `params.py:387-390` (`workspace_xy_min/max`, `workspace_z_min/max`) | Same numeric intent; port filters SAMPLES not the ACTUAL EE (DELTA B) |
| `workspace_margins` | `0.02` | Not enforced on live EE, only on sample rejection | `sampling_c3plus_options.yaml:33` |
| `robot_radius_limits` | `[0.25, 0.75]` | present in `params.py` but no live-state check | `sampling_c3plus_options.yaml:32` |
| `ee_velocity_limits` | present as C3 linear constraint (`cc:1028-1035`) | Port does NOT add C3 velocity constraint | DELTA — inner-solve constraint set differs |
| `u_horizontal_limits` | `[-50, 50]` | via `PORT_U_HORIZONTAL` env or scalar `torque_limit` | `sampling_c3plus_options.yaml:34`, `inner_solve.py:308-315` |
| `u_vertical_limits` | `[-50, 50]` | via `PORT_U_VERTICAL` env | `sampling_c3plus_options.yaml:35`, `inner_solve.py:308-315` |
| `num_outer_threads` | `4` | `1` unless `PORT_NUM_THREADS_TO_USE` set | `sampling_c3plus_options.yaml:6`, `inner_solve.py:358-367` — DELTA (serial vs parallel, but bit-equivalent) |
| `num_threads_to_use_` | derived from `num_outer_threads` | `1` | Port parallel path exists but disabled by default; not an amplifier for G-on but does change RNG order |
| `solve_time_filter_alpha` | `0.95` (heavy filter) | Not implemented | `sampling_c3plus_options.yaml:14` |
| `control_loop_delay_ms` | (yaml, non-zero to sync with hardware) | Not implemented | `sampling_c3_controller_params.h:37` |
| `travel_cost_per_meter` | (yaml value, weighted `‖xy_travel‖`) | Present in `params.py` as `travel_cost_per_meter` (progress_params) | Matches structurally |
| `finished_reposition_cost` | (yaml value) | Port default `1.0e9` (`params.py:203`) — DIFFERENT magnitude, port uses it as a "gigantic inflation" to force mode transitions | DELTA — see mode-switch notes |
| `hyst_c3_to_repos_frac` etc. | (yaml values) | Present, matches structurally | Values conf-mapped 2026-07-14 §5 |
| `use_relative_hysteresis` | `false` (push_t) | Port default `False` | Matches |
| `use_quaternion_dependent_cost` | `true` | Port supports via `task_costs.py` build (near-goal only) | Present, see conformance-map §4 |
| `q_quaternion_dependent_weight` | `1000` | Port config `tasks.yaml` for T | Present |
| `Kp_for_ee_pd_rollout` / `Kd_for_ee_pd_rollout` | `100 / 0.5` | Port `100.0 / 0.5` at `inner_solve.py:323-326` — but only used for `use_cost_lcs_ranking` path (tshape) | Matches structurally |
| `pos_error_sample_retention` | (yaml value) | Port default `0.05 m` (`params.py:271`) — matches ref |
| `ang_error_sample_retention` | (yaml value) | Port default `0.30 rad` (`params.py:272`) — matches ref |
| `unsuccessful_radius` | `0.05` (ref anything/push_t) | Port default `0.010 m` (`params.py:288`) — port is 5× tighter | DELTA — port flags this as "matches reference"; needs re-check against push_t yaml |
| `unsuccessful_pos/ang_error_sample_retention` | (ref values) | Port `0.006 m / 0.05 rad` — port claims ref-matched at `params.py:289-290` | Verify vs `push_t/parameters/sampling_params.yaml` |
| `avoid_choosing_unsuccessful_samples` | `true` | Port `True` | Matches |
| `consider_best_buffer_sample_when_leaving_c3` | `true` | Port `True` | Matches |
| `N_sample_buffer` | (yaml value) | Port `5` (`params.py:270`) |
| `pwl_waypoint_height`, `speed`, etc. | Reference reposition trajectory params | Port has both PWL tracker + IK tracker, controlled by `use_reposition_pwl_trajectory` | See conformance-map §2 |

**Port-only params (no reference equivalent):**

- `w_align` (`params.py:512`, default `30000.0`) — dominant sample-ranking bonus
- `w_travel` (default `200.0`), `w_rot` (default `0.0`)
- `nominal_push_force` (`5.0`), `min_push_force` (`4.0` per recent commit b8f31f9)
- `W_force`, `use_force_tracking` — OSC force-tracking specifics
- `contact_entry_threshold`, `contact_entry_surface_threshold`, `entry_align_threshold`, `use_contact_entry_gate`, `use_surface_entry_gate`
- `contact_loss_threshold_default_s`, `contact_loss_threshold_with_override_s`, `contact_loss_threshold_phaseA_ltd_s`, `contact_loss_threshold_phaseB_ltd_s`, `phaseC_stall_threshold_s`, `phaseC_hard_cap_s`, `phaseC_progress_eps`
- `watchdog_steps_since_improve_threshold`
- `use_commit_face_gate`, `commit_face_align_threshold`
- `pos_regression_threshold`
- `sampling_setback`, `sample_reject_clearance`, `box_half_extent`, `sampling_height`, `sampling_radius`, `repos_target_radius` — geometric shims for the local `kFaceNormal` strategy
- `use_cost_lcs_ranking`, `cost_lcs_pgs_max_iter`, `cost_lcs_pgs_tol`, `cost_lcs_pgs_reg`, `Kp_for_ee_pd_rollout`, `Kd_for_ee_pd_rollout` — tshape-only cost-LCS rollout

**Reference params missing on port:**

- `use_predicted_x0_c3`, `use_predicted_x0_reset_mechanism`, `x_pred_curr_plan_` (c3 side only)
- `solve_time_filter_alpha`
- `control_loop_delay_ms`
- `radio` inputs (Xbox channels 6, 7, 11, 12, 14)
- `include_walls`, `include_back_wall`
- `sample_projection_clearance` (analog to port's `sample_reject_clearance`)
- `max_tilt_angle` (EE tilt away from workspace center in c3-mode traj)

---

## 5 — Ranked outer-loop divergences most likely to amplify inner-loop G-on instability

Ranked highest→lowest by causal proximity to the G-diag ADMM instability signature (70-80% non-monotone iters, workspace crashes at ρ=1, dual explosion at ρ=10).

### Rank 1 — Asymmetric per-sample ADMM iters (DELTA E)

**Signal:** `inner_solve.py:467, 1120` — k=0 gets `base_admm_iter=25`, k≥1 get
`surrogate_admm_iters=1`. Reference: `cc:971-1085` runs all samples at `admm_iter=3`
(same value). See `sampling_c3plus_options.yaml:2`.

**Why amplifies G-on instability:** The 2026-07-23 investigation memo notes item-#7's
real issue is coupled ρ/G/Q. When G augmentation destabilizes ADMM, the k=0 25-iter
solve accumulates leakage over 25 iterations; k≥1 solves stop before leakage
accumulates. The bonus-inflated `c_sample` argmin then compares apples to oranges,
and the k=0 cost is over-penalized by both leakage AND bonus. Under baseline
(G-off), the 25-vs-1 asymmetry is masked by convergence at ρ=10 (arc-1 finding);
under G-on, no ρ converges (arc-2 finding), so the asymmetry becomes visible.

**Falsifier design:** cross the port's `admm_iter` (25) with the reference's
outer setting `admm_iter=3` and test whether ρ-sweep behaviour matches the
reference. If the port converges more at admm_iter=3-across-all-samples, DELTA E
is a first-order amplifier.

### Rank 2 — c3-mode `x_pred_curr_plan_` not implemented (DELTA A + DELTA K)

**Signal:** `cc:1406-1454` (`ResolvePredictedEEState`) + `cc:1457-1472`
(`ClampEndEffectorAcceleration`) run every tick. The clamped `x_lcs_curr` is
what gets used for LCS linearization at `cc:855-857`. Port `_solve_plan` never
clamps — it uses raw Drake state. Also `cc:1718, 1727-1732` interpolates
`x_pred_curr_plan_` from the plan knots at `filtered_solve_time_ / dt`, which
smooths the tick-to-tick input to the plan.

**Why amplifies G-on instability:** The ADMM subproblem is a linearization of
the LCS at x0. If x0 is jitter-noisy (Drake state fresh every tick), the
linearization coefficients (A, B, D, E, F, H, c) jitter too. G-diag augmentation
adds a per-slot penalty on `(λ, η)` distance to `(λ_prev, η_prev)` — but there
is no `λ_prev`/`η_prev` across ticks (warm-start off both sides, item §3e). The
jitter feeds directly into the augmented objective. Reference's α-filtered
`filtered_solve_time_ = 0.95` + `x_pred_curr_plan_` interpolation dampens this.

**Falsifier design:** implement a repos-mode-style `_x_pred_c3_plan_` in the
port (interpolated from `last_x_seq` at `_dt_ctrl` since last solve) and clamp
`current_q, current_v` head-3 to `x_pred ± nominal_ee_accel · dt²`. Re-run the
G-on ρ-sweep and check whether non-monotone ADMM iter rate drops.

### Rank 3 — Bonus-inflated cost drives mode-switch, raw cost drives progress (DELTA G + DELTA H)

**Signal:** `sampling_based_c3_controller.py:1543-1547` feeds `c_C3_raw` to progress
tracker; `mode_switch.decide_mode` sees `c_curr = c_samples[0]` (bonus-inflated).
Reference `cc:2236-2240` and `cc:1147-1149` both use `all_sample_costs_[kCurrentLocation]`
— the SAME cost value.

**Why amplifies G-on instability:** G-on regressions typically show as slow / non-existent
progress on the primary metric while the argmin sample flips per tick. The port's
dual-metric split means `met_progress` says "still improving" (raw quadratic slowly
decreasing) while the cost gap says "another sample is 30k below" (align_bonus deals
away 30k). Result: `kToReposCost` fires while progress tracker thinks C3 is winning.
Under noisy G-on ADMM, sample ranking flips more often, so `kToBetterRepos` cascades
more often, which invokes `AddToUnsuccessfulBuffer` more often, which erodes sample
diversity.

**Falsifier design:** temporarily route BOTH progress tracker and mode-switch through
the same cost expression (either both raw or both bonus-inflated). Sample flip rate
and mode-transition rate should drop.

### Rank 4 — `_no_ee_box_streak` disengage under G-on noise

**Signal:** `sampling_based_c3_controller.py:2087-2141` — after `contact_loss_threshold_*_s`
consecutive c3 steps with no admitted EE-BOX pair, force `kToReposUnproductive`.
Reference has NO equivalent gate.

**Why amplifies G-on instability:** When G-on causes ADMM to emit noisy or zero
λ, `_last_contact_info` reports no EE-BOX pair (because Drake's 2 mm signed-distance
threshold rejects sub-mm mis-linearized configs). The streak counter fires, forcing
c3→free. This is not a bug per se — it's the "safety" gate the port added to escape
ADMM-stuck states. But under G-on, the "safety exit" becomes the primary transition
driver: c3 modes never last long enough to accumulate physical box motion.

**Falsifier design:** set `PORT_DISABLE_CONTACT_LOSS_GATE=1` (existing env override
at `sampling_based_c3_controller.py:2100`) and re-run G-on p82. If push distance
improves, the gate is a secondary amplifier.

### Rank 5 — `finished_reposition_cost=1e9` mode-transition lever

**Signal:** Port default `1e9` (`params.py:203`) vs reference push_t yaml (smaller,
verify — commonly O(10-100)). Port uses this cost inflation as the primary trigger
for `kToC3ReachedReposTarget` (`mode_switch.py:183-190`).

**Why amplifies:** Port relies on `finished_reposition_cost` to inflate slot-1 cost
so that the c3-entry hysteresis check `c3_cost + gap_back < best_other_cost` fires.
When G-on ADMM produces near-random `c_C3_raw`, this `1e9` inflation dominates
regardless. The result: mode transitions happen on inflation arithmetic, not on
actual "did the EE reach the target" physics. Reference's smaller inflation +
IK/PWL closed-form "finished" signal decouple this.

---

## 6 — Concrete falsification probes (env-var-driven, do NOT run)

Design three probes to isolate the top-3 candidates. Each is a single env-var
switch that toggles port behavior toward reference behavior on ONE axis.

### Probe P1 — Symmetric ADMM iters across samples

- **Purpose:** Falsify Rank-1 DELTA E.
- **Env:** `PUSHA_SYMMETRIC_ADMM_ITERS=1` (**new — must be added at `inner_solve.py:467`**).
  When set, override `admm_iter_k = base_admm_iter` for ALL k, regardless of `is_current_ee` / `full_iters`.
- **Also set:** `--admm-iter 3` on the main.py CLI (matches reference push_t yaml).
- **Prediction if amplifier:** G-on ρ-sweep — trans/rot metrics improve at ρ=10 (arc-1's ρ=10 sweet spot); non-monotone ADMM-iter rate (%) drops from 70-80% to <30%.
- **Prediction if not amplifier:** metrics unchanged; ADMM-iter rate unchanged.
- **Cost of implementation:** ~5 lines in `inner_solve.py` behind env-gate.

### Probe P2 — c3-mode `x_pred` clamp

- **Purpose:** Falsify Rank-2 DELTA A + DELTA K.
- **Env:** `PUSHA_C3_XPRED=1` (**new — must be added at `sampling_based_c3_controller.py:1256-1258`** just after `SetPositions/Velocities`).
  When set, if `_last_x_seq is not None` and `_last_plan_tick > 0`, interpolate `x_pred = _last_x_seq[interp_idx]` at `interp_idx = (self._step - _last_plan_tick) * _dt_ctrl / dt`; clamp `current_q, current_v` head-3 to `x_pred_head3 ± nominal_ee_accel · dt²` per axis. `nominal_ee_accel` from `getattr(base_mpc, 'nominal_ee_accel', 2.0)`.
- **Prediction if amplifier:** G-on p82 numbers approach G-off ρ=10 (arc-1 baseline: trans=0.021, rot=0.064); jitter in `[X-SEQ-PROBE]` output diagnostic tightens.
- **Prediction if not amplifier:** metrics essentially unchanged (jitter in x0 was not the dominant driver of G-diag divergence).

### Probe P3 — Single-cost mode-switch (unify progress + gate)

- **Purpose:** Falsify Rank-3 DELTA G + DELTA H.
- **Env:** `PUSHA_UNIFIED_COST=1` (**new — must be added at `sampling_based_c3_controller.py:1543-1547` and `:1900`**).
  When set: progress tracker feeds `c_curr = c_samples[0]` (bonus-inflated, same as mode gate); mode gate unchanged. This makes both metrics live in the SAME space.
- **Prediction if amplifier:** `kToReposCost` transition rate drops; `_no_ee_box_streak` firings drop (fewer premature exits); box translation improves.
- **Prediction if not amplifier:** minor change; the true issue is deeper.

**Falsification matrix:**

| P1 fires only | P2 fires only | P3 fires only | Multiple fire | Nothing improves |
|---|---|---|---|---|
| Asymmetric iters is root; fix at inner_solve.py | x0 jitter is root; port `x_pred` for c3 mode | Cost-space split is root; unify feed | Composite effect; run 2-way A/B/C to attribute | Investigate ranks 4-5 or return to inner-solver diff |

---

## 7 — Direct citation table (audit)

Every claim in this report is grounded at file:line-line. Cross-reference here.

| Claim | Reference (cc / h / yaml) | Port |
|---|---|---|
| Per-tick entry `ComputePlan` / `compute_control` | `cc:722-1399` | `sampling_based_c3_controller.py:1055-1142` (thin), `:1237-2500+` (`_solve_plan`) |
| Sample ordering `[current, prev_repos?, samples...]` | `cc:930-938` | `sampling_based_c3_controller.py:898-988` (`_build_samples`) |
| `AugmentSamplesWithBuffer` | `cc:2106-2158` | `sampling_based_c3_controller.py:1417-1442` |
| `MaintainSampleBuffers` (prune + insert + sort) | `cc:2002-2102` | `sampling_based_c3_controller.py:988-1054` + `sample_buffer.py:95-125` |
| `AddToUnsuccessfulBuffer` | `cc:2161-2205` | `sampling_based_c3_controller.py:1945-1976` |
| `ResolvePredictedEEState` / clamp | `cc:1406-1472` | Repos-only at `sampling_based_c3_controller.py:4001-4043`; c3 missing |
| `CheckForWorkspaceLimitViolations` | `cc:1476-1494` | Missing (sample filter only at `sampling.py:809`) |
| Cost-switching threshold + buffer wipes | `cc:800-848` | `sampling_based_c3_controller.py:1588-1608` (partial: no buffer wipe) |
| `achieved_fixed_goal_` | `cc:887-897, 1160-1164, 1279-1283` | `sampling_based_c3_controller.py:1559-1579, 1931-1933` |
| Per-sample ADMM iters | `cc:1055-1058` + `sampling_c3plus_options.yaml:2` (admm_iter=3, uniform) | `inner_solve.py:467, 1120` (k=0 → 25, k≥1 → 1) |
| `CalcCost` cost-type variants | `cc:493-720` (6 types) | `inner_solve.py:213-830` (1 type, +tshape cost-LCS) |
| `travel_cost_per_meter` addition | `cc:1074-1078` | `inner_solve.py:792-793` (`w_travel * travel_dist`) |
| Bonus (`w_align`, `w_rot`, `w_travel`) | Not present in reference | `inner_solve.py:753-810`, `params.py:20-25` |
| `KeepTrackOfC3ModeProgress` | `cc:2208-2305` | `progress.py:118-215`, feed at `sampling_based_c3_controller.py:1543-1547` |
| `finished_reposition_cost` inflation | `cc:1081-1084` | `sampling_based_c3_controller.py:1410-1415` |
| Mode-switch order (c3 branch) | `cc:1146-1201` | `mode_switch.py:158-168` (in `decide_mode`), plus port-only gates around it |
| Mode-switch order (free branch) | `cc:1202-1310` | `mode_switch.py:170-214` |
| `use_predicted_x0_c3/repos` YAML defaults | `sampling_c3plus_options.yaml:38-40` | Port implements only repos side |
| `warm_start: false` | `sampling_c3plus_options.yaml:10` | `admm_solver.py:112` OSQP warm_starting=1 (inner) but no outer ω/δ carry |
| Reference outer TODO for warm-start | `cc:1325` | N/A |
| Reference `#pragma omp parallel for` | `cc:971-1085` | `inner_solve.py:1108-1144` serial default |
| `UpdateC3ExecutionTrajectory` LCM traj | `cc:1700-1836` | Missing — `_run_osc` shortcuts to `x_seq[1]` |
| `UpdateRepositioningExecutionTrajectory` | `cc:1839-1940` | `sampling_based_c3_controller.py:3900-4050`, `reposition_trajectory.py`, `reposition_ik.py` |
| `filtered_solve_time_` α-filter | `cc:1387-1391, 1718` | Missing |
| Contact-loss disengage | Not present | `sampling_based_c3_controller.py:2087-2141` |
| Contact-proximity entry gate | Not present | `sampling_based_c3_controller.py:1724-1749`, `params.py:contact_entry_*` |
| Goal-align gate L1 | Not present | `sampling_based_c3_controller.py:1759-1785` |
| Commit-face gate L2 | Not present | `sampling_based_c3_controller.py:1800-1811, 2030-2048`, `commit_face_gate.py` |
| Watchdog force-c3 | Not present | `sampling_based_c3_controller.py:2187-2203` |
| PHASE A/B/C state machine | Not present | `sampling_based_c3_controller.py:2087-2185`, `params.py:contact_loss_threshold_phase*`, `phaseC_*` |
| EE-z altitude gate (tshape) | `cc:1290-1293` | `sampling_based_c3_controller.py:1866-1888` |
| `SwitchReason` enum | `.h:58-65` (5 non-none reasons) | `mode_switch.py:34-48` (7 non-none reasons including 2 port-only) |
| Bug: 8-sample-limited strategies | `generate_samples.cc:24-165` (8 strategies) | `sampling.py:90-131` (5 strategies) |

---

## 8 — Handoff to inner-solver sibling agent

Findings observed while walking the outer pipeline that BELONG to the inner
solver diff — do not act on these here, hand to the sibling:

- **Reference `push_t` YAML settings**: `admm_iter=3`, `rho_scale=3`, `warm_start=false`,
  `end_on_qp_step=false`, `delta_option=1`, `gamma=1.0`, `qp_projection_alpha=0.01`,
  `qp_projection_scaling=1` (`sampling_c3plus_options.yaml:2-25`). Sibling should
  cross these against port `admm_solver.py` C3Solver constructor defaults + kwargs
  passed at `ci_mpc_c3plus.py`.
- **Reference uses TWO LCS objects per sample**: `lcs_candidates[i]` for the C3 solve
  and `lcs_candidates_for_cost[i]` for the CalcCost reduction — different `resolve_contacts_to`
  and `mu_per_contact` (`cc:1620-1698`). Port uses ONE LCS. This affects the ADMM
  inner objective's F, H, c matrices, not the outer pipeline.
- **Reference `cost_type` for push_t**: check `progress_params.yaml` — likely `kSimImpedanceObjectCostOnly`
  (`cc:592-609`). This drives the sample cost ranking; irrelevant to ADMM but
  relevant to what the outer pipeline uses as "the truth" for sample selection.
- **G-diag augmentation**: reference `SetPositionTrackingOptions` / `SetPoseTrackingOptions`
  (`sampling_c3_options.h:427-505`) explicitly populate `g_lambda`, `g_lambda_n`, `g_lambda_t`,
  `g_eta_*` from yaml lists. Port's G-diag `_use_g_matrix=True` path likely populates a
  different structure. This IS the item-#7 arc-2 blocker; deep dive belongs to sibling.

---

**End of report.** 903 lines pre-linewrap.
