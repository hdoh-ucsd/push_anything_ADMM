# Port TODO — reference-conformance work not yet landed

Tracks reference mechanisms identified during T-push bug hunts that remain
unported. Ordered by expected impact on T-push success rate.

**Status audit 2026-07-22 (arc af44d06):** items #1, #2, #3, #4, #5,
#6 all landed on `main` (verified against HEAD). Item #7 (full
reference q_vector) is STRUCTURALLY BLOCKED on a G-matrix ADMM port
+ LCS state layout parity — see §7 for the empirical migration
receipts. See each section for the file:line evidence + landing
commit.

## 1. Parallel sample evaluation (LANDED — commit `af44d06`)

**Reference**: `sampling_based_c3_controller.cc:971`
`#pragma omp parallel for num_threads(num_threads_to_use_)`
with `num_outer_threads` resolved at :415-422.

**Port** (LANDED): `InnerSolver._lazy_init_worker_kits(n)` in
`control/sampling_c3/inner_solve.py` builds n per-worker kits, each
holding its own `plant_ctx`, `plant_ad` context, `LCSFormulator` (own
`_last_contact_info` cache), and `C3Solver` (own `_u_prev_solve`,
`_last_lambda_*`). `evaluate_samples` runs k=0 serially (keeps its
diagnostic stream) then dispatches k>=1 through a `ThreadPoolExecutor`
fed from a queue of kits. Stdout is globally suppressed around the
pool because `contextlib.redirect_stdout` swaps process-global
`sys.stdout` — per-worker suppression would race.

Config: `SamplingC3Params.num_threads_to_use: int = 1` (default serial;
`num_outer_threads` alias honoured on load). Env override
`PORT_NUM_THREADS_TO_USE` for A/B smoke tests without reloading yaml.

Push_t canonical validation (30 s):
- Serial baseline: avg_per_step_ms=105.0.
- Parallel-3: avg_per_step_ms=54.6 (1.92× speedup), no exceptions,
  sim completes. QP-failure rate ticks 0.02%→1.00% (OSC main-thread,
  attributable to ADMM ordering drift under parallel dispatch);
  trajectory nondeterminism within noise per the push_t plan doc.

Default remains serial; push_t yaml opt-in deferred pending a
coordinated regression sweep before promoting `num_threads_to_use`
above 1 as the default.

## 2. `use_predicted_x0_c3` (ALREADY PORTED)

**Reference**: `sampling_based_c3_controller.cc:1418-1450`
**Port**: `control/ci_mpc_c3plus.py:118-142` +
`control/ci_mpc_c3plus.py:326-345` (clamp block).

Marked here for cross-reference — no work needed. The clamp implementation
uses `x_seq[1]` as `_x_pred_curr_plan`, which is functionally equivalent to
reference's interpolation between `knots.col(last_passed_index)` and
`knots.col(last_passed_index + 1)` when `filtered_solve_time_ / dt ≈ 1`.

## 3. `BufferedSample` carrying full `SampleResult` (LANDED 2026-07-19)

**Reference**: `sample_costs_buffer_` matrix stores position + cost, but the
per-sample C3 plan (`c3_buffer_plan_`) is stored alongside.
**Port** (LANDED): `BufferedSample.result: Optional[object]` at
`control/sampling_c3/sample_buffer.py:56`; write site
`control/sampling_c3/sampling_based_c3_controller.py:1023-1031` populates
`result=r` from the per-sample SampleResult; `AugmentSamplesWithBuffer`
re-enabled at controller `:1417-1442`, gated by
`consider_best_buffer_sample_when_leaving_c3` (default True; set True
in `config/sampling_c3_kik_t.yaml`). Match to reference cc:2106-2158.

## 4. `kRandomOnPerimeter` sampler (LANDED — commit `fc582aa`)

**Reference** push_t uses `sampling_strategy: 4` (`kRandomOnPerimeter`).
**Port** (LANDED): implemented at `control/sampling_c3/sampling.py:657`;
`SamplingStrategy.kRandomOnPerimeter = 4` at `params.py:77`; push_t yaml
opts in via `sampling_strategy: kRandomOnPerimeter` at
`config/sampling_c3_kik_t.yaml:278`. Reference-parity `grid_x/y_limits`
window added at commit `fc582aa`.

**Historical probe 2026-07-19 (yaw_face_bias_strength)**: dropping
`yaw_face_bias_strength: 500 → 0` under the old `kFaceNormal` sampler
gave uniform 10/10 face coverage but killed c3 entries entirely. Landed
solution was to swap to `kRandomOnPerimeter` + tune cost pipeline
(commits `3d335f3`, `fc582aa`). Baseline evidence:
`push_t_show_20260719_022819.txt` (old bias) vs
`push_t_show_yawbias0_20260719_023523.txt` (probe).

## 5. `CheckForWorkspaceLimitViolations` (HARD-PORTED — commit `6563e37`)

**Reference**: DRAKE_DEMAND aborts on workspace-axis or radius-shell exit
(`sampling_based_c3_controller.cc:1476-1494`).
**Port** (LANDED): `SamplingParams.strict_workspace: bool = False` and
`SamplingParams.robot_radius_limits: [float, float] = [0.0, 100.0]` at
`control/sampling_c3/params.py`; controller raises RuntimeError on first
violation when `strict_workspace=True`. push_t canonical yaml opts in
(`strict_workspace: true`, `robot_radius_limits: [0.25, 0.75]`). Default
preserves existing soft-log behaviour for all other yaml callers.

## 6. `avoid_choosing_unsuccessful_samples` (LANDED)

**Reference**: `sampling_based_c3_controller.cc:2161-2205`. Tracks failed
samples (samples where c3 mode was entered but progress not made) in a
buffer keyed by object pose. When generating new samples, rejects any within
`unsuccessful_radius` of an active entry.

**Port** (LANDED): `UnsuccessfulSampleBuffer` at
`control/sampling_c3/sample_buffer.py:172-253`; `sample_avoids_bad_spots`
filter at controller `:501-521` (fires when
`avoid_choosing_unsuccessful_samples` is True — default True); append on
free→c3 transitions at controller `:1963` (mirrors reference cc:2177).
Matches reference generate_samples.cc:187-205 sample-avoids-bad-spots
contract.

## 7. Full reference `q_vector` Q construction (STRUCTURALLY BLOCKED)

**Status audit 2026-07-22 (arc af44d06):** structurally blocked, not a
simple `use_reference_q_vector: true` flip. Requires two paired
structural ports before the reference Q can converge on the port's
ADMM. Every surface-level attempt has diverged.

**Cost pipeline is already ported.** `task_costs.py:785-802`
build_ee_space assembles `Q_base = w_Q · diag(q_vector)` from
`q_vector_ee_pos / obj_quat / obj_pos / ee_vel / obj_ang_vel /
obj_lin_vel`, remapped from reference layout `[ee_pos, quat, obj_pos,
ee_vel, obj_ang_vel, obj_lin_vel]` into the port's `[quat, obj_pos,
ee_pos, obj_ang_vel, obj_lin_vel, ee_vel]`. Gated by
`use_reference_q_vector` (default false).

**Empirical migration attempts (all diverged, all 2026-07-22):**

| Attempt | Config | Outcome |
|---|---|---|
| p39/p40/p41 | Single-parameter migrations (obj_pos alone, obj_quat alone, etc.) | Each regressed or blew up in isolation |
| p44 | Full coordinated + admm_iter=3 (reference iter count) | trans=174.8 mm, T over-pushed |
| p46 | Full coordinated + admm_iter=25 (port iter count) | Dual explodes 1000×/solve |
| p68 | Full ref G matrix with `g_x=g_u=0` (reference values), q_vec off | Trans regressed |
| p69 | Full ref G matrix, q_vec on | T physically tipped over |

See `config/tasks.yaml:209-214` and `control/admm_solver.py:159-166`
for the receipts.

**Root cause (as noted in `config/tasks.yaml:213`):** "port scalar-ρ
ADMM vs ref per-slot G". Reference's ADMM applies `w_G · diag(g_vector)`
augmentation (`g_x=0, g_u=0, g_lambda=2, g_eta=1`) via per-slot ρ
scaling; port's ADMM applies uniform scalar ρ to all slots. Under the
reference's dense q_vector (which relies on the per-slot ρ to keep
velocity slots well-conditioned), the port's uniform ρ over-augments
state/input slots by ~100× and under-augments λ slots by ~50×, which
ill-conditions the whole system.

**Required for landing** (NOT attempted in this arc):

1. **Per-slot G-matrix ADMM.** `admm_solver.py:1057-1076` already has
   a `_use_g_matrix` code path gated by `REFCONF_USE_G_MATRIX=1`. Enabling
   it (attempts p68/p69) destabilises even without the q_vector change,
   suggesting the current implementation needs additional work
   (possibly per-iter ρ ramp, dt-dependent G tuning, or a different QP
   formulation).
2. **LCS state layout parity.** Reference LCS state order is
   `[ee_pos(3), quat(4), obj_pos(3), ee_vel(3), obj_ang_vel(3),
   obj_lin_vel(3)]`; port is `[quat(4), obj_pos(3), ee_pos(3),
   obj_ang_vel(3), obj_lin_vel(3), ee_vel(3)]`. Structural — touches
   `lcs_formulator.linearize_discrete_ee_space`, wrapper x0 construction,
   OSC state slots, viz plumbing. Multi-file, multi-day.
3. **Re-tune after both above land** — reference q_vector values were
   sized against reference velocity dynamics; may need adjustment for
   port's discretisation.

**Recommendation:** treat item #7 as a separate multi-arc project.
Not tractable in a single-session port push. When you're ready to
tackle it, start with a G-matrix debugging arc (make `_use_g_matrix=1`
non-destabilising in isolation) before touching q_vector at all.

**Deep investigation (2026-07-23, arc af44d06):** see
`docs/superpowers/investigations/2026-07-23-item7-deep-investigation.md`.
Key correction to prior framing: the LCS state layout is NOT the
actual blocker (port's Q remap is bijective). The real blocker is a
**three-way coupled configuration mismatch** — every single-flag
migration attempt (p39-p46, p68, p69) is a compensation for a
compensation, none stepping toward the reference regime.

Coordinated triple flip never attempted:
- `rho = 1.0` (was 100.0)
- `_use_g_matrix = True` (was False)
- `use_reference_q_vector = True` (was False)

Suggested next arc: ρ sweep in *current* config
(`_use_g_matrix=False`, `use_reference_q_vector=False`) — establishes
baseline ADMM behavior vs ρ before any coordinated flip.

---

## Not blocking, but worth noting

- **Reference uses `include_walls: false` for push_t**. Port doesn't have wall
  contact at all. Match.
- **Reference `nominal_ee_accel: 2` inherited from anything**. Port hardcodes
  2.0 in `ci_mpc_c3plus.py:__init__` (commit `d601f96`). Could be made
  yaml-loaded but not urgent.
- **Reference `lcs_dt_resolution: 4`** — LCS integrated at planning_dt/4.
  Port uses planning_dt directly. Would require refactoring
  `linearize_discrete_ee_space` to accept a sub-dt argument.
