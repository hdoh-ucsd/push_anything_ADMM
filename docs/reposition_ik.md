# RepositionIKTracker: defaults, decisions, and the diagnostic arc that produced them

This document covers the Drake-IK reposition planner in `control/sampling_c3/reposition_ik.py`, the `RepositionIKParams` struct in `control/sampling_c3/params.py`, and the decisions and findings from its initial implementation. The findings include four bugs caught during integration, the load-bearing parameter defaults arrived at via bisection, and the diagnostic arc that surfaced them.

Out of scope: the §IV-D mode-switch logic in `control/sampling_c3/wrapper.py`, the inner C3 ADMM solve in `control/admm_solver.py`, wrapper-level integration tests, and the upstream dairlib equivalents that this work does not modify. Mode-switch behavior appears here only where it interacts with the tracker (poison-clear sites, mode-driven tracker resets).

Navigation by reader intent:

- Debugging a regression in this area: §Bug catalog.
- Looking up what the tracker exposes: §Tracker API surface.
- Operating the system or interpreting diagnostic output: §Operational notes.
- Considering removing or modifying something that looks like a cleanup candidate: §Refactor-protection notes.
- Writing a new test for this subsystem or a similar one: §Test design patterns.
- Understanding why the current design exists or how the bugs were found: §Diagnostic discipline.


## Bug catalog

### `ik_min_distance_lower_bound` over-enforcement on contact warm-starts

Symptom: every free-mode tracker call reported infeasibility. The controller chattered between c3 and free modes and never settled. Mode-switch metrics showed 142 flips over a 200-loop run that should have produced single-digit flips.

Root cause: `AddMinimumDistanceLowerBoundConstraint(d_min, infl)` is evaluated against the warm-start configuration. With `d_min = 0.005` and the pusher in resting contact with the manipuland (signed distance approximately 0), the warm-start was itself a constraint violation. The IK had no feasible region without first lifting away from the contact, which conflicted with the position constraint pinning the EE to the guide path's first knot. For contact-rich manipulation tasks, the configuration the controller wants to plan from is exactly the configuration the constraint rejects.

Fix: the original single field `min_distance_lower_bound` was split into two: `ik_min_distance_lower_bound` (default 0.0, disables IK-side enforcement) and `fk_min_distance` (separate knob for the FK sweep on tail knots; see entry 4). Both default to 0.0, matching dairlib upstream which relies on the lift-traverse-descend trajectory shape rather than per-knot signed-distance constraints. See `control/sampling_c3/params.py:307-308` for the field definitions and the class docstring above (`params.py:250+`) for the rationale and migration error on the old field name. The two-test pair `test_obstacle_in_path_without_dmin` and `test_obstacle_in_path_with_dmin` in `tests/test_reposition_ik.py` provides regression protection for both branches.

Diagnostic arc: V-7 was the experiment that disabled the IK-side constraint and observed the flip count drop from 142 to 3. V-1 through V-6 had eliminated other hypotheses without reaching this one. See §Diagnostic discipline for the full sequence.

### Quaternion drift on floating-base bodies

Symptom: Drake's `InverseKinematics` rejected the `AddBoundingBoxConstraint` pin on the manipuland's floating-base quaternion in under 1 ms, returning `is_success = False` with no diagnostic output beyond the constraint-violation flag. Every IK call failed before the solver could attempt the position constraint.

Root cause: simulator integration drifts floating-base quaternions away from unit norm by approximately 1e-7 per hundred timesteps. Drake's IK validates pin constraints against the unit-norm invariant (tolerance approximately 1e-6) and refuses inputs that violate it. The drifted quaternion was numerically valid for forward simulation but invalid as an IK constraint target.

Fix: `_normalize_floating_quaternions` at `control/sampling_c3/reposition_ik.py:115` renormalizes every floating-base quaternion in the warm-start configuration before the pin is applied. Called from `_solve_single_knot_ik:349`, immediately before the `AddBoundingBoxConstraint`. The helper iterates body-by-body so RPY-floating bodies (no quaternion) are skipped cleanly, and includes a post-norm assertion that catches the case where renormalization itself fails. The refactor-protection docstring on the helper documents why the call cannot be removed; see §Refactor-protection notes.

Diagnostic arc: V-5 was the inspection of `q_full[7:11]` that surfaced the drift; V-6 landed the fix. V-1 through V-4 had ruled out other hypotheses without inspecting the quaternion slice. See §Diagnostic discipline.

### `GetInfeasibleConstraintNames` state-corrupting side effect

Symptom: after approximately 90 calls in a 200-loop simulation, the simulator's plant context q vector diverged to NaN. The divergence reproduced under a fixed seed and propagated forward into all subsequent computations. Removing the call left the simulation byte-correct.

Root cause: `result.GetInfeasibleConstraintNames(prog)` mutates plant-context state when called repeatedly in a hot loop. The exact mutation path is not characterized; the empirical bisection that surfaced the side effect is sufficient evidence that the call is unsafe in the repositioning IK's per-loop budget.

Fix: the hot path emits sentinel strings (`"WALL_CLOCK_TIMEOUT"` when the wall-clock cap is breached, `"DRAKE_INTROSPECT_DEFERRED"` otherwise) instead of calling `GetInfeasibleConstraintNames`. Out-of-band introspection is exposed via `RepositionIKTracker.diagnose_failure_at` at `control/sampling_c3/reposition_ik.py:721`, which builds a fresh diagram context for a one-shot post-run query and is safe to call after the simulation completes. The comment block at `reposition_ik.py:442+` documents the workaround and warns against re-introducing the call. The workaround is the supported configuration; upstream behavior may change in future Drake releases, in which case the workaround can be revisited.

Diagnostic arc: V-2.5. The NaN propagation was first observed during V-2 and initially attributed to quaternion drift. V-2.5 isolated the cause to `GetInfeasibleConstraintNames` by removing the call and observing the NaN disappear. See §Diagnostic discipline.

### `fk_min_distance` default mismatch with upstream

Symptom: 8 within-free-mode hysteresis swaps in a 200-loop run, versus 21 in the V-7 baseline. The drop indicated the controller was being kicked out of free mode by spurious wall-clock timeouts caused by the FK sweep's per-loop signed-distance computations.

Root cause: V-8 introduced an FK-side `d_min` check on tail knots K..N-1 with a 5 mm default, framed as a "safety net" complementing the IK-side fix from V-7. Each free-mode loop performed approximately 19 `ComputeSignedDistancePairwiseClosestPoints` calls, occasionally pushing the IK budget past the 8 ms wall-clock cap. The added timeouts inflated the controller's poison-stash rate, reducing within-free hysteresis swaps and changing the observable mode-switch behavior.

Fix (V-9): revert the default to 0.0, matching dairlib upstream's design choice that the trajectory's geometric shape (lift-traverse-descend with safe-height clearance) is the safety primitive; per-knot FK-side enforcement is opt-in for trajectories that need it. See `control/sampling_c3/params.py:308` for the default and the field docstring above it for the rationale including the cost-of-opting-in note. The byte-identical `reason_hist` over 200 loops in V-9 versus V-7 is the canonical evidence that the revert restores correct behavior.

Diagnostic arc: V-8 introduced the regression while landing the V-7 fix; V-9 reverted with the byte-identical `reason_hist` confirmation from a side-by-side comparison against V-7. See §Diagnostic discipline.

### Closed-bound rejection in `is_in_workspace` workspace filter

Symptom: across 200-loop diagnostic runs (`scripts/probe_5f_smoke.py diag-kik`), 176 of 200 free-mode loops produced a sample list of exactly `(current, prev_repos)` — no `strat_*` candidates. The wrapper's reposition logic had no fresh alternatives to consider beyond the previously-committed target. The rich-spy capture instrumented in step 8.0.3 made the pattern visible; aggregate metrics had not surfaced it because mode-switch decisions and IK feasibility were unaffected.

Root cause: `is_in_workspace` at `control/sampling_c3/sampling.py:162-172` used closed bounds on each axis (`workspace_xy_min[i] <= p[i] <= workspace_xy_max[i]`). The `kRandomOnCircle` strategy with `n_samples=1` and a non-`None` `g_hat` produces only the mandatory "behind-box" proxy at `obj_xy − r·g_hat`. With `obj_xy ≈ (0,0)` and `target = (0.3, 0)`, float arithmetic gives `obj_xy[1] ≈ +1.5e-6` (sim drift) and `g_hat[1] ≈ −5.0e-6` (goal-direction normalization picks up a small negative ε on y). The proxy y-coordinate computes to `+1.5e-6 − 0.18·(−5.0e-6) ≈ +2.4e-6 m`. With `workspace_xy_max[1] = 0.0` as a closed bound, the proxy fails `p[1] <= 0.0` by 2.4 µm. The 8.1.3 spy verification recorded `proxy_y > 0` and `proxy_passes_filter == False` in 176/176 free-mode 2-tuple loops, with the `proxy_y` value byte-stable at `+2.390e-06 m` across all 176.

Fix: `_WORKSPACE_BOUND_TOL = 1e-3 m` symmetric tolerance applied to all six bounds in `is_in_workspace`. Three orders of magnitude headroom over the observed 2.4 µm ε while staying well within physically-meaningful precision for a robot workspace bound. See `control/sampling_c3/sampling.py:158-180` for the constant and the inline receipt-naming comment. After fix: 0 of 200 free-mode 2-tuple loops; `strat_*` samples now survive filtering and appear in the wrapper's sample list as designed.

Diagnostic arc: 8.0.3 (rich-spy spec, C1/C2/C3 evidence patterns) → 8.1.1 (read of `_build_samples` revealed mode-conditional sample structure) → 8.1.3 (proxy-y verification, 176/176 confirmed) → 8.2 (fix shipped, 176→0 verified across 3 runs each side). The "uncomputable statistic IS the finding" pattern: the C1 metric (travel-penalty ratio between prev_repos and strat_* samples) was uncomputable because the two never co-occurred in the same loop, and the absence of qualifying data was the diagnostic evidence rather than a measurement gap.


## Tracker API surface

### Knot-0 feasibility report

Two properties expose knot 0's IK outcome from the most recent `compute_torque` call. `last_knot0_feasible` (`control/sampling_c3/reposition_ik.py:672`) returns the feasibility boolean. `last_knot0_failure_msg` (`reposition_ik.py:688`) returns the failure-cause string when infeasible, or `None` when feasible. The failure-cause string is one of `"WALL_CLOCK_TIMEOUT"`, `"DRAKE_INTROSPECT_DEFERRED"`, or a pipe-joined Drake constraint-name list; see §Operational notes for the diagnostic-codes table.

Both properties are valid to read after any `compute_torque` call and overwritten by the next call. The single-attribute interface is what `wrapper.py` reads to decide whether to stash a poison target; the wrapper never reads the plural list memos described in the asymmetry entry below.

### Failure-input capture and post-run reconstruction

`last_knot0_failure_inputs` (`reposition_ik.py:677`) returns the `(q_warm_full, p_target)` tuple fed into the most recent failed knot-0 IK solve, or `None` on success or before any `compute_torque` has run. The tuple is the input to `RepositionIKTracker.diagnose_failure_at` at `reposition_ik.py:721`, which builds a fresh diagram context for a one-shot post-run query and returns a dict of solve-state fields (warm-start position error, signed-distance pair, IPOPT solve outcome).

`diagnose_failure_at` is post-run only: never invoke from inside the per-loop control path. The method runs through the Drake introspection APIs that the sentinel-emission workaround in §Bug catalog entry 2 routes around for the same reason. The freshly-allocated diagram context isolates the introspection from the live simulator's plant context, which is what makes the call safe after the run completes.

### Singular and plural naming asymmetry

Properties prefixed `last_knot0_*` return knot 0's value. Attributes named `last_q_knots`, `last_ee_knots`, `last_feasible`, `last_knots_solve_ms`, `last_failure_msgs`, and `last_failure_inputs` hold the per-knot list or array across all `K` IK-solved knots and the `N - K` FK-tail knots.

`wrapper.py` reads only the singular properties, since its single-attribute interface needs knot 0 alone. Diagnostic harnesses and tests that need full-trajectory data read the plural attributes directly. Tests in `tests/test_reposition_ik.py` exercise both conventions: `last_knot0_feasible` for canary assertions and `last_q_knots[:, :K]` for `test_joint_limit_continuity`'s inter-knot-delta checks. See §Test design patterns for how the two conventions interact in test bodies.


## Operational notes

### Test pickle freshness

When reusing a pickled metrics dict from a prior smoke run, audit the pickle's mtime against the source files in the relevant control stack: `wrapper.py`, `reposition_ik.py`, `params.py`, and any other source whose behavior the metrics depend on. If any source mtime is newer than the pickle, the pickle was generated against a different version of the code and may not reflect current behavior. Force regeneration by re-running the harness rather than silently reusing.

The b.pkl staleness incident is the canonical example. Step 5's (b) sub-step initially expected to reuse `results/probe_5f_b.pkl` from a prior run. The existing pickle predated V-6's quaternion fix. Without the mtime audit, step 5's (c) sub-step would have read the stale pickle and based its poison-lifecycle reconstruction on metrics that no longer reflected the working code. The audit caught the staleness, forced regeneration via the diag-kik run, and the fresh pickle became the V-9 reference state.

### Diagnostic codes

The tracker emits one of three diagnostic codes per failed knot-0 IK solve. Two are produced from the per-loop hot path; the third is only ever produced by `diagnose_failure_at` after the simulation completes.

| Code                                  | When emitted        | What it means                                                                                                                                                                                                                  |
|---------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WALL_CLOCK_TIMEOUT`                  | Per-loop hot path   | Elapsed wall-clock exceeded `per_knot_solve_timeout_s` before IPOPT returned `is_success`.                                                                                                                                       |
| `DRAKE_INTROSPECT_DEFERRED`           | Per-loop hot path   | IPOPT exited unsuccessfully but the wall-clock cap did not fire. The per-loop diagnostic deliberately does not call `GetInfeasibleConstraintNames` (§Bug catalog entry 2); offline introspection via `diagnose_failure_at` is required. |
| Pipe-joined Drake constraint name list| Post-run only       | Output of `diagnose_failure_at` on a fresh diagram context; lists the constraints violated at the warm-start. The only path that reaches Drake's introspection API safely.                                                       |

`WALL_CLOCK_TIMEOUT` and IPOPT's max-iter exit can co-occur on the same `Solve` call. IPOPT terminates after `max_iter` and prints a Drake-side warning to stderr; the wall-clock cap also fires because elapsed exceeded the cap. Both signals are real, but `failure_msg` reports `WALL_CLOCK_TIMEOUT` because the cap check supersedes the IPOPT-internal exit code. Step 5's V-9 run produced one such co-occurrence: Drake's stderr showed `IPOPT terminated after exceeding the maximum iteration limit` while `failure_msg` recorded `WALL_CLOCK_TIMEOUT`.

### Site B and IK-failure interaction

The wrapper's poison-clear logic at `control/sampling_c3/wrapper.py:445` (Site B) clears `_infeasible_repos_target` unconditionally on every c3-mode loop, on the reasoning that once the controller is back in c3, the prior repos target's feasibility is irrelevant. A second clear path at `wrapper.py:472` (Site D, helper at `wrapper.py:295`) fires when the controller commits to a new pursued target outside `infeasibility_match_radius_m`. The two sites cooperate: an IK failure during free mode that contributes to a `kToC3Cost` mode flip is cleared by Site B on the next loop without Site D needing to fire.

Step 5's (c) reconstruction is the worked example:

| Loop | Mode | isnone | feas_lr | Note                                              |
|------|------|--------|---------|---------------------------------------------------|
| 36   | free | True   | True    | —                                                 |
| 37   | free | True   | True    | —                                                 |
| 38   | free | False  | False   | IK fails (`WALL_CLOCK_TIMEOUT`), poison stashed   |
| 39   | c3   | True   | False   | Site B fires on mode-flip to c3; poison cleared   |
| 40   | free | True   | True    | —                                                 |
| 41   | free | True   | True    | —                                                 |

The IK failure at loop 38 contributes to the cost trajectory that triggers the `kToC3Cost` flip at loop 39, and Site B fires for that flip's standard reason. The poison-clear logic and the mode-switch logic agree on the same underlying event without needing to coordinate. The single-loop dwell time between stash and clear is a tight bound: Site B's idempotent firing ensures any free-mode poison is cleaned up the moment the controller flips back to c3 for any reason.

### Why test (e) was dropped

The original step-5 test (e) was to verify the poison lifecycle by dropping the manipuland into the table at a fixed loop, asserting poison stash within `≤ 2` loops, restoring the manipuland at a later loop, and asserting poison clear within `≤ 5` loops. Under C-2's defaults the test does not exercise its target code path.

Two reasons compound. First, `ik_min_distance_lower_bound = 0.0` means the IK does not constrain against the manipuland's position; moving the manipuland anywhere, including inside the table, does not produce an IK failure. Second, when the manipuland is inert, the controller stays in c3 mode for the duration of the drop window. The kIK tracker is never invoked, regardless of what it would do if reached. The original failure-induction approach is structurally disconnected from the system under test.

Step 5's (c) sub-step provides equivalent lifecycle coverage organically. The natural `WALL_CLOCK_TIMEOUT` event at loop 38 produced a stash-and-Site-B-clear sequence with the same shape test (e) was designed to verify.

### Why test_collision_filter_excludes_resting_contact was dropped

The original test name conflated two distinct filtering mechanisms. At the tracker level, the test would have verified that EE-manipuland pairs are excluded from the IK's min-distance constraint when the warm-start is in resting contact. At the LCSFormulator level, the same name could have referred to filtering applied to the contact-pair output of `ComputeSignedDistancePairwiseClosestPoints`.

Neither interpretation maps to a test that belongs in `tests/test_reposition_ik.py` under C-2. The tracker-level interpretation reduces to verifying that no constraint is enforced, since `ik_min_distance_lower_bound = 0.0` is the production default; testing the absence of code is not behavior verification, and the `without_dmin` and `with_dmin` test pair already covers both branches of that behavior. The LCSFormulator-level interpretation is out of scope for the tracker's test file; a `tests/test_lcs_filter.py` would be the appropriate home, deferred to step 8+ scope.

Equivalent coverage for the original test's intent lives in the `without_dmin` and `with_dmin` pair in `tests/test_reposition_ik.py`, whose regression-protection role is described in §Bug catalog entry 3.

### kIK and PWL trackers settle at different z equilibria under default gains

Under the default `RepositionParams` gains, the two reposition trackers reach different steady-state EE z-positions despite both consuming the same wrapper-chosen `p_target = (x, y, 0.050)`. In a 200-loop pushing-task diag run starting from `INITIAL_ARM_Q` (FK at z=200 mm), the kIK tracker settles at z ≈ 25 mm — 25 mm below the wrapper's z=50 mm reference. The PWL tracker settles at z ≈ 75 mm — 25 mm above. Both fall short of the target in opposite directions.

The mechanism is a Type-0 control-law steady-state error compounded by the per-loop IK target's truncation to one stride. `compute_torque` at `reposition_ik.py:1172-1192` is a P-I-D law with gravity comp, no feedforward acceleration term, and no velocity reference in the D-term (`u_d = -Kd_q · v_arm_now` rather than `-Kd_q · (v_arm_now − v_target)`). With `num_full_ik_knots = 1`, the joint-PD reference is recomputed each loop from `current_q + one stride along the guide path` — there is no persistent "lift to z=50 mm" command, only a chain of "lift 10 mm from wherever you are" commands. When the controller under-tracks per loop, the next loop recomputes a fresh one-stride reference from the new (lower) current_q, and the steady-state position is wherever `Kp_q · q_err + Ki_q · integral` balances the gravity-load mismatch.

The kIK/PWL z-asymmetry comes from each tracker's IK method producing a different `q_target` configuration for the same Cartesian point. kIK uses Drake's full constrained `InverseKinematics`; PWL uses DLS pseudoinverse on a Cartesian guide. Both are valid inverse kinematics solutions; the joint-PD landing equilibrium under the same gains differs because the two configurations have different gravity-load shapes and different inertia couplings. The 8.4.3 TS4 verification confirmed that kIK's IK lands at `p_guide[:, 0]` to within `√3 · 1 mm = 1.732 mm` (the 3D Euclidean norm of the IK solver's per-axis tolerance) on every prev_repos-winning loop across three runs — the IK is correct; the joint-PD does not reach the IK's q in one 10 ms control loop.

### Joint-PD covers ~0% of per-stride distance per 10 ms control loop

A direct measurement from the 8.4.3 spy capture: the median gap between `FK(last_q_knots[:, 0])` (where the IK says the EE should be) and the end-of-loop EE position is 11.1 mm, while the per-stride distance `ds = repos_params.speed · dt_planning = 0.20 · 0.05 = 10.0 mm`. The controller covers essentially 0% of each 10 mm stride in a single 10 ms control loop. Across the 178 prev_repos-winning loops in a representative run, `‖TS4 − TS3‖` (IK solution minus achieved EE) ranges 10.146 mm to 27.580 mm with median 11.142 mm.

This means the IK keeps issuing "lift 10 mm from where you are" each loop, the arm does not lift appreciably, and the IK solves for the same lift command on the next loop from the same starting EE. The arm sits in a near-static equilibrium where Kp_q · q_err + Ki_q · integral balances the gravity-load mismatch; raising `Kp_q` past the saturation regime does not move the equilibrium (see §Refactor-protection notes). The integral builds up to ~1.0 rad·s over ~125 loops and at that point provides Ki_q · 1.0 = 8 Nm of corrective torque, which matches the 7.39 Nm gravity-load shift between current_q and q_target measured in the 8.4.2.2 correctness check. The integral is doing its theoretical job at equilibrium; the residual gap remains because the proportional response saturates per-joint torque before the arm can traverse the 10 mm lift in 10 ms.

### Torque-clip ceiling at √6 · 30 Nm under default Kp_q

In 200-loop diag runs at `Kp_q = 60` (default), `‖u‖_max ≈ 68 Nm = √(5.14) · 30 Nm`. At `Kp_q = 120` (Fix 5, reverted), `‖u‖_max = 73.54 Nm = √6 · 30 Nm` exactly, the unmistakable signature of 6 of 7 joints simultaneously saturated at the ±30 Nm per-joint clip. The np.clip at `reposition_ik.py:1190` caps each joint at `±torque_limit` before output, so doubling `Kp_q` does not double the actual torque commanded — it just saturates more joints. At default Kp_q the joint-PD law was already operating in the saturation regime on most joints; raising the gain produces more clipping, not more proportional response. The receipt for not raising Kp_q beyond 60 lives in §Refactor-protection notes.

### C3 ADMM trajectory does not intersect the box for the pushing task

The 8.4.6 contact-geometry check is the load-bearing finding for why the wrapper's repositioning targets do not produce box motion. Across the 24 c3-mode loops in a 200-loop pushing-task run, the pusher (sphere, radius 25 mm) is never in contact with the box (100 mm cube at obj_xy ≈ (0, 0), z = 50 mm). The minimum signed distance between pusher surface and box surface is +12.95 mm at t=13 — the pusher comes within 13 mm of the box's south-top edge but does not touch. Median signed distance over the 24 c3-mode loops is +56 mm; max is +256 mm.

The trajectory shape is the diagnosis. In the early c3-mode phase (t=0..20), while EE xy is still near the box's xy-vicinity, EE z descends from 200 mm to 96 mm — staying above the box top (z=100 mm) for most of the descent. Closest approach at t=13: EE = (-26, -70, 132), box top-south corner = (-26, -50, 100), Cartesian distance 38 mm minus pusher radius = 13 mm. After t=13 the EE swings outbound in xy: by t=20, EE = (-76, -157, 96), well clear of the box. By the time EE z drops into the box's vertical extent (t=46+, z ≈ 27 mm — within the box's z=0 to z=100 mm range), EE xy is at (-145, -315), 280 mm from the box. The two trajectories — z and xy — never simultaneously intersect the box.

This is a sampling-C3 inner-solver finding, not a tracker finding. The C3 ADMM at `control/admm_solver.py` and the LCS formulation at `control/lcs_formulator.py` are producing torque trajectories that route the EE around the box rather than into it, given the goal at +x and the box at origin. The four step 8 fixes (closed-bound, gravity-comp at q_target, I_max increase, reverted Kp) removed real defects in the wrapper and the kIK tracker but cannot move the box because they're improving tracking of references that do not intersect the manipuland. The investigation surface for that bottleneck is documented in `docs/step9_c3_admm_proposal.md`; it is out of scope for the kIK delivery and step 8.

### Verification gap from step 5/7's V-9 receipt

The V-9 byte-identity claim ("byte-identical `reason_hist` over 200 loops in V-9 versus V-7") measured what it measured: mode-switch decisions are deterministic under the same control-law and seed within a single bracket of identical invocations. Step 8 strengthened the IK feasibility receipt — `FK(last_q_knots[:, 0])` lands within 1.7 mm of the requested guide knot (within solver per-axis tolerance) on every prev_repos-winning loop. What V-9 did not measure: steady-state position-tracking accuracy of the joint-PD law, contact-attempt success rate of the resulting EE trajectory, or whether the C3 ADMM's commanded targets intersect the manipuland.

The kIK delivery's mechanical receipts are not weakened by the step 8 findings. They are scoped: V-9 is determinism of mode-switch decisions and IK feasibility, not whole-system task success. The step 8 findings extend the verification matrix in directions V-9 did not cover; they do not invalidate the V-9 region. Future tracker work that touches the joint-PD law or the IK warm-start should preserve V-9's byte-identity within a bracket of identical invocations as a regression baseline, with the new step 8 measurements (TS4↔TS2 ≈ √3 mm, contact rate during c3-mode) as additional canaries.

### Variance floor on bracketed runs is non-zero

Step 8 measured the simulation's run-to-run noise floor across multiple identical invocations. Within a single bracket of identical `python scripts/probe_5f_smoke.py diag-kik` calls — same script bytes, same `.pyc` cache state, same seed=42, no edits between invocations — three runs produce variance=0 across every categorical metric (flips, reason_hist, free-mode 2-tuple count, V-2.5 first IK fail pair) but produce byte-different pickles (different md5 hashes). Across brackets that span script edits or harness reloads, variance is small but non-zero: flip counts in the range {1, 3, 5, 7, 11} have been observed at the same seed under structurally-identical scripts, with kToBetterRepos counts ranging {6, 8, 15, 21}. The 8.1.4 result of variance=0 within a bracket was the special case, not the general one; the 8.2 baseline run produced flips {3, 5, 5} across three back-to-back invocations of the same script.

The operational implication: fix-verification protocols must use multi-run statistical comparison (3+ runs each side of the fix, compare distributions) rather than byte-identity unless both sides are within the same bracket. Step 8's 3+3+1 protocol on each fix landing (3 pre-fix runs, 3 post-fix runs, 1 PWL verification) was sized to clear this noise floor. The signal sizes that mattered (closed-bound 2-tuple count: 176→0, contact count during c3-mode: 0/24 in every variant) were many standard deviations above the noise. Smaller signals would need wider brackets or controlled within-bracket comparison.

### Step 8 closure (8.6 full-math examination)

IK trajectories in free-mode never route within 102 mm of the box surface, min clearance four pusher-radii. The controller-side fixes shipped in step 8 (closed-bound, gravity-comp at q_target, I_max increase) address real defects but cannot move the box because the IK targets they're tracking are geometrically far from contact. The bottleneck is upstream of IK: either in the wrapper's target selection (the proxy at 130 mm behind box surface is the algorithm's staging-target design per `docs/step9_findings.md` Finding C; the wrapper's surrogate-C3 evaluation rejects it via `align_bonus=0` per `docs/step9_findings.md` mechanism α reframing) or in the C3 ADMM (which produces non-contact trajectories for c3-mode per 8.4.6 independently of free-mode IK behavior). See `docs/step9_findings.md` for the deferred investigation scope.

The 8.6 instrumentation captured `last_q_knots`, `last_ee_knots`, gravity-at-target, current and post-dt arm state, and the reconstructed torque breakdown across 200 loops. P1 surfaced that consecutive-loop `q_arm_target` shifts are well-behaved (median 0.72 mm Cartesian motion per loop, geometrically coherent). P2 surfaced asymmetric saturation: q[5] saturates 49.7% of free-mode loops, tau_g on q[1] is already 31 Nm before any control input (above the 30 Nm per-joint clip), and ‖u_post‖₂ median is √2·30 Nm indicating ~2 joints simultaneously saturated. P3 confirmed plant integration is internally consistent. P4 surfaced per-joint tracking failures: q[5] has 0.81 rad of error with 49.7% saturation ("working hard but failing"); q[1]–q[3] have high error with low saturation ("genuine under-gained"). P5+P6 surfaced the load-bearing finding: every free-mode IK trajectory's closest approach to the box is ≥ 102 mm. P2 and P4 are real second-order defects; P5+P6 is the first-order gating issue.

### kIK tracker tail-knot behavior at `num_full_ik_knots = 1`

With `K = 1`, tail extrapolation velocity is zero (K<2 fallback at `_fk_sweep_tail:1014-1017` sets `v_arm = zeros`), producing identical q across all 20 knots in the captured horizon. The horizon is effectively a single IK-solved knot. This is consistent with the `K = 1` design intent (one full IK solve per loop), but worth noting because instrumentation that captures `last_ee_knots` or `last_q_knots` under production defaults will see identical-across-knots data — the "20-knot trajectory" is one knot repeated. Consumers requiring a true 20-knot horizon must set `K ≥ 2`.

A secondary consequence: under `fk_min_distance = 0` (the V-9 default — see §Bug catalog entry 4), `last_ee_knots[:, 1:]` are zero-init values rather than FK output (the FK sweep is gated on `fk_min_distance > 0` per `_fk_sweep_tail:1037-1043`). Diagnostic code that needs FK for tail knots must either set `fk_min_distance > 0` (re-introducing the V-8 regression) or compute FK harness-side from `last_q_knots`. The 8.6.5 analysis took the latter approach.


## Refactor-protection notes

### Both `d_min` defaults are 0.0

`RepositionIKParams.ik_min_distance_lower_bound` and `RepositionIKParams.fk_min_distance` both default to `0.0` (`control/sampling_c3/params.py:307-308`). A reader doing a hardening pass or parameter sweep might raise both to a small positive value (5 mm is a common starting point) on the reasoning that some signed-distance margin is the responsible default and zero looks under-defended.

This is wrong for the system the tracker is integrated into. Raising `ik_min_distance_lower_bound` above zero rejects every warm-start in which the pusher is in resting contact with the manipuland, because the warm-start is itself a constraint violation and the IK has no feasible region without lifting away from the contact, which conflicts with the position constraint pinning the EE to the guide path's first knot. The kIK reposition path becomes unusable for contact-rich manipulation. Raising `fk_min_distance` above zero adds approximately 19 `ComputeSignedDistancePairwiseClosestPoints` calls per free-mode loop, occasionally pushing the IK budget past the wall-clock cap and changing the controller's observable mode-switch behavior. The receipts are in §Bug catalog entries 3 and 4.

The defaults match dairlib upstream's design choice that the lift-traverse-descend trajectory shape is the safety primitive, not per-knot signed-distance enforcement. Per-knot enforcement is opt-in for trajectories that need it, with the cost-of-opting-in noted in the field docstring.

### Quaternion renormalization at the IK warm-start

The single call to `_normalize_floating_quaternions` at `control/sampling_c3/reposition_ik.py:349`, immediately before the `AddBoundingBoxConstraint` pin in `_solve_single_knot_ik`, looks unmotivated to a reader who trusts the simulator's quaternion output. The simulator advances floating-base bodies under integration that nominally preserves unit norm; one might expect to read the integrator's output and pass it directly to Drake's IK without re-normalization.

This is wrong in practice. Simulator integration drifts floating-base quaternions away from unit norm by approximately 1e-7 over hundreds of timesteps. Drake's `InverseKinematics` validates pin constraints against unit-norm tolerance approximately 1e-6 and rejects inputs that violate it. Without the normalization, every IK call after the first 100 simulation steps fails in under 1 ms with a quaternion-norm-validation error, well upstream of the position constraint or solver iteration. The receipt is in §Bug catalog entry 1.

The helper at `reposition_ik.py:115` is the canonical implementation: it iterates body-by-body, skips RPY-floating bodies, and asserts post-norm. Renaming the call site or moving the helper is unconstrained; removing the call re-introduces V-6's failure mode.

### Warm-up `WARNING` line is intentional

The constructor's warm-up `Solve` (around `control/sampling_c3/reposition_ik.py:659+`) emits a `[RepositionIK] WARNING: IPOPT warm-up solve completed in X ms (>Y ms cap)` line when the warm-up exceeds the configured `per_knot_solve_timeout_s`. A reader encountering the WARNING in test output might convert it to a raised exception ("warnings should be errors") or treat it as an unfinished TODO.

Neither is correct. The warm-up `Solve` is a startup throwaway whose purpose is pre-paying IPOPT's 15 to 25 ms cold-start cost so per-loop solves hit the warm path. When `per_knot_solve_timeout_s` is set tight (1 ms in `test_infeasibility_marks_target`), the warm-up necessarily overshoots. The result tuple is discarded; no state mutates that affects subsequent solves. The WARNING is observable noise, part of the same diagnostic-codes ecosystem that §Bug catalog entry 2's sentinel emission established (visible, parseable diagnostics rather than silent failure or Drake-API introspection).

Tests that deliberately tighten the timeout will see one WARNING per tracker construction. Reformatting the print is unconstrained; raising on overshoot would break tests that exercise the timeout failure path on purpose.

### Why `I_max = 2.0` (not 0.5)

`RepositionParams.I_max` defaults to 2.0 (`control/sampling_c3/params.py:233`). A reader doing a hardening pass might lower this to 0.5 on the reasoning that integral wind-up is a known failure mode and a tighter clamp is the responsible default. Dairlib upstream's analog uses a tighter clamp; matching upstream looks like the correct conservative choice.

This is wrong for the system the tracker is integrated into. The 8.4.5 measurement showed the integral converges to a per-joint magnitude of ~1.0 rad·s at equilibrium under default `Kp_q = 60`, `Ki_q = 8.0`, and the contact-rich pushing task. With `Ki_q · 1.0 = 8 Nm`, the integral correction at equilibrium matches the gravity-load shift between `q_arm_now` (z=24 mm equilibrium) and `q_arm_target` (z=35 mm one-stride lift) measured at 7.39 Nm in the 8.4.2.2 correctness check, to within 10%. The integral is doing its theoretical job: it's compensating for the gravity-load mismatch that the proportional term cannot eliminate (Type-0 system, see §Operational notes).

With `I_max = 0.5`, the integral was clamped at 50% of the value it would naturally reach, capping the integral correction at 4 Nm — half of what the task requires. Raising the clamp to 2.0 lets the integral reach its natural equilibrium without clamping; in the 8.4.5 measurement, the integral plateaued at ~1.0 (50% of the new clamp), confirming that 2.0 is loose enough not to bind. Lowering it back to 0.5 reintroduces the clamp at the value the integral wants to be at, undoing Fix 6's contribution.

PWL is byte-identical pre/post Fix 6 because PWL's tracker reaches a different equilibrium where the integral never approaches 0.5; the larger clamp is unused on PWL but unnecessary to revert.

### Why gravity comp uses `q_arm_target` (not `current_q`)

`compute_torque` computes gravity compensation against the IK solution at knot 0 (`reposition_ik.py:1185-1192`), not against the measured configuration. A reader following Drake's standard usage of `CalcGravityGeneralizedForces` might revert this to use the caller's `plant_ctx` (which is set to `current_q` at `reposition_ik.py:1087`), on the reasoning that gravity comp is conventionally evaluated at the actual measured configuration since that's where the gravity force actually applies.

This is the standard convention for *regulator* control where the system holds a fixed setpoint. For *tracking* control where the system traverses a trajectory toward a moving reference, the convention is to anticipate the gravity load at the reference rather than the measurement. The 8.4.2.2 correctness check measured the difference: at `current_q = INITIAL_ARM_Q` (EE z=200 mm) versus `q_arm_target` from IK to (same xy, z=50 mm), gravity-comp shifts by 8.73 Nm in 2-norm, with the largest single-joint shift of +7.39 Nm on q[1] (shoulder). The shift sign is consistent with anticipating a more-extended-down configuration's larger shoulder anti-gravity load.

The fix at `reposition_ik.py:1185-1192` reads `q_arm_target` into a local `q_full_target`, sets the tracker's private `_plant_ctx_ik` to that q (the caller's `plant_ctx` is intentionally untouched), and computes gravity comp against the IK-resolved configuration. Reverting to `plant_ctx` re-introduces a 7-8 Nm gravity-comp mismatch that the joint-PD's Kp term is forced to absorb via steady-state error. The receipt is in §Operational notes (the Type-0 mechanism entry); the kIK README's bug-catalog framing was avoided since this was a structural improvement rather than a bug, but the magnitude and direction of the shift are independently verifiable via the 8.4.2.2 procedure.

### Why `Kp_q = 60` (not higher)

`RepositionParams.Kp_q` defaults to 60.0 (`control/sampling_c3/params.py:230`). A reader who has measured the joint-PD's tracking error and wants to close the loop tighter might raise this to 90 or 120 on the reasoning that tighter proportional gain produces tighter tracking — standard control intuition.

This is wrong in the operating regime the tracker actually visits. The 8.4.4 measurement showed that at default `Kp_q = 60`, `‖u‖_max ≈ 68 Nm = √(5.14) · 30 Nm`, indicating ~5 of 7 joints already saturating at the per-joint `torque_limit = 30 Nm`. At `Kp_q = 120`, `‖u‖_max = 73.54 Nm = √6 · 30 Nm` exactly — 6 of 7 joints saturated. The np.clip at `reposition_ik.py:1190` caps each joint at `±torque_limit` before output, so increasing `Kp_q` past the saturation regime does not increase the actual torque commanded — it just produces more clipping. Tracking did not improve under `Kp_q = 120` (TS4↔TS3 median went from 11.17 mm to 11.61 mm — slightly worse). Mode flips tripled (3 → 11) due to instability surfacing in the saturation regime. PWL regressed: EE z mean 83.6 mm → 106.9 mm, further from the 50 mm target.

The fix surface for the tracking gap is not proportional gain. The integral-clamp expansion (Fix 6, see §Refactor-protection note above) addresses the steady-state mismatch directly. Future investigations of the joint-PD law should consider the velocity-reference D-term or the feedforward acceleration term (neither currently present); raising `Kp_q` past 60 is closed off by the saturation receipt.

### Why the workspace-bound tolerance is 1 mm

`_WORKSPACE_BOUND_TOL = 1e-3` in `control/sampling_c3/sampling.py:166`. A reader doing a numerical-rigor pass might tighten this to `1e-6` (microns) or remove it entirely on the reasoning that workspace bounds should be exact and float-arithmetic ε within micron range is a code-correctness issue rather than something to tolerate.

This is wrong for the operating regime: the 8.1.3 measurement recorded the actual ε on the rejected proxy y-coordinate at +2.4e-6 m, byte-stable across 176 loops. The tolerance of 1 mm is three orders of magnitude above the observed ε, providing headroom for future numerical drift while staying well below physically-meaningful workspace precision (a robot's reachable workspace is meaningful at the millimeter scale, not the micrometer scale). Tightening to 1e-6 leaves no safety margin against future float-arithmetic drift in `obj_xy` or `g_hat` computation; removing the tolerance reintroduces §Bug catalog entry 5's failure mode wholesale.

The tolerance is symmetric (applied to both min and max bounds on all six axis-pairs) because the closed-bound trap is direction-agnostic: any axis-aligned bound at exactly zero produces the same ε-rejection failure. The bug surfaced on `workspace_xy_max[1] = 0.0` for the pushing task; the tolerance prevents the same issue on any future scenario with axis-aligned zero bounds.


## Test design patterns

### Two-branch tests must verify both outcomes

A regression-canary test paired with a mirror failure-canary test must verify both outcomes; passing the success branch alone is structurally insufficient evidence that the underlying behavior is correct.

V-8 is the worked receipt. V-7's fix made the IK succeed in resting contact when `ik_min_distance_lower_bound = 0.0`. V-8 added an FK-side `d_min` default that did not break V-7's success criteria (the IK still succeeded under the same scenarios) but did change observable mode-switch behavior measurable only by side-by-side comparison against V-7's full run. Without a failure branch verifying that the FK-side default actually rejected something when set positive, the regression was visible only retrospectively in aggregate metrics.

The canonical regression protection is the `test_obstacle_in_path_without_dmin` and `test_obstacle_in_path_with_dmin` pair in `tests/test_reposition_ik.py`. The first asserts that the C-2 default succeeds in resting contact. The second asserts that flipping the override to `0.005` actually fails. A passing first test alone could not distinguish "the constraint is wired correctly and the default is right" from "the constraint is silently disabled and the default does not matter." See §Diagnostic discipline for the V-8 → V-9 receipt that produced this pattern.

### Threshold calibration via documented rationale

When a test threshold cannot be reached without expanding the test's scope to cover behaviors beyond its primary purpose, relax the threshold with documented rationale rather than expand the scope.

`test_joint_limit_continuity` is the receipt. The initial draft used `(0, 0.15, 0.10)` as the target, producing a max delta of 0.026 rad (above the "test is moving" floor of 0.01) and a min margin to joint limits of 1.18 rad (above the "test is stressing the joint-limit branch" ceiling of 0.5 rad). Tuning by overriding `pwl_waypoint_height` and pre-positioning the arm forward via DLS-IK reached margin 0.68 rad, a 42% reduction. Going under 0.5 rad would have required either ignoring Panda's reach geometry (knot 2 was already at the 0.85 m reach edge) or adding orientation-cone or kinematic-singularity constraints, both of which would have made the test exercise behaviors orthogonal to inter-knot continuity.

The chosen response was to accept 0.68 rad with the geometric explanation documented inline. The threshold was set tight initially to validate that the test was meaningful; the calibration step measured whether the test was meaningful in practice and recorded the answer. Future maintainers reading the relaxed threshold see the rationale for the relaxation in the same place they see the threshold itself.

### Drake-required test fixture pattern

Tests requiring a Drake plant should use a session-scoped fixture for the world (one expensive build per pytest session), function-scoped fixtures for per-test contexts (cheap reset for isolation), and a factory pattern for parameter overrides applied via `dataclasses.replace` rather than in-place mutation.

Drake plant construction including `ToAutoDiffXd` takes approximately 1.5 s. Building per-test would inflate a six-test file from ~1 s to ~10 s and discourage adding tests. Function-scoping per-test contexts isolates state without re-paying the build cost (`Simulator(diagram).get_mutable_context()` is essentially free compared to constructing the diagram). The factory pattern with `dataclasses.replace` prevents test bodies from accidentally mutating fixture-shared parameter instances; if a sibling test mutated `default_params.repos_ik_params` directly, the next test's setup would inherit the mutation. The replace-on-copies contract forecloses this entire class of leakage.

The canonical implementation is the fixture stack at the top of `tests/test_reposition_ik.py`: `_world` (session-scoped, returns a NamedTuple of plant, scene_graph, and frame handles), `root_ctx` (function-scoped fresh `Simulator` context), `default_params` (function-scoped `SamplingC3Params`), and `tracker_factory` (function-scoped callable applying overrides via `dataclasses.replace`). The pickle-freshness audit in §Operational notes operates against fixtures of this shape: the session-scoped world is rebuilt only between pytest sessions, the same granularity at which source-file changes invalidate cached metrics.

### Test scenery via unconstrained DLS-IK

Tests requiring a specific kinematic state for setup should drive the arm into that state using the unconstrained damped-pseudoinverse IK at `control/sampling_c3/ik.py:solve_ik_to_ee_pos`, not the constrained Drake IK at `RepositionIKTracker._solve_single_knot_ik` that the test is designed to exercise.

The constrained IK is the system under test. Using it to set up its own test scenarios entangles scenario validity with system-under-test correctness: a test passing because the setup-time IK already moved the arm out of the configuration the test is supposed to exercise produces no behavioral signal. The DLS helper is a different code path with different failure modes; its convergence on a target position is independent confirmation that the warm-start state is what the test claims.

The canonical example is `_setup_resting_contact_scenario` in `tests/test_reposition_ik.py`. The helper drives the pusher into resting contact with the manipuland's top face by calling `solve_ik_to_ee_pos` against a target position computed from the manipuland's configured size and pose. The resulting `q_solved` is the warm-start fed into `RepositionIKTracker.compute_torque` for the test's actual measurement. Substituting the constrained IK for setup would conflate "the warm-start was achievable" with "the tracker's IK accepts the warm-start," which are the two properties the test is designed to keep distinct.

### IK-failure pinning via direct-line shortcut

An IK-failure test that wants to exercise a constraint-violating warm-start should set the planner's `p_target` equal to the warm-start EE position itself, so the guide-path generator's direct-line shortcut returns `p_target` verbatim. The first knot's IK target collapses to the warm-start, leaving the position constraint with no room to satisfy other constraints by displacing.

Without this pin, IPOPT can satisfy a positive `d_min` constraint by lifting the EE off the contact while still hitting position tolerance on a free-space target. The first attempt at `test_obstacle_in_path_with_dmin` used `p_target = ee_now + (5 cm, 0, 10 cm)`. With `d_min = 0.005`, IPOPT lifted the pusher approximately 5 mm off the manipuland while hitting the 1 mm position tolerance: `d_min` satisfied, position satisfied, IK reported success. The constraint was wired correctly but the test verified nothing about its enforcement.

Replacing `p_target` with `ee_now` exploits `next_waypoint`'s direct-line shortcut at `control/sampling_c3/reposition.py:63-69`: when the displacement is below `straight_line_thresh = 8 mm`, the first guide knot returns `p_target` verbatim. Satisfying `d_min` would then require lifting the EE by 5 mm, violating the 1 mm position tolerance. No feasible point exists, and IK reports infeasibility.


## Diagnostic discipline (V-1 → V-9)

The repositioning IK tracker's first integration with the §IV-D wrapper produced two visible failure modes: mode-switch chatter (179 flips in a 200-loop reference run, where the expected count was single-digit) and divergence of the simulator's plant-context q vector to NaN after approximately 100 loops. Both reproduced under a fixed seed. The arc from V-1 to V-9 narrowed the hypotheses, surfaced two confounding factors, and produced three production fixes plus one pre-shipping revert.

### V-1 to V-3: hypothesis disambiguation

The initial hypotheses were drawn from the most visible suspects: an interaction between the new tracker's PD integrator and the wrapper's mode-switch logic, an off-by-one in the kIK guide-path knot indexing, and a sign error in the joint-PD-with-grav-comp control law. V-1 inspected the mode-switch decisions and found the controller flipping for legitimate reasons (every flip carried a valid `SwitchReason`); the chatter was an effect, not the cause. V-2 measured the per-call IK feasibility report and found `last_knot0_feasible` returning `False` more often than a working tracker should; the question shifted from "why is the wrapper flipping" to "why is the IK reporting infeasible." V-3 attempted a four-way bisection across plausible IK-failure causes (wrong constraint, wrong warm-start, wrong target, wrong solver options) and observed that none of the four cells produced a working configuration when toggled in isolation.

### V-2.5: confounding factor in the instrumentation

While V-2 was running, the harness produced a NaN-propagation event that interrupted later loops and corrupted downstream metrics. V-2.5 isolated the cause: removing the call to `result.GetInfeasibleConstraintNames(prog)` from the per-loop diagnostic path left the simulation byte-correct over the same fixed seed. The introspection call itself was state-corrupting. Drake's behavior in this scenario is not characterized; the empirical bisection that surfaced the side effect is the receipt. The fix was to emit sentinel strings in the hot path and expose post-run introspection separately. After V-2.5, the harness's diagnostic output was trustworthy for subsequent V-steps; before V-2.5, several rounds of V-1/V-2 metrics had been quietly corrupted by NaN propagation, which is part of why the four-way bisection at V-3 produced inconclusive results.

### V-4: bisection outside the predicted hypothesis tree

V-4 expanded the V-3 bisection into a more thorough enumeration covering combinations of constraint, warm-start, and solver-option toggles, and observed that no single-variable change recovered a working configuration. The conclusion was that either the cause was a multi-variable interaction not captured by single-toggle bisection, or the cause was outside the predicted hypothesis tree entirely. V-4 closed without identifying the cause but ruled out the single-variable hypotheses inside the tree. The next experiment would either pursue multi-variable interactions or step outside the tree.

### V-5: out-of-tree experiment

V-5 was an inspection of the IK warm-start at the per-knot level: print `q_warm_full` immediately before the IPOPT solve, with the floating-base quaternion slice annotated (`q_full[7:11]` for this plant). The print revealed that the quaternion was drifting from unit norm by approximately 1e-7 over hundreds of simulation timesteps. Drake's IK validates pin constraints against unit-norm tolerance approximately 1e-6 and rejects inputs that violate it. The IK was failing in under 1 ms because the warm-start was rejected before any solver iteration began; the failure was upstream of any constraint or solver configuration the V-1 to V-4 experiments had been varying. V-5's inspection was not in the V-3 or V-4 hypothesis enumeration and would not have been reached by continuing the predicted bisection.

### V-6: quaternion-renormalization fix

V-6 implemented `_normalize_floating_quaternions` and called it before every IK pin. Mode-flips dropped from 179 to 142. The improvement was real but partial: the fix eliminated the upstream-1ms-failure path entirely (every IK call now reached the solver), and the 142 remaining flips reflected actual IK behavior rather than warm-start rejection. The next question was why the IK, given a valid warm-start, was still reporting infeasibility on enough calls to produce 142 flips.

### V-7: `d_min` disabled, root cause for the remaining flips

V-7 disabled `ik_min_distance_lower_bound` (set to 0.0) and re-ran. Mode-flips dropped from 142 to 3. With `d_min = 0.005`, every warm-start in resting contact with the manipuland (signed distance approximately 0) was a constraint violation, and the IK had no feasible region without lifting away from the contact, which conflicted with the position constraint pinning the EE to the guide path's first knot. V-7's result motivated the field split documented in §Bug catalog entry 3: `ik_min_distance_lower_bound` and `fk_min_distance` as separate knobs, both defaulting to 0.0 to match upstream.

### V-8: FK-side safety net, regression

V-8 attempted to add an FK-side `d_min` check on tail knots K..N-1, framed as a safety net complementing V-7's IK-side fix, with a 5 mm default. Re-running the same reference scenario produced 3 mode flips (matching V-7) but a different `kToBetterRepos` count: 8 versus V-7's 21. The drop indicated the controller was being kicked out of free mode by spurious wall-clock timeouts caused by the FK sweep's added per-loop signed-distance computations. V-8 passed every assertion that V-7 had passed; the regression was visible only by comparing V-8 to V-7 directly rather than to the original failing pre-V-7 state.

### V-9: revert to upstream parity

V-9 reverted the FK-side default to 0.0, matching dairlib upstream's choice that the trajectory's geometric shape is the safety primitive rather than per-knot signed-distance enforcement. The reference scenario produced byte-identical `reason_hist` over 200 loops compared to V-7. Byte-identity at this scale (200 deterministic mode-switch decisions in a fixed seed) is the strongest confirmation available that the revert restored the prior working state.

The arc surfaces three reusable patterns. First, instrumentation that mutates state under repeated calls is a confounder that can corrupt several rounds of bisection before being identified; sentinel-emission with post-run introspection is the standard alternative when in-loop diagnostic calls are suspect. Second, when a single-variable bisection across the predicted hypothesis tree fails to recover a working configuration, the next experiment should step outside the tree rather than expand to multi-variable interactions inside it; multi-variable expansion grows combinatorially and rarely surfaces upstream-of-the-tree causes. Third, a fix should be verified by side-by-side comparison against the prior working state and not only against the original failing state; V-8 passed every assertion that V-7 had passed but introduced a new regression visible only against V-7 directly. The third pattern is the basis of `test_obstacle_in_path_without_dmin` paired with `test_obstacle_in_path_with_dmin`, which verifies both the success branch and the failure branch as a single structural assertion. See §Test design patterns for the generalization.


## Diagnostic discipline (S8.0 → S8.4): the wrapper-side arc that closed at the inner solver

Step 8 followed the V-9 close-out and was scoped to wrapper-level investigation: verdict-A (the kIK live-sim run and the PWL live-sim run both ended with the box not moving) had ruled out a kIK-specific regression and pointed at shared wrapper-level mechanisms. The arc from S8.0 to S8.4 shipped four real fixes, identified the actual blocker as the inner C3 ADMM solver, and closed step 8 with the box still not moving but the controller-side mechanisms understood. The arc parallels V-1 to V-9 in shape (multi-round bisection, mid-arc reframing, ending with a clear next-phase scope) but lands at a different surface — the C3 ADMM rather than the kIK tracker.

### S8.0: instrumentation for cost-balance evidence

S8.0 instrumented the wrapper's per-loop sample list and counterfactual mode-switch decisions via spy-side monkey-patching of `wrapper._build_samples`, `wrapper.inner_solver.evaluate_samples`, and `wmod.decide_mode`. The C1/C2/C3 candidates from the verdict-A planning doc (`prev_repos` discount, alignment-vs-travel calibration, kToC3Cost threshold) were each given a specific evidence pattern. Three runs of `diag-kik` produced uncomputable C1 (no loops with both prev_repos and strat_* samples), small-magnitude C2 (align/travel ratio 623, much larger than required to differ from prev_repos winning), and tautological C3 (cf_reason==kToC3Cost in 144/144 kStayInRepos loops because excluding prev_repos from the counterfactual left no other samples). The per-loop sample-tuple distribution surfaced the actual shape: `(current, prev_repos)` in 178 of 200 loops, no co-occurrence with strat_*. The "uncomputable statistic IS the finding" pattern — when a planned diagnostic measurement has no qualifying data points, the absence is itself diagnostic.

### S8.1: structural read of `_build_samples`, closed-bound mechanism

S8.1.1 read `wrapper._build_samples` and identified the mode-conditional sample structure: free-mode loops with prev_repos slot non-None call `generate_samples` with `n_samples = num_additional_samples_repos = 1`, which produces only the mandatory `kRandomOnCircle` proxy at `obj_xy − r·g_hat`, which is then passed through `is_in_workspace`. S8.1.3 verified the proxy-rejection hypothesis with a spy on `wmod.generate_samples`: 176/176 free-mode 2-tuple loops had `proxy_y > 0` (median +2.390e-6 m, byte-stable across all 176 loops) and `proxy_passes_filter == False`. The closed-bound trap at `workspace_xy_max[1] = 0.0` rejecting an ε-positive proxy y-coordinate was the structural cause. The "1.732 mm" in S8.4.3 (the byte-stable TS4↔TS2 gap) and the "2.390e-6 m" in S8.1.3 (the byte-stable proxy_y) were the kind of clean numerical fingerprints that confirm a mechanism without needing further bisection.

### S8.2: closed-bound fix shipped

S8.2 added `_WORKSPACE_BOUND_TOL = 1e-3 m` symmetric tolerance to `is_in_workspace`. 3+3+1 verification protocol (3 pre-fix kik runs, 3 post-fix kik, 1 post-fix pwl) confirmed: 2-tuple count 176→0 across all post-fix runs, no PWL regression, byte-identical PWL pickle pre/post. The fix is in `control/sampling_c3/sampling.py:158-180`. **Box did not move post-fix.** The closed-bound was a real bug — strat_* samples had been systematically rejected for an unknown number of prior runs — but fixing it did not produce object motion. Necessary but not sufficient; the next pivot was driven by data showing the wrapper still wasn't moving the box despite now having real alternatives in the sample list.

### S8.3: the prev_repos drift finding and TS4 disambiguation

S8.3.2 measured per-loop EE-to-target trajectory across the 175 prev_repos-winning loops (post-fix). Direct observation: when prev_repos won (which it did in 175/200 loops), the EE-to-target gap *grew* from 75 mm to 216 mm at +0.86 mm/loop, monotonically. The wrapper's sample selection was correct (prev_repos wins on cost), but the EE was not converging to the chosen target. Both kIK and PWL fail to move the box, but kIK settles at z=25 mm (below target) while PWL settles at z=75 mm (above) — the asymmetry pointed at the IK methods producing different `q_target` configurations rather than at the joint-PD law itself. S8.3.4 deferred a TS4 capture in favor of an initial-condition experiment (S8.3.5) which the receiving Claude correctly stopped on a premise mismatch: the framing "EE starts at 25 mm, lift to 50 mm" was wrong because the actual EE z trajectory is "starts at 200 mm, falls to 25 mm in ~50 loops, stays there." The pivot to controller-side investigation (S8.4) followed from the corrected premise.

### S8.4: gravity-comp, gain-tuning ceiling, integral expansion, contact-geometry verdict

S8.4.2 shipped Fix 4: gravity comp anticipated at `q_arm_target` rather than `current_q`. Correctness check measured 8.73 Nm shift in tau_g for the extreme test case (current at z=200 mm, target at z=50 mm). 3+3+1 verification: structurally correct, magnitude small at typical operating equilibria, no PWL regression. Box did not move; kept in place.

S8.4.3 applied the deferred TS4 spy (capturing `tracker.last_ee_knots[:, 0]` and `tracker.last_q_knots[:, 0]` per loop). TS4↔TS2 = 1.732 mm = √3 · 1 mm exactly across 526 prev_repos-winning loops in three runs — the IK lands at the guide-knot target within solver per-axis tolerance every loop. The kIK delivery's IK feasibility receipt is *strengthened* (Hypothesis E, IK silent failure, ruled out). TS4↔TS3 = 11.1 mm = one full per-stride distance: the joint-PD covers ~0% of each 10 mm stride per 10 ms control loop. Verdict: Hypothesis F, joint-PD failure to track the IK solution.

S8.4.4 attempted Fix 5: `Kp_q` 60→120 to add proportional authority. The 3-run verification surfaced the saturation receipt: `‖u‖_max = 73.54 Nm = √6 · 30 Nm` exactly, signature of 6 of 7 joints simultaneously clipping at the per-joint torque_limit. Tracking did not improve (TS4↔TS3 went the wrong way by 0.45 mm). Mode-flips tripled. PWL regressed (EE z mean 83.6 → 106.9 mm). Reverted. The "calibrate to torque headroom" framing in the S8.4.4 prompt was wrong on its premise — the headroom didn't exist; default Kp_q was already in the saturation regime. The receiving Claude's data-driven stop was the right call.

S8.4.5 applied Fix 6: `I_max` 0.5→2.0 with the Type-0-system framing (integral capacity insufficient to reject gravity-load mismatch). Mechanism verified: integral converges to ~1.0 rad·s per joint at equilibrium, providing 8 Nm of corrective torque that matches the 7.39 Nm gravity-load shift on q[1] from the S8.4.2.2 measurement to within 10%. The integral is doing its theoretical job. Practical magnitude small (EE z mean +0.77 mm). PWL byte-identical. Kept in place.

S8.4.6 verified contact geometry. Pusher (radius 25 mm) and box (100 mm cube at obj_xy ≈ 0, z=50 mm). Across the 24 c3-mode loops in a representative post-fix run, contact count = 0/24. Min signed distance +12.95 mm at t=13. The C3 ADMM trajectory swings the EE around the box's south face during the descent phase, then far outbound after settling; the z and xy trajectories never simultaneously intersect the box's volume. **The blocker is in the inner solver, not the wrapper.**

### Step 8 closure

Five fixes attempted, four shipped (closed-bound, gravity-comp at q_target, I_max raised, Kp revert), one reverted (Kp doubling). The box did not move; the bottleneck is upstream of every controller-side mechanism investigated. The deferred investigation surface (the C3 ADMM solver, LCS formulation, contact-mode selection) is documented in `docs/step9_c3_admm_proposal.md`. The kIK delivery's V-9 receipts stand and are strengthened by S8.4.3's TS4 verification. The four shipped fixes are kept in place as structurally-correct improvements with measurable receipts.

The arc surfaces three reusable patterns beyond V-1 → V-9's three. Fourth: when investigating a multi-mechanism failure mode, "necessary but not sufficient" findings are the dominant outcome; each fix removes one defect without producing the win condition until enough mechanisms have been peeled to expose the gating one. Fifth: the saturation signature (`‖u‖_max = √n · torque_limit`) is a standard-form receipt — when that exact arithmetic appears in measured aggregate torque, n joints are simultaneously clipping and proportional-gain increases will not produce more proportional response. Sixth: stopping on a premise mismatch (S8.3.5's "starts at 25 mm" framing being wrong) is the right call even when the prompter has explicitly approved the next step; "the data picks the verdict" supersedes prompter intent when they conflict, and the cost of stopping is small relative to the cost of executing a wrong-shaped task.
