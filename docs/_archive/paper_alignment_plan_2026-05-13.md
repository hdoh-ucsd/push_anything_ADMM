# Archive — paper_alignment_plan.md (Pattern-1 migration, 2026-05-13)

Extracted from `docs/paper_alignment_plan.md` on 2026-05-13 during the
Pattern-1 current-state / journal split. See git log for prior versions
of `paper_alignment_plan.md`. The verbatim prose below is preserved in
source order. Round-summary entries in `docs/journal.md` cite specific
dates from this archive; per-discipline extracts live in
`docs/lessons.md`.

------------------------------------------------------------

## 9.3.3 — α-mechanism verification (Item 1.1 inline)

Verification (9.3.3 data): c_C3 cost gap inverted on both paths. Path A +75k → −10.6k at step 800. Path D +69k → −181.6k. Mode switches collapsed (A: 10 → 4, D: 78 → 6). Empty-LCS fraction collapsed (A: 94.9% → 50.1%, D: 95.0% → 2.75%). Object moved 33.6mm (A) and 47.4mm (D). Direction: NW (away from goal at +x). PARTIALLY LOAD-BEARING.

------------------------------------------------------------

## 9.3.4 — Finding-C C-fix verification (Item 1.2 inline)

Verification (9.3.4 data): change applied correctly, verified via `[proxy]` log on Path D at step 800. NO-CHANGE in behavior on either path. Path A: EE never enters stage-1 regime (stuck at ee_to_box ≈ 0.17m). Path D: EE overshoots new staging target by ~10cm, invariant to coefficient.

Implication: Finding C as framed was either not load-bearing for verdict-A, or its locus is downstream of the binding constraint (see 9.4 / 9.4.1).

------------------------------------------------------------

## 9.4.5-B Attempt 1 + sub-attempt 1.5 (Effect C adversarial interaction)

**Effect C creates an adversarial interaction between Effect B fixes and verdict-A on Path D.** 9.4.5-B Attempt 1 (commit `161b7d9`, I_max 2.0→4.0) produced PARTIAL Effect B improvement (clamped integrators 3→1, EE displacement 197mm→132mm at t=30s, joint 1 q_err -0.405→-0.343 rad) and a minor Path D shift (empty-LCS 1.43%→80.9%, obj motion 35.2mm→48.7mm). 9.4.5-B sub-attempt 1.5 (I_max 4.0→7.0, reverted) confirmed Effect B is solvable by integrator authority alone (joint 1 unclamped at -6.12, EE displacement 25.25mm at t=30s, q_err -0.0151 rad — closed-form predicted, see analysis above) but **regressed Path D verdict-A to 0 mm box motion / 100% empty-LCS / 70 mode switches**. The Path D empty-LCS evolution is monotonic in I_max, ruling out the measurement-artifact hypothesis that was open after Attempt 1. **Fixing Effects A and B in isolation does not unblock verdict-A.**

The 9.4.5-B premise — that fixing executor home-hold would unblock verdict-A — is **falsified**. 9.4.5-D Candidate 2 sub-option 1a (`w_align` decay) was queued and then attempted; 9.4.5-E (sub-option 1c, K-loop lock-in) and 9.4.5-F (sub-option 1d, watchdog) followed. All four wrapper-side sub-options targeting Effect C are now falsified — see the next subsection.

------------------------------------------------------------

## 9.4.5-D / -E / -F — wrapper-side fix sub-options falsified

**Wrapper-side fix sub-options falsified (9.4.5-D through 9.4.5-F).**

Per-attempt result, against the verdict-A Path D / Path A regression criteria:

- **1a — `w_align` decay (9.4.5-D):** falsified by direct cost arithmetic. The c_C3_raw gap between fresh `strat_*` samples and `prev_repos` measures ~185k in the late-run state; the alignment bonus maxes at `w_align · max(0, n̂·ĝ) = 30000`. Reducing the alignment bonus by any factor cannot flip prev_repos's `c_sample` advantage when c_C3_raw dominates by 6×. The lock-in is not bonus-driven.
- **1b — raise `w_travel`:** skipped per the same c_C3_raw reasoning. The travel-penalty term `w_travel · ‖p_sample − p_ee_now‖` is two orders of magnitude smaller than the c_C3_raw gap (in the verdict-A regime travel penalties run 10–30 cost units, vs the ~185k gap), and raising `w_travel` to a magnitude that bridges the gap would dominate the cost over realistic Cartesian distances.
- **1c — K-loop lock-in (9.4.5-E):** NO PROGRESS on Path D (object motion 36 mm; `prev_repos` wins 750/801, essentially unchanged from baseline 761/801). Path A regressed (35.8 mm → 0.0 mm). The lock-in mechanism worked (15 exclusion events fired correctly on Path D, 14 on Path A), but fresh `strat_*` samples did not win the subsequent loop — only 5 `strat_0` wins out of 51 non-`prev_repos` wins on Path D. Removing `prev_repos` from the sample list does not promote fresh strategy samples; the wrapper picks `current` or `buffer` instead, neither of which is contact-seeking.
- **1d — `steps_since_improve` watchdog (9.4.5-F):** NO PROGRESS on Path D in spirit (object moved 73 mm NW, *away* from goal at +x; goal_dist 0.300 → 0.2995 m, essentially unchanged; c3-mode time 61/801 = 7.6 %, far below the 50 % threshold for FULLY RESOLVED). Path A regressed (10.8 mm at HEAD baseline → 0.0 mm with watchdog; switches 2 → 14). **All 581 `[C3]` dispatches during forced c3-mode showed `n_λ=0` (empty LCS).** The watchdog mechanism worked as designed (3 fires on Path D, 6 on Path A; `kForceC3Watchdog` switch reason emitted correctly), but the resulting c3-mode received an LCS with zero contact pairs in every dispatch. Forcing mode does not produce contact.

**Finding A promoted to binding constraint.** The 1d empty-LCS observation makes step 9 Finding A (empty-LCS gradient decoupling — see `docs/step9_findings.md` Finding A) directly visible at runtime: the LCS is empty at all EE positions the wrapper commits to. Effect C's mechanism (`prev_repos` lock-in) is downstream of Finding A — the wrapper picks samples, the kIK moves the EE there, and the LCS at that geometry has zero contact pairs. No wrapper arbitration fix (cost-tuning + sample arbitration + mode arbitration) can produce contact when the geometry at commanded positions has no contact pairs. Finding A is promoted from "structural observation" to "binding constraint for verdict-A under current configuration."

------------------------------------------------------------

## 9.4.5-G — wrapper-side path closed, architectural pause

**Fix-direction implications (post-9.4.5-F).** Wrapper-side path is **CLOSED** — sub-options 1a / 1b / 1c / 1d are exhausted, and the falsification chain (cost gap too large → fresh samples don't win → forced c3-mode finds no contact) is mechanism-level evidence that the issue is contact geometry, not arbitration. The next investigation surfaces are upstream:

- **Option U1 — Finding A directly:** characterize why the LCS is empty at commanded geometry. Cross-ref `docs/step9_findings.md` Finding A and the verdict-A trajectory through the stable point.
- **Option U2 — Sampler bias:** why `prev_repos` lands southwest (9.4.5-C unexplored surface 1 — what geometric or numerical property of `kRandomOnCircle` + the cost composition consistently selects the same off-axis target).
- **Option U3 — c_C3_raw mechanism:** why `prev_repos` has structurally lower `c_C3_raw` than fresh samples in the verdict-A regime. The 6× gap is the proximate cause of every wrapper-side falsification; characterize whether this is α-mechanism residue, surrogate-iteration bias on fresh-sample warm starts, or a deeper property of the LCS linearization at the sampled positions.
- **Option G5 — OSC architectural pivot:** replace the wrapper + kIK + joint-PD stack with a paper-faithful operational-space controller at 1 kHz. The deferred-architecture item in step 8's executor-tuning catalog.

**Status: PAUSED for architectural decision (U1 / U2 / U3 / G5).** Direction is the user's call; this entry is the institutional record of the four-sub-option exhaustion.

------------------------------------------------------------

## 9.4.6 LCS-contents probe + 9.4.7 F2 reframe

**9.4.7 F2 reframe (Finding A resolved; Effect C standalone binding constraint).**

9.4.6 (probe `bd18003`) identified the empty-LCS mechanism as a 5 mm geometric mismatch: `sampling_radius` 0.18 m minus box half-extent 0.05 m minus pusher radius 0.025 m = 0.105 m pusher-to-box surface clearance at every strategy sample, just 5 mm outside Drake's 0.10 m extraction threshold (`control/lcs_formulator.py:181`). The probe ran a 100-step WEST scenario and recorded 0/341 Class-A calls (Drake returns 0 pairs) but 189/341 Class-B calls (Drake returns pairs, project filter excludes all because no ee-box pair is within threshold). The `sampling_radius=0.18` value had no documented rationale.

9.4.7 F2 (commit `761f9f8`) reduces `sampling_radius` to 0.13 m. Target pusher-to-box surface clearance: 0.055 m (inside the 0.10 m threshold by 45 mm, above the 0.075 m hard-collision floor by 55 mm). Verified post-commit on the standard 800-step verdict-A scenarios:

- **Path D (kIK config)** — empty-LCS 80.9% → 0.0%; λ_n_max median 0 → 0.569 (99.4% non-zero across 796 dispatches); `best_src=strat_*` 0/801 → 128/801; obj_xy 48.7 mm (NW) → 10.4 mm (SW); switches 8 → 0.
- **Path A (PWL config)** — empty-LCS high → 0.0%; λ_n_max median 0 → 0.101 (99.6% non-zero across 798 dispatches); obj_xy 10.8 mm (S) → 106.5 mm (W); switches 2 → 0. Magnitude up ~10× vs HEAD baseline; direction still West (wrong sign for the +x goal) — not a regression per the 25 mm floor criterion, but the wrong-direction motion is the surface symptom of the residual Effect C.

**Finding A is RESOLVED.** Across 1594 `[C3]` dispatches between the two runs, zero `n_λ=0` lines. The empty-LCS gradient-decoupling condition is closed at its geometric mechanism. The four upstream options enumerated in the PAUSED section above are reassessed:

- **U1 (Finding A directly):** RESOLVED by F2. The 5 mm mismatch was the load-bearing mechanism.
- **U2 (sampler bias):** OPEN, deprioritized. Sample geometry now consistently produces non-empty LCS; whether the wrapper still picks the same off-axis target is downstream of arbitration, not sampler bias.
- **U3 (c_C3_raw mechanism):** OPEN. The 6× cost gap reported in 9.4.5-D was measured under empty-LCS conditions, where contact-active samples saw `n_λ=0` and thus had no contact-coupled cost-to-go. Whether the gap persists with non-empty LCS is an open question.
- **G5 (OSC architectural pivot):** DEFERRED. The project's architecture demonstrably produces motion (Path A 106.5 mm) when geometry is correct — the problem is direction of motion, not capability. Architectural pivot becomes appropriate only if the sub-option re-evaluation chain below also exhausts.

**Effect C reframe — now standalone binding constraint.** Yesterday's 9.4.5-D / -E / -F sub-option falsifications all rested on Finding A being unresolved (the wrapper's chosen targets sat at empty-LCS geometry, so no arbitration choice could produce contact). With F2 closing Finding A, the four sub-options are candidates for re-evaluation — they are not pre-judged as RESOLVED or REOPENED; each falsification was correct at the time given empty-LCS conditions, and re-testing under the post-F2 regime is the only way to know.

- **1a (`w_align` decay)** — falsified by direct c_C3_raw gap arithmetic. That gap was measured under empty-LCS conditions; the post-F2 c_C3_raw landscape is uncharacterized. Re-evaluable contingent on a c_C3_raw landscape measurement.
- **1b (raise `w_travel`)** — skipped on the same c_C3_raw reasoning. Same re-evaluation status as 1a.
- **1c (K-loop lock-in)** — falsified because fresh `strat_*` samples did not win after `prev_repos` exclusion. Path A regressed. Whether the cost landscape under non-empty LCS would let fresh samples win is open; re-evaluable contingent on c_C3_raw landscape.
- **1d (`steps_since_improve` watchdog)** — falsified because forced c3-mode produced `n_λ=0` in every dispatch. **This falsification reason is directly invalidated by F2** — post-F2, the LCS is non-empty in 99.4-99.6% of dispatches. 1d has the strongest a priori case for re-evaluation.

**Fix-direction implications (post-F2).** Wrapper-side investigation is **REOPENED**. The architectural pivot (G5) is less urgent — one bounded parameter fix upstream of arbitration (F2 itself) closed one binding constraint without architectural change, and demonstrated that the project's stack can produce substantial Cartesian motion. The next investigation surfaces are wrapper-side, in three natural starting orders:

- **Option A — Re-test 1d watchdog under F2 regime.** Lowest-effort, strongest a priori case; the prior falsification reason is directly invalidated. A successful re-run would also incidentally answer whether forced c3-mode now drives the box toward the goal under non-empty LCS.
- **Option B — Characterize c_C3_raw landscape post-F2.** Highest information yield; would inform 1a/1b/1c re-evaluation simultaneously. A per-sample `c_C3_raw` and `align_bonus` log over a 200-step run is the natural instrumentation.
- **Option C — Direct per-sample cost-breakdown analysis.** Cheapest insight into why `prev_repos` still wins (best_src=prev_repos 84-99.7% of loops post-F2). Read-only inspection of `[GS-table]` lines plus targeted instrumentation in `InnerSolver.evaluate_samples` if needed.

Direction is the user's call; no auto-proceed.

------------------------------------------------------------

## 9.4.7 Option A / B / C — executed results

**9.4.7 Option A / B / C results (executed):**

- **Option A (re-test 1d watchdog under F2):** REFINED-FALSIFIED. `kForceC3Watchdog` added to `mode_switch.py`, `watchdog_steps_since_improve_threshold` field added to `params.py` (default 0, opt-in via yaml), wrapper override + summary line wired in. Path D (kIK, threshold=100): 4 fires, c3-time 4/802 = 0.5%, mean λ_n_max = 0.594 (99.4% non-zero — empty-LCS condition stays closed), obj_xy 15.7 mm SW vs F2 baseline 10.4 mm SW. Path A (PWL): 5 fires, c3-time 86/802 = 10.7%, obj_xy 97.7 mm West vs 106.5 mm West baseline. **Yesterday's falsification reason (`n_λ=0` in forced c3-mode) is invalidated by F2; today's failure mode is c3-mode non-persistence — wrapper exits via `kToReposCost` within 1–17 steps because non-current samples still have ~50% lower c_sample.** Watchdog code is retained (default off) for future re-test infrastructure.

- **Option B / C (c_C3_raw landscape characterization):** RAN. `scripts/probe_9_4_7_B_c3_landscape.py` (per-sample CSV instrumentation, watchdog disabled) + `scripts/probe_9_4_7_C_gs_table_analysis.py` (read-only `[GS-table]` parser across F2 + Option A logs). **The 6× c_C3_raw gap reported in 9.4.5-D is GONE.** Combined post-F2 data: prev_repos vs strat_0 c_C3_raw median gap = 0.80, mean −7.58, range −85 to +23 cost units — statistically indistinguishable. `c_sample` is now dominated by `align_bonus` (~30k) and small `travel_penalty` (~25), not by a c_C3_raw differential. **strat_0 wins WHEN it is generated**: 5/40 sampled blocks for Path D F2, strat_0 won every time it appeared. The new binding observation is that **strat_0 is generated only 7/40 sampled blocks (17%)** because `SamplingParams.workspace_xy_max[1] = 0.0` rejects all y > 0 random samples on the 0.13 m circle around the box (which sits near `obj_y = 0`). Across 33/40 sampled blocks for Path D F2, `prev_repos` was the *only* non-current candidate.

------------------------------------------------------------

## Diagnostic chain 9.4 → 9.4.4 (Hypothesis F re-derivation)

Diagnostic chain that re-derived the mechanism:

- **9.4 (commit 0b0ee69)** — Standalone kIK probe drove the tracker toward verdict-A's W1 and W2 targets with no wrapper, no C3, no box. Both converged to essentially the same EE position (~(-0.016, -0.084, 0.025)) regardless of commanded target. IK reports feasible (knot-0 IK lands within √3·1mm of the guide knot per `docs/reposition_ik.md:150`).
- **9.4.1 (commit 61f064c)** — Per-joint torque breakdown ruled OUT saturation as the cause: 6 saturated joint-steps out of 5,607 (joint 5 transient only). Steady-state demand sub-clamp on every joint. Reframed as PD steady-state equilibrium issue.
- **9.4.2 (commit dd49e59)** — 30s integrator probe ruled OUT anti-windup, leak, reset-events, and different-error hypotheses (H1/H3/H4/H5). Confirmed H2 (slow update) — integrator winds linearly toward clamp.
- **9.4.3 (commit d87f386)** — 90s clamped-integrator probe + analytical Path B sanity check. Joint 1 clamps at t≈60s with residual q_err -0.033 rad. **EE-to-target distance is invariant from t=1s through t=90s at 0.159m**, indistinguishable from verdict-A's 16cm miss. Ki sufficiency and gravity-model error both ruled out (Path A and Path B agree on closed-loop force balance to within rounding). The closed-loop fixed point IS the binding constraint.
- **9.4.4** — `_build_guide_path` trace surfaced the structural cause: the IK targets `p_guide[:, 0] = next_waypoint(ee_now, p_target, z_safe=0.20, ds=0.01m)`, recomputed from current `ee_now` each control step. With `num_full_ik_knots=1` (default), only knot 0 is sent to IK. The PD reaches the per-loop "1 cm ahead of where I am" target; the guide rebuilds with new `ee_now` ≈ old `ee_now`; the cycle is stable at a fixed point.

------------------------------------------------------------

## Critical-takeaway narrative (paragraphs 2-3)

The day's load-bearing methodological lesson is **prior-art rediscovery, not novel diagnosis**. The 9.4 → 9.4.3 chain re-derived a mechanism the project had already characterized empirically (TS4↔TS3 = 11.1 mm = one full per-stride distance). The chain was rigorous but inefficient — searching `docs/reposition_ik.md` for "guide" or "stride" or "Hypothesis F" earlier in the chain would have surfaced the prior characterization. The institutional discipline note added to step 9's backlog: when investigating a mechanism, search project docs early for prior-art entries, especially docs that catalog hypotheses or defects (e.g., `docs/reposition_ik.md` Bug catalog, Operational notes, Step 8 closure sections).

What is genuinely new is the **promotion**: prior to α + C-fix shipping, Hypothesis F was second-order behind "wrapper picks wrong targets." After α + C-fix, the wrapper picks contact-seeking targets and Hypothesis F becomes binding. Future paper-alignment work on Items 3.1-3.4 (polish) should remain gated behind the Hypothesis F fix — the same gating step 8 identified, now with a promoted urgency.
