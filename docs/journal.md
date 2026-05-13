# Research journal

Append-only log of investigation rounds. One entry per round, newest at
the bottom. Each entry is one paragraph (~100–200 words), datestamped,
with a one-line headline. Cross-link to commits, scripts, and the
current-state docs (paper_alignment_plan.md, step9_findings.md) when
relevant — those docs say what's true *now*; this file says what
happened.

Entry template:

    ## YYYY-MM-DD — <one-line headline>
    <round id, e.g. 9.4.7 A/B/C> — <commit hash if applicable>.
    <One paragraph: what was tested, what the data showed, what
    decision it produced. Skip background; assume the reader has the
    current-state docs open. Cross-link upstream/downstream rounds by
    date if causal.>

------------------------------------------------------------

## 2026-05-11 — α fix verified (Item 1.1, partially load-bearing)

9.3.3 verification of mechanism α — commit `3e58cc6` (`surrogate_admm_iters` 1→3). Path A: c_C3 cost gap +75k → −10.6k at step 800; mode switches 10 → 4; empty-LCS fraction 94.9% → 50.1%; object moved 33.6 mm. Path D: cost gap +69k → −181.6k; switches 78 → 6; empty-LCS 95.0% → 2.75%; object moved 47.4 mm. Box direction: NW (away from goal at +x). Verdict: PARTIALLY LOAD-BEARING — the α fix inverts the surrogate-C3 cost gap as predicted, but the box still doesn't reach the goal. The α fix opens the door for the wrapper to command contact-seeking targets, which then exposes Hypothesis F (see 2026-05-11 diagnostic chain entry) as the next binding constraint.

## 2026-05-11 — Finding-C C-fix shipped, NO-CHANGE on either path

9.3.4 verification of Finding C — commit `8f0b738` (`pre_approach_3d` coefficient 0.18 → 0.16, moving the stage-1 staging target from 5 mm outside Drake's 100 mm contact threshold to 15 mm inside). Change applied correctly, verified via `[proxy]` log on Path D at step 800. NO-CHANGE in behavior on either path. Path A: EE never enters stage-1 regime (stuck at ee_to_box ≈ 0.17 m, upstream of the C-fix's locus). Path D: EE overshoots the new staging target by ~10 cm, invariant to the coefficient. Implication: Finding C as framed was either not load-bearing for verdict-A, or its locus is downstream of the binding constraint. The 9.4 / 9.4.1 chain (see next entry) localized the binding constraint elsewhere; Finding C reframes as "structural observation correct, not load-bearing under current configuration."

## 2026-05-11 — kIK isolation chain re-derives Hypothesis F (9.4 → 9.4.4)

Four-round diagnostic chain that localized verdict-A's binding constraint to the kIK guide-path. **9.4 (`0b0ee69`)** — standalone kIK probe drove the tracker toward verdict-A's W1 and W2 targets with no wrapper, no C3, no box. Both converged to essentially the same EE position (~(-0.016, -0.084, 0.025)) regardless of commanded target; IK reports feasible. **9.4.1 (`61f064c`)** — per-joint torque breakdown ruled OUT saturation (6 saturated joint-steps out of 5,607). Reframed as PD steady-state equilibrium issue. **9.4.2 (`dd49e59`)** — 30 s integrator probe ruled OUT anti-windup, leak, reset-events, and different-error hypotheses (H1/H3/H4/H5); confirmed H2 (slow update). **9.4.3 (`d87f386`)** — 90 s clamped-integrator probe + analytical Path B sanity check. Joint 1 clamps at t ≈ 60 s with residual q_err −0.033 rad. **EE-to-target distance invariant from t=1 s through t=90 s at 0.159 m**, indistinguishable from verdict-A's 16 cm miss. Ki sufficiency and gravity-model error both ruled out. The closed-loop fixed point IS the binding constraint. **9.4.4** — `_build_guide_path` trace surfaced the structural cause: knot-0 IK target `p_guide[:, 0] = next_waypoint(ee_now, p_target, ds=0.01 m)` is rebuilt from the current `ee_now` every loop, producing a stable fixed point. The re-derived mechanism matches step 8 Hypothesis F.

## 2026-05-12 — 9.4.5-B Attempt 1 + sub-attempt 1.5 (Effect C surfaced)

9.4.5-B Attempt 1 — commit `161b7d9` (I_max 2.0 → 4.0). PARTIAL Effect B improvement: clamped integrators 3 → 1, EE displacement at t=30 s 197 mm → 132 mm, joint 1 q_err −0.405 → −0.343 rad. Minor Path D shift: empty-LCS 1.43% → 80.9%, obj motion 35.2 mm → 48.7 mm. Sub-attempt 1.5 (I_max 4.0 → 7.0, reverted) confirmed Effect B is solvable by integrator authority alone (joint 1 unclamped at −6.12, EE displacement 25.25 mm at t=30 s — within 2% of the closed-form 2× sizing prediction), but **regressed Path D verdict-A to 0 mm box motion / 100% empty-LCS / 70 mode switches**. Path D empty-LCS evolution is monotonic in I_max, ruling out the measurement-artifact hypothesis. **Fixing Effects A and B in isolation does not unblock verdict-A.** The 9.4.5-B premise — that fixing executor home-hold would unblock verdict-A — is falsified. Effect C (wrapper-internal `prev_repos` lock-in) surfaces as the new binding constraint, with step-8 Candidates 2 and 3 promoted from "Investigation needed" to "next investigation pass."

## 2026-05-12 — 9.4.5-D / -E / -F wrapper-side sub-options falsified; Finding A promoted

Three sub-options against Effect C, all falsified. **1a `w_align` decay (9.4.5-D)** — defeated by direct cost arithmetic. The c_C3_raw gap between fresh `strat_*` and `prev_repos` measured ~185k in the late-run state; alignment bonus maxes at 30k. Reducing alignment cannot flip prev_repos's advantage when c_C3_raw dominates 6×. **1b raise `w_travel`** — skipped on the same arithmetic. Travel penalty (10–30 cost units) is two orders of magnitude smaller than the gap. **1c K-loop lock-in (9.4.5-E)** — NO PROGRESS on Path D (object motion 36 mm; prev_repos wins 750/801, unchanged from baseline 761/801); Path A regressed (35.8 mm → 0.0 mm). Exclusion mechanism worked (15 fires), but fresh `strat_*` did not win the subsequent loop; wrapper picked `current` or `buffer` instead. **1d `steps_since_improve` watchdog (9.4.5-F)** — NO PROGRESS on Path D (obj +73 mm NW, away from goal); Path A regressed (10.8 mm → 0.0 mm). **All 581 `[C3]` dispatches during forced c3-mode showed `n_λ=0`** — empty LCS. Forcing mode does not produce contact. **Finding A promoted to binding constraint**: the LCS is empty at all EE positions the wrapper commits to. Effect C is downstream of Finding A under this regime.

## 2026-05-12 — 9.4.5-G architectural pause (wrapper-side path closed)

Commit `a2112d7`. The 9.4.5-D / -E / -F chain produced four falsified sub-options on the wrapper layer, with the falsification chain reading as a single mechanism statement: cost gap too large → fresh samples don't win → forced c3-mode finds no contact. This is mechanism-level evidence that the issue is contact geometry, not arbitration. Four upstream options enumerated for direction: U1 (characterize Finding A directly), U2 (sampler bias — why `prev_repos` lands SW), U3 (c_C3_raw mechanism — why prev_repos has structurally lower raw cost), G5 (OSC architectural pivot). **Status: PAUSED for architectural decision.** Subsequent rounds (2026-05-13) overturned this pause: 9.4.6 chose Option U1 as a cheap upstream investigation and 9.4.7 F2 closed Finding A as a 5 mm parameter mismatch, demonstrating that the pause-justifying premise (no wrapper-side option works) had an unexamined upstream alternative.

## 2026-05-13 — 9.4.6 probe + 9.4.7 F2 closes Finding A

Two commits: `bd18003` (9.4.6 LCS-contents probe) and `761f9f8` (F2 fix). The 9.4.6 probe identified the empty-LCS mechanism as a 5 mm geometric mismatch: `sampling_radius=0.18 m` minus box half-extent 0.05 m minus pusher radius 0.025 m = 0.105 m pusher-to-box surface clearance at every strategy sample, just 5 mm outside Drake's 0.10 m extraction threshold. The probe ran a 100-step WEST scenario: 0/341 Class-A calls (Drake returns 0 pairs), but 189/341 Class-B calls (Drake returns pairs, project filter excludes all). The 0.18 value had no documented rationale. F2 reduces `sampling_radius` to 0.13 m, targeting clearance 0.055 m (inside the threshold by 45 mm, above the 0.075 m hard floor by 55 mm). Verified post-commit on 800-step verdict-A scenarios: **Path D** — empty-LCS 80.9% → 0.0%, λ_n_max median 0 → 0.569, `best_src=strat_*` 0/801 → 128/801, obj_xy 48.7 mm NW → 10.4 mm SW. **Path A** — empty-LCS high → 0.0%, λ_n_max median 0 → 0.101, obj_xy 10.8 mm S → 106.5 mm W. **Finding A is RESOLVED**: zero `n_λ=0` across 1594 `[C3]` dispatches. The four upstream options reassessed: U1 RESOLVED; U2 OPEN deprioritized; U3 OPEN; G5 DEFERRED.

## 2026-05-13 — 9.4.7 Option A / B / C executed (sub-option re-evaluation under F2)

Commit `e2f4099`. Three follow-on rounds under the post-F2 regime, executing in order. **Option A — re-test 1d watchdog under F2.** REFINED-FALSIFIED. `kForceC3Watchdog` enum + opt-in `watchdog_steps_since_improve_threshold` param added. Path D (threshold=100): 4 fires, c3-time 4/802 = 0.5%, mean λ_n_max 0.594 (99.4% non-zero — empty-LCS condition stays closed), obj_xy 15.7 mm SW. Path A: 5 fires, c3-time 86/802 = 10.7%, obj_xy 97.7 mm West. Prior falsification reason (`n_λ=0` in forced c3) is invalidated by F2; today's failure mode is c3-mode non-persistence — wrapper exits via `kToReposCost` within 1–17 steps. **Option B/C — c_C3_raw landscape characterization.** `probe_9_4_7_B_c3_landscape.py` (per-sample CSV) + `probe_9_4_7_C_gs_table_analysis.py` (read-only `[GS-table]` parser). **The 6× c_C3_raw gap is GONE**: prev_repos vs strat_0 c_C3_raw median gap = 0.80, mean −7.58 (range −85 to +23). `c_sample` is dominated by `align_bonus` (~30k) and small `travel_penalty` (~25). strat_0 wins WHEN it is generated (5/5 sampled blocks); but workspace_xy_max[1] = 0.0 rejects all y > 0 samples on the 0.13 m circle, so strat_0 reaches the candidate list in only 7/40 sampled blocks. The new fix surface F3 (workspace y-bound relaxation) is enumerated.

## 2026-05-14 — F3 ships: workspace_xy_max[1] 0.0 → 0.13

Phase 1 (read-only audit, 2026-05-13) verdict was BRANCH-RELAX. Phase 2 swept y_max ∈ {0.00, 0.05, 0.10, 0.13, 0.15, 0.20} × {Path D, Path A}. M1 (strat_0 generation) climbed 17% → 100% (D saturates at y=0.05, A at y=0.10) confirming workspace was the binding generation constraint. M2 (strat_0 win-rate when generated) stabilized at 47.5% D / 0.0% A — the tracker-conditional asymmetry is flagged for follow-up. M3 (downstream outcome) bit-identical across y_max ∈ {0.10, 0.13, 0.15, 0.20}: Path D (-0.013, -0.009), Path A (-0.026, +0.055). No movement toward goal on either path. Outcome-B verdict: F3 necessary, not sufficient. Persistence (kToReposCost exit within 1-17 steps of any c3-mode entry) is the next binding constraint, surfaced first by 9.4.7 Option A and now confirmed standalone post-F3. Shipped at y=0.13 as the principled value (matches sampling_radius). Tests updated. CLI override --workspace-y-max retained for future ablation.
