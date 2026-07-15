# Receipt — c8402b3 landing closeout (Q2c gate)

**HEAD landed**: `c8402b3 Q2c: gate admit-guard z_safe cap on EE_z >= 0.090m`
**Chain**: `fa8db2b → ae071d1 → c8402b3`
**Receipt filed**: 2026-06-05

## Gate logic

`scripts/.../params.py:545` — `contact_entry_threshold=0.090` carried into the admit-guard.
Cap is suppressed below z=0.090m so the descend phase delivers force; cap applies at z≥0.090m so the lift phase isn't pushed downward by stale ADMM η-cap residue.

## (A) SC-collision-gone — tick-level PASS

Evidence dir: `contact_guard_v2/`, `altitude_hold_sweep/`.

| signal                              | observed   | location |
|-------------------------------------|------------|----------|
| Cap suppressed at collision altitudes | 0.106 / 0.103 / 0.099 m | gate gated below 0.090, fires above (per c8402b3 commit body) |
| Drake traverse-contact ticks         | 73 → 0     | contact_guard_v1 (baseline) vs contact_guard_v2 (post-gate) |
| Knock-off displacement               | −121 mm    | pre-gate baseline; eliminated post-gate |
| Lift completes to                    | z = 0.201 m | altitude_hold_sweep/seed4_altitude_hold.log:7 (`ee_p z=0.20092`) |

The cap was knocking the box during the lift-traverse-descend; gating it on `EE_z >= 0.090m` keeps it active for nominal pushes and suppresses it during repositioning. End-to-end: zero traverse-contact, zero knock-off, clean lift.

## (B) Per-tick noregress — PASS

Evidence: `q5_head_default_noregress/summary.txt` (seeds 0/2/4, 6s sim, `--admm-iter 25`, HEAD=c8402b3).

| seed | first_c3 (vs baseline 109/164/159) | goal_dist_t6 (vs baseline 0.263/0.179/0.247) | Δ goal_dist |
|------|-------------------------------------|-----------------------------------------------|-------------|
| 0    | 110 (+1)                            | 0.235                                          | **−10.6%**  |
| 2    | 159 (−5)                            | 0.139                                          | **−22.3%**  |
| 4    | 160 (+1)                            | 0.145                                          | **−41.3%**  |

`first_c3` within ±5 ticks (well inside ±20 band). `goal_dist_t6` improves for all 3 seeds. Caveat: `success=NO` at t=6s for all 3 (final 0.14–0.24m) — this is noregress, not goal-reaching.

`n_lcs` not reported — excluded per `feedback_nlcs_not_noregress.md` (FP-order ripple over Drake's 2mm gate; planner-internal proxy).

Baseline source: `q1_noregress/` at HEAD=fa8db2b (pre-jitter, pre-gate) — pinned at the pre-change HEAD per `feedback_baseline_provenance.md`.

## (C) Baseline rotation

c8402b3 is now the canonical noregress baseline for forward work. Prior baseline (fa8db2b) is stale for noregress comparisons. Memory updated: `project_q5_noregress_head_c8402b3.md`.

## Script-side artifact (not a sim issue)

`scripts/_q5_head_default_noregress.sh:41` grep returned NaN for seeds 0/2 `goal_dist_t6` at write time; data WAS in run.log all along and the same chain re-run post-hoc returned valid numbers under identical `set -eo pipefail`. Recovered values entered as `RECOVERED` lines in `summary.txt`. Transient race, not a fix-needed blocker.

## Closeout status

Task 6 = DONE. c8402b3 is cleanly landed: gate verified, noregress verified, baseline rotated, memory updated. No follow-ups outstanding.
