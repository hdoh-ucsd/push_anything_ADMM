# pathT_smoke_v11 — Per-sample plant-context update (Delta-1 gap fix) — CRUX RESULT

**Date:** 2026-07-05
**Baseline:** pathT_smoke_v10_5pair — 5-pair cost-LCS `force_top_k_ee_box=True` but cost-LCS linearized at *current* plant context. East strat_0 lost by 15% because contact geometry was extrapolated ~15 cm from the current arm to the east-face sample.
**This change:** Per-sample plant-context update before every cost-LCS build. Solves IK to place arm at `sample_pos`, sets plant, linearizes cost-LCS, restores plant. Mirrors reference `UpdateContext(plant, current_v, candidate_states[i])` at `sampling_based_c3_controller.cc:1628-1631`.

## Pre-registered check

The user's directive:
> Per-sample linearization → east < north + EE steers east + T pushes → the Delta-1 gap WAS the crux; the T reproduces the reference.
> Still mis-ranks → the last candidate (cost-rollout gains / planner). The per-sample update is STILL banked as a fidelity fix.

## The four outcome tests

### Test 1 — CRUX: does per-sample linearization make east < north?

**PARTIAL YES.** East sample now scores as low as **92,305** (best), routinely 100–110 k. North-close sample still scores 106–110 k. At specific ticks east decisively beats north — e.g., **step 120: prev_repos EAST (+0.150,+0.001) c_C3=102,544 (WIN) vs SW alternatives 129,432**.

Direct v10-vs-v11 comparison at the east strat_0 sample (+0.1501, +0.0007, 0.034):

| metric               | v10 (shared LCS at current arm)           | v11 (per-sample IK)                                     |
|----------------------|--------------------------------------------|---------------------------------------------------------|
| ee_t_phi (row 0)     | +0.017 → +0.026 m (extrapolated)          | **+0.0005 m** (imminent contact — at sample face)       |
| ee_t_phi (row 1)     | +0.024 → +0.033 m (also extrapolated)     | +0.1605 m (far — 2nd bar not near sample)               |
| Forward-sim dT_dx    | (not logged — added in v11)                | **-0.017 m** (WEST, correct direction toward goal)      |
| Forward-sim dT_dy    | (not logged)                               | -0.002 m (near-zero)                                    |
| c_C3_sim minimum     | ~120,000                                   | **92,305** (v11 range: 92 k – 153 k)                     |
| ik_iters             | (skipped)                                  | 3 (~0.4 mm ik_err)                                       |

**Direction restored:** v10's east sample sim moved T "in the wrong direction" (0.018 m of motion but off-goal). v11's east sample sim moves T **-0.017 m in x** (west, toward goal) with per-sample-linearized contact geometry.

### Test 2 — DISPATCH: EE steers east + wrapper enters c3?

**YES.**

- `switches=8` (v10 baseline: **0**)
- At step=400 (t=4 s): EE reached **(+0.150, +0.002, +0.057) — precisely at the east face**
- At step=193, 595, 688 (t=1.93, 5.95, 6.88 s): wrapper entered c3-mode via `kToC3Cost` or `kToC3ReachedReposTarget`
- At step=688: **contact=Y productive=Y lam_n=16.325 lam_t=13.915 f_cmd=(+4.90,-0.99)** — real Drake contact with productive-direction force command

### Test 3 — PUSH: T displacement / XY / YAW closure?

**MARGINAL / NO.**
```
[RESULT] method=sampling-c3  final_obj_xy=(0.0022, 0.0093)  goal_dist=0.2062m  success=NO
```
- Initial goal_dist: 0.206 m → Final: 0.2062 m (**+0.6 mm** — within noise)
- T ended at (0.002, 0.009, 0.020) — 2 mm +x, 9 mm +y from start; small drift NORTH (wrong direction)
- Yaw closure: not measured; T ended near +0.005 rad, goal +0.7854 rad — no meaningful rotation

Only **5 productive-contact Drake events** across 8 s (v10: 0). Contact bursts were brief (arm quickly disengaged), so no sustained push accumulated.

### Test 4 — TIP: T tips under sustained push?

**NO.** No sustained push occurred. μ=1.0 tip validation remains deferred per pre-registration.

## Verdict (per pre-registered check)

**East < north is now achievable but not consistent enough** to sustain a c3-mode push. The per-sample linearization fix:

- **Restored the direction of T motion** in the forward-sim (dT_dx=-0.017 m WEST for east sample vs v10's wrong-direction 18 mm).
- **Enabled 8 c3-mode dispatches** (v10 baseline: 0) with 5 productive-contact Drake events.
- **Enabled EE steering to the east face** (v10 baseline: never left north-side).

But the T ended within 1 mm of its start position — the improvement is **architectural, not yet sufficient** for T closure. Per the pre-registered fork:

> Still mis-ranks (marginally, T doesn't push) → **the last candidate: cost-rollout gains / planner**. The per-sample update is banked as a fidelity fix.

**Remaining candidates for actual T closure:**

1. **Cost-rollout PD gains** — `Kp_for_ee_pd_rollout=100, Kd_for_ee_pd_rollout=0.5` may not match the port's actual OSC behavior. If the forward-sim's PD tracking of the plan is too tight (or too loose), it doesn't represent what the executor would produce, so `c_C3_sim` mis-ranks.
2. **Planner-side cost weights** — `w_ee_approach=8000` biases the plan toward reposition; `w_yaw=800` may over-penalize small yaw excursions from east push. If the plan itself can't produce a productive-force sequence for the sample's LCS, no cost-LCS fix will help.
3. **Cost-LCS forward-sim gains + steps** — planner produces N=20 knots; the forward-sim rolls them out with PD tracking. If knot count / integration behavior mis-scales for the T, contact bursts (like the one at step 688) never sustain.

## Committing the fidelity banks

Per user directive: "**Commit the 5-pair + this per-sample fix as fidelity banks once validated.**"

**Validated as mechanisms:**
- 5-pair: `[COST-LCS] n_ee_t=2` on every sample; both EE-manipuland pairs admitted regardless of setback distance.
- Per-sample: `ik_iters=3 ik_err=0.0004 m`; cost-LCS `ee_t_phi=[+0.0005, +0.161]` (linearized at sample, not extrapolated).
- Together: **8 c3-dispatches + 5 productive contacts vs 0 baseline**.

**Not fully validated as sufficient for T closure** — goal_dist unchanged. That's the next-candidate work (rollout gains / planner).

## Artifacts

- `pathT_smoke_v11_persample/run.log` — 8 s sim, 801 [STEP] lines, ~120 east-sample [COST-LCS] traces, 5 productive contacts
- `pathT_smoke_v10_5pair/run.log` + `CRUX_ANALYSIS.md` — 5-pair-only baseline
- `pathT_smoke_v8/run.log` — pre-5-pair baseline (cost-LCS ranking silently no-op)
