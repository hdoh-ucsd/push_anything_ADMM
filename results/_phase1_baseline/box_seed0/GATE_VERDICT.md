# Phase-1 box seed=0 gate verdict

**Branch/HEAD:** `reproduce-dairlib @ aa42789` (Task 6 tip; Task 7 is this evaluation).
**Phase-0 anchor:** commit `de14138` — `goal_dist=0.1289m` (57% closure), 0 QP failures, 6.16% saturation, orient_err=1.68 rad.

## Phase-1 measurement

`[RESULT] method=sampling-c3  final_obj_xy=(-0.2523, -0.0257)  goal_dist=0.0541m  orient_err=1.6302rad  success=NO  ref_gate=FAIL`

`[OSC-SUMMARY] calls=601  qp_failures=0 (0.00%)  saturation=1 (0.17%)  avg_solve_ms=0.31`

## Gate evaluation (§3 Phase 1 go/no-go)

| Metric | Bar | Phase-0 | Phase-1 | Result |
|---|---|---|---|---|
| `goal_dist` | ≤ 0.135 m | 0.1289 m | **0.0541 m** | ✅ PASS (60% margin) |
| Closure (of 0.3 m init) | — | 57% | **82%** | +25 pp |
| `qp_failures` | == 0 | 0 | **0** | ✅ PASS |
| OSC saturation | ≤ 8% | 6.16% | **0.17%** | ✅ PASS (97% reduction) |
| Overshoot (final x vs goal x=-0.30) | — | −0.12 m past | **−0.05 m short** | Undershoot instead of overshoot |
| `orient_err` | not gated | 1.68 rad (96°) | 1.63 rad (93°) | ~same tumble |

**Verdict: GO.** All three gate criteria pass with margin. Not just no-regression — Phase-1 improves translation from 57% to 82% closure while reducing OSC saturation by 36× (from 6.16% → 0.17%).

## Mechanism read (what the change achieved)

The port's pre-Phase-1 c3-mode executor used Kp=400/W_track=100 (compound position authority = 40,000) plus the older `use_force_tracking` scaffolding. That produced a 200× over-drive vs the reference's 1:1 position:force weighting (200), causing the box to be *hammered* past the goal (−0.42 m past a −0.30 m goal) with high saturation (6.16%).

Post-Phase-1, c3-mode uses Kp=200/W_track=1 (200) with joint-2 posture pull to 1.1 rad and a trajectory-shaped input. The box arrives short of the goal (−0.25 m vs −0.30 m target) instead of overshooting past it — a fundamentally different failure mode. Saturation drops 36× because the QP isn't fighting a hyperactive position task.

## What Phase 1 does *not* fix

The orientation error stays at ~93°: the box tumbles under the aligned executor almost identically to the un-aligned executor. That matches the alignment-project characterization ("the box tumbles under the reference's own controller") and is Phase 3's territory (LCS↔Drake contact-model interaction is the named §4 risk; tumble is likely downstream of that).

## Next

Task 8 (T seed=0 diagnostic — advisory). Then Task 9 close-out.
