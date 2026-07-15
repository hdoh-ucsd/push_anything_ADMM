# Stage 2 — 20-seed contact-rate sweep, post-fix

**Date:** 2026-05-29.
**Commit under test:** `b9a3c2a` (gravity-comp ownership fix — committed atomically before sweep).
**Pre-registered bar:** ≥ 15 / 20 seeds form EE-box contact (locked at `PREREGISTERED_BAR.md` BEFORE results came back).
**Baseline:** `38dbf18` (pre-fix), `0 / 20` seeds formed any contact.

## Verdict — PASS

**17 / 20 seeds (85%) formed EE-box contact.** Comfortably clears the ≥ 15 / 20 bar.

| metric | pre-fix `38dbf18` | post-fix `b9a3c2a` |
|---|---|---|
| seeds with ≥1 contact step | **0 / 20 (0%)** | **17 / 20 (85%)** |
| seeds with 0 contact steps | 20 / 20 | 3 / 20 (seeds 3, 8, 9) |

The fix unblocked the substrate. The arm descends, reaches the box, Drake admits the EE-box contact pair. For the first time in the branch's history, contact forms reproducibly across the seed distribution.

## Per-seed tally

```
seed   steps  contact_Y  notes
  0    222     63        cap (timeout at 942s, 88% of sim)
  1    236     65        cap
  2    226     64        cap
  3    251      0        clean run, no contact ← FAIL (face-picker TOP)
  4    251     31        clean run
  5    229     63        cap
  6    248     41        cap
  7    224     63        cap
  8    196      0        partial run, no contact ← FAIL (face-picker TOP)
  9    185      0        partial run, no contact ← FAIL (face-picker TOP)
 10    243     45        cap
 11    242     56        cap
 12    244     56        cap
 13    251     31        clean run
 14    227     66        cap
 15    220     62        cap
 16    231     65        cap
 17    251     26        clean run
 18    251     33        clean run
 19    251     75        clean run
```

(The "cap" seeds hit the 15-min wall-clock per-run timeout but completed 88-99% of the 2.5 s sim and made substantial contact within the truncated run; their contact verdict is unambiguous. Seeds 8 and 9 hit an external interruption during the first run-attempt; their truncated 185/196 steps with 0 contact are a conservative "no contact" verdict that the protocol allows — they were not re-run because at 74-78% of the sim duration without contact they had already failed the contact-formation test.)

Contact-event counts on the 17 PASS seeds range 26–75 per run (median ~63). These are not single-frame brushes — they are sustained contact windows of dozens of timesteps. The dynamics are now working.

## Pre-registered failure-mode discrimination

Per `PREREGISTERED_BAR.md`:

| outcome | reading | actual |
|---|---|---|
| ≥15/20 | PASS — fix unblocked substrate | **17/20 — landed here** |
| 8–14/20 | PARTIAL — face-picker still picks TOP for some seeds | — |
| 1–7/20 | FAIL — deeper contact-formation issue | — |
| 0/20 | UNEXPECTED — likely regression | — |

The PASS landed exactly where predicted with the secondary-issue clause: 3/20 still fail, and they fail by the documented face-picker mechanism. See next section.

## The 3 failed seeds — face-picker TOP-face selection

Seeds 3, 8, 9 all show `[APPROACH-OVERRIDE] step=1 face_axis=2 face_sign=+1` — the face picker selected the TOP face of the box on the very first step. This is the bug fingered in `project_l26_unrunnable_root_cause.md` and the bug that `stash@{0}` (the directional face-picker patch, still not applied) was written to fix.

Mode-distribution comparison for the 3 failed seeds:
- seed 3: 132 c3 / 119 free (47% free — dispatcher kept retreating)
- seed 8: 138 c3 / 58 free (30% free)
- seed 9: 104 c3 / 81 free (44% free)

vs. the contacting seeds which sit much more in c3 mode (seed 0 has 56 c3 / 4 free in its first 60 steps). The TOP-face pick leads to bad cost gradients, dispatcher bouncing between modes, no contact formation. This is the known stash@{0} bug surfacing exactly as predicted in the "next wall" framing of the Stage-1 report.

## What this fix did NOT do

This is exactly what the Stage-1 report flagged. The fix unblocked testing the next layer, it did NOT make pushing work:

- **Box motion**: still ~0 mm on every seed. None of the 17 contacting seeds pushed the box meaningfully. Sustained contact (median ~63 contact steps per seed) did not translate to box translation. The post-contact push direction / magnitude is the next investigation, and the original SC3 statistics from REPORT.md (60% mis-directed, 68% recoil — n=1) now need re-measurement on real data.
- **Goal**: 0/20 seeds reached the goal. `goal_dist = 0.3000m` on every completed seed.
- **Face-picker**: 3/20 seeds still fail at TOP-face selection. The directional face-picker (`stash@{0}`) is the next concrete lever; the Stage-1 report's "stash now needs testing under a descending arm" framing applies literally.
- **ADMM convergence**: not measured this stage, but still 25/25 iters per the architecture (per CLAUDE.md), still ~96% of control-loop wall time. Untouched.

The right reading: **the kinematic prerequisite for pushing now holds**. The dynamics-coupled controller bugs (face-picker direction, force-command quality, ADMM) are still ahead.

## Tree state at end of Stage 2

- `git rev-parse HEAD` = `b9a3c2a` (the Stage-1 fix commit; clean).
- Working-tree modifications:
  - `control/sampling_c3/wrapper.py` — WIRE-PROBE instrumentation (prior probe; no semantic change, env-gated).
- Stash list unchanged. `stash@{0}` (facepicker_experiment_no_op_2026-05-29) still NOT applied — that is the next stage's lever.

No further commits. No face-picker patch applied. Stopping per directive.

## Three numbers to remember

| | value |
|---|---|
| Pre-fix contact rate | 0 / 20 (0%) |
| Post-fix contact rate | **17 / 20 (85%)** |
| Pre-registered bar | ≥ 15 / 20 — **cleared** |
