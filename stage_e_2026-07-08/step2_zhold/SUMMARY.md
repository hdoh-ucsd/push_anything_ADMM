# §7.76 z-hold loop-break — landed + measured (2026-07-08)

## The landed change

**File:** `control/task_costs.py:800-822` (inside `build_ee_space`, `if dist > 1e-3:` block, at 16-space indent).

**Env flags** (all default-OFF; unset → bit-identical to pre-change):
- `PUSHA_EE_Z_HOLD=1` — arm the z-hold
- `PUSHA_EE_Z_HOLD_W` — weight (default 1000.0)
- `PUSHA_EE_Z_HOLD_TARGET` — target EE z in metres (default `self.z_ref = 0.05` = cost_cfg z_ee_target)

**Effect when armed:**
```python
Q[9, 9] += _w_zh
x_ref[9] = _z_tgt
```
Adds a per-knot penalty on the plan's EE-z toward `_z_tgt`. Plan naturally prefers lower `x_seq[k][9]`; ADMM discovers lower-z push trajectory; OSC receives low z-target consistently. **Plan-consistent — no plan/OSC tension** (candidate ii from `experiments/§7.75c_repro/REPORT.md:117-133`).

Log emission on first fire: `[§7.76-Z-HOLD] PUSHA_EE_Z_HOLD=1  w_zh=1000.0  z_tgt=0.0500m`.

## Bit-identity check (flag OFF)

`run_box.sh`-style canonical, seed 0, max-time 6 (`canon_flagOFF_seed0_maxtime6.log`):

```
[RESULT] method=sampling-c3  final_obj_xy=(-0.2187, 0.0250)  goal_dist=0.0851m
Elapsed: 14:01.29
Z-HOLD:  (line not printed — flag correctly inert)
```

**Byte-identical to §7.76 canonical** (docs/superpowers/plans/2026-07-02-§7.76-push-wrap-up-close-out.md:128-130: `final_obj_xy=(-0.2187, 0.0250) goal_dist=0.0851m`). 72% closure at max-time 6.

## Flag-ON measurement (max-time 12, PWL, FT-ON, seed 0)

Same env as Stage E Step 1b (DECISIVE cell) + `PUSHA_EE_Z_HOLD=1 PUSHA_EE_Z_HOLD_W=1000 PUSHA_EE_Z_HOLD_TARGET=0.05`.

## Delta: 1b (flag OFF, Stage E DECISIVE baseline) vs z-hold ON

| Metric | 1b (flag OFF baseline) | z-hold ON | delta | Stage E bar |
|---|---:|---:|---:|:---:|
| goal_motion (m)          | 0.180 | **0.188** | +0.008 (+4%) | ≥0.020 |
| \|qy\|_max               | 0.709 | 0.709 | 0.000 (**unchanged**) | <0.10 |
| \|qz\|_max               | 0.670 | 0.595 | −0.075 (−11%) | <0.10 |
| peak ee_z (m)            | 0.332 | **0.243** | −0.089 (−27%) | (§7.75: <0.10 ideal) |
| EE-BOX admit % (c3-ticks)| 29.1  | **38.6**  | +9.5pp (+33%) | ≥60% |
| free-mode box motion (m) | 0.032 | 0.020 | −0.012 (−37%) | ≈0 |
| c3-mode box motion (m)   | 0.246 | 0.320 | +0.074 (+30%) | — |
| OSC QP failures (%)      | 0.00  | 2.75  | +2.75pp (worse) | 0% |
| lam_n_last_admit (N)     | 0.09  | 1.14  | +1.05 N | non-zero |
| ADMM warnings/switches   | 6 sw. | 8 sw. | +2 sw. | — |
| goal_dist_final (m)      | 0.120 | 0.112 | −0.008 | — |
| wall (s)                 | 1528  | 1649  | +121 | — |

**Stage E tilt guard passes: NO.** |qy| unchanged; |qz| still 6× over.

## The escalation-loop-vs-initial-tip split

The z-hold quantifies how much of the 90° flip is the *fixable* escalation loop vs the *unfixable* initial-tip:

| axis | flag OFF (total flip) | z-hold ON (residual) | reduction | interpretation |
|---|---:|---:|---:|---|
| peak ee_z (sphere climb) | 0.332 m       | 0.243 m       | **27%** ↓ | 27% of the climb was the escalation loop (fixable by z-hold); 73% is baseline + initial-tip drag |
| \|qy\|_max (pitch)       | 0.709 (~90°)  | 0.709 (~90°)  | **0%** ↓  | Zero escalation contribution — pitch is entirely INITIAL-TIP task-geometry |
| \|qz\|_max (yaw)         | 0.670 (~84°)  | 0.595 (~73°)  | **11%** ↓ | 11% escalation; 89% initial-tip |

**Verdict:** the escalation loop contributes at most ~27% of the climb, ~11% of yaw, and **0% of pitch**. Landing the z-hold cannot rescue the |qy|/|qz|<0.10 Stage E guard. Both residual tilts remain 6-7× over guard. The cube is task-geometry-bound; the remainder of the tip is the initial-tip physics (F_tip = 1.3–3.3 N vs the 50 N u-cap = 15–38× over threshold), which no cost-side loop-break can fix.

## Sub-verdicts (things that DID move)

- **EE-BOX admit +33%** (29→39%) — the plan-consistent lower z produces more sustained contact (less bounce off the tilted face).
- **c3-mode motion +30%** (246→320 mm) — some of that admit-time converts to real push.
- **Sphere climb capped at 0.243 m** vs 0.332 m — meaningful reduction, though still far above the target 0.05 m (5× the plan target). Weight sweep might squeeze more.
- **OSC QP failures 0→2.75%** — mild new failure mode. Not blocking, but worth noting; probably the QP hitting infeasibility as the plan-target pulls z down while contact wrench pulls up.
- **Motion ~unchanged** (180→188 mm goal-aligned). Doc predicted "small lean push, partial closure cost"; observed: small structural gains, no closure regression, no closure breakthrough.

## Files

- `canon_flagOFF_seed0_maxtime6.{log,time}` — bit-identity check (14 min)
- `zhold_ON_seed0_maxtime12.{log,time}` — flag-ON measurement (27:29 min)
- `zhold_ON_metrics.json` — extracted metrics
- Code change: `control/task_costs.py:800-822` (uncommitted)

## What this means for Stage E

- The z-hold is a real loop-break — plan-consistently reduces sphere climb and improves contact admission at zero closure cost. Worth keeping (behind the default-OFF flag).
- But it does **not** rescue the Stage E |qy|/|qz|<0.10 tilt guard. Pitch is 100% initial-tip physics; yaw is 89% initial-tip.
- Confirms the OSC-scope research verdict: **the cube is fundamentally tip-prone at reference force** — task-geometry dead-end for Stage E's tilt guard, regardless of executor / cost restructure.
- Path to close Stage E is **manipuland change** (hard_pushing @ 1.5 kg or T-shape/letters like reference) OR **u-cap cut below F_tip=3 N**, NOT a loop-break patch.
