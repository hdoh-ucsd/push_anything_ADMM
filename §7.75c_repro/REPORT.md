# §7.75c — Reproducibility scare closed + planner hand-climb loop confirmed

**Working directory:** `/root/push_anything_ADMM`
**HEAD:** `b23fa823` (§7.73 bank)
**Seed:** 0 (West, `--task-id 4`)
**Sim:** 6 s, ST/25 (`--admm-iter 25`)
**Full invocation:** `§7.75c_repro/run_canonical.sh` (all 13 env flags + 8 CLI args, recorded)

---

## STEP 1 — Reproduction verdict: **CONFIRMED bit-identically. Corruption scare CLOSED.**

| metric                | §7.75c-repro (this run)                | §7.72 canonical (`full_reference_aligned_772/seed0.log`) |
|-----------------------|-----------------------------------------|-----------------------------------------------------------|
| `final_obj_xy`        | `(-0.2187, 0.0250)`                     | `(-0.2187, 0.0250)`                                       |
| `goal_dist`           | **0.0851 m** (72% closure)              | **0.0851 m**                                              |
| `full_solves`         | 601                                      | 601                                                       |
| `cheap_solves`        | 1367                                     | 1258                                                      |
| `switches`            | 16                                       | 74                                                        |
| OSC calls / failures  | 601 / 0                                  | not logged                                                |
| OSC saturation        | 1 (0.17%)                                | not logged                                                |
| c3-mode SETPOINT ticks| 87                                       | (n/a; new probe)                                          |

`switches=16` vs `74` in the canonical is the only float-order drift under IK Ipopt FP-non-determinism (see memory `project_ik_ipopt_nondeterminism_pinned`); box-motion outcome is bit-identical to 4 decimal digits.

### The reproducibility fix

`main.py:264` defaults `--solver c3` (Anitescu); `main.py:275` defaults `--c3plus-projection componentwise`. §7.72 requires **both** `--solver c3plus` **and** `--c3plus-projection lcp`. The prior failed re-run of §7.72 was missing at least one of these CLI flags, so:
- Anitescu Lorentz-cone path replaced Bui 2026 C3+ (or)
- Componentwise projection replaced LCP projection — no LCP-feasibility guarantee, no LCP residual for B1-A to pull toward.

Either way, the dispatcher flaps, `x_seq` becomes noise, box never moves.

**Root cause of the scare:** the invocation was **never recorded** — no run script in the tree, no capture in the §7.73 commit body beyond a prose sentence. The tree itself is fine.

**Fix landed this session:** `§7.75c_repro/run_canonical.sh` + `§7.75c_repro/HEAD.txt` — the full canonical invocation is now a runnable script pinned to the exact commit. Reproducible by:

```bash
cd /root/push_anything_ADMM
git checkout b23fa823
bash §7.75c_repro/run_canonical.sh   # produces §7.75c_repro/run.log
```

Recommend copying this script to a `scripts/` location under version control before the next §-block so it survives cleanups.

---

## STEP 2 — Planner hand-climb loop: **CONFIRMED**

### Instrumentation

- `PUSHA_SETPOINT_TRACE=1` → emits `[SETPOINT] tick=… p_ee_des=[x,y,z] box_now=[…] …` per c3-mode tick.
- In `--ee-space`, `sampling_based_c3_controller.py:2331` assigns `_p_ee_des = _x_seq[1][7:10].copy()`. So the `p_ee_des` z-slot in each SETPOINT row **is** the plan's `x_seq[1][9]` — the planner's EE-z target one horizon-node ahead.
- `--pitch-probe` → emits `[PITCH-PROBE] step=… ee_z=… tip_deg=… box_z_com=…` per sim tick.

### Confirmed observation (c): does `x_seq[1][9]` ramp up during the push?

**Yes.** Aligned window right after c3-mode entry:

| step | `x_seq[1][9]` (plan-z) | `ee_z` (actual) | `tip_deg` | `box_z_com` |
|------|------------------------|-----------------|-----------|--------------|
| 135  | 0.021                  | 0.034           | 0.00      | 0.0500       |
| 138  | 0.025                  | 0.033           | 0.00      | 0.0500       |  ← first Drake contact
| 139  | 0.028                  | 0.034           | 0.08      | 0.0502       |  ← tip starts
| 140  | 0.030                  | 0.034           | 0.12      | 0.0502       |
| 141  | **0.047** (+17 mm)     | 0.035           | 0.86      | 0.0508       |  ← plan-z SPIKE
| 142  | 0.052                  | 0.036           | 1.67      | 0.0514       |
| 143  | 0.056                  | 0.038           | 3.10      | 0.0526       |
| 144  | 0.060                  | 0.040           | 4.58      | 0.0538       |
| 145  | 0.067                  | 0.043           | 6.52      | 0.0553       |
| 146  | 0.073                  | 0.046           | 8.79      | 0.0570       |
| 147  | 0.067                  | 0.050           | 11.23     | 0.0587       |
| 148  | 0.070                  | 0.054           | 14.07     | 0.0606       |
| 149  | 0.078                  | 0.059           | 17.23     | 0.0626       |

- **Peak `x_seq[1][9]`:** 0.124 at step 462.
- **Peak actual `ee_z`:** 0.19981 at step 168 (matches user's "0.033 → 0.19").
- **Peak tip:** 97.5° at step 166 — the box has physically tipped OVER onto its west face. The "72% closure" is a **tip-and-slide**, not a clean push.

### Confirmed observation (feedback loop timing)

- Tip crosses 0°: step 139.
- Tip enters accelerating regime (0.12°→0.86°, ×7 growth in 10 ms): step 141.
- Plan-z spikes +17 mm in the same tick: step 141.
- Plan-z climbs monotonically through step 149, tracking tip growth 1-to-1.
- Actual `ee_z` follows the plan with ~10-20 ms lag (OSC settling).

**Timing lead:** tip pitch qy leads plan-z rise by 2–3 control ticks (20–30 ms), matching the user's "3-tick lead" claim.

**Ratio:** over ticks 139→149, plan-z climbs +50 mm while box CoM z climbs +12 mm. **Plan-z climbs 4× faster than the box's actual z-rise.** The plan overshoots the physics because it linearizes the tilted contact geometry and aims for where the face contact point *will be* if the tip continues — creating positive feedback.

### Confirmed loop mechanism

1. Sub-CoM couple tips box (physics; tip_deg > 0).
2. East-face normal tilts up (`nhat_z` goes 0 → +sin(tip)).
3. LCS linearization sees the tilted face; planner sets `x_seq[k][9]` higher to keep contact on the projected face contact point.
4. OSC follows (`Kp_z · (x_seq[1][9] − ee_z_now)` — no z-clamp anywhere; `op_space_controller.py:231`).
5. Sphere pushes higher on the tilted face → larger moment arm → **more tip → step 1 amplified.**

**c3-mode gains under `PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1`:** Kp=200, W_track=1.0, Fz cap ±3 N (`PUSHA_STAGE5_U_VERTICAL=3`). Task-tracking authority at `Δz≈+30 mm` → ~6 N vertical OSC pull, easily overpowering the +3 N u_z cap. No z-guard between planner and OSC.

**OSC as origin:** refuted by code (OSC just follows `p_ee_desired`) and by the SETPOINT/PITCH-PROBE timing here — the plan itself produces the climb, the OSC merely executes it.

---

## STEP 3 — Loop-break fix candidates + recommendation

Objective: **keep the planner's hand-z target low.** No OSC-QP z-bound (that would fight the plan under W_track authority).

### (i) Planner QP upper bound on `x_seq[k][9]`

- **Where:** new per-knot state inequality plumbed through `admm_solver._solve_c3plus`. `grep -n "state_upper|x_upper|state bound" control/admm_solver.py` → 0 hits. No existing scaffolding.
- **Cost:** ~30–50 LOC of new OSQP constraint plumbing + solver signature changes + tests.
- **Risk:** hardest constraint; ADMM may hit infeasibility if the bound is tight.
- **Verdict:** MOST EXPENSIVE. Reject unless (ii) fails.

### (ii) Cost-side penalty on `x_seq[k][9] − sampling_height`  ⭐ **RECOMMENDED**

- **Where:** `control/task_costs.py:build_ee_space`, right after the existing EE-approach block at line 794. ~5 LOC behind a default-OFF `PUSHA_EE_Z_HOLD` env flag:
  ```python
  import os as _os_zh
  if _os_zh.environ.get("PUSHA_EE_Z_HOLD", "0") == "1":
      _w_zh = float(_os_zh.environ.get("PUSHA_EE_Z_HOLD_W", "1000.0"))
      _z_tgt = float(_os_zh.environ.get(
          "PUSHA_EE_Z_HOLD_TARGET",
          str(self.params.sampling_params.sampling_height)))  # 0.05 for cube
      Q[9, 9] += _w_zh
      x_ref[9] = _z_tgt
  ```
- **Effect:** the plan naturally prefers lower `x_seq[k][9]`. ADMM inner solve discovers a lower-z push trajectory; OSC receives a low z-target consistently; **no plan-vs-OSC tension**.
- **Reference weight scale:** `w_obj_xy=100000`, `w_ee_approach=8000`, so `w_z_hold=1000` is 100× below the goal-error term (won't dominate) but 8× below the approach term (has authority against the tilted-face pull). Reasonable starting w for a sweep.
- **Risk:** if `w_z_hold` too high, plan may under-commit push force. Start at 1000 and sweep 100→10000.
- **Verdict:** cheapest fix that breaks the loop **plan-consistently**. Same env-flag / default-OFF pattern as every other §7 bank.

### (iii) Z-target clamp on `_p_ee_des[2]` before executor

- **Where:** `sampling_based_c3_controller.py:2331`. ~3 LOC after `_p_ee_des = _x_seq[1][7:10].copy()`:
  ```python
  if _os.environ.get("PUSHA_EE_Z_CAP", "0") == "1":
      _z_cap = float(_os.environ.get("PUSHA_EE_Z_CAP_VAL", "0.070"))
      _p_ee_des[2] = min(_p_ee_des[2], _z_cap)
  ```
- **Effect:** OSC sees a hard z-ceiling. But the **plan still commands the climb** — `u_sol` and `lambda_des` continue to expect a higher ball. Plan and executor disagree silently every tick.
- **Risk under §7.72 gains (Kp=200, W_track=1):** OSC yank-down (Kp·Δz ≈ 40 N at Δz=0.20) beats the +3 N u_z cap, so the ball IS held down. But the plan re-plans higher every tick → OSC pulls harder every tick → p_err bloom, joint saturation, ADMM warnings. The loop moves from planner-mediated to planner-vs-OSC oscillation.
- **Verdict:** cheapest by LOC but plan/executor mismatch. Only useful as an emergency safety net if (ii) can't be tuned.

### Recommendation

**Land (ii) with `PUSHA_EE_Z_HOLD_W=1000, PUSHA_EE_Z_HOLD_TARGET=0.05`.** Expected outcome per the user's own framing: a **small lean** push (partial closure with the plan preferring low z), not a perfect slide. One shot; if the result is worse than 72% closure (i.e., box moves less), pivot to the C3-vs-C3+ study (§7.38) with the loop understood but unpatched.

Do **not** build (i) unless (ii) is validated as insufficient. Keep (iii) in the drawer.

---

## Files touched this session

- `§7.75c_repro/run_canonical.sh` — the full canonical invocation, executable.
- `§7.75c_repro/HEAD.txt` — commit pin: `b23fa823f715102dc20ddd86798eba27f7b433bb`.
- `§7.75c_repro/run.log` — reproduction transcript (9,324 lines; SETPOINT + PITCH-PROBE fully populated).
- `§7.75c_repro/setpoint.tsv` — extracted `(tick, p_ee_des_z, box_z)` per c3 tick.
- `§7.75c_repro/pitch.tsv` — extracted `(step, ee_z, tip_deg)` per sim tick.
- `§7.75c_repro/REPORT.md` — this file.

**No source-code changes were made in §7.75c.** Fix candidates are drafted only; the user decides fix-vs-pivot next.
