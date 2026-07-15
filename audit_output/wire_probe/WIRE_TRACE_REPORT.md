# STEP-1 wire trace — does the Lever-3 override reach the executor?

**Date:** 2026-05-29
**Tree state at probe launch:** pristine `38dbf18`. Face-picker working-tree patch saved to `stash@{0}` ("facepicker_experiment_no_op_2026-05-29") before probing. WIRE-PROBE instrumentation added on top of pristine (37-line diff on `control/sampling_c3/wrapper.py`).
**Run:** `python main.py pushing --task-id 4 --solver c3plus --sampling-c3 --admm-iter 25 --max-time 0.6 --no-record --seed 0`. Output: `audit_output/wire_probe/seed0_pristine_with_probe.log`.

## Verdict

**(c) == (a) in 100% (56/56) of c3-mode ticks.** The Lever-3 override IS consumed by the executor in c3 mode.

The override OVERWRITES `_p_ee_des` after the FK assignment. Code flow at `wrapper.py:1414-1541`:

```
1414  if mode == "c3":
1417-1427     _p_ee_des = FK(x_seq[1])              # (b) planner setpoint set FIRST
1466-1517     if no EE-BOX pair admitted by LCS:
1517              _p_ee_des = ee_pos_now + advance*(ee_to_face/dist)   # (a) override OVERWRITES (b)
1532-1541     executor.compute_torque(p_ee_desired=_p_ee_des, ...)    # (c) whichever survived
```

Static read confirmed by probe: in every c3-mode tick of the 0.6 s run, the override fired and overwrote the FK setpoint before the executor was called.

## Match distribution (56 c3-mode ticks; 4 free-mode ticks not probed)

| ovr_fired | match_xseq | match_ovr | count |
|---|---|---|---|
| Y | N | Y | **56 / 56** |

So in c3 mode, the executor's `p_ee_desired` is **always** the override's value, never `FK(x_seq[1])`, whenever `_no_admitted_pair = True` — which (per canonical baseline) is the regime the whole run sits in (`lam_n = 0`, `n_ee_box = 0` everywhere, override fires every c3 tick).

## Per-tick z-trace (z values in metres; sampled steps)

| step | ee_z (now) | pee_xseq_z (FK, b) | pee_ovr_z (override, a) | pee_exec_z (c) |
|---:|---:|---:|---:|---:|
| 1 | +0.2001 | +0.1849 | +0.1901 | +0.1901 |
| 2 | +0.2003 | +0.1861 | +0.1903 | +0.1903 |
| 3 | +0.2010 | +0.1817 | +0.1910 | +0.1910 |
| 5 | +0.2028 | +0.1861 | +0.1928 | +0.1928 |
| 10 | +0.2084 | +0.1935 | +0.1984 | +0.1984 |
| 15 | +0.2138 | +0.1975 | +0.2038 | +0.2038 |
| 20 | +0.2195 | +0.2043 | +0.2096 | +0.2096 |
| 30 | +0.2309 | +0.2151 | +0.2210 | +0.2210 |
| 40 | +0.2432 | +0.2252 | +0.2333 | +0.2333 |
| 50 | +0.2553 | +0.2391 | +0.2454 | +0.2454 |
| 55 | +0.2609 | +0.2437 | +0.2511 | +0.2511 |
| 60 | +0.2673 | +0.2506 | +0.2575 | +0.2575 |

- ee_z drift over 60 steps (0.6 s): **+0.0672 m** (climbed 6.7 cm).
- Max planner-FK requested descent (ee_z − pee_xseq_z): **+0.0193 m**.
- Max override requested descent (ee_z − pee_ovr_z): **+0.0100 m** (advance-step cap).
- `pee_exec_z == pee_ovr_z` exactly at every probed step.

Both setpoints — planner FK (b) and override (a) — commanded ee_z BELOW the current ee_z at every tick. The executor received the override's setpoint and the EE rose anyway.

## What this rules in and out

**Ruled out:**
- Override is silently discarded in c3 mode. **No** — it is consumed. 56/56 ticks.
- FK(x_seq[1]) overwrites the override's setpoint. **No** — the direction of overwrite is the opposite (override overwrites FK).
- The framing "the Lever-3 override is decorative in this regime" — **wrong**. It is the executor's actual tracked target in every c3 tick of the run.

**Ruled in (not investigated here, per directive):**
- The override is wired and consumed AND the EE still climbs ⇒ per the branch the user gave, the cause is downstream of `p_ee_desired` — in how the climb is generated despite a position setpoint pointed downward. Candidates (not investigated): OSC position-tracking weight too low relative to other terms (force-tracking λ_ext command, posture, acc-reg); gravity-comp or bias term sign; planner u_seq dominating via the dynamics constraint inside the QP; force-tracking λ_ext pushing the arm against the descent command.

Per the methodology rule and your explicit instruction: STEP-2 (LCS / planner parking x_seq[1] in the air) is **not** ruled in by this trace — quite the opposite, the planner's FK was asking for descent every tick. STEP-2 is the next thing to investigate only after we understand why the executor doesn't track a descent setpoint it is given.

## Tree state at end of probe

- `git rev-parse HEAD` = `38dbf18`
- Working-tree modified file: `control/sampling_c3/wrapper.py` (WIRE-PROBE instrumentation only, +37 lines, no semantic change to control behaviour).
- Stash list:
  - `stash@{0}`: facepicker_experiment_no_op_2026-05-29 (the directional-face-picker patch — separate from this probe)
  - `stash@{1..3}`: pre-existing.

No commits. No fixes applied. Reporting and stopping per directive.
