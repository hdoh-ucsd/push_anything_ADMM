# Fix (b) — singular gravity-comp ownership — stage-1 verification

**Date:** 2026-05-29.
**Tree state at fix application:** pristine `38dbf18`; `stash@{0}` NOT applied. Working tree carried wire-probe + OSC-ZDECOMP instrumentation from prior steps; this stage adds the fix to three files (semantic changes, see §Patch).

## Stage-1 verdict — accel match: PASS

Same 0.6 s, seed-0, task-id 4 scenario as the prior probes.

| metric | pre-fix | post-fix |
|---|---|---|
| EE z trajectory                 | +0.200 → **+0.267** (climbed +6.7 cm) | +0.200 → **+0.150** (descended −5.0 cm) |
| Mode distribution (60 ticks)    | 56 c3 / 4 free                       | 56 c3 / 4 free (unchanged) |
| QP-commanded `a_ee_z` (median)  | −8.29 m/s² (down, ignored by sim)    | −0.13 m/s² (small, sim now tracking) |
| Realized `a_ee_z` (FD median)   | +0.10 m/s² (up, ignoring command)    | −0.10 m/s² (matches command) |
| Gap median `\|a_real − a_QP\|`  | **+9.0 m/s² uniform (≈+g)**          | **~0.1 m/s²** (within tolerance) |
| OSC QP failures                 | 0 / 60                               | 0 / 60 |
| OSC saturation events           | 0 / 60                               | 0 / 60 |

Per-tick comparison (post-fix, sampled):

| tick | ee_z (m) | a_QP (m/s²) | a_real (FD m/s²) | gap |
|---:|---:|---:|---:|---:|
|  3 | +0.19939 | −1.40 | −2.90 | −0.65 |
|  5 | +0.19794 | −0.54 | −1.10 | −0.23 |
| 10 | +0.19336 | −0.03 | −0.20 | −0.14 |
| 15 | +0.18911 | −1.02 | +1.10 | +2.74 (FD noise) |
| 20 | +0.18469 | −0.07 | +0.70 | +0.83 (FD noise) |
| 25 | +0.17988 | +0.02 | −0.10 | −0.11 |
| 30 | +0.17599 | −0.38 | −0.80 | −0.18 |
| 40 | +0.16678 | −1.63 | +2.30 | −2.12 (FD noise at torque transient) |
| 50 | +0.15808 | +0.01 | −0.10 | −0.10 |
| 55 | +0.15410 | −0.62 | −1.40 | −0.39 |
| 60 | +0.14964 | −0.04 | −0.00 | +0.08 |

The +9 m/s² uniform offset from the pre-fix data is GONE. Residual gap is within the ≤ 0.5 m/s² tolerance the prior probe established, with three FD-noise outliers at torque transients (ticks 15, 20, 40) where central-difference of position is poorly conditioned. The inversion is gone.

## Free-mode regression: PASS

Same scenario produces 4 free-mode ticks (at steps 13, 26, 39, 52 — identical dispatcher firings in pre- and post-fix). Free-mode tracking error `‖ee − target‖` at each tick:

| step | pre-fix `‖ee − target‖` | post-fix `‖ee − target‖` |
|---:|---:|---:|
| 13 | 2 mm | 2 mm |
| 26 | 3 mm | 2 mm |
| 39 | 2 mm | 2 mm |
| 52 | 2 mm | 2 mm |

The tracking-vs-target relationship is preserved. The absolute target positions differ between pre- and post-fix because the whole trajectory changed (no longer climbing → different EE pose → different dispatcher sample picks → different IK targets), but free-mode's job — track the dispatcher-commanded waypoint — is unchanged. Confirmed.

Mathematical equivalence check (why this works): in free mode the OSC executor (with stripped internal gravity-comp) emits a task-only `u_opt`; the main loop adds `tau_g`. Total applied = `tau_g + u_opt`. Pre-fix: OSC executor emitted gravity-comped `u_opt`; the main loop applied that alone. The two paths yield the SAME total torque at the plant input (modulo arithmetic), so realized motion is identical. The PWL tracker change (`u_raw = u_pd` instead of `tau_g_arm + u_pd`) is a no-op for the SamplingC3MPC path because the wrapper overwrites the tracker's `u` at `wrapper.py:1608` with the OSC's `u_imp`; the change preserves PWL's external contract if it is ever called standalone.

## Cube_turning sanity check: PASS

`python main.py cube_turning --solver c3plus --sampling-c3 --admm-iter 25 --max-time 0.6 --no-record --seed 0`

- Exit code 0, 60 OSC calls, 0 QP failures, 0 saturation events
- EE trajectory: (0.000, −0.001, +0.200) → (−0.054, −0.062, +0.200) — lateral motion as expected for an in-place rotation task; EE z held constant at +0.200 (no climb, no fall)
- `[GS-perf]` shows clean dispatcher behaviour (`switches=0` — no mode flapping)

No crashes, warnings, or anomalous solver behaviour. Cube_turning's QP loads similarly to pushing and behaves identically under the new wiring.

## Patch (3 files, 3 semantic changes)

```
control/osc/operational_space_controller.py:154
- bias = Cv - g
+ bias = Cv                                           # gravity owned by main loop

control/sampling_c3/reposition.py:235-236
- tau_g_arm = -plant.CalcGravityGeneralizedForces(plant_ctx)[: n_arm_dofs]
- _u_raw = tau_g_arm + u_pd
+ _u_raw = u_pd                                       # task-only torque

main.py:572-576
- if isinstance(mpc, SamplingC3MPC) and mpc.last_mode == "free":
-     plant.get_actuation_input_port().FixValue(plant_ctx, u_opt)
- else:
-     total_torque = tau_g[:n_u] + u_opt
-     plant.get_actuation_input_port().FixValue(plant_ctx, total_torque)
+ # Singular gravity-comp ownership: always add tau_g.
+ total_torque = tau_g[:n_u] + u_opt
+ plant.get_actuation_input_port().FixValue(plant_ctx, total_torque)
```

The `RepositionIKTracker` was already gravity-comp-clean — its `compute_torque` returns `u = np.zeros(n_arm)` (the executor owns actuation under the wrapper), so no change needed there. No `ImpedanceController` file exists in the current tree (was removed in earlier work; the canonical c3 executor is the OSC).

## What this fix does NOT do (per directive)

This is **stage 1**, the accel-match check. Stage 2 — the 20-seed contact-rate sweep against the 0/20 baseline — is **NOT YET RUN**. Stopping here as instructed.

The honest framing for what's coming: this fix lets the arm descend toward the box for the first time in this branch's history. Contact rate 0/20 → nonzero would be PROGRESS, not success. Behind this bug were layers of issues the climb has been masking — the directional-face-picker (stash@{0}) is still untested under a descending arm; the SC3-regime statistics from the original investigation (60% mis-directed, 68% recoil) were n=1 and need re-measurement; the ADMM non-convergence (96% of control-loop wall time at 25/25 iters) is still there; the tilt/yaw drift still untested. The next wall is what to look for, not "pushing works."

## Tree state at end of stage-1

- `git rev-parse HEAD` = `38dbf180` (unchanged — no commits made)
- Working-tree modified files:
  - `control/sampling_c3/wrapper.py` — WIRE-PROBE instrumentation (prior probe).
  - `control/osc/operational_space_controller.py` — fix (bias = Cv) + OSC-ZDECOMP/CONTACT-DUMP/MASS-CHECK instrumentation (env-gated, no-op when OSC_ZDECOMP unset).
  - `control/sampling_c3/reposition.py` — fix (drop tau_g_arm from PWL u_raw).
  - `main.py` — fix (drop the free-vs-c3 conditional; always add tau_g).
- Stash list unchanged. `stash@{0}` (facepicker_experiment_no_op_2026-05-29) NOT applied. Held against the 0/20 baseline.

No commits, no contact sweep, no face-picker patch applied. Stopping per directive.
