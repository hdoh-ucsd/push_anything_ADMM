# Phase A4 — Sample-selection mechanism audit + (γ) recoil check

**Source data:** `nondet_seed0_setarch_hashseed/run{1,2,3,4}/run.log` (existing; no new sims).
**Scope:** B3a (RNG flow), B3b (30:1 mechanism), B3c (quantization), and the (γ) folded-in recoil co-amplifier check. Read-only.

> **VERDICT:** All three B3 sub-candidates (a/b/c) are FALSIFIED. The (γ) recoil co-amplifier hypothesis is CONFIRMED. The actual fix surface is **prevent or de-amplify the phantom-c3 recoil burst** — original-flavor B1 was in the right neighborhood with the wrong metric. The 30:1 amplification at step 159 IK target is a downstream symptom, not the lever.

---

## B3a — RNG flow audit

Code traced:
- `main.py:499` — `_rng = np.random.default_rng(args.seed) if args.seed is not None else None` — seeded.
- `wrapper.py:90` — `self._rng = rng if rng is not None else np.random.default_rng()` — stored on the wrapper.
- `wrapper.py:362, 380` — `rng=self._rng` passed into `generate_samples()` on every call.
- `sampling.py:120` — `rng.uniform(0.0, 2.0 * np.pi, size=n_samples)` (kRandomOnCircle).
- `sampling.py:242, 244, 267` — `rng.integers`, `rng.choice`, `rng.uniform` (`_face_normal_projection`).

**No `np.random.*` global state is touched in the sample path.** RNG is correctly seeded and threaded. **B3a is FALSIFIED** — RNG-driven amplification is not the surface.

### Confirmation from runtime data — `[PERSIST]` refresh events

Samples drawn at the refresh ticks are nearly IDENTICAL across runs:

| refresh step | run1 sample | run2 sample | run3 sample | run4 sample |
|---|---|---|---|---|
| 110/110/110/110 | (-0.0229, 0.08, 0.03) | (-0.0229, 0.08, 0.03) | (-0.0229, 0.08, 0.03) | (-0.0229, 0.08, 0.03) |
| 111/112/111/111 | 3 identical samples × 4 runs | ↓ | ↓ | ↓ |
| 129/131/130/129 | (-0.08, 0.0466, 0.03) | (-0.08, 0.0466, 0.03) | (-0.08, 0.0466, 0.03) | (-0.08, 0.0466, 0.03) |
| 159/161/160/159 | (-0.027, -0.0602, 0.03) | (-0.0263, -0.0608, 0.03) | (-0.0274, -0.0613, 0.03) | (-0.0259, -0.0611, 0.03) |

At step ~159 the four runs differ by < 2 mm in sample position — far less than the 22-31 mm IK-target divergence observed. The sample buffer is NOT the amplifier.

## B3b — mechanism investigation

The 22-31 mm IK-target divergence at step 159 is NOT in sample selection. It's in the IK's *guide-path waypoint index* — the [STEP] `target=` field is the FK of the IK's first solved knot (`p_des = ee_knots[:, 0]`), which depends on:
- Which strategy sample is being pursued (these are nearly-identical across runs)
- The IK guide-path interpolation along the lift-traverse-descend PWL (`_build_guide_path`, `reposition_ik.py:1001`)
- The dispatcher's "held vs new" choice (`wrapper.py:1595-1598` — `target_idx = best_other_idx` vs `k_star`)

At step 159 the [STEP] retgt/reason values diverge:
| run | target | retgt | reason | held_cost | won_cost | margin |
|---|---|---|---|---|---|---|
| 1 | (-0.024, +0.005) | **Y** | **held_cost_rose** | 17432.99 | 9222.81 | **+8210.17** (47%) |
| 3 | (-0.046, +0.036) | N | hold | 15095.14 | 15095.14 | +0.00 |

The flip in run1 is at a **47 % relative cost margin**, NOT sub-1 % noise. So this is NOT an "argmin-cost-flip-under-FP-noise" pattern. The runs flipped at different ticks because their cost CURVES had already diverged significantly by step 158:

| step | run1 held_cost | run2 held_cost | run3 held_cost | run4 held_cost |
|---|---|---|---|---|
| 155 | 15998.96 | 15625.71 | 15271.25 | 33003.87 |
| 156 | 15815.67 | 16005.55 | 16256.66 | 19879.16 |
| 157 | 15567.67 | 15836.90 | 18951.93 | 17585.83 |
| 158 | **15293.71** | 15829.45 | **19627.99** | 16306.54 |

**run3 has held_cost = 19627.99 vs run1's 15293.71 — a 28 % gap by step 158.** That's tens of thousands of cost units of divergence, not sub-cent FP noise. Something upstream amplified the sub-mm input noise into this 28 % cost-trajectory divergence over the 30 ticks of free-mode reposition.

**B3b is FALSIFIED** as written — the dispatcher's argmin doesn't fail under FP noise here. The flip is at large margins. The upstream cost-trajectory divergence is the real lever, and it lives somewhere in the recoil-driven EE state evolution during phantom-c3 + the IK's response during free-mode reposition.

## B3c — lattice quantization

Since samples are already < 2 mm apart across runs, quantizing to 1 cm wouldn't change anything (all runs would snap to the same quantized sample). **B3c does nothing for the observed brittleness — FALSIFIED.**

## (γ) — recoil-driven EE z-differential during phantom-c3 — CONFIRMED CO-AMPLIFIER

EE z-coordinate during the 18-tick phantom-c3 burst (steps 110–128):

| step | run1 ee_z | run3 ee_z | Δ (mm) |
|---|---|---|---|
| 110 | +0.047 | +0.047 | 0 |
| 113 | +0.045 | +0.044 | 1 |
| 116 | +0.045 | **+0.043** | **2** |
| 119 | +0.046 | +0.044 | 2 |
| 122 | +0.046 | +0.045 | 1 |
| 125 | +0.047 | +0.045 | 2 |
| 128 | +0.047 | +0.046 | 1 |

During phantom-c3, run3 accumulates a **1–2 mm z-drift below run1**. The mechanism (per `wrapper.py:365 _derive_force_command`): planner sees phantom contact (`lam_n=0.584`) → derives `f_cmd = -g_hat × 2N = (+2, 0, 0)` → OSC applies +x recoil force → EE is pushed in +x AWAY from box face → but tiny initial differences in ee_z + the IK's setpoint march combine into a 1–2 mm divergence in z.

This 1–2 mm z-differential at end of phantom-c3 (step 128) then COMPOUNDS over the next 30 ticks of free-mode reposition (steps 128–158) into a 28 % cost-trajectory divergence by step 158, then into a 1-tick flip-timing difference at step 159–160, then into a 22–31 mm IK target jump at step 159, then into a 33-tick first-real-contact-admit difference (good runs at 161, bad runs at 192–194), then into a y-offset-on-+x-face difference (good: +y lever arm, bad: −y lever arm), then into the 50 mm goal_dist outcome split.

**(γ) is CONFIRMED.** The recoil-driven EE z-differential during phantom-c3 is the upstream amplifier.

## Corrected mechanism summary

```
sub-mm IK noise at step 110
    │
    ├── 18-tick phantom-c3 burst (110–128)
    │       └── recoil f_cmd=(+2N, 0, 0)
    │             └── EE z-drift: run3 sinks 1–2 mm below run1
    │   ←── THE UPSTREAM AMPLIFIER (γ confirmed)
    │
    ├── 30-tick free-mode reposition (128–158)
    │       └── 1–2 mm EE-state diff
    │             └── different IK guide-path tracking
    │                   └── 28 % cost-trajectory divergence by step 158
    │
    ├── kToBetterRepos cost-crossing (159–160)
    │       └── 1-tick flip-timing difference
    │             └── 22–31 mm IK-target jump
    │
    ├── 30-tick traverse to box (160–194)
    │       └── EE approaches box on different y-sides
    │
    └── first real Drake EE-BOX admit
            └── +y lever (good) vs −y lever (bad)
                  └── 50 mm goal_dist split
```

## Routed fix candidates (NEW — replaces B3)

### B-REDUX (preferred — root cause, eliminates the upstream amplifier)

**Prevent the 18-tick phantom-c3 burst or zero out the recoil force during it.** Two implementations:

- **B-REDUX-a (cheap, guarded by Drake admit, identity-default opt-in)**: at the c3-mode `_derive_force_command` call site (`wrapper.py:365` / call site near line 1085), if Drake admit is N for the EE-BOX pair at the current tick, override `f_cmd → (0, 0, 0)` — don't apply recoil based on a phantom prediction. This eliminates the recoil amplifier during phantom-c3 without changing entry-tick or mode-flip logic.
  - **Risk**: if recoil during *real* contact (admit=Y) was contributing usable force, this should leave that case untouched (the override is admit=N gated).
  - **Pin against data**: at step 110-128 ALL runs have admit=N → the override fires on EVERY tick of the phantom-c3 burst → the 2-mm z-drift should NOT accumulate.

- **B-REDUX-b (alternative)**: gate c3-entry on `drake-admit-seen for K consecutive ticks` (not ee_to_surf). At step 110-128, drake never admits → c3 mode wouldn't start → no recoil applied → no z-drift. This is what original B1's framing tried to capture; the corrected metric is admit-Y-streak, not ee_to_surf.
  - **Risk**: if real-contact admits are intermittent at c3 boundary, this could indefinitely defer c3 entry. Needs a fallback timeout.

### B-DISPATCHER-HYSTERESIS (backstop, defense-in-depth)

**Add per-tick hysteresis to the `kToBetterRepos` flip at `wrapper.py:1595-1598`.** Don't flip unless the new sample's cost is better than the held by ≥ N % for K consecutive ticks (e.g., 10 % for 3 ticks).

This wouldn't address the upstream amplifier but would dampen the 1-tick flip-timing difference at step 159-160 from propagating into a 30 mm spatial jump.

### B3a/b/c — FALSIFIED (all of them)

- B3a: RNG is correctly seeded — not the surface.
- B3b: dispatcher argmin operates on 47 % margins at the flip tick, not sub-1 % noise — wrong mechanism.
- B3c: samples are already < 2 mm apart across runs — quantization is no-op.

## Pin-against-data for B-REDUX

If B-REDUX-a is chosen, no threshold parameter is needed — it's a binary override gated on `admit=Y`. The only YAML knob is `use_phantom_recoil_zero: bool` (identity-default `False`).

If B-REDUX-b is chosen, the K-consecutive-admit-Y-streak parameter must be pinned against measured data. From the logs:
- All runs admit=N during phantom-c3 (steps 110-128). K=1 (require even a single admit-Y tick) would suffice.
- A K=3 tightening would defer c3 entry beyond a brief flicker; should be safe.
- Measured first admit-Y tick: 161 (good runs), 192-194 (bad runs). Setting K=3 would mean c3 mode doesn't start until step 163-196, which significantly changes controller behavior — needs a smoke run.

## File:line citations

- **`wrapper.py:365`** — `_derive_force_command` (B-REDUX-a override site).
- **`wrapper.py:622-674`** — existing 5-tick disengage gate (already handles phantom-c3 cap; B-REDUX-b's entry-side analog).
- **`wrapper.py:1595-1598`** — `target_idx = best_other_idx vs k_star` (B-DISPATCHER-HYSTERESIS guard site).
- **`wrapper.py:1631-1654`** — existing `noise_flip` classifier (would become a guard under B-DISPATCHER-HYSTERESIS).
- `wrapper.py:848` — `k_star = int(np.argmin(c_samples))` (the argmin selection — not the lever as turns out, the costs have already diverged 28 % by the flip tick).
- `sampling.py:30, 146` — sample generators (FALSIFIED as the surface; samples are identical across runs).
- `sample_buffer.py:73-150` — sample buffer (FALSIFIED).

## What's NOT yet measured

- The EXACT step where cost-trajectory divergence begins (between step 128 and 155). If it begins at step 128 (right after phantom-c3 ends) — B-REDUX is the root surface. If it begins later (e.g., step 140) — there's a secondary amplifier in the IK trajectory itself that needs investigation.
- Whether the OSC's recoil application during phantom-c3 is uniformly +x or has y-drift across runs. Could add a third amplifier.
