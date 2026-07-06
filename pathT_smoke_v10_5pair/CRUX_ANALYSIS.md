# pathT_smoke_v10 — Faithful 5-pair cost-LCS (2 EE-T + 3 T-GND) — CRUX RESULT

**Date:** 2026-07-05
**Baseline:** pathT_smoke_v8 — cost-LCS ranking active but `n_ee_top_k=2` silently no-op'd because the always-on injection gate was OFF.
**This change:** `force_top_k_ee_box=True` in `inner_solve.py` cost-LCS build → `extract_lcs_contacts` unconditionally replaces auto-admitted EE-manipuland pairs with top-K by phi. Cost-LCS now truly has 2 EE-T rows (confirmed via new `[COST-LCS]` trace: `n_ee_t=2 ee_t_phi=[+0.017,+0.023]` etc).

## Pre-registered check

The user's directive:
> 5-pair distinguishes faces (east < north) + EE steers east + T pushes → the cost-LCS WAS the crux; the T reproduces the reference.
> 5-pair still doesn't distinguish (east ≈ north) → the cost-LCS is NOT the crux → the cost-rollout gains (soft press) or the planner. The 5-pair is STILL banked as a real fidelity fix; isolate the remaining 2.

## The three outcome tests

### Test 1 — does 5-pair make the ranking distinguish productive faces (east < north)?

**NO.** East loses by **12 %** (larger gap than v8's 6 %).

Evidence — GS-table step=100:
```
k=0 (current   ) pos=(-0.003,+0.059,+0.073)  c_C3=117,894 align=0.0000
k=1 (prev_repos) pos=(-0.010,+0.050,+0.034)  c_C3=109,270 align=0.0000  ← WIN
k=2 (strat_0   ) pos=(+0.150,+0.003,+0.034)  c_C3=122,268 align=0.9710  ← EAST, productive
```

**Per-sample [COST-LCS] trace across ~35 mpc solves (east strat_0 vs prev_repos north):**

East sample at (+0.150, +0.003, +0.034), align=0.97:
- n_ee_t = 2 (both bars admitted) ✓
- ee_t_phi ≈ [+0.017, +0.026] (contact rows in cost-LCS ARE active for east)
- Forward-sim dT_xy = **0.015–0.018 m** (T actually moves ~2× more than north sample's)
- box_v_peak = 0.03–0.15 m/s
- **c_C3_sim = 121,000 – 124,000** (consistently HIGHER)

North sample at (-0.010, +0.050, +0.034), align=0.00:
- n_ee_t = 2 (same auto-admit set)
- ee_t_phi = [+0.017, +0.023] (same phi set — cost-LCS is shared)
- Forward-sim dT_xy = **0.008–0.009 m** (less T motion)
- box_v_peak = 0.16–0.20 m/s
- **c_C3_sim = 108,000 – 113,000** (consistently LOWER — WIN)

**Interpretation:** East's forward-sim actually reproduces the productive-face push (2× more T linear motion, real λ_n activated in the LCP). But the object-only cost (Q_obj weighting `w_obj_xy=100k` and `w_yaw=800`) scores the resulting T trajectory as WORSE than north's near-zero-motion baseline. The east push may be rotating the T yaw or translating it in a direction that increases distance to the +45° yaw + (-0.2,+0.05) xy target more than doing nothing does.

### Test 2 — does the EE steer east + dispatch c3 there?

**NO.** The wrapper NEVER dispatches to c3 in v10.

Evidence:
- `grep -c 'mode=c3' pathT_smoke_v10_5pair/run.log` → **0**
- At t=1.5s (step=150), still `mode=free switch=kStayInRepos won_src=prev_repos`
- v8 baseline entered c3 at step=94 (t=0.94s)

Because east's c_C3_sim consistently exceeds prev_repos's c_C3_sim by 10–15 k units, the dispatcher never fires `kToC3ReachedReposTarget` or `kToC3Cost`. The 5-pair cost-LCS's inverted ranking has actually **regressed dispatcher behavior** compared to v8.

### Test 3 — does the T push?

**NO.** T stays at (0.001, 0.006, 0.016) at t=1.5s (initial (0, 0, 0.020)); `goal_dist=0.206 m` unchanged from initial 0.206 m through step 150.

## Root cause

The 5-pair cost-LCS mechanism works exactly as specified — `[COST-LCS] n_ee_t=2 ee_t_phi=[+0.017,+0.026]` proves the top-2 EE-T admission fires unconditionally at every sample. The forward-sim resolves the LCP with those 2 rows and produces real dT_xy (0.018 m for east vs 0.009 m for north — the sim DOES capture that east presses more effectively than north).

**But `c_C3_sim` inverts the ranking** because:

1. **Cost-LCS is linearized at CURRENT plant context**, not at each sample. All samples in a given tick see the SAME `A, B, D, E, F, H, phi, J_n, J_t`; only `x0[7:10]=p_ee_sample` differs. Confirmed in the log: all three samples at any given step share `ee_t_phi=[+0.017,+0.023]`.

2. **Linearization is anchored to the current EE position** (typically near the stem-north side, since prev_repos target keeps arm there). So `J_n_ee` and `phi` encode stem-side contact geometry, and the LCP extrapolates that geometry ~15 cm to reach the east sample's `p_ee`. The forward-sim's contact model is a large linear extrapolation from the current arm config, not a locally-consistent linearization at the east sample.

3. **The object-only cost** `Q_obj` (`w_obj_xy=100k`, `w_yaw=800`) evaluates the resulting T trajectory. If east's sim pushes T in a direction that increases `||box_xy - goal_xy||` or worsens yaw tracking (moderate rotation opposed to +45° goal_yaw), even though `dT_xy` is larger, the scored cost is higher. North's near-zero motion sim scores as "T stays close to reference" better than east's actual-motion-in-wrong-direction sim.

## Pre-registered verdict, actioned

> **5-pair still doesn't distinguish → the crux is the gains/planner (the 5-pair still banked).**

**Bank the 5-pair as a real fidelity fix**:
- `control/lcs_formulator.py::extract_lcs_contacts` — added `force_top_k_ee_box: bool = False` param; when True, unconditionally replaces auto-admitted EE-manipuland pairs with top-K closest candidates (reference `GetResolvedContactPairs` semantics).
- `control/lcs_formulator.py::linearize_discrete_ee_space` — forwards `force_top_k_ee_box` to `extract_lcs_contacts`.
- `control/sampling_c3/inner_solve.py` — cost-LCS build calls `linearize_discrete_ee_space(..., n_ee_top_k=2, force_top_k_ee_box=True)`.
- `main.py` — boot-time `[GS] cost-LCS ranking: ... force_top_k_ee_box=True ...` visibility line.
- `PUSHA_COST_LCS_TRACE=1` — new instrumentation env-flag that emits per-sample `[COST-LCS]` lines with sim-side motion (dT_xy, box_v_peak) + cost-LCS admission (n_ee_t, ee_t_phi).

**Isolate the remaining 2 candidates for the actual crux:**

1. **Per-sample plant-context update before cost-LCS linearization** (the reference does this via `UpdateContext(plant, current_v, candidate_states[i])` before each `LCSFactory(...).GenerateLCS()` call — `sampling_based_c3_controller.cc:1628-1631`). The port skips this for EE-space samples (`inner_solve.py:371-378`, noted as a known Delta-1 gap). Without it, the top-K EE-T rows are anchored to the WRONG contact geometry when the sample is 15 cm away from where the LCS was linearized.

2. **Cost-rollout PD gains** — `Kp_for_ee_pd_rollout=100, Kd_for_ee_pd_rollout=0.5` (reference values from `push_t/parameters/sampling_c3plus_options.yaml`). If the port's forward-sim behaves stiffer than the reference (`W_track=1` in the c3-mode gains bundle changes what the plan looks like), the sim's PD tracking of the planner path may not represent what the OSC would actually produce.

## Deferred: μ=1.0 real-sustained-push validation

Not banked — per user directive, the μ=1.0 real sustained push is only banked after a real T push is reproduced. This run showed **no T push at all** (goal_dist stayed 0.206 m through t=1.5s), so μ=1.0 remains deferred.

## Final result

```
[RESULT] method=sampling-c3  final_obj_xy=(0.0048, 0.0210)  goal_dist=0.2068m  success=NO
[GS-perf] avg_per_step_ms=2321.3  full_solves=801  cheap_solves=1601  switches=0
```

- Initial goal_dist: 0.206 m
- Final goal_dist: 0.207 m (slightly WORSE — T drifted 21 mm in +y direction, i.e., NORTH, away from goal)
- Mode switches: **0** (v8 baseline had at least the first switch to c3 at t=0.94s)
- T did not move toward the goal at any point in 8 s

## Artifacts

- `pathT_smoke_v10_5pair/run.log` — full 8 s sim log, 801 [STEP] lines, 2402 [COST-LCS] traces
- `pathT_smoke_v8/run.log` — baseline (5-pair silently no-op) for direct comparison
- Code diff: `git diff HEAD -- control/lcs_formulator.py control/sampling_c3/inner_solve.py main.py`
