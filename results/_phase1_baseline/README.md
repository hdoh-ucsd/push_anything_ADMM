# Phase 1 — Cartesian-force OSC executor (reproduce-dairlib scope §3)

**Branch:** `reproduce-dairlib` **Close-out HEAD:** `465016f` (Task 8 tip)
**Scope doc:** `docs/superpowers/plans/2026-07-13-reproduce-dairlib-scope.md` §3 Phase 1
**Plan doc:** `docs/superpowers/plans/2026-07-13-reproduce-dairlib-phase1-cartesian-osc.md`

Phase 1 = the go/no-go phase. It swaps the port's ℝ⁷-torque impedance-shaped executor for a Cartesian-force OSC structurally equivalent to dairlib's `franka_osc_controller.cc`. Validates against Phase 0 (`de14138`) baselines.

---

## Verdict: GO — with margin

Box seed=0 go/no-go gate passes all three bars simultaneously, with material improvement over Phase 0.

| Metric | Bar | Phase 0 | Phase 1 | Verdict |
|---|---|---|---|---|
| `goal_dist` (box) | ≤ 0.135 m | 0.1289 m | **0.0541 m** | ✅ PASS (60% margin) |
| Closure of 0.3 m init | — | 57% | **82%** | +25 pp |
| `qp_failures` | == 0 | 0 | **0** | ✅ PASS |
| OSC saturation | ≤ 8% | 6.16% | **0.17%** | ✅ PASS (36× reduction) |
| Overshoot pattern | — | −0.12 m past | **−0.05 m short** | Different failure mode |
| `orient_err` (box) | not gated | 1.68 rad (96°) | 1.63 rad (93°) | ~same tumble (Phase 3) |
| `orient_err` (T) | not gated | — (partial) | 0.72 rad (41°) | Recorded |
| T translation | not gated | none (partial) | **none** (full 8s) | Expected — Phase 2 target |

Per scope §3: "if the Cartesian-force OSC cannot reproduce the box push (a known-achievable target), the architecture path is not converging as expected — STOP and reassess." The reverse held here: the OSC swap *improved* the box push. **The architecture path is converging.**

---

## What Phase 1 landed (six code commits + two baseline commits)

Commits on `reproduce-dairlib` between `de14138` (Phase 0 close-out) and `465016f`:

1. `33d8208` — feat(osc): joint-2 posture cost term
2. `a8e77c6` — feat(osc): plumb q_arm/v_arm through compute_torque
3. `91cc587` — config(osc): joint-2 gains in osc_franka.yaml
4. `18498c1` — feat(osc): reference c3-gains default-on, opt-out flag
5. `491ad1a` — feat(osc): trajectory-shaped input interface (Phase-2 prep)
6. `aa42789` — feat(wrapper): dispatch c3-mode OSC through trajectory interface
7. `524ecf0` — feat(phase1): box seed=0 gate PASS (this baseline artifact)
8. `465016f` — feat(phase1): T seed=0 diagnostic

Changed subsystems:
- `control/osc/qp_builder.py` — `OscGains` extended with joint-2 fields; new Cost 6 term.
- `control/osc/operational_space_controller.py` — reference c3-gains default; new `compute_torque_from_trajectory` method; `q_arm`/`v_arm` plumbed.
- `config/osc_franka.yaml` — joint-2 gains configured (Kp=200/Kd=10/W=1/target=1.1).
- `control/sampling_c3/sampling_based_c3_controller.py` — c3-mode dispatch through the trajectory interface.

Interface change:
- `PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1` is deprecated (inert with warning). New opt-out: `PUSHA_OSC_C3_MODE_LEGACY_GAINS=1`.

Test coverage:
- `tests/test_osc_joint2_posture.py` — 2 tests (pull-toward-target + weight-zero-inert).
- `tests/test_osc_default_gains.py` — 3 tests (YAML defaults, ref c3 default, legacy opt-out).
- `tests/test_osc_trajectory_interface.py` — 2 tests (delegation + all-optional-args forwarding).
- Total 18 OSC unit tests + 33 other tests pass. Pre-existing 8 `test_progress` + 2 `test_osc_unit` failures unrelated (verified via `git stash` retest at `de14138`).

---

## Mechanism read

**Why the box improved:** The port's pre-Phase-1 c3-mode executor used Kp=400/W_track=100 → compound position authority of 40 000 vs the reference's 1×200 = 200. That's a 200× over-drive at any nonzero position error, causing the OSC to *hammer* the arm past the goal (12 cm overshoot) with 6% joint-torque saturation. Post-Phase-1 (Kp=200, W_track=1, joint-2 posture, trajectory-shaped input), the compound authority matches the reference and the arm settles short of the goal (5 cm undershoot) with 0.17% saturation. Fundamentally different failure regime.

**Why the box didn't reach ref_gate:** Orientation error (~93°) is unchanged — the box still tumbles during descent. This is the alignment-project characterization ("box tumbles under the reference's own controller"), attributable to the LCS↔Drake contact-model interaction (scope §4 named risk), not the executor.

**Why T didn't translate:** Reposition + admission subsystems are still the port's. The T's transient contact pattern (Drake ee_box_normal peaks at 335 N briefly, then dissolves) is downstream of the executor and awaits Phase 2 + Phase 3.

---

## What Phase 2 can now assume

- The executor is reference-shaped and doesn't over-drive. Phase 2's full-PWL reposition can hand a proper trajectory into a QP that will follow it without pathological saturation.
- The `compute_torque_from_trajectory` interface signature is stable — Phase 2 flips the caller side from a single-knot ZOH PP to the full N-knot PWL from `reposition.cc` without changing the executor.
- Free-mode gains (Kp=400, W_track=100) still handle repos + IK-tracker; the §7.47 IK→c3 handoff mechanism is untouched.
- Joint-2 posture (target=1.1 rad, Kp=200, Kd=10, W=1) is active in c3 mode. Extend to free mode too if Phase 2 shows benefit.

## What Phase 1 does *not* address (deferred by scope, not by oversight)

- Full-PWL trajectory (Phase 2).
- LCS↔Drake contact-model interaction (Phase 3 — the box tumble likely lives here).
- Friction cone on `λ_ext` — reference constrains `lambda_c_` (contact-constraint force), not `lambda_e_`; no cone needed on `λ_ext`.
- `W_input_smoothing` — reference `w_input_reg = 0` makes the smoothing weight vanish; no cost term needed.
- Multi-seed sweep (Phase 4).

---

## Artifacts

- `results/_phase1_baseline/box_seed0/` — box gate verdict, RESULT, OSC-SUMMARY, manifest.
- `results/_phase1_baseline/t_seed0/` — T diagnostic RESULT, OSC-SUMMARY, log tail.
- `scripts/_phase1_box_seed0.sh` — canonical Phase-1 box invocation.
- `scripts/_phase1_t_seed0.sh` — canonical Phase-1 T invocation.

Raw run.log files (~2.8 MB total) intentionally uncommitted — regenerable from HEAD.
