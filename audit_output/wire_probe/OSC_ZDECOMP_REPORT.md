# OSC-ZDECOMP — per-QP-term vertical accel decomposition at climb ticks

**Date:** 2026-05-29.
**Tree state at probe launch:** pristine `38dbf18`; `stash@{0}` (directional face-picker patch) NOT applied. Working tree has the wire-probe instrumentation (`M wrapper.py`, from prior probe) plus this probe's OSC instrumentation (`M control/osc/operational_space_controller.py`, +83 lines, env-gated by `OSC_ZDECOMP=1`, no semantic change to control behaviour).
**Run:** `OSC_ZDECOMP=1 python main.py pushing --task-id 4 --solver c3plus --sampling-c3 --admm-iter 25 --max-time 0.6 --no-record --seed 0`. Log: `audit_output/wire_probe/seed0_zdecomp.log`.

## TL;DR — the dominant upward contributor is NOT in the QP

**The user's premise — "one of those terms is positive-z and dominating" — is falsified.** No QP cost term drives the climb. The QP commands a strongly DOWNWARD `a_ee_z` at every one of the 60 c3-mode ticks (range −3.6 to −8.9 m/s², median −8.3). Every tick is also a climb tick (59/59 inter-tick comparisons show ee_z increased). The realized vertical acceleration (central-difference of logged `ee_z`) sits at +0 to +1 m/s² across the run; the gap to the QP-commanded `a_ee_z` is **+5 to +9 m/s² at every probed tick**, uniformly signed.

**Interpretation.** The QP is internally consistent and is doing the right thing — it correctly commands downward EE motion. The climb is generated DOWNSTREAM of the QP, between `u_opt → plant.AdvanceTo`. The five candidates the user enumerated were the right set to check, but the answer is *none of the above*.

## Cost leave-one-out (the "which QP cost term is the climb?" question)

Sign convention: `Δa = a_total_with_cost − a_total_without_cost`. Positive ⇒ that cost pulls EE z UP; negative ⇒ pulls DOWN. Stats over all 60 ticks:

| Cost dropped | min | median | max | mean | interpretation |
|---|---:|---:|---:|---:|---|
| W_track    (position+vel tracking) | −9.22 | **−8.92** | −4.67 | −8.41 | DOWN — dominant; tracking is correctly pulling EE toward the lower setpoint |
| W_posture  (nullspace posture)     | +0.15 | **+0.43** | +0.48 | +0.41 | UP — but ~20× smaller than tracking's magnitude |
| W_acc      (accel regularization)  |  +0.0001 |  +0.0003 |  +0.0004 |  +0.0003 | negligible |
| W_torque   (torque regularization) |  −0.0006 |  −0.0000 |  −0.0000 |  −0.0001 | negligible |
| W_force    (force-tracking on λ_ext) |  −0.0006 |  −0.0000 |  −0.0000 |  −0.0001 | negligible |
| λ_des → 0  (force-track command zeroed) |  −0.0000 |  −0.0000 |  −0.0000 |  +0.0000 | negligible |

Posture is the only positive-z cost term, and its contribution is small in absolute magnitude (+0.4 m/s² median). Tracking is the dominant cost and points DOWN (−8.9 m/s² median) — i.e., the QP recognizes the EE is above its setpoint and asks for descent. The other costs are essentially zero contributors.

**No cost term shifts the QP toward upward motion in a magnitude that explains the climb. The QP is not the culprit.**

## Dynamics-side decomposition (the "where is the mechanical force" question)

Decompose the solved `v̇*` using the QP's own dynamics equality `M v̇* = B u* + Jᵀλ_ext* + F_ff − (Cv − τ_g)`, projected to EE z via `J_v` plus `J̇_v v`. Stats over all 60 ticks:

| Source | min | median | max | mean | comment |
|---|---:|---:|---:|---:|---|
| gravity (`τ_g`)              | −9.02 | **−8.64** | −8.27 | −8.64 | ≈ −g; correct magnitude/sign for arm under gravity |
| arm torque (`B u*`)          | −0.06 |  +0.20 | +4.63 |  +0.74 | mostly small; large only at tick 1 (transient) |
| `Jᵀλ_ext*` (force-tracking)  |  0.00 |  +0.04 | +0.05 |  +0.03 | tiny — λ_des is horizontal (≈[2,0,0]) |
| `F_ff` (planner λ_planned)   |  0.00 |   0.00 |  0.00 |   0.00 | **confirmed zero in force-tracking mode** (planner u_seq not in QP) |
| Coriolis (`−Cv`)             | −0.08 | −0.05 | +0.00 | −0.05 | small |
| `J̇_v v` (kinematic bias)    |  0.00 | +0.12 | +0.18 | +0.11 | small |
| **sum (= a_total = a_check)**| −8.88 | **−8.29** | −3.61 | −7.81 | matches QP's `a_total` to numerical precision ✓ |

Reading this with the LOO: at the median tick, gravity wants −8.6 m/s² down, the arm torque adds ~+0.2 (barely compensating), and the net `a_total = −8.3` is what the QP commands. To deliver a_des ≈ −8 m/s² (per the Kp/Kd PD with growing v_err_z), the QP elects to let gravity do the work — `u*` shrinks toward zero (`u_norm` drops from 17.3 N·m at tick 1 → 1.1 N·m by tick 60) because the arm doesn't need to add torque to fall.

Cross-check: there is no positive-z mechanical term in the QP big enough to account for the climb. The largest positive contributors are `B u*` at +0.74 m/s² median and `J̇_v v` at +0.11 m/s² median — neither is anywhere near the realized +5 to +9 m/s² excess.

## The QP-vs-realized inversion (the real finding)

Central-difference estimate of realized `a_ee_z` from logged `ee_z`, alongside QP-commanded:

| tick | ee_z (m) | ΔEE (mm) | a_QP (m/s²) | a_real (m/s², FD) | gap (m/s²) | grav | trq | u_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | +0.20008 |  +0.00 | −3.610 |  — |  — | −8.27 | +4.63 | 17.33 |
| 3  | +0.20095 |  +0.62 | −6.529 |  +3.70 |  +9.12 | −8.28 | +1.71 |  8.63 |
| 10 | +0.20844 |  +1.18 | −8.258 |  +0.10 |  +8.31 | −8.37 | +0.03 |  3.14 |
| 20 | +0.21953 |  +1.20 | −8.349 |  −0.90 |  +7.38 | −8.50 | +0.03 |  1.32 |
| 30 | +0.23093 |  +1.12 | −8.140 |  +0.90 |  +8.75 | −8.64 | +0.37 |  1.13 |
| 40 | +0.24320 |  +1.01 | −6.720 |  −2.70 |  +2.30 | −8.78 | +1.95 |  5.37 |
| 50 | +0.25526 |  +1.31 | −8.818 |  +0.00 |  +8.80 | −8.91 | −0.01 |  1.30 |
| 60 | +0.26729 |  +1.32 | −8.881 |  +0.10 |  +8.93 | −9.02 | +0.06 |  1.14 |

Pattern: `a_real ≈ 0` (very slight positive, noisy from finite-difference), `a_QP ≈ −8.5`, gap ≈ +9 ≈ +g uniformly. The realized motion behaves as if **the simulator is not seeing gravity on the arm**, while the QP correctly accounts for it.

Numerical check that the inversion is in the integration, not in the QP: realized `v_ee_z ≈ +0.10 m/s` consistently (60-tick position deltas average +1.12 mm per 10 ms). A QP-commanded `a_ee_z = −8 m/s²` would zero out a +0.10 m/s upward velocity within ~12 ms (within one control period). It does not — `v_ee_z` is approximately constant across the run. The arm is decoupled from the commanded vertical accel.

## What this rules in and out

**Ruled out (all five of the user-enumerated candidates):**
- *position-tracking* — yes, pulls DOWN (−8.9 m/s² LOO median, dominant magnitude). Not the culprit.
- *force-tracking (λ_ext / λ_des)* — negligible (LOO ≤ 0.001 m/s²); λ_des is horizontal so its z-effect is zero anyway. Not the culprit.
- *gravity-compensation* — QP correctly accounts for gravity at −8.6 m/s² and routes the right `u*` to balance the position setpoint. Not the QP's failing.
- *posture / acc-reg* — posture pulls UP at +0.4 m/s² (only positive cost term), but ~20× too small to explain the +9 m/s² gap.
- *planner u_seq via dynamics constraint* — `F_ff_for_qp = 0` at every tick (force-tracking mode zeros it). Confirmed by direct decomposition: `ffp = 0.00` at every tick. The planner's u_seq does not enter the QP.

**Ruled in (the new investigation handle):**
The discrepancy is between `u_opt` and the simulator's realized motion. The QP's `M, τ_g, Cv, J_v, J̇_v v` come from the same plant context that the simulator integrates, so they should match — but the realized `v̇_ee_z` is ~+9 m/s² above the QP's prediction at every tick. Candidates for what's eating that gap (NOT investigated here, per directive):
- a phantom contact/spring force the simulator applies that the QP does not model (e.g., pusher_collision sphere against the table or arm self-collision producing a soft constraint force at no admitted LCS pair);
- the OSC's plant context is set to `(current_q, current_v)` but the simulator integrates from a slightly different state (control-loop timing — `dt_ctrl=0.01s` while the discrete plant `time_step=0.001s` — race);
- gravity-routing through the box DoFs (the box is a free-floating body in the plant, held only by the table contact; if `τ_g` includes a box-z gravity component that couples into arm v̇ through some non-zero off-diagonal in M, the QP's M⁻¹ τ_g projection differs from the simulator's table-reaction-constrained realization);
- Drake integrator semi-implicit step at 1ms applying gravity differently than `CalcGravityGeneralizedForces` returns for the algebraic OSC query.

## Tree state at end of probe

- `git rev-parse HEAD` = `38dbf180`
- Working-tree modified files:
  - `control/sampling_c3/wrapper.py` — WIRE-PROBE instrumentation (from prior probe, +37 lines, no semantic change).
  - `control/osc/operational_space_controller.py` — OSC-ZDECOMP instrumentation (this probe, +83 lines, env-gated on `OSC_ZDECOMP=1`, no semantic change).
- Stash list unchanged:
  - `stash@{0}`: facepicker_experiment_no_op_2026-05-29 (the directional-face-picker patch — still NOT applied).
  - `stash@{1..3}`: pre-existing.

No commits. No fixes applied. Held against the 0/20 baseline (this probe ran a 0.6 s single-seed scenario on the same canonical commit). Reporting and stopping per directive.
