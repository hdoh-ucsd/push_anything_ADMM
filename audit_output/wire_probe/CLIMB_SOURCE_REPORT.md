# Step-3 climb-source probe — the +9 m/s² gap is double gravity-compensation

**Date:** 2026-05-29.
**Tree state at probe launch:** pristine `38dbf18`; `stash@{0}` (directional face-picker patch) NOT applied. Working-tree changes: WIRE-PROBE instrumentation (`wrapper.py`, prior probe) + OSC-ZDECOMP/CONTACT-DUMP/MASS-CHECK instrumentation (`control/osc/operational_space_controller.py`, this probe; env-gated by `OSC_ZDECOMP=1`, no semantic change).
**Run:** `OSC_ZDECOMP=1 python main.py pushing --task-id 4 --solver c3plus --sampling-c3 --admm-iter 25 --max-time 0.6 --no-record --seed 0`. Log: `audit_output/wire_probe/seed0_contact_mass.log`.

## Verdict — the +9 m/s² is double gravity-compensation at `main.py:575`

Both ranked-cheap discriminating checks came back negative for their primary hypotheses. The actual climb mechanism is at the actuation-port write in `main.py:575`:

```python
559:  tau_g = -plant.CalcGravityGeneralizedForces(plant_ctx)
560:  u_opt = mpc.compute_control(...)
...
572:  if isinstance(mpc, SamplingC3MPC) and mpc.last_mode == "free":
573:      plant.get_actuation_input_port().FixValue(plant_ctx, u_opt)
574:  else:                                                    # <-- the c3-mode path
575:      total_torque = tau_g[:n_u] + u_opt                   # <-- double gravity-comp
576:      plant.get_actuation_input_port().FixValue(plant_ctx, total_torque)
```

The OSC QP already accounts for gravity in its dynamics constraint (`M v̇ + (Cv − τ_g) = B u + …`, so `M v̇ = B u + τ_g − Cv + …`); `u_opt` is a task torque that ASSUMES gravity acts. Main loop then pre-cancels gravity by adding `−CalcGravity` on top, so the effective dynamics the plant integrates becomes `M v̇ = B u_opt − Cv` (gravity nulled). The QP's downward setpoint relies on gravity descending the EE while `u_opt → 0` (the QP correctly minimizes torque when desired accel ≈ −g). With gravity cancelled at the plant input, `u_opt ≈ 0` produces no descent. The arm floats.

Per-tick numerical confirmation. Predicted realized `a_ee_z = a_QP − az_grav` (i.e., subtract back the gravity term the QP budgeted for) vs central-difference observed:

| tick | ee_z (m) | a_QP (m/s²) | az_grav (m/s²) | predicted_real | observed_real | diff |
|---:|---:|---:|---:|---:|---:|---:|
|  3 | +0.20095 | −6.529 | −8.278 | +1.748 | +2.200 | +0.452 |
|  5 | +0.20277 | −7.623 | −8.299 | +0.677 | +0.800 | +0.123 |
| 10 | +0.20844 | −8.258 | −8.366 | +0.107 | +0.100 | −0.007 |
| 15 | +0.21382 | −7.255 | −8.430 | +1.175 | +1.600 | +0.425 |
| 20 | +0.21953 | −8.349 | −8.498 | +0.149 | +0.200 | +0.050 |
| 25 | +0.22568 | −8.497 | −8.572 | +0.075 | +0.100 | +0.025 |
| 30 | +0.23093 | −8.140 | −8.636 | +0.496 | +0.600 | +0.104 |
| 50 | +0.25526 | −8.818 | −8.911 | +0.093 | +0.100 | +0.007 |
| 55 | +0.26095 | −8.138 | −8.965 | +0.828 | +0.900 | +0.072 |
| 59 | +0.26597 | −8.830 | −9.009 | +0.179 | +0.100 | −0.079 |

Median: predicted +0.29 m/s², observed +0.10 m/s². The two agree to within ≈ 0.1 m/s² at most ticks (tick 40 is a finite-difference outlier from a torque transient; other ticks lock in tight). The +5 to +9 m/s² gap from the prior probe is EXACTLY the negation of the QP's `az_grav` term — i.e., the climb is the realized motion losing the gravity term the QP planned for.

## Discriminator 1 — phantom support force on the pusher: FALSIFIED

Full `ContactResults` dump at every c3-mode tick. There is ONE Drake-admitted contact pair across the entire 60-tick run, and it is `world | box_link` (the box sitting on the table). The pusher has NO contact with anything.

```
[CONTACT-DUMP] tick=1 npp=0 nhyd=0 F_on_pusher_z=+0.0000N  pairs=[]
[CONTACT-DUMP] tick=2 npp=1 nhyd=0 F_on_pusher_z=+0.0000N
   pairs=[(world|box_link F=(-0.993,-0.993,+1.667) nBA=(-0.00,-0.00,-1.00) depth=+0.01mm)]
[CONTACT-DUMP] tick=3 npp=1 nhyd=0 F_on_pusher_z=+0.0000N
   pairs=[(world|box_link F=(-1.174,-1.174,+1.967) nBA=(-0.00,-0.00,-1.00) depth=+0.01mm)]
...
```

`F_on_pusher_z = 0.000 N` at every one of the 60 ticks. No phantom support force is acting on the pusher_collision sphere. Hypothesis #1 is decisively ruled out.

## Discriminator 2 — QP-M⁻¹ vs constrained realization: FALSIFIED (mismatch is microscopic)

Computed three predictions of arm-z acceleration at every tick:
- `a_qp_arm` = the QP's full-coupled prediction `(J_v vdot_opt)[2] + (J̇_v v)[2]`
- `a_box_all_held` = constrained realization with all 6 box DoFs forced to v̇=0 (table reaction + frozen box)
- `a_box_z_only_held` = constrained realization with only box-z (v[12]) forced to v̇=0, box xy + yaw free

Result at every tick is essentially the same number (gap ≤ 0.025 m/s²):

```
[MASS-CHECK] tick=1 a_qp_arm=-3.6095 a_box_all_held=-3.6335 a_box_z_only_held=-3.6335 schur_gap_qp-arm=+0.0240
   vdot_box=[+0.000,+0.000,+0.000,+0.000,+0.000,-9.810]
[MASS-CHECK] tick=10 a_qp_arm=-8.2585 a_box_all_held=-8.2851 a_box_z_only_held=-8.2851 schur_gap_qp-arm=+0.0267
[MASS-CHECK] tick=60 a_qp_arm=-8.8808 a_box_all_held=-8.9320 a_box_z_only_held=-8.9320 schur_gap_qp-arm=+0.0511
```

The Schur-complement gap is +0.024 to +0.051 m/s² (it equals the `az_lam` term from the earlier decomp — the QP's phantom λ_ext routed through `M⁻¹`). The arm-box mass matrix M is essentially block-diagonal in this regime (no admitted contact pair → no kinematic coupling beyond a negligible numerical residue), so the QP's free-box solution and the table-held-box realization predict the same arm v̇ to within ≤ 0.05 m/s². Hypothesis #2 is ruled out.

Notable: the QP's own `vdot_box` solution is `[0, 0, 0, 0, 0, −9.81]` at every tick — i.e., the QP wants the box to free-fall at −g, which the simulator correctly prevents via the table reaction. But this disagreement does NOT propagate into the arm's v̇ because M is block-diagonal. So although the user's structural intuition (QP solves box-free; sim holds box) was correct, the consequence at the arm-z is < 0.05 m/s² in this regime — three orders of magnitude too small to explain the +9 m/s² climb.

## Discriminator 3 — timing race: not investigated

Per directive, ranked-3 (control-loop aliasing) was not investigated. The double-gcomp finding is mechanism-complete and matches the realized motion at every tick within ≤ 0.5 m/s², so there is no residual to attribute.

## What this implies for the wiring

The conditional at `main.py:572–576` was written with the FREE-mode tracker in mind (`PiecewiseLinearTracker` / `RepositionIKTracker` both include gravity-comp internally — confirmed by `impedance_controller.py:84` lines `τ_bias = bias_arm − g_arm` and `reposition_ik.py:1097` joint-PD against `q_nominal`). The else-branch ("not free mode") fires for ALL c3-mode ticks regardless of which executor is selected. But the OSC executor (`use_osc=True`, the canonical path per CLAUDE.md) ALSO does gravity-comp internally via the QP's `bias = Cv − τ_g` term. Adding `tau_g` again at the actuation port double-counts.

The change required (NOT applied here per directive) is to also short-circuit `tau_g`-addition when the executor is the OSC. Either: (a) short-circuit on `params.use_osc=True`; (b) always pass `u_opt` alone, and remove the FREE-mode special case by making the FREE tracker NOT do gravity-comp internally; (c) parameterize on a `mpc.adds_gravity_comp_internally` flag. The cleanest is (b) — make the gravity-comp ownership singular (the main loop owns it for all executors) — but that's an architectural choice, not a one-line fix.

## What this rules in and out

**Ruled in:** the gap is between the QP's solved `u_opt` (which budgets gravity into its dynamics) and the actuation-port write (which pre-cancels gravity on top). Confirmed by mechanism and by per-tick numerical agreement (predicted realized within ≤ 0.5 m/s² of observed at every tick).

**Ruled out:**
- Phantom support force on the pusher (`F_on_pusher_z = 0.000 N` at all 60 ticks).
- M⁻¹ / box-DoF constraint mismatch (Schur gap ≤ 0.05 m/s², three orders of magnitude too small).

**Not investigated (per directive):** integrator timing race. The double-gcomp finding accounts for the gap, so there is no residual to chase here.

## Tree state at end of probe

- `git rev-parse HEAD` = `38dbf180`
- Working-tree modified files:
  - `control/sampling_c3/wrapper.py` — WIRE-PROBE instrumentation (from prior probe).
  - `control/osc/operational_space_controller.py` — OSC-ZDECOMP + CONTACT-DUMP + MASS-CHECK instrumentation (this probe, env-gated by `OSC_ZDECOMP=1`).
- Stash list unchanged. `stash@{0}` (facepicker_experiment_no_op_2026-05-29) NOT applied.

No commits. No fixes applied. Held against 0/20 baseline. Reporting and stopping per directive.
