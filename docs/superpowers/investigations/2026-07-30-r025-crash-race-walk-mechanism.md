# 2026-07-30 — r<0.25 crash race: full mechanism of the diag_walk_probe failure

**Run analyzed:** `results/diag_walk_probe_60s.{txt,launcher.log}` (HEAD=3d403bf,
push_t, c3plus, admm-iter 3, crashed step 278 / t=27.7 s).
**Context:** 5th consecutive run killed by the radial workspace guard since
`fdc7e41`; zero full-length runs on the current stack. Prior session launched
this probe (`DIAG_FORCE_ROUTE_TRACE=1 DIAG_SETPOINT_TRACE=1`) and the WSL VM
restarted (18:58:21) before analysis — the EIO the user saw was the dead-VM
channel, not a disk or code failure.

## Verdicts

1. **The crash guard is reference-conformant and stays.** Reference
   `CheckForWorkspaceLimitViolations` (sampling_based_c3_controller.cc:1476-1494)
   hard-DRAKE_DEMANDs the **live EE** per tick against the axis limits AND the
   radial shell `robot_radius_limits=[0.25, 0.75]` (push_t yaml), with **no
   margin**. Port semantics + values match.

2. **One clean conformance defect found and fixed (95be21f):** reference
   `IsSampleInWorkspace` (generate_samples.cc:760-775) insets samples by
   `workspace_margins=0.02` on x/y and the radial shell (z has no margin);
   the port's `is_in_workspace` applied a 1 mm tolerance *outward*. The
   reference therefore keeps every commanded sample target ≥2 cm clear of the
   abort boundary; the port didn't. **Not the trigger of this crash** (the EE
   wasn't tracking a boundary sample) but it is the reference's designed
   defense layer.

## The walk mechanism (what actually kills runs)

Timeline from the probe:

- t=0-5 s: free-mode approach; c3 entry ~t=5 s. **Box never moves the entire
  run** (`goal_dist=0.150` constant; `[DRAKE-CONTACT] ee_box_normal=0.000` on
  every sampled step — zero real contact force, ever).
- t=5-17 s: c3 stint #1, EE hovers near box (r≈0.5), phantom pushes.
- t=18-20 s: free interlude (progress/counter state resets on mode exit).
- t=20-27.7 s: c3 stint #2. EE **walks monotonically base-ward at ~1.8 cm/s**
  (x: 0.386 → 0.247 over 7.7 s) and crosses r=0.25 → RuntimeError.

Per-tick state during the walk (step 277 exemplar):

- Planner: λ_n≈25 N with EE **16 cm** from the box (`phi=0.16355`), ADMM badly
  non-converged (`pr≈23`, `iters=3/3`). The EE-BOX LCS row is hard-wired
  always-on (`lcs_formulator.py:106`), so phantom λ at distance is possible
  whenever ADMM hasn't converged.
- Force route: `env=u_sol` → `λ_des = u_seq0 ≈ (+4.7,+13.1,-1.3)` N (14 N,
  goal-ward) → OSC `lam_ext = lam_des` exactly.
- Executor: `util_max=0.016` — **1.6 % torque utilization** despite the
  printed setpoint being 26 cm away. The QP's dynamics equality
  `M v̇ = B u + Jᵀ λ_ext − bias` lets the fictitious λ_ext "explain" task
  acceleration, so it commands near-zero torque; with no real contact to
  react against, the arm drifts under the unmodeled residual. Direction of
  drift ≈ −λ_des (base-ward) — matches.

  **CORRECTION (2026-07-31, p145 re-derivation):** the λ_ext-in-dynamics
  claim above is WRONG at this HEAD. `qp_builder.py:160-165` keeps
  `A_eq = [M, −B]` in force-tracking mode — λ_ext is a cost-only shadow
  (matching reference `UpdateDynamics(x, active, {})`, conformance-map 1.f
  Tier-3), so it cannot explain away task accel. The true walk carrier is
  the **plan-following treadmill**: phantom λ recoil on the LCS EE row
  (`m_ee·v̇_ee = u + J_n_eeᵀλ`, west-face normal → −x) makes `x_seq[1]`
  step base-ward ~1.4 mm/tick; the OSC tracks it faithfully
  (p145 `[IMP]` |x_err| = 0.6–3 mm, τ_out < 1 Nm — the "26 cm setpoint" is
  NOT what the tracking cost sees). Consequence for the lever menu: the
  contact-loss gate and PORT_C3_PHANTOM_TRAJ_OVERRIDE directly break the
  treadmill; force-route-side changes would do nothing. Entry geometry
  also re-verified conformant: the fatal p145 sample sat 0.040 m off the
  T's west crossbar face ≈ the reference 0.0395 standoff.

## Why no rescue arrives (all three nets accounted for)

1. **Contact-loss disengage gate**: streak DOES increment on physical
   distance (φ>0.04 → `_no_ee_box_streak+=1`, controller :2553-2556), but the
   gate is **skipped by default** — `PORT_DISABLE_CONTACT_LOSS_GATE` defaults
   `"1"` (§7.55 reference-faithful; the reference has no such gate). One-shot
   `[§7.55]` log confirmed in the run.
2. **kConfigCostDrop progress timeout**: port metric feed is conformant
   (object-only pose cost — flat when the box doesn't move → would correctly
   flag no-progress). BUT reference cc:2265 enforces **only when the history
   window is full**, and the window resets on every mode switch (cc:2313).
   Port window = 13.5 s (180 ref loops × 0.075 s/loop mapping) = 135 c3
   ticks. The fatal stint lasted **77 ticks** — grace never expired. The walk
   covers r=0.39→0.25 in ~7.7 s < 13.5 s, so the crash always wins the race.
   The window-full grace is itself reference-conformant; the 0.075 s/loop
   mapping is an interpretation (reference loop is LCM-driven by the 1 kHz
   franka state channel, throttled by solve wall-time — actual ref cadence is
   hardware-dependent and not pinnable from yaml).
3. **`num_control_loops_to_wait` (≈4 ticks)**: not the configured metric
   (push_t uses `track_c3_progress_via: 3` = kConfigCostDrop, matching
   reference).

## Verification run (post-95be21f)

`verify_wsmargin_p144_60s`: crashed step 280 / t≈28 s at r=0.2490 — same walk,
as predicted (margin fix targets samples, not the drift). 6th consecutive
r<0.25 abort. Diagnosis confirmed: the walk is the run-killer and none of the
in-conformance fixes stop it.

## Open levers (all need a user decision — each is off-reference or
previously flagged needs-auth)

- Re-arm the phantom watchdog (p118 mechanism) — flagged "needs auth" in the
  2026-07-28 phantom-park closure.
- `PORT_DISABLE_CONTACT_LOSS_GATE=0` — re-enables the port-only disengage
  net; would exit c3 ~0.5 s into any no-real-contact stint.
- `PORT_C3_PHANTOM_TRAJ_OVERRIDE=1` — replaces phantom-plan trajectory with a
  geometric approach ramp (default-off).
- Shorten the kConfigCostDrop window mapping (e.g. assume faster ref loop
  cadence) — interpretation change, not a clean conformance fix.
- Accept crashes as reference-faithful: the reference would DRAKE_DEMAND
  identically if its EE walked below r=0.25.

## RESOLUTION (2026-07-31, e83ad4b) — verdict 1 RETRACTED

The crash race was NOT reference-faithful. Root cause: the EE-space LCS
builds c at u=0 but subtracted H@u_star anyway (ST :2060, Anitescu :2158),
shifting every EE-coupled complementarity row by −H·u* whenever the full
solve linearized at u_lin=_last_u ≠ 0. The corrupted EE-BOX gap row +
_last_u feedback drove the monotone walk. Ground rows (H≈0) and surrogate
solves (u_lin=0) were unaffected. Discovered via ADMM-dump self-consistency
(ground rows exactly at LCS scale, EE-BOX rows off by exactly −H@u_lin) and
a live plant repro; fixed in e83ad4b with an identity regression test.
p147 (180s canonical): first full-length run after 8 consecutive crashes,
min EE r=0.333, no workspace violations. Remaining failure mode is
no-real-contact idle (knot-0 phantom λ under reference-config soft
complementarity) — 0.1906/0.6263 FAIL, zero real-contact ticks.
