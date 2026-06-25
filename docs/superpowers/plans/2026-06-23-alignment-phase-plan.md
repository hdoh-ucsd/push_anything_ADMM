# Alignment Phase — Canonical Plan

**Status:** RATIFIED 2026-06-23.
**Branch:** `reference-conformance`. **HEAD at ratification:** `3fda887`.
**Substrate:** port (`/root/push_anything_ADMM`) vs reference (`/root/reference_repos/dairlib_sampling_c3` on `push_anything_dev` @ `257e3ede`).

This file is the **canonical source-of-truth** for the Alignment Phase plan AND its conformance state. Future alignment plans index against this file; they do NOT re-derive from the logic trees (`/d/projects/ERL/push_anything_ADMM/understand_logic_tree/{reference,port}/`). When a stage passes or a flagged item resolves, the relevant row in §1 / §3 is updated AS PART OF that stage's definition of done — not later. The anti-stale discipline (§7) is the whole point of this file.

The HTML rendering the user maintains is a synced view of this markdown; this file is the source.

---

## 0. TOP-LEVEL PROPERTY — two deliberate exceptions to full conformance

Full conformance with the reference is the goal, with **TWO explicitly held exceptions**:

| # | Exception | Rationale |
|---|---|---|
| 1 | **The box-pushing task** | The unique research goal — the cube is what we want to push; the reference's `anything` task (alphabet letters / jacktoy / push_t) is a different end. The port targets the cube; the reference targets its tasks. |
| 2 | **The ADMM solver internals** (iter count, adaptive-ρ schedule, projection details, E-matrix tangent-row zeroing) — **CORRECTION 2026-06-23: previous-block PROMOTION RETRACTED.** The held research target stays HELD — the disambiguation probe refuted the premise that the modeling exception was the gate. See §0 retraction subsection below + §3 Stage C outcome. | The non-convergence IS still load-bearing on the no-push, but the cause is NOT the §0 #2 modeling exception (LCP at u=0 has a complementarity-feasible solution; the deck's "tangent rows zeroed" structural claim is refuted on the live instance). The real gate is an isolable algorithmic defect in the C3+ iteration scheme (sub-hyps 1a/1b/1c), NOT the held research target. |

**Everything else aligns.** `iter=25` is therefore NOT an unexplained divergence to justify away — it is the held research target. **CORRECTION (2026-06-23):** the original "Its adequacy FOR PUSHING is evidenced by the planar sandbox success (port's C3+ reaches 99.2% in planar). The 3D pushing failure was the reposition→admit→executor chain, not the solver." is OVERTURNED by Probe B. The planar sandbox is a substantially different problem (lower n_lambda, smaller condition number, no z-axis contact); planar adequacy did NOT generalize to 3D pushing, and the 3D pushing failure includes the solver as a LOAD-BEARING contributor, NOT just the reposition-admit-executor chain. Anti-stale per §7.

---

### §0 exception #2 — RETRACTION of the PROMOTION (2026-06-23, disambiguation probe)

**The previous block PROMOTED §0 #2 from "post-cube-pushed research target" to "IMMEDIATE GATE" on the premise that the ADMM non-convergence was caused by the E-matrix modeling exception. The disambiguation probe REFUTES that premise on two independent counts. Per the same §7 anti-stale discipline that motivated the promotion, the promotion is RETRACTED here BEFORE the subsequent isolation probe runs.**

The original Probe-B FACTS preserved (the non-convergence reading is correct):
- `admm_pr` (final primal residual) median **4.95**, max 14.32 — **3-4 orders above tol=1e-3**, every c3 tick.
- `admm_dr` (final dual residual) median **83.3**, max 398.5 — **5 orders above tol=1e-3**.
- `admm_iters = 25 / 25` on **119 / 119** c3 ticks. `converged = 0 / 119`.
- Behavioral manifestation (preserved): u_mag median 6.6 N; ee_step1_dot_box median −0.94 (predicted EE retreats); box_total_xy median 234 mm/horizon; box_dot_goal median −0.999 (predicted box moves toward goal). **234 mm of fictional predicted motion per horizon; 0 mm of real box motion over 12 s.** The non-convergence IS load-bearing on the no-push — that part of Probe B's reading stands.

**What was WRONG in the previous block's promotion (now retracted):**

**(a) MODELING REFUTED — direct fixed-point existence at u=0.** The disambiguation probe captured the live c3-tick-50 ADMM instance (`stage_c/admm_dump/seed0_full50.npz`; dump gate commit `bbcb6c8`) and ran a brute-force enumeration of all 2⁶ = 64 complementary bases on the knot-0 LCP at u=0. **A feasible LCP solution exists:**
- `λ = [γ=0.146, λ_n=0.584 N, λ_t={0, 0.117, 0, 0.117}]`
- `w = M·λ + q = [-0, -0, 0.292, -0, 0.292, -0]`
- `max |λ_i · w_i| = 6.94e-17` (complementarity satisfied to machine precision)
- `λ ≥ 0` ✓; `w ≥ 0` ✓.

A complementarity-feasible λ exists. **The ADMM failure is NOT "no fixed point exists."** The §0 #2 modeling exception ("E-matrix tangent-row-zeroing → no joint fixed point exists at this linearization") is REFUTED on the captured live instance.

**(b) E-MATRIX STRUCTURE — deck claim REFUTED on the live instance.** The captured E is 6 × 19 with exactly 1 zero row out of 6. **But that zero row is row 0 — the γ Stewart-Trinkle slack slot, NOT the tangent rows.** The γ-zero-row is a STANDARD feature of the Stewart-Trinkle reformulation and does NOT structurally prevent the LCP from having a feasible point (verified by (a)). The deck's "tangent rows zeroed, E zero-rows=1/6" structural claim was wrong about the LIVE pipeline's E.

**Therefore §0 #2 is NOT the gate, and the previous block's boundary call was PREMATURE — RETRACTED.**

### §0 exception #2 — STATUS AFTER RETRACTION

- §0 #2 stays HELD as the original research target (the post-cube-pushed framing of "study/vary the held ADMM solver internals AFTER the cube-pushed gate" is RESTORED — the modeling exception was not what was blocking us).
- §1 row 3 (ADMM/C3+ solver) cell text is updated below to reflect the RETRACTION: the cell is no longer "PROMOTED HELD-EXCEPTION → IMMEDIATE GATE." It is "BLOCKED ON THE ITERATION-SCHEME DEFECT (an isolable algorithmic bug, sub-hyps 1a/1b/1c below), NOT on §0 #2."
- §1 rows 2, 4, 5 + Stage E row are updated below to RE-TAG from "research-gated on §0 #2 modeling" → "blocked on the ADMM iteration-scheme defect (isolable; sub-hyps 1a/1b/1c)."
- The planar-99.2%-doesn't-generalize correction immediately above the (now-retracted) promotion subsection STANDS — planar adequacy still did not generalize to 3D pushing; that conclusion does not depend on the (refuted) modeling-is-the-gate claim.

### Deck layers 1 & 2 — ALSO refuted as the deck framed them

The disambiguation probe also overturns the deck's two non-modeling layer claims:

**Layer 1 (implementation) as framed — "adaptive-ρ decided-but-never-applied; ρ silently 100→25":** REFUTED. The dump captured `rho_initial = 100.0`; the live solver's end-state `rho = 25.0` is the adaptive rule FIRING CORRECTLY (`100 → 50 → 25` over the every-10-iters trigger within `max_iter=25`). The rule IS applied; the convergence still fails. The deck's "decided 0 applied" framing was wrong.

**Layer 2 (tuning) as framed — "max_iter=25 vs cond≈10⁵; budget 1-2 orders too low":** REFUTED as the sole gate. A 4 × 4 sweep on the captured instance over `ρ ∈ {100, 10, 1, 0.1}` × `max_iter ∈ {25, 100, 500, 1000}` converges in **0 / 16 cells**. `pr_final` stays in `[4.94, 23.12]` across all cells (4 orders above tol=1e-3); `dr_final` in `[2.74, 84.42]` (3 orders above tol). Residuals OSCILLATE non-monotonically with iteration count (`ρ=100, max_iter 25→100→500: pr 4.94→12.65→14.37`). A 40× iteration budget AND 3 orders of ρ variation are insufficient.

### THE ACTUAL CAUSE (corrected)

The real cause is an **isolable algorithmic defect** in the C3+ iteration scheme: it **oscillates around a feasible point it provably cannot lock onto**. This is NEITHER the fundamental modeling gap (a solution demonstrably exists — `λ_n = 0.584 N`) NOR a trivial tuning/implementation bug (iteration-budget and ρ-fixes do not help — the iteration scheme itself oscillates around the feasible point).

**Sub-hypotheses (OPEN; the isolation probe in the next block disambiguates):**
- **(1a)** Bui componentwise δ-projection (eq. 12) has a sign or implementation bug on the Stewart-Trinkle slot layout (γ, λ_n, λ_t).
- **(1b)** OSQP block P/q construction has a sign error or missing term that prevents the augmented-Lagrangian fixed point from being a feasible LCP point.
- **(1c)** ρ-adaptation pathologically destabilizes when residuals oscillate (the halve-on-dr>>pr trigger could be flipping the iteration into the opposite regime, re-triggering on the next check).

### Sub-fork (a) ALIGNMENT — status after retraction

The previous block's note "(a) DECLINED-as-silent-failure" STANDS for now — but for a different reason: while ADMM does not converge, porting `GetNClosestContactPairs` would still silently fail (the planner would still ship a retreating x_seq). Re-opens IFF the iteration-scheme isolation probe identifies which of (1a)/(1b)/(1c) is the bug AND the fix is landed live; then the planner-side admission question becomes the next gate (per row 2 plan).

### Next: the isolation probe

The isolation probe runs in the next block — projection-swap (Aydinoglu per-contact Lorentz, the existing C3-not-plus path) vs Bui componentwise (C3+ current) on the captured `seed0_full50.npz` instance, diffed against the **brute-force LCP oracle** `λ_n = 0.584 N`. If C3 Lorentz converges to the oracle where C3+ componentwise does not, (1a) is isolated. NOT actioned in this plan-doc edit.

---

## 1. Convergence map (port vs reference) — live status

| # | Mechanism | Status | Source of truth (file:line) | Flips when |
|---|---|---|---|---|
| 1 | Dispatch (mode-switch) | **RECONCILED** | `mode_switch.py:95-144` ↔ `sampling_based_c3_controller.cc:1145-1310` | (already at goal) |
| 2 | LCS admission | **PARTIAL — executor-side SKIP stands; PLANNER-SIDE admission blocked on the ADMM iteration-scheme defect (1a/1b/1c) — NOT on §0 #2 modeling (CORRECTED 2026-06-23; see §0 retraction)** | `lcs_formulator.py:390-456` (2 mm threshold — executor-side contact filter AND planner-side LCS-build path) ↔ `LCSFactory::GetNClosestContactPairs` (N-closest) | Blocked on the isolable C3+ iteration-scheme defect (oscillation around a feasible LCP point), NOT on §0 #2 modeling. Re-opens IFF the isolation probe identifies the bug AND it lands live. Executor-side: no flip — already skip-justified. |
| 3 | ADMM / C3+ solver | **(CORRECTED 2026-06-23) HELD EXCEPTION STAYS HELD — earlier promotion RETRACTED; the "C3+ iteration-scheme defect (1a/1b/1c)" framing is now RE-TAGGED to the cadence discriminator pending the 1 kHz re-measure** (LCP live verification + §3 Stage C outcome). | `admm_solver.py:_solve_c3plus`, `iter=25`, adaptive-ρ | Live non-convergence at 100 Hz (pr median 4.95 / dr median 83.3 / converged 0/119) IS load-bearing on the no-push, but the projection-bug attribution is **100Hz-PROVISIONAL**. The cadence discriminator (componentwise @ 1 kHz, the reference's actual config) decides whether the projection defect is real and cadence-independent OR a symptom of a starved warm-start at 100 Hz. Sub-hyps (1a) componentwise-projection bug; (1b) OSQP block; (1c) ρ-adaptation may dissolve if cadence is the cause. |
| 4 | Control input u | PARTIAL — wired but not default; **mechanism probe-confirmed reference-EXACT** (Stage C probe 2026-06-23; see §3 Stage C outcome) | `wrapper._derive_force_command` (`-g_hat`+mag, env-gated `PUSHA_FORCE_ROUTING=u_sol` for u_seq[0]) ↔ `sampling_based_c3_controller.cc:1822-1832` (`force_samples = u_sol[i]`) | **RECONCILED flip BLOCKED on the cadence discriminator (RE-TAGGED 2026-06-23 from "iteration-scheme defect 1a/1b/1c"; see row 8 promotion + §7.2).** The mechanism is reference-exact; the row-flip gate is gap-closing / Stage E motion-bar. The LCP live-verification at 100 Hz produced 34 mm of real motion but not the gap-closed verdict; whether the projection is the fix OR cadence is the cause remains unresolved. |
| 5 | Executor (OSC + force-tracking) | PARTIAL — **mechanism probe-confirmed reference-EXACT**; Reading 2 (executor/compliance bottleneck) REFUTED (phi_act < setpoint_sd on 119/119 — executor BEATS its own commanded position by ~15 mm) (Stage C probe 2026-06-23) | `osc/qp_builder.py:73` + `params.W_force=100.0` ↔ `franka_osc_controller.cc:167-170` + `osc_params.W_ee_lambda = I_3` (scalar 1.0; port W_force/W_track ratio 100/100 = reference's 1/1 ratio preserved) | **RECONCILED flip BLOCKED on the cadence discriminator (RE-TAGGED 2026-06-23 from "iteration-scheme defect 1a/1b/1c"; see row 8 promotion + §7.2).** Same gate as row 4. |
| 6 | Reposition mechanism | **PARTIAL — wired (descent reference-aligned); residuals deferred to Stage E** | `reposition_trajectory.py` + `sampling_based_c3_controller.py:2502-2528` (gated PWL path; default OFF) ↔ `Reposition(...) + UpdateRepositioningExecutionTrajectory + LcmTrajectoryReceiver` (see Stage A outcome subsection at end of §3 Stage A) | Stage E motion-decomp + force-tracking confirm residuals (NOT this stage) |
| 7 | Push-point height computation | OPEN | `config/tasks.yaml:22 pushing.sampling_height = 0.03` (hand-coded) ↔ `sampling_params.yaml:64 z_height` auto-generated per object | Stage D passes |
| 8 | Entry cadence + multi-process | **ACTIVE FRONT — UPDATED 2026-06-25: constant-level tick→sim-time conversion LANDED + 100Hz-BYTE-VERIFIED; ≥1 BEHAVIORAL coupling in the EE-landing chain remains, still blocking 1 kHz c3 engagement (see §7.5)** | (a) tick-vs-sim-time semantics — CONSTANT-LEVEL conversion DONE (15 constants → seconds; SMOKE 1 5/5 PASS, mode_match 1201/1201 byte-equivalent at 100 Hz). BEHAVIORAL coupling REMAINS: the PWL rebuild gate (sampling_based_c3_controller.py:2643) fires per-tick starting at step ~1630 (just before c3 entry at step ~1700) — most likely `_refresh_buffer_on_arrival` → next-target-selection produces a new target every tick once EE is in proximity. (b) RATE + ARCHITECTURE — `main.py:571` + single-process loop ↔ `LcmDrivenLoop` 3-process LCM-coupled. | (a)' constant-level: DONE. (a)" behavioral coupling: trace probe next (the audit's SECOND incompleteness — grep is structurally blind to call-frequency couplings). (b) discriminator + Stage F multi-process — DEFERRED until (a)" lands. Row stays ACTIVE FRONT. |
| 9 | Contact-proximity entry-gate | **RETIRED (2026-06-23; 0% firing rate measured across NEW/OLD/baseline seeds; see §3 Stage B outcome)** | `wrapper.py:1038-1147` (no reference equivalent — reference uses EE-z-close clause at `sampling_based_c3_controller.cc:1290-1293` instead) | (already retired) |

**Maintenance:** each row above updates AS PART OF the corresponding stage's definition of done. A stage that passes its bar without flipping the row's status is the stage's bar being wrong, not the status row.

---

## 2. Pre-flight

### P0 — Probe disposition (user's separate call, not part of this plan to action)

The 7 uncommitted Stage-1 / Stage-2 / Stage-2d / Stage-5 files (`config/tasks.yaml`, `control/admm_solver.py`, `control/ci_mpc_c3plus.py`, `control/lcs_formulator.py`, `control/sampling_c3/wrapper.py`, `control/task_costs.py`, `main.py`; +321 / -15) plus their env-gated knobs (`PUSHA_LOOKAHEAD_STEP`, `LCS_EXPLICIT_BOX_GND`, `PUSHA_STAGE5_U_HORIZONTAL/VERTICAL`, `PUSHA_STAGE5_R_VECTOR`, `PUSHA_FORCE_ROUTING`, `PUSHA_HORIZON_LAM_DUMP`) are all default-OFF — the port runs baseline behavior under no env vars. **Before Stage A**: decide per knob: keep / commit / revert. Surfaced here, NOT actioned here.

### P1 — Substrate lock

Alignment baseline pinned at HEAD `3fda887` on branch `reference-conformance`. Substrate: `--ee-space --solver c3plus --admm-iter 25 --sampling-c3 config/sampling_c3_kik.yaml`, mass=0.2 kg, cube 0.10 m, pusher sphere 0.025 m. Seeds {0, 1, 2, 4} (seed-3 intentionally excluded; see §6). 12 s sim window (pitch develops in 5-12 s).

---

## 3. Stages (chain-first, NOT top-down on §1)

Per-stage bar = **LOCAL mechanism effect** (the change the stage was supposed to cause moved). End-to-end motion (≥20 mm goal-aligned) is **cumulative at Stage E**, not at every stage. This resolves the per-stage-vs-cumulative question plan-wide: requiring 20 mm motion at A would falsely fail a correct reposition that needs Stage C's executor to translate force into motion.

The inter-stage "wired" signal (§4) catches the dead-planner false positive at every stage.

---

### Stage A — Reposition mechanism port

**Reference mechanism (port this):** the reference builds a full N-knot Cartesian trajectory via `Reposition(...)` (`reposition.cc:13`) dispatching on `traj_type` (kPiecewiseLinear default for `anything`); packages it into `end_effector_position_target` / `end_effector_orientation_target` / `end_effector_force_target` LCM trajectories at `UpdateRepositioningExecutionTrajectory` (`sampling_based_c3_controller.cc:1839-1928`); OSC consumes via `LcmTrajectoryReceiver` + `TransTaskSpaceTrackingData("end_effector_target", K_p, K_d, W_end_effector, ...)` (`franka_osc_controller.cc:101-103, 149-158`).

**Port replaces:** the per-tick setpoint-march + per-tick single-knot IK + joint-PD path (`reposition_ik.py:1318, 1355-1359, 352, 1219`). Replacement: per planner tick build an N-knot trajectory, feed it to the OSC as an in-process equivalent of `LcmTrajectoryReceiver` (since the port is single-process Python, this is direct trajectory handoff into the OSC tracking-data structure — NOT LCM IPC).

**LOCAL pass bar (HARD, four conditions):**
1. **EE lands within the 2 mm admit window at c3 handoff**: at first c3 entry, `|p_ee_now - p_box_surface| ≤ 2 mm` (vs baseline ~35 mm shy; the 20 mm IK tolerance + 5 mm setback × wedge gives the current gap).
2. **Admit rate UP vs baseline**: `c3 steps with EE-BOX lam_n>0` / `c3 steps total` strictly increases over baseline measurement on the same seed.
3. **Contact-proximity entry-gate inert**: `wrapper.py:1038-1147` `[ENTRY-GATE]` log line fires on < 5 % of free→c3 candidate transitions across the 12 s window (the Sink-4 band-aid should retire as the mechanism it patches retires).
4. **|qy| and |qz| not worse than baseline**: per-seed `max|qy|` and `max|qz|` ≤ baseline_no_iii values + small noise band.

End-to-end motion (≥ 20 mm) is **NOT** a Stage A bar — held cumulative to Stage E.

**Wired signal (inter-stage rule check):** conditions 1 and 2 above must move vs baseline. If neither moves → port has an integration bug → STOP. (Catches the dead-planner case.)

**On pass:** §1 row 6 (Reposition) flips OPEN → RECONCILED. §1 row 9 (Contact-proximity entry-gate) flips PORT-ONLY → RETIRED if condition 3 holds. If row 9 doesn't flip but row 6 does, note row 9 as awaiting Stage B confirmation.

---

#### Stage A — Actual outcome (2026-06-23, anti-stale record per §7)

Stage A did NOT cleanly pass. **WIRED with four deferred residuals.** Row 6 status set to PARTIAL accordingly (NOT RECONCILED). Default stays OFF (`use_reposition_pwl_trajectory=False`); the gated path runs only when `PUSHA_REPOSITION_PWL=1`.

**Descent mechanism reference-aligned (the mechanism IS correct).** The port's `RepositionTrajectory` (`control/sampling_c3/reposition_trajectory.py`) is architecturally aligned with the reference: constant-per-leg `seg_durations = seg_lengths / speed`, mirroring `reposition.cc:391-467 RepositionPiecewiseLinear` (`step_size = speed × dt`). Only the speed *value* diverged: `config/sampling_c3_kik.yaml` ran at `0.40 m/s` (2.2× reference). The free-mode brush diagnosis (descent at vz≈0.44 m/s past phi=6 mm → Drake compliant-contact halo applies force at sub-LCS-admit distances → box yawed before c3 engaged) was resolved by adding a separate `pwl_speed: 0.18 m/s` field on `RepositionParams` (reference push_t value; kept distinct from `.speed` which the legacy IK tracker consumes as a planning-lookahead stride, so flag-OFF baseline is bit-identical). Commits: **848cc79** (descent fix) + **e65338c** (analysis tooling `scripts/_analyze_stage_a_fork.py`).

**Four deferred residuals** (these are NOT Stage A failures to retry; they are next-stage diagnostics):

  (i) **Seed-0 |qz| 0.038 residual after the fix** — 127× baseline. Descend-velocity matched the reference (vz≈-0.14 m/s vs reference 0.18) yet seed-0 still tips partially, hinting a **geometric** component remains alongside the eliminated velocity component (descent path geometry vs vertical-velocity-only fix). Origin geometric-vs-residual-velocity is **TBD**. — *Update 2026-06-23 (Stage C probe — see §3 Stage C outcome):* seed-0 `max|qz|` **COLLAPSED under u_sol routing**: baseline (`-g_hat × 100`) 0.0379 → u_sol (`u_sol × 100`) **2.6e-4** (a 146× drop, indistinguishable from seed-4's clean baseline). Unpredicted positive side effect of the routing change. Origin question is RETIRED in the u_sol-active regime; remains TBD in the legacy `(-g_hat)` regime. Carries no implication for the planner-side bottleneck.

  (ii) **The MOTION-SOURCE question** — the OLD fast-descent's 25-29 mm box motion may have been a brush-SHOVE artifact (EE impacting the box during reposition) rather than genuine contact-phase push; the reference-aligned gentle descent eliminated BOTH brush AND motion together (seed 4: 25.7 mm → 0 mm; seed 0: 29 mm → 10 mm), consistent with them being the same uncontrolled-impact phenomenon. **Resolvable ONLY post-Stage-C**: force-tracking either RESTORES motion on the gentle landing (→ brush-shove confirmed, gentle landing was correct, motion was always supposed to come from the executor) OR fails to (→ engagement-starvation is real, the gentle descent kills c3 throughput too aggressively, needs late-deceleration shape: slow only the last ~30 mm). Re-sequenced as Stage E's FIRST diagnostic (see Stage E entry below). — *Update 2026-06-23 (Stage C probe):* **STILL UNRESOLVED at Stage C completion.** The probe run (seed 0, u_sol × 100, 12 s) produced 0 mm box motion AND the gap-closed VERDICT FAILED, so the Stage-C plan §5.E.1.b motion-diagnostic interpretation table was gated-off (the table requires gap-closed = PASS as a precondition; with FAIL neither *brush-shove refuted-as-necessary* nor *brush-shove confirmed* applies). The question stays deferred to Stage E, but its scope is now narrower: the planner-side LCS-build bottleneck (row 2 reopened) is upstream of the motion-source question — until the planner predicts contact, the executor cannot produce post-contact motion, so the brush-shove-vs-engagement-starvation read is uninformative under the current LCS. The Stage E first diagnostic must be re-attacked AFTER the §3 Stage C sub-fork resolves.

  (iii) **Rebuild-on-flip churn (~3-4 Hz)** — orthogonal to the descent fix. Confirmed clear-on-transition firing per mode-flip (not per-tick-march). Separate follow-on fix: soften clear-on-transition for 1-tick c3 islands. Not blocking; not addressed in Stage A.

  (iv) **Cond-1 EE-landing WINDOW un-evaluable** — first c3 episodes are 38 ticks on both seeds, < `WINDOW_MIN_TICKS=50` floor. Separate window-strategy decision (lower the floor to 30, or change episode-selection from first-only to merge-fragments-separated-by-<3-free-ticks, or first-≥30-tick). Not addressed in Stage A.

**Carry-forward state (binding on subsequent stages):**
- Stages B/C/D/E run with `PUSHA_REPOSITION_PWL=1` (reference-aligned reposition path active during their measurements).
- Default stays OFF (`use_reposition_pwl_trajectory=False` in dataclass + YAML).
- Promotion-to-default deferred to Stage E pass (i.e., only flip the default after Stage E confirms the gentle landing + executor combination clears the cumulative motion bar).
- Cond 4 (|qz|/|qy| not worse than baseline) was NOT relaxed — Read X (no-force-regulation as sole cause) was falsified by the timing diagnostic; the bar still stands, the residual is correctly downstream (Stage C/E).

---

### Stage B — LCS admission port (MEASURE-THEN-DECIDE, not mandatory)

**Decision-first protocol:** after Stage A lands the EE on-surface, MEASURE whether the 2 mm threshold (`lcs_formulator.py:400`) still rejects the EE-box pair at meaningful rates.

- If A makes the 2 mm threshold rarely bite (EE-BOX pair admitted at ≥ 80 % of c3-mode ticks): **B is REDUNDANT — SKIP.** Note: N-closest-always is MORE permissive than the 40 mm-threshold ablation that already regressed motion 29 mm → 10 mm; porting it risks reintroducing that at-distance regression for zero motion gain. §1 row 2 status remains "PARTIAL — measure-then-decide" with a note "post-A measurement: skip-justified."
- If A leaves the 2 mm threshold still rejecting at meaningful rates (EE-BOX pair admitted at < 80 % of c3 ticks): **port N-closest-always**, replacing the 2 mm filter with the reference's `GetNClosestContactPairs(plant, ctx, geoms[i], num_to_select=resolve_contacts_to_lists[group_idx])` mechanism.

**Reference mechanism (if porting):** `sampling_based_c3_controller.cc:1582-1614 GetResolvedContactPairs(...)` + `LCSFactory::GetNClosestContactPairs`. The reference always picks N pairs per group regardless of absolute distance.

**LOCAL pass bar (HARD, conditional — applies ONLY if porting):**
1. **EE-BOX pair present in LCS at ≥ 50 % of c3-mode ticks** (verified via horizon-λ probe: `tags` includes `EE-BOX`).
2. **Held pair carries non-zero λ_n**: `λ_n > 0` on the EE-BOX row at ≥ 50 % of admitted ticks. **Present-and-dead = FAIL.** Forcing a far pair into the LCS where the complementarity gives `λ_n = 0` is exactly the phantom-contact case (the banked admit-layer-structurally-dead finding from the closed investigation). Presence alone is the trap — λ_n > 0 is the real signal.
3. **Stage A's bars must still pass** at the new admit mechanism — no regression on (1)-(4) of Stage A.

**On pass:** §1 row 2 flips PARTIAL → RECONCILED. §1 row 9 (entry-gate) flips RETIRED if it wasn't retired by A.

**On skip-justified:** §1 row 2 stays PARTIAL with the justification note; row 9 status decided by Stage A's condition 3.

---

#### Stage B — Actual outcome (2026-06-23, anti-stale record per §7)

**Decision: SKIP** — hybrid rationale (not "threshold rarely bites" and not "engagement starved"; "extending admit would phantom"). Measurement was read-only over the NEW Stage A flag-ON logs (`stage_a_speed018/seed{0,4}/run.log`); no new sim, no port code, no N-closest implementation.

**Row 2 (LCS admission) appended status note:**

> Post-A (PWL flag-ON, pwl_speed=0.18) measurement, seeds 0/4: EE-BOX admit-active (λ_n>0) rate **0.9% / 1.7%**, both far under the 80% redundant-skip bar. The non-admit gap is NOT engagement-starvation (the EE sits at phi=2-5mm above the surface, geometrically close, all c3 episode) — it is **99-100% threshold-rejection** (phi ∈ (2mm, 10mm], a present pair just outside the 2mm filter). By the literal decision tree this reads PORT-N-closest, BUT overruled on evidence: today's under-2mm admits **already phantom 20-83%** (λ_n=0, C3+ predicts no binding contact), and the phi 2-10mm band N-closest would open is even further from binding → N-closest would inflate phantoms (present-and-dead, failing bar item 2) without delivering force. SKIP confirmed via the hybrid rationale (not threshold-rarely-bites, not engagement-starved, but extending-admit-would-phantom). ROOT CAUSE of the low admit-active is UPSTREAM: the EE settles at phi=2-5mm without closing into physical contact under the gentle 0.18 descent — resolvable at Stage C (force-tracking pushes the EE the last few mm into contact) or via Stage A descent-target tuning (deferred residuals (i)/(ii)). Measured 2026-06-23 from stage_a_speed018 logs, read-only.

> **Update 2026-06-23 (Stage C post-FAIL probe — see §3 Stage C outcome):** The probe (seed 0, 119 c3 ticks, `stage_c/seed0_usol_100_setpoint.log`) FALSIFIED the premise of this SKIP. The SKIP assumed Stage A's gentle landing closed the geometric gap, so widening admission would only inflate phantoms — but that reasoning measured the EXECUTOR-side admission filter and never measured the PLANNER-side. The admission threshold lives in TWO places: the executor's contact filter AND the planner's LCS-build path (`lcs_formulator.py:245` + contact-pair filter). The probe shows the binding constraint is PLANNER-SIDE: `phi_pred_min ≥ 12 mm on 0/119 c3 ticks` (the planner NEVER predicts the EE approaching within 5 mm over any of 20 horizon knots); `setpoint_sd ≥ 10 mm on 119/119` (the planner ALWAYS ships a position target ≥ 10 mm off the surface, median +18.7 mm); `argmin@k=1 on 119/119` (the horizon monotonically RETREATS from contact — the planner predicts its own EE leaping from the current +2.7 mm to +12-30 mm at the next knot). **The executor-side SKIP STANDS** (today's under-2mm admits still phantom 20-83%; widening the executor φ-gate would still phantom). But **a distinct PLANNER-SIDE admission question is now OPEN and implicated**: does porting the reference's `GetNClosestContactPairs` (or lifting the 2 mm planner-side LCS-build filter) make the planner's `x_seq` predict contact? Pending the B phase-detector probe, which determines whether this is an alignable LCS-construction gap (port N-closest, sub-fork (a)) or the reserved ADMM E-matrix inconsistency (§0 #2, sub-fork (b)).

**Row 9 (Contact-proximity entry-gate) RETIRED status note:**

> Structurally inert at 0% firing rate across NEW/OLD/baseline seeds (2026-06-23). The contact-proximity problem this gate patched is now handled by the reposition mechanism landing the EE close (phi 2-5mm), not by this gate firing. Stage B's SKIP does not alter this — the gate is inert regardless of admit mechanism. The canonical plan's Stage A "On pass → row 9 RETIRED if condition 3 holds" is satisfied.

---

### Stage C — Executor force-tracking port

**FIRST action (block the rest of the stage on this):** READ the reference's `W_ee_lambda` from `examples/sampling_c3/shared_parameters/osc_params.yaml` (currently TBD in the port — `osc/qp_builder.py:73`'s `W_force` is hardcoded at 100.0 per `params.py:528`). If `W_ee_lambda` differs materially from `W_force=100.0`, record the gap as a finding. **Do NOT run the force-tracking port until `W_ee_lambda` is read and set.**

**Reference mechanism (port this):** planner packs `u_sol` (the C3+ solved Cartesian EE force) into `end_effector_force_target` LCM trajectory at `sampling_based_c3_controller.cc:1822-1832` `force_samples.col(i) = u_sol[i]`. OSC consumes via `ExternalForceTrackingData("end_effector_force", W_ee_lambda, plant, plant, kEndEffectorName, Vector3d::Zero())` (`franka_osc_controller.cc:167-170`); registered with `osc->AddForceTrackingData(...)` (`:188`).

**Port replaces:** `wrapper._derive_force_command` — retire the `-g_hat`-direction + `Σ|λ_n|`-magnitude-floor-cap path. Promote the env-gated `PUSHA_FORCE_ROUTING=u_sol` branch (`wrapper.py:450-471`) to default behavior. Tune `params.W_force` to the reference's `W_ee_lambda` value (set in Stage C's first action above).

**LOCAL pass bar (HARD):**
1. **Planner u_sol consumed (not derived)**: `lambda_des == base_mpc._last_u_seq[0]` at every c3-mode tick (verified by the `[FORCE-ROUTE]` diagnostic).
2. **u_z non-trivial**: planner-emitted `u_sol[2]` varies tick-to-tick (non-zero variance over the c3 window). This confirms force-routing decouples F_z from the OSC z-error mechanism (under `-g_hat`, F_z is structurally zero).
3. **Stage A's bars and Stage B's bars (if B ran) must still pass** — no regression.

End-to-end motion is **NOT** a Stage C bar — held to Stage E. But: if Stages A + B + C combined still produce zero motion, Stage E will catch it.

**On pass:** §1 rows 4 (Control input u) and 5 (Executor) flip PARTIAL → RECONCILED.

---

#### Stage C — Actual outcome (2026-06-23, anti-stale record per §7)

**Executor-force-routing sub-goal DONE (probe-confirmed reference-EXACT). Gap-closed verdict FAILED — cause LOCALIZED to the PLANNER's LCS-build, NOT the executor.**

**Phase 1 cell (u_sol × 100, PUSHA_REPOSITION_PWL=1, seeds {0, 4}, 12 s):** the GAP-CLOSED VERDICT defined in `docs/superpowers/plans/2026-06-23-stage-c-executor-force-tracking-port.md` §5.E.1.a FAILED on both seeds. V3 (F_z NON-TRIVIAL, live `[FORCE-ROUTE]` trace) PASSED on both seeds (`var(u_z)` 0.42/0.44, `max|u_z|` 3.4 N / 4.1 N — real non-zero F_z delivered). D-1 (lambda_des == `_last_u_seq[0]`) PASSED on both seeds (119/119 and 155/155 c3 ticks `eq=True`). V1 (sustained-contact `φ < 2 mm ∧ lam_n_ee_box > 0` ≥ 30%) FAILED (5.0% / 3.2%). V2 (admit-active `lam_n_ee_box > 0.5 N` ≥ 15%) FAILED (5.0% / 3.2%). V4 vacuously satisfied (V4 PRECISION NOTE — only 5-6 admits across both seeds, V1∧V2 carry the contact-happened load and they failed). The EE hovered in Stage B's 2-5 mm band on ~86-87% of c3-ticks under u_sol routing. Commits: `d43daa1` (trace), `eff91b9` (Phase 1 metrics artifacts).

**Post-FAIL localization probe (seed 0, `PUSHA_SETPOINT_TRACE=1`, 119 c3 ticks):** the named bottleneck (W_force pull-toward-u_sol vs W_track pull-toward-p_ee_des) was MEASURED, splitting the fork:
- **Reading 2 (executor / compliance) — REFUTED.** `phi_act < setpoint_sd` on 119/119 c3 ticks. The executor BEATS its own commanded position by ~15 mm via the W_force pull-toward-u_sol — outperforming, not bottlenecking. Sphere-radius / contact-stiffness / QP-feasibility are NOT the issue. Commit: `ecd74d8` (probe trace).
- **Reading 1 (UPSTREAM / planner) — CONFIRMED hard.** `phi_pred_min ≥ 12 mm on 0/119 c3 ticks` (median +29.5 mm); `setpoint_sd ≥ 10 mm on 119/119` (median +18.7 mm, min +12.4 mm); `argmin@k=1 on 119/119` (planner's horizon monotonically RETREATS from contact). The planner's LCS solve outputs an `x_seq` that retreats from contact — it does NOT predict the EE landing on the box. The OSC is FAITHFULLY tracking that retreating setpoint; the W_force pull does as much as can be done given the position-task disagreement.

**Phase 2 (W_force = 1.0 to align with `W_ee_lambda = I_3`) DECLINED-as-wrong-knob.** The §0 ratio analysis exonerated the weight (port `W_force/W_track = 100/100 = 1.0`; reference `W_ee_lambda/W_end_effector = 1/1 = 1.0`); the probe MECHANICALLY confirmed (not just ratio-inferred) that lowering W_force at fixed W_track would drag `phi_act` toward the +15 mm retreating setpoint and AWAY from contact, strictly worse for V1/V2. Not auto-run.

**Sub-fork OPEN (the B phase-detector probe runs next to decide):**
- **(a)** LCS-admit / planner-side alignment — port `GetNClosestContactPairs` (or lift the 2 mm planner-side LCS-build filter) so the planner's `x_seq` can predict contact. This re-opens §1 row 2 planner-side (see §3 Stage B outcome update). The Stage B SKIP only retired the executor-side filter, not the planner-side.
- **(b)** ADMM E-matrix inconsistency — the held research exception §0 #2. Even if the LCS admits, the E-matrix's tangent-row zeroing may prevent ADMM from finding a solution that closes the gap, making the planner unable to model contact in principle.

The B phase-detector probe (`u_sol`-vs-`x_seq` consistency + ADMM convergence reading) is the next move. NOT actioned in this stage.

**Status:**
- §1 row 2 — REOPENED on the planner side; executor-side SKIP STANDS (see §3 Stage B outcome update).
- §1 rows 4 and 5 — ANNOTATED as *mechanism probe-confirmed reference-EXACT*, but RECONCILED flip DEFERRED to gap-actually-closing / Stage E promotion (the row-flip gate is the gap-closed verdict, which still FAILS via the planner-side bottleneck).
- §1 row 9 — RETIRED unchanged.
- Default-OFF unchanged (`PUSHA_FORCE_ROUTING` NOT promoted to default; deferred to Stage E pass).
- Phase 2 DECLINED-as-wrong-knob (probe-mechanically confirmed).
- The `[FORCE-ROUTE]` and `[SETPOINT]` traces are in-source under env gates `PUSHA_FORCE_ROUTE_TRACE=1` and `PUSHA_SETPOINT_TRACE=1` (default-OFF; baseline log unaffected).

**Carry-forward (binding on subsequent stages):**
- Stage D / Stage E run with `PUSHA_REPOSITION_PWL=1` AND `PUSHA_FORCE_ROUTING=u_sol` (the reference-exact force-routing path active during their measurements).
- `W_force = 100.0` held (the §0 ratio analysis + the probe agree the weight is not the lever).
- Stage A residual (i) (seed-0 `|qz| = 0.038`) COLLAPSED under u_sol routing (0.0379 → 2.6e-4) — surfaced as an unpredicted positive side effect of the routing change. Origin question retired in the u_sol-active regime; remains TBD in the legacy `(-g_hat)` regime.
- Stage A residual (ii) (MOTION-SOURCE) **remains UNRESOLVED**. The probe run produced 0 mm box motion and gap-closed FAILED, so the §5.E.1.b motion-diagnostic interpretation table was gated-off (neither *brush-shove refuted-as-necessary* nor *confirmed* applies — the table requires gap-closed = PASS as precondition). Stays deferred; will be re-attacked at Stage E after the sub-fork is resolved.

---

### Stage D — Push-point per-task computation (MECHANISM-PORT, NOT z=0.05 hardcode)

**Reference mechanism (port this):** `sampling_params.yaml:64 z_height` is annotated `# This ^ will be overwritten by generating new files.` The reference auto-generates `z_height` per task from object geometry (likely via `examples/sampling_c3/sampling_generation/full_script.py` and/or `multiyaml_rewrite.py`). The MECHANISM is "compute z_height from object geometry," not "set z_height to 0.05."

**Port replaces:** the hand-coded `sampling_height: 0.05` (in `config/sampling_c3_kik.yaml:193`) AND the per-task override `sampling_height: 0.03` (in `config/tasks.yaml:22`). New code: a small generator that reads `task_cfg["size"]` (box half-extent) + pusher sphere radius + ground-z + (optional) face-normal direction and computes a natural CoM-z contact height per task. For our cube (size=[0.1,0.1,0.1] → half=0.05, CoM-z=0.05): auto-computed value should land within ±1 mm of 0.050 m.

**LOCAL pass bar (HARD):**
1. **Auto-computed z_height per task is within ±1 mm of the geometric CoM-z** for our cube.
2. **Cube_turning task regression check**: auto-computed value for cube_turning (which currently uses `sampling_height: 0.03`) is recomputed from cube_turning's geometry; if the geometric reasoning differs (e.g., due to a different goal-orientation), document the difference. If identical, current cube_turning behavior should be preserved.
3. **Stage A/B/C bars must still pass** at the auto-computed height — no regression.

**On pass:** §1 row 7 (Push-point height) flips OPEN → RECONCILED.

---

### Stage E — Multi-seed validation (the cumulative motion bar)

**NOT a mechanism port — the validation gate that retires the Alignment Phase.**

**FIRST action (block the rest of the stage on this): MOTION-DECOMPOSITION DIAGNOSTIC.** This re-sequences Stage A's deferred residual (ii) — the motion-source question — to the stage where it can actually be answered (motion is attributable only once Stage C's force-tracking is in place). Run a 4-quadrant decomposition across the canonical seed set, all with `PUSHA_REPOSITION_PWL=1`:

| Cell | Force-tracking (Stage C result) | Descent shape | What it isolates |
|---|---|---|---|
| FT-OFF + gentle-descent (current state) | OFF (legacy `-g_hat` derive) | pwl_speed=0.18 (current) | Baseline for the diagnostic — gentle landing + no force = current Stage A flag-ON behavior (box moves ~0-10 mm) |
| **FT-ON + gentle-descent** | ON (Stage C, `lambda_des = u_sol`) | pwl_speed=0.18 (current) | **THE DECISIVE TEST** — if motion is RESTORED to ≥ 20 mm → gentle landing was correct, motion always came from the executor → Stage A residual (ii) resolved as "brush-shove confirmed"; if motion stays ~0 → engagement-starvation real, descent needs late-deceleration shape |
| FT-OFF + fast-descent | OFF | pwl_speed=0.40 (reverted) | The OLD Stage A behavior — sanity check that the 25-29 mm motion reproduces with `-g_hat` derive at the buggy speed (confirms motion-source attribution) |
| FT-ON + fast-descent | ON | pwl_speed=0.40 (reverted) | Confirms FT-ON doesn't introduce a new pathology at the buggy speed; expected to retain motion AND retain the brush-induced |qz| tip (FT can't damp pre-c3 free-mode brush per the Stage A diagnostic) |

Decompose per cell: free-mode box motion (descent-phase only) vs post-landing c3-mode box motion. Attribution becomes possible because FT-ON gives the c3 phase a real push (separable from the descent-phase brush). Decision:
- FT-ON + gentle ≥ 20 mm and free-mode box motion ≈ 0 → gentle landing CORRECT, advance to the cumulative bar below.
- FT-ON + gentle < 20 mm and c3-mode tick count low → engagement starvation; iterate Stage A with late-deceleration descent shape (slow only the last ~30 mm) before clearing the cumulative bar.
- Other quadrants inform whether the brush-shove hypothesis was right.

This diagnostic is *gating* on Stage E: do not measure the cumulative bar below until the motion source is attributed.

**Pre-registered bar (HARD, motion-guarded, cumulative across Stages A+B(?)+C+D):**

| Condition | Threshold |
|---|---|
| |qy|_max | < 0.10 per seed across 12 s |
| |qz|_max | < 0.10 per seed across 12 s |
| Sustained contact | EE-BOX pair admitted at ≥ 60 % of c3-mode ticks per seed |
| **Goal-aligned motion** | **≥ 20 mm** per seed at t=12 s (this is the cumulative motion bar held back from Stages A-D) |
| Cross-seed | ≥ 3/4 seeds in {0, 1, 2, 4} pass all four conditions |
| Replicates | ≥ 2/3 replicates per seed (FP-noise floor per `feedback_serial_required_for_determinism`) |
| Regression check | Reference's `cube_turning` task equivalent in port shows no regression |

**On pass:** Alignment Phase complete. Resume the held research targets (the ADMM solver internals — §0 exception 2 — per the original investigation framing).

**BLOCKED ON THE CADENCE DISCRIMINATOR (RE-TAGGED 2026-06-23 from "C3+ iteration-scheme defect"; see §7.2 LCP-verification banking):** Stage E's cumulative-motion bar cannot be met until the cadence discriminator (componentwise @ 1 kHz vs LCP @ 100 Hz vs componentwise @ 100 Hz) resolves whether the 100Hz-PROVISIONAL projection-defect framing survives or dissolves. The LCP live-verification @ 100 Hz produced 34.18 mm of real box motion on seed 0 — **clearing the canonical Stage E ≥ 20 mm cumulative bar ON THIS SINGLE SEED** — but did NOT achieve formal convergence (pr 4.96 / dr 34.6 / converged 0/65 / in-pipeline λ_n wanders) and did NOT clear the gap-closed verdict (V1 20 % < 30 %, V2 6 % < 15 %), and it changed the projection without changing the cadence so the cause-attribution is unresolved. **Do NOT run {0, 4} multi-seed validation under either projection at 100 Hz expecting the verdict to clear** until the cadence discriminator resolves which axis is load-bearing. Stage E re-opens IFF the discriminator produces a clean fix (either projection or cadence) that converges in the closed loop.

---

### Stage F (LAST resort) — Cadence + multi-process

**Only run if Stages A-E don't suffice.** Reference uses LCM-coupled multi-process at ~1 kHz franka publish driving `LcmDrivenLoop` + `DeclareForcedDiscreteUpdateEvent`. Port uses single-process 100 Hz Python loop with `simulator.AdvanceTo`. **Cheapest divergence; LAST to address.** May be irrelevant if A-E pass — that's the whole point of chain-first ordering.

---

## 4. Inter-stage decision rule (the dead-planner false-positive guard)

After each stage:

| Outcome | What it means | Action |
|---|---|---|
| **Stage's LOCAL bar PASSED** (mechanism effect moved as predicted) | The port mechanism is wired and producing the predicted local change. | Advance to the next stage. Update §1 row(s) per the stage's "On pass" instruction. |
| **Stage's LOCAL bar FAILED, but the mechanism IS wired** (some predicted effect moved, just not enough to clear the bar) | Mechanism is necessary-but-insufficient at this stage. | Continue to the next stage; cumulative motion is held to Stage E. Note the under-target measurement in §1 row's status note. |
| **Stage's LOCAL bar FAILED AND the mechanism is NOT wired** (no behavior change vs baseline; |qy| ≈ 0 because the planner is dead, not because the lever was zeroed) | Port has a deeper integration bug. Five prior false positives — iter-8, Stage-2 lookahead, Stage-5 bounds, push-point alone, force-routing AT-CoM — all fit this category. | **STOP.** Do not chain another stage on top. Diagnose the wiring before continuing. |

The motion guard's whole purpose is to catch the third case. |qy|≈0 from a box that does not move is NEVER a pass — it's the most-frequent dead-planner artifact.

---

## 5. Plan boundaries (explicit)

- **In scope:** chain-first mechanism port from contact-free dispatch through contact-rich push at the cube substrate (Stages A-F).
- **Out of scope:** the uncommitted-probes disposition (P0, user's separate call); Stage-2d → main reconciliation if any probes get promoted; benchmarking against the reference's `anything` task numbers; deciding whether to migrate the cube to a different substrate; **anything inside the §0 exceptions** (the box-pushing target stays the goal; the ADMM solver internals stay the held research target).

---

## 6. Seed set

Canonical alignment seeds: **{0, 1, 2, 4}**. Seed-3 is **intentionally excluded** — it was used in the Q-series recovery / ablation runs (Q1 recovery, Q2 jitter/force-sweep) during the closed investigation, but the canonical 4-seed pre-registration in `docs/superpowers/plans/2026-06-18-port-cartesian-force-migration-map.md` § Stage 5 was {0, 1, 2, 4}, covering 4 mechanism-distinct failure modes (lateral-drift seeds 0/4 + plateau seed 1 + wrong-face seed 2). Adding seed-3 would be a Q-series recovery-from-bad-init coverage — not part of alignment validation.

If a stage wants seed-3 coverage for diagnostic reasons (not pass-bar), that's allowed but flagged as a Q-series add-on, not part of §3's pass criterion.

---

## 7. Maintenance discipline (the anti-stale rule — the whole point of this file)

**Every stage's definition of done includes flipping the relevant row in §1 and adding a one-line note to §3's "On pass" record.** Without this, the file becomes the next "NO CODE WRITTEN" — a memory record that drifts away from the codebase state and competes with current direction.

Concrete:
- When Stage A passes, §1 row 6 status updates OPEN → RECONCILED **in this commit/edit**, and row 9 follows the entry-gate firing-rate measurement.
- When Stage B is skip-justified, §1 row 2 gets the post-A measurement note **in this edit**.
- When Stage C reads W_ee_lambda, the read value goes into §3 Stage C's "FIRST action" subsection **before the rest of the stage runs**.
- When Stage E passes, this whole file is annotated "ALIGNMENT PHASE COMPLETE 20YY-MM-DD" with the seed/replicate results.

If you're reading this file and §1's status rows look outdated against the codebase, **trust the codebase, not this file** — and flag it as a maintenance failure of the stage that should have updated it.

---

### 7.1 — NOT the research boundary; isolable algorithmic defect (2026-06-23, CORRECTED from previous-block "BOUNDARY REACHED")

**The previous block (Probe B) recorded "BOUNDARY REACHED — the terminus IS the held §0 exception #2 research target." That recording is RETRACTED.** The disambiguation probe (`stage_c/admm_dump/seed0_full50.npz` + `scripts/_stage_c_admm_harness.py`) refuted the premise. Per the same §7 anti-stale discipline that motivated the boundary recording, this correction is banked here before any subsequent work proceeds.

**What the alignment-by-stage walk actually traced** — reposition lands the EE (Stage A, descent reference-aligned at `pwl_speed=0.18`); the executor is reference-exact and beats its setpoint by ~15 mm (Stage C executor probe); the planner never asks for contact (localization probe — `phi_pred_min ≥ 12 mm on 0/119 c3 ticks`); the planner cannot ask because its ADMM solve oscillates without finding the **demonstrably-existing** complementarity-feasible contact force (Probe B's `pr` median 4.95 / `dr` median 83.3 / `converged 0/119` reading STANDS — the non-convergence is real; what's REFUTED is the cause-attribution to §0 #2 modeling).

**The terminus is NOT the held §0 exception #2 research target.** The disambiguation probe proved:
- LCP at u=0 has a complementarity-feasible solution (`λ_n = 0.584 N`; `max|λ_i · w_i| = 6.94e-17`). Modeling REFUTED.
- E-matrix has 1 zero row (row 0, γ slack — a standard Stewart-Trinkle feature), NOT tangent rows. Deck's E-structure claim REFUTED on the live instance.
- Iter × ρ sweep: 0/16 cells converge across `max_iter ∈ {25, 100, 500, 1000}` × `ρ ∈ {100, 10, 1, 0.1}`. Layer-2 tuning REFUTED as sole gate.
- The deck's "ρ decided-but-never-applied" layer-1 framing also REFUTED — the rule fires correctly (100→50→25 in the live 25-iter call); convergence fails anyway.

**The actual gate is an isolable algorithmic defect in the C3+ iteration scheme** — the solver oscillates around a feasible point it cannot lock onto. Sub-hyps (1a) Bui componentwise projection bug; (1b) OSQP block construction bug; (1c) ρ-adaptation pathology.

**KEY ASSET — the disambiguation produced a KNOWN-CORRECT answer.** The brute-force LCP solution (`λ = [γ=0.146, λ_n=0.584 N, λ_t={0, 0.117, 0, 0.117}]`, `w` machine-precision-feasible) is now a **ground-truth ORACLE** for diagnosing the iteration scheme. Every prior probe in this chain INFERRED correctness from behavior; this one COMPUTED the target. That's a step-change in diagnostic capability.

**Implications for the Alignment Phase (CORRECTED from the previous-block status):**
- §1 row 2 (LCS admission, planner-side): blocked on the C3+ iteration-scheme defect, NOT on §0 #2 modeling.
- §1 row 3 (ADMM / C3+ solver): previous-block promotion (HELD-EXCEPTION → IMMEDIATE GATE) **RETRACTED**. Held exception STAYS HELD; the live blocker is the iteration-scheme defect, not the held research target.
- §1 rows 4 + 5 (control input u + executor): RECONCILED flip BLOCKED on the iteration-scheme defect, NOT on §0 #2 modeling.
- Stage E cumulative motion bar: blocked on the iteration-scheme defect (same correction).
- Sub-fork (a) ALIGNMENT (port `GetNClosestContactPairs`): still declined-as-silent-failure until the iteration scheme converges, but the gate is no longer "modeling research" — it is "isolate-the-defect."

**Next gate: the isolation probe.** Projection-swap — Aydinoglu per-contact Lorentz (the existing C3-not-plus path in `admm_solver.py`) vs Bui componentwise (current C3+) on the captured instance, diffed against the brute-force LCP oracle. If C3 Lorentz converges to the oracle where C3+ componentwise does not, (1a) is isolated. **ON-THESIS angle:** the deck framed C3+ componentwise as "4-5 orders faster" than C3 Lorentz; if C3 Lorentz converges where C3+ does not on this instance, the finding INVERTS speed-vs-correctness — a comparative claim that lands a research-quality result inside the Alignment Phase without invoking the held §0 #2 modeling target.

**Anti-stale binding:** any subsequent §3 stage entry that uses the previous block's "boundary-reached, research-gated on §0 #2 modeling" framing is operating on a STALE record. The current gate is the iteration-scheme defect (1a/1b/1c) — isolable, not fundamental.

---

### 7.2 — Live LCP verification: 34 mm motion AND a cadence confound (2026-06-23, CORRECTS the §7.1 framing)

**The LCP live-verification probe produced the first non-zero box motion of the investigation AND surfaced a cadence confound that re-tags §7.1's "iteration-scheme defect (1a/1b/1c)" framing.** Per the same anti-stale discipline, the result + the confound are banked here BEFORE the cadence discriminator runs.

#### Live result (LCP @ 100 Hz, seed 0, 12 s)

**First non-zero box motion of the investigation: 34.18 mm goal-aligned (vs 0.00 mm under componentwise), clearing the canonical Stage E ≥ 20 mm cumulative bar ON THIS SINGLE SEED.** Qualitative regime change vs Probe B's componentwise reading:
- V1 sustained-contact 5.04 % → **20.00 %** (4×).
- V3 F_z var 0.42 → **1.60** (3.8×); max 3.39 → **6.53** (1.9×).
- `phi_pred_min` median +29.5 mm → **+13.62 mm**; min +12.5 mm → **-2.91 mm**.
- `phi_pred_min < 2 mm`: **17 / 65** c3 ticks (26 %) — vs Probe B's 0 / 119 (0 %).
- `phi_pred_min < 0` (penetration predicted): **9 / 65** (14 %) — never happened under componentwise.
- `phi_act` actual EE-box: median +1.27 mm; min **-1.56 mm** (actual EE penetrates the box at deepest contact).
- **The planner now asks for contact and the box moves.**

The gap-closed VERDICT still FAILS (V1 20 % vs 30 %; V2 6.15 % vs 15 %) — but the SHAPE of every signal is unambiguously the right direction.

#### The CONFOUND (load-bearing — do NOT bank this run as the LCP-fix-confirmed conclusion)

**This run does NOT establish the LCP projection as the fix on two independent counts.**

**(a) Formal convergence STILL UNMET.** LCP @ 100 Hz pr_final median **4.96** ≈ componentwise's 4.95; dr 34.6; converged 0/65; in-pipeline `λ_n` wanders (median 0.70, max 3.76) rather than locking to the brute-force oracle 0.5839. **The offline machine-precision oracle-match (Cell B, diff 6.22e-8) did NOT translate to the live closed loop.** Primal-locks-dual-lags was the offline pattern; in the live loop both primal AND dual stay loose.

**(b) Projection-vs-cadence UNRESOLVED.** This run changed the PROJECTION (componentwise → LCP) but NOT the CADENCE (still 100 Hz). The three-way picture:

| Cell | Projection | Cadence | Box motion | Convergence |
|---|---|---|---|---|
| componentwise @ 100 Hz | componentwise | 100 Hz | 0 mm | NO (Probe B) |
| LCP @ 100 Hz | LCP-per-knot | 100 Hz | 34 mm | NO (this run) |
| componentwise @ 1 kHz | componentwise | 1 kHz | **UNTESTED** | **UNTESTED — this is the reference's actual config** |

**The 34 mm's CAUSE is less clear, not more.** It could be:
- LCP projection helping despite not converging (dr median 34.6 < componentwise's 83.3 — a 60 % drop alone may shift planner behavior), OR
- Cadence is the real missing ingredient and LCP @ 100 Hz only partially compensates, AND we still don't know whether plain componentwise @ 1 kHz would converge.

**The formal-non-convergence is the tell:** if LCP were the clean fix it should have converged in the closed loop as it did on the frozen instance — it did not. This is consistent with the real missing ingredient being the 1 kHz warm-start quality (state moves ~10× less per control tick → warm-start lands near previous solution → reference converges in iter=3).

#### Re-tagging the §7.1 ADMM/projection findings as 100Hz-PROVISIONAL

The isolation probe's "componentwise oscillates / LCP-per-knot finds the oracle" was measured on a **100Hz-STALE warm-start** (the dump was captured live at 100 Hz). It never cleanly separated *"the componentwise projection is buggy"* from *"the warm-start is starved by the slow cadence."*

- **The componentwise-projection-bug finding is 100Hz-PROVISIONAL**, pending the 1 kHz re-measure, and MAY DISSOLVE if componentwise @ 1 kHz converges cleanly.
- **The `--c3plus-projection=lcp` candidate fix is held PROVISIONAL, not confirmed.** The live verification produced the strongest qualitative signal yet (34 mm vs 0 mm) but did NOT close formal convergence and did NOT clear the gap-closed verdict.

#### LCP-verification status

**PARTIAL** — real 34 mm motion (Stage E ≥ 20 mm bar cleared on seed 0), but INCOMPLETE convergence (formal pr/dr unmet, λ wanders) AND CADENCE-CONFOUNDED (projection changed, cadence not; projection-vs-cadence unresolved). Gap-closed verdict still FAILS (V1 20 % vs 30 %, V2 6 % vs 15 %).

#### Holds (all downstream steps PAUSED pending cadence discriminator)

- Seed-4 reproduction at 100 Hz — HELD.
- `main.py:248` default flip `componentwise → lcp` — HELD.
- {0, 4} multi-seed validation — HELD.
- §1 rows 4 / 5 RECONCILED flip — HELD.
- Stage D advance / N-closest port (sub-fork (a)) — HELD.

These are NOT abandoned — they are PAUSED until the cadence discriminator resolves whether the LCP path is the real fix or a symptom-treatment. Spending them at 100 Hz on the unconfirmed projection burns the validation budget on a confounded baseline.

#### SURVIVES vs RE-BASES (precise scoping of what carries forward)

**SURVIVES (cadence-independent — these stand regardless of the 1 kHz outcome):**
- The brute-force LCP oracle (`λ_n = 0.5839 N`, machine-precision feasible at the captured `seed0_full50.npz` knot-0 LCP at u=0) — per-instance math, will remain the ground-truth for whatever instances arise at 1 kHz.
- §0 #2 modeling REFUTATION (LCP at u=0 admits a feasible point).
- E-matrix structure observation (1 zero row = γ slack, not tangent rows).
- Stage A reposition / Stage C executor mechanism findings (reference-EXACT, executor outperforms its setpoint by ~15 mm).
- The instrumentation (`[FORCE-ROUTE]`, `[SETPOINT]`, `[CONSISTENCY]`, `[ADMM-DUMP]` gates) and the offline harness.

**RE-BASES (all measured at 100 Hz — re-measure at 1 kHz):**
- All `phi_pred_min` / `phi_pred1` distributions.
- All live convergence numbers (pr/dr, iters/max, converged frac).
- The four-link causal-chain verification.
- In-pipeline `λ_n_max` vs oracle distance.
- The gap-closed V1/V2/V3/V4 fractions.

#### Anti-stale binding

Any subsequent entry that treats "`--c3plus-projection=lcp` is the confirmed fix" is operating on a PROVISIONAL record until the 1 kHz cadence discriminator lands. Any entry that treats "the iteration-scheme defect (1a/1b/1c) is the gate" is operating on a 100Hz-STALE record until the cadence discriminator runs.

**Next gate: the cadence discriminator.** Run `componentwise @ 1 kHz` (the reference's actual config) — does the live ADMM now formally converge under unchanged projection? If YES → cadence/warm-start was the gate; the projection-defect finding dissolves; LCP becomes a 100Hz-symptom-treatment. If NO → componentwise is genuinely defective regardless of cadence; LCP becomes the real candidate fix and earns the multi-seed validation. NOT actioned in this plan-doc edit — next block's work.

---

### 7.3 — Cadence discriminator STOP-AT-SCOPE: tick-vs-sim-time semantics is a separate alignment target (2026-06-24)

**The 1 kHz cadence smoke STOP-AT-SCOPE'd: raising the rate exposed that the dispatcher never enters c3 mode because `sample_buffer_lifetime` is tick-counted. The decision is to RECONCILE the tick-vs-sim-time semantics PROPERLY (sim-time conversion) before running the discriminator — NOT a scaled-gate hack.** Per the same §7 anti-stale discipline, the finding + the direction are banked here before the reconciliation plan is authored.

#### The tick-coupling finding (from the scope-stop)

The 1 kHz cadence smoke (commit `02abed9`, `PUSHA_CONTROL_HZ` gate; smoke run committed at `308f188`) confirmed the rate flips cleanly: 3001 STAGE-A-TRACE steps in 3 s = 10× the 100 Hz rate; the EE physically reached the correct φ ≈ 4.98 mm (matching the 100 Hz baseline φ ≈ 4.94 mm at sim_t = 1.73 s — actual landing geometry preserved). BUT the dispatcher **NEVER ENTERED c3 mode**: `[GS-perf] switches=0`, `finished_repos=0` throughout.

**ROOT CAUSE:** `sample_buffer_lifetime = 30` (`config/sampling_c3_kik.yaml:188` + `params.py:291`) is measured in CONTROL LOOPS, not sim-time.
- At 100 Hz: `30 ticks × 10 ms = 300 ms` between sample-buffer refreshes → the PWL trajectory (~1.4 s) substantially completes before a new sample target is proposed → Stage A works.
- At 1 kHz: `30 ticks × 1 ms = 30 ms` between sample-buffer refreshes → the dispatcher picks a NEW `_current_repos_target` every 30 ms → the rebuild gate (`sampling_based_c3_controller.py:2643`, target-moved > 5 mm) fires → a fresh 1.4 s PWL rebuilds before the previous can land.
- Smoke log evidence: `[STAGE-A-PWL]` rebuilds at step = 1, 31, 61 with different p_targets each time (`(-0.023,-0.080)`, `(+0.046,+0.080)`, `(+0.080,0)`).

The reposition-isolation precondition (EE lands the same regardless of rate) is **STRUCTURALLY VIOLATED** by this tick-coupling. **This is at least the THIRD tick-coupled constant** — after the `dt_ctrl`/`dt_osc` coupling at `main.py:511` (T-arch Stage-2d) and the 5-tick contact-loss disengage at `sampling_based_c3_controller.py:237` ("5 no-EE-BOX ticks" = 5 ms @ 1 kHz vs 50 ms @ 100 Hz). An audit (next block) will enumerate the rest.

#### Direction decision

**Reconcile the tick-vs-sim-time cadence semantics PROPERLY** (convert the tick-counted constants to sim-time durations matched to the reference's sim-time cadence) **BEFORE running the cadence discriminator**.

**NOT the scaled-gate probe-hack** (auto-scaling `sample_buffer_lifetime` by `100 / control_hz` at construction): that papers over the tick coupling and confounds future readers — it leaves "lifetime in control loops" still meaning "30 ticks", just scaled at construction, which masks the alignment delta rather than resolving it.

**Rationale.** The tick-vs-sim-time divergence is a GENUINE port-vs-reference alignment target: the reference decouples cadence by sim-time across its multi-process LCM pipeline; the port hardcodes loop counts. The sim-time conversion is alignment work that **SURVIVES IN EITHER discriminator branch** — even if the discriminator later shows the projection is the bug and cadence is not load-bearing for convergence, the port should still match the reference's sim-time cadence. Deferring the reconciliation does not save budget; it delays an inevitable alignment step and leaves the discriminator on a hacked base. **The discriminator runs on the RECONCILED base.**

#### §1 row 8 (Entry & cadence) — REVISED mechanism content

The row is no longer "a single rate flag". Row 8 has TWO sub-divergences (now reflected in the row cell):

- **(a) Tick-vs-sim-time SEMANTICS.** Multiple constants hardcoded in control-loop counts (`sample_buffer_lifetime`, the 5-tick contact-loss disengage, the rebuild gate, the dt_ctrl/dt_osc coupling, possibly more — to be enumerated by the audit) which make ALL rate-dependent behavior diverge at any rate ≠ 100 Hz. **Being reconciled now via the tick → sim-time conversion.**
- **(b) RATE + ARCHITECTURE.** 1 kHz multi-process LCM vs 100 Hz single-loop. Separate; resolved by the cadence discriminator (rate-on-warm-start, sub-question of (b)) + later Stage F (multi-process, sub-question of (b)).

**Reconciling (a) does NOT reconcile (b).** The row stays ACTIVE FRONT until both land.

#### Status

- Row 8 (Entry & cadence): **ACTIVE FRONT** — both sub-divergences still open.
- Sub-divergence (a) tick→sim-time semantics: **IN PROGRESS** — reconciliation plan authored next; implementation + verification follow.
- Sub-divergence (b) rate/architecture + the cadence discriminator: **DEFERRED** until (a) lands. The discriminator's reposition-isolation guard requires the sim-time reconciliation to hold first; running the discriminator now would just reproduce the c3-never-entered scope-stop reading.

#### Holds — discriminator ADDED to the §7.2 hold list

The §7.2 holds (seed-4 reproduction, `main.py:248` default flip, {0, 4} validation, §1 rows 4 / 5 RECONCILED flip, Stage D advance / N-closest port) STILL PAUSED. **NEW HOLD: the cadence discriminator (componentwise @ 1 kHz) is itself NOW ALSO gated behind the sim-time reconciliation** — it cannot run cleanly until c3 engages at 1 kHz with the reposition landing the same.

#### SURVIVES vs RE-BASES — addition to §7.2 SURVIVES

**ADD to §7.2 SURVIVES (cadence-independent):** *the tick → sim-time reconciliation itself, once implemented + verified, is a cadence-INDEPENDENT alignment win — it survives regardless of the discriminator outcome (it reconciles a real port-vs-reference divergence in cadence SEMANTICS, separate from the rate question).*

**RE-BASES unchanged** — all 100 Hz convergence / phi numbers re-measure at 1 kHz on the reconciled base.

#### Anti-stale binding

Any subsequent entry that treats "the cadence discriminator is the next gate" without first landing the sim-time reconciliation is operating on a STALE record — the cadence-discriminator-as-immediate-next-gate framing (from §7.2's "Next gate" line) is REVISED: the sim-time reconciliation is the next gate; the cadence discriminator runs on the reconciled base.

**Next gate (corrected from §7.2):** the sim-time reconciliation plan + implementation + verification. Then the cadence discriminator. NOT actioned in this plan-doc edit — the reconciliation plan opens in the next block.

---

### 7.4 — Reconciliation plan audit findings (2026-06-25, banked WITH the implementation block)

**The reconciliation plan (`docs/superpowers/plans/2026-06-25-tick-to-simtime-cadence-reconciliation.md`) was REVIEWED + GREENLIT with two label adjustments.** The audit surfaced findings that don't fit "reconciliation"; they are banked here so the §1 row 8 update + the reconciliation plan label honestly reflect what the conversion does and does NOT close.

**Adjustment 1 — Class B is NOT "reconciled"; it is "RATE-INDEPENDENCE-ONLY".** B1-B11 makes port-only machinery rate-independent. It does NOT reconcile to the reference: the reference has NO contact-loss disengage counter (`grep "consecutive|streak|no_contact|disengage|hard_cap"` on `sampling_based_c3_controller.cc` = 0 hits); the reference's sample buffer is event-driven via LCM output ports (`sampling_based_c3_controller.cc:358-388`), not a tick-aged Python structure; `TARGET_STABLE_TICKS` is port-only. **Status: Class B converts to seconds for rate-independence; alignment-status-OPEN** (real question for Class B: *does the port need this machinery at all?* — candidate band-aids of the same kind as the already-retired contact-proximity entry-gate, row 9 RETIRED). Bank OPEN; do NOT silently close it.

**Adjustment 2 — A3 LOCATED (not held as unverified).** Targeted grep found A3 on the reference: `examples/sampling_c3/anything/parameters/progress_params_c3plus.yaml:45 progress_enforced_over_n_loops: 16` = 16 ms @ 1 kHz. Port: 30 ticks @ 100 Hz = 300 ms — **19× longer in sim-time** than reference. This pass PRESERVES the port's 300 ms; adopting the reference's 16 ms is a separate later alignment decision.

**Banked finding (i) — A1/A2/A3 100× / 60× / 19× dispatch-TIMING gap.**

| Constant | Port @ 100 Hz | Reference @ 1 kHz (anything/c3plus) | Sim-time gap |
|---|---|---|---|
| A1 `num_control_loops_to_wait` | 60 ticks = 600 ms | 5 ticks = 5 ms | **120×** |
| A2 `num_control_loops_to_wait_position` | 30 ticks = 300 ms | 5 ticks = 5 ms | **60×** |
| A3 `progress_enforced_over_n_loops` | 30 ticks = 300 ms | 16 ticks = 16 ms | **19×** |

Real dispatch-TIMING divergence, separate from the cost/hysteresis decision logic (row 1 RECONCILED). Alignment OPEN. Adopting the reference's 5/5/16 ms values requires its own measurement (the reference dispatcher fires far more aggressively). PRESERVED at the port's 100 ms-equivalent on this pass.

**Banked finding (ii) — Disengage family + tick-aged sample buffer NO reference analog.** B3-B7 contact-loss thresholds, B1 sample_buffer_lifetime, B10 TARGET_STABLE_TICKS — port-only candidate band-aids of the same kind as the retired contact-proximity entry-gate. The conversion makes them rate-independent so the cadence discriminator can run; it does NOT decide whether they should exist. Alignment-status-OPEN.

**Scope of §1 row 8 sub-divergence (a) flip on conversion-pass:** narrows to "tick→sim-time semantics RATE-INDEPENDENT" — explicitly **NOT** "Class B reconciled to reference" AND **NOT** "Class A dispatch-timing reconciled." Both Class A (sim-time interval values) AND Class B (whether the machinery should exist) stay OPEN. Substantive alignment questions are FOLLOW-ON.

---

### 7.5 — Tick→sim-time conversion: SMOKE 1 byte-equivalence PASS, primary scope-stop RESOLVED, audit's SECOND incompleteness (grep is blind to behavioral couplings) (2026-06-25)

**The constant-level tick→sim-time conversion landed and is byte-equivalent at 100 Hz. The primary scope-stop (sample_buffer_lifetime tick-coupling) is resolved at 1 kHz. But SMOKE 2 failed 0/4 — NOT because the conversion failed, but because the audit MISSED a fourth coupling that is BEHAVIORAL (call-frequency) rather than constant-numbered, and grep is structurally blind to that class.** Banked per §7 before the next-block trace probe.

#### (1) SMOKE 1 SUCCESS — the regression gate (byte-equivalent at 100 Hz)

| Pass bar | Baseline | Candidate | Δ | Result |
|---|---|---|---|---|
| first_c3_step ±2 | 173 | 173 | **0** | ✓ |
| first_c3_phi ±0.1 mm | +4.940 mm | +4.940 mm | **0.000 mm** | ✓ |
| switches ±2 | 36 | 36 | **0** | ✓ |
| mode_match_rate ≥95% | — | **1201/1201 = 1.0000** | PERFECT tick-by-tick | ✓ |
| final_obj_xy ±0.5 mm | (0, 0) | (0, 0) | 0.000 mm | ✓ |

The 4 `[YAML-COMPAT]` shims fired (legacy int fields auto-converted to `_s` seconds).

*NOTE on baseline correction:* initial SMOKE-1 attempt failed against `stage_a_speed018/seed0/run.log` because THAT baseline lacked `PUSHA_FORCE_ROUTING=u_sol` — WRONG baseline; corrected to `stage_c/seed0_usol_100.log` (the Phase-1 same-config pre-conversion run), against which it is perfect. The multi-file conversion (`params.py`, `progress.py`, `sampling_based_c3_controller.py` + the comparator + the YAML unit test, 6/6 pass) is **SOUND and REVERSIBLE**.

#### (2) PRIMARY SCOPE-STOP RESOLVED

The `sample_buffer_lifetime` tick-coupling that blocked the first 1 kHz attempt is **FIXED**: at 1 kHz the PWL now builds at steps 1 / 301 / 601 = 300 ms wall-time, identical to 30 ticks / 300 ms at 100 Hz. The 30-tick churn that prevented c3 entry is **gone**. The sim-time conversion did its job for the constant it targeted.

#### (3) SMOKE 2 FAILED 0/4 — a NEW, FOURTH tick-coupling (NOT one of the 15 converted)

SMOKE 2 (1 kHz reconciliation) failed 0/4, but NOT because the conversion failed — because a **FOURTH** tick-coupling surfaces near EE landing. Starting at step ~1630 (sim_t 1.63 s, just before the expected c3 entry at step ~1700), the PWL rebuild gate (`sampling_based_c3_controller.py:2643`, target-moved-5mm) fires PER-TICK (1630, 1631, 1632, 1633, ...), so `finished_repos` never latches and c3 never engages.

**Most likely mechanism:** `_refresh_buffer_on_arrival` (line 427) → next-target selection produces a NEW target every tick once the EE is in proximity, tripping the 5 mm rebuild gate. To be PINNED by the next-block trace probe; not yet confirmed.

#### (4) THE AUDIT'S SECOND INCOMPLETENESS (methodological finding)

The cadence audit enumerated **15 tick-counted CONSTANTS** but MISSED this coupling — because it is NOT a constant (there is no integer field to grep), it is a **BEHAVIORAL / call-frequency coupling** (a refresh call firing per-tick under a proximity condition). **Grep-based enumeration is STRUCTURALLY BLIND to this class.** A runtime trace caught it instantly.

**METHODOLOGICAL CORRECTION (binding on future cadence-coupling enumeration):** cadence-coupling enumeration must use **RUNTIME TRACE, not grep**, for behavioral/call-frequency couplings. The constant-grep finds integer-field couplings (15 in Class A/B/C); it cannot see call-frequency couplings (≥1 found by trace). This is the audit's **SECOND incompleteness** — the first was "3 known constants → 15 enumerated by grep"; the second is "0 behavioral couplings found by grep, ≥1 found by trace."

#### (5) RECONCILIATION STATUS

The tick→sim-time SEMANTICS conversion (the 15 constants) is **DONE and 100Hz-byte-verified**. **§1 row 8 sub-divergence (a) does NOT flip RECONCILED yet** — the 1 kHz reposition-isolation guard is STILL UNMET (the secondary per-tick rebuild storm blocks c3 at 1 kHz). Sub-(a) stays **IN PROGRESS**: *"constant-level tick→sim-time conversion landed + 100Hz-verified; ≥1 behavioral coupling in the EE-landing chain remains, blocking 1 kHz c3 engagement."* The cadence discriminator stays gated behind the full reconciliation.

#### (6) CONVERGENCE SHAPE (the positive read)

Each 1 kHz attempt fails **LATER**: the first failed at ~30 ms (sample buffer; commit `308f188`), this one reaches step ~1630 — **JUST before the expected c3 entry at ~1700** — before the next coupling bites. Failing later = each fix is real, the remaining surface is shrinking and **LOCALIZED** to the EE-landing approach chain (refresh-on-arrival → next-target → rebuild gate). This is one chain revealing its couplings in sequence as the EE nears contact, **not** whack-a-mole across the system — a tractable shape.

#### (7) §7.4 confirm + NEW finding + DEFERRED constant

**Confirmed banked in §7.4 (still standing):**
- A3 LOCATED on the reference (`progress_params_c3plus.yaml:45 = 16` ticks @ 1 kHz = 16 ms; port preserves 300 ms = 19× sim-time gap, alignment OPEN).
- A1/A2/A3 dispatch-timing gaps (100×/60×/19×, alignment OPEN, separate question).
- Disengage family + tick-aged sample buffer NO reference analog (Class B alignment-status-OPEN).

**NEW (this run):** the per-tick PWL rebuild storm at EE-landing approach (the BEHAVIORAL coupling the audit missed) — banked; mechanism to be pinned by the next-block trace probe.

**DEFERRED (recorded so it is not lost):** B10/B11 `TARGET_STABLE_TICKS` in the legacy IK tracker (`reposition_ik.py:709`) — bypassed by `PUSHA_REPOSITION_PWL=1`, so does NOT bite under the current Stage A flag-ON regime. Will need conversion if the legacy path is ever re-enabled.

#### Anti-stale binding

Any subsequent entry that treats "the tick-vs-sim-time semantics sub-divergence is RECONCILED" without first landing the behavioral-coupling fix is operating on a stale record. The next gate is **the trace probe to pin the behavioral coupling** in the `_refresh_buffer_on_arrival` → next-target → rebuild chain. After the behavioral coupling is identified and fixed AND SMOKE 2 passes at 1 kHz, sub-divergence (a) can flip RECONCILED and the cadence discriminator runs.

**Next gate (corrected from §7.4):** the trace probe to pin the behavioral tick-coupling, then a focused fix commit, then re-run SMOKE 2. NOT actioned in this plan-doc edit.

#### Update 2026-06-25 — MECHANISM PINNED + BUNDLED FIX (FIX-A + FIX-B(2))

The runtime trace pinned **TWO independent couplings** in the EE-landing chain (trace artifacts on commits `580c716` + `2e57e78`). Both are addressed in a single small commit landing WITH this banking note.

**COUPLING 1 — THE STORM (items 1-7; level-trigger bug):** the refresh at `sampling_based_c3_controller.py:1977`
```python
if self._last_repos_finished:
    self._refresh_buffer_on_arrival()
```
is LEVEL-triggered — fires every tick while `_last_repos_finished == True` (**67 fires / 152 ticks in the window**). Combined with `finished_reposition_cost = 1e9` inflating `prev_repos`'s cost, the argmin is FORCED to pick a fresh random strategy sample each tick (137-167 mm away from the previous target) → PWL rebuilds → `t_end` extends (`1.6523 → 3.7020 → ...`) → self-sustaining.

**COUPLING 2 — THE DEADLOCK (item 9, INDEPENDENT; finished-semantics divergence):** `_last_repos_finished` (set at `:1972`, DISTANCE-ONLY, the tracker) and `PWL.is_finished()` (consumed at `:1072`, TIME+DISTANCE, the dispatcher) DIVERGE. At step 1629 the tracker says "arrived" (4.9 mm < 5 mm); the dispatcher says "not arrived" (sim_t 1.629 < t_end 1.652). The two halves of the dispatch chain disagree about whether the reposition finished, so **no mode flip to c3 even with the storm gone**.

**Why both fix together:** fixing only COUPLING 1 stops the storm but does NOT engage c3 (COUPLING 2 blocks the mode flip independently). The trace enumeration surfaced this second head BEFORE the fix, avoiding fix-the-storm → re-discover-the-deadlock.

**FIX-A (edge-latch the refresh):** add a `_arrival_handled` sentinel; fire on False→True transition only; reset on False so a re-arrival re-fires. Edits at `:1977`.

**FIX-B(2) (unify the finished-criterion):** under the PWL path, compute `_last_repos_finished` from `PWL.is_finished(sim_t, ee_now, tol=0.005)` — the SAME criterion the dispatcher uses at `:1072`. Legacy IK tracker path preserved unchanged. **RECONCILIATION not strictness:** the tracker was wired to a WEAKER (distance-only) criterion than the dispatcher already consumes (time+distance); B2 makes them agree on the criterion the dispatcher was designed around. **B1 (distance-only for both) would be WRONG** — declares arrival while PWL velocity is non-zero. Edits at `:1972`.

**Verification gate (BOTH smokes):** the fix touches LIVE dispatch logic — the 100 Hz byte-equivalence gate applies. SMOKE 1 (100 Hz no-op) must clear 5/5 against `stage_c/seed0_usol_100.log`. SMOKE 2 (1 kHz, short, to ~step 1750) must clear 4/4 AND surface the explicit INTERMEDIATE READ: with the storm gone (FIX-A) and the criteria unified (FIX-B), at step ~1652 sim_t reaches the original t_end → `is_finished` should fire → mode flips to c3. **If `is_finished` does NOT fire even with stable t_end → THIRD coupling; trace remains on HEAD (re-enable `PUSHA_LANDING_TRACE=1`) and we report the ACTUAL mechanism, not infer from null.**

---

## 8. Memory pointer

This file (`docs/superpowers/plans/2026-06-23-alignment-phase-plan.md`) is the **canonical source-of-truth** for the Alignment Phase plan + conformance state. It is updated as each stage / flagged item resolves, per §7. Future alignment plans index against this file; they do NOT re-derive from the logic trees at `/d/projects/ERL/push_anything_ADMM/understand_logic_tree/{reference,port}/`.

Memory record `project_canonical_alignment_plan.md` in `/root/.claude/projects/-root-push-anything-ADMM/memory/` points to this file. The logic trees are FROZEN snapshots from 2026-06-22/23; this file tracks LIVE state.

---

## 9. Ratification record (2026-06-23)

The four review items folded in:

| # | Item | Resolution location |
|---|---|---|
| 1 | Stage A bar = LOCAL mechanism effect; 20 mm motion held CUMULATIVE to Stage E | §3 Stage A "LOCAL pass bar" + §4 inter-stage rule + the per-stage-vs-cumulative resolution stated plan-wide |
| 2 | Stage B = MEASURE-THEN-DECIDE; conditional bar requires λ_n > 0 (present-and-dead = fail) | §3 Stage B "Decision-first protocol" + "LOCAL pass bar" item 2 |
| 3 | Stage C FIRST action = READ W_ee_lambda; don't run until set | §3 Stage C "FIRST action" |
| 4 | TWO deliberate exceptions to full conformance: (1) box-pushing task, (2) ADMM solver internals | §0 (top-level property) |
| 5 | seed-3 intentionally excluded from {0, 1, 2, 4} | §6 |

**Status:** ratified. Next gate is P0 (uncommitted-probe disposition), the user's separate call. No stage has begun.
