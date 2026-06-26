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
| 3 | ADMM / C3+ solver | **(UPDATED 2026-06-25 — see §7.9) Leading-candidate-fix SHIFTED again: SOLVER-level convergence is NOT the root. Contact-model alignment (LCS missing box-ground) FIRST. Model-plant consistency test shows the LCS-with-oracle predicts box motion Drake does NOT render (1.7e7× mismatch).** | `admm_solver.py:_solve_c3plus`, `lcs_formulator.py` LCS construction | The non-convergence is real (DIAGNOSTIC 1(c)), but the deeper fact is: the captured LCS has `n_lambda = 6` = 1 EE-BOX contact only, NO box-ground. Solving any LCS-consistent λ, including the brute-force oracle, predicts the box FALLS (LCS lacks the floor). Reference uses contact_model: anitescu + 12 object-ground contacts always; port uses Stewart-Trinkle + LCS_EXPLICIT_BOX_GND OFF by default. Next gate: contact-model alignment, then re-test consistency. Reference-settings precondition is MOOT until LCS matches plant. |
| 4 | Control input u | PARTIAL — wired but not default; **mechanism probe-confirmed reference-EXACT** (Stage C probe 2026-06-23; see §3 Stage C outcome) | `wrapper._derive_force_command` (`-g_hat`+mag, env-gated `PUSHA_FORCE_ROUTING=u_sol` for u_seq[0]) ↔ `sampling_based_c3_controller.cc:1822-1832` (`force_samples = u_sol[i]`) | **RECONCILED flip BLOCKED on the cadence discriminator (RE-TAGGED 2026-06-23 from "iteration-scheme defect 1a/1b/1c"; see row 8 promotion + §7.2).** The mechanism is reference-exact; the row-flip gate is gap-closing / Stage E motion-bar. The LCP live-verification at 100 Hz produced 34 mm of real motion but not the gap-closed verdict; whether the projection is the fix OR cadence is the cause remains unresolved. |
| 5 | Executor (OSC + force-tracking) | PARTIAL — **mechanism probe-confirmed reference-EXACT**; Reading 2 (executor/compliance bottleneck) REFUTED (phi_act < setpoint_sd on 119/119 — executor BEATS its own commanded position by ~15 mm) (Stage C probe 2026-06-23) | `osc/qp_builder.py:73` + `params.W_force=100.0` ↔ `franka_osc_controller.cc:167-170` + `osc_params.W_ee_lambda = I_3` (scalar 1.0; port W_force/W_track ratio 100/100 = reference's 1/1 ratio preserved) | **RECONCILED flip BLOCKED on the cadence discriminator (RE-TAGGED 2026-06-23 from "iteration-scheme defect 1a/1b/1c"; see row 8 promotion + §7.2).** Same gate as row 4. |
| 6 | Reposition mechanism | **PARTIAL — wired (descent reference-aligned); residuals deferred to Stage E** | `reposition_trajectory.py` + `sampling_based_c3_controller.py:2502-2528` (gated PWL path; default OFF) ↔ `Reposition(...) + UpdateRepositioningExecutionTrajectory + LcmTrajectoryReceiver` (see Stage A outcome subsection at end of §3 Stage A) | Stage E motion-decomp + force-tracking confirm residuals (NOT this stage) |
| 7 | Push-point height computation | OPEN | `config/tasks.yaml:22 pushing.sampling_height = 0.03` (hand-coded) ↔ `sampling_params.yaml:64 z_height` auto-generated per object | Stage D passes |
| 8 | Entry cadence + multi-process | **(UPDATED 2026-06-25 post-diagnostics — see §7.6) constant-level tick→sim-time conversion STAYS (byte-equivalent at 100 Hz, real alignment work). Storm fix DEFERRED — not on the no-push critical path. Cadence-discriminator MOOT for the no-push (the no-push is projection, cadence-independent).** | (a) tick-vs-sim-time semantics — CONSTANT-LEVEL conversion KEPT (15 constants → seconds, 100Hz-byte-equivalent, separate commits proven). BEHAVIORAL coupling (storm at EE-landing) — pinned mechanism (level-vs-edge refresh at `:1977`), DEFERRED — it is a 1 kHz c3-blocker, but at 100 Hz the 6-tick window self-corrects (DIAGNOSTIC 1(a); storm thrashes target then cycles back). FIX-A+FIX-B(2) bundle REVERTED — see §7.6. (b) RATE + ARCHITECTURE — `main.py:571` + single-process loop ↔ `LcmDrivenLoop` 3-process LCM-coupled. | (a)' constant-level: DONE, RECONCILED-as-rate-independent (NOT no-push, NOT Class B alignment). (a)" storm fix: DEFERRED until 1 kHz or multi-seed work needs it. (b) discriminator + Stage F multi-process — MOOT for no-push; Stage F still independent (multi-process architecture is separate from cadence). Row no longer the critical front; the no-push critical front is row 3. |
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

### 7.6 — Two diagnostics RESOLVED the fork: EE well-placed; no-push = Probe B at 100 Hz; REVERT ccb71f5; preserve the conversion (2026-06-25)

**The two diagnostics from §7.5's outstanding "what does SMOKE 1's regression mean" question ran and are decisive.** They overturn the prior block's "regression-is-improvement" reading. Per the §7 anti-stale discipline, banked here BEFORE the revert block executes.

#### (1) FORK RESOLVED — EE WELL-PLACED at c3 entry; landing-bug REFUTED at 100 Hz

**DIAGNOSTIC 1(b)** (pre-fix HEAD `580c716` at 100 Hz, seed 0, 2.5 s): at c3 entry (step 173) `ee_now = [+0.0799, -0.0002, +0.0354]` — east face of the box, centered, valid push height; phi = 4.94 mm. The 100 Hz storm thrashed targets for 6 ticks (164-170) then ACCIDENTALLY cycled BACK to the original target `[+0.080, 0, +0.030]` by step 171 — the EE landed correctly. **The EE-MISPOSITIONING hypothesis is REFUTED:** the storm did NOT misposition the EE at 100 Hz (the 6-tick window self-corrects; at 1 kHz the 152-tick window does not, which is why 1 kHz blocked c3). **The 100 Hz no-push is NOT a landing bug.**

#### (2) THE NO-PUSH = PROBE B componentwise non-convergence, CONFIRMED at 100 Hz

**DIAGNOSTIC 1(c)** — with the EE well-placed and c3 engaged, the ADMM does NOT converge:
- pr_final median **4.94**, dr_final median **80** (tol = 1e-3)
- `iters 25/25` every solve; `converged = 0 / 31` c3 ticks
- `u_dot_box = -0.86` (force points away from box — recoil convention)
- `ee_step1_dot_box = -0.98` (**planner predicts the EE RETREATING**)
- `box_total_xy` PREDICTED median 234 mm per horizon (fictional) / ACTUAL **0 mm**
- `lam_n_ee_box = NaN every tick` (LCS never admits the EE-BOX pair)
- phi closes 4.94 → 2.79 mm over 8 c3 ticks (EE keeps approaching despite retreating setpoint)

**This is the SAME componentwise non-convergence the 1 kHz scope-stop and the LCP-isolation probe both pinned (Probe B).** It manifests at 100 Hz too: EE well-placed, c3 engages, ADMM produces a non-converged solution whose 234 mm predicted box motion is fictional while reality renders 0 mm. **The 100 Hz no-push IS the projection/convergence question.**

#### (3) CONFOUND COLLAPSED — un-provisionalizes the projection finding

This UN-PROVISIONALS the projection-defect attribution that §7.5 marked "100Hz-PROVISIONAL" pending the 1 kHz re-measure: **the no-push is now CONFIRMED to be componentwise non-convergence at BOTH rates, EE well-placed, CADENCE-INDEPENDENT**. The 100Hz-vs-1kHz confound that has distorted the projection question since the LCP-live-verification 34 mm run is **COLLAPSED**: cadence was NEVER the cause of the no-push. The original cadence-discriminator question (does cadence fix convergence) is **MOOT for the no-push** — the no-push is the projection, not cadence.

#### (4) FIX-A is a REGRESSION; the bundle's 28 mm is an ARTIFACT

**DIAGNOSTIC 2** (temp branch from `ccb71f5`, FIX-B reverted, FIX-A only, 100 Hz, 2.5 s): `switches = 0`. **FIX-A ALONE BREAKS c3 engagement at 100 Hz.** With the refresh edge-latched, the FIRST random storm target (`[-0.0316, +0.0800, +0.0300]`, 137 mm away from the box) is FROZEN as the target permanently → the EE travels toward this wrong target → c3 never fires. The pre-fix per-tick storm was an ACCIDENTAL noisy correction (thrashed targets until cycling back to `[+0.080, 0, +0.030]` by luck at step 171); FIX-A removes the storm AND the accidental correction → **WORSE than the buggy baseline at 100 Hz**.

**FIX-B(2) carries the entire 100 Hz behavior change** in the `ccb71f5` bundle — by changing WHEN `_last_repos_finished` latches (dispatcher STATE), not the projection. The bundle's 28 mm at 100 Hz is a **DOWNSTREAM consequence** of FIX-B's state-trajectory change interacting with the NON-CONVERGENT ADMM. **28 mm of real motion from a solver that predicts 234 mm and never converges is NOISE that happened to point the right way on one seed — NOT a working push, NOT a no-push fix.**

#### (5) FAIR CREDIT (both true)

The prior block's read was RIGHT on one narrow point and WRONG on the important one:
- **RIGHT:** the storm IS a real bug (level-vs-edge refresh trigger at `:1977`), so strict byte-equivalence to the buggy 100 Hz baseline is the wrong gate FOR THE STORM FIX.
- **WRONG:** the fix does NOT address the no-push (the no-push is Probe B non-convergence, which persists; the 28 mm is artifact).

Both hold simultaneously.

#### (6) DECISION — REVERT ccb71f5; preserve the conversion + findings

**REVERT `ccb71f5` (FIX-A + FIX-B(2)).** Rationale: FIX-A alone regresses (DIAGNOSTIC 2); FIX-B(2) alone was not isolated; the bundle's 28 mm is a non-convergent-ADMM artifact, not a working push. The bundle is not on the no-push critical path.

**PRESERVE:**
- The tick→sim-time conversion (the 15-constant work; separate commits — proven byte-equivalent at 100 Hz, genuine alignment work that holds independently of the storm fix).
- The diagnostic FINDINGS (the storm mechanism pin from §7.5; the EE-well-placed read from DIAGNOSTIC 1(b); the Probe-B-at-100 Hz confirmation from DIAGNOSTIC 1(c)).
- The instrumentation gates (`PUSHA_LANDING_TRACE`, `PUSHA_CONTROL_HZ`, `PUSHA_FORCE_ROUTE_TRACE`, `PUSHA_SETPOINT_TRACE`, `PUSHA_CONSISTENCY_TRACE`, `PUSHA_ADMM_DUMP`) — read-only / default-OFF.

**DEFERRED:** the storm fix (likely FIX-B isolated + tested) — the storm is a 1 kHz c3-blocker, NOT a no-push fix; at 100 Hz it self-corrects; the projection investigation proceeds at seed-0 100 Hz where c3 engages without it. Re-address the storm IF/WHEN 1 kHz or multi-seed work needs it.

#### (7) STRATEGIC RESOLUTION

The cadence-reconciliation sub-arc produced REAL alignment work (the tick→sim-time conversion, byte-equivalent at 100 Hz) but was **NOT on the no-push critical path**. The no-push is — and always was — the projection/convergence question (Probe B componentwise non-convergence), cadence-independent, EE-well-placed.

The leading candidate fix is **LCP-per-knot** (offline harness matched the brute-force oracle to 6.22e-8 / 8 significant digits; live LCP produced 34 mm at 1 kHz seed 0 + partial four-link chain closure). **BUT** the LCP path is NOT yet proven: the live LCP solve ALSO did not formally converge (pr 4.96, λ wandered around the oracle but didn't lock to it). **"LCP is the fix" remains a HYPOTHESIS with a known gap** — offline-correct projection does not lock in the live closed loop. The next investigation interrogates THAT gap.

#### (8) §1 / §7 updates folded in WITH this banking entry

- **§1 row 3 (ADMM solver)**: projection-defect attribution UN-PROVISIONALIZED — confirmed cadence-independent at both rates. Cell text updated above.
- **§1 row 8 (Entry & cadence)**: conversion STAYS (real alignment work). Storm fix DEFERRED. Cadence-discriminator MOOT for the no-push. Cell text updated above.
- **§7.5 banking entry**: the SMOKE 1 "regression" framing is superseded by DIAGNOSTIC 1+2 here. The §7.5 "Update 2026-06-25 — MECHANISM PINNED + BUNDLED FIX" subsection's claim that FIX-A+FIX-B(2) addresses the chain is RETIRED — the fix addresses only the cosmetic landing storm at 1 kHz; it does NOT address the no-push at either rate.

#### Anti-stale binding

Any subsequent entry that treats "`ccb71f5` is the EE-landing fix" or "the 28 mm box motion at 100 Hz vindicates the fix" is operating on a stale record — DIAGNOSTIC 1+2 refute both. The current state-of-truth: the no-push is Probe B componentwise non-convergence, cadence-independent; LCP-per-knot is the leading hypothesis but has its own convergence gap (offline-oracle-correct, live-not-locked).

**Next gate (corrected):** the revert block + a probe of the LCP convergence gap (why does the live LCP wander when offline it locks to the oracle?). NOT actioned in this plan-doc edit.

---

### 7.7 — Revert landed + 100 Hz byte-equivalence restored + LCP convergence-gap probe FRAMED (2026-06-25)

**The revert block executed:** commit `5168ddd` surgically restored `control/sampling_c3/sampling_based_c3_controller.py` to its `580c716` state (FIX-A + FIX-B(2) removed; LANDING-TRACE gates + tick→sim-time conversion underneath preserved untouched). The §7.5 + §7.6 doc content stays.

**100 Hz byte-equivalence CONFIRMED restored.** Post-revert smoke (`/tmp/diag_runs/d3_post_revert_100hz.log`) vs `stage_c/seed0_usol_100.log` baseline:

| Bar | Baseline | Candidate (post-revert) | Δ | Pass |
|---|---|---|---|---|
| first_c3_step ±2 | 173 | 173 | 0 | ✓ |
| first_c3_phi ±0.1 mm | 4.940 mm | 4.940 mm | 0.000 mm | ✓ |
| switches ±2 | 36 | 36 | 0 | ✓ |
| mode_match_rate ≥95% | — | **1201/1201 = 1.0000** | PERFECT | ✓ |
| final_obj_xy ±0.5 mm | (0, 0) | (0, 0) | 0.000 mm | ✓ |

**5/5 PASS, mode_match 1201/1201 byte-equivalent.** The revert removed the fix without damaging the conversion. The tick→sim-time conversion sub-arc is in its final landed state.

#### NEXT PROBE — FRAMED (not executed): the LCP convergence-gap question

**Scope as a PROBE that interrogates an open question — NOT an assumed fix.** The leading hypothesis (LCP-per-knot is the no-push fix) has a KNOWN GAP that the probe is designed to interrogate, not paper over.

**The open question:** **why does the live LCP-per-knot solve FIND the oracle λ (`λ_n = 0.5839`, offline match to 8 significant digits — `lam_n_diff = 6.22e-08`) but NOT formally converge in the closed loop (live pr = 4.96, dr = 6.66; λ_n_max wanders around the oracle with median 0.703, max 3.76, never locking precisely)?** Offline machine-precision match did NOT translate to the live closed loop. This is the gap.

**Evidence base (preserved on HEAD):**
- **Offline harness** `scripts/_stage_c_admm_harness.py` Cell B: LCP-per-knot matched the brute-force oracle to 6.22e-8 on the captured `stage_c/admm_dump/seed0_full50.npz` instance. `pr_final = 0.74` (not below tol=1e-3) but `λ_n_first = 0.5839` (oracle).
- **Live LCP verification** (`stage_c/lcp_verify/seed0_lcp_full.log`): 1 kHz seed-0 12 s with `--c3plus-projection=lcp` → 34.2 mm box motion (vs 0 mm under componentwise), partial four-link chain closure (φ_pred_min minimum -2.91 mm crossing penetration, V1 sustained-contact 5%→20%), but formal convergence still 0/65 and in-pipeline λ_n_max wandering median 0.703 vs oracle 0.5839 (Δ = 17.4 %, max Δ = 3.76).
- **Flag exists**: `main.py:248 --c3plus-projection {componentwise, lcp}` default `componentwise`. Promotion deferred.

**Candidate sub-questions (to investigate, not assume):**
- **(a) Warm-start interaction.** The offline harness solves the captured instance once (no warm-start). The live loop solves at every tick — the prior tick's solution is the warm-start of the next. Does the LCP projection interact badly with a stale warm-start (e.g. the LCP basis at tick N pivots to one fixed point; tick N+1 starts from there and pivots to a different one; oscillation around the oracle)?
- **(b) Per-tick x₀ variability.** The offline harness uses a fixed captured `x₀` (one knot-0 LCP at a single instance). The live loop has a fresh `x₀` per tick — the LCP `q = E·x₀ + H·u + c` shifts every tick, so the LCP basis the live solver finds is slightly different per tick. Does this per-tick shift drive the in-pipeline λ_n wandering even though the underlying solution is correct?
- **(c) Primal-dual divergence (Cell B's precision caveat).** Cell B converged in the primal sense (λ_n locked to oracle to 8 sig figs) but not in the dual sense (pr = 0.74 ≫ tol = 1e-3; the dual residual stays above tol while the primal has found the answer). In the live loop, the same divergence is sharper: primal also wanders. Is the ADMM iteration scheme's convergence criterion (pr < tol AND dr < tol) over-strict for the LCP path's correctness signature?

**Substrate decision:** **seed-0 at 100 Hz.** Per DIAGNOSTIC 1, c3 engages at 100 Hz on seed-0 without the storm fix (the 6-tick window self-corrects via accidental target cycle-back). This is the cleanest substrate to interrogate the LCP convergence gap — stripped of the 1 kHz cadence confound AND the storm-fix dependency. Use the existing `[CONSISTENCY]` + `[FORCE-ROUTE]` + `[SETPOINT]` gates (all default-OFF) to capture the live LCP's per-tick λ trajectory, the warm-start state, and the q-shift.

**This is NOT a fix run.** The probe interrogates the gap: under what conditions does the live LCP lock to the oracle vs wander? The answer routes the actual fix: (a) warm-start-conscious LCP wrapper, (b) tol relaxation matching the LCP's primal-correctness signature, (c) a deeper algorithmic redesign. The next block opens this probe; NOT actioned in this plan-doc edit.

#### Anti-stale binding

Any subsequent entry that treats "the LCP path is the confirmed no-push fix" without first interrogating the convergence gap is operating on a stale record — the 34 mm at 1 kHz partial chain closure is suggestive, NOT confirming. The next gate is the probe, NOT a default flip / multi-seed validation / etc.

**Next gate (corrected from §7.6):** the LCP convergence-gap probe at seed-0 100 Hz, scoped as a probe of the offline-vs-live divergence (sub-questions a/b/c). NOT actioned in this plan-doc edit.

---

### 7.8 — Cheap artifact-reread CLOSED the LCP path; reference-componentwise fork is the leading gate (2026-06-25)

**The cheap reread of `stage_c/lcp_verify/seed0_lcp_full.log` + `scripts/_stage_c_admm_harness.py` Cell B was decisive AND plan-changing.** It closes the projection-switch fix path and re-points the route to the banked port-vs-reference componentwise fork. Banked here BEFORE the precondition probe.

#### (1) LCP-per-knot REJECTED as the fix — same fictional-prediction signature as componentwise

**The cheap reread routed to BRANCH 3, not BRANCH 1.**

**(A) PRIMAL WANDER, NOT primal-locks-dual-lags.** The LCP sub-step (Lemke per-knot) still locks to machine precision per-knot (median 7.4e-8 = oracle match), but the ADMM envelope around it does NOT produce a stable λ:

| Signal | Live (1 kHz seed-0) | Cell B (offline) |
|---|---|---|
| LCP sub-step residual | median 7.4e-8 | 6.22e-8 (oracle) |
| ADMM primal_final | median 4.96 | 0.74 |
| ADMM dual_final | median 21.0 | 6.66 |
| λ_n_max output (oracle = 0.5839) | median **0.703**, max 4.35 | 0.5839 (oracle) |
| solves within 5 % of oracle | **2.2 %** (28/1266) | (offline locked) |
| solves with λ_n > 1.0 (large wander) | **20.3 %** (257/1266) | — |

Cell B's offline primal-locks-dual-lags signature **DID NOT translate to the live closed loop**. The live primal-on-λ wanders dramatically.

**(B) PREDICTION FICTIONAL.** Even when the LCP λ is approximately right, the planner's overall x_seq is non-physical:

| Signal | Live (1 kHz LCP) | Componentwise (Probe B context) |
|---|---|---|
| Predicted `box_total_xy` median (per horizon) | **228 mm** | 234 mm |
| Actual box motion over 12 s | **38 mm** | 0 mm |
| Planner-extrapolated (228 mm/s × 12 s) | ~2736 mm | — |
| **actual / planner-extrapolated** | **1.4 %** | 0 % |

**SAME fictional-prediction signature as componentwise.** The 34 mm at 1 kHz was NEVER faithful tracking — it is a **1.4 % rendering of a fictional 228 mm/horizon prediction**. The same shape that defines the no-push at componentwise also defines the no-push at LCP.

**CONCLUSION:** switching projections (LCP-per-knot) is **NOT the fix**. LCP produces the same fictional-prediction signature as componentwise at the live envelope. The over-strict-criterion path (BRANCH 1) is also dead — the primal does not lock AND the prediction is fictional, so there is no primal-correct solution for a relaxed criterion to accept.

#### (2) Warm-start REFUTED by CODE (not inference)

`_solve_c3plus` zeros `delta = omega = delta_prev = np.zeros(total_dim)` at the top of EVERY call (`admm_solver.py:951-953`). The port's ADMM cold-starts the dual/primal variables every MPC tick — NO warm-start carryforward. Sub-question (a) warm-start interaction is **REFUTED by code review** without a new trace. Sub-question (b) per-tick x0 variability stays mechanistically plausible but the cold-start **BOUNDS its impact** (each solve is a FRESH problem with no inheritance from the prior tick's iteration state — so the wandering λ is a consequence of cold-started ADMM not reaching a fixed point in 25 iters for shifting problem structure, NOT stale-warm-start contamination).

#### (3) THE SHARED FICTIONAL-PREDICTION SIGNATURE (deeper hint, banked as HYPOTHESIS)

The fictional-prediction signature (predicted ≫ actual, ~1-1.4 % rendered) is **SHARED** between componentwise (234/0) and LCP (228/38, 1.4 %). This suggests the problem may NOT be in the PROJECTION at all but in the **LCS/dynamics the planner optimizes against**. If the planner predicts 228 mm/horizon that reality renders at 1.4 %, the MODEL the planner optimizes over may be **disconnected from the PLANT it controls**. This is plausibly the `λn_ee_box=NaN` thread (the LCS never admits the EE-BOX pair) surfacing as a prediction/reality gap.

Banked as a HYPOTHESIS, NOT the immediate move — the reference-componentwise fork is the leading gate. But it is the thing to watch if the reference-componentwise fork does not resolve it.

#### (4) ROUTE — the reference-componentwise fork is now the LEADING gate

Both the criterion path (BRANCH 1) and the LCP-switch path are dead. **The fix is NOT switching projections — it is making the port's componentwise WORK like the reference's.** The reference runs the SAME componentwise projection (Bui 2026 eq 12) at `iter=3` and CONVERGES; the port's fails at `iter=25`. The fix surface is the port-vs-reference componentwise SETUP difference: E-matrix construction, ρ schedule, iteration-scheme initialization, OSQP block coefficients.

#### (5) THE PRECONDITION (unbundled — the cheapest first move of the fork)

**BEFORE comparing the port's iter-25 trajectory to the reference's iter-3 trajectory**, the load-bearing precondition is:

> **Does the reference's componentwise converge TO THE ORACLE (`λ_n = 0.5839`) on the captured `seed0_full50.npz` instance, or just to a small formal residual?**

This decides whether the brute-force LCP oracle is the right LIVE target.

- The brute-force LCP oracle is per-instance math (the complementarity-feasible solution for the captured LCS at u=0, found by enumeration of all 64 bases; `max|λ_i · w_i| = 6.94e-17`, complementarity satisfied to machine precision).
- The reference's componentwise either lands on the same `λ_n = 0.5839` value, or it doesn't — independent of projection family.
- **If the reference converges to 0.5839** → the oracle is confirmed as the live target, the port genuinely fails to reach a reachable solution, and the trajectory/setup comparison is the next step.
- **If the reference converges to something else** → the oracle was NOT the right live target and the framing shifts (possibly to the §7.8(3) shared-signature LCS-disconnect hypothesis).

The precondition is **unbundled from (and precedes)** the trajectory comparison.

#### (6) LCP TRACE DROPPED

The live LCP trace for x0 / per-tick wandering characterization (BRANCH 2(b)) is **DROPPED**. BRANCH 3 dominance makes it low-value: characterizing why LCP's λ wanders is characterizing an abandoned path. **LCP is not the fix.**

#### (7) §1 / progress-table updates

- **§1 row 3 (ADMM solver)**: leading-candidate-fix shifts from "LCP-per-knot (leading-but-unproven)" to **"make the port's componentwise CONVERGE like the reference's (eq 12, iter 3) — a SETUP-difference fix"**. LCP is REJECTED. Cell text updated above.
- **Progress-table ground-truth/gap line (informational; banked here)**: the brute-force LCP oracle EXISTS and the per-knot LCP matches it offline (machine precision), BUT the live LCP envelope renders 1.4 % — LCP is not the fix; the reference-componentwise setup-difference is the path. The next gate is the precondition (does the reference's componentwise hit the oracle on the captured instance).

#### Anti-stale binding

Any subsequent entry that treats "LCP-per-knot is the leading no-push candidate" or "the 34 mm at 1 kHz is partial chain closure" is operating on a stale record — DIAGNOSTIC §7.6 + this §7.8 read close that path. The current state-of-truth: **both componentwise (port-default) and LCP-per-knot produce the same fictional-prediction signature at the live envelope** (~1 % rendered); the fix surface is the port-vs-reference COMPONENTWISE setup-difference, not switching projections; the precondition probe runs first.

**Next gate (corrected from §7.7):** the PRECONDITION probe — does the reference's componentwise converge to `λ_n = 0.5839` on `stage_c/admm_dump/seed0_full50.npz`? This is offline (no sim), per-instance, decides whether the oracle is the live target. The trajectory/setup comparison is the SEPARATE step after. NOT actioned in this plan-doc edit.

---

### 7.9 — Reframe + model-plant consistency probe: MODEL-BROKEN (2026-06-25)

**The reference-settings precondition probe is DEMOTED to contingent-after. The cheaper, more fundamental model-plant consistency test ran first and is decisive: the captured LCS predicts box motion that Drake does NOT render, by a factor of 1.7e7×.** The brute-force oracle is a feasible solution to the WRONG LCS.

#### Reframe — port and reference build DIFFERENT LCS objects

The port's `lcs_formulator.py` and the reference's `c3/multibody/lcs_factory.h` (the latter not on disk in this clone — its source lives in a separate `c3` module the reference repo depends on) construct different LCS objects:
- **Port:** Stewart-Trinkle reformulation; `n_lambda = 2·num_normals + n_t` (γ slack + λ_n + λ_t). Default ONLY admits pairs within the 2 mm threshold via Drake's `ComputeSignedDistancePairwiseClosestPoints`. `box_ground_drag = 10.0` viscous approximation in the A matrix (lcs_formulator.py:93). The `LCS_EXPLICIT_BOX_GND` env knob (lcs_formulator.py:71) exists to synthesize N explicit box-vertex ↔ ground contact rows but is **DEFAULT OFF**.
- **Reference:** `contact_model: anitescu` (sampling_c3plus_options.yaml:8), `resolve_contacts_to_lists: [[0, 1, 12, ...]]` (12 object-ground contacts ALWAYS — anything/c3plus parameters:contacts), `scale_lcs: true` (sampling_c3plus_options.yaml:9). Source for the actual LCS construction is in the c3 module not on disk; settings are READABLE.

**The brute-force oracle (λ_n = 0.5839) was computed on the captured port LCS at u=0.** The port LCS at that knot has `n_lambda = 6 = 1·γ + 1·λ_n + 4·λ_t` — exactly ONE contact pair admitted (the EE-BOX pair). **No box-ground contact in the LCS at all.**

#### Why this comes BEFORE the reference-settings precondition

The no-push has TWO candidate levels: (i) SOLVER — the ADMM does not reach the oracle (the demoted reference-settings precondition tests this); (ii) MODEL — the LCS the solver would converge to does not match the plant. **The MODEL level is more fundamental AND cheaper to test, so it comes FIRST**; the reference-settings precondition is DEMOTED to contingent-after (only meaningful IF the LCS matches the plant).

Evidence the MODEL level is live: the fictional-prediction signature is SHARED across projections (componentwise 234/0, LCP 228/1.4 %; §7.8); `λn_ee_box=NaN` (the live LCS never admits the EE-BOX pair at non-penetrating distance; §7.6 DIAGNOSTIC 1(c)).

**KEY ISOLATION:** the live planner's 228 mm-predicted / 1.4 %-rendered CONFLATES solver (non-converged λ) and model (LCS ≠ plant); using the ORACLE λ (the converged/best feasible solution to the LCS) removes the solver confound and isolates the model.

#### THE CONSISTENCY TEST — MODEL-BROKEN, 1.7e7× mismatch

`scripts/_stage_c_model_plant_consistency.py` (committed):

| Signal | LCS-with-oracle prediction | Drake-rendered |
|---|---|---|
| Δ box xyz | (-1.313, -1.313, **-17.226**) mm | (-0.000, +0.000, **+0.000**) mm |
| total \|Δ\| | **17.33 mm** | **0.000 mm** |
| box z-velocity after | **-0.345 m/s** | +0.000 m/s |
| z-drop ratio | — | **1.7e7×** |

**The LCS predicts the box FALLS 17 mm in 0.05 s; Drake renders the box STATIONARY.** The brute-force oracle, the gold-standard complementarity-feasible λ for the captured LCS, produces a fictional next-state.

The reason: the captured LCS has `n_lambda = 6` (1 EE-BOX contact only, no box-ground). The `d_const[15]` row carries gravity's impulse (≈ -0.49 m/s/step); λ_n=0.5839 N at the EE-BOX contact contributes +0.146 m/s/step; nothing in the LCS opposes the fall. Drake's compliant contact at the floor holds the box up.

#### STEP 1b — E-matrix tangent-row-zeroed claim REFUTED on the live instance

On the captured E:

| Slot | Row index | Nonzero count |
|---|---|---|
| γ slack | 0 | **0 (zero row)** |
| λ_n | 1 | 5 |
| λ_t[0] | 2 | 4 |
| λ_t[1] | 3 | 4 |
| λ_t[2] | 4 | 4 |
| λ_t[3] | 5 | 4 |

**Only the γ slack row is zeroed — the standard Stewart-Trinkle slot.** Tangent rows (slots 2-5) are NOT zeroed; tangent friction IS enforced via the LCS complementarity. The deck's framing "E-matrix tangent rows ZEROED (v1, friction unenforced through η)" is **REFUTED on the captured live instance**. The actual LCS structural bug is the MISSING box-ground contact, not zeroed tangent rows.

#### Route: MODEL-BROKEN

| Route | Verdict |
|---|---|
| MODEL-OK (LCS-with-oracle matches plant) | **REFUTED** — 1.7e7× mismatch |
| MODEL-BROKEN (oracle is a solution to the WRONG LCS) | **CONFIRMED** |

The no-push is a MODEL problem. **Making componentwise converge gets you the oracle = a solution to the WRONG LCS = still fictional. The reference-settings path is MOOT until the LCS matches the plant.**

#### Implications for the §1 row 3 / §7.8 banking

- §7.8 "leading-candidate-fix = make componentwise converge like reference's (eq 12, iter 3)" is **DEMOTED**. Converging to the oracle on this LCS does NOT solve the no-push because the oracle itself is fictional.
- §7.6 "the no-push is Probe B componentwise non-convergence" framing stays correct at the SOLVER LEVEL but is **NOT THE ROOT** — the deeper level is the LCS missing box-ground.
- The shared-signature hint in §7.8(3) is now **STRONGLY ELEVATED**: componentwise (234/0) and LCP (228/1.4 %) shared the same fictional-prediction signature because they were both solving the SAME wrong LCS.

#### Next gate — contact-model alignment (NOT actioned this block)

The fix is at the **CONTACT-MODEL / LCS-construction level**, not the solver-settings level:
- Enable `LCS_EXPLICIT_BOX_GND` env knob (lcs_formulator.py:71) — adds explicit box-ground contact rows.
- Possibly adopt the reference's `contact_model: anitescu` (different reformulation; fewer slack variables than Stewart-Trinkle).
- Possibly adopt `scale_lcs: true` (preprocessing the LCS matrices before ADMM; details in c3 module not on disk).

Re-test consistency AFTER the LCS-construction alignment: does the LCS-with-(new-)oracle predict what Drake renders?

#### §1 / progress-table updates

- §1 row 3 (ADMM solver) cell text updated: leading-candidate-fix shifted from "make componentwise converge like reference's" to "contact-model alignment FIRST (enable LCS_EXPLICIT_BOX_GND + possibly anitescu); reference-settings precondition is MOOT until LCS matches plant."

#### Anti-stale binding

Any subsequent entry that treats "the brute-force oracle (λ_n = 0.5839) is the live target" or "make componentwise converge" as the path forward is operating on a STALE record. The oracle is a solution to a LCS missing the box-ground contact; converging to it is converging to fiction. The next gate is contact-model alignment, then re-test consistency, then (if consistent) revisit convergence.

**Next gate (corrected from §7.8):** contact-model alignment — enable `LCS_EXPLICIT_BOX_GND` and possibly adopt anitescu; re-extract the LCS on the captured x0; recompute the oracle on that new LCS; re-run the consistency test. NOT actioned in this plan-doc edit.

---

### 7.9 — augmentation (2026-06-25): CONFIRM core + tangent-row CORRECTION (prominent) + BOTH-AXES gate + strategic reframe

#### (1) CONFIRM the MODEL-BROKEN core

The §7.9 core stands: LCS-with-oracle predicts the box FALLS **17.33 mm in 0.05 s** (vz = -0.345 m/s) while Drake renders **0 mm** (box stationary, floor holds) — a **1.7e7× mismatch**. ROOT: `n_lambda = 6` = exactly ONE contact pair (the EE-BOX pair); NO box-ground contact in the LCS. Gravity pulls the box down (`d_const[15] ≈ -0.49 m/s/step`); the single EE-BOX λ_n contributes only +0.146 m/s/step; nothing opposes the fall. Drake's compliant floor contact holds the box up; the LCS does not model it. **The no-push is a MODEL problem (LCS missing box-ground), NOT a solver problem.**

#### (2) TANGENT-ROW CORRECTION (prominent — a standing assumption FALSIFIED)

**The deck's framing "E-matrix tangent rows zeroed, friction unenforced through η (v1)" is WRONG on the captured live instance.** Direct matrix read on `seed0_full50.npz`:

- row 0 (γ slack) — **zeroed** (standard Stewart-Trinkle slack)
- row 1 (λ_n) — 5 nonzeros
- rows 2-5 (λ_t[0..3]) — **4 nonzeros each**

Tangent friction IS enforced via the LCS complementarity on this instance. The actual LCS structural bug is the **MISSING box-ground contact**, NOT zeroed tangent rows. This corrects an assumption carried from the slides AND repeated in the probe framing (the §7.6 reframe-paragraph banked "E-matrix tangent rows ZEROED (v1, friction unenforced through η)"). **SUPERSEDED — do not cite "tangent rows zeroed" again.**

#### (3) THE SHARED FICTIONAL-PREDICTION SIGNATURE — finally EXPLAINED, elevated to ROOT

§7.8(3) banked this as a hypothesis: componentwise (234/0) and LCP (228/1.4 %) produced the SAME fictional-prediction signature. The §7.9 result is the answer: **both projections were solving the SAME wrong LCS**. The projection choice could NEVER have fixed this — the bug is structural in the LCS, upstream of how the LCS is solved. **This is why every projection probe (componentwise non-convergence, the LCP oracle-match, the over-strict-criterion question) was operating one level BELOW the fault.** The shared signature is no longer a hypothesis; it is the diagnostic identity of the root cause.

#### (4) CONFIRM THE DEMOTIONS (now canonical, not provisional)

- §7.8 "make componentwise converge like the reference (eq 12, iter 3)" — **DEMOTED**. Converging to the oracle on this LCS = a precise solution to a box falling through the floor = still fictional.
- §7.6 "no-push = Probe B componentwise non-convergence" — stays correct at the SOLVER level but is **NOT THE ROOT**. The root is the LCS missing box-ground.
- The reference-settings precondition (§7.7's "next gate" before §7.9) — **DEMOTED to contingent-after** (only meaningful once the LCS matches the plant).

#### (5) NEXT GATE + the BOTH-AXES requirement

The route is **CONTACT-MODEL ALIGNMENT FIRST.** Leading hypothesis (cheaply testable): the `LCS_EXPLICIT_BOX_GND` synthesis (`lcs_formulator.py:71`, default-OFF) adds floor-contact rows (n_lambda 6 → 78) so the oracle on THAT LCS includes floor forces balancing gravity.

**The falsification test:** enable `LCS_EXPLICIT_BOX_GND`, re-extract the LCS at the captured x0, recompute the oracle, re-run the consistency test.

**GATE for "model now consistent" — BOTH axes must close:**
- **(a) VERTICAL.** LCS-with-oracle predicts ≈ 0 vertical motion (floor holds, matching Drake's 0). The predicted fall must go from **-17.33 mm to ≈ 0**.
- **(b) HORIZONTAL.** LCS-with-oracle predicts horizontal box motion that Drake renders under EE push. Fixing the fall (vertical) is **NECESSARY but NOT SUFFICIENT** — the goal is horizontal pushing.

**Do NOT pre-commit a contact count (4 vs 12) — read it off the result.** The natural choices per `lcs_formulator.py:67` are 4 (bottom corners only), 8 (all vertices), or 12 (8 vertices + 4 bottom-face centers).

**NOTE on oracle computation cost:** the brute-force oracle was a 2⁶ = 64 enumeration on the 6-dim λ. At `n_lambda = 78` a 2⁷⁸ enumeration is intractable. **The new oracle must use the per-knot LCP/Lemke solve (Cell B's method) or a complementarity/feasibility solve, NOT brute-force.**

#### (6) STRATEGIC REFRAME

This is a **TRACTABLE result.** A solver convergence problem is murky (tune ρ/iters and hope); a missing contact constraint is concrete, has a sharp falsifiable prediction (the box should stop falling), and the fix-lever already exists in-code behind a flag (`LCS_EXPLICIT_BOX_GND`). **The project moved from "the ADMM will not converge for reasons we are chasing" to "the LCS is missing the floor, here is the switch that adds it."** The deeper bug is the more tractable one.

#### (7) Progress-table note (for next regeneration)

When the progress table is next regenerated, the ADMM-solver row's framing shifts:
- **ROOT:** the LCS missing box-ground (a MODEL bug), NOT the solver.
- **Leading fix:** contact-model alignment (`LCS_EXPLICIT_BOX_GND`), NOT convergence.
- **Projection/convergence question:** DEMOTED to contingent-on-a-correct-LCS.

#### Anti-stale binding (canonical addition)

Any subsequent entry that cites "E-matrix tangent rows zeroed" without correction, or proposes "make componentwise converge" as the next gate, is operating on a stale record. The current state-of-truth (canonical):
- The captured LCS's E has only the **γ slack row zeroed** — tangent friction IS enforced via complementarity.
- The no-push is a MODEL problem (LCS missing box-ground).
- The next gate is `LCS_EXPLICIT_BOX_GND` enable + re-extract + recompute oracle + re-run consistency, with the **BOTH-AXES** pass bar (vertical fall → 0 AND horizontal pushing matches Drake).

**Next gate (corrected from §7.9 main body):** contact-model alignment via `LCS_EXPLICIT_BOX_GND`, with the BOTH-AXES gate above. Oracle recomputation uses Lemke-per-knot or feasibility solve (NOT brute-force enumeration, intractable at n_lambda = 78). NOT actioned in this plan-doc edit.

---

### 7.10 — Contact-model falsification probe: VERTICAL CONFIRMED at count=4; horizontal DEGENERATE/untested; status VERTICAL-ONLY (2026-06-25)

**The §7.9 sharp falsifiable prediction (adding box-ground stops the predicted fall) is CONFIRMED at LCS_EXPLICIT_BOX_GND=4. But the horizontal/push axis is UNTESTED at this captured state (degenerate — EE not in contact). Status is VERTICAL-ONLY, NOT MODEL-FIXED.** Banked here (the falsification probe + output are committed in `58ec6df`'s successor; this section completes the doc record).

#### (1) VERTICAL CONFIRMED at count=4

| Signal | Before (n_λ=6) | After count=4 (n_λ=24) | Drake |
|---|---|---|---|
| Δ box xyz | (-1.313, -1.313, **-17.226**) mm | (-0.001, -0.001, **+0.009**) mm | (0, 0, 0) mm |
| LCS box vz | -0.345 m/s | -0.0002 m/s | 0 m/s |
| Δz vs Drake | 17.226 mm | **0.009 mm** | — |
| VERTICAL CLOSE | False (1.7e7× off) | **TRUE** (< 1 mm bar) | — |

**MECHANISM (not just the number moving):** the 4 corner contacts each carry ≈ 0.5 N normal force; ~2 N total ≈ m·g = 1.96 N. **Gravity is balanced; the box is held up for the right physical reason.** The §7.9 falsifiable prediction is CONFIRMED. The MODEL-BROKEN root cause is now **ESTABLISHED, not hypothesized**.

#### (2) count=12 FAILS — instructive layout reason

| Count | n_λ | Lemke | LCS Δz |
|---|---|---|---|
| 4 (cube bottom corners) | 24 | Found feasible (res 1.0e-8) | +0.009 mm |
| 12 (vertices + face centers) | 72 | **FAILED** (res inf; λ=0 everywhere) | -24.525 mm |

The 12-vertex synthesis includes 4 TOP vertices at distance +0.1 m (NOT penetrating); Lemke cannot cleanly handle the mix of penetrating + non-touching contacts on this instance. **count=4 (exactly the corners touching the floor) is the right configuration** — read off the result, not guessed. The §7.9-aug discipline ("do NOT pre-commit a contact count") was vindicated. **ORACLE-INTRACTABLE is REFUTED**: Lemke worked at count=4; the count=12 failure is a LAYOUT mismatch, not solver intractability.

#### (3) HORIZONTAL is DEGENERATE — NOT a fidelity confirmation

At the captured x0 the **EE is 5 cm ABOVE the box top (NOT in contact)**. Both LCS and Drake show ≈ 0 horizontal motion (LCS Δxy ≈ 0, Drake Δxy ≈ 0, |Δ| = 0.002 mm) simply because **NOTHING IS PUSHING the box**. The horizontal "close" is a **VACUOUS PASS** — two things agreeing that an unpushed box does not move — **NOT a faithful test of horizontal-push fidelity**. A non-degenerate horizontal test REQUIRES a state where the EE is in contact with the box face.

#### (4) STATUS = VERTICAL-ONLY, NOT MODEL-FIXED

The contact-model fix (`LCS_EXPLICIT_BOX_GND=4`) is CONFIRMED for the **SUPPORT axis (vertical)**, UNTESTED for the **PUSH axis (horizontal)**. This matters because the ENTIRE POINT of the system is horizontal pushing — **a box correctly held up but with untested push dynamics is still UNVALIDATED for the task**. The §7.9-aug BOTH-AXES gate was specifically designed so a vertical-only fix could not be read as MODEL-FIXED; this is exactly that situation.

- Vertical = **fixed (confirmed)**.
- Horizontal = **untested (degenerate capture state)**.
- **DO NOT call this MODEL-FIXED — call it VERTICAL-ONLY.**

#### (5) Demotions / holds carried forward

- **§7.8(3) shared-fictional-prediction explanation is OPERATIONALLY VINDICATED.** Solving any LCS-consistent λ on the count=0 LCS produced the same fictional fall regardless of projection family **because the LCS itself was wrong** (componentwise + LCP both fictional → the LCS was the bug, not the projection).
- **Option 2 (LIVE verification) HELD / REJECTED for now.** Flipping `LCS_EXPLICIT_BOX_GND=4` live conflates the contact-model fix with solver non-convergence + cadence + storm; a null result would not isolate cause. The offline horizontal test isolates the contact model — **do NOT go live until offline horizontal confirms.**
- **Option 3 (reference-settings convergence) HELD.** Convergence is only meaningful on a CORRECT LCS, and the LCS is confirmed on ONE axis only; running it now tests the solver against a half-validated model (the same trap one axis down). **Re-promote only after horizontal is validated.**

#### (6) NEXT GATE — the non-degenerate horizontal probe

The captured instance is EE-above (degenerate for horizontal); the test needs an **EE-in-contact** state.

**Pre-registered sharp prediction:** with the EE in contact + floor modeled (count=4), the LCS-with-oracle should predict horizontal box motion **in the push direction matching Drake** (the push-axis analog of the vertical prediction that just confirmed).

- **If yes** → MODEL-FIXED for real; convergence question (option 3) becomes meaningful on a fully-correct LCS.
- **If the LCS predicts push Drake does not render (or vice versa)** → a SECOND contact-model gap on the push axis.

#### (7) Strategic framing (honest)

A **MAJOR, REAL** result — the root cause is CONFIRMED and the fix-lever works on the support axis — **but it is HALF-VALIDATED, and the unvalidated half (horizontal/push) is the half that matters for the task.** The "deeper bug is the more tractable one" thesis (§7.9-aug) holds — one env-knob flip fixed the fall. **But "fixes the fall" and "fixes the push" are DIFFERENT claims; only the first is established.** Resisting the temptation to call it done here is the same discipline that caught the prior comfortable re-interpretations (the 28 mm "improvement", the 34 mm "partial chain closure", the LCP "leading candidate").

#### (8) Progress-table note (for next regeneration)

ADMM-solver row should read:
- **ROOT:** LCS missing box-ground (**CONFIRMED**).
- **Fix-lever:** `LCS_EXPLICIT_BOX_GND=4` fixes vertical (CONFIRMED offline, gravity balanced).
- **Horizontal/push axis:** UNTESTED (degenerate capture).
- **Status:** **VERTICAL-ONLY**, NOT MODEL-FIXED.
- **Next:** non-degenerate horizontal probe.

#### Anti-stale binding

Any subsequent entry that cites the falsification result as "MODEL-FIXED" or proposes the live `LCS_EXPLICIT_BOX_GND=4` flip / the reference-settings precondition as the next gate WITHOUT first running the non-degenerate horizontal probe is operating on a STALE record. The current state-of-truth: vertical confirmed at count=4, horizontal degenerate at the captured state, status VERTICAL-ONLY.

**Next gate (corrected from §7.9-aug):** non-degenerate horizontal probe — capture or construct an EE-in-contact state, re-run consistency under `LCS_EXPLICIT_BOX_GND=4`, verify the push-axis analog of the vertical confirmation. NOT actioned in this plan-doc edit.

---

### 7.11 — Horizontal consistency probe: HORIZONTAL-GAP CONFIRMED; status PARTIAL not MODEL-FIXED (2026-06-25)

**Result: LCS-with-oracle at LCS_EXPLICIT_BOX_GND=4 captures only 27% of Drake's horizontal push (under-predicts magnitude by 3.7×). Direction is right; magnitude is not. Vertical still holds under push. Status is PARTIAL — vertical fixed, horizontal GAP CONFIRMED. The §7.10 "do NOT call this MODEL-FIXED" discipline is operationally vindicated; the bare claim is now FALSIFIED, not just cautioned-against.** Banked here (horizontal probe + output are committed; this section completes the doc record).

#### (1) HORIZONTAL-GAP CONFIRMED

**Constructed contact state (offline, clean — no live capture needed):**
- Box at rest at (0, 0, 0.05); quat = identity.
- EE at (+0.0749, 0, 0.05) — east face, 0.1 mm penetration.
- EE velocity = (-0.05, 0, 0) m/s — westward push (matches task-id 4).
- u = 0 (no commanded force; momentum drives the contact).
- IK Δ on EE pose: 0.173 mm.

**LCS at LCS_EXPLICIT_BOX_GND=4 + Lemke (residual 1.97e-8, clean):**
- `n_λ = 30` (1 EE-BOX + 4 BOX-VERT × 6 slots).
- λ_n EE-BOX: **0.821 N** (push contact force).
- λ_n floor: **(0.294, 0.294, 0.687, 0.687) N** — UNEQUAL corner loading consistent with horizontal push; sum = 1.962 N ≈ m·g = 1.96 N (gravity balanced under push).

**Push result (Δt = 0.05 s):**

| Signal | LCS | Drake |
|---|---|---|
| Δ box xyz | (-0.451, +0.012, **-0.000**) mm | (-1.684, +0.165, **-0.495**) mm |
| Δ box vel | (-0.0090, +0.0002, 0) m/s | (-0.0128, -0.0021, +0.0304) m/s |
| xy gap vs Drake | — | **1.242 mm (> 1 mm bar)** |
| z gap vs Drake | — | 0.495 mm (< 1 mm bar) |

**Direction RIGHT (both westward); magnitude UNDER by 3.7×. LCS captures 27% of Drake's horizontal box motion. HORIZONTAL CLOSE = FALSE.**

#### (2) VERTICAL still holds at the contact state

LCS Δz = 0 (box stays on floor under push); Drake Δz = -0.495 mm (slight settling under push transient) — within the 1 mm bar. **The count=4 floor support is PRESERVED under push conditions**; the vertical confirmation extends to this new state.

#### (3) STATUS = PARTIAL, NOT MODEL-FIXED

The contact-model fix is **NECESSARY** (vertical holds at both the EE-above and EE-in-contact states) but **NOT SUFFICIENT** (horizontal under-predicts 3.7×). MODEL-FIXED-REAL is REFUTED. The prior §7.10 "do NOT call this MODEL-FIXED" discipline is **OPERATIONALLY VINDICATED** — the bare claim is now FALSIFIED, not just cautioned-against.

| Route | Verdict |
|---|---|
| MODEL-FIXED-REAL | **REFUTED** (horizontal does not close). |
| HORIZONTAL-GAP (vertical holds, horizontal doesn't) | **CONFIRMED.** |
| CAPTURE-HARD | REFUTED (clean offline construction). |
| ORACLE-ISSUE | REFUTED (Lemke worked at the contact state). |

Status: contact model **PARTIAL** — vertical fixed (both states), horizontal GAP CONFIRMED (27% / 3.7×).

#### (4) THREE GAP HYPOTHESES (framed, NOT isolated this block)

**(a) `box_ground_drag = 10.0/s` viscous damping in the LCS A matrix (lcs_formulator.py:93).** Per-step multiplier `(1 - c·dt) = 0.5×` — HALVES box translational velocity per planner step. Could explain ~2× of the 3.7× under-prediction.
- **CRITICAL caveat:** `box_ground_drag` is a PORT-SPECIFIC approximation (the stand-in for box-ground contact). With explicit floor contacts now present at count=4, the drag may be a **REDUNDANT BAND-AID** double-counting ground interaction. **The reference has NO box_ground_drag** (anitescu + explicit contacts) — so removing it is a move TOWARD the reference, not a port-specific hack.
- **LEADING hypothesis.**

**(b) Discrete-time LCS Euler step over 0.05 s vs Drake's 1 ms substeps.** Nonlinearities accumulate; Drake captures intermediate contact transitions the LCS does not. A DIFFERENT kind of error (integration, not a wrong term).

**(c) Stewart-Trinkle vs Drake compliant contact.** μ matches (both 0.4), but instantaneous force balance differs; the deepest/rarest, about the enforcement *mechanism* not the *coefficient*.

**Residual caveat:** even if drag is ~2× of the gap, that leaves **~1.85× unexplained** — the gap likely has more than one contributor. Don't expect a single-contributor closure.

#### (5) CONVERGENCE STAYS HELD

Re-promoting the reference-settings convergence question now would test the solver against a model that under-predicts the push by 3.7× — **the same "test the solver on a half-validated model" trap §7.10 warned against**. Convergence HELD until the horizontal gap closes (the model quantitatively right on the push axis).

#### (6) Strategic framing (continuing the honesty)

The "deeper bug is the more tractable one" thesis (§7.9-aug) HELD for vertical (one env flag flipped it) but is visibly **NOT holding for horizontal** — the contact model is **structurally present but quantitatively wrong**, and quantitative contact-model errors are the **genuinely hard part of contact-implicit work**. This is the half that matters for the task.

**Not a setback — the problem getting honest about its actual difficulty.**

ON THE HORIZON: if the gap is fundamental to the Stewart-Trinkle-vs-compliant difference (not removable drag or discretization), matching the reference's LCS may eventually require the **anitescu reformulation** (the reference's actual contact model). Deferred — heavy lift — on the other side of the cheap structural tests.

#### (7) NEXT GATE — the drag-removal probe (cheapest first move, framed structurally)

**Is `box_ground_drag` a redundant band-aid the count=4 fix superseded?** Disable it offline (`box_ground_drag=0` in the formulator constructor), re-test horizontal at the same constructed state.

**THREE-OUTCOME pre-registration:**
- **Gap closes to ~1×** (LCS now matches Drake) → drag was the whole story. Revisit the residual arithmetic — possibly the integration error was much smaller than feared.
- **Gap closes to ~1.85×** → drag was ~2× as predicted; SECOND contributor remains. Δt sub-stepping is the next probe.
- **Gap barely moves** → the 0.5× multiplier was a red herring. The gap is in integration (b) or friction enforcement (c). Δt next; friction audit deepest.

Then (b) Δt sub-stepping on the residual; (c) friction audit held as deepest / last.

#### (8) Progress-table note (for next regeneration)

ADMM-solver row:
- **ROOT:** LCS missing box-ground (CONFIRMED).
- **Fix-lever:** `LCS_EXPLICIT_BOX_GND=4` fixes vertical (CONFIRMED at both EE-above and EE-in-contact states; gravity balanced; holds under push).
- **Horizontal/push axis:** **PARTIAL** — LCS captures **27%** of Drake's push, under-predicts **3.7×** (GAP CONFIRMED).
- **Status:** **PARTIAL**, not MODEL-FIXED.
- **Next:** HORIZONTAL-GAP investigation, starting with the drag-removal probe.
- **Convergence:** HELD until the model is quantitatively right on the push axis.

#### Anti-stale binding

Any subsequent entry that cites the contact-model fix as "MODEL-FIXED" or proposes the live `LCS_EXPLICIT_BOX_GND=4` flip / the reference-settings precondition as the next gate WITHOUT first closing the horizontal gap is operating on a STALE record. The current state-of-truth: vertical fixed (both states), horizontal under-predicts 3.7× at count=4, status PARTIAL, drag-removal probe is the cheapest next move.

**Next gate (corrected from §7.10):** drag-removal probe — set `box_ground_drag = 0` in the formulator constructor offline, re-extract LCS at the constructed contact state under `LCS_EXPLICIT_BOX_GND=4`, recompute the oracle via Lemke, re-step Drake from the same state, compare horizontal box motion. Three-outcome pre-registered: gap closes to 1× / 1.85× / barely. NOT actioned in this plan-doc edit.

---

### 7.12 — Drag-removal probe: REDHERRING (byte-identical, drag inert at v_box=0); two structural findings distinct (2026-06-25)

**Result: drag=10 vs drag=0 produced BITWISE-IDENTICAL LCS predictions at the constructed contact state. The 3.73× under-prediction did NOT move with drag — not "barely", literally byte-equivalent.** Cleanest possible REDHERRING (DRAG-NOT-IT). Commit `36da966` (`scripts/_stage_c_drag_removal.py` + `stage_c/drag_removal_output.txt`).

#### (1) DRAG-REDHERRING (DRAG-NOT-IT) — byte-identical

drag=10 vs drag=0 at the constructed contact state produced **BITWISE-IDENTICAL** LCS predictions — same Δbox **(-0.451, +0.012, -0.000) mm**, same λ down to every nonzero (λ_n EE-BOX **0.821**, floor **(0.294, 0.294, 0.687, 0.687)**), same Lemke residual **1.97e-8**, same horizontal xy-gap to Drake **1.2423 mm**. The 3.73× under-prediction **did NOT move with drag** — not "barely moved", **literally byte-equivalent**.

**OUTCOME: REDHERRING, cleanest possible.**

#### (2) THE MECHANISM — drag multiplies a zero

`box_ground_drag` enters via the A matrix as `(1 - c·Δt) = 0.5×` on the box **TRANSLATIONAL VELOCITY**. At the constructed contact-entry state **v_box = 0** — there is **nothing for the multiplier to scale**. The 0.5× single-step arithmetic (the "~2×" prediction) was applied to a term that multiplies zero — **definitionally irrelevant at the contact-entry instant**.

**CORRECTION:** the prior §7.11 §(4)(a) "could explain ~2×" prediction was **WRONG-ON-ARRIVAL**; the residual caveat ("~1.85× unexplained") **UNDERSTATED** it — **ALL 3.73× is unexplained by drag**. (The "~2×" was a reviewer prediction from the §7.11 framing; the three-outcome pre-registration caught it against expectation — recording the falsification, not the original guess.)

#### (3) THE TWO STRUCTURAL FINDINGS (kept DISTINCT)

**(a)** `box_ground_drag` is dynamically **INERT** at this contact state (multiplies v_box=0) — **NOT the gap closer**.

**(b)** Drag IS **redundant** with the explicit count=4 contacts on the **VERTICAL** axis — with drag=0 the floor still holds (Δz=0, gravity balanced by λ_n floor sum 1.962 N ≈ m·g); removing it is a **move TOWARD the reference** (no drag term) with **NO cost** to the vertical-holds property. But given (a) it is NOT the gap closer either.

**One frame:** drag was a band-aid the explicit contacts superseded, **AND** it never was the cause of the horizontal gap at this state.

#### (4) THE CAVEAT (scope limit, for the Δt block)

Drag may STILL matter once **v_box ≠ 0** (steady-state coast). The test here was at the contact-entry **INSTANT** (v_box=0). The Δt sub-stepping test will pass through states with **NON-ZERO v_box**, where `box_ground_drag` (which multiplies v_box) **RE-ENTERS** — so the Δt test and the drag question are now **ENTANGLED**. The drag question is **NOT fully closed** for trajectory-mean comparisons, only at this entry instant.

#### (5) CONVERGENCE STAYS HELD

The model is still quantitatively wrong on the push axis (**3.73× under, UNCHANGED** by drag). Status STAYS PARTIAL not MODEL-FIXED. Convergence **HELD**.

#### (6) Strategic framing (starker)

**The cheapest structural explanation (drag) is SPENT and EMPTY** — drag was already dynamically inert; the gap is NOT a redundant approximation we can delete. The 3.73× lives in EITHER the **integration (Δt)** OR the **contact mechanism itself (Stewart-Trinkle vs Drake compliant)** — the second is the genuinely hard one (does not yield to a flag/parameter).

- If Δt closes it → fixable (discretization mismatch).
- If Δt does NOT → the reference's actual contact formulation (**anitescu**) may be required, because Stewart-Trinkle may not reproduce Drake's compliant-contact push at this scale.

**The search space has narrowed to "discretization or fundamental contact-model difference"; only one is cheap.**

#### (7) THE NEXT GATE — the Δt test as a 2×2 FACTORIAL (not a single comparison)

**Δt ∈ {0.05, 0.005} × drag ∈ {0, 10}.** The factorial:
- ISOLATES sub-stepping from drag-on-a-moving-box (sub-stepping passes through v_box≠0 states where drag re-enters)
- CLOSES the drag caveat (tests drag exactly where it claims to still matter, v_box≠0)

Only the **Δt=0.005 row is NEW** (the Δt=0.05 row is the two known byte-identical baselines).

**THREE-OUTCOME pre-registration (on the Δt effect):**
- **Δt-IS-IT (WHOLE ~1×)** → sub-stepping was the dominant contributor; live implication = planner-Δt vs sim-Δt mismatch (consider sub-stepping or Δt match).
- **Δt-IS-HALF (~1.85×)** → Δt is half the gap; SECOND contributor is friction (Stewart-Trinkle vs compliant). Friction audit next.
- **Δt-NOT-IT (barely)** → gap is in the friction mechanism itself; skip to friction audit (deepest).

**VERTICAL must still hold** under Δt change (sanity). Friction audit held as last/deepest.

#### (8) Progress-table note (for next regeneration)

ADMM-solver row:
- **HORIZONTAL/push axis:** PARTIAL, **3.73× under UNCHANGED** by drag (drag inert at contact entry, multiplies v_box=0).
- **Drag redundant-on-vertical:** band-aid superseded by count=4 (a move toward the reference).
- **The gap is:** discretization (Δt) **OR** contact-mechanism (Stewart-Trinkle vs compliant).
- **Next:** Δt 2×2 factorial.
- **Convergence:** HELD.

#### Anti-stale binding

Any subsequent entry that proposes `box_ground_drag` as a horizontal-gap explanation is operating on a STALE record — the term is dynamically inert at v_box=0; the 3.73× is in discretization (Δt) or contact mechanism (Stewart-Trinkle vs compliant). Any "drag is the band-aid that superseded the explicit contacts" framing must distinguish (3)(a) [inert here, not the gap closer] from (3)(b) [redundant-on-vertical, move toward reference]; conflating them is unsupported.

**Next gate (corrected from §7.11):** Δt 2×2 factorial (Δt ∈ {0.05, 0.005} × drag ∈ {0, 10}) at the same constructed contact state under `LCS_EXPLICIT_BOX_GND=4`. Three-outcome pre-registered on the Δt effect; vertical-must-hold sanity; friction audit last. NOT actioned in this plan-doc edit.

---

### 7.13 — Δt × drag 2×2 factorial: Δt-DOMINANT (3.73× → 1.43×, contact-burst mechanism); residual 1.43× = STATUS Δt-DOMINANT-residual, NOT MODEL-FIXED-REAL (2026-06-25)

**Result: sub-stepping the LCS at Δt=0.005 (10× sub-steps, re-extracted each step) closes ~57% of the original horizontal gap — factor 3.73× → 1.43×. The Δt main effect (-0.7057 mm on box_x) is DOMINANT; drag is tiny (~6%); the 1.43× lands WHOLE per pre-registered threshold (<1.5×) but at the boundary, not 1.00×.** Commit `238a70e` (`scripts/_stage_c_dt_factorial.py` + `stage_c/dt_factorial_output.txt`).

#### (1) Δt-DOMINANT (Δt-IS-IT, WHOLE per threshold)

The Δt × drag 2×2 factorial:

| Cell | Δt | drag | LCS Δbox_x (mm) | factor |Drake/LCS_x| |
|---|---|---|---|---|
| Drake (1ms substeps) | 0.05 | — | **-1.684** | 1.00× |
| A  | 0.05  | 10 | -0.451 | **3.73×** |
| B  | 0.05  | 0  | -0.451 | **3.73×** (byte-identical to A) |
| C  | 0.005 (10× RE-EXTRACT) | 10 | **-1.135** | **1.48×** |
| D  | 0.005 (10× RE-EXTRACT) | 0  | **-1.178** | **1.43×** |
| Cf | 0.005 (10× FIXED-LCS ref) | 10 | -1.166 | 1.44× |
| Df | 0.005 (10× FIXED-LCS ref) | 0  | -1.211 | 1.39× |

**Δt main effect on box_x: -0.7057 mm** — sub-stepping closes ~57% of the original 1.23 mm gap. **Best factor 1.43×** (was 3.73×). Per pre-registered threshold (<1.5 → WHOLE): **Δt-IS-IT (WHOLE)** — Δt is the **DOMINANT** contributor.

#### (2) THE HONEST RESIDUAL

**1.43× is at the WHOLE boundary, NOT 1.00×.** ~41% of the original gap (0.51 mm of 1.23 mm) STILL REMAINS at Δt=0.005. Δt was DOMINANT but the gap is **NOT fully closed** — a SECOND smaller contributor (friction the leading candidate) is in play.

**STATUS = Δt-DOMINANT-residual-1.43×, NOT MODEL-FIXED-REAL.** The model cannot be called fixed with 43% of the gap unexplained. The threshold says WHOLE; the residual says there is more; **the residual wins.**

#### (3) THE MECHANISM (the real finding, bigger than the factor)

The per-sub-step λ history shows a much richer contact picture than the single-step LCS:
- **First sub-step λ_max = 2.342, nnz = 14**  (vs single-step λ_max = 0.821)
- **Last  sub-step λ_max = 0.687, nnz = 14**

**The single-step LCS AVERAGES a contact-burst-then-relax over 0.05 s into a single quasi-static force; the sub-stepped path captures the burst.** THIS IS WHY the model under-predicted: contact-implicit dynamics over 50 ms are **TRANSIENT**, and a single linearized step smears the transient into its mean.

The 3.73 → 1.43 is **a discretization artifact identified and removed, NOT a tuning win.** This mechanism survives even if the residual story changes.

#### (4) DRAG-MATTERS-MOVING (small) — the §7.12 caveat empirically CLOSED

- At Δt=0.05 (v_box=0 the whole step):       A-B = **+0.0000 mm** (drag inert, consistent with §7.12 byte-identical).
- At Δt=0.005 (v_box≠0 after first sub-step): C-D = **+0.0434 mm** (drag re-enters).

**DRAG-MATTERS-MOVING confirmed** — drag DOES re-enter once v_box≠0 — but magnitude ~6% of the Δt main effect (0.043 / 0.706); drag main effect (avg over Δt) is **+0.0217 mm**, tiny. The §7.12 drag caveat closes empirically: **drag at v_box≠0 is real but small; NOT a re-chase target.**

#### (5) VERTICAL HOLDS in all six cells

Δz = **0.0000 mm in EVERY cell** (A, B, C, D, Cf, Df). The count=4 floor support is stable across Δt and drag. Vertical sanity **HOLDS.**

#### (6) FAITHFULNESS — re-extract vs fixed-LCS

- C (re-extract, drag=10): -1.135 mm  vs  Cf (fixed-LCS): -1.166 mm → diff **+0.031 mm**
- D (re-extract, drag=0):  -1.178 mm  vs  Df (fixed-LCS): -1.211 mm → diff **+0.032 mm**

Within **~3%**. Both close most of the gap; **re-extract closes marginally LESS** (the stale linearization gives a marginally larger numerical push — interesting). The faithfulness caveat is empirically small at this scale; **C/D are the faithful values.** No IK fallback was triggered.

#### (7) CONVERGENCE STAYS HELD

Testing the solver against a model still **1.43× off on push** is the same trap (a smaller version of §7.10/§7.11/§7.12). **HELD until the residual is investigated.**

#### (8) Strategic framing (encouraging)

The gap is **MOSTLY discretization — the fixable kind.** Going into the factorial the worry was that the 3.73× lived in the contact mechanism (the hard reformulation — anitescu); ~57% turns out to be **discretization** — a known fixable mismatch with a concrete live implication (the planner steps at 0.05 s while the sim captures 1-ms transients; planner sub-stepping or Δt-match is the fix). Only the residual ~41% is murkier, **with a bounded ceiling.**

**Moved from "the gap might be fundamental" to "most is discretization, the remainder is one friction audit from being either closed or accepted."**

#### (9) THE NEXT GATE — the friction audit (the ONLY remaining candidate at this state)

By elimination: at this state, count=4, with Δt dominant and drag tiny, the remaining mechanism difference is **friction enforcement (Stewart-Trinkle instantaneous λ_t resolution vs Drake compliant friction)**. μ matches (0.4) — about the **MECHANISM**, not the coefficient.

**MUST be measured at Δt=0.005** (where the residual lives), NOT at Δt=0.05 (inside the discretization error).

**THREE-OUTCOME pre-registration (including ACCEPT-AND-STOP as a LEGITIMATE ending):**
- **friction-CLOSES (~1×)** → model-matches-plant; **re-promote convergence**.
- **friction-PART (sub-10% floor remains)** → likely the **IRREDUCIBLE** Stewart-Trinkle-vs-compliant friction floor → **ACCEPT** the LCS as good-enough. A **legitimate stopping point, NOT a failure**.
- **friction-BARELY-MOVES** → residual uncharacterized; **anitescu becomes live** (the reference's actual contact formulation).

#### (10) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **Δt-DOMINANT, 3.73× → 1.43×** (sub-stepping closes ~57%, contact-burst mechanism).
- **Residual 1.43×** (friction the leading remaining candidate).
- **Drag confirmed-but-tiny** (DRAG-MATTERS-MOVING ~6% of Δt).
- **Status:** Δt-DOMINANT-residual-1.43×, **NOT MODEL-FIXED-REAL**.
- **Next:** friction audit on the sub-stepped path.
- **Convergence:** HELD.

#### Anti-stale binding

Any subsequent entry that calls the contact model MODEL-FIXED-REAL on the basis of the §7.13 Δt result alone is operating on a STALE record — the residual 1.43× says ~41% of the gap is unexplained at the finest tested Δt; the threshold-passing WHOLE classification is dominated by the residual. Any "discretization closed it" framing must distinguish (1) [Δt-IS-IT WHOLE per threshold] from (2) [residual 1.43× says NOT MODEL-FIXED-REAL]. The contact-burst mechanism (3) survives independently of how the residual story resolves.

**Next gate (corrected from §7.12):** friction audit on the sub-stepped path (Stewart-Trinkle λ_t vs Drake compliant friction) at Δt=0.005, NOT at Δt=0.05. Three-outcome pre-registered including ACCEPT-AND-STOP as a legitimate ending. NOT actioned in this plan-doc edit.

### 7.14 — Friction-audit setup: AUDIT-SETUP-BROKEN — phantom panda_link7↔box at 37× the pusher impulse contaminated the §7.11→§7.13 Drake reference; MECHANISMS survive, NUMERICS need re-measurement; friction + anitescu DEFERRED; convergence HELD (2026-06-25)

**Result: the friction audit did NOT isolate friction-mechanism; it surfaced an upstream contamination of the §7.11→§7.13 chain. At the constructed contact state Drake renders THREE contact pairs on the box, not two — a phantom panda_link7↔box arm-wrist contact carrying 37× the pusher impulse that the LCS does not model at all. The §7.11→§7.13 quantitative factors (3.73×, 1.43×) compared an LCS-pusher-and-floor model to a Drake-with-arm-artifact reference — apples-to-oranges. MECHANISMS (drag inert, Δt captures the contact burst, count=4 stabilizes vertical) SURVIVE; NUMERICS need re-measurement on a clean (pusher-only) setup; the clean re-run may close the gap with no friction work at all.** Banked here. Commit `9680429` (`scripts/_stage_c_friction_audit.py` — the instrument that caught the contamination; the contact-pair-count check the script performs is the new precondition gate).

#### (1) AUDIT-SETUP-BROKEN

The friction audit did NOT isolate friction-mechanism; it surfaced an upstream contamination. **The friction explanation of the 1.43× residual is UNANSWERABLE on this setup because the Drake reference push is not what we thought.**

#### (2) THE CONTAMINATION — phantom arm↔box contact

At the constructed contact state, Drake finds **THREE contact pairs** on the box at t=0.005s, not two:

| Pair | Bodies | Impulse on box_x (N·s) | Notes |
|---|---|---|---|
| 0 | world / ground | floor normal **+0.000**, floor tangent **+0.806** | legit floor (4 corner contacts aggregated) |
| 1 | **panda_link7** ↔ box | **-0.788** | **LCS-EXCLUDED ARTIFACT** (arm wrist intrudes the box) |
| 2 | pusher / EE ↔ box | **-0.021** | legit |

**The arm-wrist contact carries 37× the pusher impulse.** Drake's -1.684 mm box displacement is driven by **panda_link7↔box, NOT pusher↔box** — the IK-set arm pose, while putting the pusher at the east face, intrudes the WRIST into the box.

The LCS's contact filter (EE body `pusher`, geom IDs `[223]`) **EXCLUDES** panda_link7, so the LCS models pusher↔box + 4 floor only. The §7.11→§7.13 chain compared an **LCS-pusher-and-floor** model to a **Drake-with-arm-artifact** reference — apples-to-oranges. The 3.73× and 1.43× are **NOT** model-vs-plant numbers; they are **model-vs-(plant+artifact)** numbers.

#### (3) CROSS-CHECK — where the LCS push actually comes from (Drake-independent)

Per the LCS λ decomposition at this state (no Drake dependence):

| Channel | Cumulative Δbox_x (mm) |
|---|---|
| EE-normal (pusher) push | **-1.067** |
| Floor tangent (friction resistance) | **+0.981** |
| Net λ-driven | -0.086 |
| Remainder (A-propagated velocity from earlier sub-steps) | -1.135 |

**The LCS's actual model is EE-push-dominated with friction nearly cancelling.** Drake's actual rendered push is **ARM-artifact-dominated** with friction nearly cancelling the arm contact: J_pusher -0.021, J_arm -0.788, J_floor_tangent +0.806, net -0.003 N·s.

If the LCS pusher↔box force is roughly the right magnitude vs Drake's pusher↔box force, the LCS may be modeling the right physics — **but we cannot test that here** because Drake's reference is dominated by something the LCS is not modeling at all.

#### (4) SURVIVES / DOESN'T-SURVIVE — per-finding audit (THE KEY)

| Section | Finding | Verdict |
|---|---|---|
| §7.11 | HORIZONTAL-GAP qualitative (LCS ≠ Drake at this state) | **SURVIVES qualitative** |
| §7.11 | "27% / 3.73×" quantitative | **DOESN'T survive** (contaminated by arm artifact) |
| §7.12 | DRAG-REDHERRING (drag inert at v_box=0) | **SURVIVES FULLY** — NO Drake dependence; LCS-internal A-matrix fact |
| §7.13 | Δt-DOMINANT contact-burst mechanism (sub-stepping captures the burst; single step averages) | **SURVIVES qualitative** |
| §7.13 | "1.43× residual" quantitative | **DOESN'T survive** — the residual was the arm/pusher mismatch, **NOT** friction-or-anitescu |
| §7.13 | DRAG-MATTERS-MOVING (~6% of Δt) | **SURVIVES** — LCS-internal A-vs-A comparison, no Drake dep |

**FRAME: MECHANISMS survive; NUMERICS need fresh measurement on a clean setup.**

#### (5) CORRECTION — the trajectory-signature misread (a reviewer interpretation, flag explicitly)

The prior reading "LCS leads the first ~15ms, then Drake takes over = the contact burst the single step averages out" was **WRONG.** The correct re-read: **Drake takes over because the ARM contact ramps up and overtakes the pusher, NOT a burst/friction mechanism.** The "LCS under-pushes Drake" is mostly Drake's arm contact ramping past the LCS's pusher contact. (A reviewer interpretation built on the contaminated reference — **superseded; do not cite the "burst" trajectory-signature story as model physics.**)

ALSO: the §7.11 unequal floor loading (0.294 / 0.294 / 0.687 / 0.687) was read as "consistent with horizontal push" but is more consistent with **the ARM pressing off-center** — a contamination signature read as confirmation.

#### (6) DEFERRALS

- **Friction audit DEFERRED** (cannot be cleanly read on a contaminated setup).
- **Anitescu DEFERRED** (was contingent on FRICTION-BARELY, which cannot be diagnosed here).
- **Convergence STAYS HELD.**

#### (7) STRATEGIC FRAMING — the consequential possibility

**The clean re-run MAY close the gap to ~1× with NO friction work** — the entire §7.11→§7.13 quantitative gap may be the arm artifact. If the LCS's pusher↔box force was roughly right all along and only *looked* 3.73× short because compared against an arm-driven reference, **the contact model may have been far closer to correct than §7.13 thought.** The cross-check above hints at this: the LCS's EE push -1.067 mm with friction nearly cancelling is a coherent model. **MODEL-FIXED-REAL may be CLOSER than believed**, just measured against a contaminated reference.

**Do NOT assume either way — the clean re-run decides.**

#### (8) METHODOLOGICAL NOTE — why it went undetected, plus the new gate

The contamination went undetected for FOUR sections (§7.11–§7.13) because **nobody checked Drake's contact-pair COUNT at the test state** — we assumed two and got three.

**NEW RULE:** any model-vs-plant comparison must FIRST assert the plant's contact-pair set matches what the model includes — a **hard precondition** before any quantitative read.

#### (9) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **AUDIT-SETUP-BROKEN** — the §7.11→§7.13 quantitative numbers (3.73×, 1.43×) were contaminated by a phantom panda_link7↔box contact (37× the pusher impulse) the LCS does not model.
- **MECHANISMS survive** (drag inert, Δt captures the burst, count=4 stabilizes vertical).
- **NUMERICS need re-measurement** on a clean (pusher-only) setup.
- **The clean re-run may close the gap with no friction work.**
- **Friction + anitescu deferred.**
- **Convergence:** HELD.

#### Anti-stale binding

Any subsequent entry that cites §7.11 "27% / 3.73×" or §7.13 "1.43×" as model-vs-plant numbers is operating on a STALE record — those factors compared an LCS-pusher-and-floor model to a Drake-with-arm-artifact reference, not to clean plant physics. The MECHANISM findings (drag inert at v_box=0; contact-burst captured by sub-stepping; count=4 stabilizes vertical; DRAG-MATTERS-MOVING) survive and remain citable as mechanism statements. The QUANTITATIVE factors do not.

**Next gate (corrected from §7.13):** re-pose probe — construct a contact state where Drake renders **exactly the contact-pair set the LCS models** (pusher↔box + floor only; no panda_link7↔box and no other arm-link↔box pair), then re-measure §7.11/§7.13 quantitatively on the clean setup. The friction audit is reopen-able only AFTER the clean reference is established (else friction-mechanism cannot be cleanly isolated). NOT actioned in this plan-doc edit.

### 7.15 — Re-pose probe (Path A) + HARD GATE: GAP-PERSISTS-LARGE — clean 3.93×/1.70× ≈ contaminated 3.73×/1.43×; arm artifact accounted for ~7% of the gap, NOT the bulk; §7.14's optimistic framing REFUTED; mechanisms SURVIVE; friction audit RE-OPENS (2026-06-25)

**Result: Path A re-pose (IK with box-pin + per-pair arm-link↔box clearance constraints + posture cost) achieved a clean pusher-only setup. HARD GATE PASS — STATIC SignedDistance + DYNAMIC ContactResults both confirm 1 pusher↔box + 1 floor↔box + 0 arm↔box pairs at the re-posed state. CLEAN-STATE READ (apples-to-apples vs §7.13 at EE_PEN_M=0.1 mm): single-step factor 3.93× (was 3.73× contaminated), sub-stepped factor 1.70× (was 1.43× contaminated). The arm artifact accounted for ~7% of the single-step gap (3.73× → 3.93× = 7% *worse* without arm) and ~19% of the sub-stepped gap (1.43× → 1.70× = 19% worse). The §7.11/§7.13 quantitative gaps are REAL model-vs-plant errors, NOT artifacts of contamination. §7.14's optimistic strategic framing ("MODEL-FIXED-REAL may be CLOSER than believed") is REFUTED.** Banked here. Commit `8ed2413` (`scripts/_stage_c_repose_clean.py` + `stage_c/repose_clean_output.txt`).

#### (1) HARD GATE: PASS at seed 0

Path A succeeded. IK with the box pinned (the box's 7 floating-base DOFs were silently free as decision vars and would have been relocated to satisfy clearance — the first attempt did exactly this, putting the box at (-0.5, -0.8, -0.7) m to clear the arm; box-pin fixes it) plus per-pair distance constraints between panda_link4–7 collision geoms and the box collision geom (24 pair constraints, ≥ 5 mm clearance) plus a posture-quadratic cost.

State at the re-posed contact instant:

| Quantity | Value |
|---|---|
| q_arm | `[0.981, 0.942, 0.447, -1.614, 0.087, 1.999, 0.785]` |
| pusher position | `(0.0749, -0.000, 0.0500)` (EE_err = 0.017 mm) |
| box position | `(0.000, 0.000, 0.050)` (pinned) |
| d(pusher ↔ box) | **-0.100 mm** (legit penetration matching EE_PEN_M) |
| d(floor ↔ box)  | **-0.077 mm** (touching/slight penetration) |
| min d(arm ↔ box) | **+18.913 mm** (panda_link7 closest) |
| min d(arm ↔ floor) | **+41.797 mm** |

Gate output (both checks):

| Check | pusher↔box | floor↔box | ARM↔box (ARTIFACT) | other |
|---|---|---|---|---|
| STATIC (SignedDistance @ t=0, ≤ 10 mm)   | 1 | 1 | **0** | 0 |
| DYNAMIC (ContactResults @ t=0.005 s)     | 1 | 1 | **0** | 0 |

**Both gates pass with 0 arm↔box pairs.** The contamination is eliminated at this state.

#### (2) CLEAN-STATE READ — the headline finding

Apples-to-apples versus §7.13 at EE_PEN_M = 0.1 mm and Δt × drag at the SAME cells:

| Cell | Δt | drag | extr | Δbox_x (mm) | Δbox_z (mm) | factor (clean) | factor (§7.13 contam) |
|---|---|---|---|---|---|---|---|
| Drake ref (1 ms substeps) | 0.05 | — | — | **-2.033** | 0.000 | 1.00× | 1.00× |
| A | 0.05 | 10 | single | -0.518 | -0.010 | **3.93×** | 3.73× |
| B | 0.05 | 0  | single | -0.518 | -0.010 | **3.93×** (drag inert ✓) | 3.73× |
| C | 0.005 | 10 | re-extract | -1.156 | 0.000 | **1.76×** | 1.48× |
| D | 0.005 | 0  | re-extract | -1.199 | 0.000 | **1.70×** | 1.43× |

**The clean gap is ESSENTIALLY THE SAME as the contaminated gap.** Single-step 3.73× → 3.93× (~7% *wider* without the arm); sub-stepped 1.43× → 1.70× (~19% wider). The arm artifact was real but its quantitative impact is modest.

#### (3) §7.14's STRATEGIC FRAMING is REFUTED

§7.14 (7) Strategic Framing said: *"The clean re-run MAY close the gap to ~1× with NO friction work — the entire §7.11→§7.13 quantitative gap may be the arm artifact. MODEL-FIXED-REAL may be CLOSER than believed, just measured against a contaminated reference."*

**REFUTED.** The clean factor is 3.93× / 1.70× — the same order as the contaminated 3.73× / 1.43×. The arm artifact was real (§7.14 stands) but its quantitative weight was ~7-19%, not the bulk. **The §7.11/§7.13 model-vs-plant gap is REAL contact-model error, not a measurement artifact.** §7.14's optimistic strategic framing is wrong, not the §7.14 audit-setup-broken finding itself — those are separable claims.

§7.14 stands on:
- the contamination IS real (panda_link7↔box at 37× pusher impulse) — confirmed
- the §7.11→§7.13 chain DID compare model-vs-(plant+artifact) — confirmed
- the methodological gate (assert contact-pair COUNT first) is essential

§7.14 falls on:
- the optimistic prediction "clean re-run MAY close the gap to ~1×" — the clean re-run does NOT close it
- "the LCS may be modeling the right physics, just measured against a contaminated reference" — REFUTED: the LCS sub-stepped at the clean state predicts only 60% of Drake's motion (1.70× off)

#### (4) The mechanism findings SURVIVE on the clean setup (the deeper good news)

The §7.12 / §7.13 mechanism findings hold quantitatively at the clean state — not just qualitatively as §7.14's survives/doesn't split implied:

| Mechanism | Source | Clean-state confirmation |
|---|---|---|
| DRAG-REDHERRING at v_box=0 (§7.12) | LCS-internal A-matrix | Cells A and B byte-identical at v_box=0 (-0.518 mm each); confirmed at clean setup |
| Contact-burst captures most of the gap (§7.13) | Sub-stepping the LCS | 3.93× → 1.70× closes 57% of the clean gap — same magnitude (~57%) as the contaminated chain |
| DRAG-MATTERS-MOVING (§7.13) at Δt=0.005 | LCS-internal A-vs-A | Drag at Δt=0.005 contributes -0.043 mm = ~7% of Δt main effect (§7.13: ~6%) |
| Vertical (count=4) holds | LCS_EXPLICIT_BOX_GND=4 | Δz ≤ 0.010 mm in all four cells |

**The §7.13 narrative (contact-burst dominates Δt main effect; residual a real model-vs-plant error) is now anchored on a clean test setup.**

#### (5) Drake-side cross-check resolution

§7.14 noted: *"Drake actual rendered push is ARM-artifact-dominated with friction nearly cancelling the arm contact (J_pusher -0.021, J_arm -0.788, J_floor_tangent +0.806, net -0.003 N·s)."* The interpretation was that Drake's clean (no-arm) push would be DIFFERENT and the LCS pusher-only model "may be modeling the right physics".

Clean-state resolution: **Drake moves MORE without the arm**, not less. Drake clean Δbox_x = -2.033 mm (vs -1.684 mm contaminated). The arm was net DECELERATING the box (J_arm + J_floor_tangent ≈ 0; pusher could only add -0.021 N·s). Without the arm pressing back, the pusher's clean impulse drives the box further. **The LCS sub-stepped under-predicts both — by 1.43× against the partly-self-canceling contaminated Drake, and by 1.70× against the cleaner Drake.** The §7.14 "LCS modeling the right physics" hypothesis is contradicted: even when Drake's contact set matches the LCS's exactly, the LCS still under-predicts by 70%.

#### (6) EE_PEN_M sweep (preliminary observation)

A prior probe variant ran with `EE_PEN_M = 1 mm` (10× deeper pusher penetration than §7.13) to give the IK headroom before the per-pair clearance constraint was correctly bounded. That variant inverted the read:

| EE_PEN_M | factor A (single) | factor D (sub-stepped) |
|---|---|---|
| 0.1 mm (§7.13 setup, apples-to-apples) | 3.93× (under) | 1.70× (under) |
| 1.0 mm (10× deeper) | 2.01× (under) | **0.48× (OVER-predict!)** |

**This implies a "sweet spot" penetration depth between 0.1 mm and 1 mm where the LCS sub-stepped matches Drake's compliant contact**. The over-correction at 1 mm penetration is consistent with the rigid-vs-compliant story: LCS Stewart-Trinkle is fully rigid, so deeper penetration → much larger λ → over-shoots; Drake compliant contact saturates softer with depth. The headline read uses 0.1 mm (apples-to-apples vs §7.13). The EE_PEN_M sweep is a next-block question, not actioned here.

#### (7) CONVERGENCE STAYS HELD

The clean setup has a real ~1.70× sub-stepped model-vs-plant gap. Re-promoting solver convergence onto the clean LCS would still test the solver against a model that is 70% off on push. **HELD until the residual closes (the model quantitatively right on the push axis at clean state).** This is identical to §7.10/§7.11/§7.12/§7.13/§7.14's HELD reasoning — none of those reads gave the model a clean bill of health.

#### (8) DEFERRALS LIFT — friction audit re-opens, anitescu still parked

- **Friction audit RE-OPENS** — on the clean setup, the §7.13 (9) three-outcome pre-registration becomes meaningful (the residual is now DEFINED, uncontaminated, and quantitatively similar to the §7.13 1.43–1.70 range). FRICTION-CLOSES / FRICTION-PARTIAL-FLOOR / FRICTION-BARELY all become live readings on the clean setup.
- **Anitescu STAYS PARKED** — re-opens only after FRICTION-BARELY at the clean state.
- **Re-pose probe** — DONE; HARD GATE PASS; clean setup is the new test substrate.

#### (9) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **GAP-PERSISTS-LARGE on the clean (pusher-only) setup** — single-step factor 3.93× (was 3.73× contaminated), sub-stepped factor 1.70× (was 1.43× contaminated); arm artifact accounted for ~7–19% of the gap, NOT the bulk.
- **§7.14's optimistic "MODEL-FIXED may be close" framing REFUTED**; the §7.11/§7.13 mechanism findings (drag-redherring at v_box=0; contact-burst; DRAG-MATTERS-MOVING; vertical count=4) all hold quantitatively on the clean setup.
- **Friction audit re-opens** as the legitimate next gate on the clean setup; anitescu still parked.
- **EE_PEN_M sweep** flagged: 1 mm gives over-prediction (factor 0.48×), suggesting a sweet spot between 0.1 mm and 1 mm — a next-block question.
- **Convergence:** HELD.

#### Anti-stale binding

Any subsequent entry that revives §7.14's optimistic framing ("clean re-run will close the gap with no friction work" / "MODEL-FIXED-REAL is close") is operating on a STALE record. The clean re-run at §7.13's apples-to-apples setup CONFIRMS a real, quantitatively similar gap (3.93× / 1.70×). The contamination is real and §7.14's audit-setup-broken finding stands; what does NOT stand is the §7.14 (7) "the entire quantitative gap may be the arm artifact" prediction. Cite §7.15 factors when characterizing the clean-state gap; cite §7.13 factors only as the contaminated baseline (with §7.14's caveat) or for the burst-mechanism quantitative story (which is robust to contamination, ~57% closure in both).

**Next gate (corrected from §7.14):** friction audit on the CLEAN sub-stepped path (Stewart-Trinkle λ_t vs Drake compliant friction) at Δt=0.005 — the §7.13 (9) three-outcome pre-registration now applies on a defined, uncontaminated residual. Parallel/orthogonal next gate: the EE_PEN_M sweep (does an intermediate penetration land closer to 1×? rigid-vs-compliant signature). NOT actioned in this plan-doc edit.

### 7.15 — augmentation (2026-06-25): routing refinement — the EE_PEN_M sign-flip is a NORMAL-COMPLIANCE signature; penetration SWEEP comes BEFORE the friction audit; friction DEMOTED; anitescu re-promotable if normal-compliance

#### (1) CONFIRM §7.15's core (already banked; restated as the augment's anchor)

The §7.15 core stands: HARD GATE PASS at the re-posed clean state via the **box-pin fix** (the box's 7 floating-base DOFs were silently free decision vars and were relocating the BOX to satisfy clearance until the pin landed) + per-link `panda_link4–7 ↔ box` distance constraints ≥ 5 mm + posture quadratic cost. BOTH static SignedDistance @ t=0 and dynamic ContactResults @ t=0.005 s confirm 1 pusher↔box + 1 floor↔box + **0 arm↔box** at the re-posed state. The gap PERSISTS clean: single-step **3.93×** (was 3.73× contaminated), sub-stepped **1.70×** (was 1.43× contaminated). The arm was net-DECELERATING (Drake clean Δbox_x = −2.033 mm vs contaminated −1.684 mm; arm impulse J_arm = −0.788 ≈ −J_floor_tangent = +0.806, mutually self-cancelling). Removing the arm WIDENED the gap ~7–19% — the §7.14 (7) optimistic "the whole gap may be the arm artifact / MODEL-FIXED-REAL is close" prediction is REFUTED; §7.14's audit-broken finding itself + the contact-pair-count gate STAND (the contamination was real and the precondition gate is essential — only the optimistic prediction falls). Mechanisms re-confirmed clean (drag byte-identical A=B at v_box=0; contact-burst sub-stepping closes 57% of the clean gap — same magnitude as the contaminated chain's 57%; DRAG-MATTERS-MOVING ~7% at Δt=0.005; vertical |Δz| ≤ 0.01 mm).

#### (2) THE ROUTING REFINEMENT (the augment)

The §7.15 (6) **EE_PEN_M sweep** observation:

| EE_PEN_M | factor A (single-step) | factor D (sub-stepped) |
|---|---|---|
| **0.1 mm** (§7.13 apples-to-apples) | 3.93× (UNDER-predicts) | **1.70× (UNDER-predicts)** |
| **1.0 mm** (10× deeper) | 2.01× (UNDER-predicts) | **0.48× (OVER-predicts!)** |

**This sign-flip across penetration depth reframes the next gate.** Re-reading the signature:

- The sub-stepped factor crosses **1× somewhere between 0.1 mm and 1 mm penetration** — there is a depth at which LCS ≈ Drake.
- The crossing is across **EE_PEN_M = the normal penetration depth**. The variable being swept is the NORMAL-direction state (how deep is the contact), not the tangent friction.
- Normal-force law: compliant Drake softens with depth (force ~ k·δ with k variable as area-of-contact / material law); rigid Stewart-Trinkle is a flat hard constraint (any depth gives any force needed to enforce non-penetration, weighted by LCP geometry). The crossing-of-1× across depth is a **rigid-vs-compliant NORMAL contact signature**.
- **Auditing friction first would be the WRONG-AXIS test** — friction is a TANGENT-force question. If the ~1.70× residual is in the normal-force-law direction, characterizing tangential friction would describe a force that is not the one carrying the residual.

**THE NEXT GATE SHIFTS.** A penetration SWEEP — map the gap across EE_PEN_M (e.g., 0.1 / 0.25 / 0.5 / 1.0 / 2.0 mm, with a per-depth contact-pair gate so each cell remains uncontaminated) — comes **BEFORE** the friction audit. The sweep disambiguates which axis carries the residual:

| Sweep outcome | Interpretation | Next direction |
|---|---|---|
| Factor crosses 1× **smoothly** with depth | Residual is **NORMAL-COMPLIANCE** (rigid-vs-compliant contact-model gap) | Contact-model conversation (a compliance term, OR the anitescu reformulation). Anitescu is **NO LONGER merely parked-pending-FRICTION-BARELY** — it becomes the indicated direction. |
| Factor stays **FLAT** at ~1.70× across depth (under-predicts at all depths) | Residual is **NOT normal-compliance** — it is in the tangent direction (or A-matrix dynamics) | Friction audit becomes the right next probe (§7.13 (9) three-outcome on the clean setup, as §7.15 originally routed). |
| Factor crosses 1× **noisily / non-monotonically** | Mixed mechanism (some normal, some tangential, possibly LCP-degeneracy regions) | Re-decompose; may need both probes. |

**The friction audit is DEMOTED.** Its scheduling is now contingent on the sweep's FLAT outcome.

#### (3) STRATEGIC FRAMING — honest, encouraging

- The gap is **REAL** (~1.70× sub-stepped at the apples-to-apples depth, not a phantom). §7.14's optimism is REFUTED.
- The mechanisms are **VALIDATED on a clean rig**. §7.12 / §7.13's mechanism findings transfer quantitatively from the contaminated chain to the clean setup.
- There is now a **SPECIFIC PHYSICAL SIGNATURE** pointing at WHAT the residual is: the EE_PEN_M sign-flip is consistent with normal-contact compliance — the rigid Stewart-Trinkle vs Drake-compliant difference. This was suspected as the hard floor all along (§7.13 (8) "going into the factorial the worry was that the 3.73× lived in the contact mechanism (the hard reformulation — anitescu)"); the sweep is the falsifiable test.
- Not lost — we have a **FINGERPRINT**. Convergence STAYS HELD (a 1.7× model is not one to tune a solver against; if the sweep names the residual as normal-compliance, the contact-model conversation precedes the solver conversation).

#### (4) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **Clean re-run GATE-PASS** (box-pin fix); gap PERSISTS clean — **3.93× single-step / 1.70× sub-stepped** (the arm artifact accounted for ~7–19%, and was net-DECELERATING).
- **Mechanisms re-confirmed clean** (drag byte-identical at v_box=0; contact-burst ~57% closure; DRAG-MATTERS-MOVING ~7%; vertical count=4 holds).
- **§7.14 optimism refuted; gate-finding stands** (contamination real, contact-pair-count gate essential).
- **The EE_PEN_M sign-flip (1.70× at 0.1 mm → 0.48× at 1 mm) is a NORMAL-COMPLIANCE signature.**
- **NEXT GATE = penetration sweep** (normal-compliance-vs-friction disambiguation) **BEFORE** the friction audit; friction DEMOTED to contingent-after; **anitescu re-promotable if sweep names normal-compliance**.
- **Convergence:** HELD.

#### Anti-stale binding (augment)

Any subsequent entry that schedules the friction audit BEFORE the penetration sweep is operating on a STALE record — §7.15-aug RE-ORDERS the next-gate sequence on the basis of the EE_PEN_M sign-flip. The sweep is the falsifier of the normal-compliance hypothesis; the friction audit is the falsifier of the residual-is-tangential hypothesis. The order matters because measuring tangent forces when the residual lives in the normal-force law would mischaracterize the residual.

**Next gate (corrected from §7.15 main body):** EE_PEN_M penetration sweep on the CLEAN setup (per-depth contact-pair gate, factor crossing-of-1× test). Friction audit becomes the next-after-that gate ONLY IF the sweep returns FLAT. Anitescu re-promotes from parked-pending-FRICTION-BARELY to indicated-direction IF the sweep returns CROSSES-SMOOTHLY (normal-compliance). NOT actioned in this plan-doc edit.

### 7.16 — Penetration sweep: NORMAL-COMPLIANCE — factor crosses 1× SMOOTHLY at ~0.55 mm penetration (1.68× under at 0.1 mm → 0.39× over at 1 mm, monotonic, all 6 depths gate-PASS); residual is rigid-vs-compliant NORMAL contact, NOT friction; friction DEMOTED; anitescu RE-PROMOTED as indicated direction (2026-06-25)

**Result: factor monotonically decreases from 1.684× (UNDER at 0.1 mm) to 0.391× (OVER at 1.0 mm), crossing 1× at ~0.549 mm. All 6 depths PASS the per-state §7.14 hard gate (STATIC SignedDistance @ t=0 AND DYNAMIC ContactResults @ t=0.005 s both show 1 pusher + 1 floor + 0 arm at every depth). The 1 mm OVER-prediction is real and clean, not arm contamination. The §7.15-aug-routed NORMAL-COMPLIANCE outcome is selected — the residual is in the rigid-vs-compliant NORMAL contact-force law, NOT friction. Friction is DEMOTED (the §7.15 friction-audit-reopens plan is overturned). Anitescu RE-PROMOTED from parked-pending-FRICTION-BARELY to the INDICATED DIRECTION.** Banked here. Commit `9e3f027` (`scripts/_stage_c_penetration_sweep.py` + `stage_c/penetration_sweep_output.txt`).

#### (1) The sweep table

Δt = 0.005 s, drag = 0, count = 4, sub-stepped, re-extracted; box-pinned IK clean state at each depth; Drake reference at 1 ms substeps over 0.05 s.

| EE_PEN_M | gate | Drake Δbox_x (mm) | LCS Δbox_x (mm) | factor | direction |
|---|---|---|---|---|---|
| 0.10 mm | PASS | -2.0331 | -1.2072 | **1.684×** | UNDER |
| 0.20 mm | PASS | -2.0144 | -1.3670 | **1.474×** | UNDER |
| 0.30 mm | PASS | -1.8729 | -1.5051 | **1.244×** | UNDER |
| 0.50 mm | PASS | -2.3288 | -2.0948 | **1.112×** | UNDER |
| 0.70 mm | PASS | -2.3029 | -3.5244 | **0.653×** | **OVER** |
| 1.00 mm | PASS | -2.2388 | -5.7311 | **0.391×** | **OVER** |

**Crossing of 1× at ~0.549 mm (linear interpolation between 0.50 mm and 0.70 mm).** Monotonic across all six depths. Spread 1.293 (1.684× − 0.391×).

#### (2) Per-state hard gate — all 6 depths CLEAN

| Depth | static (pusher / floor / arm / other) | dynamic | d(pusher↔box) | d(floor↔box) |
|---|---|---|---|---|
| 0.10 mm | 1 / 1 / **0** / 0 | 1 / 1 / **0** / 0 | -0.100 mm | -0.076 mm |
| 0.20 mm | 1 / 1 / **0** / 0 | 1 / 1 / **0** / 0 | -0.199 mm | -0.064 mm |
| 0.30 mm | 1 / 1 / **0** / 0 | 1 / 1 / **0** / 0 | -0.281 mm | -0.046 mm |
| 0.50 mm | 1 / 1 / **0** / 0 | 1 / 1 / **0** / 0 | -0.478 mm | -0.043 mm |
| 0.70 mm | 1 / 1 / **0** / 0 | 1 / 1 / **0** / 0 | -0.700 mm | -0.071 mm |
| 1.00 mm | 1 / 1 / **0** / 0 | 1 / 1 / **0** / 0 | -0.999 mm | -0.057 mm |

**GATE-FAILS-DEEP is RULED OUT** — no arm-link↔box pair forms at any tested depth. The §7.15-aug concern that "the 1 mm inversion may have been a contaminated read" is laid to rest: the 1 mm OVER-prediction is clean.

#### (3) The mechanism — rigid LCS vs compliant Drake

Drake's box motion is stable in the **-1.87 to -2.33 mm range** across all six depths (variation ~25%). LCS sub-stepped scales nearly linearly with depth, from **-1.21 mm to -5.73 mm** (variation ~5×). The driver:

- **Drake's compliant point-contact**: normal force scales with penetration depth but ALSO with a finite stiffness that saturates with area; deeper penetration → more force, but the contact softens. Net result: stable box motion across depths.
- **LCS Stewart-Trinkle**: rigid non-penetration; λ_n adjusts to enforce zero penetration at the LCP equilibrium. Deeper "target" penetration → larger λ_n required to satisfy the complementarity → more impulse delivered. Result: roughly linear scaling with depth.

**The 1× crossing at ~0.55 mm is the depth at which the rigid LCS delivers the same impulse over Δt = 0.05 s as the compliant Drake.** It is a quantitative fingerprint of the rigid-vs-compliant mismatch, not a tunable parameter of the LCS.

#### (4) §7.13 burst mechanism: REINTERPRETED

§7.13 (3) said the sub-stepping mechanism is "the single-step LCS AVERAGES a contact-burst over 0.05 s; the sub-stepped path captures the burst." The contact-burst language survives but its physical interpretation is now sharpened:

**The burst is the rigid LCS's response to penetration.** When the LCP is solved at finer Δt, the rigid contact delivers a sharp impulse to enforce non-penetration AS IF IT IS HAPPENING NOW; Drake's compliant contact spreads the same momentum over a finite-stiffness time constant. At Δt=0.05, the LCS averages the burst into a smaller mean force (under-predicts); at Δt=0.005, the LCS resolves the burst and the magnitude is governed by depth (over- or under-predicts per depth). **The contact-burst mechanism is a DOWNSTREAM CONSEQUENCE of rigid-vs-compliant, not an independent finding.**

#### (5) §7.14 finally settled

§7.14's audit-broken finding (contamination real, contact-pair-count gate essential) STANDS. §7.14's optimistic framing (§7.15 already refuted: "the whole gap may be the arm") is REFUTED. §7.14's deferral of friction + anitescu pending a clean residual is now: **friction stays DEFERRED indefinitely** (the residual was never tangential to begin with), and **anitescu is RE-PROMOTED** to the indicated direction (the reference's velocity-level convex compliance reformulation is the exact structural axis the sweep names).

#### (6) Convergence STAYS HELD

The model is now characterized as **rigid-where-Drake-is-compliant**. Tuning the C3+ ADMM against this model would tune to a wrong contact model, not a solver-convergence issue. Convergence stays held until the contact model is brought into the compliant family (anitescu reformulation, or an explicit compliance term).

#### (7) Routing consequences — friction DEMOTED, anitescu RE-PROMOTED

| Item | Pre-sweep status | Post-sweep status |
|---|---|---|
| Friction audit | RE-OPENED at clean setup (§7.15 main body) | **DEMOTED** — residual is normal, not tangential; friction audit re-opens ONLY IF next-block characterization rules out compliance |
| Anitescu | Parked-pending-FRICTION-BARELY (§7.13, §7.15) | **RE-PROMOTED** — the indicated direction; scope the port (separate block) |
| Penetration sweep | NEXT GATE (§7.15-aug) | DONE — NORMAL-COMPLIANCE routed |
| Convergence | HELD (§7.15) | **STILL HELD** — but for a sharpened reason (contact model wrong, not solver wrong) |
| Sweet-spot depth | ~0.5–1 mm hypothesized (§7.15) | **~0.549 mm measured** — quantitative handle on the rigid-vs-compliant mismatch |

#### (8) Strategic framing — confirmed

The §7.15-aug strategic position was: "we now have a FINGERPRINT pointing at WHAT the residual is." The sweep confirms the fingerprint quantitatively. The §7.13 mechanism findings (drag-redherring, contact-burst, DRAG-MATTERS-MOVING, vertical count=4) all stand. The §7.11→§7.16 chain converges on a single physical statement: **the LCS contact model is rigid where Drake is compliant; the gap is in the normal-force law; the fix is to reformulate the contact model (anitescu) or add an explicit compliance term, NOT to tune friction or the solver.**

#### (9) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **NORMAL-COMPLIANCE confirmed** — factor crosses 1× SMOOTHLY at ~0.549 mm penetration, monotonic 1.684× → 0.391× across 0.1–1.0 mm; all 6 depths gate-PASS.
- **The residual is rigid-vs-compliant NORMAL contact**, NOT friction; the §7.15 friction-audit-reopens plan is overturned.
- **Friction DEMOTED**; **anitescu RE-PROMOTED** as the indicated direction (the reference's velocity-level convex compliance reformulation).
- **Sweet-spot depth measured at ~0.549 mm** — a quantitative handle on the rigid-vs-compliant mismatch.
- **§7.13 contact-burst mechanism REINTERPRETED**: a downstream consequence of rigid-vs-compliant, not an independent finding.
- **Convergence:** HELD — model is rigid-where-Drake-is-compliant; contact-model reformulation precedes solver tuning.

#### Anti-stale binding

Any subsequent entry that schedules the friction audit at this stage is operating on a STALE record — §7.16 demotes friction to contingent-only-if-compliance-is-ruled-out. Any entry that treats anitescu as "parked-pending-FRICTION-BARELY" (§7.13 / §7.15 language) is also stale — §7.16 promotes anitescu to the indicated direction. The §7.13 contact-burst mechanism may still be cited as a sub-stepping-resolves-the-burst statement, but the burst itself is now identified as the rigid-LCS response to enforced non-penetration, not a property of the contact independent of the model class.

**Next gate (corrected from §7.15-aug):** characterize the normal-compliance gap quantitatively (sub-step force/velocity profile vs Drake's compliant force history at the SAME state), then scope the anitescu port direction (a separate block — not just a probe, an implementation scoping decision). Friction audit re-opens only if characterization rules out compliance. NOT actioned in this plan-doc edit.

### 7.16 — augmentation (2026-06-25): force-level-confirmation framing for the characterization + scoping-discipline guardrail for the anitescu phase-transition + milestone framing

#### (1) CONFIRM §7.16's core (banked; restated as the augment's anchor)

The gap factor crosses 1× **SMOOTHLY and MONOTONICALLY at ~0.549 mm** across the swept depths [0.10, 0.20, 0.30, 0.50, 0.70, 1.00 mm]; all six depths PASS the per-state §7.14 hard gate (the 1 mm OVER-prediction is REAL, GATE-FAILS-DEEP ruled out, the §7.15-aug "1 mm inversion may have been arm-contaminated" concern laid to rest). **ROUTE NORMAL-COMPLIANCE.** Drake's box motion is stable across depths (-1.87 to -2.33 mm) while the LCS scales nearly linearly (-1.21 to -5.73 mm, ~5× over a 10× depth change) — the rigid-vs-compliant normal-force signature (Drake compliant point-contact softens with depth; LCS Stewart-Trinkle is rigid). The 0.549 mm crossing is the quantitative fingerprint: the depth at which the rigid LCS delivers the same impulse Drake's compliant contact delivers over Δt = 0.05 s. Routing consequences (banked): **friction DEMOTED**; **anitescu RE-PROMOTED to indicated** (the reference's velocity-level convex compliance reformulation is the named axis); §7.13 contact-burst REINTERPRETED as a downstream consequence of rigid-vs-compliant, not an independent finding; §7.14 settled (gate stands, friction deferred indefinitely, anitescu indicated); **convergence HELD** for the sharpened reason (contact-model wrong, not solver — reformulation precedes tuning).

#### (2) THE FORCE-LEVEL-CONFIRMATION FRAMING (augment, for the characterization)

The sweep diagnosed rigid-vs-compliant at the **DISPLACEMENT level** (box Δx vs depth). The characterization should LOCK the diagnosis at the **FORCE level**: compare the LCS's normal force/impulse profile over the 10 sub-steps vs Drake's compliant force history at the SAME state (re-posed clean, box-pinned, at one or more swept depths — the 0.549 mm sweet-spot and an off-spot pair, e.g., 0.10 mm and 1.00 mm, to map the under-spot AND the over-spot).

**Pre-registered force-level signatures:**

| Side | Expected signature if rigid-vs-compliant |
|---|---|
| LCS sub-step λ_n history | **rigid-impulsive** — sharp early spike, magnitude scaling with penetration depth, decaying quickly as the box accelerates away from the contact |
| Drake ContactResults @ 1 ms ticks | **soft-spread** — depth-stable peak force, distributed over the contact's stiffness time-constant, weaker depth-dependence |
| LCS vs Drake at the sweet-spot (~0.549 mm) | total impulse over Δt = 0.05 s should match by construction (that's what the displacement-level crossing was); time-profiles still differ |

**Three-outcome pre-registration:**

- **CONFIRMED-COMPLIANCE** — force profiles match the pre-registered rigid-vs-compliant shapes (LCS impulsive, Drake soft-spread; impulse magnitudes match at the sweet-spot; magnitudes diverge per depth in opposite directions for the over-spot and under-spot). The diagnosis is **locked at the force level**; anitescu scoping proceeds on solid ground.
- **UNEXPECTED-MATCH** — force profiles are similar (LCS not impulsive OR Drake not soft-spread), yet displacements diverge as §7.16 showed. The mechanism is something else (e.g., an A-matrix dynamics artifact masquerading as compliance). **DO NOT proceed to anitescu** — re-examine.
- **PARTIAL** — force profiles partially match (e.g., LCS impulsive ✓ but Drake also impulsive ✗, or vice versa). Probably mixed mechanism. Characterize the residual before anitescu.

**The discipline: force-level confirmation BEFORE the reformulation.** The displacement-level evidence is strong but not load-bearing alone for a structural model change. The force-level test is cheap (offline, same constructed state, just dump per-tick force/impulse and plot) and is the natural follow-up to the sweep.

#### (3) THE SCOPING-DISCIPLINE GUARDRAIL (augment, the phase-transition caution)

Anitescu is the **FIRST genuinely LARGE structural change** in this arc. Everything prior has been:
- offline diagnosis on constructed states (near-zero compute cost; reversible by definition — nothing edited in `lcs_formulator.py`'s primary path)
- one-flag toggles (`LCS_EXPLICIT_BOX_GND`, `EE_PEN_M`, `Δt`, `box_ground_drag`) — small surface area, each isolating one variable

Porting the velocity-level convex compliance reformulation touches the **LCS construction itself** — the structure of `linearize_discrete_ee_space`, the meaning of E/F/H/c, the relationship between λ and v_box. It is real engineering with its own bug surface, and it **changes the model that the entire pipeline runs on** (the OSC executor, the C3+ ADMM solver, the sampling controller all consume the LCS).

**DISCIPLINE for the phase transition (the guardrail):**

| Discipline | Application |
|---|---|
| **(a) SCOPE, not BUILD** | The next block is a SCOPING block, not an implementation block. What does anitescu's reformulation entail? How much of `lcs_formulator.py` does it touch? What is the reference's exact construction (read it carefully)? What does it change in E/F/H/c semantics? What are the API touch-points downstream? Produce a written scope, not code. |
| **(b) STAGE BEHIND A FLAG** | Like `LCS_EXPLICIT_BOX_GND`, the anitescu path goes behind a `LCS_USE_ANITESCU` (or similar) flag, **default-OFF**. The Stewart-Trinkle path remains the default until validated. This makes the change reversible without git revert. |
| **(c) VALIDATE OFFLINE FIRST** | Validate the anitescu LCS against Drake on the clean (box-pinned) constructed state — re-run the §7.16 sweep, but with `LCS_USE_ANITESCU=1`. If compliance is the right diagnosis, the factor should close to ~1× **ACROSS depths** (not just at the 0.549 mm sweet spot — the sweep's smooth crossing was the rigid signature; if the new model is right, the crossing should flatten near 1× everywhere). Then re-run the friction audit (cheap, well-defined now) ONLY if any residual remains. Then — and only then — touch the live pipeline. |
| **(d) CHEAP BEFORE EXPENSIVE; ONE MECHANISM AT A TIME** | The §7.10–§7.16 chain enforced this and it kept the contamination + sign-flip recoverable. Here the surface area is larger so the discipline matters MORE. **Do not combine anitescu with a friction-audit fix, or with a Δt change, or with a solver toggle.** One axis at a time. |

**The diagnosis phase is DONE.** The build phase has the most risk — gate it the same way everything else was gated.

#### (4) STRATEGIC FRAMING — the milestone

The no-push was a MODEL problem, and the model problem is now **FULLY NAMED**:

| Mechanism | Status | Evidence |
|---|---|---|
| Missing floor contact | **FIXED** | §7.10 vertical-only confirmation at `LCS_EXPLICIT_BOX_GND=4`; vertical |Δz| ≤ 0.01 mm across all subsequent probes |
| Rigid-vs-compliant normal contact | **DIAGNOSED; anitescu indicated** | §7.16 monotonic crossing of 1× at 0.549 mm across six gated-clean depths |

The port under-pushed because its LCS used Stewart-Trinkle rigid contact where Drake + the reference use compliant contact; the sweep gives the quantitative crossing (~0.549 mm) that proves it. **A clean, defensible root-cause story for the entire no-push arc, pointing at a specific reference-aligned fix.** The DIAGNOSIS phase is done; the BUILD phase (port anitescu) is substantial but well-defined.

#### (5) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **NORMAL-COMPLIANCE diagnosed** (gap crosses 1× SMOOTHLY at 0.549 mm penetration, all six depths gate-clean; Drake-stable / LCS-linear = rigid-vs-compliant normal-force signature).
- **Friction DEMOTED**; **anitescu RE-PROMOTED to indicated** (the reference's velocity-level convex compliance reformulation is the named axis).
- **§7.13 contact-burst REINTERPRETED** as a downstream consequence of rigid-vs-compliant, not an independent finding.
- **Convergence: HELD** for the sharpened reason (model wrong, not solver).
- **NEXT GATE** = characterize at the FORCE LEVEL (lock the diagnosis at force) + then SCOPE anitescu (scope-not-build, stage-behind-flag, validate-offline-first); each is a separate block.
- **The no-push root cause is FULLY NAMED**: missing floor [FIXED] + rigid-vs-compliant normal contact [DIAGNOSED].

#### Anti-stale binding (augment)

Any subsequent entry that proceeds to anitescu port implementation WITHOUT the force-level-confirmation pre-step is operating on a STALE record — §7.16-aug requires the displacement-level diagnosis to be locked at the force level before committing to a structural model change. Any entry that lands an anitescu path WITHOUT the (a)–(d) guardrail (scope-first, flag-staged, offline-validated, one-axis) is also stale — the §7.10–§7.16 cheap-before-expensive discipline applies MORE at the build phase, not less, because the surface area to go wrong is larger. Any entry that frames "the no-push is solved" without distinguishing the two sub-mechanisms (floor [FIXED] vs compliance [DIAGNOSED, not yet fixed]) is stale — only the first sub-mechanism is closed in the codebase; the second is closed in the diagnosis but open in the build.

**Next gate (corrected from §7.16 main body):** force-level confirmation probe — dump LCS sub-step λ_n history + Drake ContactResults force history at the same (box-pinned, clean) state at three depths (0.10, ~0.55, 1.00 mm) and verify rigid-impulsive (LCS) vs soft-spread (Drake) profiles match the pre-registered shapes. **Only after CONFIRMED-COMPLIANCE** does anitescu scoping open (and that is itself a SCOPE block, not a BUILD block). NOT actioned in this plan-doc edit.

### 7.17 — Force-level confirmation probe (Part A): FORCE-DISCONFIRMS — LCS shape IS impulsive (rigid signature ✓) but scaling sub-linear; Drake force is OSCILLATING (intermittent contact, not soft-spread); discovered §7.16's 1 mm number was a PARTIAL LCS run; anitescu Part B PAUSED (2026-06-25)

**Result: the §7.16 displacement-level NORMAL-COMPLIANCE diagnosis is NOT cleanly locked at the force level. The LCS λ_n sub-step profile DOES show rigid-impulsive shape (sharp spike at step 0, decay after) — one of the pre-registered signatures matches. But the other three pre-registered signatures FAIL: (i) LCS peak λ_n scales SUB-linearly with depth (2.65× growth for 10× depth change, not the rigid-linear 10×); (ii) Drake's compliant ContactResults force history is OSCILLATING, not soft-spread, consistent with intermittent contact (the box bounces off the pusher); (iii) Drake peak force slightly DECREASES with depth (30.4 N → 23.1 N), not depth-stable. Additionally: at 1.00 mm the LCS sub-step IK FAILS at step 4, meaning §7.16's 1 mm Δbox_x = −5.73 mm and factor 0.391× came from a 3-4-substep PARTIAL LCS run, not the full 10-step sub-stepping — a §7.16 confound that was previously unnamed. Per §7.16-aug discipline, the displacement-level evidence does not lock the mechanism cleanly enough to justify the anitescu phase transition. ROUTE FORCE-DISCONFIRMS. Anitescu Part B PAUSED.** Banked here. Commit `8b55c94` (`scripts/_stage_c_force_level_probe.py` + `stage_c/force_level_probe_output.txt`).

#### (1) The four pre-registered force-level signatures — split outcome

§7.16-aug (2) pre-registered four signatures of rigid-vs-compliant at the force level. Outcome:

| Pre-registered signature | Expected | Observed | Match? |
|---|---|---|---|
| LCS λ_n profile SHAPE | rigid-impulsive (sharp early spike, decay) | spike at step 0 (3.470 at 0.10 mm; 9.200 at 1.00 mm), λ_n[0]/λ_n[-1] = 5.31× at 0.10 mm | ✓ |
| LCS peak λ_n scaling with depth | rigid-linear (~10× growth for 10× depth) | 3.47 → 9.20 = **2.65× growth** | ✗ sub-linear |
| Drake force profile SHAPE | soft-spread (depth-stable peak, distributed) | OSCILLATING force, F_x at every 5 ms = `[0, -6.4, 0, -5.2, -2.2, -0.8, -8.2, -1.4, -5.1, -7.2, -0.9]` N at 0.10 mm | ✗ intermittent contact |
| Drake peak force scaling with depth | compliant-stable (~1× growth) | 30.4 N → 23.1 N = **0.76× growth** (slightly DECREASING) | ✗ inverse |

**Only 1 of 4 pre-registered signatures matched.** The §7.16 displacement-level diagnosis is NOT cleanly locked at the force level.

#### (2) THE §7.16 1 mm CONFOUND — now named

At EE_PEN_M = 1.00 mm, the LCS sub-step machinery FAILS at step 4 (sub-step IK cannot converge after the box accelerates fast under the deep-penetration contact force). The probe records `λ_n = [9.200, 0.000, 0.000, 0.000, NaN, NaN, NaN, NaN, NaN, NaN]` — only 3-4 sub-steps complete before the trajectory aborts.

**§7.16's 1 mm Δbox_x = −5.731 mm came from a PARTIAL LCS run** (terminated at sub-step ~3 = t=0.015 s), not the full 10-step sub-stepping over Δt=0.05 s. The reported factor 0.391× (the "OVER-prediction") may therefore be an ARTIFACT of partial-trajectory truncation: the LCS delivered a big first-step impulse, then aborted before the rest of the dynamics could play out. A clean 10-step run would have produced a smaller Δbox_x, possibly closer to Drake's reference, possibly weakening or reversing the §7.16 monotonic crossing-of-1×.

This confound was not visible in the §7.16 sweep output because that probe printed `fail` counts but did NOT correlate them with the reported factor. The force-level probe surfaced it via direct λ_n NaN markers.

#### (3) Drake's intermittent-contact dynamics — the unmodeled mechanism

Drake's force history at 0.10 mm shows the box-pusher contact ENGAGING and DISENGAGING multiple times across the 50 ms window — `F_x = [0, -6.4, 0, -5.2, -2.2, -0.8, -8.2, -1.4, -5.1, -7.2, -0.9]` N. The zeros at intermediate ticks (5 ms, 10 ms) indicate contact loss; the spikes (-6.4, -5.2, -8.2 N) indicate re-engagement. The box is BOUNCING.

This is consistent with: the box's compliant floor contact lets it lift slightly when pushed; the pusher's compliant contact releases when the box lifts; the box falls back; contact re-engages. Drake's "compliant point-contact" at this geometry produces an OSCILLATORY, not soft-spread, force trace.

**Drake's actual mechanism is intermittent-bouncing-contact, not the textbook "spread distributed force" of compliant contact.** The §7.16-aug pre-registration ("soft-spread depth-stable") was a CARTOON of compliant contact that doesn't match the actual Drake dynamics at this state. The §7.16 displacement-level signature may have been comparing rigid-Stewart-Trinkle to dynamic-bouncing-Drake, not rigid-vs-compliant in the textbook sense.

#### (4) 0.549 mm sweet-spot IK FAIL — a side note

The probe also tried the §7.16 interpolated crossing depth (0.549 mm) but the box-pinned IK with the per-pair clearance constraints + posture cost failed on both seeds tried. The §7.16 sweep at 0.50 mm worked (factor 1.112×), so 0.549 mm should be feasible with more seeds. Not pursued here — the 0.10 mm and 1.00 mm pair is sufficient to read the depth-scaling signature.

#### (5) ROUTE: FORCE-DISCONFIRMS — anitescu Part B PAUSED

Per the §7.16-aug discipline ("force-level confirmation BEFORE the reformulation") and the §7.17 result (1-of-4 signatures + 1 mm partial-run confound + Drake intermittent-contact unmodeled), **the diagnosis is NOT locked at the force level**. Anitescu Part B does NOT open.

Routing consequences:

| Item | Pre-§7.17 | Post-§7.17 |
|---|---|---|
| Anitescu Part B (scoping) | OPEN after §7.16 (per §7.16-aug) | **PAUSED** — force-level confirmation failed |
| §7.16 displacement-level diagnosis | NORMAL-COMPLIANCE (banked) | **UNDER SUSPICION** — 1 mm point was a partial LCS run; the monotonic crossing may be partly artifact |
| Friction audit | DEMOTED (§7.16) | **STILL DEMOTED** — no evidence of tangential signal; not the indicated direction |
| Convergence | HELD (§7.16) | **STILL HELD** — model is wrong for some mechanism, just not cleanly named |

#### (6) Strategic framing — honest correction

§7.16's NORMAL-COMPLIANCE framing was REAL at the displacement level but possibly OVERINTERPRETED in mechanism. The §7.17 cleanup reveals:
- A real partial-LCS-run confound at deeper depths (§7.16's 1 mm number is partially an artifact)
- Drake's actual dynamics at this state include intermittent contact (bouncing), not just smooth compliance
- The LCS sub-step force shape IS rigid-impulsive, but the magnitude scaling is sub-linear

**The "no-push root cause is FULLY NAMED" milestone (§7.16-aug (4)) is RESCINDED.** The floor sub-mechanism is still FIXED (§7.10). The contact-axis sub-mechanism is **DIAGNOSED-WITH-CONFOUNDS** — there is a real model-vs-plant gap (the §7.11 / §7.13 / §7.15 factors stand at the displacement level) but the precise mechanism is not cleanly named.

This is the same discipline the §7.10 / §7.14 audits enforced: cautious-against-comfortable-re-interpretation. The §7.16 "we have a fingerprint" framing was attractive; §7.17 forces us back to "we have an unresolved residual that includes an unnamed mechanism."

#### (7) Next gate — re-examine §7.16's deep-depth confounds, NOT anitescu

The §7.16 sweep must be re-run with sub-step IK robustness improvements (more seeds per sub-step, fall-back q_arm warm starts, or accept-partial-and-flag) so the deep-depth points are not partial-LCS artifacts. Also: characterize Drake's intermittent-contact dynamics at this state — is the box bouncing off the FLOOR (rigid-ish floor contact in Drake → vertical bounce → horizontal contact intermittency)? If yes, the §7.16 "Drake stable" framing may have been an integral-over-bouncing average, not a clean compliant response.

**If §7.16 SURVIVES this cleanup** → re-do this force-level probe with the cleanup; if force-level CONFIRMS-COMPLIANCE then → anitescu Part B opens. **If §7.16 does NOT survive** → the diagnosis itself revisits; possibly the "rigid-vs-compliant" framing is too simple, and the actual gap is in dynamics (a mixture of contact-model and integration-method differences).

#### (8) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **Force-level confirmation FAILED** — only 1 of 4 pre-registered signatures matched (LCS shape impulsive ✓; LCS scaling sub-linear ✗; Drake oscillating not spread ✗; Drake scaling slightly inverse ✗).
- **§7.16 1 mm number = PARTIAL LCS run** (sub-step IK fails at step 4) — the monotonic crossing was partly an artifact of partial sub-stepping at deeper depths.
- **Anitescu Part B PAUSED** — diagnosis not locked at the force level.
- **§7.16 displacement-level signature** UNDER SUSPICION pending §7.16 sweep cleanup.
- **The "no-push root cause is FULLY NAMED" milestone is RESCINDED.** Floor [FIXED] stands; contact-axis [DIAGNOSED-WITH-CONFOUNDS, not cleanly named].
- **Friction:** still DEMOTED. **Convergence:** still HELD. **Anitescu:** PAUSED, not parked-pending-friction; awaiting §7.16 cleanup outcome.

#### Anti-stale binding

Any subsequent entry that proceeds to anitescu scoping or implementation based on §7.16 alone is operating on a STALE record — §7.17 demotes §7.16's force-level interpretation pending the deep-depth sub-step IK cleanup. Any entry that cites §7.16's 1 mm Δbox_x = −5.73 mm or factor 0.391× as a clean force-level signal is stale — it is a partial-LCS-run artifact. Any entry that cites the "no-push root cause is FULLY NAMED" milestone (§7.16-aug (4)) is stale — §7.17 rescinds that framing; the floor sub-mechanism is FIXED but the contact-axis sub-mechanism is DIAGNOSED-WITH-CONFOUNDS, not cleanly named.

**Next gate (corrected from §7.16-aug):** §7.16 sweep cleanup — re-run the penetration sweep with sub-step IK robustness improvements + correlate fail-counts with reported factors (the deep-depth points must be flagged or re-solved if partial). If the §7.16 monotonic crossing SURVIVES the cleanup, re-do this force-level probe; only on CONFIRMED-COMPLIANCE does anitescu Part B open. NOT actioned in this plan-doc edit.

### 7.17 — augmentation (2026-06-25): cartoon correction + cleanup discipline (partial-IK INVALIDATES its factor, not flags it) + dual question (crossing-survival ≠ mechanism-name) + honest strategic framing

#### (1) CONFIRM §7.17's core (banked; restated as the augment's anchor)

Part A FORCE-DISCONFIRMS: only **1 of 4** force-level signatures matched (LCS λ_n spike-decay shape ✓; LCS peak scaling 2.65× not ~10× ✗ sub-linear; Drake force OSCILLATING / intermittent not soft-spread ✗; Drake peak scaling 0.76× slightly inverse ✗). Two confounds surfaced: (a) §7.16's 1 mm number came from a PARTIAL sub-step run — IK fails at step 4 (`λ_n = [9.2, 0, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN]`); the −5.73 mm / 0.391× came from a 3–4-substep partial trajectory, so the §7.16 monotonic crossing may be partly a partial-sub-stepping artifact at deeper depths; the sweep printed fail counts but did NOT correlate them with factors; (b) Drake's actual dynamics are INTERMITTENT-CONTACT bouncing (zeros at intermediate ticks = contact loss; spikes = re-engagement), not textbook compliant softness. ROUTE FORCE-DISCONFIRMS → Part B (anitescu scoping) PAUSED, not opened. The §7.16-aug "FULLY NAMED" milestone RESCINDED — floor sub-mechanism FIXED (§7.10); contact-axis sub-mechanism DIAGNOSED-WITH-CONFOUNDS, not cleanly named. Friction DEMOTED; convergence HELD; anitescu PAUSED, not parked-pending-friction (awaiting §7.16 cleanup outcome).

#### (2) THE CARTOON CORRECTION — pre-registered signatures must come from system behavior, not textbook idealization

The §7.16-aug (2) pre-registered Drake signature — *"soft-spread depth-stable peak force distributed over the contact's stiffness time-constant"* — was a **CARTOON** of compliant contact, a textbook mental model that does NOT match Drake's actual behavior at this state. Drake here shows **intermittent-contact BOUNCING** (oscillating force, contact loss / re-engagement) — a real dynamical phenomenon driven by the box-mass / contact-stiffness / push-velocity / sub-mm-bounce geometry of the constructed state.

This was a reviewer framing baked into the pre-registered signatures; **superseded.** The lesson generalises: **pre-registered "expected signatures" must come from the actual system's observed behavior, NOT from a textbook idealization.** An idealized expectation can make a real result look like a mismatch (a real soft-spread response convolved with bouncing reads as "doesn't match the cartoon" → false UNEXPECTED-MATCH) or it can let a coincidence look like a match (a partial-LCS run with an impulsive shape happens to match "rigid-impulsive" → false CONFIRMED-COMPLIANCE). Either way, the cartoon biases the read.

Concretely: future pre-registrations must derive expected force/displacement signatures by **running the actual system first on a known case, observing the shape, and only then writing the prediction.** The §7.16-aug pre-registration skipped this and got penalised.

#### (3) THE CLEANUP DISCIPLINE — partial-IK INVALIDATES its factor (stronger than "flag")

The §7.16 cleanup must treat a partial-IK sub-step run as **INVALIDATING** its factor, NOT merely flagging it. The original sweep's error was not that the IK failed — IK failure is a legitimate outcome — it is that **the failure was printed but the resulting factor was reported as if clean** (the −5.73 mm / 0.391× from a 3–4-substep partial trajectory was tabulated alongside the clean factors and entered the monotonic-crossing determination).

This is the **§7.14 lesson one level deeper**: an unchecked precondition (full-trajectory completion) silently corrupting a number. The fix has the same shape as the contact-pair gate: **assert the precondition PER-POINT before the number counts.** A factor from a partial trajectory does NOT enter the crossing determination — **it is EXCLUDED, not annotated.**

Operationally, the cleanup probe must:
- Per depth, attempt the sub-stepped LCS with multiple seed q_arm warm-starts (try several IK starting postures per sub-step, accept the first that converges).
- If the sub-stepped trajectory fails to complete all 10 sub-steps **even with retries**, the factor for that depth is **EXCLUDED from the table**, not reported.
- Correlate the per-depth fail count with the reported factor in the output (every printed row must be marked clean / partial / failed; only `clean` rows enter route determination).
- The factor table at the end shows only clean depths; the route logic operates only on those.

The §7.14 contact-pair-count gate stopped a model-vs-plant comparison from running on contaminated state. The §7.17 partial-IK gate stops a crossing determination from including partial-trajectory factors.

#### (4) THE DUAL QUESTION — crossing-survival ≠ mechanism-name

The cleanup answers TWO separate questions that **must not be conflated**:

| Question | What it tests | What the cleanup decides |
|---|---|---|
| **(i) Does the §7.16 monotonic 1× crossing SURVIVE on clean full-trajectory runs?** | Whether the displacement-level signature itself stands once partial-LCS contributions are excluded. | Empirical re-measurement at clean depths. |
| **(ii) IF it survives, is "rigid-vs-compliant" still the right MECHANISM NAME, given Drake shows INTERMITTENT CONTACT (bouncing) not smooth compliance?** | Whether the surviving signature can be attributed to the compliance-stiffness axis vs the bouncing-dynamics axis. | A separate force-level / contact-loss probe, not the cleanup itself. |

**Survival of the crossing does NOT automatically restore the compliance diagnosis.** A momentum-driven push that produces genuine contact-loss-and-re-engagement is a DIFFERENT phenomenon than a compliance-stiffness mismatch — even if both can produce the same monotonic crossing of factor 1× across penetration depth (one because deeper penetration → stiffer rigid LCS response; the other because deeper penetration → faster box recoil → different bounce timing → different time-averaged Drake impulse).

**Do NOT let "the crossing survived" re-assert "compliance".** The mechanism name is a SEPARATE determination, **re-opened by the intermittent-contact observation** in §7.17 (3). Even on a fully clean §7.16 sweep with a surviving crossing, the contact-axis sub-mechanism would still be DIAGNOSED-WITH-CONFOUNDS until a separate probe distinguishes compliance from bouncing.

#### (5) STRATEGIC FRAMING — honest

We are further from done than the §7.16-aug "FULLY NAMED" claimed. This is the difference between a real diagnosis and a premature one — the kind of gap the discipline exists to catch.

Anitescu (a turn from being scoped toward a build) is correctly PAUSED, which likely SAVED a substantial wasted effort: porting a compliance reformulation against a diagnosis that had not survived its own force-level check would have produced a working anitescu LCS that still didn't match Drake — and the failure mode (a clean reformulation that doesn't close the gap) would have been hard to attribute (is it a port bug? is it the diagnosis being wrong? is it something else?). Pausing returns the question to the diagnosis layer, where it's tractable.

**The §7.16-aug guardrail (confirm-at-force-level-before-reformulating) paid for itself.** One careful Part A probe vs a full anitescu port built on a partly-artifactual diagnosis. This pattern should be the default for every future "we have a fingerprint → let's port the reference's mechanism" turn.

Convergence stays HELD — the model is still not validated on push, now for a more honest reason (the gap is real; the mechanism is not cleanly named; the displacement-level signature has a partial-run confound at deep depths; the textbook compliant-contact cartoon doesn't describe Drake at this state).

#### (6) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **Part A FORCE-DISCONFIRMS** — 1-of-4 force-level signatures; the §7.16 crossing is partly a partial-IK artifact at deep depths; Drake shows intermittent-contact bouncing, not the soft-spread cartoon.
- **Milestone "FULLY NAMED" RESCINDED** → DIAGNOSED-WITH-CONFOUNDS.
- **Anitescu PAUSED** (correctly — not built against a partly-artifactual diagnosis).
- **NEXT** = §7.16 sweep cleanup with the **partial-IK INVALIDATES its factor** discipline + per-point partial/fail correlation; then the **dual question** (crossing-survival ≠ compliance-name re-assertion).
- **Convergence:** HELD.

#### Anti-stale binding (augment)

Any subsequent entry that reports a sub-step LCS factor without verifying full-trajectory completion is operating on a STALE record — §7.17-aug requires the partial-IK INVALIDATES discipline at every printed factor. Any entry that, after a §7.16 cleanup with a SURVIVING crossing, jumps directly to "compliance is confirmed, anitescu scoping opens" is also stale — §7.17-aug separates crossing-survival from mechanism-name; the bouncing observation requires a separate determination before the anitescu phase transition re-opens. Any entry that reuses a textbook-idealization "expected signature" without first observing the actual system's behavior is also stale — §7.17-aug (2) records the cartoon correction explicitly.

**Next gate (corrected from §7.17 main body):** §7.16 sweep cleanup, with two specific differences from the original sweep: (a) per-sub-step IK retry with multiple seed warm-starts; partial-trajectory factors are EXCLUDED from the table, not annotated; (b) per-depth `clean / partial / failed` marker on every printed row; only `clean` rows enter the crossing determination. AFTER the cleanup, the dual question — first did the crossing survive (re-measure), second is the mechanism still compliance (a separate intermittent-contact-vs-compliance probe, not the cleanup itself). NOT actioned in this plan-doc edit.

### 7.18 — §7.16 sweep cleanup: CROSSING-SURVIVES — all 6 depths complete 10/10 sub-steps CLEAN under robust IK; factors IDENTICAL to §7.16 (within 1e-4); §7.17 partial-trajectory finding qualitatively true but factor-invariant; mechanism-name still OPEN per the §7.17-aug dual question (2026-06-26)

**Result: with robust per-sub-step IK (5 seeds: warm-start, posture, 3 random perturbations), all 6 §7.16 depths complete 10/10 sub-steps as CLEAN full trajectories — no partial-trajectory exclusions, no gate failures. The reported factors are IDENTICAL to §7.16 to 4 decimal places (max |diff| = 4e-4). The §7.16 monotonic 1× crossing at ~0.549 mm SURVIVES the cleanup.** Per the §7.17-aug DUAL QUESTION discipline, this answers question (i) "does the displacement-level signature survive" — YES — but does NOT answer question (ii) "is the mechanism name still rigid-vs-compliant given Drake shows intermittent contact" — that question stays OPEN, awaiting a separate mechanism-name probe. **§7.17's force-level disconfirmation STILL STANDS.** Anitescu Part B stays PAUSED — survival of the crossing does NOT auto-restore compliance. Banked here. Commit `5037820` (`scripts/_stage_c_sweep_cleanup.py` + `stage_c/sweep_cleanup_output.txt`).

#### (1) The cleanup table

| EE_PEN | status | n_sub_completed | fail_step | Drake Δx (mm) | LCS Δx (mm) | factor (clean) | §7.16-original | diff |
|---|---|---|---|---|---|---|---|---|
| 0.10 mm | CLEAN | **10/10** | -1 | -2.0331 | -1.2072 | **1.684×** | 1.684× | +0.000 |
| 0.20 mm | CLEAN | **10/10** | -1 | -2.0144 | -1.3670 | **1.474×** | 1.474× | -0.000 |
| 0.30 mm | CLEAN | **10/10** | -1 | -1.8729 | -1.5051 | **1.244×** | 1.244× | +0.000 |
| 0.50 mm | CLEAN | **10/10** | -1 | -2.3288 | -2.0948 | **1.112×** | 1.112× | -0.000 |
| 0.70 mm | CLEAN | **10/10** | -1 | -2.3029 | -3.5244 | **0.653×** | 0.653× | +0.000 |
| 1.00 mm | CLEAN | **10/10** | -1 | -2.2388 | -5.7311 | **0.391×** | 0.391× | -0.000 |

**CLEAN depths: 6/6. PARTIAL depths: 0. GATE FAIL depths: 0. Monotonic. Crosses 1× at ~0.549 mm — identical to §7.16.**

#### (2) §7.17's partial-trajectory finding — qualitatively true but factor-invariant; nuance

§7.17 observed that at EE_PEN_M = 1 mm, its `lcs_force_profile` probe had `λ_n = [9.200, 0.000, 0.000, 0.000, NaN, NaN, NaN, NaN, NaN, NaN]` — IK failed at step 4. The cleanup did NOT reproduce this failure (warm-start IK succeeded for all 10 sub-steps at 1 mm).

The difference: §7.17's force-level probe used only 2 setup seeds for `setup_state_at_depth` (vs the cleanup's 5), so it landed in a slightly different basin of attraction at the initial q_arm. Its sub-step IK then ran into the failure that the cleanup's q_arm (different basin, then warm-started) avoided.

**However: the §7.17 partial-trajectory factor (had it been computed at the truncation point) would have been THE SAME as the clean full-trajectory factor at 1 mm.** Why: in §7.17's partial trajectory, after step 0's big λ_n = 9.2 push, sub-steps 1–3 had λ_n = 0 (no contact admitted), so the LCS just propagated state via `A·x_curr` — free dynamics with no contact force. Whether the trajectory truncated at step 4 (partial) or continued to step 10 (clean), the box's final position was the same because nothing was pushing it. The factor 0.391× emerges from the same physics either way.

**The discipline §7.17-aug articulated stands** — partial-IK INVALIDATES its factor as a rule, because you cannot know a priori that the missing steps would have been zero-contribution. The specific §7.16 1 mm number, on its own evidence, happened to be invariant under this confound. So the §7.17 generalisation ("the §7.16 deep numbers are partly artifactual") was over-stated, but the §7.17-aug discipline ("partial-IK is invalid by precondition, not by outcome") survives.

#### (3) The §7.17 force-level disconfirmation STILL STANDS

This cleanup answers ONE question: does the §7.16 monotonic 1× crossing survive on clean full-trajectory runs? **YES.** This cleanup does NOT touch the §7.17 force-level findings:
- LCS λ_n profile shape IS impulsive (rigid signature ✓) — §7.17 (1)
- LCS peak λ_n scaling sub-linear (2.65× for 10× depth, not 10× rigid-linear) — §7.17 (1)
- Drake force profile OSCILLATING with intermittent contact, NOT soft-spread plateau — §7.17 (3)
- Drake peak force scaling slightly inverse (0.76× growth) — §7.17 (1)

The mechanism-name question — is the gap rigid-vs-compliant, or intermittent-contact-bouncing, or something else? — remains OPEN.

#### (4) The §7.17-aug DUAL QUESTION — question (i) ANSWERED, question (ii) STILL OPEN

| Question | Status |
|---|---|
| (i) Does the §7.16 monotonic crossing SURVIVE on clean full-trajectory runs? | **YES — confirmed at all 6 depths, factors identical to §7.16 to 1e-4** |
| (ii) IF it survives, is "rigid-vs-compliant" still the right MECHANISM NAME given Drake shows intermittent contact? | **STILL OPEN — requires a separate mechanism-name probe, not this cleanup** |

Per §7.17-aug (4): "Survival of the crossing does NOT automatically restore the compliance diagnosis. Even on a fully clean §7.16 sweep with a surviving crossing, the contact-axis sub-mechanism would still be DIAGNOSED-WITH-CONFOUNDS until a separate probe distinguishes compliance from bouncing." That guidance applies as written.

#### (5) Routing consequences

| Item | Pre-§7.18 | Post-§7.18 |
|---|---|---|
| §7.16 displacement-level signature | UNDER SUSPICION (§7.17) | **REVALIDATED — crossing survives cleanup; all factors clean** |
| §7.17 force-level FORCE-DISCONFIRMS | STANDS | **STILL STANDS — cleanup doesn't address it** |
| §7.17 "1 mm was partial-LCS artifact" claim | Banked as confound | **NUANCED — partial qualitatively true, factor-invariant; §7.17-aug discipline survives at the principle level** |
| Anitescu Part B (scoping) | PAUSED (§7.17) | **STILL PAUSED — awaiting mechanism-name determination** |
| Friction | DEMOTED | STILL DEMOTED |
| Convergence | HELD | STILL HELD |
| Milestone "FULLY NAMED" | RESCINDED (§7.17) | **Floor [FIXED] stands; contact-axis [DIAGNOSED-WITH-CONFOUNDS, mechanism-name still OPEN]** |

#### (6) Strategic framing — recalibration

§7.17 swung hard against §7.16 ("MILESTONE RESCINDED; FULLY NAMED gone; partial-LCS artifact"); §7.18 recovers some of that ground (the crossing IS real; the §7.16 factors are clean). The truth is in the middle:

- The §7.16 **displacement-level signature** is real and clean (§7.18).
- The §7.17 **force-level signatures** still don't cleanly match the pre-registered rigid-vs-compliant cartoon (§7.17 stands).
- Drake's intermittent-contact bouncing is real (§7.17), and it remains the most important alternative mechanism to compliance for explaining the surviving crossing.

**The conservative reading: the gap is real, monotonic in depth, and crosses 1× at ~0.549 mm — but we still don't know if it's compliance-stiffness mismatch or intermittent-contact bouncing.** Anitescu stays paused; mechanism-name probe is the next gate.

This is also a lesson on probe sensitivity: §7.17's "1 mm = partial" alarm was set off by a 2-seed setup that happened to land in a sub-step-IK-fragile basin; the 5-seed cleanup didn't reproduce the failure. The right takeaway is that **robust IK with multiple seeds is now a standard precondition for any sub-step LCS comparison**, codified in the cleanup script for re-use.

#### (7) Progress-table note (for next regeneration)

ADMM-solver row, HORIZONTAL/push axis:
- **§7.16 crossing SURVIVES cleanup** — all 6 depths CLEAN under robust IK; factors identical to §7.16 (within 1e-4); monotonic crossing at ~0.549 mm confirmed.
- **§7.17 force-level FORCE-DISCONFIRMS still STANDS** — the cleanup does not address it.
- **§7.17 "1 mm was partial artifact" claim NUANCED** — qualitatively true (was partial in §7.17's probe) but factor-invariant; the §7.17-aug discipline ("partial-IK invalidates its factor by precondition") still applies as a rule.
- **Mechanism-name question OPEN** — survival of crossing does NOT restore compliance per §7.17-aug dual question; intermittent-contact-vs-compliance probe is the next gate.
- **Anitescu Part B:** STILL PAUSED.
- **Floor sub-mechanism:** FIXED. **Contact-axis sub-mechanism:** DIAGNOSED-WITH-CONFOUNDS — signature real, mechanism not named.
- **Convergence:** HELD.

#### Anti-stale binding

Any subsequent entry that cites §7.17 to claim "the §7.16 crossing is partly artifactual" without the §7.18 nuance is operating on a STALE record — §7.18 establishes that §7.16's factors are clean and the crossing survives on full trajectories. Any entry that cites §7.18 to claim "compliance is back as the mechanism name" is also stale — §7.17-aug separates crossing-survival from mechanism-name; §7.18 answers only crossing-survival, not mechanism-name. The §7.17 force-level findings (LCS impulsive shape + sub-linear scaling; Drake intermittent contact; Drake inverse peak scaling) remain on the table as evidence that the displacement-level signature is not cleanly explained by rigid-vs-compliant in the textbook sense.

**Next gate (corrected from §7.17-aug):** mechanism-name probe — characterize Drake's contact-state time series across penetration depths (does the pusher↔box contact disengage and re-engage multiple times in the 50 ms window?); if intermittent contact is the dominant feature, the "compliance vs rigid" framing is incomplete and the dynamics are dominated by impulse-and-recoil. If intermittent contact is NOT dominant (continuous compliant push with smooth force history), compliance regains standing as the mechanism. NOT actioned in this plan-doc edit.

### 7.18 — augmentation (2026-06-26): reviewer-calibration note (the §7.17 alarm was an over-reaction cutting the OTHER way) + the discipline-survives-correction articulation + truth-in-the-middle framing

#### (1) CONFIRM §7.18's core (banked; restated as the augment's anchor)

§7.16 cleanup CROSSING-SURVIVES: **6/6 CLEAN** (0 PARTIAL, 0 GATE-FAIL), both deep points (0.7 mm, 1.0 mm) complete full 10-step trajectories under robust IK, every factor matches §7.16 to 4 decimals, crossing at ~0.549 mm IDENTICAL. The §7.17 partial-IK confound was **OVER-STATED**: §7.17's 2-seed setup hit an IK-fragile basin at 1 mm (failed at sub-step 4); the 5-seed cleanup didn't reproduce it; AND even in §7.17's partial trajectory the factor would have been THE SAME — the post-burst sub-steps had λ_n = 0 (no contact admitted), so the LCS propagated free dynamics (`A·x`) and the box position was INVARIANT under truncation. §7.17's generalisation ("the deep numbers are partly artifactual") was overstated. **BUT the §7.17-aug discipline (INVALIDATE-by-precondition-not-outcome) STILL APPLIES** — you cannot know a priori the missing steps would be zero-contribution (codified in the cleanup script via 5-seed retry + partial-EXCLUDES). Routing: §7.16 displacement-signature REVALIDATED; §7.17 force-level FORCE-DISCONFIRMS STILL STANDS (the cleanup doesn't touch it); mechanism-name STILL OPEN (crossing-survival does NOT auto-restore compliance); anitescu STILL PAUSED; floor [FIXED], contact-axis [DIAGNOSED-WITH-CONFOUNDS — signature real, mechanism not named]; friction DEMOTED; convergence HELD.

#### (2) THE REVIEWER-CALIBRATION NOTE — over-reaction in BOTH directions is the failure mode

The §7.17 partial-IK alarm was a reviewer **OVER-REACTION**: the reviewer read the partial run as undermining the deep factors, and amplified it to *"the §7.16 monotonic crossing may be partly an artifact."* The §7.18 cleanup shows the deep factors were clean (the truncated steps in §7.17 carried zero contact force, so the factor was invariant under truncation).

**This is the THIRD time in this arc a reviewer reading was overturned by the next probe** — but this one cuts the OTHER way:

| Prior reviewer overturn | Direction |
|---|---|
| §7.11 / §7.13 "Δt-DOMINANT, MODEL-FIXED close" → §7.14 audit broken | over-TRUST (premature victory lap) |
| §7.16 "fingerprint, FULLY NAMED, anitescu indicated" → §7.17 force-disconfirms | over-TRUST (premature victory lap) |
| §7.17 "1 mm = PARTIAL, crossing maybe artifact" → §7.18 cleanup clean | **over-DISTRUST (premature alarm)** |

**The calibration lesson: the failure mode is over-reaction in BOTH directions.** The base rate on "the comfortable reading is correct" applies equally to **comfortable over-corrections** — an alarm can be as premature as a victory lap. Both are forms of reasoning past the evidence; both feel earned in the moment; both get overturned by the next probe.

**The discipline (verify before concluding in EITHER direction) is the same fix for both.** §7.14 added the contact-pair-count gate to prevent over-trust; §7.17-aug added the partial-IK-INVALIDATES rule to prevent silent corruption; the calibration here adds **explicit acknowledgement that alarms also need verification before generalisation**. A partial trajectory in one probe is local evidence about that probe; it is NOT evidence that the §7.16 factor table is corrupted; that generalisation step requires a separate verification (which §7.18 supplied, after the fact).

#### (3) THE DISCIPLINE-SURVIVES-CORRECTION articulation

The §7.17-aug rule — *"partial-IK INVALIDATES its factor by precondition, not outcome"* — survives even though THIS partial turned out factor-invariant. The reasoning is precise:

- A rule "exclude partials UNLESS you've checked they don't matter" is **incoherent**: checking whether the missing steps contribute requires the trajectory completion the partial lacks. You'd be running the full trajectory to decide whether to use the partial — at which point you have the full trajectory and don't need the partial.
- A rule "exclude all partials, full stop" is the **correct rule** PRECISELY because zero-contribution is unverifiable a priori. The cost of the rule (some partials that would have been correct anyway get excluded) is cheap (re-run with robust IK); the benefit (no silent corruption) is essential.

**The specific alarm was wrong; the discipline attached to it is right; both are true, no contradiction.** This pattern — *generalisable rule survives a specific instance of the rule turning out unnecessary* — is itself worth recording: rules earn their place by what they prevent across the population of cases, not by what they prevent in this case.

The cleanup script's 5-seed retry + partial-EXCLUDES codifies the rule operationally. Re-use is the proof: the next sub-step LCS comparison inherits the discipline by default.

#### (4) STRATEGIC FRAMING — truth in the middle, better-calibrated than two turns ago

Better-calibrated in BOTH directions than two turns ago:

- **The gap is MORE solid than the §7.17 over-reaction implied** — deep factors clean, crossing real, monotonic at 0.549 mm.
- **The mechanism is LESS settled than the §7.16 over-confidence implied** — still genuinely unknown between compliance-stiffness mismatch (rigid LCS vs compliant Drake softening) and intermittent-contact bouncing (impulse-and-recoil dynamics where Drake's box bounces off the pusher / floor).

The investigation has converged on a **SHARP, well-posed question with a clean instrument to answer it**: at varying penetration depths, does Drake's pusher↔box contact disengage and re-engage multiple times in the 50 ms window, or does it sustain a continuous compliant force? The instrument is the same robust per-depth setup the cleanup uses; the addition is reading Drake's contact-state time series at 1 ms resolution.

This is the kind of question that **either probe-result would be informative**: continuous compliance regains compliance as the mechanism name; intermittent contact rules compliance out and routes to a different mechanism (impulse-and-recoil / discrete event dynamics). No comfortable outcome on either side; the mechanism-name probe is genuinely a discriminator.

Convergence STAYS HELD (still no validated model on push); anitescu STAYS PAUSED (mechanism not named); the arc continues with sharper questions and a calibrated read of where the evidence sits.

#### (5) Progress-table note (for next regeneration)

ADMM-solver row, contact-axis:
- **§7.16 cleanup CROSSING-SURVIVES** — 6/6 clean, crossing revalidated at 0.549 mm; the §7.17 partial-IK confound was over-stated (truncated steps carried zero contact force, factor invariant); BUT the INVALIDATE-by-precondition discipline stands as a rule.
- **§7.17 force-level FORCE-DISCONFIRMS STILL STANDS** (the cleanup doesn't address it).
- **Mechanism-name STILL OPEN** — compliance vs intermittent-contact-bouncing, genuinely unknown.
- **Anitescu STILL PAUSED**.
- **NEXT GATE** = mechanism-name probe (Drake pusher↔box contact-state time series across depths — continuous vs intermittent).
- **Convergence:** HELD.

#### Anti-stale binding (augment)

Any subsequent entry that frames the §7.17 partial-IK alarm as having been "vindicated" by §7.18 is operating on a STALE record — §7.18 shows the §7.17 generalisation was OVER-STATED; the discipline attached to the alarm survives, the specific alarm's prediction (that §7.16 factors were corrupted) does not. Any entry that uses §7.18's rehabilitation of §7.16 to skip a mechanism-name probe is also stale — §7.17-aug's dual question separates crossing-survival from mechanism-name; §7.18 settles only the first. Any reasoning that follows the pattern "the last probe showed Y, so Y is settled" (in EITHER direction) without verification of the generalisation step is the §7.18-aug (2) failure mode: over-reaction without a verification gate.

**Next gate (corrected from §7.18 main body):** mechanism-name probe — at each of the §7.16 depths (clean, full-trajectory), dump Drake's pusher↔box contact state at 1 ms ticks across the 50 ms window and classify each tick as in-contact / out-of-contact; map the contact-time-fraction and the contact-loss-count vs depth. The pre-registered routes (continuous-compliance / intermittent-dominant / mixed) come BEFORE running. NOT actioned in this plan-doc edit.

### 7.19 — Mechanism-name probe (contact-state time series): AMBIGUOUS-leans-DEPTH-DEPENDENT — neither binary route fires; the "disengages" are sub-30-micron grazing at the contact threshold (per the signed-distance co-trace), NOT bouncing; §7.17 force-disconfirms WEAKENS (|F|≈0 mostly on engaged contact); c55ee03's fifth label DEMOTED to interpretation; anitescu STILL PAUSED (2026-06-26)

Artifacts on disk (committed `c55ee03`): script `scripts/_stage_c_contact_state_probe.py`, output `stage_c/contact_state_probe_output.txt`. Method: per §7.16 sweep depth (0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 1.0 mm), set the clean box-pinned state (§7.14 per-depth gate), run Drake `AdvanceTo(0.001 · k)` for k=1..50, classify the pusher↔box pair as PRESENT (in `ContactResults`) / ABSENT each tick; dump signed-distance + |F| co-traces at 5 ms stride. Discriminator is **`ContactResults` presence**, not force value (a force VALUE of zero is ambiguous; presence is not).

#### (1) The result — AMBIGUOUS-leans-DEPTH-DEPENDENT (neither binary route fires)

Per-depth pusher↔box continuity (50 ms window, clean box-pinned, count=4):

| EE_PEN | engaged | engaged % | longest run | disengages | re-engages |
|---|---|---|---|---|---|
| 0.10 mm | 44/51 | 86.3% | 13 | **3** | 4 |
| 0.20 mm | 49/51 | 96.1% | 33 | 1 | 2 |
| 0.30 mm | 48/51 | 94.1% | 25 | 1 | 2 |
| 0.50 mm | 46/51 | 90.2% | 30 | 1 | 2 |
| 0.70 mm | 45/51 | 88.2% | 40 | 2 | 2 |
| 1.00 mm | 50/51 | 98.0% | 50 | **0** | 1 |

Route classification against the §7.18-aug pre-registration:

- **CONTINUOUS-COMPLIANT REJECTED** — only 1 mm hits 50/51; shallow depths have 1–3 gap events. The strict "engaged the full 50 ms at all depths" criterion is NOT met.
- **INTERMITTENT-DOMINATES REJECTED** — max 3 disengages per depth, not flicker-dominant; longest contiguous engaged run GROWS 13 → 50 with depth.
- **DEPTH-DEPENDENT PARTIAL** — 1.0 mm essentially continuous (50/51, 0 disengage) ↔ 0.10 mm most-fragmented (86%, 3 disengage, longest run 13); but middle depths break strict monotonicity (the script reports `monotonic w/ depth: NO, spread 3`).
- **AMBIGUOUS MATCHED** — the script's own verdict (`stage_c/contact_state_probe_output.txt:119`).

Routed outcome: **AMBIGUOUS, leaning DEPTH-DEPENDENT.**

#### (2) The signed-distance co-trace — the key reinterpretation

The disengage ticks correspond to near-grazing POSITIVE separations of **0.000 – 0.028 mm** — sub-30-micron geometric gaps, NOT bounce-out-and-return excursions. The continuous signed-distance co-trace (dumped alongside the binary contact indicator) converts the binary AMBIGUOUS into actionable substructure: the apparent intermittent disengage events at shallow depths are **borderline-GRAZING recoils at the geometric contact threshold**, NOT impulse-and-recoil bouncing. Deep penetration (≥ 1.0 mm) is continuous-compliant. The **0.549 mm §7.16 crossing sits INSIDE the depth band where 1–2 grazing events occur per 50 ms window** (between 0.3 mm and 0.7 mm in the table).

#### (3) What the route rejects

**INTERMITTENT-DOMINATES rejected** ⇒ the "rigid-vs-compliant is the wrong frame; anitescu not indicated" branch does NOT fire on this evidence — the dynamics are not impulse-and-recoil bouncing.

**Pure CONTINUOUS-COMPLIANT rejected** ⇒ the unconditional "compliance regains standing, anitescu re-promoted" path does NOT fire either — the grazing events at shallow depths are still a mechanism component to account for before re-promoting anitescu.

#### (4) §7.17 force-disconfirms — partial rehabilitation

§7.17 force-disconfirms WEAKENS: most of §7.17's |F|≈0 ticks coincide with **ENGAGED contact** — the |F| co-trace shows the pusher↔box pair PRESENT at ticks reading 0.00 N. They are force-MAGNITUDE variation on MAINTAINED contact, NOT contact loss. The §7.17 "Drake is bouncing / soft-spread is a cartoon" inference partly misattributed force-on-continuous-contact as intermittent-contact. §7.17 force-disconfirms does NOT fully collapse (the sub-linear scaling and impulsive LCS shape findings stand), but a substantial chunk of its |F|≈0 evidence is reinterpreted.

#### (5) Fifth-label demotion (the anti-stale binding fires against the toolchain)

The c55ee03 commit introduced a fifth label **"MOSTLY-CONTINUOUS-WITH-GRAZING"** — an accurate DESCRIPTION but **OUT-OF-PRE-REGISTRATION**. Per the §7.18-aug anti-stale binding, it is banked as an **INTERPRETATION of AMBIGUOUS + DEPTH-DEPENDENT**, NOT as a sui-generis route. The verdict-of-record the next block scopes against is the STRICT pre-registration (**AMBIGUOUS-leans-DEPTH-DEPENDENT**), not the interpretive re-label.

This is the anti-stale binding firing against the toolchain's own tidy phrasing — a probe result that didn't cleanly match a registered route got a new clean-sounding label minted post-hoc; the discipline caught it and demoted it. The c55ee03 script and output remain valid artifacts; the LABEL is demoted in the doc, not removed from the commit (no amend).

#### (6) The per-depth gate — earned its keep again

All six depths pass the §7.14 contact-pair gate (`pusher=1, floor=1, arm=0`). The discriminator is uncontaminated; no arm-link↔box phantom pair at any depth — the §7.14 failure mode is closed at every depth in this probe.

#### (7) State at stop

- **Anitescu Part B STILL PAUSED** — AMBIGUOUS does not authorise re-promotion.
- **Mechanism-name PARTIALLY NARROWED** — not flicker-bouncing, not strict continuous-compliance; the residual at the 0.549 mm crossing has a **GRAZING-INSTABILITY component superimposed on a mostly-MAINTAINED contact**, leaning depth-dependent.
- **§7.16 crossing-survives STANDS.**
- **§7.17 force-disconfirms WEAKENS** (|F|≈0-mostly-on-engaged-contact reinterpretation).
- **§7.18 STANDS.**
- Floor [FIXED]; contact-axis [DIAGNOSED-WITH-CONFOUNDS, mechanism-name LEANING **depth-dependent-grazing-on-continuous-base**, still not strictly NAMED].
- Friction DEMOTED; convergence HELD.

#### (8) Next gate

Per the AMBIGUOUS-route pre-registration (finer time-resolution OR continuous penetration-depth time series): the dt=1 ms binary indicator may UNDER-SAMPLE sub-millisecond grazing dynamics — the gaps may be artifacts of 1 kHz polling against a continuous-but-near-zero-distance signal that briefly grazes φ=0.

**Next probe (NOT actioned in this plan-doc edit):** classify on **continuous φ** (φ ≤ 0 in contact vs φ > 0 separated) at every tick, sub-ms resolution, **DROPPING the binary indicator** (the boolean was thresholding a continuous quantity at exactly its noisy value). Weighted on the **0.2 – 0.5 mm LIVE band** (live runs penetrate ~0.2 – 0.5 mm, sitting in the 1–2-grazing-event band near the crossing). Per-depth §7.14 contact-pair gate retained.

#### (9) Progress-table note (for next regeneration)

ADMM-solver row, contact-axis: **mechanism-name probe AMBIGUOUS-leans-DEPTH-DEPENDENT** — neither binary fires; the "disengages" are sub-30-micron grazing at the contact threshold (per the signed-distance co-trace), not bouncing; deep (≥ 1 mm) continuous-compliant, shallow has 1–2 grazing events, the 0.549 mm crossing sits in the grazing band; §7.17 force-disconfirms WEAKENS (|F|≈0 mostly on engaged contact); the commit's fifth label demoted to interpretation; mechanism PARTIALLY NARROWED (grazing-on-continuous-base, not strictly named); anitescu STILL PAUSED; NEXT = continuous-φ classification at sub-ms resolution, weighted on the live band; convergence HELD.

#### Anti-stale binding (§7.19)

Any subsequent entry that cites c55ee03's "MOSTLY-CONTINUOUS-WITH-GRAZING" label as a verdict-of-record (rather than as a description of the AMBIGUOUS-leans-DEPTH-DEPENDENT route's substructure) is operating on a STALE record — that label was demoted in §7.19 (5) per the §7.18-aug anti-stale binding. Any entry that uses §7.19's partial rehabilitation of §7.17 to re-promote anitescu is also stale — §7.19 explicitly does NOT authorise re-promotion (the grazing-on-continuous-base mechanism is partially narrowed, not strictly named, and the AMBIGUOUS route's next gate is a finer-resolution continuous-φ probe, not an anitescu port). Any reasoning that follows the pattern "the binary contact indicator gave verdict X" without reading the signed-distance co-trace and the §7.14 gate result is the §7.19 (2) failure mode: thresholding a continuous quantity at its noisy value and inheriting the threshold's noise as substantive structure.

### 7.21 — Force-level RE-probe under §7.20-pinned sub-ms Drake sampling: FORCE-DISCONFIRMS (strict, binary 3≠4) but SUBSTANTIALLY WALKED BACK from §7.17 — 3-of-4 signatures now confirm (was 1-of-4); Drake-side fully rehabilitated (depth-stable + contact-MAINTAINED ≥95%); only surviving residual is LCS sub-linear normal-force depth-scaling; anitescu STILL PAUSED; convergence HELD (2026-06-26)

Artifacts on disk (committed this block): script `scripts/_stage_c_force_level_reprobe.py`, output `stage_c/force_level_reprobe_output.txt`. Method: per the §7.20-pinned protocol (Drake `dt=0.00025 s`, **4× finer than §7.17's 1 ms**, 5-seed robust IK retry per knot, 3 depths spanning the §7.16 crossing — UNDER `0.10 mm`, SWEET `0.549 mm`, OVER `1.00 mm`), re-read the four pre-registered Part-A signatures from §7.17. Pre-registration was BINARY: all four match → force-level LOCKED → anitescu Part B opens; otherwise FORCE-DISCONFIRMS, anitescu STAYS PAUSED.

#### (1) The re-probe result (force-level read under §7.20-pinned sampling)

**3-of-4 pre-registered signatures now CONFIRM (was 1-of-4 in §7.17).**

| # | Signature | §7.17 verdict | §7.21 verdict | Quant |
|---|---|---|---|---|
| 1 | LCS λ_n impulsive (peak at sub-step 0, decay) | ✓ | ✓ | unchanged — rigid-cartoon shape MATCHES at both depths |
| 2 | LCS λ_n LINEAR scaling with depth (10× depth ⇒ 10× peak) | ✗ | ✗ | residual unchanged — **2.65×** over 10× depth (both runs) |
| 3 | Drake force depth-stable on contact | ✗ (0.76×) | ✓ (**1.22×**) | **OVERTURNED** by §7.20-pinned sampling |
| 4 | Drake contact MAINTAINED (not oscillating) | ✗ (oscillating) | ✓ (**95.0% / 99.5% / 99.5%**) | **OVERTURNED** by §7.20-pinned sampling |

Per-depth detail:

- **UNDER `0.10 mm`** — LCS peak λ_n = **3.47** at sub-step 0, decay to ~0.6 by step 4; Drake on-contact **95.0 %**, peak |F| = **51.5 N** at t = 0.25 ms, sustained ~2–4 N; Δbox_x LCS −1.21 mm vs Drake −2.03 mm → factor **0.594× UNDER**.
- **SWEET `0.549 mm`** *(new depth — §7.17 had a re-pose FAIL here)* — LCS peak λ_n = **6.14**, decays to 0 by step 1, **10/10 valid**; Drake on-contact **99.5 %** (1/201 ticks separated), peak |F| = **58.8 N**; Δbox_x LCS −2.43 mm vs Drake −3.28 mm → factor **0.740× UNDER**.
- **OVER `1.00 mm`** — LCS peak λ_n = **9.20**, but **only 4/10 sub-steps valid** (sub-step 4 IK fails — the §7.17 partial-LCS degradation, NOT cleared by this run's 5-seed retry; §7.18's exact robust-IK formulation was NOT matched here); Drake on-contact **99.5 %**, peak |F| = **62.7 N**; Δbox_x LCS −5.73 mm vs Drake −3.74 mm → factor **1.531× OVER**.

#### (2) The strict verdict (binary gate, 3 ≠ 4)

The pre-registered Part-A gate was BINARY — **all four signatures match → LOCKED; else FORCE-DISCONFIRMS**. 3-of-4 ≠ LOCKED. **Strict verdict: FORCE-DISCONFIRMS. Part B does NOT open. Anitescu STILL PAUSED.**

**Qualification — substantial walkback.** §7.17 force-disconfirms has been substantially walked back: 3 of 4 signatures reversed (2 of the 3 §7.17 disconfirmations rehabilitated by §7.20-pinned sampling; the §7.20 grazing-is-artifact finding propagates directly into the force read — on contact-maintained sampling, Drake's force IS depth-stable). The **only surviving disconfirmation** is the LCS sub-linear depth-scaling (Sig 2, peak λ_n 2.65× over 10× depth).

#### (3) Label demotion — anti-stale binding applied to the re-probe script

The re-probe script invented a fifth label **"FORCE-CONFIRMS-PARTIAL"**. Per the **§7.18-aug anti-stale binding**, this is **DEMOTED to interpretation, NOT promoted as a sui-generis route** — it is the same toolchain-tidy post-hoc relabeling that c55ee03 performed with "MOSTLY-CONTINUOUS-WITH-GRAZING" in §7.19. Verdict-of-record: **FORCE-DISCONFIRMS (strict)**, with the walkback qualification.

#### (4) The inversion — what the residual now is

The §7.17 framing — *"LCS impulsive ✓ but everything else ✗"* — **INVERTS** to **"everything Drake-side ✓, LCS depth-scaling ✗"**. The residual is concentrated entirely on the LCS side:

- Drake's compliant force IS depth-stable on contact (newly confirmed, 1.22×).
- Drake MAINTAINS contact (newly confirmed, ≥95% across all three depths).
- The LCS impulsive shape matches the rigid cartoon (confirmed at both runs).
- **BUT** the LCS peak λ_n grows only **2.65×** over **10×** depth, not the 10× the rigid cartoon predicts.

The open question NARROWS from *"is it compliance?"* (Drake side confirms YES, in the sense that Drake's behaviour on contact is depth-stable and contact-maintained) to **"why does the LCS rigid normal force scale SUB-LINEARLY in depth?"**

#### (5) The 1 mm partial-LCS confound (precondition for trusting the 2.65×)

The `1.00 mm` LCS run completed only **4/10 sub-steps** — sub-step 4 IK fails, reprising the §7.17 partial-LCS degradation. §7.18's robust-IK recovered this on the displacement sweep, but this force-level retry **did not match §7.18's exact IK formulation**. Consequences:

- The peak-λ_n-at-sub-step-0 reading is unaffected (peak occurs before the IK failure).
- BUT Σλ_n is artificially low (sub-steps 4-9 missing).
- The **2.65× scaling ratio uses the deep-penetration peak**, so the scaling number itself may be partial-contaminated by the failed sub-steps' invisible contribution to Σλ_n.

This is the **§7.14 lesson one level deeper**: a partial run's number cannot be trusted just because the visible part of it looks clean. **Full 1 mm-trajectory recovery (via §7.18's exact robust-IK formulation re-applied to the force-level re-probe) is the precondition before the 2.65× is trusted.**

#### (6) The dt-dependent crossing confound (new, NOT actioned)

The finer Drake `dt=0.00025 s` shifted the `1.00 mm` box motion: Δbox_x was −2.24 mm at `dt=1 ms` (§7.17) and is **−3.74 mm at `dt=0.25 ms`** (§7.21). The **§7.16 displacement-crossing depth (`0.549 mm`) may itself be dt-DEPENDENT** — a new confound, not acted on yet.

**Implication for anitescu validation:** if the crossing depth moves with Drake's timestep, then *"the gap closes ACROSS depths"* is the right validation target and *"the gap closes at the sweet spot"* was always the wrong one — reinforcing the §7.16-aug guardrail. **The crossing was never the thing to match; the depth-STABILITY is.** This confound has NOT been actioned in this block; it is recorded for the next probe.

#### (7) Convergence stays HELD

- floor **[FIXED]**
- contact-axis **[DIAGNOSED — Drake-side compliant CONFIRMED, LCS-side rigid-residual OPEN]**
- friction **DEMOTED**

The model is **not locked** (3-of-4, not 4-of-4), so **convergence stays HELD**. But the open question is now a **single LCS-internal question**: why sub-linear normal-force depth-scaling?

#### (8) Strategic framing — the guardrail paying off TWICE

The §7.17 force-disconfirms could have been:

- declared **final** (abandoning a correct compliance diagnosis on artifact-driven evidence), OR
- **papered over** (declared locked at 3-of-4 with a freshly-minted label).

**Neither happened.** The artifacts were cleaned (§7.18–§7.20), the re-probe re-ran, the strict verdict held, AND the residual narrowed to one precise LCS-side question. The guardrail (confirm-at-force-level-before-reformulating) paid off **TWICE** — it stopped a premature LOCK in §7.17 AND stopped a premature final-DISCONFIRMS once the artifacts cleared.

#### (9) Progress-table note (for next regeneration)

ADMM-solver row, **HORIZONTAL/push axis**: force-level RE-probe under sub-ms Drake sampling = **3-of-4 signatures** (was 1-of-4 in §7.17); §7.17 force-disconfirms SUBSTANTIALLY walked back (Drake-side now confirmed depth-stable + contact-maintained; the 2 §7.17 disconfirmations were sampling artifacts); strict verdict **FORCE-DISCONFIRMS** (binary gate, 3 ≠ 4), anitescu STILL PAUSED; the ONLY surviving residual is **LCS sub-linear depth-scaling** (peak λ_n 2.65× over 10× depth); **1 mm partial-run confound** + **dt-dependent-crossing confound** flagged; NEXT = clear the 1 mm confound (re-apply §7.18's exact robust-IK formulation to the force-level retry) + check λ_t-coupling; convergence HELD.

#### Anti-stale binding (§7.21)

Any subsequent entry that cites the re-probe script's invented label "FORCE-CONFIRMS-PARTIAL" as a verdict-of-record (rather than as DEMOTED-to-interpretation per §7.21 (3)) is operating on a STALE record — the binary pre-registration gate is the verdict-of-record, and 3-of-4 ≠ 4-of-4. Any entry that uses §7.21's substantial walkback of §7.17 to re-promote anitescu Part B is also stale — §7.21 explicitly does NOT authorise re-promotion (3-of-4 ≠ LOCKED; the residual is a precise LCS-internal question, not a green light). Any reasoning that follows the pattern "the 2.65× is trustworthy because the visible sub-steps were clean" without re-applying §7.18's exact robust-IK formulation to the 1 mm force-level run is the §7.21 (5) failure mode: trusting a partial-LCS scaling ratio just because its peak occurs in the clean prefix. Any reasoning that takes the §7.16 0.549 mm crossing as a fixed depth-anchor without acknowledging the §7.21 (6) dt-dependent-crossing confound is the §7.16-aug failure mode revived: chasing a sweet-spot match when **depth-STABILITY across depths** is the right validation target.

### 7.22 — LCS residual probe (Part A 1mm-confound + Part B λ_t-coupling): IS-DYNAMICS — 1mm confound CLEARED under §7.18's exact warm-aware IK (the §7.21 fixed-seed retry was the formulation gap, not a geometric limit); peak-λ_n 2.65× HOLDS on clean full-trajectory data; λ_t-coupling cleanly DISCONFIRMED (non-monotonic, sign-flipped — the friction-cone hypothesis is not the home of the residual); two of four candidate homes ELIMINATED (partial-run + friction-cone); residual localized to the A-matrix dynamics-propagation channel; anitescu STAYS PAUSED; convergence HELD (2026-06-26)

Artifacts on disk (committed `b414aff`): script `scripts/_stage_c_lcs_residual_probe.py`, output `stage_c/lcs_residual_probe_output.txt`. Method: replace the §7.21 re-probe's fixed-seed IK retry with §7.18 sweep-cleanup's EXACT warm-aware perturbation recipe (`[q_arm_warm, POSTURE_NOMINAL, q_arm_warm + ±0.1 rad rand1, +rand2, +rand3]`, `rng = np.random.default_rng(seed=0)`). LCS-only (Drake forward not required for either question). Extract per sub-step at each depth `{0.10, 0.549, 1.00 mm}`: `λ_n`, `Σλ_t`, plus the impulse-channel decomposition `D[box_v_x, :] @ λ` split into normal (`D · λ_n`) and tangent (`D · λ_t`) contributions. Pre-registered routes (the next block scopes against these; this probe does not execute the next block): RESOLVES-INTO-COMPLIANCE / IS-DYNAMICS / PERSISTS-UNEXPLAINED / 1mm-STILL-FAILS.

#### (1) The 1 mm confound — CLEARED

Under §7.18's EXACT IK formulation (warm + posture + 3 random ±0.1 rad/joint perturbations of warm, `rng seed=0` — the recipe the §7.21 re-probe did **NOT** match), **1.00 mm recovers full 10/10 sub-steps CLEAN**. Every sub-step's IK succeeded on **seed 0** (the warm-start; no perturbation was even needed once the IK was warm-aware) — so the §7.21 partial-run failure at sub-step 4 was a **FORMULATION gap (not warm-aware)**, NOT a hard geometric limit.

Peak-λ_n depth-scaling on clean data: `3.470 → 6.137 → 9.200` at `0.10 / 0.549 / 1.00 mm`. The `0.10 → 1.00 mm` ratio is **2.65× — STAYS** (identical to §7.21 to 0.1%).

**Why the §7.21 partial reading was unaffected:** the peak occurred at sub-step 0, and sub-steps 4-9 turned out to have **λ_n = 0** (contact SEPARATED after the impulse), so `Σλ_n at 1 mm = peak λ_n = 9.200` either way. The §7.21 (5) precondition was correct (a partial run's number cannot be trusted just because the visible prefix looks clean), but the specific worry it raised (Σλ_n artificially low → 2.65× contaminated) turned out invariant under truncation. **CONFOUND CLEARED — the residual is REAL on clean full-trajectory data.**

#### (2) λ_t-coupling — DISCONFIRMED

`λ_t` at sub-step 0 goes **NON-MONOTONIC** with depth (`1.388 → 0.587 → 0.806` at `0.10 / 0.549 / 1.00 mm` — magnitude does **NOT** grow). Tangent-channel `D · λ_t → box_v_x` scales **0.77× over 10× depth** (essentially FLAT). Total channel `D · λ → box_v_x` scales **3.87× over 10× depth** (sub-linear, **61.3% deviation from linear**).

The tangent contribution **OPPOSES** the normal (sign-flipped); the ratio `t/n` DECREASES with depth (`−0.393 → −0.159 → −0.114`).

**λt-DOES-NOT-ACCOUNT.** The hypothesis ("at deeper penetration, more impulse routes into the friction-cone tangent channel") is CLEANLY disconfirmed — sign-flipped, not merely weak. The friction-cone coupling is not the home of the residual.

#### (3) Route IS-DYNAMICS

Sig 2 (LCS `λ_n` LINEAR normal scaling) stays **DISCONFIRMED** on clean full-trajectory data; the sub-linear residual lives in the **A-matrix dynamics-PROPAGATION channel (`A · x`)**, NOT the `D · λ` contact-channel split. Two of the four candidate homes for the residual are now ELIMINATED:

| candidate home | status | source |
|---|---|---|
| partial-run artifact (§7.21 (5) confound) | **CLEARED** | §7.22 (1) |
| friction-cone coupling (λ_t channel) | **DISCONFIRMED** | §7.22 (2) |
| A-matrix dynamics propagation | **OPEN** | next block |
| beyond-pure-normal-compliance (mechanism reopens) | **OPEN** | conditional on A-matrix |

The residual is localized to the dynamics-propagation channel.

#### (4) Honest-flag honored — no relabeling minted

Per the pre-registered CRITICAL HONESTY FLAG (§7.21 (3) anti-stale binding extended to this probe): even if λ_t-coupling **had** succeeded, it would have been recorded as a **REFINEMENT** of the cartoon (*"linear TOTAL impulse with a normal/tangential split"*), NOT a 4th confirm. Since λ_t-coupling did **NOT** succeed, that distinction is **moot** — Sig 2 stays disconfirmed in **FACT**, not just literally. **No relabeling minted.** The same demotion that bound `"FORCE-CONFIRMS-PARTIAL"` (§7.21 (3)) and `"MOSTLY-CONTINUOUS-WITH-GRAZING"` (§7.19 (5)) is preserved here without needing to be applied.

#### (5) Convergence stays HELD

- floor **[FIXED]**
- contact-axis **[DIAGNOSED — Drake-side compliant CONFIRMED; LCS-side rigid-residual now LOCALIZED to the A-matrix propagation channel, OPEN]**
- friction **DEMOTED**

The model is **not locked**. **Convergence stays HELD.** **Anitescu STAYS PAUSED** — IS-DYNAMICS does not authorise re-promotion (the residual is now precisely localized, but localization is not naming, and the A-matrix candidate may itself rule the residual *out of* pure normal compliance once measured).

#### (6) Progress-table note (for next regeneration)

ADMM-solver row, **HORIZONTAL/push axis**: the LCS sub-linear depth-scaling residual is REAL on clean data (1 mm confound CLEARED via §7.18's exact warm-aware IK; **2.65× holds**); **λ_t-coupling DISCONFIRMED** (non-monotonic, sign-flipped); route **IS-DYNAMICS** (residual in the A-matrix propagation channel, not the contact-channel split); two of four candidate homes eliminated; NEXT = LCS-vs-Drake box-velocity propagation across depths (A-matrix contribution to `box_v_x` at the linearization point); anitescu PAUSED; convergence HELD.

#### Anti-stale binding (§7.22)

Any subsequent entry that cites the §7.21 1 mm partial-run worry as still-open is operating on a STALE record — §7.22 (1) cleared it on clean full-trajectory data under §7.18's exact warm-aware IK, AND showed the original Σλ_n concern was truncation-invariant (sub-steps 4-9 carried λ_n = 0). Any entry that re-opens λ_t-coupling as a candidate for the residual without addressing the §7.22 (2) non-monotonic + sign-flipped reads is also stale — λ_t was cleanly disconfirmed (the channel runs opposite the normal, not parallel; the ratio shrinks with depth, not grows). Any entry that treats IS-DYNAMICS as authorising anitescu re-promotion is the same staleness mode as §7.21's "3-of-4 ≠ 4-of-4": localization is not naming, and the A-matrix candidate has not yet been measured — once it is, the residual may resolve into compliance OR reopen beyond it. Any reasoning that follows the pattern "λ_t was the candidate and it didn't work, so compliance is dead" is also stale — IS-DYNAMICS leaves compliance *and* a different-mechanism explanation BOTH live; only the A-matrix probe in the next block settles which.

**Correction-of-record (added in §7.23 (2)):** §7.22 (3) claimed the residual lives in "the A-matrix dynamics-propagation channel (`A · x`), NOT the `D · λ` contact-channel split." That attribution is **WRONG** and is corrected by §7.23 — the A·x contribution to box_v_x is ZERO at sub-step 0 (box at rest; the affine/gravity term is vertical), so ALL of the LCS box_v_x at sub-step 0 is the `D · λ` channel. §7.22 internally already measured `D · λ → box_v_x` at 3.87×, which locates the residual in `D · λ`; the "A·x not D·λ" wording in §7.22 (3) was self-contradictory with §7.22's own numbers. The §7.22 λ_t-flat reading STANDS (Sig 2 still disconfirmed; tangent channel does not absorb the impulse). The corrected localization is: within `D · λ`, the residual is in the NORMAL part (λ_n scales 2.65×, D-column normal-mapping amplifies that to 3.87× in velocity — a normal-channel D-column depth-dependence). The "A-matrix" framing was the wrong channel name for a box-at-rest horizontal-velocity comparison.

### 7.23 — A-matrix probe (LCS-vs-Drake box-velocity propagation across depths, Drake dt=0.25 ms FIXED): PROPAGATION-DIVERGES at depth (LCS 3.87× vs Drake 1.20× over 10× depth), BUT the route is COMPLIANCE-CONFIRMED — not mechanism-reopens-beyond-compliance. A·x = 0 (box at rest) re-localizes the residual to the D·λ NORMAL-impulse-to-velocity map and CORRECTS §7.22 (3)'s "A·x not D·λ" attribution. Drake's depth-stable box-velocity is the velocity-side SHADOW of §7.21's depth-stable on-contact force — same compliance signature, third observable. Normal compliance is now TRIPLE-confirmed (displacement §7.16 / force §7.21 / velocity §7.23). Anitescu DEMOTED from presumptive-fix to one CANDIDATE (changes the cone, not rigid-vs-compliant). Diagnosis phase effectively COMPLETE; entering FIX phase. Anitescu PAUSED; convergence HELD (2026-06-26)

Artifacts on disk (committed `29bf054`): script `scripts/_stage_c_a_matrix_probe.py`, output `stage_c/a_matrix_probe_output.txt`. Method: clean box-pinned state at each depth (§7.18 5-seed re-pose, §7.14 contact-pair gate per-depth — `pusher=1, floor=4` from the `LCS_EXPLICIT_BOX_GND=4` synthesized `BOX-VERT-{0..3}` pairs, `arm=0`); LCS side runs one sub-step (Δt = 5 ms) via `linearize_discrete_ee_space` + LCP, reports `box_v_x` channel-decomposed into `A·x` + `D·λ` + `d` contributions; Drake side runs from the SAME configuration with `dt = 0.25 ms FIXED` (§7.20-pinned, held fixed across ALL depths to avoid the §7.21 (6) dt-dependent-crossing confound), AdvanceTo(5 ms), reports actual `box_v_x`. Pre-registered routes: PROPAGATION-MATCHES (cartoon was wrong about propagation, compliance survives, anitescu re-promotes) / PROPAGATION-DIVERGES (dynamics-matrix gap, mechanism reopens) / A-MATRIX-INCONCLUSIVE.

#### (1) The DIVERGES result — LCS vs Drake box_v_x at t = 5 ms

| depth (mm) | LCS box_v_x | Drake box_v_x | LCS/Drake |
|---|---|---|---|
| 0.100 | −0.052685 | −0.053988 | **0.976 ≈ MATCH** |
| 0.549 | −0.129102 | −0.064133 | **2.013** (LCS 2× too fast) |
| 1.000 | −0.203710 | −0.064927 | **3.138** (LCS 3× too fast) |

**LCS box_v_x scales 3.87× over 10× depth; Drake scales 1.20×** (essentially FLAT). Drake's actual box-velocity is **depth-invariant once contact engages**. Per-depth gate CLEAN at all three (`pusher = 1`, `floor = 4` from `LCS_EXPLICIT_BOX_GND=4`, `arm = 0`; Drake `dt = 0.00025 s` fixed).

#### (2) A·x = ZERO re-localization — corrects §7.22 (3)'s attribution

**The `A · x` contribution to box_v_x is ZERO at sub-step 0** (box starts at rest; the affine / gravity term is vertical), so ALL of the LCS box_v_x at sub-step 0 is the `D · λ` channel.

**This CORRECTS §7.22 (3).** §7.22 said *"residual in the A-matrix dynamics-propagation channel (A·x), NOT the D·λ contact-channel split."* That is **WRONG** and is corrected here. The residual is **entirely in the D·λ normal-impulse-to-velocity map**. (§7.22 internally already reported *"D·λ → box_v_x scales 3.87×"*, which locates it in `D · λ`; the "A·x not D·λ" conclusion was self-contradictory with §7.22's own numbers.)

The §7.22 λ_t-flat finding STANDS (Sig 2 disconfirmed; tangent channel does not absorb the impulse) but does NOT move the residual out of `D · λ`. Within `D · λ`, the residual is in the **NORMAL** part: `λ_n` scales 2.65×, and the D-matrix normal-column mapping amplifies that to 3.87× in velocity = **a D-column depth-dependence in the normal-impulse channel**. The "A-matrix" framing was the wrong channel name for a box-at-rest horizontal-velocity comparison; the residual is in the **rigid-normal-impulse-to-velocity mapping**.

#### (3) Route: COMPLIANCE-CONFIRMED — NOT mechanism-reopens-beyond-compliance

The mechanism is Drake's compliant normal contact **SATURATING** (once engaged, force AND velocity become depth-invariant) while the LCS rigid normal impulse keeps **SCALING** with penetration. This is the **THIRD triangulated confirmation** of the SAME signature:

| § | observable | Drake | LCS | Drake/depth scaling |
|---|---|---|---|---|
| §7.16 | displacement (Δbox_x) | depth-stable bench | rigid scaling | sweet at ~0.549 mm |
| §7.21 | on-contact force \|F\| | 1.22× (depth-stable) | λ_n 2.65× | ≈ flat |
| **§7.23** | **box_v_x at 5 ms** | **1.20× (depth-stable)** | **3.87×** | **≈ flat** |

The §7.23 Drake-side result is the **velocity-side SHADOW of §7.21's force-side** (depth-stable on-contact force ⇒ depth-stable box-velocity in finite time, necessarily — the same physical fact from a different angle).

**Correction to the probe report's auto-routing:** the probe report said *"mechanism REOPENS beyond pure normal compliance with a new specific target"* — this **conflates** two statements: *"anitescu may not be the right fix"* (TRUE — see §7.23 (4)) with *"the mechanism is beyond compliance"* (FALSE — contradicted by the report's own description *"Drake's compliance spreads the impulse over time"*). This is NOT the mechanism reopening beyond compliance — it is the §7.16 normal-compliance diagnosis getting its **cleanest, most direct confirmation**, now at the velocity level.

#### (4) Anitescu DEMOTED to candidate — re-attaching the probe's correct insight

Anitescu changes the friction-**CONE** formulation (polyhedral pyramid → smooth/SOCP); it does NOT obviously introduce a depth-stable normal-impulse / finite-stiffness behaviour. So anitescu is **DEMOTED** from presumptive-fix (the *"PROPAGATION-MATCHES → anitescu re-promotes"* expectation carried through §§7.15-aug, 7.16-aug, 7.17, 7.21, 7.22) to **ONE candidate to evaluate against the normal-compliance target**, NOT the presumptive fix.

The fix is **RE-TARGETED** to a **normal-compliance representation** — concrete candidates:
- compliant Stewart-Trinkle (φ regularized with a stiffness term so the impulse saturates with depth),
- smoothed / penalty-regularized contact in the D-column derivation,
- a stiffness / compliance-time-constant term that bends the normal-impulse-to-velocity map in the deep regime.

Anitescu remains on the candidate list only if its velocity-level convex formulation can be shown to introduce a depth-stable normal-impulse signature (not its primary advertised effect, so the scoping must establish this rather than assume it).

#### (5) The shallow anchor

LCS and Drake **AGREE at 0.10 mm to 2.4%** and only diverge at depth. This is the unambiguous shallow-limit anchor: **rigid is the SHALLOW LIMIT of compliant**. The LCS rigid-impulse map is a correct local linearization until Drake's compliance saturates.

The fix is therefore a **depth-ONSET compliance term** (leaves the shallow regime alone, bends the deep regime DOWN to Drake's flat box_v_x ≈ −0.064 m/s), **NOT an LCS rebuild**. This is the smallest-surface-area fix consistent with the data, and it preserves §7.14's contact-pair-gate, §7.18's robust IK, §7.20's sub-ms sampling discipline, and §7.22's tangent-channel reads unchanged.

#### (6) Diagnosis effectively COMPLETE — entering FIX phase

The **no-push diagnosis phase is effectively COMPLETE**:

- **Mechanism NAMED** — normal compliance.
- **TRIPLE-confirmed** — displacement (§7.16) + force (§7.21) + velocity (§7.23).
- **Residual PRECISELY localized** — the `D · λ` normal-impulse-to-velocity map (the rigid-impulse assumption breaking down when Drake's compliance saturates beyond ~0.5 mm depth).
- **Shallow-anchored** — 0.10 mm match to 2.4% gives a clean "what the LCS does right" baseline; the fix only needs to bend the deep regime.

**Entering the FIX phase.**

| axis | state |
|---|---|
| floor | **[FIXED]** |
| contact-axis | **[DIAGNOSED — normal compliance confirmed across 3 observables; residual = rigid-impulse-vs-compliant-saturation in the D·λ normal channel; FIX open]** |
| friction | **DEMOTED** |
| anitescu | **PAUSED (demoted to candidate, not presumptive fix)** |
| convergence | **HELD** (model still not fixed) |

#### (7) Progress-table note (for next regeneration)

ADMM-solver row, **HORIZONTAL/push axis**: A-matrix LCS-vs-Drake propagation probe = **PROPAGATION-DIVERGES** (Drake box_v_x depth-FLAT 1.20×, LCS 3.87×); **A·x = 0** (box at rest) re-localizes the residual to the **D·λ normal-impulse-to-velocity** map (CORRECTS §7.22's A·x attribution); mechanism = **normal compliance, now TRIPLE-confirmed** (displacement §7.16 / force §7.21 / velocity §7.23) — Drake's compliant normal contact saturates depth-flat, LCS rigid normal impulse keeps scaling; shallow agreement (0.10 mm 2.4%) → rigid is the shallow limit, fix = **depth-onset compliance term**; anitescu **DEMOTED to candidate** (changes the cone, not rigid-vs-compliant); diagnosis effectively **COMPLETE**, entering FIX phase; convergence HELD.

#### Anti-stale binding (§7.23)

Any subsequent entry that cites the probe report's auto-routing *"mechanism REOPENS beyond pure normal compliance"* as a verdict-of-record is operating on a STALE record — §7.23 (3) explicitly corrected that auto-routing as a conflation of *"anitescu may not be the right fix"* (TRUE) with *"the mechanism is beyond compliance"* (FALSE). The verdict-of-record is **COMPLIANCE-CONFIRMED, triple-triangulated**, with anitescu demoted to candidate; not mechanism-reopens. Any entry that cites §7.22 (3)'s *"residual in A·x, not D·λ"* without acknowledging the §7.23 (2) correction is also stale — A·x = 0 in the box-at-rest comparison, so the residual was always in `D · λ`. Any entry that treats anitescu as the presumptive fix is the same staleness mode as §7.21's "3-of-4 ≠ 4-of-4" extended to fix-selection: cone-formulation is not stiffness-mechanism, and the §7.23 (4) re-targeting binds the fix scope to normal-compliance representations. Any entry that uses the §7.16-aug *"gap closes across depths"* validation target without acknowledging the §7.23 (5) shallow-anchor constraint is also stale — the shallow agreement is part of the constraint: a correct fix must NOT degrade the 0.10 mm match while bending the deep regime down to Drake's ~−0.064 m/s flat.

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
