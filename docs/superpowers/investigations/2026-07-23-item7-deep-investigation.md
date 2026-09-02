# Item #7 — Full reference `q_vector` Q construction: deep investigation

**Date:** 2026-07-23 (arc af44d06)
**Author:** deep-investigation session, Hyungjun Doh (hdoh@ucsd.edu)
**Trigger:** port-todo #7 marked STRUCTURALLY BLOCKED after this arc's audit; user requested deep investigation of the blocker rather than another naive migration attempt.
**Scope:** research + report ONLY. No implementation, no re-runs of failed configurations.

---

## Executive summary

Item #7 is genuinely blocked by a **three-way coupled configuration mismatch**, not a single knob. Every prior migration attempt (p39, p40, p41, p44, p46 for the q_vector path; p68, p69 for the G-matrix path) flipped one dimension while leaving the other two at port defaults — placing the ADMM in a half-migrated regime that has no reason to converge, and typically doesn't.

The three coupled dimensions are:

| Dimension | Port default | Reference | File:line |
|---|---|---|---|
| ADMM starting ρ | `100.0` | `~1.0` (never explicit in yaml; implicit through `G = w_G · diag(g_vector)` scale) | `main.py:584` |
| ADMM augmentation matrix | `rho · I` (uniform on all slots) | `w_G · diag(g_vector)` — `g_x=0, g_u=0, g_λ=2, g_η=1` (per-slot) | `control/admm_solver.py:1057-1076` (impl) + `c3/core/c3.cc:389-390` (ref) |
| Q matrix values | Sparse: `w_obj_xy=12500, w_box_z=12490` only, no velocity slots | Dense diagonal: `Q = w_Q · diag(q_vector)` — `[0.01×3 ee_pos, 0.1×4 quat, 200 200 120 obj_pos, 5×3 ee_vel, 0.05×3 obj_ω, 0.05×3 obj_v]` | `control/task_costs.py:783-826` |

**Key insight:** the port's `rho=100 · I` uniform augmentation puts every slot (state + input + λ + η) at ~100× reference iter-0 augmentation. To compensate, the port has been TUNED with 25× larger `w_obj_xy` (12500 vs 10000) and ZERO velocity weights. Enabling `use_reference_q_vector=True` alone reduces Q while keeping the over-augmentation, un-balancing the QP. Enabling `_use_g_matrix=True` alone drops state augmentation to 0 (`g_x=0`) while keeping tuned Q (12500), un-balancing the other direction. Neither works in isolation because both are compensations for the ρ=100 choice.

**Minimum unblock recipe (not attempted):** three coordinated flips — `rho=1.0` + `_use_g_matrix=True` + `use_reference_q_vector=True` — with `admm_iter=3` (already ref) and `rho_scale=3` (already ref). This puts the port's C3+ ADMM in the actual reference operating regime for the first time.

---

## 1. Reference architecture

**Reference C3+ (`push_t/parameters/sampling_c3plus_options.yaml`):**

```yaml
admm_iter: 3
rho: 0            # comment: "This isn't used anywhere!"
rho_scale: 3

w_Q: 50           # Q = w_Q · diag(q_vector)
w_G: 0.01         # G_iter0 = w_G · diag(g_vector)
w_R: 1
w_U: 0.26

# q_vector — 19-slot dense diagonal:
q_vector: [0.01, 0.01, 0.01,    # EE position  (slots 0-2)
           0.1, 0.1, 0.1, 0.1,  # object orientation (slots 3-6)
           200, 200, 120,       # object position (slots 7-9)  ← pose regime
           5, 5, 5,             # EE linear velocity (slots 10-12)
           0.05, 0.05, 0.05,    # object angular velocity (slots 13-15)
           0.05, 0.05, 0.05]    # object linear velocity (slots 16-18)

# g_vector — per-slot G:
g_x: [0]*19                     # NO state augmentation
g_u: [0]*3                      # NO input augmentation
g_lambda_list: [[2]*16, [2]*20] # per-λ augmentation = 2
g_eta_list:    [[1]*16, [1]*20] # per-η augmentation = 1
```

**Reference ADMM loop (`c3/core/c3.cc:322-392`):**

```cpp
G = cost_matrices_.G   // starts as w_G · diag(g_vector), i.e. very small

for (int iter = 0; iter < admm_iter; iter++) {
  ADMMStep(x0, &delta, &w, &G, iter);
    // SolveQP  with augmentation G
    // SolveProjection with G
    // w = w + z - delta
    // w = w / rho_scale             ← scale down ω
    // G = G * rho_scale              ← scale up G
}
```

**Effective augmentation values across 3 iters (reference C3+ push_t):**

| Iter | G on λ | G on η | G on state | G on input |
|---|---|---|---|---|
| 0 | 0.01·2 = 0.02 | 0.01·1 = 0.01 | 0 | 0 |
| 1 | 0.06 | 0.03 | 0 | 0 |
| 2 | 0.18 | 0.09 | 0 | 0 |

**Reference Q matrix values (obj slots dominate):**

| Slot | Weight |
|---|---|
| ee_pos (0-2) | 0.5 |
| obj_quat (3-6) | 5.0 |
| obj_x, obj_y (7, 8) | 10000 (pose) or 12500 (position) |
| obj_z (9) | 6000 (pose) or 12500 (position) |
| ee_vel (10-12) | 250 |
| obj_ω (13-15) | 2.5 |
| obj_v_lin (16-18) | 2.5 |

**Reference state layout (from `sampling_based_c3_controller.cc:742, 768, 773`):**

```
[ee_pos(3), obj_quat(4), obj_pos(3), ee_vel(3), obj_ω(3), obj_v_lin(3)]  → n_x = 19
```

The reference q_vector is written IN this order — first three slots are ee_pos, slot 3-6 is quat, etc.

---

## 2. Port architecture

**Port `SamplingC3Params` defaults + push_t task overrides:**

```yaml
# ADMM (main.py:584, admm_solver.py defaults)
rho: 100.0                     # ← Port's tuned constant, NOT reference-derived
rho_scale: 3.0                 # (matches reference)
admm_iter: 3                   # (matches reference from 4c3bad5)
_use_g_matrix: False           # env-gated REFCONF_USE_G_MATRIX; default OFF

# Q values (config/tasks.yaml push_t)
w_obj_xy: 12500                # sparse — port drives Q via individual weights
w_obj_z:  10
w_box_z:  12490                # so Q[obj_z] = 10 + 12490 = 12500
w_box_rp: 5.0                  # weight on quat qx, qy (small)
w_yaw:    0.0                  # yaw handled by near-goal Hessian override
use_reference_q_vector: false  # ← default OFF; the dense-diag path is a no-op
```

**Port ADMM augmentation (default, `_use_g_matrix=False`):**

```python
_P_aug = rho * _eye_total    # admm_solver.py:1118
```

Uniform ρ on every slot. Same weight on state, input, λ, η.

**Effective augmentation values across 3 iters (port default):**

| Iter | Aug on λ | Aug on η | Aug on state | Aug on input |
|---|---|---|---|---|
| 0 | 100 | 100 | 100 | 100 |
| 1 | 300 | 300 | 300 | 300 |
| 2 | 900 | 900 | 900 | 900 |

**Port state layout (`control/task_costs.py:743-755`):**

```
[obj_quat(4), obj_pos(3), ee_pos(3), obj_ω(3), obj_v_lin(3), ee_vel(3)]  → n_x = 19
```

**Different order from reference.** Port has obj_quat first (Drake floating-body convention: `[qw, qx, qy, qz, x, y, z, ...]`); reference has ee_pos first.

The port CORRECTLY remaps `use_reference_q_vector` values into port's layout at `task_costs.py:791-802`:

```python
q_diag[_NEW_OBJ_QW:_NEW_OBJ_QZ+1] = q_vec_obj_quat  # port slot 0:4 = ref slot 3:7
q_diag[_NEW_OBJ_X]  = q_vec_obj_pos[0]              # port slot 4  = ref slot 7
q_diag[_NEW_PEE_SLOT]   = q_vec_ee_pos              # port slot 7:10 = ref slot 0:3
q_diag[_NEW_VBOX_OMEGA] = q_vec_obj_ang_vel         # port slot 10:13 = ref slot 13:16
q_diag[_NEW_VBOX_LIN_X] = q_vec_obj_lin_vel[0]      # port slot 13    = ref slot 16
q_diag[_NEW_VEE_SLOT]   = q_vec_ee_vel              # port slot 16:19 = ref slot 10:13
```

State layout is NOT the actual blocker — the Q construction remap is bijective + the LCS matrices are built in port's layout consistently. The blocker is purely the ρ / G / Q tuning mismatch.

---

## 3. Exact mechanism of the coupling failure

**ADMM primal step (both port and reference solve the same QP structure):**

$$
\min_{z} \tfrac{1}{2} z^T P z + q^T z \quad \text{s.t. constraints}
$$

where $P = 2 \cdot \text{blockdiag}(Q, R, ..., Q, R) + \rho \cdot G_{\text{aug}}$ and $q = q_{\text{ref}} - \rho \cdot G_{\text{aug}} \cdot (\delta - \omega)$.

**Port default:** $G_{\text{aug}} = I$, $\rho = 100$, so per-slot augmentation is 100 (state), 300 (state at iter 1), etc. Q weights $w_{\text{obj\_xy}}=12500$ are ~125× larger than augmentation — cost dominates, ADMM converges toward the cost minimum with weak constraint enforcement.

**Reference:** $G_{\text{aug}} = \text{diag}([0, 0, ..., 2, 2, ..., 0, 0, ..., 1, 1, ...])$, $\rho \approx 1$ (implicit), so per-slot augmentation is 0.02 (λ), 0 (state). Q weights $w_Q \cdot 200 = 10000$ on obj_pos are 500,000× larger than λ augmentation, ∞× larger than state augmentation. Cost dominates hard on state; λ/η get their own ADMM sub-optimization.

**What breaks when a single knob flips:**

**(a) Enable `use_reference_q_vector=True` alone (p39, p40, p41, p44):**
- Q shifts from sparse-huge (12500) to dense-moderate (10000 obj + 250 ee_vel + tiny quat).
- Augmentation stays at $\rho \cdot I = 100 \cdot I$ (uniform).
- Now Q on ee_vel (250) is 2.5× smaller than augmentation on ee_vel (100 at iter 0, 900 at iter 2).
- ADMM is dominated by the augmentation term for ee_vel → drives ee_vel toward δ (LCS predicted value) instead of toward zero (Q reference).
- The QP no longer weights EE motion by q_ee_vel — augmentation drowns it out.
- Empirical result: p44 dual residual explodes to ~1000 within 60s (`results/tight_goal_p44_qvec_full_migration_240s.txt`); p46 with admm_iter=25 goes to ~1.5M.

**(b) Enable `_use_g_matrix=True` alone (p68, p69):**
- Augmentation drops to reference values on λ/η but ZERO on state (since $g_x=0$).
- Q keeps port's tuned sparse-huge (12500 obj_xy, 0 on ee_vel).
- State ADMM step: primal residual is just `Ax_i + Bu_i + Dλ_i + d - x_{i+1}` with no augmentation to smooth the update. The QP now sees state as unconstrained by ADMM — pure Q-driven — but Q has zero weight on ee_pos and velocity slots.
- The primal x is under-constrained; velocity slots drift.
- Empirical result: p68 rot_err=0.047 (BEST rot seen), trans_err=0.058 — decent but the p69 latching artifact shows the T physically tipped over (rot_err=3.14 rad = 180°) as the planner over-corrected without state-side augmentation.

**(c) Enable `_use_g_matrix=True` + `use_reference_q_vector=True` (never attempted):**
- Would reduce Q obj_xy from 12500 → 10000 AND zero out state augmentation AND drop λ augmentation from 100/300/900 → 0.02/0.06/0.18.
- State is now purely Q-driven with reference weights.
- Should be closer to reference. But **still has ρ=100 for the λ/η per-slot G scaling**, so λ/η augmentation would be $100 \cdot 0.01 \cdot 2 = 2$ at iter 0 (100× reference's 0.02).
- Likely still divergent — but never tested.

**(d) Coordinated: `rho=1.0` + `_use_g_matrix=True` + `use_reference_q_vector=True`:**
- Reference-equivalent operating regime. Never tested.

---

## 4. Empirical receipts

All runs use `git=6f2d74a` or `git=fc582aa`, `push_t` task, `admm_iter=3` unless noted, `rho_init=100.0` throughout.

| Run | Config change | trans_err | rot_err | tight | Notes |
|---|---|---|---|---|---|
| p39 | `use_reference_q_vector=True` only | 0.0769 m | 0.191 rad | FAIL | q_vec migration alone; ee_vel damping visible in slower push |
| p40 | q_vec, ee_vel zeroed | 0.0505 m | 0.273 rad | FAIL | Better trans (proves ee_vel term is real problem), worse rot |
| p41 | q_vec + ee_mass tweak | (no [RESULT] line — did not converge in log) | — | — | Intermediate |
| p44 | Full coordinated q_vec, admm_iter=3 | (diverged) | (diverged) | — | Dual residual 970-1200 per solve |
| p46 | Full q_vec, admm_iter=25 | (diverged catastrophically) | — | — | Dual residual 1.5M — port's ρ ramp `100·3^24` explodes at high iter counts |
| p68 | `_use_g_matrix=True`, q_vec off | 0.0580 m | **0.047 rad** | FAIL | BEST rot_err seen — G-matrix alone helps rotation but doesn't close translation |
| p69 | `_use_g_matrix=True` + regfrac=0.1, q_vec off | 0.0218 m | 3.1415 rad | PASS(latched) | T physically tipped over (rot_err = π means upside-down); "tight_goal PASS" is a latching artifact from crossing the xy tolerance during the tip |

**Never attempted (based on log inventory):**
- `_use_g_matrix=True` + `use_reference_q_vector=True` at any ρ.
- Any run with `rho ≠ 100.0`. Every log line shows `rho_init: 100.0`.

---

## 5. Minimum viable unblock path

**Not this arc, not this session.** The coordinated triple flip requires methodical staging so each intermediate state is stable enough to diagnose. Suggested sequence for a future dedicated arc:

1. **ρ sweep in current (baseline) config.** Vary `rho ∈ {1, 3, 10, 30, 100}` on the *default* port config (`_use_g_matrix=False`, `use_reference_q_vector=False`). If ρ=100 was empirically chosen, this should show WHY — presumably lower ρ diverges under uniform-I augmentation. Establishes baseline ADMM behavior curve vs ρ.

2. **G-matrix isolation at ρ=1.** Enable `_use_g_matrix=True` with ρ=1 (not 100). Reference-conformant λ/η augmentation should be `1·0.01·2=0.02` on λ, matching reference iter 0 exactly. Q stays at port's sparse-huge. If ADMM converges here, the "G matrix destabilizes" narrative is falsified — it was only destabilizing because of the ρ=100 amplification.

3. **Q-only migration at ρ=1 + G-on.** Add `use_reference_q_vector=True` on top of step 2. Reference-conformant Q + reference-conformant G + reference-conformant ρ. This is the actual reference regime. Compare to reference C++ push_t runs (if available in external) for the sanity check.

4. **Regression coverage.** Only after step 3 shows convergence, run box regression to confirm box path isn't broken (box uses different task cfg but shares the C3Solver / _use_g_matrix flag).

**Estimated scope:** 4 discrete arcs, each ~1-2 hours. Each arc produces a single receipt and either advances the chain or falsifies a hypothesis. No coordinated multi-flip attempt without the isolated data.

**Not-a-blocker (verified this investigation):** the LCS state layout is different (reference `[ee_pos, quat, obj_pos, ...]` vs port `[quat, obj_pos, ee_pos, ...]`) but the port's Q remap correctly places reference q_vector values into port's layout. This is bijective and doesn't affect the ADMM math. The `tasks.yaml:203` comment claiming "LCS layout parity needed" is misleading — what's actually needed is ρ / G / Q consistency.

---

## 6. Arc 1 empirical readout (2026-07-25, git=3759820)

Executed via `PORT_RHO=<r> python main.py push_t --admm-iter 3 --max-time 180
--solver c3plus --ee-space --sampling-c3 config/sampling_c3_kik_t.yaml`.
Baseline config unchanged (`_use_g_matrix=False`, `use_reference_q_vector=False`).
Anchors p70/p71 predate the env override; p72–p74 are new.

**Goal-hitting curve:**

| ρ | run | trans_err (m) | rot_err (rad) | tight | loose |
|---|---|---|---|---|---|
| 1 | p71_rho1 | 0.052 | 0.308 | FAIL | FAIL |
| 3 | p72_rho3 | 0.172 | π (tipped) | PASS(latched) | FAIL |
| **10** | **p73_rho10** | **0.021** | **0.064** | **PASS** | **PASS** |
| 30 | p74_rho30 | 0.122 | 0.048 | FAIL | FAIL |
| 100 | p70_sanity_240s | 0.033 | 0.230 | FAIL | PASS |

**ADMM residual curve (median over run, C3+ mode):**

| ρ | primal_end (med) | dual_end (med) | mono violations | consensus-steps |
|---|---|---|---|---|
| 1 | 3.13 | 14.8 | 98/2572 (3.8%) | 2572 |
| 3 | 2.27 | 36.8 | 8/2177 (0.4%) | 2177 |
| 10 | 1.58 | 103 | 0/2176 | 2176 |
| 30 | 1.14 | 278 | 0/3090 | 3090 |
| 100 | 1.11 | 902 | 0/3874 | 3874 |

**Key finding:** ADMM residual quality is monotone in ρ (as theory
predicts — larger ρ tightens primal at the cost of dual growth), but
goal-hitting is *not* — the peak sits at ρ=10 (previously untested).
The "port was tuned for ρ=100" framing in §3 is contradicted: ρ=100
delivers only loose_goal PASS at 0.033m/0.230rad, while ρ=10 delivers
tight+loose PASS at 0.021m/0.064rad.

The ρ=3 tip-over (rot=π) matches p69's failure mode (T physically tips
during push) and is the only run in this sweep that reproduces the
"T-tumble" pathology previously attributed to G-matrix ON. Since
`_use_g_matrix=False` throughout this sweep, the tumble is
mechanically about Q-vs-ρ balance, not G-matrix presence.

**Immediate implication for arc 2:** the prescribed "G-matrix at ρ=1"
step (§5.2) may be measuring the wrong ρ. Arc 2 should probably ALSO
include G-matrix at ρ=10 for a matched comparison against the new
sweet spot. Left as a user decision — the original arc-2 spec still
stands as-written.

## 7. Arc 2 empirical readout (2026-07-26, git=3759820)

Executed via `REFCONF_USE_G_MATRIX=1 PORT_RHO=<r> …` matching arc-1
protocol otherwise. `[G-MATRIX] active: per-slot x=0 λ=0.02 u=0 η=0.01`
banner confirms infrastructure engaged as designed.

**Goal-hitting (G-on):**

| ρ | run | trans_err (m) | rot_err (rad) | outcome |
|---|---|---|---|---|
| 1 | p75_gON_rho1 | — | — | **CRASH** — workspace violation at sim_t=26.7 s (EE r=0.769 > cap 0.75) |
| 10 | p76_gON_rho10 | 0.135 | 1.106 | tight FAIL, loose FAIL |

**Matched arc-1 anchors (G-off, same ρ):**

| ρ | trans_err | rot_err | outcome |
|---|---|---|---|
| 1 | 0.052 | 0.308 | FAIL |
| 10 | **0.021** | **0.064** | tight+loose PASS |

**ADMM residual comparison (median over solves):**

| config | n_cs | mono_false | primal_end | dual_end |
|---|---|---|---|---|
| G-off, ρ=10 | 2176 | 0 (0%) | 1.58 | 103 |
| G-on, ρ=10 | 1930 | 1348 (70%) | 9.16 | 247 |
| G-off, ρ=1 | 2572 | 98 (3.8%) | 3.13 | 14.8 |
| G-on, ρ=1 | 282 (crash) | 227 (80%) | 13.7 | 33 |

**§5.2 hypothesis falsified — in the opposite direction:**

§5.2 predicted: "if ADMM converges here [G-on at ρ=1], the 'G matrix
destabilizes' narrative is falsified — it was only destabilizing
because of the ρ=100 amplification."

Actual observation: G-on at ρ=1 destabilizes ADMM *more*, not less,
than at ρ=100. Non-monotone iteration rate climbs from ~0% (G-off) to
70–80% (G-on) at both ρ tested. Primal residual 6–10× larger. G-on
at ρ=1 crashes via downstream reposition targets drifting out of the
workspace (port-todo #5 DRAKE_DEMAND abort); G-on at ρ=10 completes
but with 6× worse translation and 17× worse rotation than the G-off
anchor at the same ρ.

**Interpretation:** the reference C++ ADMM must have additional
convergence machinery the port lacks — candidates: (a) adaptive-ρ
firing within the 3-iter budget (port's fires every 10 iters →
never in admm_iter=3), (b) warm-start of ω between solves, (c)
different δ projection order/form, (d) different LCS scaling
interaction with G-diag augmentation. Without one of these, the
per-slot G-diag augmentation actively fights ADMM convergence in
the port at the reference's `w_G · g_λ = 0.02` scale.

## 8. What this investigation does NOT do

- Does not attempt arcs 3–4.
- Does not modify `admm_solver.py`, `task_costs.py`, `tasks.yaml`, or
  the runtime code paths in `main.py` beyond the arc-1 PORT_RHO env
  override (commit `3759820`) and the arc-2 REFCONF_USE_G_MATRIX env
  gate (already in place from commit `f484607`).
- Sims are single-run per config, unseeded, matching arc-1 protocol.

## 9. Bottom line (post-arc-2)

Item #7's original 4-arc plan is **compromised at arc 2**. The
"G-matrix destabilizes only at high ρ" story is refuted; G-matrix at
any tested ρ produces catastrophic ADMM non-monotonicity in the port,
so arc 3 (add q_vector migration on top of G-on + ρ=1) cannot proceed
in good faith without first diagnosing *why* G-diag augmentation
breaks the port's ADMM. Adding a third experimental variable to a
provably unstable substrate compounds uncertainty rather than
resolving item #7.

**Two independent paths forward, either or both:**

1. **Capitalize on arc-1's ρ=10 sweet spot** as a `main`-track
   improvement, independent of item #7. Confirm the win on box +
   push_t alternate goals; if it holds, land it as a default
   change. Zero code beyond a rho_init default flip.
2. **Diagnose why G-diag destabilizes port ADMM** before arc 3.
   Compare port vs reference `_solve_c3plus` iter-by-iter with G-on
   and a simple 1-contact toy LCS: what does the reference do at
   iter 1 that the port doesn't? Candidates listed in §7's
   interpretation paragraph.

Item #7 stays STRUCTURALLY BLOCKED, and now with a specific mechanism:
port ADMM cannot tolerate reference-conformant G-diag augmentation
without additional convergence machinery not yet identified.
