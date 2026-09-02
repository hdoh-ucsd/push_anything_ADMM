# Arc-2 §A — LCS-matrix extraction structural diff (port ↔ reference)

**Date:** 2026-07-26
**Scope:** LCS extraction ONLY. `(A, B, D, d, E, F, H, c)` construction from a
common Drake plant state. Inner ADMM covered by sibling §D
(`docs/superpowers/investigations/2026-07-26-arc2-D-admm-diff.md`); outer loop
covered by sibling §E (`2026-07-26-arc2-E-outer-diff.md`).

**Conformance-map baseline (do not re-derive):** `docs/conformance-map.md`
subsystem 3 (pair admission) at 2026-07-14/17/25 states. That map deferred
the LCSFactory internals as "external — clone denied." **THAT PREMISE IS NOW
STALE**: `/root/reference_repos/c3/` is fully populated locally at this
session's baseline (`multibody/lcs_factory.cc`, `core/lcs.cc/h`, `multibody/
lcs_factory.h`). This report closes the gap by reading the reference
LCSFactory directly. Deltas below therefore include material the map could
not observe.

**Port baseline:** working tree HEAD (2026-07-25 sweep + WIP D3/E1/D1 on
`admm_solver.py`/`inner_solve.py`) — none of those WIPs touch
`lcs_formulator.py`, which has been quiet since 2026-07-25 arc-2 close.

**Reference baseline:**
- `c3/multibody/lcs_factory.cc` (857 lines) + `.h` (392 lines)
- `c3/core/lcs.cc` (101 lines) + `.h` (171 lines)
- `dairlib_sampling_c3 @ push_anything_dev 257e3ede` —
  `systems/controllers/sampling_based_c3_controller.cc:1580-1698` (LCS
  construction call sites)
- `examples/sampling_c3/push_t/parameters/sampling_c3_options.yaml`
  (contact_model, planning_dt_pose, resolve_contacts_to, num_friction_directions)

**Runtime path locked in by the port's canonical scripts (`scripts/run_*.sh`
grep `--ee-space`, all present)** — the active LCS builder is
`LCSFormulator.linearize_discrete_ee_space` with the `_contact_model
= 'anitescu'` overwrite path (`lcs_formulator.py:1875-1960`), NOT the R^7
`linearize_discrete` Stewart-Trinkle path. This report treats EE-space +
Anitescu as the primary comparison and calls out R^7 differences only where
they matter.

---

## 0 — TL;DR (top-3 LCS deltas ranked by likelihood of amplifying G-on instability)

1. **State layout is architecturally different (paper §IV-A vs full-plant LCS).**
   The reference LCS is built on the **full multibody plant** (Franka arm
   + manipuland), n_x = 19 for push_t (7 arm q + 7 obj q, 7 arm v + 6 obj
   v, but the yaml `q_vector` at push_t/…c3_options.yaml:70-75 shows the
   reference layout is EE-position(3) + object quat(4) + object xy(3) +
   arm/EE lin-vel(3) + object ang-vel(3) + object lin-vel(3) = 19). The
   port's `linearize_discrete_ee_space` uses **only** box+EE point-mass:
   x = [box_q(7), p_ee(3), box_v(6), v_ee(3)] = 19, ignoring arm dynamics
   entirely (arm Jacobian doesn't appear in B or D). The count matches by
   coincidence, the semantics do not — port's B and D are **configuration-
   independent** in arm q (see `lcs_formulator.py:1414-1419` comment: "B_ctrl
   is CONFIGURATION-INDEPENDENT"). Reference's B_ctrl includes `dt·Jf_u`
   which folds through the full-plant M⁻¹, so it IS arm-configuration
   dependent. **Suspicion**: the port's simplified planner LCS produces
   **artificially small D_ee_columns** (`(dt/m_ee)·J_c_ee` with m_ee=1 kg
   vs the reference's `(dt·Jf_u)` term that carries the full arm
   inertia). `ScaleComplementarityDynamics` on the port rescales D up by
   `||A||/||D||`; this scale interacts with the G-matrix penalty at ADMM
   time (see §D report §2A). Under G-on, if D is being over-scaled to
   compensate for the small point-mass EE inertia, then `E/scale` shrinks
   in the same tick, and the augmented λ-block reads a smaller predicted
   gap — the projection then permits larger λ, which the G-augmentation
   amplifies. **Not a trivial delta; likely load-bearing under G-on.**

2. **Reference has an `E`-block cross term the port drops on both paths.**
   Reference Anitescu (`lcs_factory.cc:531-534`):
   `E[:, 0:n_q] = dt·J_c·Jf_q + E_t^T·J_n·vNqdot/dt` — the trailing
   `E_t^T·J_n·vNqdot/dt` is the linearization of the `phi/dt` position
   forcing term evaluated at `q_k` (contributes to `η`'s ∂/∂q). The port
   EE-space Anitescu at `lcs_formulator.py:1924-1927` writes
   `E[:, BOX_Q_SLOT] = dt·J_c_box·df_box_dboxq` only — no `J_n·N/dt`
   analog. The port then compensates in `c` via `E_t^T·phi/dt`
   (`lcs_formulator.py:1942`) and subtracts `E·x*` at :1945 to fold the
   linearization-point contribution back in. **This algebraically closes
   for x = x\*** (holds at the linearization point) **but diverges as
   soon as x moves away from x\***. Under a horizon-N solve, knots 2..N-1
   are away from x*, so the port's η(x) has a missing linear term in q
   compared to the reference. Effect: `η_predicted` is off by
   `-E_t^T·J_n·vNqdot/dt · (q - q*)` per knot. Impact on λ magnitude
   depends on scale of `vNqdot` for the object quaternion (large, since
   quat integration has `Nq ≈ H(q)/2` structure) — under G-on this
   translates to λ-projection targets drifting knot-to-knot.

3. **Reference has NO signed-distance threshold; port's pipeline hard-codes 0.002 m.**
   Reference `LCSFactory` (`lcs_factory.cc:263-288 ComputeContactJacobian`)
   iterates over pre-specified `contact_pairs_` and calls each pair's
   `ContactEvaluator::Eval(context_)` — the evaluator returns `(phi_i,
   J_i)` regardless of whether phi is positive, negative, or zero. All
   pre-specified pairs enter the LCS at every tick. Port
   (`lcs_formulator.py:481-492`, `:537-575`) starts with
   `query_obj.ComputeSignedDistancePairwiseClosestPoints(threshold=0.002)`
   and only keeps EE-manipuland / manipuland-ground pairs. This is the
   `conformance-map §3.a / §3.b / §3.k / §3.l / §3.m` "five-mechanism"
   pair-admission stack. Under Anitescu the port DOES have the
   always-on/top-K workarounds (`_always_on_ee_box=True` default,
   `_ref_pair_admission_planner_lcs=True`), but the `n_ee_top_k=1`
   default means the port still admits **exactly 1 EE-manipuland pair**
   whereas the reference push_t admits `resolve_contacts_to=[0, 1, 3]`
   meaning 0 pairs from group 0 (ee-ground), 1 from group 1
   (ee-object), 3 from group 2 (object-ground). **The n_c count and per-
   pair identity match by design**, but the port's variable-count vs
   reference's fixed-count creates ripples in `n_lambda_ = 4·n_c` and
   consequently in the G-matrix width (per-slot G-diag length is
   λ-dimension-dependent, `admm_solver.py:1060-1091`). This is documented
   in the 2026-07-14 map as CONFIRMED-DIFFERENT but was there tagged
   "unclear if runtime-affecting"; here we assert it IS runtime-affecting
   under G-on because the G-diag rebuild path in
   `admm_solver.py:1417-1426` depends on `n_lambda`.

---

## 1 — Function map

| Concern | Reference (file:lines) | Port (file:lines) | Status |
|---|---|---|---|
| LCS container (`{A_i, B_i, D_i, d_i, E_i, F_i, H_i, c_i}` per-knot) | `c3/core/lcs.h:24-169`, `.cc:11-58` | 12-tuple returned; per-knot replication done by ADMM (`admm_solver.py:1150-`) — **NO LCS container class** | PORT-DIVERGENT (container-less; ADMM replicates matrices to per-knot on the fly) |
| Time-invariant LCS ctor (single A, replicated to horizon N) | `lcs.h:49-53`, `lcs.cc:33-39` | Implicit — port passes the 12-tuple through `admm_solver.py:solve`, which uses the SAME matrices for every knot (`admm_solver.py:927-2130` per-i loop reads shared A/B/D) | PRESENT-IMPLICIT |
| `ScaleComplementarityDynamics` (scale = ‖A‖/‖D‖) | `c3/core/lcs.cc:46-58` | 2 sites: EE-space path `lcs_formulator.py:1976-1991`; ADMM-side rescale `admm_solver.py:1117-1124` | PRESENT — **DUPLICATED** (both fire; see §2H) |
| `LCSFactory` (options + contact pair init) | `c3/multibody/lcs_factory.cc:119-227` | `LCSFormulator.__init__` `lcs_formulator.py:55-209` | PRESENT (semantically) |
| `GetNClosestContactPairs` | `lcs_factory.cc:229-261` (nth_element by signed distance) | `lcs_formulator.py:603-647` (`force_top_k_ee_box` sort+slice; EE-manipuland ONLY) | PARTIAL — port covers only the EE-manipuland group |
| `ComputeContactJacobian` (per-pair Eval) | `lcs_factory.cc:263-288` (loops per-pair, packs Jn/Jt block) | `lcs_formulator.py:670-769` (Drake `ComputeSignedDistancePairwiseClosestPoints` → per-pair `CalcJacobianTranslationalVelocity` → dot with nhat/tangent basis) | PRESENT (different call structure; same output) |
| `GenerateLCS` — main matrix assembly | `lcs_factory.cc:303-422` | R^7: `lcs_formulator.py:1069-1322`; EE-space: `lcs_formulator.py:1367-2027` | PRESENT (two separate paths in port, one in ref) |
| Autodiff Jacobian `Jf = ∂(M⁻¹(τ_g+τ_u+f_app-Cv))/∂(q,v,u)` | `lcs_factory.cc:310-343` (single `plant_ad_.CalcMassMatrix + CalcBiasTerm + MakeActuationMatrix` on AD plant; `M.ldlt().solve(...)` inside AD) | R^7: `lcs_formulator.py:231-290` (separate AD passes for M, Cv, τ_g; explicit chain-rule `df/dx = M⁻¹[d(rhs)/dx - dM/dx·f]` at `:284-288`); EE-space: `:1607-1636` (SAME manual chain-rule pattern applied to box-block slice) | PRESENT — manual chain rule (vs ref's `M.ldlt().solve` inside AD tape) |
| `FormulateStewartTrinkleContactDynamics` (E/F/D/H/c) | `lcs_factory.cc:438-494` | R^7: `lcs_formulator.py:1234-1279`; EE-space ST (pre-Anitescu overwrite): `:1743-1853` | PRESENT (structural mirror) |
| `FormulateAnitescuContactDynamics` (E/F/D/H/c) | `lcs_factory.cc:496-545` | EE-space overwrite: `lcs_formulator.py:1875-1960` | PRESENT (structural mirror) |
| `FormulateFrictionlessSpringContactDynamics` | `lcs_factory.cc:423-436` | **MISSING in port** (never needed; `_contact_model` limited to anitescu/stewart_trinkle) | MISSING (inert) |
| `f_app = plant.CalcForceElementsContribution` (spring / damper / gravity elements) | `lcs_factory.cc:327-328` — INCLUDED in `f_qvu` | `lcs_formulator.py:257-260`, `:1518` — **MISSING** — port uses `rhs = B·u − Cv + τ_g` only, no `f_app` term | **STRUCTURAL DELTA** (see §2E) |
| `MakeVelocityToQDotMap` (q̇ = N(q)·v) | `lcs_factory.cc:353-354` (`plant_.MakeVelocityToQDotMap`) | R^7: `lcs_formulator.py:1152-1156` (loops columns via `MapVelocityToQDot`); EE-space: `:1508-1513` (box-slice loop) | PRESENT (functionally equivalent) |
| `MakeQDotToVelocityMap` (v = N⁺(q)·q̇) | `lcs_factory.cc:385-387` — used in `E` cross term | **MISSING in port** — port doesn't build `vNqdot`, effectively N⁺ = 0 | **DELTA** (§2C) |
| LCS simulator (`Simulate`, uses MobyLcpSolver) | `lcs.cc:60-75` | **MISSING in port** — port never simulates the LCS forward; ADMM solves the trajectory directly | MISSING (unused in main pipeline) |
| Fixed-modes reduction (`FixSomeModes`) | `lcs_factory.cc:191-192` decl (impl elsewhere) | **MISSING in port** | MISSING (unused) |

---

## 2 — Structural diffs per matrix (ordered by ADMM impact)

### 2.A — A (state dynamics matrix)

Reference (`lcs_factory.cc:367-371`, Anitescu path — same block for ST):
```cpp
A(0:n_q,0:n_q)      = I + dt² · qdotNv · Jf_q
A(0:n_q,n_q:n_x)    = dt · qdotNv + dt² · qdotNv · Jf_v
A(n_q:n_x,0:n_q)    = dt · Jf_q
A(n_q:n_x,n_q:n_x)  = I + dt · Jf_v
```

Port R^7 (`lcs_formulator.py:1181-1185`):
```python
A[:n_q,:n_q] = I + dt² · N · J_q
A[:n_q,n_q:] = dt · N @ (I + dt·J_v)                # == dt·N + dt²·N·J_v
A[n_q:,:n_q] = dt · J_q
A[n_q:,n_q:] = I + dt · J_v
```

**Structurally identical** for R^7 path (**semi-implicit Euler on q; explicit
Euler on v; then substitute v_{k+1} into q_{k+1} = q + dt·N·v_{k+1}**).

Port EE-space (`lcs_formulator.py:1649-1664`): same shape but restricted to
the box-only block via `df_box_dboxq` / `df_box_dboxv` — arm columns of A
are absent by construction. EE positional block is A[p_ee,p_ee]=I,
A[p_ee,v_ee]=dt·I; A[v_ee,v_ee]=I (no damping). This is where the "point-
mass EE" approximation lives — the reference has an arm Jacobian contribution
here through the full-plant Jf_v (arm mass-matrix inversion enters).

Verdict: **structurally identical for R^7; simplification in EE-space
(architecturally intended per paper §IV-A) that suppresses arm contribution
to A entirely**. This is likely a benign approximation on its own but
compounds with §2B below.

Optional box-ground drag knob in port (`lcs_formulator.py:1196-1212`,
`_box_drag_c`) has no reference analog. Currently OFF (`_box_drag_c=0.0` at
`:108`), inert on the runtime path — safe to ignore.

### 2.B — B_ctrl (input matrix)

Reference (`lcs_factory.cc:373-375`):
```cpp
B(0:n_q,   :) = dt² · qdotNv · Jf_u
B(n_q:n_x, :) = dt · Jf_u
```
where `Jf_u = ∂f/∂u = M⁻¹·B_actuation` at (q\*,v\*,u\*) — carries the
full-plant M⁻¹.

Port R^7 (`lcs_formulator.py:1215-1217`): same shape — `J_u = J_f[:, n_q+n_v:]`
comes from `M⁻¹·rhs`. **Identical**.

Port EE-space (`lcs_formulator.py:1675-1680`):
```python
B_ctrl[P_EE_SLOT, :] = (dt² / m_ee) · I_3      # arm dynamics DROPPED
B_ctrl[V_EE_SLOT, :] = (dt  / m_ee) · I_3
# box_q, box_v rows: ZERO (u doesn't enter box dynamics)
```
Reference has `dt·Jf_u` on the v_box rows (arm actuation reaching box
through contact-free f, ≈ 0 for decoupled trees but not identically zero
under Drake numerics). Port sets these to 0 by construction. **Delta:
architectural — see TL;DR §0.1.**

There is a `[LCS-COND-WARN]` guard at `lcs_formulator.py:1691-1701` flagging
`dt/m_ee > 0.5` as unsafe; currently `dt/m_ee = 0.075/1.0 = 0.075`, safe.

### 2.C — d (state-side bias)

Reference (`lcs_factory.cc:378-379`):
```cpp
d(:n_q) = dt² · qdotNv · d_v
d(n_q:) = dt · d_v
```
where `d_v = f(q*,v*,u*) - Jf · [q*;v*;u*]` (`:350`).

Port R^7 (`lcs_formulator.py:1231-1232`): identical shape, identical
`d_v_offset` construction at `:1176`. **Identical.**

Port EE-space (`lcs_formulator.py:1731-1737`): box-side same;
`d_v_ee_offset = 0` by construction (`f_ee = u/m_ee` is linear in u, no
constant part). Reference has no analog cell — arm q̈ takes gravity here.
Under EE-space this is an **intentional simplification** (paper §IV-A
"the planner does not model gravity on the EE").

### 2.D — D (contact-Jacobian → state)

Reference Anitescu (`lcs_factory.cc:527-529`):
```cpp
D(0:n_q,  0:n_λ) = dt² · qdotNv · M⁻¹ · J_c^T           where J_c = E_t^T·Jn + diag(μ)·Jt
D(n_q:n_x,0:n_λ) = dt   · M⁻¹ · J_c^T
```
Single-block Anitescu (all-λ), size (n_x × 4·n_c).

Port EE-space Anitescu (`lcs_formulator.py:1908-1913`):
```python
D_an[BOX_Q_SLOT, :] = (dt²) · (N_box @ Minv_JcT_box)
D_an[P_EE_SLOT,  :] = (dt² / m_ee) · J_c_ee.T
D_an[BOX_V_SLOT, :] = dt · Minv_JcT_box
D_an[V_EE_SLOT,  :] = (dt / m_ee) · J_c_ee.T
```
Structurally identical for the box block; the port ALSO writes into
`P_EE_SLOT` and `V_EE_SLOT` (EE position/velocity rows of the state get
D-coupling), which the reference **does implicitly** through the full-plant
M⁻¹ (arm columns of Jn/Jt fold through the arm slice of M⁻¹). The port's
`(dt/m_ee)·J_c_ee.T` is the **explicit EE-point-mass rewrite** of that
reference-side implicit path.

**Numerical delta**: for the same J_c, the port's D_v_ee entries scale
with `1/m_ee = 1.0` while the reference's would scale with the effective
inverse-arm-inertia projected onto the Cartesian direction. Franka's
diagonal joint inertias are O(1..10) kg-m²; the effective Cartesian
inertia at fingertip is O(1..3) kg. So port `1/m_ee = 1.0` under-shoots
by ~2-3× on average — port's D_v_ee rows are LARGER than reference's
equivalent, meaning the port model predicts larger v_ee changes per unit
λ than the reference. Under G-on this translates to over-predicting the
next-knot state cost from contact forces, which the ADMM projection then
tries to counteract via smaller λ — but the ρ_scale=3 ramp then swings
too far. **Non-negligible but does not obviously destabilize.**

Port R^7 Stewart-Trinkle (`lcs_formulator.py:1219-1227`): γ-cols are zero
(matches ref ST layout `lcs_factory.cc:596-599`); λ_n and λ_t cols
identically written with `(dt²)·N·M⁻¹·J^T` and `dt·M⁻¹·J^T`. **Identical
to ref ST.**

### 2.E — E, F, H, c (complementarity block)

**Reference Anitescu** (`lcs_factory.cc:531-544`):
```cpp
E(:, 0:n_q)   = dt · J_c · Jf_q + E_t^T · J_n · vNqdot / dt      # ← position-forcing term
E(:, n_q:n_x) = J_c + dt · J_c · Jf_v
F              = dt · J_c · M⁻¹ · J_c^T
H              = dt · J_c · Jf_u
c              = E_t^T · phi / dt + dt · J_c · d_v
                    - E_t^T · J_n · vNqdot · plant_.GetPositions(context_) / dt
```

**Port EE-space Anitescu** (`lcs_formulator.py:1915-1945`):
```python
E_an[:, BOX_Q_SLOT] = dt · (J_c_box @ df_box_dboxq)
E_an[:, BOX_V_SLOT] = J_c_box + dt·(J_c_box @ df_box_dboxv)
E_an[:, V_EE_SLOT]  = J_c_ee
# NOTE: E_an[:, P_EE_SLOT] left ZERO
F_an              = dt·(J_c_box·M_box⁻¹·J_c_box^T) + (dt/m_ee)·(J_c_ee·J_c_ee^T)
H_an              = (dt/m_ee) · J_c_ee
c_an              = E_t_an^T · phi / dt
                    + J_c_box · (box_v + dt · d_box_v_offset)
                    + J_c_ee  · (v_ee  + dt · d_v_ee_offset)
c_an             -= E_an·x_star + H_an·u_star           # subtract linearization-point value
```

**Deltas**:

**E1 — missing `E_t^T · J_n · vNqdot / dt` term in port E-column for q.**
Reference splits the position-forcing term into an `E`-column (dependence
on knot's q) and a `c`-constant. Port bakes ALL of the position-forcing
term into `c` (line 1942: `E_t^T · phi/dt`), then subtracts `E·x*` at
:1945. **For x = x* this is algebraically equivalent**. For x ≠ x* the
port drops the linear-in-q term from η. Specifically, define η(x) =
E·x + F·λ + H·u + c. Reference η'(q) has slope `dt·J_c·Jf_q + E_t^T·J_n·N/dt`
in q. Port η'(q) has slope only `dt·J_c_box·df_box_dboxq`. The missing
`E_t^T·J_n·N/dt` acts on the **object quaternion / position part of q**
directly (through `J_n` on the object DOFs), not through Jf_q. Under a
horizon-N solve at x_1 .. x_N, this differs by ~`||J_n||·||N||/dt · Δq`
per knot. For push_t with `q_vector[3:7]=0.1` on quat and `[7:10]=200`
on obj position, this is a large per-knot mismatch in η.

**E2 — port includes `E_an[:, V_EE_SLOT] = J_c_ee`** — the EE-velocity
contribution to `J_c · v_{k+1}`. Reference doesn't need this explicitly
because `v_{k+1}` in the ref covers the full plant velocity including
the EE via arm; port must add it because v_ee is a separate state slot.
Correct compensation for the EE-space simplification.

**E3 — port F has structurally SAME form** but the two summands
(`dt·J_c_box·M_box⁻¹·J_c_box^T` + `(dt/m_ee)·J_c_ee·J_c_ee^T`) approximate
the reference's `dt·J_c·M⁻¹·J_c^T` under the assumption M is block-
diagonal between arm and box AND the EE contribution folds via a scalar
`m_ee`. The block-diagonality assumption is correct for a fixed-base
arm + free-floating box; the point-mass EE assumption is the paper §IV-A
approximation. **F under ee-space is a slight LOW-rank re-parameterization
of ref-F; magnitudes match under the m_ee=1 kg approximation** discussed
in §2D.

**H1 — port H = `(dt/m_ee) · J_c_ee`** (`lcs_formulator.py:1938`) —
Reference H is `dt · J_c · Jf_u`, where `Jf_u = M⁻¹·B_actuation` folds
arm actuation through the full mass matrix. Port's H maps `u ∈ R^3` (EE
force) into η via a point-mass scalar. Under EE-space this is
architecturally correct (u is Cartesian force, not joint torque). The
port's implicit assumption: reference u is joint torque, so `Jf_u` is
required to translate joint torque → Cartesian EE motion. **No delta
here** since u is defined differently on the two sides. But note that
the ratio `‖H‖` between the two paths differs by ~`(J_arm^T·J_arm)/m_ee`,
which affects the ADMM q-block "sensitivity to u" — under G-on the
input-slot g-diag reads this differently.

Port R^7 Stewart-Trinkle (`lcs_formulator.py:1234-1279`):
- γ rows (`SG:SG+n_c`): `F[γ, λ_n] = μ·I`, `F[γ, λ_t] = -E_t` — MATCHES
  reference ST at `lcs_factory.cc:473-474`.
- λ_n rows: E, F, H, c match reference ST at `lcs_factory.cc:464-476`
  **except** port uses the "simple gap discretization" comment at
  `lcs_formulator.py:1258-1263`: `η_n = phi/dt + J_n·v_{k+1}`. Reference
  has `E.block(n_c, 0, n_c, n_q) = dt²·Jn·Jf_q + Jn·vNqdot` and `c.segment(n_c,
  n_c) = phi + dt²·Jn·d_v - Jn·vNqdot·q_star`. **Deltas**:
  - Ref has explicit `Jn·vNqdot` cross-term in E and c. Port drops it
    (comment at `:1260-1262` explicitly cites this as intentional to
    "avoid the floating-base n_q ≠ n_v mismatch from Aydinoglu's explicit
    (1/dt)·J_n·(q − q*) term"). **Same E1 delta as EE-space above**.
  - Ref uses `phi + dt²·Jn·d_v` (units of length). Port uses `phi/dt +
    dt·J_n·d_v` (units of length/time). Both are algebraically consistent
    within their own respective `E`/`F` scaling — port divides through
    by dt because it's tracking `η_n` in velocity units, ref keeps it
    in position units. **Not a delta; two equivalent formulations
    modulo a dt scale.** (This DOES double-book on the port's separate
    `ScaleComplementarityDynamics` divide of c by `‖A‖/‖D‖` — see §2H.)

### 2.F — Contact-pair filtering & signed-distance threshold

**Reference**: pre-specified pair list at controller construction
(`franka_sampling_c3_controller.cc:124-239`) — for push_t: 2 ee-object
pairs + 3 ground-object pairs. `GetResolvedContactPairs` (`sampling_based_
c3_controller.cc:1582-1615`) picks the N-closest from each group per
tick via `LCSFactory::GetNClosestContactPairs` (`lcs_factory.cc:229-261`).
push_t's `resolve_contacts_to=[0,1,3]` selects 0 ee-ground + 1
ee-object + 3 object-ground = 4 pairs per tick, always. **No distance
threshold** — the closest pair is always admitted, no matter how far.

**Port**: threshold-based via
`ComputeSignedDistancePairwiseClosestPoints(threshold=0.002)`
(`lcs_formulator.py:537-539`). Then:
- Group-filter to (EE,manipuland) + (manipuland,ground) only.
- `_always_on_ee_box=True` default at `:106` injects a top-K EE-manipuland
  pair when the 2-mm filter admitted none.
- `_ref_pair_admission_planner_lcs=True` default at `:107`, tshape-only
  gate at `:1140-1143` and `:1382-1386`, routes tshape planner LCS
  through `force_top_k_ee_box=True, n_ee_top_k=1`.
- Cost-LCS (`inner_solve.py`) uses `force_top_k_ee_box=True, n_ee_top_k=2`.
- BOX path: default 2-mm auto-admit + `_always_on_ee_box` fallback. **NOT
  the reference top-K**. Box's `n_c` is thus variable tick-to-tick.

**Delta significance**: (a) tshape-path is *reference-conformant on
n_c* (top-K), (b) box-path is *variable-n_c*. Under G-on, box path's
variable-n_c means the G-diag length changes tick-to-tick, forcing
`admm_solver.py` to rebuild G on the fly. That rebuild path is
untested for correctness under `_g_diag_c3p_cache` invalidation —
possible interaction with the ρ_scale=3 per-iter update.

### 2.G — Autodiff strategy

Reference (`lcs_factory.cc:311-337`): builds `M`, `C`, `τ_g`, `f_app`,
`Bu` on the AD plant, does `M.ldlt().solve(...)` INSIDE the autodiff
tape, extracts `f_qvu_norminal = ExtractValue(f_qvu)` and `Jf =
ExtractGradient(f_qvu)`. Total forward AD passes: 1.

Port (`lcs_formulator.py:262-290`, R^7): builds `M`, `Cv`, `τ_g`, `B` on
the AD plant SEPARATELY, then does the chain rule manually for M⁻¹:
`J_f[:, k] = M⁻¹ @ (J_rhs[:, k] - J_M[:, :, k] @ f_eval)`. Total forward
AD passes: 3 (M, Cv, τ_g). Manual reassembly of `df/dx = M⁻¹[drhs/dx -
(dM/dx)f]`.

**Numerical delta**: both should produce the same J_f up to FP order. Port
does one extra M⁻¹ per column of J_f which adds ~1e-14 relative FP noise.
Not load-bearing.

### 2.H — LCS scaling (`ScaleComplementarityDynamics`) — **DUPLICATED IN PORT**

Reference (`c3/core/lcs.cc:46-58`): computes `scale = ‖A[0]‖ / ‖D[0]‖`,
mutates D *= scale, E /= scale, c /= scale, H /= scale. Called once
per Solve from `c3.cc:81 ScaleLCS()` inside `C3::UpdateLCS`.

Port (**TWO CALL SITES**):
1. `lcs_formulator.py:1976-1991` — EE-space path, mutates the tuple
   RETURNED to the caller. Fires unconditionally when `_scale_lcs=True`
   (default `REFCONF_SCALE_LCS=1`).
2. `admm_solver.py:1117-1124` — ADMM `_solve_c3plus`, mutates the local
   copy AGAIN. Fires unconditionally when `n_lambda > 0 and ‖D‖ > 0`.

**Compounding**: if formulator scales D by `s_1 = ‖A‖/‖D‖`, then ADMM
sees D_scaled_once and computes `s_2 = ‖A‖/‖D_scaled_once‖ = 1`. In that
case ADMM's `_lcs_scale = 1.0` and nothing further happens — algebraically
correct, no double-scaling. BUT the port also stores `_last_lambda_scale`
downstream in `admm_solver.py` for physical-λ recovery; if _lcs_scale=1
inside ADMM, the recovered λ is not un-scaled to physical units. **Bug
suspicion**: the executor may be consuming λ that is still in the
scaled coordinate system. See sibling §D report §2 for downstream λ
consumption.

Runtime evidence: `[LCS-SCALE] scale_lcs active` prints once per run
(`lcs_formulator.py:1986-1990`) — confirms path 1 fires. Path 2 is
silent by default; no `[LCS-SCALE-ADMM]` banner.

### 2.I — `f_app = plant.CalcForceElementsContribution` — MISSING in port

Reference (`lcs_factory.cc:327-332`):
```cpp
MultibodyForces<AutoDiffXd> f_app(plant_ad_);
plant_ad_.CalcForceElementsContribution(context_ad_, &f_app);
AutoDiffVecXd f_qvu = M.ldlt().solve(tau_g + tau_u + f_app.generalized_forces() - C);
```
`f_app` picks up spring/damper elements attached via URDF (Franka joint
spring dampers, if defined; workspace-wall compliance for `anything`
task).

Port (`lcs_formulator.py:257-260`, `:1518`): `rhs = B·u − Cv + τ_g` only.
No `f_app`. For the Franka URDF used in the port
(`sim/env_builder.py`), a quick check of the URDF would confirm whether
spring elements are attached; if none, port and ref match. If any spring
elements exist (walls or joint compliance), port's LCS predicts NO
restoring force from them. **Impact currently unknown — needs URDF
audit** (out of scope; flagged for follow-up).

---

## 3 — Common pitfall audit

| Pitfall | Port state | Reference state | Delta |
|---|---|---|---|
| **Sign convention on nhat** | `lcs_formulator.py:690-699`: `nhat = sdp.nhat_BA_W` (from B to A); `nhat_onto_box = nhat if A is box else -nhat`. J_n row = `nhat @ (J_A - J_B)`. | `lcs_factory.cc` delegates to `ContactEvaluator::Eval` which returns Jn as row-0 of a per-pair Jacobian block (`.h:44-49`). The evaluator's sign convention is fixed at ctor time (see `PlanarContactEvaluator` / `PolytopeContactEvaluator`). | **Assumed same** (Drake convention: nhat_BA points from B to A; force ON A from contact is +nhat·λ_n). No verified delta. |
| **Anitescu vs Stewart-Trinkle** | Both implemented; runtime path via `_contact_model = 'anitescu'` (default). ST path lives at `lcs_formulator.py:1240-1279` (R^7) and `:1743-1853` (EE-space pre-overwrite). | Both implemented at `lcs_factory.cc:438-494` (ST) and `:496-545` (Anitescu). push_t default = anitescu. | Same default. |
| **Contact-pair filtering** | 2-mm signed-distance threshold + geom-set filter + optional top-K + optional always-on. | Pre-specified pair list + `GetNClosestContactPairs` (top-K, no threshold). | **§2F delta** (see above). |
| **Signed-distance threshold** | 0.002 m (`lcs_formulator.py:491`); intentional per docstring at :482-490. | NONE (all pre-specified pairs at every tick). | **Structural delta**; the 2-mm threshold has no reference analog. |
| **dt vs planning_dt_pose swap under crossed_switching_threshold** | Port: `_crossed_switching_threshold` flag in `ci_mpc_c3plus.py:204`; swaps `dt → dt_pose` (push_t: 0.1→0.05). | Reference: `sampling_based_c3_controller.cc:800-848` swaps `dt_ = planning_dt_pose` on `crossed_cost_switching_threshold_`. Also swaps `resolve_contacts_to` list via `GetC3Options(crossed)`. | **Present-partial**: port swaps dt but does NOT swap `resolve_contacts_to` (port uses `resolve_contacts_to=[0,1,3]` fixed via the `n_ee_top_k=1` gate). Reference swaps to `resolve_contacts_to_for_cost=[0,2,3]` for the cost-LCS in the pose regime. |
| **LCS scaling AnDn = ‖A‖/‖D‖** | `_scale_lcs=True` default (EE-space path only); ALSO applied in `admm_solver.py` (see §2H). | Applied once from `C3::UpdateLCS` via `LCS::ScaleComplementarityDynamics`. `scale_lcs: true` in ref YAML. | **§2H — port has TWO application points; ADMM one is nominally a no-op after the formulator one but the λ un-scale path is unclear**. |

---

## 4 — Verdict

**Are port LCS matrices structurally identical to reference's?**

**R^7 Stewart-Trinkle path (default when `--ee-space` not passed):**
STRUCTURALLY YES for A, B, D, d, F (γ block), H (all rows), c (γ block).
STRUCTURALLY DIVERGES from reference on E (missing `Jn·vNqdot` cross-term
per §2E) and c (`phi/dt + Jn·d_v·dt` vs `phi + Jn·d_v·dt²`). Both diverges
are algebraically compatible with the ref choice for x = x* but non-
equivalent for x away from x*.

**EE-space Anitescu path (runtime default when `--ee-space` is passed —
all canonical scripts):** ARCHITECTURALLY DIFFERENT (paper §IV-A: point-
mass EE + box-only dynamics vs reference full-plant LCS). D/E/F have
different **magnitudes** on the EE rows due to the m_ee=1 kg point-mass
substitution for the reference's full arm inertia. **Not a bug; a
paper-sanctioned approximation.**

**Impact ranking under G-on:**

1. **§2A architectural difference (EE-space vs full-plant)** — HIGH.
   The point-mass EE gives D_v_ee rows ~2-3× larger than reference's
   arm-inertia-corrected equivalent. Under G-on the augmented cost sees
   an inflated predicted-state term, ADMM projection targets a smaller
   λ, ρ_scale=3 ramp overcompensates. Combined with the ρ=100 baseline
   this is exactly the amplification path sibling §D report §2A
   diagnoses as "200× the reference augmentation strength on λ at iter 0."

2. **§2E (E1) missing `Jn·vNqdot` cross-term** — MEDIUM. Off by
   ~`||J_n||·||N||/dt · Δq_object_quat` per knot. Reference has
   `q_vector` heavy on object quat (`0.1`) and object position (`200`);
   the missing E-column linearizes those into the ADMM cost differently
   than the reference. Under G-on the augmented λ-target absorbs this
   drift over the 25-iter port ramp (vs the reference's 3-iter cap).

3. **§2H double-scaling** — LOW-to-MEDIUM. Formulator-scale is
   nominally a no-op at the ADMM-scale (since D is already unit-normed
   after path 1). But the λ un-scale path in `admm_solver.py:_last_lambda_scale`
   consumes `_lcs_scale` from within `_solve_c3plus` — if that reads the
   ADMM-side (which is 1.0 after path 1 already normalized), the
   downstream executor's `_derive_force_command` gets λ in
   SCALED coordinates instead of physical N. Needs a trace to confirm.

4. **§2F variable-n_c on box path** — LOW-to-MEDIUM. Only affects box
   task; tshape uses the top-K gate. But the G-diag rebuild per tick
   under variable-n_c may interact with the WIP D1 rho-override.

5. **§2I missing `f_app`** — UNKNOWN. Depends on URDF; likely inert for
   the Franka push_t URDF (no spring elements normally attached).

---

## 5 — Concrete falsification tests (env-var probes; DO NOT run here)

Each probe is a **single-knob** ablation isolating one of the top deltas.
All are gated behind env vars (existing or to-be-added) so no code needs
to be committed to run them.

### Probe A1 — Bypass ScaleComplementarityDynamics at ADMM
Set `REFCONF_SCALE_LCS=1` (formulator scaling on) AND add a new env-guard
in `admm_solver.py:1117-1124` (`PUSHA_ADMM_SCALE_LCS=0`) to skip the
second scaling. Expected: no runtime change (algebraically no-op), but
confirms the double-book isn't secretly diverging. IF results DIFFER,
the ADMM-side re-scale is producing FP-drift λ.

### Probe A2 — Disable formulator ScaleLCS, keep ADMM ScaleLCS
`REFCONF_SCALE_LCS=0`. Then `admm_solver.py` sees unscaled D and applies
its own scale. Expected: same result as A1 if the double-scale is truly
a no-op. IF results DIFFER, the formulator-side scale is doing extra
work beyond just applying scale (e.g., leaking into `_last_lambda_scale`).

### Probe E1 — Restore `Jn·vNqdot` cross-term in E and c
Not env-gated currently; would require a `PUSHA_LCS_REF_E_CROSSTERM=1`
gate around `E_lcs[SLN:SLN+n_c, :n_q]` (R^7) at `:1264` and around
`E_an[:, BOX_Q_SLOT]` (EE-space) at `:1924`. Add `E_t^T·J_n·N_mat/dt` to
the E-block, and matching `-E_t^T·J_n·N_mat·q_star/dt` to `c_lcs`.
Expected: measurable change in ADMM iter-1 residuals under G-on; if
residuals drop, this term is the missing linear compensation.

### Probe D1 — EE-space m_ee sweep against reference D magnitude
Add env `PUSHA_EE_MASS_M_EE=X` around `_EE_MASS = 1.0` at
`lcs_formulator.py:1365`. Sweep `m_ee ∈ {0.5, 1.0, 2.0, 3.0}`. If G-on
performance is a monotone function of m_ee (bigger m_ee = closer to
reference D magnitude = more stable ADMM), we've isolated §2A.
**Caveat**: prior p41 attempt at m_ee=0.057 blew up; current baseline
comment (`:1358-1364`) explicitly cites 1.0 as "reverted to 1.0 due to
ill-conditioning at other values." Sweeping DOWN from 1.0 is likely
safe; sweeping UP toward the ref-equivalent Cartesian inertia
(~2-3 kg) is the direction of interest.

### Probe F1 — Force uniform top-K admission on box path
Toggle `_always_on_ee_box=True` (already default) AND set env
`PUSHA_LCS_BOX_TOPK=4` (would need new env-gate around
`lcs_formulator.py:1140-1143` to enable for box too, not just tshape).
Expected: box path's `n_c` becomes constant tick-to-tick, matching
reference. If G-on then behaves more like tshape, §2F is confirmed
load-bearing.

### Probe I1 — Include `CalcForceElementsContribution`
Not env-gated. Modify `lcs_formulator.py:257-260` and `:1518` to add
`plant.CalcForceElementsContribution(context, &f_app)` and include
`f_app.generalized_forces()` in `rhs`. If port's URDF has no spring
elements, this is a no-op (0 additional force). If it has any, this
brings the port in line with reference.

---

## 6 — Citations (per-claim file:line)

Reference:
- `c3/core/lcs.h:24-171` — LCS class
- `c3/core/lcs.cc:11-58` — LCS ctors + `ScaleComplementarityDynamics`
- `c3/multibody/lcs_factory.h:70-393` — LCSFactory declaration
- `c3/multibody/lcs_factory.cc:119-227` — ctors + `InitializeContactEvaluators`
- `c3/multibody/lcs_factory.cc:229-261` — `GetNClosestContactPairs`
- `c3/multibody/lcs_factory.cc:263-288` — `ComputeContactJacobian`
- `c3/multibody/lcs_factory.cc:290-299` — `UpdateStateAndInput`
- `c3/multibody/lcs_factory.cc:303-422` — `GenerateLCS` (main matrix build)
- `c3/multibody/lcs_factory.cc:423-436` — FormulateFrictionlessSpring
- `c3/multibody/lcs_factory.cc:438-494` — FormulateStewartTrinkle
- `c3/multibody/lcs_factory.cc:496-545` — FormulateAnitescu
- `c3/multibody/lcs_factory.cc:547-561` — `LinearizePlantToLCS`
- `c3/core/c3.cc:81, 204-232` — ScaleLCS + UpdateLCS
- `dairlib_sampling_c3/systems/controllers/sampling_based_c3_controller.cc:1582-1615` — GetResolvedContactPairs
- `.cc:1619-1698` — CreateLCSObjectsForSamples
- `.cc:800-848` — dt / planning_dt swap
- `examples/sampling_c3/franka_sampling_c3_controller.cc:124-239` — contact_pairs initialization for push_t
- `examples/sampling_c3/push_t/parameters/sampling_c3_options.yaml:2-51` — canonical parameters

Port:
- `control/lcs_formulator.py:43-209` — LCSFormulator __init__
- `control/lcs_formulator.py:212-228` — extract_dynamics
- `control/lcs_formulator.py:231-290` — extract_dynamics_with_jacobian (manual chain rule for M⁻¹)
- `control/lcs_formulator.py:375-457` — synthesize_manipuland_ground_contacts
- `control/lcs_formulator.py:481-850` — extract_lcs_contacts (2mm threshold + top-K + always-on)
- `control/lcs_formulator.py:1069-1322` — linearize_discrete (R^7 ST)
- `control/lcs_formulator.py:1234-1279` — R^7 E/F/H/c ST block
- `control/lcs_formulator.py:1367-2027` — linearize_discrete_ee_space (EE-space)
- `control/lcs_formulator.py:1649-1664` — EE-space A
- `control/lcs_formulator.py:1675-1680` — EE-space B_ctrl (point-mass EE)
- `control/lcs_formulator.py:1875-1960` — EE-space Anitescu overwrite (D/E/F/H/c)
- `control/lcs_formulator.py:1976-1991` — formulator-side ScaleLCS
- `control/admm_solver.py:1117-1124` — ADMM-side ScaleLCS (**duplicate**)
- `control/ci_mpc_c3plus.py:190-226` — planner dt swap and linearize_discrete call
- `main.py:610-649` — canonical (N, dt, dt_pose, use_ee_space) settings

---

## 7 — Notes for future readers

- The conformance-map (2026-07-14 baseline, refreshed 2026-07-25 for
  contact-model cluster) marked the LCSFactory-internal audit as
  **DEFERRED** on the grounds that the c3 lib clone was denied. That
  premise is now false — the c3 lib is fully accessible at
  `/root/reference_repos/c3/`. Future arc reports on subsystem 3 can
  and should treat `c3/multibody/lcs_factory.cc` as reference-source.
- The port's EE-space Anitescu path (`_contact_model = 'anitescu'` set
  in `__init__` at :86) is the runtime default. The R^7 ST path
  (`linearize_discrete_with_complementarity`) is walked only if
  `--ee-space` is NOT passed AND `_contact_model` stays at whatever it
  is; currently the R^7 path uses ST because the Anitescu overwrite
  only exists inside `linearize_discrete_ee_space`. If someone
  disables `--ee-space`, they silently switch from Anitescu to
  Stewart-Trinkle — an easy regression to miss.
- The port's `_synthesize_manipuland_ground_contacts` opt-in path
  (`lcs_explicit_manipuland_ground_contacts` env knob) is currently
  active by default for both box (4 vertices) and tshape (3 T-vertices).
  Reference relies on URDF-defined sphere witnesses (see
  `franka_sampling_c3_controller.cc:213-227` — `top_left_sphere_geoms`,
  `top_right_sphere_geoms`, `bottom_sphere_geoms` — these are
  URDF-declared sphere collision elements on `vertical_link`, indexed
  [1], [2], [3]). Port synthesizes these programmatically at
  `_tshape_vertex_set_body_frame(3)` (`lcs_formulator.py:345-373`); as
  documented in conformance-map §3.o the SIGNATURE matches (3 witness
  points), the underlying geometry does not. **NOT a delta identified
  under G-on**; a downstream concern about box-ground friction realism.
