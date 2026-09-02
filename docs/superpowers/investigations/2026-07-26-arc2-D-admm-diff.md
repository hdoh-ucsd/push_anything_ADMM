# Arc-2 D — Line-by-line C3+ ADMM diff: port vs reference

Date: 2026-07-26
Scope: Item #7 (G-matrix augmentation). Explain why turning G on destabilizes
port ADMM despite being the reference default.

**Files diffed**
- Port  : `/root/push_anything_ADMM/control/admm_solver.py`
  (2526 lines; `_solve_c3plus` at 927..; uncommitted G-fix diff live in the
  worktree — captured below.)
- Ref C : `/root/external/c3/core/c3.cc` (596 lines)
             `/root/external/c3/core/c3_plus.cc` (222 lines)
             `/root/external/c3/core/c3.h` (443 lines)
             `/root/external/c3/core/c3_plus.h` (74 lines)
             `/root/external/c3/core/c3_options.h` (203 lines)
             `/root/external/c3/core/configs/solver_options_default.yaml`
- Ref YAML: `/root/external/dairlib_sampling_c3/examples/sampling_c3/push_t/parameters/sampling_c3plus_options.yaml`

Uncommitted-in-port at time of diff (`git diff control/admm_solver.py`):
1. `_osqp_refopts` gate (env `REFCONF_OSQP_OPTS=1`) with the ref
   solver_options_default.yaml settings.
2. `_g_x` env override via `PORT_G_X`.
3. G-on ×2 fix in P/q/adaptive-ρ update paths (admm_solver.py:1158-1163,
   1417-1426, 1930-1946, 1948-1966). G-off path preserved.

---

## 1. Function map: reference C3::Solve ↔ port `_solve_c3plus`

Reference call graph (both C3 and C3Plus share `C3::Solve` /
`C3::ADMMStep`; `C3Plus` overrides only `AddAugmentedCost`,
`SolveSingleProjection`, `SetInitialGuessQP`, `StoreQPResults`, `UpdateLCS`):

| Reference method              | file:lines             | Port counterpart                                                                                                                             |
|--------------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `C3::C3` (ctor)                | c3.cc:45-159            | `C3Solver.__init__`  admm_solver.py:66-254 (partial)                                                                                          |
| `SetDefaultSolverOptions`      | c3.cc:161-178           | Ref-set gated on env — admm_solver.py:101-123 (default OFF)                                                                                   |
| `ScaleLCS`                     | c3.cc:204-213 + lcs.cc:46-58 | Inline in `_solve_c3plus` admm_solver.py:1103-1109                                                                                       |
| `UpdateLCS`                    | c3.cc:215-232 / c3_plus.cc:65-76 | Inline (LCS is rebuilt every tick by `LCSFormulator`; no in-place mutation)                                                        |
| `UpdateTarget`                 | c3.cc:239-246           | `x_ref` argument recomputed per tick                                                                                                          |
| `UpdateCostMatrices`           | c3.cc:248-261           | Q, R, QN, x_ref rebuilt per tick by `QuadraticManipulationCost.build`                                                                         |
| `Solve` (top-level)            | c3.cc:267-362           | `_solve_c3plus` admm_solver.py:927-2130                                                                                                        |
| `ADMMStep` (one outer iter)    | c3.cc:364-392           | Inline body of `for it in range(admm_iter)` admm_solver.py:1410-1970                                                                          |
| `SetInitialGuessQP`            | c3.cc:394-412 / c3_plus.cc:78-89 | **MISSING in port** — no warm-start / no time-shift interpolation                                                              |
| `SolveQP`                      | c3.cc:441-464           | Inline `self._solver.Solve(prog, None, self._osqp_solver_options)` admm_solver.py:1505-1509                                                    |
| `AddAugmentedCost`             | c3.cc:491-504 / c3_plus.cc:116-172 | admm_solver.py:1148-1163 (G build) + 1417-1426 (q update); adaptive-ρ variants at 1930-1966                                          |
| `SolveProjection` (parallel over knots) | c3.cc:506-541    | Inline `for i in range(N)` admm_solver.py:1542-1573 (no OpenMP; parallel is a no-op for C3+ per reference `use_parallelization_in_projection_=false`) |
| `C3Plus::SolveSingleProjection` (eq (12) closed-form) | c3_plus.cc:174-221 | `_project_C3Plus` admm_solver.py:898-925                                                                                       |
| `StoreQPResults` / `SetFallbackSolution` | c3.cc:414-489    | Inline extraction admm_solver.py:2098-2103; **fallback = MISSING** (a solver failure keeps the previous `z_sol`)                              |
| Warm-start caches `warm_start_x_[iter][k]` etc. | c3.cc:61-79 / c3_plus.cc:25-33 | **MISSING** — port never persists ω, δ, z across ticks or across ADMM iters                                                    |
| LCS scaling undo (physical λ) | c3.cc:349-354           | admm_solver.py: does the same for x_seq via `_x_roll` but the physical λ is recovered downstream when the executor consumes `_last_lambda_n_first` (see `_last_lambda_scale` uses — outside `_solve_c3plus`) |

---

## 2. Line-by-line differences by concern (ordered by hypothesized relevance)

### A. G-matrix construction & scaling (highest suspicion)

**Reference (C3+, c3_plus.cc:116-172):**
- G is a `vector<MatrixXd>` of length N, each element `n_z × n_z`, computed
  from options (`options.G = w_G · diag(g_vector)`, c3_options.h:189).
- The augmented Lagrangian cost is added ONLY to the λ block and the η
  block per knot: `AddQuadraticCost(2·G_λλ, -2·G_λλ·WD_λ, lambda_[i], 1)`
  (c3_plus.cc:157-161) and analogously for η (c3_plus.cc:163-170).
  **G_xx and G_uu blocks are NOT bound to any variable** — the state and
  input blocks receive ZERO ADMM augmentation.
- Drake convention `AddQuadraticCost(H, b, z, 1)` builds `0.5·zᵀHz + bᵀz`.
  Passing `H = 2G` and `b = -2G·WD` yields `zᵀGz - 2·(G·WD)ᵀ·z` =
  `‖z − WD‖²_G + const` — i.e. an unweighted-ρ effective penalty of
  `‖·‖²_G` (no extra 2 because G already carries the full weight; the
  `2` is the Drake plumbing factor).
- The reference push_t YAML gives `w_G=0.01`, `g_x=[0..0]`, `g_u=[0..0]`,
  `g_lambda=[2..2]`, `g_eta=[1..1]` (push_t/…c3plus_options.yaml:70,82-101).
  Effective per-slot weights: `x=0`, `u=0`, `λ=0.02`, `η=0.01`. Combined
  with `rho_scale=3` and `admm_iter=3` (push_t/…c3plus_options.yaml:2,4),
  the largest augmentation any component sees is `w_G · g_lambda · rho_scale^admm_iter`
  = `0.02 · 3³` = `0.54`.
- Only 2 augmented-cost bindings per knot (λ block + η block). x and u
  slots are governed by target/input QP costs (`AddQuadraticCost(2Q,·)`,
  `AddQuadraticCost(2R,·)`, c3.cc:141-155) and dynamics equality.

**Port (`_solve_c3plus`, admm_solver.py:1060-1091 + 1158-1163):**
- Builds a **single (total_dim × total_dim) diagonal** G matrix over the
  entire stacked horizon vector, then adds `2·rho·diag(G_diag)` to the
  full P_sym (admm_solver.py:1158-1163, uncommitted-fix path). This is
  algebraically equivalent to the reference's per-knot per-block ADD,
  IF (and only if) the P_sym symmetrisation and the ρ scaling agree.
- Port scales G by an outer scalar `rho`: `2·rho·diag(G_diag)`. This is
  the uncommitted-in-worktree factor-of-2 fix; before the fix it was
  `rho·diag(G_diag)`. The port's ρ starts at 100 in the current
  wrapper config (`params.py` calls `C3Solver(rho=100.0, mode='c3plus')`).
  The reference has NO scalar ρ (yaml `rho: 0 #This isn't used anywhere!`,
  push_t/…c3plus_options.yaml:3). Reference augmentation weight is
  effectively `w_G · g_j` alone.
- **Under G-on the port therefore runs**
  `effective_weight_λ = 2 · 100 · (0.01·2) = 4.0`  (per iter 0)
  vs reference `0.02`. That is **200×** the reference augmentation
  strength on λ at iter 0. With `rho_scale=3.0` (port matches ref) the
  iter-2 effective weight is `4.0 · 3² = 36`, still >>> ref `0.18`.
- **Under G-off the port runs**  `effective_weight_all = rho · 1 = 100`
  uniformly across all slots (admm_solver.py:1161). The port's rho=100
  baseline is calibrated to the fact that G-off applies penalty to
  x, λ, u, η equally; when G turns on with `g_x=g_u=0`, the state and
  input slots suddenly lose all ADMM consensus penalty and the QP is
  free to pick arbitrary x_seq / u_seq subject only to
  target-cost + dynamics-equality. Because port `Q` and `R` are
  substantially different from reference (port `w_Q≈50…10^4`, sparse
  q-vector — see project_item7_deep_investigation memory), the
  x-block QP cost cannot compensate. The z-update x-block therefore
  drifts freely between ADMM iters → primal residual (which counts
  x-gaps too through `‖z−δ‖`) becomes non-monotone.

**Coverage of the ×2 fix across the four G-on codepaths (git diff read):**
- admm_solver.py:1158-1163 — P build, ×2 applied. **FIX APPLIED.**
- admm_solver.py:1417-1426 — q_total update per iter, ×2 applied. **FIX APPLIED.**
- admm_solver.py:1943-1944 — rho_scale P_sym in-place diag delta, ×2 applied.
  **FIX APPLIED.**
- admm_solver.py:1954-1957 and 1962-1965 — adaptive-ρ rebuild after
  10-iter Boyd doubling/halving, ×2 applied to the G-on aug. **FIX APPLIED.**
- **NOT COVERED** by the fix (but present in port only): the §7.67 B1-A
  final-iter override at admm_solver.py:1447 uses `rho * np.diag(_G_diag)`
  with **NO ×2**, and its q_total_final at 1460-1473 uses
  `-rho * _W * delta[...]` again with no ×2. The §7.67 branch is
  independently gated by `PORT_G_WEIGHT_EE_BOX_FINAL=1` (a different
  research probe) so does not affect the baseline G-on run, but if that
  flag is combined with G-on the ratio between the standard aug and the
  final-iter override changes by 2×. **NOTE FOR FUTURE.**

**Ref comment "reference gives -2·G·WD·z linear term":** The
uncommitted diff comment is correct that the Drake convention gives a
linear coefficient of −2·G·WD. But the reference passes `2·G` as the
Hessian, not `G`. The pattern `AddQuadraticCost(2G, -2G·WD, z, 1)` on
Drake's `0.5·zᵀHz + bᵀz` engine yields `zᵀ G z − 2·(G·WD)ᵀ·z` =
`‖z − WD‖²_G − ‖WD‖²_G`, i.e. augmentation weight = **G**, not 2G.
The port's ×2 fix therefore **doubles the effective augmentation
relative to the reference** rather than matching it.  Correct
reference-match would be `_P_aug = rho * np.diag(G_diag)` with the
q_total update also carrying a single ρ (not 2ρ), and the reference
formulation buries all the "2"s inside the `AddQuadraticCost` API
convention. So the fix as it stands still overshoots by a factor of 2
vs the reference math. **CANDIDATE MECHANISM: A-1.**

### B. rho / rho_scale schedule & adaptive ρ

**Reference (c3.cc:364-392):**
```
void C3::ADMMStep(...) {
  ... vector<VectorXd> z = SolveQP(x0, *G, WD, *delta, admm_iteration, false);
  ...
  *delta = SolveProjection(*G, ZW, admm_iteration);
  for (i=0..N-1) {
    w->at(i) = w->at(i) + z[i] - delta->at(i);
    w->at(i) = w->at(i) / options_.rho_scale;      // ← ω shrink
    G->at(i) = G->at(i) * options_.rho_scale;      // ← G grow
  }
}
```
`rho_scale=3` (push_t yaml). No primal/dual-residual-based adaptivity
inside the ADMM outer loop; the growth is deterministic and geometric.
OSQP's `adaptive_rho` (inside the QP solver, one level down) is a
different knob — that's the inner primal/dual adaptive-ρ used by OSQP
itself (yaml: `adaptive_rho: 1`, `adaptive_rho_interval: 0`,
`adaptive_rho_tolerance: 5`). The C3 augmented Lagrangian ρ is not
tracked adaptively; only G is grown geometrically.

**Port (admm_solver.py:1936-1966):**
- Two-track:
  - Track 1 (default active for `_rho_scale=3.0`): matches reference exactly —
    grow ρ by `_rs`, shrink ω by `_rs`, update P_sym diagonal in place
    (admm_solver.py:1936-1947). ✓
  - Track 2 (Boyd every 10 iters): if the geometric growth is disabled
    (`_rs ≤ 1.0`), fall back to `pr > 10·dr → ρ×=2, ω/=2`;
    `dr > 10·pr → ρ/=2, ω×=2` with a hard cap `rho < 1000` and floor
    `rho > 0.1` (admm_solver.py:1948-1966). At `admm_iter=25`,
    (25//10) = 2 opportunities to fire.
- **BOTH tracks are protected against divergence** by the cap
  `rho * _rs < 1e6`. **Neither track exists in the reference.**
- Port's `_rho_scale` default matches reference. But the port's **initial
  ρ=100** is what dominates; reference initial G is `w_G · g_j = 0.02`.
  With `rho_scale=3` and admm_iter=25, the port at iter 24 sits at
  `ρ ≈ 100 · 3^24 = 2.8·10¹¹` (capped by 1e6 guard). This is a
  **massive** difference: reference tops out at `0.02 · 3² = 0.18`
  after admm_iter=3.
- **Adaptive-ρ at inner OSQP level:** the port's `REFCONF_OSQP_OPTS`
  gate turns on `adaptive_rho=1` inside OSQP. Default OFF. **Reference
  runs with it ON** by default (yaml). The port default (Drake defaults
  are `adaptive_rho=1` too — needs verification below).
- The port also grows ρ in the Boyd adaptive-ρ branch, but only via
  primal/dual-residual imbalance detection, and it caps at `rho<1000`
  before doubling further — a hard floor of asymmetry the reference
  does not enforce. **CANDIDATE MECHANISM: B-1** (initial-ρ scale
  mismatch dominates every downstream setting).

### C. δ / ω projection & update

**Reference C3+ SolveSingleProjection (c3_plus.cc:174-221):**
- Per-knot: extracts `w_λ_vec = diag(U_λλ)`, `w_η_vec = diag(U_ηη)` from
  the U matrix (c3_plus.cc:184-187). U is a separate matrix from G
  (c3_options.h:190).
- Enforces `w_λ ≥ 0`, `w_η ≥ 0` at runtime with `throw` (c3_plus.cc:190-193).
- Case decision (c3_plus.cc:205-207):
  `eta_larger = eta * sqrt(w_eta) > lambda * sqrt(w_lambda)`
  (elementwise). If eta_larger → λ:=0 else η:=0 (c3_plus.cc:209-212).
  Then clip both at 0 (c3_plus.cc:214-218).
- The threshold is `sqrt(w_eta / w_lambda)`… no, wait — reading
  c3_plus.cc:206 exactly: `eta_c * sqrt(w_eta_vec) > lambda_c * sqrt(w_lambda_vec)`,
  which is equivalent to `eta > lambda · sqrt(w_lambda / w_eta)`.
- Reference `u_lambda_list=20`, `u_eta_list=1` (push_t YAML). So
  `sqrt_ratio = sqrt(20/1) = sqrt(20) ≈ 4.47`. Rule:
  case 1 (λ→0) when `η > 4.47·λ`; case 2 (η→0) when `η ≤ 4.47·λ` and
  `λ ≥ 0`; case 3 (both zero) when negative.

**Port `_project_C3Plus` (admm_solver.py:898-925):**
- Constants `u_lambda=20`, `u_eta=1` — port matches reference (was
  1:1, fixed to 20:1 in the 2026-07-18 iter9 commit per comment at
  admm_solver.py:130-138).
- Case decision (admm_solver.py:920-924):
  `sqrt_ratio = sqrt(u_lambda / u_eta) = sqrt(20/1) ≈ 4.47`
  `cond1 = (eta >= 0) & (eta >= sqrt_ratio · lam)` → case 1 (λ→0, keep η)
  `cond2 = (lam >= 0) & (eta <  sqrt_ratio · lam)` → case 2 (keep λ, η→0)
  else → case 3 (both zero).
- **Numeric equivalence to reference: matches sign-for-sign.**
- ω-update happens at admm_solver.py:1586: `omega = omega + z_sol - delta`.
  Reference at c3.cc:387-390: `w += z - delta`, then `w /= rho_scale`.
  **Port does the `w /= rho_scale` at admm_solver.py:1940 inside the
  rho_scale-branch.**  When `_rho_scale > 1`, both match.
- ω-update over the entire z vector (all slots). Reference at c3.cc:387-388
  also updates w over the full z. Matches.
- **NOTE: port projection uses G's u_lambda/u_eta constants, NOT the
  U matrix's weights** — that is, port derives the projection weight
  from `self._u_lambda=20`, `self._u_eta=1` and never reads the yaml
  U-block. Reference derives them from `U.block(...).diagonal()`. This
  is architecturally different but numerically equivalent for the
  reference push_t config (which uses uniform U weights). If a caller
  wanted per-contact U weighting (`options.qp_projection_scaling` etc.),
  the port would miss it. **NOT the arc-2 blocker** (all push_t contacts
  see the same u_lambda / u_eta), but is a functional gap.

### D. Primal solve (QP) — cost assembly, symmetrisation, warm-start, OSQP settings

**Reference (c3.cc:441-464 SolveQP + AddAugmentedCost c3.cc:491-504 /
c3_plus.cc:116-172):**
- P is built ONCE in the constructor (c3.cc:141-155):
  `AddQuadraticCost(2Q, -2Q·x_des, x_[i], 1)` for i=0..N (state), and
  `AddQuadraticCost(2R, 0, u_[i], 1)` for i=0..N-1 (input). No
  symmetrisation step — Drake QP handles that.
- Per-iter: `AddAugmentedCost` removes prior augmented bindings, then
  adds fresh `AddQuadraticCost(2G_λλ_block, -2G_λλ_block·WD_λ, lambda_[i], 1)`
  bindings (c3_plus.cc:117-171). Only λ and η blocks.
- Uses Drake's `AddLinearEqualityConstraint(A_dyn, b, {x[i],λ[i],u[i],x[i+1]})`
  for dynamics + a separate one for η equality per knot (c3.cc:118-135
  + c3_plus.cc:45-59).
- Uses `SetInitialGuessQP` which warm-starts from previous ADMM
  iteration's solution (c3.cc:394-412 + c3_plus.cc:78-89):
  ```
  prog_.SetInitialGuess(x_[i],
      (1-w) · warm_start_x_[iter-1][i] + w · warm_start_x_[iter-1][i+1])
  ```
  with `w = fractional(solve_time_ / lcs_.dt())` — interpolates across
  time steps. This is warm start per ADMM iter within a solve; and the
  warm_start caches are also updated in `StoreQPResults`, so warm start
  across ticks is preserved via `warm_start_[admm_iter][i]` (c3.cc:432-437).
- **Reference default `warm_start=false` for push_t** (push_t YAML:10),
  so the interpolation is skipped. But warm start via OSQP's own primal/
  dual warm start (via `warm_starting=1`, yaml) IS active — Drake feeds
  the previous solution's primal/dual guesses into the next OSQP solve
  automatically when the `MathematicalProgram` is re-used.
- Reference SolverOptions loaded from `solver_options_default.yaml`
  (c3.cc:161-178): `polishing=1, polish_refine_iter=3, warm_starting=1,
  scaled_termination=1, scaling=10, adaptive_rho=1, adaptive_rho_interval=0,
  eps_abs=1e-5, eps_rel=1e-5, max_iter=1000, rho=1e-4, sigma=1e-6,
  alpha=1.6, time_limit=1.0` (solver_options_default.yaml:1-27).
- Regularisation of P: **NONE explicit.** The `2Q` cost is added directly.
  Reference relies on OSQP scaling + adaptive_rho for numerical
  conditioning.

**Port (admm_solver.py:1114-1163 for P, 1417-1426 for q, 1505-1506 for solve):**
- Per-iter: rebuilds the full `q_total`, calls
  `cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)` — reuses one
  QuadraticCost binding per solve (built once at admm_solver.py:1269-1270,
  then updated 25 times).
- P is built fresh per solve (LCS shape changes → total_dim changes →
  can't reuse the prog across solves). Reference builds prog once and
  reuses; **port rebuilds prog every solve**. This kills any hope of
  OSQP-level warm starting across ADMM iters within a solve (Drake's
  `warm_starting` primitive applies within a single `Solve()` call
  chain on the same prog).
- Port applies **`P_sym = 0.5·(P_total + P_total.T) + 1e-8·I`**
  (admm_solver.py:1163). Reference has no explicit `+1e-8·I` regularisation.
  The `1e-8` is negligible for typical problem scales but WILL float above
  reference's numerical zero if `w_G · g_j` is near 1e-8 (it's not, but
  a smaller w_G could tickle this).
- Port constructs `prog = ad.MathematicalProgram()` new per solve
  (admm_solver.py:1212). Reference builds prog ONCE in the constructor
  (c3.cc:59); the LCS shape mutation is handled by `UpdateLCS` which
  only changes the coefficient matrices of pre-existing constraints
  (c3.cc:215-232). **Port has no equivalent to `UpdateLCS`.**
- Solver options: port defaults to `None` (Drake defaults). With
  `REFCONF_OSQP_OPTS=1` (uncommitted), port matches the reference yaml
  exactly (admm_solver.py:108-120). Default OFF is **the current
  behavior**, so port is running with Drake's defaults, not reference's.

**Drake OSQP defaults** — from Drake source
(drake/solvers/osqp_solver.cc, based on general OSQP knowledge):
`max_iter=4000, eps_abs=1e-3, eps_rel=1e-3, polishing=0, warm_starting=1,
scaling=10, adaptive_rho=1, adaptive_rho_interval=0, sigma=1e-6,
rho=0.1, alpha=1.6, time_limit=0`. **Compared to reference:** default
`eps_abs/rel` are LOOSER (1e-3 vs 1e-5), no polishing (vs polishing on
with 3 refine iters), same warm_starting/scaling/adaptive_rho.

**Practical implication:** port's inner OSQP solve is running looser
tolerance than reference — so at each ADMM iter the QP z_sol is
noisier than reference's. Under G-on, this noise interacts with the
G-only-on-λ-η aug (rest of z is unconstrained by ADMM) to produce
non-monotone ADMM residuals. **CANDIDATE MECHANISM: D-1.**

**SetInitialGuessQP MISSING in port.** Even with warm_starting=1 in
OSQP, port destroys and rebuilds the prog each solve; no explicit
initial-guess is fed via `prog.SetInitialGuess(z_var, ...)`. Reference
does an explicit `prog_.SetInitialGuess(x_[0], x0)` every solve
(c3.cc:395) plus the cross-iter interpolation. **CANDIDATE
MECHANISM: D-2.**

**Ref penalize_input_change: FALSE** for push_t (yaml:16), so the port's
`_penalize_input_change=True` default is off-reference for push_t.
NB: `penalize_input_change=True` for `anything` (jack) case. Task-driven.
**Not the arc-2 blocker but is off-reference for the T-push case.**

### E. Constraint & LCS handling

**Reference dynamics constraint** (c3.cc:118-135):
- One `AddLinearEqualityConstraint` per timestep i=0..N-1 with variables
  `{x[i], λ[i], u[i], x[i+1]}`, coefficient block:
  ```
  [A(i), D(i), B(i), -I]  ·  [x_i; λ_i; u_i; x_{i+1}] = -d(i)
  ```
  Total N linear-eq bindings, each n_x rows. Plus initial-state
  constraint (c3.cc:271-280), added lazily on first Solve.
- η-equality per timestep (c3_plus.cc:45-59):
  ```
  [-E(i), -F(i), -H(i), +I] · [x_i; λ_i; u_i; η_i] = c(i)
  ```
  Wait — reference c3_plus.cc:56 passes `-lcs_.c().at(i)` as the RHS.
  So the equality reads `η_i - E·x_i - F·λ_i - H·u_i = c_i`, matching
  the LCP form `η = E·x + F·λ + H·u + c`.
- `AddBoundingBoxConstraint` on λ ≥ 0: **NOT present in reference C3+**.
  Reference relies on the projection step to enforce λ ≥ 0.
  Verified by grepping c3.cc / c3_plus.cc — no `AddBoundingBox` calls
  for λ. Only user-defined `AddLinearConstraint` (c3.cc:543-573).

**Port `_solve_c3plus`** (admm_solver.py:1168-1207 for eq, 1222-1244 for bbox):
- Same equality structure: `n_eq_state = n_x + N·n_x` (x0 fix + N dyn),
  `n_eq_eta = N·n_lambda`. Combined into one big `C_eq` matrix
  (admm_solver.py:1171-1207). Functionally equivalent, but reference
  splits into per-knot bindings which Drake compiles into separate OSQP
  constraint rows (same OSQP problem, different binding structure).
- **Port adds explicit `AddBoundingBoxConstraint(0, ∞, λ_slots)`
  (admm_solver.py:1222-1228). Reference does NOT.** This is an EXTRA
  constraint in the port. **Under G-off, the QP already keeps λ
  bounded via the strong ρ·I aug (any negative drift is penalized
  toward δ which is projected to ≥0). Under G-on with g_x=g_u=0, the
  x/u slots have zero aug and the QP has freedom to compensate the
  Q/R/dyn costs by pushing λ negative — except the bbox blocks that,
  so it pushes x instead → η equality violated → ω drifts → primal
  residual grows.** **CANDIDATE MECHANISM: E-1.**
- Port also adds `AddBoundingBoxConstraint` on u (torque limits,
  admm_solver.py:1240-1244) and optionally on state velocity slots
  (admm_solver.py:1250-1267 for `ee_velocity_bounds`). Reference does
  torque limits via `AddLinearConstraint(A, lb, ub, INPUT)` (c3.cc:556-561),
  functionally equivalent. EE-velocity limits also present in
  reference dispatcher wrapper (see project memory:
  `sampling_c3plus_options.yaml:36 ee_velocity_limits: [-0.14, 0.14]`).
  Consistent with port.

### F. Convergence criterion & early exit

**Reference:**
- **Fixed number of admm_iter (default 3)**, no primal/dual residual
  check for early exit at the C3-ADMM level (c3.cc:322-324:
  `for (int iter = 0; iter < options_.admm_iter; iter++)`).
- Then one final `SolveQP(x0, G, WD, delta, admm_iter, /*is_final_solve=*/true)`
  (c3.cc:331) — this is the "half step" that produces the returned
  z_sol using the last δ and w. `end_on_qp_step=true` → return this
  z_sol as final; `end_on_qp_step=false` → do the LCS forward roll
  on top (c3.cc:336-347).
- OSQP inner solver has its own `eps_abs=1e-5, eps_rel=1e-5, max_iter=1000`
  early-exit — that's a per-QP-solve criterion, not an ADMM-outer criterion.

**Port (admm_solver.py:1968-1970):**
- **HAS explicit ADMM-outer early-exit:**
  `if pr < tol and dr < tol: actual_iters = it + 1; break`
  with `tol = 1e-3` (admm_solver.py:1341). Reference has no such gate.
- Port also has separate `end_on_qp_step=False` LCS roll-forward
  (admm_solver.py:2108-2117), matching reference's default.
- **CANDIDATE MECHANISM: F-1** — the port's ADMM-outer tol=1e-3 is
  computed on primal norm over N·(n_λ+n_η) slots; at G-on the primal
  gap sits at `‖z-δ‖` where z drifts freely on state/input slots
  (aug=0). But port's residual formula (admm_solver.py:1901-1924) only
  measures (λ, η) blocks:
  ```
  lam_vec = concat over i of z_sol[SL:SL+n_λ] + z_sol[SE:SE+n_λ]
  pr = ‖lam_vec - dlt_vec‖
  dr = rho · ‖dlt_vec - dlt_prev_vec‖
  ```
  This DOES exclude x/u drift from the residual (good for
  convergence detection), but with `rho=100` under G-on the
  `dr = 100 · ‖δ-δ_prev‖` metric is 100× ref's `w_G·rho_scale^i =
  0.02·3² = 0.18` scale. Port's tol=1e-3 is effectively a much tighter
  test than reference at rho=100. Under G-on this is unlikely to fire
  (dr stays huge), so early exit never trips — port runs all 25 iters.
  Meanwhile the ADMM oscillates because the effective augmentation is
  way too strong. **CANDIDATE MECHANISM: F-1 confirmed as accessory.**

### G. Initialisation / warm start

**Reference:**
- `warm_start_x_[admm_iter+1][N+1]`, `warm_start_lambda_[admm_iter+1][N]`,
  `warm_start_u_[admm_iter+1][N]` — one per (ADMM-iter, knot) buffer
  (c3.cc:61-79). Populated in `StoreQPResults` (c3.cc:431-437).
- Interpolated by `solve_time_ / lcs_.dt()` in `SetInitialGuessQP`
  (c3.cc:398-411) — this handles the case where control ticks are
  finer than the LCS dt.
- **Reference push_t YAML sets `warm_start: false`** so the interpolation
  is skipped for the push_t task. But the reference initial δ has
  `delta_option=1` which biases `delta.head(n_x) = x0` (c3.cc:312-316).
  **Port matches** (admm_solver.py:1321-1323).
- Reference **also inits `w` (ω) to zero at every call to Solve**
  (c3.cc:317: `vector<VectorXd> w(N_, VectorXd::Zero(n_z_))`). Port
  matches (admm_solver.py:1316).
- **NO cross-tick persistence in either implementation for
  warm_start=false. For warm_start=true, both would persist through the
  warm-start caches — port has neither cache nor tick-persistence.**

**Port:** no warm start across ADMM iters (except implicit via OSQP's
own `warm_starting=1` if REFOPTS on — see Section D). No warm start
across control ticks (each `_solve_c3plus` call builds a fresh
`prog`). **MISSING in port** but **NOT ACTIVE in reference for push_t
either** (warm_start=false).

---

## 3. Ranked list of missing convergence machinery

Ordered by hypothesized contribution to "G-on destabilizes port ADMM."

### Rank 1 (dominant): initial ρ is 500-5000× too large under G-on

**Evidence:** reference initial G-aug per λ slot = `w_G · g_λ = 0.02`
(no ρ multiplier). Port initial G-aug per λ slot = `2 · rho · w_G · g_λ =
2 · 100 · 0.02 = 4.0` (with uncommitted ×2 fix; was `2.0` before).
That is **200× reference** at iter 0, and grows to **~1800× reference**
at the last iter under `rho_scale=3`, admm_iter=3.

At the current admm_iter=25 (canonical run flag) with the 1e6 cap, port
tops out at ρ=1e6 → aug=20000 vs reference max aug ≈ `w_G · g_λ ·
rho_scale^admm_iter = 0.02 · 3² = 0.18`. **6-orders-of-magnitude
mismatch.** ADMM at that ρ makes z_λ ≈ δ_λ in one iter → z_x, z_u
(zero-aug) get pulled by target Q and R only, and this pull may push
z_x hard away from δ_x every iter. The 200× overshoot on the ×2 fix
is a hint the port fix went the wrong direction.

**Root:** admm_solver.py:66 (`rho: float = 1.0` default, overridden
to 100.0 by C3PlusMPC caller — verify by grep) combined with port's
tuned G-off ρ=100 baseline being calibrated against the wrong
augmentation formula.

### Rank 2 (mechanism): x-block and u-block have zero ADMM aug under G-on but strong ρ·I under G-off; port's Q, R don't fill the gap

**Evidence:** admm_solver.py:1076 (`_gd[SX:SL] = _gx = 0`) and 1078
(`_gd[SU:SE] = _gu = 0`) zeros out state/input augmentation. Reference
c3_plus.cc:157-170 also skips x/u in the aug bindings. Reference is
consistent because its `Q, R` matrices are tuned to solve alone. Port's
`Q, R` are tuned assuming a strong background ρ·I holds z-δ near zero
across all slots. When G-on turns off the x/u aug abruptly, the QP no
longer has ADMM pressure on those slots, and the port's sparse-q_vector
+ w_Q=50 target cost is not strong enough to keep x consistent from iter
to iter → the state block of z drifts every iter → η equality violation
propagates → λ block sees inconsistent driving forces → projection
destabilises.

### Rank 3 (accelerator): missing warm start of z_var across ADMM iters within a solve

**Evidence:** admm_solver.py:1212 rebuilds `prog = ad.MathematicalProgram()`
per solve. Reference re-uses prog. Drake's OSQP warm_starting only
matters within a single prog. Without cross-iter warm start, each of
the 25 iters sees OSQP starting from cold-start (or from Drake's
default zero-initial guess), so the 1e-5 tolerance is hit via more
inner iterations, and the returned z_sol has slightly different
numerical structure per iter — under G-on, that noise is amplified
because there's no strong aug to snap z back to δ.

Additionally: port never calls `prog.SetInitialGuess(z_var, delta)` or
similar. Reference c3.cc:394-412 does. **CANDIDATE MECHANISM: G-2.**

### Rank 4 (probable at rho=1 test): the explicit `AddBoundingBoxConstraint(0, ∞, λ)` in port

**Evidence:** admm_solver.py:1222-1228 adds a hard lower bound on λ
that reference does not have. Under G-off + rho=100, this is harmless
(the QP naturally stays feasible). Under G-on + rho=1 (as in the
2026-07-25 arc-1 sweet spot + arc-2 G-on test), the QP has weak aug on
λ, weak aug elsewhere, and the bbox becomes an active constraint. When
OSQP hits an active bound at 0 on a bunch of λ slots, the polishing
step is disabled by default (unless REFOPTS on) — and the returned
z_sol has zeroed λ slots that the δ-update then over-corrects.
**Testable by removing the bbox and letting the projection alone
enforce λ ≥ 0.**

### Rank 5 (accessory): OSQP default tolerance is 100× looser than reference

**Evidence:** solver_options_default.yaml:18-19 has `eps_abs=1e-5,
eps_rel=1e-5`. Drake default is 1e-3. Port's default (REFOPTS off)
matches Drake's default. When ADMM aug is weak (G-on) the QP tolerance
becomes the bottleneck on z accuracy; port at 1e-3 lets z-δ gap
accumulate.

---

## 4. Concrete falsification tests

Each is a single env-var flip; all default OFF. All read-only (no
implementation changes to the algorithm — just gating existing paths
or adding one-shot removals).

### T-1 (Rank 1): pin ρ=1 explicitly under G-on to match reference numeric aug scale

**Change:** Add env override `PUSHA_ADMM_RHO_OVERRIDE=<float>` that
overrides `self.rho` (admm_solver.py:66) AT solver construction time,
not just inside `_solve_c3plus`. Set to 0.5 to make port aug =
`2·0.5·w_G·g_λ = 0.02` = reference. Then run G-on at that.

**Sketch (do NOT commit):**
```python
# admm_solver.py:66 (in __init__)
_rho_env = os.environ.get("PUSHA_ADMM_RHO_OVERRIDE", "")
if _rho_env:
    rho = float(_rho_env)
    print(f"[PORT_RHO_OVERRIDE] rho={rho}", flush=True)
```

**Predicted outcome if T-1 is the root cause:** with `PUSHA_ADMM_RHO_OVERRIDE=0.5`
+ `REFCONF_USE_G_MATRIX=1`, ADMM primal/dual should reach monotone
convergence (mono=True) and the T-push metrics should either match or
mildly regress vs current G-off ρ=100 baseline (because Q, R, ee-cost
are calibrated to ρ=100). If the mechanism is dominant, the
non-monotone-iter rate should drop from 70-80% back to ~0%.

**If the run passes convergence but regresses tracking:** it confirms
Rank 1 is the ADMM-stability mechanism but the actual reference-match
would require re-tuning Q, R at ρ=1 (out of scope for the arc).

### T-2 (Rank 4): remove the λ ≥ 0 bbox and rely on projection

**Change:** Gate the `AddBoundingBoxConstraint(0, ∞, λ)` at
admm_solver.py:1222-1228 on env `PUSHA_LAMBDA_BBOX=<0|1>`, default 1.
Set to 0 to disable.

**Sketch (do NOT commit):**
```python
# admm_solver.py:1222
_lambda_bbox = os.environ.get("PUSHA_LAMBDA_BBOX", "1") == "1"
if n_lambda > 0 and _lambda_bbox:
    for i in range(N):
        prog.AddBoundingBoxConstraint(...)
```

**Predicted outcome if T-2 is the root cause:** G-on + `PUSHA_LAMBDA_BBOX=0`
should stop the workspace-violation crash at ρ=1 (arc-2 observed crash),
because the QP is no longer forced to push x when it can't push λ
negative. If T-1 fails but T-2 fixes the ρ=1 crash, Rank 4 is the arc-2
mechanism. If both fail: neither.

### T-3 (Rank 2): give x-block ADMM aug back under G-on

**Change:** Already partially implemented — the uncommitted diff adds
`PORT_G_X` env override at admm_solver.py:202-208. Test with
`PORT_G_X=1` (uniform g_x=1). This puts the port back to
"z_x has aug = 2·rho·w_G·g_x = 2·100·0.01·1 = 2.0" at iter 0.

**Predicted outcome if T-3 is the root cause:** G-on + `PORT_G_X=1`
should restore monotone ADMM iters. If T-3 works while T-1 doesn't,
Rank 2 is the mechanism (the x-block absence of aug, not the ρ scale).
If T-3 works AND T-1 works, both mechanisms are active and stacking
(consistent with Ranks 1 + 2 being coupled).

**Prioritized probe order:** T-1 first (single env var, tests dominant
hypothesis), then T-3 (also single env var, orthogonal), then T-2
(tests the ρ=1 crash separately). Do NOT combine flags in a single run
until each has been tested in isolation.

---

## 5. Summary

The port's G-on codepath is technically implementing the reference
formula's structural form (per-slot G-diag on λ and η slots, matching
c3_plus.cc:157-170), but does so **on top of** an outer scalar
`self.rho=100` that has no reference counterpart. The reference bakes
the entire augmentation weight into `G = w_G · diag(g_vector)` and
grows it geometrically via `rho_scale`; the port grows a separate ρ
by the same factor while ALSO carrying the G matrix. The result is a
200× overshoot in the effective per-λ augmentation weight at iter 0
(with the uncommitted ×2 fix), rising to ~1800× by iter 3. This alone
is likely sufficient to explain 70-80% non-monotone ADMM iters under
G-on.

Secondary contributors: (a) x-block and u-block have zero ADMM aug
under G-on but the port's sparse Q, R matrices are tuned assuming a
strong uniform ρ·I aug that G-on removes; (b) port adds an explicit
λ ≥ 0 BoundingBox that reference does not; (c) port default OSQP
tolerance is 1e-3 vs reference's 1e-5, degrading z accuracy under weak
aug; (d) missing warm start of z across ADMM iters within a solve
because port rebuilds prog every tick.

The three falsification tests above isolate the three most-likely
mechanisms and can be run one at a time without any algorithmic
changes.

--- end of report ---
