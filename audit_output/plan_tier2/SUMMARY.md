# PLANNER / ADMM / MODE-SWITCH TIER-2 EVIDENCE (2026-07-14)

Diagnostic commit: `67232d7` — `PUSHA_PLAN_T2_DIAG=1`, env-gated, default-OFF byte-identical.

## Runs
- `run_default.log`: `PUSHA_PLAN_T2_DIAG=1`, pushing --task-id 4 seed 0 admm-iter 25 c3plus --sampling-c3 config/sampling_c3_kik.yaml --max-time 0.5.

## Reference-side deep reads (Tier-2 (a), c3 lib + dairlib repos)

### C3 solver family (c3 lib `core/`)
- `c3.cc:267-362 C3::Solve(x0)`: main entry. Steps: (1) update initial-state constraint. (2) if `h_is_zero_` (passive system), pre-solve LCP via `MobyLcpSolver::SolveLcpLemke` on `F[0]`. (3) init `delta` (`delta_option=1` → head=x0; else 0) and `w = 0`. (4) `for iter in [0, admm_iter): ADMMStep(x0, &delta, &w, &G, iter)`. (5) final `SolveQP(x0, G, WD, delta, admm_iter, true)`. (6) if not `end_on_qp_step`, compute "half step" via rollout: `z_sol[i] = A·x_sol + B·u_sol + D·λ_sol + d`. (7) scale `lambda_sol *= AnDn_`.
- `c3.cc:364-392 ADMMStep`: (a) `WD = delta - w`. (b) `z = SolveQP(x0, G, WD, delta, admm_iter, false)`. (c) `ZW = w + z`. (d) `delta = SolveProjection(G or U, ZW, admm_iter)`. (e) `w = (w + z - delta) / rho_scale; G = G * rho_scale`.
- `c3.cc:394-412 SetInitialGuessQP`: within-Solve warm-start ACROSS ADMM iterations. Uses `warm_start_x_[admm_iter-1]` interpolated by `solve_time_ / lcs_.dt()` fraction. Only fires for `admm_iteration > 0` AND `warm_start_ = true`.
- `c3_plus.cc:174-221 C3Plus::SolveSingleProjection`: **exactly** Bui 2026 eq (12) componentwise projection:
  ```cpp
  eta_larger = eta * sqrt(w_eta) > lambda * sqrt(w_lambda);
  delta.λ = eta_larger.select(0, lambda_c);   // case 2: λ wins
  delta.η = eta_larger.select(eta_c, 0);      // case 1: η wins
  delta.λ = delta.λ.cwiseMax(0);              // clip to ≥ 0
  delta.η = delta.η.cwiseMax(0);
  ```
  Case 3 (both zero) implicit via the ≥0 clip when both pre-clip values were negative.
- `c3_miqp.cc`, `c3_qp.cc`: alternate projection classes. `MIQP` is the reference default per `anything/push_t` YAML `projection_type: 'MIQP'`.

### YAML values (reference `anything` config)
```
admm_iter: 3
rho_scale: 3
delta_option: 1
projection_type: 'MIQP'
warm_start: false
end_on_qp_step: false
N: 5
planning_dt_position: 0.1
```
Reference: horizon 5 × 0.1s = **0.5s**, **3 ADMM iters**, MIQP projection, cold-start cross-tick.

### Mode-switch (dairlib `sampling_based_c3_controller.cc:1140-1320`)
- 8 branches: (c3→ target-met=achieved_fixed_goal), (c3→ not-progressing=kToReposUnproductive), (c3→ cost-gap=kToReposCost), (free→ xbox=kToC3Xbox), (free→ target-met=stay-in-repos), (free→ cost-gap AND altitude-gate=kToC3ReachedReposTarget OR kToC3Cost), (free→ better-repos=kToBetterRepos or kStayInRepos), (free→ collision=kNewSample).
- **Cost-gap into c3** is AND-gated by `x_lcs_curr[2] < z_height + c3_min_clearance + wall_offset OR NOT ee_z_close` — an ALTITUDE gate: only switch to c3 if EE is near contact height OR the ee_z_close feature is off.
- **In-repos re-target** uses `hyst_repos_to_repos` (absolute) or `hyst_repos_to_repos_frac` (relative).
- **kToC3ReachedReposTarget vs kToC3Cost**: distinguished by `repos_target_cost > progress_params_.finished_reposition_cost` — the "declared finished" one gets the ReachedReposTarget reason; the cost-gap one gets kToC3Cost. Both fire only after the altitude gate passes.
- Port `mode_switch.py:decide_mode` mirrors this structure but omits: `pursued_target_source_`, `achieved_fixed_goal_`, `AddToUnsuccessfulBuffer`, `force_c3_mode` (xbox override), `wall_offset`. Adds: `kForceC3Watchdog`, `ee_z_gate_pass` explicit arg (T1a port of the reference altitude gate).

## Runtime captures (port default, `PUSHA_PLAN_T2_DIAG=1`)

```
[C3] Solver mode: c3plus (planner: R^7 joint torque, c3+ projection: componentwise)
[C3+] planner construction verified: use_ee_space=False solver.n_x=27 solver.n_u=7
[PLAN-T2] planner_class=C3PlusMPC solver_class=C3Solver solver_mode='c3plus'
          projection='componentwise' (reference default: 'MIQP' via C3MIQP class)
[PLAN-T2] horizon N=20 dt=0.05 horizon_time=1.000s admm_iter=25
          (reference: N=5 dt=0.1 horizon_time=0.5s admm_iter=3 for anything+push_t)
[PLAN-T2] torque_limit=30.0 use_ee_space=False warm_start_across_ticks=False
          (reference: warm_start=false in YAML — matches port cold-start)
[PLAN-T2] solver.rho_init=100.0 (reference: rho_scale=3 adaptive per iter)
```

Per-step:
```
[C3+] step=1 |u[0]|=34.28Nm ... primal=4.933 iters=25/25 lcp_res_max=nan
[C3+] step=3 |u[0]|=34.29Nm ... primal=3.872 iters=25/25 lcp_res_max=nan
[C3+] step=6 |u[0]|=34.38Nm ... primal=3.874 iters=25/25 lcp_res_max=nan
```
Port ADMM hits `iters=25/25` on every solve (no early termination on convergence). Primal residual stabilizes around 3.87 — non-zero, non-decreasing → **NON-CONVERGENT** ADMM. This is the mechanically-proven consequence of 3.g (Stewart-Trinkle with rank-deficient F[γ,γ]) confirmed at runtime.

## 2.k caution applied

Verified all "inert-tagged" areas individually:
- `warm_start` — both agents OFF cross-tick. But WITHIN-Solve warm-start via SetInitialGuessQP is ALWAYS-ON in reference (only gated by `admm_iteration == 0`). Port `_solve_c3plus` carries `delta` and `omega` across iterations naturally in the for-loop (line 1119) — implicit within-Solve warm-start. Both conformant.
- `end_on_qp_step` (reference YAML `false`) — reference computes a "half step" via rollout at `c3.cc:336-347`. Port `_solve_c3plus` returns the solved z directly (the QP step). **Divergence**: port does NOT do the reference's rollout half-step; port returns `x_seq` from the direct QP solution, reference computes `x_seq[i] = A·x[i-1] + B·u[i-1] + D·λ[i-1] + d`. Both should give similar results at ADMM convergence; at non-convergence they differ. This is a NEW divergence surfaced by the c3 lib deep read — 4.g.
- `delta_option = 1` (reference initializes delta head with x0) — port `_solve_c3plus` initializes delta with zeros. **NEW divergence 4.f, small effect (just an ADMM initial-guess).**
- `penalize_input_change` (reference config `anything=true`, `push_t=false`) — port has no analog. NEW 4.j — port always penalizes `u` (absolute); reference toggles between `u` and `u - u_prev`.

None of these individually load-bearing at nominal convergence, but at the port's non-convergent regime (iters=25/25 primal ~3.87), all four could contribute to the divergent trajectory the port ADMM converges to vs reference MIQP would.
