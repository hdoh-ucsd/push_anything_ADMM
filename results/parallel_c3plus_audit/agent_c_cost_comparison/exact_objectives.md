# Exact objectives at HEAD 43d5efb96 (single_obstacle / shelf_gap demos)

State layout x (n=19): [EE pos 0-2 | obj quat wxyz 3-6 | obj pos 7-9 | EE vel 10-12 | obj omega 13-15 | obj v 16-18].
Active weights (per-demo sampling_c3plus_options.yaml): w_Q=50, gamma=1.0,
q = [0.01x3 EE | 0.1x4 quat | 200,200,120 obj pos | 5x3 EE vel | 0.013x3 omega | 0.05x3 v], R = 1*diag(0.01x3).
Horizon N=5. ADMM: standard C3+ (eta-slack elementwise projection), G/U gammas = 1.

## 6A. Inner C3+ objective (all variants; identical in every canonical arm)

J_C3 = sum_{k=0..N-1} (x_k - x_des)' Q_k (x_k - x_des) + u_k' R u_k  + ADMM consensus terms
       G||z - z_proj||^2 (lambda/eta projection), with Q_k = w_Q * gamma^k * diag(q).

Decomposition per knot: EE pos (0.5 eff), quat (5), obj pos (10000/10000/6000),
EE vel (250), omega (0.65), v (2.5), input (0.01). In lcs_contact mode the LCS is
augmented with N_closest=2 frictionless obstacle contacts: lambda_obs/eta_obs enter
the complementarity constraints and D/E/F/H blocks; they carry NO objective weight
(no Q/R rows). VERIFIED: no obstacle proximity cost is active in lcs_contact —
inner-QP potential guarded at cc:2133 (&& !lcs_contact), ranking potential guarded
at cc:2317; conflict throw at cc:785 prevents soft potentials coexisting.

## 6B. Candidate-ranking objective (route_sel_mode=0 "current", all V0/V1/V3)

For each candidate i (fresh C3 solve at relocated EE):
  J_rank,i = ComputeQuadraticTrajectoryCost(XX_i, x_des, Q_zeroed, UU_i, R_zeroed)
             + travel_cost_per_meter * ||EE_i - EE_now||_xy    [= 0: weight is 0]
             + finished_reposition_cost * I[i==repos_target just finished] [1e9, 1 loop]
where XX_i = kSimImpedanceObjectCostOnly rollout (cost_type 5): PD-impedance
forward simulation of the candidate's C3 plan through the LCS; Q_zeroed = Q with
EE-position and EE-velocity blocks and R zeroed => ONLY object terms count:
  quat error (w 5/knot), obj pos error (w 10000/10000/6000), omega (0.65), v (2.5),
summed over N+1 knots, no separate terminal weight, quadratic norm throughout.
Units: position m^2 * weight; orientation quaternion-component error^2 * weight.
Special penalties: rejection hard-filters set cost 1e12 (not additive).

## 6B'. delta_v_lexicographic (V2/V4; requires channel_v1)

PRIMARY: R_i = dV_i / (T_repos,i + T_push + 1e-3), dV_i = route-value decrease of the
predicted terminal object position along the active channel (calibrated: x0.61 when
ROUTE_PREDICTION_MODE=calibrated_v1); T_repos from PWL geometry ((2*0.06+xy)/0.18),
T_push = 5.0 s. Eligibility: dV_i > 0.002 m; yaw regression bound: predicted terminal
yaw error <= current + 0.15 rad; hard-filtered candidates excluded. SECONDARY: within
the near-best set (dV >= 0.8*dV_max), lowest J_rank wins. Runs ONLY while pushing
(is_doing_c3_); during repositioning the pursued target is held (fidelity fix).

## 6B''. delta_v_weighted (mode 3, pilot): J_rank,i - 20000 * dV_i (additive credit).

## 6C. Reposition score / mode switching

current (V0/V1/V2): raw J_rank + relative hysteresis. C3->repos: best_other <
curr*(1-0.9*frac...) via hyst_c3_to_repos_frac; repos->repos stickiness 0.7-frac;
repos->C3 0.9-frac; kToBetterRepos target changes only.
transaction_v1 (V3/V4 base): C3->repos exit additionally requires best_other < curr
(no-churn gate) and the c3->repos hysteresis is scaled by (1 + T_repos/T_push)
(full transaction value); buffer candidate NOT argmin-eligible (stale-cost skip).
transaction_v1_1 (V3/V4): + timeout (repos_loop_count > 1100 -> force C3, mark
unsuccessful) + symmetric gate (best_other >= curr -> return to C3 immediately).
P5 (all arms, default ON): timeout leg active regardless of mode.

## 6D. Hard filters and state-machine guards (NOT cost terms)

- LCS nonpenetration of obstacle contacts (complementarity, solver-level).
- Candidate-object penetration filter (in_collision rejection).
- Pusher-obstacle endpoint filter (PUSHER_FILTER, off by default).
- Reposition swept-path veto (P5, ON): PWL lateral+descend legs vs obstacles below
  obs_top_z=0.12; 4 cm escape zone at start; placeholder (0,0,0) slots excluded.
- Workspace bounds: inner/outer robot radius hard abort; z limits (T config z_min
  floor 10 mm below ground -> z hard abort cannot fire, known).
- Reposition timeout 1100 loops (P5 default ON).
- No-progress detector: config cost (cost_type 3 first-knot vs FINAL goal) drop
  0.5 over 16 C3 loops -> unproductive-push reposition trigger.
- finished_reposition_cost 1e9 for one loop after arrival (forces fresh compare).
- No topple filter exists. No contact-acquisition requirement exists (arrival is
  5 mm EE-position only). No failed-sector memory beyond the unsuccessful buffer.

## 6E. Shadow L1 definitions (offline only; never live)

L1 TASK:  J = w_xy*||p_N - p_ref||_1/d_ref + w_yaw*|wrap(th_N - th_ref)|/pi + w_vel*||v_N||_1/v_ref, weights (1,1,0.2).
L1 ROUTE: J = w_route*V_active(p_N)/V_ref + w_cross*d_perp(p_N,route)/d_ref + w_yaw*|yaw_err|/pi, weights (1,0.5,0.5).
L1 TXN:   J = J_L1_route + w_time*T_repos/T_ref + w_length*L_repos/L_ref + w_switch*I[i!=current] + w_failure*fail_hist, weights (0.5,0.25,0.25,0.5); blocked paths INFEASIBLE (excluded), not high-cost.
Normalization: d_ref = current position-to-sub-goal distance; V_ref = current route value; T_ref/L_ref = median feasible reposition time/length over the bank. Weights NOT tuned to make any candidate win; sensitivity sweep (x0.5/x2 per weight) left top-1 unchanged at all three states — rankings are weight-stable because all L1 forms consume the same predicted terminal state as J_rank.
