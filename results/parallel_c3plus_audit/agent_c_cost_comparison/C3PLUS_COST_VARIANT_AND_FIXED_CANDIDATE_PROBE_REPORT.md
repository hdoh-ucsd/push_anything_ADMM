# C3+ Cost Variant and Fixed-Candidate Probe — Agent C

Date: 2026-09-05. Branch `audit/cost-variant-probe`, outer-repo base commit `4316b1ea7325b14f1de3dac6fa197e3b6aa4a0d3`. Controller repo: `external/oim_c++_anything` worktree `obstacle-lcs-contact`, branch `c3plus-channel-route-v1`, HEAD `0abc0eb7c`. No live cost change was implemented; no weights were tuned; no campaign runs were launched.

## 0. Provenance and reuse

The 2026-09-05 variant inventory + cost probe (`results/c3plus_variant_cost_probe/`, executed at controller commit `43d5efb96`) already performed this task's score-only probe, candidate banks, deterministic replays, and shadow-L1 comparison. An independent verification pass (this audit) confirmed that exactly **one commit** separates `43d5efb96` from controller HEAD `0abc0eb7c`, and it is **docs/results only** — no change to ranking cost, reposition score, route generation, P5 recovery, obstacle formulation, buffer semantics, or progress detector. All probe artifacts therefore remain valid at HEAD and are re-shipped here with the task-required schemas. (The hash `43d5edb96` circulating in some notes is a typo for `43d5efb96`.)

## 1. Variant registry and layer classification (§2-3)

`variant_registry.csv` (task schema, H0-H4 / V0-V6 / S1-S3 + deprecated delta_v_weighted) and `variant_registry_detailed.csv` (30-row per-flag registry with parse sites) plus `variant_stack_matrix.csv` and `configuration_diff_matrix.csv`. Every historical formulation is env-reachable at HEAD (one-shot parse in `ObsCfg()`, cc:719-812). Key classifications:

- **`lcs_contact` (H4) is NOT a cost-function variant.** It constrains dynamics/feasibility as a frictionless closest-footprint LCS contact; both obstacle-objective code paths are guarded off in this mode (cc:2133, cc:2317-2329). `J_rank_obstacle_total` is a passive diagnostic column only.
- **V5 (approach fix) is not an independent factor** — unconditional inside channel_v1 (cc:1966-1982), no opt-out flag.
- **V6 (post-P5 plain controller)** = V0 + default-on repos timeout (cc:804-805, 1100 loops), swept-path veto (cc:800-801), 4 cm escape zone (cc:854/861), placeholder exclusion (cc:2419-2423) — all OUTER_LOOP_OR_RECOVERY, not cost.
- V1→V2 is a two-switch package (SELECTION_MODE + PREDICTION_MODE) by design of the fidelity arc; not a clean single-factor delta.

## 2. The four decision quantities (§4) — never one "the cost"

Full formulas with weights, units, and source lines in `exact_objectives.md`. Summary:

- **A. Inner C3 objective**: identical across all canonical arms. Per-demo `sampling_c3plus_options.yaml`: w_Q=50, q=[.01×3 EE | .1×4 quat | 200/200/120 obj pos | 5×3 EE vel | .013×3 ω | .05×3 v], R=1·diag(.01), N=5, no terminal weighting; obstacle participates as LCS contact (λ_obs/η_obs), never as an objective term, in H4.
- **B. Ranking objective** (cost_type 5, `kSimImpedanceObjectCostOnly`, cc:1484-1505): object-only quadratic tracking of a PD-impedance forward simulation; effective per-knot weights quat 5, obj pos 10000/10000/6000, ω 0.65, v 2.5; travel term weight **0**; unchanged at HEAD (verified against source).
- **C. Progress score**: enforced cost drop 0.5 over 16 C3 loops (`progress_params_c3plus.yaml:44-45`; base yaml uses 0.1/100).
- **D. Reposition score**: same ranking cost + travel term (weight 0) + hard gates; time enters only via transaction_v1 hysteresis scaling (cc:2664-2671). Hard filters (never costs): collision filter, P5 swept-path veto + escape zone + placeholder exclusion, workspace aborts, repos timeout, 1e9 finished-reposition marker, no-progress detector.

## 3. Historical issues audited independently at HEAD (§5)

`historical_issue_status.csv`, all verified from source at `0abc0eb7c` with file:line evidence: stale buffer costs **PARTIALLY REMOVED** (skipped only in transaction modes; argmin-eligible in default mode), zero reposition travel cost **PRESENT** (weight 0 in every shipped config), current-contact exclusion **PRESENT** (min_element starts at index 1; env-gated compensations feed only the v1.1 return gate), one-cycle 1e9 finished-reposition **PRESENT** (by design, one-loop discriminator). Plus one registered defect: the (0,0,0) placeholder is unguarded on obstacle-free scenes.

## 4. Probe states, candidate banks, score-only probe (§6-8)

Three real states (`probe_states/`): **S0** open-table reposition churn (t=9.0 s; run later succeeded), **S1** single-obstacle north-face fixed point (t=119.7 s), **S2** shelf-corridor plateau (t=150.0 s). One fixed deterministic bank of 17 candidates per state (`candidate_banks/`) — current contact + all logged event candidates + dense reachable face contacts — used identically by every score. Scores computed on the SAME fixed logged rollouts, no re-solving: quadratic J_rank and its components, calibrated ΔV, transaction score, and shadow L1 task/route/transaction (definitions + normalization + weight-sensitivity in `exact_objectives.md` §6E; blocked paths treated as infeasible, weights untuned and rank-stable under ×0.5/×2 sweeps). Outputs: `candidate_rank_comparison.csv`, `candidate_rank_correlation.csv`, `cost_component_breakdown.csv`.

## 5. Rank correlation and replay ground truth (§10-12)

Spearman = **+1.0 among J_rank, ΔV_cal, L1_task, L1_route at every state** — they all consume the same predicted terminal state. L1_txn deviates only via the reposition-time term (anti-correlates at S0/S2). Winners:

| State | J_rank / ΔV / L1 task / L1 route | L1 txn | Actual best (deterministic replay) |
|---|---|---|---|
| S0 | logged_0 | logged_0 | **logged_2**; dense bank stem_east +37 mm |
| S1 | logged_0 | logged_0 | logged_0 ✓ among drawn; only truly productive contact (stem-tip circumnavigation, +8 mm) was **never drawn** and all alternates were hard-filtered to 1e12 |
| S2 | logged_1 | logged_0 | dense bank stem_tip +24 mm / stem_west +22 mm; the pursued contact moved the object **25 mm backward** |

Actual replays: deterministic pydrake kinematic-pusher rig, exact SDFs, 2 s @ 0.05 m/s, 3 reps (`actual_replay_performance.csv`, `predicted_vs_actual_candidate_rank.csv`). Predicted displacements reach 340 mm against realized motion under 80 mm; at S2 predictions are FLAT (max 9 mm predicted, 2 % cost spread) so every score ranks noise. Component forensics (`cost_component_breakdown.csv`): position term dominates (10000/10000/6000 vs quat 5) so orientation does NOT dominate; ΔV rewards predicted-but-unrealized motion; transaction time changes ordering only at S0/S2 (its anti-correlation); L1 removes squared-error domination but **does not change any winner**, because the inputs, not the norm, are wrong.

## 6. Cost failure vs prediction failure — decision gate (§13)

- **S0 → CASE C2**: the actually-best candidate is ranked LAST by the current cost AND by every shadow L1 form, identically — all scores are downstream of the same dishonest predictor. (Self-healing via outer-loop retries; the run succeeded.)
- **S1 → CASE C3/C4 territory (generation/filters, not cost)**: the one productive contact was never in the drawn set and the feasible pool was emptied by hard filters — the fixed point is enforced by starvation, not mis-ranking.
- **S2 → CASE C2 + partial C3**: predictions flat; productive contacts exist in the dense bank but weren't drawn; the pursued contact was actively counterproductive.
- **CASE C5 also holds**: the scenes have different primary failures.

## PRIMARY VERDICT

**ALL_COSTS_DEPEND_ON_INACCURATE_PREDICTIONS** (secondary: DIFFERENT_SCENES_HAVE_DIFFERENT_PRIMARY_FAILURES).

No shadow L1 score ranked an actually-productive candidate above unproductive ones anywhere the current cost failed — they inverted the actual order at S0 exactly as J_rank did. Per the gate: **recommended_live_cost_change = null**. Required before any cost redesign: (1) honest rollout predictions at plateau states (S2 flat-landscape mechanism; in-contact ρ≈0.61, traveling-promise≈0), (2) route-forward stem-tip-class contacts must actually enter the candidate pool at S1/S2-type states, (3) guard the placeholder on obstacle-free scenes. Handoff: `agent_c_handoff.json`; minimal Agent B replay set: `replay_candidate_manifest.json`.
