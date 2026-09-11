# Small C3+ cost-tuning study

## Executive result

The study stopped at the mandatory reproducibility gate and launched **zero**
live simulations. Benchmark v2 is ready (90 synchronized, feasible cases and
zero C3+-OIM pose/geometry mismatches), and logger v2 has demonstrated full
candidate-pool/N+1 logging in a short ICRA smoke. However, the authoritative
C3+ sampler still constructs `mt19937` engines from `std::random_device` and
does not accept the task seed. A one-run-per-setting study would therefore
confound cost parameters with unrelated candidate draws. In addition, the
logger's initial-C3 transaction bootstrap fix was compiled after its only
runtime smoke, so that final acceptance item remains unverified. In accordance
with the task's stopping rules, no passive replay, orientation perturbation,
epsilon selection, obstacle-weight test, confirmation run, or 90-case campaign
was launched.

## Scope and provenance

- Outer repository: `4316b1ea7325b14f1de3dac6fa197e3b6aa4a0d3`
- Authoritative C3+ worktree:
  `external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics`
- C3+ commit: `bd5ce7b52d7133d54065578e42874d92d3cce95c`
  plus preserved uncommitted forensics-logger work
- CPU inventory: 16 logical CPUs, 8 physical cores, AMD Ryzen 7 7800X3D
- Worker count: 0; no run-to-worker mapping exists because the launch gate
  failed
- Live-run budget: 36 maximum
- Live runs actually executed by this study: **0**
- Historical logger-v2 smoke output was audited but is not counted as a run in
  this study and is not a synchronized rot15 pilot replay.

No controller, sampler, cost, solver, benchmark, scene, or runtime parameter
was changed by this study. Existing dirty-tree changes and research outputs
were preserved.

## Prerequisite gates

| Gate | Result | Evidence |
|---|---|---|
| New rot15 benchmark | PASS | 90 canonical rows; 15 per each of six scenes; 90/90 goal-feasible; 90/90 parity PASS; focused tests 11/11 PASS |
| Logger-v2 source capability | PASS | Full candidate pool, exact ranking rollout, N+1 knots, cost decomposition, exact live obstacle scalar, geometry/contact/diversity/event signals implemented and short-smoke validated |
| Logger-v2 final runtime acceptance | BLOCKED | Initial-C3 transaction bootstrap fix was compiled after the only smoke; a populated transaction row has not yet been demonstrated |
| Matched C3+ randomness | BLOCKED | `GenerateSampleStates` has no seed/RNG parameter and its sampling paths instantiate `mt19937` from `std::random_device` |
| Live launch | BLOCKED | Hard prerequisites require both logger acceptance and controlled matched randomness |

The canonical task seed (`17000 + 100*scene_index + start_index`, invariant
across the three rotations) is correctly stored in the benchmark adapters.
OIM consumes its seed, but current C3+ does not. The environment variable
`SAMPLING_C3_FORENSICS_SEED_INFO` records provenance only; it does not control
the sampler.

Authoritative RNG evidence:

- `examples/sampling_c3/generate_samples.cc:517`:
  `std::mt19937 gen(std::random_device{}())`
- The multi-object path repeats random-device construction at lines 651 and
  656-657.
- `SamplingC3Controller::ComputePlan` calls `GenerateSampleStates` without a
  seed.

The separate deterministic seed facilities under the OIM/full-sampling code
path are not used by this authoritative C3+ `GenerateSampleStates` call and
were not treated as evidence of C3+ reproducibility.

## Current reference cost audit

All selected legacy g02 source demos use the same C3+ cost configuration:

- Horizon `N=5`, hence six (`N+1`) state-cost knots.
- `gamma=1.0`.
- Position-regime planning dt `0.1 s`; pose-regime planning dt `0.05 s`.
- Pose-regime `w_Q=50` and object-position diagonal
  `[200, 200, 120]`, giving the exact `[10000, 10000, 6000]`
  translation block.
- Position-regime `w_Q_position=50` and object-position diagonal
  `[250, 250, 250]`, giving `[12500, 12500, 12500]` before the existing
  pose-regime switch. This existing regime behavior remains frozen.
- `use_quaternion_dependent_cost=true`.
- Quaternion orientation weight `q_quaternion_dependent_weight=510`.
  `UpdateCostMatrices` overwrites the quaternion block with 510 times the
  regularized Hessian returned by
  `hessian_of_squared_quaternion_angle_difference`; this is not a scalar
  `510*yaw_error^2` surrogate.
- Baseline multipliers are `alpha_p=1` and `alpha_theta=1`. The planned
  orientation multipliers would map to quaternion-Hessian weights 255, 510,
  and 1020 without changing the formulation.

Obstacle semantics require special provenance:

- Obstacle-scene YAMLs contain the historical exponential parameters
  `obstacle_cost_weight=5000` and `obstacle_cost_decay=0.04`.
- The current scene launcher defaults to
  `SAMPLING_C3_OBSTACLE_MODE=lcs_contact`. In that mode the baseline live
  ranking obstacle potential is deliberately zero; obstacle interaction is in
  the augmented LCS contact model.
- The optional footprint-aware ReLU rank branch defaults to
  `epsilon=0.01 m`, `w=0`, and activates only when
  `SAMPLING_C3_RANK_OBS_MODE=relu_footprint` plus a positive
  `SAMPLING_C3_OBS_RELU_W` are supplied.
- Therefore `5000` was not mislabeled as a current ReLU weight. No live ReLU
  reference weight can be selected until the synchronized forensic replay
  yields the required `w_prevent_star` evidence.

## Pilot panel selected offline

| Alias | Canonical task | Rationale |
|---|---|---|
| OPEN_A | `open_task/s01_g02_{rot000,rotCCW090,rotCW090}` | Start-to-g02 distance 0.831154 m is the median of the five starts; no obstacle or edge pathology was identified |
| SINGLE_BLOCKED | `single_obstacle/s04_g02_rot000` | The sampled straight footprint path has the most negative clearance of the five starts, -0.048326 m at the central box, while the requested final pose is feasible |
| SHELF_CORRIDOR | `shelf_gap/s01_g02_rot000` | Widest sampled valid straight-corridor clearance among the five starts, 0.050010 m; conservative control for rejecting over-repulsion |
| ICRA_BLOCK | `icra_sign/s05_g02_rot000` | Retains historical ICRA05 start `s05`, uses mandatory g02, crosses the glyph-R region, and has a feasible final C-glyph pose |
| YCB_F1_CONTROL | `ycb_clutter/s02_g02_rot000` | Exact historical pair02 start/translation goal with zero relative rotation; defensible F1 analogue rather than an invented label |

The straight-route values are diagnostic samples of the orientation-aware
footprint along 201 equally spaced SE(2) poses. They identify obstacle relevance
and do not assert that the controller follows the straight path.

For a future Phase D, the deterministic second open start is `s05`: it is the
nearest travel-distance neighbor to median start `s01` (0.814050 m) and is not
OPEN_A. Slalom remains held out exactly as required.

The complete 33-run intended layout is recorded in `experiment_manifest.csv`.
Every entry is explicitly `NOT_RUN` or `NOT_MATERIALIZED`; unresolved values
are `TBD`, never fabricated.

## Experimental phases

### Phase A — passive forensics

Planned: ICRA_BLOCK, SINGLE_BLOCKED, and YCB_F1_CONTROL at the baseline.
Executed: **0/3**. No synchronized `t_first_bad`, `t_persistent_basin`,
`t_last_safer_candidate`, or event clearances exist.

### Offline epsilon and critical weights

Candidate epsilons `0.01`, `0.02`, `0.03`, and `0.05 m` are recorded but were
not evaluated. With no Phase A candidate pools, `Delta J_task`, `Delta Phi`,
`w_prevent_star`, and `w_escape_star` are unavailable. Consequently there is
no `epsilon_star` and no evidence-derived obstacle-weight grid.

### Phase B — orientation

Planned `alpha_theta` values: 0.5, 1.0, and 2.0 on OPEN_A's three rotations.
Executed: **0/9**. No `alpha_theta_star` was selected; the operational baseline
remains 1 solely because no comparative evidence was collected.

### Phase C — obstacle cost

Executed: **0/9 planned**. No values of `w_obs` were tested live. There is no
SINGLE/SHELF/ICRA result table beyond `NOT_RUN`, and no setting can be accepted
or rejected for corridor floor, over-repulsion, false improvement, or lack of
decision effect.

| Task | Live settings | Result |
|---|---:|---|
| SINGLE_BLOCKED | 0 | NOT_RUN |
| SHELF_CORRIDOR | 0 | NOT_RUN |
| ICRA_BLOCK | 0 | NOT_RUN |

### Phase D — confirmation

Executed: **0/12 planned**. No tuned tuple exists, so baseline-versus-tuned
confirmation and failure-class transition claims are unavailable.

## Parameter and mechanistic conclusions

- `alpha_theta_star`: not selected; baseline 1 retained pending evidence.
- `epsilon_star`: not selected.
- `w_obs_star`: not selected.
- `w_prevent_star` / `w_escape_star`: unavailable because synchronized Phase A
  pools were not generated.
- Failure-class transitions: none measured.
- Operational unchanged baseline tuple: `alpha_p=1`, `alpha_theta=1`, optional
  ReLU `epsilon=0.01 m` dormant with `w_obs=0`; obstacle scenes continue to use
  the unchanged LCS-contact formulation.
- Final tuned tuple: **none**.
- Ready for separate 90-case validation: **no**.

No required figure was created: all five requested plots depend on uncollected
live or Phase A candidate data, and plotting placeholders would falsely imply
experimental evidence.

## Commands executed

No controller-launch command was executed. The read-only verification commands
were:

```bash
python3 scripts/audit_cost_tuning_prerequisites.py
python3 -m benchmarks.rot15.benchmark validate
python3 -m pytest tests/test_rot15_benchmark.py -q
```

Observed results:

- prerequisite audit exit code: 2 (intentional blocked gate)
- rot15 artifact validation: PASS (`cases=90`, geometry PASS, parity PASS)
- focused pytest: 11 passed

The audit can be rerun after separately approved seed plumbing and the logger
transaction smoke. A zero exit code is the required launch authorization. No
Phase A/B/C/D command is supplied as runnable while that authorization is
false, preventing accidental unpaired experimentation.

## Files created or modified by this study

- `.gitignore` (narrow exception for compact `results/cost_tuning_small/**`)
- `scripts/audit_cost_tuning_prerequisites.py`
- `results/cost_tuning_small/prerequisite_audit.csv`
- `results/cost_tuning_small/pilot_task_selection.csv`
- `results/cost_tuning_small/experiment_manifest.csv`
- `results/cost_tuning_small/forensic_counterfactuals.csv`
- `results/cost_tuning_small/epsilon_selection.csv`
- `results/cost_tuning_small/orientation_selection.csv`
- `results/cost_tuning_small/obstacle_weight_selection.csv`
- `results/cost_tuning_small/obstacle_weight_selection.md`
- `results/cost_tuning_small/all_run_outcomes.csv`
- `results/cost_tuning_small/all_run_cost_parameters.csv`
- `results/cost_tuning_small/failure_transitions.csv`
- `results/cost_tuning_small/phaseA_forensics/README.md`
- `results/cost_tuning_small/phaseB_orientation/README.md`
- `results/cost_tuning_small/phaseC_obstacle/README.md`
- `results/cost_tuning_small/phaseD_confirmation/README.md`
- `results/cost_tuning_small/REPRODUCIBILITY_BLOCKER.md`
- `results/cost_tuning_small/COST_TUNING_SMALL_REPORT.md`

No C3+ or OIM source/config file was modified in this study.

## Git status and diff summary

The outer worktree was already dirty before this study, including numerous
user-owned deleted media/results and unrelated untracked research files; none
was restored, removed, or edited. The scoped status for this study is:

```text
M  .gitignore
?? scripts/audit_cost_tuning_prerequisites.py
?? results/cost_tuning_small/
```

This study adds one read-only audit script and 17 compact result/report files.
The result directory is about 22 kB. The repository-wide `.gitignore` diff is
10 added lines relative to HEAD, of which this study added the four-line narrow
exception/comment for `results/cost_tuning_small`; the other additions predate
this study and support benchmark-v2 artifacts. The active C3+ worktree remains
dirty from the earlier logger and unrelated experiment outputs, but this study
made no change inside it.

## Smallest unblock sequence

1. Separately approve deterministic C3+ seed plumbing. Pass an experiment seed
   into the existing generator without changing candidate distributions, then
   test that identical seeds reproduce the full candidate pool and that
   rotation labels retain the same seed.
2. Run one short logger-on ICRA smoke to validate the compiled initial-C3
   transaction bootstrap and require `scripts/validate_forensics_logs.py` PASS
   with at least one transaction row.
3. Rerun `scripts/audit_cost_tuning_prerequisites.py`; proceed only when it
   exits zero.
4. Execute the three Phase A synchronized passive replays before selecting
   epsilon or any live obstacle weight.

These are measurement/reproducibility changes, not cost tuning. They require
separate authorization because this task prohibited silently altering sampling
logic. The full 90-case validation remains unlaunched.

E. TUNING_BLOCKED_BY_MEASUREMENT_OR_REPRODUCIBILITY
