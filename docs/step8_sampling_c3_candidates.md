# Step 8 candidates: sampling-C3 wrapper-level issues identified during kIK live-sim verification

This document is forward-looking scope, recorded so it is not lost between sessions. None of the candidates below have been investigated; this is the diagnosis-first artifact, separate from the kIK delivery's preserved knowledge in `docs/reposition_ik.md`.

The candidates were surfaced during the kIK live-sim mechanical verification run (step 7 closure). The mechanical receipt for kIK stands; the issues below are orthogonal to the tracker and apply equally to both reposition trajectory types.

## Shared context: verdict-A evidence

A controlled comparison was run with identical scenario, identical config, differing only in `reposition_params.traj_type`:

| | kIK | PWL |
|---|---|---|
| Command | `python main.py pushing --sampling-c3 config/sampling_c3_kik.yaml --name kik_live_test --no-record` | `python main.py pushing --sampling-c3 config/sampling_c3_params.yaml --name pwl_live_test --no-record` |
| Logs | `results/kik_live_test.txt` | `results/pwl_live_test.txt` |
| `final_obj_xy` | `(0.0000, 0.0000)` | `(0.0000, -0.0000)` |
| `goal_dist` | `0.3000m` | `0.3000m` |
| Box displacement over 8s | **0.000m** | **0.000m** |
| Total mode switches | 4 | 10 |
| free-mode fraction (full run) | 780/801 = 97.4% | 769/801 = 96.0% |
| free-mode fraction (last 100 steps) | 100/100 | 98/100 |
| Unique repos targets (last 200 steps) | 1 | 1 |
| Repos target (last 200 steps) | `(-0.135, -0.120, +0.050)` | `(-0.170, -0.059, +0.050)` |

Both wrappers settle into free-mode-forever on a single fixed `prev_repos` target that places the EE southwest of the box. Neither target produces useful contact when reached. The two trackers fail in mechanically distinct ways — kIK reaches its target and parks (`ee_to_target=0.092m` steady), PWL oscillates ~14cm short — but the *gating* failure is the wrapper's repos-target selection and stall-recovery logic, which is shared.

This rules out a kIK-specific regression at the wrapper integration level. The kIK delivery closes; the issues below are step 8 scope.

## Candidate 1: `prev_repos` sample wins despite being unreachable for productive contact

**Observation.** Across the last 200 steps of both runs, `best_src=prev_repos` wins every selection (`best_k=1`). `[GS-table]` snapshots show `prev_repos` consistently scoring lower `c_sample` than the strategy-generated `strat_*` samples — but the EE positions implied by `prev_repos` do not lead to box motion when the wrapper subsequently re-enters c3 mode.

**Hypothesis.** The travel-cost discount (`w_travel * ‖p_sample − p_ee_now‖`) makes `prev_repos` artificially cheap once the wrapper has driven the EE near it, regardless of whether that target is still useful. From the last `[GS-table]` of `kik_live_test.txt` (step 800):

```
k=0 (current   ) ... c_C3=257871.84 align=0.0000(bonus=0)         travel=0.000m(pen=0.00)  c_sample=257871.84
k=1 (prev_repos) ... c_C3=197662.88 align=0.7718(bonus=23155.27)  travel=0.092m(pen=18.33) c_sample=174525.94 ← WIN
k=2 (strat_0   ) ... c_C3=372954.77 align=0.0000(bonus=0)         travel=0.134m(pen=26.88) c_sample=372981.64
```

`prev_repos` wins by a 60k+ margin via a combination of lower `c_C3` and the alignment bonus, while `strat_0` (the only fresh strategy sample) is 200k worse on `c_C3` alone.

**Investigation needed.** Why are `strat_*` samples scoring so much higher on `c_C3` than `prev_repos` from a similar EE position? Is the cheap-solve ADMM iteration count (`surrogate_admm_iters=1`) producing systematically pessimistic estimates for fresh samples?

> **Status (resolved by step 9):** Candidate 1's hypothesis is correct. `surrogate_admm_iters=1` produces systematically pessimistic `c_C3_raw` estimates for symmetric behind-box samples (the proxy), giving `align_bonus=0` to the sample geometrically designed for contact-seeking. See `docs/step9_findings.md` for the full mechanism analysis.

## Candidate 2: `w_align` / `w_travel` calibration in late-run states

**Observation.** The `align` term (max `30000 × 1.0 = 30000` bonus) and the `travel` term (`200 × distance_m`) operate on very different scales. In the V-9 example above, the alignment bonus contributes 23,155 to `prev_repos`'s win while travel contributes only 18 to its disadvantage — alignment dominates for any sample with a non-zero alignment score.

**Hypothesis.** The current weighting may be correct for the directional-push regime that `w_align=30000` was empirically tuned for (see `config/sampling_c3_params.yaml` comments and CLAUDE.md "Polyhedral Friction-Cone Bias" note), but breaks down when the wrapper has already committed to a target that is alignment-positive but unreachable for productive contact.

**Investigation needed.** Whether `w_align` should decay or be conditioned on `steps_since_improve`, and whether `w_travel` is too small to overcome a sticky `prev_repos` choice. The `met_progress=N steps_since_improve=75` state at the end of the kIK run shows the wrapper is aware it is not making progress but the cost function does not penalize the stuck sample.

## Candidate 3: `kToReposCost` / `kToC3Cost` threshold interaction in stuck states

**Observation.** Across the last 100 [GS] steps:

| Reason | kIK | PWL |
|---|---|---|
| `kStayInRepos` | 100 | 97 |
| `kStayInC3` | 0 | 1 |
| `kToC3Cost` | 0 | 1 |
| `kToReposCost` | 0 | 1 |

Neither wrapper produces a sustained switch back to c3-mode once it has settled on the bad `prev_repos`. PWL fires `kToC3Cost` once but immediately fires `kToReposCost` back — the c3 attempt does not persist long enough to make contact.

**Hypothesis.** The cost-comparison thresholds (`hyst_repos_to_c3_frac=0.30`, `hyst_repos_to_c3_frac_position=0.50` in the relative-hysteresis branch) require c3-mode cost to undercut repos-mode cost by 30–50% before the switch fires. Once the wrapper is in a stuck state where every c3 candidate evaluates poorly because the EE is parked far from the box, this gap is unreachable.

**Investigation needed.** Whether a `steps_since_improve`-aware fallback should force a c3-mode attempt (or a fresh-sample-only repos-mode attempt) after N stuck loops, bypassing the cost-comparison hysteresis. The current `num_control_loops_to_wait=60` / `num_control_loops_to_wait_position=30` may not be triggering as expected; worth tracing through `progress.py`'s `met_progress` logic to confirm.

## Reproducing the verdict-A comparison

```bash
# kIK
/root/miniconda3/envs/push_anything_ADMM/bin/python main.py pushing \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --name kik_live_test --no-record

# PWL
/root/miniconda3/envs/push_anything_ADMM/bin/python main.py pushing \
    --sampling-c3 config/sampling_c3_params.yaml \
    --name pwl_live_test --no-record
```

Configs differ only on `reposition_params.traj_type` (`kIK` vs `kPiecewiseLinear`); confirm with:

```bash
diff config/sampling_c3_params.yaml config/sampling_c3_kik.yaml
```

Each run takes ~150–180 seconds wall on the reference machine.
