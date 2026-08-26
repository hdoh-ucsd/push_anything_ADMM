# Figure 8 C3+ campaign — failure classification

Completion date: **2026-08-25** (America/Los_Angeles)  
Experiment ID: `fig8_mesh_28_c3plus_2026-08-25`

## Campaign outcome

| Outcome | Count |
|---|---:|
| Objects reaching 28/28 successes | 21 / 25 (84%) |
| Objects marked terminal failure | 4 / 25 (16%) |
| Successful attempts | 615 |
| Failed attempts | 254 |
| Incomplete/interrupted attempts | 12 |
| Attempt success rate, excluding incomplete | 70.8% |

The four terminal objects retained partial successes. They were stopped by operator decision after repeated failures; they were not counted as completed objects.

## Terminal-object failure modes

Classification uses the final campaign classifier first, then log evidence:

- **Topple / planar unreachable:** failed log contains `[PLANAR-UNREACHABLE]`.
- **Tight-gate miss:** failed log has no planar-unreachable event and its final result is `tight_goal=FAIL`, `loose_goal=PASS`.
- Incomplete logs are reported separately and excluded from failure-mode percentages.

| Object | Success | Failure | Incomplete | Topple / unreachable | Tight-gate miss | Success rate* |
|---|---:|---:|---:|---:|---:|---:|
| `E_shape_video` | 1 | 19 | 1 | 5 | 14 | 5.0% |
| `tape` | 2 | 18 | 1 | 0 | 18 | 10.0% |
| `gallon_milk` | 6 | 32 | 2 | 32 | 0 | 15.8% |
| `eraser` | 18 | 90 | 2 | 90 | 0 | 16.7% |
| **Total** | **27** | **159** | **6** | **127** | **32** | **14.5%** |

\* Success / (success + failure), excluding incomplete attempts.

Across the terminal objects, **127/159 failures (79.9%)** were associated with toppling and loss of planar reachability. The remaining **32/159 (20.1%)** were tight-gate misses after satisfying the loose positional/orientation criterion.

## Interpretation

### 1. Mesh stability / toppling — primary failure mode

`gallon_milk` and `eraser` are unambiguous: every classified failure contains a planar-unreachable event. Their final rotational errors are commonly near or above 1.6 rad, consistent with the object leaving the intended planar manipulation envelope. This supports the earlier mesh mass/inertia/contact-stability hypothesis more strongly than a goal-distance explanation.

`E_shape_video` is mixed: five failures toppled, while fourteen remained planar and ended near the target but outside the tight gate.

### 2. Tight success-gate miss — secondary failure mode

All eighteen classified `tape` failures passed the loose goal but failed the tight goal. Typical final errors are about 0.015–0.019 m translation and 0.19–0.21 rad rotation. This is not the same symptom as the mesh-topple failures: the object generally reaches the target neighborhood but does not converge tightly enough.

Fourteen `E_shape_video` failures have the same gate-limited pattern, typically around 0.012–0.015 m translation and 0.16–0.17 rad rotation.

### 3. No evidence that the 75-minute watchdog dominated results

None of the 159 terminal-object failures was classified as a campaign wall-timeout failure. The watchdog bounded pathological runtime without creating the dominant failure categories above.

## Presentation frames

The retained `gallon_milk` seed 3 video provides the clearest representative sequence:

1. Initial planar configuration.
2. First planar-unreachable/topple event at simulation time 7.875 s.
3. Temporary recovery at simulation time 124.950 s.
4. Re-topple at simulation time 127.950 s.
5. Final miss: 0 goals reached, 0.2473 m translation error, 1.6029 rad rotation error.

Use `gallon_milk_topple_sequence.png` as the primary slide figure. Individual full-resolution frames are in `frames/`.

## Top two examples per terminal object

Presentation sheets and their individual source frames are in `top2_examples/`.

| Object | Example 1 | Example 2 |
|---|---|---|
| `E_shape_video` | Seed 0: settled unreachable pose after 1.779 rad tilt event | Seed 2: second settled unreachable pose after toppling |
| `tape` | Seed 18: closest tight miss, 0.0152 m / 0.1856 rad | Seed 19: largest failed rotation residual, 0.0162 m / 0.2150 rad |
| `gallon_milk` | Seed 3: persistent unreachable pose after toppling | Seed 35: persistent unreachable pose after re-toppling |
| `eraser` | Seed 3: settled unreachable pose after immediate topple | Seed 81: displaced, toppled, unreachable pose |

The paired sheets are `E_shape_video_top2.png`, `tape_top2.png`, `gallon_milk_top2.png`, and `eraser_top2.png`.

The unreachable examples use a wide oblique camera and are sampled after the object has settled beyond the planar-reachability envelope, not at the instant the threshold is crossed. `tape` never produced a planar-unreachable event; its two sheets are explicitly labeled reachable tight-gate misses.

## Provenance

- Raw campaign logs: `results/fig8_mesh_28_c3plus/<object>/`
- Terminal markers: each failed object's `CAMPAIGN_FAILURE.json`
- Representative video: `results/fig8_gallon_milk_seed00003_failure_sidepanel_s4.mp4`
- Frame timestamps account for the video's 3× playback rate.
