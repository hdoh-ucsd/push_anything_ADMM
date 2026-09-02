# Experiments

This page separates the repository's two current experimental tracks. The
results below are measured simulation artifacts, not claims of general task
success or complete paper reproduction.

## 3D Jack Manipulation

The 3D Jack experiment tests non-prehensile manipulation with a full SE(3)
goal. Unlike planar pushing, success requires both translating the Jack and
rolling it onto the requested support tripod. The sampling controller draws
candidate end-effector locations on a sphere around the object, C3+ plans the
contact-rich motion, and the operational-space controller executes the selected
trajectory.

The current seed-0 ablation retains C3+ but adopts one control setting from the
published Jack C3 configuration: it penalizes the absolute input `u` instead
of the change from the previous solution `u-u_prev`. No additional task phase,
controller frequency, or output port is introduced.

```bash
python main.py push_jack \
  --sampling-c3 config/sampling_c3_kik_jack.yaml \
  --solver c3plus \
  --seed 0 \
  --max-time 600 \
  --name jacktoy_c3control_seed0_600s
```

Two continuous 600 s runs have now been completed with the same controller:

| Seed | Tight goals reached | First goal | Second goal by 600 s |
|---:|---:|---:|---:|
| 0 | 1 | 585.7 s | No |
| 1 | 1 | 419.3 s | No |

Each run drew a second random goal immediately after completing the fixed boot
goal. Consequently, both logs correctly report `goals_reached=1`, while their
final `tight_goal` and `loose_goal` fields are failures measured against the
second active goal.

Artifacts:

- [Detailed Jack ablation report](experiments/jacktoy_c3_absolute_input_c3plus.md)
- [Seed-0 run log](../results/jacktoy_c3control_seed0_600s.txt)
- [Seed-1 continuous run log](../results/jacktoy_continuous_c3control_seed1_600s.txt)
- [Seed-0 video at 10× speed](../results/jacktoy_c3control_seed0_600s_sidepanel_s10.mp4)

Both tested seeds completed exactly one goal, but two trials are still too few
for a general success-rate claim. Neither run demonstrated sustained
multi-goal throughput within 600 s.

## Single-Object Pushing

The single-object experiments test planar pushing from object geometry and a
goal pose. The controller samples candidate contact locations, constructs a
local complementarity model for each candidate, selects between contact MPC
and collision-free repositioning, and tracks the chosen trajectory with the
Franka operational-space controller.

The stored Fig. 8-style campaign currently contains completed fixed-goal runs
for the following objects:

| Object | Fastest recorded time to goal |
|---|---:|
| Chicken Broth | 27.450 s |
| Milk Bottle | 54.600 s |
| Clamp | 131.100 s |
| Push T | 19.650 s |

These entries come from `FIG8_SUCCESS_RUNS.csv`. The manifest intentionally
retains repeated runs, including two Chicken Broth and two Push T artifacts.
The table above reports the fastest stored completion for orientation; it is
not a success-rate comparison. Failed and incomplete runs are not represented
in that success-only manifest.

A configured single-object run follows this pattern:

```bash
python main.py milk \
  --sampling-c3 config/sampling_c3_kik.yaml \
  --solver c3plus \
  --seed 0 \
  --max-time 600 \
  --name milk_seed0
```

For the T-shaped benchmark:

```bash
python main.py push_t \
  --sampling-c3 config/sampling_c3_kik_t.yaml \
  --solver c3plus \
  --seed 0 \
  --max-time 600 \
  --name push_t_seed0
```

Artifacts and provenance:

- [Successful-run manifest](../FIG8_SUCCESS_RUNS.csv)
- [Stored campaign figure](figures/fig8_fixed_goal_result.png)
- [Reproducibility policy](REPRODUCIBILITY.md)

The fixed-goal campaign is a port-replication diagnostic. It does not reproduce
the paper's randomized-goal evaluation protocol and should not be presented as
a generalization or success-rate benchmark.
