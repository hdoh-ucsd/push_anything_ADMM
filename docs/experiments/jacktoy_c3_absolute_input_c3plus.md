# Jacktoy: Reference C3 Input Regularization with C3+

Date: 2026-08-22  
Task: `push_jack`  
Seed: `0`  
Simulation duration: `600 s`

## Question

Can the published Jack C3 control settings improve Jacktoy behavior while
retaining C3+ as the contact-implicit optimizer?

This is a one-variable ablation. It does not add a task phase, another
controller, a new output port, or a second control frequency.

## Reference observation

The published Jack C3 configuration sets:

```yaml
penalize_changes_in_u_across_solves: false
```

Therefore its input term penalizes the absolute command,

\[
J_u = \sum_k u_k^T R u_k,
\]

instead of the port's previous Jack behavior,

\[
J_{\Delta u} = \sum_k (u_k-u_k^{\mathrm{prev}})^T
R(u_k-u_k^{\mathrm{prev}}).
\]

The reference does **not** provide a fixed C3 control vector that can be
replayed. C3 computes its command online. Consequently, this experiment uses
the published input-cost convention rather than inventing a nominal
`u_C3_ref` or running C3 alongside C3+.

Reference file:
[DAIRLab Jack `sampling_c3_options.yaml`](https://github.com/DAIRLab/dairlib/blob/sampling_based_c3_public/examples/sampling_c3/jacktoy/parameters/sampling_c3_options.yaml)

## Change

The sampling-C3 task parameters gained an optional
`penalize_input_change` field. Jack sets:

```yaml
penalize_input_change: false
```

Everything else in the tested controller remains on the Jack C3+ path:

- solver and projection: C3+
- horizon: 5
- planning timestep: 0.1 s position / 0.05 s pose
- ADMM iterations: 3
- force limits: ±50 N horizontally and vertically
- sampling: two additional contact samples and one reposition sample
- mode-switch and controller/OSC update periods: unchanged

Runtime confirmation:

```text
[C3] Solver mode: c3plus
[C3] Input regularization: absolute-u (task override=False)
```

## Command

```bash
python main.py push_jack \
  --sampling-c3 config/sampling_c3_kik_jack.yaml \
  --solver c3plus \
  --seed 0 \
  --max-time 600 \
  --name jacktoy_c3control_seed0_600s
```

## Result

The controller completed one tight SE(3) goal at `t = 585.7 s` and then
generated a second random goal. It did not complete that second goal before
the 600 s cutoff.

Immediately before the goal-completing tripod transition:

```text
t = 585.6 s
translation error = 0.0136 m
rotation error = 0.1097 rad
C3+ command norm = 1.94 N
measured EE–Jack normal force = 0.531 N
tripod transition = RedDown -> AllUp
```

The transition at the next simulation step satisfied the tight orientation
criterion and triggered re-goaling:

```text
[GOAL-GEN] goal #1 REACHED at t=585.700s
```

The final result line is measured against the newly generated second goal,
not the completed first goal:

```text
goals_reached=1
tight_goal=FAIL
loose_goal=FAIL
```

## Comparison with the previous seed-0 run

| Metric | Previous Jack C3+ | Absolute-input C3+ |
|---|---:|---:|
| Tight SE(3) goals reached | 0 | **1** |
| First goal completion | — | 585.7 s |
| Mode switches | 151 | 225 |
| Median C3+ command | 2.27 N | 3.46 N |
| Maximum C3+ command | 12.99 N | 20.25 N |
| Median final primal residual | 11.80 | 16.49 |
| Median final dual residual | 29.60 | 45.61 |
| OSC saturation events | 5 | 259 |
| Average outer step | 82.7 ms | 106.9 ms |

All 10,916 logged C3+ solves in the new run still ended above the `1e-3`
residual tolerance after three ADMM iterations. The change therefore improved
task outcome without improving numerical convergence.

## Interpretation

The result supports a narrow conclusion: using the reference C3 absolute-input
cost can help C3+ complete the Jack pose task. It is not sufficient evidence
that the variant is robust. Goal completion occurred very late, mode switching
increased, solver residuals worsened, and the run contained high-force contact
episodes (for example, approximately 35 N at `t = 87.4 s`).

This experiment should be described as a successful seed-0 ablation, not as a
general Jacktoy solution. The unchanged-controller replication below tests a
second seed without adding task phases.

## Continuous seed-1 replication

An unchanged seed-1 run subsequently completed the same fixed boot goal at
`419.3 s`, then generated a second random goal and did not complete it before
the `600 s` cutoff.

| Metric | Seed 0 | Seed 1 |
|---|---:|---:|
| Goals reached | 1 | 1 |
| First-goal time | 585.7 s | 419.3 s |
| Mode switches | 225 | 219 |
| Median C3+ command | 3.46 N | 2.29 N |
| Maximum C3+ command | 20.25 N | 11.86 N |
| Median final primal residual | 16.49 | 14.21 |
| Median final dual residual | 45.61 | 37.77 |
| OSC saturation events | 259 | 0 |

This establishes repeatability of one boot-goal completion across two seeds,
but not continuous multi-goal performance: both runs reached exactly one goal.

## Artifacts

- Log: [`results/jacktoy_c3control_seed0_600s.txt`](../../results/jacktoy_c3control_seed0_600s.txt)
- Seed-1 log: [`results/jacktoy_continuous_c3control_seed1_600s.txt`](../../results/jacktoy_continuous_c3control_seed1_600s.txt)
- 10× video: [`results/jacktoy_c3control_seed0_600s_sidepanel_s10.mp4`](../../results/jacktoy_c3control_seed0_600s_sidepanel_s10.mp4)
- Previous comparison log: [`results/jacktoy_refinit_krandom_seed0_600s.txt`](../../results/jacktoy_refinit_krandom_seed0_600s.txt)
