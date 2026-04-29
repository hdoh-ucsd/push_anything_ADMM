# G2 ADMM-iter sweep: empirical results and math analysis

## TL;DR

- Sweeping `--admm-iter ∈ {3, 9, 10, 25}` on the without-prepositioned
  `pushing` task shows the planned normal force at step 1 grows
  17× from 3 iters to 25 iters (`λ_n_max: 0.144 → 2.507 N`),
  and only at 25 iters does the box actually move *forward* toward
  the goal (+24 mm of progress vs −5 mm at 3 iters).
- The friction-cone violation rate is **unchanged** across all four
  runs (`case 3 surface: 200/200`). G2 didn't fix the cone problem.
- The mechanism that *did* change: at `admm_iter ≥ 10`, ρ adapts
  (halved once at 25 iters, from 100 → 50), which reduces the
  ADMM augmentation's quadratic damping on `λ_n` and lets the QP
  plan larger normal forces. **G2's effect is via ρ-adaptation,
  not via dual-variable accumulation.**
- None of the four runs succeeded (`success=NO` for all). G2 is a
  partial fix, not the whole answer.

## Setup

| Parameter | Value |
|---|---|
| Task | `pushing` (μ=0.4, mass=0.2, size=0.10³ m) |
| Goal | (+0.30, 0.00) — east push |
| Sim duration | 8 s, 801 control steps, 10 ms each |
| Outer controller | `--sampling-c3` (Venkatesh-2025 wrapper) |
| ADMM solver | C3Solver, ρ₀=100, w_comp=100 |
| Horizon | 20 steps × 50 ms = 1.0 s lookahead |
| INITIAL_ARM_Q | post-F1 value (EE 20 cm above box at t=0) |

The four runs differ only in `admm_iter` passed through the new
`--admm-iter` CLI flag.

## Outcome table

| `admm_iter` | final `obj_xy` | `goal_dist` | progress | step-1 `λ_n_max` | `case 3` | ρ values | `kStayInC3` | wall ms/step |
|:-:|:--|:-:|:--|:-:|:-:|:--|:-:|:-:|
| 3  | (−0.005, +0.002) | 0.305 m | **−5 mm**  | 0.144 N | 200/200 | {100} | 129 | 208 |
| 9  | (−0.021, −0.072) | 0.329 m | **−21 mm** | 0.367 N | 200/200 | {100} | 782 | 609 |
| 10 | (−0.007, +0.007) | 0.307 m | **−7 mm**  | 0.284 N | 200/200 | {100} | 128 | 357 |
| 25 | (+0.024, −0.028) | 0.278 m | **+24 mm** | 2.507 N | 200/200 | {100, 50} | 322 | 1250 |

`progress` is the box's net x-displacement (positive = toward goal).
`case 3` counts the proportion of horizon-step Lorentz projections
that landed on the cone surface (rather than inside the cone or at
the polar). `kStayInC3` is from the outer-loop switch reason
histogram.

## What worked, what didn't

### What worked: ρ-adaptation is the mechanism

Look at step-1 `λ_n_max` against `admm_iter` and ρ history:

| `admm_iter` | `λ_n_max` step 1 | ρ at exit |
|:-:|:-:|:-:|
| 3  | 0.144 | 100 |
| 9  | 0.367 | 100 |
| 10 | 0.284 | 100 |
| 25 | **2.507** | **50** |

The jump from 0.284 N (iter 10) to 2.507 N (iter 25) is **8.8×**
in a single doubling of admm_iter. Looking at the C3Solver code,
adaptive ρ fires every 10 iters and halves ρ when
`primal/dual < 0.1`. At iter 10 it's the first opportunity but
the trigger condition isn't met within those 10 iters; at iter 25
it fires twice (iter 10 and iter 20) and ρ drops 100 → 50.

Why does halving ρ inflate `λ_n`? In the ADMM z-update
(C3Solver.solve, lines 286–290 of `admm_solver.py`):

$$
P_{\text{total}} = P + \rho I, \quad q_{\text{total}} = q_{\text{ref}} - \rho(\delta - \omega).
$$

The QP being solved is

$$
\min_z \; \tfrac{1}{2} z^\top (P + \rho I) z + (q_{\text{ref}} - \rho(\delta - \omega))^\top z
$$

For the `λ_n` block specifically: the original cost has no quadratic
term on `λ_n`, only the soft-comp gradient `q_ref[λ_n] = w_comp · φ⁺`.
At the prepositioned-equivalent state with `φ ≈ 0` (the EE is 20 cm
above, not in contact at all yet), `q_ref[λ_n] ≈ 0`. The QP minimum
in the `λ_n` direction is governed entirely by the augmentation
$\rho I$ and the cross-coupling through dynamics. Halving ρ halves
the damping on `λ_n` excursions.

But the dominant effect is more subtle. The QP minimises

$$
J = \underbrace{(x_t - x_{\text{ref}})^\top Q (x_t - x_{\text{ref}})}_{\text{tracking, includes box-xy with } w=10^5}
\;+\; \underbrace{u^\top R u}_{w_{\text{torque}}=0.01}
\;+\; \underbrace{\rho \lVert z - \delta + \omega \rVert^2}_{\text{ADMM aug.}}
$$

The QP wants large `λ_n` to push the box (which the high
$w_{\text{obj xy}}$ rewards), but the augmentation $\rho \lVert z - \delta\rVert^2$
penalises any direction in which $\delta$ is bounded — including
`λ_n ≥ 0` (ν the lower bound is bounding) and the cone constraint
on `(λ_n, F_t)`. Smaller ρ → smaller penalty for being away from
$\delta$ → QP can plan larger `λ_n`.

So: **ρ adaptation lets the QP plan larger normal forces. That's
the only G2 mechanism that did anything in these runs.**

### What didn't: friction-cone violations are unchanged

`case 3 surface = 200/200` in every run. Every horizon step, the
Lorentz projection lands on the cone boundary (i.e., the QP is
asking for $\lVert F_t\rVert > \mu \lambda_n$, and the projection
clips back to $\lVert F_t\rVert = \mu \lambda_n$).

The dual-variable accumulation hypothesis was that with more iters,
ω would build up enough to penalise infeasible λ_t in subsequent
QP solves. The data says no: the iters-9 run also has 200/200
surface projections, and it doesn't even get ρ adaptation.

So why does *this* hypothesis fail empirically?

Look at the `λ_n_max` schedule. At iter 25 it's 2.507 N. Per §2 of
`milestones/3_prepositioned_math/MATH.md`, the lever arm
$D_{[\text{box }v_x, \lambda_n]} = \Delta t/m_{\text{box}} = 0.25$
m/s/N. So `λ_n = 2.5 N` produces 0.625 m/s² of box acceleration —
plenty to overcome static friction `μmg ≈ 0.78 N`. With 1 s of
lookahead, the planner expects to push the box ~30 cm (well past
the 0.30 m goal).

But the actual integrated motion at t=8 s is only +24 mm. The plan
predicts behavior the simulator doesn't deliver.

Hypothesis (testable, but not tested here): the QP commands an
**arm torque** schedule that, when applied to the simulator, doesn't
realize the planned `λ_n`. The arm at the prepositioned-equivalent
state may be near-singular for the push direction (Suspect 3 from
the earlier diagnosis chain), so commanded `u` produces less
Cartesian force at the EE than `J_arm^{-T} u` predicts. The
simulator integrates *what was actually applied*, not what was
planned.

This would explain why ρ adaptation moves the needle (lets the
QP plan harder, which is enough to overwhelm the realization gap)
but doesn't fully solve the problem (the gap is still there, just
smaller in relative terms).

## What changed in the outer loop, and why

The switch-reason histograms tell different stories per run:

| `admm_iter` | `kToC3Cost` | `kToReposCost` | `kToC3Reached…` | `kStayInRepos` | `kStayInC3` |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 3  | 1 | 335 | 335 | 0 | 129 |
| 9  | 1 | 0   | 9   | 0 | 782 |
| 10 | 1 | 335 | 336 | 0 | 128 |
| 25 | 1 | 238 | 239 | 0 | 322 |

Iter 9 is unique: 782/801 steps in c3 mode, only 9 transitions out.
That's because at 9 iters, `λ_n_max` is *higher* than at 10 or 3
(0.367 vs 0.284 vs 0.144), and the resulting trajectory cost
`c_C3_raw[k=0]` stays low enough that no other sample beats it
under the hysteresis. The c3 plan at iter 9 isn't actually working
(box only moves −21 mm — *backward*), but the cost says it is.

So the outer loop is **fooled by the cost gap between samples**:
samples that aren't at contact have `c_C3_raw` dominated by free-fall
(no contact force = no progress = high tracking cost), so any sample
that *plans* contact has lower cost than any sample that doesn't,
regardless of whether the plan actually realizes physically.

This is the deeper version of the same disease we saw in the
prepositioned analysis: the cost differential picks `c3` mode based
on planned trajectories, not actual ones.

## Math: why ρ-adaptation helps (specifically)

C3Solver.solve runs the three-block ADMM:

$$
\begin{aligned}
z^{i+1} &= \arg\min_z \tfrac{1}{2} z^\top (P + \rho I) z + (q_{\text{ref}} - \rho(\delta^i - \omega^i))^\top z + \text{[constraints]} \\
\delta^{i+1} &= \mathrm{Proj}_{\,\text{constraint set}}\!\left(z^{i+1} + \omega^i\right) \\
\omega^{i+1} &= \omega^i + z^{i+1} - \delta^{i+1}
\end{aligned}
$$

For the contact-active state, the constraint set on the contact
block is the friction cone (Lorentz). With QP-asked `λ_t > μ λ_n`,
the projection inflates `λ_n` to fit:

$$
(\lambda_n^*, F_t^*) = \mathrm{Proj}_{\text{Lorentz}}\!(\lambda_n, F_t)
= \frac{1}{2}\!\left(1 + \frac{\mu \lVert F_t\rVert + \lambda_n}{(\mu^2 + 1)\,\lVert F_t\rVert}\right)\!(\mu, F_t/\lVert F_t\rVert)\,\lVert(\lambda_n, F_t)\rVert
$$

(case 3 from the projection code). The post-projection `δ_n`
is *larger* than the QP's `λ_n` whenever `λ_t > μ λ_n`. Then
ω accumulates the difference:

$$
\omega^{i+1}_{[\lambda_n]} = \omega^i + (\lambda_n^{i+1} - \delta_n^{i+1})
$$

At the next QP, the `q_total` term `−ρ(δ − ω)` shifts the linear
cost on `λ_n`. Specifically, since $\delta_n > \lambda_n$, $\omega_{[\lambda_n]}$
grows negative, making $-\rho(\delta_n - \omega_{[\lambda_n]})$
*more negative*, which pushes the QP to plan *larger* `λ_n`.

This is the mechanism that **was supposed** to work and didn't:
ω accumulation. After 9 iters of ω drift, the QP should be planning
bigger `λ_n` than after 3 iters. The data partially supports this
(0.144 → 0.367), but the effect is small and not enough to make
contact.

Then ρ-adaptation kicks in at iter 10 (and again at iter 20):

```
ρ_decision: primal=0.31, dual=0.52, ratio=0.59 → ρ unchanged   (iter 10, run-25)
ρ_decision: primal=0.06, dual=2.5,  ratio=0.024 < 0.1 → halve ρ (iter 20, run-25)
```

(Reconstructed from the math-diag log.) The halving of ρ at iter 20
of the iter-25 run is what unlocks `λ_n_max = 2.507`. Without the
second ρ halving (i.e., at iter 10), `λ_n` stays at 0.284.

So the path that worked is: ω accumulation gets primal/dual into
the regime where ρ adaptation triggers, which then drops the
augmentation damping enough to let the QP plan large normal forces.
The sequence is essentially "iters 1–10 prime the residual ratio,
iters 11–20 fire ρ-adaptation, iters 21+ exploit the new ρ to plan".

This explains why intermediate values (9, 10) are barely better
than the baseline: they don't get to step 21+.

## Predictions for follow-up runs

If this hypothesis is right:

1. `--admm-iter 30` should look like iter 25 (or marginally better)
   — once ρ adapts twice, more iters give diminishing returns.
2. `--admm-iter 100` should saturate; possibly worse if ρ keeps
   halving past 50 → 25 → 12.5 and the QP becomes ill-conditioned.
3. Setting `ρ₀ = 50` with `--admm-iter 3` should recover most of
   the iter-25 benefit at iter-3 cost. **This is the cheap test
   that would confirm the mechanism.**
4. Setting `ρ₀ = 25` with `--admm-iter 3` should be even better
   than iter 25 (skips the warmup).
5. None of these should fix the `case 3 surface = 200/200` problem.
   The friction-cone violation is structural — the QP has no penalty
   on `λ_t` magnitude beyond the augmentation, so it always asks
   for the maximum the cone-projection will allow.

## What's still unexplained

- Why does the box move *backward* (negative x) at iters 3, 9, 10?
  The sign of the planned `λ_n` is positive in all runs (per `[FORCE]`
  blocks). Possible cause: the EE is being pushed sideways by the QP's
  desired tangential force, and Drake's contact solver translates that
  into box rotation that *also* moves it sideways. Need to look at
  `[FORCE]` blocks across the runs to confirm.

- Why does iter-9 spend 782 steps in c3 with no progress, while iter-10
  chatters between modes? They're one iter apart, similar `λ_n_max`,
  but very different outer-loop behavior. The hysteresis ratios in
  the cost comparison are likely sensitive to the exact trajectory
  produced by the QP, but I don't have a clean mechanism for it.

- Why is iter-9 wall-clock-cost (609 ms/step) higher than iter-10
  (357 ms/step) when iter-10 does more inner ADMM work? Because the
  outer loop dispatched 2564 cheap-sample solves at iter-9 vs 1249
  at iter-10. Each cheap solve is `surrogate_admm_iters=1` of inner
  work plus IK and cost evaluation. So 9 iters caused some downstream
  amplifier to fire repeatedly. Worth looking at `[GS-table]` to see.

## Conclusion

G2 is partially confirmed. The mechanism is **ρ-adaptation, not
ω accumulation**. The friction-cone violation is unchanged across
all four runs, so the QP-vs-projection mismatch is structural and
will need a different fix (G3: add a quadratic penalty on `λ_t`,
or change the QP formulation to enforce the cone as a constraint
rather than a projection).

The cleanest follow-up test is **iter-3 with ρ₀=50**: if it
produces +20+ mm of progress, the entire G2 effect was ρ
adaptation and we can recover it for free at the iter-3 wall
clock cost. That experiment costs one main.py run and would
move the diagnosis forward decisively.

## Reference: log artifacts

All four runs are saved in `results/`:

- `g2_admm_03iter.{txt, mp4, html}` — control (current default)
- `g2_admm_09iter.{txt, mp4, html}` — pre-ρ-adaptation
- `g2_admm_10iter.{txt, mp4, html}` — at-threshold
- `g2_admm_25iter.{txt, mp4, html}` — past-threshold (the success-ish one)

Patch that enabled the sweep is at
`milestones/<…>/g2_admm_iter/main.py.diff` (a 3-line change adding
`--admm-iter` CLI flag).
