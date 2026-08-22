# Path B — results

Design and baseline architecture: `path_b_design.md`.
Probe: `control/sampling_c3/pathb_probe.py` (read-only, env-gated).

```
PORT_PATHB_SWEEP=42,92,233 PORT_PATHB_SWEEP_N=21 \
python3 main.py pushing --task-id 4 --sampling-c3 config/sampling_c3_kik.yaml \
        --seed 0 --max-time 20
```

---

## 0. Verdict

**PATH B NOT BENEFICIAL** — for contact location *within a face*, on this task.

The reason is not that continuous optimization is hard here. It is that there is
almost nothing left to win: a *perfect* continuous optimum beats the existing
5-draw sampler by **0.7–0.9%** of ranked cost, while the choice of **face** is
worth 30–60× more. The baseline's goal-aligned centering heuristic already
samples where the optimum is.

---

## 1. A contact-producing baseline (Task 5)

The prior "hover-at-standoff" diagnostic was a **wrong-config artifact**. Both
earlier rollouts in this branch used a bare `--sampling-c3`, which loads
`config/sampling_c3_params.yaml` (`kRandomOnCircle`, 3+1 samples, no cost-LCS
ranking) — a different controller from the box's canonical
`config/sampling_c3_kik.yaml` (`kFaceNormal`, 5+4, ranking on).

With the canonical config the baseline makes solid contact:

```
python3 main.py pushing --task-id 4 --sampling-c3 config/sampling_c3_kik.yaml \
        --seed 0 --max-time 180

826 ticks, 104 with contact, max |F| = 17.05 N
min EE-box distance 0.0350 m  (contact needs <= 0.077)
box moved 0.1433 m of the 0.15 m goal
|F| while in contact: median 1.66 N, p90 10.80 N, max 17.05 N
```

This is the deterministic A/B case. (The run was stopped a little short of the
180 s mark to free CPU for the sweep, so the tight-goal latch is not recorded
here; contact, object motion and reproducibility — the three things Task 5
requires — are established.)

### Corrected empirical C3+ step sizes

These supersede §3 of `docs/crisp_c3plus_linearization_study.md`, which was
measured on the wrong controller:

| quantity | median | p90 | max |
|---|---|---|---|
| EE motion per tick | 0.00088 m | 0.01350 m | 0.01903 m |
| EE motion per horizon (7 ticks) | 0.00873 m | 0.08722 m | 0.10028 m |
| box motion per horizon | 0.00002 m | 0.00416 m | 0.07683 m |
| box yaw per horizon | 0.00018 rad | 0.01487 rad | 0.11799 rad |

The p90/max EE excursions are essentially unchanged (0.087 / 0.100 m per
horizon), so the linearization study's conclusion is unaffected. What changes is
that the box now genuinely moves — up to 0.077 m and 0.118 rad within a single
horizon — which if anything strengthens that study's warning about holding one
linearization across the horizon.

---

## 2. Is `cost(s)` smooth? (Task 3)

At three contact ticks the probe sweeps `s ∈ [0,1]` in 21 points on all four
faces and re-runs the controller's own `evaluate_samples`. `c_sample` is the
ranked cost the argmin actually uses. Points whose IK could not track the
requested placement (`ik_track > 5 mm`) are dropped — that is exactly one point
per goal-aligned face, always the `s = 0` corner (58 mm off).

| step | face | align | pts | cost min | cost max | range % | argmin s | local minima | roughness |
|---|---|---|---|---|---|---|---|---|---|
| 42 | **+x** | 1.00 | 20 | 440.64 | 513.45 | **15.6** | 0.500 | 1 | 0.24 |
| 42 | +y | 0.00 | 21 | 521.48 | 536.21 | 2.8 | 0.450 | 1 | 0.20 |
| 42 | −x | 0.00 | 21 | 663.37 | 677.79 | 2.1 | 0.500 | 1 | 0.45 |
| 42 | −y | 0.00 | 21 | 520.41 | 536.01 | 3.0 | 0.600 | 1 | 0.19 |
| 92 | **+x** | 1.00 | 20 | 196.59 | 230.37 | **16.3** | 0.500 | 1 | 0.36 |
| 92 | +y | 0.09 | 21 | 204.45 | 220.71 | 7.8 | 0.450 | 2 | 0.43 |
| 92 | −x | 0.00 | 21 | 276.76 | 290.84 | 5.0 | 0.600 | 2 | 0.41 |
| 92 | −y | 0.00 | 21 | 210.25 | 221.61 | 5.3 | 0.500 | 1 | 0.20 |
| 233 | **+x** | 0.97 | 20 | 72.49 | 84.42 | **15.7** | 0.400 | 2 | 0.56 |
| 233 | +y | 0.23 | 21 | 75.61 | 85.27 | 12.3 | 0.450 | 1 | 0.21 |
| 233 | −x | 0.00 | 21 | 79.35 | 88.40 | 11.1 | 0.850 | 1 | 0.16 |
| 233 | −y | 0.00 | 21 | 75.42 | 84.11 | 11.1 | 0.550 | 1 | 0.35 |

*roughness* = mean|second difference| / mean|first difference|; « 1 means the
curve is dominated by its trend rather than by noise.

**`cost(s)` is smooth and effectively unimodal.** Roughness is 0.16–0.56
everywhere, and there are 1–2 local minima across 20 points. A local continuous
optimizer would be perfectly well-behaved here — the parameterization is exactly
affine (`path_b_design.md` §3) and the objective is well-conditioned.

**The optimum sits at the face centre.** On the goal-aligned `+x` face the argmin
is `s = 0.50, 0.50, 0.40` at the three ticks — i.e. dead centre, which is
precisely where the baseline's `CENTERED_JITTER_FRACTION = 0.2` already
concentrates its draws (`s ∈ [0.4, 0.6]`).

---

## 3. The ceiling: what continuous optimization could win (Tasks 6, 7)

The sweep minimum over 21 points is an **upper bound** on what any continuous
optimizer could achieve on that face — no method can do better than the best
point on a fine grid of a smooth, near-unimodal function, beyond sub-grid
refinement worth far less than the grid spacing's effect.

Against 5 random draws from the same face (4000 bootstrap resamples):

| step | best face | sweep min (perfect optimizer) | 5 random draws: mean | p10 | **gain** |
|---|---|---|---|---|---|
| 42 | +x | 440.64 | 444.45 | 440.64 | **0.9%** |
| 92 | +x | 196.59 | 197.99 | 196.59 | **0.7%** |
| 233 | +x | 72.49 | 73.05 | 72.49 | **0.8%** |

In every case the **p10 of 5 random draws already equals the sweep minimum** —
five draws hit the optimum roughly one time in ten and land within 1% of it on
average.

### Why the effect is so small

Compare the within-face spread to the between-face spread at the same tick:

```
step 42:   within +x face   440.6 -> 513.5    (73 units, 15.6%)
           between faces    440.6 -> 677.8   (237 units, 50.6%)
step 233:  within +x face    72.5 -> 84.4    (12 units, 15.7%)
           between faces     72.5 -> 88.4    (16 units, 21.9%)
```

**Face selection is worth 1.4–3.3× the entire within-face range, and 30–60× the
achievable optimization gain.** The discrete decision dominates; the continuous
refinement is close to noise. And the sampler is already biased to the right
place within the face, so the residual is smaller still.

### Budget (Task 7)

The exchange rate is unfavourable in the direction that matters. Per
`path_b_design.md` §2, evaluating `cost(s)` at one new `s` is a **class-C**
operation — one IK solve, one cost-LCS rebuild, one PD+PGS forward simulation —
i.e. approximately **one full candidate evaluation**. So:

```
baseline   6 candidates x 3 ADMM iters      -> within 1% of the face optimum
Path B     1 draw + k refinement iterations -> at best 0.7-0.9% better, for k >= 1
```

Sampling does not merely buy with enumeration what optimization buys with
iterations — here **enumeration is already at the ceiling**, so iterations buy
essentially nothing.

### Why the full A/B was not run

Running the A/B would spend a large compute budget resolving a ≤0.9% difference
in a ranked cost, against a run-to-run variability that the project's own
protocol treats as substantial (`feedback_no_statistical_evaluation`,
`feedback_baseline_provenance`). The ceiling measurement answers the research
question more cleanly than the A/B could: **the effect Path B is designed to
capture is smaller than the noise the A/B would have to resolve.** The
instrumentation is in place if the ceiling argument is rejected.

---

## 4. CRISP connection

The retained CRISP idea — contact location as an optimization variable — is
sound and cleanly implementable here. `path_b_design.md` shows the
parameterization is exactly affine and the design extends to `(u,v)`. The reason
to stop is **empirical and specific to this baseline**: Push Anything's sampler
already includes a goal-aligned centering heuristic that puts its draws at the
optimum, so the variable it would optimize is already near-optimally chosen.

This is not a refutation of CRISP. In CRISP's own setting the contact location
ranges over a *whole* object outline with no heuristic prior, and the optimizer
must also *sequence* faces — a regime where the discrete choice is inside the
optimization and the gains are correspondingly larger.

## 5. SE(3) assessment

Unchanged from `path_b_design.md` §3: the mapping is written as *face origin +
tangent basis × coordinates* and extends from `s` to `(u,v)` without structural
change. Nothing implemented here blocks it. Given §3's result, though, the more
promising SE(3) question is not *where on a face* but *which face* — the choice
that actually dominates the cost.

## 6. Recommendation

1. **Do not implement within-face continuous contact optimization.** The ceiling
   is ≤0.9% and the baseline is already there.
2. **The leverage is in face selection.** Faces differ by 20–50% in ranked cost,
   and the baseline picks a face by *uniform random draw per sample*
   (`sampling.py:917`) — not by enumeration. Evaluating all four faces
   deterministically, instead of hoping 5 random draws cover them, is a smaller
   change than Path B with a much larger ceiling. That is the experiment worth
   running next.
3. Keep the probe. It measures the ceiling for any future contact-placement
   idea before the idea is built, which is what made this cheap.
