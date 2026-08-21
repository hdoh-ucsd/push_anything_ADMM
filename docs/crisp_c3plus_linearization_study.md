# Is a locally linearized CRISP B-B model usable inside C3+?

**Question.** A C3+ adapter for CRISP Appendix B-B would have to linearize the
B-B dynamics about a nominal `(q̄, c̄, λ̄)` once per tick and hold that
linearization across the whole horizon. Is that accurate enough at the step
sizes C3+ actually takes?

**Scope.** Diagnosis only. No LCS adapter is implemented here, and
`control/crisp/push_box.py` dynamics are untouched — the study adds one public
accessor, `dynamics_jacobian()`, and reads the model as shipped.

Repro: `python3 tools/crisp/linearization_study.py --log <c3plus_run.log>`
Tests: `tests/test_crisp_linearization.py`

---

## 0. Answer

**LINEARIZATION MARGINAL — prototype only**, and the marginality is
channel-specific rather than diffuse.

- The remainder is **exactly** `k_rot(Δc_x Δλ_y − Δc_y Δλ_x)` in yaw plus an
  `O(Δθ²)` trig term in translation. Nothing else. Verified to machine precision.
- **Per-tick re-linearization is fine**: at the measured C3+ per-tick contact
  motion (6–16 mm) the relative error is 1.4–4.5%.
- **Holding one linearization across C3+'s 7-step horizon is not**: at the
  measured per-horizon contact motion (21–100 mm) it is 13–46% mean and up to
  160% — the yaw error becomes as large as the yaw signal. C3+ does exactly this.
- All of that error is in the **rotation** channel, which is also the channel that
  is rank-deficient at the all-zero nominal. Two independent findings landing on
  the same channel.
- Translation is comfortable throughout; if the task is `goal_yaw = 0` (our
  canonical box task) the linearization is not the binding constraint.

Because the remainder is closed-form, two mitigations are cheap and worth
prototyping before anything larger: **re-linearize per knot** rather than per
tick, or **add the exact cross term back** as a correction to the affine model.
Either removes the dominant error at negligible cost.

---

## 1. The error is exactly characterizable, not merely O(δ²)

This is the finding that decides the question, and it is structural rather than
empirical. B-B's dynamics are **bilinear**, not generically nonlinear:

```
f_px = k_t [ (λ₂+λ₄) cosθ − (λ₁+λ₃) sinθ ]      linear in λ, trigonometric in θ
f_py = k_t [ (λ₂+λ₄) sinθ + (λ₁+λ₃) cosθ ]
f_θ  = k_r [ c_x(λ₁+λ₃) − c_y(λ₂+λ₄) ]          exactly bilinear in (c, λ)
```

Three consequences follow immediately, and all three are verified numerically in
`tests/test_crisp_linearization.py`:

1. **`f_θ` does not depend on `q` at all.** Perturbing the pose — including θ —
   produces *zero* linearization error in the yaw row. Ever.
2. **A perturbation in `c` alone, or in `λ` alone, is reproduced exactly.** `f`
   is linear in each separately, so the first-order model is not an
   approximation in those directions — the error is zero to machine precision,
   not small.
3. **The entire remainder in the yaw row is one cross term**, with no
   higher-order tail:

   ```
   f_θ(c̄+Δc, λ̄+Δλ) − f_θ,lin  =  k_r ( Δc_x·Δλ_y − Δc_y·Δλ_x )
   ```

So the only sources of linearization error in the whole model are (i) the
`Δc·Δλ` cross term in yaw, and (ii) the trigonometric remainder in the two
translation rows, which is `O(Δθ²) + O(Δθ·Δλ)`.

That is a far better situation than a generic nonlinear contact model. It also
means the error is *predictable in closed form* — an adapter could bound it
without sampling, and could even add the exact cross term back as a correction.


## 2. Measured error structure (Tasks 1 and 3)

Perturbation sweep, 24 random draws per (nominal point, mode, magnitude), across
7 nominal points for our box and 9 for the reference. Max absolute error over all
points, at δ = 1e-2:

| mode | our box | reference box | where the error lands |
|---|---|---|---|
| `Δc` only | **1.4e-17** | **5.6e-17** | machine zero — exact |
| `Δλ` only | **2.8e-17** | **1.6e-16** | machine zero — exact |
| `Δθ` only | 3.3e-06 | 2.0e-05 | translation rows only; yaw exactly 0 |
| `Δc + Δλ` | **7.2e-04** | **1.3e-04** | **yaw row only**; px, py exactly 0 |
| `Δθ + Δλ` | 3.3e-05 | 4.8e-05 | translation rows only; yaw exactly 0 |
| all three | 7.2e-04 | 1.3e-04 | dominated by the yaw term |

The predicted structure is reproduced to machine precision. `Δc·Δλ` dominates
`Δθ·Δλ` by ~20x, and the two channels never mix: **all yaw error comes from
`Δc·Δλ`, all translation error from `Δθ`.**

The closed form is exact, not asymptotic. At the rotation nominal
(`c = (−0.05, 0.03)`, `λ₂ = 0.3`, `k_rot = 5.2232`), predicted `k_rot·δ²` against
measured:

```
delta     measured yaw err   predicted k_rot*d^2   rel. to |f_theta| = 0.0470
1e-03     +0.00001           +0.00001                0.0%
1e-02     +0.00052           +0.00052                1.1%
3e-02     +0.00470           +0.00470               10.0%
1e-01     +0.05223           +0.05223              111.1%
```

Relative error over all nominal points (mode = all):

| δ | our box: mean / max | reference: mean / max |
|---|---|---|
| 1e-4 | 0.02% / 0.24% | 0.004% / 0.05% |
| 1e-3 | 0.26% / 2.4% | 0.03% / 0.46% |
| **1e-2** | **4.5% / 30%** | 0.57% / 9.2% |
| 3e-2 | 13% / 66% | 1.8% / 14% |
| **1e-1** | **46% / 160%** | 6.9% / 53% |

Our 0.1 m box is roughly 7x worse than the reference 1.0 x 0.5 m box at the same
*absolute* δ, as expected: the same displacement is a much larger fraction of a
small box, and `k_rot` scales as 1/r.

Plots: `docs/figs/lin_error_{ours,ref}.png`, `docs/figs/lin_relerror.png`.

## 3. Empirical C3+ step sizes (Task 3, second half)

From a 90 s `pushing --sampling-c3` rollout, 1200 planner ticks, parsed from
`[GATE-CONTACT]`:

| quantity | median | p90 | max |
|---|---|---|---|
| EE motion per tick | 0.0059 m | 0.0145 m | 0.0162 m |
| **EE motion per horizon (7 ticks)** | **0.0212 m** | **0.0873 m** | **0.1003 m** |
| box motion per tick | 1e-5 m | 2e-5 m | 5e-5 m |
| box yaw per horizon | 1.2e-4 rad | 3.2e-4 rad | 6.0e-4 rad |

**Caveat, stated plainly: this rollout never made contact.** The end-effector
closed to 0.0912 m of the box centre against the ~0.075 m needed, and hovered —
the known hover-at-standoff class. So the box-motion and force rows are noise,
not measurements, and no empirical `Δλ` scale is available from the sim. The **EE
motion is a real measurement** and it is the one that matters, because the
end-effector position *is* the contact location `c`.

Force magnitudes therefore come from converged CRISP plans at the same geometry
instead: `|λ| ≈ 0.33` for our box, `≈ 1.48` for the reference. In the reference
plan the per-knot contact-point jump reaches **0.728 m** on a box with
`a = 0.5` — the contact teleport that follows from B-B having no continuity
constraint on `c`, now measured rather than argued.

### Reading the two together

| regime | δ | relative error |
|---|---|---|
| re-linearize **every tick** | 0.006 – 0.016 | **1.4% – 4.5%** (max 30%) |
| hold one linearization across the **horizon** | 0.021 – 0.100 | **13% – 46%** (max 160%) |

C3+ does the second. It builds one LCS per tick and uses it for all N = 7 steps.
At the p90 per-horizon excursion the yaw channel's linearization error is
comparable to the yaw signal itself.

## 4. Zero-initialization degeneracy (Task 4)

**Analytically.** `f_θ = k_rot(c_x λ_y − c_y λ_x)` is a product of two variables
that are both zero at the all-zero nominal, so

```
∂f_θ/∂c   = k_rot (λ_y, −λ_x)  = (0, 0)      at λ = 0
∂f_θ/∂λ   = k_rot (c_x, −c_y, c_x, −c_y) = 0  at c = 0
```

Both vanish simultaneously. **Numerically confirmed**: `dtheta_dc = [0, −0]`,
`dtheta_dlam = [0, −0, 0, −0]`, yaw row norm exactly 0.0, Jacobian **rank 2
instead of 3** — at both our and the reference geometry. Every other nominal
point tested has rank 3.

Rank is restored by *any* nonzero force, but usefully only by a nonzero **moment
arm**:

| seed | yaw row norm | rank |
|---|---|---|
| A all-zero | 0.0000 | 2 |
| B small λ, `c = 0` | 0.0052 | 3 |
| C face-centre `c`, λ | 0.3730 | 3 |
| D **offset** contact `c_y = b/2`, λ | 0.4162 | 3 |

### Does the seed change what the solve finds?

Pure-rotation goal, 90°, our box — diagnosis only, no warm-start policy implemented:

| seed | yaw reached | moved | faces | status |
|---|---|---|---|---|
| A all-zero | 0.0000 / 1.5708 | 0.0000 | — | reports **success**, does nothing |
| B small λ at `c = 0` | 0.2930 / 1.5708 | 0.0118 | −x | penalty max out |
| C face-centre `c` + λ | 0.0010 / 1.5708 | 0.0010 | −x | penalty max out |
| D **offset contact** + λ | **1.5403 / 1.5708** | 0.0084 | +y | penalty max out |

Seeding an *offset* contact takes a goal the all-zero guess cannot see at all to
**98% of target rotation**. Face-centre seeding (C) fails almost as badly as
all-zero, because `c_y = 0` gives zero moment arm — which is exactly what the
Jacobian analysis predicts. The mechanism is the moment arm, not merely nonzero
force.

On the translation goal the picture inverts: A, C and D all reach 0.145–0.150 m,
while B (λ seeded at `c = 0`) collapses to 0.0010 m — seeding force without
putting the contact on a face starts you at a violated face gate, which is worse
than starting feasible. **No single seed is good for both goals**, which is why
this stays diagnosis and not yet a policy.


---

## 5. Exclusivity options (Task 5)

CRISP enforces "only one face pushes at a time" with six pairwise constraints
`λ_i·λ_j = 0` (`SolvePushbox.cpp:83-99`, a separate
`pushboxContactSingleForceConstraints` function). **These cannot be represented
in C3+.** Our LCS row is `0 ≤ λ ⊥ E·x + F·λ + H·u + c ≥ 0`, which is *affine in
λ*; a product of two λ's cannot appear. And the ADMM δ-step projects onto
`C = {λ_n ≥ 0, ‖λ_t‖₂ ≤ μ·λ_n}`, a friction cone — not a "at most one component
nonzero" set. Neither the row structure nor the projection can express it.

| | **A. No exclusivity** | **B. Outer face selection** | **C. Soft penalty** |
|---|---|---|---|
| **Fidelity to CRISP** | Low — drops a constraint the reference explicitly adds; adjacent faces may push together, which physically means two pushers | High by construction — see below | Medium; exact only as weight → ∞ |
| **C3+ compatibility** | Perfect; face gates alone are exact LCP rows | Excellent; mirrors the existing sampler/dispatcher shape | **Poor** — a `λ_iλ_j` cost term is bilinear, so the QP Hessian becomes indefinite and ADMM's convex-QP step breaks |
| **Cost** | Zero | ≤ 4 structured solves/tick, *cheaper* than today's `num_additional_samples_c3: 5` + `repos: 4` | Cheap per iteration, but adds a weight to tune |
| **Linearization burden** | None beyond the dynamics | None — `c` is confined to one edge, a box constraint on one coordinate | Penalty must itself be linearized, re-introducing locality |
| **Contact discovery** | *Worse* — the solver blends faces rather than committing (measured: 17 face switches on the diagonal goal) | Cannot discover multi-face sequences within one solve | Best of the three: soft penalties let the iterate pass through blended states, i.e. a homotopy |
| **GPU batching** | Ideal | Ideal — 4 independent fixed-structure problems, which is the regime that won in the GPU-ADMM arc | Fine |

**Recommendation: B**, agreeing with the stated preference but for a sharper
reason than "it's convenient."

Under B the exclusivity constraint is not approximated — it is made
**structurally vacuous**. If only one face's λ is present in the problem at all,
there is nothing left to enforce. A discards the physics; C keeps the physics but
destroys the convexity that C3+'s ADMM step depends on. B is the only option that
keeps both.

The honest cost of B: the face choice leaves the optimizer and becomes a discrete
outer decision — exactly what Push Anything already does. So B **reduces** the
sampling layer (from N random perimeter draws to 4 structured, exactly-solved
face problems) but does not **replace** it. Any Phase-5 comparison must say so:
B-vs-sampling is a fair test of contact *placement*, and neither side does
contact *sequencing*.

---

## 6. Path A vs Path B (Task 6)

**Path A** — transplant full B-B dynamics into C3+.
**Path B** — keep C3+'s arm, friction and LCS dynamics; replace only the sampled
contact-point selection with continuous contact-location optimization.

| | Path A | Path B |
|---|---|---|
| Implementation effort | High — new adapter, exclusivity decision, new cost/weights, new OSC bridge, and re-adding everything B-B omits | Medium — add `c` to the decision vector; face gates become extra LCP rows |
| Model assumptions changed | **Many**: arm removed, pusher friction removed, velocities removed, planar only, quasi-static | **One**: contact location continuous instead of sampled |
| Fairness of the comparison | **Poor.** A 3-state quasi-static planner against a 19-state second-order one — any difference is attributable to the model, not the method | **Excellent.** Single-variable change, everything else held fixed |
| Solver difficulty | Bilinear dynamics need linearization; exclusivity unrepresentable; valid only quasi-statically | Contact Jacobians become functions of `c`, so a new bilinearity appears — not free either |
| Contact realism | Low — no tangential force, no pusher friction | High — unchanged |
| Path to SE(3) | Contact *parametrization* generalizes (c → ℝ³, 4 faces → 6, wrench `[f; r×f]`), but the quasi-static planar **limit surface does not** — SE(3) needs real rigid-body dynamics, so A's dynamics are a dead end | **Direct** — the LCS already carries a 6-DoF object pose; a continuous contact location on a 3-D surface is the natural next step |
| Research clarity | Muddled: "we replaced the whole model and results differ" is not a finding | Crisp: "does continuous contact-location optimization beat sampling, all else equal?" |

**Path B isolates the research question better on every axis except reproducing
CRISP itself.** The core question — *can continuous contact-location optimization
reduce or replace the sampled placement layer while preserving C3+ latency?* —
requires holding the dynamics fixed. Path A changes them, so it cannot answer it.

Path A retains one distinct value: it is the only way to check our B-B port
against the published benchmark. That is a validation exercise, not the research
programme, and §4.4 of `control/crisp/README.md` already covers most of it.

---

## 7. The iteration gap (Task 7)

| | CRISP reference | C3+ |
|---|---|---|
| Outer iterations | ~218 (measured, reference config) | **3, fixed** |
| Termination | Convergence-checked against `constraintTol = 1e-6` | None — never checks |
| Inner QP | Interior point (PIQP), solved accurately | OSQP, warm-started |
| Regime | Cold-start global-ish search | Warm-started local refinement |

One caveat on the headline "3 vs 218": they count different things. C3+ runs 3
*ADMM* iterations, but each spawns QPs whose own inner iterations are large — the
port's time budget measured ~8000 OSQP iterations per tick. The gap is in
**outer contact-mode reasoning**, not raw linear algebra.

What the gap implies:

- **Contact discovery.** Finding a genuinely new contact sequence means escaping
  a combinatorial local minimum. The reference needed 218 iterations to find a
  four-face sequence from an all-zero guess. Three iterations cannot do that, and
  no amount of tuning will change the order of magnitude. **This reframes Push
  Anything's sampler: it is the compensation for a low iteration budget.**
  Sampling buys, by enumeration, the mode exploration that an iterative solver
  would buy with iterations.
- **Local vs global.** C3+ is warm-started by the receding horizon and refines an
  existing plan; CRISP starts cold and searches. These are different regimes, and
  a method that wins in one need not win in the other.
- **Wall clock.** C3+'s simulated control period is 75 ms, but the measured
  wall-clock cost in this study's rollout was **176 ms/step** — the port already
  runs ~2.3× slower than real time. Any added per-tick cost lands on top of that.
- **MPC suitability.** At the reference's own horizon (N=100) a full CRISP solve
  took ~180 s. At MPC horizons (N=10) the authors report 80 ms in C++. Horizon
  length, not the algorithm, dominates.
- **Fairness.** Reporting "CRISP discovers a four-face sequence, C3+ does not"
  while the budgets differ by ~70× in outer iterations is not a fair comparison.
  Any Phase-5 result must either equalize the budget or report both budgets
  alongside the outcome. This is worth stating explicitly in any write-up,
  because it is the most likely reviewer objection.

---

## 8. Corrected CRISP reference constants

From `ComputationalRobotics/CRISP` @ `src/examples/pushbox/SolvePushbox.cpp:9-18`
and `:125-150`. **Keep `c` and `r` separate** — the SE(3) extension needs `r` to
generalize independently.

```
a = 0.5      b = 0.25     m = 1        mu = 0.5     g = 9.8
c = 0.4      r = sqrt(a^2 + b^2)       dt = 0.02    N = 100
Q = diag(100, 100, 100)   terminal knot only
R = diag(0.001) x 4       on the four lambdas only
```

No cost on the contact location. **No ½ factor** on the objective — the
reference accumulates `eᵀQe`, so a codebase using `½eᵀQe` must double `Q`.

## 9. Corrected 18-goal benchmark description

The benchmark is **SE(2)**, not a translation-direction sweep. The goal is

```
theta_i = i * 2*pi / 18 ,  i = 0..17
x_goal  = [ 3*cos(theta_i),  3*sin(theta_i),  theta_i ]
```

— the third component is that same `theta_i`, so target *orientation* rotates
with target *position*, up to a full turn. The shipped example hard-codes
`i = 12` (θ = 4.1888 rad = 240°).

## 10. N = 100 (code) vs N = 200 (paper)

`SolvePushbox.cpp:17` sets `const size_t N = 100`. The paper states "all other
examples are optimized over 200 steps with dt = 0.02 seconds" (§IV-B, following
the push-T horizon of 50). The discrepancy is unresolved and matters: at
dt = 0.02 the horizon is 2 s in code and 4 s in the text, which changes the force
magnitude needed to cover 3 m by 2×. **Use N = 100 and say so.**

## 11. Is the full LCS adapter worth implementing?

**No — not as specified, and not next.** Three independent lines converge:

1. The linearization is marginal precisely in the rotation channel (§0, §2), and
   C3+'s per-horizon step sizes sit in the bad regime (§3).
2. That same channel is rank-deficient at the natural initialization (§4), so the
   adapter would inherit a second rotation-specific pathology.
3. Path B isolates the actual research question better on every axis (§6), while
   Path A changes so many assumptions that a comparison against Push Anything
   could not attribute any result to the contact-location method.

**Recommended next step** is not the B-B adapter but the Path B prototype: keep
C3+'s dynamics, arm, and friction, and make the contact location a continuous
decision variable with outer face selection (§5, option B). That is a
single-variable change against the existing baseline, it answers the research
question directly, and it extends to SE(3) without discarding the model.

If B-B-in-C3+ is still wanted afterwards, do it in this order: per-knot
re-linearization first, exact cross-term correction second, and only then the
full adapter — and scope it to `goal_yaw = 0` tasks, where §0's translation-only
comfort holds.

One further caution for any Phase-4/5 comparison: the 18-goal benchmark is SE(2)
(§9), so it exercises the rotation channel on **every** target. This study says
that is the channel a linearized adapter handles worst.
