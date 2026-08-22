# CRISP Appendix B-B on our box push — implementation and measured delta

**What this is.** A working port of the Push Box contact-implicit formulation from
Li, Han, Kang, Ma & Yang, *On the Surprising Robustness of Sequential Convex
Optimization for Contact-Implicit Motion Planning* (arXiv:2502.01055v3), together
with the paper's solver (Algorithm 1), measured against our box task and then
audited line-by-line against the authors' released C++ implementation.

**Status: off-reference experimental arm.** This is not in the dairlib
sampling-C3 lineage the rest of the port conforms to, it is not wired into the
live control loop, and nothing reaches it unless explicitly constructed. The
standing conformance mission is untouched.

Code: `control/crisp/scp.py` (Algorithm 1), `control/crisp/push_box.py`
(eqs 52–68 + execution bridge). Tests: `tests/test_crisp_scp.py`,
`tests/test_crisp_push_box.py`. Repro: `tools/crisp/analyse_push_box.py`.

A reader-facing version of this report is published at
<https://claude.ai/code/artifact/3942579c-54ed-473e-bbd3-78d3a24b41b7>, source in
`control/crisp/report.html`. To revise it, edit that file and republish **passing
that URL as `url`** — republishing without it creates a second artifact.

---

## 0. Reference-source audit (2026-08-21)

The official implementation — <https://github.com/ComputationalRobotics/CRISP>,
RSS 2025 — was read after this port was written. Everything below is checked
against `src/examples/pushbox/SolvePushbox.cpp` and
`src/core/include/solver_core/SolverInterface.h`. **The port has been corrected
where it diverged; every number in §2–§5 was re-measured afterwards.**

### 0.1 Confirmed — inferences the source vindicates

| claim | reference evidence |
|---|---|
| `eps_c` is 1e-6, Table I's "1e6" is a typo | `SolverParameters.h:32` `constraintTol = 1e-6` |
| dynamics eqs 52–54 | `SolvePushbox.cpp:41-43` — identical term for term, including `1/(mu*m*g*c*r)` |
| `r_char = sqrt(a²+b²)` default | `SolvePushbox.cpp:14` |
| second-order correction form | `SolverInterface.h:240-241` sets `beq = -(c(z+p) − J·p)` — exactly our `c_eq_hat` |
| QP subproblem shape (slacks v/w/t, ∞-norm box trust region) | `SolverInterface.h:390-431` |
| per-constraint penalties | their `weightedMode`, and it is **ON** for push box (`SolvePushbox.cpp:197`) |
| all-zero initial guess | `SolvePushbox.cpp:190` `xInitialGuess.setZero()` |
| **PIQP is the inner QP** | `SolverInterface.h:9, 476-485`; README §Features names it explicitly |
| **no contact-point continuity constraint** | absent in *both* `SolvePushbox.cpp` and `SolvePushT.cpp` — §5 stands |

The PIQP confirmation matters most: §3.2's suspicion that our accuracy-vs-feasibility
gap is an inner-solver artefact is now a code-level fact, not an inference from
Remark 10.

### 0.2 Corrected — divergences found and fixed in this port

1. **Complementarity is an inequality, not an equality.** The reference writes
   `-(λ₁)(c_y+b) ≥ 0` and registers it through `addInequalityConstraint`
   (`SolvePushbox.cpp:68-99`). With λ ≥ 0 and the box bounds separately enforced
   the product is ≥ 0, so both encodings agree on the *true* constraint value —
   but inside the QP the inequality form leaves the **linearised** product free
   to go negative, while our equality form penalised both sides. Now
   `complementarity_as_inequality=True` by default.
2. **We invented a cost on the contact point.** Reference `R` is
   `diag(0.001)` on the four λ only (`SolvePushbox.cpp:132-137`); there is no
   cost on `c_x, c_y`. Our `r_contact = 1e-3` pulled the contact toward the box
   centre and so *deepened the very barrier documented in §3.1*. Now `0.0`.
3. **The trust region is not reset after a penalty escalation.** We inferred a
   reset to `Δ₀` and called it "standard exact-penalty behaviour". The reference
   ships that line **commented out** — `SolverInterface.h:554`
   `// trustRegionRadius_ = trustRegionInitRadius_;` — refreshing only the merit
   and model under the new penalties. Now switchable via
   `reset_trust_region_on_escalation`; see §0.4 for what it costs.

### 0.3 Flagged — divergences kept, with reasons

| # | reference | this port | note |
|---|---|---|---|
| 1 | `muMax = 1e8` (`SolverParameters.h:27`) | `mu_max = 1e6` | Table I prints 1e6; the **code disagrees with the paper** |
| 2 | `maxIterations = 5000` | `k_max = 1000` | same — Table I prints 1000 |
| 3 | convergence on `‖p‖₂/‖x‖₂ < trailTol` (`:508`) | `‖p‖∞ < eps_p` | theirs is **relative to the iterate**; ill-defined at the all-zero start (‖x‖ = 0) |
| 4 | also rejects when `reduction_actual < 1e-9` (`:490`) | rejects on `ared < 0` only | |
| 5 | escalates constraints above `10 × constraintTol` (`weightedTolFactor`) | escalates above `eps_c` | leaves a 1e-6…1e-5 band that blocks convergence without escalating |
| 6 | objective is `eᵀQe` | `½ eᵀQe` | so their `Q = 100` ≡ our `q_pos = 200` |

Defaults 1 and 2 are left at the *published* values so the port matches the
paper as written; the discrepancy is the paper's, and both are constructor
arguments.

### 0.4 What the source changes about the findings

**The push-box benchmark demands 240° of rotation.** `SolvePushbox.cpp:199-200`
sets `theta = 12·2π/18` and the goal to `[3cos θ, 3sin θ, θ]` — the *third*
component is that same θ = 4.1888 rad. So the paper's own push-box benchmark is
"translate 3 m **and rotate 240°**", which materially narrows §4.1: our
"rotation is invisible" result is about **pure** rotation with no travel, a case
the reference never tests. Once translation is present, λ leaves zero, the
`c × f` moment arm acquires a gradient, and rotation becomes reachable — at the
reference config this port turns the box 2.60 rad.

**Our benchmark constants were guesses, and three were wrong**: reference
`mu = 0.5` (we used 0.3), `c = 0.4` (we used 0.6), `g = 9.8`, and **`N = 100`**
— the paper's text says 200 steps, the code says 100.

**The shipped example solves one target, not eighteen.** The `for` loop over
`num_segments = 18` has been deleted, leaving orphaned indentation and a stray
brace (`SolvePushbox.cpp:198-205`): as committed it solves segment 12 only. The
repository therefore does not ship the sweep that produces Table II's 100%
push-box success rate.

**The authors acknowledge the sensitivity** documented in §3: the README closes
its example list with *"Feel free to try different hyperparameters, and the
weighted mode… For local solver, the hyperparameters are important for the
numerical performance."*

**`constraintTol` is loosened to 1e-3 for push T** (`SolvePushT.cpp:401`) — a
1000× looser feasibility bar than push box's 1e-6, worth knowing when reading
Table II's per-task violations.

---

## 1. What B-B actually models

| | our stack (sampling-C3 / LCS / ADMM C3+) | CRISP B-B |
|---|---|---|
| planner state | `n_x = 19` — `[box_q(7), p_ee(3), box_v(6), v_ee(3)]` | **3** — `p_x, p_y, θ` |
| planner control | `n_u = 3` (EE-space) | **6** — `c_x, c_y` + 4 face forces |
| contact geometry | witness points from mesh/box geometry, admitted by a 2 mm Drake signed-distance query | 4 analytic face half-planes; the contact point is a **decision variable** |
| ground friction | explicit ground-witness rows (4-corner box / 3-sphere mesh tables) | ellipsoidal **limit surface**, no witnesses at all |
| which face to push | chosen upstream by the sampler (`kMeshNormal`, face tables, projection) | an **output of the solve** (complementarity eqs 58–61) |
| pusher | in the model (EE state, μ_EE-BOX = 0.42, tangential Jacobians `J_t`) | **not in the model**; normal force only |
| arm | in the model (7-DoF, torque limits, null space) | absent |
| dynamics | second-order LCS with velocities | quasi-static, first-order, planar |

The consequence that matters: **B-B deletes the entire witness-point and sampler
apparatus** by construction, because the object's contact geometry enters as four
inequalities on a decision variable rather than as sampled points on a mesh.

---

## 2. It works on our canonical box task

`config/tasks.yaml: pushing` is a 0.1 m cube, μ_gnd 0.46, (0.45, 0) → (0.60, 0),
`goal_yaw: 0` — a 0.15 m pure translation. N = 40, dt = 0.05 (2 s horizon),
all-zero initial guess:

```
canonical  +x 0.15m   optimization successful  iters=17  2.39 s
                      pos_err=4.57 mm  yaw_err=0.0000 rad
                      faces=['-x']  switches=0  peak EE speed 0.07 m/s
reverse    -x 0.15m   optimization successful  iters=17  2.35 s
                      pos_err=4.57 mm  faces=['+x']
```

The face choice is genuinely an output: reversing the goal flips the chosen face
from `-x` to `+x` with no other change. Terminal accuracy is 4.6 mm against our
tight gate of 20 mm, and the plan asks the arm for 0.07 m/s — trivially
executable.

**This is the easiest possible case for B-B**: one face, no switching, no
rotation. Our canonical box task never exercises the part of the formulation
that is hard.

---

## 3. A calibration trap and a validation gap

### 3.1 The all-zero guess can silently do nothing

The all-zero point is *feasible*: no force, no motion, every complementarity
product zero. Raising a face force off zero costs linearised face-gate violation,
because at `λ = 0` the gate's own contact-point gradient
`∂/∂c_x [λ_2x (c_x + a)] = λ_2x` is itself zero — the trial step cannot slide the
contact onto the face for free. Per knot and per unit force the trade is

```
penalty   μ_0 · a                    benefit   q · d · dt · k_trans
```

which is knot-count independent, so the origin is escapable iff

```
q > μ_0 · a / (d · dt · k_trans)          (min_terminal_weight())
```

For our cube this predicts **q\* = 300.8**. The barrier is real but the
threshold is a **scale, not a sharp switch** — under the reference-faithful
encoding the transition is non-monotone:

```
q=150 (< q*)  parked   moved=0.0000 m   viol 1.4e-17  converged
q=250 (< q*)  ESCAPED  moved=0.1048 m   viol 6.5e-02  penalty max out
q=290 (< q*)  parked   moved=0.0000 m   viol 6.7e-17  converged
q=310 (> q*)  parked   moved=0.0045 m   viol 3.9e-03  penalty max out
q=400 (> q*)  ESCAPED  moved=0.1492 m   viol 4.9e-17  converged
q=600 (> q*)  ESCAPED  moved=0.1495 m   viol 3.7e-17  converged
```

**Retracted:** an earlier revision of this note reported a clean 290-parks /
310-escapes bracket. That measurement used the equality-form complementarity and
the port-only contact-point cost, both since corrected (§0.2); under the
reference formulation q = 250 escapes and q = 310 does not. What survives is the
qualitative claim — well below q\* the box does not move at all, well above it
the goal is reached — and it is enough to make the failure mode dangerous,
because at q = 150 and q = 290 the solver returns a **feasible, converged,
useless** plan and reports success. That is the same "feasible but impractical"
outcome the paper criticises IPOPT for in its transport example. Since the paper
publishes no Q or R for push box, anyone reproducing this rediscovers the
threshold; we auto-size to 10× q\* and expose `min_terminal_weight()`.

### 3.2 Accuracy and feasibility still trade — and the inner QP is why

At the paper's own benchmark scale and its own constants (3 m target, N = 200,
dt = 0.02, a = 0.5, mu = 0.5, c = 0.4), the auto weight (10x q\*) now gives:

```
moved 2.9592 m of 3.000 m   err 0.0408 m (1.36%)   viol 7.53e-05
faces=['-x']   iters=65   78 s
```

That is a large improvement on the pre-audit measurement, which reached only
2.633 m (12.22% error) and needed a 100x weight to get to 1.36%. Correcting the
constants and the encoding (§0.2) bought it. Sweeping `r_lambda` across two
decades still changes nothing, so the binding trade remains the terminal cost
against the l1 penalty on the complementarity residual, not the control cost.

**But Table II is still not reproduced, and the failure is scale-dependent.**
Sweeping the terminal weight at both scales:

```
bench 3 m   q x10    err 0.0408 m (1.36%)   viol 7.5e-05   inner QP failed
            q x100   err 0.2449 m (8.16%)   viol 1.7e+00   penalty max out
            q x1000  err 0.0000 m (0.00%)   viol 1.3e+00   inner QP failed
our box     q x10    err 0.0046 m (3.05%)   viol 2.1e-14   converged
            q x100   err 0.0006 m (0.43%)   viol 6.7e-02   penalty max out
            q x1000  err 0.0000 m (0.00%)   viol 1.0e-12   converged
```

On **our** box the trade is escapable: q x1000 delivers a perfect trajectory
*and* 1e-12 feasibility. At the reference's 3 m scale no weight tested delivers
both, and the sweep is not even monotone — x100 is worse than x10 on *both*
axes. The paper reports push box at tracking error 0.02 with violation 8.3e-9
and its own `constraintTol` is 1e-6; our best feasible benchmark point is
7.53e-5.

The inner QP is now the prime suspect on evidence rather than inference. The
reference uses interior-point PIQP (`SolverInterface.h:9`, README §Features);
this port uses OSQP, a first-order ADMM solver — exactly what Remark 10 warns
against. Under the now-faithful formulation OSQP needs ~4x the iterations and
stops converging at short horizons (§5) on a problem the reference solves
routinely. Q and R are no longer guesses (§0.2), so the weighting explanation is
spent; the solver explanation is not.

---

## 4. Where B-B stops

Everything above is the single-face case. Multi-face sequencing does happen —
at the reference's own configuration this port discovers a **four-face** contact
sequence from an all-zero guess (§4.4) — but on our small box it does not
converge, and pure rotation is unreachable outright.

### 4.1 Pure rotation is invisible to an all-zero guess (proven, not observed)

Yaw enters the dynamics as `θ̇ = k_rot (c_x f_y − c_y f_x)` — a product of two
variables that both start at zero. At the all-zero guess the Jacobian is

```
d(ṡ)/du  =  [ 0  0  0       0.2216  0       0.2216 ]   <- translation, first order
            [ 0  0  0.2216  0       0.2216  0      ]
            [ 0  0  0       0       0       0      ]   <- rotation, IDENTICALLY ZERO
```

Translation is first-order in the step (it needs only a force); rotation is
second-order (it needs a force *and* a nonzero moment arm). So a pure-rotation
goal has no first-order descent direction at all, and the solve terminates at a
genuine stationary point:

```
pure yaw 90deg   q ×10 / ×100 / ×1000   pos_err 0.00 mm, yaw_err 1.5708 rad
                                        faces=['-'],  optimization successful
pure yaw 20deg   q ×10 / ×100 / ×1000   pos_err 0.00 mm, yaw_err 0.3500 rad
                                        faces=['-'],  optimization successful
```

(Re-measured under the reference-faithful formulation of §0.2 — **unchanged to
four decimals**. This is the one §4 finding the reference audit left untouched,
as expected: it is a property of the dynamics both implementations share.)

The full goal error, zero motion, no face ever activated — reported as
**success**, at every weight tested. This is completely insensitive to the knob
that fixes the translation case, which is what distinguishes it from §3.1.
Warm-starting the contact onto a face (`c_x = −a`, small force) restores the
gradient and the solver does move (20° goal: 0.35 → 0.2752 rad residual), though
it then fails to converge.

Note our canonical box task has `goal_yaw: 0`, so it never touches the rotation
channel — which is exactly why §2 looks so clean. Any orientation-carrying task
(the T, the jack, most of Fig 8) would land here. In receding-horizon use the
cold start only bites on the first tick, but a box at rest with a do-nothing plan
never generates the motion that would warm-start the next one.

### 4.2 Two-face goals do not converge

A diagonal goal needs both the `-x` and `-y` faces, and the exclusivity
constraints (62–67) allow only one to act per knot, so it must be sequenced:

```
diagonal +x+y 0.10m (goal norm 141 mm)
  q ×10    pos_err 126.73 mm   faces=['-x','-y']   penalty max out
  q ×100   pos_err  88.95 mm   faces=['-x','-y']   penalty max out
  q ×1000  pos_err  49.93 mm   faces=['-x','-y']   penalty max out
```

Raising the weight helps monotonically — and under the reference formulation it
helps considerably more than the equality form managed (49.9 mm at ×1000 against
the 87.9 mm first measured) — but it never converges. The solve smears force
across two faces (17 face switches at ×10) instead of committing to a sequence.

### 4.3 Translation plus rotation demands an arm we do not have

```
translate +x 0.15m and rotate 90deg
  pos_err 55.62 mm, yaw_err 0.0325 rad, penalty max out
  peak commanded EE speed 1.76 m/s (88 mm per 50 ms knot)
```

It nearly achieves the rotation but violates constraints, and the plan asks the
end-effector for **1.76 m/s** — 25× the 0.07 m/s of the canonical push. Part of
that is the rotating box carrying the contact point around, which a real pusher
must genuinely track.

### 4.4 At the reference's own configuration, sequencing works — feasibility does not

`tools/crisp/analyse_push_box.py refcfg` runs `SolvePushbox.cpp` verbatim
(a = 0.5, b = 0.25, μ = 0.5, c = 0.4, g = 9.8, N = 100, dt = 0.02, Q = 100,
R = 0.001 on λ, all-zero guess, goal 3 m at 240° **including** 240° of box
rotation):

```
penalty max out   iters=218   180 s   viol 6.91e-01
  moved 2.3227 m of 3.000 m      turned 3.6877 rad of 4.1888 rad
  faces = ['+x', '+y', '-x', '-y']
```

This is the paper's headline claim reproduced qualitatively: **an entirely new
four-face contact sequence discovered from an all-zero initialization**, carrying
the box 77% of the way and through 88% of a 240° turn. What it does not
reproduce is feasibility — violation 0.69, against the reference's own
`constraintTol` of 1e-6. Under the earlier non-conformant encoding the same
configuration reached only 1.01 m and two faces, so §0.2's corrections bought
most of that ground.

The remaining gap now sits almost entirely on the flags of §0.3 plus the inner
QP: OSQP against PIQP.

---

## 5. What executing a B-B plan would require

B-B plans a contact point on the box *surface*; a finite pusher sits one radius
outside it — the same offset our sampler calls `sampling_setback`.
`to_execution_plan()` performs that conversion, emitting the EE waypoints and
world-frame push forces that `OperationalSpaceController.compute_torque` already
consumes as `p_ee_desired` and `lambda_des`. So the executor layer transfers
unchanged.

What does **not** transfer:

1. **No continuity constraint on the contact point.** Appendix B-B constrains
   `c_k` only through the face gates (58–61) and the box bounds; nothing links
   `c_k` to `c_{k+1}`. The plan may teleport the pusher between faces at no
   modelled cost. Our whole reposition apparatus — PWL tracker, lift-traverse-
   descend, `finished_reposition_cost`, the c3/free mode switch and its
   hysteresis — exists to pay exactly that bill. B-B does not remove that
   problem; it stops accounting for it.
2. **No pusher friction.** The paper states one contact point "with only the
   corresponding normal force applied". Our LCS carries EE-BOX μ = 0.42 and
   tangential Jacobians. B-B cannot represent sticking contact or the tangential
   force our pusher actually applies.
3. **No arm, no reachability, no workspace bounds, no torque limits.** The
   joint2 null-space pin, the wrist-saturation reach limit, and the workspace
   abort all live outside the model.
4. **Quasi-static and planar.** No velocities, no momentum, no tumble — our
   above-COM contact tumble analysis is unrepresentable — and no out-of-plane
   tilt, which is precisely the Fig 8 rot-error floor.

### Real-time budget

Our control loop runs at 13.3 Hz — N = 7, dt = 0.075, **75 ms** per tick.

```
N= 7 dt=0.075   1511.5 ms/solve   iters=30   inner QP failed   <- our horizon
N=10 dt=0.05    3702.7 ms/solve   iters=50   inner QP failed
N=20 dt=0.05    1049.6 ms/solve   iters=20   optimization successful
```

**These got worse, not better, under the reference-faithful encoding** —
previously 166.8 / 443.5 / 1044.7 ms, all converging. Caching the inequality
Jacobian's sparsity pattern changed nothing, so the cost is iteration count, not
assembly: N = 7 now needs 30 iterations where the equality form needed 7, and
both short horizons stop converging. The inequality form hands OSQP a ~2.3x
larger constraint block and a visibly harder problem — itself further evidence
for the inner-QP hypothesis of §3.2, since the reference runs this exact
formulation through PIQP without trouble.

At our own horizon this Python + OSQP implementation is therefore **20x over
budget and not converging**. The paper's C++ implementation solves the *harder*
push-T problem (8 faces, 29 states) at 80 ms per MPC cycle with PIQP and a
code-generated CppAD derivative library, so a real-time B-B is plausible — but
nothing in this implementation demonstrates it.

---

## 6. Verdict

For the canonical box task B-B is accurate (4.6 mm against a 20 mm gate), fast
to converge (17 iterations), and needs no sampler, no witness table, and no mode
switch. That is
a real simplification of a subsystem that has cost us several root-cause arcs
(the 3-sphere ground triangle falsification, the VHACD-bevel sampler starvation,
the face-table bug).

Multi-face sequencing is real — at the reference's own configuration this port
finds a four-face sequence from an all-zero guess (§4.4) — but on our small box
it does not converge (§4.2), and **pure** rotation is unreachable outright and
provably so (§4.1), returning success having done nothing. B-B also models
strictly less than our LCS: no pusher friction, no arm, no velocities, no
out-of-plane freedom, and no cost for moving the pusher between faces.

So: a planner for *where to push*, not a replacement for the stack. The
defensible use is as an upstream contact-sequence proposer feeding our existing
OSC — replacing the sampler and mode switch, not the LCS executor.

Before that is worth pursuing, two things must be settled, in this order:

1. **Re-run against an interior-point inner QP.** No longer a hypothesis: the
   reference uses PIQP (`SolverInterface.h:9`), and under the now-faithful
   formulation our OSQP build needs 4x the iterations and stops converging at
   short horizons (§5) while the reference runs the same problem fine. Until the
   inner solver is off the suspect list, none of the §4 limits — except the
   rotation Jacobian result, which is formulation-independent — is safely
   attributable to B-B itself. Drake ships Clarabel and SCS.
2. **Decide what pays for repositioning.** B-B leaves `c_k` unconstrained
   between knots, so it does not model the cost our whole dispatcher exists to
   pay. A contact-point rate constraint would be the minimal honest addition,
   and it is off-reference for both lineages.
