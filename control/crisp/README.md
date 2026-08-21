# CRISP Appendix B-B on our box push — implementation and measured delta

**What this is.** A working port of the Push Box contact-implicit formulation from
Li, Han, Kang, Ma & Yang, *On the Surprising Robustness of Sequential Convex
Optimization for Contact-Implicit Motion Planning* (arXiv:2502.01055v3), together
with the paper's solver (Algorithm 1), measured against our box task.

**Status: off-reference experimental arm.** This is not in the dairlib
sampling-C3 lineage the rest of the port conforms to, it is not wired into the
live control loop, and nothing reaches it unless explicitly constructed. The
standing conformance mission is untouched.

Code: `control/crisp/scp.py` (Algorithm 1), `control/crisp/push_box.py`
(eqs 52–68 + execution bridge). Tests: `tests/test_crisp_scp.py`,
`tests/test_crisp_push_box.py`. Repro: `tools/crisp/analyse_push_box.py`.

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
canonical  +x 0.15m   optimization successful  iters=15  3.08 s
                      pos_err=0.10 mm  yaw_err=0.0000 rad
                      faces=['-x']  switches=0  peak EE speed 0.07 m/s
reverse    -x 0.15m   optimization successful  iters=15  3.14 s
                      pos_err=0.10 mm  faces=['+x']
```

The face choice is genuinely an output: reversing the goal flips the chosen face
from `-x` to `+x` with no other change. Terminal accuracy is 0.1 mm against our
tight gate of 20 mm, and the plan asks the arm for 0.07 m/s — trivially
executable.

**This is the easiest possible case for B-B**: one face, no switching, no
rotation. Our canonical box task never exercises the part of the formulation
that is hard.

---

## 3. Two calibration traps, both measured

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

For our cube this predicts **q\* = 300.8**, and the measurement brackets it
sharply:

```
q=290 (< q*)  parked   moved=0.0000 m
q=310 (> q*)  ESCAPED  moved=0.1490 m   (goal 0.15 m)
```

Below the threshold the solver returns a **feasible, converged, useless** plan
and reports success — the same "feasible but impractical" outcome the paper
criticises IPOPT for in its transport example. The paper does not publish Q or R
for push box, so anyone reproducing this has to rediscover the threshold. We
auto-size the terminal weight to 10× q\* and expose `min_terminal_weight()`.

### 3.2 Tracking accuracy is set against μ, not against R

At the paper's own benchmark scale (3 m target, N = 200, dt = 0.02):

```
q ×10    err 0.3666 m (12.22%)   viol < 1e-6      optimization successful
q ×100   err 0.0408 m ( 1.36%)   viol 1.86e-3     inner QP failed
q ×1000  err 0.0041 m ( 0.14%)   viol > 1e-6      inner QP failed
```

and `r_lambda` swept 1e-2 → 1e-4 changes the answer by **nothing** (0.3666 m to
four decimals throughout). The binding trade is the terminal cost against the ℓ1
penalty on the complementarity residual, not against the control cost.

**This implementation does not reproduce Table II.** The paper reports push box
at 100% success, tracking error 0.02, and constraint violation 8.3e-9
simultaneously. We get accuracy *or* feasibility, never both: at ×10 the
violation is clean but tracking is 12%, and at ×100 tracking is respectable while
the violation blows out — 1.86e-3 pushing +x, and **1.241** pushing +y (err
0.0041 m, 51 iterations). The higher the terminal weight, the better the
trajectory looks and the less it satisfies its own constraints.

The paper predicts this failure precisely. Remark 10 notes that first-order QP
solvers give "insufficient solution quality ... [that] may hurt motion planning",
and says CRISP therefore "chose the interior-point solver PIQP to make sure the
inner QPs are solved to sufficient accuracy". **We used OSQP, which is exactly
the first-order ADMM solver that remark warns against.** So the most likely cause
of the gap is the inner QP, not the formulation — and testing that is the first
thing to do before drawing any conclusion about B-B's ceiling. (Q and R are also
unpublished for push box, so both are guesses here.)

`tests/test_crisp_push_box.py` marks the tracking-quality assertion `xfail` with
this reason, so the gap stays visible rather than being asserted away.

---

## 4. Where B-B stops

Everything above is the single-face case. The moment a goal needs the solver to
*sequence* faces, it stops working — and not because of the weights.

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
  q ×100   pos_err  92.22 mm   faces=['-x','-y']   penalty max out
  q ×1000  pos_err  87.91 mm   faces=['-x','-y']   penalty max out
  warm     pos_err 139.61 mm   faces=['-x']        penalty max out
```

Raising the weight helps monotonically but never converges; the solve smears
force across two faces (19 face switches at ×10) instead of committing to a
sequence. Warm-starting onto one face makes it worse.

### 4.3 Translation plus rotation demands an arm we do not have

```
translate +x 0.15m and rotate 90deg
  pos_err 34.37 mm, yaw_err 0.0513 rad, penalty max out
  peak commanded EE speed 1.16 m/s (58.2 mm per 50 ms knot)
```

It nearly achieves the rotation but violates constraints, and the plan asks the
end-effector for **1.16 m/s** — 17× the 0.07 m/s of the canonical push. Part of
that is the rotating box carrying the contact point around, which a real pusher
must genuinely track.

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
N= 7 dt=0.075    166.8 ms/solve   iters= 7    <- our exact horizon
N=10 dt=0.05     443.5 ms/solve   iters=14
N=20 dt=0.05    1044.7 ms/solve   iters=15
```

At our own horizon this Python + OSQP implementation is **2.2× over budget**. The
paper's C++ implementation solves the harder push-T problem (8 faces, 29 states)
at 80 ms per MPC cycle with PIQP and a code-generated CppAD derivative library,
so a real-time B-B is plausible — but not in this implementation.

---

## 6. Verdict

For the canonical box task B-B is accurate (0.1 mm), fast to converge (15
iterations), and needs no sampler, no witness table, and no mode switch. That is
a real simplification of a subsystem that has cost us several root-cause arcs
(the 3-sphere ground triangle falsification, the VHACD-bevel sampler starvation,
the face-table bug).

But that case — one face, no rotation — is the whole of what works here. The
moment a goal needs a contact *sequence*, this port either fails to converge
(§4.2) or reports success having done nothing (§4.1), and the rotation limit is
provable rather than tunable. B-B also models strictly less than our LCS: no
pusher friction, no arm, no velocities, no out-of-plane freedom, and no cost for
moving the pusher between faces.

So: a planner for *where to push*, not a replacement for the stack. The
defensible use is as an upstream contact-sequence proposer feeding our existing
OSC — replacing the sampler and mode switch, not the LCS executor.

Before that is worth pursuing, two things must be settled, in this order:

1. **Re-run against an interior-point inner QP** (PIQP, or Drake's Clarabel /
   SCS), per §3.2. The paper's Remark 10 predicts our accuracy-vs-feasibility
   tension as a first-order-solver artefact, so the §4 limits are not safely
   attributable to B-B until the inner solver is off the suspect list.
2. **Decide what pays for repositioning.** B-B leaves `c_k` unconstrained
   between knots, so it does not model the cost our whole dispatcher exists to
   pay. A contact-point rate constraint would be the minimal honest addition,
   and it is off-reference for both lineages.
