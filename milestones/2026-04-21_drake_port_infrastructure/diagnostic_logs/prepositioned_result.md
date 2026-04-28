# Pre-positioned Diagnostic Result
# Generated: 2026-04-20

## Outcome: 2 — Contact established, box moved in wrong direction (ADMM divergence)

## IK Solution (scripts/find_prepositioned_q.py)
Best seed: seed 4, err=4.05mm
PREPOSITIONED_ARM_Q = [+0.652976, +1.565831, +0.947045, -1.532373, +0.559969, +1.786368, +0.785000]
Actual pusher at t=0: (-0.072, -0.003, 0.050) — 3mm y-offset, on target in x/z
Contact at t=0: nhat_onto_box=[1, 0, 0], distance=-0.00270m (2.7mm penetration, in contact)

## Key metrics
- First contact time: t=0.00s (arm started in contact, [SANITY] nc=1 at step 1)
- Contact steps: 2000 / 2000 (geometry pair detected throughout)
  - λ_n_max > 0: steps 1–~350 (t=0–3.5s)
  - λ_n_max = 0: steps ~350–2000 (arm lost useful contact force, box frozen off-axis)
- Initial box: (0.000, 0.000) at z=0.050
- Final box: (0.006, -0.476) at z=0.154  (t=18s–20s plateau)
- Initial goal_dist: 0.300 m
- Final goal_dist: 0.560 m
- Progress: -86.7%  (box moved AWAY from goal)
- ADMM median iters: 80 / 80  (cap hit at every step)
- Fraction monotone: 0%  (mono=False every single step)
- Max |u[0]|: 35.56 Nm  (t=4.00s)
- Cone violations: 0  (all cone(δ)=OK in every C3diag table)

## Trajectory
```
t=0.00s  ee=(-0.072, -0.003, 0.050)  obj=(0.000,  0.000, 0.050)  goal_dist=0.300m
t=0.50s  ee=(-0.089, -0.097, 0.078)  obj=(-0.024,-0.112, 0.088)  goal_dist=0.342m
t=1.00s  ee=(-0.182, -0.394, 0.105)  obj=(-0.117,-0.403, 0.119)  goal_dist=0.580m
t=1.50s  ee=(-0.188, -0.438, 0.122)  obj=(-0.124,-0.452, 0.137)  goal_dist=0.620m
t=2.00s  ee=(-0.175, -0.425, 0.136)  obj=(-0.112,-0.450, 0.142)  goal_dist=0.610m
t=2.50s  ee=(-0.168, -0.424, 0.154)  obj=(-0.108,-0.461, 0.156)  goal_dist=0.616m
t=3.00s  ee=(-0.124, -0.389, 0.163)  obj=(-0.082,-0.450, 0.161)  goal_dist=0.590m
t=3.50s  ee=(-0.102, -0.419, 0.157)  obj=(-0.058,-0.481, 0.165)  goal_dist=0.600m
t=4.00s  ee=(-0.051, -0.403, 0.156)  obj=(-0.029,-0.480, 0.165)  goal_dist=0.582m
t=4.50s  ee=(0.010,  -0.411, 0.107)  obj=(0.001, -0.475, 0.149)  goal_dist=0.561m
t=5.00s  ee=(0.013,  -0.411, 0.108)  obj=(0.003, -0.475, 0.151)  goal_dist=0.560m
... (stable at goal_dist=0.560m for t=5–20s)
t=18.00s ee=(0.010,  -0.414, 0.106)  obj=(0.006, -0.476, 0.154)  goal_dist=0.560m
```

## First 10 [ADMM] lines after contact (steps 1–10)
```
[ADMM] primal: 0.0057->0.0480  dual: 161.8544->0.0170  mono=False  iters=80/80  rho=25.0
[ADMM] primal: 0.0059->0.0372  dual: 157.0338->0.0201  mono=False  iters=80/80  rho=25.0
[ADMM] primal: 0.0064->0.0515  dual: 147.5510->0.0807  mono=False  iters=80/80  rho=50.0
[ADMM] primal: 0.0072->0.0946  dual: 140.5685->0.1377  mono=False  iters=80/80  rho=50.0
[ADMM] primal: 0.0081->0.1428  dual: 134.4403->0.2113  mono=False  iters=80/80  rho=50.0
[ADMM] primal: 0.0093->0.1933  dual: 128.9513->0.1970  mono=False  iters=80/80  rho=50.0
[ADMM] primal: 0.0108->0.2362  dual: 123.4355->0.2743  mono=False  iters=80/80  rho=50.0
[ADMM] primal: 0.0122->0.2788  dual: 119.3199->0.3200  mono=False  iters=80/80  rho=50.0
[ADMM] primal: 0.0138->0.3299  dual: 114.9467->0.3951  mono=False  iters=80/80  rho=50.0
[ADMM] primal: 0.0156->0.3841  dual: 110.7155->0.4648  mono=False  iters=80/80  rho=50.0
```
Pattern: primal residual GROWS from initial to final at every step (diverging, not converging).
Dual residual starts very high (~160) and decreases (opposite of normal convergence).
ρ is increasing (25→50→100) as adaptive rule fires, but primal keeps growing.

## First contact [FORCE] diagnostic
```
[FORCE] First nonzero contact at step 0:
  contact 0: λ_n=1.6312  nhat_onto_box=[ 1. -0. -0.]  F_world=[ 1.631 -0.    -0.   ]
  F·g_hat=1.6312 (→goal ✓)
```
Force direction is CORRECT at step 1 (+x = eastward = toward goal).
Box still drifted SOUTH (-y) and UP (+z): the arm moved south under ADMM control,
dragging the box through contact. Box eventually tipped (z=0.154 vs initial 0.050).

## Diagnosis
The ADMM primal residual grows monotonically within every 80-iteration solve —
it never decreases. This is ADMM divergence, not slow convergence.
Root cause: the prepositioned arm pose ([0.65, 1.57, 0.95, -1.53, 0.56, 1.79, 0.785])
produces a badly conditioned LCS linearization. The Jacobian at this pose has very
different column structure from the default pose, and the LCS A/B matrices apparently
produce a QP that OSQP cannot reduce the primal residual on within 80 iterations.
Concretely: the approach cost proxy ([proxy] effective=[-0.159, 0, 0.05]) is 8.7cm
west of the EE, pulling the arm west INTO the box. The ADMM plans large normal forces
(λ_n up to 3.99), but the lateral (|λ_t|_max up to 1.19) forces dominate arm motion,
deflecting the arm south and dragging the box with it.

## Comparison against targets
Metric                         | Target | Actual       | PASS/FAIL
-------------------------------|--------|--------------|----------
ADMM median iters              | < 20   | 80 (cap)     | FAIL
Fraction monotone              | > 50%  | 0%           | FAIL
Fraction hit iter cap          | < 20%  | 100%         | FAIL
Max |u[0]|                     | < 25Nm | 35.56 Nm     | FAIL
Progress toward goal           | > 30%  | -86.7%       | FAIL
Cone violations                | 0      | 0            | PASS

5/6 targets fail. Classification: Outcome 2.

## What this means
The prepositioned diagnostic establishes that C3 CAN make contact (unlike Branch B),
but the ADMM fails to converge in this arm configuration. Two separate issues exist:

Issue 1 (Branch B, already known): Approach heuristic fails to reach the box from
default INITIAL_ARM_Q due to workspace geometry. Fix: reduce pre-approach distance
from 0.30m to 0.18m (as recommended in next_steps.md).

Issue 2 (this diagnostic): Once in contact, ADMM primal residuals grow (diverge) in
the prepositioned arm configuration. This is specific to this unusual arm pose and
the mismatch between the approach cost proxy and the contact-phase goal. The
prepositioned pose is so different from the default that the Jacobian and LCS
structure produce a poorly conditioned QP.

## Conclusion
Drake C3 port has a remaining algorithmic issue: contact is established throughout
but ADMM diverges (primal residual grows every solve, 0% monotone convergence,
100% iter cap) causing uncontrolled lateral forces that push the box 47cm south
and tip it off the table surface; the prepositioned diagnostic cannot isolate whether
the port is correct because the unusual arm pose invalidates the normal operating
assumptions. The correct next step is to fix the approach distance (0.30m → 0.18m)
so the DEFAULT arm configuration reaches contact naturally, then re-run the full
smoke test to exercise C3 under normal operating conditions.
