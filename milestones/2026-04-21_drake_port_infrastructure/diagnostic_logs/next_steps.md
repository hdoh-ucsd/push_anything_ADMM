# Drake C3 Port — Next Steps
# Generated: 2026-04-20

## Classification: Branch B (broken)

## Full Summary

See summary.txt. Condensed:

| Metric            | Target | Actual       | Pass? |
|-------------------|--------|--------------|-------|
| ADMM median iters | < 20   | N/A (no nc)  | FAIL* |
| Fraction monotone | > 50%  | N/A (no nc)  | FAIL* |
| Iter cap fraction | < 20%  | N/A (no nc)  | FAIL* |
| Max |u[0]|        | < 25Nm | 72.07 Nm     | FAIL  |
| Progress          | > 30%  | 0.0%         | FAIL  |
| Cone violations   | 0      | 0            | PASS  |

*FAIL because the arm never made contact — ADMM/projection never ran.

## Step 4 Actions (Branch B diagnostics)

### Variant 1 — w_obj_xy = 20000 (×20 instead of ×100)
File: results/drake_port_smoke/variant_w20k.txt

Result:
  - 2000 steps, n_c=0 throughout, box at (0,0,0.05) start and end (0% progress)
  - Arm froze at perp=0.078m, ee_to_box=0.215m, along=-0.200m (same freeze point)
  - Max |u|: 72.07 Nm, mean: ~24 Nm

Conclusion: Backing off w_obj_xy by 5× had zero effect. The freeze is not a
cost weight issue.

### Variant 2 — w_ee_approach = 80000 (proxy × 10, w_obj_xy restored to 100000)
File: results/drake_port_smoke/variant_proxy10x.txt

Result:
  - 2000 steps, n_c=0 throughout, box essentially stationary (2mm drift)
  - perp at end: 0.022m (was 0.078m in Variants 0/1 — 10× proxy DID help)
  - ee_to_box at end: 0.258m (stuck just above 0.25m stage-1→2 threshold)
  - stage=1 at end: targeting pre-approach at (-0.30, 0, 0.05)
  - Arm reached (-0.258, -0.025, 0.033) — within 51mm of stage-1 target but
    can't close: 42mm too far east (arm reach limit) and 17mm too low in z.
  - Max |u|: 73.72 Nm (start), settled to ~30 Nm

Key finding: the 10× proxy brought the arm nearly on-axis (perp=0.022m vs
0.078m) but the arm still cannot reach the 30cm pre-approach position due to
workspace geometry. The stage transition (stage1→stage2 at ee_to_box<0.25m)
never fires because the arm stabilises at ee_to_box≈0.258m.

Conclusion: the pre-approach target (0.30m behind box, hardcoded) is at the
edge of the arm's workspace from the starting configuration. The arm CAN
align laterally (perp→0.022m) but cannot get the approach distance below
0.258m from this joint configuration.

### Variant 3 — Projection diff
File: results/drake_port_smoke/projection_diff.md

Result: Drake and sandbox projections are IDENTICAL — same 3 cases, same
(1+μ²) denominator, same tolerances (1e-12 on Cases 1/2). The Branch B failure
is NOT a projection bug.

## Root Cause Diagnosis (synthesised across all three variants)

Arm starts at EE=(0.347, -0.600, 0.047). Box at (0,0,0.05). Goal (0.3,0).
Push axis is y=0. Arm starts 60cm south (perp=0.60m).

Stage 1 approach target: (-0.30, 0, 0.05) — 30cm west of box, on-axis.
This point is at distance 0.686m from the robot base (within 0.855m reach).
HOWEVER: the arm can get perp=0.022m (Variant 2) but stalls at
ee_to_box=0.258m — it cannot simultaneously reach x=-0.30 at z=0.05.
The arm appears to be kinematically limited from reaching x<-0.26 at z=0.05
from this configuration; the joint velocities run out of Jacobian authority.

Variant comparison summary:
  Variant 0: w_obj_xy=100000, w_ee=8000  → perp=0.078m, stuck (joint limit)
  Variant 1: w_obj_xy=20000,  w_ee=8000  → perp=0.078m, stuck (same limit)
  Variant 2: w_obj_xy=100000, w_ee=80000 → perp=0.022m, stuck at stage-1/2
                                             boundary (workspace edge)

Projection is correct (Variant 3). ADMM is never exercised. The port itself
(ρ floor, logging, cost scaling) is correct but untestable until approach works.

## Recommendation (one sentence)

Reduce the stage-1 pre-approach distance from 0.30m to 0.18m in
task_costs.py line 233 (`pre_approach_3d = obj - 0.18 * g_hat`) so the
arm's reachable workspace includes the approach waypoint, allowing
stage 1→2→3 transitions to complete and ADMM to be exercised under contact.
