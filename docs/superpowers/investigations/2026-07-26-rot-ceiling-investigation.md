# Investigation — rot metric stuck near 0.4 rad through the arc-2 arc

**Date:** 2026-07-26 (post commit `4bd9c99`)
**Trigger:** every arc-2 experiment (p85 → p90 → p95 → p98) has rot ∈ [0.387, 0.564] rad. Trans metric moves; rot is essentially unchanged. Task target: rot < 0.02 rad (~20× further reduction needed).

## Finding

The c3-mode geometric contact target is **hard-coded to face-center of the pushed side** with **zero moment arm** relative to box CoM. This means every c3-mode push under the port's IK-projected-target regime produces pure translation and **zero yaw authority**.

## Evidence

### Code site — `sampling_based_c3_controller.py:3097-3150`

For tshape (push_t task):
```python
if abs(g_hat_3d[1]) >= abs(g_hat_3d[0]):
    # dominant y push: target vertical bar's y face
    _face_offset = 0.02
else:
    # dominant x push: target vertical bar's east face
    _face_offset = (0.13 if g_hat_3d[0] < 0 else 0.07)
...
_p_ee_geom = np.array([
    _box_xy_now[0] - g_hat_3d[0] * _contact_offset,   # box CoM x
    _box_xy_now[1] - g_hat_3d[1] * _contact_offset,   # box CoM y  ← face-center
    float(self._c3_geom_z_target),
])
```

The XY position is `box_center - g_hat · offset` — literally the face's center point along the goal-aligned normal. **No tangent-direction component to generate a moment arm.**

### Consequence

- Contact force on box: `F = ±λ · nhat_onto_box` — magnitude ~25 N in the goal direction (verified p90 log: `f_cmd=(-4.63, +24.35, +2.35)N` with `lam_n=55`).
- Contact point: face-center → moment arm from box CoM = **~0**.
- Torque on box: `r × F ≈ 0` → **no yaw correction**.
- Cost function has `w_yaw = 800` (`config/tasks.yaml:172` sets `goal_yaw = -0.7379` rad, `-42.28°`). Q gradient wants yaw fixed. **But no actuator authority is exercised at face-center contact.**

### Why the sampler doesn't help

`sampling.py:_face_normal_projection` DOES have a `yaw_delta` bias (line 472-487) that:
- Biases face selection toward faces whose push generates goal-aligned yaw torque
- Reduces tangent jitter under strong g_hat alignment (samples cluster near center for "goal-aligned" faces)

But **c3 mode doesn't use the sampler for its OSC target**. It uses the IK-projected `_p_ee_geom` (fixed face-center) OR the planner's `_x_seq[i][7:10]` (which is also aggregated toward face-center by the planner's LCS-informed prediction). The sampler's off-center samples influence **mode-switch selection** but not the actual push contact point once c3 mode fires.

### Empirical trace

p90 (best ref-conform baseline): 58 c3-STEPs, final rot=0.387 rad. Every c3 STEP entry shows `lam_t=0.000` — zero tangential force. Only normal force applied at face-center → confirmation of zero moment arm.

p98 (LCS_ALWAYS_ON=0, 128 c3-STEPs — 2.2× more contact attempts): rot=0.400 rad. **Doubling c3 engagement did nothing for rot** because each push was still face-center.

## Root cause

The port's c3-mode contact-target computation was designed to guarantee a **reachable** contact point (IK-projected onto box face, safe height). It was NOT designed to give the planner-controller yaw authority. The design implicitly assumes the ADMM planner's `x_seq` will provide off-center variation, but under the port's converged-plan regime x_seq settles to face-center too.

**Reference** likely handles yaw differently — reference plans off-center contact naturally via ADMM optimization over the full-plant LCS (which includes rotational box dynamics AND arm reach constraints). Reference's ADMM sees both cost gradients (yaw + position) and picks a contact point that trades both.

## Fix candidates

### A — Yaw-aware c3 contact target (off-reference heuristic)

Add a tangent-direction offset to `_p_ee_geom` proportional to yaw error:

```python
# Current yaw from box quaternion, target from tasks.yaml goal_yaw.
_yaw_err = wrap_pi(target_yaw - current_yaw)

# Perpendicular to g_hat in xy plane.
_tangent = np.array([-g_hat_3d[1], g_hat_3d[0], 0.0])

# Face tangent-half-width (tshape vertical bar y-face: 0.08m; horiz: 0.02m).
_face_tan_hw = _shape_specific_tangent_half_width(g_hat_3d)

# Sign convention: torque_z = (r_x · F_y - r_y · F_x). With F = -F_mag · g_hat,
# and r_tangent = lever · _tangent, torque_z = lever * F_mag * (g_hat_x² + g_hat_y²)
# = lever * F_mag (unit-norm g_hat). So lever's sign directly determines torque
# sign. Positive lever ⇒ positive torque_z (CCW rotation).
_lever_gain = 0.03  # m per rad of yaw error
_lever = np.clip(_yaw_err * _lever_gain,
                 -_face_tan_hw * 0.8, +_face_tan_hw * 0.8)

_p_ee_geom = _p_ee_geom + _tangent * _lever
```

Env-gate `PORT_C3_YAW_LEVER=1` (default OFF). Off-reference heuristic — doesn't match reference's LCS-optimized contact selection, but injects yaw authority the port currently lacks.

**Expected impact:** rot should drop meaningfully (from 0.4 to hopefully <0.1) if the arm-Cart contact-loss cycle doesn't prevent off-center pushes from sticking.

**Risk:** off-center contact under intermittent contact might reduce trans progress (arm loses purchase faster off-center). Requires empirical A/B.

### B — Increase sampling diversity, hope planner picks off-center (reference-conformant)

Enable stronger tangent jitter under yaw_delta > 0 (currently REDUCES jitter under strong g_hat). Let the sampler generate more off-center candidates. Reference-conformant if the sampler's `sample_projection_clearance` and jitter distribution mirror reference.

Effort: modify `sampling.py:_face_normal_projection` tangent-jitter logic. Small code change.

**Risk:** if the argmin over c_samples still picks face-center (because face-center has lower c_C3), off-center samples never win.

### C — Fix the planner to optimize contact-point selection (structural)

Have the planner's ADMM explore off-center contacts through the LCS's tangential-friction structure. This is what reference does. Port's simplified EE-space LCS may lose this because `p_ee` is decoupled from `arm_q` after A1 fix.

Effort: substantial — LCS refactor. Same magnitude as full-plant LCS attempt from prior discussion.

## Recommendation

**Try A first** as a diagnostic probe (env-gated, default OFF). Confirms whether off-center push CAN correct yaw under arm-Cart. If yes, we know the port's mechanism gap. If no, arm-Cart is stronger blocker than I think and only full-plant LCS (C) helps.

**A is off-reference** — needs user authorization per `feedback_no_off_reference_knob_probes`. But it's a probe, not a landed default.

## Not investigated in this pass

- Whether reference's off-center contact selection lands at explicit lever values matching our proposed gain
- Whether reducing `w_yaw` (currently 800) would let the planner "give up" on yaw and get better trans
- Whether contact-point selection is actually solvable given tshape geometry (tshape has small T bar tangent half-width; may not fit enough lever for 42° rotation)
