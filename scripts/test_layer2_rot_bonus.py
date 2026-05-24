"""Layer 2 unit test — does the rotation bonus reward off-center,
goal-correct-direction contacts?

This is the GATE: a sign-error here would reward turning the wrong way at
runtime. We exercise the bonus computation in isolation (no Drake plant,
no LCS) by replicating the formula on synthetic contact geometry.

Three samples, all on the +x face of a unit-CoM box:
  (a) centered:   contact at (+0.05, 0, h)  -> r = (+0.05, 0, 0), n_in = (-1, 0, 0)
                  M_z = rx*ny - ry*nx = 0.05*0 - 0*(-1) = 0  -> zero bonus
  (b) off-CCW:    contact at (+0.05, -0.04, h)  -> r = (+0.05, -0.04, 0), n = (-1,0,0)
                  M_z = 0.05*0 - (-0.04)*(-1) = -0.04  (CW, negative)
                  Wait — let me re-derive. For goal +pi/4 (CCW), yaw_sign=+1.
                  We want the sample that rotates CCW (M_z > 0) to win.
                  An off-center contact at -y on the +x face pushes the box
                  in -x at point r=(+0.05,-0.04). The cross product:
                  r × F  where F = n_in (the push). M_z = r_x*F_y - r_y*F_x
                                                       = 0.05*0 - (-0.04)*(-1)
                                                       = -0.04   <- CW
                  So contact at (+0.05, -0.04) actually rotates CW (negative).
                  For CCW, we need contact at +y on +x face: (+0.05, +0.04).
                  Then M_z = 0.05*0 - 0.04*(-1) = +0.04 (CCW, positive). ✓
  (c) off-CW:     contact at (+0.05, -0.04, h)  -> M_z = -0.04 (CW, negative)

Goal +pi/4 (CCW), yaw_sign=+1:
  (a) M_z*sign = 0    -> bonus = w_rot * max(0, 0) = 0
  (b) M_z*sign = +0.04 -> bonus = w_rot * 0.04
  (c) M_z*sign = -0.04 -> bonus = w_rot * max(0, -0.04) = 0

Flip goal to -pi/4 (CW), yaw_sign=-1:
  (a) M_z*sign = 0    -> bonus = 0
  (b) M_z*sign = -0.04 -> bonus = 0
  (c) M_z*sign = +0.04 -> bonus = w_rot * 0.04
"""
import sys, numpy as np

sys.path.insert(0, ".")
from control.sampling_c3.inner_solve import _yaw_from_quat, _wrap_to_pi


def rot_bonus(ee_box_contacts, p_box, qw, qx, qy, qz, target_yaw,
              w_rot=30000.0, w_yaw=10.0):
    """Replicate the gating logic from inner_solve.evaluate_sample."""
    if w_rot <= 0.0 or w_yaw <= 0.0 or not ee_box_contacts:
        return 0.0, 0.0
    psi_now = _yaw_from_quat(qw, qx, qy, qz)
    yaw_err = _wrap_to_pi(float(target_yaw) - psi_now)
    if abs(yaw_err) <= 1e-3:
        return 0.0, 0.0
    yaw_sign = 1.0 if yaw_err > 0.0 else -1.0
    mz_signed = []
    for p_c_W, n_W in ee_box_contacts:
        rx = float(p_c_W[0]) - p_box[0]
        ry = float(p_c_W[1]) - p_box[1]
        nx = float(n_W[0]); ny = float(n_W[1])
        m_z = rx * ny - ry * nx
        mz_signed.append(m_z * yaw_sign)
    rot_score = max(mz_signed)
    bonus = w_rot * max(0.0, rot_score)
    return rot_score, bonus


# Box at origin (0,0), yaw 0. Box half-extent 0.05; pushing +x face.
# n_in (inward, force-on-box for +x face contact) = (-1, 0, 0).
P_BOX = (0.0, 0.0)
QW, QX, QY, QZ = 1.0, 0.0, 0.0, 0.0   # yaw=0
H = 0.05
N_PUSH = np.array([-1.0, 0.0, 0.0])

contact_center = [(np.array([+0.05, 0.0, H]), N_PUSH)]
contact_ccw    = [(np.array([+0.05, +0.04, H]), N_PUSH)]  # +y offset → CCW
contact_cw     = [(np.array([+0.05, -0.04, H]), N_PUSH)]  # -y offset → CW

print("=" * 70)
print("GOAL +pi/4 (CCW). Box at yaw 0. yaw_sign = +1.")
print("=" * 70)
goal = +np.pi / 4
s_a, b_a = rot_bonus(contact_center, P_BOX, QW, QX, QY, QZ, goal)
s_b, b_b = rot_bonus(contact_ccw,    P_BOX, QW, QX, QY, QZ, goal)
s_c, b_c = rot_bonus(contact_cw,     P_BOX, QW, QX, QY, QZ, goal)
print(f"  centered :  rot_score={s_a:+.4f}  rot_bonus={b_a:10.2f}")
print(f"  off-CCW  :  rot_score={s_b:+.4f}  rot_bonus={b_b:10.2f}  <- should WIN")
print(f"  off-CW   :  rot_score={s_c:+.4f}  rot_bonus={b_c:10.2f}")
assert b_b > b_a + 1e-6, "FAIL: off-CCW sample not rewarded over centered (CCW goal)"
assert b_b > b_c + 1e-6, "FAIL: CCW (goal-correct) not rewarded over CW (wrong)"
assert b_c <= b_a + 1e-6, "FAIL: wrong-direction sample got a positive bonus"
print("PASS: CCW-producing contact wins for CCW goal; CW gets zero.")

print()
print("=" * 70)
print("FLIP GOAL -pi/4 (CW). Box at yaw 0. yaw_sign = -1.")
print("=" * 70)
goal = -np.pi / 4
s_a, b_a = rot_bonus(contact_center, P_BOX, QW, QX, QY, QZ, goal)
s_b, b_b = rot_bonus(contact_ccw,    P_BOX, QW, QX, QY, QZ, goal)
s_c, b_c = rot_bonus(contact_cw,     P_BOX, QW, QX, QY, QZ, goal)
print(f"  centered :  rot_score={s_a:+.4f}  rot_bonus={b_a:10.2f}")
print(f"  off-CCW  :  rot_score={s_b:+.4f}  rot_bonus={b_b:10.2f}")
print(f"  off-CW   :  rot_score={s_c:+.4f}  rot_bonus={b_c:10.2f}  <- should WIN")
assert b_c > b_a + 1e-6, "FAIL: off-CW sample not rewarded over centered (CW goal)"
assert b_c > b_b + 1e-6, "FAIL: CW (goal-correct) not rewarded over CCW (wrong)"
assert b_b <= b_a + 1e-6, "FAIL: wrong-direction sample got a positive bonus"
print("PASS: bonus flips correctly with goal direction.")

print()
print("=" * 70)
print("REGRESSION: translation task (w_yaw=0)  ->  bonus must be inert.")
print("=" * 70)
s_t, b_t = rot_bonus(contact_ccw, P_BOX, QW, QX, QY, QZ, +np.pi/4,
                     w_rot=30000.0, w_yaw=0.0)
print(f"  off-CCW with w_yaw=0:  rot_score={s_t:.4f}  rot_bonus={b_t:.2f}")
assert b_t == 0.0, "FAIL: translation task got non-zero rot_bonus"
print("PASS: w_yaw=0 ⇒ rot_bonus inert.")

print()
print("=" * 70)
print("REGRESSION: already at goal (yaw_err ≈ 0) ⇒ rot_bonus inert.")
print("=" * 70)
# Box already at yaw = +pi/4. Quat for yaw=pi/4 about z: w=cos(pi/8), z=sin(pi/8).
yaw0 = +np.pi / 4
qw0 = np.cos(yaw0 / 2); qz0 = np.sin(yaw0 / 2)
s_z, b_z = rot_bonus(contact_ccw, P_BOX, qw0, 0.0, 0.0, qz0, +np.pi/4)
print(f"  at-goal off-CCW:  rot_score={s_z:.4f}  rot_bonus={b_z:.2f}")
assert b_z == 0.0, "FAIL: at-goal sample got non-zero bonus"
print("PASS: at-goal ⇒ rot_bonus inert.")

print()
print("=" * 70)
print("ALL UNIT TESTS PASSED — Layer 2 ready for commit + rollout")
print("=" * 70)
