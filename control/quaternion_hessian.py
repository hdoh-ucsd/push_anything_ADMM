"""
Hessian of squared quaternion-angle difference.

Port of dairlib `common/quaternion_error_hessian.cc` (push_anything_dev @
257e3ed). Used by the reference dispatcher's near-goal cost switch
(`sampling_based_c3_controller.cc:1517-1570`) to replace the linear-in-
quaternion yaw cost with the actual Hessian of θ²(q, r) where
θ = 2·atan2(‖vec(q·r*)‖, scalar(q·r*)).

Near-goal, this Hessian steers the planner to reduce quaternion-angle
error (proper 3D rotation metric) rather than the small-angle linear
approximation. In the port, this is called from
`task_costs.QuadraticManipulationCost.build_ee_space` when the wrapper
sets `crossed_switching_threshold = True`.

Numpy transcription of the closed-form 4×4 symbolic Hessian. Symbols
match the C++ source verbatim for auditability.
"""
from __future__ import annotations

import numpy as np


def hessian_of_squared_quaternion_angle_difference(
        quat: np.ndarray, quat_desired: np.ndarray) -> np.ndarray:
    """Return the 4×4 Hessian of θ²(q, r) w.r.t. q, evaluated at (q, r).

    Parameters
    ----------
    quat         : (4,) current quaternion [w, x, y, z]
    quat_desired : (4,) goal quaternion    [w, x, y, z]

    Returns
    -------
    H : (4, 4) symmetric Hessian. Not guaranteed PSD — caller must
        regularize (see reference cc:1540-1544).
    """
    quat = np.asarray(quat, dtype=float).reshape(4)
    quat_desired = np.asarray(quat_desired, dtype=float).reshape(4)

    q_w, q_x, q_y, q_z = quat
    r_w, r_x, r_y, r_z = quat_desired

    # Reusable sub-expressions (verbatim from quaternion_error_hessian.cc:31-63).
    exp_1 = np.arctan2(
        np.sqrt(
            (q_w * r_x - q_x * r_w + q_y * r_z - q_z * r_y) ** 2 +
            (q_w * r_y - q_x * r_z - q_y * r_w + q_z * r_x) ** 2 +
            (q_w * r_z + q_x * r_y - q_y * r_x - q_z * r_w) ** 2),
        q_w * r_w + q_x * r_x + q_y * r_y + q_z * r_z,
    )
    exp_2 = (q_w ** 2 * r_x ** 2 + q_w ** 2 * r_y ** 2 + q_w ** 2 * r_z ** 2
             - 2 * q_w * q_x * r_w * r_x - 2 * q_w * q_y * r_w * r_y
             - 2 * q_w * q_z * r_w * r_z
             + q_x ** 2 * r_w ** 2 + q_x ** 2 * r_y ** 2 + q_x ** 2 * r_z ** 2
             - 2 * q_x * q_y * r_x * r_y - 2 * q_x * q_z * r_x * r_z)
    exp_3 = (q_y ** 2 * r_w ** 2 + q_y ** 2 * r_x ** 2 + q_y ** 2 * r_z ** 2
             - 2 * q_y * q_z * r_y * r_z
             + q_z ** 2 * r_w ** 2 + q_z ** 2 * r_x ** 2 + q_z ** 2 * r_y ** 2)
    exp_4 = q_w ** 2 + q_x ** 2 + q_y ** 2 + q_z ** 2
    exp_5 = (exp_4 ** 2) * ((exp_2 + exp_3) ** 2.5)
    exp_6 = (exp_2 + exp_3) ** 1.5
    exp_7 = (q_w * q_x * r_x + q_w * q_y * r_y + q_w * q_z * r_z
             - q_x ** 2 * r_w - q_y ** 2 * r_w - q_z ** 2 * r_w)
    exp_8 = (q_w ** 2 * r_y - q_w * q_y * r_w + q_x ** 2 * r_y
             - q_x * q_y * r_x - q_y * q_z * r_z + q_z ** 2 * r_y)
    exp_9 = (q_w ** 2 * r_x - q_w * q_x * r_w - q_x * q_y * r_y
             - q_x * q_z * r_z + q_y ** 2 * r_x + q_z ** 2 * r_x)
    exp_10 = (q_w * r_w * r_z + q_x * r_x * r_z + q_y * r_y * r_z
              - q_z * r_w ** 2 - q_z * r_x ** 2 - q_z * r_y ** 2)
    exp_11 = (q_w ** 2 * r_z - q_w * q_z * r_w + q_x ** 2 * r_z
              - q_x * q_z * r_x + q_y ** 2 * r_z - q_y * q_z * r_y)
    exp_12 = (q_w * r_w * r_y + q_x * r_x * r_y - q_y * r_w ** 2
              - q_y * r_x ** 2 - q_y * r_z ** 2 + q_z * r_y * r_z)
    exp_13 = (q_w * r_w * r_x - q_x * r_w ** 2 - q_x * r_y ** 2
              - q_x * r_z ** 2 + q_y * r_x * r_y + q_z * r_x * r_z)

    # Guard against div-by-zero at exact quat alignment (exp_2+exp_3 = 0).
    # In that limit the Hessian is the linearization H = 8·(r rᵀ - I·(rᵀr))
    # style; but the atan2 branch also degenerates. Returning zeros is a
    # safe fallback (no cost contribution when perfectly aligned); the
    # regularizer (part 2) still gives Q[0:4,0:4] = weight · frac · r rᵀ.
    if exp_5 <= 1e-30:
        return np.zeros((4, 4))

    # Diagonal Hessian elements (verbatim from cc:66-82).
    H_ww = 8.0 * (
        -(2.0 * q_w * exp_7 * (exp_2 + exp_3)
          - (q_x * r_x + q_y * r_y + q_z * r_z) * exp_4 * (exp_2 + exp_3)
          + exp_4 * (q_w * r_x ** 2 + q_w * r_y ** 2 + q_w * r_z ** 2
                     - q_x * r_w * r_x - q_y * r_w * r_y - q_z * r_w * r_z)
          * exp_7) * (exp_2 + exp_3) * exp_1
        + (exp_7 ** 2) * exp_6
    ) / exp_5
    H_xx = 8.0 * (
        (2.0 * q_x * exp_9 * (exp_2 + exp_3)
         + (q_w * r_w + q_y * r_y + q_z * r_z) * exp_4 * (exp_2 + exp_3)
         - exp_4 * exp_9 * exp_13) * (exp_2 + exp_3) * exp_1
        + (exp_9 ** 2) * exp_6
    ) / exp_5
    H_yy = 8.0 * (
        (2.0 * q_y * exp_8 * (exp_2 + exp_3)
         + (q_w * r_w + q_x * r_x + q_z * r_z) * exp_4 * (exp_2 + exp_3)
         - exp_4 * exp_8 * exp_12) * (exp_2 + exp_3) * exp_1
        + (exp_8 ** 2) * exp_6
    ) / exp_5
    H_zz = 8.0 * (
        (2.0 * q_z * exp_11 * (exp_2 + exp_3)
         + (q_w * r_w + q_x * r_x + q_y * r_y) * exp_4 * (exp_2 + exp_3)
         - exp_4 * exp_11 * exp_10) * (exp_2 + exp_3) * exp_1
        + (exp_11 ** 2) * exp_6
    ) / exp_5

    # Off-diagonal Hessian elements (cc:83-106).
    H_wx = 8.0 * (
        (-2.0 * q_x * exp_7 * (exp_2 + exp_3)
         + (q_w * r_x - 2.0 * q_x * r_w) * exp_4 * (exp_2 + exp_3)
         + exp_4 * exp_7 * exp_13) * (exp_2 + exp_3) * exp_1
        - exp_9 * exp_7 * exp_6
    ) / exp_5
    H_wy = 8.0 * (
        (-2.0 * q_y * exp_7 * (exp_2 + exp_3)
         + (q_w * r_y - 2.0 * q_y * r_w) * exp_4 * (exp_2 + exp_3)
         + exp_4 * exp_7 * exp_12) * (exp_2 + exp_3) * exp_1
        - exp_8 * exp_7 * exp_6
    ) / exp_5
    H_wz = 8.0 * (
        (-2.0 * q_z * exp_7 * (exp_2 + exp_3)
         + (q_w * r_z - 2.0 * q_z * r_w) * exp_4 * (exp_2 + exp_3)
         + exp_4 * exp_7 * exp_10) * (exp_2 + exp_3) * exp_1
        - exp_11 * exp_7 * exp_6
    ) / exp_5
    H_xy = 8.0 * (
        (2.0 * q_y * exp_9 * (exp_2 + exp_3)
         + (q_x * r_y - 2.0 * q_y * r_x) * exp_4 * (exp_2 + exp_3)
         - exp_4 * exp_9 * exp_12) * (exp_2 + exp_3) * exp_1
        + exp_9 * exp_8 * exp_6
    ) / exp_5
    H_xz = 8.0 * (
        (2.0 * q_z * exp_9 * (exp_2 + exp_3)
         + (q_x * r_z - 2.0 * q_z * r_x) * exp_4 * (exp_2 + exp_3)
         - exp_4 * exp_9 * exp_10) * (exp_2 + exp_3) * exp_1
        + exp_9 * exp_11 * exp_6
    ) / exp_5
    H_yz = 8.0 * (
        (2.0 * q_z * exp_8 * (exp_2 + exp_3)
         + (q_y * r_z - 2.0 * q_z * r_y) * exp_4 * (exp_2 + exp_3)
         - exp_4 * exp_8 * exp_10) * (exp_2 + exp_3) * exp_1
        + exp_8 * exp_11 * exp_6
    ) / exp_5

    # Assemble symmetric matrix (cc:109-113).
    H = np.array([
        [H_ww, H_wx, H_wy, H_wz],
        [H_wx, H_xx, H_xy, H_xz],
        [H_wy, H_xy, H_yy, H_yz],
        [H_wz, H_xz, H_yz, H_zz],
    ])
    return H


def build_regularized_quaternion_cost(
        quat: np.ndarray, quat_desired: np.ndarray,
        weight: float, regularizer_fraction: float = 0.0) -> np.ndarray:
    """Return the 4×4 near-goal quaternion cost block Q_quat.

    Mirrors reference `sampling_based_c3_controller.cc:1529-1568`:
      Q_quat = weight · (H + reg1 + fraction · reg2)
    where
      H    = hessian_of_squared_quaternion_angle_difference(quat, quat_desired)
      reg1 = max(0, -min_eigenvalue(H)) · I(4)     — PSD projector
      reg2 = quat_desired · quat_desired.T         — reference-tangent shim

    Parameters
    ----------
    quat                  : (4,) current quaternion
    quat_desired          : (4,) goal quaternion
    weight                : `q_quaternion_dependent_weight` (reference: 1000)
    regularizer_fraction  : `q_quaternion_dependent_regularizer_fraction`
                            (reference push_t: 0)

    Returns
    -------
    Q_quat : (4, 4) PSD cost matrix on the quat slot.
    """
    H = hessian_of_squared_quaternion_angle_difference(quat, quat_desired)
    # Regularizer 1: shift eigenvalues so H+reg1 ≽ 0.
    min_eig = float(np.linalg.eigvals(H).real.min()) if np.any(H) else 0.0
    reg1 = max(0.0, -min_eig) * np.eye(4)
    # Regularizer 2: outer product of the goal quaternion.
    r = np.asarray(quat_desired, dtype=float).reshape(4)
    reg2 = np.outer(r, r)

    Q = float(weight) * (H + reg1 + float(regularizer_fraction) * reg2)
    # Symmetrize (guard against roundoff drift).
    Q = 0.5 * (Q + Q.T)
    return Q
