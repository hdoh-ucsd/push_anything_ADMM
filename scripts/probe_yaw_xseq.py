"""Read-only x_seq yaw probe.

Drives a short cube_turning rollout via main.main() but monkey-patches
the per-step trajectory-marker call so we can log the planner's predicted
box quaternion (qw, qz) over the horizon. Pure observation -- no
behavioral change to the controller. The probe prints one
[YAW-PROBE] line per control step.

Run:
    python scripts/probe_yaw_xseq.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
sys.argv = [
    "main.py",
    "cube_turning",
    "--admm-iter", "25",
    "--math-diag",
    "--max-time", "1.5",
    "--no-record",
]

import main as M

_orig_update = M._update_predicted_trajectory
_step_counter = {"n": 0}


def _probe(meshcat, x_seq, obj_x_idx, obj_y_idx, obj_z_idx):
    # Box floating-base layout: q = [qw, qx, qy, qz, x, y, z]
    # obj_x_idx = ps + 4, so quaternion is at obj_x_idx-4 .. obj_x_idx-1.
    ps = obj_x_idx - 4
    qw0, qz0 = float(x_seq[0, ps + 0]), float(x_seq[0, ps + 3])
    qwN, qzN = float(x_seq[-1, ps + 0]), float(x_seq[-1, ps + 3])
    # Convert qz -> yaw angle psi (assuming upright: psi = 2 * atan2(qz, qw))
    psi0 = 2.0 * np.arctan2(qz0, qw0)
    psiN = 2.0 * np.arctan2(qzN, qwN)
    _step_counter["n"] += 1
    n = _step_counter["n"]
    if n % 10 == 0 or n <= 3:
        print(
            f"[YAW-PROBE] step={n:4d}  "
            f"x_seq[0]:  qw={qw0:+.5f} qz={qz0:+.5f} psi={np.degrees(psi0):+7.2f}deg   "
            f"x_seq[N]:  qw={qwN:+.5f} qz={qzN:+.5f} psi={np.degrees(psiN):+7.2f}deg   "
            f"dpsi(over horizon)={np.degrees(psiN - psi0):+7.2f}deg",
            flush=True,
        )
    return _orig_update(meshcat, x_seq, obj_x_idx, obj_y_idx, obj_z_idx)


M._update_predicted_trajectory = _probe

# Sanity: goal qz for psi=pi/4 is sin(pi/8) = 0.382683
print(f"[YAW-PROBE] GOAL: target_yaw=pi/4 = 45.00 deg -> qz_target = "
      f"sin(pi/8) = {np.sin(np.pi/8):.6f}", flush=True)

M.main()
