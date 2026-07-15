"""Same as probe_yaw_xseq.py but runs the EXISTING 'pushing' task with
w_yaw=0 — control check that without the yaw cost, the planner's
predicted yaw is ~zero (proving the cube_turning rotation predictions
are caused by the new cost, not noise)."""
import sys
import numpy as np

sys.path.insert(0, ".")
sys.argv = [
    "main.py",
    "pushing",
    "--admm-iter", "25",
    "--math-diag",
    "--max-time", "1.5",
    "--no-record",
]

import main as M

_orig_update = M._update_predicted_trajectory
_step_counter = {"n": 0}


def _probe(meshcat, x_seq, obj_x_idx, obj_y_idx, obj_z_idx):
    ps = obj_x_idx - 4
    qw0, qz0 = float(x_seq[0, ps + 0]), float(x_seq[0, ps + 3])
    qwN, qzN = float(x_seq[-1, ps + 0]), float(x_seq[-1, ps + 3])
    psi0 = 2.0 * np.arctan2(qz0, qw0)
    psiN = 2.0 * np.arctan2(qzN, qwN)
    _step_counter["n"] += 1
    n = _step_counter["n"]
    if n % 10 == 0 or n <= 3:
        print(
            f"[YAW-PROBE-CTL] step={n:4d}  "
            f"x_seq[0]: psi={np.degrees(psi0):+7.2f}deg   "
            f"x_seq[N]: psi={np.degrees(psiN):+7.2f}deg   "
            f"dpsi={np.degrees(psiN - psi0):+7.2f}deg",
            flush=True,
        )
    return _orig_update(meshcat, x_seq, obj_x_idx, obj_y_idx, obj_z_idx)


M._update_predicted_trajectory = _probe
print(f"[YAW-PROBE-CTL] CONTROL: pushing task (w_yaw=0, goal_yaw=0) - expect dpsi~0", flush=True)
M.main()
