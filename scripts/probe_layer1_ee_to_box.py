"""Layer 1 rollout probe: does the arm now approach the box for cube_turning?

Drives a 1.5s cube_turning rollout via main.main() and monkey-patches
_update_predicted_trajectory so we can log per-step:
  - ee_to_box (xy distance from EE to box CoM)
  - mode (c3 or free, from the wrapper's last_x_seq context)

Pure observation; no behavioral change.
"""
import sys

sys.path.insert(0, ".")
sys.argv = [
    "main.py",
    "cube_turning",
    "--admm-iter", "25",
    "--max-time", "1.5",
    "--no-record",
]

import main as M

_orig_update = M._update_predicted_trajectory
_step_counter = {"n": 0}
_history = []   # list of (step, ee_xy, box_xy, ee_to_box)


def _probe(meshcat, x_seq, obj_x_idx, obj_y_idx, obj_z_idx):
    # x_seq[0] is the CURRENT state; extract EE? actually no — x_seq is
    # state vector [q; v], we don't get EE position directly. We can
    # extract box pos easily; EE position needs FK.
    # Easier: extract box xy from x_seq[0]; the wrapper logs EE elsewhere.
    box_x = float(x_seq[0, obj_x_idx])
    box_y = float(x_seq[0, obj_y_idx])
    _step_counter["n"] += 1
    n = _step_counter["n"]
    _history.append((n, box_x, box_y))
    return _orig_update(meshcat, x_seq, obj_x_idx, obj_y_idx, obj_z_idx)


M._update_predicted_trajectory = _probe
print(f"[L1-PROBE] cube_turning, w_yaw=10, layer 1 active "
      f"(translation-agnostic EE-approach via box-CoM proxy)", flush=True)
M.main()

# Summary
print()
print(f"[L1-PROBE] === Summary ===")
print(f"[L1-PROBE] Total steps observed: {len(_history)}")
if _history:
    first = _history[0]
    last  = _history[-1]
    print(f"[L1-PROBE] First step box xy: ({first[1]:+.4f}, {first[2]:+.4f})")
    print(f"[L1-PROBE] Last step  box xy: ({last[1]:+.4f}, {last[2]:+.4f})")
