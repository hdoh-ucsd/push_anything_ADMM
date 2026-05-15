#!/usr/bin/env python3
"""Generate 15 contact-free sweep task config records.

Each YAML captures the `pushing` task definition with a goal_xy
override along +x and an explicit seed. These files are records of
what was run (the actual runs use main.py's --goal-xy and --seed
CLI overrides, not these files).
"""
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent.parent
BASE = yaml.safe_load((REPO / "config" / "tasks.yaml").read_text())
PUSH = BASE["tasks"]["pushing"]

INIT_X, INIT_Y, _ = PUSH["init_xyz"]  # (0, 0, 0.05)

DISTANCES_CM = [5, 10, 15, 20, 30]
SEEDS = [0, 1, 2]

out_dir = REPO / "results" / "contact_free_sweep" / "configs"
out_dir.mkdir(parents=True, exist_ok=True)

for d_cm in DISTANCES_CM:
    d_m = d_cm / 100.0
    goal = [round(INIT_X + d_m, 6), round(INIT_Y, 6)]
    for s in SEEDS:
        task_cfg = dict(PUSH)
        task_cfg["goal_xy"] = goal
        task_cfg["seed"] = s
        task_cfg["sweep_distance_cm"] = d_cm
        wrapper = {"tasks": {"pushing": task_cfg}}
        path = out_dir / f"sweep_d{d_cm:02d}cm_s{s}.yaml"
        path.write_text(yaml.safe_dump(wrapper, sort_keys=False))
        print(f"  wrote {path.relative_to(REPO)}  goal={goal}  seed={s}")

print(f"[OK] {len(DISTANCES_CM)*len(SEEDS)} configs in {out_dir.relative_to(REPO)}")
