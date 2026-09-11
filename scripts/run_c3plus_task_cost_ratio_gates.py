#!/usr/bin/env python3
"""Run clean, short reproducibility/logger/rate gates for the ratio study."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
C3 = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
RUNNER = C3 / "tools/scene_smoke/run_scene_smoke.sh"
OUT = ROOT / "results/c3plus_task_cost_ratio/reproducibility/clean"
OPEN = "matched_rot15_ratio_open_task_s01_g02_rot000"
ICRA = "anything_icra_c_rot15_ratio_s01_g02_rot000"


def active_sims():
    result = subprocess.run(["pgrep", "-af", "franka_sim"], text=True, capture_output=True)
    return result.stdout.splitlines() if result.returncode == 0 else []


def run(name, demo, obj, goal, seed, alpha_p, alpha_theta, port, cap=15):
    folder = OUT / name
    if folder.exists():
        raise RuntimeError(f"refusing to overwrite gate output: {folder}")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SAMPLING_C3_OBS_RELU") or key in {
            "SAMPLING_C3_INNER_OBS_MODE", "SAMPLING_C3_OBJ_NONPEN",
            "SAMPLING_C3_RANK_OBS_MODE",
        }:
            env.pop(key)
    env.update({
        "SAMPLING_C3_EXPERIMENT_SEED": str(seed),
        "SAMPLING_C3_TRANSLATION_COST_SCALE": str(alpha_p),
        "SAMPLING_C3_ORIENTATION_COST_SCALE": str(alpha_theta),
        "SAMPLING_C3_FORENSICS_LOG_DIR": str(folder / "forensics"),
        "SAMPLING_C3_FORENSICS_TEXT_EVENTS": "0",
        "SAMPLING_C3_FORENSICS_TASK_ID": name,
        "SAMPLING_C3_FORENSICS_ROBOT_MODEL": "xarm6",
        "SAMPLING_C3_OBSTACLE_MODE": "lcs_contact",
    })
    env_file = ""
    if demo == ICRA:
        env_file = str(C3 / "tools/scene_smoke/env_icra_sign.sh")
    command = ["bash", str(RUNNER), demo, obj, *map(str, goal), str(cap),
               str(port), str(folder / "run"), env_file]
    folder.mkdir(parents=True)
    with (folder / "launcher_stdout.log").open("w") as stream:
        result = subprocess.run(command, cwd=C3, env=env, text=True,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"gate run failed: {name}")


def main():
    running = active_sims()
    if running:
        raise RuntimeError("unrelated live simulations active; gates deferred:\n" + "\n".join(running))
    settings = [("TR03", .5, 2), ("TR05", 1, 1), ("TR07", 2, .5)]
    for offset, (condition, ap, at) in enumerate(settings):
        run(f"{condition}_perimeter", OPEN, "G_shape_video",
            (0.397, -0.431, 0), 17001, ap, at, 22500 + offset)
    for offset, (condition, ap, at) in enumerate(settings):
        run(f"{condition}_icra", ICRA, "G_shape_video",
            (0.523, -0.4262, 0), 17401, ap, at, 22510 + offset)
    if active_sims():
        raise RuntimeError("unexpected process remained after sequential gates")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run, f"control_rate_concurrency2/lane{lane}", OPEN,
                        "G_shape_video", (0.397, -0.431, 0), 17001, 1, 1,
                        22520 + lane)
            for lane in (1, 2)
        ]
        for future in futures:
            future.result()
    print("clean gate runs complete")


if __name__ == "__main__":
    main()
