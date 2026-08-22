#!/usr/bin/env python3
"""Collect 28 successful randomized C3+ trials for each Fig. 8 block object.

Objects are processed in roster order, beginning with the canonical Block-T.
Within the active object, several independent seeds run concurrently.  A trial
counts only when the random-goal generator reports one achieved goal; timeouts
and interrupted attempts remain as provenance and a fresh seed is allocated.

The campaign is resumable: completed logs are rescanned at startup.  Create
``results/fig8_block_28_c3plus/STOP`` to stop launching new trials; active
trials are allowed to finish.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/miniconda3/envs/push_anything_ADMM/bin/python3")
OUT = ROOT / "results" / "fig8_block_28_c3plus"
ROSTER = [
    "push_t",
    "book_block", "lotion_block", "baby_toy_block", "clamp_block",
    "I_shape_texture_block", "H_shape_texture_block",
    "E_shape_video_block", "Y_shape_video_block", "3_shape_video_block",
    "C_shape_texture_block", "G_shape_video_block",
    "A_shape_video_block", "B_shape_video_block", "R_shape_texture_block",
]
SUCCESS_MARKER = "[GOAL-GEN] COMPLETE: 1 goals achieved"


def classify(path: Path) -> str:
    if not path.is_file():
        return "missing"
    text = path.read_text(errors="replace")
    if SUCCESS_MARKER in text:
        return "success"
    if "[RESULT]" in text:
        return "failure"
    return "incomplete"


def logs_for(task: str) -> list[Path]:
    return sorted((OUT / task).glob(f"{task}_seed*.txt"))


def seed_of(path: Path) -> int:
    match = re.search(r"_seed(\d+)\.txt$", path.name)
    return int(match.group(1)) if match else -1


def snapshot(active: dict[int, tuple[subprocess.Popen, Path]], target: int,
             workers: int) -> None:
    tasks = {}
    for task in ROSTER:
        states = [classify(p) for p in logs_for(task)]
        tasks[task] = {
            "successes": states.count("success"),
            "failures": states.count("failure"),
            "incomplete": states.count("incomplete"),
            "target": target,
        }
    payload = {
        "updated_unix": time.time(),
        "workers": workers,
        "active": [str(path.relative_to(ROOT)) for _, path in active.values()],
        "tasks": tasks,
    }
    tmp = OUT / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(OUT / "status.json")


def launch(task: str, seed: int, max_time: float) -> tuple[subprocess.Popen, Path]:
    task_dir = OUT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    log = task_dir / f"{task}_seed{seed:05d}.txt"
    env = os.environ.copy()
    env.update({
        "PORT_GOAL_MODE": "kRandom",
        "PORT_GOAL_DRAW_INITIAL": "1",
        "PORT_GOALGEN_N": "1",
    })
    name = f"fig8_block_28_c3plus/{task}/{task}_seed{seed:05d}"
    cmd = [
        str(PYTHON), "main.py", task,
        "--solver", "c3plus",
        "--sampling-c3", "config/sampling_c3_kik_t.yaml",
        "--max-time", str(max_time),
        "--seed", str(seed),
        "--name", name,
    ]
    stream = log.open("w")
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    print(f"[campaign] launched task={task} seed={seed} pid={proc.pid} log={log}",
          flush=True)
    return proc, log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--target", type=int, default=28)
    parser.add_argument("--max-time", type=float, default=600.0)
    args = parser.parse_args()
    if args.workers < 1 or args.target < 1 or args.max_time <= 0:
        parser.error("workers, target, and max-time must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[subprocess.Popen, Path]] = {}

    for task in ROSTER:
        existing = logs_for(task)
        next_seed = max((seed_of(p) for p in existing), default=-1) + 1
        successes = sum(classify(p) == "success" for p in existing)
        print(f"[campaign] task={task} resume successes={successes}/{args.target} "
              f"next_seed={next_seed}", flush=True)

        while successes < args.target or active:
            stop_requested = (OUT / "STOP").exists()
            remaining = max(0, args.target - successes - len(active))
            while not stop_requested and len(active) < args.workers and remaining > 0:
                proc, log = launch(task, next_seed, args.max_time)
                active[proc.pid] = (proc, log)
                next_seed += 1
                remaining -= 1
            snapshot(active, args.target, args.workers)

            if not active:
                if stop_requested:
                    print("[campaign] STOP acknowledged; no active trials", flush=True)
                    return 0
                continue

            time.sleep(5)
            for pid, (proc, log) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                state = classify(log)
                if state == "success":
                    successes += 1
                print(f"[campaign] finished task={task} pid={pid} rc={rc} "
                      f"state={state} successes={successes}/{args.target}", flush=True)
                del active[pid]

        print(f"[campaign] COMPLETE task={task} successes={successes}", flush=True)

    snapshot(active, args.target, args.workers)
    print("[campaign] ALL OBJECTS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
