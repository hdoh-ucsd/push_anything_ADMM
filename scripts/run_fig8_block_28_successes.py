#!/usr/bin/env python3
"""Run one uninterrupted 28-goal C3+ manipulation session per Fig. 8 object.

Despite the historical filename, this is not a multi-seed success collector.
Each object is launched exactly once. The object, robot, controller, and random
goal-generator state remain continuous until all requested SE(2) goals are
reached or that object's session fails. A failed session is never replaced.

Independent one-goal results remain under ``results/fig8_mesh_28_c3plus``.
Corrected sessions are written to ``results/fig8_consecutive_28_c3plus``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/miniconda3/envs/push_anything_ADMM/bin/python3")
OUT = ROOT / "results" / "fig8_consecutive_28_c3plus"
ROSTER = [
    "I_shape_texture", "C_shape_texture", "R_shape_texture", "A_shape_video",
    "Y_shape_video", "G_shape_video", "B_shape_video", "3_shape_video",
    "H_shape_texture", "E_shape_video", "S_shape", "expo_box", "lotion_block",
    "wood_block", "tape", "eraser", "milk", "clamp_block", "chicken_broth",
    "egg_carton", "book", "baby_toy", "gallon_milk", "xbox", "push_t_mesh",
]


def completion_marker(target: int) -> str:
    return f"[GOAL-GEN] COMPLETE: {target} goals achieved"


def diagnose(path: Path, target: int) -> dict[str, object]:
    """Return a mutually exclusive, evidence-backed terminal classification."""
    result: dict[str, object] = {
        "state": "missing", "failure_category": None, "goals_reached": 0,
        "failed_goal_index": 1, "translational_error_m": None,
        "rotational_error_rad": None, "loose_goal": None,
    }
    if not path.is_file():
        return result
    text = path.read_text(errors="replace")
    reached = [int(value) for value in re.findall(
        r"\[GOAL-GEN\] goal #(\d+) REACHED", text)]
    goals_reached = max(reached, default=0)
    campaign = re.search(
        r"\[CAMPAIGN-RESULT\] requested_goals=(\d+) goals_reached=(\d+) "
        r"status=(PASS|FAIL)", text)
    if campaign:
        goals_reached = int(campaign.group(2))
    pose = re.findall(
        r"translational_error=([\d.eE+-]+)m\s+rotational_error=([\d.eE+-]+)rad"
        r".*?loose_goal=(PASS|FAIL)", text)
    if pose:
        result["translational_error_m"] = float(pose[-1][0])
        result["rotational_error_rad"] = float(pose[-1][1])
        result["loose_goal"] = pose[-1][2]
    result["goals_reached"] = goals_reached
    result["failed_goal_index"] = None if goals_reached >= target else goals_reached + 1

    if (completion_marker(target) in text and campaign
            and campaign.group(1) == str(target)
            and campaign.group(2) == str(target) and campaign.group(3) == "PASS"):
        result.update(state="success", failure_category=None)
    elif campaign and campaign.group(1) != str(target):
        result.update(state="failure", failure_category="protocol_mismatch")
    elif "consecutive campaign wall timeout" in text:
        result.update(state="failure", failure_category="wall_clock_timeout")
    elif "[ABORT-UNREACHABLE]" in text:
        result.update(state="failure", failure_category="persistent_topple")
    elif "[WORKSPACE-VIOLATION]" in text and "Traceback" in text:
        result.update(state="failure", failure_category="workspace_limit")
    elif "[ABORT] Object position blowup" in text:
        result.update(state="failure", failure_category="numerical_divergence")
    elif "[WARN] NaN in state" in text:
        result.update(state="failure", failure_category="nonfinite_state")
    elif "[GOAL-TIMEOUT]" in text and result["loose_goal"] == "PASS":
        result.update(state="failure", failure_category="tight_tolerance_timeout")
    elif "[GOAL-TIMEOUT]" in text:
        result.update(state="failure", failure_category="goal_timeout")
    elif campaign and campaign.group(3) == "FAIL" and result["loose_goal"] == "PASS":
        result.update(state="failure", failure_category="tight_tolerance_timeout")
    elif campaign and campaign.group(3) == "FAIL":
        result.update(state="failure", failure_category="goal_timeout")
    elif "Traceback (most recent call last)" in text:
        result.update(state="failure", failure_category="software_exception")
    else:
        result.update(state="incomplete", failure_category="interrupted_or_truncated")
    return result


def classify(path: Path, target: int) -> str:
    return str(diagnose(path, target)["state"])


def write_status(path: Path, *, task: str, seed: int, target: int,
                 diagnosis: dict[str, object], log: Path,
                 returncode: int | None = None) -> None:
    payload = {
        "protocol": "consecutive_se2_goals", "task": task, "seed": seed,
        "requested_goals": target,
        "log": str(log.relative_to(ROOT)), "returncode": returncode,
        "updated_unix": time.time(),
        **diagnosis,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def launch(task: str, seed: int, target: int, max_time: float,
           goal_timeout: float) -> tuple[subprocess.Popen, Path]:
    task_dir = OUT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    log = task_dir / f"{task}_consecutive{target}.txt"
    env = os.environ.copy()
    env.update({
        "PORT_GOAL_MODE": "kRandom",
        "PORT_GOALGEN_N": str(target),
        "PORT_GOAL_TIMEOUT_S": str(goal_timeout),
    })
    name = f"fig8_consecutive_28_c3plus/{task}/{task}_consecutive{target}"
    cmd = [str(PYTHON), "main.py", task, "--solver", "c3plus",
           "--sampling-c3", "config/sampling_c3_kik_t.yaml",
           "--max-time", str(max_time), "--seed", str(seed), "--name", name]
    stream = log.open("w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
    stream.close()
    print(f"[campaign] launched task={task} seed={seed} pid={proc.pid} log={log}",
          flush=True)
    return proc, log


def wait_for_session(proc: subprocess.Popen, log: Path, wall_timeout: float) -> int:
    started = time.monotonic()
    while proc.poll() is None:
        if time.monotonic() - started >= wall_timeout:
            elapsed = time.monotonic() - started
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            with log.open("a") as stream:
                stream.write(f"\n[RESULT] FAILURE: consecutive campaign wall timeout "
                             f"after {elapsed:.1f}s (limit={wall_timeout:.1f}s)\n")
            return int(proc.returncode or 1)
        time.sleep(5)
    return int(proc.returncode or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0,
                        help="Single reproducibility seed for each continuous session.")
    parser.add_argument("--goal-timeout", type=float, default=1200.0,
                        help="Simulated seconds allowed for each individual goal.")
    parser.add_argument("--max-time", type=float, default=None,
                        help="Total simulated seconds (default: target * goal-timeout).")
    parser.add_argument("--wall-timeout", type=float, default=43200.0,
                        help="Wall-clock seconds allowed per object (default: 12 hours).")
    parser.add_argument("--start-object", type=int, default=1)
    parser.add_argument("--end-object", type=int, default=len(ROSTER))
    args = parser.parse_args()
    if args.max_time is None:
        args.max_time = args.target * args.goal_timeout
    if (args.target < 1 or args.goal_timeout <= 0 or args.max_time <= 0
            or args.wall_timeout <= 0):
        parser.error("target, goal-timeout, max-time, and wall-timeout must be positive")
    if not (1 <= args.start_object <= args.end_object <= len(ROSTER)):
        parser.error(f"object range must satisfy 1 <= start <= end <= {len(ROSTER)}")

    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for task in ROSTER[args.start_object - 1:args.end_object]:
        task_dir = OUT / task
        # Deliberately seed-independent path: changing --seed cannot create a
        # replacement session beside an existing outcome for the same object.
        log = task_dir / f"{task}_consecutive{args.target}.txt"
        status = task_dir / "status.json"
        prior = classify(log, args.target)
        if prior != "missing":
            print(f"[campaign] preserving existing task={task} state={prior} log={log}",
                  flush=True)
            diagnosis = diagnose(log, args.target)
            write_status(status, task=task, seed=args.seed, target=args.target,
                         diagnosis=diagnosis, log=log)
            failures += prior != "success"
            continue
        if (OUT / "STOP").exists():
            print("[campaign] STOP acknowledged before next object", flush=True)
            break

        proc, log = launch(task, args.seed, args.target, args.max_time,
                           args.goal_timeout)
        write_status(status, task=task, seed=args.seed, target=args.target,
                     diagnosis={"state": "running", "failure_category": None,
                                "goals_reached": 0, "failed_goal_index": None},
                     log=log)
        rc = wait_for_session(proc, log, args.wall_timeout)
        diagnosis = diagnose(log, args.target)
        state = str(diagnosis["state"])
        write_status(status, task=task, seed=args.seed, target=args.target,
                     diagnosis=diagnosis, log=log, returncode=rc)
        print(f"[campaign] finished task={task} rc={rc} state={state} "
              f"failure_category={diagnosis['failure_category']} "
              f"goals_reached={diagnosis['goals_reached']}", flush=True)
        failures += state != "success"

    print(f"[campaign] finished protocol=consecutive_se2_goals failures={failures}",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
