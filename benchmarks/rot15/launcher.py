#!/usr/bin/env python3
"""Resolve verified rot15 cases into native C3+ and OIM launcher commands.

Dry-run is the default-safe smoke path.  A real launch is refused unless the
offline manifest, geometry preflight, and cross-method parity artifacts pass.
No 90-case campaign is started by benchmark generation or validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .benchmark import DEFAULT_OUTPUT, ROOT, SCENES, validate_artifacts

DEFAULT_C3_ROOT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
DEFAULT_OIM_ROOT = ROOT / "external/Object-Informed-Manipulation-MJX"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _load_adapters(output: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    c3 = yaml.safe_load((output / "c3plus_cases.yaml").read_text())["cases"]
    oim = yaml.safe_load((output / "oim_cases.yaml").read_text())["cases"]
    index = lambda rows: {(str(r["scene"]), str(r["case_id"])): r for r in rows}
    return index(c3), index(oim)


def _replace_scalar(path: Path, key: str, replacement: str) -> None:
    text = path.read_text()
    updated, count = re.subn(rf"(?m)^{re.escape(key)}:.*$", f"{key}: {replacement}", text)
    if count != 1:
        raise RuntimeError(f"{path}: expected one {key!r}, found {count}")
    path.write_text(updated)


def materialize_c3plus_case(case: Mapping[str, Any], c3_root: Path) -> str:
    """Create one additive per-case config directory; never overwrite it."""
    source_name = str(case["demo_source"])
    demo_name = str(case["generated_demo"])
    base = c3_root / "examples/sampling_c3"
    source, destination = base / source_name, base / demo_name
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing generated demo {destination}; "
            "use it only after independent inspection"
        )
    if not source.is_dir():
        raise FileNotFoundError(f"legacy g02 source demo is missing: {source}")
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        if path.is_file():
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            if source_name in text:
                path.write_text(text.replace(source_name, demo_name))
    params = destination / "parameters"
    start_q = [float(case[f"start_q{k}"]) for k in "wxyz"]
    start = start_q + [float(case[f"start_{axis}"]) for axis in "xyz"]
    goal_q = [float(case[f"goal_q{k}"]) for k in "wxyz"]
    goal = [float(case[f"goal_{axis}"]) for axis in "xyz"]
    fmt = lambda values: "[" + ", ".join(f"{value:.17g}" for value in values) + "]"
    _replace_scalar(params / "sim_params.yaml", "q_init_object", fmt(start))
    _replace_scalar(params / "sim_params.yaml", "q_init_objects", f"[{fmt(start)}]")
    _replace_scalar(params / "goal_params.yaml", "fixed_target_position", fmt(goal))
    _replace_scalar(params / "goal_params.yaml", "fixed_target_positions", f"[{fmt(goal)}]")
    _replace_scalar(params / "goal_params.yaml", "fixed_target_orientation", fmt(goal_q))
    _replace_scalar(params / "goal_params.yaml", "fixed_target_orientations", f"[{fmt(goal_q)}]")
    receipt = {
        "benchmark": "benchmark_v2_rot15",
        "scene": case["scene"], "case_id": case["case_id"],
        "seed_requested": int(case["seed"]),
        "seed_delivery": "manifest only; current C3+ perimeter sampler uses std::random_device",
        "source_demo": source_name,
    }
    (destination / "ROT15_TASK_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return demo_name


def _c3_command(case: Mapping[str, Any], output: Path, c3_root: Path, cap: int, port: int, materialize: bool) -> list[str]:
    demo = materialize_c3plus_case(case, c3_root) if materialize else str(case["generated_demo"])
    run_dir = output / "runs/c3plus" / str(case["scene"]) / str(case["case_id"])
    env_file = str(case.get("env_file", ""))
    if env_file:
        env_file = str(c3_root / env_file)
    return [
        "bash", str(c3_root / "tools/scene_smoke/run_scene_smoke.sh"),
        demo, "G_shape_video", str(case["goal_x"]), str(case["goal_y"]),
        str(2.0 * math.atan2(float(case["goal_qz"]), float(case["goal_qw"]))),
        str(cap), str(port), str(run_dir), env_file,
    ]


def _oim_command(case: Mapping[str, Any], output: Path, oim_root: Path, steps: int) -> list[str]:
    scene = str(case["oim_scene"])
    script = oim_root / "examples/pusht" / f"{scene}.py"
    poses = output / "oim_pose_overrides"
    return [
        sys.executable, "-m", "benchmarks.rot15.oim_entry",
        "--oim-root", str(oim_root), "--poses-dir", str(poses),
        "--script", str(script), "--", "admm", "--headless",
        "--start", str(case["start_key"]), "--goal", str(case["goal_key"]),
        "--seed", str(case["seed"]), "--steps", str(steps),
    ]


def _select(
    rows: Mapping[tuple[str, str], Mapping[str, Any]], scene: str | None,
    case_id: str | None, smoke_triplet: bool, all_cases: bool,
) -> list[Mapping[str, Any]]:
    if all_cases:
        return [rows[key] for key in sorted(rows)]
    if smoke_triplet:
        selected_scene = scene or "open_task"
        wanted = [
            f"s01_g02_{label}" for label in ("rot000", "rotCCW090", "rotCW090")
        ]
        return [rows[(selected_scene, value)] for value in wanted]
    if not scene or not case_id:
        raise ValueError("one case needs both --scene and --case-id")
    return [rows[(scene, case_id)]]


def _run_one(command: Sequence[str], cwd: Path, log: Path) -> int:
    if log.exists() or (log.parent.exists() and any(log.parent.iterdir())):
        raise FileExistsError(f"refusing to overwrite run output: {log.parent}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=cwd, text=True, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("c3plus", "oim", "both"), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", dest="all_cases")
    group.add_argument("--smoke-triplet", action="store_true")
    group.add_argument("--case-id")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize-c3plus", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c3-root", type=Path, default=DEFAULT_C3_ROOT)
    parser.add_argument("--oim-root", type=Path, default=DEFAULT_OIM_ROOT)
    parser.add_argument("--cap", type=int, default=600)
    parser.add_argument("--lanes", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--port-base", type=int, default=18000)
    args = parser.parse_args(argv)

    validate_artifacts(args.output)
    if args.all_cases and not args.dry_run:
        raise RuntimeError(
            "full campaign safety gate: remove this guard only after the C3+ "
            "random_device seed-delivery blocker in benchmark_sync_report.md is resolved"
        )
    c3_index, oim_index = _load_adapters(args.output)
    selected = _select(c3_index, args.scene, args.case_id, args.smoke_triplet, args.all_cases)
    commands: list[tuple[str, Mapping[str, Any], list[str], Path]] = []
    methods = ("c3plus", "oim") if args.method == "both" else (args.method,)
    for index, c3_case in enumerate(selected, start=1):
        key = (str(c3_case["scene"]), str(c3_case["case_id"]))
        for method in methods:
            if method == "c3plus":
                command = _c3_command(
                    c3_case, args.output, args.c3_root, args.cap,
                    args.port_base + index, args.materialize_c3plus and not args.dry_run,
                )
                cwd = args.c3_root
            else:
                oim_case = oim_index[key]
                command = _oim_command(oim_case, args.output, args.oim_root, args.cap)
                cwd = ROOT
            commands.append((method, c3_case, command, cwd))

    for method, case, command, _ in commands:
        print(json.dumps({
            "method": method, "scene": case["scene"], "case_id": case["case_id"],
            "seed": int(case["seed"]), "command": command,
        }, sort_keys=True))
    if args.dry_run:
        print(f"LAUNCHER_SMOKE_PASS commands={len(commands)} simulations_started=0")
        return 0

    jobs = []
    with ThreadPoolExecutor(max_workers=min(args.lanes, len(commands))) as pool:
        for method, case, command, cwd in commands:
            log = args.output / "runs" / method / str(case["scene"]) / str(case["case_id"]) / "launcher.log"
            jobs.append((method, case, pool.submit(_run_one, command, cwd, log)))
        failed = 0
        for method, case, future in jobs:
            rc = future.result()
            print(f"{method}/{case['scene']}/{case['case_id']}: rc={rc}")
            failed += rc != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
