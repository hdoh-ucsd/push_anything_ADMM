#!/usr/bin/env python3
"""Fail-closed serial supervisor for the final 450-run C3+ study.

Normal invocation performs read-only preflight.  Live execution requires both
``--execute`` and an approval receipt created only after explicit user
approval.  A per-run helper is a session/process-group leader; all run-owned
children inherit that group and Linux parent-death signaling.  Cleanup never
uses process-name-wide signals.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
OUT = ROOT / "results/c3plus_geometry_rotation_cost_replication"
MANIFEST = OUT / "manifest/experiment_manifest.csv"
CONFIG = OUT / "configs/cost_configurations.json"
APPROVAL = OUT / "USER_APPROVAL.json"
BIN = WT / "bazel-bin/examples/sampling_c3"
PYTHON = Path("/root/miniconda3/envs/push_anything_ADMM/bin/python3")
RECORDER = WT / "tools/scene_smoke/record_metrics.py"
VALIDATOR = WT / "scripts/validate_forensics_logs.py"
OWNED_RUNNER = Path(__file__).resolve()
MAX_CONCURRENT_LIVE_SIMULATIONS = 1
CONTROLLER_CPUS = {2, 3, 4, 5}
SIM_CPUS = {6, 7}
RECORDER_CPUS = {0, 1}
TERM_GRACE_S = 3.0
PARENT_HELPER_GRACE_S = 20.0
PR_SET_PDEATHSIG = 1
GATE = {
    "mean_hz_min": 68.885794604,
    "median_period_s_max": 0.0154,
    "p95_period_s_max": 0.0176,
    "sim_wall_min": 0.571598162,
}
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "3", "OMP_THREAD_LIMIT": "3", "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1", "GOTO_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 450 or len({r["run_id"] for r in rows}) != 450:
        raise RuntimeError("manifest must contain 450 unique run IDs")
    return rows


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_event(run_id: str, status: str, **extra: Any) -> None:
    path = OUT / "state/status_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_id": run_id,
             "status": status, **extra}
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        close = stat.rfind(")")
        fields = stat[close + 2:].split()
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        allowed = next(line.split(":", 1)[1].strip() for line in status if line.startswith("Cpus_allowed_list:"))
        return {"pid": pid, "comm": stat[stat.find("(") + 1:close], "state": fields[0],
                "ppid": int(fields[1]), "pgid": int(fields[2]), "allowed_cpus": allowed,
                "command": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()}
    except (FileNotFoundError, ProcessLookupError, StopIteration, ValueError):
        return None


def processes_in_group(pgid: int) -> list[dict[str, Any]]:
    result = []
    for path in Path("/proc").glob("[0-9]*"):
        item = proc_identity(int(path.name))
        if item and item["pgid"] == pgid:
            result.append(item)
    return sorted(result, key=lambda row: row["pid"])


def unrelated_heavy_stack() -> list[dict[str, Any]]:
    blockers = []
    tokens = ("franka_sim", "franka_sampling_c3_controller", "run_scene_smoke.sh",
              "run_c3plus_obstacle_cost_tuning", "run_c3plus_task_cost_ratio")
    for path in Path("/proc").glob("[0-9]*"):
        item = proc_identity(int(path.name))
        if item and any(token in item["command"] for token in tokens):
            if item["pid"] not in {os.getpid(), os.getppid()}:
                blockers.append(item)
    return blockers


def instantaneous_cpu_load(interval_s: float = 0.5) -> dict[str, Any]:
    def cpu_totals() -> tuple[int, int]:
        values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    def process_ticks() -> dict[int, tuple[int, str]]:
        result: dict[int, tuple[int, str]] = {}
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                text = path.read_text(); close = text.rfind(")"); fields = text[close + 2:].split()
                result[int(path.parent.name)] = (int(fields[11]) + int(fields[12]), text[text.find("(") + 1:close])
            except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
                continue
        return result
    total0, idle0 = cpu_totals(); proc0 = process_ticks(); started = time.monotonic()
    time.sleep(interval_s)
    elapsed = time.monotonic() - started; total1, idle1 = cpu_totals(); proc1 = process_ticks()
    busy_fraction = 1.0 - (idle1 - idle0) / max(1, total1 - total0)
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    heavy = []
    for pid, (ticks1, comm) in proc1.items():
        if pid in proc0 and pid not in {os.getpid(), os.getppid()}:
            percent_one_cpu = (ticks1 - proc0[pid][0]) / ticks_per_second / elapsed * 100.0
            if percent_one_cpu >= 50.0:
                item = proc_identity(pid)
                if item: heavy.append({**item, "instant_cpu_percent": percent_one_cpu})
    return {"system_busy_fraction": busy_fraction, "heavy_processes": sorted(heavy, key=lambda row: -row["instant_cpu_percent"])}


def machine_gate() -> dict[str, Any]:
    named = unrelated_heavy_stack()
    load = instantaneous_cpu_load()
    # A half-core unrelated process is meaningful beside a latency-sensitive
    # three-core lane.  The total-load check catches distributed experiments.
    passed = not named and not load["heavy_processes"] and load["system_busy_fraction"] < 0.20
    return {"passed": passed, "named_simulation_blockers": named, **load}


def source_environment(base: dict[str, str], path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(base)
    result = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; env -0', "bash", str(path)],
        cwd=WT, env=base, check=True, capture_output=True,
    )
    env: dict[str, str] = {}
    for item in result.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def child_setup(cpus: set[int]) -> Callable[[], None]:
    def setup() -> None:
        libc = ctypes.CDLL(None)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
            os._exit(126)
        os.sched_setaffinity(0, cpus)
    return setup


def parse_cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); result.update(range(lo, hi + 1))
        elif part:
            result.add(int(part))
    return result


def verify_affinity(pid: int, expected: set[int]) -> str:
    item = proc_identity(pid)
    if item is None:
        raise RuntimeError(f"PID {pid} exited before affinity verification")
    actual = parse_cpu_list(str(item["allowed_cpus"]))
    if actual != expected:
        raise RuntimeError(f"PID {pid} affinity {sorted(actual)} != {sorted(expected)}")
    return str(item["allowed_cpus"])


def run_environment(row: dict[str, str], attempt: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("SAMPLING_C3_OBS_RELU") or key.startswith("SAMPLING_C3_OBS_EXP") or key in {
            "SAMPLING_C3_INNER_OBS_MODE", "SAMPLING_C3_OBJ_NONPEN", "SAMPLING_C3_RANK_OBS_MODE",
            "SAMPLING_C3_EXPERIMENT_SEED", "SAMPLING_C3_TRANSLATION_COST_SCALE",
            "SAMPLING_C3_ORIENTATION_COST_SCALE", "SAMPLING_C3_FORENSICS_LOG_DIR",
        }:
            env.pop(key)
    env_file = WT / row["env_file"] if row["env_file"] else None
    env = source_environment(env, env_file)
    tmp = attempt / "tmp"; tmp.mkdir()
    forensics = attempt / "forensics"; forensics.mkdir()
    env.update(THREAD_LIMITS)
    env.update({
        "TMPDIR": str(tmp),
        "SAMPLING_C3_EXPERIMENT_SEED": row["seed"],
        "SAMPLING_C3_TRANSLATION_COST_SCALE": row["alpha_p"],
        "SAMPLING_C3_ORIENTATION_COST_SCALE": row["alpha_theta"],
        "SAMPLING_C3_FORENSICS_LOG_DIR": str(forensics),
        "SAMPLING_C3_FORENSICS_TEXT_EVENTS": "0",
        "SAMPLING_C3_FORENSICS_GIT_COMMIT": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WT, text=True).strip(),
        "SAMPLING_C3_FORENSICS_TASK_ID": row["run_id"],
        "SAMPLING_C3_FORENSICS_ROBOT_MODEL": "xarm6",
        "SAMPLING_C3_OBSTACLE_MODE": "lcs_contact",
        "SAMPLING_C3_STOP_RECORDER_ON_PLANNER_EXIT": "1",
    })
    if row["obstacle_family"] == "relu_footprint":
        env.update({"SAMPLING_C3_RANK_OBS_MODE": "relu_footprint",
                    "SAMPLING_C3_OBS_RELU_EPS": row["obstacle_range_m"],
                    "SAMPLING_C3_OBS_RELU_W": row["obstacle_weight"]})
    elif row["obstacle_family"] == "exponential_footprint":
        env.update({"SAMPLING_C3_RANK_OBS_MODE": "exponential_footprint",
                    "SAMPLING_C3_OBS_EXP_SIGMA": row["obstacle_range_m"],
                    "SAMPLING_C3_OBS_EXP_W": row["obstacle_weight"]})
    else:
        raise ValueError(row["obstacle_family"])
    return env


def terminate_owned_group(
        pgid: int,
        processes: dict[str, subprocess.Popen[Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"pgid": pgid, "sigterm_sent": False, "sigkill_pids": []}
    recorder = processes.get("recorder")
    if recorder is not None and recorder.poll() is None:
        recorder.send_signal(signal.SIGTERM)
        try: recorder.wait(timeout=1.0)
        except subprocess.TimeoutExpired: pass
    owned = [p for p in processes_in_group(pgid) if p["pid"] != os.getpid()]
    if owned:
        previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            os.killpg(pgid, signal.SIGTERM); result["sigterm_sent"] = True
        except ProcessLookupError:
            pass
        finally:
            signal.signal(signal.SIGTERM, previous)
    deadline = time.monotonic() + TERM_GRACE_S
    while time.monotonic() < deadline:
        if not [p for p in processes_in_group(pgid) if p["pid"] != os.getpid()]: break
        time.sleep(0.05)
    remaining = [p for p in processes_in_group(pgid) if p["pid"] != os.getpid()]
    for item in remaining:
        try:
            os.kill(item["pid"], signal.SIGKILL); result["sigkill_pids"].append(item["pid"])
        except ProcessLookupError:
            pass
    # Reap direct children before inspecting /proc.  A killed-but-unreaped
    # child remains visible as a zombie in the owned process group and must
    # not be mistaken for a live cleanup failure.
    for proc in processes.values():
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(0.1)
    result["remaining"] = [p["pid"] for p in processes_in_group(pgid) if p["pid"] != os.getpid()]
    result["passed"] = not result["remaining"]
    return result


def owned_run(row_path: Path, attempt: Path) -> int:
    row = json.loads(row_path.read_text())
    if attempt.exists():
        raise FileExistsError(attempt)
    attempt.mkdir(parents=True)
    live = attempt / "live"; live.mkdir()
    env = run_environment(row, attempt)
    url = f"udpm://239.255.76.67:{row['lcm_port']}?ttl=0"
    common = ["--is_simulation=true", f"--demo_name={row['demo_name']}", "--robot_model=xarm6", f"--lcm_url={url}"]
    commands = {
        "osc": [str(BIN / "franka_osc_controller"), *common],
        "planner": [str(BIN / "franka_sampling_c3_controller"), *common],
        "recorder": [str(PYTHON), str(RECORDER), "--goal", row["goal_x"], row["goal_y"], row["goal_yaw_rad"],
                     "--object-name", "G_shape_video", "--out-steps", str(live / "steps_raw.jsonl"),
                     "--out-trace", str(live / "state_trace.jsonl"), "--url", url, "--duration", row["cap_wall_s"],
                     "--pos-tol", "0.02", "--ang-tol", "0.1", "--exit-on-success"],
        "sim": [str(BIN / "franka_sim"), f"--demo_name={row['demo_name']}", "--robot_model=xarm6", "--matched_mu", f"--lcm_url={url}"],
    }
    affinity = {"osc": SIM_CPUS, "planner": CONTROLLER_CPUS, "recorder": RECORDER_CPUS, "sim": SIM_CPUS}
    processes: dict[str, subprocess.Popen[Any]] = {}
    logs: dict[str, Any] = {}
    pgid = os.getpgrp()
    terminal_reason = "UNKNOWN"
    recorder_started = math.nan
    try:
        for role in ("osc", "planner", "recorder"):
            logs[role] = (live / f"{role}.log").open("x")
            processes[role] = subprocess.Popen(commands[role], cwd=WT, env=env, stdin=subprocess.DEVNULL,
                stdout=logs[role], stderr=subprocess.STDOUT, preexec_fn=child_setup(affinity[role]))
            if role == "recorder": recorder_started = time.monotonic()
        time.sleep(3.0)
        logs["sim"] = (live / "sim.log").open("x")
        processes["sim"] = subprocess.Popen(commands["sim"], cwd=WT, env=env, stdin=subprocess.DEVNULL,
            stdout=logs["sim"], stderr=subprocess.STDOUT, preexec_fn=child_setup(affinity["sim"]))
        verified = {role: verify_affinity(proc.pid, affinity[role]) for role, proc in processes.items()}
        receipt = {"run_id": row["run_id"], "launcher_pid": os.getpid(), "process_group_id": pgid,
                   "child_pids": {role: proc.pid for role, proc in processes.items()}, "verified_affinity": verified,
                   "commands": commands, "lcm_url": url, "environment": {k: env[k] for k in sorted(env) if k.startswith("SAMPLING_C3_") or k in THREAD_LIMITS},
                   "manifest_row": row, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        atomic_json(attempt / "run_receipt.json", receipt)
        while True:
            recorder = processes["recorder"]
            if recorder.poll() is not None:
                terminal_reason = "RECORDER_EXIT"
                break
            failed = next((role for role in ("planner", "sim", "osc") if processes[role].poll() is not None), None)
            if failed:
                terminal_reason = f"{failed.upper()}_EXIT"
                break
            if time.monotonic() - recorder_started >= float(row["cap_wall_s"]):
                terminal_reason = "SUPERVISOR_WALL_TIMEOUT"
                break
            time.sleep(0.5)
    except BaseException as error:
        terminal_reason = f"LAUNCHER_EXCEPTION:{type(error).__name__}:{error}"
    finally:
        cleanup = terminate_owned_group(pgid, processes)
        for stream in logs.values(): stream.close()
        atomic_json(attempt / "owned_cleanup.json", {"terminal_reason": terminal_reason, **cleanup,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    valid_terminal = terminal_reason in {"RECORDER_EXIT", "SUPERVISOR_WALL_TIMEOUT"}
    return 0 if cleanup["passed"] and valid_terminal else 2


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] * (hi - index) + values[hi] * (index - lo)


def timing_result(attempt: Path, wall_s: float) -> dict[str, Any]:
    path = attempt / "forensics/forensics_cycle.csv"
    with path.open(newline="") as stream: cycles = list(csv.DictReader(stream))
    times = [float(row["time"]) for row in cycles]
    periods = [b - a for a, b in zip(times, times[1:]) if b > a]
    if len(periods) < 10: raise RuntimeError("insufficient timing cycles")
    sim_s = times[-1] - times[0]
    median = statistics.median(periods)
    metrics = {"cycle_count": len(cycles), "simulation_time_s": sim_s,
               "wall_time_s": wall_s, "mean_controller_hz": len(periods) / sim_s,
               "median_controller_hz": 1.0 / median, "median_period_s": median,
               "p95_period_s": percentile(periods, .95), "p99_period_s": percentile(periods, .99),
               "max_period_s": max(periods), "late_cycle_fraction": sum(p > 2 * median for p in periods) / len(periods),
               "sim_wall_ratio": sim_s / max(1e-9, wall_s - 3.0)}
    checks = {"mean_hz": metrics["mean_controller_hz"] >= GATE["mean_hz_min"],
              "median_period": metrics["median_period_s"] <= GATE["median_period_s_max"],
              "p95_period": metrics["p95_period_s"] <= GATE["p95_period_s_max"],
              "sim_wall": metrics["sim_wall_ratio"] >= GATE["sim_wall_min"]}
    return {**metrics, "gate": GATE, "checks": checks, "timing_valid": all(checks.values())}


def bounded_validator_tail(stdout: str) -> bool:
    errors = [line for line in stdout.splitlines() if line.startswith("- ")]
    allowed = ("truncated CSV row", "candidate rows, expected", "selected candidate count is 0, expected 1")
    if not errors or not all(any(fragment in line for fragment in allowed) for line in errors): return False
    cycles = re.search(r"\bcycles=(\d+)", stdout)
    event_ids = [int(value) for value in re.findall(r"\bevent (\d+)", "\n".join(errors))]
    return bool(cycles and event_ids and min(event_ids) >= int(cycles.group(1)) - 4)


def recorder_final(attempt: Path) -> dict[str, Any]:
    text = (attempt / "live/recorder.log").read_text(errors="replace")
    matches = re.findall(r"(?m)^FINAL (\{.*\})$", text)
    if not matches: return {"first_success_t": None, "missing_final": True}
    return json.loads(matches[-1])


def accepted_attempt(row: dict[str, str]) -> Path | None:
    run_root = OUT / "runs" / row["run_id"]
    marker = run_root / "ACCEPTED_RUN.json"
    if not marker.exists(): return None
    return run_root / json.loads(marker.read_text())["accepted_attempt"]


def first_event(attempt: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    with (attempt / "forensics/forensics_cycle.csv").open(newline="") as stream:
        cycle = next(csv.DictReader(stream))
    with (attempt / "forensics/forensics_candidates.csv").open(newline="") as stream:
        reader = csv.DictReader(stream); first = next(reader); event_id = first["event_id"]
        candidates = [first]
        for row in reader:
            if row["event_id"] != event_id: break
            candidates.append(row)
    return cycle, candidates


def close_number(a: str, b: str, atol: float = 1e-7, rtol: float = 1e-7) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return math.isclose(x, y, abs_tol=atol, rel_tol=rtol)


def check_open_negative_control(block_index: str, rows: list[dict[str, str]]) -> None:
    block = [row for row in rows if row["block_index"] == block_index and row["scene"] == "open_task"]
    if not block: return
    pairs = (("C1", "C2"), ("C3", "C4"))
    audit_dir = OUT / "negative_control"; audit_dir.mkdir(parents=True, exist_ok=True)
    for left_id, right_id in pairs:
        left = next(row for row in block if row["config_id"] == left_id)
        right = next(row for row in block if row["config_id"] == right_id)
        left_attempt, right_attempt = accepted_attempt(left), accepted_attempt(right)
        output = audit_dir / f"block_{int(block_index):03d}_{left_id}_{right_id}.json"
        if output.exists() or left_attempt is None or right_attempt is None: continue
        lc, lp = first_event(left_attempt); rc, rp = first_event(right_attempt)
        checks: dict[str, Any] = {
            "same_task_base": left["task_base"] == right["task_base"],
            "same_candidate_count": len(lp) == len(rp),
            "initial_object_state": all(close_number(lc[key], rc[key], atol=2e-6) for key in ("obj_x", "obj_y", "obj_yaw", "obj_z")),
            "all_obstacle_costs_zero": all(abs(float(row["J_obstacle_applied_live"])) <= 1e-12 for row in lp + rp),
        }
        identity_fields = ("candidate_id", "candidate_source", "face_id", "face_bin_id", "contact_sector_id")
        sample_fields = ("candidate_object_frame_x", "candidate_object_frame_y")
        cost_fields = ("J_translation", "J_orientation", "J_task_code", "J_rank_code_total")
        checks["candidate_pool_identity"] = len(lp) == len(rp) and all(
            all(a[field] == b[field] for field in identity_fields)
            and all(close_number(a[field], b[field], atol=1e-9) for field in sample_fields)
            for a, b in zip(lp, rp)
        )
        checks["candidate_cost_identity"] = len(lp) == len(rp) and all(
            all(close_number(a[field], b[field], atol=1e-6, rtol=1e-7) for field in cost_fields)
            for a, b in zip(lp, rp)
        )
        left_selected = [row["candidate_id"] for row in lp if row["selected"] == "1"]
        right_selected = [row["candidate_id"] for row in rp if row["selected"] == "1"]
        checks["same_initial_selection"] = left_selected == right_selected and len(left_selected) == 1
        payload = {"block_index": int(block_index), "pair": [left_id, right_id], "checks": checks,
                   "passed": all(checks.values()), "left_attempt": str(left_attempt), "right_attempt": str(right_attempt)}
        atomic_json(output, payload)
        if not payload["passed"]:
            atomic_json(OUT / "state/PAUSED_OPEN_TASK_NEGATIVE_CONTROL.json", payload)
            raise RuntimeError(f"open-task negative control failed: {left_id}/{right_id}")


def next_attempt(root: Path) -> Path:
    attempts = root / "attempts"; attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts / f"attempt_{number:02d}").exists(): number += 1
    return attempts / f"attempt_{number:02d}"


def launch_one(row: dict[str, str]) -> str:
    load_gate = machine_gate()
    if not load_gate["passed"]:
        atomic_json(OUT / "state/PAUSED_FOR_MACHINE_LOAD.json", {"status": "PAUSED_FOR_MACHINE_LOAD", **load_gate})
        raise RuntimeError("PAUSED_FOR_MACHINE_LOAD")
    run_root = OUT / "runs" / row["run_id"]
    accepted = run_root / "ACCEPTED_RUN.json"
    if accepted.exists(): return "ALREADY_ACCEPTED"
    attempt = next_attempt(run_root)
    row_path = OUT / "state/current_run.json"
    atomic_json(row_path, row)
    append_event(row["run_id"], "RUNNING", attempt=str(attempt))
    started = time.monotonic()
    helper = subprocess.Popen([str(PYTHON), str(OWNED_RUNNER), "--owned-run", str(row_path), "--attempt", str(attempt)],
                              cwd=ROOT, start_new_session=True)
    helper_pgid = helper.pid
    try:
        rc = helper.wait(timeout=float(row["cap_wall_s"]) + PARENT_HELPER_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(helper_pgid, signal.SIGTERM)
        try: helper.wait(timeout=TERM_GRACE_S)
        except subprocess.TimeoutExpired: os.killpg(helper_pgid, signal.SIGKILL); helper.wait()
        rc = 124
    wall = time.monotonic() - started
    remaining = processes_in_group(helper_pgid)
    if remaining:
        for item in remaining:
            try: os.kill(item["pid"], signal.SIGKILL)
            except ProcessLookupError: pass
        append_event(row["run_id"], "CLEANUP_FAILURE", remaining=[p["pid"] for p in remaining])
        return "CLEANUP_FAILURE"
    cleanup_path = attempt / "owned_cleanup.json"
    if rc != 0 or not cleanup_path.exists() or not json.loads(cleanup_path.read_text()).get("passed"):
        append_event(row["run_id"], "RUNTIME_FAILURE", helper_returncode=rc)
        return "RUNTIME_FAILURE"
    validation = subprocess.run([str(PYTHON), str(VALIDATOR), str(attempt / "forensics")], cwd=WT, text=True, capture_output=True)
    (attempt / "validation.log").write_text(f"returncode={validation.returncode}\n{validation.stdout}{validation.stderr}")
    bounded_tail = validation.returncode != 0 and bounded_validator_tail(validation.stdout)
    if validation.returncode != 0 and not bounded_tail:
        append_event(row["run_id"], "RUNTIME_FAILURE", reason="forensics_validation")
        return "RUNTIME_FAILURE"
    timing = timing_result(attempt, wall)
    atomic_json(attempt / "timing.json", timing)
    if not timing["timing_valid"]:
        append_event(row["run_id"], "TIMING_INVALID", timing=timing)
        return "TIMING_INVALID"
    final = recorder_final(attempt)
    status = "SUCCESS" if final.get("first_success_t") is not None else "TIMEOUT"
    acceptance = {"status": status, "accepted_attempt": str(attempt.relative_to(run_root)),
                  "timing": timing, "recorder_final": final, "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_json(accepted, acceptance)
    append_event(row["run_id"], status, accepted_attempt=acceptance["accepted_attempt"])
    return status


def preflight() -> int:
    rows = read_manifest(); config = json.loads(CONFIG.read_text())
    demos = [WT / "examples/sampling_c3" / row["demo_name"] for row in rows]
    errors = []
    if config["max_concurrent_live_simulations"] != 1: errors.append("concurrency is not one")
    if len({row["lcm_port"] for row in rows}) != 450: errors.append("ports are not unique")
    if any(not demo.is_dir() for demo in demos): errors.append("materialized demo missing")
    if set(row["seed"] for row in rows) != {"0", "1"}: errors.append("seed set differs from {0,1}")
    costs = {item["config_id"]: item for item in config["costs"]}
    if (costs["C1"]["task_base"], costs["C1"]["alpha_p"], costs["C1"]["alpha_theta"]) != \
       (costs["C2"]["task_base"], costs["C2"]["alpha_p"], costs["C2"]["alpha_theta"]):
        errors.append("C1/C2 task bases differ")
    if (costs["C3"]["task_base"], costs["C3"]["alpha_p"], costs["C3"]["alpha_theta"]) != \
       (costs["C4"]["task_base"], costs["C4"]["alpha_p"], costs["C4"]["alpha_theta"]):
        errors.append("C3/C4 task bases differ")
    counts = {(cost, scene): sum(r["config_id"] == cost and r["scene"] == scene for r in rows)
              for cost in ("C1", "C2", "C3", "C4", "C5") for scene in ("open_task", "single_obstacle", "shelf_gap")}
    if any(value != 30 for value in counts.values()): errors.append("cost/scene cells are not 30")
    payload = {"status": "PASS" if not errors else "FAIL", "errors": errors, "manifest_sha256": sha256(MANIFEST),
               "rows": len(rows), "cost_scene_counts": {f"{key[0]}/{key[1]}": value for key, value in counts.items()},
               "machine_gate": machine_gate(), "scientific_runs_started": False,
               "approval_present": APPROVAL.exists()}
    atomic_json(OUT / "provenance/latest_preflight.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 1


def execute() -> int:
    if not APPROVAL.exists():
        raise RuntimeError(f"explicit approval receipt absent: {APPROVAL}")
    approval = json.loads(APPROVAL.read_text())
    if approval.get("manifest_sha256") != sha256(MANIFEST) or approval.get("approved") is not True:
        raise RuntimeError("approval receipt does not authorize this exact manifest")
    rows = read_manifest()
    queue = deque(row for row in rows if not (OUT / "runs" / row["run_id"] / "ACCEPTED_RUN.json").exists())
    consecutive_invalid = 0
    while queue:
        row = queue.popleft()
        outcome = launch_one(row)
        if outcome in {"SUCCESS", "TIMEOUT", "ALREADY_ACCEPTED"}:
            consecutive_invalid = 0
            check_open_negative_control(row["block_index"], rows)
        elif outcome == "TIMING_INVALID":
            consecutive_invalid += 1; queue.append(row)
            if consecutive_invalid >= 2:
                raise RuntimeError("PAUSED_AFTER_REPEATED_TIMING_FAILURES")
        else:
            raise RuntimeError(f"campaign paused after {outcome}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--owned-run", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.owned_run:
        if args.attempt is None: parser.error("--owned-run requires --attempt")
        return owned_run(args.owned_run, args.attempt.resolve())
    if args.execute: return execute()
    return preflight()


if __name__ == "__main__":
    raise SystemExit(main())
