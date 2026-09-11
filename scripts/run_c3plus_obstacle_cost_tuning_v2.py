#!/usr/bin/env python3
"""Serial, resumable live runner for C3+ obstacle-cost tuning v2.

The runner changes only the environment-gated candidate-ranking obstacle
potential.  It refuses to launch beside any existing franka_sim and never
overwrites an attempt or admitted result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
OUT = ROOT / "results/c3plus_obstacle_cost_tuning_v2"
RATIO = ROOT / "results/c3plus_task_cost_ratio"
RUNNER = WT / "tools/scene_smoke/run_scene_smoke.sh"
VALIDATOR = WT / "scripts/validate_forensics_logs.py"
SELECTION = OUT / "selection/live_parameter_manifest.csv"
MANIFEST = OUT / "selection/stage1_experiment_manifest.csv"
CONTROL_RATE = RATIO / "reproducibility/control_rate_validation.csv"
MAX_CONCURRENT_LIVE_SIMULATIONS = 1

TASK_COSTS = {"TR07": (2.0, 0.5), "TR08": (2.0, 1.0)}
# User-approved extension pruning: skip only task-cost/config pairs whose
# completed core (single_obstacle+shelf_gap) success rate was exactly 0/2.
# Completed attempts remain immutable; this is consulted only before launch.
SKIP_EXTENSION_PAIRS = {
    ("TR07", "OC02"), ("TR07", "OC03"), ("TR07", "OC06"), ("TR07", "OC07"),
    ("TR08", "OC05"), ("TR08", "OC06"),
}
PILOT_SCENES = ("single_obstacle", "shelf_gap", "box_clutter", "ycb_clutter")
OBJECT_NAMES = {scene: "G_shape_video" for scene in (*PILOT_SCENES, "open_task", "icra_sign", "slalom")}
SCENE_SEEDS = {
    "open_task": 17001, "single_obstacle": 17101, "shelf_gap": 17201,
    "ycb_clutter": 17301, "icra_sign": 17401, "slalom": 17501,
    "box_clutter": 17601,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def ratio_manifest_rows() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(RATIO / "experiment_manifest.csv")
    return {(r["condition_id"], r["scenario"]): r for r in rows}


def prepare() -> None:
    if MANIFEST.exists():
        print(f"manifest already exists: {MANIFEST}")
        return
    configs = [r for r in read_csv(SELECTION) if r["stage1_status"] == "PLANNED"]
    if len(configs) != 8:
        raise RuntimeError(f"expected 8 screened obstacle configs, found {len(configs)}")
    base = ratio_manifest_rows()
    rows: list[dict[str, Any]] = []
    index = 0
    for config in configs:
        for task_id in TASK_COSTS:
            for scene in PILOT_SCENES:
                index += 1
                source = base[(task_id, scene)]
                rows.append({
                    "run_index": index, "obstacle_config_id": config["obstacle_config_id"],
                    "family": config["family"], "range_parameter_m": config["range_parameter_m"],
                    "raw_weight": config["raw_weight"],
                    "beta_calibration_target": config["beta_calibration_target"],
                    "task_cost_id": task_id, "alpha_p": TASK_COSTS[task_id][0],
                    "alpha_theta": TASK_COSTS[task_id][1], "scenario": scene,
                    "seed": SCENE_SEEDS[scene], "demo_name": source["demo_name"],
                    "case_id": source["case_id"], "env_file": source["env_file"],
                    "port": 24100 + index, "cap_wall_s": 600, "status": "PLANNED",
                })
    write_csv_new(MANIFEST, rows)
    provenance = {
        "created_at_unix": time.time(), "live_concurrency": 1,
        "controller_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WT, text=True).strip(),
        "controller_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=WT, text=True).strip()),
        "controller_source_sha256": sha256(WT / "systems/controllers/sampling_based_c3_controller.cc"),
        "controller_binary_sha256": sha256(WT / "bazel-bin/examples/sampling_c3/franka_sampling_c3_controller"),
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "only_varied_controller_fields": ["candidate-ranking obstacle family", "range_parameter_m", "raw_weight"],
        "frozen": {"obstacle_mode": "lcs_contact", "seed_by_scene": SCENE_SEEDS, "max_concurrency": 1},
    }
    (OUT / "selection/stage1_run_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"prepared {len(rows)} serial jobs")


def active_franka_sim() -> list[str]:
    result = subprocess.run(["pgrep", "-af", "franka_sim"], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("process audit failed")
    return [line for line in result.stdout.splitlines() if "pgrep -af" not in line]


def process_snapshot() -> list[str]:
    return subprocess.check_output(["ps", "-eo", "pid,pcpu,pmem,etime,comm,args", "--sort=-pcpu"], text=True).splitlines()[:21]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values: return math.nan
    x = (len(values) - 1) * q; lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] * (hi-x) + values[hi] * (x-lo)


def timing(forensics: Path, wall: float) -> dict[str, Any]:
    cycles = read_csv(forensics / "forensics_cycle.csv")
    times = [float(r["time"]) for r in cycles]
    periods = [b-a for a,b in zip(times,times[1:]) if b>a]
    reference = next(r for r in read_csv(CONTROL_RATE) if r["concurrency_level"] == "1")
    duration = times[-1] - times[0]
    recorder_wall = max(1e-9, wall - 4.0)
    sim_wall = duration / recorder_wall
    relative = sim_wall / float(reference["simulation_wall_ratio"])
    return {
        "timing_valid": relative >= .95,
        "criterion": "serial; no competing franka_sim; sim/wall>=0.95x fixed clean serial gate",
        "concurrency_level": 1, "requested_control_frequency": "event_driven (publish_frequency=0)",
        "achieved_mean_control_frequency": len(periods)/duration,
        "achieved_median_control_frequency": 1/statistics.median(periods),
        "control_period_median": statistics.median(periods),
        "control_period_p95": percentile(periods,.95), "control_period_p99": percentile(periods,.99),
        "late_cycle_count": sum(p > 2*statistics.median(periods) for p in periods),
        "simulation_time": duration, "wall_time": wall, "simulation_wall_ratio": sim_wall,
        "sim_wall_vs_single": relative,
    }


def bounded_tail(stdout: str) -> bool:
    errors = [line for line in stdout.splitlines() if line.startswith("- ")]
    if not errors: return False
    allowed = ("truncated CSV row", "candidate rows, expected", "selected candidate count is 0, expected 1")
    if not all(any(fragment in line for fragment in allowed) for line in errors): return False
    cycles = re.search(r"\bcycles=(\d+)", stdout)
    event_ids = [int(x) for x in re.findall(r"\bevent (\d+)", "\n".join(errors))]
    return bool(cycles and event_ids and min(event_ids) >= int(cycles.group(1))-4)


def launch(row: dict[str, str], run_root: Path | None = None, cap_override: int | None = None) -> bool:
    if active_franka_sim():
        raise RuntimeError("WAITING_FOR_EXISTING_SIMULATION_CAMPAIGN: franka_sim active")
    family, config = row["family"], row["obstacle_config_id"]
    root = run_root or (OUT / "runs" / row["task_cost_id"] / family / config / row["scenario"])
    admitted = root / "ADMITTED_RUN.json"
    if admitted.exists():
        print(f"SKIP {row['task_cost_id']}/{config}/{row['scenario']}", flush=True)
        return True
    attempts = root / "attempts"; attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts / f"attempt_{number:02d}").exists(): number += 1
    run = attempts / f"attempt_{number:02d}"; run.mkdir()
    forensics, live = run / "forensics", run / "live"
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SAMPLING_C3_OBS_RELU") or key.startswith("SAMPLING_C3_OBS_EXP") or key in {
            "SAMPLING_C3_INNER_OBS_MODE", "SAMPLING_C3_OBJ_NONPEN", "SAMPLING_C3_RANK_OBS_MODE"
        }: env.pop(key)
    env.update({
        "SAMPLING_C3_EXPERIMENT_SEED": str(row["seed"]),
        "SAMPLING_C3_TRANSLATION_COST_SCALE": str(row["alpha_p"]),
        "SAMPLING_C3_ORIENTATION_COST_SCALE": str(row["alpha_theta"]),
        "SAMPLING_C3_FORENSICS_LOG_DIR": str(forensics),
        "SAMPLING_C3_FORENSICS_TEXT_EVENTS": "0",
        "SAMPLING_C3_FORENSICS_GIT_COMMIT": subprocess.check_output(["git","rev-parse","HEAD"],cwd=WT,text=True).strip(),
        "SAMPLING_C3_FORENSICS_TASK_ID": f"{row['scenario']}/{row['case_id']}/{config}",
        "SAMPLING_C3_FORENSICS_ROBOT_MODEL": "xarm6",
        "SAMPLING_C3_OBSTACLE_MODE": "lcs_contact",
        "SAMPLING_C3_STOP_RECORDER_ON_PLANNER_EXIT": "1",
    })
    if family == "relu":
        env.update({"SAMPLING_C3_RANK_OBS_MODE":"relu_footprint", "SAMPLING_C3_OBS_RELU_EPS":row["range_parameter_m"], "SAMPLING_C3_OBS_RELU_W":row["raw_weight"]})
    elif family == "exponential":
        env.update({"SAMPLING_C3_RANK_OBS_MODE":"exponential_footprint", "SAMPLING_C3_OBS_EXP_SIGMA":row["range_parameter_m"], "SAMPLING_C3_OBS_EXP_W":row["raw_weight"]})
    else: raise ValueError(family)
    ratio_row = ratio_manifest_rows()[(row["task_cost_id"], row["scenario"])]
    # Goals are copied from the already-audited ratio manifest/demos.
    if row["scenario"] == "box_clutter": gx,gy,gyaw = .381,-.305,math.pi/2
    else:
        import yaml
        cases = yaml.safe_load((ROOT/"results/benchmark_sync_rot15/c3plus_cases.yaml").read_text())["cases"]
        case = next(c for c in cases if c["scene"] == row["scenario"] and c["case_id"] == "s01_g02_rot000")
        gx,gy = float(case["goal_x"]),float(case["goal_y"])
        gyaw = 2*math.atan2(float(case["goal_qz"]),float(case["goal_qw"]))
    env_file = str(WT / row["env_file"]) if row["env_file"] else ""
    cap = str(cap_override or int(row["cap_wall_s"]))
    command = ["bash",str(RUNNER),ratio_row["demo_name"],OBJECT_NAMES[row["scenario"]],str(gx),str(gy),str(gyaw),cap,row["port"],str(live),env_file]
    receipt = {
        "command": command, "manifest_row": row, "environment": {k:env[k] for k in sorted(env) if k.startswith("SAMPLING_C3_")},
        "controller_source_sha256": sha256(WT/"systems/controllers/sampling_based_c3_controller.cc"),
        "controller_binary_sha256": sha256(WT/"bazel-bin/examples/sampling_c3/franka_sampling_c3_controller"),
        "prelaunch_franka_sim": [], "prelaunch_top_processes": process_snapshot(), "started_at_unix":time.time(),
    }
    (run/"run_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(f"START {row['run_index']} {row['task_cost_id']}/{config}/{row['scenario']}",flush=True)
    start=time.monotonic()
    with (run/"launcher_stdout.log").open("w") as stream:
        result=subprocess.run(command,cwd=WT,env=env,text=True,stdout=stream,stderr=subprocess.STDOUT)
    wall=time.monotonic()-start
    val=subprocess.run([sys.executable,str(VALIDATOR),str(forensics)],cwd=WT,text=True,capture_output=True)
    planner=(live/"planner.log").read_text(errors="replace") if (live/"planner.log").exists() else ""
    status={"returncode":result.returncode,"wall_time_s":wall,"validator_returncode":val.returncode,"validator_stdout":val.stdout,"validator_stderr":val.stderr,"planner_aborted":"terminate called" in planner or "Aborted" in planner,"completed_at_unix":time.time()}
    tail=(val.returncode!=0 and result.returncode==0 and bounded_tail(val.stdout))
    status["validator_bounded_final_tail_exception"]=tail
    if result.returncode!=0 or (val.returncode!=0 and not tail):
        (run/"RUN_FAILED.json").write_text(json.dumps(status,indent=2)+"\n")
        raise RuntimeError(f"validation failed: {row['task_cost_id']}/{config}/{row['scenario']}")
    if status["planner_aborted"]:
        status["timing"]={"timing_valid":True,"reason":"clean serial runtime termination with valid forensic logs","concurrency_level":1}
        complete="RUN_RUNTIME_FAILURE.json"
    else:
        status["timing"]=timing(forensics,wall)
        if not status["timing"]["timing_valid"]:
            (run/"RUN_TIMING_INVALID.json").write_text(json.dumps(status,indent=2)+"\n")
            print(f"TIMING_INVALID {row['task_cost_id']}/{config}/{row['scenario']}",flush=True)
            return False
        complete="RUN_COMPLETE.json"
    (run/complete).write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    admitted.write_text(json.dumps({"attempt":str(run.relative_to(root)),"completion_file":complete,"selected_at_unix":time.time(),"reason":"clean serial; timing and logger gates passed"},indent=2,sort_keys=True)+"\n")
    print(f"DONE {row['task_cost_id']}/{config}/{row['scenario']} wall={wall:.1f}s",flush=True)
    return True


def run_rows(rows: list[dict[str,str]]) -> None:
    for row in rows:
        for attempt in range(3):
            try:
                if launch(row): break
            except RuntimeError as error:
                # A simulator can remain visible to pgrep for a short,
                # non-scientific shutdown interval.  Preserve the safety gate
                # and retry; never bypass it or treat this as a run failure.
                if "WAITING_FOR_EXISTING_SIMULATION_CAMPAIGN" not in str(error):
                    raise
                print("WAITING_FOR_EXISTING_SIMULATION_CAMPAIGN: retrying after shutdown", flush=True)
                time.sleep(5.0)
        else: raise RuntimeError("three timing-invalid attempts")


def negative_controls() -> None:
    base = ratio_manifest_rows()
    configs = {r["obstacle_config_id"]:r for r in read_csv(SELECTION)}
    choices = (configs["OC04"], configs["OC08"])
    rows=[]
    for ti,task in enumerate(TASK_COSTS):
        for ci,cfg in enumerate(choices):
            source=base[(task,"open_task")]
            rows.append({"run_index":f"NC{ti*2+ci+1}","obstacle_config_id":cfg["obstacle_config_id"],"family":cfg["family"],"range_parameter_m":cfg["range_parameter_m"],"raw_weight":cfg["raw_weight"],"beta_calibration_target":cfg["beta_calibration_target"],"task_cost_id":task,"alpha_p":TASK_COSTS[task][0],"alpha_theta":TASK_COSTS[task][1],"scenario":"open_task","seed":SCENE_SEEDS["open_task"],"demo_name":source["demo_name"],"case_id":source["case_id"],"env_file":source["env_file"],"port":str(24080+ti*2+ci),"cap_wall_s":"60","status":"NEGATIVE_CONTROL"})
    for row in rows:
        root=OUT/"runs"/row["task_cost_id"]/row["family"]/row["obstacle_config_id"]/"open_task_negative_control"
        for _ in range(3):
            if launch(row,root,60): break


def main() -> int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare"); sub.add_parser("negative-controls")
    run=sub.add_parser("run"); run.add_argument("--phase",choices=("core","extension","all"),default="all"); run.add_argument("--start-at",type=int,default=1); run.add_argument("--stop-after",type=int)
    args=parser.parse_args()
    if args.command=="prepare": prepare(); return 0
    if args.command=="negative-controls": negative_controls(); return 0
    rows=read_csv(MANIFEST)
    scenes=("single_obstacle","shelf_gap") if args.phase=="core" else (("box_clutter","ycb_clutter") if args.phase=="extension" else PILOT_SCENES)
    selected=[r for r in rows if r["scenario"] in scenes and int(r["run_index"])>=args.start_at]
    if args.phase in ("extension", "all"):
        selected=[r for r in selected if (r["task_cost_id"], r["obstacle_config_id"]) not in SKIP_EXTENSION_PAIRS]
    if args.stop_after is not None: selected=selected[:args.stop_after]
    run_rows(selected); return 0


if __name__=="__main__": raise SystemExit(main())
