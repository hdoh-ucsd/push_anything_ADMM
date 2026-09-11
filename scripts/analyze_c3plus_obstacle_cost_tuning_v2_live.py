#!/usr/bin/env python3
"""Aggregate admitted C3+ obstacle-cost tuning v2 runs.

This script is deliberately offline.  It reads immutable admitted attempts,
uses the logger's exact live-applied obstacle scalar, and never reconstructs a
different obstacle formula under the same name.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
OUT = ROOT / "results/c3plus_obstacle_cost_tuning_v2"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
MANIFEST = OUT / "selection/stage1_experiment_manifest.csv"
POS_TOL = 0.05
YAW_TOL = 0.10
SUSTAINED_S = 1.0
PROGRESS_WINDOW_S = 30.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


RATIO = load_module("ratio_analysis_for_obsv2", ROOT / "scripts/analyze_c3plus_task_cost_ratio.py")
CHOMP = RATIO.CHOMP


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(values) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def med(values) -> float:
    values = finite(values)
    return float(np.median(values)) if values else math.nan


def quant(values, q: float) -> float:
    values = finite(values)
    return float(np.quantile(values, q)) if values else math.nan


def admitted_attempt(row: dict[str, str]) -> Path | None:
    base = OUT / "runs" / row["task_cost_id"] / row["family"] / row["obstacle_config_id"] / row["scenario"]
    marker = base / "ADMITTED_RUN.json"
    if not marker.exists():
        return None
    return base / json.loads(marker.read_text())["attempt"]


def selected_rows(path: Path, committed: set[int]) -> list[dict[str, str]]:
    selected = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                event = int(row["event_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if event in committed and row.get("selected") == "1":
                selected.append(row)
    return selected


def first_sustained(times: np.ndarray, condition: np.ndarray, duration: float = SUSTAINED_S) -> int | None:
    """First sample beginning a true interval lasting at least duration."""
    start = None
    for index, active in enumerate(condition):
        if active and start is None:
            start = index
        if not active:
            start = None
        if start is not None and times[index] - times[start] >= duration:
            return start
    return None


def nearest_value(times: np.ndarray, values: np.ndarray, at: float | None) -> float:
    if at is None or not len(times):
        return math.nan
    return float(values[int(np.argmin(np.abs(times - at)))])


def retained_and_success(live: Path, goal: tuple[float, float, float, float, int]):
    gx, gy, gz, gyaw, _ = goal
    hits: list[tuple[float, int, float, float]] = []
    all_rows: list[tuple[float, int, float, float]] = []
    for row in read_jsonl(live / "steps_raw.jsonl"):
        objects = row.get("objects", {})
        if not objects:
            continue
        pose = next(iter(objects.values()))
        pos = math.sqrt((pose[4]-gx)**2 + (pose[5]-gy)**2 + (pose[6]-gz)**2)
        yaw = RATIO.quat_error(pose[:4], gyaw)
        item = (float(row["sim_time"]), int(row["control_step"]), pos, yaw)
        all_rows.append(item)
        if pos < POS_TOL and yaw < YAW_TOL:
            hits.append(item)
    first = hits[0] if hits else None
    final = all_rows[-1] if all_rows else None
    retained = bool(final and final[2] < POS_TOL and final[3] < YAW_TOL)
    return first, retained


def progress_loss(cycles: list[dict[str, str]]) -> float | None:
    if not cycles:
        return None
    t = np.asarray([float(r["time"]) for r in cycles])
    x = np.asarray([float(r["obj_x"]) for r in cycles])
    y = np.asarray([float(r["obj_y"]) for r in cycles])
    pe = np.asarray([float(r["position_error"]) for r in cycles])
    ye = np.asarray([float(r["yaw_error"]) for r in cycles])
    stagnant = np.zeros(len(t), dtype=bool)
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] - PROGRESS_WINDOW_S))
        if t[i] - t[j] < 0.9 * PROGRESS_WINDOW_S:
            continue
        displacement = math.hypot(x[i]-x[j], y[i]-y[j])
        error_gain = max(pe[j]-pe[i], 0.5*(ye[j]-ye[i]))
        stagnant[i] = displacement < 0.005 and error_gain < 0.005
    idx = first_sustained(t, stagnant, 5.0)
    return float(t[idx]) if idx is not None else None


def cost_threshold_metrics(selected: list[dict[str, str]], cycles: list[dict[str, str]]) -> dict[str, Any]:
    usable = []
    for row in selected:
        try:
            task = float(row["J_task_code"])
            obs = float(row["J_obstacle_applied_live"])
            time = float(row["time"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(task) and math.isfinite(obs):
            usable.append((time, obs/max(abs(task), 1e-12), task, obs))
    if not usable:
        return {key: math.nan for key in (
            "d_cost_first_10pct", "d_cost_first_50pct", "d_cost_crossover", "d_cost_2x", "d_cost_5x",
            "median_Jobs_Jtask", "q25_Jobs_Jtask", "q75_Jobs_Jtask", "max_Jobs_Jtask")}
    st = np.asarray([r[0] for r in usable])
    ratios = np.asarray([r[1] for r in usable])
    ct = np.asarray([float(r["time"]) for r in cycles])
    clear = np.asarray([float(r["physical_object_obstacle_clearance"]) for r in cycles])
    out = {}
    threshold_names = ((.1,"d_cost_first_10pct"),(.5,"d_cost_first_50pct"),(1,"d_cost_crossover"),(2,"d_cost_2x"),(5,"d_cost_5x"))
    for threshold, name in threshold_names:
        idx = first_sustained(st, ratios >= threshold)
        out[name] = nearest_value(ct, clear, float(st[idx])) if idx is not None else math.nan
    out.update({
        "median_Jobs_Jtask": float(np.median(ratios)),
        "q25_Jobs_Jtask": float(np.quantile(ratios,.25)),
        "q75_Jobs_Jtask": float(np.quantile(ratios,.75)),
        "max_Jobs_Jtask": float(np.max(ratios)),
    })
    return out


def classify(cycles, events, success: bool, retained: bool, planner: str, clearance: np.ndarray, cost_metrics, loss_t):
    if success:
        return ("SUCCESS" if retained else "TRANSIENT_SUCCESS"), "SUCCESS"
    if "terminate called" in planner or "Traceback" in planner or not cycles:
        return "RUNTIME_FAILURE", "WORKSPACE_ABORT"
    t = np.asarray([float(r["time"]) for r in cycles])
    pe = np.asarray([float(r["position_error"]) for r in cycles])
    ye = np.asarray([float(r["yaw_error"]) for r in cycles])
    j = int(np.searchsorted(t, t[-1]-PROGRESS_WINDOW_S))
    progressing = pe[j]-pe[-1] > 0.005 or ye[j]-ye[-1] > 0.05
    outcome = "TIMEOUT_PROGRESSING" if progressing else "TRUE_STALL"
    names = [str(e.get("flag_name", "")) for e in events if e.get("enter_or_exit") != "EXIT"]
    c3 = [r for r in cycles if r.get("is_c3_mode") == "1"]
    c3_contact = np.mean([int(r["physical_contact"]) for r in c3]) if c3 else 0.0
    contact = np.mean([int(r["physical_contact"]) for r in cycles])
    median_tail = med(clearance[t >= t[-1]-10]) if len(clearance) else math.nan
    remote = (loss_t is not None and nearest_value(t, clearance, loss_t) > 0.05 and
              cost_metrics["median_Jobs_Jtask"] >= 1)
    if remote:
        mechanism = "REMOTE_COST_STALL"
    elif len(clearance) and np.nanmin(clearance) <= 0 and contact > .05:
        mechanism = "GEOMETRIC_BLOCK"
    elif c3_contact < .02:
        mechanism = "F1_ACQUISITION"
    elif names.count("FAILED_SECTOR_RESELECT"):
        mechanism = "F3_RESELECTION"
    elif names.count("REPOS_REACHED_NO_CONTACT") or names.count("PLANNER_PHYSICAL_CONTACT_MISMATCH") or names.count("NO_PROGRESS_EXIT"):
        mechanism = "F2_PHANTOM_CHURN"
    elif math.isfinite(median_tail) and median_tail < .05:
        mechanism = "NEAR_OBSTACLE_STALL"
    else:
        mechanism = "TRUE_STALL"
    return outcome, mechanism


def process(row: dict[str, str], goals) -> dict[str, Any] | None:
    run = admitted_attempt(row)
    if run is None:
        return None
    live, forensic = run / "live", run / "forensics"
    cycles = read_csv(forensic / "forensics_cycle.csv")
    events = read_jsonl(forensic / "forensics_events.jsonl")
    committed = {int(r["event_id"]) for r in cycles}
    selected = selected_rows(forensic / "forensics_candidates.csv", committed)
    first, retained = retained_and_success(live, goals[row["scenario"]])
    success = first is not None
    t = np.asarray([float(r["time"]) for r in cycles])
    pos = np.asarray([float(r["position_error"]) for r in cycles])
    yaw = np.asarray([float(r["yaw_error"]) for r in cycles])
    clear = np.asarray([float(r["physical_object_obstacle_clearance"]) for r in cycles])
    pusher_clear = np.asarray([float(r["physical_pusher_obstacle_clearance"]) for r in cycles])
    gap = np.asarray([float(r["physical_pusher_object_gap"]) for r in cycles])
    normalized = np.maximum(pos/POS_TOL, yaw/YAW_TOL)
    cm = cost_threshold_metrics(selected, cycles)
    loss_t = progress_loss(cycles)
    event_names = [(float(e.get("time", math.nan)), str(e.get("flag_name", ""))) for e in events if e.get("enter_or_exit") != "EXIT"]
    f2_times = [et for et,name in event_names if name in {"STALL_PERSISTENT", "NO_PROGRESS_EXIT"} and math.isfinite(et)]
    f2_t = min(f2_times) if f2_times else None
    outcome, failure = classify(cycles, events, success, retained, (live/"planner.log").read_text(errors="replace"), clear, cm, loss_t)
    transaction_path = forensic / "forensics_transactions.csv"
    transactions = read_csv(transaction_path) if transaction_path.exists() else []
    displacements = [r.get("object_displacement") for r in transactions]
    duties = [r.get("physical_contact_fraction") for r in transactions]
    no_progress = sum(r.get("exit_reason") == "TO_REPOS_UNPRODUCTIVE" for r in transactions)
    chomp = CHOMP.eval_run(row["scenario"], str(live))
    completion = run/"RUN_COMPLETE.json"
    if not completion.exists(): completion = run/"RUN_RUNTIME_FAILURE.json"
    status = json.loads(completion.read_text())
    timing = status.get("timing", {})
    first_time = first[0] if first else math.nan
    first_step = first[1] if first else ""
    final_stall = med(clear[t >= t[-1]-10]) if len(t) else math.nan
    d_loss = nearest_value(t, clear, loss_t)
    d_f2 = nearest_value(t, clear, f2_t)
    crossover = cm["d_cost_crossover"]
    crossover_times = []
    if math.isfinite(crossover):
        # Recover the first sustained crossover time for temporal/spatial lag.
        usable = [(float(r["time"]), float(r["J_obstacle_applied_live"])/max(abs(float(r["J_task_code"])),1e-12)) for r in selected if math.isfinite(float(r["J_task_code"]))]
        st = np.asarray([x[0] for x in usable]); sr = np.asarray([x[1] for x in usable])
        ix = first_sustained(st, sr >= 1)
        if ix is not None: crossover_times.append(float(st[ix]))
    cross_t = crossover_times[0] if crossover_times else None
    return {
        "obstacle_config_id": row["obstacle_config_id"], "family": row["family"],
        "range_parameter": float(row["range_parameter_m"]), "weight": float(row["raw_weight"]),
        "beta_target": float(row["beta_calibration_target"]), "task_cost_id": row["task_cost_id"],
        "alpha_p": float(row["alpha_p"]), "alpha_theta": float(row["alpha_theta"]),
        "scenario": row["scenario"], "seed": int(row["seed"]),
        "success": int(success), "retained_success": int(retained), "T_goal": first_time,
        "first_success_step": first_step, "position_success": int(np.any(pos < POS_TOL)),
        "orientation_success": int(np.any(yaw < YAW_TOL)),
        "best_position_error": float(np.min(pos)), "best_yaw_error": float(np.min(yaw)),
        "final_position_error": float(pos[-1]), "final_yaw_error": float(yaw[-1]),
        "E_best": float(np.min(normalized)), "E_final": float(normalized[-1]),
        "CHOMP_raw": chomp["M_CHOMP"], "CHOMP_normalized": chomp["M_CHOMP_norm"],
        "minimum_physical_clearance": float(np.min(clear)),
        "minimum_pusher_clearance": float(np.nanmin(pusher_clear)),
        "minimum_pusher_object_gap": float(np.nanmin(gap)),
        "time_below_10mm": RATIO.duration_below(t, clear, .010),
        "time_below_5mm": RATIO.duration_below(t, clear, .005),
        "maximum_penetration": max(0.0, -float(np.min(clear))),
        "transactions": len(transactions), "median_transaction_displacement": med(displacements),
        "mean_transaction_contact_duty": float(np.mean(finite(duties))) if finite(duties) else math.nan,
        "physical_contact_fraction": float(np.mean([int(r["physical_contact"]) for r in cycles])),
        "contactless_c3_fraction": float(np.mean([not int(r["physical_contact"]) for r in cycles if r["is_c3_mode"] == "1"])),
        "repositions": sum(1 for a,b in zip(cycles, cycles[1:]) if a["mode_name"] != b["mode_name"] and b["mode_name"] == "REPOSITION"),
        "unproductive_exits": no_progress, "runtime_status": outcome, "failure_class": failure,
        "controller_rate": timing.get("achieved_mean_control_frequency", math.nan),
        "timing_valid": int(timing.get("timing_valid", True)),
        **cm, "d_progress_loss": d_loss, "d_f2_onset": d_f2, "d_final_stall": final_stall,
        "distance_traveled_crossover_to_stall": abs(crossover-final_stall) if math.isfinite(crossover) else math.nan,
        "time_crossover_to_stall": (float(t[-1])-cross_t) if cross_t is not None else math.nan,
        "progress_loss_time": loss_t if loss_t is not None else math.nan,
        "f2_onset_time": f2_t if f2_t is not None else math.nan,
        "admitted_attempt": str(run.relative_to(ROOT)),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["obstacle_config_id"], row["task_cost_id"])].append(row)
    result = []
    for (config, task), group in sorted(groups.items()):
        valid = [r for r in group if r["timing_valid"]]
        failed = [r for r in valid if not r["success"]]
        result.append({
            "obstacle_config_id": config, "family": group[0]["family"],
            "range_parameter": group[0]["range_parameter"], "weight": group[0]["weight"],
            "beta_target": group[0]["beta_target"], "task_cost_id": task,
            "pilot_valid_count": len(valid), "pilot_success_count": sum(r["success"] for r in valid),
            "retained_success_count": sum(r["retained_success"] for r in valid),
            "median_T_goal": med(r["T_goal"] for r in valid if r["success"]),
            "median_E_best": med(r["E_best"] for r in failed),
            "median_CHOMP_norm": med(r["CHOMP_normalized"] for r in valid),
            "worst_clearance": min(finite(r["minimum_physical_clearance"] for r in valid), default=math.nan),
            "penetration_count": sum(r["maximum_penetration"] > 0 for r in valid),
            "median_Jobs_Jtask": med(r["median_Jobs_Jtask"] for r in valid),
            "median_progress_loss_radius": med(r["d_progress_loss"] for r in failed),
            "median_final_stall_radius": med(r["d_final_stall"] for r in failed),
            "remote_stall_count": sum(r["failure_class"] == "REMOTE_COST_STALL" for r in valid),
            "F2_count": sum(r["failure_class"] == "F2_PHANTOM_CHURN" for r in valid),
            "other_failure_count": sum(not r["success"] and r["failure_class"] not in {"REMOTE_COST_STALL","F2_PHANTOM_CHURN"} for r in valid),
            "status": "CORE_COMPLETE" if {r["scenario"] for r in valid} >= {"single_obstacle","shelf_gap"} else "IN_PROGRESS",
        })
    return result


def core_pruning(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predeclared conservative rule: reject only pathology on both core scenes."""
    rows = []
    for item in summary:
        complete = item["status"] == "CORE_COMPLETE"
        both_failed = complete and item["pilot_success_count"] == 0
        both_pathological = both_failed and item["pilot_valid_count"] == 2 and item["remote_stall_count"] >= 2
        status = "PILOT_REJECTED" if both_pathological else ("ADVANCE" if complete else "INCOMPLETE")
        rows.append({
            "obstacle_config_id": item["obstacle_config_id"], "task_cost_id": item["task_cost_id"],
            "core_status": status,
            "criterion": "reject only if both single_obstacle and shelf_gap are valid, unsuccessful, and REMOTE_COST_STALL",
            "evidence": f"success={item['pilot_success_count']}/2; remote_stall={item['remote_stall_count']}/2",
        })
    return rows


def figures(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r["timing_valid"]]
    if not valid:
        return
    colors = {"exponential":"tab:blue", "relu":"tab:orange"}
    plt.figure(figsize=(8,5))
    for r in valid:
        plt.scatter(r["CHOMP_normalized"], r["success"]+.02*(1 if r["task_cost_id"]=="TR08" else -1), c=colors[r["family"]], alpha=.7)
    plt.xlabel("CHOMP normalized"); plt.ylabel("First-hit success"); plt.yticks([0,1]); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIGURES/"success_vs_CHOMP.png",dpi=180); plt.close()
    failed = [r for r in valid if not r["success"]]
    plt.figure(figsize=(8,5))
    for r in failed:
        plt.scatter(r["d_final_stall"], r["E_best"], c=colors[r["family"]], alpha=.7)
    plt.xlabel("Final stall footprint clearance [m]"); plt.ylabel("Best normalized task error"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIGURES/"success_vs_stall_radius.png",dpi=180); plt.close()
    plt.figure(figsize=(8,5))
    for family in ("exponential","relu"):
        subset=[r for r in valid if r["family"]==family]
        plt.scatter([r["minimum_physical_clearance"] for r in subset],[r["median_Jobs_Jtask"] for r in subset],label=family,alpha=.75)
    plt.yscale("symlog",linthresh=.01); plt.xlabel("Minimum footprint clearance [m]"); plt.ylabel("Median exact live Jobs/Jtask"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIGURES/"Jobs_over_Jtask_vs_clearance.png",dpi=180); plt.close()
    by=defaultdict(list)
    for r in valid: by[(r["obstacle_config_id"],r["task_cost_id"])].append(r["success"])
    configs=sorted({r["obstacle_config_id"] for r in valid})
    x=np.arange(len(configs)); width=.36
    plt.figure(figsize=(10,5))
    for offset,task in ((-.18,"TR07"),(.18,"TR08")):
        plt.bar(x+offset,[sum(by[(c,task)])/len(by[(c,task)]) if by[(c,task)] else 0 for c in configs],width,label=task)
    plt.xticks(x,configs); plt.ylabel("Pilot success fraction"); plt.ylim(0,1.05); plt.legend(); plt.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(FIGURES/"TR07_TR08_obstacle_interaction.png",dpi=180); plt.close()


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("core","all"),default="all")
    args=parser.parse_args()
    manifest=read_csv(MANIFEST)
    if args.phase=="core": manifest=[r for r in manifest if r["scenario"] in {"single_obstacle","shelf_gap"}]
    goals=RATIO.load_goals()
    rows=[]
    for index,row in enumerate(manifest,1):
        result=process(row,goals)
        if result: rows.append(result)
        print(f"[{index}/{len(manifest)}] {row['task_cost_id']} {row['obstacle_config_id']} {row['scenario']} {'OK' if result else 'missing'}",flush=True)
    METRICS.mkdir(parents=True,exist_ok=True)
    write_csv(METRICS/"per_run_results.csv",rows)
    summary=summarize(rows)
    write_csv(METRICS/"obstacle_pair_summary.csv",summary)
    pruning=core_pruning(summary)
    write_csv(METRICS/"core_pruning_decisions.csv",pruning)
    stall_fields=["obstacle_config_id","task_cost_id","family","scenario","failure_class","d_cost_first_10pct","d_cost_first_50pct","d_cost_crossover","d_cost_2x","d_cost_5x","d_progress_loss","d_f2_onset","d_final_stall","distance_traveled_crossover_to_stall","time_crossover_to_stall"]
    write_csv(METRICS/"stall_radius_summary.csv",[r for r in rows if not r["success"]],stall_fields)
    figures(rows)
    print(json.dumps({"admitted_rows":len(rows),"expected":len(manifest),"timing_invalid":sum(not r["timing_valid"] for r in rows),"successes":sum(r["success"] for r in rows)},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
