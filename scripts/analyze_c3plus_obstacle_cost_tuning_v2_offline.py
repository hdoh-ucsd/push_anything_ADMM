#!/usr/bin/env python3
"""Offline range and candidate-ranking analysis for obstacle-cost tuning v2.

This program never launches or modifies a controller.  It combines the immutable
historical 600 s campaign with the admitted TR07/TR08 logger-v2 candidate pools.
Historical costs are explicitly labelled measured-state/center-distance proxies;
all new family comparisons use the logged orientation-aware footprint clearance
at every N+1 ranked rollout knot.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_obs_tuning_v2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
RATIO = ROOT / "results/c3plus_task_cost_ratio"
HIST_RUNS = WT / "results/xarm6_c3plus_scene_smoke/runs"
FAILURE = WT / "results/failure_classification"
HIST_COST = WT / "results/cost_analysis"
DEFAULT_OUTPUT = ROOT / "results/c3plus_obstacle_cost_tuning_v2"

SIGMAS = (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050)
EPSILONS = (0.010, 0.020, 0.030, 0.040, 0.050)
TASK_IDS = ("TR07", "TR08")
SCENES = ("single_obstacle", "shelf_gap", "box_clutter", "ycb_clutter", "icra_sign", "slalom")
PILOT_SCENES = ("single_obstacle", "shelf_gap", "box_clutter", "ycb_clutter")
W_DIAGNOSTIC = 5000.0
SUSTAINED_SECONDS = 1.0
REFERENCE_CLEARANCE = (0.010, 0.050)
NA = "NA"


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def percentile(values: Sequence[float], q: float) -> float:
    a = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    return float(np.percentile(a, q)) if a.size else math.nan


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration:
        if fields is None:
            raise ValueError(f"empty CSV without schema: {path}")
        with path.open("x", newline="") as stream:
            csv.DictWriter(stream, fieldnames=list(fields)).writeheader()
        return
    names = list(fields or first.keys())
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def admitted_attempt(task_id: str, scene: str) -> Path:
    marker = RATIO / "runs" / task_id / scene / "ADMITTED_RUN.json"
    data = json.loads(marker.read_text())
    attempt = marker.parent / data["attempt"]
    if not (attempt / "RUN_COMPLETE.json").exists():
        raise FileNotFoundError(f"admitted run incomplete: {attempt}")
    return attempt


def knot_groups(path: Path) -> Iterator[tuple[tuple[int, int], list[float]]]:
    """Yield ((event,candidate), clearance trajectory) from ordered knot CSV."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        key: tuple[int, int] | None = None
        values: list[float] = []
        for row in reader:
            new = (int(row["event_id"]), int(row["candidate_id"]))
            if key is not None and new != key:
                yield key, values
                values = []
            key = new
            values.append(fnum(row["object_obstacle_signed_clearance"]))
        if key is not None:
            yield key, values


@dataclass
class Candidate:
    event: int
    cid: int
    time: float
    jtask: float
    jrank: float
    live_obs: float
    progress: float
    selected: bool
    hard: bool
    rejection: str
    clearances: np.ndarray

    @property
    def base_rank(self) -> float:
        return self.jrank - self.live_obs

    @property
    def min_clearance(self) -> float:
        finite = self.clearances[np.isfinite(self.clearances)]
        return float(np.min(finite)) if finite.size else math.inf


def candidate_stream(attempt: Path) -> Iterator[Candidate]:
    fdir = attempt / "forensics"
    kg = iter(knot_groups(fdir / "forensics_candidate_knots.csv"))
    try:
        kkey, kvals = next(kg)
    except StopIteration:
        return
    with (fdir / "forensics_candidates.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["event_id"]), int(row["candidate_id"]))
            while kkey < key:
                try:
                    kkey, kvals = next(kg)
                except StopIteration:
                    return
            if kkey != key:
                # A bounded logger shutdown tail may have candidate rows without knots.
                continue
            yield Candidate(
                event=key[0], cid=key[1], time=fnum(row["time"]),
                jtask=fnum(row["J_task_code"]), jrank=fnum(row["J_rank_code_total"]),
                live_obs=fnum(row["J_obstacle_applied_live"], 0.0),
                progress=fnum(row["terminal_xy_progress"]),
                selected=row["selected"] == "1", hard=row["hard_filtered"] == "1",
                rejection=row.get("rejection_reason", "") or "",
                clearances=np.asarray(kvals, dtype=float),
            )
            try:
                kkey, kvals = next(kg)
            except StopIteration:
                return


def groups_by_event(attempt: Path) -> Iterator[list[Candidate]]:
    current: list[Candidate] = []
    event: int | None = None
    for candidate in candidate_stream(attempt):
        if event is not None and candidate.event != event:
            yield current
            current = []
        event = candidate.event
        current.append(candidate)
    if current:
        yield current


def eligible(c: Candidate) -> bool:
    return (not c.hard and math.isfinite(c.base_rank) and c.base_rank < 1e11)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 1.0
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def process_candidate_run(args: tuple[str, str, str]) -> dict[str, Any]:
    task_id, scene, part_dir_string = args
    part_dir = Path(part_dir_string)
    attempt = admitted_attempt(task_id, scene)
    part = part_dir / f"{task_id}_{scene}_sigma.csv"
    fields = [
        "task_cost_id", "scene", "time", "event_id", "sigma", "candidate_count",
        "baseline_selected", "baseline_global_argmin", "winner_new", "winner_sigma040",
        "top1_changed_vs_no_obs", "top1_changed_vs_sigma040", "rank_correlation",
        "winner_clearance", "winner_predicted_task_progress", "winner_Jobs_Jtask",
        "task_best_candidate", "task_best_rejected", "task_best_Jobs_Jtask",
    ]
    sigma_acc: dict[float, dict[str, list[float]]] = {
        s: defaultdict(list) for s in SIGMAS
    }
    eps_acc: dict[float, dict[str, list[float]]] = {
        e: defaultdict(list) for e in EPSILONS
    }
    reference: dict[tuple[str, float], list[float]] = defaultdict(list)
    earliest: dict[str, Any] | None = None
    events = 0
    baseline_selection_mismatch = 0
    # Passive detector anchor only; this state never affected the controller.
    persistent_anchor = math.nan
    events_path = attempt / "forensics/forensics_events.jsonl"
    if events_path.exists():
        with events_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if (event.get("flag_name") == "STALL_PERSISTENT" and
                        event.get("enter_or_exit") in ("ENTER", "EVENT")):
                    persistent_anchor = fnum(event.get("time"))
                    break
    f2_anchor_recorded = {epsilon: False for epsilon in EPSILONS}
    with part.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for pool in groups_by_event(attempt):
            valid = [c for c in pool if eligible(c)]
            if not valid:
                continue
            events += 1
            selected = next((c for c in pool if c.selected), None)
            base = min(valid, key=lambda c: (c.base_rank, c.cid))
            finite_task = [c for c in valid if math.isfinite(c.jtask)]
            task_best = min(finite_task, key=lambda c: (c.jtask, c.cid)) if finite_task else base
            if selected is not None and selected.cid != base.cid:
                baseline_selection_mismatch += 1
            exp_phi = {
                s: np.asarray([float(np.exp(-c.clearances / s).sum()) for c in valid])
                for s in SIGMAS
            }
            relu_phi = {
                e: np.asarray([float(np.square(np.maximum(0.0, (e - c.clearances) / e)).sum()) for c in valid])
                for e in EPSILONS
            }
            base_scores = np.asarray([c.base_rank for c in valid])
            jtasks = np.asarray([c.jtask for c in valid])
            ids = np.asarray([c.cid for c in valid], dtype=int)
            historical_scores = base_scores + W_DIAGNOSTIC * exp_phi[0.040]
            hist_idx = int(np.argmin(historical_scores))
            for sigma in SIGMAS:
                scores = base_scores + W_DIAGNOSTIC * exp_phi[sigma]
                wi = int(np.argmin(scores))
                winner = valid[wi]
                winner_task = abs(jtasks[wi]) if math.isfinite(jtasks[wi]) else abs(winner.base_rank)
                ratio = W_DIAGNOSTIC * exp_phi[sigma][wi] / max(winner_task, 1e-12)
                task_idx = int(np.where(ids == task_best.cid)[0][0])
                task_denom = abs(jtasks[task_idx]) if math.isfinite(jtasks[task_idx]) else abs(task_best.base_rank)
                task_ratio = W_DIAGNOSTIC * exp_phi[sigma][task_idx] / max(task_denom, 1e-12)
                row = {
                    "task_cost_id": task_id, "scene": scene, "time": pool[0].time,
                    "event_id": pool[0].event, "sigma": sigma,
                    "candidate_count": len(valid),
                    "baseline_selected": selected.cid if selected else NA,
                    "baseline_global_argmin": base.cid, "winner_new": winner.cid,
                    "winner_sigma040": valid[hist_idx].cid,
                    "top1_changed_vs_no_obs": int(winner.cid != base.cid),
                    "top1_changed_vs_sigma040": int(winner.cid != valid[hist_idx].cid),
                    "rank_correlation": spearman(base_scores, scores),
                    "winner_clearance": winner.min_clearance,
                    "winner_predicted_task_progress": winner.progress,
                    "winner_Jobs_Jtask": ratio,
                    "task_best_candidate": task_best.cid,
                    "task_best_rejected": int(winner.cid != task_best.cid),
                    "task_best_Jobs_Jtask": task_ratio,
                }
                writer.writerow(row)
                acc = sigma_acc[sigma]
                acc["changed"].append(float(winner.cid != base.cid))
                acc["changed_vs_040"].append(float(winner.cid != valid[hist_idx].cid))
                acc["corr"].append(row["rank_correlation"])
                acc["task_best_rejected"].append(float(winner.cid != task_best.cid))
                acc["clearance_improved"].append(float(winner.min_clearance > base.min_clearance + 1e-6))
                acc["task_progress_worsened"].append(float(winner.progress < base.progress - 1e-6))
                acc["winner_ratio"].append(ratio)
                acc["winner_gt_1"].append(float(ratio > 1.0))
                acc["winner_gt_5"].append(float(ratio > 5.0))
            if earliest is None and valid[hist_idx].cid != base.cid:
                earliest = {
                    "task_cost_id": task_id, "scene": scene, "time": pool[0].time,
                    "event_id": pool[0].event, "baseline_candidate": base.cid,
                    "historical_footprint_exp_winner": valid[hist_idx].cid,
                    "candidates": [
                        {
                            "candidate_id": c.cid, "selected": int(c.selected),
                            "J_translation_plus_orientation_task": c.jtask,
                            "J_rank_without_obstacle": c.base_rank,
                            "min_footprint_clearance": c.min_clearance,
                            "J_obstacle_sigma040_w5000": W_DIAGNOSTIC * exp_phi[0.040][j],
                            "J_total_sigma040_w5000": historical_scores[j],
                            "terminal_xy_progress": c.progress,
                        }
                        for j, c in enumerate(valid)
                    ],
                }
            chosen = selected if selected is not None and eligible(selected) else base
            ci = valid.index(chosen)
            for epsilon in EPSILONS:
                phi = relu_phi[epsilon]
                scores = base_scores + W_DIAGNOSTIC * phi
                wi = int(np.argmin(scores))
                acc = eps_acc[epsilon]
                acc["decision_states"].append(1.0)
                acc["selected_inside"].append(float(chosen.min_clearance < epsilon))
                acc["any_inside"].append(float(any(c.min_clearance < epsilon for c in valid)))
                acc["near_collision"].append(float(chosen.min_clearance < 0.010))
                acc["near_collision_inside"].append(float(
                    chosen.min_clearance < 0.010 and chosen.min_clearance < epsilon))
                acc["rank_changed"].append(float(valid[wi].cid != base.cid))
                if (math.isfinite(persistent_anchor) and not f2_anchor_recorded[epsilon]
                        and pool[0].time >= persistent_anchor):
                    acc["f2_onset_inside"].append(float(chosen.min_clearance < epsilon))
                    f2_anchor_recorded[epsilon] = True
            if REFERENCE_CLEARANCE[0] <= chosen.min_clearance <= REFERENCE_CLEARANCE[1] and chosen.jtask > 0:
                for sigma in SIGMAS:
                    phi = float(np.exp(-chosen.clearances / sigma).sum())
                    if phi > 0:
                        reference[("exponential", sigma)].append(chosen.jtask / phi)
                for epsilon in EPSILONS:
                    phi = float(np.square(np.maximum(0.0, (epsilon - chosen.clearances) / epsilon)).sum())
                    if phi > 0:
                        reference[("relu", epsilon)].append(chosen.jtask / phi)
    return {
        "task_id": task_id, "scene": scene, "part": str(part), "events": events,
        "baseline_selection_mismatch": baseline_selection_mismatch,
        "sigma_acc": sigma_acc, "eps_acc": eps_acc, "reference": reference,
        "earliest": earliest, "attempt": str(attempt),
    }


def load_cost_module():
    path = WT / "tools/scene_smoke/cost_diagnostics_v2.py"
    spec = importlib.util.spec_from_file_location("cost_diag_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_sustained(times: np.ndarray, mask: np.ndarray, seconds: float) -> int | None:
    start: int | None = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active:
            start = None
        if start is not None and times[i] - times[start] >= seconds:
            return start
    return None


def historical_forensics() -> list[dict[str, Any]]:
    mod = load_cost_module()
    audit = read_csv(FAILURE / "failure_audit.csv")
    rows: list[dict[str, Any]] = []
    for a in audit:
        scene = a["scene"]
        if scene not in {"single_obstacle", "shelf_gap", "icra_sign", "slalom", "ycb_clutter"}:
            continue
        pair = f"pair{int(a['start_id'][1:]):02d}"
        _, _, m = mod.read_metrics(scene, pair)
        t = m["sim_time"]
        dx, dy = m["object_x"] - m["goal_x"], m["object_y"] - m["goal_y"]
        eyaw = mod.wrap(m["object_yaw"] - m["goal_yaw"])
        epos = m["position_error_m"]
        latched = np.zeros(len(t), bool)
        if np.any(epos < mod.LATCH):
            latched[np.where(epos < mod.LATCH)[0][0]:] = True
        jt = np.where(latched, mod.W_XY * (dx * dx + dy * dy), mod.W_PRE * (dx * dx + dy * dy))
        jr = np.where(latched, mod.W_ROT_POST * eyaw * eyaw, mod.W_ROT_PRE * eyaw * eyaw)
        jtask = jt + jr
        discs, polys = mod.load_scene_obstacles(scene)
        jobs = mod.j_obs_series(m["object_x"], m["object_y"], discs, polys)
        center = np.full(len(t), math.inf)
        for ox, oy, radius in discs:
            center = np.minimum(center, np.hypot(m["object_x"] - ox, m["object_y"] - oy) - radius)
        for i in range(len(t)):
            for poly in polys:
                center[i] = min(center[i], mod.poly_signed_dist(m["object_x"][i], m["object_y"][i], poly))
        metric_files = list((HIST_RUNS / scene / pair).glob("*_metrics.csv"))
        with metric_files[0].open(newline="") as stream:
            metric_rows = list(csv.DictReader(stream))
        footprint = np.asarray([fnum(r.get("min_obstacle_clearance")) for r in metric_rows])
        ratio = jobs / np.maximum(np.abs(jtask), 1e-12)
        values: dict[str, Any] = {}
        for label, threshold in (("10pct", .1), ("50pct", .5), ("crossover", 1), ("2x", 2), ("5x", 5)):
            idx = first_sustained(t, ratio >= threshold, SUSTAINED_SECONDS)
            values[f"d_first_{label}"] = footprint[idx] if idx is not None else NA
            values[f"t_first_{label}"] = t[idx] if idx is not None else NA
        progress_idx = None
        for i in range(len(t)):
            j = int(np.searchsorted(t, t[i] + 10.0))
            if j >= len(t):
                break
            disp = math.hypot(m["object_x"][j] - m["object_x"][i], m["object_y"][j] - m["object_y"][i])
            error_drop = (epos[i] - epos[j]) / max(epos[i], 1e-9)
            if disp < 0.002 and error_drop < 0.005:
                progress_idx = i
                break
        f2_time = fnum(a["evidence_start_s"]) if a["primary_failure_class"].startswith("F2_") else math.nan
        f2_idx = int(np.argmin(abs(t - f2_time))) if math.isfinite(f2_time) else None
        final_start = np.searchsorted(t, max(t[-1] - 10.0, t[0]))
        row = {
            "run_id": f"{scene}/{pair}", "scene": scene, "outcome": a["task_outcome"],
            "primary_failure": a["primary_failure_class"],
            "geometry_basis_historical": "center_to_disc_or_polygon",
            "geometry_basis_new": "orientation_aware_footprint_min_sdf",
            "cost_signal_status": "MEASURED_STATE_KNOT0_PROXY_NOT_PREDICTED_RANKING_ROLLOUT",
            **values,
            "d_progress_loss": footprint[progress_idx] if progress_idx is not None else NA,
            "t_progress_loss": t[progress_idx] if progress_idx is not None else NA,
            "d_f2_onset": footprint[f2_idx] if f2_idx is not None else NA,
            "t_f2_onset": t[f2_idx] if f2_idx is not None else NA,
            "d_final_stall": float(np.nanmedian(footprint[final_start:])),
            "d_center_final": float(np.nanmedian(center[final_start:])),
            "Jtask_final_proxy": float(np.nanmedian(jtask[final_start:])),
            "Jobs_historical_final_proxy": float(np.nanmedian(jobs[final_start:])),
            "Jobs_over_Jtask_final_proxy": float(np.nanmedian(ratio[final_start:])),
            "sustained_window_s": SUSTAINED_SECONDS,
        }
        rows.append(row)
    return rows


def corridor_rows() -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from scripts.marginal_obstacle_basin_forensics import transformed_clearances
    probes = {
        "shelf_gap_center": ("shelf_gap", (0.39, 0.0, 0.0)),
        "slalom_gate1_center": ("slalom", (0.27, 0.22, 0.0)),
        "slalom_gate2_center": ("slalom", (0.46, 0.0, 0.0)),
        "slalom_gate3_center": ("slalom", (0.27, -0.22, 0.0)),
    }
    rows = []
    for probe, (scene, pose) in probes.items():
        distances = transformed_clearances(scene, pose)
        dmin = min(distances.values())
        for sigma in SIGMAS:
            parts = {name: math.exp(-d / sigma) for name, d in distances.items()}
            rows.append({
                "probe": probe, "scene": scene, "family": "exponential", "range_m": sigma,
                "raw_min_clearance_m": dmin, "min_sdf_unit_potential": math.exp(-dmin / sigma),
                "multi_obstacle_sum_unit_potential": sum(parts.values()),
                "individual_contributions_json": json.dumps(parts, sort_keys=True),
            })
        for epsilon in EPSILONS:
            parts = {name: max(0.0, (epsilon - d) / epsilon) ** 2 for name, d in distances.items()}
            rows.append({
                "probe": probe, "scene": scene, "family": "relu", "range_m": epsilon,
                "raw_min_clearance_m": dmin,
                "min_sdf_unit_potential": max(0.0, (epsilon - dmin) / epsilon) ** 2,
                "multi_obstacle_sum_unit_potential": sum(parts.values()),
                "individual_contributions_json": json.dumps(parts, sort_keys=True),
            })
    return rows


def aggregate(parts: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rescoring = out / "forensics/sigma_candidate_rescoring.csv"
    with rescoring.open("x") as target:
        first = True
        for result in sorted(parts, key=lambda r: (r["task_id"], r["scene"])):
            with Path(result["part"]).open() as source:
                header = source.readline()
                if first:
                    target.write(header)
                    first = False
                shutil.copyfileobj(source, target)
    sigma_rows = []
    for sigma in SIGMAS:
        for task_id in TASK_IDS:
            for scene in SCENES:
                result = next(r for r in parts if r["task_id"] == task_id and r["scene"] == scene)
                acc = result["sigma_acc"][sigma]
                sigma_rows.append({
                    "task_cost_id": task_id, "scene": scene, "sigma": sigma,
                    "decision_events": len(acc["changed"]),
                    "fraction_ranking_decisions_changed": np.mean(acc["changed"]),
                    "fraction_changed_vs_sigma040": np.mean(acc["changed_vs_040"]),
                    "median_rank_correlation": median(acc["corr"]),
                    "fraction_task_best_candidates_rejected": np.mean(acc["task_best_rejected"]),
                    "fraction_changed_decisions_improving_clearance": np.mean(acc["clearance_improved"]),
                    "fraction_changed_decisions_worsening_task_progress": np.mean(acc["task_progress_worsened"]),
                    "median_winner_Jobs_Jtask": median(acc["winner_ratio"]),
                    "fraction_decisions_Jobs_gt_Jtask": np.mean(acc["winner_gt_1"]),
                    "fraction_decisions_Jobs_gt_5x_Jtask": np.mean(acc["winner_gt_5"]),
                })
    eps_rows = []
    for epsilon in EPSILONS:
        for task_id in TASK_IDS:
            for scene in SCENES:
                result = next(r for r in parts if r["task_id"] == task_id and r["scene"] == scene)
                acc = result["eps_acc"][epsilon]
                eps_rows.append({
                    "task_cost_id": task_id, "scene": scene, "epsilon": epsilon,
                    "decision_events": len(acc["decision_states"]),
                    "fraction_selected_states_inside_support": np.mean(acc["selected_inside"]),
                    "fraction_candidate_pools_with_any_inside_support": np.mean(acc["any_inside"]),
                    "fraction_actual_near_collision_states_inside_support": (
                        sum(acc["near_collision_inside"]) / max(1, sum(acc["near_collision"]))
                    ),
                    "fraction_F2_onset_states_inside_support": median(acc.get("f2_onset_inside", [])),
                    "fraction_rank_changes_at_w5000": np.mean(acc["rank_changed"]),
                })
    detail = []
    for result in parts:
        if result["earliest"]:
            event = result["earliest"]
            for candidate in event.pop("candidates"):
                detail.append({**event, **candidate})
    return sigma_rows, eps_rows, detail


def derive_weights(parts: list[dict[str, Any]], selected: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, float], list[float]] = defaultdict(list)
    for result in parts:
        if result["scene"] not in PILOT_SCENES:
            continue
        for key, values in result["reference"].items():
            merged[key].extend(values)
    rows = []
    for family, ranges in selected.items():
        for value in ranges:
            base = merged[(family, value)]
            for beta in (0.25, 0.50, 1.00):
                weights = [beta * x for x in base]
                rows.append({
                    "family": family, "range_parameter_m": value, "beta_target": beta,
                    "derived_weight": median(weights), "reference_event_count": len(weights),
                    "weight_q25": percentile(weights, 25), "weight_q75": percentile(weights, 75),
                    "reference_clearance_min_m": REFERENCE_CLEARANCE[0],
                    "reference_clearance_max_m": REFERENCE_CLEARANCE[1],
                    "reference_geometry": "orientation_aware_footprint_min_sdf",
                })
    return rows


def make_figures(out: Path, sigma_rows: list[dict[str, Any]], eps_rows: list[dict[str, Any]], corridor: list[dict[str, Any]], historical: list[dict[str, Any]]) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    d = np.linspace(0, .20, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SIGMAS:
        ax.plot(d, np.exp(-d/s), label=f"{s:.3f}")
    ax.set(xlabel="footprint clearance d (m)", ylabel="unit exponential potential", title="Exponential range")
    ax.grid(alpha=.25); ax.legend(title="sigma (m)", ncol=2)
    fig.tight_layout(); fig.savefig(figdir / "sigma_potential_vs_clearance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for e in EPSILONS:
        ax.plot(d, np.square(np.maximum(0, (e-d)/e)), label=f"ReLU {e:.3f}")
    for s in (0.015, 0.025, 0.040):
        ax.plot(d, np.exp(-d/s), ls="--", label=f"Exp {s:.3f}")
    ax.set(xlabel="footprint clearance d (m)", ylabel="unit potential", title="Footprint ReLU vs exponential")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "relu_vs_exp_potential_shapes.png", dpi=180); plt.close(fig)

    aggregate_sigma = defaultdict(list)
    for row in sigma_rows:
        aggregate_sigma[row["sigma"]].append(row["fraction_ranking_decisions_changed"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = list(SIGMAS); ys = [np.mean(aggregate_sigma[x]) for x in xs]
    ax.plot(xs, ys, marker="o"); ax.set(xlabel="sigma (m)", ylabel="fraction of ranking events changed", title="Footprint exponential, w=5000")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir / "sigma_candidate_rank_change.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for probe in ("shelf_gap_center", "slalom_gate1_center", "slalom_gate2_center", "slalom_gate3_center"):
        rows = [r for r in corridor if r["probe"] == probe and r["family"] == "exponential"]
        ax.plot([r["range_m"] for r in rows], [r["multi_obstacle_sum_unit_potential"] for r in rows], marker="o", label=probe)
    ax.set(xlabel="sigma (m)", ylabel="sum of unit obstacle potentials", title="Valid-corridor exponential tail floor")
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / "shelf_corridor_floor_vs_sigma.png", dpi=180); plt.close(fig)

    aggregate_eps = defaultdict(list)
    for row in eps_rows:
        aggregate_eps[row["epsilon"]].append(row["fraction_selected_states_inside_support"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(EPSILONS, [np.mean(aggregate_eps[e]) for e in EPSILONS], marker="o")
    ax.set(xlabel="epsilon (m)", ylabel="fraction selected trajectories inside support", title="ReLU support coverage")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir / "epsilon_support_coverage.png", dpi=180); plt.close(fig)

    for scene, filename in (("single_obstacle", "historical_single_cost_ratio_vs_clearance.png"), ("shelf_gap", "historical_shelf_cost_ratio_vs_clearance.png")):
        rows = [r for r in historical if r["scene"] == scene]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter([fnum(r["d_final_stall"]) for r in rows], [fnum(r["Jobs_over_Jtask_final_proxy"]) for r in rows], c="tab:red")
        for r in rows: ax.annotate(r["run_id"].split("/")[-1], (fnum(r["d_final_stall"]), fnum(r["Jobs_over_Jtask_final_proxy"])), fontsize=7)
        ax.axhline(1, color="black", ls="--"); ax.set(xlabel="final footprint clearance (m)", ylabel="historical measured-state Jobs/Jtask proxy", title=f"Historical {scene}: final state")
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir / filename, dpi=180); plt.close(fig)

    # Analytic radius figure.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fraction, label in ((.5,"50%"),(.1,"10%"),(.05,"5%"),(.01,"1%")):
        ax.plot(SIGMAS, [-s*math.log(fraction) for s in SIGMAS], marker="o", label=label)
    ax.set(xlabel="sigma (m)", ylabel="clearance radius (m)", title="Analytic exponential influence radii")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "sigma_influence_radius.png", dpi=180); plt.close(fig)


def render_selection(out: Path, sigma_rows: list[dict[str, Any]], eps_rows: list[dict[str, Any]], corridor: list[dict[str, Any]], selected: Mapping[str, Sequence[float]]) -> None:
    def agg_sigma(s: float, field: str) -> float:
        vals = [fnum(r[field]) for r in sigma_rows if r["sigma"] == s and r["scene"] in PILOT_SCENES]
        return float(np.mean(vals))
    shelf_floor = {r["range_m"]: r["multi_obstacle_sum_unit_potential"] for r in corridor if r["probe"] == "shelf_gap_center" and r["family"] == "exponential"}
    exp = selected["exponential"]
    relu = selected["relu"]
    sigma_text = f"""# Sigma range selection

All rankings below use the same logged candidate pools and orientation-aware footprint minimum signed distance. The diagnostic weight is fixed at 5000 only to compare spatial shape; it is not the live tuning weight.

Selected range bracket: **SHORT={exp[0]:.3f} m**, **MIDDLE={exp[1]:.3f} m**, and **HISTORICAL=0.040 m**. Stage-1 live finalists are SHORT and MIDDLE; the historical center-distance sigma/w pair is retained as the provenance anchor and sigma=.040 remains in every offline comparison.

| sigma (m) | pilot rank-change fraction | winner Jobs/Jtask median (mean across runs) | shelf multi-obstacle unit floor |
|---:|---:|---:|---:|
"""
    for s in SIGMAS:
        sigma_text += f"| {s:.3f} | {agg_sigma(s, 'fraction_ranking_decisions_changed'):.4f} | {agg_sigma(s, 'median_winner_Jobs_Jtask'):.4g} | {shelf_floor[s]:.6f} |\n"
    sigma_text += "\nSelection rule: SHORT=.015 retains measurable candidate discrimination while reducing the shelf center tail by about an order of magnitude versus .040. MIDDLE=.025 provides a deliberate early-awareness bracket while keeping less than half the historical unit corridor floor. Weight is calibrated separately.\n"
    (out / "selection/sigma_selection.md").write_text(sigma_text)

    def agg_eps(e: float, field: str) -> float:
        vals = [fnum(r[field]) for r in eps_rows if r["epsilon"] == e and r["scene"] in PILOT_SCENES]
        return float(np.mean(vals))
    relu_floor = {r["range_m"]: r["multi_obstacle_sum_unit_potential"] for r in corridor if r["probe"] == "shelf_gap_center" and r["family"] == "relu"}
    eps_text = f"""# ReLU support selection

Selected live-range finalists: **{relu[0]:.3f} m** and **{relu[1]:.3f} m**.

| epsilon (m) | selected-trajectory support coverage | rank-change fraction at diagnostic w=5000 | shelf corridor floor |
|---:|---:|---:|---:|
"""
    for e in EPSILONS:
        eps_text += f"| {e:.3f} | {agg_eps(e, 'fraction_selected_states_inside_support'):.4f} | {agg_eps(e, 'fraction_rank_changes_at_w5000'):.4f} | {relu_floor[e]:.6f} |\n"
    eps_text += "\nSelection rule: bracket support that is active in recorded obstacle-relevant decisions while remaining exactly zero at the valid shelf/slalom center probes.\n"
    (out / "selection/epsilon_selection.md").write_text(eps_text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {out}")
    for directory in ("forensics", "selection", "metrics", "figures", "report", "handoff", "_parts"):
        (out / directory).mkdir(parents=True, exist_ok=True)

    influence = [{
        "sigma": s, "d_50": -s*math.log(.5), "d_25": -s*math.log(.25),
        "d_10": -s*math.log(.1), "d_05": -s*math.log(.05), "d_01": -s*math.log(.01),
    } for s in SIGMAS]
    write_csv(out / "forensics/sigma_influence_radius.csv", influence)
    historical = historical_forensics()
    write_csv(out / "forensics/historical_failure_clearance.csv", historical)
    corridor = corridor_rows()
    write_csv(out / "forensics/corridor_floor_diagnostics.csv", corridor)

    jobs = [(task, scene, str(out / "_parts")) for task in TASK_IDS for scene in SCENES]
    parts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(process_candidate_run, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            parts.append(result)
            print(f"processed {result['task_id']}/{result['scene']}: {result['events']} events", flush=True)
    sigma_rows, eps_rows, detail = aggregate(parts, out)
    write_csv(out / "forensics/sigma_decision_summary.csv", sigma_rows)
    write_csv(out / "forensics/epsilon_support_summary.csv", eps_rows)
    write_csv(out / "forensics/historical_decision_events.csv", detail)

    # Evidence-selected brackets.  A 1 cm ReLU is nearly inert, while 3--5 cm
    # spans the observed useful support and remains exactly zero at the valid
    # shelf/slalom center probes.  Exponential .015/.025 bracket short/middle
    # response; .040 is always retained separately as the historical anchor.
    selected_exp = (0.015, 0.025)
    selected_relu = (0.030, 0.050)
    selected = {"exponential": selected_exp, "relu": selected_relu}
    render_selection(out, sigma_rows, eps_rows, corridor, selected)
    weights = derive_weights(parts, selected)
    write_csv(out / "selection/derived_weight_table.csv", weights)

    # Manifest deliberately retains only beta=.25 and .50 for live screening;
    # beta=1 remains calibrated/reportable but is withheld unless low/mid fail
    # to bracket any decision effect without pathological dominance.
    manifest = []
    config_id = 0
    for family, ranges in selected.items():
        for range_value in ranges:
            for beta in (.25, .50):
                row = next(r for r in weights if r["family"] == family and r["range_parameter_m"] == range_value and r["beta_target"] == beta)
                config_id += 1
                manifest.append({
                    "obstacle_config_id": f"OC{config_id:02d}", "family": family,
                    "range_parameter_m": range_value, "raw_weight": row["derived_weight"],
                    "beta_calibration_target": beta, "geometry": "orientation_aware_footprint_min_sdf",
                    "stage1_status": "PLANNED", "historical_reference": 0,
                })
    manifest.append({
        "obstacle_config_id": "HIST_EXP_CENTER", "family": "historical_center_exponential",
        "range_parameter_m": .04, "raw_weight": 5000, "beta_calibration_target": NA,
        "geometry": "object_center_sum_over_obstacles", "stage1_status": "REFERENCE_ONLY_NOT_PRIMARY_FAMILY_COMPARISON",
        "historical_reference": 1,
    })
    write_csv(out / "selection/live_parameter_manifest.csv", manifest)
    make_figures(out, sigma_rows, eps_rows, corridor, historical)

    summary_rows = []
    shelf = {r["range_m"]: r for r in corridor if r["probe"] == "shelf_gap_center" and r["family"] == "exponential"}
    for s in SIGMAS:
        inf = next(r for r in influence if r["sigma"] == s)
        single = next((r for r in historical if r["run_id"] == "single_obstacle/pair01"), {})
        shelf_hist = next((r for r in historical if r["run_id"] == "shelf_gap/pair02"), {})
        sr = [r for r in sigma_rows if r["sigma"] == s]
        summary_rows.append({
            "sigma": s, "d50": inf["d_50"], "d25": inf["d_25"], "d10": inf["d_10"], "d05": inf["d_05"], "d01": inf["d_01"],
            "single_d_crossover": single.get("d_first_crossover", NA), "single_d_progress_loss": single.get("d_progress_loss", NA), "single_d_stall": single.get("d_final_stall", NA),
            "shelf_d_crossover": shelf_hist.get("d_first_crossover", NA), "shelf_d_progress_loss": shelf_hist.get("d_progress_loss", NA), "shelf_d_stall": shelf_hist.get("d_final_stall", NA),
            "corridor_floor_shelf": shelf[s]["multi_obstacle_sum_unit_potential"],
            "corridor_floor_slalom": max(r["multi_obstacle_sum_unit_potential"] for r in corridor if r["scene"] == "slalom" and r["family"] == "exponential" and r["range_m"] == s),
            "fraction_candidate_rank_changes": np.mean([r["fraction_ranking_decisions_changed"] for r in sr]),
            "fraction_Jobs_gt_Jtask": np.mean([r["fraction_decisions_Jobs_gt_Jtask"] for r in sr]),
            "verdict": "LIVE_RANGE_FINALIST" if s in selected_exp else ("HISTORICAL_REFERENCE" if s == .04 else "OFFLINE_ONLY"),
        })
    write_csv(out / "forensics/sigma_forensic_summary.csv", summary_rows)

    provenance = {
        "analysis_type": "offline_only_no_controller_launch", "root_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "controller_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WT, text=True).strip(),
        "workers": min(args.workers, len(jobs)), "task_cost_ids": TASK_IDS, "scenes": SCENES,
        "sigma_grid_m": SIGMAS, "epsilon_grid_m": EPSILONS, "diagnostic_weight": W_DIAGNOSTIC,
        "sustained_window_s": SUSTAINED_SECONDS, "selected_ranges": selected,
        "baseline_live_obstacle_assertion": "all TR07/TR08 J_obstacle_applied_live expected zero under lcs_contact",
        "candidate_run_inventory": [{k: r[k] for k in ("task_id", "scene", "attempt", "events", "baseline_selection_mismatch")} for r in sorted(parts, key=lambda x:(x['task_id'],x['scene']))],
    }
    (out / "forensics/offline_analysis_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(out / "_parts")
    print(json.dumps({"output": str(out), "selected": selected, "weights": len(weights), "events": sum(r["events"] for r in parts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
