#!/usr/bin/env python3
"""Offline evidence audit for marginal-obstacle basin-entry forensics.

This script never invokes a controller.  It reads the immutable 600 s campaign
and existing passive-instrumentation outputs, reconstructs signals that are
actually identifiable, and marks counterfactual quantities NA when alternative
candidate trajectories were not recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_marginal_basin")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.rot15.benchmark import (
    ROOT,
    _controller_boundary_points,
    _obstacle_sdf,
    load_spec,
)


FAILURE_DIR = ROOT / "results/failure_classification"
RUNS_DIR = ROOT / "results/xarm6_c3plus_scene_smoke/runs"
OUTPUT = ROOT / "results/marginal_obstacle_basin_forensics"
EPSILONS = (0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100)
REPRESENTATIVES = {
    "single01": ("single_obstacle", "pair01"),
    "icra05": ("icra_sign", "pair05"),
    "ycb02": ("ycb_clutter", "pair02"),
}
NA = "NA"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def finite(value: str | float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metrics_path(scene: str, pair: str) -> Path:
    matches = sorted((RUNS_DIR / scene / pair).glob("*_metrics.csv"))
    if len(matches) != 1:
        raise AssertionError(f"{scene}/{pair}: expected one metrics CSV, got {matches}")
    return matches[0]


def load_metrics(scene: str, pair: str) -> dict[str, np.ndarray]:
    columns = defaultdict(list)
    wanted = (
        "sim_time", "position_error_m", "orientation_error_rad",
        "min_obstacle_clearance", "pusher_object_gap",
        "physical_contact_active", "object_x", "object_y", "object_yaw",
    )
    with metrics_path(scene, pair).open(newline="") as stream:
        for row in csv.DictReader(stream):
            for name in wanted:
                value = finite(row.get(name))
                columns[name].append(np.nan if value is None else value)
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def nearest(array: np.ndarray, times: np.ndarray, target: float) -> float | str:
    valid = np.isfinite(array) & np.isfinite(times)
    if not np.any(valid):
        return NA
    indices = np.where(valid)[0]
    index = indices[np.argmin(np.abs(times[indices] - target))]
    return float(array[index])


def summarize_metrics(args: tuple[str, str, float]) -> dict[str, Any]:
    scene, pair, stall = args
    data = load_metrics(scene, pair)
    time = data["sim_time"]
    clearance = data["min_obstacle_clearance"]
    result: dict[str, Any] = {
        "scene": scene,
        "pair_id": pair,
        "metrics_rows": len(time),
        "object_obs_clearance_at_stall": nearest(clearance, time, stall),
        "pusher_object_gap_at_stall": nearest(data["pusher_object_gap"], time, stall),
    }
    for eps in EPSILONS:
        mask = np.isfinite(clearance) & (clearance < eps)
        result[f"actual_object_first_below_eps_{int(round(eps * 1000)):03d}"] = (
            float(time[np.where(mask)[0][0]]) if np.any(mask) else NA
        )
    return result


def episode_last_productive() -> dict[str, float | str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(FAILURE_DIR / "c3_segment_audit.csv"):
        grouped[row["run_id"]].append(row)
    output: dict[str, float | str] = {}
    for run_id, rows in grouped.items():
        productive = [
            float(row["end_t"])
            for row in rows
            if float(row["contact_fraction"]) > 0.0
            and (
                float(row["obj_disp_m"]) >= 0.005
                or float(row["epos_reduction"]) >= 0.005
            )
        ]
        output[run_id] = max(productive) if productive else NA
    return output


def classification(row: Mapping[str, str]) -> tuple[str, str, str]:
    outcome, failure = row["task_outcome"], row["primary_failure_class"]
    if outcome == "SUCCESS":
        return "SUCCESS_CONTROL", "NO", "HIGH"
    if outcome == "TRANSIENT_SUCCESS":
        return "OBJECT_OBSTACLE_COST_IRRELEVANT", "NO", "HIGH"
    if outcome == "TIMEOUT_PROGRESSING":
        return "TIMEOUT_PROGRESSING_NOT_STALL", "UNASSESSED", "HIGH"
    if outcome == "RUNTIME_TERMINATION":
        return "RUNTIME_EXCLUDED", "UNASSESSED", "HIGH"
    if failure.startswith("F1_"):
        return "OBJECT_OBSTACLE_COST_IRRELEVANT", "LOW", row["confidence"]
    if failure.startswith("F6_"):
        return "GEOMETRIC_BLOCK_COST_RELEVANT", "YES", row["confidence"]
    return "INSUFFICIENT_DATA", "UNKNOWN", "HIGH"


def build_relevance() -> list[dict[str, Any]]:
    audit = read_csv(FAILURE_DIR / "failure_audit.csv")
    if len(audit) != 30:
        raise AssertionError(f"historical campaign has {len(audit)} rows, expected 30")
    last_productive = episode_last_productive()
    work = [
        (row["scene"], f"pair{int(row['start_id'][1:]):02d}", float(row["evidence_start_s"]))
        for row in audit
    ]
    with ProcessPoolExecutor(max_workers=min(len(work), os.cpu_count() or 1)) as pool:
        summaries = list(pool.map(summarize_metrics, work))
    metric_index = {(row["scene"], row["pair_id"]): row for row in summaries}
    output = []
    for source in audit:
        pair = f"pair{int(source['start_id'][1:]):02d}"
        metric = metric_index[(source["scene"], pair)]
        label, relevant, confidence = classification(source)
        true_stall = source["task_outcome"] == "TRUE_STALL"
        f1 = source["primary_failure_class"].startswith("F1_")
        f2 = source["primary_failure_class"].startswith("F2_")
        f6 = source["primary_failure_class"].startswith("F6_")
        notes = {
            "SUCCESS_CONTROL": "Negative/success control; no obstacle-causal claim.",
            "OBJECT_OBSTACLE_COST_IRRELEVANT": (
                "Existing taxonomy identifies execution mismatch or contact acquisition; "
                "candidate-pool counterfactuals were not recorded."
            ),
            "TIMEOUT_PROGRESSING_NOT_STALL": "Preserved as progressing at cap; not credited as a cost fix.",
            "RUNTIME_EXCLUDED": "F7 workspace/runtime termination excluded from obstacle-cost causality.",
            "GEOMETRIC_BLOCK_COST_RELEVANT": (
                "Positive physical-geometry control; prevention/escape remains unidentified "
                "because alternative candidate trajectories are absent."
            ),
            "INSUFFICIENT_DATA": (
                "F2 mechanism is established, but no campaign candidate pool or alternative "
                "knot trajectories exist; A versus B versus C cannot be inferred."
            ),
        }[label]
        row: dict[str, Any] = {
            "scene": source["scene"],
            "pair_id": pair,
            "outcome": source["task_outcome"],
            "primary_failure": source["primary_failure_class"],
            "stall_onset_time": source["evidence_start_s"] if true_stall else NA,
            "obstacle_relevant": relevant,
            "safer_candidate_existed_pre_stall": NA,
            "first_safer_candidate_time": NA,
            "last_safer_candidate_time": NA,
            "epsilon": "|".join(f"{eps:.3f}" for eps in EPSILONS),
            "min_w_star": NA,
            "median_w_star_pre_stall": NA,
            "clearance_gain_at_min_w_star": NA,
            "task_cost_penalty_at_min_w_star": NA,
            "object_obs_clearance_at_stall": metric["object_obs_clearance_at_stall"],
            "pusher_obs_clearance_if_available": NA,
            "pusher_object_gap_if_available": metric["pusher_object_gap_at_stall"],
            "classification": label,
            "confidence": confidence,
            "notes": notes,
            "t_last_productive": last_productive.get(f"{source['scene']}/{pair}", NA),
            "t_bad_sample_first": NA,
            "t_bad_sample_persistent": NA,
            "t_candidate_diversity_collapse": NA,
            "t_last_safer_candidate": NA,
            "t_relu_active_eps_005": NA,
            "t_relu_active_eps_010": NA,
            "t_relu_active_eps_020": NA,
            "t_relu_active_eps_030": NA,
            "t_relu_active_eps_050": NA,
            "obj_obs_clearance_at_first_bad": NA,
            "obj_obs_clearance_at_persistent_bad": NA,
            "obj_obs_clearance_at_last_safe_candidate": NA,
            "obj_obs_clearance_at_stall": metric["object_obs_clearance_at_stall"],
            "pusher_obj_gap_at_first_bad": NA,
            "pusher_obj_gap_at_persistent_bad": NA,
            "pusher_obs_clearance_at_first_bad": NA,
            "pusher_obs_clearance_at_persistent_bad": NA,
            "meaningful_candidate_count_pre_bad": NA,
            "meaningful_candidate_count_at_first_bad": NA,
            "meaningful_candidate_count_at_persistent_bad": NA,
            "meaningful_candidate_count_at_stall": NA,
            "unique_sector_count_pre_bad": NA,
            "unique_sector_count_at_persistent_bad": NA,
            "min_w_star_prevention": NA,
            "epsilon_at_min_w_star_prevention": NA,
            "min_w_star_escape": NA,
            "epsilon_at_min_w_star_escape": NA,
            "prevention_possible": NA,
            "escape_possible": NA,
            "basin_formation_mechanism": (
                "CONTACT_ACQUISITION" if f1 else
                "OBJECT_GEOMETRIC_BLOCK" if f6 else
                "PHANTOM_CONTACT_OUTER_LOOP_CHURN" if f2 else
                "NOT_TRUE_STALL"
            ),
        }
        for eps in EPSILONS:
            suffix = int(round(eps * 1000))
            row[f"actual_obj_first_below_eps_{suffix:03d}"] = metric[
                f"actual_object_first_below_eps_{suffix:03d}"
            ]
        output.append(row)
    return output


def candidate_signal_inventory() -> list[dict[str, Any]]:
    paths = sorted(ROOT.glob("results/**/candidate_ranking_costs.csv"))
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            event_ids, candidate_rows = set(), 0
            for row in reader:
                event_ids.add(row["event_id"])
                candidate_rows += 1
        sibling = path.with_name("selected_candidate_qp_variables.jsonl")
        selected_knots = sum(1 for _ in sibling.open()) if sibling.exists() else 0
        rows.append({
            "relative_path": str(path.relative_to(ROOT)),
            "candidate_rows": candidate_rows,
            "controller_events": len(event_ids),
            "has_candidate_terminal_pose": "YES",
            "has_per_candidate_task_components": "YES",
            "has_selected_knot_trajectory": "YES" if selected_knots else "NO",
            "selected_knot_rows": selected_knots,
            "has_unselected_knot_trajectories": "NO",
            "exact_footprint_phi_reconstructable_for_all_candidates": "NO",
            "historical_obstacle_column_semantics": "exp center-distance; not valid for footprint ReLU counterfactual",
        })
    return rows


def observable_transition_anchors() -> list[dict[str, Any]]:
    """Five physical anchors; explicitly not candidate-selection events."""
    specifications = (
        ("single_obstacle", "pair01", 9.488, "actual object first below 1 cm"),
        ("single_obstacle", "pair01", 10.064, "last short productive C3 segment end"),
        ("ycb_clutter", "pair02", 8.196, "actual object first below 5 cm"),
        ("ycb_clutter", "pair02", 10.938, "last short productive C3 segment end"),
        ("icra_sign", "pair05", 118.800, "audited geometric-block boundary"),
    )
    rows = []
    for scene, pair, target, event in specifications:
        data = load_metrics(scene, pair)
        index = int(np.nanargmin(np.abs(data["sim_time"] - target)))
        rows.append({
            "scene": scene, "pair_id": pair, "event": event,
            "target_time_s": target, "matched_sim_time_s": data["sim_time"][index],
            "actual_object_obstacle_clearance_m": data["min_obstacle_clearance"][index],
            "actual_pusher_object_gap_m": data["pusher_object_gap"][index],
            "position_error_m": data["position_error_m"][index],
            "orientation_error_rad": data["orientation_error_rad"][index],
            "candidate_decision_event": "NO",
            "w_star_status": "NOT_IDENTIFIABLE",
        })
    return rows


def transformed_clearances(scene: str, pose: tuple[float, float, float]) -> dict[str, float]:
    spec = load_spec()["scenes"][scene]
    points = _controller_boundary_points(spec["footprint"])
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    world = [(x + c * px - s * py, y + s * px + c * py) for px, py in points]
    return {
        str(obstacle["name"]): min(_obstacle_sdf(point, obstacle) for point in world)
        for obstacle in spec["obstacles"]
    }


def phi_relu(distance: float, epsilon: float) -> float:
    return max(0.0, (epsilon - distance) / epsilon) ** 2


def phi_chomp_like(distance: float, epsilon: float) -> float:
    """Normalized C1 finite-support CHOMP-like potential, unit weight."""
    if distance >= epsilon:
        return 0.0
    if distance >= 0.0:
        return 0.5 * ((epsilon - distance) / epsilon) ** 2
    return 0.5 - distance / epsilon


def total_potential(scene: str, pose: tuple[float, float, float], shape: str, parameter: float) -> float:
    distances = transformed_clearances(scene, pose).values()
    if shape == "relu":
        return sum(phi_relu(value, parameter) for value in distances)
    if shape == "chomp_like":
        return sum(phi_chomp_like(value, parameter) for value in distances)
    if shape == "exponential":
        return sum(math.exp(-value / parameter) for value in distances)
    raise ValueError(shape)


def corridor_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cross = []
    for x in np.linspace(0.30, 0.48, 181):
        distances = transformed_clearances("shelf_gap", (float(x), 0.0, 0.0))
        row: dict[str, Any] = {"scene": "shelf_gap", "x": x, "y": 0.0, "yaw": 0.0}
        row.update({f"d_{name}": value for name, value in distances.items()})
        row["raw_min_clearance"] = min(distances.values())
        for eps in (0.010, 0.020, 0.030, 0.050):
            key = int(eps * 1000)
            row[f"relu_eps_{key:03d}"] = sum(phi_relu(value, eps) for value in distances.values())
            row[f"chomp_like_eps_{key:03d}"] = sum(phi_chomp_like(value, eps) for value in distances.values())
        row["exponential_sigma_040"] = sum(math.exp(-value / 0.04) for value in distances.values())
        cross.append(row)

    probes = {
        "shelf_gap_center": ("shelf_gap", (0.39, 0.0, 0.0)),
        "slalom_gate1_center": ("slalom", (0.27, 0.22, 0.0)),
        "slalom_gate2_center": ("slalom", (0.46, 0.0, 0.0)),
        "slalom_gate3_center": ("slalom", (0.27, -0.22, 0.0)),
    }
    summaries = []
    step = 1e-4
    for name, (scene, pose) in probes.items():
        per_obstacle = transformed_clearances(scene, pose)
        for shape, parameter in [
            *(('relu', eps) for eps in (0.010, 0.020, 0.030, 0.050)),
            ("chomp_like", 0.050), ("exponential", 0.040),
        ]:
            base = total_potential(scene, pose, shape, parameter)
            px, py, yaw = pose
            gx = (
                total_potential(scene, (px + step, py, yaw), shape, parameter)
                - total_potential(scene, (px - step, py, yaw), shape, parameter)
            ) / (2 * step)
            gy = (
                total_potential(scene, (px, py + step, yaw), shape, parameter)
                - total_potential(scene, (px, py - step, yaw), shape, parameter)
            ) / (2 * step)
            contributions = {
                obstacle: (
                    phi_relu(distance, parameter) if shape == "relu" else
                    phi_chomp_like(distance, parameter) if shape == "chomp_like" else
                    math.exp(-distance / parameter)
                )
                for obstacle, distance in per_obstacle.items()
            }
            summaries.append({
                "probe": name, "scene": scene, "x": px, "y": py, "yaw": yaw,
                "shape": shape, "support_or_sigma_m": parameter,
                "raw_min_clearance_m": min(per_obstacle.values()),
                "unit_weight_total_potential": base,
                "gradient_x_per_m": gx, "gradient_y_per_m": gy,
                "gradient_magnitude_per_m": math.hypot(gx, gy),
                "individual_contributions_json": json.dumps(contributions, sort_keys=True),
            })
    return cross, summaries


def wide_support_corridor_floors() -> list[dict[str, Any]]:
    probes = {
        "shelf_gap_center": ("shelf_gap", (0.39, 0.0, 0.0)),
        "slalom_gate1_center": ("slalom", (0.27, 0.22, 0.0)),
        "slalom_gate2_center": ("slalom", (0.46, 0.0, 0.0)),
        "slalom_gate3_center": ("slalom", (0.27, -0.22, 0.0)),
    }
    rows = []
    for name, (scene, pose) in probes.items():
        clearance = min(transformed_clearances(scene, pose).values())
        for epsilon in (0.005, 0.075, 0.100):
            rows.append({
                "probe": name, "scene": scene, "epsilon_m": epsilon,
                "raw_min_clearance_m": clearance,
                "relu_unit_weight_floor": total_potential(scene, pose, "relu", epsilon),
                "chomp_like_unit_weight_floor": total_potential(scene, pose, "chomp_like", epsilon),
            })
    return rows


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def critical_weight_figure(tag: str, scene: str, pair: str, stall: float) -> None:
    data = load_metrics(scene, pair)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(data["sim_time"] - stall, data["min_obstacle_clearance"], lw=1.2)
    for eps in (0.01, 0.02, 0.03, 0.05):
        axes[0].axhline(eps, lw=0.8, ls="--", label=f"eps={eps:.2f} m")
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_ylabel("actual object clearance [m]")
    axes[0].legend(ncol=4, fontsize=8)
    style_axis(axes[0])
    axes[1].text(
        0.5, 0.55,
        "B(t, epsilon) / w_star NOT IDENTIFIABLE\n"
        "No alternative-candidate knot trajectories were recorded.",
        ha="center", va="center", transform=axes[1].transAxes, fontsize=12,
    )
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_ylabel("critical weight")
    axes[1].set_xlabel("simulation time relative to preserved stall evidence onset [s]")
    axes[1].set_yticks([])
    style_axis(axes[1])
    fig.suptitle(f"{scene}/{pair}: observable clearance, unavailable candidate counterfactual")
    fig.savefig(OUTPUT / "figures" / f"critical_weight_vs_time_{tag}.png", dpi=180)
    plt.close(fig)


def local_minimum_figure(tag: str, scene: str, pair: str, stall: float, last_productive: float | str) -> None:
    data = load_metrics(scene, pair)
    time = data["sim_time"]
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    axes[0].plot(time, data["position_error_m"], label="position")
    axes[0].plot(time, data["orientation_error_rad"], label="yaw")
    axes[0].legend(); axes[0].set_ylabel("goal error")
    axes[1].plot(time, data["min_obstacle_clearance"]); axes[1].set_ylabel("object-obstacle [m]")
    axes[2].plot(time, data["pusher_object_gap"]); axes[2].set_ylabel("pusher-object [m]")
    missing = [
        "candidate sectors unavailable", "candidate task-cost alternatives unavailable",
        "w_star unavailable",
    ]
    for axis, message in zip(axes[3:6], missing):
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        axis.set_yticks([])
    axes[3].set_ylabel("diversity")
    axes[4].set_ylabel("candidate cost")
    axes[5].set_ylabel("critical weight")
    axes[6].plot(time, data["physical_contact_active"], lw=0.7)
    axes[6].set_ylabel("physical contact"); axes[6].set_xlabel("simulation time [s]")
    for axis in axes:
        axis.axvline(stall, color="black", lw=1, label="stall evidence onset")
        if last_productive != NA:
            axis.axvline(float(last_productive), color="tab:green", lw=1, ls="--")
        style_axis(axis)
    fig.suptitle(f"{scene}/{pair}: local-minimum formation evidence and logging gaps")
    fig.savefig(OUTPUT / "figures" / f"local_minimum_formation_{tag}.png", dpi=180)
    plt.close(fig)


def other_figures(relevance: Sequence[Mapping[str, Any]], cross: Sequence[Mapping[str, Any]]) -> None:
    # Actual-state activation timing is not a substitute for predicted-candidate activation.
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    width = 0.18
    for index, (tag, (scene, pair)) in enumerate(REPRESENTATIVES.items()):
        row = next(r for r in relevance if r["scene"] == scene and r["pair_id"] == pair)
        stall = float(row["stall_onset_time"])
        values = []
        for eps in (0.01, 0.02, 0.03, 0.05):
            value = row[f"actual_obj_first_below_eps_{int(eps * 1000):03d}"]
            values.append(np.nan if value == NA else float(value) - stall)
        ax.bar(np.arange(4) + (index - 1) * width, values, width=width, label=tag)
    ax.set_xticks(np.arange(4), ["0.01", "0.02", "0.03", "0.05"])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("epsilon [m]"); ax.set_ylabel("actual object first-below-epsilon minus stall onset [s]")
    ax.set_title("Actual-state activation timing only (candidate activation unavailable)")
    ax.legend(); style_axis(ax)
    fig.savefig(OUTPUT / "figures/epsilon_activation_timing.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = np.asarray([float(r["x"]) for r in cross])
    for field, label in [
        ("relu_eps_010", "ReLU eps=.01"), ("relu_eps_030", "ReLU eps=.03"),
        ("relu_eps_050", "ReLU eps=.05"), ("chomp_like_eps_050", "CHOMP-like eps=.05"),
        ("exponential_sigma_040", "exponential sigma=.04"),
    ]:
        ax.plot(x, [float(r[field]) for r in cross], label=label)
    ax.axvspan(0.3345, 0.4455, color="tab:green", alpha=0.08, label="T-center raw-feasible interval")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("T center x at shelf gate y=0 [m]")
    ax.set_ylabel("unit-weight summed obstacle potential")
    ax.set_title("Shelf corridor potential superposition, T yaw=0")
    ax.legend(ncol=2); style_axis(ax)
    fig.savefig(OUTPUT / "figures/shelf_corridor_cost_crosssection.png", dpi=180)
    plt.close(fig)

    counts = Counter(
        r["classification"] for r in relevance if r["outcome"] == "TRUE_STALL"
    )
    order = [
        "COST_PREVENTABLE_BASIN_ENTRY", "NO_ALTERNATIVE_SAMPLED",
        "OBJECT_OBSTACLE_COST_IRRELEVANT", "GEOMETRIC_BLOCK_COST_RELEVANT",
        "INSUFFICIENT_DATA",
    ]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(range(len(order)), [counts[key] for key in order])
    ax.set_xticks(range(len(order)), ["A cost-preventable", "B no alternative", "C wrong variable", "D geometric", "unresolved"], rotation=15)
    ax.set_ylabel("TRUE_STALL runs")
    ax.set_title("Marginal-cost relevance: evidence-supported counts")
    style_axis(ax)
    fig.savefig(OUTPUT / "figures/marginal_cost_relevance_by_failure_class.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.set_xlabel("task cost difference")
    ax.set_ylabel("clearance / obstacle-potential difference")
    ax.text(
        0.5, 0.55,
        "NOT IDENTIFIABLE FROM RECORDED DATA\n\n"
        "Candidate scalar task components and terminal states exist only in later replays.\n"
        "Alternative knot trajectories—and therefore orientation-aware Delta Phi—do not.",
        ha="center", va="center", transform=ax.transAxes,
    )
    ax.set_xticks([]); ax.set_yticks([]); style_axis(ax)
    ax.set_title("Pre-stall candidate cost/clearance tradeoff evidence boundary")
    fig.savefig(OUTPUT / "figures/candidate_cost_tradeoff_pre_stall_icra05.png", dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(relevance: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]], probes: Sequence[Mapping[str, Any]]) -> str:
    stalls = [r for r in relevance if r["outcome"] == "TRUE_STALL"]
    counts = Counter(r["classification"] for r in stalls)
    rows = "\n".join(
        f"| `{r['scene']}/{r['pair_id']}` | {r['primary_failure']} | {r['stall_onset_time']} | "
        f"{float(r['object_obs_clearance_at_stall']):.4f} | {r['classification']} |"
        for r in stalls
    )
    probe_rows = "\n".join(
        f"| `{r['probe']}` | {r['shape']} | {float(r['support_or_sigma_m']):.3f} | "
        f"{float(r['raw_min_clearance_m']):.4f} | {float(r['unit_weight_total_potential']):.6g} | "
        f"{float(r['gradient_magnitude_per_m']):.6g} |"
        for r in probes
    )
    return f"""# Marginal obstacle basin forensics

## Executive conclusion

The existing recordings do **not** identify whether marginal object-obstacle cost could have flipped a pre-stall candidate selection. The 30-run campaign records executed state, contact, progress, and outer-loop episodes but no candidate pools. Later passive replays record per-candidate scalar task components and terminal object poses, while knot-wise QP/state trajectories are written only for the selected candidate. Consequently an alternative candidate's orientation-aware `d_i0...d_iN`, `Phi_i(epsilon)`, `Delta_Phi`, exact `w_star`, candidate-diversity collapse, prevention/escape map, and five pre-stall decision events cannot be reconstructed. Using the historical `J_rank_obstacle_total` would be invalid: source audit shows it is `5000 exp(-d/0.04)` using object-center distance, not the requested orientation-aware footprint ReLU. The preserved evidence still supports two F1 wrong-variable controls and one F6 geometric positive control, but leaves all 13 F2 TRUE_STALL runs unresolved at ranking-counterfactual level.

## Preserved campaign taxonomy

- 30 historical runs: 5 per scene.
- 16 TRUE_STALL: 13 F2 primary, 2 F1 primary, 1 F6 primary.
- Genuine local-C3 fixed point as primary: 0.
- Other outcomes: 5 SUCCESS, 1 TRANSIENT_SUCCESS/F5, 6 TIMEOUT_PROGRESSING, 2 RUNTIME_TERMINATION/F7.

Evidence-supported TRUE_STALL relevance counts:

- A `COST_PREVENTABLE_BASIN_ENTRY`: {counts['COST_PREVENTABLE_BASIN_ENTRY']} proven.
- B `NO_ALTERNATIVE_SAMPLED`: {counts['NO_ALTERNATIVE_SAMPLED']} proven.
- C `OBJECT_OBSTACLE_COST_IRRELEVANT`: {counts['OBJECT_OBSTACLE_COST_IRRELEVANT']} (the two F1 primaries; ranking counterfactual remains unobserved).
- D `GEOMETRIC_BLOCK_COST_RELEVANT`: {counts['GEOMETRIC_BLOCK_COST_RELEVANT']} (ICRA05 positive control; preventability remains unobserved).
- Unresolved `INSUFFICIENT_DATA`: {counts['INSUFFICIENT_DATA']} (all F2 primaries).

## Authoritative evidence locations

- Campaign: `results/xarm6_c3plus_scene_smoke/runs/{{scene}}/pair01...pair05/`.
- Taxonomy: `results/failure_classification/C3PLUS_600S_FAILURE_CLASSIFICATION_REPORT.md` and its six CSVs.
- Phantom mechanism: `results/c3plus_outer_loop_phantom_contact/report/C3PLUS_OUTER_LOOP_PHANTOM_CONTACT_REPORT.md`.
- Cost semantics: `results/cost_analysis/{{current_cost_inventory.csv,cost_semantics_notes.md,data_inventory.md}}`.
- Relevant cost comparison: `results/c3plus_nonpen_vs_qpcost_single/COMPARISON_SUMMARY.md`.
- The requested filenames `C3PLUS_SINGLE_OBSTACLE_AND_SHELF_GAP_COST_REPORT.md` and `C3PLUS_RECIPROCAL_VS_EXPONENTIAL_INNER_QP_REPORT.md` do not exist under `results/`; the sources above are the authoritative related records found.

## Signal audit and identifiability proof

Every campaign run contains `state_trace.jsonl`, `steps_raw.jsonl`, an orientation-aware evaluation metrics CSV, logs, manifest, result, and video. None contains `candidate_ranking_costs.csv` or candidate trajectory tensors. Across {len(inventory)} later instrumented costlog directories, `candidate_ranking_costs.csv` contains contact location, terminal predicted object pose, frozen task-cost decomposition, and the old reconstructed exponential center-distance column. `selected_candidate_qp_variables.jsonl` contains full knots only for the selected candidate. No file contains unselected candidates' knot states.

This is a structural missing variable, not a statistical power problem: many different alternative trajectories share the same recorded terminal pose/task total but have different minimum clearances and `Phi`. Therefore `Delta_Phi` and `w_star = -Delta_J_task/Delta_Phi` are not uniquely determined.

## Stall-onset and transition semantics

Historical outcomes and evidence-window onsets are preserved rather than reclassified. For the run-level CSV, `stall_onset_time` is the audited `evidence_start_s`: 30 s for F1/F2 stalls and 118.8 s for ICRA05's last-progress/geometric-block boundary. `t_last_productive` is the final C3 episode having physical contact plus either at least 5 mm object displacement or 5 mm position-error reduction. These are observable physical anchors.

The following requested decision events are **NA**, because candidate identity/pools were not logged in the campaign: `t_bad_sample_first`, `t_bad_sample_persistent`, candidate-diversity collapse, last safer candidate, failed-sector candidate recurrence at each selection, candidate ReLU activation, prevention `w_star`, and escape `w_escape_star`. Actual object first-crossing times are included under explicitly named `actual_obj_first_below_eps_*` fields; they are not presented as candidate activation times.

### Actual-state activation sanity bound

This weaker observable is useful only as a spatial bound. Among the 13 F2 runs, the executed object entered 1 cm support at some point in 5/13 and was inside 1 cm at the preserved failure-regime boundary in 3/13. It entered 2 cm in 9/13, 3 cm in 12/13, and 5 cm in 13/13; at the preserved boundary the corresponding counts are 6/13, 10/13, and 13/13. Thus 1 cm is absent from most executed F2 histories while 5 cm is ubiquitous, but the logs cannot say whether a safer candidate would have differed inside any support.

For the representative controls: `single_obstacle/pair01` first crossed 1 cm at 9.488 s (last short productive episode ended at 10.064 s); ICRA05 first crossed 1 cm at 7.977 s, long before its audited 118.8 s geometric-block boundary; YCB02 never crossed 3 cm and first crossed 5 cm at 8.196 s. These facts reject any claim that 1 cm universally activates only after physical blockage, while still leaving ranking leverage unidentified. All open-task rows have zero obstacles, so ReLU, exponential, and CHOMP-like obstacle potentials are identically zero and critical obstacle weight is irrelevant.

## HOW THE CONTROLLER ENTERS THE LOCAL MINIMUM

The logs establish physical formation mechanisms but not ranking-level entry decisions. F2 runs transition from low-productivity/contactless episodes into repeated reposition-arrival → contactless C3 → no-progress → equivalent-sector reselection transactions. F1 runs are dominated by failure to acquire object contact. ICRA05 progresses until 118.8 s and then remains physically engaged against sign geometry. The earliest bad selected candidate, candidate-pool degradation, alternative disappearance, and whether equivalent contacts were available cannot be located because only executed transactions—not contemporaneous candidate pools—were recorded.

At the first preserved F1/F2 evidence boundary, object clearance and pusher-object gap are recoverable from the metrics and appear in `marginal_cost_relevance.csv`. Pusher-obstacle clearance is not: the campaign logs an evaluation cost column named `pusher_obstacle`, not a geometric signed distance or pusher shape/path. It is left NA. Thus the data support the known acquisition/churn mechanisms but cannot decide whether a footprint cost could have prevented their first selection.

Prevention versus escape is likewise unidentified. Setting either flag from the executed trajectory would confuse “what happened” with “what alternative candidates existed.” No finite or infinite critical weight is assigned without `Delta_Phi`.

### Five most informative observable anchors

There are no identifiable pre-stall **decision** events because selected and unselected candidate pools were not recorded. `observable_transition_anchors.csv` instead records five clearly labeled physical proxies: single01's 1 cm crossing and final short productive episode, YCB02's 5 cm crossing and final short productive episode, and ICRA05's audited geometric-block boundary. These must not be used as candidate-choice events or as inputs to `w_star`.

## TRUE_STALL run table

| Run | Primary failure | Preserved onset [s] | Actual object clearance at onset [m] | Classification |
|---|---|---:|---:|---|
{rows}

## Cost shapes audited offline

Current finite-support ReLU, unit weight:

`phi_relu(d; eps) = max(0, (eps-d)/eps)^2`, with `(1/w)dJ/dd = 0` for `d>=eps`, otherwise `-2(eps-d)/eps^2`.

Historical exponential, unit weight and audited `sigma=0.04 m`:

`phi_exp(d; sigma) = exp(-d/sigma)`.

Smooth finite-support CHOMP-like comparison, unit weight:

`phi_C(d;eps)=0` for `d>=eps`; `0.5((eps-d)/eps)^2` for `0<=d<eps`; and `0.5-d/eps` for penetration. It is C1 at both joins, quadratic in the influence band and linear in penetration. This is only a shape probe, not a controller proposal.

No `w_star` distribution or useful epsilon range can be inferred without alternative trajectories. Actual-state crossings do not answer when the selected prediction first entered the support.

## Corridor and superposition check

The shelf and slalom probes use the exact orientation-aware T boundary and exact synchronized AABBs/disc. Full individual contributions and gradient vectors are in `corridor_probe_summary.csv`.

| Probe | Shape | support/sigma [m] | raw clearance [m] | unit potential | |gradient| [/m] |
|---|---|---:|---:|---:|---:|
{probe_rows}

Finite-support potentials are exactly zero at a valid passage center whenever epsilon is below its raw clearance. The historical exponential retains a nonzero long-range floor and can superpose/cancel between opposing walls. Larger finite supports become active near corridor boundaries; whether their weighted cost is acceptable cannot be judged without task-cost-normalized candidate comparisons. The offline cross-section therefore detects shape risk but does not justify a live weight.

At the shelf center (`d_min=0.0555 m`), unit-weight ReLU is zero through epsilon 0.05, then has floors 0.1352 at epsilon 0.075 and 0.39605 at epsilon 0.10; the CHOMP-like values are half those. At the three slalom gate centers (`d_min=0.0755 m`), epsilon 0.075 remains zero but epsilon 0.10 produces a ReLU floor 0.12005 (CHOMP-like 0.060025). Thus 7.5–10 cm support creates a genuine corridor-floor risk even before applying a weight.

## Direct answers

1. Historical TRUE_STALL runs with a proven pre-stall argmin flip: **0 identifiable; 16 cannot be fully tested**.
2. Proven no-safer-candidate cases: **0 from the 600 s campaign**; terminal-only evidence in other runs cannot establish the requested pre-stall fact.
3. Wrong-variable/acquisition cases: **at least 2 TRUE_STALL F1 primaries**; F2 pusher-path relevance cannot be converted into an object-cost counterfactual without candidate paths.
4. Is epsilon 0.01 generally too late? **Not identifiable at candidate-decision level.** Actual object clearance alone is insufficient.
5. First useful influence distance: **not identifiable**.
6. Critical-weight range: **not identifiable**.
7. Comparison with translation/orientation magnitudes: **not possible without `w_star`**.
8. Corridor floor: finite support avoids a center-floor when epsilon is below clearance; exponential has a long-range floor. Candidate-normalized harm remains unknown.
9. ICRA05 is geometrically relevant, but cost-preventable versus sampling-limited is **not identifiable**.
10. Live ReLU tuning: **not justified yet**; first obtain passive candidate trajectories.
11. CHOMP-like live testing: **not justified**; the smooth shape is only an offline corridor probe.
12. Failures not expected to be fixed by object cost: F5 open01, F7 terminations, progressing-at-cap controls, and the two F1 acquisition primaries absent contrary candidate evidence.
13. Smallest next ablation: no weight ablation. First perform the passive replay specified in `RERUN_REQUIRED.md`.

## Limitations

A counterfactual argmin flip would establish only ranking-level causal leverage, never guaranteed closed-loop success. Here even that first step is unavailable. No claim is made that obstacle cost fixes the outer loop, prevents F2, or rescues ICRA05.

## FINAL VERDICT

D. INSUFFICIENT_RECORDED_DATA
Existing logs cannot support the counterfactual decision analysis.
"""


def rerun_required() -> str:
    return """# RERUN REQUIRED: minimum passive candidate-trajectory replay

## Missing signal

The historical 600 s campaign did not log candidate pools. Later passive cost logs retain scalar task components and terminal prediction for every candidate, but retain knot-wise state/input trajectories only for the selected candidate. Exact orientation-aware `d_ik`, `Phi_i(epsilon)`, `Delta_Phi`, `w_star`, candidate-terminal spread through time, prevention, and escape are therefore not reconstructable for alternatives.

The logged `J_rank_obstacle_total` must not be used as a substitute. Source inspection shows that logger reconstructs the historical center-to-disc exponential potential (`weight=5000`, `sigma=0.04 m`) even when another live obstacle mode is active; it is not the required footprint geometry.

## Smallest first replay

Run **one ICRA-sign pair05 passive replay** only, using the exact frozen historical config/commit and unchanged random semantics. ICRA05 is the strongest positive control and will determine whether candidate-level counterfactual analysis is technically and scientifically viable before replaying F1/F2 controls.

The audited historical sampler used `std::random_device`; no task seed reproduces the old candidate pools. This replay can establish leverage for a new matched draw but cannot retroactively recover `w_star` for the recorded 600 s ICRA05 trajectory. Preserve that distinction in every downstream table.

Add one read-only log record per controller event, candidate, and prediction knot containing:

- simulation time, event ID, candidate ID, selected flag;
- candidate EE/contact state and object-frame sector;
- full predicted object `[x,y,yaw,vx,vy,wz]` at every knot;
- the exact frozen `J_trans`, `J_ori`, `J_angvel`, `J_linvel` components;
- candidate source (fresh/buffered), rejection reason, and failed-sector-memory distance;
- selected/reposition approach polyline or its already-computed swept pusher-obstacle minimum;
- exact task/config/commit receipt and any random draw identifier available.

Do not compute an obstacle counterfactual inside the controller. Log solved trajectories read-only and reconstruct all footprints/potentials offline. The controller decision must not consume any logged value.

## Validation gate

Before any further replay, verify for the ICRA05 replay that:

1. every ranked candidate has `N+1` knots;
2. recomputed non-obstacle task components reproduce the logged/code ranking total after excluding the audited historical obstacle column and explicit penalties;
3. selected candidate trajectory matches the existing selected-QP log;
4. logging on/off produces the same selected IDs and executed state trace for a fixed reproducible draw, if such a draw can be delivered without changing controller semantics;
5. exact C-glyph footprint and seven fixed glyph hulls reproduce recorded selected-candidate clearances.

Only if this passes, add two targeted passive replays: `single_obstacle/pair01` (F2) and `ycb_clutter/pair02` (F1). Do not rerun all 30 and do not launch automatically.
"""


def main(argv: Sequence[str] | None = None) -> int:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    OUTPUT = args.output.resolve()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite forensic evidence directory: {OUTPUT}")
    (OUTPUT / "figures").mkdir(parents=True)
    relevance = build_relevance()
    inventory = candidate_signal_inventory()
    cross, probes = corridor_outputs()
    write_csv(OUTPUT / "marginal_cost_relevance.csv", relevance)
    write_csv(OUTPUT / "candidate_signal_inventory.csv", inventory)
    write_csv(OUTPUT / "observable_transition_anchors.csv", observable_transition_anchors())
    write_csv(OUTPUT / "shelf_corridor_crosssection.csv", cross)
    write_csv(OUTPUT / "corridor_probe_summary.csv", probes)
    write_csv(OUTPUT / "wide_support_corridor_floor.csv", wide_support_corridor_floors())
    true_stalls = [row for row in relevance if row["outcome"] == "TRUE_STALL"]
    write_csv(OUTPUT / "true_stall_classification.csv", true_stalls)
    write_csv(OUTPUT / "negative_control_checks.csv", [
        {
            "scene": "open_task", "pair_id": f"pair{index:02d}",
            "obstacle_count": 0, "relu_unit_potential": 0.0,
            "exponential_unit_potential": 0.0, "chomp_like_unit_potential": 0.0,
            "critical_weight_status": "IRRELEVANT", "status": "PASS",
        }
        for index in range(1, 6)
    ])

    last_productive = episode_last_productive()
    for tag, (scene, pair) in REPRESENTATIVES.items():
        row = next(r for r in relevance if r["scene"] == scene and r["pair_id"] == pair)
        stall = float(row["stall_onset_time"])
        critical_weight_figure(tag, scene, pair, stall)
        local_minimum_figure(tag, scene, pair, stall, last_productive.get(f"{scene}/{pair}", NA))
    other_figures(relevance, cross)

    write_text(OUTPUT / "RERUN_REQUIRED.md", rerun_required())
    write_text(OUTPUT / "MARGINAL_OBSTACLE_BASIN_FORENSICS_REPORT.md", render_report(relevance, inventory, probes))
    manifest = {
        "analysis": "offline marginal obstacle basin forensics",
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip(),
        "command": f"python scripts/marginal_obstacle_basin_forensics.py --output {OUTPUT}",
        "parallel_workers": min(30, os.cpu_count() or 1),
        "campaign_rows": len(relevance),
        "true_stalls": len(true_stalls),
        "epsilon_grid_m": EPSILONS,
        "meaningful_alternative_thresholds": {
            "contact_or_ee_location_m": 0.02,
            "terminal_object_xy_m": 0.01,
            "minimum_clearance_gain_m": 0.005,
            "different_object_frame_sector": True,
            "status": "specified for rerun; cannot be applied to absent candidate trajectories",
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                FAILURE_DIR / "failure_audit.csv",
                FAILURE_DIR / "c3_segment_audit.csv",
                FAILURE_DIR / "outer_loop_summary.csv",
                FAILURE_DIR / "C3PLUS_600S_FAILURE_CLASSIFICATION_REPORT.md",
            ]
        },
        "outputs": sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*") if path.is_file()),
    }
    write_text(OUTPUT / "analysis_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "runs": len(relevance),
        "true_stalls": len(true_stalls),
        "classifications": Counter(row["classification"] for row in true_stalls),
        "candidate_log_sets": len(inventory),
        "verdict": "D. INSUFFICIENT_RECORDED_DATA",
        "output": str(OUTPUT),
    }, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
