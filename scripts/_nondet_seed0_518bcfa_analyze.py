#!/usr/bin/env python3
"""Bimodality + per-mode contact analysis for nondet_seed0_518bcfa runs.

Outputs: per-run [RESULT], gate counts, [DRAKE-CONTACT] n_pairs distribution,
admission-edge count, mode classification (median split + k=2 cluster check),
gate-firing x goal_dist correlation, and mode-gap vs prior-deltas table.
"""
from __future__ import annotations
import os, re, sys, statistics, math
from pathlib import Path

OUTBASE = Path("/root/push_anything_ADMM/nondet_seed0_518bcfa")
RUNS = [1, 2, 3, 4, 5, 6]

RE_RESULT = re.compile(r"^\[RESULT\] .*final_obj_xy=\(([^)]+)\)\s+goal_dist=([-0-9.]+)m")
RE_DRAKE = re.compile(r"^\[DRAKE-CONTACT\] step=(\d+) n_pairs=(\d+) ee_box_normal=([-+0-9.]+)")
RE_GS_C3 = re.compile(r"^\[GS\] step=(\d+) mode=c3")
RE_GS = re.compile(r"^\[GS\] step=(\d+)")
RE_GATE_PRE = re.compile(r"^\[GATE-COMMIT-FACE\] step=(\d+)")
RE_GATE_POST = re.compile(r"^\[GATE-COMMIT-FACE-POST\] step=(\d+)")
RE_STEP_FINISHED = re.compile(
    r"^\[STEP\] step=(\d+) .*finished_val=([0-9.]+)m finished_thresh=([0-9.]+)m"
)


def parse_run(idx):
    log = OUTBASE / f"run{idx}" / "run.log"
    if not log.exists():
        return None
    state = {
        "idx": idx, "goal_dist": None, "final_xy": None, "complete": False,
        "gate_pre": 0, "gate_post": 0,
        "gate_pre_first_step": None, "gate_post_first_step": None,
        "first_c3_step": None,
        "n_pairs_hist": {}, "drake_lines": 0,
        "admission_rises": 0, "admission_falls": 0,
        "first_n_pairs_ge2_step": None,
        # IK-finished gate trace around step ~110 (the bifurcation window)
        "finished_val_at_step": {},   # step -> finished_val
    }
    prev_np = -1
    with log.open("r", errors="replace") as f:
        for line in f:
            m = RE_RESULT.match(line)
            if m:
                state["final_xy"] = m.group(1).strip()
                state["goal_dist"] = float(m.group(2))
                state["complete"] = True
                continue
            m = RE_DRAKE.match(line)
            if m:
                step, n_pairs, eebn = int(m.group(1)), int(m.group(2)), float(m.group(3))
                state["n_pairs_hist"][n_pairs] = state["n_pairs_hist"].get(n_pairs, 0) + 1
                state["drake_lines"] += 1
                if prev_np >= 0:
                    if prev_np == 0 and n_pairs >= 1:
                        state["admission_rises"] += 1
                    if prev_np >= 1 and n_pairs == 0:
                        state["admission_falls"] += 1
                if n_pairs >= 2 and state["first_n_pairs_ge2_step"] is None:
                    state["first_n_pairs_ge2_step"] = step
                prev_np = n_pairs
                continue
            m = RE_GS_C3.match(line)
            if m and state["first_c3_step"] is None:
                state["first_c3_step"] = int(m.group(1))
                continue
            m = RE_GATE_PRE.match(line)
            if m:
                state["gate_pre"] += 1
                if state["gate_pre_first_step"] is None:
                    state["gate_pre_first_step"] = int(m.group(1))
                continue
            m = RE_GATE_POST.match(line)
            if m:
                state["gate_post"] += 1
                if state["gate_post_first_step"] is None:
                    state["gate_post_first_step"] = int(m.group(1))
                continue
            m = RE_STEP_FINISHED.match(line)
            if m:
                step = int(m.group(1))
                if 100 <= step <= 130:
                    state["finished_val_at_step"][step] = float(m.group(2))
                continue
    return state


def fmt_state(s):
    if s is None:
        return "MISSING"
    if not s["complete"]:
        return "INCOMPLETE (no [RESULT])"
    return (
        f"run{s['idx']}: gd={s['goal_dist']:.4f} obj=({s['final_xy']}) "
        f"gate_pre={s['gate_pre']} (first@{s['gate_pre_first_step']}) "
        f"gate_post={s['gate_post']} (first@{s['gate_post_first_step']}) "
        f"drake_lines={s['drake_lines']} "
        f"n_pairs_hist={dict(sorted(s['n_pairs_hist'].items()))} "
        f"rises={s['admission_rises']} falls={s['admission_falls']} "
        f"first_n_pairs_ge2@{s['first_n_pairs_ge2_step']} "
        f"first_c3@{s['first_c3_step']}"
    )


def classify_modes(states):
    completed = [s for s in states if s and s["complete"]]
    if len(completed) < 2:
        return {}, 0.0
    gd = sorted(s["goal_dist"] for s in completed)
    # Median split as the mode boundary
    mid = (gd[len(gd) // 2] + gd[(len(gd) - 1) // 2]) / 2.0
    classes = {}
    for s in completed:
        cls = "q7-mode" if s["goal_dist"] < mid else "baseline-mode"
        classes[s["idx"]] = cls
    mode_gap = max(gd) - min(gd)
    return classes, mode_gap


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    states = [parse_run(i) for i in RUNS]
    print("=== Per-run summary ===")
    for s in states:
        print(fmt_state(s))
    print()

    completed = [s for s in states if s and s["complete"]]
    if not completed:
        print("No completed runs yet.")
        return

    classes, mode_gap = classify_modes(states)
    print(f"=== Mode classification (median split) ===")
    print(f"mode_gap (max-min goal_dist): {mode_gap:.4f}m")
    n_q7 = sum(1 for c in classes.values() if c == "q7-mode")
    n_bl = sum(1 for c in classes.values() if c == "baseline-mode")
    print(f"q7-mode count: {n_q7}/{len(completed)}")
    print(f"baseline-mode count: {n_bl}/{len(completed)}")
    for s in completed:
        print(f"  run{s['idx']}: gd={s['goal_dist']:.4f} -> {classes[s['idx']]}")
    print()

    print("=== Gate-firing x goal_dist correlation ===")
    xs_gd = [s["goal_dist"] for s in completed]
    xs_pre = [s["gate_pre"] for s in completed]
    xs_post = [s["gate_post"] for s in completed]
    xs_tot = [s["gate_pre"] + s["gate_post"] for s in completed]
    r_pre = pearson(xs_pre, xs_gd)
    r_post = pearson(xs_post, xs_gd)
    r_tot = pearson(xs_tot, xs_gd)
    print(f"  r(gate_pre, goal_dist)  = {r_pre:+.4f}")
    print(f"  r(gate_post, goal_dist) = {r_post:+.4f}")
    print(f"  r(gate_total, goal_dist) = {r_tot:+.4f}")
    print()

    print("=== Per-mode [DRAKE-CONTACT] aggregates ===")
    for mode in ("q7-mode", "baseline-mode"):
        members = [s for s in completed if classes[s["idx"]] == mode]
        if not members:
            print(f"  {mode}: (no members)"); continue
        n = len(members)
        # Sum histograms
        agg = {}
        for s in members:
            for k, v in s["n_pairs_hist"].items():
                agg[k] = agg.get(k, 0) + v
        avg_rises = sum(s["admission_rises"] for s in members) / n
        avg_falls = sum(s["admission_falls"] for s in members) / n
        avg_np2 = sum(s["n_pairs_hist"].get(2, 0) for s in members) / n
        avg_np3 = sum(s["n_pairs_hist"].get(3, 0) for s in members) / n
        avg_np4 = sum(s["n_pairs_hist"].get(4, 0) for s in members) / n
        firsts = [s["first_n_pairs_ge2_step"] for s in members if s["first_n_pairs_ge2_step"] is not None]
        print(f"  {mode} (n={n}):")
        print(f"    aggregate n_pairs_hist: {dict(sorted(agg.items()))}")
        print(f"    avg n_pairs=2: {avg_np2:.1f}  n_pairs=3: {avg_np3:.1f}  n_pairs=4: {avg_np4:.1f}")
        print(f"    avg admission rises: {avg_rises:.1f}  falls: {avg_falls:.1f}")
        print(f"    first_n_pairs>=2 step (min/max/median): "
              f"{min(firsts) if firsts else '-'}/{max(firsts) if firsts else '-'}/"
              f"{statistics.median(firsts) if firsts else '-'}")
    print()

    print("=== Bifurcation-step (~110) IK-finished gate FP-flicker check ===")
    print("  finished_thresh = 0.0200 m (reposition_ik.py:1299)")
    print("  Steps shown: 105..115 (just before/at/after the bifurcation)")
    header_cols = ["run{0}_fv".format(s["idx"]) for s in completed]
    print("  Header: step | " + " | ".join(header_cols))
    for step in range(105, 116):
        row = [f"  {step}"]
        for s in completed:
            v = s["finished_val_at_step"].get(step)
            row.append(f"{v*1000:6.2f}mm" if v is not None else "  -   ")
        print("  | ".join(row))
    print(f"  first_c3_step per run: " + ", ".join(f"run{s['idx']}={s['first_c3_step']}" for s in completed))
    print("  Interpretation: at the threshold, ADMM/IK-Ipopt FP noise tips one side vs the")
    print("  other across runs -> different repos->c3 entry step -> mode bifurcation.")
    print()

    print("=== Prior single-seed deltas vs mode gap (absolute mm) ===")
    print(f"  mode_gap observed: {mode_gap*1000:.1f}mm")
    # (label, delta_m, sign_convention) — delta_m signed; positive = improvement (goal_dist down)
    prior_deltas = [
        ("Q5 seed-0 c8402b3 vs fa8db2b (0.263 -> 0.235)",   +0.028),
        ("Q5 seed-2 c8402b3 vs fa8db2b (0.179 -> 0.139)",   +0.040),
        ("Q5 seed-4 c8402b3 vs fa8db2b (0.247 -> 0.145)",   +0.102),
        ("Q7 seed-0 0.177 vs conformance 0.2024",           +0.0254),
        ("Q7 seed-2 0.345 vs baseline 0.179 (REGRESSION)",  -0.166),
        ("Q7 seed-4 0.227 vs baseline 0.247",               +0.020),
        ("Q7 seed-4 0.227 vs c8402b3 0.145 (REGRESSION)",   -0.082),
    ]
    print("  delta_name                                              size      survives_mode_gap?")
    for name, size_m in prior_deltas:
        survives = abs(size_m) > mode_gap
        print(f"    {name:<54s} {size_m*1000:+6.1f}mm  {'YES (likely real)' if survives else 'NO (possible mode artifact)'}")
    print()
    print("  Interpretation:")
    print(f"  - Deltas with |size| <= mode_gap ({mode_gap*1000:.1f}mm) are inseparable from mode-occupancy noise")
    print(f"    at seed=0 (the same-seed bimodality observed here).")
    print(f"  - Multi-seed deltas (Q5) are computed as 1-vs-1 per seed; if EACH seed has its")
    print(f"    own bimodality, each delta is also a mode-pair compared at fixed seed.")
    print()


if __name__ == "__main__":
    main()
