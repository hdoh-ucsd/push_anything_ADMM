#!/usr/bin/env python3
"""Parse a push_anything log file into per-step JSONL telemetry.

Usage:
    python parse_log_to_jsonl.py <input.log> [output.jsonl]

Produces one JSON object per simulation step with fields:
    step              int
    sim_t             float (seconds; step * dt_ctrl, default 0.01)
    mode              "free" | "c3" | null
    switch_reason     str | null
    ee_pos            [x, y, z] | null   (latest known)
    obj_pos           [x, y] | null      (latest known)
    p_repos           [x, y, z] | null
    ee_to_target      float | null        (legacy [GS-tgt] field)
    ee_stride         float | null        ([STEP] free: EE→next-knot stride)
    ee_to_ptarget     float | null        ([STEP] free: EE→p_target goal gap)
    ee_to_box         float | null
    g_hat             [x, y] | null      (goal direction)
    contact_active    bool
    contact_normal    [x, y, z] | null   (when contact active)
    contact_distance  float | null
    n_lambda          int | null          (n_λ from C3+ line)
    lambda_n_max      float | null
    u_arm_norm        float | null        (|u[0]| from C3+ line)
    tau_imp           float | null        (impedance feedback torque norm)
    tau_ff            float | null        (impedance feedforward torque norm)
    x_err             float | null        (Cartesian error norm)
    osc_saturated     bool | null
    curr_cost         float | null
    repos_cost        float | null
    best_other_cost   float | null
    attribution       "planned_productive" | "planned_unproductive"
                    | "accidental_contact" | "no_contact_free"
                    | null
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# Regex patterns for each significant log line type.
RE_GS_STEP = re.compile(
    r"^\[GS\] step=(?P<step>\d+) mode=(?P<mode>\w+) switch=(?P<switch>\w+)"
    r" .*? curr_cost=(?P<curr_cost>[\d.]+) repos_cost=(?P<repos_cost>[\d.\-]+)"
    r" best_other=(?P<best_other>[\d.]+)"
)
RE_GS_TGT = re.compile(
    r"^\[GS-tgt\] step=(?P<step>\d+) ee=\((?P<ex>[+\-\d.]+),(?P<ey>[+\-\d.]+),(?P<ez>[+\-\d.]+)\)"
    r" p_repos=\((?P<rx>[+\-\d.]+),(?P<ry>[+\-\d.]+),(?P<rz>[+\-\d.]+)\)"
    r" ee_to_target=(?P<dist>[\d.]+)m target_label=(?P<label>\S+)"
    r"(?: target_changed=(?P<changed>[YN]))?"
)
RE_C3PLUS_FULL = re.compile(
    r"^\[C3\+\] step=(?P<step>\d+) \|u\[0\]\|=(?P<u>[\d.]+)Nm"
    r" λ_n_max=(?P<lam_n_max>[\d.+\-e]+)"
)
RE_C3PLUS_NOFORCE = re.compile(
    r"^\[C3\+\] step=(?P<step>\d+) n_λ=(?P<n_lam>\d+)\s+\|u\[0\]\|=(?P<u>[\d.]+) Nm"
)
RE_CONTACT_RUN = re.compile(
    r"^\[CONTACT-RUN\] step=(?P<step>\d+) nhat_BA_W=\[(?P<nx>[+\-\d.]+),(?P<ny>[+\-\d.]+),(?P<nz>[+\-\d.]+)\]"
    r" .*? distance=(?P<dist>[+\-\d.]+)"
)
RE_EEREL = re.compile(
    r"^\[EErel\] along_push=(?P<along>[+\-\d.]+)m"
    r".*? obj=\[(?P<ox>[+\-\d.eE]+)\s+(?P<oy>[+\-\d.eE]+)\]"
    r"\s+g_hat=\[(?P<gx>[+\-\d.eE]+)\s+(?P<gy>[+\-\d.eE]+)\]"
)
RE_EEREL_BOX = re.compile(r"ee_to_box=(?P<ee_to_box>[\d.]+)m")
RE_IMP = re.compile(
    r"^\[IMP\] step=(?P<step>\d+) mode=(?P<mode>\w+) \|x_err\|=(?P<xerr>[\d.]+)m"
    r" \|tau_imp\|=(?P<tau_imp>[\d.]+)Nm \|tau_ff\|=(?P<tau_ff>[\d.]+)Nm"
    r" \|tau_out\|=(?P<tau_out>[\d.]+)Nm sat=(?P<sat>\w+)"
)
RE_TIME_SNAPSHOT = re.compile(
    r"^\s+t=(?P<t>[\d.]+)s \| ee=\((?P<ex>[+\-\d.]+), (?P<ey>[+\-\d.]+), (?P<ez>[+\-\d.]+)\)"
    r" \| obj=\((?P<ox>[+\-\d.]+), (?P<oy>[+\-\d.]+), (?P<oz>[+\-\d.]+)\)"
)
RE_DT = re.compile(r"\[MPC\]\s+Horizon:.*?dt:\s*(?P<dt>[\d.]+)\s*s")

# OSC's per-step line if present (older runs use [IMP], newer may use [OSC]).
RE_OSC = re.compile(
    r"^\[OSC\] step=(?P<step>\d+) mode=(?P<mode>\w+) \|x_err\|=(?P<xerr>[\d.]+)m"
    r".*?sat=(?P<sat>\w+)"
)

# Goal coordinates from [ENV] line.
RE_GOAL = re.compile(r"\[ENV\]\s+Goal coords:\s*\[(?P<gx>[+\-\d.]+),\s*(?P<gy>[+\-\d.]+)\]")

# Unified [STEP] line — common prefix (always present).
RE_STEP_PREFIX = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) mode=(?P<mode>\w+) t=(?P<t>[\d.]+)s"
    r" ee=\((?P<ex>[+\-\d.]+),(?P<ey>[+\-\d.]+),(?P<ez>[+\-\d.]+)\)"
    r" obj=\((?P<ox>[+\-\d.]+),(?P<oy>[+\-\d.]+),(?P<oz>[+\-\d.]+)\)"
    r" goal_dist=(?P<goal_dist>[\d.]+)m switch=(?P<switch>\w+)"
)
# Free-mode payload appended after the prefix.
RE_STEP_FREE = re.compile(
    r" target=\((?P<tx>[+\-\d.]+),(?P<ty>[+\-\d.]+),(?P<tz>[+\-\d.]+)\)"
    r" ee_stride=(?P<ee_stride>[\d.]+)m"
    r" ee_to_ptarget=(?P<ee_to_ptarget>[\d.nNaA]+)m"
    r" landing_err=(?P<landing>[\d.nNaA]+)m"
    r" ik_ok=(?P<ik_ok>[YN?])"
    r" ik_resid=(?P<ik_resid>[\d.nNaA]+)m"
    r" pd_sat=(?P<pd_sat>[YN?])"
    r" q_max_resid_deg=(?P<qmax>[\-\d.nNaA]+)"
    r" target_changed=(?P<changed>[YN])"
)
# Why-replan payload — sub-pattern, appended after RE_STEP_FREE on free-mode lines.
RE_STEP_RETGT = re.compile(
    r" retgt=(?P<retgt>[YN?])"
    r" held_idx=(?P<held_idx>[\-\d?]+)"
    r" held_cost=(?P<held_cost>[\-\d.?]+)"
    r" won_cost=(?P<won_cost>[\-\d.]+)"
    r" margin=(?P<margin>[\+\-\d.?]+)"
    r" won_src=(?P<won_src>[\w_?\-]+)"
    r" held_valid=(?P<held_valid>[YN?])"
    r" reason=(?P<reason>[\w_]+)"
)
# C3-mode payload appended after the prefix.
RE_STEP_C3 = re.compile(
    r" c3_cost=(?P<cost>[\d.]+)"
    r" lam_n=(?P<lam_n>[\d.\-]+)"
    r" lam_t=(?P<lam_t>[\d.\-]+)"
    r" contact=(?P<contact>[YN])"
    r" productive=(?P<productive>[YN])"
    r" f_cmd=\((?P<fx>[+\-\d.]+),(?P<fy>[+\-\d.]+),(?P<fz>[+\-\d.]+)\)"
)


def _maybe_float(s: str) -> float | None:
    """Parse a float, tolerating 'nan' tokens."""
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_log(log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a log file into (metadata, list_of_per_step_records).

    Strategy:
        1. First pass: read all lines, group by 'step' where step is parseable.
        2. Forward-fill carry-over fields (ee_pos, obj_pos, g_hat) when missing
           on a given step.
        3. Compute attribution at the end.
    """
    text = log_path.read_text(errors="replace")

    meta: dict[str, Any] = {
        "file": str(log_path),
        "goal": None,
        "dt": 0.01,  # default per [MPC]
    }
    per_step: dict[int, dict[str, Any]] = {}

    # Pre-scan for metadata.
    for line in text.splitlines():
        if m := RE_GOAL.search(line):
            meta["goal"] = [float(m["gx"]), float(m["gy"])]
        if m := RE_DT.search(line):
            # Note: MPC dt is planning horizon dt (e.g. 0.05),
            # the control loop dt is typically 0.01.
            pass
    # Use 0.01 as control dt (sim dt for per-step granularity)
    # since [GS] step=N is per-control-loop.
    meta["dt_ctrl"] = 0.01

    # Track the most recent values to forward-fill.
    latest_ee = None
    latest_obj = None
    latest_g_hat = None

    # Walk the log linearly. Keep a 'current_step' anchor, updated from
    # [GS] step=N lines (the per-MPC-step record). All other per-step data
    # ([CONTACT-RUN], [C3+], [IMP], [EErel]) gets attached to the most
    # recently anchored step. We deliberately DON'T trust the step numbers
    # on [C3+], [CONTACT-RUN] etc. because they may use different counters
    # (e.g., [C3+] step numbers increment by 3 due to MPC horizon).
    current_step: int | None = None
    for line in text.splitlines():
        # [GS] step=N — the anchor for all subsequent step-scoped data.
        if m := RE_GS_STEP.match(line):
            step = int(m["step"])
            current_step = step
            rec = per_step.setdefault(step, {"step": step})
            rec["mode"] = m["mode"]
            rec["switch_reason"] = m["switch"]
            rec["curr_cost"] = float(m["curr_cost"])
            # repos_cost may be "-" for c3 mode (no repos cost computed)
            rc = m["repos_cost"]
            rec["repos_cost"] = float(rc) if rc not in ("-", "") else None
            rec["best_other_cost"] = float(m["best_other"])
            continue

        # [GS-tgt] step=N ee=... p_repos=...
        if m := RE_GS_TGT.match(line):
            step = int(m["step"])
            current_step = step
            rec = per_step.setdefault(step, {"step": step})
            ee = [float(m["ex"]), float(m["ey"]), float(m["ez"])]
            rec["ee_pos"] = ee
            rec["p_repos"] = [float(m["rx"]), float(m["ry"]), float(m["rz"])]
            rec["ee_to_target"] = float(m["dist"])
            rec["target_label"] = m["label"]
            changed = m.groupdict().get("changed")
            if changed is not None:
                rec["target_changed"] = (changed == "Y")
            latest_ee = ee
            continue

        # [STEP] unified per-step line (additive — sits alongside [GS]/[GS-tgt]).
        # Prefix is always present; the suffix is mode-aware.
        if m := RE_STEP_PREFIX.match(line):
            step = int(m["step"])
            current_step = step
            rec = per_step.setdefault(step, {"step": step})
            rec.setdefault("mode", m["mode"])
            ee = [float(m["ex"]), float(m["ey"]), float(m["ez"])]
            obj = [float(m["ox"]), float(m["oy"]), float(m["oz"])]
            rec["ee_pos"] = ee
            rec["obj_pos"] = obj[:2]
            rec["obj_z"] = obj[2]
            rec["goal_dist"] = float(m["goal_dist"])
            rec.setdefault("switch_reason", m["switch"])
            latest_ee = ee
            latest_obj = obj[:2]
            # Try the free-mode suffix.
            m_free = RE_STEP_FREE.search(line, pos=m.end())
            if m_free is not None:
                rec["p_repos"] = [
                    float(m_free["tx"]), float(m_free["ty"]), float(m_free["tz"])]
                rec["ee_stride"]     = float(m_free["ee_stride"])
                rec["ee_to_ptarget"] = _maybe_float(m_free["ee_to_ptarget"])
                rec["landing_err"]   = _maybe_float(m_free["landing"])
                rec["ik_ok"]         = (m_free["ik_ok"] == "Y")
                rec["ik_resid"]      = _maybe_float(m_free["ik_resid"])
                rec["pd_sat"]        = (m_free["pd_sat"] == "Y")
                rec["q_max_resid_deg"] = _maybe_float(m_free["qmax"])
                rec["target_changed"] = (m_free["changed"] == "Y")
                # Why-replan payload (always present on free-mode lines).
                m_rt = RE_STEP_RETGT.search(line, pos=m_free.end())
                if m_rt is not None:
                    rec["retgt"]      = (m_rt["retgt"] == "Y")
                    rec["held_idx"]   = (int(m_rt["held_idx"])
                                         if m_rt["held_idx"] not in ("-", "?")
                                         else None)
                    rec["held_cost"]  = _maybe_float(m_rt["held_cost"])
                    rec["won_cost"]   = _maybe_float(m_rt["won_cost"])
                    rec["margin"]     = _maybe_float(m_rt["margin"])
                    rec["won_src"]    = m_rt["won_src"]
                    rec["held_valid"] = (m_rt["held_valid"] == "Y")
                    rec["replan_reason"] = m_rt["reason"]
            # Try the c3-mode suffix.
            m_c3 = RE_STEP_C3.search(line, pos=m.end())
            if m_c3 is not None:
                rec["c3_cost"]      = float(m_c3["cost"])
                rec["lam_n"]        = float(m_c3["lam_n"])
                rec["lam_t"]        = float(m_c3["lam_t"])
                rec["contact_active"] = (m_c3["contact"] == "Y")
                rec["productive"]   = (m_c3["productive"] == "Y")
                rec["f_cmd"]        = [
                    float(m_c3["fx"]), float(m_c3["fy"]), float(m_c3["fz"])]
            continue

        # [C3+] with full force info — attach to current_step (ignore C3+ step num)
        if m := RE_C3PLUS_FULL.match(line):
            if current_step is not None:
                rec = per_step.setdefault(current_step, {"step": current_step})
                rec["u_arm_norm"] = float(m["u"])
                rec["lambda_n_max"] = float(m["lam_n_max"])
            continue

        # [C3+] with no force (n_λ=0)
        if m := RE_C3PLUS_NOFORCE.match(line):
            if current_step is not None:
                rec = per_step.setdefault(current_step, {"step": current_step})
                rec["u_arm_norm"] = float(m["u"])
                rec["n_lambda"] = int(m["n_lam"])
                rec.setdefault("lambda_n_max", 0.0)
            continue

        # [CONTACT-RUN] step=N — has its own step number that matches GS step
        if m := RE_CONTACT_RUN.match(line):
            step = int(m["step"])  # contact-run step does match GS step
            rec = per_step.setdefault(step, {"step": step})
            rec["contact_active"] = True
            rec["contact_normal"] = [float(m["nx"]), float(m["ny"]), float(m["nz"])]
            rec["contact_distance"] = float(m["dist"])
            continue

        # [EErel] — has obj and g_hat, attach to current_step
        if m := RE_EEREL.match(line):
            obj = [float(m["ox"]), float(m["oy"])]
            g_hat = [float(m["gx"]), float(m["gy"])]
            ee_to_box = None
            if m2 := RE_EEREL_BOX.search(line):
                ee_to_box = float(m2["ee_to_box"])
            latest_obj = obj
            latest_g_hat = g_hat
            if current_step is not None:
                rec = per_step.setdefault(current_step, {"step": current_step})
                rec.setdefault("obj_pos", obj)
                rec.setdefault("g_hat", g_hat)
                if ee_to_box is not None:
                    rec.setdefault("ee_to_box", ee_to_box)
            continue

        # [IMP] step=N — per-step impedance controller info
        if m := RE_IMP.match(line):
            step = int(m["step"])
            current_step = step
            rec = per_step.setdefault(step, {"step": step})
            rec.setdefault("mode", m["mode"])
            rec["x_err"] = float(m["xerr"])
            rec["tau_imp"] = float(m["tau_imp"])
            rec["tau_ff"] = float(m["tau_ff"])
            rec["osc_saturated"] = m["sat"].lower() == "true"
            continue

        # [OSC] step=N — newer executor output
        if m := RE_OSC.match(line):
            step = int(m["step"])
            current_step = step
            rec = per_step.setdefault(step, {"step": step})
            rec.setdefault("mode", m["mode"])
            rec["x_err"] = float(m["xerr"])
            rec["osc_saturated"] = m["sat"].lower() == "true"
            continue

        # `  t=X.XXs | ee=... | obj=...` — snapshot lines (every 0.5s usually)
        if m := RE_TIME_SNAPSHOT.match(line):
            # Snap obj/ee for forward-fill
            latest_ee = [float(m["ex"]), float(m["ey"]), float(m["ez"])]
            latest_obj = [float(m["ox"]), float(m["oy"])]
            continue

    # Drop any step records that have step > some sane upper bound from C3+
    # noise (in case the older code path created step=709 entries). Use the
    # max step from a [GS] line as the upper bound.
    valid_steps = [s for s in per_step.keys()]
    if valid_steps:
        max_gs_step = max(
            s for s, r in per_step.items()
            if "mode" in r and "switch_reason" in r
        )
        per_step = {s: r for s, r in per_step.items() if s <= max_gs_step}

    steps_sorted = sorted(per_step.keys())
    if not steps_sorted:
        return meta, []

    # Forward-fill ee_pos, obj_pos, g_hat for steps that lack them.
    prev_ee = None
    prev_obj = None
    prev_g_hat = None
    for step in steps_sorted:
        rec = per_step[step]
        if rec.get("ee_pos") is None and prev_ee is not None:
            rec["ee_pos"] = prev_ee
        else:
            prev_ee = rec.get("ee_pos", prev_ee)
        if rec.get("obj_pos") is None and prev_obj is not None:
            rec["obj_pos"] = prev_obj
        else:
            prev_obj = rec.get("obj_pos", prev_obj)
        if rec.get("g_hat") is None and prev_g_hat is not None:
            rec["g_hat"] = prev_g_hat
        else:
            prev_g_hat = rec.get("g_hat", prev_g_hat)

    # Compute attribution per step.
    records: list[dict[str, Any]] = []
    dt = meta["dt_ctrl"]
    for step in steps_sorted:
        rec = per_step[step]
        rec.setdefault("contact_active", False)
        rec.setdefault("contact_normal", None)
        rec["sim_t"] = step * dt

        mode = rec.get("mode")
        contact = rec.get("contact_active", False)
        lam_n = rec.get("lambda_n_max", 0.0)
        normal = rec.get("contact_normal")
        g_hat = rec.get("g_hat")

        # Productive direction: contact normal · (-g_hat) > 0 means the EE
        # is on the side of the box that pushes it toward the goal.
        productive_direction = None
        if normal is not None and g_hat is not None:
            # Contact normal in 3D, g_hat in 2D — use xy only.
            nx, ny = normal[0], normal[1]
            # Goal direction is -g_hat (since g_hat points away from goal in
            # the [EErel] convention: g_hat = -obj_direction-of-goal).
            # Actually, from EErel: g_hat=[-1, 0] for West task. Goal is at
            # x=-0.3, so g_hat=[-1,0] points TOWARD the goal.
            # To push box toward goal, contact_normal · g_hat should be > 0
            # (force into box is along normal; to move box toward goal, force
            # direction should be g_hat).
            # nhat_BA_W points from B to A (from box to EE). Force on box is
            # -nhat (push into box). To move box toward goal direction g_hat,
            # we want -nhat aligned with g_hat, i.e. nhat anti-aligned with
            # g_hat, i.e. nhat · g_hat < 0.
            dot = nx * g_hat[0] + ny * g_hat[1]
            productive_direction = dot < -0.3  # threshold for "well-aligned"

        if contact and mode == "c3" and lam_n > 0.5 and productive_direction:
            attr = "planned_productive"
        elif contact and mode == "c3":
            attr = "planned_unproductive"
        elif contact and mode == "free":
            attr = "accidental_contact"
        else:
            attr = "no_contact_free"
        rec["attribution"] = attr

        # Tidy record for output
        out = {
            "step": rec["step"],
            "sim_t": round(rec["sim_t"], 4),
            "mode": rec.get("mode"),
            "switch_reason": rec.get("switch_reason"),
            "ee_pos": rec.get("ee_pos"),
            "obj_pos": rec.get("obj_pos"),
            "p_repos": rec.get("p_repos"),
            "ee_to_target": rec.get("ee_to_target"),  # legacy [GS-tgt] field
            "ee_stride": rec.get("ee_stride"),
            "ee_to_ptarget": rec.get("ee_to_ptarget"),
            "target_label": rec.get("target_label"),
            "target_changed": rec.get("target_changed"),
            "ee_to_box": rec.get("ee_to_box"),
            "g_hat": rec.get("g_hat"),
            "contact_active": rec.get("contact_active", False),
            "contact_normal": rec.get("contact_normal"),
            "contact_distance": rec.get("contact_distance"),
            "n_lambda": rec.get("n_lambda"),
            "lambda_n_max": rec.get("lambda_n_max"),
            "u_arm_norm": rec.get("u_arm_norm"),
            "tau_imp": rec.get("tau_imp"),
            "tau_ff": rec.get("tau_ff"),
            "x_err": rec.get("x_err"),
            "osc_saturated": rec.get("osc_saturated"),
            "curr_cost": rec.get("curr_cost"),
            "repos_cost": rec.get("repos_cost"),
            "best_other_cost": rec.get("best_other_cost"),
            "attribution": rec.get("attribution"),
            # Unified [STEP] payload — free-mode reposition diagnostics.
            "landing_err": rec.get("landing_err"),
            "ik_ok": rec.get("ik_ok"),
            "ik_resid": rec.get("ik_resid"),
            "pd_sat": rec.get("pd_sat"),
            "q_max_resid_deg": rec.get("q_max_resid_deg"),
            "goal_dist": rec.get("goal_dist"),
            # Why-replan payload.
            "retgt": rec.get("retgt"),
            "held_idx": rec.get("held_idx"),
            "held_cost": rec.get("held_cost"),
            "won_cost": rec.get("won_cost"),
            "margin": rec.get("margin"),
            "won_src": rec.get("won_src"),
            "held_valid": rec.get("held_valid"),
            "replan_reason": rec.get("replan_reason"),
            # Unified [STEP] payload — c3-mode contact diagnostics.
            "c3_cost": rec.get("c3_cost"),
            "lam_n": rec.get("lam_n"),
            "lam_t": rec.get("lam_t"),
            "productive": rec.get("productive"),
            "f_cmd": rec.get("f_cmd"),
        }
        records.append(out)

    return meta, records


def attribution_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary attribution stats from records."""
    by_cat: dict[str, int] = {
        "planned_productive": 0,
        "planned_unproductive": 0,
        "accidental_contact": 0,
        "no_contact_free": 0,
    }
    motion_by_cat: dict[str, float] = {k: 0.0 for k in by_cat}

    prev_obj = None
    g_hat = None

    for rec in records:
        cat = rec.get("attribution") or "no_contact_free"
        by_cat[cat] = by_cat.get(cat, 0) + 1
        obj = rec.get("obj_pos")
        gh = rec.get("g_hat")
        if gh is not None:
            g_hat = gh
        if obj is not None and prev_obj is not None and g_hat is not None:
            # West-progress increment = -(dx) for west task (g_hat = [-1,0]).
            dx = obj[0] - prev_obj[0]
            dy = obj[1] - prev_obj[1]
            # Project displacement onto g_hat direction.
            progress = dx * g_hat[0] + dy * g_hat[1]
            motion_by_cat[cat] = motion_by_cat.get(cat, 0.0) + progress
        if obj is not None:
            prev_obj = obj

    return {
        "step_counts": by_cat,
        "goal_progress_mm_by_category": {
            k: round(v * 1000, 3) for k, v in motion_by_cat.items()
        },
        "total_steps": sum(by_cat.values()),
        "total_progress_mm": round(sum(motion_by_cat.values()) * 1000, 3),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_log_to_jsonl.py <input.log> [output.jsonl]")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else in_path.with_suffix(".jsonl")
    )
    meta, records = parse_log(in_path)
    summary = attribution_summary(records)

    # Write metadata + records (JSONL: one JSON object per line)
    with out_path.open("w") as f:
        f.write(json.dumps({"_meta": meta, "_summary": summary}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} step records to {out_path}")
    print(f"Metadata: {json.dumps(meta, indent=2)}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
