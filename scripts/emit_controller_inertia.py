#!/usr/bin/env python3
"""Emit per-task `controller_inertia` into config/tasks.yaml from each
imported object's *_controller.sdf <inertial> block (2026-08-15 planner-
inertia conformance — the reference plans with the DECLARED tensor, which
is not proportional to the sim model's).

Idempotent: re-running replaces existing controller_inertia blocks.
Only touches tasks that have `object_sdf` AND a sibling *_controller.sdf.
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "config" / "tasks.yaml"


def parse_inertial(sdf_path: Path):
    txt = sdf_path.read_text()
    m = re.search(r"<inertial>(.*?)</inertial>", txt, re.S)
    if not m:
        return None
    blk = m.group(1)

    def val(tag, default=None):
        mm = re.search(rf"<{tag}>([^<]+)</{tag}>", blk)
        return float(mm.group(1)) if mm else default

    pose = re.search(r"<pose>([^<]+)</pose>", blk)
    com = [0.0, 0.0, 0.0]
    if pose:
        parts = [float(v) for v in pose.group(1).split()]
        com = parts[:3]
        if any(abs(v) > 1e-12 for v in parts[3:6]):
            raise SystemExit(f"{sdf_path}: rotated inertial pose unsupported")
    return dict(
        mass=val("mass"),
        com=com,
        moments=[val("ixx"), val("iyy"), val("izz")],
        products=[val("ixy", 0.0), val("ixz", 0.0), val("iyz", 0.0)],
    )


def main():
    raw = yaml.safe_load(TASKS.read_text())
    tasks = raw["tasks"]
    lines = TASKS.read_text().splitlines(keepends=True)

    updates = {}
    for name, blk in tasks.items():
        sdf = blk.get("object_sdf")
        if not sdf:
            continue
        ctrl = ROOT / str(sdf).replace(".sdf", "_controller.sdf")
        if not ctrl.exists():
            print(f"  [skip] {name}: no {ctrl.name}")
            continue
        inr = parse_inertial(ctrl)
        if inr is None or inr["mass"] is None or None in inr["moments"]:
            print(f"  [skip] {name}: unparsable inertial in {ctrl.name}")
            continue
        updates[name] = inr

    # Textual splice: after each task's `controller_mass:` line (or, if
    # absent, after `object_sdf:`), insert/replace the controller_inertia
    # mapping at the same indent.
    out = []
    i = 0
    cur_task = None
    task_hdr = re.compile(r"^  (\w[\w-]*):\s*(#.*)?$")
    while i < len(lines):
        ln = lines[i]
        mh = task_hdr.match(ln)
        if mh:
            cur_task = mh.group(1)
        if (cur_task in updates
                and re.match(r"^    controller_inertia:", ln)):
            # drop existing block (this line + following deeper-indented)
            i += 1
            while i < len(lines) and re.match(r"^(      |\t)", lines[i]):
                i += 1
            continue
        out.append(ln)
        if (cur_task in updates
                and re.match(r"^    controller_mass:", ln)):
            inr = updates.pop(cur_task)
            mo = inr["moments"]
            pr = inr["products"]
            com = inr["com"]
            out.append("    # Declared planner tensor from the reference "
                       "*_controller.sdf <inertial>\n")
            out.append("    # (NOT proportional to the sim tensor — see "
                       "scripts/emit_controller_inertia.py).\n")
            out.append(f"    controller_inertia:\n")
            out.append(f"      mass: {inr['mass']}\n")
            out.append(f"      com: [{com[0]}, {com[1]}, {com[2]}]\n")
            out.append(f"      moments: [{mo[0]}, {mo[1]}, {mo[2]}]\n")
            out.append(f"      products: [{pr[0]}, {pr[1]}, {pr[2]}]\n")
        i += 1

    if updates:
        print(f"  [warn] tasks with parsed inertia but no controller_mass "
              f"anchor line: {sorted(updates)}")
    TASKS.write_text("".join(out))
    print(f"[emit_controller_inertia] wrote {TASKS}")


if __name__ == "__main__":
    sys.exit(main())
