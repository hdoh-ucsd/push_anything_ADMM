#!/usr/bin/env python3
"""Classify LCS contact-opening events across the 20-run seed-sign sweep.

A "contact-opening event" = consecutive control ticks (K, K+1) where:
  CONTACT-RUN[K]  has contact_type=EE-BOX  (LCS admitted EE-box pair)
  CONTACT-RUN[K+1] has contact_type=NONE   (LCS no longer admits it)

For each event, decompose using:
  nhat[K]          : outward face normal at the last admitted tick
                     (from CONTACT-RUN nhat_BA_W; box→EE)
  Δee = ee_p[K+1] - ee_p[K]    (from GATE-CONTACT)
  Δbox = box_p[K+1] - box_p[K]
  d_geom(K+1)      : estimated sphere-box SDF on the same face at K+1
                     d_geom = (ee_p - box_p) · nhat - (box_half + pusher_radius)

Classification (in order):
  LCS-DROP      : d_geom(K+1) <= 0.003 m   (geometry still in contact — LCS
                  threshold is 0.002 m; 0.003 m gives a ~1 mm slack for
                  estimation error)
  EE-RETREAT    : d_geom(K+1) > 0.003 AND Δee · nhat > -Δbox · nhat
                  (EE moved outward more than the box ran away in push dir)
  BOX-ESCAPE    : d_geom(K+1) > 0.003 AND -Δbox · nhat >= Δee · nhat

Constants:
  box_half_extent  = 0.05 m   (config/sampling_c3_kik.yaml:148)
  pusher_radius    = 0.025 m  (Franka EE sphere, sim/env_builder.py)
  admit_threshold  = 0.002 m  (lcs_formulator.py:245)
  drop_slack       = 0.001 m  (estimation slack above admit_threshold)
"""
from __future__ import annotations

import argparse
import re
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass

CR_RE = re.compile(
    r"\[CONTACT-RUN\] step=(?P<step>\d+) "
    r"nhat_BA_W=\[(?P<nx>[+\-\d.eE]+),(?P<ny>[+\-\d.eE]+),(?P<nz>[+\-\d.eE]+)\] "
    r"p_BCb=\[(?P<bx>[+\-\d.eE]+),(?P<by>[+\-\d.eE]+),(?P<bz>[+\-\d.eE]+)\] "
    r"distance=(?P<dist>[+\-\d.eE]+) contact_type=(?P<ctype>\S+)"
)
GC_RE = re.compile(
    r"\[GATE-CONTACT\] step=(?P<step>\d+) "
    r"F_W=\([+\-\d.eE,]+\) "
    r"F_on_box=\([+\-\d.eE,]+\) "
    r"n_face_out=\([+\-\d.eE,]+\) "
    r"A_is_ee=(?P<aee>\d+) "
    r"box_q=\([+\-\d.eE,]+\) "
    r"box_p=\((?P<bxp>[+\-\d.eE]+),(?P<byp>[+\-\d.eE]+),(?P<bzp>[+\-\d.eE]+)\) "
    r"ee_p=\((?P<exp>[+\-\d.eE]+),(?P<eyp>[+\-\d.eE]+),(?P<ezp>[+\-\d.eE]+)\)"
)

BOX_HALF       = 0.05
PUSHER_RADIUS  = 0.025
CONTACT_OFFSET = BOX_HALF + PUSHER_RADIUS    # 0.075 m, sphere-cube face contact
ADMIT_THR      = 0.002                       # LCS admit threshold
DROP_SLACK     = 0.001                       # estimation slack
DROP_THR       = ADMIT_THR + DROP_SLACK      # 0.003 m

SIGN_DOMINANCE = 1.0                         # strict: larger of two wins


@dataclass
class StepData:
    ee:    tuple[float, float, float] | None = None    # GATE-CONTACT ee_p
    box:   tuple[float, float, float] | None = None    # GATE-CONTACT box_p
    aee:   int | None = None                           # GATE A_is_ee
    nhat:  tuple[float, float, float] | None = None    # CONTACT-RUN nhat_BA_W
    dist:  float | None = None                         # CONTACT-RUN distance
    ctype: str | None = None                           # CONTACT-RUN contact_type


@dataclass
class OpeningEvent:
    log:      str
    arm:      str
    seed:     int
    step_K:   int
    nhat:     tuple[float, float, float]
    dist_K:   float                # SDF at K (small, EE-BOX admitted)
    d_geom_K1: float               # estimated SDF at K+1
    ee_outward: float              # Δee · nhat   (positive = EE moved out)
    box_runaway: float             # -Δbox · nhat (positive = box ran from EE)
    delta_ee: tuple[float, float, float]
    delta_box: tuple[float, float, float]
    label: str                     # 'LCS-DROP' / 'EE-RETREAT' / 'BOX-ESCAPE'


def parse_log(path: pathlib.Path) -> dict[int, StepData]:
    steps: dict[int, StepData] = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            m = CR_RE.search(line)
            if m:
                k = int(m["step"])
                sd = steps.setdefault(k, StepData())
                sd.nhat = (float(m["nx"]), float(m["ny"]), float(m["nz"]))
                sd.dist = float(m["dist"])
                sd.ctype = m["ctype"]
                continue
            m = GC_RE.search(line)
            if m:
                k = int(m["step"])
                sd = steps.setdefault(k, StepData())
                sd.ee  = (float(m["exp"]),  float(m["eyp"]),  float(m["ezp"]))
                sd.box = (float(m["bxp"]),  float(m["byp"]),  float(m["bzp"]))
                sd.aee = int(m["aee"])
    return steps


def find_openings(steps: dict[int, StepData],
                  log_path: pathlib.Path,
                  arm: str,
                  seed: int) -> list[OpeningEvent]:
    events: list[OpeningEvent] = []
    sorted_keys = sorted(steps)
    for i, k in enumerate(sorted_keys[:-1]):
        k1 = sorted_keys[i + 1]
        if k1 != k + 1:
            continue  # gap in log (don't classify across discontinuities)
        sd_k  = steps[k]
        sd_k1 = steps[k1]
        if sd_k.ctype != "EE-BOX":
            continue
        if sd_k1.ctype != "NONE":
            continue
        # Need positions at both ticks to classify
        if sd_k.ee is None or sd_k.box is None or sd_k1.ee is None or sd_k1.box is None:
            continue
        nhat = sd_k.nhat
        if nhat is None:
            continue

        def dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        delta_ee  = tuple(sd_k1.ee[j]  - sd_k.ee[j]  for j in range(3))
        delta_box = tuple(sd_k1.box[j] - sd_k.box[j] for j in range(3))

        ee_outward  = dot(delta_ee,  nhat)
        box_runaway = -dot(delta_box, nhat)

        sep_vec_K1 = tuple(sd_k1.ee[j] - sd_k1.box[j] for j in range(3))
        d_geom_K1  = dot(sep_vec_K1, nhat) - CONTACT_OFFSET

        if d_geom_K1 <= DROP_THR:
            label = "LCS-DROP"
        elif ee_outward >= SIGN_DOMINANCE * box_runaway:
            label = "EE-RETREAT"
        else:
            label = "BOX-ESCAPE"

        events.append(OpeningEvent(
            log=str(log_path), arm=arm, seed=seed,
            step_K=k, nhat=nhat,
            dist_K=sd_k.dist if sd_k.dist is not None else float("nan"),
            d_geom_K1=d_geom_K1,
            ee_outward=ee_outward, box_runaway=box_runaway,
            delta_ee=delta_ee, delta_box=delta_box,
            label=label,
        ))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="probe output directory (with seed*_*.log)")
    ap.add_argument("--detail", action="store_true",
                    help="dump every opening-event row")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    logs = sorted(d.glob("seed*_*.log"))
    if not logs:
        print(f"No seed*_*.log in {d}", file=sys.stderr); sys.exit(2)

    all_events: list[OpeningEvent] = []
    per_log_counts: list[tuple[str, int, int, int, int, int, int]] = []
    # (arm, seed, n_lcs_ee_box, n_drake_aee1, n_openings, lcs_drop, ee_retreat, box_escape)

    fn_re = re.compile(r"seed(\d+)_(off|alpha05)\.log")
    for p in logs:
        m = fn_re.match(p.name)
        seed = int(m.group(1)) if m else -1
        arm  = m.group(2)     if m else "?"
        steps = parse_log(p)
        events = find_openings(steps, p, arm, seed)
        all_events.extend(events)

        n_lcs = sum(1 for s in steps.values() if s.ctype == "EE-BOX")
        n_dr  = sum(1 for s in steps.values() if s.aee == 1)
        c = Counter(e.label for e in events)
        per_log_counts.append((
            arm, seed, n_lcs, n_dr, len(events),
            c.get("LCS-DROP", 0), c.get("EE-RETREAT", 0), c.get("BOX-ESCAPE", 0)
        ))

    print("=== PER-LOG CONTACT-OPENING SUMMARY ===")
    print(f"{'arm':>7s} {'seed':>4s}  "
          f"{'n_LCS':>5s} {'n_Drake':>7s} {'n_open':>6s}  "
          f"{'LCS-DROP':>8s} {'EE-RETR':>7s} {'BOX-ESC':>7s}")
    for row in sorted(per_log_counts, key=lambda r: (r[0], r[1])):
        arm, seed, nlcs, ndr, nop, ld, er, be = row
        print(f"{arm:>7s} {seed:>4d}  "
              f"{nlcs:>5d} {ndr:>7d} {nop:>6d}  "
              f"{ld:>8d} {er:>7d} {be:>7d}")

    print()
    print("=== AGGREGATE BY ARM ===")
    by_arm: dict[str, Counter] = {}
    by_arm_lcs: dict[str, int] = {}
    by_arm_drake: dict[str, int] = {}
    by_arm_openings: dict[str, int] = {}
    for arm, seed, nlcs, ndr, nop, ld, er, be in per_log_counts:
        by_arm_lcs[arm] = by_arm_lcs.get(arm, 0) + nlcs
        by_arm_drake[arm] = by_arm_drake.get(arm, 0) + ndr
        by_arm_openings[arm] = by_arm_openings.get(arm, 0) + nop
        c = by_arm.setdefault(arm, Counter())
        c["LCS-DROP"]   += ld
        c["EE-RETREAT"] += er
        c["BOX-ESCAPE"] += be
    print(f"{'arm':>7s}  {'n_LCS_total':>11s} {'n_Drake_total':>13s} {'openings':>8s} "
          f"  {'LCS-DROP':>9s} {'EE-RETR':>8s} {'BOX-ESC':>8s}")
    for arm in sorted(by_arm):
        c = by_arm[arm]
        n_open = by_arm_openings[arm]
        ld, er, be = c["LCS-DROP"], c["EE-RETREAT"], c["BOX-ESCAPE"]
        def pct(x): return f"{(100.0*x/n_open):5.1f}%" if n_open else "  -  "
        print(f"{arm:>7s}  {by_arm_lcs[arm]:>11d} {by_arm_drake[arm]:>13d} {n_open:>8d}   "
              f"{ld:>4d}({pct(ld)}) {er:>3d}({pct(er)}) {be:>3d}({pct(be)})")

    print()
    print("=== AGGREGATE (BOTH ARMS POOLED) ===")
    pooled = Counter()
    for arm, c in by_arm.items():
        pooled.update(c)
    total_n = sum(pooled.values())
    if total_n:
        for label in ("LCS-DROP", "EE-RETREAT", "BOX-ESCAPE"):
            n = pooled[label]
            print(f"  {label:>10s}  {n:>4d}  ({100.0*n/total_n:5.1f}%)")
        print(f"  {'TOTAL':>10s}  {total_n:>4d}")
    print()
    print(f"=== LCS vs Drake gap (model-fidelity signal) ===")
    for arm in sorted(by_arm_lcs):
        nl, nd = by_arm_lcs[arm], by_arm_drake[arm]
        ratio = (nl / nd) if nd else float("inf")
        print(f"  [{arm}] LCS-admitted EE-BOX ticks = {nl:5d}  vs  "
              f"Drake A_is_ee=1 ticks = {nd:5d}  "
              f"(LCS/Drake = {ratio:.2f}x)")

    if args.detail:
        print()
        print("=== EVERY OPENING EVENT (debug) ===")
        for e in all_events:
            print(f"  arm={e.arm} seed={e.seed} step={e.step_K} "
                  f"nhat=({e.nhat[0]:+.2f},{e.nhat[1]:+.2f},{e.nhat[2]:+.2f}) "
                  f"dist_K={e.dist_K:+.5f} d_geom_K1={e.d_geom_K1:+.5f} "
                  f"ee_out={e.ee_outward*1000:+6.2f}mm box_runaway={e.box_runaway*1000:+6.2f}mm "
                  f"-> {e.label}")


if __name__ == "__main__":
    main()
