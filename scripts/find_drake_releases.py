#!/usr/bin/env python3
"""Find Drake-side contact release events: A_is_ee=1 -> A_is_ee=0 transitions.

These are the REAL contact-release events (where Drake actually had a
resolved contact force and then lost it). Distinct from LCS-side openings
(CONTACT-RUN EE-BOX -> NONE), which are dominated by phantom drops.

For each Drake-release event, decompose K->K+1:
  - EE motion in nhat direction
  - box motion
  - F_W at K (Drake-realized force magnitude / direction)
  - mode, override, planner λ_n
"""
from __future__ import annotations
import argparse, re, pathlib
from collections import Counter
from dataclasses import dataclass

GC_RE = re.compile(
    r"\[GATE-CONTACT\] step=(?P<step>\d+) "
    r"F_W=\((?P<fx>[+\-\d.eE]+),(?P<fy>[+\-\d.eE]+),(?P<fz>[+\-\d.eE]+)\) "
    r"F_on_box=\([+\-\d.eE,]+\) "
    r"n_face_out=\((?P<nfx>[+\-\d.eE]+),(?P<nfy>[+\-\d.eE]+),(?P<nfz>[+\-\d.eE]+)\) "
    r"A_is_ee=(?P<aee>\d+) "
    r"box_q=\([+\-\d.eE,]+\) "
    r"box_p=\((?P<bxp>[+\-\d.eE]+),(?P<byp>[+\-\d.eE]+),(?P<bzp>[+\-\d.eE]+)\) "
    r"ee_p=\((?P<exp>[+\-\d.eE]+),(?P<eyp>[+\-\d.eE]+),(?P<ezp>[+\-\d.eE]+)\)"
)
CR_RE = re.compile(
    r"\[CONTACT-RUN\] step=(?P<step>\d+) "
    r"nhat_BA_W=\[(?P<nx>[+\-\d.eE]+),(?P<ny>[+\-\d.eE]+),(?P<nz>[+\-\d.eE]+)\] "
    r"p_BCb=\[[+\-\d.eE,]+\] "
    r"distance=(?P<dist>[+\-\d.eE]+) contact_type=(?P<ctype>\S+)"
)
OVR_RE = re.compile(r"\[APPROACH-OVERRIDE\] step=(?P<step>\d+) phase=(?P<phase>\S+)")
STEP_C3_RE = re.compile(
    r"\[STEP\] step=(?P<step>\d+) mode=c3 .*?lam_n=(?P<lam_n>[+\-\d.eE]+) "
    r"lam_t=[+\-\d.eE]+ contact=(?P<contact>[YN])"
)
STEP_MODE_RE = re.compile(r"\[STEP\] step=(?P<step>\d+) mode=(?P<mode>\S+)")


@dataclass
class StepData:
    aee:    int | None = None
    fW:     tuple[float, float, float] | None = None
    n_face_out: tuple[float, float, float] | None = None
    ee:     tuple[float, float, float] | None = None
    box:    tuple[float, float, float] | None = None
    ctype:  str | None = None
    nhat:   tuple[float, float, float] | None = None
    dist:   float | None = None
    mode:   str | None = None
    lam_n:  float | None = None
    override_phase: str | None = None


def parse_log(path):
    steps = {}
    def sd(k): return steps.setdefault(k, StepData())
    with path.open(errors="replace") as fh:
        for line in fh:
            m = GC_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.aee = int(m["aee"])
                d.fW = (float(m["fx"]), float(m["fy"]), float(m["fz"]))
                d.n_face_out = (float(m["nfx"]), float(m["nfy"]), float(m["nfz"]))
                d.ee = (float(m["exp"]), float(m["eyp"]), float(m["ezp"]))
                d.box = (float(m["bxp"]), float(m["byp"]), float(m["bzp"]))
                continue
            m = CR_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.ctype = m["ctype"]
                d.nhat = (float(m["nx"]), float(m["ny"]), float(m["nz"]))
                d.dist = float(m["dist"])
                continue
            m = OVR_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.override_phase = m["phase"]
                continue
            m = STEP_C3_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.mode = "c3"
                d.lam_n = float(m["lam_n"])
                continue
            m = STEP_MODE_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                if d.mode is None:
                    d.mode = m["mode"]
    return steps


def find_drake_releases(steps, arm, seed):
    events = []
    ks = sorted(steps)
    for i, k in enumerate(ks[:-1]):
        k1 = ks[i+1]
        if k1 != k + 1:
            continue
        s, sn = steps[k], steps[k1]
        if s.aee != 1 or sn.aee != 0:
            continue
        # Use Drake n_face_out at K as the outward normal (most reliable here)
        nhat = s.n_face_out
        if nhat is None or all(abs(c) < 1e-6 for c in nhat):
            # fallback to CONTACT-RUN nhat
            nhat = s.nhat if s.nhat is not None else (0,0,0)

        def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

        delta_ee = tuple(sn.ee[j]-s.ee[j] for j in range(3)) if (s.ee and sn.ee) else None
        delta_box = tuple(sn.box[j]-s.box[j] for j in range(3)) if (s.box and sn.box) else None
        ee_out_mm = (1000*dot(delta_ee, nhat)) if delta_ee else float("nan")
        box_runaway_mm = (-1000*dot(delta_box, nhat)) if delta_box else float("nan")
        fW_mag = (sum(c*c for c in s.fW)**0.5) if s.fW else float("nan")
        fW_inward = (-dot(s.fW, nhat)) if (s.fW and nhat) else float("nan")
        events.append({
            "arm": arm, "seed": seed, "step_K": k,
            "nhat": nhat,
            "ee_out_mm": ee_out_mm, "box_runaway_mm": box_runaway_mm,
            "F_W_mag": fW_mag, "F_W_inward_N": fW_inward,
            "ctype_K": s.ctype, "ctype_K1": sn.ctype,
            "dist_K": s.dist, "dist_K1": sn.dist,
            "mode_K": s.mode, "mode_K1": sn.mode,
            "lam_n_K": s.lam_n, "lam_n_K1": sn.lam_n,
            "override_K": s.override_phase, "override_K1": sn.override_phase,
        })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    logs = sorted(d.glob("seed*_*.log"))
    fn_re = re.compile(r"seed(\d+)_(off|alpha05)\.log")
    all_events = []
    per_log_aee = {}
    for p in logs:
        m = fn_re.match(p.name)
        if not m: continue
        seed = int(m.group(1)); arm = m.group(2)
        steps = parse_log(p)
        aee1 = sum(1 for s in steps.values() if s.aee == 1)
        per_log_aee[(arm, seed)] = aee1
        evs = find_drake_releases(steps, arm, seed)
        all_events.extend(evs)
        if evs:
            print(f"[{arm} seed={seed}] Drake-realized contact ticks={aee1}; "
                  f"release events found: {len(evs)} at steps {[e['step_K'] for e in evs]}")
        else:
            if aee1 > 0:
                print(f"[{arm} seed={seed}] aee=1 ticks={aee1} but no K→K+1 transitions "
                      f"detected (gap in log or contact persists to end-of-run)")
            # silent for runs with aee=0

    print()
    print(f"=== TOTAL DRAKE RELEASE EVENTS ACROSS 20 RUNS: {len(all_events)} ===")
    if not all_events:
        print("No Drake-side contact-release events detected.")
        return

    print()
    print("--- PER-EVENT DETAIL ---")
    print(f"{'arm':>7s} {'seed':>4s} {'stepK':>5s}  "
          f"{'nhat':>22s}  {'F_W_mag':>7s} {'F_inward':>8s}  "
          f"{'ee_out':>7s} {'box_run':>7s}  "
          f"{'cT->cT+1':>11s} {'mode':>4s}  {'ovr_K':>11s} {'ovr_K+1':>11s}")
    for e in all_events:
        nh = e["nhat"]
        print(f"{e['arm']:>7s} {e['seed']:>4d} {e['step_K']:>5d}  "
              f"({nh[0]:+.2f},{nh[1]:+.2f},{nh[2]:+.2f})  "
              f"{e['F_W_mag']:7.3f} {e['F_W_inward_N']:+8.3f}  "
              f"{e['ee_out_mm']:+6.2f} {e['box_runaway_mm']:+6.2f}  "
              f"{str(e['ctype_K']):>5s}->{str(e['ctype_K1']):<4s} "
              f"{str(e['mode_K']):>4s}  "
              f"{str(e['override_K']):>11s} {str(e['override_K1']):>11s}")

    print()
    # Classify: EE-RETREAT (ee_out > box_runaway), BOX-ESCAPE (other way),
    # LCS-DROP (CONTACT-RUN was EE-BOX at K+1 — LCS still admitted while
    # Drake released; this means Drake stopped resolving force despite LCS
    # still seeing contact). These align with the user's original categories.
    cnt = Counter()
    for e in all_events:
        if e["ctype_K1"] == "EE-BOX":
            cnt["LCS-still-admits"] += 1
        elif e["ee_out_mm"] > e["box_runaway_mm"]:
            cnt["EE-RETREAT"] += 1
        elif e["box_runaway_mm"] > 0.2:
            cnt["BOX-ESCAPE"] += 1
        else:
            cnt["AMBIGUOUS"] += 1
    print("--- CLASSIFIED DRAKE RELEASE EVENTS ---")
    for k, v in cnt.most_common():
        print(f"  {k:<20s}  {v}")


if __name__ == "__main__":
    main()
