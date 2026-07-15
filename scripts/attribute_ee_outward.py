#!/usr/bin/env python3
"""For each LCS contact-opening event (CONTACT-RUN EE-BOX -> NONE), attribute
the EE-outward micro-motion at K -> K+1 to executor-side causes:

  TARGET-RETREAT : dot(p_ee_des[K+1] - p_ee_des[K], nhat) > +0.5 mm/tick
                   (the executor's commanded EE position moved outward; the
                    impedance PD pulls EE outward to track)
  MODE-FLIP      : mode[K] != mode[K+1]      (control law source changed)
  LAM_N-DROP     : lam_n[K] = True, lam_n[K+1] = False
                   (LCS contact-feedforward τ_ff turned OFF at K+1; the
                    inward-driving recoil compensation disappeared)
  VFF-OUTWARD    : (alpha05 only) dot(v_des[K], nhat) > +0.05 m/s
                   (v_ff command at K was outward; advected onto EE)
  UNATTRIBUTED   : none of the above dominates

For each event, multiple flags may trip; we report flag combinations and
the dominant primary cause by priority: TARGET-RETREAT > MODE-FLIP >
LAM_N-DROP > VFF-OUTWARD > UNATTRIBUTED. The priority reflects how directly
each acts on the EE-outward velocity over a 50 ms control tick.

Inputs:
  Same sweep dir as classify_contact_openings.py.

Outputs:
  Stdout: per-event attribution row + per-arm aggregate.
"""
from __future__ import annotations

import argparse
import re
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass

CR_RE = re.compile(
    r"\[CONTACT-RUN\] step=(?P<step>\d+) "
    r"nhat_BA_W=\[(?P<nx>[+\-\d.eE]+),(?P<ny>[+\-\d.eE]+),(?P<nz>[+\-\d.eE]+)\] "
    r"p_BCb=\[[+\-\d.eE,]+\] "
    r"distance=(?P<dist>[+\-\d.eE]+) contact_type=(?P<ctype>\S+)"
)
GC_RE = re.compile(
    r"\[GATE-CONTACT\] step=(?P<step>\d+) .*?"
    r"A_is_ee=(?P<aee>\d+) .*?"
    r"box_p=\((?P<bxp>[+\-\d.eE]+),(?P<byp>[+\-\d.eE]+),(?P<bzp>[+\-\d.eE]+)\) "
    r"ee_p=\((?P<exp>[+\-\d.eE]+),(?P<eyp>[+\-\d.eE]+),(?P<ezp>[+\-\d.eE]+)\)"
)
# c3-mode [STEP] (no target field):
#   [STEP] step=K mode=c3 t=... ee=(...) obj=(...) goal_dist=Xm switch=... \
#     c3_cost=... lam_n=F lam_t=F contact=Y/N productive=Y/N f_cmd=(...)
STEP_C3_RE = re.compile(
    r"\[STEP\] step=(?P<step>\d+) mode=c3 .*?"
    r"lam_n=(?P<lam_n>[+\-\d.eE]+) "
    r"lam_t=[+\-\d.eE]+ "
    r"contact=(?P<contact>[YN])"
)
# free-mode [STEP] (has target field; we use it):
STEP_FREE_RE = re.compile(
    r"\[STEP\] step=(?P<step>\d+) mode=(?P<mode>free) .*?"
    r"target=\((?P<tx>[+\-\d.eE]+),(?P<ty>[+\-\d.eE]+),(?P<tz>[+\-\d.eE]+)\)"
)
# Mode-only prefix (works for both):
STEP_MODE_RE = re.compile(r"\[STEP\] step=(?P<step>\d+) mode=(?P<mode>\S+)")
# APPROACH-OVERRIDE: emits p_ee_des every step it fires
OVR_RE = re.compile(
    r"\[APPROACH-OVERRIDE\] step=(?P<step>\d+) phase=(?P<phase>\S+) .*?"
    r"p_ee_des=\((?P<tx>[+\-\d.eE]+),(?P<ty>[+\-\d.eE]+),(?P<tz>[+\-\d.eE]+)\)"
)
IMP_RE = re.compile(
    r"\[IMP\] step=(?P<step>\d+) mode=(?P<mode>\S+) "
    r"\|x_err\|=(?P<xerr>[\d.eE+\-]+)m "
    r"\|tau_imp\|=(?P<ti>[\d.eE+\-]+)Nm "
    r"\|tau_ff\|=(?P<tf>[\d.eE+\-]+)Nm "
    r"\|tau_out\|=(?P<to>[\d.eE+\-]+)Nm "
    r"sat=(?P<sat>\S+) "
    r"lam_n=(?P<lam_n>\S+) "
    r"lam_t=(?P<lam_t>\S+)"
)
VFF_RE = re.compile(
    r"\[VFF\] step=(?P<step>\d+) mode=(?P<mode>\S+) "
    r"alpha=(?P<alpha>[\d.]+) "
    r"v_des=\((?P<vx>[+\-\d.eE]+),(?P<vy>[+\-\d.eE]+),(?P<vz>[+\-\d.eE]+)\)"
)

BOX_HALF       = 0.05
PUSHER_RADIUS  = 0.025
CONTACT_OFFSET = BOX_HALF + PUSHER_RADIUS
ADMIT_THR      = 0.002
DROP_THR       = ADMIT_THR + 0.001

TARGET_RETREAT_THR_M    = 0.0005   # +0.5 mm/tick outward on p_ee_des
VFF_OUTWARD_THR_MPS     = 0.05     # +0.05 m/s outward on v_des


@dataclass
class StepDiag:
    nhat:    tuple[float, float, float] | None = None
    dist:    float | None = None
    ctype:   str | None = None
    box:     tuple[float, float, float] | None = None
    ee:      tuple[float, float, float] | None = None
    aee:     int | None = None        # GATE-CONTACT A_is_ee (Drake-side)
    mode:    str | None = None        # [STEP] mode (overrides [IMP] mode)
    step_contact: str | None = None   # [STEP] contact=Y/N field (c3 mode)
    step_lam_n: float | None = None   # [STEP] lam_n float (planner λ_n_first)
    p_ee_des: tuple[float, float, float] | None = None   # APPROACH-OVERRIDE p_ee_des
    override_phase: str | None = None # APPROACH-OVERRIDE phase
    tau_imp: float | None = None
    tau_ff:  float | None = None
    lam_n:   bool | None = None       # [IMP] lam_n boolean
    v_des:   tuple[float, float, float] | None = None


def parse_log(path: pathlib.Path) -> dict[int, StepDiag]:
    steps: dict[int, StepDiag] = {}

    def sd(k):
        return steps.setdefault(k, StepDiag())

    with path.open(errors="replace") as fh:
        for line in fh:
            m = CR_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.nhat = (float(m["nx"]), float(m["ny"]), float(m["nz"]))
                d.dist = float(m["dist"])
                d.ctype = m["ctype"]
                continue
            m = GC_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.ee  = (float(m["exp"]),  float(m["eyp"]),  float(m["ezp"]))
                d.box = (float(m["bxp"]),  float(m["byp"]),  float(m["bzp"]))
                d.aee = int(m["aee"])
                continue
            m = STEP_C3_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.mode = "c3"
                d.step_contact = m["contact"]
                d.step_lam_n = float(m["lam_n"])
                continue
            m = STEP_FREE_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.mode = "free"
                d.p_ee_des = (float(m["tx"]), float(m["ty"]), float(m["tz"]))
                continue
            m = STEP_MODE_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                if d.mode is None:
                    d.mode = m["mode"]
                continue
            m = OVR_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.override_phase = m["phase"]
                d.p_ee_des = (float(m["tx"]), float(m["ty"]), float(m["tz"]))
                continue
            m = IMP_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                if d.mode is None:
                    d.mode = m["mode"]
                d.tau_imp = float(m["ti"])
                d.tau_ff  = float(m["tf"])
                d.lam_n   = (m["lam_n"] == "True")
                continue
            m = VFF_RE.search(line)
            if m:
                d = sd(int(m["step"]))
                d.v_des = (float(m["vx"]), float(m["vy"]), float(m["vz"]))
    return steps


@dataclass
class AttribRow:
    arm: str
    seed: int
    step_K: int
    nhat: tuple[float, float, float]
    ee_out_mm: float
    p_ee_des_outward_mm: float
    mode_K: str | None
    mode_K1: str | None
    mode_flip: bool
    aee_K: int | None          # Drake A_is_ee at K
    aee_K1: int | None
    step_lam_n_K: float | None # planner's λ_n_first at K (from [STEP])
    step_lam_n_K1: float | None
    override_phase_K: str | None  # 'A_lift_trav', 'A_descend', 'B_*', etc.
    vff_outward_mps: float | None
    flags: list[str]
    primary: str


def attribute(steps: dict[int, StepDiag], arm: str, seed: int) -> list[AttribRow]:
    rows: list[AttribRow] = []
    ks = sorted(steps)
    for i, k in enumerate(ks[:-1]):
        k1 = ks[i + 1]
        if k1 != k + 1:
            continue
        s = steps[k]; sn = steps[k1]
        if s.ctype != "EE-BOX" or sn.ctype != "NONE":
            continue
        if s.ee is None or s.box is None or sn.ee is None or sn.box is None:
            continue
        if s.nhat is None:
            continue
        nhat = s.nhat

        def dot(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

        delta_ee = tuple(sn.ee[j]-s.ee[j] for j in range(3))
        ee_out_mm = 1000.0 * dot(delta_ee, nhat)

        # Skip events with effectively no outward motion (the K→K+1 step had
        # ee inward or ~0 — those are flicker-coincident but not "EE moved
        # outward" events. Keep flag for completeness but mark them out.
        # Use ALL events that meet classify_contact_openings criteria
        # (CONTACT-RUN EE-BOX → NONE) — be consistent.

        flags = []

        # Flag 0 (DOMINANT — check first): PHANTOM-LCS — LCS admitted EE-BOX
        # but Drake did not realize a contact force at K. This means the
        # contact was admitted by the model only; nothing physical to "open"
        # in the Drake sense. EE-outward motion at K+1 is not "contact
        # release" but trajectory continuation past the brushed face.
        phantom = (s.aee == 0)
        if phantom:
            flags.append("PHANTOM-LCS")

        # Flag 0b: PLANNER-LAMBDA-ZERO — at K, planner's first-knot λ_n_first
        # is zero (no force intent). The "contact" exists in LCS but the
        # solver chose zero contact force. Often coincident with phantom.
        if s.step_lam_n is not None and s.step_lam_n < 1e-6:
            flags.append("PLANNER-LAMBDA-ZERO")

        # Flag 0c: OVERRIDE-ACTIVE at K — APPROACH-OVERRIDE phase fires at K.
        # Often SUPPRESSED at admitted ticks (LCS contact admission disables
        # the override). Check K-1 (just before admission) and K+1 (after
        # drop) too — if override fired before and resumes after, this is
        # the trajectory-brush pattern.
        override_K   = s.override_phase
        if override_K is not None:
            flags.append(f"OVERRIDE@K[{override_K}]")
        # Look at K-1 and K+1 for surrounding context (set in attribute loop)

        # Flag 1: TARGET-RETREAT — p_ee_des moved outward by ≥+0.5 mm
        p_ee_des_outward_mm = float("nan")
        if s.p_ee_des is not None and sn.p_ee_des is not None:
            dp = tuple(sn.p_ee_des[j] - s.p_ee_des[j] for j in range(3))
            v_out = dot(dp, nhat)
            p_ee_des_outward_mm = 1000.0 * v_out
            if v_out >= TARGET_RETREAT_THR_M:
                flags.append("TARGET-RETREAT")

        # Flag 2: MODE-FLIP
        mode_flip = (s.mode is not None and sn.mode is not None and s.mode != sn.mode)
        if mode_flip:
            flags.append("MODE-FLIP")

        # Flag 3: VFF-OUTWARD (alpha05 only)
        vff_out = None
        if s.v_des is not None:
            vff_out_m = dot(s.v_des, nhat)
            vff_out = vff_out_m
            if vff_out_m >= VFF_OUTWARD_THR_MPS:
                flags.append("VFF-OUTWARD")

        # Primary cause selection — PHANTOM-LCS dominates if it tripped,
        # since the event isn't really a "contact release" but trajectory
        # brush. Otherwise priority TARGET-RETREAT > MODE-FLIP > VFF-OUTWARD.
        if phantom:
            primary = "PHANTOM-LCS"
        elif "TARGET-RETREAT" in flags:
            primary = "TARGET-RETREAT"
        elif "MODE-FLIP" in flags:
            primary = "MODE-FLIP"
        elif "VFF-OUTWARD" in flags:
            primary = "VFF-OUTWARD"
        else:
            primary = "REAL-CONTACT-OPENING"   # the rare case: A_is_ee=1 at K

        # Override neighborhood: K-1, K, K+1
        s_prev = steps.get(k - 1)
        override_Km1 = s_prev.override_phase if s_prev is not None else None
        override_K1  = sn.override_phase
        # If override was active at K-1 and resumes at K+1, classify as
        # TRAJ-BRUSH (the override trajectory was running, briefly suppressed
        # at admitted ticks, and resumes after drop).
        traj_brush = (override_Km1 is not None and override_K1 is not None
                      and override_K is None)
        if traj_brush:
            flags.append("TRAJ-BRUSH")
            primary = "TRAJ-BRUSH"

        rows.append(AttribRow(
            arm=arm, seed=seed, step_K=k, nhat=nhat,
            ee_out_mm=ee_out_mm,
            p_ee_des_outward_mm=p_ee_des_outward_mm,
            mode_K=s.mode, mode_K1=sn.mode,
            mode_flip=mode_flip,
            aee_K=s.aee, aee_K1=sn.aee,
            step_lam_n_K=s.step_lam_n, step_lam_n_K1=sn.step_lam_n,
            override_phase_K=override_K,
            vff_outward_mps=vff_out,
            flags=flags, primary=primary,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    logs = sorted(d.glob("seed*_*.log"))
    if not logs:
        print(f"No seed*_*.log in {d}", file=sys.stderr); sys.exit(2)

    fn_re = re.compile(r"seed(\d+)_(off|alpha05)\.log")
    all_rows: list[AttribRow] = []
    for p in logs:
        m = fn_re.match(p.name)
        if not m: continue
        seed = int(m.group(1)); arm = m.group(2)
        steps = parse_log(p)
        rows = attribute(steps, arm, seed)
        all_rows.extend(rows)

    print(f"=== EE-OUTWARD ATTRIBUTION over {len(all_rows)} contact-opening events ===")
    print()

    # Primary cause distribution
    print("--- PRIMARY CAUSE (priority: PHANTOM-LCS > TARGET-RETREAT > MODE-FLIP > VFF-OUTWARD > REAL-CONTACT-OPENING) ---")
    primary_by_arm: dict[str, Counter] = defaultdict(Counter)
    for r in all_rows:
        primary_by_arm[r.arm][r.primary] += 1
    primary_pooled = Counter()
    for c in primary_by_arm.values():
        primary_pooled.update(c)
    print(f"{'cause':>22s}  {'all':>5s} {'alpha05':>9s} {'off':>5s}")
    total = sum(primary_pooled.values())
    for cause in ["TRAJ-BRUSH", "PHANTOM-LCS", "TARGET-RETREAT", "MODE-FLIP",
                  "VFF-OUTWARD", "REAL-CONTACT-OPENING"]:
        a = primary_by_arm["alpha05"].get(cause, 0)
        o = primary_by_arm["off"].get(cause, 0)
        t = primary_pooled.get(cause, 0)
        frac = (100.0*t/total) if total else 0.0
        print(f"{cause:>22s}  {t:>5d} {a:>9d} {o:>5d}   ({frac:5.1f}% pooled)")

    print()
    print("--- PHANTOM-LCS vs REAL-CONTACT — A_is_ee at K ---")
    aee1 = [r for r in all_rows if r.aee_K == 1]
    aee0 = [r for r in all_rows if r.aee_K == 0]
    print(f"  events with Drake-realized contact at K (A_is_ee=1): {len(aee1)}  ({100.0*len(aee1)/len(all_rows):5.1f}%)")
    print(f"  events with phantom LCS contact at K  (A_is_ee=0):   {len(aee0)}  ({100.0*len(aee0)/len(all_rows):5.1f}%)")

    print()
    print("--- PLANNER λ_n_first at K (from c3-mode [STEP]) ---")
    lambda_zero = [r for r in all_rows
                   if r.step_lam_n_K is not None and r.step_lam_n_K < 1e-6]
    lambda_pos  = [r for r in all_rows
                   if r.step_lam_n_K is not None and r.step_lam_n_K >= 1e-6]
    lambda_none = [r for r in all_rows if r.step_lam_n_K is None]
    print(f"  λ_n_first = 0 at K:    {len(lambda_zero):4d}  ({100.0*len(lambda_zero)/len(all_rows):5.1f}%)")
    print(f"  λ_n_first > 0 at K:    {len(lambda_pos):4d}  ({100.0*len(lambda_pos)/len(all_rows):5.1f}%)")
    print(f"  K not in c3 mode (no [STEP] λ_n_first): {len(lambda_none):4d}  ({100.0*len(lambda_none)/len(all_rows):5.1f}%)")
    if lambda_pos:
        print(f"  λ_n_first values >0: {sorted(r.step_lam_n_K for r in lambda_pos)[:10]}{'...' if len(lambda_pos)>10 else ''}")

    print()
    print("--- APPROACH-OVERRIDE PHASE at K ---")
    override_counter = Counter(r.override_phase_K or "<no override active>" for r in all_rows)
    for k, v in override_counter.most_common():
        print(f"  {k:<30s}  {v:>4d}  ({100.0*v/len(all_rows):5.1f}%)")

    print()
    print("--- FLAG-COMBINATION DISTRIBUTION (every event, all flags that tripped) ---")
    combo_counter: Counter = Counter()
    for r in all_rows:
        key = "+".join(r.flags) if r.flags else "<none>"
        combo_counter[key] += 1
    for k, v in combo_counter.most_common():
        print(f"  {k:<40s}  {v:>4d}  ({100.0*v/len(all_rows):5.1f}%)")

    print()
    print("--- INDIVIDUAL FLAG TRIP RATES (multi-counted) ---")
    flag_counter = Counter()
    for r in all_rows:
        for f in r.flags:
            flag_counter[f] += 1
    for k, v in flag_counter.most_common():
        print(f"  {k:<20s}  {v:>4d}  ({100.0*v/len(all_rows):5.1f}% of events)")

    print()
    print("--- EE-OUTWARD MOTION MAGNITUDE BREAKDOWN ---")
    pos_evt = [r for r in all_rows if r.ee_out_mm > 0]
    neg_evt = [r for r in all_rows if r.ee_out_mm < 0]
    zero_evt = [r for r in all_rows if r.ee_out_mm == 0]
    print(f"  events with EE moving OUTWARD (ee_out_mm > 0): {len(pos_evt)}")
    print(f"  events with EE moving INWARD  (ee_out_mm < 0): {len(neg_evt)}")
    print(f"  events with EE stationary in normal dir       : {len(zero_evt)}")
    if pos_evt:
        vals = sorted(r.ee_out_mm for r in pos_evt)
        print(f"  median outward motion (mm/tick): {vals[len(vals)//2]:.3f}")
        print(f"  mean   outward motion (mm/tick): {sum(vals)/len(vals):.3f}")

    if args.detail:
        print()
        print("--- EVERY EVENT (debug) ---")
        for r in sorted(all_rows, key=lambda r: (r.arm, r.seed, r.step_K)):
            tgt = (f"{r.p_ee_des_outward_mm:+6.2f}mm"
                   if r.p_ee_des_outward_mm == r.p_ee_des_outward_mm else "  N/A ")
            lambda_str = (f"{r.step_lam_n_K:.3f}" if r.step_lam_n_K is not None else "N/A")
            ovr = r.override_phase_K or "-"
            print(f"  arm={r.arm:>7s} seed={r.seed} step={r.step_K:>3d} "
                  f"ee_out={r.ee_out_mm:+6.2f}mm  "
                  f"p_ee_des_out={tgt}  "
                  f"aee {str(r.aee_K):>4s}->{str(r.aee_K1):<4s}  "
                  f"λn={lambda_str:>5s}  "
                  f"ovr={ovr:>11s}  primary={r.primary}")


if __name__ == "__main__":
    main()
