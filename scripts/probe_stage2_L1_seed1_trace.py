"""Seed-1 alignment-trace probe — beyond the standard SC probe.

The standard probe (probe_stage2_L1.py) counts [GATE-ALIGN] refusals but does
NOT answer the seed-1 puzzle: n_gate=2 means the gate evaluated only twice,
not that it refused on contact ticks.

This probe extracts, per seed:

(1) First c3 entry: step, switch_reason, EE-box distance at that tick.
    Tests whether finished_repos -> c3 happens WHILE the EE is far from the
    box (contact_info empty -> alignment gate SKIPPED by wrapper.py:917).

(2) First A_is_ee=1 GATE-CONTACT tick: step, n_face_out, alignment.
    Tests whether the wrong-face contact would have FAILED the alignment
    threshold if the gate had run.

(3) Distinct g_hat values observed across the run (logged only on
    GATE-ALIGN refusals -- but we also reconstruct g_hat per tick from
    obj_xy logged in [STEP] lines).
    Tests hypothesis (b) — g_hat-chatter defeating the gate.

(4) Per-tick alignment trace over the whole run (reconstructed):
       align(t) = nhat_onto_box(t) · g_hat(t)
       g_hat = unit(target - obj_xy)
    Count threshold crossings of 0.7. If many — chatter is real.
    If g_hat stable + nhat_onto_box stable + alignment never above 0.7 once
    locked — the gate is structurally bypassed, not chattering.

Usage: python scripts/probe_stage2_L1_seed1_trace.py
"""
from __future__ import annotations
import math
import re
from pathlib import Path

REPO = Path("/root/push_anything_ADMM")
SWEEP = REPO / "stage2_L1_sweep"
ALIGN_THR = 0.70
GOAL_XY = (-0.3, 0.0)   # task-id=4 (west) goal

RE_GATE_CONTACT = re.compile(
    r"^\[GATE-CONTACT\]\s+step=(\d+).*?"
    r"n_face_out=\(([+\-0-9.,eE]+)\).*?A_is_ee=(\d).*?"
    r"box_p=\(([+\-0-9.,eE]+)\)"
)
RE_STEP = re.compile(
    r"^\[STEP\]\s+step=(\d+)\s+mode=(\w+)\s+t=([0-9.]+)s\s+"
    r"ee=\(([+\-0-9.,eE]+)\)\s+obj=\(([+\-0-9.,eE]+)\).*?"
    r"switch=(\w+)"
)
RE_GATE_ALIGN = re.compile(
    r"^\[GATE-ALIGN\]\s+step=(\d+)\s+refused:\s+align=([+\-0-9.eE]+).*?"
    r"nhat_xy=\(([+\-0-9.,eE]+)\)\s+g_hat=\(([+\-0-9.,eE]+)\)"
)
RE_RESULT = re.compile(r"^\[RESULT\].*?final_obj_xy=\(([+\-0-9.,eE ]+)\)")


def parse_seed(seed: int):
    path = SWEEP / f"seed{seed}_stage2_L1.log"
    if not path.exists():
        return None
    out = {
        "seed": seed,
        "gate_contacts": [],   # (step, n_face_out, A_is_ee, box_xy)
        "steps": [],           # (step, mode, t, ee_xy, obj_xy, switch)
        "gate_aligns": [],     # (step, align, nhat_xy, g_hat)
        "result_xy": None,
    }
    with path.open() as f:
        for line in f:
            m = RE_GATE_CONTACT.match(line)
            if m:
                step = int(m.group(1))
                nfo = tuple(float(x) for x in m.group(2).split(","))
                A = int(m.group(3))
                bp = tuple(float(x) for x in m.group(4).split(","))
                out["gate_contacts"].append((step, nfo, A, bp[:2]))
                continue
            m = RE_STEP.match(line)
            if m:
                step = int(m.group(1)); mode = m.group(2); t = float(m.group(3))
                ee = tuple(float(x) for x in m.group(4).split(","))
                obj = tuple(float(x) for x in m.group(5).split(","))
                switch = m.group(6)
                out["steps"].append((step, mode, t, ee[:2], obj[:2], switch))
                continue
            m = RE_GATE_ALIGN.match(line)
            if m:
                step = int(m.group(1)); align = float(m.group(2))
                nh = tuple(float(x) for x in m.group(3).split(","))
                gh = tuple(float(x) for x in m.group(4).split(","))
                out["gate_aligns"].append((step, align, nh, gh))
                continue
            m = RE_RESULT.match(line)
            if m:
                xy = [float(x) for x in m.group(1).split(",")]
                out["result_xy"] = (xy[0], xy[1])
    return out


def g_hat_from_obj(obj_xy):
    dx = GOAL_XY[0] - obj_xy[0]
    dy = GOAL_XY[1] - obj_xy[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return (0.0, 0.0)
    return (dx/n, dy/n)


def first_c3_entry(steps):
    prev_mode = None
    for step, mode, t, ee, obj, switch in steps:
        if mode == "c3" and prev_mode != "c3":
            return (step, t, switch, ee, obj)
        prev_mode = mode
    return None


def first_aee1(gate_contacts):
    for step, nfo, A, bxy in gate_contacts:
        if A == 1:
            return (step, nfo, bxy)
    return None


def ee_to_box(ee_xy, obj_xy):
    return math.hypot(ee_xy[0]-obj_xy[0], ee_xy[1]-obj_xy[1])


def per_tick_alignment(steps, gate_contacts):
    """Match GATE-CONTACT ticks to STEP ticks (closest step), compute
    align = nhat_onto_box · g_hat where g_hat reconstructed from obj_xy."""
    # Build a step->obj_xy map
    obj_by_step = {s: obj for s, _, _, _, obj, _ in steps}
    out = []
    for step, nfo, A, bxy in gate_contacts:
        if A != 1: continue
        obj = obj_by_step.get(step)
        if obj is None:
            # try nearest step
            for s_off in (1, -1, 2, -2, 3, -3):
                obj = obj_by_step.get(step + s_off)
                if obj is not None:
                    break
        if obj is None:
            continue
        gh = g_hat_from_obj(obj)
        # nhat_onto_box = -n_face_out (face normal points out of box)
        nob_xy = (-nfo[0], -nfo[1])
        align = nob_xy[0]*gh[0] + nob_xy[1]*gh[1]
        out.append((step, align, nob_xy, gh))
    return out


def threshold_crossings(align_trace, thr=ALIGN_THR):
    crossings = 0
    prev_above = None
    for _, align, _, _ in align_trace:
        above = align > thr
        if prev_above is not None and above != prev_above:
            crossings += 1
        prev_above = above
    return crossings


def quantize_unique(vals, decimals=3):
    return sorted(set(round(v, decimals) for v in vals))


def main():
    print(f"Seed-1 alignment-trace probe (threshold = {ALIGN_THR:.2f})\n")
    for seed in [0, 1, 2, 3, 4]:
        d = parse_seed(seed)
        if d is None:
            print(f"=== seed={seed}: log missing ===\n"); continue
        print(f"=== seed={seed} ===")
        if d["result_xy"]:
            print(f"  final_obj_xy=({d['result_xy'][0]:+.4f}, {d['result_xy'][1]:+.4f})")

        # (1) First c3 entry
        fc = first_c3_entry(d["steps"])
        if fc:
            step, t, switch, ee, obj = fc
            d_ee_box = ee_to_box(ee, obj)
            print(f"  first_c3_entry: step={step} t={t:.3f}s switch={switch}  "
                  f"ee_to_box={d_ee_box*1000:.1f}mm (LCS admit threshold = 2mm)")
            if d_ee_box > 0.002 and switch == "kToC3ReachedReposTarget":
                print(f"    GATE SKIPPED: contact_info empty at entry "
                      f"(EE >> 2mm). L1 gate condition `_nhat_xy is not None` "
                      f"(wrapper.py:925) fails -> alignment check bypassed.")

        # (2) First admitted contact
        fa = first_aee1(d["gate_contacts"])
        if fa:
            step, nfo, bxy = fa
            obj_at = next((s_obj for (s_step, _, _, _, s_obj, _) in d["steps"]
                           if s_step == step), bxy)
            gh = g_hat_from_obj(obj_at)
            nob = (-nfo[0], -nfo[1])
            align_at = nob[0]*gh[0] + nob[1]*gh[1]
            print(f"  first_admitted_contact: step={step} "
                  f"n_face_out=({nfo[0]:+.3f},{nfo[1]:+.3f},{nfo[2]:+.3f}) "
                  f"nhat_onto_box_xy=({nob[0]:+.3f},{nob[1]:+.3f})")
            print(f"    reconstructed g_hat=({gh[0]:+.3f},{gh[1]:+.3f})  "
                  f"align@first = {align_at:+.3f}  "
                  f"{'PASSES' if align_at > ALIGN_THR else 'WOULD-FAIL'} thr=0.70")

        # (3) Distinct g_hat values
        gh_logged = [tuple(round(v,3) for v in gh) for (_,_,_,gh) in d["gate_aligns"]]
        gh_recon = []
        for s, _, _, _, obj, _ in d["steps"]:
            g = g_hat_from_obj(obj)
            gh_recon.append((round(g[0],3), round(g[1],3)))
        gh_recon_unique = sorted(set(gh_recon))
        print(f"  g_hat values: logged_distinct={len(set(gh_logged))}  "
              f"reconstructed_distinct={len(gh_recon_unique)}")
        if gh_recon_unique:
            print(f"    g_hat range: [{gh_recon_unique[0]}, ..., {gh_recon_unique[-1]}]  "
                  f"(spread = how much goal-direction rotates as box moves)")

        # (4) Per-tick alignment trace + threshold crossings
        align_trace = per_tick_alignment(d["steps"], d["gate_contacts"])
        if align_trace:
            aligns = [a for _, a, _, _ in align_trace]
            n_above = sum(1 for a in aligns if a > ALIGN_THR)
            n_below = sum(1 for a in aligns if a <= ALIGN_THR)
            crossings = threshold_crossings(align_trace)
            amin = min(aligns); amax = max(aligns); amed = sorted(aligns)[len(aligns)//2]
            print(f"  per-tick alignment trace: n_contact_ticks={len(align_trace)}  "
                  f"above_thr={n_above}  below_thr={n_below}  "
                  f"threshold_crossings={crossings}")
            print(f"    align range: min={amin:+.3f}  med={amed:+.3f}  max={amax:+.3f}")
            if crossings >= 5:
                print(f"    HYPOTHESIS (b) CONFIRMED: alignment crosses thr "
                      f"{crossings} times -> gate-chatter defeats L1")
            elif n_above == 0:
                print(f"    HYPOTHESIS (a),(b) FALSIFIED for held contact: "
                      f"alignment NEVER above thr "
                      f"-> gate would refuse on every tick if it ran")
            elif n_above < 10:
                print(f"    Brief admissible windows ({n_above} ticks) — "
                      f"gate would refuse on most ticks if it ran")

        # GATE-ALIGN events that DID fire
        if d["gate_aligns"]:
            print(f"  [GATE-ALIGN] refusals fired: {len(d['gate_aligns'])}")
            for (s, a, nh, gh) in d["gate_aligns"][:3]:
                print(f"    step={s}: align={a:+.3f} nhat_xy=({nh[0]:+.3f},{nh[1]:+.3f}) "
                      f"g_hat=({gh[0]:+.3f},{gh[1]:+.3f})")
        else:
            print(f"  [GATE-ALIGN] refusals fired: 0  "
                  f"(gate ran ZERO times during the run)")

        print()


if __name__ == "__main__":
    main()
