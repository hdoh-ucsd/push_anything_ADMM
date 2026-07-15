"""Stage 2 L1 sweep verdict — pre-registered SCs + the discriminators.

Reads stage2_L1_sweep/seed{0-4}_stage2_L1.log (commit 4285814,
entry_align_threshold=0.7).

Pre-registered (docs/superpowers/plans/2026-06-01-stage2-L1-gate-cardinality.md):
  SC-L1-cardinal     : every c3-admission has alignment > 0.7 (gate enforces).
  SC-L1-lambda       : projection foundation unconditionally sound.
                       median lcp_res_max ≤ 5e-07, p99 ≤ 5e-06, zero seed-3-
                       style cracks (zero ticks with lcp_res > 1e-05 OR λ_n > 10).
  SC-L1-noregress-working : seeds 0/3/4 still admit +x AND final.x stays at
                       Stage-1 levels (seed 0 < -0.45, seed 3 < -0.50,
                       seed 4 < -0.26).
  SC-L1-variance      : final_obj_xy spread shrinks vs Stage 1 (controller-
                       property signal — not just pass count).
  SC-distributional   : 5 seeds, no cherry-pick.

Three outcomes:
  A : seeds 1/2/4 swing to +x face, cardinal+goal-aligned, push -x → L1 sufficient.
  B : seeds 1/2/4 STOP CONTACTING (gate refuses, approach can't deliver +x)
      → routing to L2, NOT failure.
  C : working seeds 0/3/4 lose contact → threshold too strict (unlikely; alignment ≈1.0).

Discriminator (structural-vs-parametric L2):
  Count [GATE-ALIGN] refusal events per seed; cluster into refuse→admit→refuse
  cycles. Single long refuse window = structural-L2; oscillation = structural-L2
  with frustration; refuse → eventual cardinal admission → success = parametric-L2.
"""
from __future__ import annotations
import re
from pathlib import Path

REPO = Path("/root/push_anything_ADMM")
SWEEP = REPO / "stage2_L1_sweep"
STAGE1 = REPO / "altitude_hold_sweep"

STAGE1_FINAL_XY = {
    0: (-0.4959, 0.1876),
    1: (+0.0218, 0.3256),
    2: (-0.0261, 0.3186),
    3: (-0.5519, 0.1355),
    4: (-0.3136, 0.2805),
}
SC_NOREG_THRESH = {0: -0.45, 3: -0.50, 4: -0.26}
SC_MODEB_FINAL_X = -0.05
LAM_VIOL = 5.0
LAM_CRACK = 10.0
LCP_CRACK = 1.0e-05

RE_GATE = re.compile(
    r"^\[GATE-CONTACT\]\s+step=(\d+).*?"
    r"n_face_out=\(([+\-0-9.,e]+)\).*?A_is_ee=(\d).*?"
    r"box_p=\(([+\-0-9.,e]+)\)"
)
RE_C3 = re.compile(
    r"^\[C3\+\]\s+step=(\d+)\s+\|u\[0\]\|=\S+\s+"
    r"λ_n_max=([0-9.eE+-]+).*?lcp_res_max=([0-9.eE+-]+)"
)
RE_ALIGN = re.compile(
    r"^\[GATE-ALIGN\]\s+step=(\d+)\s+refused:\s+align=([+\-0-9.eE]+)\s+"
    r"<=\s+thr=([0-9.eE+-]+)\s+nhat_xy=\(([+\-0-9.,e]+)\)"
)
RE_RESULT = re.compile(
    r"^\[RESULT\]\s+method=\S+\s+final_obj_xy=\(([+\-0-9.,e ]+)\)\s+"
    r"goal_dist=([0-9.eE+-]+)m\s+success=(\w+)"
)


def parse(path):
    ticks = []
    inner_lams = []
    inner_res = []
    align_refusals = []   # list of (step, align, nhat_xy)
    result = None
    with path.open() as f:
        for line in f:
            mg = RE_GATE.match(line)
            if mg:
                nfo = tuple(float(x) for x in mg.group(2).split(","))
                bp = tuple(float(x) for x in mg.group(4).split(","))
                ticks.append((int(mg.group(1)), int(mg.group(3)), nfo, bp[:2]))
                continue
            mc = RE_C3.match(line)
            if mc:
                inner_lams.append(float(mc.group(2)))
                inner_res.append(float(mc.group(3)))
                continue
            ma = RE_ALIGN.match(line)
            if ma:
                align_refusals.append((
                    int(ma.group(1)), float(ma.group(2)),
                    tuple(float(x) for x in ma.group(4).split(",")),
                ))
                continue
            mr = RE_RESULT.match(line)
            if mr:
                xy = [float(x) for x in mr.group(1).split(",")]
                result = (xy[0], xy[1], float(mr.group(2)), mr.group(3))
    return ticks, inner_lams, inner_res, align_refusals, result


def first_contact_face(ticks):
    """First A_is_ee=1 tick's n_face_out + dominant axis."""
    for step, A, nfo, _ in ticks:
        if A == 1:
            nx, ny, nz = nfo
            if abs(nx) > 0.9: face = "+x" if nx > 0 else "-x"
            elif abs(ny) > 0.9: face = "+y" if ny > 0 else "-y"
            else: face = "off-cardinal"
            return step, face, nfo
    return None, None, None


def refuse_reapproach_cycles(refusals):
    """Count clusters of [GATE-ALIGN] refusals: a "cycle" is a contiguous
    run of refusals (consecutive ticks within 5 of each other). Distinct
    cycles == distinct refuse-reapproach attempts."""
    if not refusals:
        return 0, []
    steps = sorted(r[0] for r in refusals)
    clusters = [[steps[0]]]
    for s in steps[1:]:
        if s - clusters[-1][-1] <= 5:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return len(clusters), [(c[0], c[-1], len(c)) for c in clusters]


def chatter_count(refusals, ticks):
    """Detect alternating admit/refuse within a few ticks (gate flickering
    around the 0.7 threshold). Counts adjacent refusal pairs with gap > 0
    (i.e. there was at least one non-refusal between them) within 3 ticks."""
    if len(refusals) < 2:
        return 0
    steps = sorted(r[0] for r in refusals)
    chatter = 0
    for a, b in zip(steps, steps[1:]):
        if 1 < b - a <= 3:
            chatter += 1
    return chatter


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    def q(p):
        i = max(0, min(n-1, int(round(p*(n-1)))))
        return xs[i]
    return {"min": xs[0], "med": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1]}


def main():
    print("Stage 2 L1 sweep verdict (commit 4285814, entry_align_threshold=0.7)\n")

    results = {}
    for seed in [0, 1, 2, 3, 4]:
        log = SWEEP / f"seed{seed}_stage2_L1.log"
        if not log.exists():
            print(f"seed={seed}: log missing"); continue
        ticks, inner_lams, inner_res, refusals, result = parse(log)
        n_contact = sum(1 for t in ticks if t[1] == 1)
        first_step, first_face, first_nfo = first_contact_face(ticks)
        n_cycles, cycle_detail = refuse_reapproach_cycles(refusals)
        chatter = chatter_count(refusals, ticks)
        qres = quantiles(inner_res)
        qlam = quantiles(inner_lams)
        n_lam_viol = sum(1 for x in inner_lams if x >= LAM_VIOL)
        n_lam_crack = sum(1 for x in inner_lams if x >= LAM_CRACK)
        n_lcp_crack = sum(1 for x in inner_res if x >= LCP_CRACK)

        results[seed] = {
            "ticks": ticks, "result": result,
            "n_contact": n_contact, "first": (first_step, first_face, first_nfo),
            "n_refusals": len(refusals), "n_cycles": n_cycles,
            "cycle_detail": cycle_detail, "chatter": chatter,
            "qres": qres, "qlam": qlam,
            "n_lam_viol": n_lam_viol, "n_lam_crack": n_lam_crack,
            "n_lcp_crack": n_lcp_crack,
        }

        print(f"=== seed={seed} ===")
        if result:
            final_x, final_y, gd, succ = result
            stage1_x, stage1_y = STAGE1_FINAL_XY[seed]
            dx = final_x - stage1_x
            print(f"  RESULT: final_obj_xy=({final_x:+.4f}, {final_y:+.4f})  goal_dist={gd:.4f}  success={succ}")
            print(f"    Stage 1 baseline: ({stage1_x:+.4f}, {stage1_y:+.4f})  Δx = {dx:+.4f}")
        print(f"  contact_ticks={n_contact}  first_contact: step={first_step} face={first_face} nfo={first_nfo}")
        print(f"  [GATE-ALIGN] refusals: {len(refusals)}  distinct cycles: {n_cycles}  chatter pairs: {chatter}")
        if cycle_detail and len(cycle_detail) <= 8:
            for s, e, n in cycle_detail:
                print(f"    cycle: steps [{s}..{e}] len={n}")
        elif cycle_detail:
            print(f"    first 4 cycles:")
            for s, e, n in cycle_detail[:4]:
                print(f"    cycle: steps [{s}..{e}] len={n}")
            print(f"    ... ({len(cycle_detail)-4} more)")
        if qres:
            print(f"  lcp_res_max: med={qres['med']:.2e}  p90={qres['p90']:.2e}  p99={qres['p99']:.2e}  max={qres['max']:.2e}")
        if qlam:
            print(f"  λ_n_max:     med={qlam['med']:.3f}  p99={qlam['p99']:.3f}  max={qlam['max']:.3f}  "
                  f"≥5: {n_lam_viol}  ≥10: {n_lam_crack}  lcp_res>1e-5: {n_lcp_crack}")
        print()

    # Variance shrink — Stage 1 vs L1
    print("=" * 70)
    print("SC-L1-VARIANCE (controller-property signal)")
    print("=" * 70)
    stage1_x = [STAGE1_FINAL_XY[s][0] for s in [0,1,2,3,4]]
    stage1_y = [STAGE1_FINAL_XY[s][1] for s in [0,1,2,3,4]]
    L1_x = [results[s]["result"][0] for s in [0,1,2,3,4] if results.get(s) and results[s]["result"]]
    L1_y = [results[s]["result"][1] for s in [0,1,2,3,4] if results.get(s) and results[s]["result"]]
    def var(xs):
        if not xs:
            return float("nan")
        m = sum(xs)/len(xs)
        return sum((x-m)**2 for x in xs)/len(xs)
    print(f"  Stage 1 (β=0): var(final.x)={var(stage1_x):.5f}  var(final.y)={var(stage1_y):.5f}  (n=5)")
    print(f"  L1 (4285814):  var(final.x)={var(L1_x):.5f}  var(final.y)={var(L1_y):.5f}  (n={len(L1_x)})")
    if L1_x and var(stage1_x) > 0:
        print(f"  spread shrink: x {(1 - var(L1_x)/var(stage1_x))*100:+.1f}%   y {(1 - var(L1_y)/var(stage1_y))*100:+.1f}%")
    else:
        print(f"  spread shrink: insufficient L1 data (need all 5 seeds)")
    # First-contact face consistency
    n_plusx = sum(1 for s in [0,1,2,3,4] if results.get(s) and results[s]["first"][1] == "+x")
    print(f"  first-contact +x face: {n_plusx}/5 (Stage 1 baseline was 3/5)")
    print()

    # SC summary
    print("=" * 70)
    print("PRE-REGISTERED SC EVALUATION")
    print("=" * 70)
    print()
    print("SC-L1-lambda (foundation unconditionally sound):")
    for seed in [0,1,2,3,4]:
        r = results.get(seed)
        if not r or not r["qres"]: continue
        ok_med = r["qres"]["med"] <= 5e-7
        ok_p99 = r["qres"]["p99"] <= 5e-6
        ok_crack = (r["n_lam_crack"] == 0 and r["n_lcp_crack"] == 0)
        verdict = "PASS" if (ok_med and ok_p99 and ok_crack) else "FAIL"
        print(f"  seed {seed}: {verdict}  med={r['qres']['med']:.2e}  p99={r['qres']['p99']:.2e}  "
              f"λ_n≥10: {r['n_lam_crack']}  lcp_res>1e-5: {r['n_lcp_crack']}")
    print()
    print("SC-L1-noregress-working (seeds 0,3,4 hold):")
    for seed in [0,3,4]:
        r = results.get(seed)
        if not r or not r["result"]: continue
        final_x = r["result"][0]
        thr = SC_NOREG_THRESH[seed]
        ok = final_x < thr
        face = r["first"][1]
        print(f"  seed {seed}: final.x={final_x:+.4f} {'<' if ok else '≥'} {thr}  → {'PASS' if ok else 'FAIL'}  "
              f"first-contact face={face}")
    print()
    print("SC-modeB-convert (seeds 1,2 — outcome A check):")
    for seed in [1,2]:
        r = results.get(seed)
        if not r or not r["result"]: continue
        final_x = r["result"][0]
        ok = final_x < SC_MODEB_FINAL_X
        face = r["first"][1]
        print(f"  seed {seed}: final.x={final_x:+.4f} {'<' if ok else '≥'} {SC_MODEB_FINAL_X}  → "
              f"{'CONVERT (outcome A)' if ok else 'no convert (outcome B)'}  first-contact face={face}")
    print()
    print("Structural-vs-parametric L2 discriminator (refuse-reapproach cycles):")
    for seed in [1,2,4]:
        r = results.get(seed)
        if not r: continue
        print(f"  seed {seed}: refusals={r['n_refusals']}  cycles={r['n_cycles']}  chatter={r['chatter']}  "
              f"final.x={r['result'][0]:+.4f}")


if __name__ == "__main__":
    main()
