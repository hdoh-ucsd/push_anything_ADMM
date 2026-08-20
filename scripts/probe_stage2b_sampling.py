"""Read-only Stage 2B verdict parser — evaluate against pre-registered SCs.

Pre-registration (docs/superpowers/plans/2026-06-01-stage2b-sampling-bias-mode-b.md):
  SC-modeB-convert : seeds 1, 2 contact +x face (n_face_out_x > 0.9 on ≥1 tick in
                     a ≥25-tick sustained window) AND final box.x < -0.05.
  SC-noregress-working : seeds 0, 3, 4 still contact +x face AND final box.x stays
                     within +0.05 m of their Stage-1 baselines (0: <-0.45, 3:
                     <-0.50, 4: <-0.26).
  SC-noregress-lambda  : median lcp_res_max ≤ 5e-07 per seed; p99 ≤ 5e-06.
  SC-distributional    : all 5 seeds reported, no cherry-pick.

NOT-A-FAILURE: goal-overshoot on converted seeds 1, 2 — that is goal-stop's job
in Stage 2C, separate plan.
"""

from __future__ import annotations
import re
from pathlib import Path

REPO = Path("/root/push_anything_ADMM")
SWEEP = REPO / "stage2b_sampling_sweep"

# Stage-1 baselines (per altitude_hold_sweep/sweep.summary)
STAGE1_FINAL_X = {0: -0.4959, 1: +0.0218, 2: -0.0261, 3: -0.5519, 4: -0.3136}
SC_NOREG_THRESH = {0: -0.45, 3: -0.50, 4: -0.26}
SC_MODEB_FINAL_X = -0.05
LAM_VIOL = 5.0

RE_GATE = re.compile(
    r"^\[GATE-CONTACT\]\s+step=(\d+).*?"
    r"n_face_out=\(([+\-0-9.,e]+)\).*?A_is_ee=(\d).*?"
    r"box_p=\(([+\-0-9.,e]+)\)"
)
RE_C3 = re.compile(
    r"^\[C3\+\]\s+step=(\d+)\s+\|u\[0\]\|=([0-9.eE+-]+)N\s+"
    r"λ_n_max=([0-9.eE+-]+)\s+η_n_max=([0-9.eE+-]+)\s+"
    r"primal=([0-9.eE+-]+)\s+iters=\d+/\d+\s+"
    r"lcp_res_max=([0-9.eE+-]+)"
)
RE_RESULT = re.compile(
    r"^\[RESULT\]\s+method=\S+\s+final_obj_xy=\(([+\-0-9.,e ]+)\)\s+"
    r"goal_dist=([0-9.eE+-]+)m\s+success=(\w+)"
)


def parse(path):
    ticks = []        # list of (gate_step, A_is_ee, n_face_out_xyz, box_xy)
    inner_lams = []   # all lam_n_max per [C3+] line (drop sub-step grouping for SC math)
    inner_residuals = []
    result = None
    with path.open() as f:
        for line in f:
            mg = RE_GATE.match(line)
            if mg:
                # Groups: (step, n_face_out, A_is_ee, box_p)
                nfo = [float(x) for x in mg.group(2).split(",")]
                bp = [float(x) for x in mg.group(4).split(",")]
                ticks.append((int(mg.group(1)), int(mg.group(3)), tuple(nfo), (bp[0], bp[1])))
                continue
            mc = RE_C3.match(line)
            if mc:
                inner_lams.append(float(mc.group(3)))
                inner_residuals.append(float(mc.group(6)))
                continue
            mr = RE_RESULT.match(line)
            if mr:
                xy = [float(x) for x in mr.group(1).split(",")]
                result = (xy[0], xy[1], float(mr.group(2)), mr.group(3))
    return ticks, inner_lams, inner_residuals, result


def contact_runs(ticks):
    runs = []
    i = 0
    while i < len(ticks):
        if ticks[i][1] == 1:
            j = i
            while j + 1 < len(ticks) and ticks[j + 1][1] == 1:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def has_plus_x_contact_in_sustained_window(ticks, runs, x_thresh=0.9, min_len=25):
    """SC operationalization: at least one ≥25-tick contact window where
    n_face_out x-component > 0.9 on at least one tick."""
    hits = []
    for s, e in runs:
        if (e - s + 1) < min_len:
            continue
        for k in range(s, e + 1):
            nx = ticks[k][2][0]
            if nx > x_thresh:
                hits.append((s, e, k, nx))
                break
    return hits


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    def q(p):
        i = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs[i]
    return {"min": xs[0], "med": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1]}


def n_face_out_buckets(ticks):
    """Bucket ALL contact ticks by which face was the live contact face."""
    pos_x = neg_x = pos_y = neg_y = mixed = none = 0
    for _, aee, nfo, _ in ticks:
        if aee == 0:
            none += 1
            continue
        nx, ny, _ = nfo
        if abs(nx) > 0.9 and nx > 0: pos_x += 1
        elif abs(nx) > 0.9 and nx < 0: neg_x += 1
        elif abs(ny) > 0.9 and ny > 0: pos_y += 1
        elif abs(ny) > 0.9 and ny < 0: neg_y += 1
        else: mixed += 1
    return {"+x": pos_x, "-x": neg_x, "+y": pos_y, "-y": neg_y, "mixed": mixed, "none": none}


def main():
    print("Stage 2B sampler-bias verdict (commit 5b0dc91, face_bias_strength=2.0)\n")

    all_seeds = []
    sc_modeb = {}; sc_noregwork = {}; sc_noreglam = {}

    for seed in [0, 1, 2, 3, 4]:
        log = SWEEP / f"seed{seed}_stage2b_sampling.log"
        if not log.exists():
            print(f"seed={seed}: log missing ({log})"); continue
        ticks, inner_lams, inner_res, result = parse(log)
        runs = contact_runs(ticks)
        n_contact = sum(1 for t in ticks if t[1] == 1)

        max_run = max((e - s + 1 for s, e in runs), default=0)
        buckets = n_face_out_buckets(ticks)
        sustained_plus_x = has_plus_x_contact_in_sustained_window(ticks, runs)

        qres = quantiles(inner_res)
        qlam = quantiles(inner_lams)

        if result:
            final_x, final_y, gd, succ = result
            print(f"=== seed={seed} ===")
            print(f"  RESULT: final_obj_xy=({final_x:+.4f}, {final_y:+.4f})  goal_dist={gd:.4f}  success={succ}")
            print(f"    Stage-1 baseline final.x = {STAGE1_FINAL_X[seed]:+.4f}")
        else:
            print(f"=== seed={seed} ===")
            print(f"  RESULT: missing")
            final_x = None

        print(f"  contact_ticks={n_contact}  contact_runs={len(runs)}  max_run_len={max_run}")
        print(f"  n_face_out buckets (contact ticks only): "
              f"+x={buckets['+x']}  -x={buckets['-x']}  +y={buckets['+y']}  -y={buckets['-y']}  mixed={buckets['mixed']}")
        if sustained_plus_x:
            s, e, k, nx = sustained_plus_x[0]
            print(f"  sustained-window +x contact: YES  first match at run [{s}..{e}] tick {k}  n_face_out.x={nx:+.3f}")
        else:
            print(f"  sustained-window +x contact: NO")
        if qres:
            print(f"  lcp_res_max  med={qres['med']:.2e}  p90={qres['p90']:.2e}  p99={qres['p99']:.2e}  max={qres['max']:.2e}")
        if qlam:
            n_viol = sum(1 for x in inner_lams if x >= LAM_VIOL)
            print(f"  λ_n_max      med={qlam['med']:.3f}  p99={qlam['p99']:.3f}  max={qlam['max']:.3f}  violations(≥5)={n_viol}")
        print()

        # SC evaluation
        if seed in (1, 2):
            ok_face = bool(sustained_plus_x)
            ok_motion = (final_x is not None and final_x < SC_MODEB_FINAL_X)
            sc_modeb[seed] = (ok_face, ok_motion)
        if seed in (0, 3, 4):
            ok_face = bool(sustained_plus_x)
            ok_motion = (final_x is not None and final_x < SC_NOREG_THRESH[seed])
            sc_noregwork[seed] = (ok_face, ok_motion)
        if qres:
            ok_lam = (qres["med"] <= 5e-7) and (qres["p99"] <= 5e-6)
            sc_noreglam[seed] = ok_lam

    # SC summary
    print("=" * 70)
    print("PRE-REGISTERED SC EVALUATION")
    print("=" * 70)
    print()
    print("SC-modeB-convert (seeds 1, 2 must convert):")
    for seed in (1, 2):
        if seed in sc_modeb:
            ok_face, ok_motion = sc_modeb[seed]
            print(f"  seed {seed}: +x face contact = {'PASS' if ok_face else 'FAIL'}  "
                  f"|  final.x < {SC_MODEB_FINAL_X} = {'PASS' if ok_motion else 'FAIL'}")
    print()
    print("SC-noregress-working (seeds 0, 3, 4 must hold):")
    for seed in (0, 3, 4):
        if seed in sc_noregwork:
            ok_face, ok_motion = sc_noregwork[seed]
            thr = SC_NOREG_THRESH[seed]
            print(f"  seed {seed}: +x face contact = {'PASS' if ok_face else 'FAIL'}  "
                  f"|  final.x < {thr} = {'PASS' if ok_motion else 'FAIL'}")
    print()
    print("SC-noregress-lambda (med ≤ 5e-7 AND p99 ≤ 5e-6):")
    for seed, ok in sorted(sc_noreglam.items()):
        print(f"  seed {seed}: {'PASS' if ok else 'FAIL'}")
    print()
    print("(NOT-A-FAILURE: goal-overshoot on converted seeds 1/2 is goal-stop's job, Stage 2C.)")


if __name__ == "__main__":
    main()
