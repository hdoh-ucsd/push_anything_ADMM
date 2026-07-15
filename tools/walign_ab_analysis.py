"""
w_align A/B analysis.

For each results/walign_ab_{wa0,wa30k}_seed{0,1,2}.txt run log:
  - final obj_xy + goal_dist
  - early-window face pick: first [GS-tgt] entry's p_repos = (x, y, z)
  - dominant face label of the first K [GS-tgt] entries (which face the
    dispatcher kept picking before any contact)
  - max obj_y reached during the run (north-push signature)

Then cross-arm verdict:
    "north-push KILLED": seeds 1/2 final obj_y back toward 0 with w_align=30000
                         -> w_align is NOT free at the dispatcher; PARTIAL REVERT.
    "north-push PERSISTS": w_align=30000 doesn't fix the failure
                          -> w_align exonerated; dig elsewhere (entry_align next).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_RESULTS = Path("/root/push_anything_ADMM/results")

_STEP_RE = re.compile(
    r"\[STEP\]\s+step=(\d+)\s+mode=(\S+)\s+t=([\d.]+)s\s+"
    r"ee=\(([+-]?[\d.]+),([+-]?[\d.]+),([+-]?[\d.]+)\)\s+"
    r"obj=\(([+-]?[\d.]+),([+-]?[\d.]+),([+-]?[\d.]+)\)\s+"
    r"goal_dist=([\d.]+)m"
)
_GSTGT_RE = re.compile(
    r"\[GS-tgt\]\s+step=(\d+)\s+ee=\([^)]+\)\s+"
    r"p_repos=\(([+-]?[\d.]+),([+-]?[\d.]+),([+-]?[\d.]+)\)"
)
_RESULT_RE = re.compile(
    r"\[RESULT\].*final_obj_xy=\(([+-]?[\d.]+),\s*([+-]?[\d.]+)\)\s+goal_dist=([\d.]+)m"
)


def _face_label(p_repos_xy: tuple[float, float],
                box_xy: tuple[float, float] = (0.0, 0.0)) -> str:
    """Classify which face the EE-sample sits on (relative to box center).

    Push direction from EE -> box is opposite the (p_repos - box_xy) vector,
    so seat at SOUTH face -> push NORTH on box, etc.
    """
    dx = p_repos_xy[0] - box_xy[0]
    dy = p_repos_xy[1] - box_xy[1]
    if abs(dx) > abs(dy):
        # East/West face dominant
        if dx > 0:
            return "EAST  (pushes box WEST  toward goal)"
        else:
            return "WEST  (pushes box EAST  away from goal)"
    else:
        if dy > 0:
            return "NORTH (pushes box SOUTH)"
        else:
            return "SOUTH (pushes box NORTH)"


def parse_run(log_path: Path) -> dict:
    steps, ts, obj_xy, goal_d, mode = [], [], [], [], []
    p_repos_early = []
    final = None
    with open(log_path) as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                mode.append(m.group(2))
                ts.append(float(m.group(3)))
                obj_xy.append((float(m.group(7)), float(m.group(8))))
                goal_d.append(float(m.group(10)))
                continue
            g = _GSTGT_RE.search(line)
            if g:
                if len(p_repos_early) < 40:
                    p_repos_early.append((int(g.group(1)),
                                          (float(g.group(2)),
                                           float(g.group(3)),
                                           float(g.group(4)))))
                continue
            r = _RESULT_RE.search(line)
            if r:
                final = (float(r.group(1)), float(r.group(2)), float(r.group(3)))
    if not steps:
        return {"empty": True}
    return {
        "empty":   False,
        "n_ticks": len(steps),
        "obj_xy":  np.array(obj_xy),
        "goal_d":  np.array(goal_d),
        "mode":    np.array(mode),
        "p_repos_early": p_repos_early,
        "final":   final,
    }


def main():
    results = {}
    # wa0 arm reuses prior overshoot_causal_pwl runs (same YAML — landed
    # conformance with w_align=0, entry_align=0, traj_type=kPiecewiseLinear).
    # wa30k arm is the new restored-w_align run.
    for arm in ["wa0", "wa30k"]:
        for seed in [0, 1, 2]:
            if arm == "wa0":
                stem = f"overshoot_causal_pwl_seed{seed}"
            else:
                stem = f"walign_ab_{arm}_seed{seed}"
            log = _RESULTS / f"{stem}.txt"
            if not log.is_file():
                results[(arm, seed)] = {"missing": True}
                continue
            run = parse_run(log)
            if run.get("empty"):
                results[(arm, seed)] = {"empty": True}
                continue

            obj_y = run["obj_xy"][:, 1]
            obj_x = run["obj_xy"][:, 0]
            results[(arm, seed)] = {
                "n_ticks":      run["n_ticks"],
                "final":        run["final"],
                "max_obj_y":    float(np.max(obj_y)),
                "min_obj_y":    float(np.min(obj_y)),
                "end_obj_xy":   (float(obj_x[-1]), float(obj_y[-1])),
                "end_goal_d":   float(run["goal_d"][-1]),
                "p_repos_early": run["p_repos_early"][:10],
            }

    print("=" * 84)
    print("w_align A/B  (kPWL, --task-id 4 west, --max-time 12, entry_align=0 in BOTH)")
    print("  wa0   = w_align=0.0     (current landed conformance)")
    print("  wa30k = w_align=30000.0 (restored, pre-conformance value)")
    print("=" * 84)

    for arm in ["wa0", "wa30k"]:
        for seed in [0, 1, 2]:
            r = results[(arm, seed)]
            tag = f"{arm:<5s} seed={seed}"
            if r.get("missing"):
                print(f"\n--- {tag}: MISSING")
                continue
            if r.get("empty"):
                print(f"\n--- {tag}: empty")
                continue
            print(f"\n--- {tag} | n_ticks={r['n_ticks']}")
            fp = r["final"]
            if fp is not None:
                print(f"    final_obj_xy=({fp[0]:+.4f}, {fp[1]:+.4f})  "
                      f"goal_dist={fp[2]:.4f} m")
            else:
                ex, ey = r["end_obj_xy"]
                print(f"    end_obj_xy=({ex:+.4f}, {ey:+.4f}) (no [RESULT] — timeout)  "
                      f"end_goal_d={r['end_goal_d']:.4f}")
            print(f"    obj_y range during run: "
                  f"[{r['min_obj_y']:+.4f}, {r['max_obj_y']:+.4f}]")
            print(f"    first 5 [GS-tgt] p_repos picks (face classification):")
            for step, p in r["p_repos_early"][:5]:
                face = _face_label((p[0], p[1]))
                print(f"      step={step:3d}  p_repos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
                      f"-> {face}")

    print()
    print("=" * 84)
    print("CROSS-ARM VERDICT")
    print("=" * 84)
    header = f"  {'seed':4s}  {'wa0  final_obj_y':18s}  {'wa30k final_obj_y':18s}  {'delta':>8s}"
    print(header)
    for seed in [0, 1, 2]:
        r0 = results[("wa0", seed)]
        rk = results[("wa30k", seed)]

        def fmt_y(r):
            if r.get("missing") or r.get("empty"):
                return "(no data)"
            if r["final"] is not None:
                return f"{r['final'][1]:+.4f}"
            return f"{r['end_obj_xy'][1]:+.4f} (to)"

        y0 = fmt_y(r0)
        yk = fmt_y(rk)
        if (not r0.get("missing") and not r0.get("empty")
            and not rk.get("missing") and not rk.get("empty")):
            y0v = r0["final"][1] if r0["final"] is not None else r0["end_obj_xy"][1]
            ykv = rk["final"][1] if rk["final"] is not None else rk["end_obj_xy"][1]
            delta = f"{ykv - y0v:+.4f}"
        else:
            delta = "—"
        print(f"  {seed:<4d}  {y0:18s}  {yk:18s}  {delta:>8s}")

    print()
    print("Interpretation (seeds 1/2 are the test cases for north-push):")
    n_killed = 0
    for seed in [1, 2]:
        r0 = results[("wa0", seed)]
        rk = results[("wa30k", seed)]
        if any(x.get("missing") or x.get("empty") for x in (r0, rk)):
            continue
        y0v = r0["final"][1] if r0["final"] is not None else r0["end_obj_xy"][1]
        ykv = rk["final"][1] if rk["final"] is not None else rk["end_obj_xy"][1]
        if y0v > 0.10 and ykv < 0.10:
            print(f"  seed {seed}: NORTH-PUSH KILLED "
                  f"(wa0 obj_y={y0v:+.3f}, wa30k obj_y={ykv:+.3f})")
            n_killed += 1
        elif y0v > 0.10 and ykv > 0.10:
            print(f"  seed {seed}: NORTH-PUSH PERSISTS "
                  f"(wa0 obj_y={y0v:+.3f}, wa30k obj_y={ykv:+.3f})")
        else:
            print(f"  seed {seed}: not clearly diagnostic "
                  f"(wa0 obj_y={y0v:+.3f}, wa30k obj_y={ykv:+.3f})")

    print()
    if n_killed >= 2:
        print("ROUTING: w_align=30000 KILLS the seed-1/2 north-push -> "
              "w_align is NOT free at the dispatcher level for these seeds.")
        print("         PARTIAL REVERT: keep traj_type=kPiecewiseLinear (the kPWL")
        print("         conformance is fine), but RESTORE w_align=30000 in YAML:23.")
        print("         The 'w_align is free' verdict held in the EARLY WINDOW on")
        print("         seeds 0/1 only — NOT at task outcome on seeds 1/2; the")
        print("         early-window ablation was structurally blind to LATE-phase")
        print("         wrong-face selection.")
    elif n_killed == 0:
        print("ROUTING: w_align=30000 does NOT fix the north-push -> w_align EXONERATED.")
        print("         Next test: entry_align_threshold=0.7 restored (alone or +w_align).")
    else:
        print(f"ROUTING: partial signal (n_killed={n_killed}/2 seeds) — manual inspection.")


if __name__ == "__main__":
    sys.exit(main())
