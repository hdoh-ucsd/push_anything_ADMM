"""
Box-velocity-vs-goal_dist braking-profile analysis for the kIK vs kPWL
overshoot-causation pass.

For each results/overshoot_causal_{kik,pwl}_seed{0,1,2}.txt run log:

  1. Parse [STEP] lines: step, sim_t, obj=(x,y,z), goal_dist.
  2. Build closing_rate = -d(goal_dist)/dt at each tick (positive = box
     is closing distance to goal; near zero = stationary; negative = box
     is moving AWAY from goal i.e. overshoot/recovery).
  3. Bin by goal_dist:
       - "approach" band   : 0.15 <= goal_dist < 0.25 (box is committed)
       - "braking"  band   : 0.05 <= goal_dist < 0.15 (deceleration window)
       - "in-goal"  band   : goal_dist < 0.05         (settled / oscillating)
     Report mean closing_rate per band.
  4. Detect overshoot: peak negative closing_rate after first time
     goal_dist drops below 0.12 m (box passed through goal vicinity then
     moved away). Magnitude in mm/s.

Verdict (printed at end):
    "BRAKES NEAR GOAL"     - braking-band closing_rate is substantially
                             LOWER than approach-band closing_rate
                             (tracker decelerates into goal)
    "DRIVES THROUGH GOAL"  - braking-band closing_rate is similar to or
                             HIGHER than approach-band (no deceleration)

Decision logic across seeds: if kPWL is "BRAKES" on >=2/3 seeds and kIK
is "DRIVES THROUGH" on >=2/3 seeds, the overshoot causation is CONFIRMED
and kPWL is the fix.  Otherwise the overshoot is FP-noise / deeper-fidelity
issue, not the tracker.
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
_RESULT_RE = re.compile(
    r"\[RESULT\].*final_obj_xy=\(([+-]?[\d.]+),\s*([+-]?[\d.]+)\)\s+goal_dist=([\d.]+)m"
)

_BANDS = [
    ("approach", 0.15, 0.25),
    ("braking",  0.05, 0.15),
    ("in_goal",  0.00, 0.05),
]


def parse_run(log_path: Path) -> dict:
    """Parse one [STEP] stream into per-tick arrays."""
    steps, ts, obj_xy, goal_d, mode = [], [], [], [], []
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
            r = _RESULT_RE.search(line)
            if r:
                final = (float(r.group(1)), float(r.group(2)), float(r.group(3)))
    if not steps:
        return {"empty": True}
    ts = np.array(ts)
    goal_d = np.array(goal_d)
    obj_xy = np.array(obj_xy)
    dt = np.diff(ts)
    # closing rate at tick k: -(goal_d[k+1] - goal_d[k]) / dt[k]
    closing = np.zeros_like(goal_d)
    closing[:-1] = -np.diff(goal_d) / np.where(dt == 0, 1e-9, dt)
    closing[-1] = closing[-2] if len(closing) > 1 else 0.0
    return {
        "empty":   False,
        "steps":   np.array(steps),
        "ts":      ts,
        "obj_xy":  obj_xy,
        "goal_d":  goal_d,
        "mode":    np.array(mode),
        "closing": closing,
        "final":   final,
    }


def band_stats(run: dict) -> dict:
    """Per-band closing-rate stats + overshoot detection."""
    out = {}
    g = run["goal_d"]
    c = run["closing"]
    for name, lo, hi in _BANDS:
        mask = (g >= lo) & (g < hi)
        if mask.sum() == 0:
            out[name] = {"n": 0, "mean_mmps": None, "med_mmps": None,
                         "peak_mmps": None}
            continue
        cs = c[mask] * 1000.0  # m/s -> mm/s
        out[name] = {
            "n": int(mask.sum()),
            "mean_mmps": float(np.mean(cs)),
            "med_mmps":  float(np.median(cs)),
            "peak_mmps": float(np.max(cs)),
        }

    # Overshoot detection: after first time goal_d crosses 0.12 m, the
    # most-negative closing rate (= rate at which box is moving AWAY from goal)
    below = np.where(g < 0.12)[0]
    if below.size:
        i0 = int(below[0])
        post = c[i0:] * 1000.0
        out["overshoot"] = {
            "first_below_step": int(run["steps"][i0]),
            "first_below_t":    float(run["ts"][i0]),
            "peak_recede_mmps": float(-np.min(post)) if post.size else None,
            "min_goal_d_after": float(np.min(g[i0:])),
        }
    else:
        out["overshoot"] = None
    return out


def fmt_band(b):
    if b["n"] == 0:
        return "  (no ticks)"
    return (f"n={b['n']:4d}  "
            f"mean={b['mean_mmps']:+6.1f} mm/s  "
            f"med={b['med_mmps']:+6.1f}  "
            f"peak={b['peak_mmps']:+6.1f}")


def classify(stats: dict) -> str:
    a = stats["approach"]
    br = stats["braking"]
    if a["mean_mmps"] is None or br["mean_mmps"] is None:
        return "INSUFFICIENT (didn't reach band)"
    if a["mean_mmps"] <= 5.0:
        return "STALLED (never approached)"
    ratio = br["mean_mmps"] / a["mean_mmps"]
    if ratio < 0.50:
        return f"BRAKES (braking/approach={ratio:.2f})"
    return f"DRIVES THROUGH (braking/approach={ratio:.2f})"


def main():
    results = {}
    for tracker in ["pwl", "kik"]:
        for seed in [0, 1, 2]:
            stem = f"overshoot_causal_{tracker}_seed{seed}"
            log = _RESULTS / f"{stem}.txt"
            if not log.is_file():
                results[(tracker, seed)] = {"missing": True}
                continue
            run = parse_run(log)
            if run.get("empty"):
                results[(tracker, seed)] = {"empty": True}
                continue
            stats = band_stats(run)
            results[(tracker, seed)] = {
                "stats":     stats,
                "final":     run["final"],
                "n_ticks":   len(run["steps"]),
                "verdict":   classify(stats),
                "min_g_d":   float(np.min(run["goal_d"])),
                "max_g_d":   float(np.max(run["goal_d"])),
                "end_g_d":   float(run["goal_d"][-1]),
            }

    print("=" * 78)
    print("OVERSHOOT-CAUSATION VELOCITY-PROFILE REPORT")
    print(f"goal=(-0.30, 0.00); --task-id 4; --max-time 12; --seed in 0/1/2")
    print(f"closing_rate = -d(goal_dist)/dt; positive=closing, "
          f"negative=receding (overshoot)")
    print("=" * 78)
    for tracker in ["pwl", "kik"]:
        for seed in [0, 1, 2]:
            r = results[(tracker, seed)]
            tag = f"{tracker.upper():<3s} seed={seed}"
            if r.get("missing"):
                print(f"\n--- {tag}: MISSING run log")
                continue
            if r.get("empty"):
                print(f"\n--- {tag}: empty (no [STEP] lines)")
                continue
            print(f"\n--- {tag} | n_ticks={r['n_ticks']}  "
                  f"goal_d range=[{r['min_g_d']:.3f}, {r['max_g_d']:.3f}]  "
                  f"end={r['end_g_d']:.3f}")
            s = r["stats"]
            for name, _, _ in _BANDS:
                print(f"    {name:8s}: {fmt_band(s[name])}")
            ov = s["overshoot"]
            if ov is not None:
                print(f"    overshoot: first_below_0.12_at_t={ov['first_below_t']:.2f}s  "
                      f"peak_recede={ov['peak_recede_mmps']:+5.1f} mm/s  "
                      f"min_goal_d_after={ov['min_goal_d_after']:.3f} m")
            else:
                print("    overshoot: box never crossed goal_d=0.12 m")
            fp = r["final"]
            if fp is None:
                print(f"    final: (run timed out — no [RESULT] line)")
            else:
                print(f"    final_goal_d={fp[2]:.4f} m  final_obj_xy=({fp[0]:+.4f},{fp[1]:+.4f})")
            print(f"    VERDICT: {r['verdict']}")

    print()
    print("=" * 78)
    print("CROSS-SEED VERDICT TABLE")
    print("=" * 78)
    print(f"  {'seed':5s}  {'PWL braking/approach':30s}  {'kIK braking/approach':30s}")
    for seed in [0, 1, 2]:
        rpwl = results[("pwl", seed)]
        rkik = results[("kik", seed)]
        vp = rpwl.get("verdict", "MISS")
        vk = rkik.get("verdict", "MISS")
        print(f"  {seed:<5d}  {vp:30s}  {vk:30s}")

    print()
    pwl_brakes = sum(1 for s in [0,1,2]
                     if results[("pwl", s)].get("verdict", "").startswith("BRAKES"))
    kik_drives = sum(1 for s in [0,1,2]
                     if results[("kik", s)].get("verdict", "").startswith("DRIVES"))
    print(f"PWL brakes on {pwl_brakes}/3 seeds; kIK drives through on {kik_drives}/3 seeds.")
    if pwl_brakes >= 2 and kik_drives >= 2:
        print("\nROUTING: overshoot causation CONFIRMED — kPWL is the fix.")
    elif pwl_brakes <= 1 and kik_drives <= 1:
        print("\nROUTING: NO braking-profile separation — overshoot is FP-noise / "
              "deeper-fidelity, NOT the tracker.  kPWL conforms but does not "
              "structurally fix overshoot.")
    else:
        print("\nROUTING: partial/mixed signal — manual inspection of per-seed "
              "profiles required.")


if __name__ == "__main__":
    sys.exit(main())
