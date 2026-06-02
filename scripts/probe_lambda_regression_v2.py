"""Extension of probe_lambda_regression.py:

  (a) Compute violation rate conditioned on contact-window length, so we can
      tell whether violations require *sustained* contact or appear at any
      contact tick.
  (b) Run all 5 seeds in both sweeps for a wider view (cheap re-parse).
  (c) Tie violation clusters to the worst single contact window — the long
      windows are where Mode-A overshoot likely arises, testing the
      shared-cause hypothesis.
"""

from __future__ import annotations
import sys
sys.path.insert(0, "/root/push_anything_ADMM/scripts")
from probe_lambda_regression import parse_log, contact_windows, LOG_DIRS, REPO, LAM_VIOL


def main():
    print(f"=== Violation rate vs window length (threshold = lam_n_max >= {LAM_VIOL}) ===")
    print()
    rows = []
    for label, (sub, fmt) in LOG_DIRS.items():
        for seed in [0, 1, 2, 3, 4]:
            p = REPO / sub / fmt.format(seed)
            if not p.exists():
                continue
            ticks = parse_log(p)
            runs = contact_windows(ticks)
            # bucket window lengths
            buckets = {
                "<= 25":  [r for r in runs if (r[1] - r[0] + 1) <= 25],
                "26-50":  [r for r in runs if 26 <= (r[1] - r[0] + 1) <= 50],
                "51-100": [r for r in runs if 51 <= (r[1] - r[0] + 1) <= 100],
                "> 100":  [r for r in runs if (r[1] - r[0] + 1) > 100],
            }
            row_buckets = {}
            for name, bs in buckets.items():
                n_ticks = sum(e - s + 1 for s, e in bs)
                n_viol = sum(1 for s, e in bs for i in range(s, e + 1) if ticks[i].violation)
                rate = n_viol / n_ticks if n_ticks else 0.0
                row_buckets[name] = (n_ticks, n_viol, rate)
            rows.append((label, seed, len(runs), row_buckets,
                         sum(1 for t in ticks if t.violation),
                         sum(1 for t in ticks if t.A_is_ee == 1)))

    print(f"{'sweep':9s} {'seed':4s} {'runs':4s}  "
          f"{'≤25 (n_t,n_v,rate%)':22s}  "
          f"{'26-50 (n_t,n_v,rate%)':22s}  "
          f"{'51-100 (n_t,n_v,rate%)':22s}  "
          f"{'>100 (n_t,n_v,rate%)':22s}  "
          f"{'total_viol':10s} {'n_contact':9s}")
    for label, seed, nr, buckets, n_viol, n_contact in rows:
        cells = []
        for name in ["<= 25", "26-50", "51-100", "> 100"]:
            n_t, n_v, r = buckets[name]
            cells.append(f"{n_t:>4d},{n_v:>3d},{r*100:5.2f}".ljust(22))
        print(f"{label:9s} {seed:>4d} {nr:>4d}  "
              + "  ".join(cells)
              + f"  {n_viol:>10d} {n_contact:>9d}")

    # Window-length analysis: do violations require long windows?
    print()
    print("=== Per-window: longest 3 contact runs in each Stage-1 seed (sorted) ===")
    for label, (sub, fmt) in LOG_DIRS.items():
        if label != "Stage1":
            continue
        for seed in [0, 1, 2, 3, 4]:
            p = REPO / sub / fmt.format(seed)
            ticks = parse_log(p)
            runs = contact_windows(ticks)
            ranked = sorted(runs, key=lambda r: -(r[1] - r[0] + 1))[:3]
            print(f"  Stage1 seed={seed} top-3 runs:")
            for s, e in ranked:
                length = e - s + 1
                n_v = sum(1 for i in range(s, e + 1) if ticks[i].violation)
                # lam_n_max trajectory: max lambda in this window
                lams = [ticks[i].lam_n_max for i in range(s, e + 1)]
                # box motion across the window
                start_box = ticks[s].box_xy
                end_box = ticks[e].box_xy
                dx = end_box[0] - start_box[0]
                dy = end_box[1] - start_box[1]
                disp = (dx * dx + dy * dy) ** 0.5
                print(f"     [{s:>4d}..{e:>4d}] len={length:>4d} viol={n_v:>3d}"
                      f"  max_lam_n={max(lams):.3f}"
                      f"  box_displacement={disp*1000:.1f}mm (dx,dy)=({dx*1000:+.1f},{dy*1000:+.1f})mm"
                      f"  start_box=({start_box[0]:+.3f},{start_box[1]:+.3f})"
                      f"  end_box=({end_box[0]:+.3f},{end_box[1]:+.3f})")

    # Shared-cause test: do violations cluster in the SAME long windows that drive box motion?
    # In Stage 1 seed 4 (overshoot mode), look at which window contains the
    # min-distance-to-goal tick and whether violations cluster there.
    print()
    print("=== Shared-cause test (Stage-1 seed 4 — Mode-A overshoot candidate) ===")
    p = REPO / "altitude_hold_sweep" / "seed4_altitude_hold.log"
    ticks = parse_log(p)
    runs = contact_windows(ticks)
    GOAL = (-0.5, 0.0)
    dists = [((t.box_xy[0] - GOAL[0]) ** 2 + (t.box_xy[1] - GOAL[1]) ** 2) ** 0.5 for t in ticks]
    min_i = min(range(len(dists)), key=lambda i: dists[i])
    print(f"  min-dist tick = {min_i}  dist = {dists[min_i]:.4f}m  box_xy = {ticks[min_i].box_xy}")
    # Which window?
    in_window = None
    for s, e in runs:
        if s <= min_i <= e:
            in_window = (s, e)
            break
    print(f"  min-dist tick {'inside' if in_window else 'outside'} contact window: {in_window}")
    # Violation indices
    viol_indices = [i for i, t in enumerate(ticks) if t.violation]
    if viol_indices:
        print(f"  violations span ticks: [{viol_indices[0]}..{viol_indices[-1]}]")
        # any violations in the min-dist window?
        if in_window:
            s, e = in_window
            v_in = [i for i in viol_indices if s <= i <= e]
            print(f"  violations inside min-dist window: {len(v_in)}/{len(viol_indices)}  indices={v_in[:10]}...")

    print()
    print("=== Per-tick velocity of |box_xy - goal| in the long-window: is overshoot continuous? ===")
    # Walk through the top long run for seed 4
    top_run = max(runs, key=lambda r: r[1] - r[0])
    s, e = top_run
    print(f"  top run [{s}..{e}] length={e-s+1}")
    sample = list(range(s, e + 1, max(1, (e - s) // 12)))
    print(f"  {'tick':>5s} {'A_is_ee':>7s} {'lam_n_max':>10s} {'dist_to_goal':>13s} {'box_xy':>22s} {'ee_xy':>22s}")
    for i in sample:
        t = ticks[i]
        d = ((t.box_xy[0] - GOAL[0]) ** 2 + (t.box_xy[1] - GOAL[1]) ** 2) ** 0.5
        viol_mark = "**" if t.violation else "  "
        print(f"  {i:>5d} {t.A_is_ee:>7d} {t.lam_n_max:>10.3f}{viol_mark} {d:>13.4f} "
              f"({t.box_xy[0]:+.4f},{t.box_xy[1]:+.4f})  ({t.ee_xyz[0]:+.4f},{t.ee_xyz[1]:+.4f})")


if __name__ == "__main__":
    main()
