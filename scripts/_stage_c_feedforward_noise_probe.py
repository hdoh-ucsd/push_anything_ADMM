"""§7.34 SOURCE/NOISE GATE — feedforward-acceleration probe.

ee-space x has NO acceleration state slot (verified at lcs_formulator.py:1196-1203
`x = [box_q(7), p_ee(3), box_v(6), v_ee(3)]`, N_X_NEW = 19). So a_ff for the
PD + a_ff = a_ff + Kp·p_err + Kd·v_err port would need a SECOND difference:
        a_ff[k] = (x_seq[2][16:19] - x_seq[1][16:19]) / dt_planner

That is a second derivative of a NON-CONVERGED (25/25) ADMM solution. The
question is whether the implied a_ff is bounded/smooth or oscillating/garbage.

We probe the existing §7.32 live log (faithful_desired_state_live/run.log,
the DISSOLVES build) — it logs the planner's predicted v_des per c3-mode
step, with the same x_seq source the feedforward would use. By taking
first-differences of consecutive v_des entries over c3-mode ticks we get a
DIRECT noise floor proxy for a_ff. (The second-difference in x_seq becomes
a first-difference once we already read out v.)

dt_planner = 0.05 s (from ci_mpc_c3plus.py:54 dt default + main.py canonical
config; the planner emits knots at 50 ms cadence; the SAME stride-bug guard
used for the velocity feedforward applies — feedforward acceleration uses
dt_planner not dt_ctrl).

The output below characterizes:
  • mean/median/max of |a_ff_component| across all c3-mode steps
  • how often it would CLIP at a defensive a_max threshold (10, 25, 50 m/s²)
  • sign-flip rate (oscillation diagnostic) — fraction of consecutive ticks
    where any component changed sign

Decision rule:
  PASS (build STEP 1) → a_ff is mostly bounded under 50 m/s² (sample-frame
    Cartesian Franka can plausibly track), sign-flip rate < ~30% (the
    planner is consistently predicting a direction).
  FAIL (STOP, decide mitigation) → noise dominates: |a_ff| typically
    saturates well above 50 m/s², sign-flip rate > 50%. Mitigation
    options: (i) feed conditionally (only when smooth), (ii) low-pass /
    clip the feedforward at low a_max, (iii) accept executor reconciled
    at pos+velocity and defer feedforward until ADMM converges.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import numpy as np

LOG = Path("faithful_desired_state_live/run.log")
DT_PLANNER = 0.05  # s — ci_mpc_c3plus.py default

RX = re.compile(
    r"^\[VFF\] step=(\d+) mode=c3 .* "
    r"v_des=\(([+\-\d\.]+),([+\-\d\.]+),([+\-\d\.]+)\)"
)


def collect_vff(log_path: Path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = RX.match(line)
            if not m:
                continue
            step = int(m.group(1))
            v = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])
            rows.append((step, v))
    return rows


def main() -> int:
    if not LOG.exists():
        print(f"missing log: {LOG}")
        return 2
    rows = collect_vff(LOG)
    print("=" * 84)
    print("§7.34 SOURCE/NOISE GATE — predicted-acceleration probe")
    print("=" * 84)
    print(f"log: {LOG}")
    print(f"c3-mode steps with [VFF] v_des emission: {len(rows)}")
    if len(rows) < 5:
        print("too few c3-mode samples to characterize")
        return 1

    # Sort by step (should already be, defensive)
    rows.sort(key=lambda r: r[0])
    steps = np.array([r[0] for r in rows])
    V = np.stack([r[1] for r in rows])  # (N, 3)

    # First-differences over CONSECUTIVE-tick c3 segments only. A gap in
    # step index means the dispatcher was in free mode between, which would
    # contaminate the implied a. We treat the planner's predicted a as
    # well-defined only across consecutive c3-mode ticks.
    diffs = []     # (step_k, step_k+1, a_component_vec, |a|)
    for i in range(len(rows) - 1):
        s_k, v_k = rows[i]
        s_k1, v_k1 = rows[i + 1]
        if s_k1 - s_k != 1:
            continue  # gap → not a usable a sample
        a = (v_k1 - v_k) / DT_PLANNER
        diffs.append((s_k, s_k1, a, float(np.linalg.norm(a))))

    if len(diffs) < 2:
        print("no consecutive-c3 pairs — feedforward acceleration cannot be probed "
              "from this log; would need a fresh run with denser c3-mode.")
        return 1

    A = np.stack([d[2] for d in diffs])    # (M, 3) implied a_ff per consecutive pair
    A_abs = np.abs(A)
    A_mag = np.linalg.norm(A, axis=1)
    print(f"consecutive-c3-mode a samples: {len(diffs)}")
    print(f"  component |a_ff|  (m/s²):  mean={A_abs.mean():.2f}  "
          f"median={np.median(A_abs):.2f}  max={A_abs.max():.2f}  p90={np.percentile(A_abs,90):.2f}")
    print(f"  vector  |a_ff|_2 (m/s²):  mean={A_mag.mean():.2f}  "
          f"median={np.median(A_mag):.2f}  max={A_mag.max():.2f}  p90={np.percentile(A_mag,90):.2f}")

    for thr in (10.0, 25.0, 50.0, 100.0):
        clip_rate = float(np.mean(A_abs >= thr))
        clip_rate_v = float(np.mean(A_mag >= thr))
        print(f"  fraction of components ≥ {thr:5.1f} m/s²: "
              f"{100*clip_rate:5.1f}% | vectors ≥ {thr:5.1f}: {100*clip_rate_v:5.1f}%")

    # Sign-flip rate — for each component, how often does Δa flip sign between
    # consecutive consecutive-c3 pairs (oscillation indicator).
    if len(diffs) >= 2:
        signs = np.sign(A)
        # consecutive pairs of (M-1) — but we also want consecutive STEP pairs;
        # for a coarse oscillation proxy on this small set we treat all pairs.
        flips_any = np.mean(np.any(signs[1:] * signs[:-1] < 0, axis=1))
        flips_x = np.mean(signs[1:, 0] * signs[:-1, 0] < 0)
        flips_y = np.mean(signs[1:, 1] * signs[:-1, 1] < 0)
        flips_z = np.mean(signs[1:, 2] * signs[:-1, 2] < 0)
        print(f"  sign-flip rate between consecutive a_ff samples: "
              f"any={100*flips_any:.1f}% x={100*flips_x:.1f}% y={100*flips_y:.1f}% z={100*flips_z:.1f}%")

    # Show a sample window
    print()
    print("Sample (first 12 consecutive-c3 a samples, components in m/s²):")
    for d in diffs[:12]:
        s_k, s_k1, a, mag = d
        print(f"  step {s_k:3d}→{s_k1:3d}  a=({a[0]:+8.2f},{a[1]:+8.2f},{a[2]:+8.2f})  |a|={mag:7.2f}")

    print()
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    # Pass criterion: median component |a_ff| ≤ 25 m/s², p90 ≤ 100 m/s²,
    # any-axis sign-flip rate ≤ 50%. Generous because we have a defensive
    # a_max clip in the build proposal.
    med = float(np.median(A_abs))
    p90 = float(np.percentile(A_abs, 90))
    flip_any = float(np.mean(np.any(np.sign(A)[1:] * np.sign(A)[:-1] < 0, axis=1)))
    smooth = (med <= 25.0) and (p90 <= 100.0) and (flip_any <= 0.50)
    print(f"  median component |a_ff|: {med:.2f} m/s²  ({'≤25' if med<=25 else '>25'})")
    print(f"  p90 component |a_ff|   : {p90:.2f} m/s²  ({'≤100' if p90<=100 else '>100'})")
    print(f"  sign-flip rate (any axis): {100*flip_any:.1f}%  ({'≤50%' if flip_any<=0.5 else '>50%'})")
    print()
    if smooth:
        print("  SOURCE-NOISE GATE: PASS — a_ff is bounded enough to feed with a "
              "defensive a_max clip. Proceed to STEP 1 build (PD + a_ff).")
        return 0
    else:
        print("  SOURCE-NOISE GATE: FAIL — implied a_ff is too noisy on the "
              "non-converged planner. STOP STEP 1 build. Mitigation options:")
        print("    (i)   feed conditionally (only when smooth — needs gating logic)")
        print("    (ii)  low-pass/clip at low a_max (would re-introduce damping)")
        print("    (iii) accept executor reconciled at pos+velocity, defer "
              "feedforward until ADMM converges (the §7.32 DISSOLVES state)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
