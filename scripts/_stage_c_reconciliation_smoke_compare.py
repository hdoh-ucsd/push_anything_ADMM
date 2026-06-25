"""Stage C tick→sim-t reconciliation — smoke comparator.

Two modes:
  --mode 100hz-noop      Smoke 1: prove byte/distribution equivalence to a
                         100 Hz baseline (the regression gate). Pass = the
                         conversion did nothing observable at 100 Hz.
  --mode 1khz-recon      Smoke 2: prove c3 mode engages at 1 kHz with the
                         reposition landing the same φ as the 100 Hz
                         baseline (the cadence-discriminator-unblocker).

Pass bars per plan
docs/superpowers/plans/2026-06-25-tick-to-simtime-cadence-reconciliation.md
§3.1 (100hz-noop) and §3.2 (1khz-recon).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RE_STAGE_A_TRACE = re.compile(
    r"\[STAGE-A-TRACE\] step=(\d+) sim_t=([+-]?\d*\.?\d+) "
    r"mode=(\w+) phi=([+-]?\d*\.?\d+(?:e[+-]?\d+)?|nan)"
)
RE_STAGE_A_PWL_BUILD = re.compile(r"\[STAGE-A-PWL\] step=(\d+) sim_t=([\d.]+) build")
RE_GS_PERF = re.compile(r"\[GS-perf\].*switches=(\d+)")
RE_RESULT = re.compile(r"\[RESULT\].*final_obj_xy=\(([+-]?[\d.]+),\s*([+-]?[\d.]+)\)")


def parse_log(path: Path) -> dict:
    steps = {}
    pwl_builds = []
    switches = None
    final_xy = None
    first_c3_step = None
    first_c3_phi = None
    with path.open() as f:
        for line in f:
            m = RE_STAGE_A_TRACE.search(line)
            if m:
                step = int(m.group(1))
                mode = m.group(3)
                try:
                    phi = float(m.group(4))
                except ValueError:
                    phi = float("nan")
                steps[step] = (mode, phi)
                if first_c3_step is None and mode == "c3":
                    first_c3_step = step
                    first_c3_phi = phi
            m = RE_STAGE_A_PWL_BUILD.search(line)
            if m:
                pwl_builds.append(int(m.group(1)))
            m = RE_GS_PERF.search(line)
            if m:
                switches = int(m.group(1))
            m = RE_RESULT.search(line)
            if m:
                final_xy = (float(m.group(1)), float(m.group(2)))
    return dict(steps=steps, pwl_builds=pwl_builds, switches=switches,
                final_xy=final_xy, first_c3_step=first_c3_step,
                first_c3_phi=first_c3_phi)


def evaluate_100hz_noop(baseline: dict, candidate: dict) -> dict:
    bars = []

    # Bar 1: first-c3-entry step ±2
    base_s = baseline["first_c3_step"]; cand_s = candidate["first_c3_step"]
    if base_s is not None and cand_s is not None:
        delta = abs(cand_s - base_s)
        bars.append(("first_c3_step_within_2",
                     delta <= 2,
                     f"baseline={base_s}, candidate={cand_s}, delta={delta}"))
    else:
        bars.append(("first_c3_step_within_2", False,
                     f"baseline={base_s}, candidate={cand_s} (missing)"))

    # Bar 2: first-c3 phi ±0.1 mm
    base_p = baseline["first_c3_phi"]; cand_p = candidate["first_c3_phi"]
    if base_p is not None and cand_p is not None:
        delta_mm = abs(cand_p - base_p) * 1000
        bars.append(("first_c3_phi_within_0.1mm",
                     delta_mm <= 0.1,
                     f"baseline={base_p*1000:.3f}mm cand={cand_p*1000:.3f}mm delta={delta_mm:.3f}mm"))
    else:
        bars.append(("first_c3_phi_within_0.1mm", False,
                     f"baseline_phi={base_p} cand_phi={cand_p} (missing)"))

    # Bar 3: switches ±2
    base_sw = baseline["switches"]; cand_sw = candidate["switches"]
    if base_sw is not None and cand_sw is not None:
        delta = abs(cand_sw - base_sw)
        bars.append(("switches_within_2",
                     delta <= 2,
                     f"baseline={base_sw}, candidate={cand_sw}, delta={delta}"))
    else:
        bars.append(("switches_within_2", False,
                     f"baseline_switches={base_sw} cand_switches={cand_sw}"))

    # Bar 4: per-tick mode-match ≥95%
    common = set(baseline["steps"].keys()) & set(candidate["steps"].keys())
    if common:
        matches = sum(
            1 for s in common
            if baseline["steps"][s][0] == candidate["steps"][s][0])
        rate = matches / len(common)
        bars.append(("mode_match_rate_ge_0.95",
                     rate >= 0.95,
                     f"matches={matches}/{len(common)} rate={rate:.4f}"))
    else:
        bars.append(("mode_match_rate_ge_0.95", False,
                     "no common steps to compare"))

    # Bar 5: final obj_xy ±0.5 mm
    base_xy = baseline["final_xy"]; cand_xy = candidate["final_xy"]
    if base_xy and cand_xy:
        dx_mm = abs(cand_xy[0] - base_xy[0]) * 1000
        dy_mm = abs(cand_xy[1] - base_xy[1]) * 1000
        max_d = max(dx_mm, dy_mm)
        bars.append(("final_obj_xy_within_0.5mm",
                     max_d <= 0.5,
                     f"baseline={base_xy} cand={cand_xy} max|Δ|={max_d:.3f}mm"))
    else:
        bars.append(("final_obj_xy_within_0.5mm", False,
                     f"baseline_xy={base_xy} cand_xy={cand_xy}"))

    return dict(mode="100hz-noop", bars=bars,
                all_pass=all(b[1] for b in bars))


def evaluate_1khz_recon(baseline: dict, candidate: dict) -> dict:
    bars = []

    # Bar 1: switches > 0 (c3 mode engaged — the thing that failed in the scope-stop)
    cand_sw = candidate["switches"]
    bars.append(("switches_gt_0",
                 cand_sw is not None and cand_sw > 0,
                 f"candidate switches={cand_sw}"))

    # Bar 2: first-c3 phi within ±0.2 mm of baseline
    base_p = baseline["first_c3_phi"]; cand_p = candidate["first_c3_phi"]
    if base_p is not None and cand_p is not None:
        delta_mm = abs(cand_p - base_p) * 1000
        bars.append(("first_c3_phi_within_0.2mm",
                     delta_mm <= 0.2,
                     f"baseline={base_p*1000:.3f}mm cand={cand_p*1000:.3f}mm delta={delta_mm:.3f}mm"))
    else:
        bars.append(("first_c3_phi_within_0.2mm", False,
                     f"baseline_phi={base_p} cand_phi={cand_p} (missing)"))

    # Bar 3: PWL builds spaced at ~300 ticks at 1 kHz (sim-time consistent)
    builds = candidate["pwl_builds"]
    if len(builds) >= 2:
        deltas = [builds[i+1] - builds[i] for i in range(len(builds) - 1)]
        median = sorted(deltas)[len(deltas) // 2]
        # ±10% of 300 ticks (= [270, 330])
        in_band = 270 <= median <= 330
        bars.append(("pwl_build_delta_median_at_300_ticks_pm10pct",
                     in_band,
                     f"deltas={deltas[:10]} median={median} (expected 270≤x≤330 at 1kHz)"))
    else:
        bars.append(("pwl_build_delta_median_at_300_ticks_pm10pct", False,
                     f"only {len(builds)} build lines"))

    # Bar 4: finished_repos=1 reached within 5% of baseline sim-time (~1.7 s)
    # At 1 kHz that's step ~1700 ± 85.
    # Find first step at which mode flipped from free to c3 — uses first_c3_step.
    # At 1 kHz a 1.7 s landing means step ~1700.
    cand_first_c3 = candidate["first_c3_step"]
    if cand_first_c3 is not None:
        # Expected ~1700 at 1 kHz (the baseline 100Hz hits c3 at step 173 = sim_t 1.73s)
        delta = abs(cand_first_c3 - 1700)
        bars.append(("first_c3_step_near_1700_pm5pct",
                     delta <= 85,
                     f"candidate first_c3_step={cand_first_c3}, expected ~1700 ±85"))
    else:
        bars.append(("first_c3_step_near_1700_pm5pct", False,
                     "candidate never entered c3"))

    return dict(mode="1khz-recon", bars=bars,
                all_pass=all(b[1] for b in bars))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["100hz-noop", "1khz-recon"], required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    b = parse_log(Path(args.baseline))
    c = parse_log(Path(args.candidate))
    if args.mode == "100hz-noop":
        out = evaluate_100hz_noop(b, c)
    else:
        out = evaluate_1khz_recon(b, c)
    print(json.dumps(out, indent=2, default=str))
    print(("PASS" if out["all_pass"] else "FAIL"),
          f"({sum(1 for x in out['bars'] if x[1])}/{len(out['bars'])} bars)",
          file=sys.stderr)
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
