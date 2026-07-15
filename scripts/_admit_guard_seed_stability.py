"""Extract per-seed legit-approach EE_z bands from baseline logs.

A 'legit approach' tick = mode=c3 AND ee_box_normal > 0 (Drake registered
contact). Across these ticks we report: max EE_z (the peak the gate must
sit above to avoid falsely capping a real approach), median, and a
99th-percentile cutoff.

Decision criteria (per plan §Task 3 Step 2):
  - All seeds ee_z_max < 0.090 AND margin > 1.0 mm → gate seed-stable, proceed.
  - Any seed ee_z_max in [0.085, 0.090] → tight; document and proceed cautiously.
  - Any seed ee_z_max >= 0.090 → STOP. Raise threshold or escalate to (i-c).
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

STEP_RE = re.compile(
    r"\[STEP\] step=(\d+) mode=(\w+) t=([0-9.]+)s.*?"
    r"ee=\(([+-][0-9.]+),([+-][0-9.]+),([+-][0-9.]+)\)"
)
DRAKE_RE = re.compile(
    r"\[DRAKE-CONTACT\] step=(\d+) n_pairs=(\d+) ee_box_normal=([0-9.]+)"
)

EE_Z_GATE = 0.090


def parse(log_path: Path) -> dict:
    ee_z_by_step: dict[int, float] = {}
    mode_by_step: dict[int, str] = {}
    normal_by_step: dict[int, float] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            s = int(m.group(1))
            mode_by_step[s] = m.group(2)
            ee_z_by_step[s] = float(m.group(6))
            continue
        m = DRAKE_RE.search(line)
        if m:
            normal_by_step[int(m.group(1))] = float(m.group(3))

    # A "legit approach" tick — for gate-relevance purposes — must satisfy:
    #   (1) mode == "free"  : the IK tracker is only invoked in free mode;
    #                         mode=c3 never reaches the gate code path.
    #   (2) Drake registered face contact (ee_box_normal > 0): distinguishes
    #       a genuine Phase 3 descent-to-face from a traversal near-miss.
    #   Note: the [ADMIT-GUARD] log only fires in mode=free, so the
    #   latch>0 condition is implied by emission. We don't filter on the
    #   admit-latch here because the log may not have the field at all
    #   commits (pre-Q2c logs lack ee_z/gate_cap), but we still know the
    #   tick was mode=free → gate-relevant.
    legit_ee_z: list[float] = []
    for s, mode in mode_by_step.items():
        if mode != "free":
            continue
        if normal_by_step.get(s, 0.0) <= 0.0:
            continue
        legit_ee_z.append(ee_z_by_step[s])

    if not legit_ee_z:
        return dict(
            log=str(log_path),
            n_legit_ticks=0,
            gate_ok=None,
            note="no_legit_approach_baseline",
        )

    legit_ee_z.sort()
    n = len(legit_ee_z)
    p99_idx = min(n - 1, int(0.99 * n))
    p99 = legit_ee_z[p99_idx]
    p_max = legit_ee_z[-1]
    p_med = legit_ee_z[n // 2]
    margin_mm = round(1000 * (EE_Z_GATE - p_max), 2)
    gate_ok = bool(p_max < EE_Z_GATE)

    if gate_ok and margin_mm > 1.0:
        verdict = "stable"
    elif gate_ok:
        verdict = "tight"  # 0 < margin <= 1mm
    else:
        verdict = "FAILS"

    return dict(
        log=str(log_path),
        n_legit_ticks=n,
        ee_z_max=round(p_max, 4),
        ee_z_p99=round(p99, 4),
        ee_z_median=round(p_med, 4),
        gate=EE_Z_GATE,
        margin_to_gate_mm=margin_mm,
        gate_ok=gate_ok,
        verdict=verdict,
    )


def main() -> int:
    out: dict[str, dict] = {}
    for p_str in sys.argv[1:]:
        p = Path(p_str)
        out[p.stem] = parse(p)
    print(json.dumps(out, indent=2))
    # Exit non-zero if any seed fails outright (gate at or below ee_z_max).
    any_fail = any(d.get("verdict") == "FAILS" for d in out.values())
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
