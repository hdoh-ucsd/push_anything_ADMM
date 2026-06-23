"""Compare Stage A flag-ON metrics against the flag-OFF baseline.

Implements the operationalized 4-condition bar (see
docs/superpowers/plans/2026-06-23-stage-a-reposition-pwl-trajectory-port.md §2)
AND the EE-landing-primary wired signal (Refinement 2)
AND the rebuild-churn check (Refinement 3).

Verdict priority: CHURN > PASS > WIRED-BUT-INSUFFICIENT >
PARTIAL-WIRED > NOT WIRED.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LANDING_THRESHOLD_M    = 0.002
LANDING_FRAC_BAR       = 0.75
ADMIT_NOISE_BAND       = 0.05
ENTRY_GATE_INERT_RATE  = 0.05
QY_QZ_NOISE_BAND       = 0.02
WIRED_LANDING_DELTA_M  = 0.001
REBUILD_RATE_HZ_BAR    = 1.0


def _load(paths: list[Path]) -> dict[int, dict]:
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        seed = int(Path(p).parent.name.replace("seed", ""))
        out[seed] = d
    return out


def evaluate(baseline: dict[int, dict], stage_a: dict[int, dict]) -> dict:
    seeds = sorted(set(baseline.keys()) & set(stage_a.keys()))
    per_seed: dict[int, dict] = {}
    for s in seeds:
        b = baseline[s]
        a = stage_a[s]

        # Cond 1: EE lands WINDOWED & sustained.
        if a.get("window_status") != "OK":
            cond1 = False
            cond1_reason = f"window_status={a.get('window_status')}"
        elif a.get("landing_median_m") is None:
            cond1 = False
            cond1_reason = "no landing series"
        else:
            cond1 = (
                a["landing_median_m"] <= LANDING_THRESHOLD_M
                and a["landing_fraction_within_2mm"] >= LANDING_FRAC_BAR
            )
            cond1_reason = (
                f"median={a['landing_median_m']*1000:.2f}mm (≤2mm? "
                f"{a['landing_median_m']<=LANDING_THRESHOLD_M}); "
                f"frac={a['landing_fraction_within_2mm']*100:.0f}% (≥75%? "
                f"{a['landing_fraction_within_2mm']>=LANDING_FRAC_BAR})"
            )

        # Cond 2: admit MAINTAINED-OR-IMPROVED.
        cond2 = a["admit_rate"] >= (b["admit_rate"] - ADMIT_NOISE_BAND)
        cond2_reason = (
            f"stage_a={a['admit_rate']*100:.2f}% vs "
            f"baseline-band={b['admit_rate']*100:.2f}%-5pp"
        )

        # Cond 3: entry-gate firing rate < 5%.
        cond3 = a["entry_gate_firing_rate"] < ENTRY_GATE_INERT_RATE
        cond3_reason = (
            f"{a['entry_gate_firing_rate']*100:.2f}% (<5%? {cond3})"
        )

        # Cond 4: |qy|, |qz| not worse than baseline + band.
        cond4_qy = a["max_abs_qy"] <= b["max_abs_qy"] + QY_QZ_NOISE_BAND
        cond4_qz = a["max_abs_qz"] <= b["max_abs_qz"] + QY_QZ_NOISE_BAND
        cond4 = cond4_qy and cond4_qz
        cond4_reason = (
            f"|qy|: stage_a={a['max_abs_qy']:.4f} vs "
            f"baseline+band={b['max_abs_qy']+QY_QZ_NOISE_BAND:.4f} "
            f"({'OK' if cond4_qy else 'WORSE'}); "
            f"|qz|: stage_a={a['max_abs_qz']:.4f} vs "
            f"baseline+band={b['max_abs_qz']+QY_QZ_NOISE_BAND:.4f} "
            f"({'OK' if cond4_qz else 'WORSE'})"
        )

        # Wired (Refinement 2): EE-landing PRIMARY, entry-gate CONFIRMATION.
        primary = (
            a.get("landing_median_m") is not None
            and b.get("landing_median_m") is not None
            and a["landing_median_m"]
                < b["landing_median_m"] - WIRED_LANDING_DELTA_M
        )
        confirmation = (
            a["entry_gate_firing_rate"]
            <= max(0.5 * b["entry_gate_firing_rate"],
                   ENTRY_GATE_INERT_RATE)
        )
        wired = bool(primary and confirmation)

        # Churn (Refinement 3): Stage A only.
        churn_flagged = bool(a.get("rebuild_churn_flagged", False))

        per_seed[s] = dict(
            cond1_ee_lands_windowed=cond1,
            cond1_reason=cond1_reason,
            cond2_admit_maintained_or_improved=cond2,
            cond2_reason=cond2_reason,
            cond3_entry_gate_inert=cond3,
            cond3_reason=cond3_reason,
            cond4_qy_qz_not_worse=cond4,
            cond4_reason=cond4_reason,
            wired_primary_ee_landing=primary,
            wired_confirmation_entry_gate=confirmation,
            wired_signal=wired,
            churn_flagged=churn_flagged,
            # snapshots:
            baseline_landing_median_m=b.get("landing_median_m"),
            stage_a_landing_median_m=a.get("landing_median_m"),
            stage_a_landing_fraction_within_2mm=a.get(
                "landing_fraction_within_2mm"),
            stage_a_window_status=a.get("window_status"),
            stage_a_window_ticks_used=a.get("window_ticks_used"),
            stage_a_first_c3_episode_len=a.get("first_c3_episode_len_ticks"),
            baseline_admit_rate=b.get("admit_rate"),
            stage_a_admit_rate=a.get("admit_rate"),
            baseline_entry_gate_rate=b.get("entry_gate_firing_rate"),
            stage_a_entry_gate_rate=a.get("entry_gate_firing_rate"),
            baseline_max_abs_qy=b.get("max_abs_qy"),
            stage_a_max_abs_qy=a.get("max_abs_qy"),
            baseline_max_abs_qz=b.get("max_abs_qz"),
            stage_a_max_abs_qz=a.get("max_abs_qz"),
            baseline_goal_motion_m=b.get("goal_motion_m"),
            stage_a_goal_motion_m=a.get("goal_motion_m"),
            stage_a_rebuild_rate_hz=a.get("rebuild_rate_hz"),
            stage_a_pwl_rebuilds_total=a.get("pwl_rebuilds_total"),
        )

    full_bar = all(
        ps["cond1_ee_lands_windowed"]
        and ps["cond2_admit_maintained_or_improved"]
        and ps["cond3_entry_gate_inert"]
        and ps["cond4_qy_qz_not_worse"]
        for ps in per_seed.values()
    ) if per_seed else False
    any_wired = any(ps["wired_signal"] for ps in per_seed.values())
    all_wired = all(ps["wired_signal"] for ps in per_seed.values())
    any_churn = any(ps["churn_flagged"] for ps in per_seed.values())

    if any_churn:
        verdict = (
            f"CHURN — rebuild_rate_hz >= {REBUILD_RATE_HZ_BAR} on at least "
            f"one seed; per-tick march pathology may be back. STOP, "
            f"investigate the rebuild trigger before claiming any Stage A "
            f"pass."
        )
    elif full_bar:
        verdict = "PASS — full bar met; flip §1 row 6 RECONCILED."
    elif all_wired:
        verdict = (
            "WIRED-BUT-INSUFFICIENT — necessary, advance to Stage B; "
            "bar held cumulative."
        )
    elif any_wired:
        verdict = (
            "PARTIAL-WIRED — one seed wired, the other not. STOP and "
            "investigate seed asymmetry BEFORE advancing."
        )
    else:
        verdict = (
            "NOT WIRED — EE landing did NOT move closer to surface. "
            "STOP. Integration bug at Stage A. Do NOT chain Stage B."
        )

    return dict(
        per_seed=per_seed,
        full_bar=full_bar,
        any_wired=any_wired,
        all_wired=all_wired,
        any_churn=any_churn,
        verdict=verdict,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", type=Path, required=True)
    ap.add_argument("--stage_a", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("stage_a/bar_evaluation.json"))
    args = ap.parse_args()
    b = _load(args.baseline)
    a = _load(args.stage_a)
    result = evaluate(b, a)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
