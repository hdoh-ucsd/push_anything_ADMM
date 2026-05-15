#!/usr/bin/env python3
"""Generate heatmap.md, envelope.png/txt, REPORT.md from summary.csv."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "results" / "contact_free_sweep"
CSV  = ROOT / "summary.csv"
HEATMAP_MD = ROOT / "heatmap.md"
ENVELOPE_PNG = ROOT / "envelope.png"
ENVELOPE_TXT = ROOT / "envelope.txt"
REPORT_MD = ROOT / "REPORT.md"


def load_rows() -> list[dict]:
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def classify(row: dict) -> str:
    """Failure-mode classification per STEP 5 of the protocol."""
    status = row.get("status", "")
    if status not in ("ok",):
        if "timeout" in status:
            return "Timeout"
        if "crash" in status:
            return "Crashed"
        if "missing" in status:
            return "Missing"
        if status == "partial_no_result":
            return "Partial"
        return f"Unknown({status})"

    def f(name: str) -> float:
        v = row.get(name, "")
        return float(v) if v not in ("", None) else float("nan")

    def i(name: str) -> int:
        v = row.get(name, "")
        return int(v) if v not in ("", None) else 0

    motion_mm = f("total_motion_mm")
    if motion_mm < 1.0:
        return "Never engaged"

    # obj z escape catastrophic — proxy via min_ee_z far below 0 implies box may have fallen,
    # but we don't have obj_z in CSV. Use final_box_err > 1m as catastrophic proxy.
    final_err = f("final_box_err_m")
    if final_err > 1.0:
        return "Catastrophic"

    if i("T1") or i("T2"):
        return "Reaches goal"
    if i("T3") or i("T4"):
        # Moved in correct direction far enough but didn't reach
        # If EE froze: Type-0 fixed point reached
        if i("ee_z_froze"):
            return "Frozen at fixed point"
        return "Stops short"
    if i("T5"):
        return "Drifts off direction"
    # T5 fails: motion < 5mm or angle off > 30 deg
    if motion_mm > 5.0:
        return "Drifts off direction"
    return "Never engaged"


def render_heatmap_md(rows: list[dict]) -> str:
    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        try:
            d = int(r["distance_cm"])
        except (KeyError, ValueError):
            continue
        by_d[d].append(r)

    def tally(rs, key):
        ok = sum(1 for r in rs if str(r.get(key, "")).strip() == "1")
        marks = "".join("✓" if str(r.get(key, "")).strip() == "1" else "·" for r in rs)
        return f"{ok}/{len(rs)} {marks}"

    out = ["| Distance | T1 (5mm) | T2 (10%) | T3 (≥80%) | T4 (≥50%) | T5 (correct dir) |",
           "|----------|----------|----------|-----------|-----------|------------------|"]
    for d in sorted(by_d.keys()):
        rs = sorted(by_d[d], key=lambda r: int(r["seed"]))
        out.append(
            f"| {d:>2} cm   | {tally(rs,'T1'):<8} | {tally(rs,'T2'):<8} "
            f"| {tally(rs,'T3'):<9} | {tally(rs,'T4'):<9} | {tally(rs,'T5'):<16} |"
        )
    return "\n".join(out)


def render_failure_breakdown(rows: list[dict]) -> dict[int, dict[str, int]]:
    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        try:
            d = int(r["distance_cm"])
        except (KeyError, ValueError):
            continue
        by_d[d].append(r)
    out: dict[int, dict[str, int]] = {}
    for d, rs in by_d.items():
        bucket: dict[str, int] = defaultdict(int)
        for r in rs:
            bucket[classify(r)] += 1
        out[d] = dict(bucket)
    return out


def render_envelope_txt(rows: list[dict], breakdown: dict) -> str:
    L = ["Contact-Free Joint-PD Operating Envelope",
         "==========================================",
         "",
         "Threshold success rates (n=3 seeds per distance):",
         ""]
    L.append(render_heatmap_md(rows))
    L.append("")
    L.append("Failure-mode breakdown by distance:")
    L.append("")
    for d in sorted(breakdown.keys()):
        modes = breakdown[d]
        total = sum(modes.values())
        parts = [f"{k}={v}/{total}" for k, v in sorted(modes.items(), key=lambda kv: -kv[1])]
        L.append(f"  {d:>2} cm: " + "  ".join(parts))
    return "\n".join(L)


def try_render_png(rows: list[dict], breakdown: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] matplotlib unavailable ({e}); skipping PNG.")
        return False

    distances = sorted({int(r["distance_cm"]) for r in rows if r.get("distance_cm","").isdigit()})
    # Panel 1: T1-T5 success rates per distance
    thresholds = ["T1", "T2", "T3", "T4", "T5"]
    rates: dict[str, list[float]] = {t: [] for t in thresholds}
    for d in distances:
        rs = [r for r in rows if r.get("distance_cm") == str(d)]
        n = max(1, len(rs))
        for t in thresholds:
            ok = sum(1 for r in rs if str(r.get(t, "")).strip() == "1")
            rates[t].append(ok / n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: grouped bars
    import numpy as np
    x = np.arange(len(distances))
    w = 0.15
    for i, t in enumerate(thresholds):
        ax1.bar(x + (i - 2) * w, rates[t], width=w, label=t)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d} cm" for d in distances])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Success rate (n=3)")
    ax1.set_title("Joint-PD success rates by threshold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2: failure-mode stacked bars
    all_modes = sorted({m for d in distances for m in breakdown.get(d, {})})
    color_map = {
        "Reaches goal":           "#1b9e77",
        "Stops short":            "#d95f02",
        "Frozen at fixed point":  "#7570b3",
        "Drifts off direction":   "#e7298a",
        "Never engaged":          "#66a61e",
        "Catastrophic":           "#e6ab02",
        "Timeout":                "#a6761d",
        "Crashed":                "#666666",
        "Missing":                "#cccccc",
        "Partial":                "#999999",
    }
    bottoms = np.zeros(len(distances))
    for mode in all_modes:
        counts = np.array([breakdown.get(d, {}).get(mode, 0) for d in distances], dtype=float)
        n_per_d = np.array([sum(breakdown.get(d, {}).values()) or 1 for d in distances], dtype=float)
        fracs = counts / n_per_d
        ax2.bar(x, fracs, bottom=bottoms, width=0.6,
                color=color_map.get(mode, None), label=mode)
        bottoms += fracs
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d} cm" for d in distances])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Fraction of seeds")
    ax2.set_title("Dominant failure mode by distance")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Contact-Free Joint-PD Operating Envelope", fontsize=13)
    fig.tight_layout()
    fig.savefig(ENVELOPE_PNG, dpi=140)
    plt.close(fig)
    print(f"[fig] wrote {ENVELOPE_PNG.relative_to(REPO)}")
    return True


def dominant_mode(modes: dict[str, int]) -> str:
    if not modes:
        return "n/a"
    return max(modes.items(), key=lambda kv: kv[1])[0]


def implications(rows: list[dict], breakdown: dict) -> tuple[str, str, str, str]:
    """Return (reliable_zone, transition_zone, failure_zone, dominant_failure)."""
    distances = sorted(breakdown.keys())
    reliable = []
    transition = []
    failure = []
    for d in distances:
        rs = [r for r in rows if r.get("distance_cm") == str(d)]
        n = len(rs)
        n_t1 = sum(1 for r in rs if str(r.get("T1", "")).strip() == "1")
        n_t2 = sum(1 for r in rs if str(r.get("T2", "")).strip() == "1")
        n_t5 = sum(1 for r in rs if str(r.get("T5", "")).strip() == "1")
        if n_t1 == n or n_t2 == n:
            reliable.append(d)
        elif n_t5 == 0:
            failure.append(d)
        else:
            transition.append(d)
    rs = lambda zone: (", ".join(f"{d} cm" for d in zone) if zone else "(none)")
    # Dominant failure mode across all non-reliable distances
    non_reliable_modes: dict[str, int] = defaultdict(int)
    for d in transition + failure:
        for m, c in breakdown.get(d, {}).items():
            non_reliable_modes[m] += c
    dom = dominant_mode(non_reliable_modes)
    return rs(reliable), rs(transition), rs(failure), dom


def write_report(rows: list[dict], breakdown: dict, fig_ok: bool):
    reliable, transition, failure, dom = implications(rows, breakdown)
    n_runs = len(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")

    headline = (
        f"Joint-PD reliably reaches goals within "
        f"**{reliable.split(',')[0] if reliable != '(none)' else '(no distance qualified)'}** "
        f"(all 3 seeds pass T1 or T2). Transition zone: **{transition}**. "
        f"Failure zone (no seeds clear T5): **{failure}**. "
        f"Dominant failure mode outside the reliable zone: **{dom}**."
    )

    L = []
    L.append("# Contact-Free Joint-PD Operating Envelope")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(headline)
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append("- 5 goal distances (5, 10, 15, 20, 30 cm along +x from box init at (0,0,0.05))")
    L.append("- 3 seeds per distance (0, 1, 2) for SamplingC3MPC sampling-circle angle draws")
    L.append("- Total: 15 sequential runs")
    L.append("- Executor: joint-PD via SamplingC3MPC + kIK guide path "
             "(`config/sampling_c3_kik.yaml`)")
    L.append("- Initial pose: `--prepositioned` (IK-aligned EE behind box)")
    L.append("- Per-run instrumentation captured: per-step EE xyz ([GS-tgt]), "
             "per-step box xy ([EErel]), mode (free|c3) ([GS]), peak torque, "
             "final box / EE state ([RESULT]).")
    L.append("- Wall-clock per run: ~3.5–4 min. Total: ~55–60 min.")
    L.append(f"- {n_ok}/{n_runs} runs completed cleanly.")
    L.append("")
    L.append("## Success thresholds")
    L.append("")
    L.append("- **T1**: box reaches goal within 5 mm (verdict-A standard)")
    L.append("- **T2**: box reaches goal within 10% of goal distance")
    L.append("- **T3**: box moves ≥ 80% of goal distance in goal direction")
    L.append("- **T4**: box moves ≥ 50% of goal distance in goal direction")
    L.append("- **T5**: box moves > 5 mm within 30° of goal direction")
    L.append("")
    L.append("## Operating envelope")
    L.append("")
    L.append(render_heatmap_md(rows))
    L.append("")
    L.append("## Failure-mode analysis")
    L.append("")
    for d in sorted(breakdown.keys()):
        modes = breakdown[d]
        total = sum(modes.values())
        parts = ", ".join(f"{k}={v}/{total}" for k, v in sorted(modes.items(), key=lambda kv: -kv[1]))
        L.append(f"- **{d} cm**: {parts}")
    L.append("")
    if fig_ok:
        L.append(f"![envelope](envelope.png)")
    else:
        L.append("(matplotlib not available; see `envelope.txt`.)")
    L.append("")
    L.append("### Interpretation")
    L.append("")
    if "Frozen at fixed point" in dom or any(
        "Frozen" in m for m in [dominant_mode(breakdown[d]) for d in breakdown]
    ):
        L.append("- EE z froze at the ~0.025 m fixed point (Type-0 signature, Effect-A) "
                 "in a substantial fraction of runs. This matches the prediction from "
                 "the OSC-pivot Type-0 analysis: gravity-mismatch between q_now and "
                 "q_target produces a steady-state error that scales with the EE "
                 "displacement required to reach behind the box. The smaller the goal, "
                 "the less the EE travels, the smaller the gravity mismatch.")
    L.append("")
    L.append("## Implications")
    L.append("")
    if reliable != "(none)":
        L.append(f"- For **verdict-A**: realistic goal distances are **{reliable}** "
                 f"with the current joint-PD + kIK stack. Distances in **{transition}** "
                 f"are unreliable across seeds. Distances in **{failure}** require "
                 f"architectural changes (e.g., the parked OSC pivot, or gain retuning).")
    else:
        L.append("- For **verdict-A**: **no goal distance in the swept range** produces "
                 "reliable success with the current joint-PD + kIK stack. The transition "
                 f"zone spans **{transition}**; the failure zone is **{failure}**.")
    L.append(f"- Dominant failure mode outside the reliable zone: **{dom}**. ")
    if dom == "Frozen at fixed point":
        L.append("  This is the Type-0 signature. Next investigation: joint-PD gain "
                 "retuning, or reviving the OSC pivot for displacement-tolerant tracking.")
    elif dom == "Drifts off direction":
        L.append("  EE took a path that imparted off-axis force. Next investigation: "
                 "the kIK chain's path-routing logic and how the sampling-circle angle "
                 "draws bias the approach geometry.")
    elif dom == "Stops short":
        L.append("  Direction is correct but motion stops before reaching the goal. "
                 "Possible causes: friction-cone limit reached, EE de-contact, or "
                 "joint-PD reaching its torque ceiling.")
    elif dom == "Never engaged":
        L.append("  The kIK chain didn't establish contact. Next investigation: "
                 "the reposition trajectory and initial pose's reach margin.")
    L.append("")
    L.append("## Raw data")
    L.append("")
    L.append("- [`summary.csv`](summary.csv) — per-run metrics")
    L.append("- [`logs/`](logs/) — 15 raw run logs")
    L.append("- [`configs/`](configs/) — 15 sweep config records")
    L.append("- [`envelope.png`](envelope.png) / [`envelope.txt`](envelope.txt) — figure")
    L.append("")
    REPORT_MD.write_text("\n".join(L) + "\n")
    print(f"[report] wrote {REPORT_MD.relative_to(REPO)}")


def main():
    if not CSV.exists():
        print(f"[report] missing {CSV}; run parse_sweep_logs.py first.")
        return 1
    rows = load_rows()
    breakdown = render_failure_breakdown(rows)

    HEATMAP_MD.write_text(render_heatmap_md(rows) + "\n")
    print(f"[report] wrote {HEATMAP_MD.relative_to(REPO)}")

    ENVELOPE_TXT.write_text(render_envelope_txt(rows, breakdown) + "\n")
    print(f"[report] wrote {ENVELOPE_TXT.relative_to(REPO)}")

    fig_ok = try_render_png(rows, breakdown)
    write_report(rows, breakdown, fig_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
