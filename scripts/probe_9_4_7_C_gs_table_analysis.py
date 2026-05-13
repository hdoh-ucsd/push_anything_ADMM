"""9.4.7 Option C — direct [GS-table] inspection across F2 + Option A logs.

Read-only analysis. Parses [GS-table] blocks from given log files,
collapses them into a long-form table (one row per sample per step),
and reports:

  - Per-label (current / prev_repos / strat_0 / ...) cost-component
    histograms: c_C3_raw, align_bonus, travel_penalty, c_sample
  - Mean/median/IQR per label
  - The post-F2 c_C3_raw gap between prev_repos and the best strat_*
  - Winner-share by label, broken down by mode
  - For winning prev_repos rows: what was the gap to the best non-
    prev_repos sample?

Output: prints summary to stdout and writes
``results/probe_9_4_7_C_gs_table_summary.txt``.
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "results"

# Path A v2 was the F2 PWL re-run; the v1 was killed at step ~398.
DEFAULT_LOGS = [
    RESULTS_DIR / "9_4_7_F2_path_d_stdout.log",
    RESULTS_DIR / "9_4_7_F2_path_a_v2_stdout.log",
    RESULTS_DIR / "9_4_7_A_path_d_stdout.log",     # Option A
    RESULTS_DIR / "9_4_7_A_path_a_stdout.log",     # Option A (if present)
]

# [GS-table] step=N
_re_table  = re.compile(r"^\[GS-table\]\s+step=(?P<step>\d+)\s*$")
#   k=0 (current   ) pos=(...) c_C3=...  align=0.8049(bonus=24148.25) travel=0.000m(pen=  0.00) c_sample=...  feas=Y ik_err=... [← WIN]
_re_row    = re.compile(
    r"^\s*k=(?P<k>\d+)\s+\((?P<label>[^)]+?)\s*\)\s+"
    r"pos=\([^)]*\)\s+"
    r"c_C3=\s*(?P<c_C3>-?[\d.]+)\s+"
    r"align=(?P<align_score>[\d.]+)\(bonus=\s*(?P<align_bonus>-?[\d.]+)\)\s+"
    r"travel=(?P<travel>[\d.]+)m\(pen=\s*(?P<travel_pen>-?[\d.]+)\)\s+"
    r"c_sample=\s*(?P<c_sample>-?[\d.]+)\s+"
    r"feas=(?P<feas>[YN])\s+"
    r"ik_err=(?P<ik_err>[\d.]+)m"
    r"(?P<win>\s+←\s+WIN)?",
)
# [GS] step=N mode=... best_src=...
_re_gs     = re.compile(
    r"^\[GS\]\s+step=(?P<step>\d+)\s+mode=(?P<mode>\w+)\s+\S+\s+"
    r"best_k=(\d+|None)\s+best_src=(?P<best_src>\w+)",
)


def parse_log(path: Path):
    """Return list of (step, mode, [rows...]) and the wrapper's full
    best_src histogram across all [GS] lines."""
    if not path.exists():
        return [], {}

    text = path.read_text(errors="replace")

    # Best_src histogram across every step (not just every-20)
    best_src_counter = defaultdict(int)
    mode_per_step = {}
    for ln in text.splitlines():
        m = _re_gs.match(ln)
        if m:
            best_src_counter[m.group("best_src")] += 1
            mode_per_step[int(m.group("step"))] = m.group("mode")

    # GS-table blocks (every-20-steps in production code)
    blocks = []
    cur_step = None
    cur_rows = []
    for ln in text.splitlines():
        mhead = _re_table.match(ln)
        if mhead:
            if cur_step is not None:
                blocks.append((cur_step, mode_per_step.get(cur_step, "?"), cur_rows))
            cur_step = int(mhead.group("step"))
            cur_rows = []
            continue
        if cur_step is not None:
            mrow = _re_row.match(ln)
            if mrow:
                cur_rows.append({
                    "k":            int(mrow.group("k")),
                    "label":        mrow.group("label").strip(),
                    "c_C3_raw":     float(mrow.group("c_C3")),
                    "align_score":  float(mrow.group("align_score")),
                    "align_bonus":  float(mrow.group("align_bonus")),
                    "travel":       float(mrow.group("travel")),
                    "travel_pen":   float(mrow.group("travel_pen")),
                    "c_sample":     float(mrow.group("c_sample")),
                    "feasible":     mrow.group("feas") == "Y",
                    "winner":       bool(mrow.group("win")),
                })
            elif ln.startswith("[") or ln.startswith("  t="):
                # Block ended; commit
                if cur_step is not None and cur_rows:
                    blocks.append((cur_step, mode_per_step.get(cur_step, "?"), cur_rows))
                cur_step = None
                cur_rows = []
    if cur_step is not None and cur_rows:
        blocks.append((cur_step, mode_per_step.get(cur_step, "?"), cur_rows))

    return blocks, dict(best_src_counter)


def _describe(name, values):
    if not values:
        return f"{name:14s} n=0"
    vs = sorted(values)
    n = len(vs)
    return (f"{name:14s} n={n:4d}  min={min(vs):10.2f}  "
            f"median={vs[n//2]:10.2f}  mean={statistics.mean(vs):10.2f}  "
            f"max={max(vs):10.2f}")


def analyze_one(path: Path, out_lines: list):
    blocks, hist = parse_log(path)
    if not blocks:
        out_lines.append(f"\n=== {path.name} — NO DATA")
        return
    out_lines.append(f"\n=== {path.name}")
    out_lines.append(f"  GS-table blocks: {len(blocks)} (≈ every-20-step samples)")
    out_lines.append(f"  Wrapper-level best_src histogram (every step):")
    total = sum(hist.values())
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
        out_lines.append(f"    {k:20s} {v:4d} / {total} ({100*v/total:.1f}%)")

    # Aggregate per-label
    by_label_c3 = defaultdict(list)
    by_label_csmp = defaultdict(list)
    by_label_bonus = defaultdict(list)
    by_label_pen   = defaultdict(list)
    winners_by_label = defaultdict(int)
    # Per-step gap: c_C3_raw[prev_repos] − min(c_C3_raw[strat_*])
    raw_gaps = []

    for step, mode, rows in blocks:
        row_by_label = {r["label"]: r for r in rows}
        for r in rows:
            by_label_c3   [r["label"]].append(r["c_C3_raw"])
            by_label_csmp [r["label"]].append(r["c_sample"])
            by_label_bonus[r["label"]].append(r["align_bonus"])
            by_label_pen  [r["label"]].append(r["travel_pen"])
            if r["winner"]:
                winners_by_label[r["label"]] += 1
        # Gap calc
        if "prev_repos" in row_by_label:
            strats = [r["c_C3_raw"] for r in rows if r["label"].startswith("strat_")]
            if strats:
                raw_gaps.append(row_by_label["prev_repos"]["c_C3_raw"] - min(strats))

    out_lines.append("  [GS-table] sampled-step winners by label:")
    blocks_n = len(blocks)
    for lbl, n in sorted(winners_by_label.items(), key=lambda kv: -kv[1]):
        out_lines.append(f"    {lbl:20s} {n:3d} / {blocks_n}")

    out_lines.append("  c_C3_raw by label:")
    for lbl in sorted(by_label_c3):
        out_lines.append("    " + _describe(lbl, by_label_c3[lbl]))
    out_lines.append("  c_sample by label:")
    for lbl in sorted(by_label_csmp):
        out_lines.append("    " + _describe(lbl, by_label_csmp[lbl]))
    out_lines.append("  align_bonus by label:")
    for lbl in sorted(by_label_bonus):
        out_lines.append("    " + _describe(lbl, by_label_bonus[lbl]))
    out_lines.append("  travel_penalty by label:")
    for lbl in sorted(by_label_pen):
        out_lines.append("    " + _describe(lbl, by_label_pen[lbl]))
    if raw_gaps:
        out_lines.append("  c_C3_raw gap (prev_repos − min(strat_*)) — positive = prev_repos higher:")
        out_lines.append("    " + _describe("gap", raw_gaps))


def main() -> int:
    out_lines = ["9.4.7 Option C — [GS-table] inspection summary"]
    logs = [Path(a) for a in sys.argv[1:]] or DEFAULT_LOGS
    for p in logs:
        analyze_one(p, out_lines)
    rep = "\n".join(out_lines)
    print(rep)
    summary_path = RESULTS_DIR / "probe_9_4_7_C_gs_table_summary.txt"
    summary_path.write_text(rep + "\n")
    print(f"\n[PROBE 9.4.7-C] wrote summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
