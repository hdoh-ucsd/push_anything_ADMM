#!/usr/bin/env python3
"""One-shot log parser for bilevel-observation prompt.

Usage: python scripts/parse_bilevel_log.py <run.log> <out_dir>

Produces, in out_dir:
    transitions.txt           kTo*, kStay* hits one per line
    gs_trace.tsv              every [GS] step → TSV row
    gs_tables.txt             [GS-table] blocks (full block, 14 lines each)
    feasibility_trace.txt     infeas/poison/fallback hits one per line
    summary.txt               aggregate stats (mode counts, pre-/post-fallback
                              regimes, ratios, switch-reason counts)
"""
import re
import sys
import statistics
from pathlib import Path

GS_RE = re.compile(
    r"\[GS\] step=(\d+) mode=(\w+) switch=(\w+) best_k=(\S+) best_src=(\S+) "
    r"curr_cost=([\d\.\-eE]+) repos_cost=(\S+) best_other=(\S+) "
    r"met_progress=(\w) steps_since_improve=(\d+) switches=(\d+)"
)


def parse(log_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = log_path.read_text()

    # --- transitions.txt ----------------------------------------------------
    trans_keys = (
        "kToC3Cost", "kToC3ReachedReposTarget", "kToReposCost",
        "kToReposUnproductive", "kToBetterRepos", "kStayInC3", "kStayInRepos",
    )
    trans_lines = [
        ln for ln in text.splitlines()
        if any(k in ln for k in trans_keys)
    ]
    (out_dir / "transitions.txt").write_text("\n".join(trans_lines) + "\n")

    # --- gs_tables.txt: grab [GS-table] and the 12 lines after each --------
    table_blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "[GS-table]" in lines[i]:
            table_blocks.append("\n".join(lines[i:i + 13]))
            i += 13
        else:
            i += 1
    (out_dir / "gs_tables.txt").write_text("\n\n".join(table_blocks) + "\n")

    # --- feasibility_trace.txt ---------------------------------------------
    feas_pat = re.compile(
        r"infeasible|poison|_infeasible_repos_target|knot-0|"
        r"match.*radius|current_fallback|kIK.*infeas|"
        r"\[REPOS-IK\]|\[KNOT-IK\]|infeas_match"
    )
    feas_lines = [ln for ln in lines if feas_pat.search(ln)]
    (out_dir / "feasibility_trace.txt").write_text("\n".join(feas_lines) + "\n")

    # --- gs_trace.tsv -------------------------------------------------------
    rows = []
    for ln in lines:
        m = GS_RE.match(ln)
        if not m:
            continue
        (step, mode, switch, best_k, best_src,
         curr_cost, repos_cost, best_other,
         met_progress, ssi, switches) = m.groups()
        rows.append((
            int(step), mode, switch, best_k, best_src,
            curr_cost, repos_cost, best_other, met_progress,
            int(ssi), int(switches),
        ))
    header = (
        "step\tmode\tswitch\tbest_k\tbest_src\tcurr_cost\trepos_cost\t"
        "best_other\tmet_progress\tssi\tswitches"
    )
    with (out_dir / "gs_trace.tsv").open("w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    # --- summary.txt --------------------------------------------------------
    n_total = len(rows)
    mode_counts = {}
    switch_counts = {}
    for r in rows:
        mode_counts[r[1]] = mode_counts.get(r[1], 0) + 1
        switch_counts[r[2]] = switch_counts.get(r[2], 0) + 1

    # transitions: count mode != prev_mode
    n_switches = 0
    prev_mode = None
    for r in rows:
        if prev_mode is not None and r[1] != prev_mode:
            n_switches += 1
        prev_mode = r[1]

    # Identify fallback regime
    first_fallback_step = None
    for r in rows:
        if r[4] == "current_fallback":
            first_fallback_step = r[0]
            break

    pre_rows = (
        [r for r in rows if first_fallback_step is None or r[0] < first_fallback_step]
        if first_fallback_step is not None else list(rows)
    )
    post_rows = (
        [r for r in rows if first_fallback_step is not None and r[0] >= first_fallback_step]
        if first_fallback_step is not None else []
    )

    def _stat(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        return f"min={min(vals):.2f}  max={max(vals):.2f}  mean={statistics.mean(vals):.2f}"

    def _floats(col: int, src: list) -> list[float]:
        out = []
        for r in src:
            v = r[col]
            if v in ("-", "inf"):
                continue
            try:
                out.append(float(v))
            except ValueError:
                continue
        return out

    pre_curr = _floats(5, pre_rows)
    pre_best = _floats(7, pre_rows)
    ratios = [c / b for c, b in zip(pre_curr, pre_best) if b > 0]
    n_curr_lt_best = sum(1 for c, b in zip(pre_curr, pre_best) if b > c)

    pre_src_counts = {}
    for r in pre_rows:
        pre_src_counts[r[4]] = pre_src_counts.get(r[4], 0) + 1

    post_recovery = any(r[4] != "current_fallback" for r in post_rows)

    # find [RESULT]
    result_line = ""
    for ln in lines:
        if "[RESULT]" in ln:
            result_line = ln
            break

    # find perf line
    perf_line = ""
    for ln in lines:
        if "[GS-perf]" in ln:
            perf_line = ln
            break

    with (out_dir / "summary.txt").open("w") as f:
        f.write(f"# {log_path}\n")
        f.write(f"# parsed rows: {n_total}\n\n")
        f.write("## Result\n")
        f.write(f"  {result_line}\n  {perf_line}\n\n")

        f.write("## Mode counts\n")
        for k, v in mode_counts.items():
            f.write(f"  {k:>6} : {v}\n")
        f.write(f"  switches (mode transitions) : {n_switches}\n\n")

        f.write("## Switch-reason counts\n")
        for k in (
            "kStayInC3", "kStayInRepos",
            "kToC3Cost", "kToC3ReachedReposTarget",
            "kToReposCost", "kToReposUnproductive",
            "kToBetterRepos", "kForceC3Watchdog",
        ):
            f.write(f"  {k:>26} : {switch_counts.get(k, 0)}\n")
        f.write("\n")

        f.write("## Pre-fallback regime\n")
        f.write(f"  first fallback step       : {first_fallback_step}\n")
        f.write(f"  pre-fallback step count   : {len(pre_rows)}\n")
        f.write(f"  curr_cost                 : {_stat(pre_curr)}\n")
        f.write(f"  best_other_cost           : {_stat(pre_best)}\n")
        if ratios:
            f.write(
                f"  curr/best ratio           : min={min(ratios):.3f}  "
                f"max={max(ratios):.3f}  mean={statistics.mean(ratios):.3f}\n"
            )
        f.write(f"  steps with best>curr      : {n_curr_lt_best}\n")
        f.write(f"  best_src distribution     : {pre_src_counts}\n\n")

        f.write("## Fallback regime\n")
        f.write(f"  step count                : {len(post_rows)}\n")
        f.write(
            f"  best_other shown as 'inf' : "
            f"{sum(1 for r in post_rows if r[7] == 'inf')}\n"
        )
        f.write(
            f"  best_other shown as '-'   : "
            f"{sum(1 for r in post_rows if r[7] == '-')}\n"
        )
        f.write(f"  recovered from fallback   : {post_recovery}\n\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: parse_bilevel_log.py <run.log> <out_dir>", file=sys.stderr)
        sys.exit(2)
    parse(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"parsed -> {sys.argv[2]}")
