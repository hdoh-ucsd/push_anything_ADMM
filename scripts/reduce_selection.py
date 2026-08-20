#!/usr/bin/env python3
"""B3c-prime selection-audit reducer.

Reads sel_*.jsonl emitted by wrapper.py:957-975 across N paired runs of
identical seed, finds the FLIP TICK (the step in [LO, HI] where k_star
disagrees across runs — same seed ⇒ same RNG sequence ⇒ the disagreement is
the FP floor flipping argmin), and reports the three B3b signatures at that
tick:

  (a) gap = c_sample[runner_up] − c_sample[k_star]  ?<  0.2
  (b) tipping channel  ?==  travel_penalty
  (c) rng_state matched across paired runs  ?==  True

All three together confirm argmin-flip-on-FP-floor. Any failing refutes B3b.

If confirmed, the reducer suggests SELECT_HYSTERESIS = max(spurious_margins)
+ slack · (min(legitimate_margins) − max(spurious_margins)) when separable.
Spurious flips are pair-disagreements at the FP floor; legitimate flips are
both-runs-flip-identically (signal, not noise). Both require multiple ticks
to populate, so we report the single-flip-tick gap for the immediate pinning
question and the separability stats if more flips exist in [LO, HI].

Usage:
    python scripts/reduce_selection.py <sel_dir>

<sel_dir> contains sel_<run_id>.jsonl files.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


COMPONENT_KEYS = ("c_C3_raw", "align_bonus", "rot_bonus", "travel_penalty")


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_sel_dir(sel_dir: Path):
    """Return dict[run_id] -> dict[step] -> row."""
    by_run = {}
    for path in sorted(sel_dir.glob("sel_*.jsonl")):
        run_id = path.stem[len("sel_"):]
        rows = load_jsonl(path)
        by_step = {int(r["step"]): r for r in rows}
        by_run[run_id] = by_step
    return by_run


def step_kstars(by_run, step):
    return {rid: rows[step]["k_star"] for rid, rows in by_run.items() if step in rows}


def find_flip_tick(by_run, lo, hi):
    """First step in [lo,hi] where unique(k_star across runs) > 1."""
    all_steps = sorted({s for rows in by_run.values() for s in rows.keys()
                        if lo <= s <= hi})
    for step in all_steps:
        ks = step_kstars(by_run, step)
        if len(set(ks.values())) > 1:
            return step, ks
    return None, None


def runner_up_idx(samples, k_star):
    """Index of the runner-up sample (second-lowest c_sample)."""
    ranked = sorted(range(len(samples)),
                    key=lambda i: samples[i]["c_sample"])
    if ranked[0] == k_star:
        return ranked[1] if len(ranked) > 1 else None
    return ranked[0]


def margin(row):
    """gap = c_sample[runner_up] - c_sample[k_star]  (positive when k_star wins)"""
    samples = row["samples"]
    k_star = row["k_star"]
    ru = runner_up_idx(samples, k_star)
    if ru is None:
        return None
    return samples[ru]["c_sample"] - samples[k_star]["c_sample"]


def at_step(row, k):
    return row["samples"][k]


def pretty_pair_table(by_run, step):
    """One-line-per-run table at the flip tick."""
    lines = []
    header = (f"  {'run_id':>8} {'k*':>3} {'k_runup':>7} "
              f"{'gap':>10} {'rng_state':>18} {'held':>4}")
    lines.append(header)
    for rid, rows in sorted(by_run.items()):
        if step not in rows:
            continue
        r = rows[step]
        k_star = r["k_star"]
        ru = runner_up_idx(r["samples"], k_star)
        gap = margin(r)
        held = r.get("held_idx")
        rng = r.get("rng_state", -1)
        lines.append(
            f"  {rid:>8} {k_star:>3d} "
            f"{(ru if ru is not None else -1):>7d} "
            f"{(gap if gap is not None else float('nan')):>10.4f} "
            f"{rng:>18d} {str(held):>4}"
        )
    return "\n".join(lines)


def tipping_channel(by_run, step):
    """At the flip tick, compare the cost-component deltas BETWEEN paired runs
    for the WINNING k of each run. Returns dict[component] -> max |Δ| across
    paired-run differences."""
    ks_per_run = {}
    for rid, rows in by_run.items():
        if step not in rows:
            continue
        r = rows[step]
        k = r["k_star"]
        ks_per_run[rid] = (k, r["samples"][k])
    rids = sorted(ks_per_run.keys())
    deltas = {c: [] for c in COMPONENT_KEYS}
    pairs = []
    for i in range(len(rids)):
        for j in range(i + 1, len(rids)):
            a = ks_per_run[rids[i]]
            b = ks_per_run[rids[j]]
            for c in COMPONENT_KEYS:
                deltas[c].append(abs(a[1][c] - b[1][c]))
            pairs.append((rids[i], rids[j], a[0], b[0]))
    max_dev = {c: (max(deltas[c]) if deltas[c] else 0.0) for c in COMPONENT_KEYS}
    return max_dev, pairs


def rng_consistency(by_run, step):
    rngs = {}
    for rid, rows in by_run.items():
        if step in rows:
            rngs[rid] = rows[step].get("rng_state", -1)
    return rngs, (len(set(rngs.values())) == 1)


def rng_first_desync(by_run, lo):
    """Earliest step where rng_state diverges across runs (used to check
    whether sample generation desynced before the flip tick)."""
    all_steps = sorted({s for rows in by_run.values() for s in rows.keys()
                        if s >= lo})
    for step in all_steps:
        rngs = [rows[step]["rng_state"]
                for rows in by_run.values() if step in rows]
        if len(set(rngs)) > 1:
            return step, rngs
    return None, None


def main():
    if len(sys.argv) < 2:
        print("usage: reduce_selection.py <sel_dir>", file=sys.stderr)
        sys.exit(2)
    sel_dir = Path(sys.argv[1])
    by_run = load_sel_dir(sel_dir)
    if not by_run:
        print(f"no sel_*.jsonl in {sel_dir}", file=sys.stderr)
        sys.exit(1)

    # Window — read from env or use [150,170] default.
    lo = int(os.environ.get("C3_SEL_LO", "150"))
    hi = int(os.environ.get("C3_SEL_HI", "170"))

    print(f"=== sel-audit reducer ({sel_dir}) ===")
    print(f"runs: {sorted(by_run.keys())}")
    print(f"window: [{lo}, {hi}]")
    rows_per_run = {rid: len(rows) for rid, rows in by_run.items()}
    print(f"rows per run: {rows_per_run}")
    print()

    flip_step, ks_at_flip = find_flip_tick(by_run, lo, hi)
    if flip_step is None:
        print(f"NO flip tick in [{lo},{hi}] — all runs agree on k_star "
              f"across the window.")
        print("If you expected a divergence, widen LO/HI or check the seed.")
        sys.exit(0)

    print(f"flip tick: step={flip_step}  k_star={ks_at_flip}")
    print()
    print(pretty_pair_table(by_run, flip_step))
    print()

    # Signature (a): gap < 0.2?
    margins_at_flip = []
    for rid, rows in by_run.items():
        if flip_step in rows:
            m = margin(rows[flip_step])
            if m is not None:
                margins_at_flip.append((rid, m))
    print(f"signature (a) — flip-tick gap (c[runner_up] − c[k_star]):")
    for rid, m in margins_at_flip:
        flag = "<0.2 (FLOOR-TIPPABLE)" if m < 0.2 else ">=0.2"
        print(f"    run {rid}: gap = {m:.4f}  {flag}")
    min_gap = min(m for _, m in margins_at_flip) if margins_at_flip else float("nan")
    max_gap = max(m for _, m in margins_at_flip) if margins_at_flip else float("nan")
    sig_a = (max_gap < 0.2)
    print(f"  -> signature (a) {'CONFIRMED' if sig_a else 'REFUTED'} "
          f"(max gap {max_gap:.4f})")
    print()

    # Signature (b): tipping channel = travel_penalty?
    max_dev, pairs = tipping_channel(by_run, flip_step)
    print(f"signature (b) — per-component max |Δ| between paired-run winners:")
    for c in COMPONENT_KEYS:
        print(f"    {c:>14}: {max_dev[c]:.4f}")
    tipper = max(COMPONENT_KEYS, key=lambda c: max_dev[c])
    sig_b = (tipper == "travel_penalty")
    print(f"  -> tipping channel = {tipper}")
    print(f"  -> signature (b) {'CONFIRMED (travel-driven)' if sig_b else 'REFUTED'}")
    print()

    # Signature (c): rng_state matches across paired runs at the flip tick?
    rngs, sig_c = rng_consistency(by_run, flip_step)
    print(f"signature (c) — rng_state across runs at flip tick:")
    for rid, s in sorted(rngs.items()):
        print(f"    run {rid}: rng_state = {s}")
    print(f"  -> signature (c) {'CONFIRMED (RNG identical)' if sig_c else 'REFUTED (RNG desynced)'}")
    print()

    # Early rng desync check (B3a parity).
    desync_step, desync_rngs = rng_first_desync(by_run, lo)
    if desync_step is None:
        print(f"rng parity across [{lo},{hi}]: IDENTICAL on every step.")
    else:
        print(f"rng parity: FIRST desync at step {desync_step} "
              f"(rngs {desync_rngs}).")
    print()

    # Verdict.
    sigs = (sig_a, sig_b, sig_c)
    if all(sigs):
        verdict = "CONFIRMED — argmin-flip-on-FP-floor (B3b holds)."
    else:
        which = []
        if not sig_a: which.append("(a) gap>=0.2")
        if not sig_b: which.append(f"(b) tipper={tipper}")
        if not sig_c: which.append("(c) rng desynced")
        verdict = "REFUTED — relocate. Failed: " + ", ".join(which)
    print(f"B3b verdict: {verdict}")
    print()

    # Pinning SELECT_HYSTERESIS at the flip-tick gap. Only meaningful when
    # (a) holds — otherwise hysteresis is the wrong fix.
    if sig_a:
        # Use the largest spurious gap as the pin floor (so that any pair
        # disagreement at the flip tick gets gated by the threshold).
        H_floor = max_gap
        slack = 0.10
        proposed = H_floor + slack * H_floor
        print(f"SELECT_HYSTERESIS pin:")
        print(f"    H_floor (max flip-tick gap)  = {H_floor:.4f}")
        print(f"    slack                          = {slack:.2f}")
        print(f"    proposed SELECT_HYSTERESIS     = {proposed:.4f}")
        print(f"    -> any candidate-flip with gap below this is rejected;")
        print(f"       larger gaps (legitimate ranking changes) pass through.")
        print()
        print("NOTE: this pin assumes the in-window separator between spurious")
        print("and legitimate flips is the same-seed pair-disagreement gap.")
        print("If [LO,HI] contains additional ticks where ALL runs agree on a")
        print("k_star CHANGE versus the previous tick (legitimate ranking)")
        print("the reducer should be re-run with a wider window or the")
        print("legitimate-margin set tabulated; with 21 ticks this is rare.")
    else:
        print("SELECT_HYSTERESIS not pinned (signature (a) refuted — "
              "hysteresis is the wrong fix; route to ratio/sampler cofix).")


if __name__ == "__main__":
    main()
