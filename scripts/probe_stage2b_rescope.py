"""Stage 2B rescope — H1/H2/H3 mechanism probe on β=2 logs.

Read-only against stage2b_sampling_sweep/seed{3,4}_stage2b_sampling.log
(commit 5b0dc91, face_bias_strength=2.0).

Three questions:
  Q-lambda  (seed 3): at λ_n≥5 / lcp_res>1e-6 ticks, what are the
            n_face_out magnitudes? Off-cardinal (∈(0.3,0.7)) confirms the
            glancing-contact-cracks-projection mechanism.
  Q-H2     (seeds 3,4): at the contact-decision ticks where dispatcher
            picks NON-+x face under β=2, what is the c_C3_raw spread among
            +x samples vs the non-+x winner? If +x samples cluster (small
            spread) AND a unique non-+x has a uniquely low c_C3_raw, the
            tie-break collapses to the non-+x — H2 confirmed, fix locus
            is dispatcher ranking, not sampler.
  Q-H1     (seeds 3,4): are buffer samples (k=4 "buffer") stale and
            unreachable at the off-+x picks? If buffer dominates the win
            when current EE is far from it, H1 contributing.
  Q-H3     (seeds 3,4): at late-run off-+x picks, is feas=N or ik_err high
            on the +x candidates? Reachability-starvation indicator.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO = Path("/root/push_anything_ADMM")
SWEEP = REPO / "stage2b_sampling_sweep"

# --- regexes -------------------------------------------------------------
RE_GATE = re.compile(
    r"^\[GATE-CONTACT\]\s+step=(\d+).*?"
    r"n_face_out=\(([+\-0-9.,e]+)\).*?A_is_ee=(\d).*?"
    r"box_p=\(([+\-0-9.,e]+)\)"
)
RE_C3 = re.compile(
    r"^\[C3\+\]\s+step=(\d+)\s+\|u\[0\]\|=\S+\s+"
    r"λ_n_max=([0-9.eE+-]+).*?lcp_res_max=([0-9.eE+-]+)"
)
RE_KSAMPLE = re.compile(
    r"^\s+k=(\d+)\s+\(([^)]+)\)\s+pos=\(([+\-0-9.,e]+)\)\s+"
    r"c_C3=\s*([0-9.eE+-]+)\s+align=([0-9.eE+-]+)\(bonus=\s*([0-9.eE+-]+)\)\s+"
    r"rot=([+\-0-9.eE]+)\(bonus=\s*([0-9.eE+-]+)\)\s+"
    r"travel=([0-9.eE+-]+)m\(pen=\s*([0-9.eE+-]+)\)\s+"
    r"c_sample=\s*([0-9.eE+-]+)\s+feas=([YN])\s+ik_err=([0-9.eE+-]+)m"
    r"(?:\s+(← WIN))?"
)


def parse_sample_block(line):
    m = RE_KSAMPLE.match(line)
    if not m:
        return None
    sx, sy, sz = (float(x) for x in m.group(3).split(","))
    return {
        "k": int(m.group(1)),
        "src": m.group(2).strip(),
        "pos": (sx, sy, sz),
        "c_C3": float(m.group(4)),
        "align": float(m.group(5)),
        "align_bonus": float(m.group(6)),
        "travel_pen": float(m.group(10)),
        "c_sample": float(m.group(11)),
        "feas": m.group(12) == "Y",
        "ik_err": float(m.group(13)),
        "win": m.group(14) is not None,
    }


def parse_log(path: Path):
    """Return list of per-tick records with paired GATE/C3+/sample-block."""
    ticks = []
    cur_block = []
    last_gate = None
    last_c3 = None
    with path.open() as f:
        for line in f:
            ksamp = parse_sample_block(line)
            if ksamp is not None:
                cur_block.append(ksamp)
                continue
            mg = RE_GATE.match(line)
            if mg:
                nfo = tuple(float(x) for x in mg.group(2).split(","))
                bp = tuple(float(x) for x in mg.group(4).split(","))
                last_gate = {
                    "gate_step": int(mg.group(1)),
                    "n_face_out": nfo,
                    "A_is_ee": int(mg.group(3)),
                    "box_p": (bp[0], bp[1], bp[2]),
                }
                continue
            mc = RE_C3.match(line)
            if mc:
                last_c3 = {
                    "c3_step": int(mc.group(1)),
                    "lam_n_max": float(mc.group(2)),
                    "lcp_res": float(mc.group(3)),
                }
                continue
            if line.startswith("[GS] step="):
                # GS line closes the tick: flush sample block + most recent
                # GATE/C3 we saw since the last GS into a tick record.
                if cur_block or last_gate or last_c3:
                    ticks.append({
                        "samples": cur_block,
                        "gate": last_gate,
                        "c3": last_c3,
                    })
                cur_block = []
                last_gate = None
                last_c3 = None
    if cur_block or last_gate or last_c3:
        ticks.append({"samples": cur_block, "gate": last_gate, "c3": last_c3})
    return ticks


def face_label(pos, box_xy):
    """Classify a sample by box-relative face (consistent with /understand)."""
    dx = pos[0] - box_xy[0]
    dy = pos[1] - box_xy[1]
    if abs(dx) > abs(dy):
        return "+x" if dx > 0 else "-x"
    else:
        return "+y" if dy > 0 else "-y"


def cardinal_score(nfo):
    """Cardinality: max(|nx|,|ny|). Cardinal contact ≈ 1.0; off-cardinal/
    glancing ≈ 0.5–0.7."""
    return max(abs(nfo[0]), abs(nfo[1]))


# -----------------------------------------------------------------------
# Q-lambda: seed 3 — n_face_out at violation ticks
# -----------------------------------------------------------------------
def q_lambda_seed3(ticks):
    print("=== Q-lambda: seed 3 n_face_out at λ_n≥5 OR lcp_res>1e-6 ticks ===")
    rows = []
    for t in ticks:
        if t["c3"] is None: continue
        lam = t["c3"]["lam_n_max"]
        res = t["c3"]["lcp_res"]
        if lam >= 5.0 or res > 1.0e-6:
            if t["gate"] is None: continue
            nfo = t["gate"]["n_face_out"]
            rows.append((t["c3"]["c3_step"], lam, res, nfo, cardinal_score(nfo)))
    if not rows:
        print("  no violation/elevated-residual ticks"); return
    print(f"  {len(rows)} elevated ticks. Cardinality breakdown:")
    cardinal = [r for r in rows if r[4] > 0.9]
    glancing = [r for r in rows if 0.3 <= r[4] <= 0.7]
    near_zero = [r for r in rows if r[4] < 0.3]
    mid_high = [r for r in rows if 0.7 < r[4] <= 0.9]
    print(f"    cardinal  (max|nx|or|ny|>0.9):   {len(cardinal)}/{len(rows)}")
    print(f"    mid-high  (0.7 < c ≤ 0.9):       {len(mid_high)}/{len(rows)}")
    print(f"    glancing  (0.3 ≤ c ≤ 0.7):       {len(glancing)}/{len(rows)}  ← H_lambda predicts here")
    print(f"    near-zero (c < 0.3):             {len(near_zero)}/{len(rows)}")
    print(f"  first 8 elevated ticks:")
    for step, lam, res, nfo, c in rows[:8]:
        print(f"    step={step:>4d}  λ_n={lam:>6.2f}  lcp_res={res:.2e}  "
              f"n_face_out=({nfo[0]:+.3f},{nfo[1]:+.3f},{nfo[2]:+.3f})  cardinality={c:.3f}")


# -----------------------------------------------------------------------
# Q-H2: c_C3 clustering of +x vs non-+x at non-+x WIN ticks
# -----------------------------------------------------------------------
def q_h2(label, ticks, focus_face_committed="-y"):
    """For ticks where the committed contact face is NOT +x (per gate
    n_face_out), inspect the sample-block: did +x samples cluster on c_C3
    while a unique non-+x sample had a uniquely low c_C3?"""
    print(f"=== Q-H2: {label} c_C3 clustering — looking at non-+x WIN ticks ===")
    found = 0
    spreads = []
    for t in ticks:
        if t["gate"] is None or t["gate"]["A_is_ee"] != 1: continue
        nfo = t["gate"]["n_face_out"]
        # Skip cardinal +x contact (we want the wrong-face ticks)
        if nfo[0] > 0.9: continue
        if not t["samples"]: continue
        box_xy = t["gate"]["box_p"][:2]
        # Classify each sample
        plus_x = []
        other = []
        winner = None
        for s in t["samples"]:
            lbl = face_label(s["pos"], box_xy)
            if lbl == "+x":
                plus_x.append(s)
            else:
                other.append(s)
            if s["win"]:
                winner = (s, lbl)
        if not plus_x or winner is None or winner[1] == "+x":
            continue
        # The interesting case: at least one +x candidate, winner is non-+x
        found += 1
        if found <= 6:
            plus_x_c3 = sorted(s["c_C3"] for s in plus_x)
            plus_x_csamp = sorted(s["c_sample"] for s in plus_x)
            spread_c3 = max(plus_x_c3) - min(plus_x_c3) if len(plus_x_c3) > 1 else 0.0
            spread_csamp = max(plus_x_csamp) - min(plus_x_csamp) if len(plus_x_csamp) > 1 else 0.0
            w = winner[0]
            print(f"  tick gate_step={t['gate']['gate_step']:>4d}  "
                  f"n_face_out=({nfo[0]:+.2f},{nfo[1]:+.2f})  box_xy=({box_xy[0]:+.3f},{box_xy[1]:+.3f})")
            print(f"    +x candidates: n={len(plus_x)}  c_C3 ∈ [{min(plus_x_c3):.1f}, {max(plus_x_c3):.1f}] (spread={spread_c3:.1f})  "
                  f"c_sample ∈ [{min(plus_x_csamp):.1f}, {max(plus_x_csamp):.1f}]")
            print(f"    winner src={w['src']!r:>14s} face={winner[1]}  c_C3={w['c_C3']:.1f}  c_sample={w['c_sample']:.1f}  align={w['align']:.3f}")
            print(f"    +x min c_sample = {min(plus_x_csamp):.1f}  vs winner c_sample = {w['c_sample']:.1f}  delta = {min(plus_x_csamp) - w['c_sample']:+.1f}")
            spreads.append((spread_c3, spread_csamp))
        else:
            plus_x_c3 = sorted(s["c_C3"] for s in plus_x)
            plus_x_csamp = sorted(s["c_sample"] for s in plus_x)
            spreads.append((
                max(plus_x_c3) - min(plus_x_c3) if len(plus_x_c3) > 1 else 0.0,
                max(plus_x_csamp) - min(plus_x_csamp) if len(plus_x_csamp) > 1 else 0.0,
            ))
    print(f"  total non-+x-WIN ticks with +x candidates available: {found}")
    if spreads:
        med_spread_c3 = sorted([s[0] for s in spreads])[len(spreads)//2]
        med_spread_csamp = sorted([s[1] for s in spreads])[len(spreads)//2]
        print(f"  median +x c_C3 spread across those ticks:    {med_spread_c3:.1f}")
        print(f"  median +x c_sample spread across those ticks: {med_spread_csamp:.1f}")


# -----------------------------------------------------------------------
# Q-H1: stale buffer winning while current EE is far?
# -----------------------------------------------------------------------
def q_h1(label, ticks):
    """Count ticks where the winner src is 'buffer' or stale prev_repos
    AND distance from current EE to winning sample is large.
    """
    print(f"=== Q-H1: {label} buffer-staleness check ===")
    buffer_wins = 0
    total_wins = 0
    far_buffer_wins = 0
    for t in ticks:
        for s in t["samples"]:
            if s["win"]:
                total_wins += 1
                if "buffer" in s["src"]:
                    buffer_wins += 1
                    # Distance from current_ee (k=0) to winner
                    cur_ee = next((c["pos"] for c in t["samples"] if c["k"] == 0), None)
                    if cur_ee is not None:
                        dx = s["pos"][0] - cur_ee[0]
                        dy = s["pos"][1] - cur_ee[1]
                        d = (dx*dx + dy*dy) ** 0.5
                        if d > 0.10:  # 10cm
                            far_buffer_wins += 1
                break
    print(f"  buffer-source wins: {buffer_wins}/{total_wins}  ({100*buffer_wins/max(1,total_wins):.1f}%)")
    print(f"  buffer wins > 10cm from current EE: {far_buffer_wins}/{buffer_wins}")


# -----------------------------------------------------------------------
# Q-H3: +x sample reachability (feasibility / ik_err) at non-+x WIN ticks
# -----------------------------------------------------------------------
def q_h3(label, ticks):
    print(f"=== Q-H3: {label} +x sample reachability at non-+x WIN ticks ===")
    plus_x_infeasible = 0
    plus_x_total = 0
    high_ik_err = 0
    for t in ticks:
        if t["gate"] is None or t["gate"]["A_is_ee"] != 1: continue
        nfo = t["gate"]["n_face_out"]
        if nfo[0] > 0.9: continue
        if not t["samples"]: continue
        box_xy = t["gate"]["box_p"][:2]
        for s in t["samples"]:
            lbl = face_label(s["pos"], box_xy)
            if lbl != "+x": continue
            plus_x_total += 1
            if not s["feas"]: plus_x_infeasible += 1
            if s["ik_err"] > 0.020: high_ik_err += 1
    print(f"  +x candidates at non-+x-WIN ticks: {plus_x_total}")
    print(f"    feas=N: {plus_x_infeasible} ({100*plus_x_infeasible/max(1,plus_x_total):.1f}%)")
    print(f"    ik_err > 20mm: {high_ik_err} ({100*high_ik_err/max(1,plus_x_total):.1f}%)")


# -----------------------------------------------------------------------
def main():
    print("Stage 2B rescope probe — β=2 logs (commit 5b0dc91)\n")
    for seed in [3, 4]:
        log = SWEEP / f"seed{seed}_stage2b_sampling.log"
        print(f"\n############### seed={seed} ###############\n")
        ticks = parse_log(log)
        print(f"parsed {len(ticks)} tick blocks")
        if seed == 3:
            q_lambda_seed3(ticks)
            print()
        q_h2(f"seed {seed}", ticks)
        print()
        q_h1(f"seed {seed}", ticks)
        print()
        q_h3(f"seed {seed}", ticks)
        print()


if __name__ == "__main__":
    main()
