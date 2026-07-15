"""Per-seed parser for Stage 2 L2 commit-face gate sweep.

Extracts per seed:
  - final_obj_xy, goal_dist, success
  - first-contact face (parsed from first EE-BOX nhat_BA_W)
  - contact_ticks (count of EE-BOX [CONTACT-RUN] lines)
  - [GATE-COMMIT-FACE] refusal count + severity-tag breakdown
  - [GATE-ALIGN] (L1) fire count
  - lambda_n_max distribution, lcp_res_max distribution

Then reports against the pre-registered SCs:
  - SC-L2-convert         (seeds 1, 2): final.x < -0.05 AND first-face = +x
  - SC-L2-noregress-working (seeds 0,3,4): final.x < {-0.45, -0.50, -0.26}
                                          AND first-face = +x
  - SC-L2-deadlock        (seeds 1,2): contact_ticks == 0 ?
                                       split by tag-dominance:
                                       near-miss-dominant -> relax threshold
                                       perpendicular/anti-goal dominant -> sampler fix
  - SC-L2-lambda          : lambda distribution preserved (gate doesn't touch face dist)
  - SC-L2-L1-redundancy   : total [GATE-ALIGN] across seeds
                            0 -> L1 redundant, Task 8 removes it
                            >0 -> L2 gap exists, report attribution

Goal direction for pushing-W (task-id 4) is West (-x); goal-aligned first-face
is +x (EE on east face, pushes west).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

RE_RESULT = re.compile(
    r"^\[RESULT\] method=\S+\s+final_obj_xy=\(([-+0-9.]+),\s*([-+0-9.]+)\)"
    r"\s+goal_dist=([0-9.]+)m\s+success=(YES|NO)"
)
RE_CONTACT_RUN_EEBOX = re.compile(
    r"^\[CONTACT-RUN\] step=(\d+) nhat_BA_W=\[([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\]"
    r".*contact_type=EE-BOX"
)
RE_GATE_L1 = re.compile(r"^\[GATE-ALIGN\] step=(\d+) refused: align=([-+0-9.]+)")
RE_GATE_L2 = re.compile(
    r"^\[GATE-COMMIT-FACE\] step=(\d+) refused: face_align=([-+0-9.]+)\s+tag=(\S+)"
)
RE_C3PLUS = re.compile(
    r"^\[C3\+\] step=(\d+) .*λ_n_max=([0-9.eE+-]+) .*primal=([0-9.eE+-]+)"
    r" iters=(\d+)/\d+ lcp_res_max=([0-9.eE+-]+)"
)


def classify_face(nx: float, ny: float) -> str:
    """Classify dominant XY axis. Returns '+x', '-x', '+y', or '-y'."""
    if abs(nx) >= abs(ny):
        return "+x" if nx > 0 else "-x"
    return "+y" if ny > 0 else "-y"


def parse_seed(log_path: Path) -> dict:
    out = {
        "log": str(log_path),
        "final_xy": None,
        "goal_dist": None,
        "success": None,
        "first_contact_step": None,
        "first_contact_face": None,
        "first_contact_nhat_xy": None,
        "contact_ticks": 0,
        "n_gate_L1": 0,
        "n_gate_L2": 0,
        "tags": {"near-miss-cone": 0, "perpendicular": 0, "anti-goal": 0},
        "l1_steps": [],
        "l2_steps": [],
        "lam_n_max_obs": [],
        "lcp_res_obs": [],
    }
    if not log_path.exists():
        return out
    with log_path.open() as fp:
        for line in fp:
            m = RE_RESULT.match(line)
            if m:
                out["final_xy"] = (float(m.group(1)), float(m.group(2)))
                out["goal_dist"] = float(m.group(3))
                out["success"] = m.group(4) == "YES"
                continue
            m = RE_CONTACT_RUN_EEBOX.match(line)
            if m:
                step = int(m.group(1))
                nx, ny = float(m.group(2)), float(m.group(3))
                out["contact_ticks"] += 1
                if out["first_contact_step"] is None:
                    out["first_contact_step"] = step
                    out["first_contact_face"] = classify_face(nx, ny)
                    out["first_contact_nhat_xy"] = (nx, ny)
                continue
            m = RE_GATE_L1.match(line)
            if m:
                out["n_gate_L1"] += 1
                out["l1_steps"].append((int(m.group(1)), float(m.group(2))))
                continue
            m = RE_GATE_L2.match(line)
            if m:
                out["n_gate_L2"] += 1
                tag = m.group(3)
                if tag in out["tags"]:
                    out["tags"][tag] += 1
                out["l2_steps"].append((int(m.group(1)), float(m.group(2)), tag))
                continue
            m = RE_C3PLUS.match(line)
            if m:
                out["lam_n_max_obs"].append(float(m.group(2)))
                out["lcp_res_obs"].append(float(m.group(5)))
                continue
    return out


def fmt_seed(seed: int, d: dict) -> str:
    lines = []
    lines.append(f"=== seed={seed} ===  log={d['log']}")
    fx = d["final_xy"]
    if fx is None:
        lines.append("  [RESULT] not found — sim may have crashed or no result line")
    else:
        lines.append(
            f"  final_obj_xy=({fx[0]:+.4f}, {fx[1]:+.4f})  "
            f"goal_dist={d['goal_dist']:.4f}m  success={d['success']}"
        )
    if d["first_contact_face"] is not None:
        nx, ny = d["first_contact_nhat_xy"]
        lines.append(
            f"  first_contact: step={d['first_contact_step']} face={d['first_contact_face']} "
            f"nhat_xy=({nx:+.3f},{ny:+.3f})"
        )
    else:
        lines.append("  first_contact: NONE (no EE-BOX contact in the run)")
    lines.append(f"  contact_ticks (EE-BOX [CONTACT-RUN] count): {d['contact_ticks']}")
    lines.append(
        f"  [GATE-COMMIT-FACE] (L2) refusals: {d['n_gate_L2']}  "
        f"tags: near-miss={d['tags']['near-miss-cone']} "
        f"perp={d['tags']['perpendicular']} "
        f"anti-goal={d['tags']['anti-goal']}"
    )
    lines.append(f"  [GATE-ALIGN] (L1) refusals: {d['n_gate_L1']}")
    if d["lam_n_max_obs"]:
        lam = d["lam_n_max_obs"]
        lcp = d["lcp_res_obs"]
        lines.append(
            f"  C3+ solves: n={len(lam)}  "
            f"λ_n_max: max={max(lam):.3f} mean={sum(lam)/len(lam):.3f}  "
            f"lcp_res_max: max={max(lcp):.2e}"
        )
    return "\n".join(lines)


SC_NO_REGRESS_X = {0: -0.45, 3: -0.50, 4: -0.26}


def verdict(per_seed: dict) -> str:
    out = []
    out.append("\n=========================================================")
    out.append(" PRE-REGISTERED SC READOUT")
    out.append("=========================================================")

    # SC-L2-convert (seeds 1, 2)
    out.append("\n[SC-L2-convert]  target: seeds 1, 2 — final.x < -0.05 AND first-face = +x")
    for s in (1, 2):
        d = per_seed.get(s, {})
        fx = d.get("final_xy")
        ff = d.get("first_contact_face")
        if fx is None:
            out.append(f"  seed {s}: NO RESULT (skip)")
            continue
        cond_x = fx[0] < -0.05
        cond_face = ff == "+x"
        verdict = "PASS" if (cond_x and cond_face) else "FAIL"
        out.append(
            f"  seed {s}: final.x={fx[0]:+.4f} ({'OK' if cond_x else 'NO'}) "
            f"first-face={ff} ({'OK' if cond_face else 'NO'}) -> {verdict}"
        )

    # SC-L2-noregress-working (seeds 0, 3, 4)
    out.append("\n[SC-L2-noregress-working]  target: seeds 0/3/4 — final.x < per-seed bar + first-face = +x")
    for s in (0, 3, 4):
        d = per_seed.get(s, {})
        fx = d.get("final_xy")
        ff = d.get("first_contact_face")
        bar = SC_NO_REGRESS_X[s]
        if fx is None:
            out.append(f"  seed {s}: NO RESULT (skip)")
            continue
        cond_x = fx[0] < bar
        cond_face = ff == "+x"
        verdict = "PASS" if (cond_x and cond_face) else "FAIL"
        out.append(
            f"  seed {s}: final.x={fx[0]:+.4f} (bar {bar:+.2f}; {'OK' if cond_x else 'NO'}) "
            f"first-face={ff} ({'OK' if cond_face else 'NO'}) -> {verdict}"
        )

    # SC-L2-deadlock (1, 2) — if contact_ticks==0, route on tag dominance
    out.append("\n[SC-L2-deadlock]  seeds 1,2 contact_ticks==0 -> route by severity-tag dominance")
    for s in (1, 2):
        d = per_seed.get(s, {})
        ct = d.get("contact_ticks", 0)
        if ct > 0:
            out.append(f"  seed {s}: contact_ticks={ct} (no deadlock)")
            continue
        t = d.get("tags", {})
        nm = t.get("near-miss-cone", 0)
        pp = t.get("perpendicular", 0)
        ag = t.get("anti-goal", 0)
        tot = nm + pp + ag
        if tot == 0:
            out.append(f"  seed {s}: contact_ticks=0 AND no L2 refusals — investigate (not classical deadlock)")
            continue
        nm_share = nm / tot
        wrong_share = (pp + ag) / tot
        if nm_share > 0.5:
            route = "near-miss-dominant -> RELAX threshold (-0.7 -> -0.6 or -0.5)"
        elif wrong_share > 0.5:
            route = "wrong-face-dominant (perp/anti-goal) -> SAMPLER FIX (sampler not producing +x face)"
        else:
            route = "mixed — needs deeper inspection"
        out.append(
            f"  seed {s}: contact_ticks=0  L2-refusals={tot} "
            f"(nm={nm} {nm_share:.0%}, perp={pp}, anti={ag}; wrong={wrong_share:.0%}) -> {route}"
        )

    # SC-L2-lambda — qualitative (max λ_n_max preserved)
    out.append("\n[SC-L2-lambda]  λ distribution unaffected by gate (gate doesn't touch face dist)")
    max_lam_per_seed = {}
    for s in sorted(per_seed):
        lam = per_seed[s].get("lam_n_max_obs", [])
        max_lam_per_seed[s] = max(lam) if lam else None
    out.append(
        "  per-seed max λ_n_max: "
        + ", ".join(f"seed{s}={v:.3f}" if v is not None else f"seed{s}=N/A"
                    for s, v in sorted(max_lam_per_seed.items()))
    )
    out.append("  (compare against L1 sweep λ distribution; gate shouldn't depress it)")

    # SC-L2-L1-redundancy — total L1 fires
    total_l1 = sum(d.get("n_gate_L1", 0) for d in per_seed.values())
    total_l2 = sum(d.get("n_gate_L2", 0) for d in per_seed.values())
    out.append("\n[SC-L2-L1-redundancy]  total [GATE-ALIGN] fires across 5 seeds = "
               f"{total_l1}  (total L2 fires = {total_l2})")
    if total_l1 == 0:
        out.append("  -> L1 confirmed REDUNDANT under L2. Task 8 removes [GATE-ALIGN] block; "
                   "re-verify variance recovery (good branch: NET GUARD DECREASE).")
    else:
        # Per-step attribution (L1-only / L2-only / both)
        l1_only = l2_only = both = 0
        for d in per_seed.values():
            l1_steps = {s for s, _ in d.get("l1_steps", [])}
            l2_steps = {s for s, _, _ in d.get("l2_steps", [])}
            both += len(l1_steps & l2_steps)
            l1_only += len(l1_steps - l2_steps)
            l2_only += len(l2_steps - l1_steps)
        out.append(
            f"  attribution across seeds (per step): L1-only={l1_only} "
            f"L2-only={l2_only} both={both}"
        )
        if l1_only > 0:
            out.append(f"  -> L2 has a GAP at {l1_only} step(s): L1 caught what L2 missed "
                       "(post-contact face that L2's pre-arrival target didn't predict). "
                       "Inspect those steps; don't remove L1.")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="stage2_L2_sweep")
    args = ap.parse_args()
    d = Path(args.dir)
    per_seed = {}
    for s in range(5):
        log = d / f"seed{s}_stage2_L2.log"
        per_seed[s] = parse_seed(log)
        print(fmt_seed(s, per_seed[s]))
        print()
    print(verdict(per_seed))


if __name__ == "__main__":
    main()
