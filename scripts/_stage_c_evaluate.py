"""Stage C Phase 1 — GAP-CLOSED VERDICT + BOX-MOTION DIAGNOSTIC evaluator.

Consumes [STAGE-A-TRACE] (the per-step trace already emitted by
sampling_based_c3_controller.py:2737) for V1, V2, V4 + box-motion + qy/qz,
and consumes [FORCE-ROUTE tick=] (the per-tick u_sol trace added in commit
d43daa1) for D-1 (eq=True at every c3 tick) and V3 (var(u_z), max|u_z|).

Verdict per §5.E.1.a (plan
docs/superpowers/plans/2026-06-23-stage-c-executor-force-tracking-port.md):
  V1 sustained-contact: c3-mode ticks with phi<2mm AND lam_n_ee_box>0 ≥ 30%
  V2 admit-active:       c3-mode ticks with lam_n_ee_box > 0.5 N           ≥ 15%
  V3 F_z NON-TRIVIAL:    var(u_z) > 0.01 N² AND max|u_z| > 0.5 N (live source)
  V4 no-phantom:         no c3 tick has lam_n_ee_box>0.5 AND phi>=2mm

Diagnostic (NOT a pass-gate, §5.E.1.b):
  Box motion: goal-aligned CoM displacement over the run window.

Cell-level pass: BOTH seeds clear V1∧V2∧V3∧V4 (SEED_PASS_FRACTION=1.0).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# --- §5.E.1.a verdict bars ---
SUSTAINED_CONTACT_BAR = 0.30   # V1
ADMIT_ACTIVE_BAR      = 0.15   # V2
V3_VAR_UZ_BAR         = 0.01   # V3
V3_MAX_ABS_UZ_BAR     = 0.5    # V3
PHI_ADMIT_M           = 0.002  # 2 mm — the LCS admit threshold

# --- §5.E.1.b diagnostic split point ---
BOX_MOTION_DIAGNOSTIC_THRESHOLD_M = 0.005

# --- Cell-pass rule (Stage C scoped to seeds {0, 4}; both must clear) ---
SEED_PASS_FRACTION = 1.0

# Goal direction for task-id 4 is west (-x in world). Box starts at origin in
# the canonical pushing task; goal_xy = (-0.30, 0.0). Goal-aligned displacement
# = (obj_xy_end − obj_xy_start) · g_hat_xy. Stage A trace gives box_xy directly.
TASK_ID_4_GOAL_XY = (-0.30, 0.0)

RE_STAGE_A_TRACE = re.compile(
    r"\[STAGE-A-TRACE\] step=(\d+) sim_t=([+-]?\d*\.?\d+) "
    r"mode=(\w+) phi=([+-]?\d*\.?\d+(?:e[+-]?\d+)?|nan) "
    r"box_xy=([+-]?\d*\.?\d+),([+-]?\d*\.?\d+) "
    r"lam_n_ee_box=([+-]?\d*\.?\d+(?:e[+-]?\d+)?|nan) "
    r"qy=([+-]?\d*\.?\d+) qz=([+-]?\d*\.?\d+) "
    r"finished_repos=(\d+)"
)
RE_FORCE_ROUTE = re.compile(
    r"\[FORCE-ROUTE\] tick=(\d+) env=(\w+) "
    r"lambda_des=\[([+-]?\d*\.?\d+),([+-]?\d*\.?\d+),([+-]?\d*\.?\d+)\] "
    r"u_seq0=\[([+-]?\d*\.?\d+),([+-]?\d*\.?\d+),([+-]?\d*\.?\d+)\] "
    r"u_z=([+-]?\d*\.?\d+) eq=(True|False)"
)


def _safe_float(s: str) -> float:
    if s == "nan":
        return float("nan")
    return float(s)


def parse_log(path: Path) -> dict:
    """Parse one run log → per-step records + per-tick force-route trace."""
    steps = {}          # step -> dict(mode, phi, box_xy, lam_n_ee_box, qy, qz)
    routes = []         # list of dicts (one per [FORCE-ROUTE] line)
    box_xy_first = None
    box_xy_last = None
    with path.open() as f:
        for line in f:
            m = RE_STAGE_A_TRACE.search(line)
            if m:
                step = int(m.group(1))
                rec = dict(
                    sim_t=float(m.group(2)),
                    mode=m.group(3),
                    phi=_safe_float(m.group(4)),
                    box_x=float(m.group(5)),
                    box_y=float(m.group(6)),
                    lam_n_ee_box=_safe_float(m.group(7)),
                    qy=float(m.group(8)),
                    qz=float(m.group(9)),
                    finished_repos=int(m.group(10)),
                )
                steps[step] = rec
                if box_xy_first is None:
                    box_xy_first = (rec["box_x"], rec["box_y"])
                box_xy_last = (rec["box_x"], rec["box_y"])
            m = RE_FORCE_ROUTE.search(line)
            if m:
                routes.append(dict(
                    tick=int(m.group(1)),
                    env=m.group(2),
                    ld=(float(m.group(3)), float(m.group(4)), float(m.group(5))),
                    u0=(float(m.group(6)), float(m.group(7)), float(m.group(8))),
                    u_z=float(m.group(9)),
                    eq=(m.group(10) == "True"),
                ))
    return dict(steps=steps, routes=routes,
                box_xy_first=box_xy_first, box_xy_last=box_xy_last)


def goal_aligned_motion_m(parsed: dict) -> float:
    if not parsed["box_xy_first"] or not parsed["box_xy_last"]:
        return 0.0
    x0, y0 = parsed["box_xy_first"]
    x1, y1 = parsed["box_xy_last"]
    gx_n, gy_n = TASK_ID_4_GOAL_XY
    norm = (gx_n ** 2 + gy_n ** 2) ** 0.5
    if norm < 1e-9:
        return 0.0
    gx_n, gy_n = gx_n / norm, gy_n / norm
    return (x1 - x0) * gx_n + (y1 - y0) * gy_n


def evaluate_seed(parsed: dict) -> dict:
    c3_steps = [s for s, r in parsed["steps"].items() if r["mode"] == "c3"]
    n_c3 = len(c3_steps)

    # V1 sustained-contact: c3 ticks with phi < 2mm AND lam_n_ee_box > 0
    # (admitted EE-BOX pair); lam_n_ee_box=nan means no EE-BOX pair admitted.
    n_v1 = sum(
        1 for s in c3_steps
        if (not _is_nan(parsed["steps"][s]["phi"]))
        and parsed["steps"][s]["phi"] < PHI_ADMIT_M
        and (not _is_nan(parsed["steps"][s]["lam_n_ee_box"]))
        and parsed["steps"][s]["lam_n_ee_box"] > 0.0
    )
    sc_rate = n_v1 / max(n_c3, 1)

    # V2 admit-active: c3 ticks with lam_n_ee_box > 0.5 N (planner-admitted
    # AND magnitude above the friction-floor threshold; admits the EE-BOX
    # pair AND commits force above the marginal-prediction noise floor).
    n_v2 = sum(
        1 for s in c3_steps
        if (not _is_nan(parsed["steps"][s]["lam_n_ee_box"]))
        and parsed["steps"][s]["lam_n_ee_box"] > 0.5
    )
    aa_rate = n_v2 / max(n_c3, 1)

    # V4 no-phantom: no c3 tick has lam_n_ee_box>0.5 AND phi>=2mm.
    # An admitted EE-BOX pair commiting force from beyond the 2mm geometric
    # window indicates a stale buffer or phantom admit.
    n_phantom = sum(
        1 for s in c3_steps
        if (not _is_nan(parsed["steps"][s]["lam_n_ee_box"]))
        and parsed["steps"][s]["lam_n_ee_box"] > 0.5
        and (not _is_nan(parsed["steps"][s]["phi"]))
        and parsed["steps"][s]["phi"] >= PHI_ADMIT_M
    )
    no_phantom = (n_phantom == 0)

    # V4 PRECISION NOTE (per execution prompt): V4 has teeth only when there
    # ARE admits. In a no-contact run V4 is vacuously satisfied — V1∧V2 carry
    # the contact-happened load. Report n_admit_count to make this explicit.
    n_admit_count = sum(
        1 for s in c3_steps
        if (not _is_nan(parsed["steps"][s]["lam_n_ee_box"]))
        and parsed["steps"][s]["lam_n_ee_box"] > 0.5
    )

    # V3 from live [FORCE-ROUTE] u_z trace (NOT from a downstream metric).
    # Restrict to c3-mode ticks: a route line's tick matches a step number
    # where mode=c3.
    c3_set = set(c3_steps)
    uz_c3 = [r["u_z"] for r in parsed["routes"] if r["tick"] in c3_set]
    if uz_c3:
        var_uz = statistics.pvariance(uz_c3) if len(uz_c3) >= 2 else 0.0
        max_abs_uz = max(abs(v) for v in uz_c3)
    else:
        var_uz = 0.0
        max_abs_uz = 0.0
    pass_V3 = (var_uz > V3_VAR_UZ_BAR) and (max_abs_uz > V3_MAX_ABS_UZ_BAR)

    # D-1 (u_sol CONSUMED at every c3 tick) — direct from eq= field.
    eq_c3 = [r["eq"] for r in parsed["routes"] if r["tick"] in c3_set]
    n_eq_true_c3 = sum(1 for e in eq_c3 if e)
    n_eq_total_c3 = len(eq_c3)
    pass_D1 = (n_eq_total_c3 > 0) and (n_eq_true_c3 == n_eq_total_c3)

    # D-3 cond-4 supplementary diagnostic: max|qy|, max|qz| over the run.
    max_abs_qy = max((abs(r["qy"]) for r in parsed["steps"].values()), default=0.0)
    max_abs_qz = max((abs(r["qz"]) for r in parsed["steps"].values()), default=0.0)

    pass_V1 = sc_rate >= SUSTAINED_CONTACT_BAR
    pass_V2 = aa_rate >= ADMIT_ACTIVE_BAR
    pass_V4 = no_phantom
    gap_closed = pass_V1 and pass_V2 and pass_V3 and pass_V4

    motion_m = goal_aligned_motion_m(parsed)
    motion_refutes_brush_shove = motion_m >= BOX_MOTION_DIAGNOSTIC_THRESHOLD_M

    return dict(
        c3_ticks=n_c3,
        sustained_contact=sc_rate,
        admit_active=aa_rate,
        n_admit_count=n_admit_count,
        var_u_z=var_uz,
        max_abs_u_z=max_abs_uz,
        no_phantom=no_phantom,
        n_phantom=n_phantom,
        pass_V1=pass_V1,
        pass_V2=pass_V2,
        pass_V3=pass_V3,
        pass_V4=pass_V4,
        gap_closed=gap_closed,
        D1_eq_true_c3=n_eq_true_c3,
        D1_eq_total_c3=n_eq_total_c3,
        pass_D1=pass_D1,
        max_abs_qy=max_abs_qy,
        max_abs_qz=max_abs_qz,
        box_motion_m=motion_m,
        motion_refutes_brush_shove=motion_refutes_brush_shove,
    )


def _is_nan(x) -> bool:
    return x != x  # NaN-test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_name", help="Label (e.g. u_sol_100, g_hat_100)")
    ap.add_argument("logs", nargs="+", help="LOG paths: seed0=PATH seed4=PATH ...")
    args = ap.parse_args()
    seed_results = {}
    for spec in args.logs:
        seed_str, _, path = spec.partition("=")
        seed = int(seed_str.replace("seed", ""))
        seed_results[seed] = evaluate_seed(parse_log(Path(path)))
    n_seeds = len(seed_results)
    seed_pass = {s: r["gap_closed"] for s, r in seed_results.items()}
    n_pass = sum(1 for v in seed_pass.values() if v)
    cell_pass = (n_pass / max(n_seeds, 1)) >= SEED_PASS_FRACTION if n_seeds else False
    n_motion_refutes = sum(
        1 for r in seed_results.values() if r["motion_refutes_brush_shove"])
    motion_interpretation = (
        "brush-shove REFUTED as necessary (motion >= 5mm on both seeds)"
        if n_seeds > 0 and n_motion_refutes == n_seeds
        else "brush-shove CONFIRMED — motion < 5mm on at least one seed; "
             "EXPECTED reading at gap-closed (NOT a Stage C failure)"
    )
    out = dict(
        cell_name=args.cell_name,
        n_seeds=n_seeds,
        n_pass_gap_closed=n_pass,
        cell_pass_gap_closed=cell_pass,
        seed_pass_gap_closed=seed_pass,
        seed_results=seed_results,
        motion_diagnostic=dict(
            n_seeds_motion_refutes_brush_shove=n_motion_refutes,
            n_seeds=n_seeds,
            interpretation=motion_interpretation,
        ),
    )
    print(json.dumps(out, indent=2))
    sys.exit(0 if cell_pass else 1)


if __name__ == "__main__":
    main()
