#!/usr/bin/env python3
"""Extract the R^7-formulation baseline signals from seed-4 off.

Pre-registered baseline for the EE-space rewrite comparison.

Captures three signals from seed4_off_source.log (HEAD=37418f67):
  1. F_W tick series — Drake-realized contact force on the box at every
     [GATE-CONTACT] step. Source of the 0-127N oscillation.
  2. ADMM convergence per solve — primal & dual residuals at termination
     (always 25/25 iters at HEAD=37418f67), n_iters always 25.
  3. n_cont — count of Drake-realized contact ticks (A_is_ee=1).

Outputs three CSVs in this directory; the rewrite's post-fix run
re-extracts the same three on the same seed-4 for direct comparison.
"""
from __future__ import annotations
import argparse, csv, re, pathlib, sys

GC_RE = re.compile(
    r"\[GATE-CONTACT\] step=(?P<step>\d+) "
    r"F_W=\((?P<fx>[+\-\d.eE]+),(?P<fy>[+\-\d.eE]+),(?P<fz>[+\-\d.eE]+)\) "
    r"F_on_box=\([+\-\d.eE,]+\) "
    r"n_face_out=\([+\-\d.eE,]+\) "
    r"A_is_ee=(?P<aee>\d+) "
    r"box_q=\([+\-\d.eE,]+\) "
    r"box_p=\((?P<bxp>[+\-\d.eE]+),(?P<byp>[+\-\d.eE]+),(?P<bzp>[+\-\d.eE]+)\) "
    r"ee_p=\((?P<exp>[+\-\d.eE]+),(?P<eyp>[+\-\d.eE]+),(?P<ezp>[+\-\d.eE]+)\)"
)
CPLUS_RE = re.compile(
    r"\[C3\+\] step=(?P<step>\d+) "
    r"\|u\[0\]\|=(?P<u0>[\d.eE+\-]+)Nm "
    r"λ_n_max=(?P<ln>[\d.eE+\-]+) "
    r"η_n_max=(?P<en>[\d.eE+\-]+) "
    r"primal=(?P<pr>[\d.eE+\-]+) "
    r"iters=(?P<it>\d+)/(?P<itmax>\d+)"
)
ADMM_WARN_RE = re.compile(
    r"\[ADMM-C3\+\] WARNING non-converged: "
    r"pr=(?P<pr>[\d.eE+\-]+) dr=(?P<dr>[\d.eE+\-]+) tol=(?P<tol>\S+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_log",
                    nargs="?",
                    default=str(pathlib.Path(__file__).parent / "seed4_off_source.log"),
                    help="path to source log (default: seed4_off_source.log alongside)")
    ap.add_argument("--outdir",
                    default=str(pathlib.Path(__file__).parent),
                    help="output dir for CSVs (default: this script's dir)")
    args = ap.parse_args()
    src = pathlib.Path(args.source_log)
    out = pathlib.Path(args.outdir)
    if not src.exists():
        print(f"missing source: {src}", file=sys.stderr); sys.exit(2)
    out.mkdir(parents=True, exist_ok=True)

    fw_rows = []
    cplus_rows = []
    admm_warn_rows = []
    n_aee1 = 0
    _i_warn = 0
    # Header values derived FROM the source log itself so the SUMMARY
    # header cannot lie about which run produced these numbers. (Earlier
    # versions hardcoded "BASELINE R^7" + a specific HEAD and were misread
    # as describing the body data when run on a different log.)
    derived_head: str | None = None       # last seen `HEAD=...` line
    derived_n_x_planner: int | None = None  # from [DEBUG-C3+] slot dump
    derived_n_u_planner: int | None = None
    derived_plant_dofs: str | None = None   # from `[C3] DOFs:` line
    derived_use_ee_space: bool | None = None
    # Match the C3PlusMPC banner that fix-1 emits:
    #   [C3+] planner construction verified: use_ee_space=True solver.n_x=19 ...
    _re_banner_c3p = re.compile(
        r"\[C3\+\] planner construction verified: "
        r"use_ee_space=(?P<ee>True|False) "
        r"solver\.n_x=(?P<nx>\d+) solver\.n_u=(?P<nu>\d+)"
    )
    # Match the [DEBUG-C3+] slot dim (the source-of-truth for the planner):
    #   per-step slots:  x=[0:19)  λ=[19:25)  u=[25:28)  η=[28:34)
    _re_slots = re.compile(
        r"\[DEBUG-C3\+\] per-step slots:\s+x=\[0:(?P<nx>\d+)\)\s+"
        r"λ=\[\d+:\d+\)\s+u=\[\d+:(?P<u_end>\d+)\)"
    )
    _re_plant_dofs = re.compile(
        r"\[C3\] DOFs:\s+(?P<txt>n_q=\d+, n_v=\d+, n_u=\d+, n_x=\d+)"
    )
    _re_head = re.compile(r"\bHEAD=(?P<sha>[0-9a-f]{7,40})")
    with src.open(errors="replace") as fh:
        # admm warnings are emitted just after the corresponding [C3+] line —
        # join by ordinal index since neither carries a step on its own here.
        last_cplus_step = None
        for line in fh:
            m = GC_RE.search(line)
            if m:
                step = int(m["step"])
                fx, fy, fz = float(m["fx"]), float(m["fy"]), float(m["fz"])
                fmag = (fx*fx + fy*fy + fz*fz) ** 0.5
                aee = int(m["aee"])
                if aee == 1: n_aee1 += 1
                fw_rows.append({
                    "step": step, "A_is_ee": aee,
                    "F_W_x": fx, "F_W_y": fy, "F_W_z": fz, "F_W_mag": fmag,
                    "box_x": float(m["bxp"]), "box_y": float(m["byp"]), "box_z": float(m["bzp"]),
                    "ee_x":  float(m["exp"]), "ee_y":  float(m["eyp"]), "ee_z":  float(m["ezp"]),
                })
                continue
            m = CPLUS_RE.search(line)
            if m:
                last_cplus_step = int(m["step"])
                cplus_rows.append({
                    "step": last_cplus_step,
                    "u0_Nm": float(m["u0"]),
                    "lambda_n_max": float(m["ln"]),
                    "eta_n_max": float(m["en"]),
                    "primal": float(m["pr"]),
                    "iters": int(m["it"]),
                    "iters_max": int(m["itmax"]),
                })
                continue
            m = ADMM_WARN_RE.search(line)
            if m:
                admm_warn_rows.append({
                    "step": last_cplus_step if last_cplus_step is not None else -1,
                    "pr": float(m["pr"]),
                    "dr": float(m["dr"]),
                    "tol": m["tol"],
                })
                continue
            # Derive header values from log content. These are best-effort;
            # if a given line type isn't present, the SUMMARY says "unknown"
            # for that field rather than reusing a stale constant.
            m = _re_banner_c3p.search(line)
            if m:
                derived_use_ee_space = (m["ee"] == "True")
                derived_n_x_planner  = int(m["nx"])
                derived_n_u_planner  = int(m["nu"])
                continue
            m = _re_slots.search(line)
            if m and derived_n_x_planner is None:
                # Fallback: derive planner n_x from the slot dump (oldest
                # logs without the banner still expose this).
                derived_n_x_planner = int(m["nx"])
                derived_n_u_planner = int(m["u_end"]) - derived_n_x_planner - (
                    int(m["u_end"]) - derived_n_x_planner - 0)  # 0; placeholder
                # Actually compute n_u directly from the slot bounds.
                m2 = re.search(
                    r"u=\[(\d+):(\d+)\)", line)
                if m2:
                    derived_n_u_planner = int(m2.group(2)) - int(m2.group(1))
                continue
            m = _re_plant_dofs.search(line)
            if m and derived_plant_dofs is None:
                derived_plant_dofs = m["txt"]
                continue
            m = _re_head.search(line)
            if m:
                derived_head = m["sha"]

    # Emit CSVs
    def write(name, rows, fields):
        p = out / name
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p} ({len(rows)} rows)")

    write("fw_tick_series.csv", fw_rows,
          ["step", "A_is_ee", "F_W_x", "F_W_y", "F_W_z", "F_W_mag",
           "box_x", "box_y", "box_z", "ee_x", "ee_y", "ee_z"])
    write("c3plus_per_solve.csv", cplus_rows,
          ["step", "u0_Nm", "lambda_n_max", "eta_n_max", "primal",
           "iters", "iters_max"])
    write("admm_nonconverged_warnings.csv", admm_warn_rows,
          ["step", "pr", "dr", "tol"])

    # Summary
    sum_path = out / "SUMMARY.txt"
    n_fw = len(fw_rows)
    fw_aee1 = [r for r in fw_rows if r["A_is_ee"] == 1]
    fmag_aee1 = [r["F_W_mag"] for r in fw_aee1]
    fmag_aee1.sort()
    iters_at_max = sum(1 for r in cplus_rows if r["iters"] == r["iters_max"])
    prs = [r["pr"] for r in admm_warn_rows]
    drs = [r["dr"] for r in admm_warn_rows]
    def stats(xs):
        if not xs: return ("n=0",)
        xs = sorted(xs)
        return (
            f"n={len(xs)}", f"min={xs[0]:.4f}", f"med={xs[len(xs)//2]:.4f}",
            f"max={xs[-1]:.4f}", f"mean={sum(xs)/len(xs):.4f}",
        )
    # Derive the formulation label from the planner dims so the header
    # always describes the body data.
    if derived_use_ee_space is True or (
            derived_n_x_planner == 19 and derived_n_u_planner == 3):
        formulation_label = "EE-SPACE (planner n_x=19, u in Newtons)"
    elif derived_use_ee_space is False or (
            derived_n_x_planner == 27 and derived_n_u_planner == 7):
        formulation_label = "R^7 (planner n_x=27, u in Nm)"
    elif derived_n_x_planner is not None:
        formulation_label = (
            f"UNKNOWN formulation "
            f"(planner n_x={derived_n_x_planner}, n_u={derived_n_u_planner})"
        )
    else:
        formulation_label = "UNKNOWN formulation (no [DEBUG-C3+] / banner found)"
    head_str = derived_head if derived_head else "unknown (no HEAD= line in log)"
    plant_str = derived_plant_dofs if derived_plant_dofs else "unknown"
    with sum_path.open("w") as fh:
        fh.write(f"{formulation_label} — seed-4 off, HEAD={head_str}\n")
        fh.write(f"Source: {src.name}  (full path: {src})\n")
        fh.write(f"Plant DOFs (Drake sim): {plant_str}\n")
        fh.write(f"Planner solver: n_x={derived_n_x_planner}  n_u={derived_n_u_planner}\n\n")
        fh.write(f"--- GATE-CONTACT step coverage ---\n")
        fh.write(f"  total ticks logged:        {n_fw}\n")
        fh.write(f"  Drake-realized contact (A_is_ee=1) n_cont: {n_aee1}\n")
        if fmag_aee1:
            fh.write(f"--- F_W magnitude on A_is_ee=1 ticks ---\n")
            fh.write(f"  {' '.join(stats(fmag_aee1))}\n")
            fh.write(f"  p90: {fmag_aee1[int(0.9*len(fmag_aee1))]:.4f}\n")
            fh.write(f"  range: [{fmag_aee1[0]:.4f}, {fmag_aee1[-1]:.4f}]\n")
        fh.write(f"\n--- ADMM (C3+) convergence ---\n")
        fh.write(f"  C3+ solves logged: {len(cplus_rows)}\n")
        fh.write(f"  solves hitting iters_max (25/25): {iters_at_max} / {len(cplus_rows)}  "
                 f"({100.0*iters_at_max/max(1,len(cplus_rows)):.1f}%)\n")
        fh.write(f"  non-converged warnings emitted: {len(admm_warn_rows)}\n")
        if prs:
            fh.write(f"  primal residual at termination: {' '.join(stats(prs))}\n")
            fh.write(f"  dual residual at termination:   {' '.join(stats(drs))}\n")
        fh.write(f"\n--- Pre-registered comparison targets (post-EE-space-rewrite) ---\n")
        fh.write(f"  SUCCESS = F_W magnitude on A_is_ee=1 collapses to a stable value\n")
        fh.write(f"           AND ADMM convergence improves (fewer iters_max hits, smaller pr/dr).\n")
        fh.write(f"  NOT-A-FAILURE = box still falls short of goal (over-admission is a separate wall).\n")
    print(f"wrote {sum_path}")


if __name__ == "__main__":
    main()
