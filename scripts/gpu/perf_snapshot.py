"""Task 6: emit a machine-comparable performance snapshot.

The CRISP / Path-B agent may change candidate count, per-candidate QP
dimension, iteration counts, solves per tick, or contact variables. Any of
those invalidates the current GPU crossover estimate, so the snapshot records
BOTH the problem shape and the latency distributions in one JSON document
that a later run can be diffed against.

    # before Path B
    python3 scripts/gpu/perf_snapshot.py --label before_pathb --sim-seconds 25
    # after Path B
    python3 scripts/gpu/perf_snapshot.py --label after_pathb  --sim-seconds 25
    # compare
    python3 scripts/gpu/perf_snapshot.py --compare before_pathb after_pathb

Deliberately records shape and latency together: a latency change is
uninterpretable without knowing whether the problem changed underneath it.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

OUT_DIR = "audit_output/perf_snapshots"
SECTIONS = ("admm.osqp_solve", "admm.final_qp", "admm.qp_build",
            "lcs.extract_dynamics", "admm.z_update")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    if not xs:
        return None
    return {"calls": len(xs), "mean": float(np.mean(xs)),
            "p50": q(xs, .50), "p90": q(xs, .90), "p95": q(xs, .95),
            "p99": q(xs, .99), "max": float(np.max(xs))}


def parse_log(path):
    """Pull shape + latency out of a completed run log."""
    t = open(path, errors="replace").read()
    out = {}

    m = re.findall(r"avg_per_step_ms=([\d.]+)\s+full_solves=(\d+)"
                   r"\s+cheap_solves=(\d+)\s+switches=(\d+)", t)
    if m:
        a = m[-1]
        out["avg_per_step_ms"] = float(a[0])
        out["full_solves"] = int(a[1])
        out["switches"] = int(a[3])

    r = re.findall(r"translational_error=([\d.]+)m\s+"
                   r"rotational_error=([\d.]+)rad\s+success=(\w+)\s+"
                   r"tight_goal=(\S+)", t)
    if r:
        out["trans_err"] = float(r[-1][0])
        out["rot_err"] = float(r[-1][1])
        out["success"] = r[-1][2]
        out["tight_goal"] = r[-1][3]

    out["gs_ticks"] = len(re.findall(r"\[GS\] step=\d+", t))
    it = re.findall(r"iters=(\d+)/(\d+)", t)
    if it:
        out["c3plus_iters"] = dist([int(a) for a, _ in it])

    # per-section latency, from the exit distribution report
    sec = {}
    for line in t.splitlines():
        for name in SECTIONS:
            if line.strip().startswith(name):
                f = line.split()
                try:
                    sec[name] = {"calls": int(f[1].replace(",", "")),
                                 "mean": float(f[2]), "p50": float(f[3]),
                                 "p90": float(f[4]), "p95": float(f[5]),
                                 "p99": float(f[6]), "max": float(f[7])}
                except (IndexError, ValueError):
                    pass
    out["sections_ms"] = sec
    return out


def problem_shape():
    """Dimensions, read from the corpus rather than assumed."""
    qps = sorted(glob.glob("audit_output/admm_corpus/inst_*_qp.npz"))
    ins = sorted(glob.glob("audit_output/admm_corpus/inst_*[0-9].npz"))
    if not qps:
        return {"note": "no corpus; run scripts/gpu/dump_admm_corpus.sh"}
    d, i0 = np.load(qps[0], allow_pickle=True), np.load(ins[0],
                                                       allow_pickle=True)
    P, C = d["P_sym"], d["C_eq"]
    nnz = lambda M: int(np.count_nonzero(np.abs(M) > 1e-14))   # noqa: E731
    return {
        "n_x": int(i0["n_x"]), "n_u": int(i0["n_u"]),
        "n_lambda": int(d["n_lambda"]), "N": int(i0["N"]),
        "contacts": int(d["num_normals"]),
        "qp_dim_n": int(d["total_dim"]),
        "qp_rows_m": int(C.shape[0]) + int(d["u_lo"].size) * int(i0["N"]),
        "P_nnz": nnz(P), "P_density_pct": 100.0 * nnz(P) / P.size,
        "C_eq_nnz": nnz(C), "C_eq_density_pct": 100.0 * nnz(C) / C.size,
        "admm_iter": int(i0["admm_iter"]),
        "rho_scale": float(i0["rho_scale"]),
        "corpus_instances": len(qps),
    }


def env():
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    e = {"python": sys.version.split()[0], "numpy": _v("numpy"),
         "scipy": _v("scipy"), "cupy": _v("cupy"), "torch": _v("torch")}
    try:
        e["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
             "--format=csv,noheader"], capture_output=True,
            text=True).stdout.strip()
    except Exception:
        e["gpu"] = None
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--sim-seconds", type=int, default=25)
    ap.add_argument("--task", default="box", choices=["box", "t"])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.compare:
        a, b = (json.load(open(f"{OUT_DIR}/{x}.json")) for x in args.compare)
        print(f"{'metric':<28}{args.compare[0]:>18}{args.compare[1]:>18}"
              f"{'change':>12}")
        print("-" * 76)
        for k in ("n_x", "n_u", "n_lambda", "N", "contacts", "qp_dim_n",
                  "qp_rows_m", "P_density_pct", "admm_iter"):
            va, vb = a["shape"].get(k), b["shape"].get(k)
            ch = "" if va in (None, 0) or vb is None else f"{vb/va:.2f}x"
            print(f"{k:<28}{str(va):>18}{str(vb):>18}{ch:>12}")
        for k in ("avg_per_step_ms", "full_solves", "gs_ticks"):
            va, vb = a["run"].get(k), b["run"].get(k)
            ch = "" if not va or vb is None else f"{vb/va:.2f}x"
            print(f"{k:<28}{str(va):>18}{str(vb):>18}{ch:>12}")
        print()
        for s in SECTIONS:
            sa = a["run"].get("sections_ms", {}).get(s)
            sb = b["run"].get("sections_ms", {}).get(s)
            if not sa or not sb:
                continue
            print(f"  {s}")
            for stat in ("mean", "p50", "p95", "p99", "max"):
                print(f"    {stat:<8}{sa[stat]:>14.2f}{sb[stat]:>18.2f}"
                      f"{sb[stat]/max(sa[stat],1e-9):>12.2f}x")
        print("\n  NOTE: the GPU crossover estimate is only valid for the "
              "shape it was measured at.\n  If any shape row changed, "
              "re-run scripts/gpu/primitive_sweep.py before quoting it.")
        return

    if not args.label:
        ap.error("--label is required unless --compare is used")

    log = f"/tmp/perf_snapshot_{args.label}.log"
    cmd = (["python3", "main.py", "pushing", "--task-id", "4",
            "--sampling-c3", "config/sampling_c3_kik.yaml"]
           if args.task == "box" else
           ["python3", "main.py", "push_t_mesh",
            "--sampling-c3", "config/sampling_c3_kik_t.yaml"])
    cmd += ["--max-time", str(args.sim_seconds), "--seed", "0"]

    e = dict(os.environ, PORT_SECTION_TIMING="1",
             PORT_SECTION_DISTRIBUTIONS="1")
    t0 = time.perf_counter()
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=e,
                       timeout=3600)
    wall = time.perf_counter() - t0

    snap = {
        "label": args.label,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True,
                                     text=True).stdout.strip(),
        "task": args.task, "sim_seconds": args.sim_seconds,
        "wall_seconds": wall, "seed": 0,
        "candidate_semantics": os.environ.get("PORT_CANDIDATE_WARMSTART",
                                              "legacy_ordered"),
        "check_termination": int(os.environ.get(
            "PORT_OSQP_CHECK_TERMINATION", "100")),
        "env": env(), "shape": problem_shape(), "run": parse_log(log),
    }
    path = f"{OUT_DIR}/{args.label}.json"
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    print(f"wrote {path}   wall={wall:.1f}s  "
          f"avg_per_step_ms={snap['run'].get('avg_per_step_ms')}  "
          f"tight={snap['run'].get('tight_goal')}")


if __name__ == "__main__":
    main()
