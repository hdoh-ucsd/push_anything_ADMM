"""Phase 2C: compact CPU golden fixtures for INDEPENDENT_BATCH semantics.

Captures the full chain a GPU backend must reproduce, at one deterministic
tick built from real corpus instances:

    1  tick-entry u_prev                     (the ONE warm start, broadcast)
    2  assembled candidate matrices          (hashed -- P_sym is 3.3 MB each)
    3  candidate initial states
    4  one candidate projection input/output (Bui eq.12)
    5  one ADMM iteration  (z, delta, omega after iteration 0)
    6  fixed-iteration C3+ solve             (u_seq, x_seq per candidate)
    7  candidate objective vector
    8  selected candidate ID
    9  first MPC action

Large matrices are stored as SHA-256 rather than verbatim: the fixture stays
compact and a GPU backend can still prove it assembled the identical problem.
Small vectors -- the ones a parity test actually compares elementwise -- are
stored in full.

Writes audit_output/gpu_golden/independent_batch/.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")

from control.admm_solver import C3Solver, project_C3Plus_eq12   # noqa: E402
from control.solver_api import CandidateSemantics               # noqa: E402

OUT = "audit_output/gpu_golden/independent_batch"
N_CAND = 5
ADMM_ITER = 3


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64)
                          .tobytes()).hexdigest()[:32]


def args_of(d):
    def _opt(k):
        v = d[k]
        return v if np.size(v) > 0 else None
    return dict(x0=d["x0"], A=d["A"], B_ctrl=d["B_ctrl"], D=d["D"], d=d["d"],
                E=d["E"], F=d["F"], H=d["H"], c_lcs=d["c_lcs"],
                J_n=d["J_n"], J_t=d["J_t"], mu=d["mu"],
                Q=d["Q"], R=d["R"], QN=d["QN"], x_ref=d["x_ref"],
                N=int(d["N"]), admm_iter=int(d["admm_iter"]),
                torque_limit=float(d["torque_limit"]),
                phi=_opt("phi"), u_lower=_opt("u_lower"),
                u_upper=_opt("u_upper"))


def main():
    files = sorted(glob.glob("audit_output/admm_corpus/inst_*[0-9].npz"))
    if len(files) < N_CAND:
        sys.exit("run scripts/gpu/dump_admm_corpus.sh first")
    os.makedirs(OUT, exist_ok=True)
    ds = [np.load(f, allow_pickle=True) for f in files[:N_CAND]]
    ref = ds[0]
    n_x, n_u = int(ref["n_x"]), int(ref["n_u"])
    N = int(ref["N"])

    # (1) the single tick-entry warm start, broadcast to every candidate.
    rng = np.random.default_rng(0)
    u_prev_entry = np.round(rng.uniform(-0.4, 0.4, (N, n_u)), 6)

    payload = {"u_prev_entry": u_prev_entry}
    per_cand = {}

    for k, d in enumerate(ds):
        solver = C3Solver(n_x=n_x, n_u=n_u, rho=float(d["rho_initial"]),
                          mode="c3plus")
        solver._u_prev_solve = u_prev_entry          # INDEPENDENT_BATCH
        u_seq, x_seq = solver._solve_c3plus(**args_of(d))
        payload[f"u_seq_{k}"] = u_seq                # (6) fixed-iter solve
        payload[f"x_seq_{k}"] = x_seq
        payload[f"x0_{k}"] = d["x0"]                 # (3) initial states

        qp = np.load(files[k].replace(".npz", "_qp.npz"), allow_pickle=True)
        per_cand[str(k)] = {                          # (2) assembled matrices
            "P_sym_sha": sha(qp["P_sym"]), "q_ref_sha": sha(qp["q_ref"]),
            "C_eq_sha": sha(qp["C_eq"]), "b_eq_sha": sha(qp["b_eq"]),
            "total_dim": int(qp["total_dim"]), "TOT": int(qp["TOT"]),
            "n_lambda": int(qp["n_lambda"]),
            "num_normals": int(qp["num_normals"]),
        }

    # (7) candidate objectives + (8) selection + (9) first action.
    # Deterministic surrogate objective: horizon control effort. The real
    # controller cost mixes in alignment/travel terms that live outside the
    # solver, so this fixture pins the SOLVER-side chain only.
    costs = np.array([float(np.sum(payload[f"u_seq_{k}"] ** 2))
                      for k in range(N_CAND)])
    best = int(np.argmin(costs))
    payload["candidate_costs"] = costs
    payload["best_candidate"] = np.int64(best)
    payload["first_action"] = payload[f"u_seq_{best}"][0]

    # (4) one projection input/output, real magnitudes.
    nl = int(ref["E"].shape[0])
    lam_in = np.round(rng.uniform(-5, 5, nl), 6)
    eta_in = np.round(rng.uniform(-5, 5, nl), 6)
    d_lam, d_eta = project_C3Plus_eq12(lam_in, eta_in, 1.0, 1.0)
    payload.update(proj_lam_in=lam_in, proj_eta_in=eta_in,
                   proj_lam_out=d_lam, proj_eta_out=d_eta)

    # (5) one ADMM iteration: state after iteration 0 for candidate 0.
    s0 = C3Solver(n_x=n_x, n_u=n_u, rho=float(ds[0]["rho_initial"]),
                  mode="c3plus")
    s0._u_prev_solve = u_prev_entry
    a1 = dict(args_of(ds[0]))
    a1["admm_iter"] = 1
    u1, x1 = s0._solve_c3plus(**a1)
    payload["iter1_u_seq"] = u1
    payload["iter1_x_seq"] = x1

    np.savez_compressed(os.path.join(OUT, "fixtures.npz"), **payload)

    meta = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True).stdout.strip(),
        "backend": "cpu",
        "candidate_semantics": CandidateSemantics.INDEPENDENT_BATCH.value,
        "candidate_count": N_CAND,
        "horizon": N,
        "n_x": n_x, "n_u": n_u,
        "n_lambda": int(ref["E"].shape[0]),
        "contacts": int(np.load(
            files[0].replace(".npz", "_qp.npz"))["num_normals"]),
        "admm_iter": ADMM_ITER,
        "rho_initial": float(ref["rho_initial"]),
        "rho_scale": float(ref["rho_scale"]),
        "u_lambda": float(ref["u_lambda"]), "u_eta": float(ref["u_eta"]),
        "osqp": {"eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 2000,
                 "check_termination": int(os.environ.get(
                     "PORT_OSQP_CHECK_TERMINATION", "100")),
                 "polishing": 0, "scaling": 1, "adaptive_rho": 1},
        "seed": 0,
        "source_corpus": [os.path.basename(f) for f in files[:N_CAND]],
        "note": ("Large matrices are stored as SHA-256; small vectors "
                 "verbatim. Objective is a solver-side surrogate (horizon "
                 "control effort), NOT the controller's ranking cost."),
        "per_candidate": per_cand,
    }
    with open(os.path.join(OUT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    sz = os.path.getsize(os.path.join(OUT, "fixtures.npz"))
    print(f"wrote {OUT}/  ({sz/1024:.1f} KB + metadata.json)")
    print(f"  candidates={N_CAND} N={N} n_x={n_x} n_u={n_u} n_lambda={nl}")
    print(f"  best_candidate={best}  costs={np.round(costs, 4)}")
    print(f"  first_action={np.round(payload['first_action'], 6)}")


if __name__ == "__main__":
    main()
