#!/usr/bin/env python3
"""Is a locally linearized CRISP B-B model accurate enough at C3+ step sizes?

Tasks 1-4 of the linearization study. Writes tables to stdout, a JSON blob of
every measurement, and plots into docs/figs/.

    python3 tools/crisp/linearization_study.py [--log RUN.log] [--out DIR]

The model is used exactly as shipped; nothing here changes its dynamics.
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from control.crisp.push_box import PushBoxParams, PushBoxProblem  # noqa: E402
from control.crisp.scp import CrispParams, CrispSolver            # noqa: E402

# Reference constants, SolvePushbox.cpp:9-18. c and r stay separate.
REF = dict(a=0.5, b=0.25, mu=0.5, mass=1.0, g=9.8, c_int=0.4, N=100, dt=0.02)
# config/tasks.yaml `pushing` -- the geometry a C3+ adapter would actually see.
OURS = dict(a=0.05, b=0.05, mu=0.46, mass=1.0, c_int=0.6, N=40, dt=0.05)

C3PLUS_DT = 0.075      # main.py planning timestep
C3PLUS_N = 7           # main.py horizon

DELTAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
MODES = ("dc", "dlam", "dtheta", "dc+dlam", "dtheta+dlam", "all")


def make(cfg, goal=(1.0, 0.0, 0.0)):
    return PushBoxProblem(PushBoxParams(**cfg), np.zeros(3), np.array(goal))


# --------------------------------------------------------------- error metric
def lin_error(prob, q_bar, u_bar, dq, du):
    """f_true - f_lin for one perturbation, plus norms. z = [q, c, lambda]."""
    J = prob.dynamics_jacobian(q_bar, u_bar)
    f0 = prob.dynamics(q_bar, u_bar)
    f_true = prob.dynamics(q_bar + dq, u_bar + du)
    f_lin = f0 + J @ np.concatenate([dq, du])
    e = f_true - f_lin
    return dict(
        err=e,
        abs=float(np.linalg.norm(e)),
        rel=float(np.linalg.norm(e) / max(1e-12, np.linalg.norm(f_true))),
        f_true_norm=float(np.linalg.norm(f_true)),
    )


def jac_report(prob, q_bar, u_bar):
    J = prob.dynamics_jacobian(q_bar, u_bar)
    return dict(
        rank=int(np.linalg.matrix_rank(J, tol=1e-12)),
        yaw_row_norm=float(np.linalg.norm(J[2])),
        trans_row_norm=float(np.linalg.norm(J[:2])),
        singular_values=[float(s) for s in np.linalg.svd(J, compute_uv=False)],
    )


# ------------------------------------------------------- TASK 2: nominal points
def nominal_points(prob, refcfg_traj=None):
    a, b = prob.p.a, prob.p.b
    lam = 2.0 if a > 0.2 else 0.3          # force scaled to the box size
    pts = [
        ("A all-zero", np.zeros(3), np.zeros(6)),
        ("B face-centre -x", np.zeros(3), np.array([-a, 0.0, 0.0, lam, 0.0, 0.0])),
        ("C near-corner -x", np.zeros(3),
         np.array([-a, 0.95 * b, 0.0, lam, 0.0, 0.0])),
        ("D small force", np.zeros(3),
         np.array([-a, 0.0, 0.0, 0.01 * lam, 0.0, 0.0])),
        ("E moderate force", np.zeros(3), np.array([-a, 0.0, 0.0, lam, 0.0, 0.0])),
        ("F translation nominal", np.array([0.4 * a, 0.0, 0.0]),
         np.array([-a, 0.0, 0.0, lam, 0.0, 0.0])),
        ("G rotation nominal", np.array([0.0, 0.0, 0.35]),
         np.array([-a, 0.6 * b, 0.0, lam, 0.0, 0.0])),
    ]
    if refcfg_traj is not None:
        s, u = refcfg_traj
        k_mid = len(u) // 2
        pts.append(("H refcfg mid-trajectory", s[k_mid].copy(), u[k_mid].copy()))
        sw = _first_face_switch(u)
        if sw is not None:
            pts.append((f"I face switch (k={sw})", s[sw].copy(), u[sw].copy()))
    return pts


def _first_face_switch(controls):
    prev, prev_k = None, None
    for k, u in enumerate(controls):
        f = PushBoxProblem.active_face(u, 1e-4)
        if f is None:
            continue
        if prev is not None and f != prev:
            return prev_k
        prev, prev_k = f, k
    return None


# ------------------------------------------------- TASK 3: perturbation sweep
def perturb(mode, d, rng):
    dq, du = np.zeros(3), np.zeros(6)
    parts = mode.split("+") if mode != "all" else ["dc", "dlam", "dtheta"]
    if "dc" in parts:
        v = rng.normal(size=2)
        du[:2] = d * v / np.linalg.norm(v)
    if "dlam" in parts:
        v = rng.normal(size=4)
        du[2:] = d * v / np.linalg.norm(v)
    if "dtheta" in parts:
        dq[2] = d * rng.choice([-1.0, 1.0])
    return dq, du


def sweep(prob, pts, n_draw=24, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for name, q_bar, u_bar in pts:
        for mode in MODES:
            for d in DELTAS:
                errs = [lin_error(prob, q_bar, u_bar, *perturb(mode, d, rng))
                        for _ in range(n_draw)]
                out.append(dict(
                    point=name, mode=mode, delta=d,
                    abs_mean=float(np.mean([e["abs"] for e in errs])),
                    abs_max=float(np.max([e["abs"] for e in errs])),
                    rel_mean=float(np.mean([e["rel"] for e in errs])),
                    rel_max=float(np.max([e["rel"] for e in errs])),
                    px=float(np.max([abs(e["err"][0]) for e in errs])),
                    py=float(np.max([abs(e["err"][1]) for e in errs])),
                    th=float(np.max([abs(e["err"][2]) for e in errs])),
                ))
    return out


# ------------------------------------------- TASK 4: zero-init degeneracy
def zero_init_analysis(prob):
    J0 = prob.dynamics_jacobian(np.zeros(3), np.zeros(6))
    out = dict(J_zero=J0.tolist(), **jac_report(prob, np.zeros(3), np.zeros(6)))
    out["dtheta_dc"] = J0[2, 3:5].tolist()
    out["dtheta_dlam"] = J0[2, 5:].tolist()
    a, b = prob.p.a, prob.p.b
    seeds = {
        "A all-zero": np.zeros(6),
        "B small lambda seed": np.array([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0]),
        "C face-centre c + lambda": np.array([-a, 0.0, 0.0, 1e-2, 0.0, 0.0]),
        "D offset contact + lambda": np.array([-a, 0.5 * b, 0.0, 1e-2, 0.0, 0.0]),
    }
    out["seeds"] = {}
    for name, u in seeds.items():
        out["seeds"][name] = dict(u=u.tolist(),
                                  **jac_report(prob, np.zeros(3), u))
    return out


def init_strategy_solves(cfg, goal, strategies):
    """Does the seed change what the SOLVE finds? Diagnosis only."""
    res = {}
    for name, seed_fn in strategies.items():
        prob = make(cfg, goal)
        r = CrispSolver(CrispParams()).solve(prob, seed_fn(prob))
        s, u = prob.unpack(r.z)
        faces = sorted({PushBoxProblem.active_face(x, 1e-4) for x in u} - {None})
        res[name] = dict(status=r.status, iters=r.iterations,
                         viol=r.max_violation,
                         moved=float(np.linalg.norm(s[-1, :2])),
                         yaw=float(abs(s[-1, 2])),
                         yaw_goal=float(abs(goal[2])),
                         faces=faces)
    return res


# ------------------------------------------- empirical C3+ scales from a log
def c3plus_scales(log_path):
    rows = []
    for ln in open(log_path):
        if "[GATE-CONTACT]" not in ln:
            continue

        def g(k):
            return [float(x) for x in
                    re.search(rf"{k}=\(([^)]*)\)", ln).group(1).split(",")]
        rows.append((g("F_on_box"), g("box_p"), g("ee_p"), g("box_q")))
    if not rows:
        return None
    F = np.array([r[0] for r in rows])
    P = np.array([r[1] for r in rows])[:, :2]
    E = np.array([r[2] for r in rows])[:, :2]
    Q = np.array([r[3] for r in rows])
    yaw = 2.0 * np.arctan2(Q[:, 3], Q[:, 0])
    fn = np.linalg.norm(F, axis=1)
    contact = fn > 1e-6

    def stats(v, h):
        step = (np.abs(v[h:] - v[:-h]) if v.ndim == 1
                else np.linalg.norm(v[h:] - v[:-h], axis=1))
        return dict(median=float(np.median(step)),
                    p90=float(np.percentile(step, 90)), max=float(step.max()))
    out = dict(
        ticks=len(rows), contact_ticks=int(contact.sum()),
        max_force=float(fn.max()),
        ee_per_tick=stats(E, 1), ee_per_horizon=stats(E, C3PLUS_N),
        box_per_tick=stats(P, 1), box_per_horizon=stats(P, C3PLUS_N),
        yaw_per_tick=stats(yaw, 1), yaw_per_horizon=stats(yaw, C3PLUS_N),
    )
    if contact.sum() > C3PLUS_N + 1:
        out["force_per_tick_incontact"] = stats(fn[contact], 1)
        out["force_per_horizon_incontact"] = stats(fn[contact], C3PLUS_N)
        out["force_median_incontact"] = float(np.median(fn[contact]))
    return out


def crisp_plan_scales(cfg, goal):
    """Force / contact / pose step magnitudes inside a CONVERGED CRISP plan.

    The C3+ rollout hovers short of contact, so it cannot supply an empirical
    lambda scale. A converged plan at the same geometry can: it says what
    magnitudes the model itself uses, and how fast they vary knot to knot.
    """
    prob = make(cfg, goal)
    r = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))
    s, u = prob.unpack(r.z)
    lam = u[:, 2:]
    lam_mag = np.abs(lam).sum(axis=1)
    dlam = np.linalg.norm(np.diff(lam, axis=0), axis=1)
    dc = np.linalg.norm(np.diff(u[:, :2], axis=0), axis=1)
    dq = np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1)
    dth = np.abs(np.diff(s[:, 2]))

    def st(v):
        return dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                    max=float(v.max()))
    return dict(status=r.status, goal=list(goal), dt=cfg["dt"],
                lam_mag=st(lam_mag), dlam_per_knot=st(dlam),
                dc_per_knot=st(dc), dq_per_knot=st(dq), dtheta_per_knot=st(dth))


# ----------------------------------------------------------------------- main
def _seed(prob, c, lam):
    s, u = prob.unpack(np.zeros(prob.n))
    u[:, 0], u[:, 1] = c[0], c[1]
    u[:, 3] = lam
    return prob.pack(s, u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="C3+ run log for empirical scales")
    ap.add_argument("--out", default="docs/figs")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== solving refcfg for nominal points H and I ===", flush=True)
    theta = 12 * 2 * np.pi / 18
    goal = [3 * np.cos(theta), 3 * np.sin(theta), theta]
    prob_ref = PushBoxProblem(
        PushBoxParams(q_pos=200.0, q_yaw=200.0, r_lambda=0.002, **REF),
        np.zeros(3), np.array(goal))
    r = CrispSolver(CrispParams(mu_0=10.0, mu_max=1e8, k_max=5000)).solve(
        prob_ref, np.zeros(prob_ref.n))
    refcfg_traj = prob_ref.unpack(r.z)
    print(f"    refcfg: {r.status}, {r.iterations} iters", flush=True)

    results = {}
    for tag, cfg in (("ours", OURS), ("ref", REF)):
        prob = make(cfg)
        pts = nominal_points(prob, refcfg_traj if tag == "ref" else None)
        print(f"=== {tag}: {len(pts)} nominal points ===", flush=True)
        results[tag] = dict(
            nominals=[dict(name=n, q=q.tolist(), u=u.tolist(),
                           face=PushBoxProblem.active_face(u, 1e-6),
                           **jac_report(prob, q, u)) for n, q, u in pts],
            sweep=sweep(prob, pts),
            zero_init=zero_init_analysis(prob),
            k_trans=prob.k_trans, k_rot=prob.k_rot,
        )

    print("=== task 4: initialisation strategies ===", flush=True)
    a = OURS["a"]
    strategies = {
        "A all-zero": lambda p: np.zeros(p.n),
        "B small lambda seed": lambda p: _seed(p, (0.0, 0.0), 1e-3),
        "C face-centre c + lambda": lambda p: _seed(p, (-a, 0.0), 1e-2),
        "D offset contact + lambda": lambda p: _seed(p, (-a, 0.5 * a), 1e-2),
    }
    results["rotation_init"] = init_strategy_solves(
        OURS, [0.0, 0.0, np.pi / 2], strategies)
    results["translation_init"] = init_strategy_solves(
        OURS, [0.15, 0.0, 0.0], strategies)

    print("=== crisp plan scales ===", flush=True)
    results["plan_scales"] = {
        "ours translate": crisp_plan_scales(OURS, [0.15, 0.0, 0.0]),
        "ref benchmark": crisp_plan_scales(
            dict(REF, **{}), [3 * np.cos(theta), 3 * np.sin(theta), theta]),
    }

    if args.log:
        results["c3plus"] = c3plus_scales(args.log)

    (out_dir / "linearization_results.json").write_text(
        json.dumps(results, indent=1))
    _plots(results, out_dir)
    _print_tables(results)
    print(f"\nwrote {out_dir}/linearization_results.json and plots")


def _plots(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for tag in ("ours", "ref"):
        rows = results[tag]["sweep"]
        pts = sorted({r["point"] for r in rows})
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ax, mode in zip(axes, ("dc+dlam", "dtheta+dlam", "all")):
            for pt in pts:
                sel = [r for r in rows if r["point"] == pt and r["mode"] == mode]
                if not sel:
                    continue
                ax.loglog([r["delta"] for r in sel],
                          [max(r["abs_mean"], 1e-20) for r in sel],
                          marker="o", ms=3, lw=1.2, label=pt)
            d = np.array(DELTAS)
            ax.loglog(d, d ** 2, "k--", lw=.8, label=r"slope 2 ($\delta^2$)")
            ax.set_title(f"{tag}: {mode}")
            ax.set_xlabel(r"perturbation magnitude $\delta$")
            ax.grid(alpha=.3, which="both")
        axes[0].set_ylabel(r"$\|f_{true}-f_{lin}\|_2$")
        axes[2].legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"lin_error_{tag}.png", dpi=140)
        plt.close(fig)

    # relative error vs delta, both configs, mode=all
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tag, style in (("ours", "-o"), ("ref", "--s")):
        rows = [r for r in results[tag]["sweep"] if r["mode"] == "all"]
        ys = [max(np.mean([r["rel_mean"] for r in rows if r["delta"] == d]), 1e-20)
              for d in DELTAS]
        ax.loglog(DELTAS, ys, style, ms=4, lw=1.3, label=tag)
    ax.axhline(0.01, color="k", ls=":", lw=.9)
    ax.text(DELTAS[0], 0.012, "1% relative error", fontsize=7)
    ax.set_xlabel(r"perturbation magnitude $\delta$")
    ax.set_ylabel("mean relative error")
    ax.grid(alpha=.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lin_relerror.png", dpi=140)
    plt.close(fig)


def _print_tables(results):
    for tag in ("ours", "ref"):
        print(f"\n### {tag}: nominal points")
        print(f"{'point':28s} {'face':6s} {'rank':>4s} {'|J_yaw|':>10s} "
              f"{'|J_trans|':>10s}")
        for n in results[tag]["nominals"]:
            flag = "  <-- YAW ROW DEAD" if n["yaw_row_norm"] < 1e-12 else ""
            print(f"{n['name']:28s} {str(n['face']):6s} {n['rank']:4d} "
                  f"{n['yaw_row_norm']:10.4f} {n['trans_row_norm']:10.4f}{flag}")
        print(f"\n### {tag}: error vs delta (mode=all, over all nominal points)")
        rows = [r for r in results[tag]["sweep"] if r["mode"] == "all"]
        print(f"{'delta':>8s} {'rel_mean':>11s} {'rel_max':>11s} {'abs_max':>11s}")
        for d in DELTAS:
            sel = [r for r in rows if r["delta"] == d]
            print(f"{d:8.1e} {np.mean([r['rel_mean'] for r in sel]):11.3e} "
                  f"{max(r['rel_max'] for r in sel):11.3e} "
                  f"{max(r['abs_max'] for r in sel):11.3e}")
        print(f"\n### {tag}: error by mode at delta=1e-2 (max over points)")
        for mode in MODES:
            sel = [r for r in results[tag]["sweep"]
                   if r["mode"] == mode and r["delta"] == 1e-2]
            print(f"  {mode:14s} abs_max={max(r['abs_max'] for r in sel):.3e} "
                  f"px={max(r['px'] for r in sel):.3e} "
                  f"py={max(r['py'] for r in sel):.3e} "
                  f"th={max(r['th'] for r in sel):.3e}")
    if results.get("c3plus"):
        c = results["c3plus"]
        print(f"\n### empirical C3+ scales ({c['ticks']} ticks, "
              f"{c['contact_ticks']} with contact, max|F|={c['max_force']:.3f} N)")
        for k, v in c.items():
            if isinstance(v, dict):
                print(f"  {k:30s} median={v['median']:.5f} p90={v['p90']:.5f} "
                      f"max={v['max']:.5f}")
    if results.get("plan_scales"):
        print("\n### magnitudes inside a converged CRISP plan")
        for name, v in results["plan_scales"].items():
            print(f"  {name} ({v['status']}, dt={v['dt']}):")
            for k in ("lam_mag", "dlam_per_knot", "dc_per_knot",
                      "dq_per_knot", "dtheta_per_knot"):
                d = v[k]
                print(f"     {k:16s} median={d['median']:.5f} "
                      f"p90={d['p90']:.5f} max={d['max']:.5f}")
    for key in ("rotation_init", "translation_init"):
        print(f"\n### {key}")
        for name, v in results[key].items():
            print(f"  {name:26s} moved={v['moved']:.4f} "
                  f"yaw={v['yaw']:.4f}/{v['yaw_goal']:.4f} "
                  f"faces={v['faces'] or ['-']} viol={v['viol']:.1e} {v['status']}")


if __name__ == "__main__":
    main()
