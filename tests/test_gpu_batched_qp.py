"""BatchedBoxQP must reach the same optimum as OSQP, and amendment A1's
explicit inverse must not cost accuracy on REAL corpus instances.
"""
import glob

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="no CUDA device")

from control.gpu.legacy_torch.batched_qp import BatchedBoxQP  # noqa: E402


def make_qp(rng, n=40, m_eq=12, m_box=10):
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    A_eq = rng.standard_normal((m_eq, n))
    b_eq = rng.standard_normal(m_eq)
    idx = rng.choice(n, m_box, replace=False)
    A = np.vstack([A_eq, np.eye(n)[idx]])
    lo = np.concatenate([b_eq, -np.ones(m_box)])
    hi = np.concatenate([b_eq, np.ones(m_box)])
    return P, rng.standard_normal(n), A, lo, hi


def osqp_ref(P, q, A, lo, hi):
    import osqp
    from scipy import sparse
    s = osqp.OSQP()
    s.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=lo, u=hi,
            eps_abs=1e-10, eps_rel=1e-10, max_iter=200000, verbose=False)
    return s.solve().x


def build(probs, sel, **kw):
    sel = list(sel)
    return BatchedBoxQP(np.stack([probs[i][0] for i in sel]),
                        np.stack([probs[i][2] for i in sel]),
                        np.stack([probs[i][3] for i in sel]),
                        np.stack([probs[i][4] for i in sel]),
                        n_eq_rows=12, **kw)


def test_matches_osqp():
    """Same optimum as the solver Drake runs on CPU."""
    rng = np.random.default_rng(0)
    probs = [make_qp(rng) for _ in range(4)]
    x, _, _ = build(probs, range(4)).solve(
        np.stack([p[1] for p in probs]), max_iter=60000, eps=1e-9,
        check_every=200)
    for b, (P, q, A, lo, hi) in enumerate(probs):
        ref = osqp_ref(P, q, A, lo, hi)
        err = np.max(np.abs(x[b].cpu().numpy() - ref))
        assert err < 1e-4, f"batch {b}: max|dx| = {err:.3e}"


def test_batch_equals_singleton():
    """Batching must not COUPLE problems.

    Note this is a tolerance test, not bit-equality, and deliberately so:
    torch.linalg.cholesky is itself not bit-identical across batch sizes
    (cuSOLVER selects different kernels for batched vs single; measured
    3.5e-15 on K, 8.7e-18 on K^-1), which compounds to ~6e-14 over a few
    thousand iterations. That is a property of the factorization kernel,
    NOT of problem coupling -- a bare torch.bmm IS bit-identical across
    batch sizes. 1e-10 is still five orders inside OSQP's own eps=1e-5.
    """
    rng = np.random.default_rng(1)
    probs = [make_qp(rng) for _ in range(3)]
    q_all = np.stack([p[1] for p in probs])
    full = build(probs, range(3)).solve(q_all, max_iter=3000, eps=0.0,
                                        check_every=10**9)[0]
    for b in range(3):
        one = build(probs, [b]).solve(q_all[b:b + 1], max_iter=3000, eps=0.0,
                                      check_every=10**9)[0]
        rel = ((full[b] - one[0]).abs().max()
               / one[0].abs().max().clamp_min(1e-300)).item()
        assert rel < 1e-10, f"batch element {b} coupled: rel {rel:.3e}"


def test_neighbours_do_not_affect_result():
    """The sharper coupling test: element 0's answer must not depend on
    WHICH other problems share its batch."""
    rng = np.random.default_rng(4)
    probs = [make_qp(rng) for _ in range(5)]
    q = np.stack([p[1] for p in probs])
    kw = dict(max_iter=3000, eps=0.0, check_every=10**9)
    with_12 = build(probs, [0, 1, 2]).solve(q[[0, 1, 2]], **kw)[0][0]
    with_34 = build(probs, [0, 3, 4]).solve(q[[0, 3, 4]], **kw)[0][0]
    rel = ((with_12 - with_34).abs().max()
           / with_12.abs().max().clamp_min(1e-300)).item()
    assert rel < 1e-10, f"result depends on batch neighbours: rel {rel:.3e}"


def test_warm_start_reaches_same_point():
    """Warm starting (cuNRTO P6) must not change the optimum."""
    rng = np.random.default_rng(5)
    probs = [make_qp(rng) for _ in range(2)]
    q = np.stack([p[1] for p in probs])
    s = build(probs, range(2))
    cold = s.solve(q, max_iter=40000, eps=1e-9, check_every=200)[0]
    s2 = build(probs, range(2))
    a = s2.solve(q, max_iter=500, eps=0.0, check_every=10**9)
    warm = s2.solve(q, max_iter=40000, eps=1e-9, check_every=200,
                    x0=a[0], z0=a[1], y0=a[2])[0]
    assert torch.allclose(cold, warm, atol=1e-6)


def test_equality_rows_are_satisfied():
    rng = np.random.default_rng(2)
    probs = [make_qp(rng) for _ in range(2)]
    x, _, _ = build(probs, range(2)).solve(
        np.stack([p[1] for p in probs]), max_iter=60000, eps=1e-9,
        check_every=200)
    for b, (_P, _q, A, lo, _hi) in enumerate(probs):
        r = A[:12] @ x[b].cpu().numpy() - lo[:12]
        assert np.max(np.abs(r)) < 1e-5, f"batch {b} equality residual {r}"


# ---------------------------------------------------------------------
# A1 acceptance: the explicit inverse must be accurate on REAL instances.
# cond(K) ~ 5 was measured on RANDOM SPD matrices; real LCS instances are
# the thing that actually has to hold.
# ---------------------------------------------------------------------
CORPUS = sorted(glob.glob("audit_output/admm_corpus/inst_*[0-9].npz"))


QPS = sorted(glob.glob("audit_output/admm_corpus/inst_*_qp.npz"))


def _qp_from_corpus(qf, device="cuda"):
    """Build a BatchedBoxQP from the REAL matrices the CPU solved, so no
    assembly is reimplemented for this check."""
    d = np.load(qf, allow_pickle=True)
    P_sym, C_eq, b_eq = d["P_sym"], d["C_eq"], d["b_eq"]
    total_dim, TOT, SX, SU = (int(d["total_dim"]), int(d["TOT"]),
                              int(d["SX"]), int(d["SU"]))
    n_u = int(d["u_lo"].size)
    N = (total_dim - (TOT - 2 * int(d["n_lambda"]) - n_u)) // TOT
    idx, lo, hi = [], [], []
    for i in range(N):                                   # per-knot u bounds
        for j in range(n_u):
            idx.append(i * TOT + SU + j)
            lo.append(float(d["u_lo"][j])); hi.append(float(d["u_hi"][j]))
    for row in d["spb"]:                                 # workspace bounds
        s, l, h = int(row[0]), float(row[1]), float(row[2])
        for i in range(N):
            idx.append(i * TOT + SX + s); lo.append(l); hi.append(h)
        idx.append(N * TOT + s); lo.append(l); hi.append(h)
    if d["ee_vel_bounds"].size == 2:                     # EE velocity bounds
        l, h = float(d["ee_vel_bounds"][0]), float(d["ee_vel_bounds"][1])
        for s in d["ee_vel_idx"].tolist():
            for i in range(N):
                idx.append(i * TOT + SX + int(s)); lo.append(l); hi.append(h)
            idx.append(N * TOT + int(s)); lo.append(l); hi.append(h)
    sel = np.zeros((len(idx), total_dim))
    sel[np.arange(len(idx)), np.asarray(idx)] = 1.0
    A = np.vstack([C_eq, sel])[None]
    return BatchedBoxQP(P_sym[None], A,
                        np.concatenate([b_eq, np.asarray(lo)])[None],
                        np.concatenate([b_eq, np.asarray(hi)])[None],
                        n_eq_rows=C_eq.shape[0], device=device)


@pytest.mark.skipif(not QPS, reason="run scripts/gpu/dump_admm_corpus.sh")
def test_A1_explicit_inverse_accuracy_on_corpus():
    """Amendment A1 acceptance. cond(K) ~ 5 was measured on RANDOM SPD
    matrices; this is the check that actually matters -- the explicit
    inverse must match cholesky_solve on every REAL instance, or A1 must
    fall back to cholesky_solve."""
    worst, worst_cond = 0.0, 0.0
    for qf in QPS:
        qp = _qp_from_corpus(qf)
        K = (qp.P + qp.sigma * qp.eye
             + qp.At @ (qp.rho[:, None] * qp.A))
        L = torch.linalg.cholesky(K)
        rhs = torch.randn(K.shape[0], qp.n, dtype=torch.float64, device=qp.dev)
        a = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        b = torch.bmm(qp.Kinv, rhs.unsqueeze(-1)).squeeze(-1)
        worst = max(worst, ((a - b).norm() / a.norm()).item())
        worst_cond = max(worst_cond, torch.linalg.cond(K).max().item())
    print(f"\n  {len(QPS)} instances | worst rel err {worst:.3e} "
          f"| worst cond(K) {worst_cond:.3e}")
    assert worst < 1e-9, (
        f"explicit inverse lost accuracy on the corpus: {worst:.3e} "
        f"(worst cond(K) {worst_cond:.3e}) -- fall back to cholesky_solve")


@pytest.mark.skipif(not QPS, reason="run scripts/gpu/dump_admm_corpus.sh")
def test_gpu_qp_matches_osqp_on_real_instance():
    """End-to-end on a REAL assembled QP: the GPU solve must reach the same
    optimum OSQP does, at OSQP's own accuracy scale."""
    from scipy import sparse
    import osqp
    qf = QPS[0]
    d = np.load(qf, allow_pickle=True)
    qp = _qp_from_corpus(qf)
    q = d["q_ref"][None]
    x, _, _ = qp.solve(q, max_iter=40000, eps=1e-9, check_every=200)
    s = osqp.OSQP()
    s.setup(P=sparse.csc_matrix(qp.P[0].cpu().numpy()), q=d["q_ref"],
            A=sparse.csc_matrix(qp.A[0].cpu().numpy()),
            l=qp.lo[0].cpu().numpy(), u=qp.hi[0].cpu().numpy(),
            eps_abs=1e-10, eps_rel=1e-10, max_iter=200000, verbose=False)
    ref = s.solve().x
    err = np.max(np.abs(x[0].cpu().numpy() - ref))
    print(f"\n  real instance n={qp.n} m={qp.m}: max|dx| vs OSQP = {err:.3e}")
    assert err < 1e-4, f"GPU QP disagrees with OSQP on a real instance: {err:.3e}"
