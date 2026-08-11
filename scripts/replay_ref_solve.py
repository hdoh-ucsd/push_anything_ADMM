#!/usr/bin/env python3
"""Replay a dumped reference C3+ solve instance through the port's ADMM.

Usage:
  DIAG_ZVEE=1 DIAG_ZVEE_PEE0=0 DIAG_ZVEE_VEE0=10 \
      python3 scripts/replay_ref_solve.py results/refdump_solve500.txt

The dump comes from the [EXPERIMENT-DUMP] patch in the reference c3.cc
(one solve, POST-AnDn-scaling, reference state layout: EE pos 0:3,
obj quat 3:7, obj pos 7:10, EE vel 10:13, obj vel 13:19).

Same instance, two implementations: if the port's loop grows the
approach here like the reference does ([ZVEE-REF] ~31->42->66 mm), the
port ADMM is exonerated and the divergence is instance CONTENT; if it
collapses (like its native 45->34->14), the implementation divergence
is trapped in this fully-inspectable offline case.

Caveats: Q is rebuilt from the dumped DIAGONAL (quaternion-Hessian
off-diagonals lost — steers rotation, secondary for approach); the
port's projection uses the scalar 1000:1 ratio (reference U is
per-slot; wall rows 1:1 differ — far/inactive).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from control.admm_solver import C3Solver  # noqa: E402


def parse_dump(path):
    blocks, cur, name = {}, [], None
    for line in open(path):
        if line.startswith("#"):
            if name is not None:
                blocks[name] = np.array(cur)
            name = line.split()[1] if len(line.split()) > 1 else line.strip("# \n")
            cur = []
        else:
            vals = [float(v) for v in line.split()]
            if vals:
                cur.append(vals)
    if name is not None:
        blocks[name] = np.array(cur)
    return blocks


def main():
    b = parse_dump(sys.argv[1] if len(sys.argv) > 1
                   else "results/refdump_solve500.txt")
    N, n_x, n_lam, n_u, AnDn = b["dims"].ravel()
    N, n_x, n_lam, n_u = int(N), int(n_x), int(n_lam), int(n_u)
    x0   = b["x0"].ravel()
    xdes = b["xdes"].ravel()
    A, B, D = b["A"], b["B"], b["D"]
    d = b["d"].ravel()
    E, F, H = b["E"], b["F"], b["H"]
    c = b["c"].ravel()
    Q  = np.diag(b["Qdiag"].ravel())
    QN = np.diag(b["QNdiag"].ravel())
    R  = np.diag(b["Rdiag"].ravel())
    print(f"instance: N={N} n_x={n_x} n_lam={n_lam} n_u={n_u} AnDn={AnDn:.4f}")
    print(f"x0 ee=({x0[0]:.4f},{x0[1]:.4f},{x0[2]:.4f}) "
          f"obj=({x0[7]:.4f},{x0[8]:.4f})")
    g = b["Gdiag"].ravel()
    u = b["Udiag"].ravel()
    print(f"ref Gdiag blocks: x={g[:n_x].max():.3f} "
          f"lam={g[n_x:n_x+n_lam].mean():.3f} "
          f"u={g[n_x+n_lam:n_x+n_lam+n_u].max():.3f} "
          f"eta={g[n_x+n_lam+n_u:].mean():.3f}")
    print(f"ref Udiag lam block: {u[n_x:n_x+n_lam]}")

    s = C3Solver(n_x=n_x, n_u=n_u, mode="c3plus")
    ret = s._solve_c3plus(
        x0=x0, A=A, B_ctrl=B, D=D, d=d, E=E, F=F, H=H, c_lcs=c,
        J_n=np.zeros((0, n_x)), J_t=np.zeros((0, n_x)), mu=0.42,
        Q=Q, R=R, QN=QN, x_ref=xdes,
        N=N, admm_iter=3, torque_limit=10.0,
        u_lower=np.array([-10.0, -10.0, -3.0]),
        u_upper=np.array([10.0, 10.0, 3.0]),
        ee_velocity_bounds=(-0.14, 0.14),
        ee_vel_state_indices=(10, 11, 12),
    )
    x_seq = ret[1]
    dee = float(np.linalg.norm(np.asarray(x_seq)[-1, 0:2]
                               - np.asarray(x_seq)[0, 0:2]))
    print(f"FINAL plan dEE_xy = {dee*1000:.1f} mm  "
          f"(reference's own loop on this class: ~31->42->66 mm growth)")


if __name__ == "__main__":
    main()
