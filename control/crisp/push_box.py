r"""CRISP Appendix B-B — the Push Box contact-implicit formulation.

arXiv:2502.01055v3 eqs (52)-(68). A planar box of size 2a x 2b slides
quasi-statically on a table. Unlike our LCS lineage, the pusher is NOT in the
model: the decision variables are the box pose, the body-frame contact point,
and four face-normal forces.

    v = [p_x, p_y, theta | c_x, c_y, lam_1y, lam_2x, lam_3y, lam_4x]
         \___ state ___/   \_____________ control _____________/

Object-ground friction is the ellipsoidal limit surface (eqs 52-54), so there
are no ground contact points to author. Which face is pushed is decided by the
complementarity constraints (eqs 58-61), which force the contact point onto a
face whenever that face's force is nonzero, plus the pairwise exclusivity
constraints (eqs 62-67) that forbid two faces acting at once.

Sign convention, straight from the paper: lam_1y >= 0 acts on the -y face,
lam_2x >= 0 on the -x face, and lam_3y <= 0, lam_4x <= 0 act on the +y and +x
faces. Every force therefore points INTO the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from control.crisp.scp import NlpProblem

_NS = 3          # p_x, p_y, theta
_NU = 6          # c_x, c_y, lam_1y, lam_2x, lam_3y, lam_4x
_NK = _NS + _NU
_N_COMP = 10     # 4 face gates (58-61) + 6 exclusivity pairs (62-67)
_N_INEQ = 8      # 4 sign conditions + 4 box-boundary conditions

#: How far above the escape threshold the auto terminal weight sits.
#: 10x clears the barrier and tracks our 0.15 m box goal to 0.07%. Escaping the
#: origin and tracking WELL are separate requirements, and the gap between them
#: widens with goal distance / box size: at the paper's benchmark ratio (3 m
#: target, 1 m box) 10x leaves 12.2% terminal error and 1000x is needed for
#: 0.14%. Retarget this if the geometry changes materially.
AUTO_WEIGHT_SAFETY = 10.0


def min_terminal_weight(mu_0, half_extent, distance, dt, k_trans) -> float:
    r"""Smallest terminal weight that can leave the all-zero initial guess.

    The all-zero point is feasible: no force, no motion, every complementarity
    product zero. Raising a face force off zero costs linearised face-gate
    violation, because at lam = 0 the gate's contact-point gradient
    d/dc_x [lam_2x (c_x + a)] = lam_2x is itself zero, so the trial step cannot
    slide the contact onto the face for free. Per knot and per unit of force the
    trade is

        penalty   mu_0 * a          (gate residual lam*(c_x+a) at c_x = 0)
        benefit   q * d * dt * k_trans   (terminal cost bought by the motion)

    which is knot-count independent, so the origin is escapable iff

        q > mu_0 * a / (d * dt * k_trans).

    Measured on the 0.1 m cube: predicted 300.8, q = 290 leaves the box at
    0.0000 m and q = 310 drives it 0.1490 m of a 0.15 m goal.
    """
    return float(mu_0 * half_extent / max(distance * dt * k_trans, 1e-12))


@dataclass
class PushBoxParams:
    a: float                       # half-extent along body x
    b: float                       # half-extent along body y
    mu: float                      # object-ground friction coefficient
    mass: float                    # object mass used by the limit surface
    N: int                         # knots
    dt: float
    c_int: float = 0.6             # integration constant c in [0, 1], eq (54)
    g: float = 9.81
    r_char: float | None = None    # characteristic distance; default sqrt(a^2+b^2)
    # Terminal weights, eq (68). None auto-sizes them above the escape
    # threshold; the paper does not publish Q or R for this problem.
    q_pos: float | None = None
    q_yaw: float | None = None
    # Running control weight, eq (68). The reference costs the four lambdas
    # only (R = 0.001*I_4, SolvePushbox.cpp:132-137) and puts NO cost on the
    # contact point; r_contact > 0 is a port-only addition that pulls the
    # contact toward the box centre and so fights the face gates.
    r_lambda: float = 1e-2
    r_contact: float = 0.0
    penalty_mu_0: float = 10.0     # must match the CrispParams.mu_0 used to solve
    #: Reference encodes the four face gates and six exclusivity products as
    #: INEQUALITIES -- e.g. -(lam_1y)(c_y+b) >= 0 -- not equalities
    #: (SolvePushbox.cpp:68-99, addInequalityConstraint). With lambda >= 0 and
    #: the box bounds also enforced, the product is >= 0, so the two agree on
    #: the true constraint value; they differ inside the QP, where the
    #: inequality form leaves the LINEARISED product free to go negative while
    #: the equality form penalises both sides.
    complementarity_as_inequality: bool = True

    def __post_init__(self):
        if self.r_char is None:
            self.r_char = float(np.hypot(self.a, self.b))


class PushBoxProblem(NlpProblem):
    def __init__(self, params: PushBoxParams, s_init, s_goal):
        self.p = params
        self.s_init = np.asarray(s_init, dtype=float)
        self.s_goal = np.asarray(s_goal, dtype=float)
        N = params.N
        self.n = _NK * N + _NS
        self.k_trans = 1.0 / (params.mu * params.mass * params.g)
        self.k_rot = 1.0 / (
            params.c_int * params.r_char * params.mu * params.mass * params.g
        )
        self._comp_ineq = bool(params.complementarity_as_inequality)
        if self._comp_ineq:
            self.m_eq = _NS + _NS * N
            self.m_ineq = (_N_INEQ + _N_COMP) * N
        else:
            self.m_eq = _NS + _NS * N + _N_COMP * N
            self.m_ineq = _N_INEQ * N
        self.q_star = self._escape_threshold()
        q_pos = params.q_pos if params.q_pos is not None else (
            AUTO_WEIGHT_SAFETY * self.q_star)
        q_yaw = params.q_yaw if params.q_yaw is not None else q_pos
        self._Q = np.array([q_pos, q_pos, q_yaw])
        self._R = np.array([params.r_contact, params.r_contact] +
                           [params.r_lambda] * 4)
        self._hess = self._build_hess()
        self._ineq_jac = self._build_ineq_jac()
        self._comp_pattern = (self._build_comp_pattern() if self._comp_ineq
                              else None)

    def _escape_threshold(self) -> float:
        """min_terminal_weight for this goal, with yaw-only goals folded in."""
        p = self.p
        d_trans = float(np.linalg.norm(self.s_goal[:2] - self.s_init[:2]))
        d_yaw = p.r_char * abs(self.s_goal[2] - self.s_init[2])
        distance = max(d_trans, d_yaw, 0.1 * p.r_char)
        return min_terminal_weight(p.penalty_mu_0, max(p.a, p.b), distance,
                                   p.dt, self.k_trans)

    # ------------------------------------------------------------- indexing
    def _si(self, k: int) -> int:
        return _NK * k

    def _ui(self, k: int) -> int:
        return _NK * k + _NS

    def pack(self, states, controls) -> np.ndarray:
        z = np.zeros(self.n)
        for k in range(self.p.N):
            z[self._si(k):self._si(k) + _NS] = states[k]
            z[self._ui(k):self._ui(k) + _NU] = controls[k]
        z[self._si(self.p.N):] = states[self.p.N]
        return z

    def unpack(self, z):
        N = self.p.N
        states = np.zeros((N + 1, _NS))
        controls = np.zeros((N, _NU))
        for k in range(N):
            states[k] = z[self._si(k):self._si(k) + _NS]
            controls[k] = z[self._ui(k):self._ui(k) + _NU]
        states[N] = z[self._si(N):]
        return states, controls

    # ------------------------------------------------------------- dynamics
    def dynamics(self, s, u) -> np.ndarray:
        """Eqs (52)-(54): quasi-static limit-surface twist from the contact wrench."""
        theta = s[2]
        cx, cy, l1, l2, l3, l4 = u
        fx, fy = l2 + l4, l1 + l3           # body-frame contact force
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            self.k_trans * (fx * ct - fy * st),
            self.k_trans * (fx * st + fy * ct),
            self.k_rot * (cx * fy - cy * fx),
        ])

    def _dynamics_jacobians(self, s, u):
        """(d f / d s, d f / d u) at one knot."""
        theta = s[2]
        cx, cy, l1, l2, l3, l4 = u
        fx, fy = l2 + l4, l1 + l3
        ct, st = np.cos(theta), np.sin(theta)
        kt, kr = self.k_trans, self.k_rot

        df_ds = np.zeros((_NS, _NS))
        df_ds[0, 2] = kt * (-fx * st - fy * ct)
        df_ds[1, 2] = kt * (fx * ct - fy * st)

        df_du = np.zeros((_NS, _NU))
        #                      c_x   c_y   lam_1y lam_2x lam_3y lam_4x
        df_du[0, 2:] = [-kt * st, kt * ct, -kt * st, kt * ct]
        df_du[1, 2:] = [kt * ct, kt * st, kt * ct, kt * st]
        df_du[2, 0] = kr * fy
        df_du[2, 1] = -kr * fx
        df_du[2, 2:] = [kr * cx, -kr * cy, kr * cx, -kr * cy]
        return df_ds, df_du

    # ---------------------------------------------------------- constraints
    def eq_constraints(self, z) -> np.ndarray:
        p = self.p
        states, controls = self.unpack(z)
        out = np.empty(self.m_eq)
        out[:_NS] = states[0] - self.s_init
        off = _NS
        for k in range(p.N):
            out[off:off + _NS] = (states[k + 1] - states[k]
                                  - p.dt * self.dynamics(states[k], controls[k]))
            off += _NS
        if not self._comp_ineq:
            for k in range(p.N):
                out[off:off + _N_COMP] = self._complementarity(controls[k])
                off += _N_COMP
        return out

    def _complementarity(self, u) -> np.ndarray:
        a, b = self.p.a, self.p.b
        cx, cy, l1, l2, l3, l4 = u
        return np.array([
            l1 * (cy + b),          # eq 58: -y face gate
            l2 * (cx + a),          # eq 59: -x face gate
            (-l3) * (b - cy),       # eq 60: +y face gate
            (-l4) * (a - cx),       # eq 61: +x face gate
            l1 * l2,                # eqs 62-67: at most one face active
            l1 * (-l3),
            l1 * (-l4),
            l2 * (-l3),
            l2 * (-l4),
            (-l3) * (-l4),
        ])

    def ineq_constraints(self, z) -> np.ndarray:
        a, b = self.p.a, self.p.b
        _, controls = self.unpack(z)
        out = np.empty(self.m_ineq)
        w = _N_INEQ + _N_COMP if self._comp_ineq else _N_INEQ
        for k in range(self.p.N):
            cx, cy, l1, l2, l3, l4 = controls[k]
            row = [
                l1, l2, -l3, -l4,               # unidirectional forces
                cy + b, cx + a, b - cy, a - cx,  # contact point stays on the box
            ]
            if self._comp_ineq:
                row += list(-self._complementarity(controls[k]))
            out[w * k:w * (k + 1)] = row
        return out

    def eq_jacobian(self, z) -> sp.spmatrix:
        p = self.p
        states, controls = self.unpack(z)
        rows, cols, vals = [], [], []

        def put(r, c, v):
            rows.append(r)
            cols.append(c)
            vals.append(v)

        for i in range(_NS):
            put(i, self._si(0) + i, 1.0)

        off = _NS
        for k in range(p.N):
            df_ds, df_du = self._dynamics_jacobians(states[k], controls[k])
            for i in range(_NS):
                put(off + i, self._si(k + 1) + i, 1.0)
                for j in range(_NS):
                    v = (-1.0 if i == j else 0.0) - p.dt * df_ds[i, j]
                    if v:
                        put(off + i, self._si(k) + j, v)
                for j in range(_NU):
                    v = -p.dt * df_du[i, j]
                    if v:
                        put(off + i, self._ui(k) + j, v)
            off += _NS

        if not self._comp_ineq:
            for k in range(p.N):
                u0 = self._ui(k)
                for r, entries in self._complementarity_terms(controls[k]):
                    for j, v in entries:
                        if v:
                            put(off + r, u0 + j, v)
                off += _N_COMP

        return sp.csc_matrix((vals, (rows, cols)), shape=(self.m_eq, self.n))

    def _complementarity_terms(self, u):
        """d(complementarity)/du as (row, [(u-slot, value), ...]) per row."""
        a, b = self.p.a, self.p.b
        cx, cy, l1, l2, l3, l4 = u
        return [
            (0, [(1, l1), (2, cy + b)]),
            (1, [(0, l2), (3, cx + a)]),
            (2, [(1, l3), (4, cy - b)]),
            (3, [(0, l4), (5, cx - a)]),
            (4, [(2, l2), (3, l1)]),
            (5, [(2, -l3), (4, -l1)]),
            (6, [(2, -l4), (5, -l1)]),
            (7, [(3, -l3), (4, -l2)]),
            (8, [(3, -l4), (5, -l2)]),
            (9, [(4, l4), (5, l3)]),
        ]

    def _build_ineq_jac(self) -> sp.spmatrix:
        """The sign/bound block, which is constant. Complementarity is added
        per-z by ineq_jacobian() when it lives on the inequality side."""
        rows, cols, vals = [], [], []
        # (row offset, u-slot, coefficient); constants drop out of the Jacobian.
        pattern = [(0, 2, 1.0), (1, 3, 1.0), (2, 4, -1.0), (3, 5, -1.0),
                   (4, 1, 1.0), (5, 0, 1.0), (6, 1, -1.0), (7, 0, -1.0)]
        w = _N_INEQ + _N_COMP if self._comp_ineq else _N_INEQ
        for k in range(self.p.N):
            for r, j, v in pattern:
                rows.append(w * k + r)
                cols.append(self._ui(k) + j)
                vals.append(v)
        return sp.csc_matrix((vals, (rows, cols)), shape=(self.m_ineq, self.n))

    #: (row within the 10 complementarity rows, u-slot) for each nonzero.
    #: The sparsity pattern is fixed; only the values depend on u.
    _COMP_NZ = ((0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 4), (3, 0), (3, 5),
                (4, 2), (4, 3), (5, 2), (5, 4), (6, 2), (6, 5), (7, 3), (7, 4),
                (8, 3), (8, 5), (9, 4), (9, 5))

    def _build_comp_pattern(self):
        """Fixed (row, col) index arrays for the complementarity block."""
        w = _N_INEQ + _N_COMP
        rows = np.empty(self.p.N * len(self._COMP_NZ), dtype=np.int64)
        cols = np.empty_like(rows)
        for k in range(self.p.N):
            base = k * len(self._COMP_NZ)
            u0 = self._ui(k)
            for i, (r, j) in enumerate(self._COMP_NZ):
                rows[base + i] = w * k + _N_INEQ + r
                cols[base + i] = u0 + j
        return rows, cols

    def _comp_values(self, controls) -> np.ndarray:
        """d(-complementarity)/du for every knot, in _COMP_NZ order."""
        a, b = self.p.a, self.p.b
        cx, cy = controls[:, 0], controls[:, 1]
        l1, l2, l3, l4 = (controls[:, 2], controls[:, 3],
                          controls[:, 4], controls[:, 5])
        v = np.stack([
            l1, cy + b,          # row 0: lam1*(cy+b)
            l2, cx + a,          # row 1: lam2*(cx+a)
            l3, cy - b,          # row 2: (-lam3)*(b-cy)
            l4, cx - a,          # row 3: (-lam4)*(a-cx)
            l2, l1,              # row 4: lam1*lam2
            -l3, -l1,            # row 5
            -l4, -l1,            # row 6
            -l3, -l2,            # row 7
            -l4, -l2,            # row 8
            l4, l3,              # row 9
        ], axis=1)
        return -v.reshape(-1)    # rows hold the NEGATED products

    def ineq_jacobian(self, z) -> sp.spmatrix:
        if not self._comp_ineq:
            return self._ineq_jac
        _, controls = self.unpack(z)
        comp = sp.csc_matrix(
            (self._comp_values(controls), self._comp_pattern),
            shape=(self.m_ineq, self.n))
        return (self._ineq_jac + comp).tocsc()

    # -------------------------------------------------------------- objective
    def objective(self, z) -> float:
        states, controls = self.unpack(z)
        err = states[-1] - self.s_goal
        return float(0.5 * (self._Q @ err ** 2)
                     + 0.5 * np.sum(self._R * controls ** 2))

    def objective_grad(self, z) -> np.ndarray:
        states, controls = self.unpack(z)
        g = np.zeros(self.n)
        g[self._si(self.p.N):] = self._Q * (states[-1] - self.s_goal)
        for k in range(self.p.N):
            g[self._ui(k):self._ui(k) + _NU] = self._R * controls[k]
        return g

    def _build_hess(self) -> sp.spmatrix:
        diag = np.zeros(self.n)
        for k in range(self.p.N):
            diag[self._ui(k):self._ui(k) + _NU] = self._R
        diag[self._si(self.p.N):] = self._Q
        return sp.diags(diag, format="csc")

    def objective_hess(self, z) -> sp.spmatrix:
        return self._hess

    # ------------------------------------------------------------- reporting
    def contact_point_world(self, s, u) -> np.ndarray:
        """Body-frame contact point mapped into the world (execution target)."""
        ct, st = np.cos(s[2]), np.sin(s[2])
        return np.array([s[0] + ct * u[0] - st * u[1],
                         s[1] + st * u[0] + ct * u[1]])

    @staticmethod
    def active_face(u, tol=1e-6):
        """Which face carries force at this knot: '-y', '-x', '+y', '+x', or None."""
        mags = np.array([u[2], u[3], -u[4], -u[5]])
        i = int(np.argmax(mags))
        return ("-y", "-x", "+y", "+x")[i] if mags[i] > tol else None


@dataclass
class ExecutionPlan:
    """A B-B solution expressed in the quantities our OSC already consumes.

    B-B plans a contact point on the box SURFACE. A real pusher has a radius, so
    the end-effector centre sits one radius further out along the outward face
    normal -- the same geometric offset the sampler calls `sampling_setback`.
    """

    times: np.ndarray        # (N,)
    ee_xyz: np.ndarray       # (N, 3) end-effector centre, world
    contact_xy: np.ndarray   # (N, 2) contact point on the surface, world
    force_world: np.ndarray  # (N, 2) force applied TO the box, world
    face: np.ndarray         # (N,) object array of '-x' | '+x' | '-y' | '+y' | None
    box_pose: np.ndarray     # (N+1, 3)
    dt: float

    def max_ee_speed(self) -> float:
        """Peak commanded end-effector speed, m/s. The executability test."""
        if len(self.ee_xyz) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.ee_xyz, axis=0), axis=1).max()
                     / self.dt)

    def face_switches(self) -> int:
        """Face changes between consecutive pushing knots -- each one is a
        reposition the B-B model itself does not pay for."""
        f = [x for x in self.face if x is not None]
        return sum(1 for i in range(1, len(f)) if f[i] != f[i - 1])


def to_execution_plan(prob: PushBoxProblem, z, ee_height, pusher_radius=0.025
                      ) -> ExecutionPlan:
    """Convert a solved B-B trajectory into end-effector targets and forces."""
    states, controls = prob.unpack(z)
    N = prob.p.N
    contact = np.empty((N, 2))
    ee = np.empty((N, 3))
    force = np.empty((N, 2))
    faces = np.empty(N, dtype=object)

    for k in range(N):
        s, u = states[k], controls[k]
        ct, st = np.cos(s[2]), np.sin(s[2])
        R = np.array([[ct, -st], [st, ct]])
        f_body = np.array([u[3] + u[5], u[2] + u[4]])
        contact[k] = prob.contact_point_world(s, u)
        force[k] = R @ f_body
        faces[k] = PushBoxProblem.active_face(u)

        norm = np.linalg.norm(f_body)
        if norm > 1e-9:
            out_body = -f_body / norm      # outward normal opposes the push
        else:
            c = np.array(u[:2])
            cn = np.linalg.norm(c)
            out_body = c / cn if cn > 1e-9 else np.array([-1.0, 0.0])
        ee[k, :2] = contact[k] + pusher_radius * (R @ out_body)
        ee[k, 2] = ee_height

    return ExecutionPlan(
        times=np.arange(N) * prob.p.dt,
        ee_xyz=ee,
        contact_xy=contact,
        force_world=force,
        face=faces,
        box_pose=states,
        dt=prob.p.dt,
    )
