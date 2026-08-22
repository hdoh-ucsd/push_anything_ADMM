#!/usr/bin/env python3
"""Measure what CRISP Appendix B-B does on our box task.

Reproduces the numbers quoted in control/crisp/README.md.

    python3 tools/crisp/analyse_push_box.py [cases|weights|barrier|hard|timing|refcfg]

`cases`   B-B on our 0.1 m cube for five goals, with the executability metrics.
`weights` terminal-weight calibration vs terminal tracking error.
`barrier` the all-zero escape threshold, predicted vs measured.
`hard`    do the multi-face goals fail on weights or structurally?
`timing`  receding-horizon solve cost against our 75 ms control period.
`refcfg`  replicate the reference SolvePushbox.cpp configuration verbatim.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from control.crisp.push_box import (  # noqa: E402
    PushBoxParams,
    PushBoxProblem,
    min_terminal_weight,
    to_execution_plan,
)
from control.crisp.scp import CrispParams, CrispSolver

OURS = dict(a=0.05, b=0.05, mu=0.46, mass=1.0, c_int=0.6)   # config/tasks.yaml: pushing
BENCH = dict(a=0.5, b=0.25, mu=0.5, mass=1.0, g=9.8, c_int=0.4)  # SolvePushbox.cpp:9-16

CASES = [
    ("canonical  +x 0.15m", [0.15, 0.0, 0.0]),
    ("reverse    -x 0.15m", [-0.15, 0.0, 0.0]),
    ("diagonal   +x+y 0.10m", [0.10, 0.10, 0.0]),
    ("translate + rotate 90deg", [0.15, 0.0, np.pi / 2]),
    ("pure yaw   90deg", [0.0, 0.0, np.pi / 2]),
]


def solve(box, goal, N=40, dt=0.05, q_mult=None, r_lambda=1e-2, z0=None):
    p = PushBoxParams(N=N, dt=dt, r_lambda=r_lambda, **box)
    prob = PushBoxProblem(p, np.zeros(3), np.array(goal))
    if q_mult is not None:
        q = q_mult * prob.q_star
        p = PushBoxParams(N=N, dt=dt, r_lambda=r_lambda, q_pos=q, q_yaw=q, **box)
        prob = PushBoxProblem(p, np.zeros(3), np.array(goal))
    t0 = time.perf_counter()
    res = CrispSolver(CrispParams()).solve(
        prob, np.zeros(prob.n) if z0 is None else z0)
    return prob, res, time.perf_counter() - t0


def cmd_cases():
    print("=== B-B on our 0.1 m cube, N=40 dt=0.05 (2 s horizon) ===")
    for name, goal in CASES:
        prob, res, wall = solve(OURS, goal)
        s, _ = prob.unpack(res.z)
        plan = to_execution_plan(prob, res.z, ee_height=0.05, pusher_radius=0.025)
        pos = np.linalg.norm(s[-1, :2] - np.array(goal[:2]))
        print(f"{name:26s} {res.status:24s} iters={res.iterations:4d} {wall:6.2f}s")
        print(f"{'':26s} pos_err={pos*1000:7.2f}mm yaw_err={abs(s[-1,2]-goal[2]):6.4f}rad "
              f"switches={plan.face_switches()} peak_ee={plan.max_ee_speed():.2f} m/s")


def cmd_weights():
    print("=== terminal weight vs terminal tracking error ===")
    for tag, box, goal, N, dt in (("bench", BENCH, [3.0, 0, 0], 200, 0.02),
                                  ("ours", OURS, [0.15, 0, 0], 40, 0.05)):
        d = np.linalg.norm(goal[:2])
        for mult in (10, 100, 1000):
            prob, res, wall = solve(box, goal, N, dt, q_mult=mult)
            s, _ = prob.unpack(res.z)
            err = np.linalg.norm(s[-1, :2] - np.array(goal[:2]))
            print(f"{tag:6s} q x{mult:<6g} err={err:8.4f}m ({100*err/d:5.2f}%) "
                  f"viol={res.max_violation:.1e} iters={res.iterations:4d} "
                  f"{wall:6.1f}s {res.status}")


def cmd_barrier():
    k_trans = 1.0 / (OURS["mu"] * OURS["mass"] * 9.81)
    q_star = min_terminal_weight(CrispParams().mu_0, OURS["a"], 0.15, 0.05, k_trans)
    print(f"=== all-zero escape threshold: predicted q* = {q_star:.1f} ===")
    for q in (150, 250, 290, 310, 400, 600):
        p = PushBoxParams(N=40, dt=0.05, q_pos=float(q), q_yaw=float(q), **OURS)
        prob = PushBoxProblem(p, np.zeros(3), np.array([0.15, 0.0, 0.0]))
        res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))
        moved = np.linalg.norm(prob.unpack(res.z)[0][-1, :2])
        print(f"q={q:5d} ({'>' if q > q_star else '<'} q*)  "
              f"{'ESCAPED' if moved > 0.075 else 'parked ':8s} moved={moved:.4f}m  "
              f"viol={res.max_violation:.1e}  {res.status}")


def cmd_hard():
    print("=== do the multi-face goals fail on weights, or structurally? ===")
    for name, goal in (("diagonal +x+y 0.10", [0.10, 0.10, 0.0]),
                       ("pure yaw 90deg", [0.0, 0.0, np.pi / 2]),
                       ("pure yaw 20deg", [0.0, 0.0, 0.35])):
        print(f"--- {name} ---")
        for mult in (10, 100, 1000):
            prob, res, _ = solve(OURS, goal, q_mult=mult)
            s, u = prob.unpack(res.z)
            faces = sorted({PushBoxProblem.active_face(x, 1e-4) for x in u} - {None})
            print(f"    q x{mult:<6g} pos_err="
                  f"{np.linalg.norm(s[-1,:2]-np.array(goal[:2]))*1000:7.2f}mm "
                  f"yaw_err={abs(s[-1,2]-goal[2]):6.4f}rad faces={faces or ['-']} "
                  f"{res.status}")


def cmd_timing():
    print("=== receding-horizon cost vs our 75 ms control period ===")
    for N, dt in ((7, 0.075), (10, 0.05), (20, 0.05)):
        _, res, wall = solve(OURS, [0.15, 0.0, 0.0], N=N, dt=dt)
        print(f"N={N:3d} dt={dt:<6g} {wall*1000:8.1f} ms/solve iters={res.iterations:3d}"
              f"  {res.status}")
    print("port C3+ tick: 75 ms (N=7, dt=0.075); paper C++ push-T MPC: 80 ms (N=10)")


def cmd_refcfg():
    """Replicate the reference's own SolvePushbox.cpp configuration verbatim.

    ComputationalRobotics/CRISP @ src/examples/pushbox/SolvePushbox.cpp:
    a=0.5 b=0.25 m=1 mu=0.5 g=9.8 c=0.4 dt=0.02 N=100; Q=diag(100,100,100) at
    the terminal knot WITHOUT a 1/2 factor; R=0.001 on the four lambdas only;
    goal = [3cos(theta), 3sin(theta), theta] with theta = 12*2pi/18 -- so the
    benchmark demands 3 m of travel AND 240 deg of rotation.
    """
    theta = 12 * 2 * np.pi / 18
    goal = np.array([3 * np.cos(theta), 3 * np.sin(theta), theta])
    ref = dict(a=0.5, b=0.25, mu=0.5, mass=1.0, g=9.8, c_int=0.4,
               N=100, dt=0.02, q_pos=200.0, q_yaw=200.0,   # Q=100, no 1/2
               r_lambda=0.002, r_contact=0.0)              # R=0.001 on lambda
    solver = CrispParams(mu_0=10.0, mu_max=1e8, k_max=5000,
                         eps_p=1e-3, eps_r=1e-3, eps_c=1e-6)

    print("=== reference SolvePushbox.cpp configuration ===")
    print(f"goal = [{goal[0]:.4f}, {goal[1]:.4f}, {goal[2]:.4f}]  "
          f"(|xy| = 3.000 m, yaw = {np.degrees(theta) % 360:.0f} deg)")
    prob = PushBoxProblem(PushBoxParams(**ref), np.zeros(3), goal)
    t0 = time.perf_counter()
    res = CrispSolver(solver).solve(prob, np.zeros(prob.n))
    wall = time.perf_counter() - t0
    s, u = prob.unpack(res.z)
    faces = sorted({PushBoxProblem.active_face(x, 1e-4) for x in u} - {None})
    print(f"{res.status:24s} iters={res.iterations} {wall:.1f}s "
          f"viol={res.max_violation:.2e}")
    print(f"  moved {np.linalg.norm(s[-1, :2]):.4f} m of 3.000 m   "
          f"turned {abs(s[-1, 2]):.4f} rad of {theta:.4f} rad")
    print(f"  pos_err={np.linalg.norm(s[-1, :2] - goal[:2]):.4f} m  "
          f"yaw_err={abs(s[-1, 2] - goal[2]):.4f} rad   faces={faces or ['-']}")


if __name__ == "__main__":
    cmds = {"cases": cmd_cases, "weights": cmd_weights, "barrier": cmd_barrier,
            "hard": cmd_hard, "timing": cmd_timing, "refcfg": cmd_refcfg}
    for name in (sys.argv[1:] or ["cases"]):
        cmds[name]()
        print()
