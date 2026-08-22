"""Path B probe — measure candidate cost as a function of contact location.

Read-only diagnostic, entirely env-gated, zero cost when off. It answers the
question that gates Path B: along one box face, is the candidate cost a smooth
function of the contact coordinate `s`, or is it rough enough that local
refinement cannot beat another random draw?

Enable with a comma-separated list of planner ticks:

    PORT_PATHB_SWEEP=120,240,360  [PORT_PATHB_SWEEP_N=21]

At each listed tick the probe re-runs the controller's own
`InnerSolver.evaluate_samples` on a deterministic sweep of contact locations
along every side face, then emits one line per point:

    [PATHB-SWEEP] step=<k> face=<f> s=<s> x= y= z= c_sample= c_raw=
                  align= rot= ik_err= ik_track= ok=<0|1>

`c_sample` is the ranked cost the argmin actually uses; `ik_track` is how far
the IK-resolved EE ended up from the requested placement, which bounds how much
of any observed roughness is the contact model versus the IK.

The sweep uses the same face geometry, setback and sampling height as
`_face_normal_projection`, so `s` is exactly the baseline's tangential jitter
rescaled to [0, 1]:  jitter = (2s - 1) * half_len.
"""

from __future__ import annotations

import os

import numpy as np

from control.sampling_c3.sampling import _quat_to_rot

#: Body-frame outward normals of a box's four side faces.
_BOX_SIDE_NORMALS = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
])
_FACE_NAMES = ("+x", "-x", "+y", "-y")


def sweep_ticks() -> set[int]:
    raw = os.environ.get("PORT_PATHB_SWEEP", "").strip()
    if not raw:
        return set()
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return out


def face_geometry(obj_xy, obj_quat, box_half):
    """(name, centre_xy, tangent_xy, normal_xy, half_len) for each side face."""
    R = _quat_to_rot(np.asarray(obj_quat, float)) if obj_quat is not None \
        else np.eye(3)
    obj2 = np.asarray(obj_xy, float).reshape(-1)[:2]
    out = []
    for name, n_body in zip(_FACE_NAMES, _BOX_SIDE_NORMALS):
        n_w = R @ n_body
        n2 = n_w[:2]
        nn = float(np.linalg.norm(n2))
        if nn < 1e-9:
            continue
        n2 = n2 / nn
        t2 = np.array([-n2[1], n2[0]])
        out.append((name, obj2 + box_half * n2, t2, n2, float(box_half)))
    return out


def contact_point(centre_xy, tangent_xy, half_len, s):
    """c(s) = p0 + s (p1 - p0) with p0/p1 the face corners. Affine in s."""
    return centre_xy + (2.0 * float(s) - 1.0) * half_len * tangent_xy


def ee_placement(centre_xy, tangent_xy, normal_xy, half_len, s, setback, z):
    c = contact_point(centre_xy, tangent_xy, half_len, s)
    xy = c + setback * normal_xy
    return np.array([xy[0], xy[1], float(z)])


def run_sweep(inner_solver, step, obj_xy, obj_quat, box_half, setback, z,
              eval_kwargs, n_points=None):
    """Sweep every face and print one [PATHB-SWEEP] line per point."""
    n = int(n_points or os.environ.get("PORT_PATHB_SWEEP_N", "21"))
    faces = face_geometry(obj_xy, obj_quat, box_half)
    s_vals = np.linspace(0.0, 1.0, n)

    positions, tags = [], []
    for name, centre, tang, norm, half in faces:
        for s in s_vals:
            positions.append(
                ee_placement(centre, tang, norm, half, s, setback, z))
            tags.append((name, float(s)))

    try:
        results = inner_solver.evaluate_samples(samples=positions, **eval_kwargs)
    except Exception as exc:                      # pragma: no cover - diagnostic
        print(f"[PATHB-SWEEP] step={step} FAILED {type(exc).__name__}: {exc}",
              flush=True)
        return

    for (name, s), p, r in zip(tags, positions, results):
        if r is None:
            print(f"[PATHB-SWEEP] step={step} face={name} s={s:.4f} ok=0",
                  flush=True)
            continue
        ik_track = float(np.linalg.norm(np.asarray(r.ee_pos_resolved) - p))
        ok = int(bool(r.feasible) and np.isfinite(r.c_sample))
        print(f"[PATHB-SWEEP] step={step} face={name} s={s:.4f} "
              f"x={p[0]:+.5f} y={p[1]:+.5f} z={p[2]:+.5f} "
              f"c_sample={r.c_sample:.6f} c_raw={r.c_C3_raw:.6f} "
              f"align={r.align_score:.4f} rot={r.rot_score:.4f} "
              f"ik_err={r.ik_err:.6f} ik_track={ik_track:.6f} ok={ok}",
              flush=True)
