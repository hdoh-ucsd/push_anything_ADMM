"""Tests for the diagnostic topple-roll plan (DIAG_JACK_TOPPLE_DRIVER).

Rolling the jack over the support edge formed by two of its three ground
tips toggles exactly ONE tripod sign — the third capsule's — so the
tripod graph is a 3-cube. To toggle sign k, push the UP end of capsule k
horizontally from above tip_k toward the midpoint of the other two
support tips. The up-end cap centre sits at CoM_z + h/sqrt(3) = 97.2 mm,
above the 55.3 mm tip-before-slide critical height, so sub-newton force
tips instead of sliding (see jack-topple-mechanics memory).
"""
import numpy as np
import pytest

from control.sampling_c3.goal_generator import (
    KNOMINAL_ORIENTATIONS_JACK,
    quat_multiply,
    topple_roll_plan,
    tripod_id,
)

JACK_HALF_LEN = 0.0625     # capsule centre -> tip-cap centre (jack.sdf)
COM_REST_Z = 0.061084


def _rotmat(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _flip_sign(tripod, k):
    t = list(tripod)
    t[k] = -t[k]
    return tuple(t)


def test_picks_first_differing_sign():
    q = KNOMINAL_ORIENTATIONS_JACK[5]           # BlueDown rest
    cur = tripod_id(q)
    for k in range(3):
        goal = _flip_sign(cur, k)
        kk, _p, _d = topple_roll_plan(q, cur, goal)
        assert kk == k
    # multi-sign difference: first differing index wins
    goal = _flip_sign(_flip_sign(cur, 1), 2)
    kk, _p, _d = topple_roll_plan(q, cur, goal)
    assert kk == 1


def test_push_point_is_up_end_above_critical_height():
    for q in KNOMINAL_ORIENTATIONS_JACK:
        cur = tripod_id(q)
        goal = _flip_sign(cur, 2)
        _k, p_B, _d = topple_roll_plan(q, cur, goal)
        assert np.linalg.norm(p_B) == pytest.approx(JACK_HALF_LEN, abs=1e-12)
        # world height of the pushed point = CoM_z + h/sqrt(3) = 97.2mm,
        # above the 55.3mm tip-before-slide critical height.
        z_world = COM_REST_Z + float((_rotmat(q) @ p_B)[2])
        assert z_world == pytest.approx(
            COM_REST_Z + JACK_HALF_LEN / np.sqrt(3.0), abs=1e-9)
        assert z_world > 0.0553


def test_direction_is_horizontal_unit_toward_far_edge():
    q = KNOMINAL_ORIENTATIONS_JACK[3]           # AllDown rest
    cur = tripod_id(q)
    R = _rotmat(q)
    h = JACK_HALF_LEN
    e = np.eye(3)
    tips = [R @ (-cur[i] * h * e[i]) for i in range(3)]
    for k in range(3):
        goal = _flip_sign(cur, k)
        _k, _p, d_W = topple_roll_plan(q, cur, goal)
        assert abs(d_W[2]) < 1e-12
        assert np.linalg.norm(d_W) == pytest.approx(1.0, abs=1e-12)
        # points from above tip_k toward the midpoint of the OTHER two tips
        i, j = [m for m in range(3) if m != k]
        edge_mid = 0.5 * (tips[i] + tips[j])
        away = (edge_mid - tips[k])[:2]
        assert float(np.dot(d_W[:2], away / np.linalg.norm(away))) > 0.99


def test_yaw_equivariance():
    # Yawing the object yaws the plan: same k, direction rotates with it.
    q = KNOMINAL_ORIENTATIONS_JACK[5]
    cur = tripod_id(q)
    goal = _flip_sign(cur, 0)
    _k0, p0, d0 = topple_roll_plan(q, cur, goal)
    yaw = 1.1
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    qy = quat_multiply(qz, q)
    _k1, p1, d1 = topple_roll_plan(qy, tripod_id(qy), goal)
    assert np.allclose(p1, p0)                  # body-frame point unchanged
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert np.allclose(d1, Rz @ d0, atol=1e-12)


def test_identical_tripods_returns_none():
    q = KNOMINAL_ORIENTATIONS_JACK[5]
    cur = tripod_id(q)
    assert topple_roll_plan(q, cur, cur) is None


def test_prefer_direction_picks_goalward_roll():
    # Goal differs in ALL THREE signs -> three candidate rolls; with a
    # preference direction the plan must pick the candidate whose push
    # direction has the largest dot with it (rolls walk the CoM along
    # d_W, so this reduces the goal-position gap flip by flip).
    q = KNOMINAL_ORIENTATIONS_JACK[5]
    cur = tripod_id(q)
    goal = tuple(-s for s in cur)
    dirs = {}
    for k in range(3):
        g1 = list(cur)
        g1[k] = -g1[k]
        _k, _p, d = topple_roll_plan(q, cur, tuple(g1))
        dirs[k] = d
    for k_want in range(3):
        prefer = dirs[k_want]
        kk, _p, d = topple_roll_plan(q, cur, goal, prefer_direction=prefer)
        assert kk == k_want
        assert float(np.dot(d, prefer)) > 0.99
    # no preference: first differing sign (unchanged default)
    kk, _p, _d = topple_roll_plan(q, cur, goal)
    assert kk == 0
