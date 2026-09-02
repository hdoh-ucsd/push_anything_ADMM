import numpy as np
import pytest

from control.sampling_c3.sampling_based_c3_controller import (
    classify_ee_box_contact,
    object_tilt_from_quaternion_wxyz,
)


def test_resolved_pair_across_gap_is_not_physical_contact():
    info = [{
        "tag": "EE-BOX",
        "distance": 0.01785,
        "nhat_BA_W": np.array([0.39, -0.087, 0.917]),
    }]

    pair, contact, index, distance, normal = classify_ee_box_contact(
        info, np.array([0.0])
    )

    assert pair
    assert not contact
    assert index == 0
    assert distance == 0.01785
    np.testing.assert_allclose(normal, [0.39, -0.087, 0.917])


def test_close_pair_with_normal_force_is_physical_contact():
    info = [{
        "tag": "EE-BOX",
        "distance": 0.001,
        "nhat_BA_W": np.array([1.0, 0.0, 0.0]),
    }]

    pair, contact, index, distance, _ = classify_ee_box_contact(
        info, np.array([0.1])
    )

    assert pair
    assert contact
    assert index == 0
    assert distance == 0.001


def test_close_pair_without_normal_force_is_not_physical_contact():
    info = [{
        "tag": "EE-BOX",
        "distance": -1e-5,
        "nhat_BA_W": np.array([1.0, 0.0, 0.0]),
    }]

    pair, contact, *_ = classify_ee_box_contact(info, np.array([0.0]))

    assert pair
    assert not contact


def test_object_tilt_ignores_yaw_but_detects_roll_pitch():
    yaw = 2.4
    upright_yaw = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    assert object_tilt_from_quaternion_wxyz(upright_yaw) == 0.0

    roll = np.deg2rad(30.0)
    rolled = [np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0]
    assert object_tilt_from_quaternion_wxyz(rolled) == pytest.approx(
        np.deg2rad(30.0)
    )
