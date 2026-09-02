"""Canonical DAIRLab cost profiles and Push Anything multi-object expansion.

Values are transcribed from
``examples/sampling_c3/multiyaml_rewrite.py`` on dairlib's
``push_anything_dev`` branch.  Keeping the profile here avoids copying stale
single-object literals into every task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_SCALES = {
    1: {"N": 10, "threads": 6, "w_R": 6.0, "w_G": 0.18},
    2: {"N": 15, "threads": 5, "w_R": 6.0, "w_G": 0.18},
    3: {"N": 7, "threads": 4, "w_R": 10.0, "w_G": 0.15},
    4: {"N": 7, "threads": 5, "w_R": 10.0, "w_G": 0.20},
}


@dataclass(frozen=True)
class PushAnythingCostProfile:
    """Expanded cost values for a Push Anything scene."""

    num_objects: int
    horizon: int
    num_outer_threads: int
    w_Q: float = 80.0
    w_Q_position: float = 50.0
    w_R: float = 6.0
    w_G: float = 0.18
    w_U: float = 0.5
    w_U_position: float = 0.26
    quaternion_weight: float = 1000.0
    quaternion_regularizer_fraction: float = 0.0
    terminal_multiplier: float = 1.0

    @property
    def include_walls(self) -> bool:
        return self.num_objects < 4

    @property
    def num_contact_pairs(self) -> int:
        object_object = self.num_objects * (self.num_objects - 1) // 2
        object_ground = 3 * self.num_objects
        object_wall = self.num_objects if self.include_walls else 0
        return 1 + object_ground + object_object + object_wall

    def q_vector(self, pose_regime: bool) -> np.ndarray:
        """Return the upstream state-cost vector for ``num_objects``.

        Layout: EE position; each object's quaternion and position; EE linear
        velocity; each object's angular and linear velocity.
        """
        values = [0.01, 0.01, 0.01]
        obj_quat = [0.1] * 4 if pose_regime else [0.0] * 4
        obj_pos = [150.0, 150.0, 120.0] if pose_regime else [200.0, 200.0, 120.0]
        for _ in range(self.num_objects):
            values.extend(obj_quat)
            values.extend(obj_pos)
        values.extend([15.0, 15.0, 10.0])
        for _ in range(self.num_objects):
            values.extend([0.05] * 3)
            values.extend([0.05] * 3)
        return np.asarray(values, dtype=float)

    def r_vector(self) -> np.ndarray:
        return np.asarray([0.01, 0.01, 1.0], dtype=float)

    def tracking_matrices(self, pose_regime: bool) -> tuple[np.ndarray, np.ndarray]:
        """Build reference-layout Q and R matrices for any supported N."""
        w_q = self.w_Q if pose_regime else self.w_Q_position
        return w_q * np.diag(self.q_vector(pose_regime)), self.w_R * np.diag(self.r_vector())

    def object_quaternion_slice(self, object_index: int) -> slice:
        if not 0 <= object_index < self.num_objects:
            raise IndexError(object_index)
        start = 3 + 7 * object_index
        return slice(start, start + 4)

    def consensus_vectors(self) -> dict[str, np.ndarray | float]:
        """Return the C3+ G/U literals emitted by upstream multi-rewrite."""
        n_lambda = 4 * self.num_contact_pairs
        n_obj_ground = 3 * self.num_objects
        n_obj_object = self.num_objects * (self.num_objects - 1) // 2
        n_walls = self.num_objects if self.include_walls else 0
        u_lambda = (
            [1000.0] * 4
            + [1000.0] * (4 * n_obj_ground)
            + [1.0] * (4 * n_obj_object)
            + [1.0] * (4 * n_walls)
        )
        assert len(u_lambda) == n_lambda
        return {
            "w_G": self.w_G,
            "w_U": self.w_U,
            "w_U_position": self.w_U_position,
            "g_x": np.zeros(6 + 13 * self.num_objects),
            "g_u": np.zeros(3),
            "g_lambda": np.full(n_lambda, 2.0),
            "g_eta": np.ones(n_lambda),
            "u_x": np.zeros(6 + 13 * self.num_objects),
            "u_u": np.zeros(3),
            "u_lambda": np.asarray(u_lambda),
            "u_eta": np.ones(n_lambda),
            "final_augmented_cost_contact_scaling": 1000.0,
            "final_augmented_cost_contact_indices": np.arange(4),
        }


def push_anything_cost_profile(num_objects: int = 1) -> PushAnythingCostProfile:
    """Return the exact upstream profile for one through four objects."""
    try:
        scales = _SCALES[int(num_objects)]
    except (KeyError, ValueError) as exc:
        raise ValueError("Push Anything supports num_objects in [1, 4]") from exc
    return PushAnythingCostProfile(
        num_objects=int(num_objects),
        horizon=int(scales["N"]),
        num_outer_threads=int(scales["threads"]),
        w_R=float(scales["w_R"]),
        w_G=float(scales["w_G"]),
    )


def jacktoy_c3plus_cost_config() -> dict:
    """Return the jacktoy C3+ tracking cost from the DAIRLab experiment.

    The pose and position regimes are transcribed from
    ``jacktoy/parameters/sampling_c3plus_options.yaml`` on
    ``push_anything_dev``. Solver-consensus values (w_G=0.03, w_U=0.26 and
    u_lambda=4) live in the sampling-controller configuration because they
    are consumed by a different subsystem in this port.
    """
    return {
        "use_reference_q_vector": True,
        "w_Q": 45.0,
        "w_Q_position": 45.0,
        "q_vector_ee_pos": [0.01, 0.01, 0.01],
        "q_vector_obj_quat": [0.1, 0.1, 0.1, 0.1],
        "q_vector_obj_pos": [200.0, 200.0, 120.0],
        "q_vector_position_obj_quat": [0.1, 0.1, 0.1, 0.1],
        "q_vector_position_obj_pos": [1500.0, 1500.0, 1500.0],
        "q_vector_ee_vel": [5.0, 5.0, 5.0],
        "q_vector_obj_ang_vel": [0.05, 0.05, 0.05],
        "q_vector_obj_lin_vel": [0.05, 0.05, 0.05],
        "q_vector_position_obj_ang_vel": [0.01, 0.01, 0.01],
        "q_vector_position_obj_lin_vel": [1.0, 1.0, 1.0],
        "w_R": 1.0,
        "r_vector": [0.01, 0.01, 0.01],
        "use_quaternion_dependent_cost": True,
        "q_quaternion_dependent_weight": 2500.0,
        "q_quaternion_dependent_regularizer_fraction": 0.0,
        "w_terminal": 1.0,
    }


def resolve_cost_config(cost_cfg: dict) -> dict:
    """Expand ``profile: push_anything`` into the current one-object API."""
    c = dict(cost_cfg)
    profile_name = c.pop("profile", None)
    if profile_name is None:
        return c
    if profile_name == "jacktoy_c3plus":
        defaults = jacktoy_c3plus_cost_config()
        overlap = defaults.keys() & c.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"jacktoy C3+ profile values cannot be overridden: {names}")
        defaults.update(c)
        return defaults
    if profile_name != "push_anything":
        raise ValueError(f"unknown cost profile: {profile_name!r}")

    profile = push_anything_cost_profile(c.pop("num_objects", 1))
    defaults = {
        "push_anything_num_objects": profile.num_objects,
        "use_reference_q_vector": True,
        "w_Q": profile.w_Q,
        "w_Q_position": profile.w_Q_position,
        "q_vector_ee_pos": [0.01, 0.01, 0.01],
        "q_vector_obj_quat": [0.1, 0.1, 0.1, 0.1],
        "q_vector_obj_pos": [150.0, 150.0, 120.0],
        "q_vector_position_obj_quat": [0.0, 0.0, 0.0, 0.0],
        "q_vector_position_obj_pos": [200.0, 200.0, 120.0],
        "q_vector_ee_vel": [15.0, 15.0, 10.0],
        "q_vector_obj_ang_vel": [0.05, 0.05, 0.05],
        "q_vector_obj_lin_vel": [0.05, 0.05, 0.05],
        "w_R": profile.w_R,
        "r_vector": [0.01, 0.01, 1.0],
        "use_quaternion_dependent_cost": True,
        "q_quaternion_dependent_weight": profile.quaternion_weight,
        "q_quaternion_dependent_regularizer_fraction": profile.quaternion_regularizer_fraction,
        "w_terminal": profile.terminal_multiplier,
    }
    overlap = defaults.keys() & c.keys()
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Push Anything profile values cannot be overridden: {names}")
    defaults.update(c)
    return defaults
