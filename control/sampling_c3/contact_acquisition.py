"""Measured-contact lifecycle for the C3 outer loop.

This is a port robustness extension, not an original Push Anything mode.
It keeps physical approach time separate from task-progress accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


def resolve_acquisition_timeout(configured, waypoint_height, target_height,
                                speed, dt):
    """Use one nominal PWL descent, rounded up to a planner interval.

    Existing box timing: (0.135 - 0.050) / 0.18 = 0.4722 s, rounded
    up at 0.075 s/planner update to 0.525 s. No task-progress window
    or new speed/geometry tuning enters this calculation.
    """
    if configured is not None:
        timeout = float(configured)
    else:
        if not math.isfinite(float(speed)) or speed <= 0:
            raise ValueError('Contact acquisition requires positive PWL speed')
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError('Contact acquisition requires positive planner dt')
        descent = max(0.0, float(waypoint_height) - float(target_height)) / speed
        timeout = max(1, math.ceil(descent / dt)) * dt
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('contact_acquisition_timeout_s must be finite and positive')
    return timeout


def measured_contact_force(plant, context, ee_ids, obj_ids):
    """Largest actual EE-object contact-force norm, in N.

    Uses the same geometry-ID filtering as main.py's F1K contact monitor.
    Contact topology/predicted impulses never participate. API failures are
    propagated rather than silently being converted into failed approaches.
    """
    def matches(a, b):
        return ((a in ee_ids and b in obj_ids)
                or (b in ee_ids and a in obj_ids))

    contacts = plant.get_contact_results_output_port().Eval(context)
    forces = [0.0]
    for i in range(contacts.num_point_pair_contacts()):
        info = contacts.point_pair_contact_info(i)
        pair = info.point_pair()
        if matches(pair.id_A, pair.id_B):
            forces.append(float(np.linalg.norm(info.contact_force())))
    for i in range(contacts.num_hydroelastic_contacts()):
        info = contacts.hydroelastic_contact_info(i)
        surface = info.contact_surface()
        if matches(surface.id_M(), surface.id_N()):
            forces.append(float(np.linalg.norm(info.F_Ac_W().translational())))
    if not all(math.isfinite(value) for value in forces):
        raise ValueError('Nonfinite measured EE-object contact force')
    return max(forces)


@dataclass
class ContactAcquisition:
    timeout_s: float
    # Existing main.py F1K contact-count precision (1 micro-newton).
    force_threshold: float = 1e-6
    active: bool = False
    contact_acquired: bool = False
    acquisition_failure: bool = False
    entry_time: Optional[float] = None
    acquired_time: Optional[float] = None
    timeout_time: Optional[float] = None
    failure_time: Optional[float] = None
    entry_object_pose: Optional[np.ndarray] = None
    entry_ee_position: Optional[np.ndarray] = None
    target_world: Optional[np.ndarray] = None
    target_body_xy: Optional[np.ndarray] = None

    def __post_init__(self):
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError('Contact acquisition timeout must be finite and positive')
        if not math.isfinite(self.force_threshold) or self.force_threshold < 0:
            raise ValueError('Contact force threshold must be finite and nonnegative')

    @property
    def phase(self):
        if not self.active:
            return 'INACTIVE'
        if self.acquisition_failure:
            return 'ACQUISITION_TIMEOUT'
        return 'TRACKING_PROGRESS' if self.contact_acquired else 'ACQUIRING_CONTACT'

    def begin(self, sim_time, object_pose, ee_position, target_world,
              target_body_xy, force_norm=0.0):
        self.active = True
        self.contact_acquired = False
        self.acquisition_failure = False
        self.entry_time = float(sim_time)
        self.timeout_time = self.entry_time + self.timeout_s
        self.acquired_time = self.failure_time = None
        self.entry_object_pose = np.asarray(object_pose, dtype=float).copy()
        self.entry_ee_position = np.asarray(ee_position, dtype=float).copy()
        self.target_world = np.asarray(target_world, dtype=float).copy()
        self.target_body_xy = np.asarray(target_body_xy, dtype=float).copy()
        self.observe(sim_time, force_norm)

    def observe(self, sim_time, force_norm):
        """Latch first real contact; contact at the deadline wins the tie."""
        if (not self.active or self.contact_acquired or self.acquisition_failure
                or float(sim_time) > self.timeout_time + 1e-9):
            return False
        if float(force_norm) > self.force_threshold:
            self.contact_acquired = True
            self.acquired_time = float(sim_time)
            return True
        return False

    def expired(self, sim_time):
        return (self.active and not self.contact_acquired
                and float(sim_time) >= self.timeout_time - 1e-9)

    def mark_failed(self, sim_time):
        if not self.expired(sim_time):
            raise ValueError('Only a timed-out unacquired approach can be marked failed')
        self.acquisition_failure = True
        self.failure_time = float(sim_time)

    def end(self):
        """Keep the episode snapshot available to passive exit logging."""
        self.active = False
