"""Operational-space controller (Khatib OSC + null-space posture).

Replaces the joint-PD executor in the sampling-C3 free-mode tracker.
See docs/osc_design.md for the design spec.

Two callers:
  - ``OperationalSpaceController`` — Drake ``LeafSystem`` with input
    ports (plant_state, cartesian_setpoint) and an actuation output
    port. For future Drake-diagram wiring.
  - ``OperationalSpaceTracker`` — synchronous tracker matching the
    ``compute_torque`` API of ``PiecewiseLinearTracker`` /
    ``RepositionIKTracker``. Used by ``SamplingC3MPC``'s free-mode
    path (Phase 3 wiring).
Both share the per-tick QP math in ``qp_builder.build_osc_qp``.
"""
from control.osc.operational_space_controller import OperationalSpaceController
from control.osc.osc_tracker import OperationalSpaceTracker

__all__ = ["OperationalSpaceController", "OperationalSpaceTracker"]
