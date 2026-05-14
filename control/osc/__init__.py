"""Operational-space controller (Khatib OSC + null-space posture).

Replaces the joint-PD executor in the sampling-C3 free-mode tracker.
See docs/osc_design.md for the design spec.
"""
from control.osc.operational_space_controller import OperationalSpaceController

__all__ = ["OperationalSpaceController"]
