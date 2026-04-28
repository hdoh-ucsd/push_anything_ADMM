"""Sampling-C3 outer controller — Venkatesh et al. RA-L 2025 (§IV-D port)."""

from control.sampling_c3.params import (
    ProgressMetric,
    ProgressParams,
    RepositionParams,
    RepositioningTrajectoryType,
    SamplingC3Params,
    SamplingParams,
    SamplingStrategy,
)
from control.sampling_c3.wrapper import SamplingC3MPC

__all__ = [
    "ProgressMetric",
    "ProgressParams",
    "RepositionParams",
    "RepositioningTrajectoryType",
    "SamplingC3MPC",
    "SamplingC3Params",
    "SamplingParams",
    "SamplingStrategy",
]
