"""CRISP: contact-implicit planning by sequential convex programming.

Port of Li, Han, Kang, Ma & Yang, "On the Surprising Robustness of Sequential
Convex Optimization for Contact-Implicit Motion Planning", arXiv:2502.01055v3.

This is an OFF-REFERENCE experimental arm: it is not part of the dairlib
sampling-C3 lineage the rest of the port conforms to. Nothing here is reached
unless a task explicitly selects it.
"""

from control.crisp.push_box import (
    ExecutionPlan,
    PushBoxParams,
    PushBoxProblem,
    min_terminal_weight,
    to_execution_plan,
)
from control.crisp.scp import CrispParams, CrispResult, CrispSolver, NlpProblem

__all__ = [
    "CrispParams",
    "CrispResult",
    "CrispSolver",
    "ExecutionPlan",
    "NlpProblem",
    "PushBoxParams",
    "PushBoxProblem",
    "min_terminal_weight",
    "to_execution_plan",
]
