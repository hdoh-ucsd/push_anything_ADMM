"""Which LCSFormulator / C3Solver constructor args does the worker-kit
clone FAIL to forward? Any such arg is a silent parent/clone divergence.
"""
import inspect

from control.admm_solver import C3Solver
from control.lcs_formulator import LCSFormulator

# Exactly what _lazy_init_worker_kits passes today.
FORMULATOR_PASSED = {"plant", "mu", "obj_body", "plant_ad", "context_ad",
                     "object_shape", "mu_per_pair_type"}
SOLVER_PASSED = {"n_x", "n_u", "rho", "mode", "math_diag",
                 "penalize_input_change"}

for cls, passed in ((LCSFormulator, FORMULATOR_PASSED),
                    (C3Solver, SOLVER_PASSED)):
    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters if p != "self"]
    missing = [p for p in params if p not in passed]
    print(f"=== {cls.__name__}: {len(params)} ctor args, "
          f"clone forwards {len(passed)}, MISSING {len(missing)}")
    for p in missing:
        print(f"      {p:38s} default={sig.parameters[p].default!r}")
    print()
