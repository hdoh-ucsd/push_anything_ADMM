"""Task 2: the GPU backend must be strictly OPTIONAL.

Three properties, each of which has bitten real projects:

1. Importing the active solver path must not drag in a CUDA framework.
   A stray top-level `import torch` in a solver module costs seconds of
   startup and megabytes of VRAM on every CPU-only run.
2. `control.gpu.cupy_primitives` must import cleanly when CuPy is absent and
   fall back to NumPy, so a machine without a GPU can still run the tests.
3. No GPU module may be reachable from the default solver path at all.

These run on CPU-only machines by design -- they are the tests that prove the
GPU work is optional.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def _run(code: str):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=600)


def test_active_solver_path_imports_no_cuda_framework():
    """Importing the solver must pull neither torch nor cupy."""
    r = _run("""
        import sys
        import control.admm_solver          # noqa: F401
        import control.solver_api           # noqa: F401
        import control.sampling_c3.inner_solve  # noqa: F401
        pulled = [m for m in ("torch", "cupy") if m in sys.modules]
        print("PULLED:" + ",".join(pulled))
        print("GPUPKG:" + str("control.gpu" in sys.modules))
    """)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "PULLED:\n" in r.stdout or "PULLED:" in r.stdout
    pulled = [l for l in r.stdout.splitlines() if l.startswith("PULLED:")][0]
    assert pulled == "PULLED:", (
        f"the active solver path imported a CUDA framework: {pulled}")
    gpupkg = [l for l in r.stdout.splitlines() if l.startswith("GPUPKG:")][0]
    assert gpupkg == "GPUPKG:False", "control.gpu must not be auto-imported"


def test_cupy_primitives_import_and_work_without_cupy():
    """With CuPy unavailable the module must still import and compute on
    NumPy arrays -- an optional backend, not a hard dependency."""
    r = _run("""
        import sys, numpy as np

        class Block:
            def find_module(self, name, path=None):
                return self if name == "cupy" or name.startswith("cupy.") else None
            def load_module(self, name):
                raise ImportError("cupy blocked")

        sys.meta_path.insert(0, Block())
        for m in [k for k in list(sys.modules) if k.startswith("cupy")]:
            del sys.modules[m]

        import control.gpu.cupy_primitives as P
        assert P.HAVE_CUPY is False, "block failed; test is not exercising the path"

        lam = np.array([1.0, -1.0, 2.0, -1.0])
        eta = np.array([2.0,  1.0, 1.0, -1.0])
        dl, de = P.project_C3Plus_batch(lam, eta, 1.0, 1.0)
        assert np.array_equal(dl, [0.0, 0.0, 2.0, 0.0]), dl
        assert np.array_equal(de, [2.0, 1.0, 0.0, 0.0]), de
        assert P.candidate_argmin(np.array([np.nan, 5.0, 2.0])) == 2
        assert isinstance(P.to_device([1.0]), np.ndarray)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "OK" in r.stdout


def test_gpu_primitives_take_plain_arrays_only():
    """No PyDrake object may cross into a GPU primitive. The signatures take
    arrays; this pins that they neither require nor accept solver objects."""
    from control.gpu import cupy_primitives as P
    lam = np.zeros((2, 3, 4))
    eta = np.zeros((2, 3, 4))
    dl, de = P.project_C3Plus_batch(lam, eta)
    assert isinstance(dl, np.ndarray) and isinstance(de, np.ndarray)
    with pytest.raises((AttributeError, TypeError)):
        P.project_C3Plus_batch(object(), object())
