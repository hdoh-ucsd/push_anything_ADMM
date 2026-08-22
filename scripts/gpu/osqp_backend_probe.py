"""Is a GPU (CUDA) algebra backend available to OSQP in this environment?

OSQP 1.x supports pluggable algebra backends, including `cuda` -- the
productionised descendant of the cuOSQP paper (Schubiger/Banjac/Lygeros).
Drake vendors its own OSQP and always uses the builtin CPU algebra, so this
asks the separate question: could we call a GPU OSQP directly?
"""
import inspect

import osqp

print("osqp version :", getattr(osqp, "__version__", "?"))
print("osqp file    :", osqp.__file__)

for name in ("algebras_available", "algebra_available", "default_algebra",
             "algebras", "constant"):
    if hasattr(osqp, name):
        attr = getattr(osqp, name)
        if callable(attr):
            try:
                print(f"  osqp.{name}() -> {attr()}")
            except Exception as exc:
                print(f"  osqp.{name}() raised {type(exc).__name__}: {exc}")
        else:
            print(f"  osqp.{name} = {attr}")

print("\npublic names:", [n for n in dir(osqp) if not n.startswith("_")])

try:
    print("\nOSQP.__init__:", inspect.signature(osqp.OSQP.__init__))
except Exception as exc:
    print("signature unavailable:", exc)

for alg in ("cuda", "mkl", "builtin"):
    try:
        m = osqp.OSQP(algebra=alg)
        print(f"  algebra={alg:8s} -> OK ({m})")
    except Exception as exc:
        print(f"  algebra={alg:8s} -> {type(exc).__name__}: "
              f"{str(exc)[:120]}")
