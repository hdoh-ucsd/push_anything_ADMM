"""GPU-accelerated pieces of the C3+ ADMM solver (cuNRTO design basis).

Everything here is opt-in behind PORT_GPU_ADMM=1 and imported lazily, so an
unset gate never pays the torch import. See
docs/superpowers/plans/2026-08-20-gpu-admm.md.
"""
