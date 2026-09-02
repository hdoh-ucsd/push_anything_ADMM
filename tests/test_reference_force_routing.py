from types import SimpleNamespace

import numpy as np

from control.sampling_c3.sampling_based_c3_controller import SamplingC3Controller


def test_reference_force_route_does_not_gate_u_sol_on_zero_lambda(monkeypatch):
    """Reference cc:1855-1867 publishes u_sol independently of lambda_n."""
    monkeypatch.delenv("PORT_FORCE_ROUTING", raising=False)
    controller = SamplingC3Controller.__new__(SamplingC3Controller)
    controller.base_mpc = SimpleNamespace(
        use_ee_space=True,
        _last_u_seq=np.array([[1.25, -2.5, 0.75]]),
    )

    force = controller._derive_force_command(
        np.zeros(1), np.array([1.0, 0.0, 0.0]))

    np.testing.assert_allclose(force, [1.25, -2.5, 0.75])
