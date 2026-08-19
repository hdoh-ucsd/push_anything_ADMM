"""Per-task ADMM solver scales (u_lambda / w_G) — jacktoy regression fix.

2026-08-16: L3 (`3202fbe`) globalized the anything-N1 solver scales
(u_lambda 20 -> 1000, w_G 0.01 -> 0.18) as C3Solver class constants. The
reference is per-demo: jacktoy/parameters/sampling_c3plus_options.yaml
pins u_lambda_list = uniform 4 (:192) and w_G = 0.03 (:73), while
anything sets 1000 / 0.18 (:114, :85). Under the globalized values the
jack's C3+ projection tiebreak (eta*sqrt(u_eta) > lambda*sqrt(u_lambda))
endorses phantom contact ~90% of the time and — with no final-QP boost
for jacktoy (correctly: the reference key is anything-only) — publishes
standoff plans: the 2026-08-16 contact-latch regression.

Fix shape: optional per-task keys `u_lambda` / `w_G` on the sampling-c3
yaml (SamplingC3Params), pushed onto the solver by the wrapper via
C3Solver.apply_task_solver_scales — same plumbing pattern as
final_augmented_cost_contact_scaling. Absent keys keep the anything-N1
defaults, so T/box/letter behavior is byte-unchanged. The PORT_U_LAMBDA /
PORT_W_G falsification env hooks keep highest precedence.
"""
import pytest

from control.admm_solver import C3Solver
from control.sampling_c3.params import SamplingC3Params


def _mk_solver(**kw):
    return C3Solver(n_x=19, n_u=3, rho=1.0, mode="c3plus", **kw)


def test_defaults_stay_anything_n1():
    # Guard: tasks whose yaml omits the keys (T/box/letters) must keep the
    # L3 anything-N1 values exactly.
    s = _mk_solver()
    assert s._u_lambda == 1000.0
    assert s._w_G == 0.18


def test_apply_task_solver_scales_sets_values():
    s = _mk_solver()
    s.apply_task_solver_scales(u_lambda=4.0, w_G=0.03)
    assert s._u_lambda == 4.0
    assert s._w_G == 0.03


def test_apply_task_solver_scales_none_keeps_defaults():
    s = _mk_solver()
    s.apply_task_solver_scales(u_lambda=None, w_G=None)
    assert s._u_lambda == 1000.0
    assert s._w_G == 0.18


def test_env_falsification_hook_beats_yaml(monkeypatch):
    # The PORT_* hooks exist for single-knob attribution runs; a per-task
    # yaml value must not silently clobber an explicit env override.
    monkeypatch.setenv("PORT_U_LAMBDA", "7.5")
    s = _mk_solver()
    assert s._u_lambda == 7.5
    s.apply_task_solver_scales(u_lambda=4.0, w_G=0.03)
    assert s._u_lambda == 7.5   # env wins
    assert s._w_G == 0.03       # non-overridden attr still applied


def test_params_from_dict_parses_scales():
    p = SamplingC3Params.from_dict({"u_lambda": 4.0, "w_G": 0.03})
    assert p.u_lambda == 4.0
    assert p.w_G == 0.03


def test_params_from_dict_absent_is_none():
    p = SamplingC3Params.from_dict({})
    assert p.u_lambda is None
    assert p.w_G is None


def test_kik_jack_yaml_carries_jacktoy_literals():
    p = SamplingC3Params.from_yaml("config/sampling_c3_kik_jack.yaml")
    assert p.u_lambda == 4.0
    assert p.w_G == 0.03


def test_anything_lineage_yamls_leave_scales_absent():
    for path in ("config/sampling_c3_kik_t.yaml",
                 "config/sampling_c3_kik.yaml"):
        p = SamplingC3Params.from_yaml(path)
        assert p.u_lambda is None, path
        assert p.w_G is None, path
