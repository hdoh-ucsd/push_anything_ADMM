"""
Unit tests for the C3+ componentwise complementarity projection
(Bui 2026 ICRA "Push Anything" §IV-B.2, eq. 12).

Per scalar pair (λ°, η°), with u_λ, u_η > 0:
    (δ_λ, δ_η)* =
      (0,  η°)   if η° ≥ 0  AND  η° ≥ √(u_λ/u_η)·λ°       (case 1: η wins)
      (λ°, 0 )   if λ° ≥ 0  AND  η° <  √(u_λ/u_η)·λ°       (case 2: λ wins)
      (0,  0 )   otherwise                                  (case 3: apex)

The cases must obey the post-projection complementarity δ_λ · δ_η = 0
and non-negativity δ_λ ≥ 0, δ_η ≥ 0.
"""
import numpy as np
import pytest


# Inline the projection so the test runs without Drake (mirrors test_projection.py)
def project_eq12(lam: np.ndarray, eta: np.ndarray,
                 u_lambda: float = 1.0, u_eta: float = 1.0):
    sqrt_ratio = float(np.sqrt(u_lambda / u_eta))
    cond1 = (eta >= 0.0) & (eta >= sqrt_ratio * lam)
    cond2 = (lam >= 0.0) & (eta <  sqrt_ratio * lam)
    delta_lam = np.where(cond2, lam, 0.0)
    delta_eta = np.where(cond1, eta, 0.0)
    return delta_lam, delta_eta


# -----------------------------------------------------------------------
# Case-by-case scalar tests with u_λ = u_η = 1 (sqrt_ratio = 1)
# -----------------------------------------------------------------------

def test_case1_eta_wins_when_eta_dominant():
    """η° = 5 ≥ λ° = 1 ≥ 0  ⇒ (0, 5)"""
    dl, de = project_eq12(np.array([1.0]), np.array([5.0]))
    assert dl[0] == 0.0
    assert de[0] == 5.0


def test_case2_lambda_wins_when_lambda_dominant():
    """λ° = 5 > η° = 1, both ≥ 0  ⇒ (5, 0)"""
    dl, de = project_eq12(np.array([5.0]), np.array([1.0]))
    assert dl[0] == 5.0
    assert de[0] == 0.0


def test_case3_apex_when_both_negative():
    """λ° = -1, η° = -1  ⇒ (0, 0). Neither cond1 (η°<0) nor cond2 (λ°<0) holds."""
    dl, de = project_eq12(np.array([-1.0]), np.array([-1.0]))
    assert dl[0] == 0.0
    assert de[0] == 0.0


def test_case3_apex_when_lambda_negative_eta_negative():
    """λ°=-3, η°=-2: cond1 fails (η<0), cond2 fails (λ<0). Apex."""
    dl, de = project_eq12(np.array([-3.0]), np.array([-2.0]))
    assert dl[0] == 0.0
    assert de[0] == 0.0


def test_origin_yields_origin():
    """(0, 0) input — case 1 fires (η°=0 ≥ 0, η° ≥ 1·0). Returns (0, 0)."""
    dl, de = project_eq12(np.array([0.0]), np.array([0.0]))
    assert dl[0] == 0.0
    assert de[0] == 0.0


def test_lambda_zero_eta_positive_picks_eta():
    """λ°=0, η°=2  ⇒ case 1: (0, 2). λ wins requires λ° > 0 strictly here."""
    dl, de = project_eq12(np.array([0.0]), np.array([2.0]))
    assert dl[0] == 0.0
    assert de[0] == 2.0


def test_eta_zero_lambda_positive_picks_lambda():
    """η°=0, λ°=2: cond1 needs η° ≥ √(1)·2 = 2. Fails. cond2: λ°≥0 (true), η°<2 (true). λ wins."""
    dl, de = project_eq12(np.array([2.0]), np.array([0.0]))
    assert dl[0] == 2.0
    assert de[0] == 0.0


# -----------------------------------------------------------------------
# Vectorized: multiple components in one call
# -----------------------------------------------------------------------

def test_vectorized_mixed_cases():
    lam = np.array([ 5.0, 1.0, -1.0, 0.5, -0.2])
    eta = np.array([ 1.0, 5.0, -1.0, 0.0,  3.0])
    dl, de = project_eq12(lam, eta)
    # Component 0: λ=5, η=1: cond1 needs η≥5 (no), cond2: λ≥0 (yes), η<5 (yes). λ wins.
    # Component 1: λ=1, η=5: η≥0 (yes), η≥1 (yes). η wins.
    # Component 2: λ=-1, η=-1: case 3 (both negative).
    # Component 3: λ=0.5, η=0: cond2: λ≥0 (yes), η<0.5 (yes). λ wins.
    # Component 4: λ=-0.2, η=3: cond1: η≥0 and η≥-0.2. η wins.
    expected_lam = np.array([5.0, 0.0, 0.0, 0.5, 0.0])
    expected_eta = np.array([0.0, 5.0, 0.0, 0.0, 3.0])
    assert np.allclose(dl, expected_lam)
    assert np.allclose(de, expected_eta)


# -----------------------------------------------------------------------
# Output invariants — must hold for all inputs
# -----------------------------------------------------------------------

def test_output_complementarity_dot_zero():
    """δ_λ · δ_η = 0 always (one of them is always zero by construction)."""
    rng = np.random.default_rng(42)
    lam = rng.uniform(-10, 10, size=200)
    eta = rng.uniform(-10, 10, size=200)
    dl, de = project_eq12(lam, eta)
    assert np.all(dl * de == 0.0)


def test_output_nonnegative():
    """δ_λ ≥ 0 and δ_η ≥ 0 for any input."""
    rng = np.random.default_rng(42)
    lam = rng.uniform(-10, 10, size=200)
    eta = rng.uniform(-10, 10, size=200)
    dl, de = project_eq12(lam, eta)
    assert np.all(dl >= 0.0)
    assert np.all(de >= 0.0)


# -----------------------------------------------------------------------
# Non-default G weights — sqrt_ratio scales the threshold
# -----------------------------------------------------------------------

def test_unequal_weights_changes_threshold():
    """u_λ=4, u_η=1 ⇒ √(u_λ/u_η)=2. λ°=2, η°=3: case 2 (η<2·2=4)."""
    dl, de = project_eq12(np.array([2.0]), np.array([3.0]),
                          u_lambda=4.0, u_eta=1.0)
    assert dl[0] == 2.0
    assert de[0] == 0.0


def test_unequal_weights_weights_other_direction():
    """u_λ=1, u_η=4 ⇒ √=0.5. λ°=2, η°=3: case 1 (η°≥0 and η°≥0.5·2=1)."""
    dl, de = project_eq12(np.array([2.0]), np.array([3.0]),
                          u_lambda=1.0, u_eta=4.0)
    assert dl[0] == 0.0
    assert de[0] == 3.0


# -----------------------------------------------------------------------
# Optional: importable from admm_solver itself (round-trip via Drake-free import)
# -----------------------------------------------------------------------

def test_module_import_matches_inline():
    """The exposed project_C3Plus_eq12 helper should agree with the
    inline reimplementation. Skips cleanly if pydrake is not available."""
    try:
        from control.admm_solver import project_C3Plus_eq12
    except Exception:
        pytest.skip("control.admm_solver requires pydrake; skipping")

    rng = np.random.default_rng(7)
    lam = rng.uniform(-3, 3, size=50)
    eta = rng.uniform(-3, 3, size=50)
    dl_inline, de_inline = project_eq12(lam, eta)
    dl_module, de_module = project_C3Plus_eq12(lam, eta)
    assert np.allclose(dl_inline, dl_module)
    assert np.allclose(de_inline, de_module)
