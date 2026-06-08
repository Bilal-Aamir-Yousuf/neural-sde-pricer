"""Tests for the adjoint / pathwise Greeks (Phase 6).

Ground truth is closed-form Black-Scholes, reached by running the Greeks code on
the constant-volatility ``GeometricBrownianSDE``.  Note the model's "Vega" is
d price / d (vol_shift) and vol_shift scales sigma multiplicatively, so it
equals  BS_vega * sigma.
"""
import pytest

from config import set_seed
from models.neural_sde import GeometricBrownianSDE
from pricing.baselines import black_scholes_greeks, black_scholes_price
from pricing.greeks import (
    finite_difference_greeks, greeks_adjoint, greeks_pathwise,
)

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.03, 0.2
BS = black_scholes_greeks(S0, K, T, R, SIG, "call")
BS_PRICE = black_scholes_price(S0, K, T, R, SIG, "call")


def test_adjoint_delta_matches_black_scholes():
    set_seed(0)
    g = greeks_adjoint(GeometricBrownianSDE(SIG), S0, K, T, R, "call",
                       n_paths=20000, n_steps=100)
    assert g["price"] == pytest.approx(BS_PRICE, abs=0.15)
    assert g["delta"] == pytest.approx(BS["delta"], abs=0.02)


def test_adjoint_vega_and_rho_match_black_scholes():
    set_seed(0)
    g = greeks_adjoint(GeometricBrownianSDE(SIG), S0, K, T, R, "call",
                       n_paths=20000, n_steps=100)
    # Vega here is d price / d vol_shift = BS_vega * sigma.
    assert g["vega"] == pytest.approx(BS["vega"] * SIG, rel=0.05)
    assert g["rho"] == pytest.approx(BS["rho"], rel=0.05)


def test_pathwise_greeks_match_black_scholes():
    set_seed(0)
    g = greeks_pathwise(GeometricBrownianSDE(SIG), S0, K, T, R, "call",
                        n_paths=20000, n_steps=100)
    assert g["delta"] == pytest.approx(BS["delta"], abs=0.02)
    assert g["gamma"] == pytest.approx(BS["gamma"], abs=0.01)
    assert g["vega"] == pytest.approx(BS["vega"] * SIG, rel=0.06)
    assert g["rho"] == pytest.approx(BS["rho"], rel=0.06)
    assert g["theta"] == pytest.approx(BS["theta"], abs=0.5)


def test_pathwise_agrees_with_finite_difference():
    """Same Brownian increments => autograd and finite-diff must nearly coincide."""
    sde = GeometricBrownianSDE(SIG)
    pw = greeks_pathwise(sde, S0, K, T, R, "call", n_paths=20000, n_steps=50, seed=7)
    fd = finite_difference_greeks(sde, S0, K, T, R, "call",
                                  n_paths=20000, n_steps=50, seed=7)
    assert pw["delta"] == pytest.approx(fd["delta"], abs=2e-3)
    assert pw["vega"] == pytest.approx(fd["vega"], abs=2e-2)
    assert pw["rho"] == pytest.approx(fd["rho"], abs=2e-2)
    assert pw["theta"] == pytest.approx(fd["theta"], abs=2e-2)
