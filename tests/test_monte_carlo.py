"""Tests for the Monte Carlo pricing engine (Phase 4).

The key tests a reviewer pokes at first:
* the engine reproduces closed-form Black-Scholes (validates simulation),
* antithetic variates reduce variance,
* pricing error shrinks ~ 1/sqrt(N) (the defining MC signature),
* exotic payoffs have the right economic ordering.
"""
import numpy as np
import pytest
import torch

from config import set_seed
from models.neural_sde import GeometricBrownianSDE
from pricing.baselines import black_scholes_price
from pricing.monte_carlo import (
    convergence_study, price_option, simulate_paths, simulate_paths_torchsde,
)

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.03, 0.2


def test_mc_matches_black_scholes():
    set_seed(0)
    sde = GeometricBrownianSDE(sigma=SIG)
    res = price_option(sde, S0, K, T, R, "call", "european",
                       n_paths=200000, n_steps=100, antithetic=True)
    bs = black_scholes_price(S0, K, T, R, SIG, "call")
    # Agree within ~3 standard errors.
    assert abs(res.price - bs) < 3 * res.stderr + 0.05


def test_torchsde_integrator_agrees():
    """The torchsde.sdeint path and the hand-rolled engine must agree."""
    set_seed(0)
    sde = GeometricBrownianSDE(sigma=SIG)
    S = simulate_paths_torchsde(sde, S0, T, n_steps=100, n_paths=40000, r=R)
    disc = np.exp(-R * T)
    price = disc * torch.clamp(S[:, -1] - K, min=0).mean().item()
    bs = black_scholes_price(S0, K, T, R, SIG, "call")
    assert abs(price - bs) < 0.2


def test_antithetic_reduces_variance():
    set_seed(0)
    sde = GeometricBrownianSDE(sigma=SIG)
    plain = price_option(sde, S0, K, T, R, "call", "european",
                         n_paths=20000, n_steps=50, antithetic=False)
    set_seed(0)
    anti = price_option(sde, S0, K, T, R, "call", "european",
                        n_paths=20000, n_steps=50, antithetic=True)
    assert anti.stderr < plain.stderr


def test_convergence_rate_half():
    set_seed(0)
    sde = GeometricBrownianSDE(sigma=SIG)
    out = convergence_study(sde, S0, K, T, R, "call",
                            path_counts=(1000, 2000, 4000, 8000, 16000, 32000),
                            n_steps=50, plot=False)
    # slope of log(SE) vs log(N) should be close to -0.5
    assert -0.62 < out["slope"] < -0.38


def test_exotic_ordering():
    """Asian < European < Lookback (floating); barrier knock-out < European."""
    set_seed(0)
    sde = GeometricBrownianSDE(sigma=SIG)
    kw = dict(n_paths=60000, n_steps=100, antithetic=True)
    euro = price_option(sde, S0, K, T, R, "call", "european", **kw).price
    asian = price_option(sde, S0, K, T, R, "call", "asian", **kw).price
    look = price_option(sde, S0, K, T, R, "call", "lookback",
                        strike="floating", **kw).price
    barr = price_option(sde, S0, K, T, R, "call", "barrier",
                        barrier=130.0, barrier_type="up-and-out", **kw).price
    assert asian < euro < look
    assert barr < euro


def test_barrier_in_out_parity():
    """Knock-in + knock-out = vanilla. This is a path-by-path identity, so on
    the SAME simulated paths it must hold to floating-point precision."""
    sde = GeometricBrownianSDE(sigma=SIG)
    kw = dict(n_paths=60000, n_steps=100, antithetic=True)
    set_seed(1)
    euro = price_option(sde, S0, K, T, R, "call", "european", **kw).price
    set_seed(1)
    ki = price_option(sde, S0, K, T, R, "call", "barrier",
                      barrier=130.0, barrier_type="up-and-in", **kw).price
    set_seed(1)
    ko = price_option(sde, S0, K, T, R, "call", "barrier",
                      barrier=130.0, barrier_type="up-and-out", **kw).price
    assert ki + ko == pytest.approx(euro, abs=1e-6)
