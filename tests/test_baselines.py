"""Tests for the closed-form baselines (Black-Scholes & Heston COS)."""
import numpy as np
import pytest

from pricing.baselines import (
    black_scholes_greeks, black_scholes_price, heston_cos_price, implied_vol,
)


def test_put_call_parity():
    S, K, T, r, sig = 100, 105, 0.75, 0.03, 0.25
    c = black_scholes_price(S, K, T, r, sig, "call")
    p = black_scholes_price(S, K, T, r, sig, "put")
    # c - p = S - K e^{-rT}
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)


def test_implied_vol_roundtrip():
    S, K, T, r, sig = 100, 100, 1.0, 0.02, 0.3
    price = black_scholes_price(S, K, T, r, sig, "call")
    assert implied_vol(price, S, K, T, r, "call") == pytest.approx(sig, abs=1e-6)


def test_bs_greeks_match_finite_difference():
    S, K, T, r, sig = 100, 100, 1.0, 0.03, 0.2
    g = black_scholes_greeks(S, K, T, r, sig, "call")
    h = 1e-4
    fd_delta = (black_scholes_price(S + h, K, T, r, sig, "call")
                - black_scholes_price(S - h, K, T, r, sig, "call")) / (2 * h)
    fd_vega = (black_scholes_price(S, K, T, r, sig + h, "call")
               - black_scholes_price(S, K, T, r, sig - h, "call")) / (2 * h)
    assert g["delta"] == pytest.approx(fd_delta, abs=1e-4)
    assert g["vega"] == pytest.approx(fd_vega, abs=1e-2)


def test_heston_reduces_to_black_scholes():
    """With negligible vol-of-vol and v0=theta=sigma^2, Heston ~ Black-Scholes."""
    S, K, T, r, sig = 100, 100, 1.0, 0.03, 0.2
    bs = black_scholes_price(S, K, T, r, sig, "call")
    hes = heston_cos_price(S, K, T, r, kappa=2.0, theta=sig**2, xi=1e-3,
                           rho=0.0, v0=sig**2, option_type="call")
    assert hes == pytest.approx(bs, abs=2e-3)


def test_heston_put_call_parity():
    S, K, T, r = 100, 110, 1.0, 0.03
    p = dict(kappa=2.0, theta=0.04, xi=0.5, rho=-0.6, v0=0.04)
    c = heston_cos_price(S, K, T, r, **p, option_type="call")
    put = heston_cos_price(S, K, T, r, **p, option_type="put")
    assert c - put == pytest.approx(S - K * np.exp(-r * T), abs=5e-2)
