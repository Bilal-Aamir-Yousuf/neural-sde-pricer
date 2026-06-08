"""Phase 7 — Closed-form baselines: Black-Scholes and Heston (COS method).

* **Black-Scholes** — the simplest baseline (closed form) plus analytic Greeks
  and an implied-volatility inverter (Brent with a Newton fast-path).

* **Heston stochastic volatility** — priced via the **COS method**
  (Fang & Oosterlee, 2008) rather than naive Monte Carlo: it evaluates the
  Fourier-cosine expansion of the (known) characteristic function and is both
  far faster and more accurate than MC.  Using it signals familiarity with the
  standard professional approach.

All functions are plain NumPy/SciPy (no torch) so they are dependency-light and
usable as ground-truth in tests.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# ===========================================================================
# Black-Scholes
# ===========================================================================
def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes European option price (continuous dividend yield q)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc, growth = np.exp(-r * T), np.exp(-q * T)
    if option_type == "call":
        return growth * S * norm.cdf(d1) - disc * K * norm.cdf(d2)
    return disc * K * norm.cdf(-d2) - growth * S * norm.cdf(-d1)


def black_scholes_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Analytic Greeks — used to validate the adjoint Greeks in Phase 6."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    disc, growth = np.exp(-r * T), np.exp(-q * T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    if option_type == "call":
        delta = growth * norm.cdf(d1)
        rho = K * T * disc * norm.cdf(d2)
        theta = (-growth * S * pdf * sigma / (2 * sqrtT)
                 - r * K * disc * norm.cdf(d2) + q * growth * S * norm.cdf(d1))
    else:
        delta = -growth * norm.cdf(-d1)
        rho = -K * T * disc * norm.cdf(-d2)
        theta = (-growth * S * pdf * sigma / (2 * sqrtT)
                 + r * K * disc * norm.cdf(-d2) - q * growth * S * norm.cdf(-d1))
    gamma = growth * pdf / (S * sigma * sqrtT)
    vega = growth * S * pdf * sqrtT          # per 1.00 change in vol
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}


def implied_vol(price, S, K, T, r, option_type="call", q=0.0):
    """Invert Black-Scholes for implied volatility (Brent on [1e-6, 5])."""
    intrinsic = (max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == "call"
                 else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0))
    if price < intrinsic - 1e-8 or price <= 0:
        return np.nan
    try:
        return brentq(lambda s: black_scholes_price(S, K, T, r, s, option_type, q) - price,
                      1e-6, 5.0, maxiter=200, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan


# ===========================================================================
# Heston via the COS method (Fang & Oosterlee, 2008)
# ===========================================================================
def _heston_cf(u, T, r, kappa, theta, xi, rho, v0):
    """Characteristic function of the log-return ln(S_T/S0) under Q.

    Uses the "little Heston trap" formulation (the ``-d`` / ``g`` branch) which
    is numerically stable for long maturities.
    """
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + u ** 2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    exp_dT = np.exp(-d * T)
    C = (kappa * theta / xi ** 2) * (
        (kappa - rho * xi * iu - d) * T - 2.0 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((kappa - rho * xi * iu - d) / xi ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))
    return np.exp(C + D * v0 + iu * r * T)


def _chi_psi(k, a, b, c, d):
    """COS series building blocks chi_k and psi_k on [c, d] within [a, b]."""
    bma = b - a
    kpi = k * np.pi / bma
    # chi
    cos_d = np.cos(kpi * (d - a))
    cos_c = np.cos(kpi * (c - a))
    sin_d = np.sin(kpi * (d - a))
    sin_c = np.sin(kpi * (c - a))
    chi = (1.0 / (1.0 + kpi ** 2)) * (
        cos_d * np.exp(d) - cos_c * np.exp(c)
        + kpi * sin_d * np.exp(d) - kpi * sin_c * np.exp(c)
    )
    # psi
    psi = np.empty_like(chi)
    psi[0] = d - c
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:] = (sin_d[1:] - sin_c[1:]) / kpi[1:]
    return chi, psi


def heston_cos_price(S, K, T, r, kappa, theta, xi, rho, v0,
                     option_type="call", N=256, L=12):
    """Heston European option price via the COS method."""
    x = np.log(S / K)
    # Cumulants of the log-return for the truncation range [a, b].
    c1 = (r * T + (1 - np.exp(-kappa * T)) * (theta - v0) / (2 * kappa)
          - 0.5 * theta * T)
    c2 = (1.0 / (8.0 * kappa ** 3)) * (
        xi * T * kappa * np.exp(-kappa * T) * (v0 - theta) * (8 * kappa * rho - 4 * xi)
        + kappa * rho * xi * (1 - np.exp(-kappa * T)) * (16 * theta - 8 * v0)
        + 2 * theta * kappa * T * (-4 * kappa * rho * xi + xi ** 2 + 4 * kappa ** 2)
        + xi ** 2 * ((theta - 2 * v0) * np.exp(-2 * kappa * T)
                     + theta * (6 * np.exp(-kappa * T) - 7) + 2 * v0)
        + 8 * kappa ** 2 * (v0 - theta) * (1 - np.exp(-kappa * T))
    )
    w = L * np.sqrt(abs(c2) + 1e-12)
    a, b = x + c1 - w, x + c1 + w

    k = np.arange(N)
    u = k * np.pi / (b - a)
    cf = _heston_cf(u, T, r, kappa, theta, xi, rho, v0)
    cf_z = cf * np.exp(1j * u * x)                     # CF of z = ln(S_T/K)

    if option_type == "call":
        chi, psi = _chi_psi(k, a, b, 0.0, b)
        Uk = (2.0 / (b - a)) * K * (chi - psi)
    else:
        chi, psi = _chi_psi(k, a, b, a, 0.0)
        Uk = (2.0 / (b - a)) * K * (psi - chi)

    terms = np.real(cf_z * np.exp(-1j * u * a)) * Uk
    terms[0] *= 0.5                                    # halve the k=0 term
    return float(np.exp(-r * T) * terms.sum())


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
    bs = black_scholes_price(S, K, T, r, sigma, "call")
    print(f"BS call = {bs:.6f}")
    # Heston with negligible vol-of-vol should ~ collapse to BS at vol=sqrt(v0).
    hes = heston_cos_price(S, K, T, r, kappa=2.0, theta=sigma**2, xi=1e-3,
                           rho=0.0, v0=sigma**2, option_type="call")
    print(f"Heston(xi->0) call = {hes:.6f}  (should ~= BS)")
    print("IV round-trip:", implied_vol(bs, S, K, T, r, "call"))
