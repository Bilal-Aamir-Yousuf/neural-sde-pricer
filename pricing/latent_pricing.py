"""Extension — risk-neutral Monte Carlo pricing from the trained latent SDE.

Measure change (the explicit modeling choice)
---------------------------------------------
The latent model is fit under P.  For pricing, the change of measure for a
stochastic-volatility model is NOT unique (market price of volatility risk).
We make the standard practical choice (zero vol-risk premium, as in textbook
Heston implementations):

  * coordinate 0 (log-price): drift replaced by  r - 1/2 g0(z)^2  — the
    martingale condition on the discounted asset;  diffusion unchanged.
  * coordinate 1 (vol state):  keeps its PRIOR drift h(z)[1] and diffusion —
    lambda_vol = 0.

The level-match knob mirrors v1's ``vol_shift``: g0 is scaled by
``(1 + vol_shift)`` (the price-coordinate vol only — scaling the vol-of-vol
would change the smile shape, not the level), and the Q drift uses the
*shifted* g0, exactly as v1's Q drift uses its shifted sigma.

Time feature
------------
Training windows span ~0.25y, and "time since window start" carries no
economic meaning (windows are arbitrary calendar slices), so the dynamics are
treated as time-homogeneous: nets are evaluated at a frozen reference time
``T_REF`` (mid-window) instead of extrapolating the t-feature past its
training range for long maturities.

Why lambda_vol = 0 is the project default (and not just convenience): the
regime benchmark's target prices are bootstrapped from realized returns
re-centred to the risk-neutral drift — by construction they contain NO vol
risk premium, so lambda=0 is the measure that matches the evaluation target.
Calibrating lambda to market option chains (the desk approach) is the natural
next step but conflates premium with model misspecification on a single
snapshot, and would mismatch the premium-free benchmark.

Initial vol state z1(0)
-----------------------
``v0_mode='prior'``  — draw from the learned p(z0) vol coordinate
                       (unconditional pricing; used in the regime benchmark).
``v0_mode=<float>``  — pin z1(0) to a value.
``v0_mode=<Tensor>`` — (n, latent_dim) FILTERED samples of the current latent
                       state from ``filtered_state_samples`` (conditional
                       pricing: today's price reflects today's vol state).
                       Pricing mixes over them — the price is an expectation
                       over the filtering posterior, not a plug-in point.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DT as DT_DAY  # noqa: E402  (one trading day in years)
from config import DTYPE, RISK_FREE_RATE  # noqa: E402
from models.latent_sde import LatentSDE  # noqa: E402
from pricing.monte_carlo import discount_and_estimate, european_payoff  # noqa: E402

T_REF = 0.125   # frozen time feature (mid training-window)


def simulate_paths_latent(model: LatentSDE, S0, T, n_steps=80, n_paths=30000,
                          r=RISK_FREE_RATE, vol_shift=0.0, antithetic=True,
                          v0_mode="prior", generator=None):
    """Price paths under Q from the latent prior. Returns (n_paths, n_steps+1)."""
    dt = T / n_steps
    sq = math.sqrt(dt)
    if antithetic and n_paths % 2:
        n_paths += 1

    # z(0): price coordinate pinned to the observed spot (window-relative 0);
    # vol coordinate(s) from the learned prior, filtered samples, or a value.
    # With antithetic noise, initial vol states are duplicated across the +/-
    # halves so each antithetic pair shares its v0 (proper pairing).
    z = torch.zeros(n_paths, model.latent_dim, dtype=DTYPE)
    m = n_paths // 2 if antithetic else n_paths

    def _paired(x):
        return torch.cat([x, x], dim=0) if antithetic else x

    if isinstance(v0_mode, torch.Tensor):
        idx = torch.randint(v0_mode.shape[0], (m,), generator=generator)
        z[:, 1:] = _paired(v0_mode[idx, 1:].to(DTYPE))
    elif v0_mode == "prior":
        eps0 = torch.randn(m, model.latent_dim - 1, dtype=DTYPE,
                           generator=generator)
        z[:, 1:] = _paired(
            model.pz0_mean[1:] + torch.exp(model.pz0_logstd[1:]) * eps0
        )
    else:
        z[:, 1] = float(v0_mode)

    t_ref = torch.tensor(T_REF, dtype=DTYPE)
    logS = [torch.zeros(n_paths, dtype=DTYPE)]
    with torch.no_grad():
        for _ in range(n_steps):
            g = model.g(t_ref, z)
            h = model.h(t_ref, z)
            g0 = g[:, 0] * (1.0 + vol_shift)
            if antithetic:
                e = torch.randn(n_paths // 2, model.latent_dim, dtype=DTYPE,
                                generator=generator)
                eps = torch.cat([e, -e], dim=0)
            else:
                eps = torch.randn(n_paths, model.latent_dim, dtype=DTYPE,
                                  generator=generator)
            dz0 = (r - 0.5 * g0 ** 2) * dt + g0 * sq * eps[:, 0]
            dz1 = h[:, 1] * dt + g[:, 1] * sq * eps[:, 1]
            z = torch.stack([z[:, 0] + dz0, z[:, 1] + dz1], dim=1)
            logS.append(z[:, 0])
    S0t = torch.as_tensor(float(S0), dtype=DTYPE)
    return S0t * torch.exp(torch.stack(logS, dim=1))


def filtered_state_samples(model: LatentSDE, window_log_prices,
                           n_samples: int = 512, generator=None) -> torch.Tensor:
    """Samples of the CURRENT latent state given a trailing observed window.

    Pipeline (approach 2 in the module docstring):
      encode window (reverse GRU) -> sample q(z0 | window) -> integrate the
      POSTERIOR SDE (context-conditioned drift f) through the window -> the
      terminal states are draws from the smoothing posterior at the boundary,
      and smoothing at the final time IS the filtering posterior p(z_t|x_{0:t})
      (there is no data beyond t).

    Feed the result to ``simulate_paths_latent(..., v0_mode=samples)``: pricing
    then averages over the filtering posterior of the vol state.  The price
    coordinate of the samples is ignored downstream — the spot is observed and
    pinned exactly (the readout's observation noise is unpriced microstructure).

    ``window_log_prices``: 1-D log-price series (absolute or relative; it is
    made window-relative here, matching training).
    """
    x = torch.as_tensor(window_log_prices, dtype=DTYPE).reshape(-1, 1, 1)
    x = x - x[0]
    T = x.shape[0]
    ts = torch.linspace(0.0, (T - 1) * DT_DAY, T, dtype=DTYPE)
    sq = math.sqrt(DT_DAY)
    with torch.no_grad():
        ctx, summary = model.encode(x)
        qm, qls = model._qz0(summary)                      # (1, d)
        model.contextualize(ts, ctx.repeat(1, n_samples, 1))
        z = qm + torch.exp(qls) * torch.randn(
            n_samples, model.latent_dim, dtype=DTYPE, generator=generator)
        for i in range(T - 1):
            f = model.f(ts[i], z)
            g = model.g(ts[i], z)
            eps = torch.randn(n_samples, model.latent_dim, dtype=DTYPE,
                              generator=generator)
            z = z + f * DT_DAY + g * sq * eps
    return z


def price_european_latent(model, S0, K, T, r=RISK_FREE_RATE, option_type="call",
                          n_paths=30000, n_steps=80, vol_shift=0.0,
                          antithetic=True, v0_mode="prior", generator=None):
    S = simulate_paths_latent(model, S0, T, n_steps, n_paths, r, vol_shift,
                              antithetic, v0_mode, generator)
    payoff = european_payoff(S, K, option_type)
    return discount_and_estimate(payoff, r, T, antithetic)


def calibrate_latent_vol_shift(model, sigma_ann, S0, r=RISK_FREE_RATE):
    """vol_shift matching the latent model's ATM IV to a target vol level.

    Same protocol (and same single degree of freedom) as v1's
    ``calibrate_vol_shift``: linear estimate plus one Newton-style correction.
    """
    from pricing.baselines import implied_vol
    import numpy as np

    def atm_iv(vshift):
        res = price_european_latent(model, S0, S0, 0.25, r, "call",
                                    n_paths=30000, n_steps=60,
                                    vol_shift=vshift)
        return implied_vol(res.price, S0, S0, 0.25, r, "call")

    base_iv = atm_iv(0.0)
    if not np.isfinite(base_iv) or base_iv <= 0:
        return 0.0
    guess = sigma_ann / base_iv - 1.0
    iv_guess = atm_iv(guess)
    if np.isfinite(iv_guess) and iv_guess > 0:
        guess = (1 + guess) * (sigma_ann / iv_guess) - 1.0
    return float(guess)


if __name__ == "__main__":
    from models.train_latent import load_trained_latent

    model = load_trained_latent()
    res = price_european_latent(model, 100.0, 100.0, 0.25)
    # Martingale sanity: discounted terminal mean must ~equal the spot.
    S = simulate_paths_latent(model, 100.0, 0.25)
    fwd = float(S[:, -1].mean()) * math.exp(-RISK_FREE_RATE * 0.25)
    print(f"ATM call {res}, discounted E[S_T] = {fwd:.3f} (spot 100)")
