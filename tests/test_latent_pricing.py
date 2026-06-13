"""Martingale sanity for risk-neutral pricing off the latent SDE.

Under Q the discounted price must satisfy  E[e^{-rT} S_T] = S0.  Our scheme
makes this hold EXACTLY at the discrete level, not just in the dt->0 limit:
each Euler step in log-space is  z' = z + (r - g0^2/2) dt + g0 sqrt(dt) eps
with g0 measurable at the step start, so E[e^{z'} | z] = e^{z + r dt}
(lognormal mgf).  The only deviation left is Monte Carlo error — the tests
assert agreement within 5 MC standard errors (antithetic-pair aware).

The property must hold BY CONSTRUCTION for *any* network weights, so the core
tests use an untrained model (no checkpoint dependency); a final test repeats
the check on the trained checkpoint when it exists.
"""
import math

import pytest
import torch

import config  # noqa: F401  (sets the float64 default dtype)
from models.latent_sde import LatentSDE
from models.train_latent import CHECKPOINT_PATH, load_trained_latent
from pricing.latent_pricing import filtered_state_samples, simulate_paths_latent

S0, R, T, STEPS, NPATHS = 100.0, 0.03, 0.5, 60, 100_000


def assert_martingale(model, v0_mode, seed=1):
    g = torch.Generator().manual_seed(seed)
    S = simulate_paths_latent(model, S0, T, n_steps=STEPS, n_paths=NPATHS,
                              r=R, v0_mode=v0_mode, antithetic=True,
                              generator=g)
    disc = math.exp(-R * T) * S[:, -1]
    half = disc.shape[0] // 2
    pairs = 0.5 * (disc[:half] + disc[half:])      # antithetic-pair means
    mean = float(pairs.mean())
    se = float(pairs.std(unbiased=True) / math.sqrt(half))
    err = abs(mean - S0)
    print(f"  discounted E[S_T] = {mean:.4f}  (spot {S0}, MC stderr {se:.4f}, "
          f"|err| = {err:.4f} = {err / max(se, 1e-12):.2f} se)")
    assert err < 5 * se + 1e-6, (
        f"martingale violated: E[disc S_T]={mean:.4f} vs S0={S0} "
        f"(err {err:.4f} > 5 x stderr {se:.4f})"
    )


def test_martingale_untrained_prior_v0():
    torch.manual_seed(0)
    assert_martingale(LatentSDE(), v0_mode="prior")


def test_martingale_untrained_filtered_v0():
    torch.manual_seed(0)
    model = LatentSDE()
    g = torch.Generator().manual_seed(7)
    window = (torch.randn(64, generator=g) * 0.01).cumsum(0)   # fake log prices
    z_t = filtered_state_samples(model, window, n_samples=256, generator=g)
    assert z_t.shape == (256, model.latent_dim)
    assert torch.isfinite(z_t).all()
    assert_martingale(model, v0_mode=z_t)


def test_martingale_pinned_v0():
    torch.manual_seed(0)
    assert_martingale(LatentSDE(), v0_mode=0.3)


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(),
                    reason="no trained latent checkpoint yet")
def test_martingale_trained_checkpoint():
    model = load_trained_latent()
    assert_martingale(model, v0_mode="prior")
