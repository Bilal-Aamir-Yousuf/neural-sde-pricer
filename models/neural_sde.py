"""Phase 2 — Neural SDE architecture.

A standard Ito SDE has the form

    dX_t = mu(X_t, t) dt + sigma(X_t, t) dW_t

Black-Scholes fixes mu = r*X and sigma = vol*X.  A *Neural* SDE replaces both
drift and diffusion with small neural networks learned from data.

Design choices (the important ones)
-----------------------------------
* **We model the log-price** ``Y = log(S)``.  This keeps prices positive for free
  and makes the learned dynamics numerically stable.  In log-space the networks
  output the drift/diffusion of ``Y`` directly.

* **Diffusion positivity** is guaranteed with a ``softplus`` head (plus a small
  floor): a diffusion coefficient can never be negative.

* **Physical (P) vs risk-neutral (Q) measure.**  The networks are fit to
  *historical* returns, i.e. under the physical measure P.  Option prices,
  however, are expectations under the *risk-neutral* measure Q.  By Girsanov's
  theorem the diffusion coefficient is invariant under an equivalent change of
  measure, so we **keep the learned diffusion** but **replace the drift** with
  the risk-neutral one when pricing.  In log-space, if dS/S = r dt + sigma dW,
  then by Ito  dY = (r - 1/2 sigma^2) dt + sigma dW.  This makes the discounted
  asset a martingale and the resulting prices arbitrage-free.  ``measure='P'``
  uses the learned drift (for realism checks / simulating real-world paths);
  ``measure='Q'`` uses the risk-neutral drift (for all pricing & Greeks).

This module implements the ``torchsde`` Ito-SDE interface: an ``nn.Module`` with
``noise_type``/``sde_type`` attributes and ``f`` (drift) / ``g`` (diffusion)
methods wrapping the two networks, so it plugs straight into ``torchsde.sdeint``
and ``torchsde.sdeint_adjoint``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RISK_FREE_RATE  # noqa: E402

_ACTIVATIONS = {"tanh": nn.Tanh, "elu": nn.ELU, "relu": nn.ReLU, "softplus": nn.Softplus}


class MLP(nn.Module):
    """Small fully-connected net: (log-price, time) -> scalar."""

    def __init__(self, in_dim=2, out_dim=1, hidden=64, n_layers=3, activation="tanh"):
        super().__init__()
        act = _ACTIVATIONS[activation]
        dims = [in_dim] + [hidden] * (n_layers - 1)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act()]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralSDE(nn.Module):
    """Neural SDE over the log-price, exposing the torchsde Ito interface.

    Parameters
    ----------
    hidden, n_layers, activation : architecture of both networks.
    """

    noise_type = "diagonal"   # one independent Brownian per state dim
    sde_type = "ito"

    def __init__(self, hidden=64, n_layers=3, activation="tanh"):
        super().__init__()
        self.drift_net = MLP(2, 1, hidden, n_layers, activation)
        self.diffusion_net = MLP(2, 1, hidden, n_layers, activation)

        # Input normalisation (set from training data via `set_normalization`).
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self.register_buffer("t_scale", torch.ones(1))

        # --- pricing configuration (mutated per-simulation, not parameters) ---
        self.measure = "P"                 # 'P' learned drift | 'Q' risk-neutral
        self.rate = float(RISK_FREE_RATE)  # may be set to a tensor for Rho
        self.vol_shift = 0.0               # parallel vol bump for Vega
        self._sigma_floor = 1e-3

    # ------------------------------------------------------------------ utils
    def set_normalization(self, y_mean, y_std, t_scale=1.0):
        self.y_mean.fill_(float(y_mean))
        self.y_std.fill_(float(max(y_std, 1e-6)))
        self.t_scale.fill_(float(max(t_scale, 1e-6)))

    def _features(self, t, y):
        """Build the (batch, 2) network input from time t and state y."""
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        yn = (y - self.y_mean) / self.y_std
        t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
        tn = (t_tensor / self.t_scale).expand(y.shape[0], 1)
        return torch.cat([yn, tn], dim=1)

    # ------------------------------------------------- drift / diffusion heads
    def diffusion(self, t, y):
        """Learned diffusion sigma(Y, t) > 0 (with optional parallel vol shift)."""
        raw = self.diffusion_net(self._features(t, y))
        sigma = F.softplus(raw) + self._sigma_floor
        # A parallel (relative) shift of the whole volatility level — the
        # natural generalisation of "Vega" for a model whose vol is a function,
        # not a scalar.  Applied consistently so the risk-neutral drift sees it
        # too.  Always multiplied (a no-op at vol_shift=0) so the shift tensor
        # stays in the autograd graph for adjoint/pathwise Vega.
        return sigma * (1.0 + self.vol_shift)

    def drift_physical(self, t, y):
        """Learned physical-measure drift mu(Y, t)."""
        return self.drift_net(self._features(t, y))

    # ----------------------------------------------- torchsde f / g interface
    def f(self, t, y):
        if self.measure == "Q":
            sigma = self.diffusion(t, y)
            rate = self.rate
            if not torch.is_tensor(rate):
                rate = torch.as_tensor(rate, dtype=y.dtype, device=y.device)
            return rate - 0.5 * sigma ** 2          # risk-neutral log-drift
        return self.drift_physical(t, y)            # physical-measure drift

    def g(self, t, y):
        return self.diffusion(t, y)

    # ------------------------------------------------------- pricing context
    def configure(self, measure=None, rate=None, vol_shift=None):
        """Set the pricing context. Only arguments that are passed are changed
        (so a previously-set vol_shift is *not* silently reset by a later
        ``configure(measure='Q', rate=r)`` call inside the MC engine).
        Returns self for chaining."""
        if measure is not None:
            self.measure = measure
        if rate is not None:
            self.rate = rate
        if vol_shift is not None:
            self.vol_shift = vol_shift
        return self


class GeometricBrownianSDE(nn.Module):
    """Constant-volatility log-price SDE (i.e. Black-Scholes dynamics).

    Same torchsde interface as ``NeuralSDE`` so it drops into the MC engine and
    the Greeks code — used as an *analytic ground truth*: prices/Greeks from this
    SDE must match the closed-form Black-Scholes results, which validates the
    whole simulation + autodiff pipeline.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, sigma=0.2):
        super().__init__()
        self.sigma = float(sigma)
        self.measure = "Q"
        self.rate = float(RISK_FREE_RATE)
        self.vol_shift = 0.0

    def diffusion(self, t, y):
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        sig = torch.full_like(y, self.sigma)
        return sig * (1.0 + self.vol_shift)

    def drift_physical(self, t, y):
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return torch.zeros_like(y)

    def f(self, t, y):
        sigma = self.diffusion(t, y)
        rate = self.rate
        if not torch.is_tensor(rate):
            rate = torch.as_tensor(rate, dtype=y.dtype if y.dim() else None)
        return rate - 0.5 * sigma ** 2          # risk-neutral log-drift

    def g(self, t, y):
        return self.diffusion(t, y)

    def configure(self, measure=None, rate=None, vol_shift=None):
        if measure is not None:
            self.measure = measure
        if rate is not None:
            self.rate = rate
        if vol_shift is not None:
            self.vol_shift = vol_shift
        return self


if __name__ == "__main__":
    # quick smoke test of shapes
    sde = NeuralSDE()
    y = torch.randn(5, 1) + 6.0
    t = torch.tensor(0.1)
    print("drift", sde.f(t, y).shape, "diffusion", sde.g(t, y).shape)
    sde.configure(measure="Q", rate=0.03)
    print("Q drift sample", sde.f(t, y).flatten()[:3])
    print("sigma > 0:", bool((sde.g(t, y) > 0).all()))
