"""Phase 6 — Greeks via adjoint sensitivity.

Why this matters
----------------
Computing Greeks by finite differences (bump-and-reprice) requires re-running
the entire Monte Carlo simulation once per parameter, and forward-mode autodiff
scales linearly with the number of parameters.  The **adjoint sensitivity
method** (Li et al., 2020) computes gradients with respect to *all* inputs in a
single backward pass with **constant memory** in the simulation length.

Because the whole pricing pipeline (SDE integration -> path simulation ->
payoff -> discounting) is differentiable PyTorch, the option price is a
differentiable function of the spot, rate, vol level and maturity, and the
Greeks fall out as gradients:

    Delta = d price / d S0        (one .backward())
    Gamma = d^2 price / d S0^2    (autograd.grad with create_graph=True)
    Vega  = d price / d (vol level shift)
    Rho   = d price / d r
    Theta = - d price / d T

Three independent implementations are provided and cross-checked:

* ``greeks_adjoint``  — uses ``torchsde.sdeint_adjoint`` (the constant-memory
  adjoint).  Delivers Delta, Vega, Rho.
* ``greeks_pathwise`` — autograd through the full (unrolled) differentiable MC
  engine.  Supports the second-order Gamma (the adjoint method does not support
  double-backprop) and the maturity-reparametrised Theta.
* ``finite_difference_greeks`` — bump-and-reprice with **common random numbers**
  (the same Brownian increments for every bump), the classical ground truth.

Validation target: adjoint/autograd Greeks should agree with finite differences
to within Monte Carlo noise.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DTYPE, RISK_FREE_RATE  # noqa: E402


# ===========================================================================
# Differentiable simulators
# ===========================================================================
def _simulate_terminal_rescaled(sde, S0, T, r, vol_shift, n_paths, n_steps, Z):
    """Terminal log-price via Euler-Maruyama on a rescaled unit-time interval.

    Time is reparametrised to tau in [0, 1] so the *maturity T enters as a
    differentiable scalar* (drift scaled by T, diffusion by sqrt(T)) — this is
    what makes a clean autograd Theta possible.  ``Z`` are fixed standard-normal
    increments (common random numbers).
    """
    sde.configure(measure="Q", rate=r, vol_shift=vol_shift)
    dtau = 1.0 / n_steps
    sqrt_dtau = math.sqrt(dtau)
    Y = torch.log(S0).reshape(1).expand(n_paths).clone()
    for n in range(n_steps):
        t_real = (n * dtau) * T            # real time (feature); ~unused (time-homog.)
        yv = Y.unsqueeze(-1)
        mu = sde.f(t_real, yv).squeeze(-1)
        sig = sde.g(t_real, yv).squeeze(-1)
        dW = sqrt_dtau * Z[:, n]
        Y = Y + mu * T * dtau + sig * torch.sqrt(T) * dW
    return Y


def _payoff(ST, K, option_type, beta=None):
    """European payoff; if ``beta`` given, a softplus-smoothed version (for the
    second-order Gamma, whose exact payoff has a non-differentiable kink)."""
    x = (ST - K) if option_type == "call" else (K - ST)
    if beta is None:
        return torch.clamp(x, min=0.0)
    return torch.nn.functional.softplus(beta * x) / beta


def _common_normals(n_paths, n_steps, seed):
    """Antithetic standard-normal increments used as common random numbers."""
    g = torch.Generator().manual_seed(seed)
    if n_paths % 2:
        n_paths += 1
    half = n_paths // 2
    Zh = torch.randn(half, n_steps, dtype=DTYPE, generator=g)
    return torch.cat([Zh, -Zh], dim=0)


# ===========================================================================
# 1) Pathwise Greeks via autograd (full graph; gives Gamma + Theta)
# ===========================================================================
def greeks_pathwise(sde, S0, K, T, r=RISK_FREE_RATE, option_type="call",
                    n_paths=20000, n_steps=50, seed=0, gamma_beta=2.0):
    Z = _common_normals(n_paths, n_steps, seed)
    n_paths = Z.shape[0]
    S0t = torch.tensor(float(S0), dtype=DTYPE, requires_grad=True)
    rt = torch.tensor(float(r), dtype=DTYPE, requires_grad=True)
    vt = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    Tt = torch.tensor(float(T), dtype=DTYPE, requires_grad=True)

    YT = _simulate_terminal_rescaled(sde, S0t, Tt, rt, vt, n_paths, n_steps, Z)
    ST = torch.exp(YT)
    disc = torch.exp(-rt * Tt)
    price = disc * _payoff(ST, K, option_type).mean()

    delta = torch.autograd.grad(price, S0t, create_graph=True, retain_graph=True)[0]
    vega = torch.autograd.grad(price, vt, retain_graph=True)[0]
    rho = torch.autograd.grad(price, rt, retain_graph=True)[0]
    theta = -torch.autograd.grad(price, Tt, retain_graph=True)[0]

    # Gamma from a smoothed payoff (exact payoff's 2nd derivative is a Dirac).
    price_s = disc.detach() * _payoff(ST, K, option_type, beta=gamma_beta).mean()
    d_s = torch.autograd.grad(price_s, S0t, create_graph=True, retain_graph=True)[0]
    gamma = torch.autograd.grad(d_s, S0t)[0]

    out = {"price": float(price.detach()), "delta": float(delta.detach()),
           "gamma": float(gamma.detach()), "vega": float(vega.detach()),
           "rho": float(rho.detach()), "theta": float(theta.detach())}
    sde.configure(rate=float(r), vol_shift=0.0)        # clear leaked grad tensors
    return out


# ===========================================================================
# 2) Adjoint Greeks via torchsde.sdeint_adjoint (constant memory)
# ===========================================================================
def greeks_adjoint(sde, S0, K, T, r=RISK_FREE_RATE, option_type="call",
                   n_paths=20000, n_steps=50, seed=0):
    """Delta, Vega, Rho from a single adjoint backward pass."""
    import torchsde

    sde.configure(measure="Q", rate=None, vol_shift=0.0)
    S0t = torch.tensor(float(S0), dtype=DTYPE, requires_grad=True)
    rt = torch.tensor(float(r), dtype=DTYPE, requires_grad=True)
    vt = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    sde.rate, sde.vol_shift = rt, vt          # tensors that we differentiate

    bm = torchsde.BrownianInterval(t0=0.0, t1=float(T), size=(n_paths, 1),
                                   dtype=DTYPE, entropy=seed)
    ts = torch.tensor([0.0, float(T)], dtype=DTYPE)
    y0 = torch.log(S0t).reshape(1, 1).expand(n_paths, 1).contiguous()

    # adjoint_params must include every tensor we want a gradient for that is
    # used *inside* f/g (here: the network params plus rate and vol_shift).
    adj_params = tuple(sde.parameters()) + (rt, vt)
    ys = torchsde.sdeint_adjoint(sde, y0, ts, bm=bm, method="euler",
                                 dt=float(T) / n_steps, adjoint_params=adj_params)
    ST = torch.exp(ys[-1].squeeze(-1))
    price = torch.exp(-rt * float(T)) * _payoff(ST, K, option_type).mean()
    price.backward()

    out = {"price": float(price.detach()), "delta": float(S0t.grad),
           "vega": float(vt.grad), "rho": float(rt.grad)}
    sde.configure(rate=float(r), vol_shift=0.0)        # clear leaked grad tensors
    return out


# ===========================================================================
# 3) Finite-difference Greeks (common random numbers) — the ground truth
# ===========================================================================
def finite_difference_greeks(sde, S0, K, T, r=RISK_FREE_RATE, option_type="call",
                             n_paths=20000, n_steps=50, seed=0,
                             h_S=1e-2, h_v=1e-3, h_r=1e-4, h_T=1e-3):
    """Central finite differences re-using the SAME Brownian increments."""
    Z = _common_normals(n_paths, n_steps, seed)
    n_paths = Z.shape[0]

    def price(S0_, T_, r_, vshift_):
        with torch.no_grad():
            YT = _simulate_terminal_rescaled(
                sde, torch.tensor(S0_, dtype=DTYPE), torch.tensor(T_, dtype=DTYPE),
                torch.tensor(r_, dtype=DTYPE), torch.tensor(vshift_, dtype=DTYPE),
                n_paths, n_steps, Z)
            ST = torch.exp(YT)
            disc = math.exp(-r_ * T_)
            return disc * _payoff(ST, K, option_type).mean().item()

    p0 = price(S0, T, r, 0.0)
    hS = h_S * S0
    delta = (price(S0 + hS, T, r, 0.0) - price(S0 - hS, T, r, 0.0)) / (2 * hS)
    gamma = (price(S0 + hS, T, r, 0.0) - 2 * p0 + price(S0 - hS, T, r, 0.0)) / hS ** 2
    vega = (price(S0, T, r, h_v) - price(S0, T, r, -h_v)) / (2 * h_v)
    rho = (price(S0, T, r + h_r, 0.0) - price(S0, T, r - h_r, 0.0)) / (2 * h_r)
    theta = -(price(S0, T + h_T, r, 0.0) - price(S0, T - h_T, r, 0.0)) / (2 * h_T)
    return {"price": p0, "delta": delta, "gamma": gamma, "vega": vega,
            "rho": rho, "theta": theta}


# ===========================================================================
# Validation table
# ===========================================================================
def validate_greeks(sde, S0=50.0, K=50.0, T=1.0, r=RISK_FREE_RATE,
                    option_type="call", n_paths=20000, n_steps=50, seed=0):
    """Compute Greeks three ways and print a comparison table."""
    fd = finite_difference_greeks(sde, S0, K, T, r, option_type,
                                  n_paths=n_paths, n_steps=n_steps, seed=seed)
    pw = greeks_pathwise(sde, S0, K, T, r, option_type,
                         n_paths=n_paths, n_steps=n_steps, seed=seed)
    adj = greeks_adjoint(sde, S0, K, T, r, option_type,
                         n_paths=n_paths, n_steps=n_steps, seed=seed + 1)

    print(f"\nGreeks @ S0={S0}, K={K}, T={T}, r={r}, {option_type}  "
          f"(N={n_paths} paths)")
    print(f"{'Greek':6s} {'adjoint':>12s} {'pathwise(AD)':>14s} "
          f"{'finite-diff':>14s} {'|AD-FD|':>10s}")
    print("-" * 60)
    for g in ("delta", "gamma", "vega", "rho", "theta"):
        a = adj.get(g, float("nan"))
        p = pw[g]
        f = fd[g]
        print(f"{g:6s} {a:12.5f} {p:14.5f} {f:14.5f} {abs(p - f):10.2e}")
    return {"adjoint": adj, "pathwise": pw, "finite_difference": fd}


if __name__ == "__main__":
    from models.train import load_trained

    sde = load_trained()
    S0 = float(torch.exp(sde.y_mean).item())   # in-domain spot
    validate_greeks(sde, S0=round(S0, 1), K=round(S0, 1), T=1.0)
