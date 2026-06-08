"""Phase 4 — Monte Carlo pricing engine.

Simulates forward price paths from the trained Neural SDE **under the
risk-neutral measure Q** and prices European and exotic options off those paths.

What's here
-----------
* ``simulate_paths``     — vectorised Euler-Maruyama / Milstein integrator with
                           **antithetic variates** and full-path tracking
                           (needed for path-dependent payoffs).  Fully
                           differentiable, so Greeks fall out via autograd.
* ``simulate_paths_torchsde`` — the same simulation via ``torchsde.sdeint``
                           (honours the spec's integrator; used to cross-check
                           the hand-rolled engine agrees).
* payoffs: European, barrier (knock-in/out), Asian (arithmetic average),
  lookback (floating & fixed strike).
* ``price_option``       — discounted MC estimate with a proper standard error
                           (antithetic-pair aware).
* ``convergence_study``  — demonstrates the defining 1/sqrt(N) Monte Carlo rate.

Everything prices under measure Q: the SDE's drift is replaced by the
risk-neutral drift r - 1/2 sigma^2 (see ``neural_sde.py``) so prices are
arbitrage-free.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DTYPE, OUTPUT_DIR, RISK_FREE_RATE  # noqa: E402


# ===========================================================================
# Path simulation
# ===========================================================================
def simulate_paths(sde, S0, T, n_steps=100, n_paths=10000, r=RISK_FREE_RATE,
                   antithetic=True, scheme="euler", generator=None):
    """Simulate price paths under Q. Returns S of shape (n_paths, n_steps+1).

    If ``antithetic`` the first half of the paths use Brownian increments Z and
    the second half use -Z (paired for variance reduction).  Differentiable in
    ``S0`` (and in ``r`` / the SDE's vol-shift) for pathwise Greeks.
    """
    sde.configure(measure="Q", rate=r)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    if antithetic:
        if n_paths % 2:
            n_paths += 1
        half = n_paths // 2
        Z = torch.randn(half, n_steps, dtype=DTYPE, generator=generator)
        Z = torch.cat([Z, -Z], dim=0)
    else:
        Z = torch.randn(n_paths, n_steps, dtype=DTYPE, generator=generator)

    S0t = torch.as_tensor(S0, dtype=DTYPE)
    Y = torch.log(S0t).reshape(1).expand(n_paths).clone()   # (n_paths,)

    prices = [torch.exp(Y)]
    t = 0.0
    for n in range(n_steps):
        tt = torch.as_tensor(t, dtype=DTYPE)
        yv = Y.unsqueeze(-1)
        mu = sde.f(tt, yv).squeeze(-1)
        sig = sde.g(tt, yv).squeeze(-1)
        dW = sqrt_dt * Z[:, n]
        if scheme == "milstein":
            # Milstein adds 1/2 * sigma * d(sigma)/dY * (dW^2 - dt).
            dsig = _dsigma_dy(sde, tt, Y)
            Y = Y + mu * dt + sig * dW + 0.5 * sig * dsig * (dW ** 2 - dt)
        else:  # euler-maruyama
            Y = Y + mu * dt + sig * dW
        t += dt
        prices.append(torch.exp(Y))
    return torch.stack(prices, dim=1)


def _dsigma_dy(sde, t, Y):
    """d sigma / d Y via autograd (for the Milstein correction)."""
    y = Y.detach().unsqueeze(-1).requires_grad_(True)
    sig = sde.g(t, y).squeeze(-1)
    (grad,) = torch.autograd.grad(sig.sum(), y, create_graph=False)
    return grad.squeeze(-1).detach()


def simulate_paths_torchsde(sde, S0, T, n_steps=100, n_paths=10000,
                            r=RISK_FREE_RATE, method="euler"):
    """Same simulation via torchsde.sdeint (spec's integrator)."""
    import torchsde

    sde.configure(measure="Q", rate=r)
    ts = torch.linspace(0.0, T, n_steps + 1, dtype=DTYPE)
    y0 = torch.full((n_paths, 1), math.log(float(S0)), dtype=DTYPE)
    ys = torchsde.sdeint(sde, y0, ts, method=method, dt=T / n_steps)
    return torch.exp(ys.squeeze(-1)).transpose(0, 1)        # (n_paths, n_steps+1)


# ===========================================================================
# Payoffs  (operate on price paths S of shape (n_paths, n_steps+1))
# ===========================================================================
def european_payoff(S, K, option_type="call"):
    ST = S[:, -1]
    return torch.clamp(ST - K, min=0.0) if option_type == "call" else torch.clamp(K - ST, min=0.0)


def asian_payoff(S, K, option_type="call", average="arithmetic"):
    avg = S[:, 1:].mean(dim=1) if average == "arithmetic" else torch.exp(torch.log(S[:, 1:]).mean(dim=1))
    return torch.clamp(avg - K, min=0.0) if option_type == "call" else torch.clamp(K - avg, min=0.0)


def lookback_payoff(S, K=None, option_type="call", strike="floating"):
    """Floating-strike (default) or fixed-strike lookback."""
    run_max, run_min, ST = S.max(dim=1).values, S.min(dim=1).values, S[:, -1]
    if strike == "floating":
        return ST - run_min if option_type == "call" else run_max - ST
    # fixed strike
    return (torch.clamp(run_max - K, min=0.0) if option_type == "call"
            else torch.clamp(K - run_min, min=0.0))


def barrier_payoff(S, K, barrier, option_type="call", barrier_type="up-and-out"):
    """Barrier option payoff. barrier_type in {up,down}-and-{in,out}."""
    direction, knock = barrier_type.split("-and-")
    breached = ((S >= barrier).any(dim=1) if direction == "up"
                else (S <= barrier).any(dim=1)).to(S.dtype)
    vanilla = european_payoff(S, K, option_type)
    alive = breached if knock == "in" else (1.0 - breached)
    return vanilla * alive


# ===========================================================================
# Discounted MC estimator (antithetic-pair aware standard error)
# ===========================================================================
@dataclass
class PriceResult:
    price: float
    stderr: float
    n_paths: int

    def __repr__(self):
        return f"PriceResult(price={self.price:.4f}, stderr={self.stderr:.4f}, N={self.n_paths})"


def discount_and_estimate(payoff, r, T, antithetic=True):
    """Return PriceResult from a payoff tensor (handles antithetic pairing)."""
    disc = math.exp(-r * T)
    n = payoff.shape[0]
    pay = payoff.detach()
    if antithetic and n % 2 == 0:
        half = n // 2
        pair = 0.5 * (pay[:half] + pay[half:])         # average each antithetic pair
        price = disc * pair.mean().item()
        se = disc * (pair.std(unbiased=True) / math.sqrt(half)).item()
        return PriceResult(price, se, n)
    price = disc * pay.mean().item()
    se = disc * (pay.std(unbiased=True) / math.sqrt(n)).item()
    return PriceResult(price, se, n)


def price_option(sde, S0, K, T, r=RISK_FREE_RATE, option_type="call",
                 kind="european", n_paths=10000, n_steps=100, antithetic=True,
                 scheme="euler", generator=None, **payoff_kwargs):
    """End-to-end MC price of a (possibly exotic) option under the Neural SDE.

    Runs under ``torch.no_grad()`` — pricing never needs the autograd graph
    (Greeks build their own in ``greeks.py``), and tracking it through tens of
    thousands of paths would waste memory.
    """
    with torch.no_grad():
        S = simulate_paths(sde, S0, T, n_steps, n_paths, r, antithetic, scheme, generator)
        if kind == "european":
            payoff = european_payoff(S, K, option_type)
        elif kind == "asian":
            payoff = asian_payoff(S, K, option_type, **payoff_kwargs)
        elif kind == "lookback":
            payoff = lookback_payoff(S, K, option_type, **payoff_kwargs)
        elif kind == "barrier":
            payoff = barrier_payoff(S, K, option_type=option_type, **payoff_kwargs)
        else:
            raise ValueError(f"unknown option kind: {kind}")
    return discount_and_estimate(payoff, r, T, antithetic)


# ===========================================================================
# Convergence study — the defining 1/sqrt(N) Monte Carlo signature
# ===========================================================================
def convergence_study(sde, S0, K, T, r=RISK_FREE_RATE, option_type="call",
                      path_counts=(500, 1000, 2000, 4000, 8000, 16000, 32000, 64000),
                      n_steps=100, plot=True):
    """Show MC standard error shrinks ~ 1/sqrt(N). Returns dict of results."""
    Ns, prices, ses = [], [], []
    for N in path_counts:
        res = price_option(sde, S0, K, T, r, option_type, "european",
                           n_paths=N, n_steps=n_steps, antithetic=False)
        Ns.append(res.n_paths); prices.append(res.price); ses.append(res.stderr)
        print(f"  N={res.n_paths:6d}  price={res.price:.4f}  stderr={res.stderr:.5f}")

    Ns_a = np.array(Ns, float)
    ses_a = np.array(ses, float)
    # Fit log(SE) = a + b*log(N); b should be ~ -0.5.
    b, a = np.polyfit(np.log(Ns_a), np.log(ses_a), 1)
    print(f"[mc] convergence slope d log(SE)/d log(N) = {b:.3f}  (theory -0.5)")

    if plot:
        plt.figure(figsize=(7, 5))
        plt.loglog(Ns_a, ses_a, "o-", label="MC standard error")
        ref = ses_a[0] * np.sqrt(Ns_a[0] / Ns_a)
        plt.loglog(Ns_a, ref, "--", color="gray", label="ideal 1/sqrt(N)")
        plt.xlabel("number of paths N"); plt.ylabel("standard error")
        plt.title(f"Monte Carlo convergence (fitted slope {b:.2f})")
        plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
        out = OUTPUT_DIR / "mc_convergence.png"
        plt.savefig(out, dpi=130); plt.close()
        print(f"[mc] wrote {out.name}")

    return {"N": Ns, "price": prices, "stderr": ses, "slope": float(b)}


if __name__ == "__main__":
    from models.train import load_trained

    sde = load_trained()
    S0 = 100.0
    print("European call MC vs exotics @ S0=100, K=100, T=1:")
    for kind, kw in [("european", {}), ("asian", {}),
                     ("lookback", {"strike": "floating"}),
                     ("barrier", {"barrier": 130.0, "barrier_type": "up-and-out"})]:
        res = price_option(sde, S0, 100.0, 1.0, kind=kind, **kw)
        print(f"  {kind:10s} {res}")
    print("\nConvergence study:")
    convergence_study(sde, S0, 100.0, 1.0)
