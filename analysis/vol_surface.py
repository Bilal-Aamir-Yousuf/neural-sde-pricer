"""Phase 5 — Implied-volatility surface reconstruction & comparison.

Pipeline
--------
1. For a grid of strikes x maturities, price European calls with the Neural-SDE
   Monte Carlo engine (Phase 4).
2. Numerically invert Black-Scholes (Brent's method) to recover an implied
   volatility for each grid point.
3. Assemble the (moneyness x maturity x IV) surface and plot it in 3D.
4. Build the *market* IV surface from the real options chain pulled in Phase 1
   and compare shapes.

The headline question: does the Neural SDE reproduce the volatility
**smile / skew** that real markets exhibit — and that Black-Scholes, with a
single constant volatility, fundamentally cannot?  A non-flat model surface is
the single strongest piece of evidence the SDE learned something real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR, RISK_FREE_RATE  # noqa: E402
from pricing.baselines import implied_vol  # noqa: E402
from pricing.monte_carlo import price_option  # noqa: E402


# ---------------------------------------------------------------------------
# Model-implied surface
# ---------------------------------------------------------------------------
def model_iv_surface(sde, spot=100.0, moneyness=None, maturities=None,
                     r=RISK_FREE_RATE, n_paths=40000, n_steps=100):
    """Return (moneyness, maturities, IV grid) for the Neural-SDE model."""
    if moneyness is None:
        moneyness = np.linspace(0.80, 1.20, 13)
    if maturities is None:
        maturities = np.array([0.1, 0.25, 0.5, 1.0])

    iv = np.full((len(maturities), len(moneyness)), np.nan)
    for i, T in enumerate(maturities):
        for j, m in enumerate(moneyness):
            K = spot * m
            # Use OTM option type for a more stable inversion (calls for K>=S).
            opt = "call" if m >= 1.0 else "put"
            res = price_option(sde, spot, K, T, r, opt, "european",
                               n_paths=n_paths, n_steps=n_steps, antithetic=True)
            iv[i, j] = implied_vol(res.price, spot, K, T, r, opt)
    return np.asarray(moneyness), np.asarray(maturities), iv


# ---------------------------------------------------------------------------
# Market surface from the real options chain
# ---------------------------------------------------------------------------
def market_iv_surface(chain, moneyness=None, maturities=None, r=RISK_FREE_RATE):
    """Invert the real chain mids to IV and grid them onto (moneyness, maturity)."""
    if moneyness is None:
        moneyness = np.linspace(0.85, 1.15, 13)
    if maturities is None:
        maturities = np.array([0.1, 0.25, 0.5, 1.0])

    spot = float(chain["spot"].iloc[0])
    pts_m, pts_T, pts_iv = [], [], []
    for _, row in chain.iterrows():
        K, T, mid, opt = row["strike"], row["maturity"], row["mid"], row["type"]
        m = K / spot
        if not (0.7 <= m <= 1.3) or T <= 0:
            continue
        # OTM options only — they carry the cleanest vol information.
        if (opt == "call" and m < 1.0) or (opt == "put" and m > 1.0):
            continue
        iv = implied_vol(mid, spot, K, T, r, opt)
        if iv is not None and np.isfinite(iv) and 0.01 < iv < 2.0:
            pts_m.append(m); pts_T.append(T); pts_iv.append(iv)

    if len(pts_iv) < 4:
        return np.asarray(moneyness), np.asarray(maturities), \
            np.full((len(maturities), len(moneyness)), np.nan), spot

    from scipy.interpolate import griddata

    grid_m, grid_T = np.meshgrid(moneyness, maturities)
    iv_grid = griddata((pts_m, pts_T), pts_iv, (grid_m, grid_T), method="linear")
    nan_mask = np.isnan(iv_grid)
    if nan_mask.any():  # fill edges with nearest-neighbour
        iv_grid[nan_mask] = griddata((pts_m, pts_T), pts_iv,
                                     (grid_m, grid_T), method="nearest")[nan_mask]
    return np.asarray(moneyness), np.asarray(maturities), iv_grid, spot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_surfaces(model, market=None, fname="vol_surface.png"):
    """3D model surface (+ market if given) and a 2D smile comparison."""
    m_x, m_T, m_iv = model
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    X, Y = np.meshgrid(m_x, m_T)
    ax1.plot_surface(X, Y, m_iv * 100, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_xlabel("moneyness K/S"); ax1.set_ylabel("maturity (yr)")
    ax1.set_zlabel("implied vol (%)"); ax1.set_title("Neural SDE IV surface")

    if market is not None:
        k_x, k_T, k_iv = market[0], market[1], market[2]
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        Xk, Yk = np.meshgrid(k_x, k_T)
        ax2.plot_surface(Xk, Yk, k_iv * 100, cmap="plasma", edgecolor="none", alpha=0.9)
        ax2.set_xlabel("moneyness K/S"); ax2.set_ylabel("maturity (yr)")
        ax2.set_zlabel("implied vol (%)"); ax2.set_title("Market IV surface (real chain)")

    # 2D smile comparison at the shortest common maturity.
    ax3 = fig.add_subplot(1, 3, 3)
    ti = 0
    ax3.plot(m_x, m_iv[ti] * 100, "o-", label=f"Neural SDE (T={m_T[ti]:.2f})")
    if market is not None:
        ax3.plot(market[0], market[2][ti] * 100, "s--",
                 label=f"Market (T={market[1][ti]:.2f})")
    # Black-Scholes is a flat line by construction — the contrast is the point.
    flat = np.nanmean(m_iv[ti]) * 100
    ax3.axhline(flat, color="gray", ls=":", label="Black-Scholes (flat)")
    ax3.set_xlabel("moneyness K/S"); ax3.set_ylabel("implied vol (%)")
    ax3.set_title("Volatility smile / skew"); ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[vol] wrote {out.name}")
    return out


def skew_metric(moneyness, iv_row):
    """Quantify skew: IV(0.9) - IV(1.1). Positive => downside skew (real markets)."""
    lo = np.interp(0.9, moneyness, iv_row)
    hi = np.interp(1.1, moneyness, iv_row)
    return float(lo - hi)


def main():
    from data.fetch_data import fetch_options_chain
    from models.train import load_trained

    sde = load_trained()
    chain = fetch_options_chain()

    # Generate the model surface at a spot INSIDE the training price domain
    # (the network is calibrated to 2010-2016 prices ~$6-178; evaluating far
    # outside that range would just hit a saturated/flat region of the net).
    # We compare SHAPES in moneyness space, which is measure-/spot-invariant.
    model_spot = float(np.exp(sde.y_mean.item()))
    print(f"[vol] model spot={model_spot:.1f} (training-domain centre); "
          f"market spot={float(chain['spot'].iloc[0]):.1f}")
    model = model_iv_surface(sde, spot=model_spot)
    market = market_iv_surface(chain)
    plot_surfaces(model, market)

    skew = skew_metric(model[0], model[2][0])
    print(f"[vol] model short-maturity skew IV(0.9)-IV(1.1) = {skew:+.3%} "
          f"({'downside skew like real markets' if skew > 0 else 'flat/inverted'})")
    print(f"[vol] model IV range: {np.nanmin(model[2]):.1%} - {np.nanmax(model[2]):.1%}")


if __name__ == "__main__":
    main()
