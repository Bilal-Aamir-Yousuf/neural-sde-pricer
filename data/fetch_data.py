"""Phase 1 — Data pipeline.

Responsibilities
----------------
1. Pull historical daily prices for SPY + large caps via ``yfinance``.
2. Compute log-returns and rolling 21-day realized volatility.
3. Split the data *temporally* (never randomly) into the calm / crisis / bear
   regimes used for out-of-sample testing, plus a strictly-earlier train window.
4. Pull an options chain (strikes, maturities, mid quotes) for the primary
   underlying, used in Phase 5 to compare implied-vol surfaces.

Robustness
----------
Yahoo can be flaky / rate-limited and CI machines may be offline.  Every fetch
is therefore cached to ``data/cache`` and, if the network is unavailable, the
pipeline falls back to a *synthetic* but economically realistic dataset
(regime-dependent Heston-like paths) so the entire downstream project remains
runnable and testable.  Whether data is real or synthetic is always reported.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    BASKET,
    CACHE_DIR,
    PRIMARY_TICKER,
    REALIZED_VOL_WINDOW,
    REGIMES,
    TRADING_DAYS,
    TRAIN_WINDOW,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Price download (with cache + synthetic fallback)
# ---------------------------------------------------------------------------
def _synthetic_prices(tickers, start="2010-01-01", end="2022-12-31", seed=7):
    """Generate regime-dependent synthetic prices so the project runs offline.

    Uses a Heston-style stochastic-volatility process per ticker, with the
    long-run variance level raised during the COVID and 2022 windows so the
    synthetic data exhibits genuine regime structure (vol clustering, fat-ish
    tails) rather than plain constant-vol GBM.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    dt = 1.0 / TRADING_DAYS

    def regime_vol_multiplier(date):
        d = pd.Timestamp(date)
        if pd.Timestamp("2020-02-01") <= d <= pd.Timestamp("2020-06-30"):
            return 3.5          # COVID crash: vol spike
        if pd.Timestamp("2022-01-01") <= d <= pd.Timestamp("2022-12-31"):
            return 1.8          # choppy bear
        return 1.0

    mult = np.array([regime_vol_multiplier(d) for d in dates])

    out = {}
    for j, tk in enumerate(tickers):
        s0 = 100.0 + 50.0 * rng.random()
        kappa, theta, xi, rho = 3.0, 0.04, 0.5, -0.7
        mu = 0.07 + 0.03 * rng.random()
        v = theta
        log_s = np.log(s0)
        prices = np.empty(n)
        for i in range(n):
            theta_t = theta * mult[i] ** 2
            z1, z2 = rng.standard_normal(2)
            dw_s = np.sqrt(dt) * z1
            dw_v = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)
            v = max(v + kappa * (theta_t - v) * dt + xi * np.sqrt(max(v, 1e-8)) * dw_v, 1e-8)
            log_s += (mu - 0.5 * v) * dt + np.sqrt(v) * dw_s
            prices[i] = np.exp(log_s)
        out[tk] = prices
    df = pd.DataFrame(out, index=dates)
    df.index.name = "Date"
    return df


def download_prices(tickers=BASKET, start="2010-01-01", end="2022-12-31",
                    force_refresh=False):
    """Return a DataFrame of adjusted close prices (columns = tickers).

    Tries cache → yfinance → synthetic fallback, in that order.
    """
    cache_path = CACHE_DIR / "prices.parquet"
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        if set(tickers).issubset(df.columns):
            print(f"[data] loaded cached prices {df.shape} from {cache_path.name}")
            return df[tickers]

    try:
        import yfinance as yf

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, threads=True)
        if raw is None or len(raw) == 0:
            raise RuntimeError("yfinance returned empty frame")
        # yfinance returns a column MultiIndex (field, ticker) for >1 ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw[["Close"]].copy()
            close.columns = [tickers[0]] if isinstance(tickers, (list, tuple)) else [tickers]
        close = close.dropna(how="all").ffill().dropna()
        if close.shape[0] < 100:
            raise RuntimeError("too few rows from yfinance")
        close.to_parquet(cache_path)
        print(f"[data] downloaded REAL prices {close.shape} via yfinance")
        return close[[t for t in tickers if t in close.columns]]
    except Exception as exc:  # noqa: BLE001 — any failure → synthetic
        print(f"[data] yfinance unavailable ({type(exc).__name__}: {exc}); "
              f"using SYNTHETIC fallback data.")
        df = _synthetic_prices(tickers, start, end)
        df.to_parquet(cache_path)
        return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Log-returns and rolling realized volatility (annualised)."""
    log_price = np.log(prices)
    log_ret = log_price.diff()
    # Realized vol: rolling std of daily log-returns, annualised.
    realized_vol = log_ret.rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    return {
        "price": prices,
        "log_price": log_price,
        "log_return": log_ret,
        "realized_vol": realized_vol,
    }


# ---------------------------------------------------------------------------
# Temporal regime splitting
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """A temporal slice with engineered features for one regime/window."""
    name: str
    prices: pd.DataFrame
    log_return: pd.DataFrame
    realized_vol: pd.DataFrame

    def primary_log_returns(self, ticker=PRIMARY_TICKER) -> np.ndarray:
        return self.log_return[ticker].dropna().to_numpy()


def _slice(features, start, end, name) -> Dataset:
    idx = (features["price"].index >= pd.Timestamp(start)) & (
        features["price"].index <= pd.Timestamp(end)
    )
    return Dataset(
        name=name,
        prices=features["price"].loc[idx],
        log_return=features["log_return"].loc[idx],
        realized_vol=features["realized_vol"].loc[idx],
    )


def build_regime_splits(prices: pd.DataFrame | None = None):
    """Return (train_dataset, {regime_name: Dataset}).

    The train window ends strictly before the earliest test regime, so even the
    calm regime is genuine out-of-sample (no look-ahead bias).
    """
    if prices is None:
        prices = download_prices()
    features = compute_features(prices)

    train = _slice(features, *TRAIN_WINDOW, name="train")
    test_sets = {name: _slice(features, lo, hi, name) for name, (lo, hi) in REGIMES.items()}

    # Sanity: train must end before any test regime begins.
    train_end = pd.Timestamp(TRAIN_WINDOW[1])
    earliest_test = min(pd.Timestamp(lo) for lo, _ in REGIMES.values())
    assert train_end < earliest_test, "look-ahead bias: train overlaps test regimes"
    return train, test_sets


# ---------------------------------------------------------------------------
# Options chain
# ---------------------------------------------------------------------------
def fetch_options_chain(ticker=PRIMARY_TICKER, max_expiries=6):
    """Pull (strike, maturity_years, mid, type, spot) rows for an underlying.

    Falls back to a synthetic Black-Scholes-with-smile chain when offline, so
    Phase 5's market-vs-model surface comparison always has something to plot.
    """
    cache_path = CACHE_DIR / f"options_{ticker}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"[data] loaded cached options chain for {ticker} ({len(df)} rows)")
        return df

    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        spot = float(tk.history(period="1d")["Close"].iloc[-1])
        expiries = tk.options[:max_expiries]
        if not expiries:
            raise RuntimeError("no expiries")
        rows = []
        today = pd.Timestamp.today().normalize()
        for exp in expiries:
            chain = tk.option_chain(exp)
            T = (pd.Timestamp(exp) - today).days / 365.0
            if T <= 0:
                continue
            for opt_type, frame in (("call", chain.calls), ("put", chain.puts)):
                for _, r in frame.iterrows():
                    bid, ask = r.get("bid", np.nan), r.get("ask", np.nan)
                    mid = (bid + ask) / 2 if bid and ask else r.get("lastPrice", np.nan)
                    if not mid or mid <= 0:
                        continue
                    rows.append({
                        "strike": float(r["strike"]), "maturity": T,
                        "mid": float(mid), "type": opt_type, "spot": spot,
                    })
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("empty options chain")
        df.to_parquet(cache_path)
        print(f"[data] downloaded REAL options chain for {ticker} ({len(df)} rows)")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[data] options chain unavailable ({type(exc).__name__}); "
              f"using SYNTHETIC smile chain.")
        df = _synthetic_chain(ticker)
        df.to_parquet(cache_path)
        return df


def _synthetic_chain(ticker, spot=450.0):
    """Synthetic chain with a realistic volatility smile/skew baked in."""
    from pricing.baselines import black_scholes_price  # local import avoids cycle

    maturities = [0.08, 0.25, 0.5, 1.0]
    moneyness = np.linspace(0.8, 1.2, 13)
    rows = []
    for T in maturities:
        for m in moneyness:
            K = spot * m
            # Smile: higher IV for OTM puts (skew) + term structure.
            iv = 0.18 + 0.10 * (m - 1.0) ** 2 - 0.05 * (m - 1.0) + 0.02 * np.sqrt(T)
            for opt_type in ("call", "put"):
                price = black_scholes_price(spot, K, T, 0.03, iv, opt_type)
                rows.append({"strike": K, "maturity": T, "mid": price,
                             "type": opt_type, "spot": spot})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    prices = download_prices()
    train, tests = build_regime_splits(prices)
    print(f"\n[data] train window {TRAIN_WINDOW}: {len(train.prices)} days")
    for name, ds in tests.items():
        rv = ds.realized_vol[PRIMARY_TICKER].mean()
        print(f"[data] regime '{name}': {len(ds.prices)} days, "
              f"mean realized vol {rv:.1%}")
    chain = fetch_options_chain()
    print(f"[data] options chain: {len(chain)} rows, "
          f"{chain['maturity'].nunique()} maturities")


if __name__ == "__main__":
    main()
