"""Shared configuration and constants for the Neural SDE Options Pricer.

Centralises paths, the equity basket, regime definitions, and numerical
defaults so every module agrees on the same conventions.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for _d in (CACHE_DIR, OUTPUT_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Market universe
# ---------------------------------------------------------------------------
# SPY is the primary underlying we model; the large caps give the drift/diffusion
# networks more cross-sectional data to learn a robust dynamics shape.
PRIMARY_TICKER = "SPY"
BASKET = ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ"]

# Temporal, never-random regime splits used for out-of-sample testing (Phase 1).
# Training uses only data strictly *before* the earliest test regime to avoid
# look-ahead bias.
REGIMES = {
    "calm":   ("2017-01-01", "2019-12-31"),   # calm / bull
    "crisis": ("2020-02-01", "2020-06-30"),   # COVID crash
    "bear":   ("2022-01-01", "2022-12-31"),   # choppy / bear
}
# Train window: everything we are willing to fit on. We deliberately stop before
# the calm test regime begins so the *earliest* test regime is genuinely OOS.
TRAIN_WINDOW = ("2010-01-01", "2016-12-31")

# ---------------------------------------------------------------------------
# Numerical / finance defaults
# ---------------------------------------------------------------------------
TRADING_DAYS = 252                # annualisation factor
DT = 1.0 / TRADING_DAYS           # one trading day in years
REALIZED_VOL_WINDOW = 21          # rolling window for realized volatility
RISK_FREE_RATE = 0.03             # default flat risk-free rate (annual)
SEED = 42

# --- Latent SDE extension (variational training) ---
LATENT_WINDOW_LEN = 64            # observed-path window length (trading days)
LATENT_WINDOW_STRIDE = 5          # window start spacing within each ticker
KL_ANNEAL_EPOCHS = 30             # beta ramps 0 -> 1 linearly over this many epochs
KL_COLLAPSE_THRESHOLD = 0.05      # warn if pathwise KL/window < this after annealing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64             # float64: pricing/Greeks need the precision

torch.set_default_dtype(DTYPE)


def set_seed(seed: int = SEED) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
