"""Tests for the data pipeline (Phase 1) — focus on no look-ahead bias."""
import numpy as np
import pandas as pd

from config import PRIMARY_TICKER, REGIMES, TRAIN_WINDOW
from data.fetch_data import build_regime_splits, compute_features, download_prices


def test_no_lookahead_bias():
    """Train window must end strictly before every test regime begins."""
    train_end = pd.Timestamp(TRAIN_WINDOW[1])
    for name, (lo, hi) in REGIMES.items():
        assert train_end < pd.Timestamp(lo), f"train overlaps regime {name}"


def test_regime_split_shapes():
    train, tests = build_regime_splits()
    assert len(train.prices) > 100
    assert set(tests) == set(REGIMES)
    for ds in tests.values():
        assert len(ds.prices) > 0


def test_features_are_finite_and_reasonable():
    prices = download_prices()
    feats = compute_features(prices)
    rv = feats["realized_vol"][PRIMARY_TICKER].dropna()
    # Annualised realized vol should sit in a sane band.
    assert (rv > 0).all()
    assert 0.02 < rv.median() < 2.0


def test_crisis_more_volatile_than_calm():
    """Sanity: the COVID crisis regime is more volatile than the calm regime."""
    _, tests = build_regime_splits()
    calm = tests["calm"].realized_vol[PRIMARY_TICKER].mean()
    crisis = tests["crisis"].realized_vol[PRIMARY_TICKER].mean()
    assert crisis > calm
