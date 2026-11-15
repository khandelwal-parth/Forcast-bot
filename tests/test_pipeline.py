"""Offline tests: feature leakage, labels, walk-forward purging, training, inference, analyzer fixes."""

import numpy as np
import pandas as pd
import pytest

from src import analyzer
from src.features import (FEATURE_COLS, HORIZON, build_dataset, build_features, build_market_context,
                          forward_return, rsi_wilder)
from src.model import (EMBARGO, CANDIDATES, bundle_from_bytes, bundle_to_bytes, predict_latest,
                       train_champion_candidate, walk_forward_folds)


def make_prices(n_days=1500, n_tickers=30, signal=0.0, seed=0, end=None):
    """Synthetic market: shared factor + idiosyncratic noise (+ optional 5-day mean reversion)."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range(end=end, periods=n_days) if end is not None else pd.bdate_range("2015-01-01", periods=n_days)
    mkt = rng.normal(0.0004, 0.01, n_days)
    bench = pd.DataFrame({"close": 100 * np.exp(np.cumsum(mkt))}, index=idx)
    bench["open"] = bench["high"] = bench["low"] = bench["close"]
    bench["volume"] = 1e6
    prices = {}
    for t in range(n_tickers):
        r = np.zeros(n_days)
        logp = np.zeros(n_days)
        for i in range(n_days):
            past5 = logp[i - 1] - logp[i - 6] if i >= 6 else 0.0
            r[i] = 0.0003 + 0.9 * mkt[i] + rng.normal(0, 0.012) - signal * past5
            logp[i] = (logp[i - 1] if i else 0) + r[i]
        close = 50 * np.exp(logp)
        high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n_days))
        prices[f"T{t}"] = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                                        "volume": rng.lognormal(13, 0.3, n_days)}, index=idx)
    return prices, bench


@pytest.fixture(scope="module")
def market_ctx():
    prices, bench = make_prices(n_days=400, n_tickers=1)
    return prices["T0"], build_market_context(bench, None), bench


def test_features_have_no_lookahead(market_ctx):
    df, ctx, bench = market_ctx
    full = build_features(df, ctx)
    for cut in (320, 350, 399):
        part_df = df.iloc[:cut + 1]
        part_ctx = build_market_context(bench.iloc[:cut + 1], None)
        part = build_features(part_df, part_ctx)
        a, b = full.iloc[cut].values.astype(float), part.iloc[cut].values.astype(float)
        assert np.allclose(a, b, equal_nan=True), f"feature leakage at row {cut}: " + \
            str([c for c, x, y in zip(FEATURE_COLS, a, b) if not (np.isnan(x) and np.isnan(y)) and abs(x - y) > 1e-6])


def test_features_are_stationary_scale_free():
    prices, bench = make_prices(n_days=400, n_tickers=1)
    df = prices["T0"]
    f1 = build_features(df)
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 37.0
    f2 = build_features(scaled)
    cols = [c for c in FEATURE_COLS if not c.startswith(("mkt_", "vix", "rel_"))]
    assert np.allclose(f1[cols].values.astype(float), f2[cols].values.astype(float), rtol=1e-3, atol=1e-4, equal_nan=True)


def test_forward_return_and_dataset_labels():
    prices, bench = make_prices(n_days=500, n_tickers=3)
    ctx = build_market_context(bench, None)
    ds = build_dataset(prices, {s: ctx for s in prices})
    df = prices["T0"]
    fwd = forward_return(df, HORIZON)
    assert np.isnan(fwd.iloc[-HORIZON:]).all()
    assert fwd.iloc[10] == pytest.approx(df["close"].iloc[10 + HORIZON] / df["close"].iloc[10] - 1)
    row = ds[ds.symbol == "T0"].iloc[50]
    assert row["y"] == int(row["fwd_ret"] > 0)
    last_date = df.index[-HORIZON - 1]
    assert ds[ds.symbol == "T0"]["date"].max() == last_date          # last HORIZON rows have no label -> dropped
    assert ds["date"].is_monotonic_increasing


def test_walk_forward_folds_are_purged():
    dates = np.array(pd.bdate_range("2010-01-01", periods=3000))
    folds = list(walk_forward_folds(dates))
    assert len(folds) == 5
    u = np.sort(np.unique(dates))
    for train_end, te0, te1 in folds:
        gap = (np.searchsorted(u, te0) - np.searchsorted(u, train_end))
        assert gap >= EMBARGO                                         # label horizon can't reach the test window
        assert te1 is None or te1 > te0
    starts = [f[1] for f in folds]
    assert starts == sorted(starts)


def test_rsi_wilder_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi_wilder(up).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi_wilder(down).iloc[-1] == pytest.approx(0.0, abs=1e-6)
    r = rsi_wilder(pd.Series(np.random.RandomState(1).normal(100, 1, 200)))
    assert r.dropna().between(0, 100).all()


FAST = {k: CANDIDATES[k] for k in ("logreg", "hgb_shallow")}


@pytest.fixture(scope="module")
def trained_signal():
    prices, bench = make_prices(n_days=1500, n_tickers=30, signal=0.08, seed=1)
    ctx = build_market_context(bench, None)
    ds = build_dataset(prices, {s: ctx for s in prices})
    return train_champion_candidate(ds, candidates=FAST, n_folds=3), prices, ctx


def test_model_finds_planted_signal(trained_signal):
    bundle, _, _ = trained_signal
    summ = bundle["metrics"]["summary"]
    assert summ["skill"] > 0 and summ["auc"] > 0.52
    assert bundle["reliable"] is True
    assert summ["positive_folds"] >= 2


def test_model_does_not_claim_edge_on_pure_noise():
    prices, bench = make_prices(n_days=1500, n_tickers=30, signal=0.0, seed=2)
    ctx = build_market_context(bench, None)
    ds = build_dataset(prices, {s: ctx for s in prices})
    bundle = train_champion_candidate(ds, candidates=FAST, n_folds=3)
    assert bundle["metrics"]["summary"]["auc"] < 0.53
    assert bundle["reliable"] is False


def test_bundle_roundtrip_and_prediction(trained_signal):
    bundle, prices, ctx = trained_signal
    restored = bundle_from_bytes(bundle_to_bytes(bundle))
    df = prices["T0"]
    p1 = predict_latest(bundle, df, ctx)
    p2 = predict_latest(restored, df, ctx)
    assert p1 == p2
    assert 0.0 < p1["prob_up"] < 1.0
    assert p1["direction"] in ("UP", "DOWN", "NEUTRAL")
    assert p1["horizon_days"] == HORIZON and p1["as_of"] == str(df.index[-1].date())
    short = predict_latest(bundle, df.iloc[:30], ctx)
    assert short["direction"] == "UNKNOWN"


def test_missing_fundamentals_do_not_trigger_assessments():
    out = analyzer.analyze_fundamentals({"DebtToEquity": "N/A", "CurrentRatio": "N/A", "PERatio": "N/A"}, {})
    assert out["financial_health"]["assessment"] == []
    assert out["valuation"]["assessment"] == []
    out = analyzer.analyze_fundamentals({"DebtToEquity": 0.3, "CurrentRatio": 0.8}, {})
    assert "Low debt levels" in out["financial_health"]["assessment"]
    assert "Low liquidity - potential concern" in out["financial_health"]["assessment"]


def test_unreliable_ml_is_not_counted_in_recommendation():
    base = analyzer.generate_overall_recommendation({"trend": []}, {}, {"direction": "UNKNOWN"})["score"]
    up = {"direction": "UP", "confidence": "HIGH", "prob_up": 0.7, "horizon_days": 5}
    assert analyzer.generate_overall_recommendation({"trend": []}, {}, {**up, "reliable": False})["score"] == base
    assert analyzer.generate_overall_recommendation({"trend": []}, {}, {**up, "reliable": True})["score"] == base + 1
    dn = {**up, "direction": "DOWN", "reliable": True, "confidence": "MEDIUM"}
    assert analyzer.generate_overall_recommendation({"trend": []}, {}, dn)["score"] == base - 0.5


def test_price_levels_are_ordered_and_volatility_scaled():
    calm = pd.Series({"close": 100.0, "atr": 1.0, "sma_20": 105.0, "sma_50": 99.0})
    wild = pd.Series({"close": 100.0, "atr": 5.0, "sma_20": 90.0, "sma_50": 80.0})
    for latest in (calm, wild):
        lv = analyzer.compute_price_levels(latest)
        assert all(z["price"] <= 100.0 for z in lv["entry_zones"])     # never "buy" above the current price
        assert all(t["price"] > 100.0 for t in lv["targets"])
        assert all(s["price"] < 100.0 for s in lv["stop_losses"])
    assert analyzer.compute_price_levels(wild)["targets"][0]["price"] > analyzer.compute_price_levels(calm)["targets"][0]["price"]
