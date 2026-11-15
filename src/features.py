"""Feature engineering for the ML model.

Everything here is *stationary* (ratios, returns, z-scores) so a model trained on many
tickers and many years can be applied to any stock at any price level. Every feature at
row t only uses data up to and including t (no look-ahead) - tests/test_pipeline.py checks that.
"""

import numpy as np
import pandas as pd

# Prediction horizon in trading days. Next-day direction is ~pure noise; 5 days has a
# noticeably better signal-to-noise ratio. Change here and the whole pipeline follows.
HORIZON = 5

FEATURE_COLS = [
    # multi-horizon returns
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_21", "ret_63",
    # trend / position
    "dist_sma20", "dist_sma50", "dist_sma200", "sma20_sma50",
    "dist_52w_high", "dist_52w_low",
    # momentum
    "rsi", "macd_pct", "macd_hist_pct", "stoch_k", "bb_pct", "bb_width",
    # volatility / range
    "atr_pct", "vol_5", "vol_21", "vol_ratio", "range_pct", "close_loc", "gap",
    # volume
    "volume_z", "volume_ratio",
    # calendar
    "dow",
    # market context
    "mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_dist_sma50", "mkt_vol_21",
    "vix", "vix_chg_5", "rel_ret_21", "rel_ret_5",
]

# Rows missing these are warm-up rows (not enough history) and are dropped from training.
REQUIRED_COLS = ["dist_sma200", "ret_63", "vol_21"]


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (matches TradingView / most charting tools)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
        return 100 - 100 / (1 + rs)


def benchmark_for(symbol: str) -> tuple:
    """(market index, volatility index) used as context for a symbol."""
    s = symbol.upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return "^NSEI", "^INDIAVIX"
    return "SPY", "^VIX"


def build_market_context(bench: pd.DataFrame, vix: pd.DataFrame = None) -> pd.DataFrame:
    """Market-wide features indexed by date, from a benchmark index (+ optional VIX) frame."""
    c = bench["close"]
    ret1 = c.pct_change()
    m = pd.DataFrame(index=bench.index)
    m["mkt_ret_1"] = ret1
    m["mkt_ret_5"] = c.pct_change(5)
    m["mkt_ret_21"] = c.pct_change(21)
    m["mkt_dist_sma50"] = c / c.rolling(50).mean() - 1
    m["mkt_vol_21"] = ret1.rolling(21).std()
    if vix is not None and len(vix):
        v = vix["close"].reindex(m.index, method="ffill")
        m["vix"] = v
        m["vix_chg_5"] = v.pct_change(5)
    else:
        m["vix"] = np.nan
        m["vix_chg_5"] = np.nan
    return m


def build_features(df: pd.DataFrame, market: pd.DataFrame = None) -> pd.DataFrame:
    """Stationary feature frame (same index as df) from an OHLCV frame."""
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    f = pd.DataFrame(index=df.index)
    ret1 = c.pct_change()

    for n in (1, 2, 3, 5, 10, 21, 63):
        f[f"ret_{n}"] = c.pct_change(n)

    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    f["dist_sma20"] = c / sma20 - 1
    f["dist_sma50"] = c / sma50 - 1
    f["dist_sma200"] = c / sma200 - 1
    f["sma20_sma50"] = sma20 / sma50 - 1
    f["dist_52w_high"] = c / c.rolling(252, min_periods=126).max() - 1
    f["dist_52w_low"] = c / c.rolling(252, min_periods=126).min() - 1

    f["rsi"] = rsi_wilder(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_pct"] = macd / c
    f["macd_hist_pct"] = (macd - macd_sig) / c

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14)
    bb_std = c.rolling(20).std()
    f["bb_pct"] = (c - (sma20 - 2 * bb_std)) / (4 * bb_std)
    f["bb_width"] = 4 * bb_std / sma20

    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["vol_5"] = ret1.rolling(5).std()
    f["vol_21"] = ret1.rolling(21).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_21"]
    f["range_pct"] = (h - l) / c
    f["close_loc"] = (c - l) / (h - l)
    f["gap"] = o / prev_c - 1

    v_mean, v_std = v.rolling(20).mean(), v.rolling(20).std()
    f["volume_z"] = (v - v_mean) / v_std
    f["volume_ratio"] = v / v_mean
    f["dow"] = df.index.dayofweek.astype("float64")

    if market is not None and len(market):
        mk = market.reindex(df.index, method="ffill")
        for col in ("mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_dist_sma50", "mkt_vol_21", "vix", "vix_chg_5"):
            f[col] = mk[col] if col in mk else np.nan
        f["rel_ret_21"] = f["ret_21"] - f["mkt_ret_21"]
        f["rel_ret_5"] = f["ret_5"] - f["mkt_ret_5"]
    else:
        for col in ("mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_dist_sma50", "mkt_vol_21",
                    "vix", "vix_chg_5", "rel_ret_21", "rel_ret_5"):
            f[col] = np.nan

    f = f.replace([np.inf, -np.inf], np.nan)
    return f[FEATURE_COLS].astype("float32")


def forward_return(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Return from close[t] to close[t+horizon]. NaN for the last `horizon` rows."""
    c = df["close"]
    return c.shift(-horizon) / c - 1


def build_dataset(prices: dict, markets: dict, horizon: int = HORIZON) -> pd.DataFrame:
    """Pooled (all tickers) training frame: date, symbol, features, fwd_ret, y.

    prices:  {symbol: OHLCV DataFrame}
    markets: {symbol: market-context DataFrame or None}
    """
    parts = []
    for sym, df in prices.items():
        if df is None or len(df) < 320:
            continue
        f = build_features(df, markets.get(sym))
        f = f.dropna(subset=REQUIRED_COLS)
        if f.empty:
            continue
        fwd = forward_return(df, horizon).reindex(f.index)
        f = f.assign(fwd_ret=fwd.astype("float32"), symbol=sym)
        f.index.name = "date"
        parts.append(f.reset_index())
    if not parts:
        return pd.DataFrame()
    ds = pd.concat(parts, ignore_index=True)
    ds = ds.dropna(subset=["fwd_ret"])           # last `horizon` rows of every ticker have no label yet
    ds["y"] = (ds["fwd_ret"] > 0).astype("int8")
    return ds.sort_values("date", kind="stable").reset_index(drop=True)
