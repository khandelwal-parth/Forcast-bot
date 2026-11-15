"""Technical and Fundamental Analysis Module."""

import pandas as pd
import numpy as np

from .features import rsi_wilder


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators to the dataframe."""
    df = df.copy()

    # Moving Averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI (Wilder smoothing)
    df["rsi"] = rsi_wilder(df["close"])

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Stochastic Oscillator
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr"] = true_range.rolling(window=14).mean()

    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Price momentum
    df["momentum"] = df["close"] - df["close"].shift(14)
    df["roc"] = ((df["close"] - df["close"].shift(14)) / df["close"].shift(14)) * 100

    return df


def generate_signals(df: pd.DataFrame) -> dict:
    """Generate buy/sell signals based on technical indicators."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    signals = {
        "trend": [],
        "momentum": [],
        "volatility": [],
        "volume": [],
    }

    # Trend Signals
    if latest["close"] > latest["sma_20"] > latest["sma_50"]:
        signals["trend"].append(("BULLISH", "Price above SMA20 and SMA50 - uptrend"))
    elif latest["close"] < latest["sma_20"] < latest["sma_50"]:
        signals["trend"].append(("BEARISH", "Price below SMA20 and SMA50 - downtrend"))

    # Golden/Death Cross
    if latest["sma_20"] > latest["sma_50"] and prev["sma_20"] <= prev["sma_50"]:
        signals["trend"].append(("STRONG BUY", "Golden Cross detected!"))
    elif latest["sma_20"] < latest["sma_50"] and prev["sma_20"] >= prev["sma_50"]:
        signals["trend"].append(("STRONG SELL", "Death Cross detected!"))

    # MACD Signals
    if latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
        signals["momentum"].append(("BUY", "MACD crossed above signal line"))
    elif latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
        signals["momentum"].append(("SELL", "MACD crossed below signal line"))

    # RSI Signals
    if latest["rsi"] < 30:
        signals["momentum"].append(("OVERSOLD", f"RSI at {latest['rsi']:.1f} - potential bounce"))
    elif latest["rsi"] > 70:
        signals["momentum"].append(("OVERBOUGHT", f"RSI at {latest['rsi']:.1f} - potential pullback"))

    # Bollinger Band Signals
    if latest["bb_pct"] < 0.1:
        signals["volatility"].append(("BUY", "Price near lower Bollinger Band - potential reversal"))
    elif latest["bb_pct"] > 0.9:
        signals["volatility"].append(("SELL", "Price near upper Bollinger Band - potential reversal"))

    # Stochastic Signals
    if latest["stoch_k"] < 20 and latest["stoch_k"] > latest["stoch_d"]:
        signals["momentum"].append(("BUY", "Stochastic oversold with bullish cross"))
    elif latest["stoch_k"] > 80 and latest["stoch_k"] < latest["stoch_d"]:
        signals["momentum"].append(("SELL", "Stochastic overbought with bearish cross"))

    # Volume Signals
    if latest["volume_ratio"] > 2:
        signals["volume"].append(("HIGH_VOLUME", f"Volume {latest['volume_ratio']:.1f}x average"))
    elif latest["volume_ratio"] < 0.5:
        signals["volume"].append(("LOW_VOLUME", f"Volume {latest['volume_ratio']:.1f}x average"))

    return signals


def analyze_fundamentals(company_info: dict, income_data: dict) -> dict:
    """Analyze fundamental data. Missing values stay None and never trigger an assessment."""
    def to_float(val):
        if val is None or val == "N/A" or val == "":
            return None
        try:
            v = float(val)
            return v if v == v else None
        except (ValueError, TypeError):
            return None

    analysis = {}

    pe_ratio = to_float(company_info.get("PERatio"))
    peg_ratio = to_float(company_info.get("PEGRatio"))
    pb_ratio = to_float(company_info.get("PriceToBookRatio"))
    analysis["valuation"] = {"pe_ratio": pe_ratio, "peg_ratio": peg_ratio, "pb_ratio": pb_ratio, "assessment": []}
    if pe_ratio is not None and 0 < pe_ratio < 15:
        analysis["valuation"]["assessment"].append("Undervalued based on P/E")
    elif pe_ratio is not None and pe_ratio > 30:
        analysis["valuation"]["assessment"].append("Potentially overvalued based on P/E")
    if peg_ratio is not None and 0 < peg_ratio < 1:
        analysis["valuation"]["assessment"].append("Good value considering growth (PEG < 1)")

    profit_margin = to_float(company_info.get("ProfitMargin"))
    operating_margin = to_float(company_info.get("OperatingMarginTTM"))
    roe = to_float(company_info.get("ReturnOnEquityTTM"))
    analysis["profitability"] = {"profit_margin": profit_margin, "operating_margin": operating_margin,
                                 "roe": roe, "assessment": []}
    if profit_margin is not None and profit_margin > 0.20:
        analysis["profitability"]["assessment"].append("Excellent profit margins (>20%)")
    elif profit_margin is not None and profit_margin > 0.10:
        analysis["profitability"]["assessment"].append("Healthy profit margins (>10%)")
    if roe is not None and roe > 0.15:
        analysis["profitability"]["assessment"].append("Strong ROE (>15%)")

    revenue_growth = to_float(company_info.get("QuarterlyRevenueGrowthYOY"))
    earnings_growth = to_float(company_info.get("QuarterlyEarningsGrowthYOY"))
    analysis["growth"] = {"revenue_growth_yoy": revenue_growth, "earnings_growth_yoy": earnings_growth, "assessment": []}
    if revenue_growth is not None and revenue_growth > 0.20:
        analysis["growth"]["assessment"].append("Strong revenue growth (>20% YoY)")
    if earnings_growth is not None and earnings_growth > 0.20:
        analysis["growth"]["assessment"].append("Strong earnings growth (>20% YoY)")

    debt_to_equity = to_float(company_info.get("DebtToEquity"))
    current_ratio = to_float(company_info.get("CurrentRatio"))
    analysis["financial_health"] = {"debt_to_equity": debt_to_equity, "current_ratio": current_ratio, "assessment": []}
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            analysis["financial_health"]["assessment"].append("Low debt levels")
        elif debt_to_equity > 2:
            analysis["financial_health"]["assessment"].append("High debt levels - monitor closely")
    if current_ratio is not None:
        if current_ratio > 1.5:
            analysis["financial_health"]["assessment"].append("Good liquidity")
        elif current_ratio < 1:
            analysis["financial_health"]["assessment"].append("Low liquidity - potential concern")

    return analysis


def compute_price_levels(latest) -> dict:
    """Volatility-scaled (ATR) entry zones, targets and stops - tied to the stock's own range."""
    close = float(latest["close"])
    atr = latest.get("atr") if hasattr(latest, "get") else None
    atr = float(atr) if atr is not None and atr == atr and atr > 0 else close * 0.02

    def num(key):
        v = latest.get(key)
        return float(v) if v is not None and v == v else None

    sma20, sma50 = num("sma_20"), num("sma_50")
    r2 = lambda x: round(x, 2)
    return {
        "entry_zones": [
            {"label": "Aggressive", "price": r2(close - 0.5 * atr)},
            {"label": "Conservative", "price": r2(min(sma20 if sma20 else close, close - 1.0 * atr))},
            {"label": "Deep Value", "price": r2(min(sma50 if sma50 else close, close - 2.0 * atr))},
        ],
        "targets": [
            {"label": "Target 1", "price": r2(close + 1.5 * atr)},
            {"label": "Target 2", "price": r2(close + 3.0 * atr)},
            {"label": "Target 3", "price": r2(close + 4.5 * atr)},
        ],
        "stop_losses": [
            {"label": "Tight", "price": r2(close - 1.0 * atr)},
            {"label": "Standard", "price": r2(close - 2.0 * atr)},
            {"label": "Wide (ATR)", "price": r2(close - 3.0 * atr)},
        ],
    }


def generate_overall_recommendation(
    technical_signals: dict,
    fundamental_analysis: dict,
    ml_prediction: dict
) -> dict:
    """Generate overall buy/sell/hold recommendation."""
    score = 0
    reasons = []

    # Technical analysis scoring
    for category, signals in technical_signals.items():
        for signal_type, _ in signals:
            if "BUY" in signal_type or "BULLISH" in signal_type or "OVERSOLD" in signal_type:
                score += 1
                reasons.append(f"Technical: {signal_type}")
            elif "SELL" in signal_type or "BEARISH" in signal_type or "OVERBOUGHT" in signal_type:
                score -= 1
                reasons.append(f"Technical: {signal_type}")

    # Fundamental scoring
    for category, data in fundamental_analysis.items():
        if "assessment" in data:
            for assessment in data["assessment"]:
                if "Undervalued" in assessment or "Strong" in assessment or "Excellent" in assessment or "Good" in assessment or "Low debt" in assessment:
                    score += 0.5
                    reasons.append(f"Fundamental: {assessment}")
                elif "overvalued" in assessment or "concern" in assessment or "High debt" in assessment or "Low liquidity" in assessment:
                    score -= 0.5
                    reasons.append(f"Fundamental: {assessment}")

    # ML prediction scoring - only counted when the model has demonstrated an out-of-sample edge
    ml = ml_prediction or {}
    direction = ml.get("direction")
    if direction in ("UP", "DOWN"):
        if ml.get("reliable"):
            weight = 1 if ml.get("confidence") == "HIGH" else 0.5
            score += weight if direction == "UP" else -weight
            p = ml.get("prob_up")
            reasons.append(f"ML ({ml.get('horizon_days', '?')}-day, P(up)={p:.0%}): {direction} with {ml.get('confidence')} confidence"
                           if p is not None else f"ML: {direction} with {ml.get('confidence')} confidence")
        else:
            reasons.append("ML signal not counted - model has no proven edge yet")
    elif direction == "NEUTRAL":
        reasons.append("ML: no clear directional edge")

    # Final recommendation
    if score >= 3:
        recommendation = "STRONG BUY"
    elif score >= 1:
        recommendation = "BUY"
    elif score >= -1:
        recommendation = "HOLD"
    elif score >= -3:
        recommendation = "SELL"
    else:
        recommendation = "STRONG SELL"

    return {
        "recommendation": recommendation,
        "score": score,
        "reasons": reasons[:10]  # Top 10 reasons
    }
