"""Stock data fetcher - yfinance (primary) with Alpha Vantage fallback for daily bars."""

import os
import time
import logging
import threading

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
OHLCV = ["open", "high", "low", "close", "volume"]

_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 15 * 60


def _cached(key, ttl, fn):
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1].copy() if isinstance(hit[1], pd.DataFrame) else hit[1]
    val = fn()
    with _cache_lock:
        _cache[key] = (now, val)
        if len(_cache) > 300:                       # crude size cap
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
                _cache.pop(k, None)
    return val.copy() if isinstance(val, pd.DataFrame) else val


def _clean_frame(hist: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance frame: lowercase OHLCV columns, tz-naive date index, no NaN rows."""
    df = hist.rename(columns=str.lower)[OHLCV].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.apply(pd.to_numeric, errors="coerce").dropna()


def resolve_symbol(symbol: str) -> str:
    """Resolve a bare ticker to an exchange-qualified symbol via Yahoo search."""
    if "." in symbol or symbol.startswith("^"):
        return symbol.upper()
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
        equities = [q for q in quotes if q.get("quoteType") == "EQUITY"]

        best_match = None
        for q in equities:
            q_symbol = q.get("symbol", "").upper()
            if q_symbol == symbol.upper() or q_symbol.startswith(f"{symbol.upper()}."):
                if q_symbol.endswith(".NS"):
                    return q_symbol
                if best_match is None:
                    best_match = q_symbol
        if best_match:
            return best_match
        if equities:
            return equities[0]["symbol"].upper()
        for q in quotes:
            if q.get("quoteType") in ("EQUITY", "ETF"):
                return q["symbol"].upper()
    except Exception as e:
        logger.warning("Symbol search failed: %s", e)
    return symbol.upper()


def get_history_df(symbol: str, period: str = "5y", retries: int = 3) -> pd.DataFrame:
    """OHLCV DataFrame (lowercase columns, ascending date index) for one symbol."""
    def fetch():
        import yfinance as yf
        last_err = None
        for attempt in range(retries):
            try:
                hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
                if hist is not None and not hist.empty:
                    return _clean_frame(hist)
                last_err = ValueError(f"No data returned for {symbol}")
            except Exception as e:
                last_err = e
            time.sleep(1.5 * (attempt + 1))
        raise last_err

    try:
        return _cached(("hist", symbol, period), CACHE_TTL, fetch)
    except Exception as e:
        logger.warning("yfinance failed for %s (%s); trying Alpha Vantage", symbol, e)
        return parse_to_dataframe(_get_av_daily_data(symbol, "full" if period not in ("1y", "6mo", "3mo") else "compact"))


def download_many(symbols: list, period: str = "15y", chunk: int = 25) -> dict:
    """Batch download OHLCV for many symbols. Returns {symbol: DataFrame}; failures are skipped."""
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        for attempt in range(3):
            try:
                raw = yf.download(batch, period=period, auto_adjust=True, group_by="ticker",
                                  progress=False, threads=True)
                break
            except Exception as e:
                logger.warning("batch download failed (%s), retry %d", e, attempt + 1)
                raw = None
                time.sleep(3 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for sym in batch:
            try:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                sub = _clean_frame(sub)
                if len(sub) > 0:
                    out[sym] = sub
            except Exception:
                continue
        time.sleep(1)
    return out


def get_market_context(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Market-wide feature frame for the benchmark associated with `symbol`."""
    from .features import benchmark_for, build_market_context
    bench_sym, vix_sym = benchmark_for(symbol)

    def build():
        bench = get_history_df(bench_sym, period)
        try:
            vix = get_history_df(vix_sym, period)
        except Exception:
            vix = None
        return build_market_context(bench, vix)

    return _cached(("mkt", bench_sym, period), CACHE_TTL, build)


# ───────────────────────── legacy / compatibility helpers ─────────────────────────

def get_daily_data(symbol: str, output_size: str = "compact") -> dict:
    """Daily bars in Alpha-Vantage dict format (kept for backwards compatibility)."""
    df = get_history_df(symbol, "2y")
    return {idx.strftime("%Y-%m-%d"): {"1. open": r.open, "2. high": r.high, "3. low": r.low,
                                       "4. close": r.close, "5. volume": r.volume}
            for idx, r in df.iterrows()}


def _get_av_daily_data(symbol: str, output_size: str = "compact") -> dict:
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": output_size, "apikey": API_KEY}
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Invalid symbol or API error: {data}")
    return data["Time Series (Daily)"]


def parse_to_dataframe(daily_data: dict) -> pd.DataFrame:
    """Convert Alpha-Vantage style dict to an OHLCV DataFrame."""
    df = pd.DataFrame.from_dict(daily_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = OHLCV
    return df.sort_index()


def _num(x):
    """float(x) or None for missing/invalid values (Yahoo uses None / 'Infinity' / strings)."""
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    except (TypeError, ValueError):
        return None


def get_company_info(symbol: str) -> dict:
    """Company profile + fundamentals. Missing values are the string 'N/A' (as before)."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}

        def g(key, default="N/A"):
            v = info.get(key)
            return default if v is None else v

        pe = _num(info.get("trailingPE"))
        if pe is None:
            pe = _num(info.get("forwardPE"))
        # Yahoo reports debtToEquity as a percentage (150 == 1.5x); convert to a ratio.
        de = _num(info.get("debtToEquity"))
        de = de / 100.0 if de is not None else None

        return {
            "Symbol": symbol,
            "Name": info.get("shortName") or info.get("longName") or symbol,
            "Description": g("longBusinessSummary"),
            "Sector": g("sector"),
            "Industry": g("industry"),
            "PERatio": pe if pe is not None else "N/A",
            "PEGRatio": g("pegRatio"),
            "PriceToBookRatio": g("priceToBook"),
            "ProfitMargin": g("profitMargins", 0),
            "OperatingMarginTTM": g("operatingMargins", 0),
            "ReturnOnEquityTTM": g("returnOnEquity", 0),
            "QuarterlyRevenueGrowthYOY": g("revenueGrowth", 0),
            "QuarterlyEarningsGrowthYOY": g("earningsGrowth", 0),
            "DebtToEquity": de if de is not None else "N/A",
            "CurrentRatio": g("currentRatio"),
            "MarketCapitalization": g("marketCap"),
            "52WeekHigh": g("fiftyTwoWeekHigh"),
            "52WeekLow": g("fiftyTwoWeekLow"),
            "DividendYield": g("dividendYield"),
            "Beta": g("beta"),
        }
    except Exception as e:
        logger.warning("yfinance company info failed (%s), trying Alpha Vantage", e)
        return _get_av_company_info(symbol)


def _get_av_company_info(symbol: str) -> dict:
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": API_KEY}
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()
