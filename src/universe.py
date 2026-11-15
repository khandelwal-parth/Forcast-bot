"""Training / daily-prediction universe: a fixed base list plus symbols users have analysed."""

BASE_UNIVERSE = [
    # US mega/large caps across sectors
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE", "CRM", "AMD",
    "INTC", "CSCO", "QCOM", "TXN", "IBM", "NFLX", "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK",
    "V", "MA", "AXP", "BRK-B", "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "AMGN",
    "WMT", "COST", "HD", "LOW", "TGT", "NKE", "MCD", "SBUX", "KO", "PEP", "PG", "CL", "PM", "MO",
    "XOM", "CVX", "COP", "SLB", "CAT", "DE", "BA", "GE", "HON", "UPS", "LMT", "RTX", "MMM",
    "NEE", "DUK", "SO", "LIN", "SHW", "FCX", "NEM", "AMT", "PLD", "DIS", "CMCSA", "VZ", "T",
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV",
    # India (NSE) large caps
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS",
    "LT.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "NTPC.NS",
    "ONGC.NS",
]

MAX_USER_SYMBOLS = 60


def build_universe(extra_symbols=None) -> list:
    """Base list + most recently analysed user symbols (capped), de-duplicated, order preserved."""
    extra = [s.upper() for s in (extra_symbols or [])][:MAX_USER_SYMBOLS]
    seen, out = set(), []
    for s in BASE_UNIVERSE + extra:
        if s not in seen and not s.startswith("^"):
            seen.add(s)
            out.append(s)
    return out
