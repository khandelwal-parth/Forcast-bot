# Quant Bot

An AI-powered stock analysis system that delivers institutional-grade analysis in seconds.

🔗 **Live Demo: [QUANT-BOT](https://quant-bot-ma3u.onrender.com/)**

## What It Does

Quant Bot analyzes any US stock and provides:

- **Buy/Sell recommendations** with confidence scores
- **Entry and exit points** with specific price levels
- **Technical analysis** (RSI, MACD, Moving Averages, Bollinger Bands)
- **Fundamental analysis** (P/E, PEG, margins, ROE, debt levels)
- **ML 5-day outlook** (probability of a higher price) that only counts toward the recommendation once it has proven an out-of-sample edge
- **Risk assessment** with stop-loss suggestions

## Features

| Feature | Description |
|---------|-------------|
| Technical Analysis | 15+ indicators including RSI, MACD, SMA, Bollinger Bands |
| Fundamental Analysis | Valuation, profitability, growth, and financial health metrics |
| ML Predictions | Calibrated gradient-boosting / logistic ensemble trained on ~100 stocks and up to 15 years, validated with purged walk-forward testing |
| Trading Signals | Clear buy/sell signals with composite scoring |
| Price Targets | Specific entry zones, take-profit levels, and stop-losses |

## What You Get

For any stock, Quant Bot tells you:

1. **Should you buy, sell, or hold?** — Clear recommendation
2. **At what price?** — Specific entry zones
3. **When to exit?** — Profit targets and stop-loss levels
4. **Why?** — Key reasons based on technicals, fundamentals, and ML

## How the ML keeps improving

A scheduled GitHub Actions job (`.github/workflows/daily.yml`, runs `python -m jobs.daily`) does this every weekday:

1. Writes a heartbeat row to Supabase (also runs on weekends, so the free project never auto-pauses).
2. Resolves past predictions against real prices (5 trading days later) and updates the accuracy page.
3. Downloads up to 15 years of data for the universe: a base list plus every ticker users have analysed.
4. Retrains weekly (or sooner if the universe grew or live accuracy degraded), picks the best model by purged
   walk-forward validation, calibrates probabilities, and promotes it unless it is clearly worse than the champion.
5. Logs a fresh prediction for every ticker, so live out-of-sample accuracy builds up automatically.

### Setup

1. Supabase: copy the **Connection pooling** string (Project Settings -> Database). It supports IPv4, which GitHub Actions needs.
2. GitHub repo -> Settings -> Secrets and variables -> Actions: add `DATABASE_URL` (and optionally `DISCORD_WEBHOOK_URL`,
   `SUPABASE_URL`, `SUPABASE_ANON_KEY`).
3. Actions tab -> "Quant Bot daily run" -> Run workflow -> `force_retrain` to train the first model immediately.
4. Render: set `DATABASE_URL` (same string) and optionally `CRON_TOKEN`; an external pinger hitting
   `/api/cron?token=...` daily is a second keep-alive (and wakes the free web service).

Local: `python -m jobs.daily --force-retrain`, `pytest`.

## Disclaimer

For educational purposes only. Not financial advice. Always do your own research.
