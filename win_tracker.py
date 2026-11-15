"""Prediction logging and outcome resolution.

A prediction is made from the close of `as_of_date` and judged on the close `horizon` trading
days later, compared with the price at prediction time. NEUTRAL calls are logged but not scored.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

import database
from src import data_fetcher

logger = logging.getLogger(__name__)


def record_prediction(symbol, analysis_result, source="user"):
    """Log the ML call from an analysis result (idempotent per symbol/as-of-date/horizon)."""
    try:
        ml = analysis_result.get("ml_prediction") or {}
        price = (analysis_result.get("quote") or {}).get("price")
        if ml.get("direction") not in ("UP", "DOWN", "NEUTRAL") or ml.get("prob_up") is None or not price:
            return False
        as_of = ml.get("as_of")
        horizon = int(ml.get("horizon_days") or 5)
        approx_check = (pd.Timestamp(as_of) + timedelta(days=int(horizon * 1.5) + 1)).date().isoformat()
        return database.save_prediction(
            symbol, ml["direction"], float(price), approx_check,
            as_of_date=as_of, horizon=horizon, prob_up=float(ml["prob_up"]),
            model_version=ml.get("model_version"), source=source,
        )
    except Exception as e:
        logger.error("could not record prediction for %s: %s", symbol, e)
        return False


def check_pending_outcomes() -> dict:
    """Resolve every prediction whose horizon has elapsed. Returns counts."""
    pending = database.get_pending_predictions()
    stats = {"pending": len(pending), "resolved": 0, "expired": 0, "waiting": 0}
    if not pending:
        return stats

    by_symbol = {}
    for p in pending:
        by_symbol.setdefault(p["symbol"], []).append(p)

    today = pd.Timestamp(datetime.now(timezone.utc).date())
    frames = data_fetcher.download_many(list(by_symbol), "1y")       # one batched request instead of one per symbol
    for symbol, rows in by_symbol.items():
        df = frames.get(symbol)
        if df is None or df.empty:
            logger.warning("no price data for %s", symbol)
            for p in rows:
                if (today - pd.Timestamp(p["as_of_date"])).days > 45:
                    database.expire_prediction(p["id"])
                    stats["expired"] += 1
                else:
                    stats["waiting"] += 1
            continue

        close = df["close"]
        close = close[close.index < today]                    # never score on an in-progress bar
        for p in rows:
            as_of = pd.Timestamp(p["as_of_date"])
            horizon = int(p["horizon"])
            if len(close) == 0 or as_of < close.index[0]:
                database.expire_prediction(p["id"])
                stats["expired"] += 1
                continue
            after = close[close.index > as_of]
            if len(after) < horizon:
                if (today - as_of).days > 45:
                    database.expire_prediction(p["id"])
                    stats["expired"] += 1
                else:
                    stats["waiting"] += 1
                continue
            exit_price = float(after.iloc[horizon - 1])
            base = float(p["price_at_prediction"])
            ret = exit_price / base - 1
            direction = p["predicted_direction"]
            if direction in ("UP", "DOWN"):
                correct = bool((exit_price > base) == (direction == "UP"))
            else:
                correct = None
            database.update_prediction_outcome(p["id"], exit_price, correct, ret)
            stats["resolved"] += 1
    logger.info("outcome check: %s", stats)
    return stats


def start_background_checker(interval_hours=6):
    """In-process safety net: heartbeat + outcome check every N hours while the web app is awake.

    The primary scheduler is the GitHub Actions workflow (works even when the web service sleeps).
    All work here is idempotent, so running in several gunicorn workers is harmless.
    """
    def loop():
        time.sleep(30)
        while True:
            try:
                from jobs.daily import run
                run(mode="light", source="web-thread")
            except Exception as e:
                logger.error("background checker error: %s", e)
            time.sleep(interval_hours * 3600)

    threading.Thread(target=loop, daemon=True, name="quant-bot-checker").start()
    logger.info("background checker started (every %sh)", interval_hours)


def get_accuracy_data():
    """Everything the /accuracy page needs."""
    stats = database.get_accuracy_stats()
    overall = stats.get("overall", {})
    by_dir = stats.get("by_direction", {})

    def d(name):
        r = by_dir.get(name, {})
        return {"total": r.get("total", 0), "correct": r.get("correct", 0), "win_rate": r.get("win_rate", 0)}

    return {
        "overall": {
            "total": overall.get("total", 0) or 0,
            "correct": overall.get("correct", 0) or 0,
            "win_rate": overall.get("win_rate", 0) or 0,
            "baseline_up_rate": overall.get("baseline_up_rate", 0) or 0,
            "edge": overall.get("edge", 0) or 0,
            "pending": overall.get("pending", 0) or 0,
            "neutral": overall.get("neutral", 0) or 0,
        },
        "by_direction": {"up": d("UP"), "down": d("DOWN")},
        "by_symbol": stats.get("per_symbol", []),
        "by_model": stats.get("by_model", []),
        "recent_predictions": stats.get("recent", []),
        "training_runs": database.get_training_runs(10),
    }
