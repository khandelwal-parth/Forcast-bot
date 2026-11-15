"""Scheduled job: keeps Supabase active, resolves outcomes, logs predictions, retrains the model.

    python -m jobs.daily                 # full run (weekday evening, after US + India markets close)
    python -m jobs.daily --mode light    # heartbeat + outcome check only (weekends / web thread)
    python -m jobs.daily --force-retrain

Everything is idempotent, so running it twice (or from several places) is harmless.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import database
import win_tracker
from src import data_fetcher, ml_service
from src.features import build_dataset, build_market_context, benchmark_for
from src.model import bundle_to_bytes, predict_latest, train_champion_candidate
from src.universe import build_universe

logger = logging.getLogger("jobs.daily")

RETRAIN_DAYS = 7              # retrain at least weekly
GROWTH_TRIGGER = 1.2          # ...or when the universe is 20% larger than the champion's training set
PROMOTE_TOLERANCE = 0.002     # reject a retrain whose walk-forward skill is clearly worse than the champion's
MAX_CHAMPION_AGE_DAYS = 30    # ...but never keep a champion older than this just because it scored higher
LIVE_DEGRADATION = -0.02      # retrain early if live accuracy trails the always-UP baseline by 2+ points


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))


def supabase_rest_ping() -> bool:
    """Optional second keep-alive signal through Supabase's REST API."""
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return False
    try:
        r = requests.get(f"{url.rstrip('/')}/rest/v1/heartbeat?select=id&limit=1",
                         headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=10)
        return r.status_code < 500
    except Exception:
        return False


def load_markets(symbols, period="15y") -> dict:
    """{symbol: market-context frame} using one download per distinct benchmark / vol index."""
    pairs = {benchmark_for(s) for s in symbols}
    needed = sorted({x for pair in pairs for x in pair})
    frames = data_fetcher.download_many(needed, period)
    ctx = {}
    for bench, vix in pairs:
        if bench in frames:
            ctx[(bench, vix)] = build_market_context(frames[bench], frames.get(vix))
    return {s: ctx[benchmark_for(s)] for s in symbols if benchmark_for(s) in ctx}


def should_retrain(n_tickers_now: int, force: bool = False):
    if force:
        return True, "forced"
    meta = database.get_champion(include_blob=False)
    if not meta:
        return True, "no champion model yet"
    age = datetime.now(timezone.utc) - meta["created_at"]
    if age >= timedelta(days=RETRAIN_DAYS):
        return True, f"champion is {age.days} days old"
    trained_on = ((meta.get("metrics") or {}).get("n_tickers")) or 0
    if trained_on and n_tickers_now >= trained_on * GROWTH_TRIGGER:
        return True, f"universe grew {trained_on} -> {n_tickers_now} tickers"
    live = database.get_live_edge(meta["version"], min_n=300)
    if live and live["edge"] < LIVE_DEGRADATION:
        return True, f"live edge {live['edge']:+.1%} over {live['n']} resolved predictions"
    return False, "champion is fresh"


def retrain(prices: dict, markets: dict, reason: str) -> dict:
    """Train a challenger and promote it unless it is clearly worse than the champion."""
    ds = build_dataset(prices, markets)
    logger.info("training on %d rows from %d tickers (%s)", len(ds), ds["symbol"].nunique() if len(ds) else 0, reason)
    bundle = train_champion_candidate(ds)
    m = bundle["metrics"]
    summ = m["summary"]
    meta = database.get_champion(include_blob=False)

    decision, why = "promoted", f"retrain ({reason})"
    if meta:
        old = ((meta.get("metrics") or {}).get("summary") or {}).get("skill")
        age_days = (datetime.now(timezone.utc) - meta["created_at"]).days
        if old is not None and summ["skill"] < old - PROMOTE_TOLERANCE and age_days < MAX_CHAMPION_AGE_DAYS:
            decision = "rejected"
            why = f"skill {summ['skill']:+.4f} clearly below champion {old:+.4f}; keeping champion"

    if decision == "promoted":
        database.save_model(bundle["version"], bundle_to_bytes(bundle), _jsonable(m), bundle["reliable"])
        database.promote_model(bundle["version"])
        ml_service.set_bundle(bundle)

    database.log_training_run(bundle["version"], decision, why, candidate=bundle["candidate"],
                              reliable=bundle["reliable"], n_rows=m["n_rows"], n_tickers=m["n_tickers"],
                              skill=summ["skill"], auc=summ["auc"], metrics=_jsonable(m))
    logger.info("model %s %s: %s | candidate=%s skill=%+.4f auc=%.4f reliable=%s", bundle["version"], decision,
                why, bundle["candidate"], summ["skill"], summ["auc"], bundle["reliable"])
    return {"version": bundle["version"], "decision": decision, "reason": why, "reliable": bundle["reliable"],
            "skill": summ["skill"], "auc": summ["auc"], "candidate": bundle["candidate"]}


def predict_universe(prices: dict, markets: dict, bundle: dict) -> list:
    """Log a prediction for every ticker so the accuracy tracker grows without any user input."""
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    rows, preds = [], []
    for sym, df in prices.items():
        if len(df) < 320 or (today - df.index[-1]).days > 6:
            continue
        try:
            p = predict_latest(bundle, df, markets.get(sym))
        except Exception as e:
            logger.warning("predict failed for %s: %s", sym, e)
            continue
        if p.get("direction") in (None, "UNKNOWN"):
            continue
        horizon = p["horizon_days"]
        rows.append({
            "symbol": sym, "direction": p["direction"], "price": float(df["close"].iloc[-1]),
            "check_date": (pd.Timestamp(p["as_of"]) + timedelta(days=int(horizon * 1.5) + 1)).date().isoformat(),
            "as_of_date": p["as_of"], "horizon": horizon, "prob_up": p["prob_up"],
            "model_version": p["model_version"], "source": "universe",
        })
        preds.append({"symbol": sym, **p})
    inserted = database.save_predictions_bulk(rows)
    logger.info("logged %d new predictions (%d computed)", inserted, len(rows))
    return preds


def run(mode: str = "full", source: str = "github-actions", force_retrain: bool = False) -> dict:
    summary = {"mode": mode, "source": source, "started": datetime.now(timezone.utc).isoformat()}

    summary["heartbeat"] = database.heartbeat(f"{source}:{mode}")
    summary["rest_ping"] = supabase_rest_ping()
    summary["outcomes"] = win_tracker.check_pending_outcomes()
    if mode == "light":
        return summary

    universe = build_universe(database.get_watchlist_symbols())
    prices = data_fetcher.download_many(universe, "15y")
    summary["tickers_downloaded"] = len(prices)
    if len(prices) < 30:
        summary["error"] = f"only {len(prices)} tickers downloaded - skipping training/prediction"
        logger.error(summary["error"])
        return summary
    markets = load_markets(list(prices))

    do_train, why = should_retrain(len(prices), force_retrain)
    if do_train:
        try:
            summary["training"] = retrain(prices, markets, why)
        except Exception as e:
            logger.exception("training failed")
            summary["training"] = {"decision": "error", "reason": str(e)}
    else:
        summary["training"] = {"decision": "skipped", "reason": why}

    bundle = ml_service.get_bundle(force=True)
    if bundle is None:
        summary["error"] = "no model available - no predictions logged"
        return summary
    preds = predict_universe(prices, markets, bundle)
    summary["predictions_logged"] = len(preds)

    try:
        from discord_alerts import send_daily_digest
        send_daily_digest(summary, preds, bundle)
    except Exception as e:
        logger.warning("digest failed: %s", e)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "light"], default="full")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--source", default="github-actions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not database.db_configured():
        print("DATABASE_URL is not set - nothing to keep alive.", file=sys.stderr)
        sys.exit(2)
    summary = run(args.mode, args.source, args.force_retrain)
    print(json.dumps(_jsonable(summary), indent=2))
    if not summary.get("heartbeat"):
        print("Heartbeat write FAILED - the database may be paused or unreachable.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
