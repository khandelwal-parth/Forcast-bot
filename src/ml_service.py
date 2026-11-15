"""Loads the current champion model (Supabase first, local cache as fallback) and serves predictions."""

import os
import time
import logging
import threading

import database
from . import data_fetcher
from .model import bundle_from_bytes, bundle_to_bytes, predict_latest

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 3600
LOCAL_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "champion.joblib")

_state = {"bundle": None, "checked_at": 0.0}
_lock = threading.Lock()


def _read_local():
    try:
        with open(LOCAL_CACHE, "rb") as fh:
            return bundle_from_bytes(fh.read())
    except Exception:
        return None


def _write_local(bundle):
    try:
        os.makedirs(os.path.dirname(LOCAL_CACHE), exist_ok=True)
        with open(LOCAL_CACHE, "wb") as fh:
            fh.write(bundle_to_bytes(bundle))
    except Exception as e:
        logger.debug("could not write local model cache: %s", e)


def _refresh_from_registry():
    """Pull the champion from the registry if it changed. Never raises."""
    try:
        current = _state["bundle"]
        meta = database.get_champion(include_blob=False)
        if meta and (current is None or meta["version"] != current["version"]):
            row = database.get_champion(include_blob=True)
            if row and row.get("blob"):
                bundle = bundle_from_bytes(row["blob"])
                with _lock:
                    _state["bundle"] = bundle
                _write_local(bundle)
                logger.info("loaded champion model %s", bundle["version"])
    except Exception as e:
        logger.error("model refresh failed: %s", e)


def get_bundle(force: bool = False):
    """Current champion bundle or None.

    The registry is re-checked at most once per REFRESH_SECONDS. When a model is already in
    memory the check runs in a background thread, so a slow or paused database never stalls a request.
    """
    now = time.time()
    with _lock:
        bundle, stale = _state["bundle"], now - _state["checked_at"] >= REFRESH_SECONDS
        if bundle is not None and not force and not stale:
            return bundle
        _state["checked_at"] = now

    if bundle is not None and not force:
        threading.Thread(target=_refresh_from_registry, daemon=True, name="model-refresh").start()
        return bundle

    _refresh_from_registry()                       # cold start or forced: synchronous
    with _lock:
        if _state["bundle"] is None:
            _state["bundle"] = _read_local()
        return _state["bundle"]


def set_bundle(bundle):
    """Used by training jobs to make a freshly promoted model available in-process."""
    with _lock:
        _state["bundle"] = bundle
        _state["checked_at"] = time.time()
    _write_local(bundle)


def predict(symbol: str, df) -> dict:
    """ML prediction for the latest bar of `df` (an OHLCV frame for `symbol`)."""
    bundle = get_bundle()
    if bundle is None:
        return {"direction": "UNKNOWN", "confidence": "N/A", "prob_up": None, "reliable": False,
                "note": "ML model is still being trained - check back after the next scheduled run."}
    try:
        market = data_fetcher.get_market_context(symbol)
    except Exception as e:
        logger.warning("market context unavailable for %s: %s", symbol, e)
        market = None
    try:
        return predict_latest(bundle, df, market)
    except Exception as e:
        logger.error("prediction failed for %s: %s", symbol, e)
        return {"direction": "UNKNOWN", "confidence": "N/A", "prob_up": None, "reliable": False,
                "note": "Prediction failed for this symbol"}


def model_info() -> dict:
    """Compact, JSON-safe description of the champion for the API / UI."""
    b = get_bundle()
    if b is None:
        return {"available": False}
    m = b.get("metrics", {})
    summ, hold = m.get("summary", {}), m.get("holdout", {})
    return {
        "available": True,
        "version": b["version"],
        "trained_at": b["trained_at"],
        "train_end": b.get("train_end"),
        "candidate": b["candidate"],
        "horizon_days": b["horizon"],
        "reliable": bool(b["reliable"]),
        "base_rate": round(b["base_rate"], 4),
        "n_rows": m.get("n_rows"),
        "n_tickers": m.get("n_tickers"),
        "skill": summ.get("skill"),
        "auc": summ.get("auc"),
        "positive_folds": summ.get("positive_folds"),
        "folds": summ.get("folds"),
        "holdout_edge": hold.get("directional_edge"),
        "holdout_directional_acc": hold.get("directional_acc"),
        "feature_importance": m.get("importance", {}),
    }
