"""Training, walk-forward evaluation, calibration and inference for the pooled ML model.

The model predicts P(price is higher in HORIZON trading days). It is trained on many tickers
at once, selected by purged walk-forward validation, probability-calibrated on out-of-sample
predictions, and only trusted (``reliable``) if it shows a real edge over the base rate.
"""

import io
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS, HORIZON, REQUIRED_COLS, build_features

logger = logging.getLogger(__name__)

N_FOLDS = 5
TEST_FRAC = 0.35          # last 35% of dates are used for walk-forward testing
EMBARGO = HORIZON + 2     # gap between train end and test start (labels look HORIZON days ahead)
MARGIN = 0.04             # UP if p >= base+MARGIN, DOWN if p <= base-MARGIN, else NEUTRAL
MIN_AUC = 0.51            # minimum mean AUC to call the model "reliable"
MIN_POSITIVE_FOLDS = 3    # of N_FOLDS must beat the constant base-rate predictor


class BlendModel:
    """Average of several probabilistic models (kept as a top-level class so it pickles)."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def _logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=500))


def _hgb_shallow():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                          min_samples_leaf=300, l2_regularization=5.0, random_state=42)


def _hgb_mid():
    return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.03, max_iter=300,
                                          min_samples_leaf=500, l2_regularization=10.0, random_state=42)


CANDIDATES = {
    "logreg": _logreg,
    "hgb_shallow": _hgb_shallow,
    "hgb_mid": _hgb_mid,
    "blend": lambda: BlendModel([_logreg(), _hgb_shallow()]),
}


# ───────────────────────────── evaluation ─────────────────────────────

def walk_forward_folds(dates: np.ndarray, n_folds: int = N_FOLDS, test_frac: float = TEST_FRAC,
                       embargo: int = EMBARGO):
    """Yield (train_end_date, test_start_date, test_end_date_exclusive_or_None).

    Expanding window. Training rows are at least `embargo` dates before the test window so
    their forward-looking labels never overlap test data.
    """
    u = np.sort(pd.unique(dates))
    n = len(u)
    start = int(n * (1 - test_frac))
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        te0 = edges[i]
        if te0 - embargo - 1 < 50:
            continue
        te1 = edges[i + 1]
        yield u[te0 - embargo - 1], u[te0], (u[te1] if te1 < n else None)


def _metrics(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    """Quality of probabilities p against outcomes y, relative to simple baselines."""
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    ll = log_loss(y, p, labels=[0, 1])
    base_ll = log_loss(y, np.full_like(p, base_rate), labels=[0, 1])
    out = {
        "n": int(len(y)),
        "logloss": float(ll),
        "baseline_logloss": float(base_ll),
        "skill": float(1 - ll / base_ll),                 # >0 means better than predicting the base rate
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
        "baseline_acc": float(y.mean()),                  # accuracy of "always UP"
        "acc": float(((p > 0.5) == (y == 1)).mean()),
    }
    sel = (p >= base_rate + MARGIN) | (p <= base_rate - MARGIN)
    out["coverage"] = float(sel.mean())
    if sel.sum() >= 30:
        pred_up = p[sel] >= base_rate + MARGIN
        out["directional_acc"] = float((pred_up == (y[sel] == 1)).mean())
        out["directional_baseline"] = float(y[sel].mean())          # always-UP on the same rows
        out["directional_edge"] = out["directional_acc"] - out["directional_baseline"]
    return out


def walk_forward(ds: pd.DataFrame, make_model, feature_cols=FEATURE_COLS, n_folds: int = N_FOLDS):
    """Evaluate a model constructor with purged walk-forward folds.

    Returns (fold_metrics, oof_frame, last_fold_fit) where oof_frame has columns
    fold, y, p_raw for every out-of-sample row and last_fold_fit = (model, X_test, y_test).
    """
    dates = ds["date"].values
    X_all = ds[feature_cols]
    y_all = ds["y"].values
    fold_metrics, oof_parts, last = [], [], None

    for k, (train_end, te0, te1) in enumerate(walk_forward_folds(dates, n_folds)):
        tr = dates <= train_end
        te = (dates >= te0) if te1 is None else ((dates >= te0) & (dates < te1))
        if tr.sum() < 5000 or te.sum() < 500:
            continue
        model = make_model()
        model.fit(X_all[tr], y_all[tr])
        p = model.predict_proba(X_all[te])[:, 1]
        base = float(y_all[tr].mean())
        m = _metrics(y_all[te], p, base)
        m["fold"] = k
        m["train_rows"] = int(tr.sum())
        m["test_start"] = str(pd.Timestamp(te0).date())
        fold_metrics.append(m)
        oof_parts.append(pd.DataFrame({"fold": k, "y": y_all[te], "p_raw": p}))
        last = (model, X_all[te], y_all[te])

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(columns=["fold", "y", "p_raw"])
    return fold_metrics, oof, last


def _summarize(fold_metrics: list) -> dict:
    if not fold_metrics:
        return {"folds": 0, "skill": -1.0, "auc": 0.5, "positive_folds": 0}
    return {
        "folds": len(fold_metrics),
        "skill": float(np.mean([m["skill"] for m in fold_metrics])),
        "auc": float(np.mean([m["auc"] for m in fold_metrics])),
        "acc": float(np.mean([m["acc"] for m in fold_metrics])),
        "baseline_acc": float(np.mean([m["baseline_acc"] for m in fold_metrics])),
        "positive_folds": int(sum(m["skill"] > 0 for m in fold_metrics)),
    }


# ───────────────────────────── training ─────────────────────────────

def train_champion_candidate(ds: pd.DataFrame, feature_cols=FEATURE_COLS, candidates=None,
                             n_folds: int = N_FOLDS) -> dict:
    """Select the best candidate by walk-forward skill, calibrate it, fit on all data.

    Returns a "bundle" dict (see predict_latest) containing the model, calibrator and metrics.
    """
    candidates = candidates or CANDIDATES
    # A feature with no data at all (e.g. a volatility index that failed to download) would crash
    # gradient boosting's binning - drop it for this model; the bundle records what was used.
    feature_cols = [c for c in feature_cols if ds[c].notna().mean() >= 0.05]
    if len(ds) < 20000 or ds["symbol"].nunique() < 10:
        raise ValueError(f"Not enough data to train ({len(ds)} rows, {ds['symbol'].nunique()} tickers)")

    results = {}
    for name, make in candidates.items():
        folds, oof, last = walk_forward(ds, make, feature_cols, n_folds)
        summ = _summarize(folds)
        results[name] = {"summary": summ, "folds": folds, "oof": oof, "last": last}
        logger.info("candidate %-12s skill=%+.4f auc=%.4f folds+=%d/%d", name, summ["skill"],
                    summ["auc"], summ["positive_folds"], summ["folds"])

    best = max(results, key=lambda n: results[n]["summary"]["skill"])
    r = results[best]
    oof = r["oof"]
    if oof.empty:
        raise ValueError("Walk-forward produced no out-of-sample predictions")

    # Honest holdout: calibrator fitted on earlier folds only, scored on the final fold.
    last_fold = oof["fold"].max()
    earlier, final = oof[oof["fold"] < last_fold], oof[oof["fold"] == last_fold]
    holdout = {}
    if len(earlier) > 1000 and len(final) > 500:
        iso_h = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(earlier["p_raw"], earlier["y"])
        holdout = _metrics(final["y"].values, iso_h.predict(final["p_raw"]), float(earlier["y"].mean()))

    # Production calibrator: all out-of-sample predictions.
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(oof["p_raw"], oof["y"])

    # Permutation importance of the last-fold model (works for any model type).
    importance = {}
    try:
        model_k, X_te, y_te = r["last"]
        idx = np.random.RandomState(0).choice(len(X_te), size=min(15000, len(X_te)), replace=False)
        pi = permutation_importance(model_k, X_te.iloc[idx], y_te[idx], scoring="neg_log_loss",
                                    n_repeats=2, random_state=0)
        s = pd.Series(pi.importances_mean, index=feature_cols).sort_values(ascending=False)
        importance = {k: float(v) for k, v in s.head(10).items()}
    except Exception as e:                                   # importance is cosmetic; never fail training on it
        logger.warning("permutation importance failed: %s", e)

    # Final model on all labelled data.
    final_model = candidates[best]()
    final_model.fit(ds[feature_cols], ds["y"].values)
    base_rate = float(ds["y"].mean())

    summ = r["summary"]
    reliable = bool(summ["auc"] >= MIN_AUC and summ["skill"] > 0 and summ["positive_folds"] >= MIN_POSITIVE_FOLDS)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M")

    return {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "candidate": best,
        "model": final_model,
        "calibrator": calibrator,
        "feature_cols": list(feature_cols),
        "horizon": HORIZON,
        "base_rate": base_rate,
        "margin": MARGIN,
        "reliable": reliable,
        "train_end": str(pd.Timestamp(ds["date"].max()).date()),
        "metrics": {
            "summary": summ,
            "holdout": holdout,
            "folds": r["folds"],
            "all_candidates": {n: results[n]["summary"] for n in results},
            "importance": importance,
            "n_rows": int(len(ds)),
            "n_tickers": int(ds["symbol"].nunique()),
        },
    }


# ───────────────────────────── persistence ─────────────────────────────

def bundle_to_bytes(bundle: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.getvalue()


def bundle_from_bytes(blob: bytes) -> dict:
    return joblib.load(io.BytesIO(bytes(blob)))


# ───────────────────────────── inference ─────────────────────────────

def predict_latest(bundle: dict, df: pd.DataFrame, market: pd.DataFrame = None) -> dict:
    """Probability / direction for the most recent bar of `df`."""
    cols = bundle["feature_cols"]
    f = build_features(df, market)
    row = f.iloc[[-1]][cols]
    # Same warm-up rule as training: rows without long-window features were never seen by the model.
    if f.iloc[-1][REQUIRED_COLS].isna().any() or row.isna().mean(axis=1).iloc[0] > 0.5:
        return {"direction": "UNKNOWN", "confidence": "N/A", "prob_up": None,
                "note": "Not enough history to compute features"}

    p_raw = float(bundle["model"].predict_proba(row)[:, 1][0])
    p = float(bundle["calibrator"].predict([p_raw])[0])
    base, margin = bundle["base_rate"], bundle["margin"]
    edge = p - base
    if edge >= margin:
        direction = "UP"
    elif edge <= -margin:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"
    confidence = "LOW" if direction == "NEUTRAL" else ("HIGH" if abs(edge) >= 2 * margin else "MEDIUM")
    return {
        "direction": direction,
        "confidence": confidence,
        "prob_up": round(p, 4),
        "base_rate": round(base, 4),
        "horizon_days": int(bundle["horizon"]),
        "model_version": bundle["version"],
        "reliable": bool(bundle["reliable"]),
        "as_of": str(pd.Timestamp(df.index[-1]).date()),
    }
