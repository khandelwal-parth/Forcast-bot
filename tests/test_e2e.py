"""End-to-end: scheduled job (train -> promote -> log predictions) and the Flask app. Needs TEST_DATABASE_URL."""

import os

import numpy as np
import pandas as pd
import pytest

from tests.test_pipeline import make_prices

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

END = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
while END.dayofweek >= 5:
    END -= pd.Timedelta(days=1)


@pytest.fixture(scope="module")
def world():
    os.environ["DATABASE_URL"] = URL
    import database
    conn = database.get_connection()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS predictions, alert_log, heartbeat, model_registry, training_runs, watchlist CASCADE")
    conn.commit(); conn.close()
    database._schema_ready = False

    prices, bench = make_prices(n_days=1400, n_tickers=40, signal=0.08, seed=5, end=END)
    vix = bench.copy(); vix["close"] = 18 + 5 * np.sin(np.arange(len(vix)) / 40)
    frames = {**prices, "SPY": bench, "^VIX": vix, "^NSEI": bench, "^INDIAVIX": vix}
    return database, frames


@pytest.fixture(scope="module")
def job(world, tmp_path_factory):
    database, frames = world
    from jobs import daily
    from src import model, ml_service
    model_fast = {k: model.CANDIDATES[k] for k in ("logreg", "hgb_shallow")}
    mp = pytest.MonkeyPatch()
    mp.setattr(model, "CANDIDATES", model_fast)
    mp.setattr(daily, "build_universe", lambda extra=None: list(k for k in frames if k.startswith("T")))
    mp.setattr(daily.data_fetcher, "download_many", lambda syms, period="15y", chunk=25: {s: frames[s] for s in syms if s in frames})
    mp.setattr(ml_service, "LOCAL_CACHE", str(tmp_path_factory.mktemp("m") / "champion.joblib"))
    yield daily
    mp.undo()


def test_full_run_trains_promotes_and_logs_predictions(world, job):
    database, _ = world
    out = job.run(mode="full", source="pytest")
    assert out["heartbeat"] is True
    assert out["training"]["decision"] == "promoted", out
    assert out["predictions_logged"] >= 35
    champ = database.get_champion(include_blob=False)
    assert champ["version"] == out["training"]["version"]
    runs = database.get_training_runs(5)
    assert runs[0]["decision"] == "promoted" and runs[0]["n_tickers"] == 40
    with database._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT as_of_date) AS d FROM predictions WHERE source='universe'")
        r = cur.fetchone()
    assert r["n"] >= 35 and r["d"] == 1


def test_second_run_skips_training_and_is_idempotent(world, job):
    database, _ = world
    with database._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM predictions")
        before = cur.fetchone()["n"]
    out = job.run(mode="full", source="pytest")
    assert out["training"]["decision"] == "skipped"
    with database._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM predictions")
        assert cur.fetchone()["n"] == before             # same as_of date -> no duplicates
        cur.execute("SELECT COUNT(*) AS n FROM heartbeat")
        assert cur.fetchone()["n"] == 2                  # but every run is a heartbeat


def test_forced_retrain_with_worse_model_is_rejected(world, job, monkeypatch):
    database, _ = world
    real = job.train_champion_candidate

    def worse(ds, **k):
        b = real(ds, **k)
        b["metrics"]["summary"]["skill"] = -0.05
        b["version"] += "-bad"
        return b

    monkeypatch.setattr(job, "train_champion_candidate", worse)
    before = database.get_champion(include_blob=False)["version"]
    out = job.run(mode="full", source="pytest", force_retrain=True)
    assert out["training"]["decision"] == "rejected"
    assert database.get_champion(include_blob=False)["version"] == before
    assert database.get_training_runs(1)[0]["decision"] == "rejected"


def test_flask_app_endpoints(world, job, monkeypatch):
    database, frames = world
    import importlib
    monkeypatch.setenv("ENABLE_BACKGROUND_THREAD", "0")
    monkeypatch.setenv("CRON_TOKEN", "s3cret")
    import app as app_module
    importlib.reload(app_module)
    from src import data_fetcher
    monkeypatch.setattr(app_module.data_fetcher, "resolve_symbol", lambda s: s.upper())
    monkeypatch.setattr(app_module.data_fetcher, "get_history_df", lambda s, p="5y": frames[s].copy())
    monkeypatch.setattr(app_module.data_fetcher, "get_market_context",
                        lambda s, p="5y": __import__("src.features", fromlist=["x"]).build_market_context(frames["SPY"], frames["^VIX"]))
    monkeypatch.setattr(app_module.data_fetcher, "get_company_info",
                        lambda s: {"Name": "Test Co", "Sector": "Tech", "PERatio": "N/A", "DebtToEquity": "N/A",
                                   "CurrentRatio": "N/A", "ProfitMargin": 0.3, "ReturnOnEquityTTM": 0.2})
    client = app_module.app.test_client()

    r = client.get("/analyze/T3")
    body = r.get_json()
    assert r.status_code == 200 and body["success"], body
    d = body["data"]
    assert d["ml_prediction"]["direction"] in ("UP", "DOWN", "NEUTRAL")
    assert 0 < d["ml_prediction"]["prob_up"] < 1 and d["ml_prediction"]["horizon_days"] == 5
    assert d["model_metrics"]["available"] and d["model_metrics"]["n_tickers"] == 40
    assert d["recommendation"]["recommendation"] in ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")
    assert d["fundamentals"]["financial_health"]["assessment"] == []          # N/A debt no longer = "Low debt"
    assert d["price_levels"]["stop_losses"][0]["price"] < d["quote"]["price"] < d["price_levels"]["targets"][0]["price"]

    import time
    for _ in range(40):                                                         # background persistence
        if "T3" in database.get_watchlist_symbols():
            break
        time.sleep(0.1)
    assert "T3" in database.get_watchlist_symbols()

    assert client.get("/accuracy").status_code == 200
    html = client.get("/accuracy").get_data(as_text=True)
    assert "Always-UP Baseline" in html and "Model Training History" in html
    api = client.get("/api/accuracy").get_json()
    assert api["success"] and "baseline_up_rate" in api["data"]["overall"]
    assert client.get("/api/model").get_json()["data"]["champion"]["available"] is True
    assert client.get("/healthz").get_json()["ok"] is True
    assert client.get("/historical/T3").get_json()["success"]

    assert client.get("/api/cron").status_code == 403
    assert client.get("/api/cron?token=wrong").status_code == 403
    assert client.get("/api/cron?token=s3cret").get_json()["started"] is True
    monkeypatch.delenv("CRON_TOKEN")
    assert client.get("/api/cron?token=x").status_code == 404

    assert client.get("/").status_code == 200
