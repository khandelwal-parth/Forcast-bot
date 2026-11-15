"""Integration tests against a real Postgres. Set TEST_DATABASE_URL to run (skipped otherwise).

    TEST_DATABASE_URL=postgresql://postgres:pw@localhost/qbtest pytest tests/test_db_integration.py
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="module")
def db():
    os.environ["DATABASE_URL"] = URL
    import database
    conn = database.get_connection()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS predictions, alert_log, heartbeat, model_registry, training_runs, watchlist CASCADE")
    conn.commit()
    conn.close()
    database._schema_ready = False
    database.ensure_schema()
    return database


def test_schema_is_idempotent_and_rls_enabled(db):
    db._schema_ready = False
    db.ensure_schema()
    db.ensure_schema()
    with db._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname = ANY(%s)", (db.TABLES,))
        rows = {r["relname"]: r["relrowsecurity"] for r in cur.fetchall()}
    assert set(rows) == set(db.TABLES) and all(rows.values())


def test_heartbeat_writes_rows(db):
    assert db.heartbeat("test", "hello") is True
    assert db.heartbeat("test") is True
    last = db.last_heartbeat()
    assert last["source"] == "test"
    with db._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM heartbeat")
        assert cur.fetchone()["n"] == 2


def test_missing_database_url_never_raises(db, monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    assert db.heartbeat("x") is False
    assert db.get_accuracy_stats()["overall"]["total"] == 0
    assert db.get_champion() is None
    monkeypatch.setenv("DATABASE_URL", URL)


def test_unreachable_database_never_raises(db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:pw@127.0.0.1:1/nope?connect_timeout=1")
    assert db.heartbeat("x") is False
    monkeypatch.setenv("DATABASE_URL", URL)


def test_prediction_dedup_and_bulk(db):
    d0 = (date.today() - timedelta(days=20)).isoformat()
    assert db.save_prediction("AAPL", "UP", 100.0, as_of_date=d0, horizon=5, prob_up=0.6, model_version="v1") is True
    assert db.save_prediction("AAPL", "DOWN", 101.0, as_of_date=d0, horizon=5, prob_up=0.4, model_version="v1") is False
    rows = [{"symbol": s, "direction": dr, "price": 50.0, "as_of_date": d0, "horizon": 5, "prob_up": 0.5,
             "model_version": "v1"} for s, dr in (("MSFT", "UP"), ("NVDA", "DOWN"), ("KO", "NEUTRAL"), ("AAPL", "UP"))]
    assert db.save_predictions_bulk(rows) == 3        # AAPL already existed
    assert db.save_predictions_bulk(rows) == 0        # fully idempotent


def test_outcome_resolution_end_to_end(db, monkeypatch):
    import win_tracker
    d0 = pd.Timestamp(date.today() - timedelta(days=20))
    idx = pd.bdate_range(d0 - pd.Timedelta(days=30), periods=40)
    prices = {   # all four go up 10% after as_of except NVDA (down)
        "AAPL": 100.0, "MSFT": 50.0, "NVDA": 50.0, "KO": 50.0,
    }

    def fake_history(symbol):
        base = prices[symbol]
        s = pd.Series(base, index=idx, dtype=float)
        after = s.index > d0
        s[after] = base * (0.9 if symbol == "NVDA" else 1.1)
        return pd.DataFrame({"close": s, "open": s, "high": s, "low": s, "volume": 1.0})

    monkeypatch.setattr(win_tracker.data_fetcher, "download_many",
                        lambda syms, period="1y", chunk=25: {s: fake_history(s) for s in syms})
    stats = win_tracker.check_pending_outcomes()
    assert stats["resolved"] == 4 and stats["pending"] == 4

    with db._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT symbol, predicted_direction, was_correct, actual_return, checked_at FROM predictions")
        got = {r["symbol"]: r for r in cur.fetchall()}
    assert got["AAPL"]["was_correct"] is True                 # predicted UP, went up
    assert got["MSFT"]["was_correct"] is True
    assert got["NVDA"]["was_correct"] is True                 # predicted DOWN, went down
    assert got["KO"]["was_correct"] is None                   # NEUTRAL is not scored...
    assert got["KO"]["checked_at"] is not None                # ...but is resolved (not pending forever)
    assert got["AAPL"]["actual_return"] == pytest.approx(0.1, abs=1e-4)
    assert win_tracker.check_pending_outcomes()["pending"] == 0


def test_accuracy_stats_use_baseline(db):
    st = db.get_accuracy_stats()
    o = st["overall"]
    assert o["total"] == 3 and o["correct"] == 3 and o["win_rate"] == 100.0
    assert o["baseline_up_rate"] == pytest.approx(66.7, abs=0.1)     # 2 of 3 rose
    assert o["edge"] == pytest.approx(33.3, abs=0.1)
    assert o["neutral"] == 1 and o["pending"] == 0
    assert st["by_direction"]["UP"]["total"] == 2 and st["by_direction"]["DOWN"]["total"] == 1
    assert st["by_model"][0]["model_version"] == "v1"
    rec = {r["symbol"]: r for r in st["recent"]}
    assert rec["NVDA"]["actual_direction"] == "DOWN" and rec["AAPL"]["actual_direction"] == "UP"
    import win_tracker
    data = win_tracker.get_accuracy_data()
    assert data["by_direction"]["up"]["win_rate"] == 100.0


def test_not_yet_elapsed_horizon_stays_pending(db, monkeypatch):
    import win_tracker
    d1 = (date.today() - timedelta(days=2)).isoformat()
    db.save_prediction("TSLA", "UP", 10.0, as_of_date=d1, horizon=5, prob_up=0.6, model_version="v1")
    idx = pd.bdate_range(end=pd.Timestamp(date.today()) - pd.Timedelta(days=1), periods=30)
    monkeypatch.setattr(win_tracker.data_fetcher, "download_many",
                        lambda syms, period="1y", chunk=25: {s: pd.DataFrame({"close": np.linspace(9, 11, 30)}, index=idx) for s in syms})
    stats = win_tracker.check_pending_outcomes()
    assert stats["resolved"] == 0 and stats["waiting"] == 1


def test_model_registry_champion_flow(db):
    blob = os.urandom(200_000)
    assert db.save_model("v1", blob, {"summary": {"skill": 0.003}, "n_tickers": 90}, True)
    assert db.save_model("v2", b"x" * 10, {"summary": {"skill": 0.001}}, False)
    assert db.promote_model("v1")
    champ = db.get_champion()
    assert champ["version"] == "v1" and champ["blob"] == blob and champ["reliable"] is True
    assert db.get_champion(include_blob=False)["metrics"]["n_tickers"] == 90
    assert db.promote_model("v2") and db.get_champion()["version"] == "v2"
    with db._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM model_registry WHERE is_champion")
        assert cur.fetchone()["n"] == 1


def test_training_runs_watchlist_alerts_live_edge(db):
    assert db.log_training_run("v1", "promoted", "first", candidate="blend", reliable=True, n_rows=1000,
                               n_tickers=90, skill=0.002, auc=0.52, metrics={"a": 1})
    assert db.get_training_runs(5)[0]["version"] == "v1"
    db.touch_symbol("infy.ns"); db.touch_symbol("INFY.NS"); db.touch_symbol("AAPL")
    syms = db.get_watchlist_symbols()
    assert syms.count("INFY.NS") == 1 and set(syms) == {"INFY.NS", "AAPL"}
    assert db.was_alert_sent_recently("AAPL", "buy_AAPL") is False
    db.log_alert("AAPL", "buy_AAPL")
    assert db.was_alert_sent_recently("AAPL", "buy_AAPL", within_hours=6) is True
    assert db.get_live_edge("v1", min_n=300) is None          # too few samples to judge
    live = db.get_live_edge("v1", min_n=3)
    assert live["n"] == 3 and live["acc"] == 1.0


def test_light_job_run_is_idempotent_keepalive(db, monkeypatch):
    from jobs import daily
    monkeypatch.setattr(daily.win_tracker.data_fetcher, "download_many", lambda syms, period="1y", chunk=25: {})
    out = daily.run(mode="light", source="pytest")
    assert out["heartbeat"] is True and "outcomes" in out
    with db._db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM heartbeat WHERE source = 'pytest:light'")
        assert cur.fetchone()["n"] == 1
