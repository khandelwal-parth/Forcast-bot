"""PostgreSQL (Supabase) persistence for predictions, model registry, training history and heartbeats.

Design notes
- Nothing connects at import time. The app boots even if Supabase is paused or DATABASE_URL is
  missing; DB functions then log a warning and return a harmless default.
- The schema is created/migrated lazily on first use (idempotent).
- For Supabase use the *pooler* connection string (IPv4, works from GitHub Actions and Render).
"""

import os
import time
import logging
import threading
from contextlib import contextmanager
from functools import wraps

import psycopg2
from psycopg2.extras import RealDictCursor, Json

logger = logging.getLogger(__name__)

_schema_ready = False
_schema_lock = threading.Lock()

TABLES = ["predictions", "alert_log", "heartbeat", "model_registry", "training_runs", "watchlist"]

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        predicted_direction TEXT NOT NULL,
        price_at_prediction REAL,
        actual_next_price REAL,
        was_correct BOOLEAN,
        created_at TIMESTAMP DEFAULT NOW(),
        check_date DATE,
        checked_at TIMESTAMP
    )""",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS as_of_date DATE",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS horizon INTEGER",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS prob_up REAL",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_version TEXT",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'user'",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_return REAL",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_symbol_asof ON predictions (symbol, as_of_date, horizon)",
    "CREATE INDEX IF NOT EXISTS ix_predictions_pending ON predictions (checked_at) WHERE checked_at IS NULL",
    """CREATE TABLE IF NOT EXISTS alert_log (
        id SERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        sent_at TIMESTAMP DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS heartbeat (
        id SERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        note TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS model_registry (
        id SERIAL PRIMARY KEY,
        version TEXT UNIQUE NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        is_champion BOOLEAN DEFAULT FALSE,
        reliable BOOLEAN DEFAULT FALSE,
        metrics JSONB,
        blob BYTEA
    )""",
    """CREATE TABLE IF NOT EXISTS training_runs (
        id SERIAL PRIMARY KEY,
        version TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        decision TEXT,
        reason TEXT,
        candidate TEXT,
        reliable BOOLEAN,
        n_rows INTEGER,
        n_tickers INTEGER,
        skill REAL,
        auc REAL,
        metrics JSONB
    )""",
    """CREATE TABLE IF NOT EXISTS watchlist (
        symbol TEXT PRIMARY KEY,
        hits INTEGER DEFAULT 1,
        first_seen TIMESTAMPTZ DEFAULT NOW(),
        last_seen TIMESTAMPTZ DEFAULT NOW()
    )""",
]


def database_url():
    return os.getenv("DATABASE_URL")


def db_configured() -> bool:
    return bool(database_url())


def get_connection():
    """Open a connection (6s timeout, one retry - Supabase may be waking from a pause)."""
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    last = None
    for attempt in range(2):
        try:
            return psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=6)
        except psycopg2.OperationalError as e:
            last = e
            time.sleep(2)
    raise last


def ensure_schema():
    """Create / migrate tables once per process; lock tables down from Supabase's public REST API."""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn = get_connection()
        try:
            for stmt in SCHEMA:
                try:
                    with conn.cursor() as cur:
                        cur.execute(stmt)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error("schema statement failed: %s | %s", stmt.split("(")[0][:60], e)
            for t in TABLES:      # RLS on + no policies = unreachable via anon key, direct SQL role still works
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
                    conn.commit()
                except Exception:
                    conn.rollback()
            _schema_ready = True
            logger.info("PostgreSQL schema ready")
        finally:
            conn.close()


@contextmanager
def _db():
    ensure_schema()
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def safe(default):
    """Log-and-return-default wrapper so a paused/unreachable DB never breaks a request."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **k):
            if not db_configured():
                logger.warning("DATABASE_URL not set - %s skipped", fn.__name__)
                return default() if callable(default) else default
            try:
                return fn(*a, **k)
            except Exception as e:
                logger.error("%s failed: %s", fn.__name__, e)
                return default() if callable(default) else default
        return wrapper
    return deco


def init_db():
    """Explicit schema initialisation (also runs lazily on first use)."""
    try:
        ensure_schema()
        return True
    except Exception as e:
        logger.error("DB init failed: %s", e)
        return False


# ───────────────────────────── heartbeat / keep-alive ─────────────────────────────

@safe(False)
def heartbeat(source: str, note: str = None) -> bool:
    """Write a row (real write activity keeps a free Supabase project from pausing)."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO heartbeat (source, note) VALUES (%s, %s)", (source, note))
        cur.execute("DELETE FROM heartbeat WHERE created_at < NOW() - INTERVAL '60 days'")
        cur.execute("SELECT COUNT(*) AS n FROM heartbeat")
        cur.fetchone()
    return True


@safe(None)
def last_heartbeat():
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT source, created_at FROM heartbeat ORDER BY created_at DESC LIMIT 1")
        return cur.fetchone()


# ───────────────────────────── watchlist (grows the training universe) ─────────────────────────────

@safe(None)
def touch_symbol(symbol: str):
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO watchlist (symbol) VALUES (%s)
                       ON CONFLICT (symbol) DO UPDATE SET hits = watchlist.hits + 1, last_seen = NOW()""",
                    (symbol.upper(),))


@safe(list)
def get_watchlist_symbols(limit: int = 60):
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT symbol FROM watchlist ORDER BY last_seen DESC LIMIT %s", (limit,))
        return [r["symbol"] for r in cur.fetchall()]


# ───────────────────────────── predictions ─────────────────────────────

@safe(False)
def save_prediction(symbol: str, direction: str, price: float, check_date: str = None, *,
                    as_of_date: str = None, horizon: int = None, prob_up: float = None,
                    model_version: str = None, source: str = "user") -> bool:
    """Insert a prediction. One row per (symbol, as_of_date, horizon): later duplicates are ignored."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO predictions (symbol, predicted_direction, price_at_prediction, check_date,
                                     as_of_date, horizon, prob_up, model_version, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, as_of_date, horizon) DO NOTHING
        """, (symbol.upper(), direction, price, check_date, as_of_date, horizon, prob_up, model_version, source))
        return cur.rowcount == 1


@safe(0)
def save_predictions_bulk(rows: list) -> int:
    """Insert many predictions in one round trip. rows: dicts with the save_prediction fields.
    Returns the number of newly inserted rows (existing symbol/as_of/horizon rows are skipped)."""
    if not rows:
        return 0
    from psycopg2.extras import execute_values
    values = [(r["symbol"].upper(), r["direction"], r["price"], r.get("check_date"), r["as_of_date"],
               r["horizon"], r.get("prob_up"), r.get("model_version"), r.get("source", "universe")) for r in rows]
    with _db() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions (symbol, predicted_direction, price_at_prediction, check_date,
                                     as_of_date, horizon, prob_up, model_version, source)
            VALUES %s ON CONFLICT (symbol, as_of_date, horizon) DO NOTHING
            RETURNING id
        """, values, fetch=True)
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


@safe(list)
def get_pending_predictions(limit: int = 2000):
    """Unresolved v2 predictions old enough that their horizon could have elapsed."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT * FROM predictions
            WHERE checked_at IS NULL AND as_of_date IS NOT NULL AND horizon IS NOT NULL
              AND as_of_date <= CURRENT_DATE - 1
            ORDER BY as_of_date
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


@safe(None)
def update_prediction_outcome(prediction_id: int, actual_price: float, was_correct, actual_return: float = None):
    """Resolve a prediction. was_correct=None for NEUTRAL calls (abstentions)."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE predictions
            SET actual_next_price = %s, was_correct = %s, actual_return = %s, checked_at = NOW()
            WHERE id = %s
        """, (actual_price, was_correct, actual_return, prediction_id))


@safe(None)
def expire_prediction(prediction_id: int):
    """Give up on a prediction whose price data never arrived (e.g. delisted)."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE predictions SET checked_at = NOW() WHERE id = %s", (prediction_id,))


def _empty_stats():
    return {"overall": {"total": 0, "correct": 0, "incorrect": 0, "pending": 0, "win_rate": 0,
                        "baseline_up_rate": 0, "edge": 0, "neutral": 0},
            "per_symbol": [], "recent": [], "by_model": [], "by_direction": {}}


@safe(_empty_stats)
def get_accuracy_stats():
    """Overall / per-symbol / per-model accuracy of resolved directional (UP/DOWN) predictions."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
              COUNT(*) FILTER (WHERE was_correct IS NOT NULL) AS total,
              COUNT(*) FILTER (WHERE was_correct = TRUE) AS correct,
              COUNT(*) FILTER (WHERE was_correct = FALSE) AS incorrect,
              COUNT(*) FILTER (WHERE checked_at IS NULL) AS pending,
              COUNT(*) FILTER (WHERE checked_at IS NOT NULL AND was_correct IS NULL
                               AND predicted_direction = 'NEUTRAL') AS neutral,
              COUNT(*) FILTER (WHERE was_correct IS NOT NULL AND actual_next_price > price_at_prediction) AS went_up
            FROM predictions WHERE as_of_date IS NOT NULL
        """)
        o = dict(cur.fetchone())
        for k in o:
            o[k] = o[k] or 0
        o["win_rate"] = round(o["correct"] / o["total"] * 100, 1) if o["total"] else 0
        o["baseline_up_rate"] = round(o["went_up"] / o["total"] * 100, 1) if o["total"] else 0
        o["edge"] = round(o["win_rate"] - o["baseline_up_rate"], 1) if o["total"] else 0

        cur.execute("""
            SELECT symbol, COUNT(*) AS total, COUNT(*) FILTER (WHERE was_correct) AS correct
            FROM predictions WHERE as_of_date IS NOT NULL AND was_correct IS NOT NULL
            GROUP BY symbol ORDER BY total DESC, symbol LIMIT 50
        """)
        per_symbol = []
        for r in cur.fetchall():
            r = dict(r)
            r["win_rate"] = round(r["correct"] / r["total"] * 100, 1) if r["total"] else 0
            per_symbol.append(r)

        cur.execute("""
            SELECT predicted_direction, COUNT(*) AS total, COUNT(*) FILTER (WHERE was_correct) AS correct
            FROM predictions WHERE as_of_date IS NOT NULL AND was_correct IS NOT NULL
            GROUP BY predicted_direction
        """)
        by_direction = {}
        for r in cur.fetchall():
            r = dict(r)
            r["win_rate"] = round(r["correct"] / r["total"] * 100, 1) if r["total"] else 0
            by_direction[r["predicted_direction"]] = r

        cur.execute("""
            SELECT model_version, COUNT(*) AS total, COUNT(*) FILTER (WHERE was_correct) AS correct
            FROM predictions WHERE as_of_date IS NOT NULL AND was_correct IS NOT NULL AND model_version IS NOT NULL
            GROUP BY model_version ORDER BY model_version DESC LIMIT 10
        """)
        by_model = []
        for r in cur.fetchall():
            r = dict(r)
            r["win_rate"] = round(r["correct"] / r["total"] * 100, 1) if r["total"] else 0
            by_model.append(r)

        cur.execute("""
            SELECT symbol, predicted_direction, price_at_prediction, actual_next_price, was_correct,
                   created_at, as_of_date, horizon, prob_up, source, actual_return
            FROM predictions WHERE as_of_date IS NOT NULL
            ORDER BY created_at DESC LIMIT 25
        """)
        recent = []
        for r in cur.fetchall():
            r = dict(r)
            r["actual_price"] = r.get("actual_next_price")
            if r["actual_price"] is not None and r["price_at_prediction"] is not None:
                r["actual_direction"] = "UP" if r["actual_price"] > r["price_at_prediction"] else "DOWN"
            else:
                r["actual_direction"] = None
            recent.append(r)

        return {"overall": o, "per_symbol": per_symbol, "recent": recent, "by_model": by_model,
                "by_direction": by_direction}


@safe(None)
def get_live_edge(model_version: str = None, min_n: int = 300):
    """Live accuracy vs the always-UP baseline on resolved directional predictions."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS n,
                   AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) AS acc,
                   AVG(CASE WHEN actual_next_price > price_at_prediction THEN 1.0 ELSE 0.0 END) AS base
            FROM predictions
            WHERE as_of_date IS NOT NULL AND was_correct IS NOT NULL
              AND (%s::text IS NULL OR model_version = %s)
        """, (model_version, model_version))
        r = cur.fetchone()
        if not r or (r["n"] or 0) < min_n:
            return None
        return {"n": r["n"], "acc": float(r["acc"]), "baseline": float(r["base"]),
                "edge": float(r["acc"]) - float(r["base"])}


# ───────────────────────────── model registry & training history ─────────────────────────────

@safe(False)
def save_model(version: str, blob: bytes, metrics: dict, reliable: bool) -> bool:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO model_registry (version, blob, metrics, reliable)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (version) DO UPDATE SET blob = EXCLUDED.blob, metrics = EXCLUDED.metrics,
                                                reliable = EXCLUDED.reliable
        """, (version, psycopg2.Binary(blob), Json(metrics), reliable))
    return True


@safe(False)
def promote_model(version: str, keep: int = 5) -> bool:
    """Make `version` the single champion and prune old non-champion blobs."""
    with _db() as conn, conn.cursor() as cur:
        cur.execute("UPDATE model_registry SET is_champion = FALSE WHERE is_champion")
        cur.execute("UPDATE model_registry SET is_champion = TRUE WHERE version = %s", (version,))
        cur.execute("""
            DELETE FROM model_registry
            WHERE NOT is_champion AND id NOT IN (SELECT id FROM model_registry ORDER BY created_at DESC LIMIT %s)
        """, (keep,))
        return cur.rowcount >= 0


@safe(None)
def get_champion(include_blob: bool = True):
    cols = "version, created_at, reliable, metrics" + (", blob" if include_blob else "")
    with _db() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {cols} FROM model_registry WHERE is_champion ORDER BY created_at DESC LIMIT 1")
        r = cur.fetchone()
        if r and include_blob and r.get("blob") is not None:
            r = dict(r)
            r["blob"] = bytes(r["blob"])
        return r


@safe(False)
def log_training_run(version: str, decision: str, reason: str, candidate: str = None, reliable: bool = None,
                     n_rows: int = None, n_tickers: int = None, skill: float = None, auc: float = None,
                     metrics: dict = None) -> bool:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO training_runs (version, decision, reason, candidate, reliable, n_rows, n_tickers,
                                       skill, auc, metrics)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (version, decision, reason, candidate, reliable, n_rows, n_tickers, skill, auc, Json(metrics or {})))
    return True


@safe(list)
def get_training_runs(limit: int = 15):
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT version, created_at, decision, reason, candidate, reliable, n_rows, n_tickers, skill, auc
                       FROM training_runs ORDER BY created_at DESC LIMIT %s""", (limit,))
        return [dict(r) for r in cur.fetchall()]


# ───────────────────────────── alerts ─────────────────────────────

@safe(None)
def log_alert(symbol: str, alert_type: str):
    with _db() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO alert_log (symbol, alert_type) VALUES (%s, %s)", (symbol.upper(), alert_type))


@safe(False)
def was_alert_sent_recently(symbol: str, alert_type: str, within_hours: int = 6) -> bool:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT COUNT(*) AS cnt FROM alert_log
                       WHERE symbol = %s AND alert_type = %s AND sent_at > NOW() - make_interval(hours => %s)""",
                    (symbol.upper(), alert_type, within_hours))
        return (cur.fetchone()["cnt"] or 0) > 0


# Alias for backwards compatibility
get_all_symbols_accuracy = get_accuracy_stats
