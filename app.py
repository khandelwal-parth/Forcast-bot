"""Flask Web Dashboard for Quant Bot."""

import os
import hmac
import math
import sys
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
from discord_alerts import check_and_send_alerts
from src import data_fetcher, analyzer, ml_service
from win_tracker import record_prediction, start_background_checker, get_accuracy_data


def clean_for_json(obj):
    """Recursively clean an object to handle NaN, Infinity, etc. for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    elif isinstance(obj, (int, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else obj
    elif hasattr(obj, "item"):                      # numpy scalars
        return clean_for_json(obj.item())
    else:
        return str(obj)


app = Flask(__name__)

# In-process safety net (heartbeat + outcome checks every 6h while the web app is awake).
# The real scheduler is the GitHub Actions workflow - it works even when the web service sleeps.
if os.getenv("ENABLE_BACKGROUND_THREAD", "1") == "1":
    start_background_checker(interval_hours=6)


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/analyze/<symbol>")
def analyze_symbol(symbol):
    """Analyze a stock symbol and return results."""
    symbol = data_fetcher.resolve_symbol(symbol)

    try:
        df = data_fetcher.get_history_df(symbol, "5y")
        company_info = data_fetcher.get_company_info(symbol)
        if len(df) < 60:
            raise ValueError(f"Not enough price history for {symbol} ({len(df)} days)")

        price_df = df                                   # raw OHLCV for the ML features
        df = analyzer.add_technical_indicators(df)

        signals = analyzer.generate_signals(df)
        fundamental_analysis = analyzer.analyze_fundamentals(company_info, {})

        # ML prediction (champion model trained offline on the whole universe)
        prediction = ml_service.predict(symbol, price_df)
        info = ml_service.model_info()

        recommendation = analyzer.generate_overall_recommendation(signals, fundamental_analysis, prediction)

        latest = df.iloc[-1].to_dict()
        prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]
        price_change = latest["close"] - prev_close
        price_change_pct = (price_change / prev_close) * 100

        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "quote": {
                "price": latest["close"],
                "change": price_change,
                "change_pct": price_change_pct,
                "volume": latest["volume"],
                "name": company_info.get("Name", symbol),
                "sector": company_info.get("Sector", "N/A"),
                "market_cap": company_info.get("MarketCapitalization", "N/A"),
            },
            "technicals": {k: latest.get(k) for k in
                           ("sma_20", "sma_50", "sma_200", "rsi", "macd", "macd_signal",
                            "stoch_k", "stoch_d", "bb_pct", "atr")},
            "signals": signals,
            "fundamentals": fundamental_analysis,
            "ml_prediction": prediction,
            "model_metrics": info,
            "recommendation": recommendation,
            "price_levels": analyzer.compute_price_levels(latest),
        }

        cleaned_result = clean_for_json(result)

        # Persist in the background so a slow/paused database never slows or breaks the response
        threading.Thread(target=_after_analysis, args=(symbol, cleaned_result), daemon=True).start()
        return jsonify({"success": True, "data": cleaned_result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


def _after_analysis(symbol, result):
    """Win-rate logging, universe growth and alerts (best effort)."""
    try:
        database.touch_symbol(symbol)                 # analysed tickers join the training universe
        record_prediction(symbol, result, source="user")
        check_and_send_alerts(symbol, result)
    except Exception as e:
        logging.getLogger(__name__).error("post-analysis tasks failed for %s: %s", symbol, e)


@app.route("/historical/<symbol>")
def get_historical(symbol):
    """Get historical price data for charting."""
    symbol = data_fetcher.resolve_symbol(symbol)
    try:
        df = data_fetcher.get_history_df(symbol, "2y")
        df = analyzer.add_technical_indicators(df).tail(90)    # indicators on full history, chart last 90 days

        chart_data = {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "close": df["close"].tolist(),
            "volume": df["volume"].tolist(),
            "sma_20": df["sma_20"].tolist(),
            "sma_50": df["sma_50"].tolist(),
        }

        # Clean data for JSON serialization
        cleaned_chart = clean_for_json(chart_data)
        return jsonify({"success": True, "data": cleaned_chart})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/accuracy")
def accuracy_page():
    """Win rate tracking page."""
    data = get_accuracy_data()
    return render_template("accuracy.html", data=data)


@app.route("/api/accuracy")
def accuracy_api():
    """API endpoint for accuracy data (JSON)."""
    data = get_accuracy_data()
    return jsonify({"success": True, "data": data})


@app.route("/api/model")
def model_api():
    """Champion model description + recent training history."""
    return jsonify({"success": True, "data": {"champion": ml_service.model_info(),
                                              "training_runs": database.get_training_runs(15)}})


@app.route("/api/cron")
def cron():
    """Token-protected keep-alive: point an external pinger (cron-job.org / UptimeRobot) at
    /api/cron?token=... daily. Writes a heartbeat row to the database and resolves outcomes."""
    expected = os.getenv("CRON_TOKEN")
    if not expected:
        return jsonify({"success": False, "error": "not found"}), 404
    if not hmac.compare_digest(request.args.get("token", ""), expected):
        return jsonify({"success": False, "error": "forbidden"}), 403

    def work():
        try:
            from jobs.daily import run
            run(mode="light", source="web-cron")
        except Exception as e:
            logging.getLogger(__name__).error("cron job failed: %s", e)

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"success": True, "started": True})


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n  Quant Bot Web Dashboard")
    print("  " + "-" * 21)
    print("  Starting server at http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", host="0.0.0.0", port=5000)
