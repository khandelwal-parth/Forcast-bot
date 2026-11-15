"""Discord Alerts for Quant Bot."""

import os
import logging
import requests
from database import log_alert, was_alert_sent_recently

logger = logging.getLogger(__name__)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord_message(content: str):
    """Send a raw message to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping alert.")
        return False
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=10)
        return response.status_code in (200, 204)
    except Exception as e:
        logger.error(f"❌ Discord send failed: {e}")
        return False


def check_and_send_alerts(symbol: str, analysis_result: dict):
    """
    Check analysis result for alert conditions and send Discord alerts.
    Call this in your /analyze route after analysis completes.
    """
    if not DISCORD_WEBHOOK_URL:
        return

    try:
        tech = analysis_result.get("technicals", {})
        rec = analysis_result.get("recommendation", {})
        quote = analysis_result.get("quote", {})
        levels = analysis_result.get("price_levels", {})

        rsi = tech.get("rsi")
        action = rec.get("recommendation", "HOLD")
        price = quote.get("price", 0)
        name = quote.get("name", symbol)

        targets = levels.get("targets", [])
        stop_losses = levels.get("stop_losses", [])
        t1 = targets[0]["price"] if targets else "N/A"
        stop = stop_losses[1]["price"] if len(stop_losses) > 1 else "N/A"

        # ── Alert 1: RSI Overbought ────────────────────────────────────────
        if rsi and rsi > 80:
            alert_type = f"overbought_{symbol}"
            if not was_alert_sent_recently(symbol, alert_type, within_hours=12):
                msg = (
                    f"🔴 **OVERBOUGHT ALERT — {symbol}**\n"
                    f"**{name}**\n"
                    f"RSI: `{rsi:.1f}` (>80 = overbought)\n"
                    f"Current Price: `${price:.2f}`\n"
                    f"⚠️ Consider taking profit or tightening stop loss."
                )
                if send_discord_message(msg):
                    log_alert(symbol, alert_type)
                    logger.info(f"🔔 Sent overbought alert for {symbol}")

        # ── Alert 2: RSI Oversold ──────────────────────────────────────────
        if rsi and rsi < 20:
            alert_type = f"oversold_{symbol}"
            if not was_alert_sent_recently(symbol, alert_type, within_hours=12):
                msg = (
                    f"🟢 **OVERSOLD ALERT — {symbol}**\n"
                    f"**{name}**\n"
                    f"RSI: `{rsi:.1f}` (<20 = oversold)\n"
                    f"Current Price: `${price:.2f}`\n"
                    f"📈 Potential bounce/entry opportunity."
                )
                if send_discord_message(msg):
                    log_alert(symbol, alert_type)
                    logger.info(f"🔔 Sent oversold alert for {symbol}")

        # ── Alert 3: Strong BUY signal ─────────────────────────────────────
        if action in ("BUY", "STRONG BUY"):
            alert_type = f"buy_{symbol}"
            if not was_alert_sent_recently(symbol, alert_type, within_hours=6):
                msg = (
                    f"✅ **BUY SIGNAL — {symbol}**\n"
                    f"**{name}**\n"
                    f"Current Price: `${price:.2f}`\n"
                    f"Target 1: `${t1}`\n"
                    f"Stop Loss: `${stop}`\n"
                    f"📊 Composite recommendation: **BUY**"
                )
                if send_discord_message(msg):
                    log_alert(symbol, alert_type)
                    logger.info(f"🔔 Sent BUY alert for {symbol}")

        # ── Alert 4: Strong SELL signal ────────────────────────────────────
        if action in ("SELL", "STRONG SELL"):
            alert_type = f"sell_{symbol}"
            if not was_alert_sent_recently(symbol, alert_type, within_hours=6):
                msg = (
                    f"🚨 **SELL SIGNAL — {symbol}**\n"
                    f"**{name}**\n"
                    f"Current Price: `${price:.2f}`\n"
                    f"📉 Composite recommendation: **SELL**\n"
                    f"⚠️ Review your position."
                )
                if send_discord_message(msg):
                    log_alert(symbol, alert_type)
                    logger.info(f"🔔 Sent SELL alert for {symbol}")

    except Exception as e:
        logger.error(f"⚠️ Alert check failed for {symbol}: {e}")


def send_daily_digest(summary: dict, preds: list, bundle: dict):
    """One short message per scheduled run: model health + strongest calls."""
    if not DISCORD_WEBHOOK_URL:
        return
    t = summary.get("training", {})
    outcomes = summary.get("outcomes", {})
    ups = sorted([p for p in preds if p["direction"] == "UP"], key=lambda p: -p["prob_up"])[:5]
    downs = sorted([p for p in preds if p["direction"] == "DOWN"], key=lambda p: p["prob_up"])[:5]
    fmt = lambda ps: ", ".join(f"{p['symbol']} ({p['prob_up']:.0%})" for p in ps) or "none"
    reliable = "✅ proven edge" if bundle.get("reliable") else "⚠️ no proven edge yet (signals not counted)"
    msg = (
        f"📊 **Quant Bot daily run** — model `{bundle['version']}` {reliable}\n"
        f"Training: {t.get('decision', 'n/a')} ({t.get('reason', '')})\n"
        f"Logged {summary.get('predictions_logged', 0)} predictions · resolved {outcomes.get('resolved', 0)} outcomes\n"
        f"Top UP (P(up)): {fmt(ups)}\nTop DOWN: {fmt(downs)}"
    )
    send_discord_message(msg[:1900])
