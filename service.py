"""Phase 6 & Phase 7: Shadow-Run Settlement, Monitoring & Standalone REST + SSE Service.

Implements:
  1. Phase 6 Shadow Run & Settlement (`--backfill-shadow` / `--settle`):
     - Logs shadow predictions (`published = FALSE`) across recent trading sessions,
       settles them against realized `intraday_bars` (`realized_return`, `realized_regime`,
       `max_adverse_move`, `settled_at`), and computes live accuracy, calibration, and
       feature-drift monitoring metrics.
  2. Phase 7 Standalone Versioned REST + SSE Service (`--serve`):
     - `GET  /v1/sentiment/nifty/latest`     : Latest prediction contract for website backend
     - `GET  /v1/sentiment/nifty/history`    : Historical predictions & settled outcomes
     - `GET  /v1/sentiment/nifty/monitoring` : Live vs backtest accuracy, drift & latency status
     - `GET  /v1/sentiment/nifty/stream`     : Server-Sent Events (SSE) push channel for live updates
     - `POST /v1/sentiment/nifty/killswitch` : Output kill-switch (`{"suppressed": true/false}`)

Usage:
    python service.py --backfill-shadow --days 5
    python service.py --serve --port 8080
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import pickle
import queue
import threading
import time
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb
import redis
import requests

from agent import (
    AI_DISCLOSURE,
    TEMPLATE_VERSION,
    TIER_LABELS,
    build_invalidation_condition,
    classify_confidence_tier,
    extract_top_drivers,
    generate_layman_explanation,
    render_compliance_template,
    run_agent,
    tool_fetch_market_snapshot,
    tool_fetch_recent_scored_news,
    tool_run_model_inference,
)
from db import ROOT, apply_schema, get_dsn
from features_and_models import add_technical_features, build_intraday_window_dataset, load_scored_articles

IST = ZoneInfo("Asia/Kolkata")
KILL_SWITCH_ACTIVE = False
_SETTLE_LOCK = threading.Lock()
_LAST_BAR_FETCH_TS = 0.0


def is_market_open(dt: datetime = None, symbol: str = "NIFTY50") -> bool:
    """Return True if the market is actively open for regular trading."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    ist_dt = dt.astimezone(IST)
    if ist_dt.weekday() > 4:  # 5=Saturday, 6=Sunday
        return False
    if symbol.upper() in ("CRUDEOIL", "GOLD", "SILVER"):
        # MCX Commodity trading session: 09:00 to 23:30 IST
        market_open = ist_dt.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = ist_dt.replace(hour=23, minute=30, second=0, microsecond=0)
    else:
        # NSE Equity trading session: 09:15 to 15:30 IST
        market_open = ist_dt.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = ist_dt.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= ist_dt < market_close


# ---------------------------------------------------------------------------
# Phase 6: Shadow Run Backfill, Settlement & Drift Monitoring
# ---------------------------------------------------------------------------
def backfill_and_settle_shadow_run(dsn: str, max_windows: int = 40, window_minutes: int = 30) -> dict:
    """Generate shadow-run predictions (`published=False`) over recent windows and settle them."""
    model_path = ROOT / "models" / "sentiment_agent_v1.pkl"
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        news_df = load_scored_articles(conn)
        df_win, _ = build_intraday_window_dataset(conn, news_df, window_minutes=window_minutes)
        recent_df = df_win.tail(max_windows).reset_index(drop=True)

        logged_count = 0
        settled_count = 0

        for _, row in recent_df.iterrows():
            as_of_utc = pd.to_datetime(row["as_of"], utc=True).to_pydatetime()
            snapshot = tool_fetch_market_snapshot(conn, as_of_utc)
            recent_news = tool_fetch_recent_scored_news(conn, as_of_utc, lookback_hours=12.0, limit=3)
            inf = tool_run_model_inference(row, bundle, window_minutes=window_minutes)
            drivers = extract_top_drivers(row, snapshot, recent_news)
            invalidation = build_invalidation_condition(row, snapshot, inf["regime"])

            ist_dt = as_of_utc.astimezone(IST)
            run_type = "pre_open" if (ist_dt.hour == 9 and ist_dt.minute <= 30) else "scheduled"

            rationale = (
                f"Shadow run ({window_minutes}m window): Nifty 50 classified as {inf['regime']} "
                f"(P(Up)={inf['prob_up']*100:.1f}%, entry={inf['entry_condition']})."
            )
            features_dict = {
                k: round(float(row.get(k, 0.0)), 5) for k in bundle["feature_columns"]
            }
            out = {
                "instrument": "NIFTY50",
                "run_type": run_type,
                "as_of": as_of_utc.isoformat(),
                "as_of_ist": ist_dt.strftime("%Y-%m-%d %H:%M IST"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "window_minutes": window_minutes,
                "horizon_start": as_of_utc.isoformat(),
                "horizon_end": min(
                    as_of_utc + timedelta(minutes=window_minutes),
                    ist_dt.replace(hour=15, minute=30, second=0, microsecond=0).astimezone(timezone.utc)
                ).isoformat(),
                "trigger_article_id": None,
                "model_version": inf["model_version"],
                "label_version": inf["label_version"],
                "regime": inf["regime"],
                "prob_bullish": inf["prob_bullish"],
                "prob_bearish": inf["prob_bearish"],
                "prob_sideways": inf["prob_sideways"],
                "prob_up": inf["prob_up"],
                "entry_condition": inf["entry_condition"],
                "entry_score": inf["entry_score"],
                "confidence": inf["confidence"],
                "invalidation": invalidation,
                "top_drivers": drivers,
                "rationale": rationale,
                "features_used": features_dict,
                "data_freshness": snapshot,
                "status": "ok",
                "template_version": TEMPLATE_VERSION,
                "published": False,
                "realized_return": round(float(row["fwd_ret"]), 6),
                "realized_regime": str(row["regime_label"]),
                "max_adverse_move": round(float(row["realized_mae"]), 6),
                "settled_at": datetime.now(timezone.utc).isoformat(),
            }
            out["rendered_text"] = render_compliance_template(out)

            conn.execute(
                """
                INSERT INTO predictions (
                    instrument, run_type, as_of, generated_at, window_minutes,
                    horizon_start, horizon_end, trigger_article_id,
                    model_version, label_version,
                    regime, prob_bullish, prob_bearish, prob_sideways, prob_up,
                    entry_condition, entry_score, confidence, invalidation,
                    top_drivers, rationale, features_used, data_freshness,
                    status, template_version, rendered_text, published,
                    realized_return, realized_regime, max_adverse_move, settled_at
                ) VALUES (
                    %(instrument)s, %(run_type)s, %(as_of)s, %(generated_at)s, %(window_minutes)s,
                    %(horizon_start)s, %(horizon_end)s, %(trigger_article_id)s,
                    %(model_version)s, %(label_version)s,
                    %(regime)s, %(prob_bullish)s, %(prob_bearish)s, %(prob_sideways)s, %(prob_up)s,
                    %(entry_condition)s, %(entry_score)s, %(confidence)s, %(invalidation)s,
                    %(top_drivers)s, %(rationale)s, %(features_used)s, %(data_freshness)s,
                    %(status)s, %(template_version)s, %(rendered_text)s, %(published)s,
                    %(realized_return)s, %(realized_regime)s, %(max_adverse_move)s, %(settled_at)s
                )
                ON CONFLICT (instrument, run_type, as_of, window_minutes) DO UPDATE SET
                    realized_return = EXCLUDED.realized_return,
                    realized_regime = EXCLUDED.realized_regime,
                    max_adverse_move = EXCLUDED.max_adverse_move,
                    settled_at = EXCLUDED.settled_at
                """,
                {
                    **out,
                    "top_drivers": Jsonb(out["top_drivers"]),
                    "features_used": Jsonb(out["features_used"]),
                    "data_freshness": Jsonb(out["data_freshness"]),
                },
            )
            logged_count += 1
            settled_count += 1

    return get_shadow_monitoring_summary(dsn)


def settle_matured_predictions(dsn: str) -> int:
    """Automatically settle any prediction whose horizon_end <= NOW() against actual 1m Nifty 50 bars."""
    global _LAST_BAR_FETCH_TS
    if not _SETTLE_LOCK.acquire(blocking=False):
        return 0
    try:
        from market_data import YFinanceAdapter, ingest_intraday_bars

        settled = 0
        with psycopg.connect(dsn, autocommit=True) as conn:
            pending = conn.execute(
                """
                SELECT prediction_id, COALESCE(horizon_start, as_of) AS h_start,
                       COALESCE(horizon_end, as_of + (window_minutes || ' minutes')::interval) AS h_end,
                       window_minutes, data_freshness, instrument
                FROM predictions
                WHERE realized_return IS NULL
                  AND COALESCE(horizon_end, as_of + (window_minutes || ' minutes')::interval) + interval '25 seconds' <= CURRENT_TIMESTAMP
                ORDER BY as_of ASC
                """
            ).fetchall()

            if not pending:
                return 0

            for pred_id, h_start, h_end, win_min, freshness, instrument in pending:
                sym = (instrument or "NIFTY50").upper()
                fresh_dict = dict(freshness or {})
                # Strictly lock p_start to the original anchor price recorded when the prediction was generated!
                p_start = float(fresh_dict.get("settled_start_price") or fresh_dict.get("start_spot_price") or fresh_dict.get("spot_price") or fresh_dict.get("nifty_close") or 0.0)
                if p_start <= 0:
                    r_start = conn.execute(
                        """
                        SELECT close FROM intraday_bars
                        WHERE symbol = %s AND interval_sec = 60 AND ts <= %s
                        ORDER BY ts DESC LIMIT 1
                        """,
                        (sym, h_start),
                    ).fetchone()
                    if r_start:
                        p_start = float(r_start[0])

                r_end = conn.execute(
                    """
                    SELECT close FROM intraday_bars
                    WHERE symbol = %s AND interval_sec = 60 AND available_at <= CURRENT_TIMESTAMP AND ts <= %s
                    ORDER BY ts DESC LIMIT 1
                    """,
                    (sym, h_end),
                ).fetchone()
                if p_start <= 0 or not r_end:
                    continue

                p_end = float(r_end[0])
                if p_end <= 0:
                    continue

                ret = math.log(p_end / p_start)
                w_m = int(win_min or 30)
                if w_m <= 1:
                    band = 0.00025
                elif w_m <= 5:
                    band = 0.00060
                elif w_m <= 15:
                    band = 0.00100
                else:
                    band = 0.00150
                # Sideways market trend commented out per user request:
                # if abs(ret) <= band:
                #     real_reg = "sideways"
                # elif ret > band:
                #     real_reg = "bullish"
                # else:
                #     real_reg = "bearish"
                if ret >= 0:
                    real_reg = "bullish"
                else:
                    real_reg = "bearish"

                # Keep the exact original p_start so the displayed Start Price NEVER changes after settlement!
                fresh_dict["settled_start_price"] = round(p_start, 2)
                fresh_dict["settled_end_price"] = round(p_end, 2)

                conn.execute(
                    """
                    UPDATE predictions
                    SET realized_return = %s,
                        realized_regime = %s,
                        settled_at = CURRENT_TIMESTAMP,
                        data_freshness = %s
                    WHERE prediction_id = %s
                    """,
                    (round(ret, 6), real_reg, Jsonb(fresh_dict), pred_id),
                )
                settled += 1
        return settled
    finally:
        _SETTLE_LOCK.release()


def get_shadow_monitoring_summary(dsn: str, symbol: str = "NIFTY50") -> dict:
    """Compute live shadow-run accuracy, Brier calibration, adverse move rate & drift metrics."""
    symbol = symbol.upper()
    settle_matured_predictions(dsn)
    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute(
            """
            SELECT prob_up, realized_return, entry_condition, max_adverse_move,
                   regime, realized_regime, as_of
            FROM predictions
            WHERE instrument = %s AND realized_return IS NOT NULL
            ORDER BY as_of DESC
            """,
            (symbol,),
        ).fetchall()

    if not rows:
        return {"settled_predictions": 0, "status": "no_settled_predictions"}

    probs = np.array([float(r[0]) for r in rows])
    rets = np.array([float(r[1]) for r in rows])
    y_true = (rets > 0).astype(int)
    y_pred = (probs >= 0.50).astype(int)

    dir_acc = round(100.0 * float((y_true == y_pred).mean()), 2)
    brier = round(float(np.mean((probs - y_true) ** 2)), 4)

    fav_rows = [r for r in rows if r[2] == "favourable"]
    fav_adverse = (
        round(100.0 * sum(1 for r in fav_rows if float(r[3] or 0) > 0.0025) / len(fav_rows), 2)
        if fav_rows
        else 0.0
    )

    return {
        "settled_predictions": len(rows),
        "shadow_directional_accuracy_pct": dir_acc,
        "backtest_reference_accuracy_pct": 59.34,
        "within_backtest_tolerance": abs(dir_acc - 59.34) <= 8.0,
        "shadow_brier_score": brier,
        "favourable_calls": len(fav_rows),
        "favourable_adverse_move_rate_pct": fav_adverse,
        "kill_switch_active": KILL_SWITCH_ACTIVE,
        "latest_settled_as_of": rows[0][6].isoformat(),
    }


_SSE_CLIENTS: list[queue.Queue] = []
_SSE_LOCK = threading.Lock()
_LAST_SIM_TICK: dict = {}
_SIM_LOCK = threading.Lock()

SUPPORTED_SYMBOLS: dict[str, dict] = {
    "NIFTY50": {
        "id": "NIFTY50",
        "name": "Nifty 50",
        "segment": 1,
        "token": 26000,
        "exchange": "NSE",
        "category": "Index",
        "market_hours": "09:15–15:30 IST",
        "ref_price": 25238.0,
    },
    "BANKNIFTY": {
        "id": "BANKNIFTY",
        "name": "Bank Nifty",
        "segment": 1,
        "token": 26001,
        "exchange": "NSE",
        "category": "Index",
        "market_hours": "09:15–15:30 IST",
        "ref_price": 52500.0,
    },
    "CRUDEOIL": {
        "id": "CRUDEOIL",
        "name": "Crude Oil",
        "segment": 51,
        "token": 573422,
        "exchange": "MCX",
        "category": "Commodity",
        "market_hours": "09:00–23:30 IST",
        "ref_price": 8350.0,
    },
    "GOLD": {
        "id": "GOLD",
        "name": "Gold",
        "segment": 51,
        "token": 495213,
        "exchange": "MCX",
        "category": "Commodity",
        "market_hours": "09:00–23:30 IST",
        "ref_price": 136500.0,
    },
    "SILVER": {
        "id": "SILVER",
        "name": "Silver",
        "segment": 51,
        "token": 495214,
        "exchange": "MCX",
        "category": "Commodity",
        "market_hours": "09:00–23:30 IST",
        "ref_price": 178500.0,
    },
}

TOKEN_TO_SYMBOL: dict[str, str] = {
    str(meta["token"]): sym for sym, meta in SUPPORTED_SYMBOLS.items()
}

_LIVE_QUOTE_CACHE: dict = {
    "spot_price": None,
    "open": None,
    "high": None,
    "low": None,
    "prev_close": None,
    "vix": None,
    "percent_change": 0.0,
    "ts": 0.0,
}
for _sym in SUPPORTED_SYMBOLS:
    _LIVE_QUOTE_CACHE[_sym] = {
        "spot_price": None,
        "open": None,
        "high": None,
        "low": None,
        "prev_close": None,
        "vix": None,
        "percent_change": 0.0,
        "ts": 0.0,
    }
_ACTIVE_WINDOW: int = 15
_LAST_PRED_RUN_TS: float = 0.0


def broadcast_sse(event_name: str, payload: dict) -> None:
    """Broadcast an SSE event payload to all currently connected clients."""
    data_str = json.dumps(payload)
    msg = f"event: {event_name}\ndata: {data_str}\n\n".encode("utf-8")
    with _SSE_LOCK:
        dead = []
        for q in _SSE_CLIENTS:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for d in dead:
            if d in _SSE_CLIENTS:
                _SSE_CLIENTS.remove(d)


XTS_BROADCASTER_API = "http://172.172.1.13:8050/api/v1"
XTS_REDIS_HOST = "172.172.1.13"
XTS_REDIS_PORT = 6379
XTS_MESSAGE_CODE = 1501
XTS_EXCHANGE_SEGMENT = 1
XTS_INSTRUMENT_ID = 26000
CSV_LIVE_FILE = r"data\live_data_ohlc.csv"


def _xts_live_feed_worker(dsn: str) -> None:
    """Real-time tick-by-tick XTS broadcaster & Redis subscriber for all supported symbols."""
    global _LIVE_QUOTE_CACHE

    def _subscribe():
        try:
            instruments = [
                {
                    "exchangeSegment": meta["segment"],
                    "exchangeInstrumentID": meta["token"],
                    "symbol": str(meta["token"]),
                    "subscribedCodes": [XTS_MESSAGE_CODE],
                }
                for meta in SUPPORTED_SYMBOLS.values()
            ]
            requests.post(f"{XTS_BROADCASTER_API}/subscriptions", json={"instruments": instruments}, timeout=5)
        except Exception:
            pass

    def _poll_touchline():
        for sym, meta in SUPPORTED_SYMBOLS.items():
            try:
                r = requests.get(
                    f"{XTS_BROADCASTER_API}/last-tick/{meta['segment']}/{meta['token']}?code={XTS_MESSAGE_CODE}",
                    timeout=2,
                )
                if r.status_code == 200:
                    data = r.json()
                    tl = data.get("payload", {}).get("Touchline") or data.get("Touchline")
                    if tl:
                        _apply_touchline(tl, sym=sym)
            except Exception:
                pass

    def _apply_touchline(tl: dict, sym: str = "NIFTY50"):
        ltp = tl.get("LastTradedPrice")
        if ltp is None:
            return
        ltp = float(ltp)
        opn = float(tl.get("Open") or ltp)
        hi = float(tl.get("High") or ltp)
        lo = float(tl.get("Low") or ltp)
        cls = float(tl.get("Close") or ltp)
        pct = float(tl.get("PercentChange") or 0.0)
        with _SIM_LOCK:
            if sym not in _LIVE_QUOTE_CACHE:
                _LIVE_QUOTE_CACHE[sym] = {}
            c_dict = _LIVE_QUOTE_CACHE[sym]
            c_dict["spot_price"] = ltp
            c_dict["open"] = opn
            c_dict["high"] = hi
            c_dict["low"] = lo
            c_dict["prev_close"] = cls
            c_dict["percent_change"] = pct
            c_dict["ts"] = time.time()
            if sym == "NIFTY50":
                _LIVE_QUOTE_CACHE["spot_price"] = ltp
                _LIVE_QUOTE_CACHE["open"] = opn
                _LIVE_QUOTE_CACHE["high"] = hi
                _LIVE_QUOTE_CACHE["low"] = lo
                _LIVE_QUOTE_CACHE["prev_close"] = cls
                _LIVE_QUOTE_CACHE["percent_change"] = pct
                _LIVE_QUOTE_CACHE["ts"] = time.time()

    def _save_candle(min_dt: datetime, c: dict, sym: str = "NIFTY50"):
        # Strictly ignore post-market or pre-market ticks to prevent flat candle contamination
        if not is_market_open(min_dt, symbol=sym):
            return
        try:
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(
                    """
                    INSERT INTO intraday_bars (symbol, interval_sec, ts, open, high, low, close, volume, available_at)
                    VALUES (%s, 60, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval_sec, ts) DO UPDATE SET
                        high = GREATEST(intraday_bars.high, EXCLUDED.high),
                        low = LEAST(intraday_bars.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        available_at = EXCLUDED.available_at
                    """,
                    (
                        sym,
                        min_dt,
                        c["open"],
                        c["high"],
                        c["low"],
                        c["close"],
                        c["volume"],
                        min_dt + timedelta(seconds=60),
                    ),
                )
                for iv_sec in (300, 900):
                    conn.execute(
                        f"""
                        INSERT INTO intraday_bars (symbol, interval_sec, ts, open, high, low, close, volume, available_at)
                        SELECT 
                            %s,
                            {iv_sec},
                            to_timestamp(floor(extract(epoch from ts) / {iv_sec}) * {iv_sec}) as bucket_ts,
                            (array_agg(open ORDER BY ts ASC))[1],
                            MAX(high),
                            MIN(low),
                            (array_agg(close ORDER BY ts DESC))[1],
                            SUM(volume),
                            to_timestamp(floor(extract(epoch from ts) / {iv_sec}) * {iv_sec}) + interval '{iv_sec} seconds'
                        FROM intraday_bars
                        WHERE symbol = %s AND interval_sec = 60 AND ts >= CURRENT_TIMESTAMP - interval '1 day'
                        GROUP BY bucket_ts
                        ON CONFLICT (symbol, interval_sec, ts) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = GREATEST(intraday_bars.high, EXCLUDED.high),
                            low = LEAST(intraday_bars.low, EXCLUDED.low),
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            available_at = EXCLUDED.available_at;
                        """,
                        (sym, sym),
                    )
        except Exception:
            pass

    _subscribe()
    _poll_touchline()

    current_minutes = {sym: None for sym in SUPPORTED_SYMBOLS}
    candles = {sym: None for sym in SUPPORTED_SYMBOLS}

    def _poll_continuous():
        while True:
            try:
                _r = redis.Redis(host=XTS_REDIS_HOST, port=XTS_REDIS_PORT, decode_responses=True)
                for sym, meta in SUPPORTED_SYMBOLS.items():
                    h = _r.hgetall(f"xts:marketdata:last:{meta['segment']}:{meta['token']}")
                    tl_str = h.get(str(XTS_MESSAGE_CODE))
                    if tl_str:
                        tl = json.loads(tl_str).get("Touchline")
                        if tl:
                            _apply_touchline(tl, sym=sym)
                            ltp = float(tl.get("LastTradedPrice") or 0.0)
                            vol = int(tl.get("TotalTradedQuantity") or 0)
                            if ltp > 0:
                                now_min = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                                if current_minutes[sym] is None:
                                    current_minutes[sym] = now_min
                                    candles[sym] = {"open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": vol}
                                elif now_min != current_minutes[sym]:
                                    _save_candle(current_minutes[sym], candles[sym], sym=sym)
                                    current_minutes[sym] = now_min
                                    candles[sym] = {"open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": vol}
                                else:
                                    candles[sym]["high"] = max(candles[sym]["high"], ltp)
                                    candles[sym]["low"] = min(candles[sym]["low"], ltp)
                                    candles[sym]["close"] = ltp
                                    candles[sym]["volume"] = vol
            except Exception:
                pass
            time.sleep(1.0)

    t_poll = threading.Thread(target=_poll_continuous, daemon=True, name="XTSContinuousPoll")
    t_poll.start()

    while True:
        try:
            r_client = redis.Redis(host=XTS_REDIS_HOST, port=XTS_REDIS_PORT, decode_responses=True)
            ps = r_client.pubsub()
            ps.subscribe("xts:marketdata:all")

            for msg in ps.listen():
                if msg["type"] != "message":
                    continue
                try:
                    tick = json.loads(msg["data"])
                except Exception:
                    continue

                if not isinstance(tick, dict):
                    continue
                tok = str(tick.get("ExchangeInstrumentID", tick.get("exchangeInstrumentID")))
                sym = TOKEN_TO_SYMBOL.get(tok)
                if not sym:
                    continue

                touchline = tick.get("Touchline")
                if not touchline:
                    continue
                ltp = touchline.get("LastTradedPrice")
                if ltp is None:
                    continue

                _apply_touchline(touchline, sym=sym)

                # Aggregate ticks into 1-minute OHLC candle per symbol
                ltp = float(ltp)
                vol = int(touchline.get("TotalTradedQuantity", 0))
                now_min = datetime.now(timezone.utc).replace(second=0, microsecond=0)

                if current_minutes[sym] is None:
                    current_minutes[sym] = now_min
                    candles[sym] = {
                        "open": ltp,
                        "high": ltp,
                        "low": ltp,
                        "close": ltp,
                        "volume": vol,
                    }
                elif now_min != current_minutes[sym]:
                    _save_candle(current_minutes[sym], candles[sym], sym=sym)
                    current_minutes[sym] = now_min
                    candles[sym] = {
                        "open": ltp,
                        "high": ltp,
                        "low": ltp,
                        "close": ltp,
                        "volume": vol,
                    }
                else:
                    candles[sym]["high"] = max(candles[sym]["high"], ltp)
                    candles[sym]["low"] = min(candles[sym]["low"], ltp)
                    candles[sym]["close"] = ltp
                    candles[sym]["volume"] = vol
        except Exception:
            time.sleep(2.0)
            _subscribe()
            _poll_touchline()
            time.sleep(2.0)
            _subscribe()
            _poll_touchline()


def fetch_live_market_tick(dsn: str, symbol: str = "NIFTY50") -> dict:
    """Fetch the absolute latest spot price, day stats, and change directly from live XTS / Redis feed."""
    global _LIVE_QUOTE_CACHE

    symbol = symbol.upper()
    now_ist = datetime.now(IST)
    q = _LIVE_QUOTE_CACHE.get(symbol) or _LIVE_QUOTE_CACHE
    ref_default = SUPPORTED_SYMBOLS.get(symbol, {}).get("ref_price", 25238.0)

    if q.get("spot_price") and (time.time() - q.get("ts", 0.0) < 60.0):
        spot_close = q["spot_price"]
        day_open = q.get("open") or spot_close
        day_high = q.get("high") or spot_close
        day_low = q.get("low") or spot_close
        prev_close = q.get("prev_close") or spot_close
        pct_chg = q.get("percent_change", 0.0)
        vix = q.get("vix") or 16.96
    else:
        with psycopg.connect(dsn, autocommit=True) as conn:
            r_bar = conn.execute(
                """
                SELECT close, open, high, low, ts, available_at 
                FROM intraday_bars 
                WHERE symbol = %s AND interval_sec = 60
                ORDER BY ts DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()

            r_day = conn.execute(
                """
                SELECT open, high, low, close
                FROM intraday_bars
                WHERE symbol = %s AND interval_sec = 60
                  AND ts >= (SELECT MAX(ts)::date FROM intraday_bars WHERE symbol = %s)
                ORDER BY ts ASC
                """,
                (symbol, symbol),
            ).fetchall()

            r_vix = conn.execute(
                """
                SELECT close FROM intraday_bars WHERE symbol = 'INDIAVIX' ORDER BY ts DESC LIMIT 1
                """
            ).fetchone()

        spot_close = float(r_bar[0]) if r_bar else ref_default
        if r_day:
            day_open = float(r_day[0][0])
            day_high = max(float(x[1]) for x in r_day)
            day_low = min(float(x[2]) for x in r_day)
        else:
            day_open = spot_close
            day_high = spot_close
            day_low = spot_close
        prev_close = day_open
        pct_chg = ((spot_close - day_open) / day_open) * 100.0 if day_open else 0.0
        vix = float(r_vix[0]) if r_vix else 16.96

    cur_spot = spot_close
    day_change = cur_spot - prev_close
    day_change_pct = pct_chg if pct_chg != 0.0 else ((day_change / prev_close) * 100.0 if prev_close else 0.0)
    last_tick_time = now_ist.strftime("%H:%M:%S IST")

    return {
        "symbol": symbol,
        "symbol_name": SUPPORTED_SYMBOLS.get(symbol, {}).get("name", symbol),
        "spot_price": round(cur_spot, 2),
        "day_open": round(day_open, 2),
        "day_high": round(day_high, 2),
        "day_low": round(day_low, 2),
        "day_change": round(day_change, 2),
        "day_change_pct": round(day_change_pct, 2),
        "vix": round(vix, 2),
        "last_tick_time": last_tick_time,
    }


def fetch_latest_prediction_payload(dsn: str, symbol: str = "NIFTY50") -> dict:
    symbol = symbol.upper()
    sym_name = SUPPORTED_SYMBOLS.get(symbol, {}).get("name", symbol)
    if KILL_SWITCH_ACTIVE:
        return {
            "instrument": symbol,
            "status": "suppressed",
            "reason": "Output kill-switch is active.",
            "ai_disclosure": AI_DISCLOSURE,
        }
    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            """
            SELECT prediction_id, instrument, run_type, as_of, generated_at,
                   window_minutes, horizon_start, horizon_end, model_version, label_version,
                   regime, prob_bullish, prob_bearish, prob_sideways, prob_up,
                   entry_condition, entry_score, confidence, invalidation,
                   top_drivers, rationale, data_freshness, status,
                   template_version, rendered_text, features_used
            FROM predictions
            WHERE instrument = %s
            ORDER BY prediction_id DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    if not row:
        pred = run_agent(dsn, run_type="scheduled", window_minutes=_ACTIVE_WINDOW, symbol=symbol, persist=True)
        pred["live_tick"] = fetch_live_market_tick(dsn, symbol=symbol)
        pred["is_market_open"] = is_market_open(symbol=symbol)
        return pred

    cols = [
        "prediction_id", "instrument", "run_type", "as_of", "generated_at",
        "window_minutes", "horizon_start", "horizon_end", "model_version", "label_version",
        "regime", "prob_bullish", "prob_bearish", "prob_sideways", "prob_up",
        "entry_condition", "entry_score", "confidence", "invalidation",
        "top_drivers", "rationale", "data_freshness", "status",
        "template_version", "rendered_text", "features_used"
    ]
    d = dict(zip(cols, row))
    for k in ("as_of", "generated_at", "horizon_start", "horizon_end"):
        if hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    for k in ("prob_bullish", "prob_bearish", "prob_sideways", "prob_up", "entry_score", "confidence"):
        if d[k] is not None:
            d[k] = float(d[k])

    live_tick = fetch_live_market_tick(dsn, symbol=symbol)
    d["live_tick"] = live_tick
    d["is_market_open"] = is_market_open(symbol=symbol)

    # 1. Solid, un-jittered model forecast probability (NO micro-tick wobble)
    base_prob = float(d.get("prob_up", 0.5))
    feat_dict = d.get("features_used") or {}
    ref_spot = float(feat_dict.get("close") or (d.get("data_freshness") or {}).get("spot_price") or (d.get("data_freshness") or {}).get("nifty_close") or live_tick["spot_price"])
    cur_spot = live_tick["spot_price"]

    # Calculate structural VWAP support/resistance level
    vwap_dist = float(feat_dict.get("true_vwap_dist", feat_dict.get("vwap_dist", 0.0)))
    vwap_level = round(ref_spot / (1.0 + vwap_dist), 1) if (1.0 + vwap_dist) != 0 else ref_spot
    base_regime = str(d.get("regime", "sideways")).lower()

    # Major Disruption Threshold: 1.5x Window ATR or India VIX spike > 8%
    atr_pct = float(feat_dict.get("atr_pct", 0.0025))
    min_atr_pts = 10.0 if symbol in ("CRUDEOIL", "GOLD", "SILVER") else 35.0
    atr_pts = max(min_atr_pts, round(ref_spot * atr_pct * 1.5, 0))
    cur_vix = float(live_tick.get("vix") or 17.0)
    ref_vix = float((d.get("data_freshness") or {}).get("india_vix") or cur_vix)
    vix_spike = (cur_vix - ref_vix) / ref_vix if ref_vix > 0 else 0.0

    # 2. Invalidation check: a major structural break against the call WITHDRAWS the view
    tier = classify_confidence_tier(base_prob)
    d["favourable_pct"] = round(base_prob * 100.0, 1)
    if base_regime == "bullish" and (cur_spot < vwap_level - atr_pts or cur_spot < ref_spot - atr_pts):
        d["regime"] = "sideways"
        tier = "NO_VIEW"
        d["entry_condition"] = "avoid"
        d["layman_explanation"] = (
            f"🚨 View invalidated: {sym_name} ({cur_spot:,.1f}) broke below VWAP support ({vwap_level:,.1f}) by "
            f"{abs(cur_spot - vwap_level):.1f} pts (> 1.5x ATR). The earlier Bullish call is withdrawn — "
            f"no clear view until the next window."
        )
    elif base_regime == "bearish" and (cur_spot > vwap_level + atr_pts or cur_spot > ref_spot + atr_pts):
        d["regime"] = "sideways"
        tier = "NO_VIEW"
        d["entry_condition"] = "avoid"
        d["layman_explanation"] = (
            f"🚨 View invalidated: {sym_name} ({cur_spot:,.1f}) surged above VWAP resistance ({vwap_level:,.1f}) by "
            f"{(cur_spot - vwap_level):.1f} pts (> 1.5x ATR). The earlier Bearish call is withdrawn — "
            f"no clear view until the next window."
        )
    elif vix_spike > 0.08:
        d["regime"] = "sideways"
        tier = "NO_VIEW"
        d["entry_condition"] = "avoid"
        d["layman_explanation"] = (
            f"🚨 Volatility disruption: India VIX spiked +{vix_spike*100:.1f}% to {cur_vix:.2f}. "
            f"Directional view withheld until volatility settles."
        )
    else:
        # Honest probability (no stretching); abstain inside the no-view band.
        if tier == "NO_VIEW":
            d["regime"] = "sideways"
        else:
            d["regime"] = "bullish" if base_prob >= 0.50 else "bearish"
        d["layman_explanation"] = generate_layman_explanation(
            d, d.get("data_freshness") or {}, pd.Series(feat_dict), symbol=symbol
        )

    d["confidence_pct"] = round(max(base_prob, 1.0 - base_prob) * 100.0, 1)
    d["confidence_tier"] = tier
    d["signal_strength"] = TIER_LABELS[tier]

    d["conviction_pct"] = round(float(d.get("entry_score", 0.5)) * 100.0, 1) if d.get("entry_score") is not None else 50.0
    d["ai_disclosure"] = AI_DISCLOSURE
    return d


def build_post_prediction_analysis(
    conn,
    as_of_dt: datetime,
    h_end_dt: datetime,
    pred_regime: str,
    prob_up: float,
    start_price: float | None,
    end_price: float | None,
    ret: float | None,
    features_used: dict | None,
    symbol: str = "NIFTY50",
) -> dict | None:
    """Generate a detailed post-window diagnostic explaining what happened during [as_of, h_end]
    and, when the market moved opposite to the prediction, why it wasn't picked earlier at entry.
    """
    if ret is None or as_of_dt is None or h_end_dt is None:
        return None

    feat = features_used or {}
    is_bull_pred = prob_up >= 0.50
    is_hit = (is_bull_pred and ret >= 0) or ((not is_bull_pred) and ret < 0)
    pred_label = "BULLISH (UP)" if is_bull_pred else "BEARISH (DOWN)"
    act_label = "UP" if ret >= 0 else "DOWN"
    ret_pct = round(ret * 100.0, 3)

    # 1. Why the model predicted this direction at entry (time t)
    vwap_d = float(feat.get("true_vwap_dist", feat.get("vwap_dist", 0.0))) * 100.0
    ema_slope = float(feat.get("ema20_slope", 0.0)) * 100.0
    adx_val = float(feat.get("adx", 18.0))
    rsi_raw = float(feat.get("rsi_14", 0.0)) * 50.0 + 50.0
    sr_pos = float(feat.get("sr_pos", 0.5))
    mtf = float(feat.get("mtf_trend_score", 0.0))
    vol_sp = float(feat.get("vol_spike", 0.0))
    top5_r = float(feat.get("top5_ret_1", 0.0)) * 100.0
    news_surp = float(feat.get("news_surprise", 0.0))

    entry_reasons = []
    if vwap_d > 0.02:
        entry_reasons.append(f"price trading +{vwap_d:.2f}% above Session VWAP")
    elif vwap_d < -0.02:
        entry_reasons.append(f"price trading {vwap_d:.2f}% below Session VWAP")
    if ema_slope > 0.005:
        entry_reasons.append(f"rising 20-EMA slope (+{ema_slope:.3f}%)")
    elif ema_slope < -0.005:
        entry_reasons.append(f"falling 20-EMA slope ({ema_slope:.3f}%)")
    if mtf > 0.15:
        entry_reasons.append("higher-timeframe (15m/60m) trend aligned bullish")
    elif mtf < -0.15:
        entry_reasons.append("higher-timeframe (15m/60m) trend aligned bearish")
    if abs(news_surp) > 0.05:
        entry_reasons.append(f"news sentiment impulse ({news_surp:+.2f})")
    if not entry_reasons:
        entry_reasons.append(f"ML baseline probability P(Up)={prob_up*100:.1f}% with ADX={adx_val:.1f}, RSI={rsi_raw:.1f}")

    why_predicted = (
        f"At entry ({as_of_dt.astimezone(IST).strftime('%H:%M IST')}), the model forecast {pred_label} "
        f"({prob_up*100:.1f}%) based on: " + "; ".join(entry_reasons) + "."
    )

    # 2. Fetch intra-window bars (prefer 60s, fallback 300s) to see what ACTUALLY happened inside [as_of, h_end]
    bars = conn.execute(
        """
        SELECT ts, open, high, low, close, volume
        FROM intraday_bars
        WHERE symbol = %s AND interval_sec = 60
          AND ts >= %s AND ts <= %s
        ORDER BY ts ASC
        """,
        (symbol, as_of_dt - timedelta(seconds=60), h_end_dt + timedelta(seconds=60)),
    ).fetchall()
    if len(bars) < 2:
        bars = conn.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM intraday_bars
            WHERE symbol = %s AND interval_sec = 300
              AND ts >= %s AND ts <= %s
            ORDER BY ts ASC
            """,
            (symbol, as_of_dt - timedelta(seconds=300), h_end_dt + timedelta(seconds=300)),
        ).fetchall()

    # Check if any breaking news arrived INSIDE the prediction window
    mid_news = conn.execute(
        """
        SELECT COALESCE(a.specific_title, a.title), a.published_at, s.sentiment_score
        FROM articles a
        JOIN article_scores s ON s.article_id = a.article_id
        WHERE a.published_at > %s AND a.published_at <= %s
        ORDER BY ABS(s.sentiment_score) DESC
        LIMIT 2
        """,
        (as_of_dt, h_end_dt),
    ).fetchall()

    s_px = float(start_price) if start_price else (float(bars[0][4]) if bars else 0.0)
    e_px = float(end_price) if end_price else (float(bars[-1][4]) if bars else s_px)
    pts_chg = round(e_px - s_px, 2)

    trajectory_notes = []
    missed_reasons = []

    if bars:
        highs = [(float(b[2]), b[0].astimezone(IST).strftime("%H:%M")) for b in bars]
        lows = [(float(b[3]), b[0].astimezone(IST).strftime("%H:%M")) for b in bars]
        win_hi, hi_t = max(highs, key=lambda x: x[0])
        win_lo, lo_t = min(lows, key=lambda x: x[0])
        mid_idx = len(bars) // 2
        mid_px = float(bars[mid_idx][4]) if mid_idx < len(bars) else e_px
        first_half_pts = mid_px - s_px
        second_half_pts = e_px - mid_px

        trajectory_notes.append(
            f"Spot moved from {s_px:,.1f} to {e_px:,.1f} ({pts_chg:+.1f} pts / {ret_pct:+.3f}%), "
            f"hitting a window high of {win_hi:,.1f} at {hi_t} IST and low of {win_lo:,.1f} at {lo_t} IST."
        )

        # Detect whether it moved our way first then reversed, or went opposite immediately
        if not is_hit:
            if (is_bull_pred and first_half_pts > 5.0 and second_half_pts < -10.0) or (
                not is_bull_pred and first_half_pts < -5.0 and second_half_pts > 10.0
            ):
                trajectory_notes.append(
                    f"Initial move followed the forecast ({first_half_pts:+.1f} pts in the first half) "
                    f"before a sharp mid-window reversal ({second_half_pts:+.1f} pts in the second half)."
                )
                missed_reasons.append(
                    "Post-entry reversal: order flow initially aligned with the model at entry, "
                    "but profit-booking / counter-flow emerged mid-window after the prediction was locked."
                )
            else:
                trajectory_notes.append(
                    f"Price moved against the forecast early ({first_half_pts:+.1f} pts in first half, "
                    f"{second_half_pts:+.1f} pts in second half)."
                )
    else:
        trajectory_notes.append(
            f"Spot settled from {s_px:,.1f} to {e_px:,.1f} ({pts_chg:+.1f} pts / {ret_pct:+.3f}%)."
        )

    if mid_news:
        headline_str = str(mid_news[0][0])[:90]
        sent_val = float(mid_news[0][2] or 0.0)
        t_news = mid_news[0][1].astimezone(IST).strftime("%H:%M IST")
        trajectory_notes.append(
            f"Mid-window news arrived at {t_news} (sentiment {sent_val:+.2f}): \"{headline_str}\"."
        )
        if not is_hit:
            missed_reasons.append(
                f"Unforeseen intra-window news catalyst published at {t_news} shifted sentiment after entry."
            )

    what_happened = " ".join(trajectory_notes)

    if is_hit:
        why_opposite = (
            f"Forecast CONFIRMED ({act_label} {ret_pct:+.3f}%): Price respected the entry structure "
            f"and stayed aligned with the predicted {pred_label} bias through settlement."
        )
    else:
        # Diagnose why the opposite move wasn't caught at entry time t
        if 0.46 <= prob_up <= 0.54:
            missed_reasons.append(
                f"Borderline conviction at entry (P(Up)={prob_up*100:.1f}%): signal strength was near the 50% coin-flip threshold."
            )
        if adx_val < 20.0:
            missed_reasons.append(
                f"Low trend strength (ADX={adx_val:.1f} < 20): market was range-bound/choppy, where short-horizon mean-reversion noise easily overrides momentum."
            )
        if is_bull_pred and (sr_pos >= 0.78 or rsi_raw >= 65.0):
            missed_reasons.append(
                f"Overhead exhaustion at entry (Channel Pos {sr_pos*100:.0f}%, RSI {rsi_raw:.1f}): momentum looked positive, but buyers ran into overhead resistance."
            )
        elif (not is_bull_pred) and (sr_pos <= 0.22 or rsi_raw <= 35.0):
            missed_reasons.append(
                f"Oversold support zone at entry (Channel Pos {sr_pos*100:.0f}%, RSI {rsi_raw:.1f}): sellers pushed into support where dip-buying absorbed the drop."
            )
        if is_bull_pred and top5_r < -0.02:
            missed_reasons.append(
                f"Heavyweight drag (Top-5 return {top5_r:+.3f}%): Nifty spot hadn't broken down yet, so the model weighted spot VWAP over lagging index heavyweights."
            )
        elif (not is_bull_pred) and top5_r > 0.02:
            missed_reasons.append(
                f"Heavyweight absorption (Top-5 return {top5_r:+.3f}%): index heavyweights were already turning positive while Nifty spot still lagged below VWAP."
            )
        if vol_sp < -0.25:
            missed_reasons.append(
                f"Low pre-entry volume (vol_spike={vol_sp:.2f}): the counter-move was driven by fresh liquidity entering *after* the candle opened."
            )
        if not missed_reasons:
            missed_reasons.append(
                "Sudden intra-window institutional order-flow shift occurred after entry that was not visible in prior closed candles."
            )
        why_opposite = " | ".join(missed_reasons)

    return {
        "status_label": "CONFIRMED" if is_hit else "OPPOSITE_MOVE",
        "is_hit": is_hit,
        "predicted": pred_label,
        "actual": f"{act_label} ({ret_pct:+.3f}% / {pts_chg:+.1f} pts)",
        "why_predicted": why_predicted,
        "what_happened": what_happened,
        "why_opposite": why_opposite,
    }


def fetch_live_chart_payload(dsn: str, window_minutes: int = 5, symbol: str = "NIFTY50") -> dict:
    """Build real-time OHLC candles + technical indicators (VWAP, EMA20, RSI14, ADX14)
    and overlay predictions + post-window diagnostics for the live interactive chart.
    """
    symbol = symbol.upper()
    is_commodity = symbol in ("CRUDEOIL", "GOLD", "SILVER")
    window_minutes = 15 if int(window_minutes) >= 15 else 5
    sec = window_minutes * 60

    with psycopg.connect(dsn, autocommit=True) as conn:
        # Load recent bars (1m preferred, rolled into window_minutes; fallback to 300s/900s)
        rows = conn.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM intraday_bars
            WHERE symbol = %s AND interval_sec = 60
              AND ts >= CURRENT_TIMESTAMP - interval '5 days'
            ORDER BY ts ASC
            """,
            (symbol,),
        ).fetchall()

        if not rows:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM intraday_bars
                WHERE symbol = %s AND interval_sec = %s
                ORDER BY ts DESC LIMIT 200
                """,
                (symbol, sec),
            ).fetchall()
            rows = list(reversed(rows))

        if not rows:
            return {"date": None, "window_minutes": window_minutes, "symbol": symbol, "candles": [], "predictions": []}

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["ist_dt"] = df["ts"].dt.tz_convert(IST)
        df["trade_date"] = df["ist_dt"].dt.date

        # Keep regular trading hours (09:00 - 23:30 IST for MCX, 09:15 - 15:30 IST for NSE)
        t_val = df["ist_dt"].dt.time
        if is_commodity:
            t_start = datetime.strptime("09:00:00", "%H:%M:%S").time()
            t_end = datetime.strptime("23:30:00", "%H:%M:%S").time()
        else:
            t_start = datetime.strptime("09:15:00", "%H:%M:%S").time()
            t_end = datetime.strptime("15:30:00", "%H:%M:%S").time()

        df = df[(t_val >= t_start) & (t_val < t_end)].reset_index(drop=True)
        if df.empty:
            return {"date": None, "window_minutes": window_minutes, "symbol": symbol, "candles": [], "predictions": []}

        if is_commodity:
            mins = ((df["ist_dt"].dt.hour * 60 + df["ist_dt"].dt.minute) - (9 * 60)).clip(lower=0)
        else:
            mins = ((df["ist_dt"].dt.hour * 60 + df["ist_dt"].dt.minute) - (9 * 60 + 15)).clip(lower=0)

        df["win_idx"] = mins // window_minutes

        win_bars = (
            df.groupby(["trade_date", "win_idx"], as_index=False)
            .agg(
                ts=("ts", "first"),
                ist_dt=("ist_dt", "first"),
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .sort_values("ts")
            .reset_index(drop=True)
        )

        # Technical indicators (warmed up across previous days, then sliced to latest trading day)
        win_bars = add_technical_features(win_bars)
        close_s = win_bars["close"].astype(float)
        win_bars["ema9"] = close_s.ewm(span=9, adjust=False).mean()
        win_bars["ema20"] = close_s.ewm(span=20, adjust=False).mean()
        win_bars["ema26"] = close_s.ewm(span=26, adjust=False).mean()
        win_bars["rsi_actual"] = win_bars["rsi_14"] * 50.0 + 50.0

        # True Session VWAP per trade_date
        tp = (win_bars["high"].astype(float) + win_bars["low"].astype(float) + win_bars["close"].astype(float)) / 3.0
        vol_pos = win_bars["volume"].astype(float).replace(0.0, 1.0)
        win_bars["_tp_v"] = tp * vol_pos
        win_bars["_v"] = vol_pos
        win_bars["vwap"] = (
            win_bars.groupby("trade_date")["_tp_v"].cumsum() / win_bars.groupby("trade_date")["_v"].cumsum()
        )

        latest_date = win_bars["trade_date"].iloc[-1]
        today_bars = win_bars[win_bars["trade_date"] == latest_date].reset_index(drop=True)
        prev_bars = win_bars[win_bars["trade_date"] < latest_date]
        pdh = round(float(prev_bars["high"].max()), 2) if not prev_bars.empty else None
        pdl = round(float(prev_bars["low"].min()), 2) if not prev_bars.empty else None
        prev_close = round(float(prev_bars["close"].iloc[-1]), 2) if not prev_bars.empty else None

        # Show up to 50 candles (includes prior session historic context + all of today's candles)
        chart_bars = win_bars.tail(max(50, len(today_bars))).reset_index(drop=True)
        chart_start_date = chart_bars["trade_date"].iloc[0]

        # Fetch predictions across the visible chart dates
        pred_rows = conn.execute(
            """
            SELECT prediction_id, as_of,
                   COALESCE(horizon_end, as_of + (window_minutes || ' minutes')::interval) AS h_end,
                   window_minutes, regime, prob_up, entry_condition, rationale,
                   realized_return, data_freshness, features_used
            FROM predictions
            WHERE instrument = %s AND (as_of AT TIME ZONE 'Asia/Kolkata')::date >= %s
            ORDER BY prediction_id ASC
            """,
            (symbol, chart_start_date),
        ).fetchall()

        candles_out = []
        today_start_idx = None
        for idx_c, r in chart_bars.iterrows():
            is_today = bool(r["trade_date"] == latest_date)
            if is_today and today_start_idx is None:
                today_start_idx = idx_c
            candles_out.append(
                {
                    "ts_epoch": int(r["ts"].timestamp()),
                    "ts_ist": r["ist_dt"].strftime("%H:%M"),
                    "date_ist": str(r["trade_date"]),
                    "is_today": is_today,
                    "open": round(float(r["open"]), 2),
                    "high": round(float(r["high"]), 2),
                    "low": round(float(r["low"]), 2),
                    "close": round(float(r["close"]), 2),
                    "vwap": round(float(r["vwap"]), 2),
                    "ema9": round(float(r["ema9"]), 2),
                    "ema20": round(float(r["ema20"]), 2),
                    "ema26": round(float(r["ema26"]), 2),
                    "rsi14": round(float(r["rsi_actual"]), 1),
                    "adx14": round(float(r["adx"]), 1),
                    "atr_pct": round(float(r["atr_pct"]) * 100.0, 3),
                }
            )

        preds_out = []
        for pr in pred_rows:
            p_id, p_as_of, p_hend, p_win, p_reg, p_prob, p_env, p_rat, p_ret, p_fresh, p_feat = pr
            prob_up_f = float(p_prob) if p_prob is not None else 0.5
            ret_f = float(p_ret) if p_ret is not None else None
            fresh_d = p_fresh or {}
            s_px = fresh_d.get("settled_start_price") or fresh_d.get("nifty_close")
            e_px = fresh_d.get("settled_end_price")
            p_ist = p_as_of.astimezone(IST)
            # Snap prediction to the nearest candle index on chart
            p_epoch = int(p_as_of.timestamp())
            best_idx = 0
            best_diff = 10**12
            for idx, c in enumerate(candles_out):
                diff = abs(c["ts_epoch"] - p_epoch)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            if best_diff > window_minutes * 120:
                continue

            post_an = None
            verdict_str = "PENDING"
            is_no_call = (p_reg == "sideways")
            if ret_f is not None and is_no_call:
                verdict_str = "NO CALL"
            elif ret_f is not None:
                is_hit = (prob_up_f >= 0.5 and ret_f >= 0) or (prob_up_f < 0.5 and ret_f < 0)
                verdict_str = "HIT" if is_hit else "MISS"
                post_an = build_post_prediction_analysis(
                    conn, p_as_of, p_hend, p_reg, prob_up_f, s_px, e_px, ret_f, p_feat, symbol=symbol
                )

            preds_out.append(
                {
                    "prediction_id": p_id,
                    "candle_idx": best_idx,
                    "ts_ist": p_ist.strftime("%H:%M"),
                    "end_ist": p_hend.astimezone(IST).strftime("%H:%M") if p_hend else "",
                    "window_minutes": int(p_win or 15),
                    "regime": "sideways" if is_no_call else ("bullish" if prob_up_f >= 0.5 else "bearish"),
                    "prob_up": round(prob_up_f * 100.0, 1),
                    "entry_condition": p_env,
                    "rationale": p_rat,
                    "verdict": verdict_str,
                    "realized_return_pct": round(ret_f * 100.0, 3) if ret_f is not None else None,
                    "post_analysis": post_an,
                }
            )

        # Build clear Historic + Live Indicator Scorecard explaining how the chart drives the prediction
        last_r = chart_bars.iloc[-1]
        cur_c = float(last_r["close"])
        cur_vwap = float(last_r["vwap"])
        cur_ema9 = float(last_r["ema9"])
        cur_ema20 = float(last_r["ema20"])
        cur_ema26 = float(last_r["ema26"])
        cur_rsi = float(last_r["rsi_actual"])
        cur_adx = float(last_r["adx"])
        cur_macd_hist = float(last_r.get("macd_hist", 0.0) or 0.0)
        ema_slope_pct = float(last_r.get("ema20_slope", 0.0) or 0.0) * 100.0
        vwap_diff = cur_c - cur_vwap
        ema_diff = cur_c - cur_ema20
        ema_cross_diff = cur_ema9 - cur_ema26

        # Visible Swing Support & Resistance Zones (20-bar window) + Pivot
        zone_bars = chart_bars.tail(20)
        res_high = round(float(zone_bars["high"].max()), 2)
        sup_low = round(float(zone_bars["low"].min()), 2)
        zone_band = max(10.0, round(cur_c * 0.0008, 1))
        res_bot = round(res_high - zone_band, 2)
        sup_top = round(sup_low + zone_band, 2)
        raw_pivot = last_r.get("pivot")
        pivot = round(float(raw_pivot) if raw_pivot is not None and not pd.isna(raw_pivot) else (res_high + sup_low + cur_c) / 3.0, 2)
        chan_span = max(res_high - sup_low, 1e-6)
        chan_pos_pct = max(0.0, min(100.0, ((cur_c - sup_low) / chan_span) * 100.0))

        latest_payload = fetch_latest_prediction_payload(dsn, symbol=symbol)
        feats = latest_payload.get("features_used") or {}
        mtf_score = float(feats.get("mtf_trend_score", 0.0) or 0.0)
        top5_ret = float(feats.get("top5_ret_1", 0.0) or 0.0) * 100.0
        bn_ret = float(feats.get("banknifty_ret_1", 0.0) or 0.0) * 100.0
        ad_ratio = float(feats.get("breadth_ad_ratio", 1.0) or 1.0)
        news_surp = float(feats.get("news_surprise", 0.0) or 0.0)
        event_shock = float(feats.get("signed_event_shock", 0.0) or 0.0)

        # Historic multi-day context
        hist_chg = (cur_c - prev_close) if prev_close else 0.0
        hist_dir = "ABOVE" if hist_chg >= 0 else "BELOW"
        pd_ctx = f" | PDH {pdh:,.1f} / PDL {pdl:,.1f}" if (pdh and pdl) else ""

        scorecard = [
            {
                "name": "EMA(9) vs EMA(26) Crossover",
                "value": f"EMA9 {cur_ema9:,.1f} vs EMA26 {cur_ema26:,.1f} ({ema_cross_diff:+.1f} pts)",
                "signal": "BULLISH" if ema_cross_diff > 2 else ("BEARISH" if ema_cross_diff < -2 else "NEUTRAL"),
                "note": "Fast 9-EMA above 26-EMA (bullish trend alignment)" if ema_cross_diff > 2 else ("Fast 9-EMA below 26-EMA (bearish trend alignment)" if ema_cross_diff < -2 else "9-EMA & 26-EMA converging (crossover watch)"),
            },
            {
                "name": "Session VWAP & EMA(20)",
                "value": f"VWAP {cur_vwap:,.1f} ({vwap_diff:+.1f}) | EMA20 {cur_ema20:,.1f} ({ema_diff:+.1f})",
                "signal": "BULLISH" if (vwap_diff > 5 and ema_diff > 0) else ("BEARISH" if (vwap_diff < -5 and ema_diff < 0) else "NEUTRAL"),
                "note": f"Spot above VWAP & 20-EMA (slope {ema_slope_pct:+.3f}%)" if (vwap_diff > 5 and ema_diff > 0) else (f"Spot below VWAP & 20-EMA (slope {ema_slope_pct:+.3f}%)" if (vwap_diff < -5 and ema_diff < 0) else f"Mixed around VWAP/20-EMA (slope {ema_slope_pct:+.3f}%)"),
            },
            {
                "name": "Support & Resistance Zones",
                "value": f"Res {res_bot:,.1f}–{res_high:,.1f} | Sup {sup_low:,.1f}–{sup_top:,.1f} ({chan_pos_pct:.0f}%)",
                "signal": "BEARISH" if chan_pos_pct >= 80 else ("BULLISH" if chan_pos_pct <= 20 else "NEUTRAL"),
                "note": f"Pivot {pivot:,.1f} · Channel {chan_pos_pct:.0f}% ({'near resistance cap' if chan_pos_pct >= 80 else ('near support floor' if chan_pos_pct <= 20 else 'mid-channel')}){pd_ctx}",
            },
            {
                "name": "MACD (12/26) & RSI(14)",
                "value": f"MACD Hist {cur_macd_hist:+.2f} | RSI {cur_rsi:.1f}",
                "signal": "BULLISH" if (cur_macd_hist > 0 and cur_rsi >= 50 and cur_rsi < 70) else ("BEARISH" if (cur_macd_hist < 0 and cur_rsi <= 50 and cur_rsi > 30) else ("BEARISH" if cur_rsi >= 70 else ("BULLISH" if cur_rsi <= 30 else "NEUTRAL"))),
                "note": "Overbought RSI (>=70)" if cur_rsi >= 70 else ("Oversold RSI (<=30)" if cur_rsi <= 30 else ("Positive MACD & RSI momentum" if cur_macd_hist > 0 else "Negative MACD & RSI momentum")),
            },
            {
                "name": "ADX(14) & MTF (15m/60m)",
                "value": f"ADX {cur_adx:.1f} | MTF Score {mtf_score:+.2f}",
                "signal": "BULLISH" if (mtf_score > 0.15 and cur_adx >= 18) else ("BEARISH" if (mtf_score < -0.15 and cur_adx >= 18) else ("TRENDING" if cur_adx >= 22 else "CHOPPY")),
                "note": f"15m/60m trend {'bullish' if mtf_score > 0.1 else ('bearish' if mtf_score < -0.1 else 'flat')}; ADX {'trending' if cur_adx >= 22 else 'range-bound (<22)'}",
            },
            {
                "name": "Heavyweights & News Sentiment",
                "value": f"Top5 {top5_ret:+.2f}% | BankNifty {bn_ret:+.2f}% | A/D {ad_ratio:.2f}",
                "signal": "BULLISH" if (top5_ret + bn_ret > 0.05 or news_surp + event_shock > 0.1) else ("BEARISH" if (top5_ret + bn_ret < -0.05 or news_surp + event_shock < -0.1) else "NEUTRAL"),
                "note": f"News surprise {news_surp:+.2f}, event shock {event_shock:+.2f} · Spot {abs(hist_chg):.1f} pts {hist_dir} prev close",
            },
        ]

        return {
            "symbol": symbol,
            "symbol_name": SUPPORTED_SYMBOLS.get(symbol, {}).get("name", symbol),
            "date": str(latest_date),
            "window_minutes": window_minutes,
            "is_market_open": is_market_open(symbol=symbol),
            "today_start_idx": today_start_idx,
            "pdh": pdh,
            "pdl": pdl,
            "prev_close": prev_close,
            "zones": {
                "res_top": res_high,
                "res_bot": res_bot,
                "sup_top": sup_top,
                "sup_bot": sup_low,
                "pivot": pivot,
            },
            "candles": candles_out,
            "predictions": preds_out,
            "scorecard": scorecard,
            "active_layman": latest_payload.get("layman_explanation"),
            "active_regime": latest_payload.get("regime"),
            "active_prob_up": latest_payload.get("favourable_pct"),
            "active_confidence": latest_payload.get("confidence_pct"),
            "signal_strength": latest_payload.get("signal_strength"),
        }


def start_background_ticker(dsn: str) -> None:
    """Daemon thread pushing live tick & sentiment updates to SSE clients every 2 seconds, and settling expired predictions."""
    global _LAST_PRED_RUN_TS, _ACTIVE_WINDOW

    # Start native real-time XTS broadcaster & Redis live tick subscriber
    t_xts = threading.Thread(target=_xts_live_feed_worker, args=(dsn,), daemon=True, name="XTSLiveFeedWorker")
    t_xts.start()

    # Track execution slots and calendar dates per (symbol, horizon) to ensure strictly one run per clock interval
    _LAST_RUN_SLOT: dict[tuple, tuple] = {}
    _LAST_RUN_DATE_BY_HORIZON: dict[tuple, str] = {}

    def _worker():
        global _LAST_PRED_RUN_TS, _ACTIVE_WINDOW
        HORIZONS = [5, 15, 30, 60, 375, 750]

        while True:
            try:
                now_utc = datetime.now(timezone.utc)
                now_epoch = time.time()
                now_ist = now_utc.astimezone(IST)
                today_str = str(now_ist.date())
                cur_hour = now_ist.hour
                cur_min = now_ist.minute
                cur_total_min = cur_hour * 60 + cur_min

                # 1. Broadcast live tick & status payload for all symbols to SSE clients
                for s in ("NIFTY50", "BANKNIFTY", "CRUDEOIL", "GOLD", "SILVER"):
                    try:
                        p = fetch_latest_prediction_payload(dsn, symbol=s)
                        broadcast_sse("tick", p)
                    except Exception:
                        pass

                # 2. Automatically settle matured predictions whose window has closed
                try:
                    settle_matured_predictions(dsn)
                except Exception:
                    pass

                # 3. Schedule and run predictions at exact canonical clock multiples
                for sym in ("NIFTY50", "BANKNIFTY", "CRUDEOIL", "GOLD", "SILVER"):
                    is_mcx = sym in ("CRUDEOIL", "GOLD", "SILVER")
                    open_min = 9 * 60 if is_mcx else (9 * 60 + 15)
                    close_min = (23 * 60 + 30) if is_mcx else (15 * 60 + 30)
                    market_open = is_market_open(now_utc, symbol=sym)

                    # --- 1 DAY HORIZON (hz=375: Current Day Close) ---
                    # Trigger: One time in a day at 9:15 am (or 9:00 for MCX), or whenever service starts during trading hours
                    if market_open and cur_total_min < (close_min - 30):
                        if _LAST_RUN_DATE_BY_HORIZON.get((sym, 375)) != today_str:
                            _LAST_RUN_DATE_BY_HORIZON[(sym, 375)] = today_str
                            snap_open_hour = 9
                            snap_open_min = 0 if is_mcx else 15
                            snap_ist = now_ist.replace(hour=snap_open_hour, minute=snap_open_min, second=0, microsecond=0)
                            as_of_utc = snap_ist.astimezone(timezone.utc)
                            try:
                                pred_res = run_agent(dsn, run_type="scheduled", window_minutes=375, symbol=sym, as_of_utc=as_of_utc, persist=True)
                                broadcast_sse("sentiment_update", pred_res)
                            except Exception:
                                pass

                    # --- NEXT DAY HORIZON (hz=750: Next Day Close) ---
                    # Trigger: At 4:00 pm (16:00 IST) for NSE/BSE after full-day data & news analysis; at 23:30 for MCX
                    trigger_next_day = (
                        (not is_mcx and (cur_hour > 16 or (cur_hour == 16 and cur_min >= 0))) or
                        (is_mcx and (cur_hour == 23 and cur_min >= 30))
                    )
                    if trigger_next_day:
                        if _LAST_RUN_DATE_BY_HORIZON.get((sym, 750)) != today_str:
                            _LAST_RUN_DATE_BY_HORIZON[(sym, 750)] = today_str
                            next_hour = 23 if is_mcx else 16
                            next_min = 30 if is_mcx else 0
                            snap_ist = now_ist.replace(hour=next_hour, minute=next_min, second=0, microsecond=0)
                            as_of_utc = snap_ist.astimezone(timezone.utc)
                            try:
                                pred_res = run_agent(dsn, run_type="scheduled", window_minutes=750, symbol=sym, as_of_utc=as_of_utc, persist=True)
                                broadcast_sse("sentiment_update", pred_res)
                            except Exception:
                                pass

                    # If market is not currently open, skip intraday intervals
                    if not market_open:
                        continue

                    mins_to_close = close_min - cur_total_min

                    # --- INTRADAY HORIZONS: 5m, 15m, 30m, 60m ---
                    for hz in (5, 15, 30, 60):
                        # Never trigger if remaining market time is less than horizon
                        if mins_to_close < hz:
                            continue

                        # Canonical Clock Multiple Check:
                        # 1) 5 Min: multiple of 5 (cur_min % 5 == 0) -> e.g. 9:30, 9:35, 9:40, ...
                        # 2) 15 Min: multiple of 15 (cur_min % 15 == 0) -> e.g. 9:30, 9:45, 10:00, ...
                        # 3) 30 Min: multiple of 30 (cur_min % 30 == 0) -> e.g. 9:30, 10:00, 10:30, ...
                        # 4) 60 Min: multiple of 60 on the hour (cur_min == 0) -> e.g. 10:00, 11:00, 12:00, ...
                        is_slot = False
                        if hz == 5 and cur_min % 5 == 0:
                            is_slot = True
                        elif hz == 15 and cur_min % 15 == 0:
                            is_slot = True
                        elif hz == 30 and cur_min % 30 == 0:
                            is_slot = True
                        elif hz == 60 and cur_min == 0:
                            is_slot = True

                        if not is_slot:
                            continue

                        slot_key = (today_str, hz, cur_hour, cur_min)
                        if _LAST_RUN_SLOT.get((sym, hz)) == slot_key:
                            continue

                        # Register slot immediately so it only executes once during this clock interval
                        _LAST_RUN_SLOT[(sym, hz)] = slot_key

                        # Snap as_of strictly to the exact canonical clock boundary (second=0, microsecond=0)
                        snap_ist = now_ist.replace(minute=cur_min, second=0, microsecond=0)
                        as_of_utc = snap_ist.astimezone(timezone.utc)

                        try:
                            pred_res = run_agent(
                                dsn,
                                run_type="scheduled",
                                window_minutes=hz,
                                symbol=sym,
                                as_of_utc=as_of_utc,
                                persist=True,
                            )
                            broadcast_sse("sentiment_update", pred_res)
                        except Exception:
                            pass

            except Exception:
                pass
            time.sleep(2.0)

    t = threading.Thread(target=_worker, daemon=True, name="RealTimeTickerWorker")
    t.start()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Nifty 50 Market Sentiment Agent — Live Test Console</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8; --green: #22c55e; --amber: #f59e0b; --red: #ef4444; }
    body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
    h1, h2, h3 { margin-top: 0; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; border-bottom: 1px solid #334155; padding-bottom: 12px; }
    .controls { display: flex; gap: 8px; flex-wrap: wrap; }
    button { background: #2563eb; color: #fff; border: none; padding: 7px 13px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }
    button:hover { background: #1d4ed8; }
    button.kill { background: #dc2626; }
    button.resume { background: #16a34a; }
    .live-ticker-banner { background: #0b1120; border: 1px solid #334155; border-radius: 10px; padding: 12px 18px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    .grid { display: grid; grid-template-columns: 1.6fr 1.4fr; gap: 18px; margin-bottom: 18px; }
    .card { background: var(--card); border-radius: 10px; padding: 16px; border: 1px solid #334155; }
    .badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-weight: 700; font-size: 12px; text-transform: uppercase; }
    .bullish, .favourable, .ok { background: rgba(34,197,94,0.2); color: var(--green); }
    .sideways, .cautious, .degraded { background: rgba(245,158,11,0.2); color: var(--amber); }
    .bearish, .avoid, .suppressed { background: rgba(239,68,68,0.2); color: var(--red); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #334155; }
    th { color: var(--muted); font-weight: 600; }
    pre { background: #090d16; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; color: #cbd5e1; white-space: pre-wrap; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
    .metric-box { background: #0f172a; padding: 10px; border-radius: 8px; border: 1px solid #334155; }
    .metric-label { font-size: 11.5px; color: var(--muted); }
    .metric-val { font-size: 18px; font-weight: 700; margin-top: 3px; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.3; transform: scale(0.85); } }
    .gauge-wrapper { background: #090d16; border-radius: 10px; padding: 10px 14px 8px 14px; border: 1px solid #334155; margin-bottom: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.25); }
    .gauge-svg-container { position: relative; width: 100%; max-width: 220px; height: 110px; margin: 0 auto; }
    .sym-pill { background: #0f172a; color: #94a3b8; border: 1px solid #334155; padding: 6px 14px; border-radius: 8px; font-size: 13px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s ease; }
    .sym-pill:hover { background: #1e293b; color: #f8fafc; border-color: #64748b; }
    .sym-pill.active { background: #0284c7; color: #ffffff; border-color: #38bdf8; font-weight: 700; box-shadow: 0 0 12px rgba(56, 189, 248, 0.4); }
  </style>
</head>
<body>
  <!-- Top Symbol Switcher Bar -->
  <div style="display:flex; align-items:center; justify-content:space-between; background:#0b1120; border:1px solid #334155; border-radius:10px; padding:10px 16px; margin-bottom:14px; flex-wrap:wrap; gap:12px;">
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
      <span style="font-size:12px; font-weight:800; color:#38bdf8; text-transform:uppercase; letter-spacing:0.8px;">⚡ SELECT INSTRUMENT:</span>
      <div id="symbol-pills" style="display:flex; gap:8px; flex-wrap:wrap;">
        <button class="sym-pill active" onclick="switchSymbol('NIFTY50')" id="btn-sym-NIFTY50">
          <span>NIFTY 50</span>
          <span style="font-size:10px; opacity:0.75; background:rgba(0,0,0,0.3); padding:2px 5px; border-radius:4px;">NSE</span>
        </button>
        <button class="sym-pill" onclick="switchSymbol('BANKNIFTY')" id="btn-sym-BANKNIFTY">
          <span>BANK NIFTY</span>
          <span style="font-size:10px; opacity:0.75; background:rgba(0,0,0,0.3); padding:2px 5px; border-radius:4px;">NSE</span>
        </button>
        <button class="sym-pill" onclick="switchSymbol('CRUDEOIL')" id="btn-sym-CRUDEOIL">
          <span>CRUDE OIL</span>
          <span style="font-size:10px; opacity:0.75; background:rgba(0,0,0,0.3); padding:2px 5px; border-radius:4px;">MCX</span>
        </button>
        <button class="sym-pill" onclick="switchSymbol('GOLD')" id="btn-sym-GOLD">
          <span>GOLD</span>
          <span style="font-size:10px; opacity:0.75; background:rgba(0,0,0,0.3); padding:2px 5px; border-radius:4px;">MCX</span>
        </button>
        <button class="sym-pill" onclick="switchSymbol('SILVER')" id="btn-sym-SILVER">
          <span>SILVER</span>
          <span style="font-size:10px; opacity:0.75; background:rgba(0,0,0,0.3); padding:2px 5px; border-radius:4px;">MCX</span>
        </button>
      </div>
    </div>
    <div style="font-size:12.5px; color:#cbd5e1; background:#1e293b; padding:5px 12px; border-radius:6px; border:1px solid #334155;">
      Active: <strong id="active-symbol-title" style="color:#38bdf8;">Nifty 50 (NSECM:26000)</strong>
    </div>
  </div>

  <div class="header">
    <div>
      <h1><span id="dash-symbol-name">Nifty 50</span> Market Sentiment Agent <span id="market-status-badge" class="badge cautious" style="font-size:12px; vertical-align:middle; margin-left:10px;">🌙 MARKET CLOSED</span></h1>
      <div style="color: var(--muted); font-size: 13.5px;">Interactive Real-Time Testing & Verification Dashboard (Phases 0–7)</div>
    </div>
    <div class="controls">
      <button style="background:#7c3aed" onclick="runPrediction(1)">⚡ Run 1m</button>
      <button style="background:#4f46e5" onclick="runPrediction(5)">Run 5m</button>
      <button onclick="runPrediction(15)">Run 15m</button>
      <button onclick="runPrediction(30)">Run 30m (Primary)</button>
      <button onclick="runPrediction(60)">Run 60m</button>
      <button style="background:#0284c7" onclick="loadAll()">🔄 Refresh Now</button>
      <a href="/v1/sentiment/nifty/backtest-5m/excel" style="text-decoration:none"><button style="background:#059669">📥 Download Excel (.xlsx)</button></a>
      <a href="/v1/sentiment/nifty/backtest-5m/csv" style="text-decoration:none"><button style="background:#0d9488">📥 Download CSV</button></a>
      <button class="kill" onclick="toggleKill(true)">Kill-Switch</button>
      <button class="resume" onclick="toggleKill(false)">Resume</button>
    </div>
  </div>

  <!-- Real-Time Spot Ticker Banner (Always Visible at Top) -->
  <div class="live-ticker-banner">
    <div style="display:flex; align-items:center; gap:22px; flex-wrap:wrap;">
      <div>
        <div id="live-symbol-name" style="font-size:11px; text-transform:uppercase; color:var(--muted); font-weight:700; letter-spacing:0.5px;">NIFTY 50 SPOT</div>
        <div style="display:flex; align-items:baseline; gap:10px;">
          <span id="live-spot-price" style="font-size:28px; font-weight:800; color:#f8fafc; font-variant-numeric:tabular-nums;">--</span>
          <span id="live-spot-chg" style="font-size:14px; font-weight:700; color:var(--green);">--</span>
        </div>
      </div>
      <div style="border-left:1px solid #334155; padding-left:18px; display:flex; gap:18px; font-size:12.5px; flex-wrap:wrap;">
        <div><span style="color:var(--muted);">Day Open:</span> <strong id="live-day-open" style="color:#f8fafc;">--</strong></div>
        <div><span style="color:var(--muted);">Day High:</span> <strong id="live-day-high" style="color:#f8fafc;">--</strong></div>
        <div><span style="color:var(--muted);">Day Low:</span> <strong id="live-day-low" style="color:#f8fafc;">--</strong></div>
        <div><span style="color:var(--muted);">India VIX:</span> <strong id="live-vix" style="color:#38bdf8;">--</strong></div>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
      <div style="display:inline-flex; align-items:center; gap:6px; background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.3); padding:4px 11px; border-radius:999px; font-size:12px; color:var(--green); font-weight:700;">
        <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); animation: pulse 1.2s infinite;"></span>
        LIVE REALTIME FEED
      </div>
      <span id="live-last-tick-time" style="font-size:11.5px; color:var(--muted);">Connecting...</span>
    </div>
  </div>

  <!-- Live Intraday Chart + Indicators + AI Prediction Overlay -->
  <div class="card" style="margin-bottom:18px; position:relative;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px;">
      <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
        <h2 style="margin:0; font-size:16px;">📈 Live <span id="chart-symbol-name">Nifty 50</span> Chart, Indicators &amp; AI Trend Analysis</h2>
        <div style="display:flex; gap:6px;">
          <button id="chart-tf-5" onclick="setChartTf(5)" style="padding:4px 10px; font-size:12px; background:#0284c7;">5m</button>
          <button id="chart-tf-15" onclick="setChartTf(15)" style="padding:4px 10px; font-size:12px; background:#334155;">15m</button>
        </div>
        <span id="chart-session-date" style="font-size:12px; color:var(--muted);">Session: --</span>
      </div>
      <div style="display:flex; gap:14px; font-size:12px; flex-wrap:wrap; align-items:center;">
        <span><span style="color:#f59e0b; font-weight:700;">━</span> VWAP: <strong id="ind-vwap" style="color:#f59e0b;">--</strong></span>
        <span><span style="color:#4ade80; font-weight:700;">━</span> EMA(9): <strong id="ind-ema9" style="color:#4ade80;">--</strong></span>
        <span><span style="color:#38bdf8; font-weight:700;">━</span> EMA(20): <strong id="ind-ema20" style="color:#38bdf8;">--</strong></span>
        <span><span style="color:#f472b6; font-weight:700;">━</span> EMA(26): <strong id="ind-ema26" style="color:#f472b6;">--</strong></span>
        <span><span style="color:#ef4444; font-weight:700;">▬</span>/<span style="color:#22c55e; font-weight:700;">▬</span> Res/Sup Zones</span>
        <span>RSI(14): <strong id="ind-rsi" style="color:#f8fafc;">--</strong></span>
        <span>ADX(14): <strong id="ind-adx" style="color:#f8fafc;">--</strong></span>
        <span style="color:#22c55e; font-weight:700;">▲ UP Pred</span>
        <span style="color:#ef4444; font-weight:700;">▼ DOWN Pred</span>
      </div>
    </div>

    <div style="position:relative;">
      <canvas id="live-chart-canvas" style="width:100%; height:360px; display:block; background:#0b1120; border-radius:8px; border:1px solid #334155; cursor:crosshair;"></canvas>
      <div id="chart-tooltip" style="display:none; position:absolute; top:10px; left:12px; background:rgba(15,23,42,0.95); border:1px solid #475569; border-radius:8px; padding:10px 12px; font-size:12px; max-width:420px; pointer-events:none; box-shadow:0 8px 24px rgba(0,0,0,0.5); z-index:20;"></div>
    </div>

    <div id="chart-ai-analysis" style="margin-top:10px; background:#0f172a; border:1px solid #334155; border-left:4px solid #38bdf8; border-radius:6px; padding:9px 14px; font-size:12.5px; color:#cbd5e1; line-height:1.5;">
      <strong style="color:#38bdf8;">🧠 Chart Signal &amp; Rationale:</strong> <span id="chart-ai-text">Loading live chart indicators and AI reasoning...</span>
    </div>

    <!-- Historic + Live Indicator Scorecard -->
    <div id="chart-scorecard" style="margin-top:10px; display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:8px;"></div>
  </div>

  <div class="grid">
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <h2 style="margin:0; font-size:16px;">Real-Time Prediction Meter & Outlook</h2>
        <div style="display:flex; align-items:center; gap:6px; font-size:11.5px; color:#22c55e; font-weight:700;">
          <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e; animation: pulse 1.5s infinite;"></span>
          LIVE AUTO-TICKING
        </div>
      </div>

      <!-- Binary Speedometer / Gauge Meter (Left Red: Bearish 0-50% | Right Green: Bullish 50-100%) -->
      <div class="gauge-wrapper">
        <div class="gauge-svg-container">
          <svg viewBox="0 0 280 140" style="width:100%; height:100%; overflow:visible;">
            <defs>
              <linearGradient id="bearishGrad" x1="0%" y1="100%" x2="50%" y2="0%">
                <stop offset="0%" stop-color="#b91c1c" />
                <stop offset="100%" stop-color="#ef4444" />
              </linearGradient>
              <linearGradient id="bullishGrad" x1="50%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#22c55e" />
                <stop offset="100%" stop-color="#15803d" />
              </linearGradient>
              <filter id="needleGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000" flood-opacity="0.8"/>
              </filter>
            </defs>

            <!-- Background Track -->
            <path d="M 35 125 A 105 105 0 0 1 245 125" fill="none" stroke="#1e293b" stroke-width="16" stroke-linecap="round" />

            <!-- Left Arc: Bearish Red (180° to 90°) -->
            <path d="M 35 125 A 105 105 0 0 1 140 20" fill="none" stroke="url(#bearishGrad)" stroke-width="15" stroke-linecap="round" />

            <!-- Right Arc: Bullish Green (90° to 0°) -->
            <path d="M 140 20 A 105 105 0 0 1 245 125" fill="none" stroke="url(#bullishGrad)" stroke-width="15" stroke-linecap="round" />

            <!-- Center 50% Divider Tick -->
            <line x1="140" y1="12" x2="140" y2="28" stroke="#f8fafc" stroke-width="2" />

            <!-- Zone Labels (Binary: BEARISH vs BULLISH) -->
            <text x="32" y="137" fill="#ef4444" font-size="10" font-weight="800" text-anchor="middle">BEARISH</text>
            <text x="140" y="10" fill="#94a3b8" font-size="9" font-weight="700" text-anchor="middle">50% THRESHOLD</text>
            <text x="248" y="137" fill="#22c55e" font-size="10" font-weight="800" text-anchor="middle">BULLISH</text>

            <!-- Calibration Percentages -->
            <text x="35" y="112" fill="#ef4444" font-size="8.5" font-weight="700" text-anchor="middle">100%</text>
            <text x="75" y="60" fill="#94a3b8" font-size="8" font-weight="600" text-anchor="middle">60%</text>
            <text x="140" y="38" fill="#f8fafc" font-size="8.5" font-weight="700" text-anchor="middle">50%</text>
            <text x="205" y="60" fill="#94a3b8" font-size="8" font-weight="600" text-anchor="middle">60%</text>
            <text x="245" y="112" fill="#22c55e" font-size="8.5" font-weight="700" text-anchor="middle">100%</text>

            <!-- Dynamic Live Needle (Pivots at center 140, 125) -->
            <g id="meter-needle" style="transform-origin: 140px 125px; transform: rotate(0deg); transition: transform 0.6s cubic-bezier(0.34, 1.3, 0.64, 1);" filter="url(#needleGlow)">
              <polygon points="137,125 140,24 143,125" fill="#f8fafc" />
              <circle cx="140" cy="125" r="8" fill="#0284c7" stroke="#f8fafc" stroke-width="2" />
              <circle cx="140" cy="125" r="3" fill="#f8fafc" />
            </g>
          </svg>
        </div>

        <!-- Digital Readout below Gauge -->
        <div style="text-align:center; margin-top:-2px;">
          <div id="meter-fav-pct" style="font-size:26px; font-weight:800; letter-spacing:-0.5px; color:#f8fafc; font-variant-numeric:tabular-nums;">50.0%</div>
          <div id="meter-regime-badge" style="margin-top:2px; display:flex; justify-content:center; gap:8px; align-items:center;">
            <span class="badge bullish" style="font-size:11px; padding:3px 12px;">BULLISH</span>
          </div>
          <div id="meter-sub-text" style="font-size:12px; color:var(--muted); margin-top:5px;">
            P(Up): <strong id="meter-fav-str" style="color:#f8fafc;">50.0%</strong> |
            P(Down): <strong id="meter-down-str" style="color:#ef4444;">50.0%</strong> |
            Confidence: <strong id="meter-conf-val" style="color:#38bdf8;">50.0%</strong> |
            Edge: <strong id="meter-conv-str" style="color:#f59e0b;">--</strong>
          </div>
        </div>

        <!-- Plain-English Layman Explanation Card -->
        <div style="background:#0f172a; border-radius:8px; border:1px solid #334155; padding:10px 14px; margin-top:10px; font-size:12.5px; line-height:1.55; color:#cbd5e1;">
          <div style="font-size:11px; text-transform:uppercase; color:var(--accent); font-weight:700; margin-bottom:4px; letter-spacing:0.5px;">
            💡 Why This Prediction Was Made (Plain-English Rationale)
          </div>
          <div id="meter-layman-text">Analyzing live spot order flow, Volume VWAP, and heavyweight momentum...</div>
        </div>
      </div>

      <div id="latest-summary">Loading...</div>
      <h3 style="margin-top:14px; font-size:14px;">Compliance-Rendered Advisory Output</h3>
      <pre id="rendered-text" style="max-height:160px; overflow-y:auto;">Loading...</pre>
    </div>
    <div class="card">
      <h2>Shadow-Run & Calibration Metrics</h2>
      <div id="monitoring-box">Loading...</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:12px;">
      <h2 style="margin:0;">Recent Predictions &amp; Settled Outcomes (Auto-Settles When Window Ends)</h2>
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <label style="font-size:12.5px; color:var(--muted); font-weight:600;">Date:</label>
        <select id="hist-date-select" onchange="loadTables()" style="background:#0f172a; color:#f8fafc; border:1px solid #334155; padding:5px 10px; border-radius:6px; font-size:12.5px; font-weight:600;">
          <option value="">All Available Dates</option>
        </select>
        <label style="font-size:12.5px; color:var(--muted); font-weight:600;">Horizon:</label>
        <select id="hist-window-select" onchange="loadTables()" style="background:#0f172a; color:#38bdf8; border:1px solid #334155; padding:5px 10px; border-radius:6px; font-size:12.5px; font-weight:700;">
          <option value="">All Horizons</option>
          <option value="5">5m (Scalp)</option>
          <option value="15">15m (Intraday)</option>
          <option value="30">30m (Swing)</option>
          <option value="60">60m (Hourly)</option>
          <option value="375">Current Day (Close 15:30)</option>
          <option value="750">Next Day (Close 15:30)</option>
        </select>
        <button onclick="loadTables()" style="padding:5px 11px; font-size:12px; background:#1e293b; border:1px solid #334155;">🔄 Refresh</button>
      </div>
    </div>
    <div style="max-height:480px; overflow-y:auto; border:1px solid #334155; border-radius:8px;">
    <table id="history-table" style="margin-top:0;">
      <thead>
        <tr>
          <th>ID</th>
          <th>Prediction Window (IST)</th>
          <th>Horizon</th>
          <th>Predicted Regime</th>
          <th>P(Up)</th>
          <th>Entry Env</th>
          <th id="th-spot-col">Spot (Start → End)</th>
          <th>Realized Move</th>
          <th>Outcome Verdict</th>
          <th>Post-Window Analysis</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
    </div>

    <!-- Detailed Post-Prediction Window Analysis Card -->
    <div id="post-analysis-panel" style="margin-top:16px; background:#0b1120; border:1px solid #334155; border-radius:10px; padding:14px 18px; display:none;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px; border-bottom:1px solid #1e293b; padding-bottom:8px;">
        <div>
          <span style="font-size:11px; text-transform:uppercase; color:var(--accent); font-weight:700; letter-spacing:0.5px;">🔍 Post-Prediction Window Breakdown</span>
          <h3 id="pa-title" style="margin:2px 0 0 0; font-size:15px;">Prediction #--</h3>
        </div>
        <span id="pa-badge" class="badge">--</span>
      </div>
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; font-size:12.5px; line-height:1.55;">
        <div style="background:#1e293b; padding:10px 12px; border-radius:8px; border-left:3px solid #38bdf8;">
          <div style="font-size:11px; color:#38bdf8; font-weight:700; text-transform:uppercase; margin-bottom:4px;">1. Why Model Predicted This at Entry</div>
          <div id="pa-why-pred" style="color:#e2e8f0;">--</div>
        </div>
        <div style="background:#1e293b; padding:10px 12px; border-radius:8px; border-left:3px solid #a855f7;">
          <div style="font-size:11px; color:#c084fc; font-weight:700; text-transform:uppercase; margin-bottom:4px;">2. What Actually Happened in This Period</div>
          <div id="pa-what-happened" style="color:#e2e8f0;">--</div>
        </div>
        <div id="pa-opp-box" style="background:#1e293b; padding:10px 12px; border-radius:8px; border-left:3px solid #f59e0b;">
          <div id="pa-opp-title" style="font-size:11px; color:#fbbf24; font-weight:700; text-transform:uppercase; margin-bottom:4px;">3. Why Opposite Move Wasn't Picked Earlier</div>
          <div id="pa-why-opp" style="color:#e2e8f0;">--</div>
        </div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <h2 style="margin-bottom:4px;">Walk-Forward Backtest Explorer (5m / 15m / 30m / 60m Candles | 0 Lookahead Violations)</h2>
        <div id="bt-summary-banner" style="font-size:13px;color:var(--muted);">Loading backtest...</div>
      </div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <label style="font-size:13px;color:var(--muted);">Candle Horizon:</label>
        <select id="bt-win-select" onchange="btDatesLoaded=false; loadBacktest('', this.value)" style="background:#0f172a;color:#38bdf8;border:1px solid #334155;padding:7px 12px;border-radius:6px;font-weight:700;">
          <option value="5" selected>5m Candles (15 Days | 1,110 Candles)</option>
          <option value="15">15m Candles (60 Days | 1,440 Candles)</option>
          <option value="30">30m Candles — Primary (60 Days | 720 Candles)</option>
          <option value="60">60m Candles (60 Days | 360 Candles)</option>
          <option value="375">Current Day (Close 15:30 IST)</option>
          <option value="750">Next Day (Close 15:30 IST)</option>
        </select>
        <label style="font-size:13px;color:var(--muted);">Trading Date:</label>
        <select id="bt-date-select" onchange="loadBacktest(this.value, document.getElementById('bt-win-select').value)" style="background:#0f172a;color:#f8fafc;border:1px solid #334155;padding:7px 12px;border-radius:6px;font-weight:600;"></select>
        <a href="/v1/sentiment/nifty/backtest-5m/excel" style="text-decoration:none"><button style="background:#059669">📥 Download Multi-Horizon Excel (.xlsx)</button></a>
      </div>
    </div>

    <h3 style="margin-top:16px;margin-bottom:6px;font-size:14px;color:var(--accent);">Daily Accuracy Summary (Click any row or use dropdown above to inspect candles/predictions)</h3>
    <div style="max-height:220px;overflow-y:auto;border:1px solid #334155;border-radius:8px;margin-bottom:16px;">
      <table id="bt-daily-table" style="margin-top:0;">
        <thead>
          <tr>
            <th>Trade Date</th>
            <th>Predictions / Candles</th>
            <th>Hits</th>
            <th>Misses</th>
            <th>Directional Accuracy</th>
            <th>Day Nifty Open</th>
            <th>Day Nifty Close</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

    <h3 id="bt-candles-title" style="margin-top:12px;margin-bottom:6px;font-size:14px;color:var(--accent);">Candle-by-Candle Predictions</h3>
    <div style="max-height:420px;overflow-y:auto;border:1px solid #334155;border-radius:8px;">
      <table id="bt-candles-table" style="margin-top:0;">
        <thead>
          <tr>
            <th>Candle Start (IST)</th>
            <th>Nifty Open</th>
            <th>Nifty Close (t)</th>
            <th>Next Close</th>
            <th>India VIX</th>
            <th>ADX(14)</th>
            <th>Breadth</th>
            <th>P(Up)</th>
            <th>Predicted</th>
            <th>Actual</th>
            <th>Realized Return</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Latest Scored News (Lisuns API + Hybrid Fin Scorer)</h2>
    <table id="news-table">
      <thead><tr><th>Published</th><th>Headline</th><th>Polarity</th><th>Theme</th><th>Event</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <script>
    let currentSymbol = 'NIFTY50';
    const SYMBOL_META = {
      'NIFTY50': { name: 'Nifty 50', tag: 'NSECM:26000' },
      'BANKNIFTY': { name: 'Bank Nifty', tag: 'NSECM:26001' },
      'CRUDEOIL': { name: 'Crude Oil', tag: 'MCXFO:573422 (19 Nov)' },
      'GOLD': { name: 'Gold', tag: 'MCXFO:495213 (04 Dec)' },
      'SILVER': { name: 'Silver', tag: 'MCXFO:495214 (04 Dec)' },
    };

    function switchSymbol(sym) {
      currentSymbol = sym;
      document.querySelectorAll('.sym-pill').forEach(b => b.classList.remove('active'));
      const btn = document.getElementById('btn-sym-' + sym);
      if (btn) btn.classList.add('active');

      const meta = SYMBOL_META[sym] || { name: sym, tag: sym };
      const titleEl = document.getElementById('active-symbol-title');
      if (titleEl) titleEl.textContent = `${meta.name} (${meta.tag})`;
      const dashEl = document.getElementById('dash-symbol-name');
      if (dashEl) dashEl.textContent = meta.name;
      const liveEl = document.getElementById('live-symbol-name');
      if (liveEl) liveEl.textContent = `${meta.name.toUpperCase()} SPOT`;
      const chartEl = document.getElementById('chart-symbol-name');
      if (chartEl) chartEl.textContent = meta.name;
      const spotTh = document.getElementById('th-spot-col');
      if (spotTh) spotTh.textContent = `${meta.name} Spot (Start → End)`;

      histDatesLoaded = false;
      const dateSel = document.getElementById('hist-date-select');
      if (dateSel) dateSel.innerHTML = '<option value="">All Available Dates</option>';

      loadAll();
    }

    function updateLiveDisplay(latest) {
      if (!latest) return;
      if (latest.instrument && latest.instrument !== currentSymbol) return;

      // 1. Live Nifty 50 Spot banner at top
      if (latest.live_tick) {
        const t = latest.live_tick;
        const spotEl = document.getElementById('live-spot-price');
        if (spotEl) spotEl.textContent = Number(t.spot_price).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        
        const chgEl = document.getElementById('live-spot-chg');
        if (chgEl) {
          const isUp = Number(t.day_change) >= 0;
          chgEl.textContent = `${isUp ? '+' : ''}${Number(t.day_change).toFixed(2)} (${isUp ? '+' : ''}${Number(t.day_change_pct).toFixed(2)}%)`;
          chgEl.style.color = isUp ? 'var(--green)' : 'var(--red)';
        }

        const openEl = document.getElementById('live-day-open');
        if (openEl) openEl.textContent = Number(t.day_open).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});

        const highEl = document.getElementById('live-day-high');
        if (highEl) highEl.textContent = Number(t.day_high).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});

        const lowEl = document.getElementById('live-day-low');
        if (lowEl) lowEl.textContent = Number(t.day_low).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});

        const vixEl = document.getElementById('live-vix');
        if (vixEl) vixEl.textContent = Number(t.vix).toFixed(2);

        const timeEl = document.getElementById('live-last-tick-time');
        if (timeEl) timeEl.textContent = `Last Tick: ${t.last_tick_time || ''}`;
      }

      if (latest.status === 'suppressed') {
        const sumEl = document.getElementById('latest-summary');
        if (sumEl) sumEl.innerHTML = `<span class="badge suppressed">SUPPRESSED</span> <p>${latest.reason || latest.rationale}</p>`;
        const renEl = document.getElementById('rendered-text');
        if (renEl) renEl.textContent = JSON.stringify(latest, null, 2);
        const needleEl = document.getElementById('meter-needle');
        if (needleEl) needleEl.style.transform = 'rotate(0deg)';
        const meterPctEl = document.getElementById('meter-fav-pct');
        if (meterPctEl) { meterPctEl.textContent = 'SUPPRESSED'; meterPctEl.style.color = 'var(--muted)'; }
        const badgeEl = document.getElementById('meter-regime-badge');
        if (badgeEl) badgeEl.innerHTML = '<span class="badge suppressed">SUPPRESSED</span>';
        return;
      }

      // 2. Speedometer Needle & Digital Readout
      const mBadge = document.getElementById('market-status-badge');
      if (mBadge) {
        const isComm = ['CRUDEOIL', 'GOLD', 'SILVER'].includes(currentSymbol);
        if (latest.is_market_open) {
          mBadge.className = 'badge bullish';
          mBadge.innerHTML = isComm ? '🟢 MCX LIVE (09:00–23:30 IST)' : '🟢 NSE LIVE (09:15–15:30 IST)';
        } else {
          mBadge.className = 'badge cautious';
          mBadge.innerHTML = isComm ? '🌙 MCX CLOSED (Resumes 09:00 IST)' : '🌙 MARKET CLOSED (Trading resumes 09:15 IST)';
        }
      }

      const favPct = Number(latest.favourable_pct != null ? latest.favourable_pct : (latest.prob_up != null ? latest.prob_up * 100.0 : 50.0));
      const downPct = Math.max(0.0, Math.min(100.0, 100.0 - favPct));
      const convPct = Number(latest.conviction_pct != null ? latest.conviction_pct : (latest.entry_score != null ? latest.entry_score * 100.0 : 50.0));

      const clamped = Math.max(0.0, Math.min(100.0, favPct));
      const angle = (clamped - 50.0) * 1.8;

      const needleEl = document.getElementById('meter-needle');
      if (needleEl) {
        needleEl.style.transform = `rotate(${angle}deg)`;
      }

      const meterPctEl = document.getElementById('meter-fav-pct');
      if (meterPctEl) {
        if (latest.regime === 'bearish') {
          meterPctEl.textContent = `${downPct.toFixed(1)}% Down`;
          meterPctEl.style.color = 'var(--red)';
        } else if (latest.regime === 'bullish') {
          meterPctEl.textContent = `${favPct.toFixed(1)}% Up`;
          meterPctEl.style.color = 'var(--green)';
        } else {
          meterPctEl.textContent = `${favPct.toFixed(1)}% (Neutral)`;
          meterPctEl.style.color = 'var(--amber)';
        }
      }

      const badgeEl = document.getElementById('meter-regime-badge');
      if (badgeEl) {
        badgeEl.innerHTML = `<span class="badge ${latest.regime}" style="font-size:12px; padding:4px 14px;">${regimeLabel(latest.regime)}</span>`;
      }

      const favStrEl = document.getElementById('meter-fav-str');
      if (favStrEl) favStrEl.textContent = `${favPct.toFixed(1)}%`;
      const downStrEl = document.getElementById('meter-down-str');
      if (downStrEl) downStrEl.textContent = `${downPct.toFixed(1)}%`;

      const confValEl = document.getElementById('meter-conf-val');
      if (confValEl) {
        const dc = Number(latest.confidence_pct != null ? latest.confidence_pct : Math.max(favPct, 100 - favPct));
        confValEl.textContent = `${dc.toFixed(1)}%`;
        confValEl.style.color = dc >= 60 ? '#22c55e' : (dc <= 53.5 ? '#f59e0b' : '#38bdf8');
      }

      const convStrEl = document.getElementById('meter-conv-str');
      if (convStrEl) {
        const strength = latest.signal_strength || (Math.abs(favPct - 50) <= 3.5 ? 'LOW EDGE (COINFLIP)' : 'MODERATE EDGE');
        convStrEl.textContent = strength;
        convStrEl.style.color = strength.includes('LOW') ? '#f59e0b' : (strength.includes('HIGH') ? '#22c55e' : '#38bdf8');
      }

      // 3. Layman Plain-English Explanation Box
      const laymanEl = document.getElementById('meter-layman-text');
      if (laymanEl && latest.layman_explanation) {
        laymanEl.innerHTML = latest.layman_explanation;
      }

      // 4. Advisory summary & compliance box
      const sumEl = document.getElementById('latest-summary');
      if (sumEl) {
        sumEl.innerHTML = `
          <div class="metrics">
            <div class="metric-box"><div class="metric-label">Market Regime (${latest.window_minutes}m)</div><div class="metric-val"><span class="badge ${latest.regime}">${regimeLabel(latest.regime)}</span></div></div>
            <div class="metric-box"><div class="metric-label">Direction P(Up)</div><div class="metric-val" style="color:${favPct>=58?'var(--green)':(favPct<=42?'var(--red)':'var(--amber)')}">${favPct.toFixed(1)}%</div></div>
            <div class="metric-box"><div class="metric-label">Entry Environment</div><div class="metric-val"><span class="badge ${latest.entry_condition}">${latest.entry_condition}</span></div></div>
          </div>
          <div style="font-size:12.5px; color:var(--muted);"><strong>Invalidation:</strong> ${latest.invalidation || 'N/A'}</div>
        `;
      }

      const renEl = document.getElementById('rendered-text');
      if (renEl && latest.rendered_text) {
        renEl.textContent = latest.rendered_text;
      }
    }

    async function pollLiveTick() {
      try {
        const res = await fetch('/v1/sentiment/nifty/latest?symbol=' + encodeURIComponent(currentSymbol));
        const data = await res.json();
        updateLiveDisplay(data);
      } catch(err){}
    }

    let histDatesLoaded = false;
    async function loadTables() {
      try {
        const dateSel = document.getElementById('hist-date-select');
        const winSel = document.getElementById('hist-window-select');
        const selDate = dateSel ? dateSel.value : '';
        const selWin = winSel ? winSel.value : '';

        let histUrl = '/v1/sentiment/nifty/history?limit=100&symbol=' + encodeURIComponent(currentSymbol);
        if (selDate) histUrl += '&date=' + encodeURIComponent(selDate);
        if (selWin) histUrl += '&window=' + encodeURIComponent(selWin);

        const [mon, hist, news] = await Promise.all([
          fetch('/v1/sentiment/nifty/monitoring?symbol=' + encodeURIComponent(currentSymbol)).then(r => r.json()),
          fetch(histUrl).then(r => r.json()),
          fetch('/v1/sentiment/nifty/news?limit=12').then(r => r.json())
        ]);

        // Populate available dates dropdown dynamically
        if (dateSel && (!histDatesLoaded || dateSel.options.length <= 1)) {
          const prevVal = dateSel.value;
          const opts = ['<option value="">All Available Dates</option>'];
          (hist.available_dates || []).forEach(d => {
            const isSel = (d === prevVal) ? 'selected' : '';
            opts.push(`<option value="${d}" ${isSel}>${d}</option>`);
          });
          dateSel.innerHTML = opts.join('');
          histDatesLoaded = true;
        }

        document.getElementById('monitoring-box').innerHTML = `
          <div class="metrics" style="grid-template-columns: 1fr 1fr;">
            <div class="metric-box"><div class="metric-label">Shadow Accuracy</div><div class="metric-val" style="color:var(--green)">${mon.shadow_directional_accuracy_pct}%</div></div>
            <div class="metric-box"><div class="metric-label">Walk-Forward Ref</div><div class="metric-val">${mon.backtest_reference_accuracy_pct}%</div></div>
            <div class="metric-box"><div class="metric-label">Shadow Brier Score</div><div class="metric-val">${mon.shadow_brier_score}</div></div>
            <div class="metric-box"><div class="metric-label">Settled Windows</div><div class="metric-val">${mon.settled_predictions}</div></div>
          </div>
          <div style="font-size:13px; color:var(--muted);">Kill-Switch Active: <strong>${mon.kill_switch_active}</strong> | Favourable Calls: <strong>${mon.favourable_calls}</strong></div>
        `;

        document.querySelector('#news-table tbody').innerHTML = (news.articles || []).map(a => `
          <tr>
            <td style="white-space:nowrap">${a.published_at || ''}</td>
            <td>${a.title}</td>
            <td style="color:${a.polarity > 0 ? '#22c55e' : (a.polarity < 0 ? '#ef4444' : '#94a3b8')};font-weight:700">${a.polarity > 0 ? '+' : ''}${a.polarity.toFixed(2)}</td>
            <td>${a.impact_level}</td>
            <td>${a.is_event ? '<span class="badge avoid">' + (a.event_type || 'EVENT') + '</span>' : '-'}</td>
          </tr>
        `).join('');

        window._HIST_PREDICTIONS = hist.predictions || [];
        document.querySelector('#history-table tbody').innerHTML = window._HIST_PREDICTIONS.map(p => {
          const priceStr = p.start_price
            ? (p.end_price ? `${Number(p.start_price).toFixed(1)} → ${Number(p.end_price).toFixed(1)}` : `${Number(p.start_price).toFixed(1)} → ...`)
            : '-';
          const retStr = p.realized_return != null
            ? `<span style="color:${p.realized_return >= 0 ? '#22c55e' : '#ef4444'};font-weight:700">${p.realized_return >= 0 ? '+' : ''}${(p.realized_return*100).toFixed(2)}%</span>`
            : '<span style="color:#94a3b8">Waiting</span>';
          const vColor = (p.verdict || '').startsWith('✅') ? '#22c55e' : ((p.verdict || '').startsWith('❌') ? '#ef4444' : '#f59e0b');
          const inspectBtn = p.post_analysis
            ? `<button style="padding:3px 9px;font-size:11.5px;background:${p.post_analysis.is_hit ? '#15803d' : '#b91c1c'}" onclick="showPostAnalysis(${p.prediction_id})">🔍 Inspect</button>`
            : '<span style="color:#64748b;font-size:11.5px;">Pending</span>';
          const hzLabel = p.window_minutes === 375 ? 'Current Day (Close)' : (p.window_minutes === 750 ? 'Next Day (Close)' : `${p.window_minutes}m`);
          return `
            <tr>
              <td>#${p.prediction_id}</td>
              <td style="white-space:nowrap;font-weight:600">${p.window_ist || ''}</td>
              <td style="white-space:nowrap;font-weight:700;color:#38bdf8;">${hzLabel}</td>
              <td><span class="badge ${p.regime}">${regimeLabel(p.regime)}</span></td>
              <td>${p.prob_up != null ? (p.prob_up*100).toFixed(1)+'%' : '-'}</td>
              <td><span class="badge ${p.entry_condition}">${p.entry_condition}</span></td>
              <td style="white-space:nowrap">${priceStr}</td>
              <td>${retStr}</td>
              <td style="color:${vColor};font-weight:700">${p.verdict || ''}</td>
              <td>${inspectBtn}</td>
            </tr>
          `;
        }).join('');

        // Automatically display detailed analysis for the latest settled prediction
        const firstSettled = window._HIST_PREDICTIONS.find(x => x.post_analysis);
        if (firstSettled) {
          showPostAnalysis(firstSettled.prediction_id);
        }
      } catch(err){}
    }

    function showPostAnalysis(predId) {
      const p = (window._HIST_PREDICTIONS || []).find(x => x.prediction_id === predId);
      const panel = document.getElementById('post-analysis-panel');
      if (!p || !p.post_analysis || !panel) return;
      const pa = p.post_analysis;
      panel.style.display = 'block';
      document.getElementById('pa-title').textContent =
        `#${p.prediction_id} (${p.window_ist}) • Predicted ${pa.predicted} vs Actual ${pa.actual}`;
      const badgeEl = document.getElementById('pa-badge');
      badgeEl.className = 'badge ' + (pa.is_hit ? 'bullish' : 'bearish');
      badgeEl.textContent = pa.is_hit ? '✅ CONFIRMED' : '❌ OPPOSITE MOVE';

      document.getElementById('pa-why-pred').textContent = pa.why_predicted;
      document.getElementById('pa-what-happened').textContent = pa.what_happened;
      document.getElementById('pa-why-opp').textContent = pa.why_opposite;

      const oppBox = document.getElementById('pa-opp-box');
      const oppTitle = document.getElementById('pa-opp-title');
      if (pa.is_hit) {
        oppBox.style.borderLeftColor = '#22c55e';
        oppTitle.style.color = '#22c55e';
        oppTitle.textContent = '3. Why the Prediction Succeeded';
      } else {
        oppBox.style.borderLeftColor = '#ef4444';
        oppTitle.style.color = '#f87171';
        oppTitle.textContent = "3. Why Opposite Move Wasn't Picked Earlier";
      }
    }

    let _CHART_TF = 5;
    let _CHART_DATA = null;

    function regimeLabel(r) {
      if (!r) return '--';
      return r === 'sideways' ? 'NO CLEAR VIEW' : r.toUpperCase();
    }

    function setChartTf(tf) {
      _CHART_TF = tf;
      document.getElementById('chart-tf-5').style.background = tf === 5 ? '#0284c7' : '#334155';
      document.getElementById('chart-tf-15').style.background = tf === 15 ? '#0284c7' : '#334155';
      loadLiveChart();
    }

    async function loadLiveChart() {
      try {
        const res = await fetch('/v1/sentiment/nifty/live-chart?window=' + _CHART_TF + '&symbol=' + encodeURIComponent(currentSymbol));
        _CHART_DATA = await res.json();
        drawLiveChart(-1);
      } catch(err){}
    }

    function drawLiveChart(hoverIdx) {
      const canvas = document.getElementById('live-chart-canvas');
      if (!canvas || !_CHART_DATA || !_CHART_DATA.candles || !_CHART_DATA.candles.length) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0) return;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const W = rect.width;
      const H = rect.height;
      const candles = _CHART_DATA.candles;
      const preds = _CHART_DATA.predictions || [];

      document.getElementById('chart-session-date').textContent =
        `Session: ${_CHART_DATA.date || '--'} (${candles.length} candles)`;

      const lastC = candles[candles.length - 1];
      document.getElementById('ind-vwap').textContent = lastC.vwap.toFixed(1);
      const ema9El = document.getElementById('ind-ema9');
      if (ema9El && lastC.ema9 != null) ema9El.textContent = lastC.ema9.toFixed(1);
      document.getElementById('ind-ema20').textContent = lastC.ema20.toFixed(1);
      const ema26El = document.getElementById('ind-ema26');
      if (ema26El && lastC.ema26 != null) ema26El.textContent = lastC.ema26.toFixed(1);
      const rsiEl = document.getElementById('ind-rsi');
      rsiEl.textContent = lastC.rsi14.toFixed(1);
      rsiEl.style.color = lastC.rsi14 >= 65 ? '#ef4444' : (lastC.rsi14 <= 35 ? '#22c55e' : '#f8fafc');
      document.getElementById('ind-adx').textContent = lastC.adx14.toFixed(1);

      const aiTxt = document.getElementById('chart-ai-text');
      if (aiTxt && _CHART_DATA.active_layman) {
        aiTxt.textContent = _CHART_DATA.active_layman;
      }

      // Render Historic + Live Indicator Scorecard
      const scBox = document.getElementById('chart-scorecard');
      if (scBox && _CHART_DATA.scorecard) {
        scBox.innerHTML = _CHART_DATA.scorecard.map(item => {
          const sig = item.signal || 'NEUTRAL';
          const badgeCol = (sig === 'BULLISH' || sig === 'TRENDING') ? '#22c55e' : (sig === 'BEARISH' ? '#ef4444' : '#f59e0b');
          return `
            <div style="background:#0f172a; border:1px solid #334155; border-left:3px solid ${badgeCol}; border-radius:6px; padding:8px 10px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
                <span style="font-size:11px; color:#94a3b8; font-weight:700; text-transform:uppercase;">${item.name}</span>
                <span style="font-size:10px; font-weight:800; color:${badgeCol}; background:rgba(255,255,255,0.05); padding:1px 6px; border-radius:4px;">${sig}</span>
              </div>
              <div style="font-size:12.5px; font-weight:700; color:#f8fafc;">${item.value}</div>
              <div style="font-size:11px; color:#94a3b8; margin-top:2px;">${item.note}</div>
            </div>
          `;
        }).join('');
      }

      // Layout: Top 70% Price + Indicators, Bottom 20% RSI(14), 10% X-Axis
      const padL = 14, padR = 64, padT = 18;
      const priceH = H * 0.70;
      const rsiTop = priceH + 22;
      const rsiH = H * 0.18;
      const plotW = W - padL - padR;

      const zones = _CHART_DATA.zones || null;
      let minP = Infinity, maxP = -Infinity;
      candles.forEach(c => {
        minP = Math.min(minP, c.low, c.vwap, c.ema20, c.ema9 || c.low, c.ema26 || c.low);
        maxP = Math.max(maxP, c.high, c.vwap, c.ema20, c.ema9 || c.high, c.ema26 || c.high);
      });
      if (zones) {
        if (zones.sup_bot) minP = Math.min(minP, zones.sup_bot);
        if (zones.res_top) maxP = Math.max(maxP, zones.res_top);
      }
      const pPad = Math.max(8, (maxP - minP) * 0.12);
      minP -= pPad; maxP += pPad;
      const yOf = p => padT + (maxP - p) / (maxP - minP) * (priceH - padT);
      const stepX = plotW / Math.max(candles.length, 1);
      const xOf = i => padL + (i + 0.5) * stepX;

      ctx.clearRect(0, 0, W, H);

      // Horizontal Price Grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillStyle = '#64748b';
      for (let g = 0; g <= 4; g++) {
        const pv = minP + (maxP - minP) * (g / 4);
        const y = yOf(pv);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillText(pv.toFixed(0), W - padR + 6, y + 4);
      }

      // Vertical Divider for Today's Session Start (separates Historic Session vs Today)
      const tIdx = _CHART_DATA.today_start_idx;
      if (tIdx != null && tIdx > 0 && tIdx < candles.length) {
        const vx = xOf(tIdx) - stepX / 2;
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(vx, padT); ctx.lineTo(vx, rsiTop + rsiH); ctx.stroke();
        ctx.fillStyle = '#94a3b8';
        ctx.font = 'bold 10px system-ui';
        ctx.fillText('HISTORIC', Math.max(padL + 6, vx - 58), padT + 10);
        ctx.fillStyle = '#38bdf8';
        ctx.fillText('TODAY 09:15 →', vx + 6, padT + 10);
        ctx.restore();
      }

      // Subtle PDH / PDL reference lines if within visible chart range
      const drawRef = (lvl, col, lbl) => {
        if (!lvl || lvl < minP || lvl > maxP) return;
        const y = yOf(lvl);
        ctx.save();
        ctx.setLineDash([3, 5]);
        ctx.strokeStyle = col;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillStyle = col;
        ctx.font = '10px system-ui';
        ctx.fillText(lbl + ' ' + lvl.toFixed(0), W - padR + 4, y + 3);
        ctx.restore();
      };
      drawRef(_CHART_DATA.pdh, 'rgba(248,113,113,0.65)', 'PDH');
      drawRef(_CHART_DATA.pdl, 'rgba(74,222,128,0.65)', 'PDL');

      // Shaded Support & Resistance Zones + Pivot Line
      if (zones) {
        ctx.save();
        if (zones.res_top && zones.res_bot) {
          const yRT = yOf(zones.res_top), yRB = yOf(zones.res_bot);
          ctx.fillStyle = 'rgba(239,68,68,0.10)';
          ctx.fillRect(padL, yRT, W - padL - padR, Math.max(3, yRB - yRT));
          ctx.strokeStyle = 'rgba(239,68,68,0.45)';
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(padL, yRT); ctx.lineTo(W - padR, yRT); ctx.stroke();
          ctx.fillStyle = '#f87171';
          ctx.font = 'bold 10px system-ui';
          ctx.fillText(`RES ZONE ${zones.res_bot.toFixed(0)}–${zones.res_top.toFixed(0)}`, padL + 6, yRT + 11);
        }
        if (zones.sup_top && zones.sup_bot) {
          const yST = yOf(zones.sup_top), ySB = yOf(zones.sup_bot);
          ctx.fillStyle = 'rgba(34,197,94,0.10)';
          ctx.fillRect(padL, yST, W - padL - padR, Math.max(3, ySB - yST));
          ctx.strokeStyle = 'rgba(34,197,94,0.45)';
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(padL, ySB); ctx.lineTo(W - padR, ySB); ctx.stroke();
          ctx.fillStyle = '#4ade80';
          ctx.font = 'bold 10px system-ui';
          ctx.fillText(`SUP ZONE ${zones.sup_bot.toFixed(0)}–${zones.sup_top.toFixed(0)}`, padL + 6, ySB - 4);
        }
        if (zones.pivot && zones.pivot >= minP && zones.pivot <= maxP) {
          const yP = yOf(zones.pivot);
          ctx.strokeStyle = 'rgba(148,163,184,0.55)';
          ctx.setLineDash([2, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(padL, yP); ctx.lineTo(W - padR, yP); ctx.stroke();
          ctx.fillStyle = '#94a3b8';
          ctx.font = '10px system-ui';
          ctx.fillText(`PIVOT ${zones.pivot.toFixed(0)}`, W - padR - 68, yP - 3);
        }
        ctx.restore();
      }

      // Candlesticks (Historic dimmed, Today bright)
      const cw = Math.max(2, Math.min(12, stepX * 0.62));
      candles.forEach((c, i) => {
        const x = xOf(i);
        const isUp = c.close >= c.open;
        const col = isUp
          ? (c.is_today ? '#22c55e' : 'rgba(34,197,94,0.45)')
          : (c.is_today ? '#ef4444' : 'rgba(239,68,68,0.45)');
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(x, yOf(c.high)); ctx.lineTo(x, yOf(c.low)); ctx.stroke();
        const yO = yOf(c.open), yC = yOf(c.close);
        ctx.fillStyle = col;
        ctx.fillRect(x - cw / 2, Math.min(yO, yC), cw, Math.max(1.5, Math.abs(yC - yO)));
      });

      // VWAP, EMA9, EMA20, EMA26 Lines
      const drawLine = (key, col, width) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = width;
        ctx.beginPath();
        let started = false;
        candles.forEach((c, i) => {
          if (c[key] == null) return;
          const x = xOf(i), y = yOf(c[key]);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
      };
      drawLine('vwap', '#f59e0b', 1.8);
      drawLine('ema9', '#4ade80', 1.3);
      drawLine('ema20', '#38bdf8', 1.6);
      drawLine('ema26', '#f472b6', 1.3);

      // AI Prediction Markers on Chart
      preds.forEach(pr => {
        const idx = Math.min(candles.length - 1, Math.max(0, pr.candle_idx));
        const c = candles[idx];
        const x = xOf(idx);
        const isBull = pr.regime === 'bullish';
        ctx.font = 'bold 12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        if (pr.regime === 'sideways') {
          // Abstained window: grey hollow marker above the candle, no HIT/MISS
          ctx.fillStyle = '#94a3b8';
          ctx.fillText('○', x, yOf(c.high) - 6);
          return;
        }
        ctx.fillStyle = isBull ? '#22c55e' : '#ef4444';
        if (isBull) {
          const y = yOf(c.low) + 15;
          ctx.fillText('▲', x, y);
          ctx.font = 'bold 9px system-ui, sans-serif';
          ctx.fillText(pr.verdict === 'HIT' ? '✓' : (pr.verdict === 'MISS' ? '✗' : '•'), x, y + 10);
        } else {
          const y = yOf(c.high) - 6;
          ctx.fillText('▼', x, y);
          ctx.font = 'bold 9px system-ui, sans-serif';
          ctx.fillText(pr.verdict === 'HIT' ? '✓' : (pr.verdict === 'MISS' ? '✗' : '•'), x, y - 11);
        }
      });
      ctx.textAlign = 'left';

      // RSI(14) Sub-Panel
      ctx.strokeStyle = '#334155';
      ctx.beginPath(); ctx.moveTo(padL, rsiTop); ctx.lineTo(W - padR, rsiTop); ctx.stroke();
      const yRsi = v => rsiTop + (100 - Math.min(100, Math.max(0, v))) / 100 * rsiH;
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(239,68,68,0.4)';
      ctx.beginPath(); ctx.moveTo(padL, yRsi(70)); ctx.lineTo(W - padR, yRsi(70)); ctx.stroke();
      ctx.strokeStyle = 'rgba(34,197,94,0.4)';
      ctx.beginPath(); ctx.moveTo(padL, yRsi(30)); ctx.lineTo(W - padR, yRsi(30)); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = '#64748b'; ctx.font = '10px system-ui';
      ctx.fillText('RSI 70', W - padR + 6, yRsi(70) + 3);
      ctx.fillText('RSI 30', W - padR + 6, yRsi(30) + 3);

      ctx.strokeStyle = '#a855f7'; ctx.lineWidth = 1.4;
      ctx.beginPath();
      candles.forEach((c, i) => {
        const x = xOf(i), y = yRsi(c.rsi14);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // X-Axis Time Labels
      ctx.fillStyle = '#94a3b8'; ctx.font = '10.5px system-ui'; ctx.textAlign = 'center';
      const labelStep = Math.max(1, Math.floor(candles.length / 8));
      for (let i = 0; i < candles.length; i += labelStep) {
        ctx.fillText(candles[i].ts_ist, xOf(i), H - 4);
      }
      ctx.textAlign = 'left';

      // Hover Crosshair & Tooltip
      const tip = document.getElementById('chart-tooltip');
      if (hoverIdx >= 0 && hoverIdx < candles.length) {
        const c = candles[hoverIdx];
        const hx = xOf(hoverIdx);
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = '#94a3b8';
        ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, rsiTop + rsiH); ctx.stroke();
        ctx.restore();

        const matchPreds = preds.filter(p => p.candle_idx === hoverIdx);
        let predHtml = '';
        if (matchPreds.length) {
          const mp = matchPreds[matchPreds.length - 1];
          const col = mp.regime === 'bullish' ? '#22c55e' : (mp.regime === 'sideways' ? '#f59e0b' : '#ef4444');
          predHtml = `<div style="margin-top:6px;padding-top:6px;border-top:1px solid #334155;">
            <strong style="color:${col};">AI Call #${mp.prediction_id}: ${regimeLabel(mp.regime)} (${mp.prob_up}%) • ${mp.verdict}</strong>
            <div style="color:#cbd5e1;margin-top:3px;">${mp.post_analysis ? mp.post_analysis.why_predicted : (mp.rationale || '')}</div>
            ${mp.post_analysis ? `<div style="color:#fbbf24;margin-top:3px;"><strong>Post-Window:</strong> ${mp.post_analysis.what_happened}</div>` : ''}
          </div>`;
        }
        tip.style.display = 'block';
        tip.innerHTML = `
          <div style="font-weight:700;color:#38bdf8;margin-bottom:4px;">${c.ts_ist} IST (${_CHART_TF}m)</div>
          <div>O: <strong>${c.open}</strong> | H: <strong>${c.high}</strong> | L: <strong>${c.low}</strong> | C: <strong>${c.close}</strong></div>
          <div style="margin-top:2px;color:#94a3b8;">VWAP: <span style="color:#f59e0b">${c.vwap}</span> | EMA9: <span style="color:#4ade80">${c.ema9 ?? '--'}</span> | EMA20: <span style="color:#38bdf8">${c.ema20}</span> | EMA26: <span style="color:#f472b6">${c.ema26 ?? '--'}</span></div>
          <div style="margin-top:2px;color:#94a3b8;">RSI(14): ${c.rsi14} | ADX(14): ${c.adx14}</div>
          ${predHtml}
        `;
      } else if (tip) {
        tip.style.display = 'none';
      }
    }

    (function bindChartEvents(){
      const canvas = document.getElementById('live-chart-canvas');
      if (!canvas) return;
      canvas.addEventListener('mousemove', function(e) {
        if (!_CHART_DATA || !_CHART_DATA.candles || !_CHART_DATA.candles.length) return;
        const rect = canvas.getBoundingClientRect();
        const padL = 14, padR = 64;
        const plotW = rect.width - padL - padR;
        const relX = e.clientX - rect.left - padL;
        const idx = Math.floor(relX / (plotW / _CHART_DATA.candles.length));
        if (idx >= 0 && idx < _CHART_DATA.candles.length) drawLiveChart(idx);
        else drawLiveChart(-1);
      });
      canvas.addEventListener('mouseleave', function() { drawLiveChart(-1); });
    })();

    async function loadAll() {
      await pollLiveTick();
      await loadTables();
      await loadLiveChart();
    }

    async function runPrediction(win) {
      document.getElementById('rendered-text').textContent = `Running ${win}m model inference for ${currentSymbol}...`;
      const res = await fetch('/v1/sentiment/nifty/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({run_type: 'scheduled', window_minutes: win, symbol: currentSymbol})
      }).then(r => r.json());
      updateLiveDisplay(res);
      await loadTables();
    }

    async function toggleKill(state) {
      await fetch('/v1/sentiment/nifty/killswitch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({suppressed: state})
      });
      await loadAll();
    }

    let btDatesLoaded = false;
    async function loadBacktest(selectedDate = '', win = '') {
      const w = win || (document.getElementById('bt-win-select') ? document.getElementById('bt-win-select').value : '5');
      const wLabel = w === '375' ? 'Current Day (Close 15:30)' : (w === '750' ? 'Next Day (Close 15:30)' : `${w}m`);
      const url = `/v1/sentiment/nifty/backtest-5m?window=${encodeURIComponent(w)}${selectedDate ? '&date=' + encodeURIComponent(selectedDate) : ''}`;
      const data = await fetch(url).then(r => r.json());
      document.getElementById('bt-summary-banner').innerHTML =
        `Horizon: <strong>${wLabel}</strong> | Total Tested: <strong>${data.total_candles}</strong> predictions across <strong>${data.daily_summary.length}</strong> trading days | Overall Accuracy: <strong style="color:var(--green)">${data.overall_accuracy_pct}%</strong> | Showing Date: <strong style="color:var(--accent)">${data.selected_date}</strong> (${data.candles.length} predictions)`;

      const sel = document.getElementById('bt-date-select');
      if (!btDatesLoaded && data.daily_summary) {
        sel.innerHTML = data.daily_summary.map(d =>
          `<option value="${d.Trade_Date}" ${d.Trade_Date === data.selected_date ? 'selected' : ''}>${d.Trade_Date} (${d.Accuracy_Pct}% acc | ${d.Hits}/${d.Candles_5m})</option>`
        ).join('');
        btDatesLoaded = true;
      } else if (selectedDate) {
        sel.value = selectedDate;
      }

      document.querySelector('#bt-daily-table tbody').innerHTML = (data.daily_summary || []).map(d => `
        <tr style="${d.Trade_Date === data.selected_date ? 'background:rgba(56,189,248,0.12)' : ''}">
          <td style="font-weight:700">${d.Trade_Date}</td>
          <td>${d.Candles_5m}</td>
          <td style="color:#22c55e;font-weight:700">${d.Hits}</td>
          <td style="color:#ef4444">${d.Misses}</td>
          <td style="color:${d.Accuracy_Pct >= 50 ? '#22c55e' : '#f59e0b'};font-weight:700">${d.Accuracy_Pct}%</td>
          <td>${Number(d.Day_Open).toFixed(2)}</td>
          <td>${Number(d.Day_Close).toFixed(2)}</td>
          <td><button style="padding:4px 10px;font-size:12px" onclick="loadBacktest('${d.Trade_Date}', '${w}')">Inspect ${d.Candles_5m} Predictions</button></td>
        </tr>
      `).join('');

      document.getElementById('bt-candles-title').textContent =
        `${wLabel} Predictions for ${data.selected_date} (${data.candles.length} Predictions)`;

      document.querySelector('#bt-candles-table tbody').innerHTML = (data.candles || []).map(c => `
        <tr>
          <td style="white-space:nowrap;font-weight:600">${c.Candle_Start_IST}</td>
          <td>${Number(c.Nifty_Open).toFixed(2)}</td>
          <td>${Number(c.Nifty_Close).toFixed(2)}</td>
          <td>${Number(c.Next_5m_Close).toFixed(2)}</td>
          <td>${Number(c.India_VIX).toFixed(2)}</td>
          <td>${Number(c.ADX_14).toFixed(1)}</td>
          <td>${Number(c.Breadth_Ratio).toFixed(2)}</td>
          <td>${Number(c.Prob_Up_Pct).toFixed(2)}%</td>
          <td><span class="badge ${c.Predicted_Dir === 'UP' ? 'bullish' : 'bearish'}">${c.Predicted_Dir}</span></td>
          <td><span class="badge ${c.Actual_Dir === 'UP' ? 'bullish' : 'bearish'}">${c.Actual_Dir}</span></td>
          <td style="color:${c.Realized_Return_Pct >= 0 ? '#22c55e' : '#ef4444'};font-weight:700">${c.Realized_Return_Pct != null ? (c.Realized_Return_Pct >= 0 ? '+' : '') + Number(c.Realized_Return_Pct).toFixed(3) + '%' : '-'}</td>
          <td style="color:${c.Verdict === 'HIT' ? '#22c55e' : '#ef4444'};font-weight:700">${c.Verdict === 'HIT' ? '✅ HIT' : '❌ MISS'}</td>
        </tr>
      `).join('');
    }

    // Connect to Server-Sent Events (SSE) stream for instant real-time live needle updates
    function connectSSE() {
      try {
        const evtSrc = new EventSource('/v1/sentiment/nifty/stream');
        evtSrc.onmessage = function(e) {
          try { updateLiveDisplay(JSON.parse(e.data)); } catch(err){}
        };
        evtSrc.addEventListener('tick', function(e) {
          try { updateLiveDisplay(JSON.parse(e.data)); } catch(err){}
        });
        evtSrc.addEventListener('sentiment_update', function(e) {
          try { updateLiveDisplay(JSON.parse(e.data)); } catch(err){}
        });
        evtSrc.onerror = function() {
          evtSrc.close();
          setTimeout(connectSSE, 3000);
        };
      } catch(err){}
    }
    connectSSE();

    // Initial load
    loadAll();
    loadBacktest();

    // Real-time ticking polling fallback (guarantees needle & price updates every 2s)
    setInterval(pollLiveTick, 2000);

    // Slower interval for monitoring, history, chart and news tables (every 10s)
    setInterval(loadTables, 10000);
    setInterval(loadLiveChart, 10000);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Phase 7: Versioned REST + SSE HTTP Server
# ---------------------------------------------------------------------------
class SentimentAPIHandler(BaseHTTPRequestHandler):
    dsn: str = ""

    def _send_json(self, status_code: int, payload: dict | list) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path in ("", "/"):
            html = DASHBOARD_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if path == "/v1/sentiment/nifty/backtest-5m/excel":
            xlsx_path = ROOT / "backtest_5m_15days.xlsx"
            data = xlsx_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", 'attachment; filename="backtest_5m_15days.xlsx"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/v1/sentiment/nifty/backtest-5m/csv":
            win_q = qs.get("window", ["5"])[0]
            csv_path = ROOT / f"backtest_{win_q}m_candles.csv"
            if not csv_path.exists():
                csv_path = ROOT / "backtest_5m_15days_all_candles.csv"
            data = csv_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{csv_path.name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/v1/sentiment/nifty/backtest-5m":
            win_q = qs.get("window", ["5"])[0]
            csv_path = ROOT / f"backtest_{win_q}m_candles.csv"
            if not csv_path.exists():
                csv_path = ROOT / "backtest_5m_15days_all_candles.csv"
            df = pd.read_csv(csv_path)
            daily = (
                df.groupby("Trade_Date", as_index=False)
                .agg(
                    Candles_5m=("Verdict", "count"),
                    Hits=("Verdict", lambda s: int((s == "HIT").sum())),
                    Misses=("Verdict", lambda s: int((s == "MISS").sum())),
                    Accuracy_Pct=("Verdict", lambda s: round(100.0 * float((s == "HIT").mean()), 2)),
                    Day_Open=("Nifty_Open", "first"),
                    Day_Close=("Nifty_Close", "last"),
                )
                .sort_values("Trade_Date", ascending=False)
            )
            sel_date = qs.get("date", [str(daily["Trade_Date"].iloc[0])])[0]
            sub_candles = df[df["Trade_Date"].astype(str) == sel_date]
            self._send_json(
                HTTPStatus.OK,
                {
                    "total_candles": int(len(df)),
                    "overall_accuracy_pct": round(100.0 * float((df["Verdict"] == "HIT").mean()), 2),
                    "selected_date": sel_date,
                    "daily_summary": daily.to_dict("records"),
                    "candles": sub_candles.to_dict("records"),
                },
            )
            return

        if path in ("/health", "/v1/health"):
            self._send_json(HTTPStatus.OK, {"status": "ok", "kill_switch": KILL_SWITCH_ACTIVE})
            return

        if path == "/v1/sentiment/nifty/news":
            limit = min(50, int(qs.get("limit", ["15"])[0]))
            with psycopg.connect(self.dsn, autocommit=True) as conn:
                rows = conn.execute(
                    """
                    SELECT a.article_id, a.published_at, a.title, a.specific_title,
                           s.sentiment_score, s.macro_theme, s.nifty_relevance, s.is_event, s.event_type, a.symbol
                    FROM articles a
                    JOIN article_scores s ON s.article_id = a.article_id
                    ORDER BY a.published_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                ).fetchall()
            items = [
                {
                    "article_id": r[0],
                    "published_at": r[1].astimezone(IST).strftime("%Y-%m-%d %H:%M IST") if r[1] else None,
                    "title": r[3] or r[2],
                    "polarity": float(r[4] or 0.0),
                    "impact_level": r[5] or "general",
                    "nifty_relevance": float(r[6] or 0.0),
                    "is_event": bool(r[7]),
                    "event_type": r[8],
                    "entities": r[9],
                }
                for r in rows
            ]
            self._send_json(HTTPStatus.OK, {"count": len(items), "articles": items})
            return

        if path == "/v1/sentiment/nifty/symbols":
            self._send_json(HTTPStatus.OK, list(SUPPORTED_SYMBOLS.values()))
            return

        if path == "/v1/sentiment/nifty/latest":
            sym = qs.get("symbol", ["NIFTY50"])[0].upper()
            payload = fetch_latest_prediction_payload(self.dsn, symbol=sym)
            self._send_json(HTTPStatus.OK, payload)
            return

        if path == "/v1/sentiment/nifty/monitoring":
            sym = qs.get("symbol", ["NIFTY50"])[0].upper()
            summary = get_shadow_monitoring_summary(self.dsn, symbol=sym)
            self._send_json(HTTPStatus.OK, summary)
            return

        if path == "/v1/sentiment/nifty/live-chart":
            win_q = int(qs.get("window", ["5"])[0])
            sym = qs.get("symbol", ["NIFTY50"])[0].upper()
            chart_payload = fetch_live_chart_payload(self.dsn, window_minutes=win_q, symbol=sym)
            self._send_json(HTTPStatus.OK, chart_payload)
            return

        if path == "/v1/sentiment/nifty/history":
            sym = qs.get("symbol", ["NIFTY50"])[0].upper()
            settle_matured_predictions(self.dsn)
            limit = min(200, int(qs.get("limit", ["50"])[0]))
            filter_date = qs.get("date", [""])[0].strip()
            filter_win = qs.get("window", [""])[0].strip()

            where_clauses = ["instrument = %s"]
            params = [sym]
            if filter_date:
                where_clauses.append("(as_of AT TIME ZONE 'Asia/Kolkata')::date = %s")
                params.append(filter_date)
            if filter_win:
                where_clauses.append("window_minutes = %s")
                params.append(int(filter_win))

            where_sql = "WHERE " + " AND ".join(where_clauses)
            params.append(limit)

            now_utc = datetime.now(timezone.utc)
            items = []
            available_dates = []
            available_windows = []

            with psycopg.connect(self.dsn, autocommit=True) as conn:
                # Fetch available dates and window options for client filters
                d_rows = conn.execute(
                    "SELECT DISTINCT (as_of AT TIME ZONE 'Asia/Kolkata')::date FROM predictions WHERE instrument = %s ORDER BY 1 DESC LIMIT 30",
                    (sym,)
                ).fetchall()
                available_dates = [str(r[0]) for r in d_rows if r[0]]

                w_rows = conn.execute(
                    "SELECT DISTINCT window_minutes FROM predictions WHERE instrument = %s ORDER BY 1 ASC",
                    (sym,)
                ).fetchall()
                available_windows = [int(r[0]) for r in w_rows if r[0] is not None]

                rows = conn.execute(
                    f"""
                    SELECT prediction_id, run_type, as_of,
                           COALESCE(horizon_end, as_of + (window_minutes || ' minutes')::interval) AS h_end,
                           window_minutes, regime, prob_up, entry_condition, confidence,
                           status, realized_return, realized_regime, data_freshness, features_used
                    FROM predictions
                    {where_sql}
                    ORDER BY prediction_id DESC
                    LIMIT %s
                    """,
                    params,
                ).fetchall()
                for r in rows:
                    w_m = int(r[4] or 30)
                    time_fmt = "%H:%M:%S" if w_m <= 5 else "%H:%M"
                    as_of_ist = r[2].astimezone(IST).strftime(time_fmt) if r[2] else ""
                    date_ist = r[2].astimezone(IST).strftime("%Y-%m-%d") if r[2] else ""
                    end_ist = r[3].astimezone(IST).strftime(f"{time_fmt} IST") if r[3] else ""
                    fresh = r[12] or {}
                    p_start = fresh.get("settled_start_price") or fresh.get("nifty_close")
                    p_end = fresh.get("settled_end_price")
                    ret = float(r[10]) if r[10] is not None else None
                    real_reg = r[11]
                    pred_reg = r[5]
                    prob_up = float(r[6]) if r[6] is not None else 0.5
                    post_an = None

                    if ret is not None and pred_reg == "sideways":
                        # Abstained window: no directional call, so it is neither a HIT nor a MISS
                        act_trend = "BULLISH" if ret >= 0 else "BEARISH"
                        verdict = f"⚪ NO CALL (Actual: {act_trend})"
                    elif ret is not None:
                        is_hit = (prob_up >= 0.5 and ret >= 0) or (prob_up < 0.5 and ret < 0)
                        act_trend = "BULLISH" if ret >= 0 else "BEARISH"
                        if is_hit:
                            verdict = f"✅ HIT ({act_trend})"
                        else:
                            verdict = f"❌ MISS (Actual: {act_trend})"
                        post_an = build_post_prediction_analysis(
                            conn, r[2], r[3], pred_reg, prob_up, p_start, p_end, ret, r[13], symbol=sym
                        )
                    else:
                        secs_left = max(1, int(math.ceil((r[3] - now_utc).total_seconds()))) if r[3] else 0
                        if secs_left <= 90:
                            verdict = f"⏳ Settles at {end_ist} ({secs_left}s left)"
                        else:
                            mins_left = int(math.ceil(secs_left / 60.0))
                            verdict = f"⏳ Settles at {end_ist} ({mins_left}m left)"

                    items.append(
                        {
                            "prediction_id": r[0],
                            "run_type": r[1],
                            "as_of": r[2].isoformat(),
                            "window_ist": f"{date_ist} {as_of_ist} → {end_ist}",
                            "window_minutes": r[4],
                            "regime": pred_reg,
                            "prob_up": prob_up,
                            "entry_condition": r[7],
                            "confidence": float(r[8]) if r[8] is not None else None,
                            "status": r[9],
                            "start_price": p_start,
                            "end_price": p_end,
                            "realized_return": ret,
                            "realized_regime": real_reg,
                            "verdict": verdict,
                            "post_analysis": post_an,
                        }
                    )
            self._send_json(
                HTTPStatus.OK,
                {
                    "count": len(items),
                    "predictions": items,
                    "available_dates": available_dates,
                    "available_windows": available_windows,
                    "selected_date": filter_date,
                    "selected_window": filter_win,
                },
            )
            return

        if path == "/v1/sentiment/nifty/stream":
            # Server-Sent Events (SSE) push channel
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Push initial payload immediately
            try:
                payload = fetch_latest_prediction_payload(self.dsn)
                event_data = f"event: sentiment_update\ndata: {json.dumps(payload)}\n\n"
                self.wfile.write(event_data.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                return

            client_q = queue.Queue(maxsize=50)
            with _SSE_LOCK:
                _SSE_CLIENTS.append(client_q)

            try:
                while True:
                    try:
                        msg = client_q.get(timeout=2.0)
                        self.wfile.write(msg)
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (ConnectionResetError, BrokenPipeError, Exception):
                pass
            finally:
                with _SSE_LOCK:
                    if client_q in _SSE_CLIENTS:
                        _SSE_CLIENTS.remove(client_q)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:
        global KILL_SWITCH_ACTIVE
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(raw_body or "{}")
        except Exception:
            import re
            try:
                cleaned = re.sub(r'([a-zA-Z0-9_]+)\s*:', r'"\1":', raw_body)
                body = json.loads(cleaned)
            except Exception:
                body = {}

        if path == "/v1/sentiment/nifty/killswitch":
            KILL_SWITCH_ACTIVE = bool(body.get("suppressed", True))
            self._send_json(
                HTTPStatus.OK,
                {"kill_switch_active": KILL_SWITCH_ACTIVE, "status": "suppressed" if KILL_SWITCH_ACTIVE else "ok"},
            )
            return

        if path == "/v1/sentiment/nifty/generate":
            global _ACTIVE_WINDOW, _LAST_PRED_RUN_TS
            run_type = body.get("run_type", "scheduled")
            if run_type not in ("pre_open", "scheduled", "event", "ad_hoc"):
                run_type = "scheduled"
            window = int(body.get("window_minutes", 15))
            sym = str(body.get("symbol", "NIFTY50")).upper()
            _ACTIVE_WINDOW = window
            _LAST_PRED_RUN_TS = time.time()
            persist = is_market_open(symbol=sym)
            out = run_agent(self.dsn, run_type=run_type, window_minutes=window, symbol=sym, persist=persist)
            broadcast_sse("sentiment_update", out)
            self._send_json(HTTPStatus.OK, out)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown POST endpoint: {path}"})

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 Shadow Run & Phase 7 Sentiment API Service.")
    parser.add_argument("--backfill-shadow", action="store_true", help="Backfill & settle shadow predictions")
    parser.add_argument("--windows", type=int, default=40, help="Number of 30m windows for shadow backfill")
    parser.add_argument("--serve", action="store_true", help="Start REST + SSE HTTP service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    args = parser.parse_args()

    dsn = get_dsn()
    if args.backfill_shadow:
        summary = backfill_and_settle_shadow_run(dsn, max_windows=args.windows)
        print(json.dumps(summary, indent=2))

    if args.serve:
        start_background_ticker(dsn)
        SentimentAPIHandler.dsn = dsn
        server = ThreadingHTTPServer((args.host, args.port), SentimentAPIHandler)
        print(f"Sentiment REST + SSE Service listening on http://127.0.0.1:{args.port}")
        server.serve_forever()

