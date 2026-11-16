"""Phase 1 Data Pipeline Worker: Historical Backfill & Scheduled Live Ingestion.

Modes:
  --backfill : Backfill all 50 Nifty 50 symbols' news history from news.lisuns.com,
               2 years of daily market bars, 5 days of 1-minute intraday bars, and
               compute Nifty 50 intraday market breadth.
  --once     : Run a single incremental ingestion pass (latest news + intraday bars + breadth)
               with retry backoff and latency monitoring.
  --daemon   : Run continuously (every 3 minutes during NSE market hours 09:00-15:35 IST,
               hourly outside market hours).

Usage:
    python pipeline_worker.py --backfill
    python pipeline_worker.py --once
    python pipeline_worker.py --daemon
"""
import argparse
from datetime import datetime, time as dtime, timezone
import logging
import os
import time
from zoneinfo import ZoneInfo

import psycopg
import requests

from audit_news_api import fetch_symbol_history
from db import apply_schema, get_dsn
from ingest import ingest as ingest_articles, parse_ts
from scorer import score_all_articles
from market_data import (
    YFinanceAdapter,
    compute_and_store_breadth,
    ingest_daily_bars,
    ingest_intraday_bars,
)

IST = ZoneInfo("Asia/Kolkata")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline")


def get_nifty50_symbols(conn) -> list[str]:
    rows = conn.execute(
        "SELECT symbol FROM instruments WHERE asset_class = 'equity' ORDER BY index_weight DESC NULLS LAST"
    ).fetchall()
    return [r[0] for r in rows]


def get_core_intraday_symbols(conn) -> list[str]:
    rows = conn.execute(
        """
        SELECT symbol FROM instruments
        WHERE symbol IN ('NIFTY50', 'BANKNIFTY', 'INDIAVIX')
           OR is_heavyweight = TRUE
        ORDER BY symbol
        """
    ).fetchall()
    return [r[0] for r in rows]


def fetch_with_retries(url: str, headers: dict, params: dict, retries: int = 3) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            if resp.status_code == 429:
                wait = float(resp.headers.get("X-Ratelimit-Reset", 1.5))
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except requests.RequestException as exc:
            if attempt == retries:
                log.warning("API request failed after %d attempts (%s): %s", retries, params, exc)
                return []
            time.sleep(1.5 * attempt)
    return []


def backfill_all_news(dsn: str, max_pages_per_symbol: int = 10) -> tuple[int, int]:
    base_url = os.environ["NEWS_API_URL"].rstrip("/")
    api_key = os.environ["NEWS_API_KEY"]
    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        symbols = get_nifty50_symbols(conn)

    log.info("Starting news backfill across %d Nifty 50 symbols...", len(symbols))
    all_articles: list[dict] = []
    for idx, sym in enumerate(symbols, 1):
        try:
            api_syms = ["TMPV", "TMCV"] if sym == "TATAMOTORS" else [sym]
            for s in api_syms:
                items = fetch_symbol_history(base_url, api_key, s, max_pages=max_pages_per_symbol)
                all_articles.extend(items)
            if idx % 10 == 0 or idx == len(symbols):
                log.info("Fetched %d/%d symbols (cumulative raw rows: %d)", idx, len(symbols), len(all_articles))
        except Exception as exc:
            log.warning("Skipping symbol %s on error: %s", sym, exc)

    # Also fetch macro / central bank stories (RBI)
    rbi_items = fetch_with_retries(
        f"{base_url}/api/news",
        headers={"x-api-key": api_key},
        params={"company": "RBI", "limit": 100},
    )
    all_articles.extend(rbi_items)

    added = ingest_articles(all_articles, dsn)
    log.info("News backfill complete: %d new articles inserted (%d total fetched).", added, len(all_articles))
    return added, len(all_articles)


def run_incremental_pass(dsn: str) -> dict:
    """Poll latest news across Nifty 50 constituents + latest 1m bars + breadth, with latency stats."""
    t0 = time.perf_counter()
    base_url = os.environ["NEWS_API_URL"].rstrip("/")
    api_key = os.environ["NEWS_API_KEY"]
    headers = {"x-api-key": api_key}

    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        symbols = get_nifty50_symbols(conn)
        intraday_syms = get_core_intraday_symbols(conn)

    latest_batch: list[dict] = []
    for sym in symbols:
        for s in (["TMPV", "TMCV"] if sym == "TATAMOTORS" else [sym]):
            items = fetch_with_retries(
                f"{base_url}/api/news",
                headers=headers,
                params={"symbol": s, "limit": 20},
            )
            latest_batch.extend(items)

    new_articles = ingest_articles(latest_batch, dsn) if latest_batch else 0
    if new_articles > 0:
        score_all_articles(dsn)

    # Measure publication-to-ingestion latency on the polled batch
    lags = []
    for a in latest_batch:
        if a.get("date") and a.get("created_at"):
            lags.append((parse_ts(a["created_at"]) - parse_ts(a["date"])).total_seconds())
    median_lag_s = sorted(lags)[len(lags) // 2] if lags else None

    adapter = YFinanceAdapter()
    with psycopg.connect(dsn) as conn:
        bars_added = ingest_intraday_bars(conn, adapter, days=1, interval="1m", symbols=intraday_syms)
        breadth_rows = compute_and_store_breadth(conn)

    elapsed_s = round(time.perf_counter() - t0, 2)
    stats = {
        "polled_articles": len(latest_batch),
        "new_articles": new_articles,
        "median_vendor_lag_sec": round(median_lag_s, 1) if median_lag_s is not None else None,
        "intraday_bars_processed": bars_added,
        "breadth_rows_updated": breadth_rows,
        "worker_pass_elapsed_sec": elapsed_s,
    }
    log.info("Incremental pass finished: %s", stats)
    return stats


def is_market_hours(now_utc: datetime | None = None) -> bool:
    now_ist = (now_utc or datetime.now(timezone.utc)).astimezone(IST)
    if now_ist.weekday() >= 5:
        return False
    return dtime(9, 0) <= now_ist.time() <= dtime(15, 35)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Data Pipeline Worker.")
    parser.add_argument("--backfill", action="store_true", help="Run full Phase 1 news & market backfill")
    parser.add_argument("--once", action="store_true", help="Run one incremental ingestion cycle")
    parser.add_argument("--daemon", action="store_true", help="Run continuous scheduled ingestion loop")
    parser.add_argument("--daily-days", type=int, default=730, help="Days of daily OHLCV history to backfill")
    parser.add_argument("--intraday-days", type=int, default=5, help="Days of 1m intraday bars to backfill")
    args = parser.parse_args()

    dsn = get_dsn()

    if args.backfill:
        backfill_all_news(dsn)
        adapter = YFinanceAdapter()
        with psycopg.connect(dsn) as conn:
            core_syms = get_core_intraday_symbols(conn)
            daily_syms = core_syms + ["SP500", "NASDAQ", "BRENT", "USDINR"]
            log.info("Backfilling %d days of daily bars for %d symbols...", args.daily_days, len(daily_syms))
            d_cnt = ingest_daily_bars(conn, adapter, days=args.daily_days, symbols=daily_syms)
            log.info("Daily bars upserted: %d", d_cnt)

            log.info("Backfilling %d days of 1m intraday bars for %d core symbols...", args.intraday_days, len(core_syms))
            i_cnt = ingest_intraday_bars(conn, adapter, days=args.intraday_days, interval="1m", symbols=core_syms)
            b_cnt = compute_and_store_breadth(conn)
            log.info("Intraday 1m bars ingested: %d | Breadth snapshots computed: %d", i_cnt, b_cnt)

    if args.once:
        run_incremental_pass(dsn)

    if args.daemon:
        log.info("Starting continuous ingestion daemon...")
        while True:
            run_incremental_pass(dsn)
            sleep_sec = 180 if is_market_hours() else 3600
            log.info("Sleeping %ds until next scheduled pass...", sleep_sec)
            time.sleep(sleep_sec)

