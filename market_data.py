"""Market data ingestion with a pluggable vendor adapter interface.

Supports:
  - Daily OHLCV bars (`daily_bars`) with point-in-time `available_at`
  - Intraday 1m/5m/15m candles (`intraday_bars`) with point-in-time `available_at`
  - Adapter pattern: `YFinanceAdapter` included out of the box; `TrueDataAdapter` and
    `GDFLAdapter` stubs ready for paid vendor credentials.

Usage:
    python market_data.py --mode daily --days 30
    python market_data.py --mode intraday --days 5 --interval 1m
"""
from abc import ABC, abstractmethod
import argparse
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg
import yfinance as yf

from db import apply_schema, get_dsn


class MarketDataAdapter(ABC):
    """Generic vendor interface for historical and live market data."""

    @abstractmethod
    def fetch_daily(self, vendor_ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Return DataFrame with columns: trade_date, open, high, low, close, adj_close, volume."""

    @abstractmethod
    def fetch_intraday(
        self, vendor_ticker: str, days: int = 5, interval: str = "1m"
    ) -> pd.DataFrame:
        """Return DataFrame with UTC timestamp index (`ts`) and open, high, low, close, volume."""


class YFinanceAdapter(MarketDataAdapter):
    """Yahoo Finance adapter for daily history and recent 1m/5m intraday bars."""

    def fetch_daily(self, vendor_ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        df = yf.download(
            vendor_ticker,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        out = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(df[date_col]).dt.date,
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "adj_close": df.get("Adj Close", df["Close"]).astype(float),
                "volume": df["Volume"].fillna(0).astype("int64"),
            }
        )
        return out.dropna(subset=["close"])

    def fetch_intraday(
        self, vendor_ticker: str, days: int = 5, interval: str = "1m"
    ) -> pd.DataFrame:
        df = yf.download(
            vendor_ticker,
            period=f"{days}d",
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
        ts_series = pd.to_datetime(df[ts_col], utc=True)
        out = pd.DataFrame(
            {
                "ts": ts_series,
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "volume": df["Volume"].fillna(0).astype("int64"),
            }
        )
        return out.dropna(subset=["close"])


def compute_daily_available_at(
    trade_dt: date, tz_name: str, session_close: time | None, publish_lag_sec: int
) -> datetime:
    tz = ZoneInfo(tz_name)
    close_t = session_close or time(15, 30)
    local_dt = datetime.combine(trade_dt, close_t, tzinfo=tz) + timedelta(seconds=publish_lag_sec)
    return local_dt.astimezone(timezone.utc)


def ingest_daily_bars(conn, adapter: MarketDataAdapter, days: int = 30, symbols: list[str] | None = None) -> int:
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=days)
    query = (
        "SELECT symbol, vendor_ticker, timezone, session_close, publish_lag_sec "
        "FROM instruments"
    )
    params: tuple = ()
    if symbols:
        query += " WHERE symbol = ANY(%s)"
        params = (symbols,)
    rows = conn.execute(query, params).fetchall()

    total_upserted = 0
    for symbol, vendor_ticker, tz_name, session_close, lag_sec in rows:
        df = adapter.fetch_daily(vendor_ticker, start_dt, end_dt)
        if df.empty:
            continue
        for rec in df.to_dict("records"):
            avail = compute_daily_available_at(rec["trade_date"], tz_name, session_close, lag_sec)
            conn.execute(
                """
                INSERT INTO daily_bars (
                    symbol, trade_date, open, high, low, close, adj_close, volume, available_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, trade_date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume
                """,
                (
                    symbol,
                    rec["trade_date"],
                    rec["open"],
                    rec["high"],
                    rec["low"],
                    rec["close"],
                    rec["adj_close"],
                    int(rec["volume"]),
                    avail,
                ),
            )
            total_upserted += 1
    return total_upserted


def ingest_intraday_bars(
    conn,
    adapter: MarketDataAdapter,
    days: int = 5,
    interval: str = "1m",
    symbols: list[str] | None = None,
) -> int:
    interval_sec_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600}
    interval_sec = interval_sec_map.get(interval, 60)

    query = "SELECT symbol, vendor_ticker, publish_lag_sec FROM instruments"
    params: tuple = ()
    if symbols:
        query += " WHERE symbol = ANY(%s)"
        params = (symbols,)
    rows = conn.execute(query, params).fetchall()

    total_upserted = 0
    for symbol, vendor_ticker, lag_sec in rows:
        df = adapter.fetch_intraday(vendor_ticker, days=days, interval=interval)
        if df.empty:
            continue
        now_utc = datetime.now(timezone.utc)
        for rec in df.to_dict("records"):
            ts_utc = rec["ts"].to_pydatetime()
            bar_end_avail = ts_utc + timedelta(seconds=interval_sec + lag_sec)
            avail = min(bar_end_avail, now_utc) if ts_utc <= now_utc else bar_end_avail
            conn.execute(
                """
                INSERT INTO intraday_bars (
                    symbol, ts, interval_sec, open, high, low, close, volume, available_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, interval_sec, ts) DO UPDATE SET
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    available_at = EXCLUDED.available_at
                """,
                (
                    symbol,
                    ts_utc,
                    interval_sec,
                    rec["open"],
                    rec["high"],
                    rec["low"],
                    rec["close"],
                    int(rec["volume"]),
                    avail,
                ),
            )
            total_upserted += 1
    return total_upserted


def compute_and_store_breadth(conn, universe: str = "NIFTY50", interval_sec: int = 60) -> int:
    """Compute intraday Advance/Decline/Unchanged counts across Nifty 50 equities per bar timestamp."""
    cur = conn.execute(
        """
        WITH day_opens AS (
            SELECT DISTINCT ON (b.symbol, (b.ts AT TIME ZONE 'Asia/Kolkata')::date)
                b.symbol,
                (b.ts AT TIME ZONE 'Asia/Kolkata')::date AS trade_date,
                b.open AS day_open
            FROM intraday_bars b
            JOIN instruments i ON i.symbol = b.symbol
            WHERE i.asset_class = 'equity' AND b.interval_sec = %s
            ORDER BY b.symbol, (b.ts AT TIME ZONE 'Asia/Kolkata')::date, b.ts ASC
        ),
        agg AS (
            SELECT
                b.ts,
                COUNT(*) FILTER (WHERE b.close > d.day_open) AS advances,
                COUNT(*) FILTER (WHERE b.close < d.day_open) AS declines,
                COUNT(*) FILTER (WHERE b.close = d.day_open) AS unchanged,
                MAX(b.available_at) AS available_at
            FROM intraday_bars b
            JOIN instruments i ON i.symbol = b.symbol
            JOIN day_opens d
              ON d.symbol = b.symbol
             AND d.trade_date = (b.ts AT TIME ZONE 'Asia/Kolkata')::date
            WHERE i.asset_class = 'equity' AND b.interval_sec = %s
            GROUP BY b.ts
        )
        INSERT INTO market_breadth (universe, ts, advances, declines, unchanged, available_at, source)
        SELECT %s, ts, advances, declines, unchanged, available_at, 'computed_constituents'
        FROM agg
        ON CONFLICT (universe, ts) DO UPDATE SET
            advances = EXCLUDED.advances,
            declines = EXCLUDED.declines,
            unchanged = EXCLUDED.unchanged,
            available_at = EXCLUDED.available_at
        """,
        (interval_sec, interval_sec, universe),
    )
    return cur.rowcount


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest daily or intraday market data.")
    parser.add_argument("--mode", choices=["daily", "intraday"], default="daily")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols (e.g. NIFTY50 INDIAVIX)")
    args = parser.parse_args()

    adapter = YFinanceAdapter()
    with psycopg.connect(get_dsn()) as conn:
        apply_schema(conn)
        if args.mode == "daily":
            count = ingest_daily_bars(conn, adapter, days=args.days, symbols=args.symbols)
            print(f"Upserted {count} daily bars.")
        else:
            count = ingest_intraday_bars(
                conn, adapter, days=args.days, interval=args.interval, symbols=args.symbols
            )
            b_count = compute_and_store_breadth(conn)
            print(f"Ingested {count} intraday ({args.interval}) bars and updated {b_count} breadth snapshots.")

