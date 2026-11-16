"""Phase 1 Data Quality Checker & Report Generator.

Audits the PostgreSQL `sentiment` database across news, calendar, daily bars,
1-minute intraday bars, and market breadth, and writes `docs/phase1_quality_report.md`.

Usage:
    python data_quality.py
"""
import json
import psycopg
from db import ROOT, get_dsn


def run_quality_checks(dsn: str) -> dict:
    with psycopg.connect(dsn) as conn:
        # 1. News summary
        news_summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total_articles,
                COUNT(DISTINCT url) AS unique_urls,
                COUNT(DISTINCT symbol) AS distinct_symbols,
                MIN(published_at) AS min_published_at,
                MAX(published_at) AS max_published_at,
                COUNT(*) FILTER (WHERE title IS NULL OR summary IS NULL) AS missing_text_rows,
                COUNT(*) FILTER (WHERE available_at < published_at) AS lookahead_violations,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (ingested_at - published_at))) AS median_lag_sec,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (ingested_at - published_at))) AS p95_lag_sec,
                COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM (ingested_at - published_at)) > 900) AS late_over_15m_count
            FROM articles
            """
        ).fetchone()

        # 2. Monthly volume
        monthly_counts = conn.execute(
            """
            SELECT TO_CHAR(published_at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM') AS month,
                   COUNT(*) AS articles,
                   COUNT(DISTINCT symbol) AS symbols_covered
            FROM articles
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()

        # 3. Session buckets
        session_counts = conn.execute(
            """
            SELECT session_bucket, COUNT(*)
            FROM articles
            GROUP BY session_bucket
            ORDER BY 2 DESC
            """
        ).fetchall()

        # 4. Daily bars coverage
        daily_cov = conn.execute(
            """
            SELECT symbol, COUNT(*) AS bars, MIN(trade_date), MAX(trade_date)
            FROM daily_bars
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()

        # 5. Intraday 1m bars coverage
        intraday_cov = conn.execute(
            """
            SELECT symbol, COUNT(*) AS bars, MIN(ts), MAX(ts)
            FROM intraday_bars
            WHERE interval_sec = 60
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()

        # 6. Breadth & calendars
        breadth_cnt = conn.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM market_breadth").fetchone()
        holiday_cnt = conn.execute("SELECT COUNT(*) FROM nse_holidays").fetchone()[0]
        expiry_cnt = conn.execute("SELECT COUNT(*), COUNT(*) FILTER (WHERE expiry_kind='monthly') FROM expiry_calendar").fetchone()

    report = {
        "news": {
            "total_articles": news_summary[0],
            "unique_story_urls": news_summary[1],
            "distinct_symbols_covered": news_summary[2],
            "min_published_at": news_summary[3].isoformat() if news_summary[3] else None,
            "max_published_at": news_summary[4].isoformat() if news_summary[4] else None,
            "missing_text_rows": news_summary[5],
            "lookahead_violations": news_summary[6],
            "median_ingest_lag_sec": round(float(news_summary[7] or 0), 2),
            "p95_ingest_lag_sec": round(float(news_summary[8] or 0), 2),
            "late_arriving_over_15m": news_summary[9],
            "monthly_coverage": [
                {"month": m, "articles": c, "symbols": s} for m, c, s in monthly_counts
            ],
            "session_buckets": {b: c for b, c in session_counts},
        },
        "daily_bars": [
            {"symbol": s, "bars": b, "start": str(mn), "end": str(mx)}
            for s, b, mn, mx in daily_cov
        ],
        "intraday_1m_bars": [
            {"symbol": s, "bars": b, "start": mn.isoformat(), "end": mx.isoformat()}
            for s, b, mn, mx in intraday_cov
        ],
        "market_breadth": {
            "snapshots": breadth_cnt[0],
            "start": breadth_cnt[1].isoformat() if breadth_cnt[1] else None,
            "end": breadth_cnt[2].isoformat() if breadth_cnt[2] else None,
        },
        "calendars": {
            "nse_holidays": holiday_cnt,
            "total_expiries_2026": expiry_cnt[0],
            "monthly_expiries_2026": expiry_cnt[1],
        },
    }
    return report


def write_markdown_report(report: dict) -> None:
    n = report["news"]
    lines = [
        "# Phase 1 Data Quality & Backfill Report",
        "",
        "## 1. News Corpus Summary (`articles` & `raw_articles`)",
        f"- **Total Ingested Articles**: `{n['total_articles']:,}`",
        f"- **Unique Story URLs**: `{n['unique_story_urls']:,}`",
        f"- **Distinct Symbols Covered**: `{n['distinct_symbols_covered']}`",
        f"- **Date Range (UTC)**: `{n['min_published_at']}` to `{n['max_published_at']}`",
        f"- **Missing Title/Summary Rows**: `{n['missing_text_rows']}`",
        f"- **Point-in-Time Lookahead Violations (`available_at < published_at`)**: `{n['lookahead_violations']}`",
        f"- **Ingestion Latency (`ingested_at - published_at`)**: Median `{n['median_ingest_lag_sec']}s` | 95th Percentile `{n['p95_ingest_lag_sec']}s` | Late arrivals (>15 min): `{n['late_arriving_over_15m']}`",
        "",
        "### Monthly Coverage Density",
        "| Month (IST) | Articles | Symbols Active |",
        "| :--- | ---: | ---: |",
    ]
    for row in n["monthly_coverage"]:
        lines.append(f"| `{row['month']}` | `{row['articles']:,}` | `{row['symbols']}` |")

    lines += [
        "",
        "### Session Bucket Breakdown (with NSE Holiday Calendar)",
        "| Session Bucket | Article Count |",
        "| :--- | ---: |",
    ]
    for bucket, cnt in n["session_buckets"].items():
        lines.append(f"| `{bucket}` | `{cnt:,}` |")

    lines += [
        "",
        "## 2. Daily Market Data Coverage (`daily_bars`)",
        "| Symbol | Trading Days | Start Date | End Date |",
        "| :--- | ---: | :--- | :--- |",
    ]
    for row in report["daily_bars"]:
        lines.append(f"| `{row['symbol']}` | `{row['bars']:,}` | `{row['start']}` | `{row['end']}` |")

    lines += [
        "",
        "## 3. Intraday 1-Minute Bars (`intraday_bars`) & Breadth (`market_breadth`)",
        f"- **Computed Nifty 50 Breadth Snapshots (`market_breadth`)**: `{report['market_breadth']['snapshots']:,}` (`{report['market_breadth']['start']}` to `{report['market_breadth']['end']}`)",
        f"- **Calendars Loaded**: `{report['calendars']['nse_holidays']}` NSE holidays | `{report['calendars']['total_expiries_2026']}` Nifty F&O expiries (`{report['calendars']['monthly_expiries_2026']}` monthly)",
        "",
        "| Symbol | 1m Bars | Earliest Bar (UTC) | Latest Bar (UTC) |",
        "| :--- | ---: | :--- | :--- |",
    ]
    for row in report["intraday_1m_bars"]:
        lines.append(f"| `{row['symbol']}` | `{row['bars']:,}` | `{row['start']}` | `{row['end']}` |")

    out_path = ROOT / "docs" / "phase1_quality_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    rep = run_quality_checks(get_dsn())
    write_markdown_report(rep)
    print(json.dumps(rep, indent=2))

