"""Phase 0 News API & Dataset Audit Tool (Lisuns News API).

Endpoint Mechanics Discovered:
  - Base URL: https://news.lisuns.com
  - News endpoint: GET /api/news
  - Directory endpoint: GET /api/companies (limit, offset)
  - Auth: Header `x-api-key: <NEWS_API_KEY>` (or query param `api_key`)
  - Rate limit: 40 requests / second (`X-Ratelimit-Limit: 40`, `X-Ratelimit-Reset: 1`)
  - Required filter: Exactly one of `symbol`, `isin`, `company`, or `scrip_code`
  - Optional filters: `from`, `to` (ISO 8601 with timezone offset), `sentiment`, `limit` (max 100), `offset`

Usage:
    python audit_news_api.py --live --ingest
    python audit_news_api.py sample_articles.json
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median, quantiles
import time

import requests

from db import ROOT, get_dsn
from ingest import clean, ingest, parse_ts

DEFAULT_NIFTY_HEAVYWEIGHTS = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "ITC",
    "TCS",
    "LT",
    "BHARTIARTL",
    "AXISBANK",
    "SBIN",
]


def fetch_symbol_history(base_url: str, api_key: str, symbol: str, max_pages: int = 10) -> list[dict]:
    """Paginate through /api/news for a single symbol using limit=100 & offset."""
    url = f"{base_url.rstrip('/')}/api/news"
    headers = {"x-api-key": api_key}
    all_articles: list[dict] = []
    for page in range(max_pages):
        offset = page * 100
        resp = requests.get(
            url,
            headers=headers,
            params={"symbol": symbol, "limit": 100, "offset": offset},
            timeout=20,
        )
        if resp.status_code == 429:
            time.sleep(1.1)
            resp = requests.get(
                url,
                headers=headers,
                params={"symbol": symbol, "limit": 100, "offset": offset},
                timeout=20,
            )
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        all_articles.extend(batch)
        if len(batch) < 100:
            break
    return all_articles


def load_from_api(symbols: list[str] | None = None, max_pages: int = 10) -> tuple[list[dict], dict]:
    base_url = os.environ.get("NEWS_API_URL")
    api_key = os.environ.get("NEWS_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("NEWS_API_URL and NEWS_API_KEY must be set in .env")

    target_symbols = symbols or DEFAULT_NIFTY_HEAVYWEIGHTS
    all_records: list[dict] = []
    per_symbol_counts: dict[str, int] = {}

    # Also inspect /api/companies metadata
    comp_resp = requests.get(
        f"{base_url.rstrip('/')}/api/companies",
        headers={"x-api-key": api_key},
        params={"limit": 1},
        timeout=15,
    )
    companies_total = comp_resp.json().get("total") if comp_resp.status_code == 200 else None
    rate_limit = comp_resp.headers.get("X-Ratelimit-Limit")

    for sym in target_symbols:
        items = fetch_symbol_history(base_url, api_key, sym, max_pages=max_pages)
        per_symbol_counts[sym] = len(items)
        all_records.extend(items)

    api_meta = {
        "endpoint": f"{base_url.rstrip('/')}/api/news",
        "companies_endpoint_total": companies_total,
        "rate_limit_per_sec": int(rate_limit) if rate_limit and rate_limit.isdigit() else rate_limit,
        "per_symbol_article_counts": per_symbol_counts,
    }
    return all_records, api_meta


def audit_articles(articles: list[dict], api_meta: dict | None = None) -> dict:
    total = len(articles)
    ids = [a.get("article_id") for a in articles if a.get("article_id") is not None]
    unique_ids = len(set(ids))
    duplicate_id_count = len(ids) - unique_ids

    # Check shared URLs across entity-specific rows
    urls = [clean(a.get("link")) for a in articles if clean(a.get("link"))]
    unique_urls = len(set(urls))

    fields = [
        "article_id",
        "title",
        "specific_title",
        "summary",
        "long_summary",
        "link",
        "source",
        "article_type",
        "sector",
        "company",
        "symbol",
        "ISIN",
        "scrip_code",
        "sentiment",
        "date",
        "created_at",
        "logo_url",
    ]
    non_null_counts = {f: 0 for f in fields}
    sentiments: Counter = Counter()
    sources: Counter = Counter()
    article_types: Counter = Counter()
    sectors: Counter = Counter()
    pub_dates = []
    lags_sec = []
    negative_lags = 0

    for a in articles:
        for f in fields:
            if clean(a.get(f)) is not None:
                non_null_counts[f] += 1
        sentiments[clean(a.get("sentiment")) or "<NULL>"] += 1
        sources[clean(a.get("source")) or "<NULL>"] += 1
        article_types[clean(a.get("article_type")) or "<NULL>"] += 1
        sectors[clean(a.get("sector")) or "<NULL>"] += 1
        if a.get("date"):
            pub = parse_ts(a["date"])
            pub_dates.append(pub)
            if a.get("created_at"):
                ing = parse_ts(a["created_at"])
                lag = (ing - pub).total_seconds()
                lags_sec.append(lag)
                if lag < 0:
                    negative_lags += 1

    completeness = {
        f: round(100.0 * cnt / total, 1) if total else 0.0
        for f, cnt in non_null_counts.items()
    }
    lag_quantiles = quantiles(lags_sec, n=4) if len(lags_sec) >= 4 else []
    report = {
        "api_metadata": api_meta or {},
        "total_records": total,
        "unique_article_ids": unique_ids,
        "duplicate_article_ids": duplicate_id_count,
        "unique_story_urls": unique_urls,
        "multi_company_shared_url_rows": len(urls) - unique_urls,
        "earliest_published_at": min(pub_dates).isoformat() if pub_dates else None,
        "latest_published_at": max(pub_dates).isoformat() if pub_dates else None,
        "ingest_lag_seconds": {
            "min": round(min(lags_sec), 2) if lags_sec else None,
            "p25": round(lag_quantiles[0], 2) if lag_quantiles else None,
            "median": round(median(lags_sec), 2) if lags_sec else None,
            "p75": round(lag_quantiles[2], 2) if lag_quantiles else None,
            "max": round(max(lags_sec), 2) if lags_sec else None,
            "negative_lag_count": negative_lags,
        },
        "vendor_sentiment_distribution": dict(sentiments),
        "top_sources": dict(sources.most_common(10)),
        "top_article_types": dict(article_types.most_common(10)),
        "top_sectors": dict(sectors.most_common(10)),
        "field_completeness_pct": completeness,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit News API payload or sample JSON.")
    parser.add_argument("path", nargs="?", help="Path to local JSON sample file")
    parser.add_argument("--live", action="store_true", help="Query live NEWS_API_URL from .env")
    parser.add_argument("--symbols", nargs="*", help="Symbols to query in --live mode")
    parser.add_argument("--ingest", action="store_true", help="Also ingest fetched articles into PostgreSQL")
    args = parser.parse_args()

    if args.live:
        records, meta = load_from_api(symbols=args.symbols)
    else:
        target = Path(args.path) if args.path else (ROOT / "sample_articles.json")
        raw = json.loads(target.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else [raw]
        meta = {"source_file": str(target)}

    report = audit_articles(records, meta)
    print(json.dumps(report, indent=2))

    if args.ingest and records:
        added = ingest(records, get_dsn())
        print(f"\nIngested into PostgreSQL: {added} new, {len(records) - added} already present.")
