"""Ingest news JSON into PostgreSQL: raw table + normalized table.

Usage:
    # DATABASE_URL is read from .env (see db.py) or the environment
    python ingest.py sample.json          # a JSON file containing one article or a list
"""
import json
import os
import re
import sys
from datetime import datetime, time, timezone
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

from db import apply_schema, get_dsn

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)
NULL_STRINGS = {"", "n/a", "na", "null", "none", "-"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(html):
    if not html:
        return None
    parser = _TextExtractor()
    # Tags become spaces so "<li>a</li><li>b</li>" doesn't glue words together.
    parser.feed(re.sub(r"<[^>]+>", lambda m: m.group(0) + " ", html))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip() or None


def clean(value):
    """Turn placeholders like 'N/A' into real NULLs."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in NULL_STRINGS else text


def parse_ts(value):
    ts = datetime.fromisoformat(value)
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def session_bucket(ts_utc, holidays=None):
    """Classify a timestamp against the NSE session in IST (including holidays)."""
    ist = ts_utc.astimezone(IST)
    if ist.weekday() >= 5:
        return "weekend"
    if holidays and ist.date() in holidays:
        return "holiday"
    t = ist.time()
    if t < MARKET_OPEN:
        return "pre_market"
    if t <= MARKET_CLOSE:
        return "in_session"
    return "post_market"


def normalize(a, holidays=None):
    published = parse_ts(a["date"])
    ingested = parse_ts(a["created_at"]) if a.get("created_at") else published
    avail = max(published, ingested)
    return {
        "article_id": int(a["article_id"]),
        "title": clean(a.get("title")) or clean(a.get("specific_title")),
        "specific_title": clean(a.get("specific_title")),
        "summary": clean(a.get("summary")),
        "body_text": html_to_text(a.get("long_summary")),
        "url": clean(a.get("link")),
        "source": clean(a.get("source")),
        "article_type": clean(a.get("article_type")),
        "subcategory": clean(a.get("subcategory")),
        "sector": clean(a.get("sector")),
        "company": clean(a.get("company")),
        "symbol": clean(a.get("symbol")),
        "isin": clean(a.get("ISIN")),
        "scrip_code": clean(a.get("scrip_code")),
        "vendor_sentiment": clean(a.get("sentiment")),
        "published_at": published,
        "ingested_at": ingested,
        # Point-in-time safety: the article can't be known before we had it.
        "available_at": avail,
        "session_bucket": session_bucket(avail, holidays),
    }


INSERT_ARTICLE = """
INSERT INTO articles (
    article_id, title, specific_title, summary, body_text, url, source, article_type, subcategory,
    sector, company, symbol, isin, scrip_code, vendor_sentiment,
    published_at, ingested_at, available_at, session_bucket
) VALUES (
    %(article_id)s, %(title)s, %(specific_title)s, %(summary)s, %(body_text)s, %(url)s, %(source)s,
    %(article_type)s, %(subcategory)s, %(sector)s, %(company)s, %(symbol)s,
    %(isin)s, %(scrip_code)s, %(vendor_sentiment)s,
    %(published_at)s, %(ingested_at)s, %(available_at)s, %(session_bucket)s
) ON CONFLICT (article_id) DO NOTHING
"""


def ingest(articles, dsn):
    import psycopg
    from psycopg.types.json import Jsonb

    new = 0
    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        holidays = {
            r[0] for r in conn.execute("SELECT holiday_date FROM nse_holidays").fetchall()
        }
        for a in articles:
            row = normalize(a, holidays)
            # First-seen wins: later edits to an article never rewrite history.
            cur = conn.execute(
                "INSERT INTO raw_articles (article_id, payload) VALUES (%s, %s) "
                "ON CONFLICT (article_id) DO NOTHING",
                (row["article_id"], Jsonb(a)),
            )
            if cur.rowcount:
                conn.execute(INSERT_ARTICLE, row)
                new += 1
    return new


if __name__ == "__main__":
    data = json.loads(Path(sys.argv[1]).read_text())
    articles = data if isinstance(data, list) else [data]
    added = ingest(articles, get_dsn())
    print(f"{added} new, {len(articles) - added} already present")