# Phase 0 News API Audit Report (`news.lisuns.com`)

**Audited on:** 11 Nov 2026  
**Sample Audited & Ingested:** 1,520 articles across the top 10 Nifty 50 heavyweights (`RELIANCE`, `HDFCBANK`, `ICICIBANK`, `INFY`, `ITC`, `TCS`, `LT`, `BHARTIARTL`, `AXISBANK`, `SBIN`).

---

## 1. Endpoints, Authentication & Rate Limits

| Property | Finding |
| :--- | :--- |
| **Base URL** | `https://news.lisuns.com` |
| **News Endpoint** | `GET /api/news` (requires exactly one of `symbol`, `isin`, `company`, or `scrip_code`) |
| **Company Directory** | `GET /api/companies` (`6,523` Indian companies with `ISIN`, `symbol`, `scrip_code`, `company_name`, `logo_url`) |
| **Authentication** | Header `x-api-key: <NEWS_API_KEY>` (or query parameter `api_key`) |
| **Rate Limit** | **40 requests / second** (`X-Ratelimit-Limit: 40`, `X-Ratelimit-Reset: 1`) |
| **Pagination** | `limit` (default `20`, max `100`) and `offset` (0-indexed integer) |
| **Date Range Queries** | `from` and `to` in ISO 8601 format **with mandatory timezone offset** (e.g., `2026-07-14T00:00:00+05:30`) |
| **Additional Filters** | `sentiment` (`positive`, `neutral`, `negative`) |

---

## 2. Historical Depth & Coverage

- **Earliest Article in API**: `2026-07-14T02:23:55+00:00` (`article_id = 544669`).
- **Latest Article at Audit**: `2026-11-11T05:53:47+00:00` (`article_id = 746772`).
- **Effective Historical Depth**: **~4 months** (mid-July 2026 to present). Queries for 2022–June 2026 return `0` articles.
- **Implication for Phase 1 & Phase 3**:
  - Market technical and regime features (`intraday_bars`, `daily_bars`) can be backfilled for 2–3 years, whereas **Market + News** joint walk-forward validation and ablations must be evaluated over the **July 2026 – present** window (and expanding forward as continuous ingestion runs).
- **Index-Level Queries**:
  - Querying `symbol=NIFTY` returns `404` because `/api/news` is keyed by listed company (`symbol`, `isin`, `scrip_code`).
  - Index-wide sentiment for Nifty 50 must therefore be constructed by **aggregating across all 50 Nifty constituent symbols** (weighted by `index_weight` and `nifty_relevance`), plus macro queries (`company=RBI`, etc.).

---

## 3. Data Structure & Multi-Company Story Splitting

- **Entity-Specific Rows**: When a single news article mentions multiple companies (e.g., *"Q1 Showdown: Analysts pick top bets as ICICI Bank, HDFC Bank, Axis Bank..."*), the API emits **one distinct `article_id` per company**, sharing the same `link`, `title`, and `date`, but with tailored:
  - `specific_title` (e.g., *"ICICI Bank Positioned for Strong Q1 Growth"* vs. *"HDFC Bank: Defensive Choice with Muted Upside"*)
  - `summary` and `long_summary` (HTML formatted with `<h2>`, `<p>`, `<ol><li>`)
  - `sentiment` (`positive` for `ICICIBANK`, `neutral` for `HDFCBANK` on the exact same article URL)
- Out of `1,520` constituent rows audited, there were `1,520` unique `article_id`s across `1,178` unique story URLs (`342` rows shared a URL with another constituent).
- **Schema Update Applied**: Migration [`003_add_specific_title.sql`](file:///d:/trading-predictions/migrations/003_add_specific_title.sql) added `specific_title` and an index on `url` to `articles` so our Phase 2 scorer and Phase 3 feature engine can use the company-specific angle and deduplicate story volume when counting distinct macro events.

---

## 4. Field Completeness, Latency & Vendor Sentiment

- **Field Completeness**: **100.0%** across `article_id`, `title`, `specific_title`, `summary`, `long_summary`, `link`, `source`, `article_type`, `company`, `symbol`, `ISIN`, `scrip_code`, `sentiment`, `date`, and `created_at`. `sector` is **91.2%** complete (`8.8%` null).
- **Publication-to-Ingestion Latency (`created_at - date`)**:
  - **Min**: `46.8s`
  - **25th Percentile**: `157.2s` (~2.6 min)
  - **Median**: `206.0s` (~3.4 min)
  - **75th Percentile**: `288.8s` (~4.8 min)
  - **Max**: `21,674.8s` (~6.0 hours for late-syndicated stories)
  - **Negative lag (`created_at < date`)**: `0` rows.
  - **Takeaway**: Using `available_at = max(published_at, ingested_at)` protects against the ~3.4-minute median pipeline delay and occasional multi-hour syndication lags during intraday backtesting.
- **Publisher Sources**: Economic Times (`38.6%`), Moneycontrol (`24.1%`), Mint (`20.9%`), Business Standard (`14.9%`), NDTV Profit (`1.5%`).
- **Article Types** (7 clean categories provided by vendor):
  - `Operations & Business Updates` (`28.6%`)
  - `Financial Performance` (`17.0%`)
  - `Strategic Actions` (`15.1%`)
  - `Analyst & Credit Ratings` (`14.0%`)
  - `Management, Governance & Legal` (`12.7%`)
  - `Bulk/Block/Insider Deals` (`7.7%`)
  - `Corporate Actions` (`4.8%`)
- **Vendor `sentiment` Field Distribution**:
  - `positive`: `757` (`49.8%`)
  - `neutral`: `486` (`32.0%`)
  - `negative`: `277` (`18.2%`)
  - **Origin & Trustworthiness**: The vendor `sentiment` is an automated LLM/classifier label generated per-company at ingestion time alongside `specific_title` and `long_summary`. It exhibits a strong positive skew (`~50%` positive vs `18%` negative) and lacks a continuous confidence score or Nifty index-level relevance weight—confirming the value of Phase 2's calibrated scorer and the Phase 4 ablation (`vendor_sentiment` vs. custom scorer).

