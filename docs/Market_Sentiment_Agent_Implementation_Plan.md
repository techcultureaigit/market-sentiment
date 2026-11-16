# Market Sentiment Agent: Implementation Plan

Prepared 11 Nov 2026. Scope: Nifty 50 first, then the wider open market. Dates assume a start on Monday 16 Nov 2026 with 1 to 2 engineers; shift them as needed.

## 1. Goal and success criteria

Build an agent on our trading platform that combines intraday market data, technical indicators and news to classify the market regime (bullish, bearish or sideways) over a chosen window, estimate whether conditions favour entering the market, and explain the drivers in plain language.

- **First target:** Nifty 50 regime (bullish, bearish, sideways) and entry conditions over a fixed intraday window (15, 30 or 60 minutes, chosen in Phase 0), with a neutral band (for example within ±0.3%).
- **Success gate (end of Phase 4):** the full model beats the baselines (always-up, yesterday's direction, market-features-only) on walk-forward validation, with calibrated probabilities. If it does not, we do not launch; we iterate.
- **Positioning:** an advisory output issued under our investment adviser licence: a market-level view of whether conditions look favourable for entering the market, never an instrument-specific “enter now” call, with no assurance of outcomes and compliance-approved wording.

## 2. Phase overview

| # | Phase | Target dates | Duration | Status |
| --- | --- | --- | --- | --- |
| 0 | Decisions, compliance design and setup | 16 Nov to 29 Nov | 2 weeks | Not started |
| 1 | Data pipeline and backfill (news and intraday data) | 30 Nov to 20 Dec | 3 weeks | Not started |
| 2 | News scoring | 7 Dec to 20 Dec | 2 weeks (overlaps Phase 1) | Not started |
| 3 | Features and models (regime, direction, entry conditions) | 21 Dec to 10 Jan | 3 weeks | Not started |
| 4 | Backtest, calibration, go/no-go | 11 Jan to 24 Jan | 2 weeks | Not started |
| 5 | Agent layer | 25 Jan to 7 Feb | 2 weeks | Not started |
| 6 | Shadow run (live, hidden from users) | 8 Feb to 21 Mar | 6 weeks | Not started |
| 7 | Platform integration and compliance | 25 Jan to 28 Mar | runs in parallel | Not started |
| 8 | Limited launch | from 29 Mar | ongoing | Not started |
| 9 | Expansion to the open market | after launch | TBD | Not started |

## 3. Phase details

### Phase 0: Decisions, compliance design and setup (16 Nov to 29 Nov)

- Fix the prediction target: regime labels (bullish, bearish, sideways) and an entry-conditions output over an intraday window, with objective label definitions (for example, return over the window relative to ATR, with low ADX for sideways).
- Decide the refresh schedule: a pre-open outlook (about 8:45 to 9:10 AM IST, using everything since the last close) plus intraday updates at fixed intervals and on significant news. A prediction uses only articles available at or before that time; later news feeds the next prediction, so nothing is ignored. Define the label to match (for example, today's close vs yesterday's close).
- Stand up PostgreSQL (local or Docker for development) and run the ingestion script on sample data.
- Audit the news API: historical queries by date range, rate limits, pagination, whether articles are edited after publishing, how far back data goes, and the origin of the existing `sentiment` field.
- Choose the market data sources: Nifty OHLCV, India VIX, FII/DII flows, options put-call ratio, GIFT Nifty, US indices, crude oil, USD/INR.

* Compliance design with the compliance team: wording of the output (a market-level view that conditions look favourable or not, never an instrument-specific “enter now” call), AI-use disclosure, record-keeping of every output and its rationale, and approval of the wording traders see.
* Choose the intraday window (15, 30 or 60 minutes) and the refresh cadence.

**Exit criteria:** target, label, window and compliance design written down; API audit complete; database running; sample articles ingested.

### Phase 1: Data pipeline and backfill, news and intraday market data (30 Nov to 20 Dec)

- Scheduled ingestion job (every few minutes during market hours, hourly otherwise) with retries and logging.
- Backfill 2 to 3 years of news; confirm coverage is even over time and note any source changes.
- Ingest market data into Postgres on the same timestamp conventions (UTC stored, IST for session logic).
- Add an NSE holiday calendar to session bucketing.
- Data quality checks: duplicate rate, missing timestamps, gaps, late-arriving articles (`ingested_at` vs `published_at`).

* Intraday market data: 1-minute Nifty candles, options chain snapshots (open interest, implied volatility, put-call ratio), India VIX, breadth and heavyweight-stock data, all with exchange timestamps.
* Real-time news path with latency monitoring, since news now acts as an event and shock input during market hours.

**Exit criteria:** continuous ingestion running; historical news and market data loaded; quality report reviewed.

### Phase 2: News scoring (7 Dec to 20 Dec)

- Decide the scorer: LLM via API (better relevance judgement on global stories) or FinBERT (cheaper, offline), or a hybrid with FinBERT as a first pass.
- Output per article: `sentiment_score`, `confidence`, `nifty_relevance` (0 to 1), `sectors[]`, `macro_theme`.
- Hand-label a sample of about 200 articles and measure scorer agreement against it; tune the prompt or model until acceptable.
- Estimate cost and runtime for the full backfill, then score history and set up scoring of new articles as they arrive.
- Store model name and prompt version with every score so results are reproducible.

* Add an event and shock flag from news (policy, results, geopolitical headlines) for the entry-conditions layer.

**Exit criteria:** all historical and new articles scored; agreement with the hand-labelled sample documented.

### Phase 3: Features and models: regime, direction, entry conditions (21 Dec to 10 Jan)

- News features per window: relevance-weighted average sentiment, latest-hours sentiment, day-over-day change, news volume, share of strongly negative items, sector-level sentiment (banking, IT, energy), macro-theme counts.
- Market features: returns over 1, 5 and 20 days, RSI, moving-average distance, VIX level and change, PCR, FII/DII flows, overnight cues.
- Models: logistic regression first, then LightGBM or XGBoost.
- Baselines: always-up, yesterday's direction, market-features-only.
- Validation: walk-forward only, never random splits.

* Intraday regime features: ADX, EMA 20/50 slope, distance from VWAP, ATR, Bollinger bandwidth, RSI and MACD, market structure, opening-range and previous-day levels.
* Participation and positioning features: breadth, sector and heavyweight participation, options open-interest buildup, put-call ratio and implied volatility.
* Three outputs: regime label, direction probability over the window, and an entry-conditions score (favourable, cautious, avoid) driven by volatility, event risk, liquidity and regime confidence.
* Keep the indicator set small and low in correlation, and add an indicator-rules-only regime classifier as an extra baseline.

**Exit criteria:** reproducible training pipeline; baseline and model results in one comparison table.

### Phase 4: Backtest, calibration and go/no-go (11 Jan to 24 Jan)

- Lookahead audit: confirm every feature uses only data available at prediction time (`available_at`).
- Metrics: accuracy, log-loss, Brier score, calibration curve; results by market regime (trending, volatile, event days).
- Ablations: market-only vs market plus news, vendor sentiment vs our scorer, pre-open vs post-close prediction time.
- Include realistic costs if the signal will ever drive trades.
- **Go/no-go decision** against the success gate in section 1.

* Evaluate regime labels and entry conditions on intraday data with costs and slippage, split by regime, expiry days and event days.
* Measure how often “favourable” conditions were followed by adverse moves, and report it openly.

**Exit criteria:** written go/no-go with supporting results. A "no-go" sends us back to Phases 2 and 3 rather than forward.

### Phase 5: Agent layer (25 Jan to 7 Feb)

- Tools for the agent: fetch recent scored news, query features, run the model, fetch market snapshot.
- The model produces the numbers; the LLM only orchestrates and explains. It never invents probabilities.
- Output format: direction, probability, confidence, top drivers, and a plain-language rationale.
- Guardrails: refuse out-of-scope questions, show data freshness, and flag low-confidence or stale-data situations.

* Guardrails against guarantee-style language and out-of-scope instruments, plus a compliance-approved template for how each output is phrased.
* Every output carries an AI-generated disclosure, an as-of time and an invalidation condition.

**Exit criteria:** agent returns consistent, explainable output on a test set of scenarios, including edge cases (holiday, no news, API outage).

### Phase 6: Shadow run (8 Feb to 21 Mar)

- Run the full system live every trading day, logging predictions without showing them to users.
- Compare predictions with realized outcomes; track live accuracy and calibration versus the backtest.
- Monitor feature drift, news volume anomalies, scorer failures and latency.
- Define retraining cadence and alert thresholds.

* Review how each output would have read to traders at different points of the day, including event days and expiry days.

**Exit criteria:** live performance within an agreed tolerance of backtest results for the full period; no unresolved monitoring issues.

### Phase 7: Platform integration and compliance (25 Jan to 28 Mar, parallel)

- Build the sentiment service as separate from the website, exposing a versioned REST API (for example GET /v1/sentiment/nifty/latest) plus a push channel for live updates (SSE, or WebSocket if two-way messaging is needed). The website backend consumes it and never reads our Postgres database directly. Agree the response contract with the website team: instrument, prediction time, direction, probabilities, confidence, top drivers and explanation, model version, data-freshness indicator, and a status field for degraded states. The UI shows an as-of timestamp and flags stale data.
- Disclaimers and wording reviewed by compliance. Confirm with compliance which adviser obligations apply to a market-level output (disclosures, record-keeping, audits) before any user-facing launch.
- Logging for audit: what the model saw, what it output, and when.
- Authentication, rate limiting and API versioning on the service, plus cost controls on LLM calls. The website may cache the latest prediction in its existing MongoDB or Redis so it keeps working if the service is briefly down.

**Exit criteria:** compliance sign-off; integration tested on staging.

### Phase 8: Limited launch (from 29 Mar)

- Release to a small cohort, collect feedback, watch live metrics.
- Gradual rollout only if live metrics hold.

### Phase 9: Expansion to the open market (after launch)

- The pipeline is built around a generic instrument interface (symbol, exchange, timezone, trading hours, data source), so new markets are configuration plus new data feeds.
- Suggested order: Bank Nifty and sector indices, then major US indices, then selected individual stocks.
- Each new market repeats the Phase 3 to 4 validation gate before launch.

## 4. Data storage rationale

The news is stored even though the API provides it, for these reasons:

- **Backtesting needs history** with stable timestamps; the API may not serve it reliably or at all.
- **Point-in-time correctness:** we record when we first saw each article (`available_at`) so backtests cannot see the future.
- **Edits and deletions:** publishers change or remove articles; stored first-seen copies keep history stable.
- **Scoring is expensive:** scores are stored once and reused rather than recomputed.
- **Reproducibility and audit:** we can show exactly what the model saw on any day.
- **Joins:** features combine news with market data in one place.

## 5. Key risks

| Risk | Mitigation |
| --- | --- |
| Lookahead bias inflates backtest results | Use available\_at, audit every feature, walk-forward validation |
| No predictive edge beyond market features | Ablation test in Phase 4; go/no-go gate; fall back to a descriptive regime indicator |
| News API gaps, edits or outages | Raw storage, monitoring, freshness flags in the agent output |
| Intraday data gaps or latency | Exchange-timestamped feeds, latency monitoring, show no view when data is stale |
| LLM scoring cost and drift | Prompt and model versioning, batching, FinBERT first pass, spot-check samples |
| Regime change degrades performance | Drift monitoring, scheduled retraining, regime-based reporting |
| Traders over-rely on the entry view | Probabilistic wording, invalidation levels, AI-use disclosure, no assurance language, output kill switch |
| Regulatory exposure | Compliance design in Phase 0 and review in Phase 7; advisory positioning under our licence; full audit logs |

## 6. Open decisions

- Scorer choice: LLM API, FinBERT, or hybrid (decide in Phase 2).
- Intraday window (15, 30 or 60 minutes), refresh cadence, and whether to also publish a pre-open outlook.
- Origin and trustworthiness of the API's `sentiment` field.
- Who owns compliance review and when it starts.
- Hosting: where the sentiment service and its Postgres database run in production, kept separate from the website's MongoDB.

* Whether a market-level view, identical for all traders, needs any per-trader suitability step (confirm with compliance).
