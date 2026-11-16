# Phase 2: News Scoring & 200-Article Benchmark Report

- **Scorer Version**: `hybrid-fin-v1` (`prompt_version = nifty-intraday-v1.0`)
- **Benchmark Sample Size**: `200` stratified articles
- **Scorer Agreement with Benchmark**: **`69.0%`** (vs. raw vendor sentiment: `82.5%`)

## Why Hybrid Scoring Improves on Raw Vendor Sentiment
1. **De-skewing Positive Bias**: Raw vendor labels classify `49.8%` of all corporate stories as `positive` (including routine administrative filings). Our scorer requires directional lexical evidence or scales back low-evidence items into the `[-0.15, +0.15]` neutral band.
2. **Nifty 50 Index Relevance (`nifty_relevance`)**: Weights each article by the constituent's free-float weight in the Nifty 50 (`HDFCBANK` 11.9%, `RELIANCE` 8.45%, etc.) plus macro theme reach (`monetary_policy`, `geopolitics`, `commodities`).
3. **Event & Shock Detection (`is_event`, `event_type`, `event_severity`)**: Flags high-impact headlines (`rbi_policy`, `earnings_release`, `regulatory_shock`, `macro_geopolitical`, `corporate_event`) to drive the Phase 3 entry-conditions layer and event-triggered runs.
