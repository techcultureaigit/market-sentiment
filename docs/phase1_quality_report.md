# Phase 1 Data Quality & Backfill Report

## 1. News Corpus Summary (`articles` & `raw_articles`)
- **Total Ingested Articles**: `4,141`
- **Unique Story URLs**: `2,990`
- **Distinct Symbols Covered**: `56`
- **Date Range (UTC)**: `2026-07-14T07:53:55+05:30` to `2026-11-11T13:05:39+05:30`
- **Missing Title/Summary Rows**: `0`
- **Point-in-Time Lookahead Violations (`available_at < published_at`)**: `0`
- **Ingestion Latency (`ingested_at - published_at`)**: Median `210.77s` | 95th Percentile `1251.26s` | Late arrivals (>15 min): `227`

### Monthly Coverage Density
| Month (IST) | Articles | Symbols Active |
| :--- | ---: | ---: |
| `2026-07` | `679` | `50` |
| `2026-08` | `1,229` | `53` |
| `2026-09` | `819` | `53` |
| `2026-10` | `1,035` | `51` |
| `2026-11` | `379` | `52` |

### Session Bucket Breakdown (with NSE Holiday Calendar)
| Session Bucket | Article Count |
| :--- | ---: |
| `in_session` | `1,687` |
| `post_market` | `1,101` |
| `pre_market` | `761` |
| `weekend` | `498` |
| `holiday` | `94` |

## 2. Daily Market Data Coverage (`daily_bars`)
| Symbol | Trading Days | Start Date | End Date |
| :--- | ---: | :--- | :--- |
| `AXISBANK` | `495` | `2024-11-11` | `2026-11-11` |
| `BANKNIFTY` | `493` | `2024-11-11` | `2026-11-11` |
| `BHARTIARTL` | `495` | `2024-11-11` | `2026-11-11` |
| `BRENT` | `504` | `2024-11-11` | `2026-11-11` |
| `HDFCBANK` | `495` | `2024-11-11` | `2026-11-11` |
| `ICICIBANK` | `495` | `2024-11-11` | `2026-11-11` |
| `INDIAVIX` | `491` | `2024-11-11` | `2026-11-11` |
| `INFY` | `495` | `2024-11-11` | `2026-11-11` |
| `ITC` | `495` | `2024-11-11` | `2026-11-11` |
| `LT` | `495` | `2024-11-11` | `2026-11-11` |
| `NASDAQ` | `501` | `2024-11-11` | `2026-11-10` |
| `NIFTY50` | `494` | `2024-11-11` | `2026-11-11` |
| `RELIANCE` | `495` | `2024-11-11` | `2026-11-11` |
| `SBIN` | `495` | `2024-11-11` | `2026-11-11` |
| `SP500` | `501` | `2024-11-11` | `2026-11-10` |
| `TCS` | `495` | `2024-11-11` | `2026-11-11` |
| `USDINR` | `518` | `2024-11-11` | `2026-11-11` |

## 3. Intraday 1-Minute Bars (`intraday_bars`) & Breadth (`market_breadth`)
- **Computed Nifty 50 Breadth Snapshots (`market_breadth`)**: `1,695` (`2026-11-04T09:15:00+05:30` to `2026-11-11T13:21:00+05:30`)
- **Calendars Loaded**: `14` NSE holidays | `53` Nifty F&O expiries (`12` monthly)

| Symbol | 1m Bars | Earliest Bar (UTC) | Latest Bar (UTC) |
| :--- | ---: | :--- | :--- |
| `AXISBANK` | `1,695` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `BANKNIFTY` | `1,747` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `BHARTIARTL` | `1,694` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `HDFCBANK` | `1,695` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `ICICIBANK` | `1,694` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `INDIAVIX` | `1,747` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `INFY` | `1,693` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `ITC` | `1,695` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `LT` | `1,693` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `NIFTY50` | `1,747` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `RELIANCE` | `1,695` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `SBIN` | `1,694` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
| `TCS` | `1,694` | `2026-11-04T09:15:00+05:30` | `2026-11-11T13:21:00+05:30` |
