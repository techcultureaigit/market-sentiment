# 15-Day 5-Minute (`5m`) Candle Walk-Forward Backtest Report

- **Horizon Evaluated**: `5 minutes` (`interval_sec = 300`)
- **Trading Sessions Tested**: `15 trading days` (`2026-10-21` through `2026-11-11`)
- **Total 5-Minute Candles**: `1,110` (`74` intraday 5-minute transitions per trading day)
- **Out-of-Sample Walk-Forward Test Candles**: `611`
- **Point-in-Time Lookahead Violations (`available_at > as_of`)**: `0`

---

## 1. Out-of-Sample Walk-Forward Model Comparison (`611` Test Candles)

| Model / Baseline | Out-of-Sample Accuracy | Macro-F1 | Log-Loss | Brier Score |
| :--- | ---: | ---: | ---: | ---: |
| `2_prev_direction` (Momentum / Prev 5m Candle) | `48.94%` | `48.66%` | `0.6971` | `0.2520` |
| `5_market_plus_vendor_news_lr` (Raw Vendor Sentiment) | `49.43%` | `48.93%` | `0.6932` | `0.2500` |
| `3_indicator_rules` (Technical Rules Only) | `49.59%` | `49.08%` | `0.6933` | `0.2501` |
| `4_market_only_lr` (Market Technicals Only) | `49.59%` | `49.08%` | `0.6932` | `0.2500` |
| `6_full_model_lr` (Market + Hybrid News Scorer LR) | `50.90%` | `48.61%` | `0.6917` | `0.2493` |
| **`7_full_model_hybrid_gbt` (Full Hybrid Agent Model)** | **`51.06%`** (`51.71%` all 1,110) | `48.35%` | **`0.6917`** | **`0.2493`** |

### Regime & Event-Active Sub-Slices (Out-of-Sample)
- **Event-Active 5m Windows (`313` candles)**: **`53.67%` accuracy** (`Brier = 0.2472`)
- **`Favourable` Entry Environment Calls (`262` candles)**: **`53.82%` directional hit rate** (`2.29%` adverse excursion rate)
- **Trending Windows (`ADX >= 20`, `533` candles)**: **`51.03%` accuracy** (`Brier = 0.2492`)
- **Sideways Windows (`ADX < 20`, `78` candles)**: **`51.28%` accuracy** (`Brier = 0.2494`)

---

## 2. Day-by-Day 5-Minute Accuracy Breakdown (15 Trading Days)

| Trade Date | 5m Candles | Directional Accuracy | Nifty Open | Nifty Close |
| :--- | ---: | ---: | ---: | ---: |
| `2026-10-21` | 74 | **58.11%** | 23,016.00 | 22,738.45 |
| `2026-10-22` | 74 | **50.00%** | 22,687.75 | 22,666.55 |
| `2026-10-23` | 74 | **56.76%** | 22,742.20 | 22,516.40 |
| `2026-10-26` | 74 | **51.35%** | 22,524.40 | 22,717.05 |
| `2026-10-27` | 74 | **45.95%** | 22,707.55 | 22,760.75 |
| `2026-10-28` | 74 | **47.30%** | 22,678.00 | 22,706.80 |
| `2026-10-29` | 74 | **52.70%** | 22,690.65 | 22,940.75 |
| `2026-10-30` | 74 | **44.59%** | 22,964.50 | 23,010.00 |
| `2026-11-02` | 74 | **45.95%** | 23,222.30 | 23,331.10 |
| `2026-11-03` | 74 | **58.11%** | 23,582.25 | 23,731.85 |
| `2026-11-04` | 74 | **52.70%** | 23,734.15 | 23,911.70 |
| `2026-11-05` | 74 | **54.05%** | 24,378.05 | 24,542.20 |
| `2026-11-06` | 74 | **59.46%** | 24,518.20 | 24,938.90 |
| `2026-11-09` | 74 | **52.70%** | 25,073.65 | 25,003.65 |
| `2026-11-11` | 74 | **45.95%** | 25,337.80 | 25,130.00 |

---

## 3. How to Re-Run or Customize

```powershell
# Run 15-day 5-minute candle walk-forward backtest
.\.venv\Scripts\python.exe main.py backtest --window 5 --days 15

# Run 10-day 5-minute candle walk-forward backtest
.\.venv\Scripts\python.exe main.py backtest --window 5 --days 10
```

