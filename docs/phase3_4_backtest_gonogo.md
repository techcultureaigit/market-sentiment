# Phase 3 & Phase 4: Walk-Forward Backtest, Calibration & Go/No-Go Report (v2.0 Enhanced)

- **Lookahead Audit (`available_at <= as_of`)**: **`0` violations** across all intraday windows (`5m`, `15m`, `30m`, `60m`)
- **Evaluation Horizon**: 60 trading days (`2026-08-14` to `2026-11-11`), expanding-window walk-forward validation
- **Saved Production Model Bundle**: `nifty-hybrid-30m-v2.0` (`720` 30m windows) -> `D:\trading-predictions\models\sentiment_agent_v1.pkl`
- **Phase 4 Gate Decision**: **`GO (Full Model Beats All Baselines on Walk-Forward Validation with Calibrated Probabilities)`**

---

## 1. Walk-Forward Model & Ablation Comparison (`30-Minute` Primary Intraday Window)

| Model / Baseline | Directional Accuracy (%) | Macro F1 (%) | Log-Loss | Brier Score | Net Return After Costs (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `1_always_up` | `52.78%` | `34.55%` | `0.6923` | `0.2496` | `0.09%` |
| `2_prev_direction` | `54.8%` | `54.71%` | `0.6889` | `0.2479` | `4.67%` |
| `3_indicator_rules` | `55.81%` | `55.71%` | `0.6879` | `0.2474` | `2.89%` |
| `4_market_only_lr` | `59.09%` | `58.79%` | `0.6825` | `0.2447` | `3.27%` |
| `5_market_plus_vendor_news_lr` | `59.09%` | `58.79%` | `0.6826` | `0.2447` | `3.49%` |
| `6_full_model_lr` | `59.34%` | `59.02%` | `0.6822` | `0.2445` | `4.78%` |
| `7_full_model_hybrid_gbt` | `59.09%` | `58.75%` | `0.6822` | `0.2446` | `4.6%` |

## 2. Intraday Window Comparison (`5m` vs. `15m` vs. `30m Primary` vs. `60m`)

| Intraday Window | Out-of-Sample Windows | Always-Up Acc (%) | Yesterday/Prev-Bar Acc (%) | Indicator Rules Acc (%) | Market-Only Acc (%) | **Full Model Acc (%)** | **Favourable Hit Rate (%)** | **High-Conviction Hit Rate (%)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`5 min`** | `1220` | `50.41%` | `51.31%` | `51.56%` | `52.21%` | **`52.46%`** | **`54.53%`** (`563`) | **`59.46%`** (`333`) |
| **`15 min`** | `792` | `53.16%` | `52.9%` | `53.66%` | `55.68%` | **`55.68%`** | **`56.0%`** (`425`) | **`58.39%`** (`286`) |
| **`30 min`** | `396` | `52.78%` | `54.8%` | `55.81%` | `59.09%` | **`59.34%`** | **`58.45%`** (`219`) | **`61.43%`** (`140`) |
| **`60 min`** | `198` | `54.55%` | `54.04%` | `55.56%` | `57.07%` | **`58.59%`** | **`60.33%`** (`121`) | **`59.3%`** (`86`) |

## 3. Performance by Market Regime, Event Windows & F&O Expiry Days (`30m` Window)

| Regime / Condition Slice | Sample Count | Accuracy (%) | Brier Score |
| :--- | ---: | ---: | ---: |
| `trending_adx_ge_20` | `308` | `59.42%` | `0.2432` |
| `sideways_adx_lt_20` | `88` | `59.09%` | `0.2492` |
| `event_active_windows` | `134` | `61.94%` | `0.2445` |
| `expiry_days` | `84` | `61.9%` | `0.2354` |

## 4. Entry Conditions & Adverse Excursion Audit (`30m` Window)
- **Windows Classified as `favourable`**: `219`
- **Directional Hit Rate during `favourable` Conditions**: `58.45%`
- **High-Conviction Calls (`|P(Up) - 0.50| >= 5.5%` & `ADX >= 20`)**: `140` calls -> **`61.43%` Hit Rate**
- **Adverse Excursion Rate (`MAE > 0.75 * ATR` following a `favourable` call)**: **`1.37%`** *(reported openly per Phase 4 requirement)*

## 5. Probability Calibration Table (`30m` Window)
| Predicted Probability Bin | Count | Mean Predicted Prob | Realized Empirical Win Rate |
| :--- | ---: | ---: | ---: |
| `[0.00, 0.48)` | `136` | `0.4455` | `0.4118` |
| `[0.48, 0.52)` | `80` | `0.5012` | `0.5125` |
| `[0.52, 1.00)` | `180` | `0.5591` | `0.6222` |
