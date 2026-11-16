# Phase 0: Decisions, Compliance Design & Vendor Checklist

## 1. Prediction Target, Window & Objective Label Definitions

- **Primary Instrument**: `NIFTY50` (Nifty 50 Index).
- **Intraday Window ($W$)**: Parameterized across the pipeline (`15`, `30`, `60` minutes). **30 minutes** is the primary operational window; 15m and 60m are evaluated side-by-side in Phase 4.
- **Point-in-Time Rule**: Every run at timestamp `as_of` queries exclusively rows where `available_at <= as_of`.

### Objective Labels over Window $[t, t + W)$
Let $r_{t,W} = \ln(P_{t+W} / P_t)$ be the realized log return over the window, and $\text{ATR}_W(t)$ be the 14-period Average True Range computed on $W$-minute bars up to $t$ (expressed in return terms: $\text{ATR}_{14}(t) / P_t$).

1. **Market Regime (`bullish` | `bearish` | `sideways`)**
   - `sideways` if $\text{ADX}_{14}(t+W) < 20$ **or** $|r_{t,W}| \le k \cdot \text{ATR}_W(t)$ (default $k = 0.5$, calibrated on training folds for balanced class priors; fixed $\pm 0.3\%$ band also reported as a baseline reference).
   - `bullish` if $r_{t,W} > k \cdot \text{ATR}_W(t)$ and $\text{ADX}_{14}(t+W) \ge 20$.
   - `bearish` if $r_{t,W} < -k \cdot \text{ATR}_W(t)$ and $\text{ADX}_{14}(t+W) \ge 20$.

2. **Direction Probability (`prob_up`)**
   - Binary target $y_{\text{dir}} = \mathbb{I}(r_{t,W} > 0)$, evaluated on non-neutral moves and across all windows.

3. **Entry Conditions (`favourable` | `cautious` | `avoid`)**
   - Let $\text{MAE}_{t,W}$ be the Maximum Adverse Excursion (worst intrawindow drawdown against the predicted regime direction using 1-minute highs/lows inside $[t, t+W)$).
   - `favourable`: Realized regime matches predicted directional regime (`bullish` or `bearish`), $\text{MAE}_{t,W} \le m \cdot \text{ATR}_W(t)$ ($m = 0.75$), and no high-severity event shock occurs within the window.
   - `avoid`: $\text{MAE}_{t,W} > m \cdot \text{ATR}_W(t)$, realized regime reverses against prediction, India VIX spikes above the 90th percentile, or an unscheduled policy/geopolitical shock (`is_event = true`) is active.
   - `cautious`: All intermediate states (including low-volatility `sideways` regimes).

---

## 2. Refresh Schedule

| Run Type | Timing (IST) | Trigger & Inputs |
| :--- | :--- | :--- |
| **`pre_open`** | `09:10 IST` | Runs after the 09:00–09:08 NSE pre-open auction settles. Incorporates all overnight news (`post_market`, `weekend`, `holiday`, `pre_market`), US close, Brent crude, USD/INR, GIFT Nifty, and pre-open equilibrium price. Covers opening window `[09:15, 09:45)`. |
| **`scheduled`** | Every 30 min (`09:45, 10:15, ..., 15:00 IST`) | Runs at each window boundary using latest 1-minute Nifty & heavyweight bars, options chain snapshot, breadth, VIX, and scored news up to `as_of`. |
| **`event`** | On-demand (cooldown: 10 min) | Triggered when a newly ingested article is scored with `is_event = true` and `event_severity >= 0.7`. Re-evaluates regime and entry conditions immediately. |

---

## 3. Compliance Design (Investment Adviser Licence Positioning)

- **Positioning**: Market-level regime and entry-environment assessment issued under the platform's Investment Adviser (IA) licence. Never issues an instrument-specific "buy/sell/enter now" instruction or price target.
- **Mandatory Output Fields**:
  1. `as_of` timestamp (IST) and `window_minutes` horizon.
  2. Calibrated regime probabilities and entry-condition category (`favourable` / `cautious` / `avoid`).
  3. Explicit **invalidation condition** (e.g., *"Bullish regime view is invalidated if Nifty 50 breaks below VWAP (24,180) or India VIX rises above 16.5"*).
  4. **AI-Use Disclosure**: *"AI-assisted market regime assessment generated from quantitative indicators and news sentiment; does not constitute an assurance of returns or an execution instruction."*
- **Auditability**: Every generated view is persisted in `predictions` with `features_used`, `data_freshness`, `template_version`, `rendered_text`, and `published` flag.
- **Output Kill Switch**: Setting `status = 'suppressed'` (automatic on stale data > 3 minutes or manual override) withholds the view and displays a neutral data-staleness notice.

---

## 4. Paid Intraday Data Vendor Checklist (TrueData vs. Global Datafeeds)

Before finalizing the vendor subscription in Phase 0, verify the following capabilities:

| Requirement | Why it is needed | TrueData | Global Datafeeds (GDFL) |
| :--- | :--- | :--- | :--- |
| **Nifty 50 & India VIX 1-min bars (2–3 yr history)** | Intraday regime features (ADX, VWAP, ATR, EMA slopes) & walk-forward backtest | Verify depth | Verify depth |
| **Top 10–50 Nifty constituent 1-min bars + volume** | Heavyweight participation & true volume-weighted VWAP (index spot has no volume) | Verify | Verify |
| **Historical Options Chain snapshots (OI, IV, Strike LTP)** | Put-Call Ratio (PCR), OI buildup, and IV skew features in Phase 3 | Check if 1-min/5-min historical chain is available | Check if historical chain or only expired contract OHLCV is provided |
| **Exchange timestamps & tick/bar lag** | Point-in-time `available_at` accuracy and live latency monitoring | Confirm | Confirm |
| **Advance / Decline breadth feed** | Intraday participation features | Confirm (or compute from Nifty 50 1-min bars) | Confirm |

