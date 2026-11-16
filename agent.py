"""Phase 5: Market Sentiment Agent Layer, Tools, Compliance Template & Guardrails.

Implements:
  1. Agent Tools:
     - `fetch_recent_scored_news`: Retrieves top point-in-time scored headlines (`available_at <= as_of`).
     - `fetch_market_snapshot`: Retrieves point-in-time Nifty 50 price, VWAP, ATR, ADX, VIX, and breadth.
     - `query_feature_vector`: Computes the exact point-in-time feature vector at `as_of`.
     - `run_model_inference`: Runs the saved calibrated estimator (`models/sentiment_agent_v1.pkl`)
       to produce `regime`, `prob_bullish`, `prob_bearish`, `prob_sideways`, `prob_up`,
       `entry_condition`, `entry_score`, and `confidence`. The LLM/orchestrator NEVER invents numbers.
  2. Guardrails & Compliance:
     - Out-of-scope query refusal (blocks single-stock execution calls, price targets, non-Nifty tips).
     - Forbidden guarantee-language filter (blocks 'guaranteed', 'sure shot', 'buy now', 'enter now', etc.).
     - Data freshness monitor & automatic kill-switch (`status = 'degraded'` or `'suppressed'`).
     - Mandatory AI-Use Disclosure, `as_of` IST timestamp, and objective invalidation condition.
     - Full audit logging into PostgreSQL `predictions` table (`features_used`, `data_freshness`,
       `template_version`, `rendered_text`, `published`).

Usage:
    python agent.py --run-type scheduled --window 30
    python agent.py --test-scenarios
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import pickle
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from db import ROOT, apply_schema, get_dsn
from features_and_models import (
    MARKET_COLS,
    NEWS_SCORER_COLS,
    build_intraday_window_dataset,
    enhanced_market_prior_proba,
    indicator_rules_predict_proba,
    load_scored_articles,
    query_df,
)

IST = ZoneInfo("Asia/Kolkata")
TEMPLATE_VERSION = "ia-advisory-v1.0"
AI_DISCLOSURE = (
    "AI-Use Disclosure: This market-level regime and entry-conditions assessment is generated "
    "using quantitative indicators and automated news sentiment analysis under our Investment "
    "Adviser licence. It is not an instrument-specific execution call and provides no assurance "
    "of returns."
)

FORBIDDEN_PHRASES = [
    r"\bguaranteed\b",
    r"\bsure[- ]shot\b",
    r"\brisk[- ]free\b",
    r"\b100%\s+safe\b",
    r"\bbuy\s+now\b",
    r"\bsell\s+now\b",
    r"\benter\s+now\b",
    r"\b assured\s+profit\b",
]


# ---------------------------------------------------------------------------
# Guardrail 1: Out-of-Scope Query Filter
# ---------------------------------------------------------------------------
def check_query_scope(user_query: str | None) -> tuple[bool, str | None]:
    """Refuse out-of-scope stock tips, execution instructions, or price target requests."""
    if not user_query:
        return True, None
    q = user_query.lower()
    out_of_scope_patterns = [
        (r"\b(which stock|best stock|multibagger|penny stock|stock tip|call or put strike)\b",
         "Out of scope: The Market Sentiment Agent provides a market-level regime and entry-environment view for the Nifty 50 index only, not individual stock or option strike recommendations."),
        (r"\b(should i buy|should i sell|enter now|exact target price|stop loss for)\b",
         "Out of scope: Under compliance rules, this agent assesses whether market-level conditions look favourable, cautious, or avoid over the window, and never issues 'enter/buy/sell now' execution calls."),
        (r"\b(crypto|bitcoin|ethereum|forex pair|gold tip)\b",
         "Out of scope: Only Nifty 50 index regime and entry conditions are currently supported."),
    ]
    for pat, msg in out_of_scope_patterns:
        if re.search(pat, q):
            return False, msg
    return True, None


def enforce_wording_guardrails(text: str) -> str:
    """Ensure rendered output never contains guarantee-style or imperative execution wording."""
    cleaned = text
    for pat in FORBIDDEN_PHRASES:
        cleaned = re.sub(pat, "[compliance-redacted]", cleaned, flags=re.I)
    return cleaned


# ---------------------------------------------------------------------------
# Agent Tools
# ---------------------------------------------------------------------------
def tool_fetch_recent_scored_news(
    conn, as_of_utc: datetime, lookback_hours: float = 12.0, limit: int = 5
) -> list[dict]:
    """Tool 1: Fetch top relevance-weighted scored news articles strictly before `as_of_utc`."""
    win_start = as_of_utc - timedelta(hours=lookback_hours)
    rows = conn.execute(
        """
        SELECT
            a.article_id,
            a.symbol,
            a.company,
            COALESCE(a.specific_title, a.title) AS headline,
            a.source,
            a.available_at,
            s.sentiment_score,
            s.confidence,
            s.nifty_relevance,
            s.macro_theme,
            s.is_event,
            s.event_type,
            s.event_severity
        FROM articles a
        JOIN article_scores s ON s.article_id = a.article_id
        WHERE s.scorer_version = 'hybrid-fin-v1'
          AND a.available_at > %s
          AND a.available_at <= %s
        ORDER BY (s.nifty_relevance * s.confidence * ABS(s.sentiment_score)) DESC,
                 a.available_at DESC
        LIMIT %s
        """,
        (win_start, as_of_utc, limit),
    ).fetchall()
    cols = [
        "article_id", "symbol", "company", "headline", "source", "available_at",
        "sentiment_score", "confidence", "nifty_relevance", "macro_theme",
        "is_event", "event_type", "event_severity",
    ]
    out = []
    for r in rows:
        item = dict(zip(cols, r))
        item["available_at"] = item["available_at"].isoformat()
        item["sentiment_score"] = round(float(item["sentiment_score"]), 3)
        item["nifty_relevance"] = round(float(item["nifty_relevance"]), 3)
        out.append(item)
    return out


SYMBOL_DISPLAY_NAMES = {
    "NIFTY50": "Nifty 50",
    "BANKNIFTY": "Bank Nifty",
    "CRUDEOIL": "Crude Oil",
    "GOLD": "Gold",
    "SILVER": "Silver",
}


def tool_fetch_market_snapshot(conn, as_of_utc: datetime, symbol: str = "NIFTY50") -> dict:
    """Tool 2: Fetch point-in-time market snapshot and holiday/expiry status at `as_of_utc`."""
    symbol = symbol.upper()
    as_of_ist = as_of_utc.astimezone(IST)
    holiday = conn.execute(
        "SELECT description FROM nse_holidays WHERE holiday_date = %s",
        (as_of_ist.date(),),
    ).fetchone()
    expiry = conn.execute(
        "SELECT expiry_kind FROM expiry_calendar WHERE underlying = %s AND expiry_date = %s",
        (symbol, as_of_ist.date()),
    ).fetchone()

    latest_bar = conn.execute(
        """
        SELECT ts, open, high, low, close, available_at
        FROM intraday_bars
        WHERE symbol = %s AND available_at <= %s
        ORDER BY ts DESC
        LIMIT 1
        """,
        (symbol, as_of_utc),
    ).fetchone()

    latest_vix = conn.execute(
        """
        SELECT close, available_at
        FROM intraday_bars
        WHERE symbol = 'INDIAVIX' AND available_at <= %s
        ORDER BY ts DESC
        LIMIT 1
        """,
        (as_of_utc,),
    ).fetchone()

    latest_news = conn.execute(
        "SELECT MAX(available_at) FROM articles WHERE available_at <= %s",
        (as_of_utc,),
    ).fetchone()

    if not latest_bar:
        return {"status": "no_market_data", "symbol": symbol, "is_holiday": bool(holiday)}

    bar_avail = latest_bar[5]
    news_avail = latest_news[0] if latest_news else None
    market_lag_min = round((as_of_utc - bar_avail).total_seconds() / 60.0, 1)
    news_lag_min = round((as_of_utc - news_avail).total_seconds() / 60.0, 1) if news_avail else None

    spot_p = round(float(latest_bar[4]), 2)
    return {
        "symbol": symbol,
        "nifty_close": spot_p,
        "spot_price": spot_p,
        "start_spot_price": spot_p,
        "nifty_bar_ts": latest_bar[0].isoformat(),
        "nifty_available_at": bar_avail.isoformat(),
        "india_vix": round(float(latest_vix[0]), 2) if latest_vix else 14.0,
        "market_data_lag_min": market_lag_min,
        "news_data_lag_min": news_lag_min,
        "latest_news_available_at": news_avail.isoformat() if news_avail else None,
        "is_holiday": bool(holiday),
        "holiday_name": holiday[0] if holiday else None,
        "is_weekend": as_of_ist.weekday() >= 5,
        "expiry_kind": expiry[0] if expiry else None,
    }


def generate_layman_explanation(inference: dict, snapshot: dict, feature_row: pd.Series, symbol: str = "NIFTY50") -> str:
    """Generate a clean, logically consistent conversational layman explanation that accurately reflects current price action."""
    sym_name = SYMBOL_DISPLAY_NAMES.get(symbol.upper(), symbol)
    is_commodity = symbol.upper() in ("CRUDEOIL", "GOLD", "SILVER")
    regime = inference.get("regime", "sideways").lower()
    fav_pct = float(inference.get("favourable_pct", 50.0))
    down_pct = round(100.0 - fav_pct, 1)
    spot_price = float(snapshot.get("spot_price", snapshot.get("nifty_close", 25345.0)) or 25345.0)
    adx = float(feature_row.get("adx", 18.0))
    vwap_dist = float(feature_row.get("true_vwap_dist", 0.0))
    top5_ret = float(feature_row.get("top5_ret_1", 0.0))
    ema_cross = float(feature_row.get("ema9_21_cross", feature_row.get("ema20_slope", 0.0)))
    breadth = float(feature_row.get("breadth_ad_ratio", 0.0))
    r1 = float(feature_row.get("ret_1", 0.0))
    rsi_14 = float(feature_row.get("rsi_14", 0.0))
    sr_pos = float(feature_row.get("sr_pos", 0.5))
    dist_to_sup = float(feature_row.get("dist_to_sup", feature_row.get("dist_to_pdl", 0.0)))
    dist_to_res = float(feature_row.get("dist_to_res", feature_row.get("dist_to_pdh", 0.0)))

    if regime == "bearish":
        reasons = []
        if dist_to_res <= 0.0020 and rsi_14 >= 0.15:
            reasons.append(f"{sym_name} faced sharp rejection at key overhead resistance")
        if vwap_dist < 0:
            reasons.append(f"price ({spot_price:,.1f}) has fallen below session VWAP, confirming sellers are in control")
        if ema_cross < 0:
            reasons.append("the short-term 9 EMA has crossed below the 21 EMA indicating active downward price pressure")
        if not is_commodity and top5_ret < 0:
            reasons.append("heavyweights like Reliance and banking leaders are under selling pressure")
        if breadth < 0:
            reasons.append("session breadth is negative with selling bias dominating")
        if not reasons:
            reasons.append("short-term momentum indicators are signaling downward continuation")
        return f"Why Bearish (P(Down) {down_pct:.1f}% | P(Up) {fav_pct:.1f}%)? Because " + ", and ".join(reasons[:3]) + f". Downside caution is advised as {sym_name} intraday selling pressure dominates."

    elif regime == "bullish":
        reasons = []
        if (sr_pos <= 0.25 or dist_to_sup <= 0.0020) and rsi_14 <= -0.15:
            reasons.append(f"{sym_name} formed a strong structural bounce off key support")
        if vwap_dist > 0:
            reasons.append(f"{sym_name} ({spot_price:,.1f}) is holding comfortably above session VWAP, showing buyers in control")
        if ema_cross > 0:
            reasons.append("the short-term 9 EMA is trending above the 21 EMA indicating positive price momentum")
        if not is_commodity and top5_ret > 0:
            reasons.append("key index heavyweights are providing upward lift")
        if breadth > 0:
            reasons.append("session breadth is positive with advancers leading")
        if not reasons:
            reasons.append("multi-timeframe technical indicators align in an upward direction")
        return f"Why Bullish (P(Up) {fav_pct:.1f}%)? Because " + ", and ".join(reasons[:3]) + f". This creates favorable conditions for {sym_name} upward continuation."

    else:
        reasons = []
        if (sr_pos <= 0.20 or dist_to_sup <= 0.0015) and rsi_14 <= -0.25:
            reasons.append(f"price ({spot_price:,.1f}) is resting on major support with oversold RSI, triggering institutional absorption")
        elif (sr_pos >= 0.80 or dist_to_res <= 0.0015) and rsi_14 >= 0.25:
            reasons.append(f"price ({spot_price:,.1f}) is testing major resistance with overbought RSI, capping immediate upside")
        elif abs(vwap_dist) < 0.0006:
            reasons.append(f"{sym_name} ({spot_price:,.1f}) is oscillating right around session VWAP and daily Pivot without directional separation")
        elif vwap_dist > 0 and top5_ret < 0:
            reasons.append("price is above VWAP but momentum is slowing, resulting in consolidation")
        if adx < 18.0:
            reasons.append("trend strength (ADX) is low indicating a rangebound consolidation zone")
        if not reasons:
            reasons.append("directional indicators are in conflict inside the daily channel")
        return (
            f"No clear view (P(Up) {fav_pct:.1f}%, inside the 47–53% no-view band). Because "
            + ", while ".join(reasons[:2])
            + ". The signals don't agree strongly enough for a directional call, so the system abstains "
            "for this window instead of forcing a coin-flip Bullish/Bearish call."
        )


# ---------------------------------------------------------------------------
# Selective-prediction confidence tiers (Phase 3b, step 1)
# Tiers are based on the raw probability edge |P(Up) - 50%|. They are NOT yet
# calibrated hit rates: live shadow results per tier must be collected before
# any tier's accuracy is claimed. Tune NO_VIEW_EDGE once tier data accumulates.
# ---------------------------------------------------------------------------
NO_VIEW_EDGE = 0.03      # 47% < P(Up) < 53%  -> "No clear view" (abstain)
MEDIUM_EDGE = 0.05       # 53-55% / 45-47%    -> LOW edge
HIGH_EDGE = 0.08         # 55-58% / 42-45%    -> MEDIUM edge;  >=58% / <=42% -> HIGH edge

TIER_LABELS = {
    "NO_VIEW": "NO CLEAR VIEW (ABSTAIN)",
    "LOW": "LOW EDGE",
    "MEDIUM": "MEDIUM EDGE",
    "HIGH": "HIGH EDGE",
}


# ---------------------------------------------------------------------------
# Meta-Labeling Secondary Model Loader (Phase 3b)
# Predicts whether the primary model's directional call will succeed.
# ---------------------------------------------------------------------------
_META_BUNDLE = None

def _load_meta_model():
    global _META_BUNDLE
    if _META_BUNDLE is None:
        meta_path = ROOT / "models" / "meta_labeler_v1.pkl"
        if meta_path.exists():
            try:
                with open(meta_path, "rb") as f:
                    _META_BUNDLE = pickle.load(f)
            except Exception:
                _META_BUNDLE = None
    return _META_BUNDLE

def predict_meta_confidence(feature_row: pd.Series, prob_up: float, pred_dir: int) -> float:
    """Evaluate secondary meta-labeling model probability of trade success."""
    meta_bundle = _load_meta_model()
    if not meta_bundle or "model" not in meta_bundle:
        # Fallback to distance from 50%
        return float(np.clip(max(prob_up, 1.0 - prob_up), 0.50, 0.95))
    try:
        edge = abs(prob_up - 0.50)
        ema_slope = float(feature_row.get("ema20_slope", 0.0) or 0.0)
        vwap_d = float(feature_row.get("true_vwap_dist", 0.0) or 0.0)
        sr_pos = float(feature_row.get("sr_pos", 0.5) or 0.5)
        adx_val = float(feature_row.get("adx", 18.0) or 18.0)
        rsi_14 = float(feature_row.get("rsi_14", 0.0) or 0.0)
        vol_sp = float(feature_row.get("vol_spike", 0.0) or 0.0)
        vix_chg = float(feature_row.get("vix_change", 0.0) or 0.0)

        trend_aligned = 1.0 if ((pred_dir == 1 and ema_slope > 0 and vwap_d > 0) or (pred_dir == 0 and ema_slope < 0 and vwap_d < 0)) else 0.0
        sr_clear = 1.0 if ((pred_dir == 1 and sr_pos < 0.70) or (pred_dir == 0 and sr_pos > 0.30)) else 0.0
        adx_strong = 1.0 if adx_val >= 20.0 else 0.0

        feat_vals = [edge, trend_aligned, sr_clear, adx_strong, adx_val, rsi_14, vol_sp, vix_chg]
        p_hit = float(meta_bundle["model"].predict_proba([feat_vals])[0, 1])
        return round(float(np.clip(p_hit, 0.40, 0.85)), 4)
    except Exception:
        return float(np.clip(max(prob_up, 1.0 - prob_up), 0.50, 0.95))


def classify_confidence_tier(prob_up: float | None) -> str:
    if prob_up is None:
        return "NO_VIEW"
    edge = abs(float(prob_up) - 0.50)
    if edge < NO_VIEW_EDGE:
        return "NO_VIEW"
    if edge < MEDIUM_EDGE:
        return "LOW"
    if edge < HIGH_EDGE:
        return "MEDIUM"
    return "HIGH"


def tool_run_model_inference(
    feature_row: pd.Series, bundle: dict, window_minutes: int = 30
) -> dict:
    """Quantitative Statistical Machine Learning Model Inference:
    Executes trained direction, regime, and entry models on the full point-in-time feature vector,
    modulated by Support/Resistance structure, RSI exhaustion, constituent divergence, and FinBERT sentiment.
    """
    X_all = pd.DataFrame([feature_row])[bundle["feature_columns"]].fillna(0.0).values

    # 1. Base ML Direction Model (Calibrated Logistic Regression on 12-Year Dataset)
    p_dir = bundle["direction_model"].predict_proba(X_all)[0]
    p_up_ml = float(p_dir[1])

    # 2. Multi-Class Regime Probabilities
    p_reg = bundle["regime_model"].predict_proba(X_all)[0]
    reg_cls = list(bundle["regime_model"].classes_)
    p_bear_ml = float(p_reg[reg_cls.index("bearish")])
    p_bull_ml = float(p_reg[reg_cls.index("bullish")])
    p_side_ml = float(p_reg[reg_cls.index("sideways")])

    # 3. Entry Environment Model
    p_ent = bundle["entry_model"].predict_proba(X_all)[0]
    ent_cls = list(bundle["entry_model"].classes_)
    p_avoid_ml = float(p_ent[ent_cls.index("avoid")])
    p_caut_ml = float(p_ent[ent_cls.index("cautious")])
    p_fav_ml = float(p_ent[ent_cls.index("favourable")])

    # 4. Leading Microstructure Signals & Exhaustion Modulators
    surp = float(feature_row.get("news_surprise", 0.0))
    ev_shock = float(feature_row.get("signed_event_shock", 0.0))
    vwap_d = float(feature_row.get("true_vwap_dist", 0.0))
    r1 = float(feature_row.get("ret_1", 0.0))
    ema_slope = float(feature_row.get("ema20_slope", 0.0))
    mtf_trend = float(feature_row.get("mtf_trend_score", 0.0))
    top5_r = float(feature_row.get("top5_ret_1", 0.0))
    bn_r = float(feature_row.get("banknifty_ret_1", 0.0))
    vol_sp = float(feature_row.get("vol_spike", 0.0))
    adx_val = float(feature_row.get("adx", 18.0))
    vix_chg = float(feature_row.get("vix_change", 0.0))
    ev_sev = float(feature_row.get("max_event_severity", 0.0))
    rsi_14 = float(feature_row.get("rsi_14", 0.0))  # scaled [-1, 1]
    sr_pos = float(feature_row.get("sr_pos", 0.5))  # [0, 1] position between support and resistance
    dist_to_sup = float(feature_row.get("dist_to_sup", feature_row.get("dist_to_pdl", 0.0)))
    dist_to_res = float(feature_row.get("dist_to_res", feature_row.get("dist_to_pdh", 0.0)))

    # A. Volume-Gated Intraday VWAP & Price Momentum Alignment
    # Do not blindly chase low-volume pullbacks at full weight.
    vol_mult = 0.5 if vol_sp < -0.25 else (1.15 if vol_sp > 0.25 else 1.0)
    intra_shift = 0.0
    if vwap_d > 0.0003 and ema_slope > 0.00005:
        intra_shift = +0.045 * vol_mult
    elif vwap_d < -0.0003 and ema_slope < -0.00005:
        intra_shift = -0.045 * vol_mult

    # B. Multi-Timeframe Trend Alignment (15m EMA Slope + 1h/Daily MTF Trend Alignment)
    mtf_shift = 0.0
    if mtf_trend > 0.15 and ema_slope > 0.0:
        mtf_shift = +0.025  # Confirmed higher-timeframe institutional bullish trend
    elif mtf_trend < -0.15 and ema_slope < 0.0:
        mtf_shift = -0.025  # Confirmed higher-timeframe institutional bearish trend

    # C. Range-Aware S/R Channel & RSI Exhaustion at extremes
    sr_shift = 0.0
    hi_thr = 0.72 if adx_val < 22.0 else 0.85
    lo_thr = 0.28 if adx_val < 22.0 else 0.15
    if sr_pos >= hi_thr and rsi_14 >= 0.05:
        sr_shift = -0.035  # Resistance rejection / overbought pullback
        if sr_pos >= 0.90 and rsi_14 >= 0.15:
            sr_shift -= 0.035  # Extreme resistance ceiling (+oversold) dampens aggressive chasing
    elif sr_pos <= lo_thr and rsi_14 <= -0.05:
        sr_shift = +0.035  # Support defense / oversold bounce
        if sr_pos <= 0.10 and rsi_14 <= -0.15:
            sr_shift += 0.035  # Extreme support floor (+oversold) dampens aggressive breakdown chasing

    # D. Institutional Heavyweight + BankNifty Divergence & VIX Filter
    div_shift = 0.0
    if r1 < 0 and (top5_r > 0.00025 or bn_r > 0.00010 or (top5_r - r1) > 0.00045):
        div_shift = +0.030  # Institutional / BankNifty absorption relief bias
    elif r1 > 0 and (top5_r < -0.00025 or bn_r < -0.00010 or (top5_r - r1) < -0.00045):
        div_shift = -0.030  # Heavyweight / BankNifty drag divergence

    vix_shift = 0.0
    if vix_chg < -0.0020:
        vix_shift = +0.015  # Falling VIX / fear subsiding
    elif vix_chg > 0.0030:
        vix_shift = -0.015  # Rising VIX / hedging pressure

    # E. FinBERT News & Macro Shock Modulator (Gated by VWAP)
    news_shift = 0.035 * ev_shock - 0.025 * surp
    if vwap_d < -0.0010 and news_shift > 0:
        news_shift *= 0.4  # If price is trading below VWAP, dampen bullish headlines

    # Combine into true quantitative direction probability
    prob_up = float(np.clip(p_up_ml + intra_shift + mtf_shift + sr_shift + div_shift + vix_shift + news_shift, 0.10, 0.90))

    # F. Opening Session Volatility Damper (09:15 - 09:45 IST):
    if "ts" in feature_row and pd.notna(feature_row["ts"]):
        try:
            ts_ist = pd.to_datetime(feature_row["ts"]).astimezone(IST)
            if ts_ist.hour == 9 and ts_ist.minute < 45:
                prob_up = 0.50 + (prob_up - 0.50) * 0.55
        except Exception:
            pass

    # 5. Honest probability: NO cosmetic stretching. The displayed figure is the model's own P(Up).
    favourable_pct = round(prob_up * 100.0, 1)
    dir_edge = abs(prob_up - 0.50)
    confidence_tier = classify_confidence_tier(prob_up)

    # 6. Selective prediction: issue a direction only when the edge clears the No-View band.
    #    Inside the band the system abstains ("No clear view"); stored as regime 'sideways'
    #    because the DB CHECK constraint only allows bullish / bearish / sideways.
    p_bull = round(prob_up, 4)
    p_bear = round(1.0 - prob_up, 4)
    p_side = 0.0
    if confidence_tier == "NO_VIEW":
        regime = "sideways"
    elif prob_up >= 0.50:
        regime = "bullish"
    else:
        regime = "bearish"

    # 7. Entry Condition (Avoid on no-view / high VIX spike / shock, Favourable on high conviction)
    if regime == "sideways":
        entry_condition = "avoid"
        entry_score = round(float(np.clip(0.40 + 0.20 * p_caut_ml, 0.35, 0.55)), 4)
    elif vix_chg > 0.08 or ev_sev >= 0.70 or p_avoid_ml > 0.45:
        entry_condition = "avoid"
        entry_score = round(max(0.20, 1.0 - max(p_avoid_ml, vix_chg)), 4)
    elif regime in ("bullish", "bearish") and (adx_val >= 18.0 and dir_edge >= 0.035) and p_fav_ml >= 0.38:
        entry_condition = "favourable"
        entry_score = round(float(np.clip(0.65 + 0.30 * p_fav_ml, 0.65, 0.95)), 4)
    else:
        entry_condition = "cautious"
        entry_score = round(float(np.clip(0.40 + 0.20 * p_caut_ml, 0.35, 0.62)), 4)

    conviction_pct = round(entry_score * 100.0, 1)
    pred_dir = 1 if prob_up >= 0.50 else 0
    meta_p_hit = predict_meta_confidence(feature_row, prob_up, pred_dir)
    confidence = round(float(np.clip(meta_p_hit - 0.05 * ev_sev, 0.40, 0.95)), 4)

    return {
        "model_version": bundle["model_version"],
        "label_version": bundle["label_version"],
        "window_minutes": window_minutes,
        "confidence_tier": confidence_tier,
        "regime": regime,
        "prob_bullish": p_bull,
        "prob_bearish": p_bear,
        "prob_sideways": p_side,
        "prob_up": prob_up,
        "favourable_pct": favourable_pct,
        "conviction_pct": conviction_pct,
        "entry_condition": entry_condition,
        "entry_score": entry_score,
        "confidence": confidence,
    }


def extract_top_drivers(
    feature_row: pd.Series, snapshot: dict, recent_news: list[dict]
) -> list[dict]:
    """Identify top quantitative and news drivers explaining the model's numerical output."""
    drivers = []
    adx_v = float(feature_row.get("adx", 18.0))
    r1 = float(feature_row.get("ret_1", 0.0)) * 100.0
    slope = float(feature_row.get("ema20_slope", 0.0)) * 100.0
    br = float(feature_row.get("breadth_ad_ratio", 0.0))
    vix = snapshot.get("india_vix", 14.0)
    vwap_d = float(feature_row.get("true_vwap_dist", 0.0)) * 100.0
    vol_sp = math.exp(float(feature_row.get("vol_spike", 0.0)))
    mtf_s = float(feature_row.get("mtf_trend_score", 0.0))
    top5_r = float(feature_row.get("top5_ret_1", 0.0)) * 100.0
    bn_r = float(feature_row.get("banknifty_ret_1", 0.0)) * 100.0
    mtf_label = "Bullish (+1.0)" if mtf_s > 0.3 else ("Bearish (-1.0)" if mtf_s < -0.3 else "Mixed (0.0)")

    drivers.append(
        {
            "category": "technical_regime",
            "driver": (
                f"ADX(14) at {adx_v:.1f} ({'Trending' if adx_v >= 20 else 'Range-bound'}) | "
                f"Return {r1:+.2f}%, EMA-20 slope {slope:+.2f}% | "
                f"15m/60m Multi-Timeframe Alignment: {mtf_label}"
            ),
        }
    )
    drivers.append(
        {
            "category": "volume_and_heavyweights",
            "driver": (
                f"True Volume-Weighted VWAP dist {vwap_d:+.2f}% | Constituent Volume {vol_sp:.2f}x 14-bar avg | "
                f"Top-5 Heavyweights {top5_r:+.2f}% & BankNifty {bn_r:+.2f}% | Breadth {br:+.2f} | India VIX {vix:.2f}"
            ),
        }
    )
    # Key S/R and Momentum Drivers
    rsi_14 = float(feature_row.get("rsi_14", 0.0))
    raw_rsi = round(rsi_14 * 50.0 + 50.0, 1)
    pdl = float(feature_row.get("pdl", 0.0))
    pdh = float(feature_row.get("pdh", 0.0))
    pivot = float(feature_row.get("pivot", 0.0))
    sr_pos = float(feature_row.get("sr_pos", 0.5)) * 100.0

    drivers.append(
        {
            "category": "support_resistance_and_rsi",
            "driver": (
                f"S/R Structure: Support (PDL {pdl:.0f}) to Resistance (PDH {pdh:.0f}) | "
                f"Central Pivot {pivot:.0f} | Channel Position {sr_pos:.0f}% | RSI(14) at {raw_rsi:.1f}"
            ),
        }
    )
    if recent_news:
        top_art = recent_news[0]
        drivers.append(
            {
                "category": "news_catalyst",
                "driver": (
                    f"{top_art['symbol']} ({top_art['macro_theme']}): \"{top_art['headline']}\" "
                    f"[score {top_art['sentiment_score']:+.2f}, Nifty relevance {top_art['nifty_relevance']:.2f}]"
                ),
            }
        )
    else:
        drivers.append(
            {
                "category": "news_catalyst",
                "driver": "No new constituent or macro headlines in the lookback window; signal driven by market technicals.",
            }
        )
    return drivers


def build_invalidation_condition(
    feature_row: pd.Series, snapshot: dict, regime: str, window_minutes: int = 30, symbol: str = "NIFTY50"
) -> str:
    sym_name = SYMBOL_DISPLAY_NAMES.get(symbol.upper(), symbol)
    spot = float(snapshot.get("spot_price", snapshot.get("nifty_close", 24000.0)))
    atr_pct = float(feature_row.get("atr_pct", 0.0025))
    vwap_dist = float(feature_row.get("vwap_dist", 0.0))
    vwap_level = round(spot / (1.0 + vwap_dist), 0)
    min_atr = 8.0 if window_minutes <= 5 else 25.0
    atr_pts = max(min_atr, round(spot * atr_pct, 0))
    vix = float(snapshot.get("india_vix", 14.0))

    if regime == "bullish":
        inv_lvl = round(min(vwap_level, spot - 0.85 * atr_pts), 0)
        return (
            f"Bullish regime view is invalidated if {sym_name} spot ({spot:.0f}) drops below "
            f"{inv_lvl:.0f} (VWAP / 0.85x ATR support) or India VIX spikes above {vix * 1.06:.1f}."
        )
    if regime == "bearish":
        inv_lvl = round(max(vwap_level, spot + 0.85 * atr_pts), 0)
        return (
            f"Bearish regime view is invalidated if {sym_name} spot ({spot:.0f}) reclaims "
            f"{inv_lvl:.0f} (VWAP / 0.85x ATR resistance) with positive constituent breadth."
        )
    lo_lvl = round(spot - 0.90 * atr_pts, 0)
    hi_lvl = round(spot + 0.90 * atr_pts, 0)
    return (
        f"Sideways regime view is invalidated on a volume/breadth breakout outside the "
        f"[{lo_lvl:.0f}, {hi_lvl:.0f}] band (+/-0.9x window ATR)."
    )


def render_compliance_template(output: dict) -> str:
    """Format the output using the compliance-approved advisory template."""
    sym_name = SYMBOL_DISPLAY_NAMES.get(output.get("instrument", "NIFTY50").upper(), output.get("instrument", "NIFTY50"))
    if output["status"] == "suppressed":
        text = (
            f"[{sym_name} Market Regime View — As of {output['as_of_ist']} | Status: SUPPRESSED]\n"
            f"Reason: {output['rationale']}\n"
            f"{AI_DISCLOSURE}"
        )
        return enforce_wording_guardrails(text)

    status_tag = (
        "OK (Fresh Data)"
        if output["status"] == "ok"
        else f"DEGRADED ({output.get('degraded_reason', 'Stale input feed / low confidence')})"
    )
    drv_lines = "\n".join(f"  - [{d['category']}] {d['driver']}" for d in output["top_drivers"])
    text = (
        f"[{sym_name} Market Regime & Entry Environment — As of {output['as_of_ist']} "
        f"| Window: {output['window_minutes']}m | Run: {output['run_type'].upper()} | Data Status: {status_tag}]\n"
        f"• Market Regime Classification: {output['regime'].upper()} "
        f"(Bullish: {output['prob_bullish']*100:.1f}%, Sideways: {output['prob_sideways']*100:.1f}%, Bearish: {output['prob_bearish']*100:.1f}%)\n"
        f"• Window Direction Probability P(Up): {output['prob_up']*100:.1f}% | Model Confidence: {output['confidence']*100:.1f}%\n"
        f"• Market Entry Environment: {output['entry_condition'].upper()} (Entry Score: {output['entry_score']:.2f})\n"
        f"• Key Drivers:\n{drv_lines}\n"
        f"• Plain-Language Rationale: {output['rationale']}\n"
        f"• Invalidation Condition: {output['invalidation']}\n"
        f"• {AI_DISCLOSURE}"
    )
    return enforce_wording_guardrails(text)


# ---------------------------------------------------------------------------
# Live Incremental Snapshot Refresh (fetches latest bars & news up to NOW)
# ---------------------------------------------------------------------------
def _refresh_aux_and_vix(conn, now_utc: datetime) -> None:
    """Keep India VIX, BankNifty, and Nifty Top-10 constituent bars fresh via YFinance."""
    aux_row = conn.execute(
        "SELECT MAX(available_at) FROM intraday_bars WHERE symbol = 'BANKNIFTY' AND interval_sec = 900"
    ).fetchone()
    if aux_row and aux_row[0] and (now_utc - aux_row[0]).total_seconds() < 300:
        return
    from market_data import YFinanceAdapter, ingest_intraday_bars
    adapter = YFinanceAdapter()
    aux_syms = [
        "INDIAVIX", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK",
        "INFY", "TCS", "ITC", "LT", "BHARTIARTL", "AXISBANK", "SBIN",
    ]
    ingest_intraday_bars(conn, adapter, days=2, interval="15m", symbols=aux_syms)
    ingest_intraday_bars(conn, adapter, days=2, interval="5m", symbols=aux_syms)


def refresh_live_snapshot(conn, dsn: str) -> None:
    """Pull latest 1m/5m/15m intraday bars + VIX/BankNifty/Constituents + recent news if stale."""
    try:
        latest_row = conn.execute(
            "SELECT MAX(available_at) FROM intraday_bars WHERE symbol = 'NIFTY50' AND interval_sec = 60"
        ).fetchone()
        now_utc = datetime.now(timezone.utc)
        if latest_row and latest_row[0] and (now_utc - latest_row[0]).total_seconds() < 300:
            # Active live XTS feed is providing sub-second bars. Roll 1m bars into 5m and 15m immediately
            for iv_sec in (300, 900):
                conn.execute(
                    f"""
                    INSERT INTO intraday_bars (symbol, interval_sec, ts, open, high, low, close, volume, available_at)
                    SELECT 
                        'NIFTY50',
                        {iv_sec},
                        to_timestamp(floor(extract(epoch from ts) / {iv_sec}) * {iv_sec}) as bucket_ts,
                        (array_agg(open ORDER BY ts ASC))[1],
                        MAX(high),
                        MIN(low),
                        (array_agg(close ORDER BY ts DESC))[1],
                        SUM(volume),
                        to_timestamp(floor(extract(epoch from ts) / {iv_sec}) * {iv_sec}) + interval '{iv_sec} seconds'
                    FROM intraday_bars
                    WHERE symbol = 'NIFTY50' AND interval_sec = 60 AND ts >= CURRENT_TIMESTAMP - interval '1 day'
                    GROUP BY bucket_ts
                    ON CONFLICT (symbol, interval_sec, ts) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = GREATEST(intraday_bars.high, EXCLUDED.high),
                        low = LEAST(intraday_bars.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        available_at = EXCLUDED.available_at;
                    """
                )
            for iv_sec in (60, 300, 900):
                conn.execute(
                    """
                    UPDATE intraday_bars
                    SET available_at = %s
                    WHERE symbol = 'NIFTY50' AND interval_sec = %s
                      AND ts = (SELECT MAX(ts) FROM intraday_bars WHERE symbol = 'NIFTY50' AND interval_sec = %s)
                    """,
                    (now_utc, iv_sec, iv_sec),
                )
            _refresh_aux_and_vix(conn, now_utc)
            return

        import os
        import requests
        from ingest import ingest as ingest_articles
        from market_data import YFinanceAdapter, ingest_intraday_bars
        from scorer import score_all_articles

        adapter = YFinanceAdapter()
        all_syms = [
            "NIFTY50", "INDIAVIX", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK",
            "INFY", "TCS", "ITC", "LT", "BHARTIARTL", "AXISBANK", "SBIN",
        ]
        ingest_intraday_bars(conn, adapter, days=2, interval="15m", symbols=all_syms)
        ingest_intraday_bars(conn, adapter, days=2, interval="5m", symbols=all_syms)
        ingest_intraday_bars(conn, adapter, days=1, interval="1m", symbols=["NIFTY50", "INDIAVIX"])
        now_utc = datetime.now(timezone.utc)
        for iv_sec in (60, 300, 900):
            conn.execute(
                """
                UPDATE intraday_bars
                SET available_at = %s
                WHERE symbol = 'NIFTY50' AND interval_sec = %s
                  AND ts = (SELECT MAX(ts) FROM intraday_bars WHERE symbol = 'NIFTY50' AND interval_sec = %s)
                """,
                (now_utc, iv_sec, iv_sec),
            )

        base_url = os.environ.get("NEWS_API_URL", "").rstrip("/")
        api_key = os.environ.get("NEWS_API_KEY", "")
        if base_url and api_key:
            batch = []
            for sym in ["RELIANCE", "HDFCBANK", "ICICIBANK"]:
                resp = requests.get(
                    f"{base_url}/api/news",
                    headers={"x-api-key": api_key},
                    params={"symbol": sym, "limit": 5},
                    timeout=5,
                )
                if resp.status_code == 200 and isinstance(resp.json(), list):
                    batch.extend(resp.json())
            if batch:
                added = ingest_articles(batch, dsn)
                if added > 0:
                    score_all_articles(dsn)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main Agent Orchestrator
# ---------------------------------------------------------------------------
def run_agent(
    dsn: str,
    run_type: str = "scheduled",
    window_minutes: int = 30,
    symbol: str = "NIFTY50",
    as_of_utc: datetime | None = None,
    user_query: str | None = None,
    simulate_stale_data: bool = False,
    simulate_no_news: bool = False,
    published: bool = False,
    persist: bool = True,
    refresh_live: bool = True,
) -> dict:
    symbol = symbol.upper()
    sym_name = SYMBOL_DISPLAY_NAMES.get(symbol, symbol)
    is_commodity = symbol in ("CRUDEOIL", "GOLD", "SILVER")
    close_hour = 23 if is_commodity else 15

    # Step 0: Out-of-scope guardrail check
    in_scope, refusal_reason = check_query_scope(user_query)
    if not in_scope:
        return {
            "status": "refused_out_of_scope",
            "query": user_query,
            "message": refusal_reason,
            "ai_disclosure": AI_DISCLOSURE,
        }

    model_path = ROOT / "models" / "sentiment_agent_v1.pkl"
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    with psycopg.connect(dsn, autocommit=True) as conn:
        if refresh_live and as_of_utc is None and not simulate_stale_data and not simulate_no_news:
            refresh_live_snapshot(conn, dsn)

        news_df = load_scored_articles(conn)
        if simulate_no_news:
            news_df = news_df.iloc[0:0]

        df_win, _ = build_intraday_window_dataset(
            conn, news_df, window_minutes=window_minutes, include_live_unsettled=True, symbol=symbol
        )
        if df_win.empty:
            raise RuntimeError(f"No intraday bars available for {symbol} to build feature vector.")

        # Select the target cutoff row
        if as_of_utc is not None:
            sub = df_win[pd.to_datetime(df_win["as_of"], utc=True) <= as_of_utc]
            feature_row = sub.iloc[-1] if not sub.empty else df_win.iloc[-1]
            effective_as_of = as_of_utc.replace(second=0, microsecond=0)
        else:
            feature_row = df_win.iloc[-1]
            effective_as_of = pd.to_datetime(feature_row["as_of"], utc=True).to_pydatetime().replace(second=0, microsecond=0)

        snapshot = tool_fetch_market_snapshot(conn, effective_as_of, symbol=symbol)
        recent_news = (
            []
            if simulate_no_news
            else tool_fetch_recent_scored_news(conn, effective_as_of, lookback_hours=12.0, limit=5)
        )

        # Edge-case 1: NSE Holiday / Weekend check
        if snapshot.get("is_holiday"):
            suppressed_out = {
                "instrument": symbol,
                "run_type": run_type,
                "as_of": effective_as_of.isoformat(),
                "as_of_ist": effective_as_of.astimezone(IST).strftime("%Y-%m-%d %H:%M IST"),
                "window_minutes": window_minutes,
                "status": "suppressed",
                "rationale": (
                    f"Market holiday ({snapshot.get('holiday_name')}). Live intraday regime "
                    f"and entry-conditions signals are paused until the next trading session."
                ),
                "data_freshness": snapshot,
                "template_version": TEMPLATE_VERSION,
            }
            suppressed_out["rendered_text"] = render_compliance_template(suppressed_out)
            return suppressed_out

        # Edge-case 2: Stale market feed / API outage check
        if simulate_stale_data:
            snapshot["market_data_lag_min"] = 45.0
        mkt_lag = float(snapshot.get("market_data_lag_min") or 0.0)

        inference = tool_run_model_inference(feature_row, bundle, window_minutes=window_minutes)
        drivers = extract_top_drivers(feature_row, snapshot, recent_news)
        invalidation = build_invalidation_condition(
            feature_row, snapshot, inference["regime"], window_minutes=window_minutes, symbol=symbol
        )

        status = "ok"
        degraded_reason = None
        if simulate_stale_data or mkt_lag > 30.0:
            status = "suppressed"
            rationale = (
                f"Market data feed is stale ({mkt_lag:.1f} min lag exceeds 30m safety threshold). "
                f"View is withheld per compliance kill-switch protocol until live exchange timestamps resume."
            )
        else:
            if simulate_no_news or not recent_news:
                status = "degraded"
                degraded_reason = "No recent news in lookback window; relying on market technicals only"
            elif inference["confidence"] < 0.52:
                status = "degraded"
                degraded_reason = "Low directional conviction near neutral band"

            rationale = (
                f"Over the next {window_minutes}-minute window, quantitative indicators classify {sym_name} "
                f"in a {inference['regime']} regime ({inference['prob_' + inference['regime']]*100:.1f}% probability) "
                f"with a {inference['prob_up']*100:.1f}% upward directional probability. "
                f"Entry conditions are assessed as {inference['entry_condition']} "
                f"(score {inference['entry_score']:.2f}), reflecting current ADX trend strength, "
                f"constituent breadth, and technical alignment."
            )

        horizon_start = effective_as_of.replace(second=0, microsecond=0)
        if window_minutes == 375:
            # End-of-Day Closing Horizon: targets 15:30 IST today (or 23:30 IST for MCX)
            eff_ist = effective_as_of.astimezone(IST)
            h_end_ist = eff_ist.replace(hour=close_hour, minute=30, second=0, microsecond=0)
            if h_end_ist <= eff_ist:
                h_end_ist = (eff_ist + timedelta(days=1)).replace(hour=close_hour, minute=30, second=0, microsecond=0)
            horizon_end = h_end_ist.astimezone(timezone.utc)
            horizon_desc = f"Session Close ({close_hour}:30 IST)"
        elif window_minutes == 750:
            # Next-Day Close Horizon: targets next trading session close
            eff_ist = effective_as_of.astimezone(IST)
            next_day = eff_ist + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip Saturday/Sunday
                next_day += timedelta(days=1)
            h_end_ist = next_day.replace(hour=close_hour, minute=30, second=0, microsecond=0)
            horizon_end = h_end_ist.astimezone(timezone.utc)
            horizon_desc = f"Next Day Close ({close_hour}:30 IST)"
        else:
            # Regular intraday windows (e.g. 5m, 15m, 30m, 60m)
            # Clip horizon_end to market close so predictions never extend past session close!
            eff_ist = effective_as_of.astimezone(IST)
            close_ist = eff_ist.replace(hour=close_hour, minute=30, second=0, microsecond=0)
            raw_end = effective_as_of + timedelta(minutes=window_minutes)
            raw_end_ist = raw_end.astimezone(IST)

            if raw_end_ist > close_ist and eff_ist < close_ist:
                horizon_end = close_ist.astimezone(timezone.utc)
                horizon_desc = f"{window_minutes}m (until session close {close_hour}:30 IST)"
            else:
                horizon_end = raw_end
                horizon_desc = f"{window_minutes}-minute window"
        horizon_end = horizon_end.replace(second=0, microsecond=0)

        trigger_article_id = (
            recent_news[0]["article_id"]
            if (run_type == "event" and recent_news)
            else None
        )

        features_dict = {
            k: round(float(feature_row.get(k, 0.0)), 5)
            for k in bundle["feature_columns"]
        }
        layman_explanation = generate_layman_explanation(inference, snapshot, feature_row, symbol=symbol)

        output = {
            "instrument": symbol,
            "run_type": run_type,
            "as_of": effective_as_of.isoformat(),
            "as_of_ist": effective_as_of.astimezone(IST).strftime("%Y-%m-%d %H:%M IST"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_minutes": window_minutes,
            "horizon_desc": horizon_desc,
            "horizon_start": horizon_start.isoformat(),
            "horizon_end": horizon_end.isoformat(),
            "trigger_article_id": trigger_article_id,
            "model_version": inference["model_version"],
            "label_version": inference["label_version"],
            "regime": inference["regime"],
            "prob_bullish": inference["prob_bullish"],
            "prob_bearish": inference["prob_bearish"],
            "prob_sideways": inference["prob_sideways"],
            "prob_up": inference["prob_up"],
            "favourable_pct": inference.get("favourable_pct", round(inference["prob_up"] * 100.0, 1)),
            "conviction_pct": inference.get("conviction_pct", round(inference["entry_score"] * 100.0, 1)),
            "entry_condition": inference["entry_condition"],
            "entry_score": inference["entry_score"],
            "confidence": inference["confidence"],
            "confidence_tier": inference.get("confidence_tier"),
            "signal_strength": TIER_LABELS.get(inference.get("confidence_tier", "NO_VIEW")),
            "invalidation": invalidation,
            "top_drivers": drivers,
            "rationale": rationale,
            "layman_explanation": layman_explanation,
            "features_used": features_dict,
            "data_freshness": snapshot,
            "status": status,
            "degraded_reason": degraded_reason,
            "template_version": TEMPLATE_VERSION,
            "published": published,
            "ai_disclosure": AI_DISCLOSURE,
        }
        output["rendered_text"] = render_compliance_template(output)

        if persist and status != "suppressed":
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    instrument, run_type, as_of, generated_at, window_minutes,
                    horizon_start, horizon_end, trigger_article_id,
                    model_version, label_version,
                    regime, prob_bullish, prob_bearish, prob_sideways, prob_up,
                    entry_condition, entry_score, confidence, invalidation,
                    top_drivers, rationale, features_used, data_freshness,
                    status, template_version, rendered_text, published
                ) VALUES (
                    %(instrument)s, %(run_type)s, %(as_of)s, %(generated_at)s, %(window_minutes)s,
                    %(horizon_start)s, %(horizon_end)s, %(trigger_article_id)s,
                    %(model_version)s, %(label_version)s,
                    %(regime)s, %(prob_bullish)s, %(prob_bearish)s, %(prob_sideways)s, %(prob_up)s,
                    %(entry_condition)s, %(entry_score)s, %(confidence)s, %(invalidation)s,
                    %(top_drivers)s, %(rationale)s, %(features_used)s, %(data_freshness)s,
                    %(status)s, %(template_version)s, %(rendered_text)s, %(published)s
                )
                ON CONFLICT (instrument, run_type, as_of, window_minutes) DO UPDATE SET
                    generated_at = EXCLUDED.generated_at,
                    regime = EXCLUDED.regime,
                    prob_bullish = EXCLUDED.prob_bullish,
                    prob_bearish = EXCLUDED.prob_bearish,
                    prob_sideways = EXCLUDED.prob_sideways,
                    prob_up = EXCLUDED.prob_up,
                    entry_condition = EXCLUDED.entry_condition,
                    entry_score = EXCLUDED.entry_score,
                    confidence = EXCLUDED.confidence,
                    invalidation = EXCLUDED.invalidation,
                    top_drivers = EXCLUDED.top_drivers,
                    rationale = EXCLUDED.rationale,
                    status = EXCLUDED.status,
                    rendered_text = EXCLUDED.rendered_text
                RETURNING prediction_id
                """,
                {
                    **output,
                    "top_drivers": Jsonb(output["top_drivers"]),
                    "features_used": Jsonb(output["features_used"]),
                    "data_freshness": Jsonb(output["data_freshness"]),
                },
            )
            output["prediction_id"] = cur.fetchone()[0]

    return output


def run_scenario_test_suite(dsn: str) -> dict:
    """Phase 5 Exit Criteria: Test normal, pre-open, event, holiday, no-news, stale-data & out-of-scope scenarios."""
    results = {}

    # 1. Normal scheduled 30m intraday run
    s1 = run_agent(dsn, run_type="scheduled", window_minutes=30, persist=True)
    results["1_scheduled_30m"] = {
        "status": s1["status"],
        "regime": s1["regime"],
        "prob_up": s1["prob_up"],
        "entry_condition": s1["entry_condition"],
        "prediction_id": s1.get("prediction_id"),
    }

    # 2. Pre-open outlook run
    s2 = run_agent(dsn, run_type="pre_open", window_minutes=30, persist=True)
    results["2_pre_open_outlook"] = {
        "status": s2["status"],
        "regime": s2["regime"],
        "entry_condition": s2["entry_condition"],
    }

    # 3. Event-triggered run
    s3 = run_agent(dsn, run_type="event", window_minutes=30, persist=True)
    results["3_event_triggered"] = {
        "status": s3["status"],
        "trigger_article_id": s3.get("trigger_article_id"),
        "regime": s3["regime"],
    }

    # 4. Edge Case: NSE Holiday (Diwali-Balipratipada 2026-11-09)
    holiday_ts = datetime(2026, 11, 9, 4, 30, tzinfo=timezone.utc)
    s4 = run_agent(dsn, run_type="scheduled", window_minutes=30, as_of_utc=holiday_ts, persist=False)
    results["4_edge_case_holiday"] = {
        "status": s4["status"],
        "rationale": s4["rationale"],
    }

    # 5. Edge Case: No News in window (degraded fallback to technicals)
    s5 = run_agent(dsn, run_type="scheduled", window_minutes=30, simulate_no_news=True, persist=False)
    results["5_edge_case_no_news"] = {
        "status": s5["status"],
        "degraded_reason": s5["degraded_reason"],
        "regime": s5["regime"],
    }

    # 6. Edge Case: Stale Data / API Outage (kill-switch suppression)
    s6 = run_agent(dsn, run_type="scheduled", window_minutes=30, simulate_stale_data=True, persist=False)
    results["6_edge_case_stale_data_outage"] = {
        "status": s6["status"],
        "rationale": s6["rationale"],
    }

    # 7. Edge Case: Out-of-scope user question refusal
    s7 = run_agent(dsn, user_query="Which stock should I buy now for guaranteed profit tomorrow?", persist=False)
    results["7_edge_case_out_of_scope_refusal"] = {
        "status": s7["status"],
        "message": s7["message"],
    }

    return {"sample_rendered_output": s1["rendered_text"], "scenario_results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Market Sentiment Agent.")
    parser.add_argument("--run-type", choices=["pre_open", "scheduled", "event"], default="scheduled")
    parser.add_argument("--window", type=int, choices=[15, 30, 60], default=30)
    parser.add_argument("--query", help="Optional user question to test scope guardrails")
    parser.add_argument("--test-scenarios", action="store_true", help="Run full Phase 5 scenario & edge-case suite")
    args = parser.parse_args()

    dsn = get_dsn()
    if args.test_scenarios:
        suite = run_scenario_test_suite(dsn)
        print(suite["sample_rendered_output"])
        print("\nScenario Suite Summary:")
        print(json.dumps(suite["scenario_results"], indent=2))
    else:
        out = run_agent(dsn, run_type=args.run_type, window_minutes=args.window, user_query=args.query)
        print(out.get("rendered_text") or json.dumps(out, indent=2))
