"""Re-analyse Historical Intraday Data for 5m, 15m, and 30m Windows.
Runs the updated statistical machine learning model across historical candles,
evaluating Directional Accuracy, Regime Accuracy, Brier Score, and Daily Performance.
Updates backtest CSV files and the backtest_5m_15days.xlsx workbook.
"""
from datetime import datetime, timezone
import math
import pickle
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import psycopg

from db import ROOT, get_dsn
from features_and_models import (
    MARKET_COLS,
    TOP5_WEIGHTS,
    add_technical_features,
    assign_labels,
    build_htf_reference,
    compute_news_features_at_cutoff,
    enhanced_market_prior_proba,
    load_aux_intraday,
    load_scored_articles,
    query_df,
)
from agent import tool_run_model_inference

IST = ZoneInfo("Asia/Kolkata")


def build_window_dataset_for_backtest(conn, news_df: pd.DataFrame, window_minutes: int, limit: int = 1500) -> pd.DataFrame:
    """Build historical window dataset with exact technical, heavyweight, news and S/R features.
    Supports standard intraday windows (5m, 15m, 30m, 60m), Current Day Close (375m), and Next Day Close (750m).
    """
    if window_minutes == 5:
        base_interval_sec = 300
        step_minutes = 5
        breadth_univ = "NIFTY50_5M"
    elif window_minutes == 15:
        base_interval_sec = 900
        step_minutes = 15
        breadth_univ = "NIFTY50_15M"
    elif window_minutes == 30:
        base_interval_sec = 900
        step_minutes = 30
        breadth_univ = "NIFTY50_15M"
    elif window_minutes == 60:
        base_interval_sec = 900
        step_minutes = 60
        breadth_univ = "NIFTY50_15M"
    elif window_minutes in (375, 750):
        base_interval_sec = 900
        step_minutes = 15
        breadth_univ = "NIFTY50_15M"
    else:
        base_interval_sec = 60
        step_minutes = window_minutes
        breadth_univ = "NIFTY50"

    nifty_raw = query_df(
        conn,
        f"""
        SELECT ts, open, high, low, close, available_at
        FROM intraday_bars
        WHERE symbol = 'NIFTY50' AND interval_sec = {base_interval_sec}
        ORDER BY ts DESC LIMIT {limit}
        """,
    ).sort_values("ts").reset_index(drop=True)

    vix_raw = query_df(
        conn,
        f"""
        SELECT ts, close AS vix_close
        FROM intraday_bars
        WHERE symbol = 'INDIAVIX' AND interval_sec = {base_interval_sec}
        ORDER BY ts DESC LIMIT {limit}
        """,
    ).sort_values("ts").reset_index(drop=True)

    breadth = query_df(
        conn,
        """
        SELECT ts, advances, declines, unchanged
        FROM market_breadth
        WHERE universe = %s
        ORDER BY ts ASC
        """,
        (breadth_univ,),
    )

    nifty_raw["ts"] = pd.to_datetime(nifty_raw["ts"], utc=True)
    vix_raw["ts"] = pd.to_datetime(vix_raw["ts"], utc=True)
    breadth["ts"] = pd.to_datetime(breadth["ts"], utc=True)

    merged = (
        nifty_raw.merge(vix_raw, on="ts", how="left")
        .merge(breadth, on="ts", how="left")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    for c in ["open", "high", "low", "close", "vix_close"]:
        merged[c] = merged[c].astype(float)
    merged["vix_close"] = merged["vix_close"].ffill().bfill().fillna(14.0)
    merged["advances"] = merged["advances"].fillna(5).astype(float)
    merged["declines"] = merged["declines"].fillna(5).astype(float)

    merged["ist_dt"] = merged["ts"].dt.tz_convert(IST)
    merged["trade_date"] = merged["ist_dt"].dt.date
    # Strictly filter to regular NSE intraday trading hours: 09:15:00 <= ts < 15:30:00 IST
    t_val = merged["ist_dt"].dt.time
    t_start = datetime.strptime("09:15:00", "%H:%M:%S").time()
    t_end = datetime.strptime("15:30:00", "%H:%M:%S").time()
    merged = merged[(t_val >= t_start) & (t_val < t_end)].reset_index(drop=True)
    mins = (
        (merged["ist_dt"].dt.hour * 60 + merged["ist_dt"].dt.minute) - (9 * 60 + 15)
    ).clip(lower=0)
    merged["win_idx"] = mins // step_minutes

    grouped = merged.groupby(["trade_date", "win_idx"], as_index=False)
    win_bars = grouped.agg(
        ts=("ts", "first"),
        as_of=("available_at", "last"),
        trade_date=("trade_date", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vix_level=("vix_close", "last"),
        advances=("advances", "last"),
        declines=("declines", "last"),
    ).sort_values("ts").reset_index(drop=True)

    win_bars = add_technical_features(win_bars)
    win_bars["vix_change"] = win_bars["vix_level"].pct_change().fillna(0.0)
    total_br = (win_bars["advances"] + win_bars["declines"]).replace(0.0, np.nan)
    win_bars["breadth_ad_ratio"] = (
        (win_bars["advances"] - win_bars["declines"]) / total_br
    ).fillna(0.0)

    # Aux heavyweight data
    aux = load_aux_intraday(conn, base_interval_sec, include_live_unsettled=True)
    if not aux.empty:
        aux["ist_dt"] = aux["ts"].dt.tz_convert(IST)
        aux["trade_date"] = aux["ist_dt"].dt.date
        t_aux = aux["ist_dt"].dt.time
        aux = aux[(t_aux >= t_start) & (t_aux < t_end)].reset_index(drop=True)
        aux_mins = (
            (aux["ist_dt"].dt.hour * 60 + aux["ist_dt"].dt.minute) - (9 * 60 + 15)
        ).clip(lower=0)
        aux["win_idx"] = aux_mins // step_minutes
        aux_grouped = aux.groupby(["trade_date", "win_idx"], as_index=False).agg(
            rupee_vol=("rupee_vol", "sum"),
            banknifty_close=("banknifty_close", "last"),
            top5_ret_1=("top5_ret_bar", "sum"),
        )
        win_bars = win_bars.merge(aux_grouped, on=["trade_date", "win_idx"], how="left")
    else:
        win_bars["rupee_vol"] = 0.0
        win_bars["banknifty_close"] = np.nan
        win_bars["top5_ret_1"] = 0.0

    win_bars["rupee_vol"] = win_bars["rupee_vol"].fillna(0.0)
    win_bars["banknifty_close"] = win_bars["banknifty_close"].ffill().bfill().fillna(win_bars["close"])
    win_bars["banknifty_ret_1"] = np.log(
        win_bars["banknifty_close"] / win_bars["banknifty_close"].shift(1)
    ).fillna(0.0)
    win_bars["top5_ret_1"] = win_bars["top5_ret_1"].fillna(win_bars["ret_1"])

    # True Session VWAP
    tp = (win_bars["high"] + win_bars["low"] + win_bars["close"]) / 3.0
    vol_pos = win_bars["rupee_vol"].replace(0.0, 1.0)
    win_bars["_tp_vol"] = tp * vol_pos
    cum_tp_vol = win_bars.groupby("trade_date")["_tp_vol"].cumsum()
    cum_vol = win_bars.groupby("trade_date")["rupee_vol"].transform(
        lambda s: s.replace(0.0, 1.0).cumsum()
    )
    true_vwap = cum_tp_vol / cum_vol
    win_bars["true_vwap_dist"] = ((win_bars["close"] - true_vwap) / win_bars["close"]).fillna(0.0)
    win_bars = win_bars.drop(columns=["_tp_vol"])

    # Relative Volume Spike
    avg_vol_14 = win_bars["rupee_vol"].rolling(14, min_periods=2).mean().replace(0.0, np.nan)
    rel_vol = (win_bars["rupee_vol"] / avg_vol_14).fillna(1.0).clip(0.25, 4.0)
    win_bars["vol_spike"] = np.log(rel_vol).fillna(0.0)

    # Multi-Timeframe Alignment
    win_bars["as_of_utc"] = pd.to_datetime(win_bars["as_of"], utc=True)
    htf15 = build_htf_reference(conn, 15, include_live_unsettled=True)
    htf60 = build_htf_reference(conn, 60, include_live_unsettled=True)
    win_bars = pd.merge_asof(
        win_bars.sort_values("as_of_utc"),
        htf15,
        left_on="as_of_utc",
        right_on="htf_as_of",
        direction="backward",
    ).drop(columns=["htf_as_of"], errors="ignore")
    win_bars = pd.merge_asof(
        win_bars.sort_values("as_of_utc"),
        htf60,
        left_on="as_of_utc",
        right_on="htf_as_of",
        direction="backward",
    ).drop(columns=["htf_as_of"], errors="ignore")
    win_bars = win_bars.sort_values("ts").reset_index(drop=True)
    for c in ["htf15_slope", "htf60_slope"]:
        if c in win_bars.columns:
            win_bars[c] = win_bars[c].fillna(0.0)
        else:
            win_bars[c] = 0.0
    win_bars["mtf_trend_score"] = (
        0.50 * np.sign(win_bars["htf15_slope"]) + 0.50 * np.sign(win_bars["htf60_slope"])
    ).fillna(0.0)

    # Forward returns & Next close/high/low mapping
    if window_minutes == 375:
        # Current Day: generated at ~10:00 AM IST in the morning session predicting 15:30 IST session close
        day_close_dict = win_bars.groupby("trade_date")["close"].last().to_dict()
        day_high_dict = win_bars.groupby("trade_date")["high"].max().to_dict()
        day_low_dict = win_bars.groupby("trade_date")["low"].min().to_dict()
        win_bars["next_close"] = win_bars["trade_date"].map(day_close_dict)
        win_bars["next_high"] = win_bars["trade_date"].map(day_high_dict)
        win_bars["next_low"] = win_bars["trade_date"].map(day_low_dict)
        win_bars["fwd_ret"] = np.log(win_bars["next_close"] / win_bars["close"])
        win_bars["ist_dt"] = win_bars["ts"].dt.tz_convert(IST)
        # Select exactly 1 prediction per trading day at 10:00 AM IST (or closest bar within 30m)
        win_bars["dist_to_10am"] = win_bars["ist_dt"].apply(
            lambda dt: abs((dt.hour * 60 + dt.minute) - (10 * 60))
        )
        idx_closest = win_bars.groupby("trade_date")["dist_to_10am"].idxmin()
        win_bars = win_bars.loc[idx_closest].sort_values("ts").reset_index(drop=True)
        win_bars = win_bars[win_bars["dist_to_10am"] <= 30].reset_index(drop=True)
        win_bars = win_bars.drop(columns=["dist_to_10am"])
    elif window_minutes == 750:
        # Next Day: generated at ~2:30 PM–3:00 PM IST (14:45 IST) in the afternoon session predicting next-day 15:30 close
        day_close_dict = win_bars.groupby("trade_date")["close"].last().to_dict()
        day_high_dict = win_bars.groupby("trade_date")["high"].max().to_dict()
        day_low_dict = win_bars.groupby("trade_date")["low"].min().to_dict()
        sorted_dates = sorted(win_bars["trade_date"].unique())
        next_date_map = {sorted_dates[i]: sorted_dates[i+1] for i in range(len(sorted_dates)-1)}
        win_bars["target_date"] = win_bars["trade_date"].map(next_date_map)
        win_bars["next_close"] = win_bars["target_date"].map(day_close_dict)
        win_bars["next_high"] = win_bars["target_date"].map(day_high_dict)
        win_bars["next_low"] = win_bars["target_date"].map(day_low_dict)
        win_bars["fwd_ret"] = np.log(win_bars["next_close"] / win_bars["close"])
        win_bars["ist_dt"] = win_bars["ts"].dt.tz_convert(IST)
        # Select exactly 1 prediction per trading day at 14:45 PM IST (between 2:30–3:00 PM)
        win_bars["dist_to_afternoon"] = win_bars["ist_dt"].apply(
            lambda dt: abs((dt.hour * 60 + dt.minute) - (14 * 60 + 45))
        )
        idx_closest = win_bars.groupby("trade_date")["dist_to_afternoon"].idxmin()
        win_bars = win_bars.loc[idx_closest].sort_values("ts").reset_index(drop=True)
        win_bars = win_bars[win_bars["dist_to_afternoon"] <= 30].reset_index(drop=True)
        win_bars = win_bars.drop(columns=["dist_to_afternoon"])
        win_bars = win_bars[win_bars["next_close"].notna()].reset_index(drop=True)
    else:
        # Standard Intraday Windows (5m, 15m, 30m, 60m)
        win_bars["next_trade_date"] = win_bars["trade_date"].shift(-1)
        win_bars["next_open"] = win_bars["open"].shift(-1)
        win_bars["next_high"] = win_bars["high"].shift(-1)
        win_bars["next_low"] = win_bars["low"].shift(-1)
        win_bars["next_close"] = win_bars["close"].shift(-1)
        win_bars["fwd_ret"] = np.log(win_bars["next_close"] / win_bars["close"])

    # News features
    news_rows = []
    for as_of_ts in win_bars["as_of"]:
        as_of_utc = pd.to_datetime(as_of_ts, utc=True).to_pydatetime()
        nf, _ = compute_news_features_at_cutoff(news_df, as_of_utc, lookback_hours=12.0, recent_hours=1.5)
        news_rows.append(nf)
    news_feat_df = pd.DataFrame(news_rows)
    full_df = pd.concat([win_bars, news_feat_df], axis=1)

    full_df["rules_prior"] = enhanced_market_prior_proba(full_df) - 0.50
    full_df["news_surprise"] = (
        full_df["latest_3h_sentiment"] - full_df["rel_weighted_sentiment"]
    )
    full_df["gated_news_sentiment"] = (
        full_df["latest_3h_sentiment"] * np.tanh(full_df["news_volume"] / 2.0)
    )
    full_df["sector_weighted_sentiment"] = (
        0.45 * full_df["sector_banking_sent"]
        + 0.30 * full_df["sector_it_sent"]
        + 0.25 * full_df["sector_energy_sent"]
    )
    full_df["signed_event_shock"] = (
        full_df["max_event_severity"] * np.sign(full_df["latest_3h_sentiment"])
    )

    if window_minutes in (375, 750):
        full_df = full_df[full_df["fwd_ret"].notna()].reset_index(drop=True)
    else:
        # Exclude cross-session gaps and latest forming candle without forward return
        same_day = (full_df["trade_date"] == full_df["next_trade_date"]) & full_df["fwd_ret"].notna()
        full_df = full_df[same_day].reset_index(drop=True)

    full_df = assign_labels(full_df, k_atr=0.35, m_atr=0.75)
    return full_df


def run_backtest_for_horizon(df: pd.DataFrame, bundle: dict, window_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Execute model inference and evaluate performance metrics for a specific window timeframe."""
    records = []
    w_band = 0.00060 if window_minutes <= 5 else (0.00100 if window_minutes <= 15 else 0.00150)

    for _, row in df.iterrows():
        inf = tool_run_model_inference(row, bundle, window_minutes=window_minutes)
        as_of_ist = pd.to_datetime(row["ts"]).astimezone(IST)
        if window_minutes == 375:
            end_ist = as_of_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        elif window_minutes == 750:
            t_dt = row.get("target_date")
            if pd.notna(t_dt):
                if isinstance(t_dt, str):
                    t_dt = datetime.strptime(t_dt, "%Y-%m-%d").date()
                end_ist = datetime.combine(t_dt, datetime.min.time().replace(hour=15, minute=30)).replace(tzinfo=IST)
            else:
                end_ist = (as_of_ist + pd.Timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)
        else:
            end_ist = as_of_ist + pd.Timedelta(minutes=window_minutes)
            close_ist = as_of_ist.replace(hour=15, minute=30, second=0, microsecond=0)
            if end_ist > close_ist:
                end_ist = close_ist

        fwd_ret = float(row["fwd_ret"])
        ret_pct = round(float(fwd_ret * 100.0), 3)

        prob_up = float(inf["prob_up"])
        pred_dir = "UP" if prob_up >= 0.50 else "DOWN"
        act_dir = "UP" if fwd_ret >= 0 else "DOWN"

        # Sideways market trend commented out per user request:
        # if abs(fwd_ret) <= w_band:
        #     act_reg = "sideways"
        # elif fwd_ret > w_band:
        #     act_reg = "bullish"
        # else:
        #     act_reg = "bearish"
        pred_reg = "bullish" if prob_up >= 0.50 else "bearish"
        act_reg = "bullish" if fwd_ret >= 0 else "bearish"

        # Pure binary result: strictly HIT or MISS based on Direction
        verdict = "HIT" if (pred_dir == act_dir) else "MISS"

        # Trade Bracket: Stop Loss (1.20x ATR) & Target (1.80x ATR, 1:1.50 R:R)
        tb = inf.get("trade_bracket") or {}
        entry_price = float(row["close"])
        sl_price = float(tb.get("stop_loss", round(entry_price - 30.0 if pred_dir == "UP" else entry_price + 30.0, 1)))
        tp_price = float(tb.get("target_1", round(entry_price + 45.0 if pred_dir == "UP" else entry_price - 45.0, 1)))

        n_high = float(row.get("next_high", row["next_close"]))
        n_low = float(row.get("next_low", row["next_close"]))
        n_close = float(row["next_close"])

        if pred_dir == "UP":
            hit_sl = n_low <= sl_price
            hit_tp = n_high >= tp_price
            if hit_sl and hit_tp:
                # Conservative order execution: assume adverse move happened first
                trade_outcome = "SL_HIT"
                trade_pnl_pct = round((sl_price - entry_price) / entry_price * 100.0, 3)
            elif hit_sl:
                trade_outcome = "SL_HIT"
                trade_pnl_pct = round((sl_price - entry_price) / entry_price * 100.0, 3)
            elif hit_tp:
                trade_outcome = "TARGET_HIT"
                trade_pnl_pct = round((tp_price - entry_price) / entry_price * 100.0, 3)
            else:
                trade_pnl_pct = round((n_close - entry_price) / entry_price * 100.0, 3)
                trade_outcome = "TIME_EXIT_WIN" if trade_pnl_pct >= 0 else "TIME_EXIT_LOSS"
        else: # DOWN / Bearish Short
            hit_sl = n_high >= sl_price
            hit_tp = n_low <= tp_price
            if hit_sl and hit_tp:
                trade_outcome = "SL_HIT"
                trade_pnl_pct = round((entry_price - sl_price) / entry_price * 100.0, 3)
            elif hit_sl:
                trade_outcome = "SL_HIT"
                trade_pnl_pct = round((entry_price - sl_price) / entry_price * 100.0, 3)
            elif hit_tp:
                trade_outcome = "TARGET_HIT"
                trade_pnl_pct = round((entry_price - tp_price) / entry_price * 100.0, 3)
            else:
                trade_pnl_pct = round((entry_price - n_close) / entry_price * 100.0, 3)
                trade_outcome = "TIME_EXIT_WIN" if trade_pnl_pct >= 0 else "TIME_EXIT_LOSS"

        records.append(
            {
                "Trade_Date": str(row["trade_date"]),
                "Candle_Start_IST": as_of_ist.strftime("%Y-%m-%d %H:%M IST"),
                "Candle_End_IST": end_ist.strftime("%H:%M IST"),
                "Nifty_Open": round(float(row["open"]), 2),
                "Nifty_High": round(float(row["high"]), 2),
                "Nifty_Low": round(float(row["low"]), 2),
                "Nifty_Close": round(float(row["close"]), 2),
                "Next_5m_Close": round(float(row["next_close"]), 2),
                "Stop_Loss": sl_price,
                "Target_Price": tp_price,
                "India_VIX": round(float(row.get("vix_level", 14.0)), 2),
                "ADX_14": round(float(row.get("adx", 18.0)), 1),
                "EMA20_Slope_Pct": round(float(row.get("ema20_slope", 0.0)) * 100.0, 3),
                "Breadth_Ratio": round(float(row.get("breadth_ad_ratio", 0.0)), 2),
                "Volume_Spike_Log": round(float(row.get("vol_spike", 0.0)), 2),
                "MTF_Trend_Score": round(float(row.get("mtf_trend_score", 0.0)), 1),
                "News_Surprise": round(float(row.get("news_surprise", 0.0)), 3),
                "Prob_Up_Pct": round(prob_up * 100.0, 2),
                "Predicted_Dir": pred_dir,
                "Actual_Dir": act_dir,
                "Predicted_Regime": pred_reg,
                "Actual_Regime": act_reg,
                "Entry_Environment": inf["entry_condition"],
                "Realized_Return_Pct": ret_pct,
                "Verdict": verdict,
                "Trade_Outcome": trade_outcome,
                "Trade_PnL_Pct": trade_pnl_pct,
            }
        )

    res_df = pd.DataFrame(records)

    # Daily aggregation summary
    daily = (
        res_df.groupby("Trade_Date", as_index=False)
        .agg(
            Candles=(f"Verdict", "count"),
            Hits=("Verdict", lambda s: int((s == "HIT").sum())),
            Misses=("Verdict", lambda s: int((s == "MISS").sum())),
            Accuracy_Pct=("Verdict", lambda s: round(100.0 * float((s == "HIT").mean()), 2)),
            Target_Hits=("Trade_Outcome", lambda s: int((s == "TARGET_HIT").sum())),
            SL_Hits=("Trade_Outcome", lambda s: int((s == "SL_HIT").sum())),
            Total_PnL_Pct=("Trade_PnL_Pct", lambda s: round(float(s.sum()), 2)),
            Day_Open=("Nifty_Open", "first"),
            Day_Close=("Nifty_Close", "last"),
        )
        .rename(columns={"Candles": f"Candles_{window_minutes}m"})
        .sort_values("Trade_Date", ascending=False)
        .reset_index(drop=True)
    )

    y_true_dir = (res_df["Realized_Return_Pct"] >= 0).astype(int)
    y_pred_prob = res_df["Prob_Up_Pct"] / 100.0
    y_pred_dir = (y_pred_prob >= 0.50).astype(int)

    dir_acc = round(100.0 * float((y_true_dir == y_pred_dir).mean()), 2)
    overall_acc = round(100.0 * float((res_df["Verdict"] == "HIT").mean()), 2)
    brier = round(float(np.mean((y_pred_prob - y_true_dir) ** 2)), 4)

    fav_df = res_df[res_df["Entry_Environment"] == "favourable"]
    fav_acc = (
        round(100.0 * float((fav_df["Verdict"] == "HIT").mean()), 2)
        if not fav_df.empty
        else 0.0
    )

    wins_mask = res_df["Trade_Outcome"].isin(["TARGET_HIT", "TIME_EXIT_WIN"])
    trade_win_rate = round(100.0 * float(wins_mask.mean()), 2)
    cum_trade_pnl = round(float(res_df["Trade_PnL_Pct"].sum()), 2)
    gross_win = float(res_df.loc[res_df["Trade_PnL_Pct"] > 0, "Trade_PnL_Pct"].sum())
    gross_loss = abs(float(res_df.loc[res_df["Trade_PnL_Pct"] < 0, "Trade_PnL_Pct"].sum()))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0
    sl_hits = int((res_df["Trade_Outcome"] == "SL_HIT").sum())
    tp_hits = int((res_df["Trade_Outcome"] == "TARGET_HIT").sum())

    metrics = {
        "window_minutes": window_minutes,
        "total_candles": len(res_df),
        "trading_days": int(res_df["Trade_Date"].nunique()),
        "overall_verdict_accuracy_pct": overall_acc,
        "directional_accuracy_pct": dir_acc,
        "brier_score": brier,
        "favourable_calls": len(fav_df),
        "favourable_accuracy_pct": fav_acc,
        "trade_win_rate_pct": trade_win_rate,
        "cum_trade_pnl_pct": cum_trade_pnl,
        "profit_factor": pf,
        "target_hits": tp_hits,
        "stop_loss_hits": sl_hits,
        "date_range": f"{res_df['Trade_Date'].min()} to {res_df['Trade_Date'].max()}",
    }
    return res_df, daily, metrics


def main():
    print("=" * 70)
    print("RE-ANALYSING HISTORIC INTRADAY DATA (5m, 15m, 30m, 60m, Current Day, Next Day)")
    print("=" * 70)

    model_path = ROOT / "models" / "sentiment_agent_v1.pkl"
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    dsn = get_dsn()
    with psycopg.connect(dsn) as conn:
        news_df = load_scored_articles(conn)
        print(f"Loaded {len(news_df)} scored news articles from database.")

        results = {}
        excel_writer = pd.ExcelWriter("backtest_5m_15days.xlsx", engine="openpyxl")

        horizons = [5, 15, 30, 60, 375, 750]
        for win in horizons:
            hz_title = "Current Day (Close 15:30)" if win == 375 else ("Next Day (Close 15:30)" if win == 750 else f"{win}m")
            print(f"\nProcessing {hz_title} horizon...")
            limit = 1500
            df_win = build_window_dataset_for_backtest(conn, news_df, window_minutes=win, limit=limit)
            print(f"Built {len(df_win)} valid feature candles for {hz_title}.")

            candles_df, daily_df, metrics = run_backtest_for_horizon(df_win, bundle, window_minutes=win)
            metrics["daily_df"] = daily_df
            metrics["title"] = hz_title
            results[win] = metrics

            # Save CSVs matching dashboard requirements
            csv_path = ROOT / f"backtest_{win}m_candles.csv"
            candles_df.to_csv(csv_path, index=False)
            if win == 5:
                candles_df.to_csv(ROOT / "backtest_5m_15days_all_candles.csv", index=False)
            print(f"Saved {csv_path.name} ({len(candles_df)} candles/predictions).")

            # Write sheets into Excel workbook
            sheet_prefix = "CurrentDay" if win == 375 else ("NextDay" if win == 750 else f"{win}m")
            daily_df.to_excel(excel_writer, sheet_name=f"{sheet_prefix}_Daily", index=False)
            candles_df.to_excel(excel_writer, sheet_name=f"{sheet_prefix}_Predictions", index=False)

        excel_writer.close()
        print("\nSuccessfully updated 'backtest_5m_15days.xlsx'.")

        print("\n" + "=" * 70)
        print("HISTORIC RE-ANALYSIS & BACKTEST SUMMARY")
        print("=" * 70)
        for win, m in results.items():
            print(f"\nHorizon: {m['title']}")
            print(f"  • Date Range: {m['date_range']} ({m['trading_days']} trading days)")
            print(f"  • Total Predictions / Candles Analysed: {m['total_candles']}")
            print(f"  • Overall Verdict Accuracy: {m['overall_verdict_accuracy_pct']}%")
            print(f"  • Directional Accuracy: {m['directional_accuracy_pct']}%")
            print(f"  • Trade Outcome Win Rate: {m['trade_win_rate_pct']}%")
            print(f"  • Strategy Profit Factor: {m['profit_factor']:.2f}")
            print(f"  • Brier Calibration Score: {m['brier_score']}")
            print(f"  • Favourable Calls Count: {m['favourable_calls']} (Accuracy: {m['favourable_accuracy_pct']}%)")
            print("  Recent Daily Breakdown (Last 5 Trading Days):")
            recent_daily = m["daily_df"].head(5)
            cnt_col = f"Candles_{win}m" if f"Candles_{win}m" in recent_daily.columns else recent_daily.columns[1]
            for _, r in recent_daily.iterrows():
                dt = r["Trade_Date"]
                cnt = r[cnt_col]
                hits = r["Hits"]
                miss = r["Misses"]
                acc = r["Accuracy_Pct"]
                opn = r["Day_Open"]
                cls = r["Day_Close"]
                print(f"    - {dt}: {cnt:2d} | Hits: {hits:2d}, Misses: {miss:2d} | Accuracy: {acc:5.2f}% | Open: {opn:.1f} -> Close: {cls:.1f}")


if __name__ == "__main__":
    main()
