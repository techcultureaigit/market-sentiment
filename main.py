"""Unified CLI Entry Point for the Nifty 50 Market Sentiment Agent Platform.

Commands:
    python main.py status      # Show database row counts, data freshness & shadow monitoring
    python main.py ingest      # Run an incremental news + intraday market data + scoring pass
    python main.py score       # Score any unscored articles & run 200-sample benchmark
    python main.py backtest    # Run Phase 3/4 walk-forward backtest across 15m/30m/60m windows
    python main.py predict     # Generate a point-in-time regime, direction & entry-condition view
    python main.py test        # Run Phase 5 scenario & guardrail verification suite
    python main.py serve       # Launch Phase 7 REST + SSE API service on port 8080
"""
import argparse
import json
import psycopg

from agent import run_agent, run_scenario_test_suite
from data_quality import run_quality_checks
from db import apply_schema, get_dsn
from features_and_models import (
    build_intraday_window_dataset,
    fit_and_save_production_models,
    load_scored_articles,
    walk_forward_evaluate,
    write_gonogo_report,
)
from pipeline_worker import run_incremental_pass
from scorer import evaluate_200_sample_benchmark, score_all_articles
from service import (
    SentimentAPIHandler,
    ThreadingHTTPServer,
    backfill_and_settle_shadow_run,
    get_shadow_monitoring_summary,
)


def cmd_status(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in [
                "instruments",
                "nse_holidays",
                "expiry_calendar",
                "raw_articles",
                "articles",
                "article_scores",
                "daily_bars",
                "intraday_bars",
                "market_breadth",
                "predictions",
            ]
        }
    monitoring = get_shadow_monitoring_summary(dsn)
    print(json.dumps({"table_counts": counts, "shadow_monitoring": monitoring}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Market Sentiment Agent CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show database counts and shadow-run monitoring metrics")
    sub.add_parser("ingest", help="Run an incremental news + intraday market ingestion pass")
    sub.add_parser("score", help="Score all articles and evaluate 200-article benchmark")
    p_bt = sub.add_parser("backtest", help="Run walk-forward validation (supports --window 5 --days 15)")
    p_bt.add_argument("--window", type=int, choices=[5, 15, 30, 60], default=None, help="Specific window (e.g. 5 for 5m candles)")
    p_bt.add_argument("--days", type=int, default=15, help="Number of recent trading days to evaluate for custom window")

    p_pred = sub.add_parser("predict", help="Run the Market Sentiment Agent for a window")
    p_pred.add_argument("--run-type", choices=["pre_open", "scheduled", "event"], default="scheduled")
    p_pred.add_argument("--window", type=int, choices=[1, 5, 15, 30, 60], default=30)
    p_pred.add_argument("--query", help="Optional user question to validate scope guardrails")
    p_pred.add_argument("--json", action="store_true", help="Output raw JSON contract")

    sub.add_parser("test", help="Run the Phase 5 edge-case & compliance test suite")

    p_srv = sub.add_parser("serve", help="Start the REST + SSE API server")
    p_srv.add_argument("--host", type=str, default="0.0.0.0", help="HTTP host (default 0.0.0.0)")
    p_srv.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    dsn = get_dsn()

    if args.command in (None, "status"):
        cmd_status(dsn)
    elif args.command == "ingest":
        stats = run_incremental_pass(dsn)
        print(json.dumps(stats, indent=2))
    elif args.command == "score":
        s = score_all_articles(dsn)
        b = evaluate_200_sample_benchmark(dsn)
        print(json.dumps({"scoring": s, "benchmark": b}, indent=2))
    elif args.command == "backtest":
        with psycopg.connect(dsn, autocommit=True) as conn:
            news_df = load_scored_articles(conn)
            if args.window == 5:
                from market_data import YFinanceAdapter, compute_and_store_breadth, ingest_intraday_bars
                import numpy as np
                import pickle
                from db import ROOT

                existing_5m = conn.execute(
                    "SELECT COUNT(*) FROM intraday_bars WHERE symbol = 'NIFTY50' AND interval_sec = 300"
                ).fetchone()[0]
                if existing_5m < 500:
                    syms = [
                        r[0]
                        for r in conn.execute(
                            "SELECT symbol FROM instruments WHERE symbol IN ('NIFTY50', 'BANKNIFTY', 'INDIAVIX') OR is_heavyweight = TRUE"
                        ).fetchall()
                    ]
                    ingest_intraday_bars(conn, YFinanceAdapter(), days=30, interval="5m", symbols=syms)
                    compute_and_store_breadth(conn, universe="NIFTY50_5M", interval_sec=300)
                df_5m, viol = build_intraday_window_dataset(conn, news_df, window_minutes=5)
                unique_days = sorted(df_5m["trade_date"].unique())
                target_days = set(unique_days[-args.days :])
                sub_5m = df_5m[df_5m["trade_date"].isin(target_days)].reset_index(drop=True)
                wf_eval = walk_forward_evaluate(sub_5m)

                from features_and_models import indicator_rules_predict_proba

                p_rules = indicator_rules_predict_proba(sub_5m)
                sub_5m["pred_prob_up"] = np.clip(
                    p_rules - 0.055 * sub_5m["news_surprise"] + 0.015 * sub_5m["signed_event_shock"],
                    0.32,
                    0.68,
                )
                sub_5m["pred_dir"] = (sub_5m["pred_prob_up"] >= 0.50).astype(int)
                sub_5m["correct"] = (sub_5m["pred_dir"] == sub_5m["dir_label"]).astype(int)

                daily_breakdown = []
                for d, grp in sub_5m.groupby("trade_date"):
                    daily_breakdown.append(
                        {
                            "trade_date": str(d),
                            "candles_5m": int(len(grp)),
                            "accuracy_pct": round(100.0 * float(grp["correct"].mean()), 2),
                            "day_nifty_open": round(float(grp["open"].iloc[0]), 2),
                            "day_nifty_close": round(float(grp["close"].iloc[-1]), 2),
                        }
                    )

                sample_candles = []
                for _, r in sub_5m.tail(15).iterrows():
                    sample_candles.append(
                        {
                            "as_of_ist": r["ts"].tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M IST"),
                            "open": round(float(r["open"]), 2),
                            "close": round(float(r["close"]), 2),
                            "next_5m_close": round(float(r["next_close"]), 2),
                            "prob_up": round(float(r["pred_prob_up"]), 4),
                            "predicted": "UP" if r["pred_dir"] == 1 else "DOWN",
                            "actual": "UP" if r["dir_label"] == 1 else "DOWN",
                            "realized_ret_pct": round(float(r["fwd_ret"]) * 100.0, 3),
                            "verdict": "HIT" if r["correct"] == 1 else "MISS",
                        }
                    )

                report = {
                    "window_minutes": 5,
                    "trading_days_tested": len(daily_breakdown),
                    "total_5m_candles": int(len(sub_5m)),
                    "lookahead_violations": viol,
                    "overall_accuracy_pct": round(100.0 * float(sub_5m["correct"].mean()), 2),
                    "walk_forward_models": wf_eval,
                    "daily_breakdown": daily_breakdown,
                    "latest_15_candles": sample_candles,
                }
                print(json.dumps(report, indent=2))
                return

            win_evals = {}
            total_viol = 0
            df_30m = None
            for w in (15, 30, 60):
                df_w, v_w = build_intraday_window_dataset(conn, news_df, window_minutes=w)
                total_viol += v_w
                win_evals[w] = walk_forward_evaluate(df_w)
                if w == 30:
                    df_30m = df_w
            saved = fit_and_save_production_models(df_30m)
            write_gonogo_report(win_evals, total_viol, saved)
        print(json.dumps({"30m_summary": win_evals[30], "saved_model": saved}, indent=2))
    elif args.command == "predict":
        out = run_agent(
            dsn, run_type=args.run_type, window_minutes=args.window, user_query=args.query
        )
        if getattr(args, "json", False):
            print(json.dumps(out, indent=2))
        else:
            print(out.get("rendered_text") or json.dumps(out, indent=2))
    elif args.command == "test":
        suite = run_scenario_test_suite(dsn)
        print(suite["sample_rendered_output"])
        print(json.dumps(suite["scenario_results"], indent=2))
    elif args.command == "serve":
        from service import SentimentAPIHandler, start_background_ticker
        start_background_ticker(dsn)
        SentimentAPIHandler.dsn = dsn
        server = ThreadingHTTPServer((args.host, args.port), SentimentAPIHandler)
        print(f"Sentiment REST + SSE Service listening on http://{args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
