# demo_pipline
# run a preprocessing + time-series demo
# One collective end-to-end demo


def demo_all():
    from pathlib import Path
    import pandas as pd

    from preprocessing import load_raw, engineer
    from time_series import build_backlog_series, build_daily_panel
    from interval_analysis import IntervalAnalysis, plot_pg_signoff_monthly_trends
    from eda_opg import EDAConfig, OPGInvestigationEDA

    raw, colmap = load_raw(Path("data/raw/raw.csv"))
    typed = engineer(raw, colmap)

    backlog = build_backlog_series(typed)

    if "backlog_available" in backlog.columns and "backlog" not in backlog.columns:
        backlog = backlog.rename(columns={"backlog_available": "backlog"})

    daily, backlog_ts, events = build_daily_panel(typed)

    di = IntervalAnalysis.build_interval_frame(typed, backlog_series=backlog_ts)
    trend = IntervalAnalysis.monthly_trend(
        di, metric="days_to_pg_signoff", agg="median", by=["case_type"]
    ).copy()
    trend["month"] = pd.to_datetime(trend["yyyymm"] + "-01")

    cfg = EDAConfig(
        id_col="case_id",
        date_received="dt_received_inv",
        date_allocated="dt_alloc_invest",
        date_signed_off="dt_pg_signoff",
    )
    eda = OPGInvestigationEDA(typed, cfg)
    overview = eda.quick_overview()

    print("=== EDA OVERVIEW ===")
    print(overview)
    print("\n=== INTERVAL TREND HEAD ===")
    print(trend.head())

    # Call your plotting function for the interval and trends
    results = plot_pg_signoff_monthly_trends(di)

    # Extract for returning
    trend_all = results["trend_all"]

    plot_paths = results["plots"]

    # # Inspect returned objects if you want
    # results["trend"].tail()
    # results["trend_all"].tail()
    # results["plots"]

    return {
        "raw": raw,
        "typed": typed,
        "daily": daily,
        "backlog": backlog_ts,
        "events": events,
        "di": di,
        "trend": trend,
        "overview": overview,
        "trend_all": trend_all,
        "plots": plot_paths,
    }
