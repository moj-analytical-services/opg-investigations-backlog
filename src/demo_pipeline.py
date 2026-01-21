# One collective end-to-end demo
# One collective end-to-end demo

from pathlib import Path
import pandas as pd
import numpy as np

from preprocessing import load_raw, engineer
from time_series import build_backlog_series, build_daily_panel
from interval_analysis import IntervalAnalysis, plot_pg_signoff_monthly_trends
from eda_opg import EDAConfig, OPGInvestigationEDA
from distributions import interval_change_distribution


def demo_all():

    raw, colmap = load_raw(Path("data/raw/raw.csv"))  # REAL DATA
    typed = engineer(raw, colmap)
    # typed["date_received_opg"] = typed["dt_received_inv"]

    # # Compute days from receipt to allocation on the REAL data
    # typed["days_to_alloc"] = (
    #     typed["dt_alloc_invest"] - typed["dt_received_inv"]
    # ).dt.days

    # Time-series panels
    backlog = build_backlog_series(typed)
    if "backlog_available" in backlog.columns and "backlog" not in backlog.columns:
        backlog = backlog.rename(columns={"backlog_available": "backlog"})
    daily, backlog_ts, events = build_daily_panel(typed)

    # Interval frame for analysis (includes wip_load, time_since_last_pickup, event_newcase, etc.)
    di = IntervalAnalysis.build_interval_frame(typed, backlog_series=backlog_ts)
    # All the “interval” analysis we want should be plugged in after di is created.
    # That guarantees we are working on real data, not synthetic.

    # ------------------------------------------------------
    # Distributions of key time intervals using REAL data
    # ------------------------------------------------------
    interval_dists_overall = IntervalAnalysis.analyse_interval_distributions(di)

    interval_dists_by_case_type = IntervalAnalysis.analyse_interval_distributions(
        di, by=["case_type"]
    )

    interval_dists_by_team = IntervalAnalysis.analyse_interval_distributions(
        di, by=["team"]
    )

    # Optional: print a quick summary so you can see something immediately
    print("\n=== Interval distributions (overall, last 4 years) ===")
    print(pd.DataFrame(interval_dists_overall).T.head())

    print("\n=== Interval distributions by case_type (example: inter_pickup_days) ===")
    # This is a nested dict -> flatten or inspect one metric:
    inter_pickup = interval_dists_by_case_type.get("inter_pickup_days", {})
    print(pd.DataFrame(inter_pickup).T.head())

    # ------------------------------------------------------
    # Probability of new case start vs workload & gap
    # ------------------------------------------------------

    # “Rules” about new case starts vs workload & time since last case:

    # - From the interval frame di:
    # - wip_load – investigators’ weighted caseload on that date
    # - time_since_last_pickup – days since they last started a new case
    # - event_newcase – indicator that a new case started that day

    # - What we want is a conditional probability table:
    #     - P(new case today | caseload band, gap since last pickup band)

    # - If caseload LOW & gap LONG → probability HIGH?

    # - If caseload HIGH & gap LONG → probability LOW?

    # etc.

    pickup_df = di[
        ["date", "staff_id", "wip_load", "time_since_last_pickup", "event_newcase"]
    ].copy()

    # Drop rows where we don't know the gap or caseload
    pickup_df = pickup_df.dropna(subset=["wip_load", "time_since_last_pickup"])

    # Define workload bands (tweak thresholds as needed for OPG)
    pickup_df["wip_band"] = pd.cut(
        pickup_df["wip_load"],
        bins=[0, 40, 80, 120, float("inf")],
        labels=["Low", "Medium", "High", "Very high"],
        right=False,
        include_lowest=True,
    )

    # Define time-since-last-pickup bands (gap in days)
    pickup_df["gap_band"] = pd.cut(
        pickup_df["time_since_last_pickup"],
        bins=[0, 7, 14, 28, 90, float("inf")],
        labels=["<1 week", "1–2 weeks", "2–4 weeks", "4–13 weeks", ">13 weeks"],
        right=False,
        include_lowest=True,
    )

    # Probability: mean of event_newcase within each (wip_band, gap_band)
    pickup_prob = (
        pickup_df.groupby(["wip_band", "gap_band"], dropna=False, observed=False)[
            "event_newcase"
        ]
        .mean()
        .unstack("gap_band")
    )

    # Counts: how many staff-days in each cell (for reliability)
    pickup_counts = (
        pickup_df.groupby(["wip_band", "gap_band"], dropna=False, observed=False)[
            "event_newcase"
        ]
        .size()
        .unstack("gap_band")
    )

    print("\n=== P(new case start | workload band, gap band) ===")
    print(pickup_prob)

    print("\n=== Number of staff-days underlying each cell ===")
    print(pickup_counts)

    # year-on-year interval changes
    alloc_change = interval_change_distribution(
        typed,
        interval_col="days_to_alloc",
        date_col="dt_received_inv",  # <- real received-date column
        group="case_type",
    )

    alloc_annual_stats = alloc_change["annual_stats"]
    alloc_yoy_change = alloc_change["yoy_change"]

    print("\n=== Annual distributions of days_to_alloc by case_type ===")
    print(alloc_annual_stats.head())
    print("\n=== Year-on-year change in median days_to_alloc by case_type ===")
    print(alloc_yoy_change.head())

    # Trend Analysis
    trend = IntervalAnalysis.monthly_trend(
        di, metric="days_to_pg_signoff", agg="median", by=["case_type"]
    ).copy()
    trend["month"] = pd.to_datetime(trend["yyyymm"] + "-01")

    print("\n=== INTERVAL TREND HEAD ===")
    print(trend.head())

    cfg = EDAConfig(
        id_col="case_id",
        date_received="dt_received_inv",
        date_allocated="dt_alloc_invest",
        date_signed_off="dt_pg_signoff",
    )
    cfg.numeric_cols = [
        "days_to_alloc",
        "days_to_signoff",
        "legal_review",
        "fte",
        "weighting",
    ]
    eda = OPGInvestigationEDA(typed, cfg)
    print("=== EDA COLUMNS ===")
    print(eda.df.columns.tolist())

    # --- EDA code from demo_eda.py ---

    print("=== EDA OVERVIEW ===")
    overview = eda.quick_overview()
    # print(overview)

    print("=== EDA MISSING ===")
    missing_pct = eda.missingness_matrix()
    missing_vs_target = eda.missing_vs_target("days_to_signoff")  # , "legal_review")
    outliers_signoff = eda.iqr_outliers("days_to_signoff")
    outliers_allocate = eda.iqr_outliers("days_to_alloc")
    # cat_summary = eda.group_summary(["case_type", "risk_band"], target="legal_review")
    # cat_summary = eda.group_summary(
    #     ["case_type", "risk_band"],
    #     metrics={"legal_rate": ("legal_review", "mean")}
    # ) #"n": ("id", "count"),
    weight_summary = eda.group_summary(
        by=["weighting"],
        metrics={  # "new_column_name": ("existing_column_name", "aggfunc")
            "n_cases": ("case_id", "count"),
            "legal_rate": (
                "legal_review",
                "mean",
            ),  # proportion of cases with legal_review=1
            "median_days_to_signoff": (
                "days_to_signoff",
                "median",
            ),  # "aggfunc" one of: "count", "mean", "median", "min", "max", "std", etc.
        },
    )

    case_weight_summary = eda.group_summary(
        by=["case_type", "weighting"],
        metrics={
            "n_cases": ("case_id", "count"),
            "median_days_to_alloc": ("days_to_alloc", "median"),
            "median_days_to_signoff": ("days_to_signoff", "median"),
        },
    )
    case_weight_summary = case_weight_summary.sort_values(["case_type", "weighting"])

    legal_review_by_case_type = eda.group_summary(
        by=["case_type"],
        metrics={
            # "avg_backlog": ("backlog", "mean"),
            "legal_rate": ("legal_review", "mean"),
        },
    )

    legal_review_by_case_status = eda.group_summary(
        by=["case_type", "status"],
        metrics={
            # "avg_backlog": ("backlog", "mean"),
            "legal_rate": ("legal_review", "mean"),
        },
    )

    staff_summary = eda.group_summary(
        by=["staff_id", "fte"],
        metrics={
            "n_cases": ("case_id", "count"),
            "mean_days_to_alloc": ("days_to_alloc", "mean"),
            "mean_days_to_signoff": ("days_to_signoff", "mean"),
            "legal_rate": ("legal_review", "mean"),
        },
    )

    fte_weight_summary = eda.group_summary(
        by=["fte"],
        metrics={
            "total_weight": ("weighting", "sum"),
            "avg_weight": ("weighting", "mean"),
        },
    )

    case_weight_full = eda.group_summary(
        by=["case_type", "weighting"],
        metrics={
            "staff_id": ("staff_id", "count"),
            # "avg_backlog": ("backlog", "mean"),
            "median_days_to_alloc": ("days_to_alloc", "median"),
            "median_days_to_signoff": ("days_to_signoff", "median"),
            "legal_rate": ("legal_review", "mean"),
        },
    )
    case_weight_full = case_weight_full.sort_values(["case_type", "weighting"])

    status_summary = eda.group_summary(
        by=["status"],
        metrics={
            "staff_id": ("staff_id", "count"),
            "legal_rate": ("legal_review", "mean"),
            "median_days_to_signoff": ("days_to_signoff", "median"),
            "median_days_to_alloc": ("days_to_alloc", "median"),
        },
    )

    legal_case_summary = eda.group_summary(
        by=["case_type", "legal_review"],
        metrics={
            "staff_id": ("staff_id", "count"),
            "median_days_to_signoff": ("days_to_signoff", "median"),
            "median_days_to_alloc": ("days_to_alloc", "median"),
        },
    )

    corrs = eda.numeric_correlations(method="spearman")
    class_balance = eda.imbalance_summary()
    leakage_hits = eda.leakage_scan(["post", "signed", "decision"])
    interaction = eda.binned_interaction_rate(
        num_col="days_to_alloc",
        cat_col="weighting",
        target="legal_review",
    )

    # ts_7d, lag_corrs = eda.resample_time_series(
    #     metrics={"days_to_alloc": ("days_to_alloc", "last"),
    #              "staff_count": ("staff_id", "count")}
    # )

    # km_q = eda.km_quantiles_by_group(group="weighting")
    # monthly_kpis = eda.monthly_kpis()
    cramers_case_type_w = eda.cramers_v(typed["case_type"], typed["weighting"])
    cramers_case_type_fte = eda.cramers_v(typed["case_type"], typed["fte"])

    # --- END EDA code ---

    def weighted_mean(values, weights):
        v = np.asarray(values)
        w = np.asarray(weights)
        mask = ~np.isnan(v) & ~np.isnan(w)
        if mask.sum() == 0:
            return np.nan
        return (v[mask] * w[mask]).sum() / w[mask].sum()

    # weighted mean days_to_signoff by case_type
    weighted_signoff = (
        eda.df.groupby("case_type")
        .apply(lambda g: weighted_mean(g["days_to_signoff"], g["weighting"]))
        .reset_index(name="w_mean_days_to_signoff")
    )

    # weighted mean days_to_alloc by case_type
    weighted_alloc = (
        eda.df.groupby("case_type")
        .apply(lambda g: weighted_mean(g["days_to_alloc"], g["weighting"]))
        .reset_index(name="w_mean_days_to_alloc")
    )

    # Call your plotting function for the interval and trends
    results = plot_pg_signoff_monthly_trends(di, "data/out/plot/plots")

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
        "backlog_ts": backlog_ts,
        "di": di,
        "trend": trend,
        "trend_all": trend_all,
        "eda": {
            "cfg": cfg,
            "overview": overview,
            "missing_pct": missing_pct,
            "missing_vs_target": missing_vs_target,
            "outliers_signoff": outliers_signoff,
            "outliers_allocate": outliers_allocate,
            "weight_summary": weight_summary,
            "case_weight_summary": case_weight_summary,
            "legal_review_by_case_type": legal_review_by_case_type,
            "legal_review_by_case_status": legal_review_by_case_status,
            "staff_summary": staff_summary,
            "fte_weight_summary": fte_weight_summary,
            "status_summary": status_summary,
            "case_weight_full": case_weight_full,
            "legal_case_summary": legal_case_summary,
            "corrs": corrs,
            "cramers_case_type_w": cramers_case_type_w,
            "cramers_case_type_fte": cramers_case_type_fte,
            "class_balance": class_balance,
            "leakage_hits": leakage_hits,
            "interaction": interaction,
            # "ts_7d": ts_7d,
            # "lag_corrs": lag_corrs,
            # "km_quantiles": km_q,
            # "monthly_kpis": monthly_kpis,
        },
        "interval_dists_overall": interval_dists_overall,
        "interval_dists_by_case_type": interval_dists_by_case_type,
        "interval_dists_by_team": interval_dists_by_team,
        "pickup_prob": pickup_prob,
        "pickup_counts": pickup_counts,
        "alloc_annual_stats": alloc_annual_stats,
        "alloc_yoy_change": alloc_yoy_change,
        "weighted_signoff": weighted_signoff,
        "weighted_alloc": weighted_alloc,
        "plots": plot_paths,
    }


# from demo_pipeline import demo_all
outputs = demo_all()

print("===========================================")
print("===== INTERVAL DISTRIBUTIONS =====")
# Interval distributions
print(outputs["interval_dists_overall"])  # overall distributions
print(
    outputs["interval_dists_by_case_type"]
)  # dict keyed by metric -> group distributions
print(outputs["interval_dists_by_team"])

print("===========================================")
print("===== “Rules” table: P(new case | workload, gap) =====")
# “Rules” table: P(new case | workload, gap)
print(outputs["pickup_prob"])  # probability grid
print(outputs["pickup_counts"])  # cell counts

print("===========================================")
print("===== YoY allocation interval distributions =====")
# YoY allocation interval distributions
print(outputs.get("alloc_annual_stats"))
print(outputs.get("alloc_yoy_change"))

print("===========================================")
print("===== EDA =====")
print(outputs["eda"]["cfg"])
print(outputs["eda"]["overview"])
print(outputs["eda"]["missing_pct"])
print(outputs["eda"]["missing_vs_target"])
print(outputs["eda"]["outliers_signoff"])
print(outputs["eda"]["outliers_allocate"])
print(outputs["eda"]["weight_summary"])
print(outputs["eda"]["case_weight_summary"])
print(outputs["eda"]["legal_review_by_case_type"])
print(outputs["eda"]["legal_review_by_case_status"])
print(outputs["eda"]["staff_summary"])
print(outputs["eda"]["fte_weight_summary"])
print(outputs["eda"]["status_summary"])
print(outputs["eda"]["case_weight_full"])
print(outputs["eda"]["legal_case_summary"])
print(outputs["eda"]["corrs"])
print(outputs["eda"]["corrs"])
print(outputs["eda"]["cramers_case_type_w"])
print(outputs["eda"]["cramers_case_type_fte"])
print(outputs["eda"]["class_balance"])
print(outputs["eda"]["leakage_hits"])
print(outputs["eda"]["interaction"])
# print(outputs["eda"]["ts_7d"])
# print(outputs["eda"]["lag_corrs"])
# print(outputs["eda"]["km_quantiles"])
# print(outputs["eda"]["monthly_kpis"])


print("===========================================")
print("===== RAW Data Overview =====")
print(outputs["raw"])
print("===== PROCESSDED DATA =====")
print(outputs["typed"])
print("===== DAILY DATA =====")
print(outputs["daily"])
print("===== BACKLOG DATA =====")
print(outputs["backlog"])
print("===== EVENTS DATA =====")
print(outputs["events"])
print("===== BACKLOG TIME SERIES DATA =====")
print(outputs["backlog_ts"])
print("===== TIME SERIES DATA =====")
print(outputs["di"])
print("===== TREND =====")
print(outputs["trend"])
print("===== TREND ANALYSIS =====")
print(outputs["trend_all"])
print("===========================================")
print("===== PLOTS =====")
print(outputs["plots"])
