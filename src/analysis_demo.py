"""Demo: last‑year interval analysis by team (non-invasive)."""

import pandas as pd

try:
    from .notebook_code import IntervalAnalysis
except Exception:
    IntervalAnalysis = None


def last_year_by_team(eng_df, backlog_series=None, bank_holidays=None):
    if IntervalAnalysis is None:
        raise ImportError("IntervalAnalysis not found in notebook code.")
    di = IntervalAnalysis.build_interval_frame(
        eng_df, backlog_series=backlog_series, bank_holidays=bank_holidays
    )
    trend = IntervalAnalysis.monthly_trend_last_year(
        di, metric="days_to_pg_signoff", agg="median", by=["team"]
    ).copy()
    trend["month"] = pd.to_datetime(trend["yyyymm"] + "-01")
    return trend[["month", "team", "days_to_pg_signoff"]]
