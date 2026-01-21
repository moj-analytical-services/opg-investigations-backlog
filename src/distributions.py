"""Distributions of interval changes over multiple years by case_type."""

import pandas as pd


def interval_change_distribution(
    interval_df: pd.DataFrame,
    interval_col: str = "days_to_alloc",
    date_col: str = "date_received_opg",
    group: str = "case_type",
) -> dict:

    df = interval_df.copy()

    # Prefer the explicit date_col if it exists
    if date_col in df.columns:
        df["year"] = pd.to_datetime(df[date_col]).dt.year
    elif "date_received_opg" in df.columns:
        df["year"] = pd.to_datetime(df["date_received_opg"]).dt.year
    elif "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
    else:
        raise ValueError(
            f"Column '{date_col}' not found, and neither 'date_received_opg' nor 'date' are present to derive year."
        )

    df = df.dropna(subset=[interval_col])
    summary = (
        df.groupby([group, "year"])[interval_col]
        .agg(
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            mean="mean",
            n="size",
        )
        .reset_index()
        .sort_values([group, "year"])
    )
    yoy = (
        summary.pivot(index="year", columns=group, values="median")
        .diff()
        .stack()
        .reset_index()
        .rename(columns={0: "yoy_median_change"})
    )
    return {"annual_stats": summary, "yoy_change": yoy}
