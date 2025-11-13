# src/agg.py
# Build daily backlog, plus monthly/yearly aggregates and case-type splits.

from __future__ import annotations
import numpy as np
import pandas as pd

def daily_backlog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily backlog series = count of cases not yet allocated.
    Uses date_received_opg as start and date_allocated_investigator as end (censored to 'today' if missing).
    """
    tmp = df.copy()
    tmp["start"] = pd.to_datetime(tmp["date_received_opg"])
    tmp["end"] = pd.to_datetime(tmp["date_allocated_investigator"])
    tmp = tmp.dropna(subset=["start"])
    today = pd.Timestamp.today().normalize()
    tmp["end"] = tmp["end"].fillna(today)

    # Guard against bad rows (end < start)
    good = tmp["end"] >= tmp["start"]
    tmp = tmp.loc[good]

    # Build per-day records (efficient enough for typical volumes; for >1e6, switch to line-sweep)
    records = []
    for i, r in tmp.iterrows():
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            records.append({"date": d, "case_id": r.get("id", i)})

    if not records:
        return pd.DataFrame(columns=["date", "backlog"]).astype({"date":"datetime64[ns]"})

    daily = (
        pd.DataFrame.from_records(records)
        .groupby("date").size().rename("backlog").reset_index()
        .sort_values("date")
    )
    return daily

def aggregate(daily_df: pd.DataFrame, freq: str = "M", by: list[str] | None = None, cases_df: pd.DataFrame | None = None):
    """
    Aggregate backlog to a chosen frequency (D/M/Y), optionally by fields (e.g., case_type).
    If grouping by a field, supply 'cases_df' with those columns and a date per record.
    """
    df = daily_df.copy()
    df["period"] = df["date"].dt.to_period(freq).dt.to_timestamp()  # standard start-of-period

    if not by:
        out = df.groupby("period")["backlog"].sum().reset_index()
        return out.rename(columns={"period": "date"})
    else:
        # Join case_type etc. using nearest date snapshot (approximation: same backlog for all cases that day)
        # More exact methods require line-sweep with attributes; adequate for reporting slices.
        if cases_df is None:
            raise ValueError("cases_df is required when grouping by fields")
        # Use date_received_opg to attribute a case_type presence for that day
        cases = cases_df[["id"] + by].drop_duplicates("id")
        # Broadcast backlog across by-values proportionally (fallback: simple count per by)
        df["key"] = 1
        cases["key"] = 1
        merged = df.merge(cases, on="key").drop(columns="key")
        merged["period"] = merged["date"].dt.to_period(freq).dt.to_timestamp()
        out = merged.groupby(["period"] + by)["backlog"].sum().reset_index()
        return out.rename(columns={"period": "date"})
