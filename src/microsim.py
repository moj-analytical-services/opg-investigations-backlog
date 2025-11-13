# src/microsim.py
# Purpose: Create CSV inputs for a downstream discrete-event / micro-simulation.
# Exports arrivals, backlog, staffing, service-time quantiles (KM), and legal-review routing.

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# ---------- ARRIVALS ----------

def arrivals_by_case_type(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Count new cases by time period and case_type.
    - freq: 'D' (daily), 'M' (monthly), 'Y' (yearly).
    """
    d = df.copy()
    d["case_type"] = d.get("case_type", pd.Series(index=d.index, dtype="object")).fillna("Unknown")
    d["date"] = pd.to_datetime(d["date_received_opg"])
    d = d.dropna(subset=["date"])
    d["period"] = d["date"].dt.to_period(freq).dt.to_timestamp()
    out = (
        d.groupby(["period", "case_type"])
         .size().rename("n")
         .reset_index()
         .rename(columns={"period": "date"})
         .sort_values(["date", "case_type"])
    )
    return out

# ---------- STAFFING ----------

def staffing_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily staffing metrics:
    - investigators_on_duty: unique investigators allocated that day
    - n_allocations: count of allocations
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date_allocated_investigator"))
    out = out.dropna(subset=["date"])
    grp = (
        out.groupby(out["date"].dt.floor("D"))
           .agg(investigators_on_duty=("investigator", pd.Series.nunique),
                n_allocations=("id", "count"))
           .reset_index(names="date")
           .sort_values("date")
    )
    return grp

# ---------- BACKLOG (DAILY) ----------

def backlog_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily backlog = number of cases not yet allocated (receipt→allocation window).
    Note: for very large data, switch to a line-sweep algorithm.
    """
    tmp = df.copy()
    tmp["start"] = pd.to_datetime(tmp["date_received_opg"])
    tmp["end"] = pd.to_datetime(tmp["date_allocated_investigator"])
    tmp = tmp.dropna(subset=["start"])
    today = pd.Timestamp.today().normalize()
    tmp["end"] = tmp["end"].fillna(today)
    tmp = tmp[tmp["end"] >= tmp["start"]]

    records = []
    for i, r in tmp.iterrows():
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            records.append({"date": d})
    if not records:
        return pd.DataFrame(columns=["date","backlog"]).astype({"date":"datetime64[ns]"})
    return (pd.DataFrame(records)
              .groupby("date").size().rename("backlog")
              .reset_index()
              .sort_values("date"))

# ---------- SERVICE-TIME QUANTILES (KAPLAN–MEIER) ----------

def km_quantiles_by_group(
    df: pd.DataFrame,
    duration_col: str = "days_to_pg_signoff",
    event_col: str = "event_pg_signoff",
    group_cols: list[str] = ["case_type", "risk"],
    quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
) -> pd.DataFrame:
    """
    Compute service-time quantiles with right-censoring via Kaplan–Meier.
    Falls back to empirical quantiles when KM fails (tiny groups).
    """
    d = df.copy()
    d["duration"] = pd.to_numeric(d[duration_col], errors="coerce")
    d["event"] = d[event_col].astype(int)
    for c in group_cols:
        if c not in d.columns:
            d[c] = "Unknown"
        d[c] = d[c].fillna("Unknown")

    rows = []
    for keys, g in d.dropna(subset=["duration"]).groupby(group_cols):
        # Skip empty or degenerate groups
        if len(g) < 10:
            # tiny group: use simple empirical quantiles of observed events
            obs = g.loc[g["event"] == 1, "duration"]
            if len(obs) == 0:
                continue
            qs = np.quantile(obs, quantiles)
            row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
            row.update({f"q{int(q*100)}": float(qv) for q, qv in zip(quantiles, qs)})
            row["n"] = int(len(g)); row["events"] = int(g["event"].sum())
            rows.append(row)
            continue

        kmf = KaplanMeierFitter()
        try:
            kmf.fit(durations=g["duration"], event_observed=g["event"])
            row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
            for q in quantiles:
                # lifelines' percentile: timeline where S(t) <= 1-q
                t = kmf.percentile(100 * q)
                row[f"q{int(q*100)}"] = float(t) if np.isfinite(t) else np.nan
            row["n"] = int(len(g)); row["events"] = int(g["event"].sum())
            rows.append(row)
        except Exception:
            # Fallback: empirical on observed events
            obs = g.loc[g["event"] == 1, "duration"]
            if len(obs) == 0:
                continue
            qs = np.quantile(obs, quantiles)
            row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
            row.update({f"q{int(q*100)}": float(qv) for q, qv in zip(quantiles, qs)})
            row["n"] = int(len(g)); row["events"] = int(g["event"].sum())
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=group_cols + [f"q{int(q*100)}" for q in quantiles] + ["n","events"])
    return pd.DataFrame(rows).sort_values(group_cols)

# ---------- ROUTING PROBABILITIES (LEGAL REVIEW) ----------

def legal_review_routing(df: pd.DataFrame, group_cols: list[str] = ["case_type", "risk"]) -> pd.DataFrame:
    """
    Estimate P(legal_review=1) by group to drive branching in the simulation.
    """
    d = df.copy()
    if "needs_legal_review" not in d.columns:
        # derive from presence of legal dates if not precomputed
        has_legal = pd.Series(False, index=d.index)
        for c in d.columns:
            if str(c).lower().startswith("date_legal"):
                has_legal |= d[c].notna()
        if "legal_approval_date" in d.columns:
            has_legal |= d["legal_approval_date"].notna()
        d["needs_legal_review"] = has_legal.astype(int)
    for c in group_cols:
        if c not in d.columns:
            d[c] = "Unknown"
        d[c] = d[c].fillna("Unknown")

    out = (
        d.groupby(group_cols)["needs_legal_review"]
         .agg(rate="mean", n="size")
         .reset_index()
         .sort_values(group_cols)
    )
    return out

# ---------- BUNDLE WRITER ----------

def write_microsim_bundle(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    Build and write the full set of CSVs required by the micro-sim model.
    Returns a small dict of summary stats and writes a metadata.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Arrivals
    a_daily = arrivals_by_case_type(df, "D")
    a_month = arrivals_by_case_type(df, "M")
    a_daily.to_csv(out_dir / "arrivals_daily.csv", index=False)
    a_month.to_csv(out_dir / "arrivals_monthly.csv", index=False)

    # 2) Backlog (daily)
    b_daily = backlog_daily(df)
    b_daily.to_csv(out_dir / "backlog_daily.csv", index=False)

    # 3) Staffing (daily)
    s_daily = staffing_daily(df)
    s_daily.to_csv(out_dir / "staffing_daily.csv", index=False)

    # 4) Service-time quantiles by case_type x risk (receipt → PG sign-off)
    if "days_to_pg_signoff" in df.columns and "event_pg_signoff" in df.columns:
        km_q = km_quantiles_by_group(df, "days_to_pg_signoff", "event_pg_signoff",
                                     group_cols=["case_type","risk"])
        km_q.to_csv(out_dir / "service_time_quantiles_pg_signoff.csv", index=False)
    else:
        km_q = pd.DataFrame()

    # 5) Routing to legal review by case_type x risk
    route = legal_review_routing(df, ["case_type","risk"])
    route.to_csv(out_dir / "routing_legal_review.csv", index=False)

    # 6) Metadata (for auditability)
    meta = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_rows_input": int(len(df)),
        "files": {
            "arrivals_daily": int(len(a_daily)),
            "arrivals_monthly": int(len(a_month)),
            "backlog_daily": int(len(b_daily)),
            "staffing_daily": int(len(s_daily)),
            "service_time_quantiles_pg_signoff": int(len(km_q)) if not km_q.empty else 0,
            "routing_legal_review": int(len(route)),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta
