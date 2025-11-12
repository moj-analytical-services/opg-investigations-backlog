from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORICAL_NULL = "Unknown"

DATE_COLS = [
    "date_received_opg",
    "date_received_investigations",
    "date_allocated_team",
    "date_allocated_investigator",
    "pg_signoff_date",
    "closure_date",
    "legal_approval_date",
]

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    df = _parse_dates(df)
    if "reallocation" in df.columns:
        df["reallocation"] = (
            df["reallocation"].fillna("No").astype(str).str.lower().isin(["yes","y","true","1"])
        )
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].fillna(CATEGORICAL_NULL).replace("", CATEGORICAL_NULL)
    if "weighting" in df.columns:
        df["weighting"] = pd.to_numeric(df["weighting"], errors="coerce").fillna(0).clip(0,5)
    return df

def engineer_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"date_received_opg","date_allocated_investigator"} <= set(df.columns):
        df["days_to_alloc"] = (df["date_allocated_investigator"] - df["date_received_opg"]).dt.days
    else:
        df["days_to_alloc"] = np.nan
    if {"pg_signoff_date","date_received_opg"} <= set(df.columns):
        df["days_to_pg_signoff"] = (df["pg_signoff_date"] - df["date_received_opg"]).dt.days
    else:
        df["days_to_pg_signoff"] = np.nan
    df["event_pg_signoff"] = df["pg_signoff_date"].notna().astype(int)
    if "status" in df.columns:
        df["is_backlog"] = df["status"].str.lower().isin(["awaiting_investigator","to_be_allocated"]).astype(int)
    else:
        df["is_backlog"] = 0
    # Legal review indicator: any legal-related date present
    has_legal_cols = [c for c in df.columns if c.startswith("date_legal")]
    has_any_legal = df[has_legal_cols].notna().any(axis=1) if has_legal_cols else pd.Series(False, index=df.index)
    if "legal_approval_date" in df.columns:
        has_any_legal = has_any_legal | df["legal_approval_date"].notna()
    df["needs_legal_review"] = has_any_legal.astype(int)
    return df

def daily_backlog_series(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["start"] = pd.to_datetime(tmp.get("date_received_opg"))
    tmp["end"] = pd.to_datetime(tmp.get("date_allocated_investigator"))
    tmp = tmp.dropna(subset=["start"])
    today = pd.Timestamp.today().normalize()
    tmp["end"] = tmp["end"].fillna(today)
    records = []
    for i, r in tmp.iterrows():
        if r["end"] < r["start"]:
            continue
        days = pd.date_range(r["start"], r["end"], freq="D")
        for d in days:
            records.append({"date": d, "case_id": r.get("id", i)})
    if not records:
        return pd.DataFrame(columns=["date","backlog"])
    daily = pd.DataFrame(records).groupby("date").size().rename("backlog").reset_index()
    return daily

def aggregate_staffing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date_allocated_investigator")).dt.date
    out = out.dropna(subset=["date"])
    grp = out.groupby(pd.to_datetime(out["date"])).agg(
        investigators_on_duty=("investigator", pd.Series.nunique),
        n_allocations=("id", "count"),
    ).reset_index(names="date")
    return grp

def generate_synthetic(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2022-01-01")
    case_types = ["Investigation", "TPO", "Fraud", "Complaint", "Reg 40", "Aspect"]
    teams = ["CW1", "CW2", "ClusterA", "ClusterB", "ClusterC"]
    investigators = [f"INV_{i:03d}" for i in range(1, 120)]
    risks = ["Low", "Medium", "High", "Very High", "Normal"]
    status_states = ["To be allocated", "Awaiting investigator", "Investigation Phase", "Closed", "Further action", "No further action"]
    df = pd.DataFrame({
        "id": np.arange(1, n+1),
        "case_no": rng.integers(10_000_000, 99_999_999, size=n),
        "investigator": rng.choice(investigators, size=n),
        "team": rng.choice(teams, size=n),
        "reallocation": rng.choice([True, False], size=n, p=[0.15, 0.85]),
        "weighting": rng.choice([0,1,2,3,4], size=n, p=[0.1,0.25,0.2,0.4,0.05]),
        "risk": rng.choice(risks, size=n, p=[0.15,0.45,0.25,0.05,0.10]),
        "case_type": rng.choice(case_types, size=n, p=[0.6,0.1,0.05,0.1,0.1,0.05]),
        "status": rng.choice(status_states, size=n, p=[0.05,0.25,0.4,0.2,0.05,0.05]),
    })
    start_offsets = rng.integers(0, 1200, size=n)
    df["date_received_opg"] = base_date + pd.to_timedelta(start_offsets, unit="D")
    alloc_delay = np.maximum(0, rng.normal(loc=25 - 2*df["weighting"], scale=10, size=n)).astype(int)
    df["date_allocated_investigator"] = df["date_received_opg"] + pd.to_timedelta(alloc_delay, unit="D")
    ct_factor = df["case_type"].map({"Fraud": 40, "TPO": 35, "Complaint": 25, "Reg 40": 30, "Aspect": 28, "Investigation": 22}).fillna(22)
    risk_factor = df["risk"].map({"Very High": 20, "High": 15, "Medium": 10, "Low": 5, "Normal": 7}).fillna(7)
    signoff_delay = np.maximum(0, rng.normal(loc=ct_factor + risk_factor, scale=10)).astype(int)
    df["pg_signoff_date"] = df["date_allocated_investigator"] + pd.to_timedelta(signoff_delay, unit="D")
    close_delay = np.maximum(0, rng.normal(loc=10, scale=5, size=n)).astype(int)
    df["closure_date"] = df["pg_signoff_date"] + pd.to_timedelta(close_delay, unit="D")
    mask_open = rng.random(n) < 0.2
    df.loc[mask_open, ["pg_signoff_date", "closure_date"]] = pd.NaT
    needs_legal = rng.random(n) < df["case_type"].map({"TPO": 0.5, "Fraud": 0.4, "Reg 40": 0.3, "Complaint": 0.2, "Aspect": 0.15, "Investigation": 0.1}).values
    approve_delay = np.maximum(0, rng.normal(loc=20, scale=7, size=n)).astype(int)
    df["legal_approval_date"] = pd.NaT
    if needs_legal.sum() > 0:
        df.loc[needs_legal, "legal_approval_date"] = df.loc[needs_legal, "date_allocated_investigator"] + pd.to_timedelta(approve_delay[:needs_legal.sum()], unit="D")
    return df
