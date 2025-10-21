#!/usr/bin/env python3
"""
Build a per‑investigator daily panel and backlog series from an OPG investigations raw extract.

Input: a CSV or XLSX/XLS with columns like (case‑insensitive, whitespace ignored):
    ID, LPA or DeputyID, Investigator, Investigator FTE, Team, Reallocated Case, Weighting,
    Client/Donor Title, Risk, Case Type, Concern Type, Status, Sub Status,
    Date Received in OPG, Date Received in Investigations, Date allocated to team,
    Date allocated to current investigator, Anticipated completion date, PG Sign off date,
    Days on Hold, Currently hold from, Multiple ID, Lead Case, Days to PG sign off,
    Closure Date, PG Sign off Hold days, PG sign off to Close days, Last Status,
    Referals made by ITAS, High Risk From, Days at high risk, Recommended Court Outcome,
    Date of Legal Review Request 1, Date Legal Rejects 1, Reason For Rejection 1, Legal Risk Rejection 1,
    Date of Legal Review Request 2, Date Legal Rejects 2, Reason For Rejection 2, Legal Risk Rejection 2,
    Date of Legel Review Request 3, LCR Request No, Times Lawyers Allocated (>1 reallocated case),
    Legal Approval Date, Legal Risk, Date Sent To CA, CA Acceptance Type, Lawyer, Allocated to Solicitor Date,
    Legal Team, PG Signatory, Court Outcome, Date Of Order, PGS Addendum Date, Flagged Date, Flagged Type,
    Flag Lawyer, Day 40 Review Date, Day 40 Review Completion Date, Day 70 Review Date, Day 70 Review Completion Date,
    Day 100 Review Date, Day 100 Review Completion Date, Further Review Date, No. Of Further Reviews, Concern Received From

Outputs:
  - investigator_daily.csv with columns:
        date, staff_id, team, role, FTE, is_new_starter, weeks_since_start,
        wip, time_since_last_pickup, mentoring_flag, trainee_flag,
        backlog_available, term_flag, season, dow, bank_holiday,
        event_newcase, event_legal, event_court
  - backlog_series.csv with columns: date, backlog_available
  - event_log.csv (one row per event with case_id + event type + owner on that day)

Usage (CLI):
    python build_investigator_daily_from_raw.py --in /path/to/raw.csv --outdir ./out
"""

from __future__ import annotations
import argparse, sys, re, hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

# ----------------------- Utilities -----------------------

NULL_STRINGS = {
    "", "na", "n/a", "none", "null", "-", "--", "unknown",
    "not completed", "not complete", "tbc", "n\\a"
}

DATE_COLS_RAW = [
    "Date Received in OPG",
    "Date Received in Investigations",
    "Date allocated to team",
    "Date allocated to current investigator",
    "Anticipated completion date",
    "PG Sign off date",
    "Closure Date",
    "High Risk From",
    "Date of Legal Review Request 1",
    "Date Legal Rejects 1",
    "Date of Legal Review Request 2",
    "Date Legal Rejects 2",
    "Date of Legel Review Request 3",
    "Legal Approval Date",
    "Date Sent To CA",
    "Allocated to Solicitor Date",
    "Date Of Order",
    "Flagged Date",
    "Day 40 Review Date",
    "Day 40 Review Completion Date",
    "Day 70 Review Date",
    "Day 70 Review Completion Date",
    "Day 100 Review Date",
    "Day 100 Review Completion Date",
    "Further Review Date",
    "Anticipated completion date"
]

def normalise_col(c: str) -> str:
    """Normalise column name for robust matching (lowercase, strip, collapse spaces)."""
    return re.sub(r"\s+", " ", str(c).strip().lower())

def parse_date_series(s: pd.Series) -> pd.Series:
    """Parse messy UK-format date strings to pandas datetime (UTC-naive)."""
    def _parse_one(x):
        if pd.isna(x): return pd.NaT
        xs = str(x).strip().lower()
        if xs in NULL_STRINGS: return pd.NaT
        # Remove ordinal suffixes (1st, 2nd, 3rd, 4th)
        xs = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', xs)
        # Common typo fix
        xs = xs.replace("legel", "legal")
        # Try pandas to_datetime with dayfirst
        try:
            return pd.to_datetime(xs, dayfirst=True, errors="raise")
        except Exception:
            # Try split by space/slash/dash
            try:
                return pd.to_datetime(xs, infer_datetime_format=True, dayfirst=True, errors="coerce")
            except Exception:
                return pd.NaT
    return s.apply(_parse_one)

def hash_id(text: str) -> str:
    """Anonymise staff names deterministically but reversibly if you keep a lookup elsewhere."""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    return "S" + hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8]

def month_to_season(m: int) -> str:
    return {12:"winter",1:"winter",2:"winter",
            3:"spring",4:"spring",5:"spring",
            6:"summer",7:"summer",8:"summer",
            9:"autumn",10:"autumn",11:"autumn"}[int(m)]

def is_term_month(m: int) -> int:
    # Crude proxy: term in Jan–Jul, Sep–Dec; off-term in Aug
    return 0 if int(m) == 8 else 1

# ------------------------ Core ---------------------------

def load_raw(path: Path) -> pd.DataFrame:
    """Load CSV/XLSX, keep strings, and trim whitespace."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, sep=None, engine="python")
    # Strip whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Build a normalised column map
    colmap = {normalise_col(c): c for c in df.columns}
    return df, colmap

def col(df: pd.DataFrame, colmap: Dict[str,str], name: str) -> pd.Series:
    """Get a column by tolerant/normalised name; return empty Series if missing."""
    key = normalise_col(name)
    if key in colmap:
        return df[colmap[key]]
    # Try some fallbacks for typos
    for k,v in colmap.items():
        if key in k or k in key:
            return df[v]
    return pd.Series([np.nan]*len(df))

def engineer(df: pd.DataFrame, colmap: Dict[str,str]) -> pd.DataFrame:
    """Create typed fields needed for events and intervals."""
    out = pd.DataFrame({
        "case_id": col(df, colmap, "ID"),
        "investigator": col(df, colmap, "Investigator"),
        "team": col(df, colmap, "Team"),
        "fte": pd.to_numeric(col(df, colmap, "Investigator FTE"), errors="coerce"),
    })
    # Dates
    out["dt_received_inv"]   = parse_date_series(col(df, colmap, "Date Received in Investigations"))
    out["dt_alloc_invest"]   = parse_date_series(col(df, colmap, "Date allocated to current investigator"))
    out["dt_alloc_team"]     = parse_date_series(col(df, colmap, "Date allocated to team"))
    out["dt_pg_signoff"]     = parse_date_series(col(df, colmap, "PG Sign off date"))
    out["dt_close"]          = parse_date_series(col(df, colmap, "Closure Date"))
    out["dt_legal_req_1"]    = parse_date_series(col(df, colmap, "Date of Legal Review Request 1"))
    out["dt_legal_rej_1"]    = parse_date_series(col(df, colmap, "Date Legal Rejects 1"))
    out["dt_legal_req_2"]    = parse_date_series(col(df, colmap, "Date of Legal Review Request 2"))
    out["dt_legal_rej_2"]    = parse_date_series(col(df, colmap, "Date Legal Rejects 2"))
    out["dt_legal_req_3"]    = parse_date_series(col(df, colmap, "Date of Legel Review Request 3"))
    out["dt_legal_approval"] = parse_date_series(col(df, colmap, "Legal Approval Date"))
    out["dt_date_of_order"]  = parse_date_series(col(df, colmap, "Date Of Order"))
    out["dt_flagged"]        = parse_date_series(col(df, colmap, "Flagged Date"))
    # Fill FTE default
    out["fte"] = out["fte"].fillna(1.0)
    # Staff ID (hashed) to avoid PII leakage
    out["staff_id"] = out["investigator"].apply(hash_id)
    # Role not present in data -> set blank
    out["role"] = ""
    return out

def date_horizon(typed: pd.DataFrame, pad_days: int = 14) -> Tuple[pd.Timestamp,pd.Timestamp]:
    """Pick a sensible horizon from min(received/alloc) to max(close/signoff/order)+pad."""
    start = pd.concat([typed["dt_received_inv"], typed["dt_alloc_invest"], typed["dt_alloc_team"]]).min()
    end   = pd.concat([typed["dt_close"], typed["dt_pg_signoff"], typed["dt_date_of_order"]]).max()
    if pd.isna(start):
        start = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
    if pd.isna(end):
        end = pd.Timestamp.today().normalize()
    end = end + pd.Timedelta(days=pad_days)
    return start.normalize(), end.normalize()

def build_event_log(typed: pd.DataFrame) -> pd.DataFrame:
    """Explode date columns into a tidy event log per case/staff/date."""
    records = []
    for _, r in typed.iterrows():
        sid = r["staff_id"]; team = r["team"]; fte = r["fte"]; cid = r["case_id"]
        def add(dt, etype, meta=None):
            if pd.isna(dt): return
            records.append({
                "date": dt.normalize(),
                "staff_id": sid,
                "team": team,
                "fte": fte,
                "case_id": cid,
                "event": etype,
                "meta": meta or ""
            })
        add(r["dt_alloc_invest"], "newcase")
        # Legal markers
        add(r["dt_legal_req_1"], "legal_request")
        add(r["dt_legal_req_2"], "legal_request")
        add(r["dt_legal_req_3"], "legal_request")
        add(r["dt_legal_approval"], "legal_approval")
        # Court marker
        add(r["dt_date_of_order"], "court_order")
    ev = pd.DataFrame.from_records(records)
    if ev.empty:
        ev = pd.DataFrame(columns=["date","staff_id","team","fte","case_id","event","meta"])
    return ev

def build_wip_series(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Line-sweep to compute daily WIP per staff: +1 at allocation, -1 at end+1."""
    # Define end date per case: closure > signoff > horizon end
    end_dt = typed["dt_close"].fillna(typed["dt_pg_signoff"]).fillna(end)
    starts = typed[["staff_id","team","dt_alloc_invest"]].dropna()
    intervals = pd.DataFrame({
        "staff_id": typed["staff_id"],
        "team": typed["team"],
        "start": typed["dt_alloc_invest"],
        "end": end_dt
    }).dropna()
    # Build deltas
    deltas = []
    for _, r in intervals.iterrows():
        s = r["start"].normalize(); e = r["end"].normalize()
        if s > end or e < start: 
            continue
        s = max(s, start); e = min(e, end)
        deltas.append((r["staff_id"], r["team"], s,  1))
        deltas.append((r["staff_id"], r["team"], e + pd.Timedelta(days=1), -1))
    if not deltas:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"date": idx, "staff_id": [], "team": [], "wip": []}).head(0)
    deltas = pd.DataFrame(deltas, columns=["staff_id","team","date","delta"])
    # Cum-sum per staff across dates
    all_dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    out_rows = []
    for (sid, team), g in deltas.groupby(["staff_id","team"]):
        gg = g.groupby("date", as_index=False)["delta"].sum()
        grid = all_dates.merge(gg, on="date", how="left").fillna({"delta":0})
        grid["wip"] = grid["delta"].cumsum()
        grid["staff_id"] = sid
        grid["team"] = team
        out_rows.append(grid[["date","staff_id","team","wip"]])
    wip = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["date","staff_id","team","wip"])
    return wip

def build_backlog_series(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Approximate central backlog = cumulative accepted (received in investigations) minus allocations to investigators."""
    accepted = typed[["dt_received_inv"]].dropna().assign(date=lambda d: d["dt_received_inv"].dt.normalize())["date"].value_counts().sort_index()
    allocated = typed[["dt_alloc_invest"]].dropna().assign(date=lambda d: d["dt_alloc_invest"].dt.normalize())["date"].value_counts().sort_index()
    idx = pd.date_range(start, end, freq="D")
    acc = accepted.reindex(idx, fill_value=0).cumsum()
    allo = allocated.reindex(idx, fill_value=0).cumsum()
    backlog = (acc - allo).rename("backlog_available").to_frame()
    backlog.index.name = "date"
    backlog = backlog.reset_index()
    return backlog

def build_daily_panel(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Combine WIP, event log, and simple calendar features into the Stage‑2 daily panel."""
    ev = build_event_log(typed)
    wip = build_wip_series(typed, start, end)
    backlog = build_backlog_series(typed, start, end)

    # Base grid: all staff x all dates appearing in either WIP or events
    staff = typed[["staff_id","team","role","fte"]].drop_duplicates()
    dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    grid = dates.assign(key=1).merge(staff.assign(key=1), on="key").drop(columns="key")

    # Merge WIP
    grid = grid.merge(wip, on=["date","staff_id","team"], how="left").fillna({"wip":0})

    # Events -> pivot to daily flags
    if not ev.empty:
        ev_flags = (ev
            .assign(flag=lambda d: 1)
            .pivot_table(index=["date","staff_id"],
                         columns="event",
                         values="flag",
                         aggfunc="max")
            .reset_index()
            .rename_axis(None, axis=1)
            )
        grid = grid.merge(ev_flags, on=["date","staff_id"], how="left")
    for c in ["newcase","legal_request","legal_approval","court_order"]:
        if c not in grid:
            grid[c] = 0
        else:
            grid[c] = grid[c].fillna(0).astype(int)

    # time_since_last_pickup per staff (based on newcase flag)
    grid = grid.sort_values(["staff_id","date"])
    grid["time_since_last_pickup"] = grid.groupby("staff_id")["newcase"].apply(
        lambda x: x.groupby((x==1).cumsum()).cumcount()
    )
    # For first run-in where no pickup has happened yet, set to large sentinel (e.g., 99)
    mask_no_pickups = grid.groupby("staff_id")["newcase"].transform("sum") == 0
    grid.loc[mask_no_pickups, "time_since_last_pickup"] = 99

    # Simple calendar
    grid["dow"] = grid["date"].dt.day_name().str[:3]
    grid["season"] = grid["date"].dt.month.map(month_to_season)
    grid["term_flag"] = grid["date"].dt.month.map(is_term_month).astype(int)
    grid["bank_holiday"] = 0  # can be filled later from a calendar

    # New starter heuristic: first 28 days since first allocation in data window
    first_alloc = (typed.dropna(subset=["dt_alloc_invest"])
                         .groupby("staff_id")["dt_alloc_invest"].min()
                         .rename("first_alloc"))
    grid = grid.merge(first_alloc, on="staff_id", how="left")
    grid["weeks_since_start"] = ((grid["date"] - grid["first_alloc"]).dt.days // 7).fillna(0).clip(lower=0).astype(int)
    grid["is_new_starter"] = (grid["weeks_since_start"] < 4).astype(int)

    # Mentoring/trainee flags not present -> default 0 (can be merged later from HR)
    grid["mentoring_flag"] = 0
    grid["trainee_flag"] = 0

    # Backlog (same for all staff on a given day)
    grid = grid.merge(backlog, on="date", how="left").fillna({"backlog_available":0}).sort_values(["staff_id","date"])

    # Rename event flags to model columns
    grid["event_newcase"] = grid["newcase"].astype(int)
    grid["event_legal"]   = ((grid["legal_request"] + grid["legal_approval"]) > 0).astype(int)
    grid["event_court"]   = grid["court_order"].astype(int)
    grid = grid.drop(columns=["newcase","legal_request","legal_approval","court_order","first_alloc"])

    # Order columns
    cols = ["date","staff_id","team","role","fte",
            "is_new_starter","weeks_since_start",
            "wip","time_since_last_pickup",
            "mentoring_flag","trainee_flag",
            "backlog_available","term_flag","season","dow","bank_holiday",
            "event_newcase","event_legal","event_court"]
    return grid[cols].sort_values(["staff_id","date"])

# ------------------------- CLI --------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Build per‑investigator daily panel from OPG investigations extract.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to raw CSV/XLSX")
    ap.add_argument("--outdir", dest="outdir", default="./out", help="Output directory")
    args = ap.parse_args(argv)

    in_path = Path(args.in_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, colmap = load_raw(in_path)
    typed = engineer(df, colmap)
    start, end = date_horizon(typed, pad_days=14)

    # Build outputs
    daily = build_daily_panel(typed, start, end)
    backlog = build_backlog_series(typed, start, end)
    events = build_event_log(typed)

    # Write
    daily.to_csv(outdir / "investigator_daily.csv", index=False)
    backlog.to_csv(outdir / "backlog_series.csv", index=False)
    events.to_csv(outdir / "event_log.csv", index=False)

    # Brief report
    print(f"Rows in daily panel: {len(daily):,}")
    print(f"Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")
    print(f"Investigators: {daily['staff_id'].nunique()}")
    print(f"Total new case events: {daily['event_newcase'].sum()}")

if __name__ == "__main__":
    main()
