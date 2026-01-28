# -*- coding: utf-8 -*-
# build_investigator_daily_from_raw_annotated.py
# Auto-generated on 2025-10-27 10:32:55Z (UTC)
# This script is a linearized and *commented* version of the notebook
# `Build_Investigator_Daily_from_Raw_JL_27_10_25.ipynb`.
# Each original line of code is preceded by a short explanatory comment.

if __name__ == "__main__":
    # Entry point if you run this file as a script
    # The code below mirrors the notebook cells in order.
    pass


# === Notebook code cell starts ===
# Original comment
#!python -m venv .venv && . .venv/bin/activate
# === Notebook code cell ends ===

# === Notebook code cell starts ===
# Import libraries/modules for use below
from pathlib import Path

# Import libraries/modules for use below
import pandas as pd
import numpy as np

# Import libraries/modules for use below
import re
import hashlib

# Blank line

# Original comment
# Configure paths
# Execute the following statement
RAW_PATH = Path("data/raw/raw.csv")
# Execute the following statement
OUT_DIR = Path("data/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Print a message or value
print(RAW_PATH.exists(), OUT_DIR)
# === Notebook code cell ends ===

# === Notebook code cell starts ===
# Blank line

# Execute the following statement
NULL_STRINGS = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "-",
    "--",
    "unknown",
    "not completed",
    "not complete",
    "tbc",
    "n\\a",
}


# Return a value from a function
def normalise_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


# Define a reusable function
def parse_date_series(s: pd.Series) -> pd.Series:
    # Define a reusable function
    def _p(x):
        # Import libraries/modules for use below
        import pandas as pd

        # Use pandas functionality
        if pd.isna(x):
            return pd.NaT
        # Execute the following statement
        xs = str(x).strip().lower()
        # Use pandas functionality
        if xs in NULL_STRINGS:
            return pd.NaT
        # Execute the following statement
        xs = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", xs).replace("legel", "legal")
        # Convert values to pandas datetime
        try:
            return pd.to_datetime(xs, dayfirst=True, errors="raise")
        # Convert values to pandas datetime
        except:
            return pd.to_datetime(
                xs, infer_datetime_format=True, dayfirst=True, errors="coerce"
            )

    # Return a value from a function
    return s.apply(_p)


# Define a reusable function
def hash_id(t: str) -> str:
    # Import libraries/modules for use below
    import pandas as pd

    # Use pandas functionality
    if pd.isna(t) or str(t).strip() == "":
        return ""
    # Return a value from a function
    return "S" + hashlib.sha1(str(t).encode("utf-8")).hexdigest()[:8]


# Return a value from a function
def month_to_season(m: int) -> str:
    return {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }[int(m)]


# Return a value from a function
def is_term_month(m: int) -> int:
    return 0 if int(m) == 8 else 1


# === Notebook code cell ends ===


# === Notebook code cell starts ===
# Original comment
# def load_raw(p:Path):
# Original comment
#     import pandas as pd
# Original comment
#     if p.suffix.lower() in ('.xlsx','.xls'): df=pd.read_excel(p,dtype=str)
# Original comment
#     else: df=pd.read_csv(p,dtype=str, sep=None, engine='python')
# Original comment
#     df=df.applymap(lambda x: x.strip() if isinstance(x,str) else x)
# Original comment
#     colmap={normalise_col(c):c for c in df.columns}; return df, colmap
# Define a reusable function
def load_raw(p: Path, force_encoding: str | None = None):
    # Execute the following statement
    """
    # Execute the following statement
        Load CSV/XLSX with robust encoding handling.
    # Execute the following statement
        - If force_encoding is given, use it.
    # Execute the following statement
        - Otherwise try common encodings in order and fall back to a safe decode.
    # Execute the following statement
        Returns: (df, colmap)
    # Execute the following statement
    """
    # Import libraries/modules for use below
    import pandas as pd
    import re

    # Blank line

    # Conditional branch
    if not p.exists():
        # Execute the following statement
        raise FileNotFoundError(p)
    # Blank line

    # Original comment
    # Excel files are not affected by CSV encoding issues
    # Conditional branch
    if p.suffix.lower() in (".xlsx", ".xls"):
        # Load an Excel sheet into a DataFrame
        df = pd.read_excel(p, dtype=str)
    # Fallback branch
    else:
        # Execute the following statement
        tried = []
        # Execute the following statement
        encodings_to_try = (
            # Execute the following statement
            [force_encoding]
            if force_encoding
            else
            # Execute the following statement
            [
                "utf-8-sig",
                "cp1252",
                "latin1",
                "iso-8859-1",
                "utf-16",
                "utf-16le",
                "utf-16be",
            ]
            # Execute the following statement
        )
        # Blank line

        # Execute the following statement
        df = None
        # Execute the following statement
        last_err = None
        # Loop over a sequence
        for enc in encodings_to_try:
            # Try a block of code that may raise errors
            try:
                # Load a CSV file into a DataFrame
                df = pd.read_csv(
                    p,
                    dtype=str,
                    sep=None,
                    engine="python",
                    # Execute the following statement
                    encoding=enc,
                    encoding_errors="strict",
                )
                # Execute the following statement
                break
            # Handle errors from the try block
            except UnicodeDecodeError as e:
                # Execute the following statement
                tried.append(enc)
                last_err = e
            # Handle errors from the try block
            except Exception as e:
                # Original comment
                # Other parse errors (separator/quotes) – keep trying other encodings
                # Execute the following statement
                tried.append(enc)
                last_err = e
        # Blank line

        # Original comment
        # Last-resort: decode with cp1252 but *replace* bad bytes
        # Conditional branch
        if df is None:
            # Try a block of code that may raise errors
            try:
                # Load a CSV file into a DataFrame
                df = pd.read_csv(
                    p,
                    dtype=str,
                    sep=None,
                    engine="python",
                    # Execute the following statement
                    encoding="cp1252",
                    encoding_errors="replace",
                )
                # Print a message or value
                print(
                    f"[load_raw] WARNING: used cp1252 with replacement after failed encodings: {tried}"
                )
            # Handle errors from the try block
            except Exception as e:
                # Execute the following statement
                raise RuntimeError(
                    # Execute the following statement
                    f"Failed to read CSV. Tried encodings {tried}. Last error: {last_err}"
                    # Execute the following statement
                ) from e
    # Blank line

    # Original comment
    # Trim whitespace across all string columns
    # Execute the following statement
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Execute the following statement
    colmap = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in df.columns}
    # Return a value from a function
    return df, colmap


# Blank line

# Blank line


# Define a reusable function
def col(df, colmap, name):
    # Import libraries/modules for use below
    import numpy as np

    # Execute the following statement
    k = normalise_col(name)
    # Return a value from a function
    if k in colmap:
        return df[colmap[k]]
    # Loop over a sequence
    for kk, v in colmap.items():
        # Return a value from a function
        if k in kk or kk in k:
            return df[v]
    # Use NumPy for numeric operations
    return pd.Series([np.nan] * len(df))


# Define a reusable function
def engineer(df, colmap):
    # Import libraries/modules for use below
    import pandas as pd

    # Use pandas functionality
    out = pd.DataFrame(
        {
            "case_id": col(df, colmap, "ID"),
            "investigator": col(df, colmap, "Investigator"),
            "team": col(df, colmap, "Team"),
            "fte": pd.to_numeric(col(df, colmap, "Investigator FTE"), errors="coerce"),
        }
    )
    # Execute the following statement
    out["dt_received_inv"] = parse_date_series(
        col(df, colmap, "Date Received in Investigations")
    )
    # Execute the following statement
    out["dt_alloc_invest"] = parse_date_series(
        col(df, colmap, "Date allocated to current investigator")
    )
    # Execute the following statement
    out["dt_alloc_team"] = parse_date_series(col(df, colmap, "Date allocated to team"))
    # Execute the following statement
    out["dt_pg_signoff"] = parse_date_series(col(df, colmap, "PG Sign off date"))
    # Execute the following statement
    out["dt_close"] = parse_date_series(col(df, colmap, "Closure Date"))
    # Execute the following statement
    out["dt_legal_req_1"] = parse_date_series(
        col(df, colmap, "Date of Legal Review Request 1")
    )
    # Execute the following statement
    out["dt_legal_rej_1"] = parse_date_series(col(df, colmap, "Date Legal Rejects 1"))
    # Execute the following statement
    out["dt_legal_req_2"] = parse_date_series(
        col(df, colmap, "Date of Legal Review Request 2")
    )
    # Execute the following statement
    out["dt_legal_rej_2"] = parse_date_series(col(df, colmap, "Date Legal Rejects 2"))
    # Execute the following statement
    out["dt_legal_req_3"] = parse_date_series(
        col(df, colmap, "Date of Legel Review Request 3")
    )
    # Execute the following statement
    out["dt_legal_approval"] = parse_date_series(col(df, colmap, "Legal Approval Date"))
    # Execute the following statement
    out["dt_date_of_order"] = parse_date_series(col(df, colmap, "Date Of Order"))
    # Execute the following statement
    out["dt_flagged"] = parse_date_series(col(df, colmap, "Flagged Date"))
    # Fill missing values with a default
    out["fte"] = out["fte"].fillna(1.0)
    out["staff_id"] = out["investigator"].apply(hash_id)
    out["role"] = ""
    return out


# Define a reusable function
def date_horizon(typed, pad_days: int = 14):
    # Import libraries/modules for use below
    import pandas as pd

    # Use pandas functionality
    start = pd.concat(
        [typed["dt_received_inv"], typed["dt_alloc_invest"], typed["dt_alloc_team"]]
    ).min()
    # Use pandas functionality
    end = pd.concat(
        [typed["dt_close"], typed["dt_pg_signoff"], typed["dt_date_of_order"]]
    ).max()
    # Use pandas functionality
    if pd.isna(start):
        start = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
    # Use pandas functionality
    if pd.isna(end):
        end = pd.Timestamp.today().normalize()
    # Use pandas functionality
    end = end + pd.Timedelta(days=pad_days)
    return start.normalize(), end.normalize()


# Define a reusable function
def build_event_log(typed):
    # Import libraries/modules for use below
    import pandas as pd

    # Execute the following statement
    rec = []
    # Loop over a sequence
    for _, r in typed.iterrows():
        # Execute the following statement
        sid, team, fte, cid = r["staff_id"], r["team"], r["fte"], r["case_id"]

        # Define a reusable function
        def add(dt, etype):
            # Use pandas functionality
            if pd.isna(dt):
                return
            # Execute the following statement
            rec.append(
                {
                    "date": dt.normalize(),
                    "staff_id": sid,
                    "team": team,
                    "fte": fte,
                    "case_id": cid,
                    "event": etype,
                    "meta": "",
                }
            )

        # Execute the following statement
        add(r["dt_alloc_invest"], "newcase")
        add(r["dt_legal_req_1"], "legal_request")
        add(r["dt_legal_req_2"], "legal_request")
        add(r["dt_legal_req_3"], "legal_request")
        add(r["dt_legal_approval"], "legal_approval")
        add(r["dt_date_of_order"], "court_order")
    # Use pandas functionality
    ev = pd.DataFrame.from_records(rec)
    # Use pandas functionality
    return (
        ev
        if not ev.empty
        else pd.DataFrame(
            columns=["date", "staff_id", "team", "fte", "case_id", "event", "meta"]
        )
    )


# Define a reusable function
def build_wip_series(typed, start, end):
    # Import libraries/modules for use below
    import pandas as pd

    # Fill missing values with a default
    end_dt = typed["dt_close"].fillna(typed["dt_pg_signoff"]).fillna(end)
    # Drop rows with missing values
    intervals = pd.DataFrame(
        {
            "staff_id": typed["staff_id"],
            "team": typed["team"],
            "start": typed["dt_alloc_invest"],
            "end": end_dt,
        }
    ).dropna()
    # Execute the following statement
    deltas = []
    # Loop over a sequence
    for _, r in intervals.iterrows():
        # Execute the following statement
        s = r["start"].normalize()
        e = r["end"].normalize()
        # Execute the following statement
        if s > end or e < start:
            continue
        # Execute the following statement
        s = max(s, start)
        e = min(e, end)
        # Use pandas functionality
        deltas.append((r["staff_id"], r["team"], s, 1))
        deltas.append((r["staff_id"], r["team"], e + pd.Timedelta(days=1), -1))
    # Use pandas functionality
    if not deltas:
        return pd.DataFrame(columns=["date", "staff_id", "team", "wip"])
    # Use pandas functionality
    deltas = pd.DataFrame(deltas, columns=["staff_id", "team", "date", "delta"])
    # Use pandas functionality
    all_dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    # Execute the following statement
    rows = []
    # Group rows and compute aggregations
    for (sid, team), g in deltas.groupby(["staff_id", "team"]):
        # Group rows and compute aggregations
        gg = g.groupby("date", as_index=False)["delta"].sum()
        # Fill missing values with a default
        grid = all_dates.merge(gg, on="date", how="left").fillna({"delta": 0})
        # Execute the following statement
        grid["wip"] = grid["delta"].cumsum()
        grid["staff_id"] = sid
        grid["team"] = team
        rows.append(grid[["date", "staff_id", "team", "wip"]])
    # Use pandas functionality
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "staff_id", "team", "wip"])
    )


# Define a reusable function
def build_backlog_series(typed, start, end):
    # Import libraries/modules for use below
    import pandas as pd

    # Drop rows with missing values
    accepted = (
        typed[["dt_received_inv"]]
        .dropna()
        .assign(date=lambda d: d["dt_received_inv"].dt.normalize())["date"]
        .value_counts()
        .sort_index()
    )
    # Drop rows with missing values
    allocated = (
        typed[["dt_alloc_invest"]]
        .dropna()
        .assign(date=lambda d: d["dt_alloc_invest"].dt.normalize())["date"]
        .value_counts()
        .sort_index()
    )
    # Use pandas functionality
    idx = pd.date_range(start, end, freq="D")
    acc = accepted.reindex(idx, fill_value=0).cumsum()
    allo = allocated.reindex(idx, fill_value=0).cumsum()
    # Rename columns for clarity/consistency
    backlog = (acc - allo).rename("backlog_available").to_frame()
    backlog.index.name = "date"
    return backlog.reset_index()


# Blank line


# Define a reusable function
def build_daily_panel(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    # Execute the following statement
    """Combine WIP, event log, and calendar features. Returns: (daily, backlog, events)."""
    # Execute the following statement
    ev = build_event_log(typed)
    # Execute the following statement
    wip = build_wip_series(typed, start, end)
    # Execute the following statement
    backlog = build_backlog_series(typed, start, end)
    # Blank line

    # Original comment
    # Base grid: all staff x all dates
    # Execute the following statement
    staff = typed[["staff_id", "team", "role", "fte"]].drop_duplicates()
    # Use pandas functionality
    dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    # Combine tables by key columns
    grid = dates.assign(key=1).merge(staff.assign(key=1), on="key").drop(columns="key")
    # Blank line

    # Original comment
    # Merge WIP
    # Fill missing values with a default
    grid = grid.merge(wip, on=["date", "staff_id", "team"], how="left").fillna(
        {"wip": 0}
    )
    # Blank line

    # Original comment
    # Event flags
    # Conditional branch
    if not ev.empty:
        # Execute the following statement
        ev_flags = (
            # Create or transform columns
            ev.assign(flag=1)
            # Execute the following statement
            .pivot_table(
                index=["date", "staff_id"],
                columns="event",
                values="flag",
                aggfunc="max",
            )
            # Reset index to turn group keys into columns
            .reset_index().rename_axis(None, axis=1)
            # Execute the following statement
        )
        # Combine tables by key columns
        grid = grid.merge(ev_flags, on=["date", "staff_id"], how="left")
    # Loop over a sequence
    for c in ["newcase", "legal_request", "legal_approval", "court_order"]:
        # Conditional branch
        if c not in grid:
            # Execute the following statement
            grid[c] = 0
        # Fallback branch
        else:
            # Cast column(s) to a specific dtype
            grid[c] = grid[c].fillna(0).astype(int)
    # Blank line

    # Original comment
    # --- SAFE time_since_last_pickup (no index mismatch) ---
    # Sort rows by specified columns
    grid = grid.sort_values(["staff_id", "date"])
    # Group rows and compute aggregations
    grp = grid.groupby("staff_id", sort=False)
    # Execute the following statement
    runs = grp["newcase"].transform(lambda s: (s == 1).cumsum())
    # Group rows and compute aggregations
    grid["time_since_last_pickup"] = grid.groupby([grid["staff_id"], runs]).cumcount()
    # Execute the following statement
    mask_no_pickups = grp["newcase"].transform("sum") == 0
    # Select/assign rows/columns by label/position
    grid.loc[mask_no_pickups, "time_since_last_pickup"] = 99
    # Blank line

    # Original comment
    # Calendar
    # Execute the following statement
    grid["dow"] = grid["date"].dt.day_name().str[:3]
    # Execute the following statement
    grid["season"] = grid["date"].dt.month.map(month_to_season)
    # Cast column(s) to a specific dtype
    grid["term_flag"] = grid["date"].dt.month.map(is_term_month).astype(int)
    # Execute the following statement
    grid["bank_holiday"] = 0
    # Blank line

    # Original comment
    # New starters
    # Execute the following statement
    first_alloc = (
        # Drop rows with missing values
        typed.dropna(subset=["dt_alloc_invest"])
        # Group rows and compute aggregations
        .groupby("staff_id")["dt_alloc_invest"].min()
        # Rename columns for clarity/consistency
        .rename("first_alloc")
        # Execute the following statement
    )
    # Combine tables by key columns
    grid = grid.merge(first_alloc, on="staff_id", how="left")
    # Execute the following statement
    grid["weeks_since_start"] = (
        (
            # Execute the following statement
            (grid["date"] - grid["first_alloc"]).dt.days
            // 7
            # Cast column(s) to a specific dtype
        )
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )
    # Cast column(s) to a specific dtype
    grid["is_new_starter"] = (grid["weeks_since_start"] < 4).astype(int)
    # Blank line

    # Original comment
    # Default flags
    # Execute the following statement
    grid["mentoring_flag"] = 0
    # Execute the following statement
    grid["trainee_flag"] = 0
    # Blank line

    # Original comment
    # Backlog (same for all staff/day)
    # Fill missing values with a default
    grid = grid.merge(backlog, on="date", how="left").fillna({"backlog_available": 0})
    # Blank line

    # Original comment
    # Final columns
    # Cast column(s) to a specific dtype
    grid["event_newcase"] = grid["newcase"].astype(int)
    # Cast column(s) to a specific dtype
    grid["event_legal"] = ((grid["legal_request"] + grid["legal_approval"]) > 0).astype(
        int
    )
    # Cast column(s) to a specific dtype
    grid["event_court"] = grid["court_order"].astype(int)
    # Execute the following statement
    grid = grid.drop(
        columns=[
            "newcase",
            "legal_request",
            "legal_approval",
            "court_order",
            "first_alloc",
        ]
    )
    # Blank line

    # Execute the following statement
    cols = [
        "date",
        "staff_id",
        "team",
        "role",
        "fte",
        # Execute the following statement
        "is_new_starter",
        "weeks_since_start",
        # Execute the following statement
        "wip",
        "time_since_last_pickup",
        # Execute the following statement
        "mentoring_flag",
        "trainee_flag",
        # Execute the following statement
        "backlog_available",
        "term_flag",
        "season",
        "dow",
        "bank_holiday",
        # Execute the following statement
        "event_newcase",
        "event_legal",
        "event_court",
    ]
    # Reset index to turn group keys into columns
    daily = grid[cols].sort_values(["staff_id", "date"]).reset_index(drop=True)
    # Blank line

    # Original comment
    # <-- IMPORTANT: return the frames
    # Return a value from a function
    return daily, backlog, ev


# Original comment
# def build_daily_panel(typed,start,end):
# Original comment
#     ev=build_event_log(typed); wip=build_wip_series(typed,start,end); backlog=build_backlog_series(typed,start,end)
# Original comment
#     staff=typed[['staff_id','team','role','fte']].drop_duplicates(); dates=pd.DataFrame({'date':pd.date_range(start,end,freq='D')})
# Original comment
#     grid=dates.assign(key=1).merge(staff.assign(key=1), on='key').drop(columns='key')
# Original comment
#     grid=grid.merge(wip, on=['date','staff_id','team'], how='left').fillna({'wip':0})
# Original comment
#     if len(ev):
# Original comment
#         ev_flags=(ev.assign(flag=lambda d:1).pivot_table(index=['date','staff_id'], columns='event', values='flag', aggfunc='max').reset_index().rename_axis(None, axis=1))
# Original comment
#         grid=grid.merge(ev_flags, on=['date','staff_id'], how='left')
# Original comment
#     for c in ['newcase','legal_request','legal_approval','court_order']:
# Original comment
#         if c not in grid: grid[c]=0
# Original comment
#         else: grid[c]=grid[c].fillna(0).astype(int)
# Blank line

# Original comment
#     # grid['time_since_last_pickup']=grid.groupby('staff_id')['newcase'].apply(lambda x: x.groupby((x==1).cumsum()).cumcount())
# Original comment
#     # mask_no=grid.groupby('staff_id')['newcase'].transform('sum')==0; grid.loc[mask_no,'time_since_last_pickup']=99
# Original comment
#     # Ensure rows are ordered
# Original comment
#     grid = grid.sort_values(["staff_id", "date"])
# Original comment
#     # Build a "run id" that increments each time there's a pickup (newcase == 1), per staff
# Original comment
#     grp = grid.groupby("staff_id")
# Original comment
#     runs = grp["newcase"].transform(lambda s: (s == 1).cumsum())
# Original comment
#     # Cumcount within each (staff_id, run) so it resets after each pickup
# Original comment
#     grid["time_since_last_pickup"] = grid.groupby(["staff_id", runs]).cumcount()
# Original comment
#     # For staff who never had a pickup in the window, set a sentinel (e.g., 99)
# Original comment
#     mask_no_pickups = grp["newcase"].transform("sum") == 0
# Original comment
#     grid.loc[mask_no_pickups, "time_since_last_pickup"] = 99
# Blank line

# Original comment
#     grid['dow']=grid['date'].dt.day_name().str[:3]; grid['season']=grid['date'].dt.month.map(month_to_season); grid['term_flag']=grid['date'].dt.month.map(is_term_month).astype(int); grid['bank_holiday']=0
# Original comment
#     first_alloc=(typed.dropna(subset=['dt_alloc_invest']).groupby('staff_id')['dt_alloc_invest'].min().rename('first_alloc'))
# Original comment
#     grid=grid.merge(first_alloc, on='staff_id', how='left'); grid['weeks_since_start']=((grid['date']-grid['first_alloc']).dt.days//7).fillna(0).clip(lower=0).astype(int); grid['is_new_starter']=(grid['weeks_since_start']<4).astype(int)
# Original comment
#     grid['mentoring_flag']=0; grid['trainee_flag']=0
# Original comment
#     grid=grid.merge(backlog, on='date', how='left').fillna({'backlog_available':0}).sort_values(['staff_id','date'])
# Original comment
#     grid['event_newcase']=grid['newcase'].astype(int); grid['event_legal']=((grid['legal_request']+grid['legal_approval'])>0).astype(int); grid['event_court']=grid['court_order'].astype(int)
# Original comment
#     grid=grid.drop(columns=['newcase','legal_request','legal_approval','court_order','first_alloc'])
# Original comment
#     cols=['date','staff_id','team','role','fte','is_new_starter','weeks_since_start','wip','time_since_last_pickup','mentoring_flag','trainee_flag','backlog_available','term_flag','season','dow','bank_holiday','event_newcase','event_legal','event_court']
# Original comment
#     daily=grid[cols].sort_values(['staff_id','date'])
# Original comment
#     daily.to_csv(OUT_DIR / 'investigator_daily.csv', index=False)
# Original comment
#     backlog.to_csv(OUT_DIR / 'backlog_series.csv', index=False)
# Original comment
#     ev.to_csv(OUT_DIR / 'event_log.csv', index=False)
# Original comment
#     daily.head()
# === Notebook code cell ends ===

# === Notebook code cell starts ===
# Load a CSV file into a DataFrame
df_test = pd.read_csv(RAW_PATH, dtype=str, sep=None, engine="python", encoding="cp1252")
# Execute the following statement
df_test.head()
# Blank line

# Execute the following statement
df_raw, colmap = load_raw(RAW_PATH)
# Print a message or value
print("df_raw: ", df_raw)
# Print a message or value
print("colmap: ", df_raw)
# Execute the following statement
typed = engineer(df_raw, colmap)
# Print a message or value
print("typed: ", typed)
# Execute the following statement
start, end = date_horizon(typed, 14)
# Print a message or value
print("start: ", start)
# Print a message or value
print("end: ", end)
# Original comment
# daily = build_daily_panel(typed, start, end)
# Original comment
# print(f"daily: ", daily)
# Original comment
# backlog = pd.read_csv(OUT_DIR / 'backlog_series.csv', parse_dates=['date'])
# Original comment
# events = pd.read_csv(OUT_DIR / 'event_log.csv', parse_dates=['date'])
# Original comment
# print(len(daily), 'daily rows')
# Execute the following statement
daily, backlog, events = build_daily_panel(typed, start, end)
# Blank line

# Original comment
# (optional) save to disk
# Save a DataFrame to CSV
daily.to_csv(OUT_DIR / "investigator_daily.csv", index=False)
# Save a DataFrame to CSV
backlog.to_csv(OUT_DIR / "backlog_series.csv", index=False)
# Save a DataFrame to CSV
events.to_csv(OUT_DIR / "event_log.csv", index=False)
# Blank line

# Print a message or value
print(f"{len(daily):,} daily rows")
# Print a message or value
print("Date range:", daily["date"].min().date(), "→", daily["date"].max().date())
# Print a message or value
print("Investigators:", daily["staff_id"].nunique())
# Print a message or value
print("Total new case events:", int(daily["event_newcase"].sum()))
# === Notebook code cell ends ===

# === Notebook code cell starts ===
# Blank line

# Blank line

# Original comment
# Inputs expected from previous stage:
# Original comment
#   /mnt/data/out/investigator_daily.csv
# Original comment
#   /mnt/data/out/backlog_series.csv
# Original comment
#
# Original comment
# Outputs created:
# Original comment
#   /mnt/data/out/backlog_forecast_bayes.csv              (date, mean, median, p05, p20, p80, p95)
# Original comment
#   /mnt/data/out/investigator_pickup_posterior.csv       (posterior for per-staff pickup rate)
# Blank line

# Blank line

# Import libraries/modules for use below
from pathlib import Path

# Import libraries/modules for use below
from scipy.stats import invgamma, gamma as gamma_dist

# Blank line

# Original comment
# ---- Locations ----
# Execute the following statement
BASE = Path("data")
# Execute the following statement
OUT = BASE / "out"
# Execute the following statement
OUT.mkdir(parents=True, exist_ok=True)
# Execute the following statement
daily_path = OUT / "investigator_daily.csv"
# Execute the following statement
backlog_path = OUT / "backlog_series.csv"
# Blank line

# Original comment
# ---- Load outputs from Stage-2 ----
# Load a CSV file into a DataFrame
daily = pd.read_csv(daily_path, parse_dates=["date"])
# Load a CSV file into a DataFrame
backlog = (
    pd.read_csv(backlog_path, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
# Blank line

# Original comment
# ---- Build daily delta series for backlog ----
# Execute the following statement
backlog["delta"] = backlog["backlog_available"].diff()
# Drop rows with missing values
backlog = backlog.dropna(subset=["delta"]).reset_index(drop=True)
# Blank line

# Original comment
# Design matrix for a conjugate Bayesian linear model:
# Original comment
# y_t = delta_t ~ N(X_t beta, sigma^2), with X_t = [1, lag_delta, sin, cos, DOW dummies]
# Execute the following statement
df = backlog.copy()
# Execute the following statement
df["lag_delta"] = df["delta"].shift(1)
# Drop rows with missing values
df = df.dropna(subset=["lag_delta"]).reset_index(drop=True)
# Blank line

# Original comment
# Weekday effects (Mon=0..Sun=6), drop_first to avoid dummy trap
# Execute the following statement
df["dow"] = df["date"].dt.dayofweek
# Use pandas functionality
dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
# Blank line

# Original comment
# Annual seasonality with sin/cos (period ~ 365.25)
# Cast column(s) to a specific dtype
day_of_year = df["date"].dt.dayofyear.astype(float)
# Use NumPy for numeric operations
df["sin_annual"] = np.sin(2 * np.pi * day_of_year / 365.25)
# Use NumPy for numeric operations
df["cos_annual"] = np.cos(2 * np.pi * day_of_year / 365.25)
# Blank line

# Use pandas functionality
X = pd.concat(
    [
        # Use pandas functionality
        pd.Series(1.0, index=df.index, name="intercept"),
        # Execute the following statement
        df[["lag_delta", "sin_annual", "cos_annual"]],
        # Execute the following statement
        dow_dummies,
        # Execute the following statement
    ],
    axis=1,
)
# Execute the following statement
y = df["delta"].to_numpy(float)
# Execute the following statement
X_mat = X.to_numpy(float)
# Blank line

# Original comment
# ---- Conjugate Normal–Inverse-Gamma posterior ----
# Original comment
# Prior: beta|sigma^2 ~ N(m0, sigma^2 V0),  sigma^2 ~ InvGamma(a0, b0)
# Execute the following statement
n, p = X_mat.shape
# Use NumPy for numeric operations
m0 = np.zeros(p)
# Use NumPy for numeric operations
V0 = np.eye(p) * 1e6  # weakly-informative
# Execute the following statement
a0 = 2.0
# Use NumPy for numeric operations
yvar = float(np.var(y)) if np.isfinite(np.var(y)) and np.var(y) > 0 else 1.0
# Execute the following statement
b0 = yvar * (a0 - 1)
# Blank line

# Execute the following statement
XtX = X_mat.T @ X_mat
# Use NumPy for numeric operations
V0inv = np.linalg.inv(V0)
# Use NumPy for numeric operations
Vn = np.linalg.inv(XtX + V0inv)
# Execute the following statement
mn = Vn @ (V0inv @ m0 + X_mat.T @ y)
# Execute the following statement
an = a0 + n / 2.0
# Use NumPy for numeric operations
bn = b0 + 0.5 * (y @ y + m0 @ V0inv @ m0 - mn @ np.linalg.inv(Vn) @ mn)
# Blank line

# Original comment
# ---- Posterior predictive: forward simulate next H days with AR(1) lag ----
# Execute the following statement
H = 90  # forecast horizon (days)
# Execute the following statement
S = 4000  # posterior draws
# Blank line

# Select/assign rows/columns by label/position
last_delta = float(df.iloc[-1]["delta"])
# Select/assign rows/columns by label/position
last_backlog = float(backlog.iloc[-1]["backlog_available"])
# Select/assign rows/columns by label/position
last_date = df.iloc[-1]["date"]
# Blank line

# Use pandas functionality
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=H, freq="D")
# Execute the following statement
future_dow = future_dates.dayofweek
# Use NumPy for numeric operations
future_sin = np.sin(2 * np.pi * future_dates.dayofyear / 365.25)
# Use NumPy for numeric operations
future_cos = np.cos(2 * np.pi * future_dates.dayofyear / 365.25)
# Execute the following statement
dow_cols = [c for c in X.columns if c.startswith("dow_")]
# Blank line


# Define a reusable function
def make_x_row(lag_delta_val, idx):
    # Original comment
    # Build X* in the same column order as training X
    # Execute the following statement
    dow = int(future_dow[idx])
    # Use NumPy for numeric operations
    dd = np.zeros(len(dow_cols))
    # Loop over a sequence
    for j, c in enumerate(dow_cols):
        # Try a block of code that may raise errors
        try:
            # Execute the following statement
            target = int(c.split("_")[1])  # 'dow_3' -> 3
        # Handle errors from the try block
        except Exception:
            # Execute the following statement
            target = None
        # Execute the following statement
        dd[j] = 1.0 if (target is not None and dow == target) else 0.0
    # Use NumPy for numeric operations
    return np.concatenate(([1.0, lag_delta_val, future_sin[idx], future_cos[idx]], dd))


# Blank line

# Use NumPy for numeric operations
rng = np.random.default_rng(2025)
# Blank line

# Original comment
# Robust Cholesky (add tiny jitter if near-singular)
# Use NumPy for numeric operations
evals = np.linalg.eigvals(Vn)
# Conditional branch
if np.min(np.real(evals)) < 1e-12:
    # Use NumPy for numeric operations
    Vn = Vn + np.eye(p) * 1e-10
# Use NumPy for numeric operations
L = np.linalg.cholesky(Vn)
# Blank line

# Original comment
# Sample (sigma^2, beta) from posterior
# Execute the following statement
sigma2 = invgamma.rvs(a=an, scale=bn, size=S, random_state=rng)
# Execute the following statement
z = rng.standard_normal((S, p))
# Use NumPy for numeric operations
beta = mn + np.sqrt(sigma2)[:, None] * (z @ L.T)
# Blank line

# Original comment
# Simulate daily deltas forward with AR lag in X
# Use NumPy for numeric operations
delta_draws = np.zeros((S, H))
# Loop over a sequence
for s in range(S):
    # Execute the following statement
    lag = last_delta
    # Use NumPy for numeric operations
    bs = beta[s]
    sig = np.sqrt(sigma2[s])
    # Loop over a sequence
    for h in range(H):
        # Execute the following statement
        xh = make_x_row(lag, h)
        # Execute the following statement
        mean_h = float(xh @ bs)
        # Execute the following statement
        delta_h = mean_h + rng.normal(0.0, sig)
        # Execute the following statement
        delta_draws[s, h] = delta_h
        # Execute the following statement
        lag = delta_h
# Blank line

# Original comment
# Transform to backlog levels; clip at zero
# Use NumPy for numeric operations
backlog_paths = last_backlog + np.cumsum(delta_draws, axis=1)
# Use NumPy for numeric operations
backlog_paths = np.clip(backlog_paths, 0, None)
# Blank line

# Original comment
# Summaries
# Execute the following statement
q = [0.05, 0.2, 0.5, 0.8, 0.95]
# Use NumPy for numeric operations
Q = np.quantile(backlog_paths, q, axis=0).T
# Use pandas functionality
forecast_df = pd.DataFrame(
    {
        # Execute the following statement
        "date": future_dates,
        # Execute the following statement
        "mean": backlog_paths.mean(axis=0),
        # Execute the following statement
        "median": Q[:, 2],
        # Execute the following statement
        "p05": Q[:, 0],
        # Execute the following statement
        "p20": Q[:, 1],
        # Execute the following statement
        "p80": Q[:, 3],
        # Execute the following statement
        "p95": Q[:, 4],
        # Execute the following statement
    }
)
# Save a DataFrame to CSV
forecast_df.to_csv(OUT / "backlog_forecast_bayes.csv", index=False)
# Blank line

# Original comment
# ---- Per-investigator pickup rates: Gamma–Poisson posteriors ----
# Original comment
# For each staff_id, y_i ~ Poisson(theta_i * T_i) with daily exposure T_i (days).
# Original comment
# Prior theta_i ~ Gamma(alpha0, beta0) (rate parameterization) => posterior Gamma(alpha0 + y, beta0 + T).
# Execute the following statement
di = daily.copy()
# Cast column(s) to a specific dtype
di["event_newcase"] = (
    pd.to_numeric(di["event_newcase"], errors="coerce").fillna(0).astype(int)
)
# Blank line

# Group rows and compute aggregations
per_staff = (
    di.groupby("staff_id", as_index=False)
    # Apply aggregation(s) to grouped data
    .agg(
        y_total=("event_newcase", "sum"),
        # Execute the following statement
        days=("date", "nunique"),
    )
)
# Blank line

# Execute the following statement
alpha0, beta0 = 1.0, 1.0
# Execute the following statement
per_staff["alpha_post"] = alpha0 + per_staff["y_total"]
# Execute the following statement
per_staff["beta_post"] = beta0 + per_staff["days"]
# Blank line

# Original comment
# Posterior summaries for daily rate theta_i
# Execute the following statement
per_staff["rate_mean"] = per_staff["alpha_post"] / per_staff["beta_post"]
# Execute the following statement
per_staff["rate_median"] = gamma_dist.ppf(
    0.5, a=per_staff["alpha_post"], scale=1.0 / per_staff["beta_post"]
)
# Execute the following statement
per_staff["rate_p05"] = gamma_dist.ppf(
    0.05, a=per_staff["alpha_post"], scale=1.0 / per_staff["beta_post"]
)
# Execute the following statement
per_staff["rate_p95"] = gamma_dist.ppf(
    0.95, a=per_staff["alpha_post"], scale=1.0 / per_staff["beta_post"]
)
# Blank line

# Original comment
# Expected pickups in next horizons
# Execute the following statement
per_staff["exp_7d_mean"] = per_staff["rate_mean"] * 7.0
# Execute the following statement
per_staff["exp_28d_mean"] = per_staff["rate_mean"] * 28.0
# Blank line

# Save a DataFrame to CSV
per_staff.to_csv(OUT / "investigator_pickup_posterior.csv", index=False)
# Blank line

# Print a message or value
print("Done.")
# Print a message or value
print("Saved:", OUT / "backlog_forecast_bayes.csv")
# Print a message or value
print("Saved:", OUT / "investigator_pickup_posterior.csv")
# Blank line

# === Notebook code cell ends ===

# === Notebook code cell starts ===
# Original comment
# === Stage 3 extension: historical "investigated so far" + 90-day daily predictions
# Original comment
# Assumes Stage-2 output exists at data/out/investigator_daily.csv
# Original comment
# "Investigated" here = daily pickups (event_newcase). Swap to a different event if needed.
# Blank line

# Import libraries/modules for use below
import pandas as pd

# Import libraries/modules for use below
import numpy as np

# Import libraries/modules for use below
from pathlib import Path

# Import libraries/modules for use below
from scipy.stats import (
    nbinom,
)  # Negative Binomial for Gamma–Poisson posterior predictive

# Blank line

# Execute the following statement
OUT = Path("data/out")
# Execute the following statement
OUT.mkdir(parents=True, exist_ok=True)
# Execute the following statement
daily_path = OUT / "investigator_daily.csv"
# Blank line

# Original comment
# ---------- Load ----------
# Load a CSV file into a DataFrame
daily = pd.read_csv(daily_path, parse_dates=["date"])
# Original comment
# If you want "investigated" to mean something else, swap this column:
# Execute the following statement
target_col = (
    "event_newcase"  # <--- change if needed (e.g., 'event_court' or a completion flag)
)
# Blank line

# Cast column(s) to a specific dtype
daily[target_col] = (
    pd.to_numeric(daily[target_col], errors="coerce").fillna(0).astype(int)
)
# Fill missing values with a default
daily["team"] = daily.get("team", pd.Series(index=daily.index)).fillna("Unknown")
# Fill missing values with a default
daily["role"] = daily.get("role", pd.Series(index=daily.index)).fillna("Unknown")
# Execute the following statement
last_date = daily["date"].max()
# Blank line

# Original comment
# =====================================================================
# Original comment
# 1) HISTORICAL: daily counts + cumulative ("so far") per entity
# Original comment
# =====================================================================
# Blank line

# Original comment
# Investigator
# Group rows and compute aggregations
hist_inv = (
    daily.groupby(["date", "staff_id", "team", "role"], as_index=False)[target_col]
    # Execute the following statement
    .sum()
    # Rename columns for clarity/consistency
    .rename(columns={target_col: "daily_pickups"})
)
# Sort rows by specified columns
hist_inv = hist_inv.sort_values(["staff_id", "date"])
# Group rows and compute aggregations
hist_inv["cum_pickups"] = hist_inv.groupby("staff_id")["daily_pickups"].cumsum()
# Save a DataFrame to CSV
hist_inv.to_csv(OUT / "hist_pickups_investigator.csv", index=False)
# Blank line

# Original comment
# Role
# Group rows and compute aggregations
hist_role = (
    daily.groupby(["date", "role"], as_index=False)[target_col]
    # Execute the following statement
    .sum()
    # Rename columns for clarity/consistency
    .rename(columns={target_col: "daily_pickups"})
)
# Sort rows by specified columns
hist_role = hist_role.sort_values(["role", "date"])
# Group rows and compute aggregations
hist_role["cum_pickups"] = hist_role.groupby("role")["daily_pickups"].cumsum()
# Save a DataFrame to CSV
hist_role.to_csv(OUT / "hist_pickups_role.csv", index=False)
# Blank line

# Original comment
# Team
# Group rows and compute aggregations
hist_team = (
    daily.groupby(["date", "team"], as_index=False)[target_col]
    # Execute the following statement
    .sum()
    # Rename columns for clarity/consistency
    .rename(columns={target_col: "daily_pickups"})
)
# Sort rows by specified columns
hist_team = hist_team.sort_values(["team", "date"])
# Group rows and compute aggregations
hist_team["cum_pickups"] = hist_team.groupby("team")["daily_pickups"].cumsum()
# Save a DataFrame to CSV
hist_team.to_csv(OUT / "hist_pickups_team.csv", index=False)
# Blank line

# Original comment
# =====================================================================
# Original comment
# 2) PREDICTIONS: 90-day daily counts per entity (Gamma–Poisson)
# Original comment
#    Posterior (rate-param Gamma prior α0=1, β0=1):
# Original comment
#      For a single day ahead, y ~ NegBinom(r=α_post, p=β_post/(β_post+1)),
# Original comment
#      E[y] = α_post / β_post, with 5–95% credible interval from NB quantiles.
# Original comment
# =====================================================================
# Blank line


# Define a reusable function
def posterior_by_key(daily_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    # Original comment
    # Aggregate to per-day counts for the entity
    # Group rows and compute aggregations
    g_daily = (
        daily_df.groupby(key_cols + ["date"], as_index=False)[target_col]
        # Execute the following statement
        .sum()
        # Rename columns for clarity/consistency
        .rename(columns={target_col: "y"})
    )
    # Original comment
    # Total counts and exposure days (T = # unique dates observed for that entity)
    # Group rows and compute aggregations
    g_total = (
        g_daily.groupby(key_cols, as_index=False)
        # Apply aggregation(s) to grouped data
        .agg(
            y_total=("y", "sum"),
            # Execute the following statement
            T=("date", "nunique"),
        )
    )
    # Blank line

    # Original comment
    # Weak prior
    # Execute the following statement
    alpha0, beta0 = 1.0, 1.0
    # Execute the following statement
    g_total["alpha_post"] = alpha0 + g_total["y_total"]
    # Execute the following statement
    g_total["beta_post"] = beta0 + g_total["T"]
    # Blank line

    # Original comment
    # Negative Binomial params for 1-day-ahead predictive:
    # Original comment
    # In scipy: nbinom(n=r, p) has mean = r*(1-p)/p. Choose p = β/(β+1) → mean = α/β
    # Execute the following statement
    g_total["p_nb"] = g_total["beta_post"] / (g_total["beta_post"] + 1.0)
    # Execute the following statement
    g_total["r_nb"] = g_total["alpha_post"]
    # Blank line

    # Original comment
    # Daily expected value and 90% credible interval
    # Execute the following statement
    g_total["mean"] = g_total["r_nb"] * (1 - g_total["p_nb"]) / g_total["p_nb"]
    # Execute the following statement
    g_total["p05"] = nbinom.ppf(0.05, n=g_total["r_nb"], p=g_total["p_nb"])
    # Execute the following statement
    g_total["p95"] = nbinom.ppf(0.95, n=g_total["r_nb"], p=g_total["p_nb"])
    # Return a value from a function
    return g_total[key_cols + ["mean", "p05", "p95"]]


# Blank line

# Execute the following statement
H = 90
# Use pandas functionality
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=H, freq="D")
# Blank line

# Original comment
# Investigator predictions
# Execute the following statement
post_inv = posterior_by_key(daily, ["staff_id"])
# Original comment
# Join convenient labels (first observed team/role for each staff)
# Execute the following statement
first_map = daily[["staff_id", "team", "role"]].drop_duplicates("staff_id")
# Combine tables by key columns
post_inv = post_inv.merge(first_map, on="staff_id", how="left")
# Blank line

# Create or transform columns
f_inv = (
    post_inv.assign(key=1)
    # Combine tables by key columns
    .merge(pd.DataFrame({"date": future_dates, "key": 1}), on="key")
    # Execute the following statement
    .drop(columns="key")
)[["date", "staff_id", "team", "role", "mean", "p05", "p95"]]
# Save a DataFrame to CSV
f_inv.to_csv(OUT / "forecast_pickups_investigator.csv", index=False)
# Blank line

# Original comment
# Role predictions
# Execute the following statement
post_role = posterior_by_key(daily, ["role"])
# Create or transform columns
f_role = (
    post_role.assign(key=1)
    # Combine tables by key columns
    .merge(pd.DataFrame({"date": future_dates, "key": 1}), on="key")
    # Execute the following statement
    .drop(columns="key")
)[["date", "role", "mean", "p05", "p95"]]
# Save a DataFrame to CSV
f_role.to_csv(OUT / "forecast_pickups_role.csv", index=False)
# Blank line

# Original comment
# Team predictions
# Execute the following statement
post_team = posterior_by_key(daily, ["team"])
# Create or transform columns
f_team = (
    post_team.assign(key=1)
    # Combine tables by key columns
    .merge(pd.DataFrame({"date": future_dates, "key": 1}), on="key")
    # Execute the following statement
    .drop(columns="key")
)[["date", "team", "mean", "p05", "p95"]]
# Save a DataFrame to CSV
f_team.to_csv(OUT / "forecast_pickups_team.csv", index=False)
# Blank line

# Print a message or value
print(
    "Saved:\n -",
    OUT / "hist_pickups_investigator.csv",
    # Execute the following statement
    "\n -",
    OUT / "hist_pickups_role.csv",
    # Execute the following statement
    "\n -",
    OUT / "hist_pickups_team.csv",
    # Execute the following statement
    "\n -",
    OUT / "forecast_pickups_investigator.csv",
    # Execute the following statement
    "\n -",
    OUT / "forecast_pickups_role.csv",
    # Execute the following statement
    "\n -",
    OUT / "forecast_pickups_team.csv",
)
# Blank line

# === Notebook code cell ends ===

# === Notebook code cell starts ===
# === Notebook code cell ends ===
