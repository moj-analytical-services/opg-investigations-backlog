# AUTO-GENERATED FROM NOTEBOOK (do not edit logic)
# Source: Build_Investigator_Daily_from_Raw_12_11_25.ipynb


# === BEGIN NOTEBOOK CODE ===


#!python -m venv .venv && . .venv/bin/activate

# Import libraries/modules for use below
from pathlib import Path
import pandas as pd
import numpy as np
import re
import hashlib

# Configure paths
# Path to the raw investigation data
RAW_PATH = Path("data/raw/raw.csv")
# Path to the output/processed investigation data
OUT_DIR = Path("data/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Print if the path exists
print(RAW_PATH.exists(), OUT_DIR)

# -----------------------------
# DATA PRE-PROCESSING
# -----------------------------


# Define a set of string patterns that represent missing or null values.
# These strings will be treated as equivalent to NaN during cleaning.
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


# -------------------------------------------------------------
# Helper: normalise_col()
# -------------------------------------------------------------
def normalise_col(c: str) -> str:
    """
    Normalize a column name for consistency.

    This function cleans up and standardises column names by:
    - convert to string, lower-case
    - Removing leading/trailing whitespace
    - Replacing multiple spaces with a single space

    Parameters
    ----------
    c : str
        The original column name.

    Returns
    -------
    str
        A cleaned and standardized version of the column name.
    """
    # Convert to string, remove extra spaces, and make lowercase.
    return re.sub(r"\s+", " ", str(c).strip().lower())


# -------------------------------------------------------------
# Helper: parse_date_series()
# -------------------------------------------------------------
def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Parse and clean a pandas Series of date strings.

    This function robustly parse a pandas Series into datetimes:
    - Handles various date formats
    - Converts known null strings to NaT
    - Removes ordinal suffixes (e.g., '1st', '2nd', '3rd')
    - Fixes known typos
    - Uses robust pandas date parsing with fallback strategies

    Parameters
    ----------
    s : pd.Series
        A pandas Series containing raw date values.

    Returns
    -------
    pd.Series
        A pandas Series of datetime64[ns] values with cleaned and parsed dates.
    """

    def _p(x):
        """Internal helper to parse a single date entry."""
        import pandas as pd

        # Return NaT if missing
        if pd.isna(x):
            return pd.NaT

        # Convert to lowercase string
        xs = str(x).strip().lower()

        # Return NaT if in known null string set
        if xs in NULL_STRINGS:
            return pd.NaT

        # Clean up common errors and ordinal suffixes
        xs = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", xs).replace("legel", "legal")

        # Try strict parsing, then flexible fallback
        try:
            return pd.to_datetime(xs, dayfirst=True, errors="raise")
        except Exception:
            return pd.to_datetime(
                xs, infer_datetime_format=True, dayfirst=True, errors="coerce"
            )

    # Apply the parser to each element of the Series
    return s.apply(_p)

    if s is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0))

    # If numeric-like (possible Excel serials), try converting via pandas
    s_num = pd.to_numeric(s, errors="coerce")
    has_numeric = s_num.notna().any()

    # First pass: assume strings with day-first ambiguity handled later
    dt1 = pd.to_datetime(
        s, errors="coerce", utc=False, dayfirst=True, infer_datetime_format=True
    )

    if has_numeric:
        # Where dt1 is NaT but we have a number, try fromordinal-like conversion via pandas
        # pandas handles Excel serials when unit='D' origin='1899-12-30'
        serial_dt = pd.to_datetime(
            s_num, unit="D", origin="1899-12-30", errors="coerce"
        )
        dt1 = dt1.fillna(serial_dt)

    # Final pass (month-first) for any remaining NaT strings
    mask_nat = dt1.isna() & s.notna()
    if mask_nat.any():
        dt2 = pd.to_datetime(
            s.where(mask_nat),
            errors="coerce",
            dayfirst=False,
            infer_datetime_format=True,
        )
        dt1 = dt1.fillna(dt2)

    # Normalise to midnight
    return dt1.dt.normalize()


# -------------------------------------------------------------
# Helper: hash_id()
# -------------------------------------------------------------
def hash_id(t: str, prefix: str = "S", length: int = 12) -> str:
    """
    Generate a short, anonymized hash-based identifier.

    Creates a pseudonymized ID for text entries using SHA1 hashing.
    Empty or missing values return an empty string.

    Parameters
    ----------
    t : str
        The input text value (e.g., name, case number).

    Returns
    -------
    str
        An anonymised hash string prefixed with 'S', e.g., 'S1a2b3c4d'.
    """
    # # Return empty string for null or blank input
    # if pd.isna(t) or str(t).strip() == '':
    #     return ''

    # # Create SHA1 hash and take first 8 characters for compact ID
    # return 'S' + hashlib.sha1(str(t).encode('utf-8')).hexdigest()[:8]

    if pd.isna(t) or str(t).strip() == "":
        return ""
    h = hashlib.sha256(str(t).strip().lower().encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:length]}"


# -------------------------------------------------------------
# Helper: month_to_season()
# -------------------------------------------------------------
def month_to_season(m: int) -> str:
    """
    Convert a numeric month into a season name.

    Parameters
    ----------
    m : int
        Month number (1–12).

    Returns
    -------
    str
        The season corresponding to the month ('winter', 'spring', 'summer', or 'autumn').

    Examples
    --------
    >>> month_to_season(4)
    'spring'
    >>> month_to_season(10)
    'autumn'
    """
    # Map month numbers to their respective seasons
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


# -------------------------------------------------------------
# Helper: is_term_month()
# -------------------------------------------------------------
def is_term_month(m: int) -> int:
    """
    Identify whether a month is a 'termination month'.

    In the current logic, August (month 8) is excluded and returns 0.
    All other months return 1, representing active/valid months.

    Parameters
    ----------
    m : int
        Month number (1–12).

    Returns
    -------
    int
        0 if the month is August, else 1.
    """
    # Return binary flag based on month value
    return 0 if int(m) == 8 else 1


# -------------------------------------
# DATA LOADING AND FEATURE ENGINEERING
# -------------------------------------

from pathlib import Path
import pandas as pd


# -------------------------------------------------------------
# Function: load_raw()
# -------------------------------------------------------------
def load_raw(p: Path, force_encoding: str | None = None):
    """
    Load a CSV or Excel file into a pandas DataFrame with robust encoding handling.

    This function attempts to open and read raw data files safely, even when
    character encodings vary or are unknown. It tries multiple encodings until
    one succeeds, trims cell whitespace, drops empty rows/columns, and returns a
    column-name map to support fuzzy lookups.

    Parameters
    ----------
    p : Path
        Path to the input file.
    force_encoding : str, optional
        If provided, forces the use of a specific encoding.

    Returns
    -------
    tuple
        (df, colmap)
        df : pd.DataFrame
            Cleaned dataframe containing the raw data.
        colmap : dict
            Mapping of normalized column names (lowercased, trimmed) to original column headers.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    RuntimeError
        If all encoding attempts fail.
    """

    # Check file existence
    if not p.exists():
        raise FileNotFoundError(p)

    # --- Read the file (Excel has no encoding issue) ---
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p, dtype=str)
    else:
        tried, df, last_err = [], None, None
        encodings_to_try = (
            [force_encoding]
            if force_encoding
            else [
                "utf-8-sig",
                "utf-8",
                "cp1252",
                "latin1",
                "iso-8859-1",
                "utf-16",
                "utf-16le",
                "utf-16be",
            ]
        )

        # Try to read using multiple encodings
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(
                    p,
                    dtype=str,
                    sep=None,
                    engine="python",
                    encoding=enc,
                    encoding_errors="strict",
                )
                break
            except Exception as e:
                tried.append(enc)
                last_err = e
        if df is None:
            # Fallback with replacement to avoid hard fail
            try:
                df = pd.read_csv(
                    p,
                    dtype=str,
                    sep=None,
                    engine="python",
                    encoding="cp1252",
                    encoding_errors="replace",
                )
                print(
                    f"[load_raw] WARNING: used cp1252 with replacement after failed encodings: {tried}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read CSV. Tried encodings {tried}. Last error: {last_err}"
                ) from e

    # --- Clean up cell text, drop fully-empty rows/columns ---
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Drop columns that are entirely blank/NaN
    df = df.dropna(axis=1, how="all")
    # Drop rows that are entirely blank/NaN
    df = df.dropna(axis=0, how="all")

    # --- Build mapping of normalised column names → original names ---
    raw_cols = list(df.columns)
    normalised = [normalise_col(c) for c in raw_cols]

    # Handle collisions (two columns normalise to the same key)
    colmap = {}
    seen = {}
    for orig, norm in zip(raw_cols, normalised):
        if norm in seen:
            # append a numeric suffix to make the key unique
            seen[norm] += 1
            key = f"{norm}__{seen[norm]}"
        else:
            seen[norm] = 0
            key = norm
        colmap[key] = orig

    return df, colmap


# -------------------------------------------------------------
# Function: col()
# -------------------------------------------------------------
def col(df: pd.DataFrame, colmap: dict, name: str) -> pd.Series:
    """
    Retrieve a column from a DataFrame by fuzzy name matching.

    This function normalises the requested column name and searches the column map
    for an exact or partial match. Returns a Series of NaNs if not found.
    - First tries exact match on a normalised key.
    - Then tries partial match (either direction).
    - Finally, falls back to an empty Series (NaNs) of correct length.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.
    colmap : dict
        Mapping of normalised column names to original names (from load_raw()).
    name : str
        Column name to look up.

    Returns
    -------
    pd.Series
        The column data if found, otherwise a Series of NaN values.
    """

    k = normalise_col(name)

    # 1) Exact match
    if k in colmap:
        return df[colmap[k]]

    # 2) Partial match (prefix/substring in either direction)
    for kk, v in colmap.items():
        if k in kk or kk in k:
            return df[v]

    # 3) If load_raw had to suffix collided keys, try any key that starts with k
    for kk, v in colmap.items():
        if kk.startswith(k):
            return df[v]

    # 4) Default: return empty column (NaNs) so downstream code doesn't crash
    return pd.Series([np.nan] * len(df), index=df.index)
    # # Exact match first
    # if k in colmap:
    #     return df[colmap[k]]

    # # Partial match fallback
    # for kk, v in colmap.items():
    #     if k in kk or kk in k:
    #         return df[v]

    # # Default: return empty column of NaNs
    # return pd.Series([np.nan] * len(df))


# -------------------------------------------------------------
# Function: engineer()
# -------------------------------------------------------------
def engineer(
    df: pd.DataFrame,
    colmap: dict,
    only_reallocated: bool = False,  # NEW: filter toggle
) -> pd.DataFrame:
    """
    Engineer standardised and typed columns from raw investigation data.

    This function extracts and converts the key variables such as case IDs, investigators,
    FTEs, and multiple date columns from the raw file using reusable helper functions.
      - selects and cleans core identifiers (case, staff, team, role, FTE)
      - parses all relevant milestone dates
      - brings in extra attributes (reallocated flag, weighting, types/status)
      - computes anonymised staff IDs
      - optionally filters to only reallocated cases via `only_reallocated`

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_raw().
    colmap : dict
        Column name mapping from load_raw().
    only_reallocated : bool, default True
        If True, return only cases where 'Reallocated Case' is truthy
        (accepts yes/y/true/1, case-insensitive). If False, return all cases.

    pd.DataFrame
        Cleaned, typed dataset ready for downstream modelling.
        Includes:
          - `is_reallocated` (bool) derived from 'Reallocated Case'
          - all date columns as datetime (normalised)
          - numeric fields coerced where applicable (fte, weighting, days_to_pg_signoff)
    """

    out = pd.DataFrame(
        {
            "case_id": col(df, colmap, "ID"),
            "investigator": col(df, colmap, "Investigator"),
            "team": col(df, colmap, "Team"),
            "fte": pd.to_numeric(col(df, colmap, "Investigator FTE"), errors="coerce"),
            "reallocated_case": col(df, colmap, "Reallocated Case"),
            "weighting": pd.to_numeric(col(df, colmap, "Weighting"), errors="coerce"),
            "case_type": col(df, colmap, "Case Type"),
            "concern_type": col(df, colmap, "Concern Type"),
            "status": col(df, colmap, "Status"),
            "days_to_pg_signoff": pd.to_numeric(
                col(df, colmap, "Days to PG sign off"), errors="coerce"
            ),
        }
    )

    # Parse and standardise relevant date columns
    out["dt_received_inv"] = parse_date_series(
        col(df, colmap, "Date Received in Investigations")
    )
    out["dt_alloc_invest"] = parse_date_series(
        col(df, colmap, "Date allocated to current investigator")
    )
    out["dt_alloc_team"] = parse_date_series(col(df, colmap, "Date allocated to team"))
    out["dt_pg_signoff"] = parse_date_series(col(df, colmap, "PG Sign off date"))
    out["dt_close"] = parse_date_series(col(df, colmap, "Closure Date"))
    out["dt_legal_req_1"] = parse_date_series(
        col(df, colmap, "Date of Legal Review Request 1")
    )
    out["dt_legal_rej_1"] = parse_date_series(col(df, colmap, "Date Legal Rejects 1"))
    out["dt_legal_req_2"] = parse_date_series(
        col(df, colmap, "Date of Legal Review Request 2")
    )
    out["dt_legal_rej_2"] = parse_date_series(col(df, colmap, "Date Legal Rejects 2"))
    out["dt_legal_req_3"] = parse_date_series(
        col(df, colmap, "Date of Legel Review Request 3")
    )
    out["dt_legal_approval"] = parse_date_series(col(df, colmap, "Legal Approval Date"))
    out["dt_date_of_order"] = parse_date_series(col(df, colmap, "Date Of Order"))
    out["dt_flagged"] = parse_date_series(col(df, colmap, "Flagged Date"))
    out["dt_sent_to_ca"] = parse_date_series(col(df, colmap, "Date Sent To CA"))

    # Fill missing FTEs with 1.0, hash investigator names for anonymization, and add placeholders
    # Defaults, anonymisation, and placeholders
    out["fte"] = out["fte"].fillna(1.0)  # assume FT when missing
    out["staff_id"] = out["investigator"].apply(hash_id)  # anonymise
    # If a 'role' column existed in raw, keep it; else initialise blank
    role_series = (
        col(df, colmap, "Role")
        if "role" in [k.split("__")[0] for k in colmap.keys()]
        else pd.Series([""] * len(out))
    )
    out["role"] = role_series.fillna("") if isinstance(role_series, pd.Series) else ""

    # Compute days_to_pg_signoff if wholly missing but dates exist
    if (
        out["days_to_pg_signoff"].isna().all()
        and ("dt_pg_signoff" in out)
        and ("dt_alloc_invest" in out)
    ):
        diff = (out["dt_pg_signoff"] - out["dt_alloc_invest"]).dt.days
        out["days_to_pg_signoff"] = pd.to_numeric(diff, errors="coerce")

    # --- NEW: derive a clean boolean, then optionally filter ---
    reall_str = out["reallocated_case"].astype(str).str.strip().str.lower()
    out["is_reallocated"] = reall_str.isin({"yes", "y", "true", "1"})

    if only_reallocated:
        out = out.loc[out["is_reallocated"]].reset_index(drop=True)

    return out


# -------------------------------------
# DATA MANIPULATION AND PROCESSING
# -------------------------------------


# -------------------------------------------------------------
# Function: date_horizon()
# -------------------------------------------------------------
def date_horizon(
    typed: pd.DataFrame, pad_days: int = 14, fallback_to_all_dates: bool = True
):
    """
    Primary rule:
      - start := earliest non-null value in 'dt_received_inv'
      - end   := latest non-null value in 'dt_pg_signoff'

    Optional fallback:
      If either start or end cannot be determined (column missing or all NaT)
      *and* fallback_to_all_dates is True, compute:
        - start := min across ALL columns starting with 'dt_'
        - end   := max across ALL columns starting with 'dt_'

    Finally, apply `pad_days` to the end date. If still missing after fallback,
    default to a 30-day lookback for start and today for end (+ padding).

    Parameters
    ----------
    typed : pd.DataFrame
        Feature-engineered dataset with standardized date columns.
    pad_days : int, default=14
        Number of days to extend the end horizon.
        pad_days adds a few days to the end date as a buffer.
    fallback_to_all_dates : bool, default=True
        Whether to fall back to scanning all `dt_` columns when the primary
        columns are unavailable or empty.
        This allows scanning all dt_… columns if the main two are missing/empty.

    Returns
    -------
    tuple of pd.Timestamp
        (start, end) normalised date range.

    Notes
    -----
    Falls back to recent 30 days if dt_received_inv or dt_pg_signoff
    are missing or contain no valid dates.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> df = pd.DataFrame({
    ...     'dt_received_inv': [pd.Timestamp('2025-01-05'), pd.NaT],
    ...     'dt_alloc_invest': [pd.NaT, pd.Timestamp('2025-01-10')],
    ...     'dt_alloc_team': [pd.NaT, pd.NaT],
    ...     'dt_close': [pd.NaT, pd.Timestamp('2025-02-01')],
    ...     'dt_pg_signoff': [pd.NaT, pd.NaT],
    ...     'dt_date_of_order': [pd.NaT, pd.NaT],
    ... })
    >>> s, e = date_horizon(df, pad_days=7)
    >>> isinstance(s, pd.Timestamp) and isinstance(e, pd.Timestamp)
    True
    >>> (e - s).days >= (pd.Timestamp('2025-02-01') - pd.Timestamp('2025-01-05')).days
    True
    """
    # start = pd.concat([typed['dt_received_inv'], typed['dt_alloc_invest'], typed['dt_alloc_team']]).min()
    # end = pd.concat([typed['dt_close'], typed['dt_pg_signoff'], typed['dt_date_of_order']]).max()

    # --- Primary computation from specified columns ---
    start = pd.NaT
    end = pd.NaT

    # Initialise start and end as “not a time” (missing).
    if "dt_received_inv" in typed:
        start = typed["dt_received_inv"].dropna().min()

    # If the “received” column exists, take the earliest non-missing date as start.
    if "dt_pg_signoff" in typed:
        end = typed["dt_pg_signoff"].dropna().max()

    # If the “PG sign-off” column exists, take the latest non-missing date as end.
    # --- Optional fallback over all dt_ columns ---
    if (pd.isna(start) or pd.isna(end)) and fallback_to_all_dates:
        dt_cols = [c for c in typed.columns if c.startswith("dt_")]
        if dt_cols:
            all_dates = pd.concat(
                [typed[c] for c in dt_cols], ignore_index=True
            ).dropna()
            if pd.isna(start) and not all_dates.empty:
                start = all_dates.min()
            if pd.isna(end) and not all_dates.empty:
                end = all_dates.max()

    # --- Final graceful defaults if still missing ---
    # If we still don’t have a start/end, default to a 30-day lookback ending at today.
    today = pd.Timestamp.today().normalize()
    if pd.isna(start):
        start = today - pd.Timedelta(days=30)
    if pd.isna(end):
        end = today

    # Add pad_days to end (the buffer) and normalize both dates to midnight (clean calendar dates).
    # --- Apply padding to end and normalise ---
    end = (end + pd.Timedelta(days=pad_days)).normalize()
    return start.normalize(), end


# -------------------------------------------------------------
# Function: build_event_log()
# -------------------------------------------------------------
def build_event_log(
    typed: pd.DataFrame, pad_days: int = 14, fallback_to_all_dates: bool = True
) -> pd.DataFrame:
    """
    Construct a staff-day event log from feature-engineered investigation data.

    Each row represents a dated event for a specific case and staff member.
    For example, “Investigator S1 picked up case C1 on 2025-01-10.”

    For each case, this function creates dated event records (e.g., new case pickup,
    legal requests/approvals, court orders) at the staff-day level.

    Events emitted (if their date exists):
      received         -> dt_received_inv
      alloc_team       -> dt_alloc_team
      newcase          -> dt_alloc_invest
      sent_to_ca       -> dt_sent_to_ca
      legal_request    -> dt_legal_req_1, dt_legal_req_2, dt_legal_req_3
      legal_reject     -> dt_legal_rej_1, dt_legal_rej_2
      legal_approval   -> dt_legal_approval
      pg_signoff       -> dt_pg_signoff
      court_order      -> dt_date_of_order
      closed           -> dt_close
      flagged          -> dt_flagged

      The output is restricted to the date horizon determined by date_horizon()
      using dt_received_inv for start and dt_pg_signoff for end (with padding).

    Parameters
    ----------
    typed : pd.DataFrame
        Output of engineer(); typically already filtered to reallocated cases.
        Expected columns include identifiers, staffing info, and the dt_* fields.
    pad_days : int, default=14
        Extra days added to end horizon via date_horizon().
    fallback_to_all_dates : bool, default=True
        If start/end cannot be derived from the primary columns, allow
        date_horizon() to fallback across all dt_* columns.

    Returns
    -------
    pd.DataFrame
        Columns: ['date','staff_id','team','fte','case_id','event','meta']

    Notes
    -------
    Includes lightweight, structured meta (JSON) with weighting, case_type, concern_type, status, and days_to_pg_signoff.
    to keep contextual attributes about that case alongside each event (for later analysis or auditing), such as:
    Case weighting (e.g., 2.5 for complexity or workload)
    Case type (Financial / Welfare / etc.)
    Concern type (Neglect / Abuse / etc.)
    Current status (Open / Closed / etc.)
    Days to PG sign-off (performance metric)

    Instead of duplicating these as separate columns for every event — which would make the event log wide, repetitive,
    and harder to serialize — we store them compactly in a single column named meta.
    Each meta cell is a JSON string encoding those extra attributes.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> typed = pd.DataFrame({
    ...     'staff_id': ['S1'],
    ...     'team': ['A'],
    ...     'fte': [1.0],
    ...     'case_id': ['C1'],
    ...     'weighting': [2.5],
    ...     'case_type': ['Financial'],
    ...     'concern_type': ['Neglect'],
    ...     'status': ['Open'],
    ...     'days_to_pg_signoff': [15],
    ...     # Key timeline dates
    ...     'dt_received_inv': [pd.Timestamp('2025-01-05')],
    ...     'dt_alloc_team': [pd.Timestamp('2025-01-08')],
    ...     'dt_alloc_invest': [pd.Timestamp('2025-01-10')],
    ...     'dt_sent_to_ca': [pd.Timestamp('2025-01-12')],
    ...     'dt_legal_req_1': [pd.Timestamp('2025-01-14')],
    ...     'dt_legal_req_2': [pd.NaT],
    ...     'dt_legal_req_3': [pd.NaT],
    ...     'dt_legal_rej_1': [pd.NaT],
    ...     'dt_legal_rej_2': [pd.NaT],
    ...     'dt_legal_approval': [pd.Timestamp('2025-01-20')],
    ...     'dt_pg_signoff': [pd.Timestamp('2025-01-25')],
    ...     'dt_date_of_order': [pd.NaT],
    ...     'dt_close': [pd.Timestamp('2025-02-01')],
    ...     'dt_flagged': [pd.NaT],
    ... })
    >>> ev = build_event_log(typed)
    >>> sorted(ev['event'].unique().tolist())
    ['alloc_team', 'closed', 'legal_approval', 'legal_request',
     'newcase', 'pg_signoff', 'received', 'sent_to_ca']
    >>> set(ev.columns) >= {'date','staff_id','team','fte','case_id','event','meta'}
    True
    >>> # Each meta cell contains structured JSON metadata:
    >>> import json
    >>> json.loads(ev.loc[0, 'meta'])
    {'weighting': 2.5,
     'case_type': 'Financial',
     'concern_type': 'Neglect',
     'status': 'Open',
     'days_to_pg_signoff': 15.0}

    Examples
    --------
    >>> import pandas as pd, json
    >>> # Two cases, two investigators, showcasing more event types
    >>> typed = pd.DataFrame({
    ...     'staff_id': ['S1', 'S2'],
    ...     'team': ['A', 'B'],
    ...     'fte': [1.0, 0.8],
    ...     'case_id': ['C1', 'C2'],
    ...     'weighting': [2.5, 1.0],
    ...     'case_type': ['Financial', 'Welfare'],
    ...     'concern_type': ['Neglect', 'Abuse'],
    ...     'status': ['Open', 'Open'],
    ...     'days_to_pg_signoff': [15, pd.NA],
    ...     # Timeline dates (C1 has a full path incl. pg_signoff; C2 shows rejects, no signoff)
    ...     'dt_received_inv': [pd.Timestamp('2025-01-05'), pd.Timestamp('2025-01-07')],
    ...     'dt_alloc_team': [pd.Timestamp('2025-01-08'), pd.Timestamp('2025-01-09')],
    ...     'dt_alloc_invest': [pd.Timestamp('2025-01-10'), pd.Timestamp('2025-01-11')],
    ...     'dt_sent_to_ca': [pd.Timestamp('2025-01-12'), pd.NaT],
    ...     'dt_legal_req_1': [pd.Timestamp('2025-01-14'), pd.Timestamp('2025-01-15')],
    ...     'dt_legal_req_2': [pd.NaT, pd.Timestamp('2025-01-18')],
    ...     'dt_legal_req_3': [pd.NaT, pd.NaT],
    ...     'dt_legal_rej_1': [pd.NaT, pd.Timestamp('2025-01-17')],
    ...     'dt_legal_rej_2': [pd.NaT, pd.NaT],
    ...     'dt_legal_approval': [pd.Timestamp('2025-01-20'), pd.NaT],
    ...     'dt_pg_signoff': [pd.Timestamp('2025-01-25'), pd.NaT],
    ...     'dt_date_of_order': [pd.NaT, pd.NaT],
    ...     'dt_close': [pd.Timestamp('2025-02-01'), pd.NaT],
    ...     'dt_flagged': [pd.NaT, pd.NaT],
    ... })
    >>> ev = build_event_log(typed)  # uses date_horizon(start=dt_received_inv, end=dt_pg_signoff+pad)
    >>> # Unique event types emitted
    >>> sorted(ev['event'].unique().tolist())
    ['alloc_team', 'closed', 'legal_approval', 'legal_reject', 'legal_request',
     'newcase', 'pg_signoff', 'received', 'sent_to_ca']
    >>> # Schema check
    >>> set(ev.columns) >= {'date','staff_id','team','fte','case_id','event','meta'}
    True
    >>> # Per-case event counts (C1 has a full pathway, C2 has requests + a reject)
    >>> ev.groupby('case_id')['event'].count().to_dict()  # doctest: +ELLIPSIS
    {'C1': 8, 'C2': 6}
    >>> # meta is JSON with contextual fields
    >>> m = json.loads(ev.loc[ev['case_id'].eq('C2')].iloc[0]['meta'])
    >>> set(m.keys()) == {'weighting','case_type','concern_type','status','days_to_pg_signoff'}
    True
    >>> m['case_type'], m['concern_type'], m['weighting']
    ('Welfare', 'Abuse', 1.0)

    """

    import json

    # Ensure expected minimal columns exist
    base_cols = ["staff_id", "team", "fte", "case_id"]
    for c in base_cols:
        if c not in typed.columns:
            raise KeyError(
                f"build_event_log: required column '{c}' missing from 'typed'."
            )

    # Compute the date horizon
    start, end = date_horizon(
        typed, pad_days=pad_days, fallback_to_all_dates=fallback_to_all_dates
    )

    # Helper to safely read a column if present
    def getcol(name: str):
        return (
            typed[name]
            if name in typed.columns
            else pd.Series([pd.NaT] * len(typed), index=typed.index)
        )

    # Pre-pull columns used in meta (safe if absent)
    weighting = (
        typed["weighting"]
        if "weighting" in typed.columns
        else pd.Series([pd.NA] * len(typed), index=typed.index)
    )
    case_type = (
        typed["case_type"]
        if "case_type" in typed.columns
        else pd.Series([pd.NA] * len(typed), index=typed.index)
    )
    concern_type = (
        typed["concern_type"]
        if "concern_type" in typed.columns
        else pd.Series([pd.NA] * len(typed), index=typed.index)
    )
    status = (
        typed["status"]
        if "status" in typed.columns
        else pd.Series([pd.NA] * len(typed), index=typed.index)
    )
    days_to_pg = (
        typed["days_to_pg_signoff"]
        if "days_to_pg_signoff" in typed.columns
        else pd.Series([pd.NA] * len(typed), index=typed.index)
    )

    # Map of event names to the corresponding date columns to scan (one or many)
    event_map = {
        "received": ["dt_received_inv"],
        "alloc_team": ["dt_alloc_team"],
        "newcase": ["dt_alloc_invest"],
        "sent_to_ca": ["dt_sent_to_ca"],
        "legal_request": ["dt_legal_req_1", "dt_legal_req_2", "dt_legal_req_3"],
        "legal_reject": ["dt_legal_rej_1", "dt_legal_rej_2"],
        "legal_approval": ["dt_legal_approval"],
        "pg_signoff": ["dt_pg_signoff"],
        "court_order": ["dt_date_of_order"],
        "closed": ["dt_close"],
        "flagged": ["dt_flagged"],
    }

    records = []
    # Iterate row-wise to emit events per case
    for i, r in typed.iterrows():
        sid, team, fte, cid = r["staff_id"], r["team"], r["fte"], r["case_id"]

        # Build the meta payload once per row
        meta_dict = {
            "weighting": None if pd.isna(weighting.iloc[i]) else weighting.iloc[i],
            "case_type": None if pd.isna(case_type.iloc[i]) else str(case_type.iloc[i]),
            "concern_type": (
                None if pd.isna(concern_type.iloc[i]) else str(concern_type.iloc[i])
            ),
            "status": None if pd.isna(status.iloc[i]) else str(status.iloc[i]),
            "days_to_pg_signoff": (
                None if pd.isna(days_to_pg.iloc[i]) else float(days_to_pg.iloc[i])
            ),
        }
        meta_json = json.dumps(meta_dict, ensure_ascii=False)

        # Emit events for each configured date column
        for etype, cols in event_map.items():
            for c in cols:
                if c in typed.columns:
                    dt = r[c]
                    if pd.notna(dt):
                        dtn = pd.to_datetime(dt).normalize()
                        # Keep only within the computed horizon
                        if start <= dtn <= end:
                            records.append(
                                {
                                    "date": dtn,
                                    "staff_id": sid,
                                    "team": team,
                                    "fte": fte,
                                    "case_id": cid,
                                    "event": etype,
                                    "meta": meta_json,
                                }
                            )

    ev = pd.DataFrame.from_records(records)

    if ev.empty:
        return pd.DataFrame(
            columns=["date", "staff_id", "team", "fte", "case_id", "event", "meta"]
        )

    # Deduplicate identical events (same staff/case/date/type)
    ev = (
        ev.drop_duplicates(subset=["date", "staff_id", "case_id", "event"])
        .sort_values(["date", "staff_id", "case_id", "event"])
        .reset_index(drop=True)
    )

    # Ensure dtypes are tidy
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["fte"] = pd.to_numeric(ev["fte"], errors="coerce")

    return ev


# -------------------------------------
# TIME SERIES ANALYSIS
# -------------------------------------


# -------------------------------------------------------------
# Function: build_wip_series()
# -------------------------------------------------------------
def build_wip_series(
    typed: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    pad_days: int = 14,
    fallback_to_all_dates: bool = True,
) -> pd.DataFrame:
    """
    Build a Work-In-Progress (WIP) daily series per staff member.

    A case is considered WIP from dt_alloc_invest (inclusive) to the earliest of:
      - dt_close
      - dt_pg_signoff
      - provided/computed `end` horizon

    If `start`/`end` are not provided, they are derived via `date_horizon()` with the
    rule: start from dt_received_inv, end from dt_pg_signoff (+ pad_days).

    Inputs and defaults:
    typed: engineered table (one row per case).
    start, end: optional date limits for the report.
    If start or end are missing, it calls date_horizon() to derive them from the data using the rule (received → pg_signoff + padding).
    Then it normalises them to whole dates (midnight).

    Output includes:
      - `wip`       : number of active cases (count-based)
      - `wip_load`  : workload proxy, defined as weighting / fte (fallbacks to 1.0 if absent)

    Parameters
    ----------
    typed : pd.DataFrame
        Expected columns:
          identifiers: ['staff_id','team','case_id'] (case_id optional for debugging)
          dates: ['dt_alloc_invest','dt_close','dt_pg_signoff'] (+ others for date_horizon)
          optional: ['weighting','fte']
    start : pd.Timestamp | None
        Start of the reporting horizon (normalised to date). If None, computed via date_horizon().
    end : pd.Timestamp | None
        End of the reporting horizon (normalised to date). If None, computed via date_horizon().
    pad_days : int, default=14
        Only used if start/end are not supplied; passed to date_horizon().
    fallback_to_all_dates : bool, default=True
        Passed to date_horizon().

    Returns
    -------
    pd.DataFrame
        Columns: ['date','staff_id','team','wip','wip_load']
        - One row per (date, staff_id, team).
        - `wip` is guaranteed non-negative.

    Examples
    --------
    >>> import pandas as pd
    >>> # Two cases for S1; second case has PG sign-off. Includes weighting & fte for wip_load.
    >>> typed = pd.DataFrame({
    ...     'staff_id': ['S1','S1'],
    ...     'team': ['A','A'],
    ...     'case_id': ['C1','C2'],
    ...     'fte': [1.0, 0.5],
    ...     'weighting': [2.0, 1.0],
    ...     'dt_received_inv': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-01')],
    ...     'dt_alloc_invest': [pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-05')],
    ...     'dt_close': [pd.Timestamp('2025-01-03'), pd.NaT],
    ...     'dt_pg_signoff': [pd.NaT, pd.Timestamp('2025-01-07')],
    ... })
    >>> # Explicit horizon
    >>> wip = build_wip_series(typed, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-10'))
    >>> set(wip.columns) == {'date','staff_id','team','wip','wip_load'}
    True
    >>> wip['wip'].ge(0).all()
    True
    >>> # On 2025-01-06, both cases are WIP -> wip >= 1
    >>> int(wip.loc[wip['date'].eq(pd.Timestamp('2025-01-06')), 'wip'].max()) >= 1
    True
    """

    # --- Compute horizon (if needed) ---
    # If you don’t pass start/end, the function figures them out using your project rule:
    # start = earliest dt_received_inv
    # end = latest dt_pg_signoff plus a small padding window
    if start is None or end is None:
        s, e = date_horizon(
            typed, pad_days=pad_days, fallback_to_all_dates=fallback_to_all_dates
        )
        if start is None:
            start = s
        if end is None:
            end = e

    # Normalise
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    # --- Guard: required columns for interval construction ---
    # Verifies key columns exist: staff_id, team, dt_alloc_invest.
    # If any are missing, it raises a helpful error explaining what’s needed.
    for c in ["staff_id", "team", "dt_alloc_invest"]:
        if c not in typed.columns:
            raise KeyError(
                f"build_wip_series: required column '{c}' missing from 'typed'."
            )

    # --- Prepare per-case start/end ---
    # Start of work
    s_col = pd.to_datetime(typed["dt_alloc_invest"], errors="coerce")

    # Earliest of dt_close and dt_pg_signoff per row; then fallback to provided/computed end
    # End of work = the earliest of dt_close and dt_pg_signoff.
    # If both are missing, the end defaults to the overall report end date (so open cases remain WIP).
    close_candidates = pd.concat(
        [
            (
                pd.to_datetime(typed["dt_close"], errors="coerce")
                if "dt_close" in typed
                else pd.Series(pd.NaT, index=typed.index)
            ),
            (
                pd.to_datetime(typed["dt_pg_signoff"], errors="coerce")
                if "dt_pg_signoff" in typed
                else pd.Series(pd.NaT, index=typed.index)
            ),
        ],
        axis=1,
    )
    row_end = close_candidates.min(axis=1)  # earliest available milestone
    row_end = row_end.fillna(end)

    # Case load for wip_load: weighting / fte (with robust fallbacks)
    # If weighting is missing, it uses 1.0 (assume average complexity).
    if "weighting" in typed.columns:
        w_series = pd.to_numeric(typed["weighting"], errors="coerce").fillna(1.0)
    else:
        w_series = pd.Series(1.0, index=typed.index)
    # If fte is missing or zero, it uses 1.0 (avoid division by zero and keep a sane baseline).
    if "fte" in typed.columns:
        fte_series = (
            pd.to_numeric(typed["fte"], errors="coerce").replace(0, pd.NA).fillna(1.0)
        )
    else:
        fte_series = pd.Series(1.0, index=typed.index)

    # Load per case = weighting ÷ fte
    load = (w_series / fte_series).astype(float)

    # Creates a small table with one row per case showing:
    # staff_id, team, start (allocation), end (close/signoff/report end), and load.
    intervals = pd.DataFrame(
        {
            "staff_id": typed["staff_id"],
            "team": typed["team"],
            "start": s_col,
            "end": row_end,
            "load": load,
        }
    ).dropna(subset=["start", "end"])

    # --- Build delta encoding (inclusive start, inclusive end) ---
    # Delta encoding (efficient daily accumulation)
    # Creates a full daily calendar and applies the cumulative sum of deltas.
    deltas = []
    horizon_start, horizon_end = start, end
    for _, r in intervals.iterrows():
        s = pd.to_datetime(r["start"]).normalize()
        e = pd.to_datetime(r["end"]).normalize()

        # Skip if outside horizon
        if s > horizon_end or e < horizon_start:
            continue

        s = max(s, horizon_start)
        e = min(e, horizon_end)

        # For each case interval: Add a +1 (and +load) on the start date.
        # Add a −1 (and −load) on the day after the end date.
        # +1 case and +load at start; -1 and -load at day after end
        deltas.append((r["staff_id"], r["team"], s, 1.0, r["load"]))
        deltas.append(
            (r["staff_id"], r["team"], e + pd.Timedelta(days=1), -1.0, -r["load"])
        )
    # This means when we later cumulatively sum these daily changes, we get the number of active cases (and total load) each day.
    if not deltas:
        return pd.DataFrame(columns=["date", "staff_id", "team", "wip", "wip_load"])

    deltas = pd.DataFrame(
        deltas, columns=["staff_id", "team", "date", "d_cases", "d_load"]
    )
    # Builds a continuous list of dates from start to end
    all_dates = pd.DataFrame(
        {"date": pd.date_range(horizon_start, horizon_end, freq="D")}
    )

    # --- Accumulate per staff/team over the horizon ---
    # For each staff × team group:
    # Merges the deltas onto the daily grid.
    # Cumulative sums to get wip (counts) and wip_load (load).
    # Clips at zero to avoid negative values if data has gaps.
    out_rows = []
    for (sid, team), g in deltas.groupby(["staff_id", "team"], sort=False):
        gg = g.groupby("date", as_index=False)[["d_cases", "d_load"]].sum()
        grid = all_dates.merge(gg, on="date", how="left").fillna(
            {"d_cases": 0.0, "d_load": 0.0}
        )
        # clip(lower=0) ensures small data glitches can’t produce negatives.
        grid["wip"] = grid["d_cases"].cumsum().clip(lower=0)  # case count
        grid["wip_load"] = grid["d_load"].cumsum().clip(lower=0.0)  # workload proxy
        grid["staff_id"] = sid
        grid["team"] = team
        out_rows.append(grid[["date", "staff_id", "team", "wip", "wip_load"]])

    out = (
        pd.concat(out_rows, ignore_index=True)
        if out_rows
        else pd.DataFrame(columns=["date", "staff_id", "team", "wip", "wip_load"])
    )

    # Ensure dtypes / normalisation
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["wip"] = pd.to_numeric(out["wip"], errors="coerce").fillna(0).astype(float)
    out["wip_load"] = (
        pd.to_numeric(out["wip_load"], errors="coerce").fillna(0.0).astype(float)
    )

    return out


# -------------------------------------
# TIME SERIES ANALYSIS
# -------------------------------------


# -------------------------------------------------------------
# Function: build_backlog_series()
# -------------------------------------------------------------
def build_backlog_series(
    typed: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    pad_days: int = 14,
    fallback_to_all_dates: bool = True,
    clip_zero: bool = True,
    compute_weighted: bool = False,
    exclude_weekends: bool = False,
    holidays: list | pd.Series | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    """
    Build a daily backlog series where:
        backlog = cumulative received − cumulative allocated.

    Definitions
    -----------
    - Received: cases entering Investigations (dt_received_inv)=(case enters Investigations queue).
    - Allocated: cases allocated to an investigator (dt_alloc_invest)=(case leaves queue to an investigator).
    - Backlog available: cases received but not yet allocated.

    Horizon
    -------
    If `start`/`end` are not provided, they are derived via `date_horizon()`:
      start := earliest dt_received_inv,
      end := latest dt_pg_signoff (+ pad_days),
      with optional fallback to all dt_* columns if primary dates are missing.

    Options
    -------
    - clip_zero:        Prevent negative backlog (recommended).
    - compute_weighted: Also compute weighted backlog using 'weighting' if present.
    - exclude_weekends: Remove Saturdays/Sundays from the time axis.
    - holidays:         Iterable of dates to exclude (e.g., UK bank holidays).
    - freq:             Optional resampling frequency (e.g., 'W-MON', 'W-FRI', 'MS').
                        For cumulative series, we take the last value per period.

    Parameters
    ----------
    typed : pd.DataFrame
        Expected columns:
          - dates: ['dt_received_inv','dt_alloc_invest']  (others allowed but not required)
          - optional: ['weighting'] if compute_weighted=True
        Note: this frame is already filtered to reallocated cases per your earlier requirement.
    start, end : pd.Timestamp | None
        Reporting horizon (inclusive). If None, computed via date_horizon().
    pad_days : int, default=14
        Only used when deriving start/end via date_horizon().
    fallback_to_all_dates : bool, default=True
        Passed to date_horizon().
    clip_zero : bool, default=True
        If True, backlog cannot go below 0 (defensive; improves interpretability).
    compute_weighted : bool, default=False
        If True and 'weighting' is present, also compute backlog_weighted
        using the same logic but summing weights instead of counts.
    exclude_weekends : bool, default=False
        If True drop Saturdays/Sundays from the series
    holidays : bool, default=False
        If True drop a custom list/series of dates (e.g., UK bank holidays)
    freq : str | None
        optional resampling (e.g., 'W-MON', 'W-FRI', 'MS' for month-start).
        For cumulative series, we take the last value per period.

    Returns
    -------
    pd.DataFrame with at least following Columns (daily):
          - date
          - received_cum      : cumulative count of received
          - allocated_cum     : cumulative count of allocated
          - backlog_available : received_cum - allocated_cum (clipped at 0 if clip_zero)
          - (optional: and, if compute_weighted) received_weighted_cum, allocated_weighted_cum, backlog_weighted

    Examples
    --------
    >>> import pandas as pd
    >>> typed = pd.DataFrame({
    ...     'dt_received_inv': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-03')],
    ...     'dt_alloc_invest': [pd.Timestamp('2025-01-02'), pd.NaT],
    ... })
    >>> backlog = build_backlog_series(typed, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-05'))
    >>> list(backlog.columns)
    ['date', 'received_cum', 'allocated_cum', 'backlog_available']
    >>> backlog.iloc[-1]['backlog_available']  # 2 received, 1 allocated -> 1
    1.0

    >>> # Weighted example (if 'weighting' present)
    >>> typed2 = pd.DataFrame({
    ...     'dt_received_inv': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-03')],
    ...     'dt_alloc_invest': [pd.Timestamp('2025-01-02'), pd.NaT],
    ...     'weighting': [2.0, 0.5],
    ... })
    >>> backlog_w = build_backlog_series(typed2, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-05'), compute_weighted=True)
    >>> {'backlog_available', 'backlog_weighted'}.issubset(backlog_w.columns)
    True

    >>> import pandas as pd
    >>> typed = pd.DataFrame({
    ...     'dt_received_inv': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-03')],
    ...     'dt_alloc_invest': [pd.Timestamp('2025-01-02'), pd.NaT],
    ... })
    >>> # Daily (default calendar)
    >>> build_backlog_series(typed, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-05')).tail(1)[['backlog_available']].iloc[0,0]
    1.0

    >>> # Business days only (excludes weekends)
    >>> business = build_backlog_series(
    ...     typed,
    ...     pd.Timestamp('2025-01-01'),
    ...     pd.Timestamp('2025-01-10'),
    ...     exclude_weekends=True
    ... )

    >>> # With holidays excluded and weekly roll-up (end-of-week values)
    >>> holidays = [pd.Timestamp('2025-01-06')]
    >>> weekly = build_backlog_series(
    ...     typed,
    ...     pd.Timestamp('2025-01-01'),
    ...     pd.Timestamp('2025-01-31'),
    ...     exclude_weekends=True,
    ...     holidays=holidays,
    ...     freq='W-FRI'
    ... )

    """
    # --- Derive horizon if needed  ---
    # If we didn’t pass start/end, we derive them with date_horizon()
    if start is None or end is None:
        s, e = date_horizon(
            typed, pad_days=pad_days, fallback_to_all_dates=fallback_to_all_dates
        )
        if start is None:
            start = s
        if end is None:
            end = e
    # Normalise them to dates (no times).
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    # --- Extract and normalise event dates ---
    rec_dates = (
        pd.to_datetime(
            typed.get("dt_received_inv", pd.Series([], dtype="datetime64[ns]")),
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
    )
    alloc_dates = (
        pd.to_datetime(
            typed.get("dt_alloc_invest", pd.Series([], dtype="datetime64[ns]")),
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
    )

    # --- Daily counts (received / allocated) ---
    # Daily counts → cumulative totals
    # Count how many received and allocated events happen per day.
    received_daily = rec_dates.value_counts().sort_index()
    allocated_daily = alloc_dates.value_counts().sort_index()

    # --- Build full daily index over the horizon ---
    idx = pd.date_range(start, end, freq="D")

    # Optional calendar filtering (weekends and/or holidays)
    if exclude_weekends:
        idx = idx[idx.weekday < 5]  # 0=Mon ... 4=Fri
    if holidays is not None and len(pd.Index(holidays)) > 0:
        hol = pd.to_datetime(pd.Index(holidays)).normalize()
        idx = idx.difference(hol)

    # Helper to reindex to possibly filtered calendar and cumulate
    def cumulate(series_counts: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        # We need the *full* daily cumsum first, then realign to filtered index
        full_range = pd.date_range(start, end, freq="D")
        full_cum = (
            series_counts.reindex(full_range, fill_value=0).cumsum().astype(float)
        )
        # If calendar is filtered, take values at the kept dates
        return full_cum.reindex(index, method="ffill").fillna(0.0)

    # --- Cumulate counts over the horizon (missing days = 0) ---
    # Reindex missing days as zeros and cumulatively sum to get “total so far”.
    received_cum = received_daily.reindex(idx, fill_value=0).cumsum().astype(float)
    allocated_cum = allocated_daily.reindex(idx, fill_value=0).cumsum().astype(float)

    # Backlog is the gap between total received and total allocated.
    backlog = received_cum - allocated_cum
    # Optionally clip at 0 (defensive, avoids negative values if historical allocations
    #  predate the first received in the window).
    if clip_zero:
        backlog = backlog.clip(lower=0.0)

    out = pd.DataFrame(
        {
            "date": idx,
            "received_cum": received_cum.values,
            "allocated_cum": allocated_cum.values,
            "backlog_available": backlog.values,
        }
    )

    # --- Optional weighted backlog ---
    # Sum weights per day at receipt and at allocation, then cumulate and subtract.
    # Same structure as counts, but with weights instead of 1s.
    if compute_weighted:
        # If weighting missing, assume 1.0 for rows with the date present, else 0
        weights = pd.to_numeric(
            typed.get("weighting", pd.Series([1.0] * len(typed))), errors="coerce"
        ).fillna(1.0)

        # Map weights to dates for received and allocated events
        def weighted_daily(dates: pd.Series, weight_series: pd.Series) -> pd.Series:
            if len(dates) == 0:
                return pd.Series(dtype=float)
            tmp = pd.DataFrame({"date": dates.reset_index(drop=True)})
            # Align weights to the same original row positions as 'dates'
            tmp["weight"] = weight_series.loc[dates.index].values
            return tmp.groupby("date")["weight"].sum().sort_index()

        # Build per-date weight sums for received and allocated
        rec_w_daily = weighted_daily(rec_dates, weights)
        alloc_w_daily = weighted_daily(alloc_dates, weights)

        # reindex
        rec_w_cum = cumulate(rec_w_daily, idx)
        alloc_w_cum = cumulate(alloc_w_daily, idx)

        # # Build per-date weight sums for received and allocated
        # rec_weights = (
        #     pd.DataFrame({'date': rec_dates.reset_index(drop=True)})
        #     .assign(weight=weights.loc[rec_dates.index].values if len(rec_dates) else [])
        #     .groupby('date')['weight'].sum()
        #     if len(rec_dates) else pd.Series(dtype=float)
        # )

        # alloc_weights = (
        #     pd.DataFrame({'date': alloc_dates.reset_index(drop=True)})
        #     .assign(weight=weights.loc[alloc_dates.index].values if len(alloc_dates) else [])
        #     .groupby('date')['weight'].sum()
        #     if len(alloc_dates) else pd.Series(dtype=float)
        # )

        # rec_w_cum = rec_weights.reindex(idx, fill_value=0).cumsum().astype(float)
        # alloc_w_cum = alloc_weights.reindex(idx, fill_value=0).cumsum().astype(float)

        backlog_w = rec_w_cum - alloc_w_cum
        if clip_zero:
            backlog_w = backlog_w.clip(lower=0.0)

        out["received_weighted_cum"] = rec_w_cum.values
        out["allocated_weighted_cum"] = alloc_w_cum.values
        out["backlog_weighted"] = backlog_w.values

    # --- Optional resampling (weekly/monthly views)
    if freq is not None:
        # Set index for resampling, then take "last" per period for cumulative metrics.
        out = out.set_index("date").sort_index()
        agg_map = {
            "received_cum": "last",
            "allocated_cum": "last",
            "backlog_available": "last",
        }
        if compute_weighted:
            agg_map.update(
                {
                    "received_weighted_cum": "last",
                    "allocated_weighted_cum": "last",
                    "backlog_weighted": "last",
                }
            )
        out = out.resample(freq).agg(agg_map).dropna(how="all").reset_index()

    return out


# -------------------------------------------------------------
# Function: build_daily_panel()
# -------------------------------------------------------------
def build_daily_panel(
    typed: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    *,
    pad_days: int = 14,
    fallback_to_all_dates: bool = True,
    # Pass-through options to backlog & WIP builders
    backlog_kwargs: dict | None = None,
    wip_kwargs: dict | None = None,
    # Panel calendar options (also forwarded into backlog unless overridden there)
    exclude_weekends: bool = False,
    holidays: list | pd.Series | None = None,
    backlog_freq: str | None = None,  # e.g. 'W-FRI', 'W-MON', 'MS'
):
    """
    Create a fully-featured daily staff panel for modelling and analytics.

    Outputs
    -------
    This function combines outputs from:
      - build_event_log()     → daily operational events (e.g., newcase, legal, sign-off)
      - build_wip_series()    → daily work-in-progress (active cases, workloads)
      - build_backlog_series()→ daily system backlog (received minus allocated)
    into one unified dataset at the **staff × date** level.

    Calendar controls
    -----------------
    exclude_weekends : if True, panel dates will exclude Saturdays/Sundays
    holidays         : iterable of dates marked as bank holidays in the panel;
                       passed to backlog as exclusions too (unless overridden).
    backlog_freq     : resampling frequency for backlog only (e.g., 'W-FRI', 'MS').
                       Daily panel remains daily (or business-day if exclude_weekends=True).

    Horizon:
    If `start` and `end` are not provided, the function automatically determines
    the date range using `date_horizon()` based on your project’s rule:
      start := earliest dt_received_inv
      end   := latest `dt_pg_signoff` (+ padding of `pad_days`)
    Set fallback_to_all_dates=True to allow scanning all dt_* if primaries are missing.

    Notes
    -----
    - Event flags derived from build_event_log(): newcase, alloc_team, sent_to_ca,
      legal_request, legal_reject, legal_approval, pg_signoff, court_order, closed, flagged.
    - Compact flags provided: event_newcase, event_legal, event_court, event_pg_signoff,
      event_sent_to_ca, event_flagged.
    - WIP uses dt_alloc_invest → earliest(dt_close, dt_pg_signoff, end).

       typed : pd.DataFrame
        Feature-engineered dataframe from `engineer()`, typically filtered
        to reallocated cases.
        Must include:
          - Identifiers: `case_id`, `staff_id`, `team`, `role`, `fte`
          - Core dates:  `dt_received_inv`, `dt_alloc_invest`, `dt_pg_signoff`,
                         `dt_close` (and optionally legal & court milestones)
        Optional columns (used if present):
          - `weighting`, `status`, `case_type`, `concern_type`,
            `days_to_pg_signoff`, etc.

    start, end : pd.Timestamp | None, default None
        Reporting horizon. If not given, derived automatically from `date_horizon()`.

    pad_days : int, default 14
        Number of days to extend the end horizon when deriving automatically.

    fallback_to_all_dates : bool, default True
        When true, allows `date_horizon()` to use all dt_* columns if the primary
        (received / PG sign-off) columns are missing or incomplete.

    backlog_kwargs : dict | None
        Extra keyword arguments forwarded to `build_backlog_series()`.
        Examples:
            {'compute_weighted': True, 'clip_zero': True,
             'exclude_weekends': False, 'holidays': holidays,
             'freq': 'W-FRI'}

    wip_kwargs : dict | None
        Extra keyword arguments forwarded to `build_wip_series()`.
        Example:
            {'pad_days': 14, 'fallback_to_all_dates': True}

    exclude_weekends : bool, default False
        If True, weekends (Saturday/Sunday) are excluded from the daily panel
        and from the backlog calculation.

    holidays : list | pd.Series | None, default None
        List or Series of public holidays to exclude from the panel timeline
        and mark with `bank_holiday = 1`.

    backlog_freq : str | None, default None
        Optional resampling frequency for backlog only.
        Examples: 'W-FRI' (weekly, Friday close), 'MS' (month-start).

    -----------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (daily, backlog, events)

        **daily** : pd.DataFrame
        One row per (date × staff × team), containing:
          - Workload:  `wip`, `wip_load`
          - Backlog context: `backlog_available`
          - Event flags: `event_newcase`, `event_legal`, `event_court`,
                         `event_pg_signoff`, `event_sent_to_ca`, `event_flagged`
          - Calendar features: `dow`, `season`, `term_flag`, `bank_holiday`
          - Tenure features: `weeks_since_start`, `is_new_starter`
          - Temporal context: `time_since_last_pickup`

        **backlog** : pd.DataFrame
        System-level backlog series built by `build_backlog_series()` with optional
        business-day or weekly/monthly resampling.

        **events** : pd.DataFrame
        Event log built by `build_event_log()`, containing granular dated events
        per staff, case, and team.

    -----------------------------------------------------------------------
    Examples
    -----------------------------------------------------------------------
    >>> import pandas as pd
    >>> typed = pd.DataFrame({
    ...     'case_id': ['C1','C2'],
    ...     'investigator': ['Alice','Bob'],
    ...     'team': ['T1','T1'],
    ...     'role': ['Investigator','Investigator'],
    ...     'fte': [1.0, 0.8],
    ...     'staff_id': ['S1','S2'],
    ...     'dt_received_inv': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02')],
    ...     'dt_alloc_invest': [pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-03')],
    ...     'dt_pg_signoff': [pd.NaT, pd.NaT],
    ...     'dt_close': [pd.NaT, pd.NaT],
    ...     'dt_legal_req_1': [pd.NaT, pd.Timestamp('2025-01-04')],
    ...     'dt_legal_approval': [pd.NaT, pd.NaT],
    ...     'dt_date_of_order': [pd.NaT, pd.NaT],
    ... })
    >>> start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-05')
    >>> daily, backlog, events = build_daily_panel(
    ...     typed,
    ...     start=start,
    ...     end=end,
    ...     exclude_weekends=True,
    ...     holidays=[pd.Timestamp('2025-01-03')],
    ...     backlog_freq='W-FRI',
    ...     backlog_kwargs={'compute_weighted': True}
    ... )
    >>> # Daily panel has one row per staff per day
    >>> set({'date','staff_id','team','fte','wip','event_newcase'}).issubset(daily.columns)
    True
    >>> # Backlog matches the number of working days
    >>> len(backlog) <= (end - start).days + 1
    True
    >>> # Event log contains expected event types
    >>> {'newcase','legal_request'}.issubset(set(events['event'].unique())) if not events.empty else True
    True

    -----------------------------------------------------------------------
    """

    backlog_kwargs = {} if backlog_kwargs is None else dict(backlog_kwargs)
    wip_kwargs = {} if wip_kwargs is None else dict(wip_kwargs)

    # 1 Determine horizon (uses your updated rule) ---
    if start is None or end is None:
        s, e = date_horizon(
            typed, pad_days=pad_days, fallback_to_all_dates=fallback_to_all_dates
        )
        start = s if start is None else start
        end = e if end is None else end
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    # 2 Build the three core artefacts from the pipeline (events, WIP, backlog)
    events = build_event_log(
        typed, pad_days=pad_days, fallback_to_all_dates=fallback_to_all_dates
    )

    # WIP stays daily across full horizon; the panel may later filter dates
    wip = build_wip_series(typed, start=start, end=end, **wip_kwargs)

    # Ensure panel-level calendar options are forwarded to backlog unless explicitly set
    backlog_defaults = {
        "pad_days": pad_days,
        "fallback_to_all_dates": fallback_to_all_dates,
        "exclude_weekends": exclude_weekends,
        "holidays": holidays,
        "freq": backlog_freq,
    }
    for k, v in backlog_defaults.items():
        backlog_kwargs.setdefault(k, v)

    backlog = build_backlog_series(typed, start=start, end=end, **backlog_kwargs)

    # 3) Panel date index (daily or business-day)
    date_index = pd.date_range(start, end, freq="D")
    if exclude_weekends:
        date_index = date_index[date_index.weekday < 5]
    if holidays is not None and len(pd.Index(holidays)) > 0:
        hol = pd.to_datetime(pd.Index(holidays)).normalize()
        date_index = date_index.difference(hol)
    dates = pd.DataFrame({"date": date_index})

    # 4) Build staff-date grid (all combinations)
    staff = typed[["staff_id", "team"]].drop_duplicates()
    grid = (
        staff.assign(_k=1)
        .merge(pd.DataFrame({"date": date_index}).assign(_k=1), on="_k", how="outer")
        .drop(columns=["_k"])
    )

    # 5) Merge WIP data (wip & wip_load). If grid has filtered dates, merge naturally subsets.
    grid = grid.merge(wip, on=["date", "staff_id", "team"], how="left")
    for c, default in [("wip", 0.0), ("wip_load", 0.0)]:
        grid[c] = (
            pd.to_numeric(grid.get(c, default), errors="coerce")
            .fillna(default)
            .astype(float)
        )

    # 6) Pivot events → daily flags per staff
    if not events.empty:
        ev_flags = (
            events.assign(flag=1)
            .pivot_table(
                index=["date", "staff_id"],
                columns="event",
                values="flag",
                aggfunc="max",
            )
            .reset_index()
        )

        # Merge at staff-day; team may differ if staff moved teams, but WIP merge above anchors team
        grid = grid.merge(ev_flags, on=["date", "staff_id"], how="left")

    # Ensure a stable set of event columns exists
    event_cols = [
        "newcase",
        "alloc_team",
        "sent_to_ca",
        "legal_request",
        "legal_reject",
        "legal_approval",
        "pg_signoff",
        "court_order",
        "closed",
        "flagged",
    ]
    for c in event_cols:
        grid[c] = grid.get(c, 0)
        grid[c] = grid[c].fillna(0).astype(int)

    # Compact event groupings useful for modelling
    grid["event_newcase"] = grid["newcase"].astype(int)
    grid["event_legal"] = (
        (grid["legal_request"] + grid["legal_approval"] + grid["legal_reject"]) > 0
    ).astype(int)
    grid["event_court"] = grid["court_order"].astype(int)
    grid["event_pg_signoff"] = grid["pg_signoff"].astype(int)
    grid["event_sent_to_ca"] = grid["sent_to_ca"].astype(int)
    grid["event_flagged"] = grid["flagged"].astype(int)

    # 7) Days since last pickup (per staff)
    grid = grid.sort_values(["staff_id", "date"])

    def _days_since_last_pickup(series: pd.Series) -> pd.Series:
        out, last = [], None
        for i, v in enumerate(series):
            if v == 1:
                last = i
                out.append(0)
            else:
                out.append(i - last if last is not None else pd.NA)
        return pd.Series(out, index=series.index)

    grid["time_since_last_pickup"] = (
        grid.groupby("staff_id", group_keys=False)["event_newcase"]
        .apply(_days_since_last_pickup)
        .fillna(99)
        .astype(int)
    )

    # 8) Calendar features
    grid["dow"] = grid["date"].dt.day_name().str[:3]
    grid["season"] = grid["date"].dt.month.map(month_to_season)
    grid["term_flag"] = grid["date"].dt.month.map(is_term_month).astype(int)
    # Bank holiday flag (1 if the date is in holidays)
    if holidays is not None and len(pd.Index(holidays)) > 0:
        hol = pd.to_datetime(pd.Index(holidays)).normalize()
        grid["bank_holiday"] = grid["date"].isin(hol).astype(int)
    else:
        grid["bank_holiday"] = 0

    # 9) New starter (tenure) features (weeks since first allocation per staff)
    first_alloc = (
        typed.dropna(subset=["dt_alloc_invest"])
        .groupby("staff_id")["dt_alloc_invest"]
        .min()
        .rename("first_alloc")
    )
    grid = grid.merge(first_alloc, on="staff_id", how="left")
    grid["weeks_since_start"] = (
        ((grid["date"] - grid["first_alloc"]).dt.days // 7)
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )
    grid["is_new_starter"] = (grid["weeks_since_start"] < 4).astype(int)
    grid = grid.drop(columns=["first_alloc"])

    # 10) Merge backlog (always by 'date'; backlog may be resampled)
    # If backlog was resampled (e.g., weekly), forward-fill to panel dates.
    if "date" in backlog.columns and backlog["date"].is_monotonic_increasing:
        back = backlog.set_index("date").sort_index()
        # Keep only the core columns we need (avoid accidental merges)
        keep_cols = [
            c
            for c in back.columns
            if c
            in {
                "received_cum",
                "allocated_cum",
                "backlog_available",
                "received_weighted_cum",
                "allocated_weighted_cum",
                "backlog_weighted",
            }
        ]
        back = back[keep_cols]
        back = back.reindex(date_index, method="ffill")  # align to panel calendar
        back = back.reset_index().rename(columns={"index": "date"})
    else:
        back = backlog.copy()

    grid = grid.merge(back, on="date", how="left")
    grid["backlog_available"] = pd.to_numeric(
        grid.get("backlog_available", 0.0), errors="coerce"
    ).fillna(0.0)

    # 11) Final tidy columns & order
    cols = [
        "date",
        "staff_id",
        "team",
        "role",
        "fte",
        "wip",
        "wip_load",
        "time_since_last_pickup",
        "weeks_since_start",
        "is_new_starter",
        "backlog_available",
        "term_flag",
        "season",
        "dow",
        "bank_holiday",
        "event_newcase",
        "event_legal",
        "event_court",
        "event_pg_signoff",
        "event_sent_to_ca",
        "event_flagged",
    ]
    cols = [c for c in cols if c in grid.columns]  # be tolerant
    daily = grid[cols].sort_values(["staff_id", "date"]).reset_index(drop=True)

    return daily, backlog, events


# -------------------------------------------------------------
# Function: summarise_daily_panel()
# -------------------------------------------------------------
def summarise_daily_panel(
    daily: pd.DataFrame,
    by: list[str] = ("date", "team"),
    *,
    freq: str | None = None,
    # How to aggregate each metric; sensible defaults provided
    agg_map: dict | None = None,
    # If resampling, how to aggregate cumulative-style fields
    resample_cum_last: tuple[str, ...] = ("backlog_available",),
) -> pd.DataFrame:
    """
    Summarise the daily staff panel by date/team (or any grouping).

    Parameters
    ----------
    daily : pd.DataFrame
        Output of build_daily_panel()[0], with columns like:
          ['date','staff_id','team','wip','wip_load','backlog_available',
           'event_newcase','event_legal','event_court','event_pg_signoff',
           'event_sent_to_ca','event_flagged','time_since_last_pickup', ...]
    by : list[str], default ('date','team')
        Grouping columns. Must include 'date' if you want a time series.
        Examples: ('date',), ('date','team'), ('date','team','role')
    freq : str | None, default None
        Optional resampling frequency over time *after* grouping.
        Examples: 'W-FRI', 'MS'. If None, returns daily resolution.
    agg_map : dict | None, default None
        Custom aggregation map. If None, a sensible default is used:
          - Sum counts/loads/events
          - Mean backlog_available
          - Median time_since_last_pickup
          - Distinct staff_count
    resample_cum_last : tuple[str,...], default ('backlog_available',)
        For resampling, fields treated as *cumulative/stateful* and aggregated
        via 'last' per period (e.g., backlog_available).

    Returns
    -------
    pd.DataFrame
        One row per group (and per period if resampled). Includes:
          - wip_sum, wip_load_sum
          - backlog_available_mean (and backlog_available_last if resampled)
          - events counts: newcase, legal, court, pg_signoff, sent_to_ca, flagged
          - staff_count (distinct staff_id)
          - time_since_last_pickup_median

    Examples
    --------
    >>> # team-level daily
    >>> team_daily = summarise_daily_panel(daily, by=['date','team'])
    >>> # team-level weekly (Friday)
    >>> team_weekly = summarise_daily_panel(daily, by=['date','team'], freq='W-FRI')
    """
    if "date" not in by:
        raise ValueError(
            "`by` must include 'date' to preserve time order (or set freq=None for a non-time summary)."
        )

    # Default aggregation plan
    default_agg = {
        "wip": "sum",
        "wip_load": "sum",
        "backlog_available": "mean",  # daily mean backlog across staff on that date
        "event_newcase": "sum",
        "event_legal": "sum",
        "event_court": "sum",
        "event_pg_signoff": "sum",
        "event_sent_to_ca": "sum",
        "event_flagged": "sum",
        "time_since_last_pickup": "median",
        "staff_id": pd.Series.nunique,  # distinct headcount working that day
    }
    if agg_map is not None:
        default_agg.update(agg_map)

    # Group and aggregate on the daily grid
    grouped = (
        daily.groupby(list(by), dropna=False)
        .agg(default_agg)
        .rename(
            columns={
                "wip": "wip_sum",
                "wip_load": "wip_load_sum",
                "backlog_available": "backlog_available_mean",
                "event_newcase": "event_newcase_sum",
                "event_legal": "event_legal_sum",
                "event_court": "event_court_sum",
                "event_pg_signoff": "event_pg_signoff_sum",
                "event_sent_to_ca": "event_sent_to_ca_sum",
                "event_flagged": "event_flagged_sum",
                "time_since_last_pickup": "time_since_last_pickup_median",
                "staff_id": "staff_count",
            }
        )
        .reset_index()
    )

    if freq is None:
        # Return daily/grouped summary as-is
        return grouped.sort_values(by).reset_index(drop=True)

    # Resampling: we need a DatetimeIndex aligned on 'date'
    out = []
    other_keys = [k for k in by if k != "date"]
    for keys, sub in grouped.groupby(other_keys, dropna=False):
        # Ensure consistent frame and index
        sub = sub.sort_values("date").set_index("date")

        # For numeric fields, decide resampling rule:
        # - For cumulative/state-like fields -> last
        # - For flow-like fields (counts) -> sum
        numeric_cols = sub.select_dtypes(include="number").columns.tolist()

        # Prepare aggregation map for resample
        resample_agg = {}
        for col in numeric_cols:
            if col in resample_cum_last:
                resample_agg[col] = "last"
            else:
                resample_agg[col] = "sum"

        sub_res = sub.resample(freq).agg(resample_agg)

        # Keep grouping keys
        if not isinstance(keys, tuple):
            keys = (keys,)
        for k, v in zip(other_keys, keys):
            sub_res[k] = v

        out.append(sub_res.reset_index())

    resampled = pd.concat(out, ignore_index=True) if out else grouped
    return resampled.sort_values(
        by if freq is None else (["date"] + other_keys)
    ).reset_index(drop=True)


# daily, backlog, events = build_daily_panel(
#     typed,
#     # optional: let it auto-derive start/end via date_horizon()
#     exclude_weekends=True,
#     holidays=[pd.Timestamp('2025-05-05'), pd.Timestamp('2025-08-25')],  # UK BHs (example)
#     backlog_freq='W-FRI',  # weekly backlog, last value each Friday
#     backlog_kwargs={'compute_weighted': True, 'clip_zero': True},  # weighted backlog too
#     wip_kwargs={'pad_days': 14, 'fallback_to_all_dates': True}
# )

from pathlib import Path

raw, colmap = load_raw(Path("data/raw/raw.csv"))
typed = engineer(raw, colmap)
start, end = date_horizon(typed)

# Works after applying the JSON patch (or using the runtime shim I used here)
events = build_event_log(typed)

wip = build_wip_series(typed, start, end)
backlog = build_backlog_series(typed, start, end)

# Requires the 'grid' insert in build_daily_panel (Patch 2)
daily, backlog2, events2 = build_daily_panel(typed, start, end)

print("Start/End:", start.date(), end.date())
print("Daily shape:", daily.shape)
print("Backlog shape:", backlog.shape)
print("Events shape:", events.shape)

print("\nDaily head:\n", daily.head())
print("\nBacklog tail:\n", backlog.tail())
print("\nEvents:\n", events.sort_values(["date", "staff_id", "event"]))

print("\nBacklog2 tail:\n", backlog2.tail())
print("\nEvents2:\n", events2.sort_values(["date", "staff_id", "event"]))

# (optional) save to disk
# Save daily DataFrame to CSV
daily.to_csv(OUT_DIR / "investigator_daily.csv", index=False)
# Save backlog DataFrame to CSV
backlog.to_csv(OUT_DIR / "backlog_series.csv", index=False)
# Save events DataFrame to CSV
events.to_csv(OUT_DIR / "event_log.csv", index=False)
# Save backlog2 DataFrame to CSV
backlog.to_csv(OUT_DIR / "backlog2_series.csv", index=False)
# Save events DataFrame to CSV
events2.to_csv(OUT_DIR / "event_log2.csv", index=False)

# summary = summarise_daily_panel(daily, by=['date','team'])
# print(summary)


# 1) Team-level daily
team_daily = summarise_daily_panel(daily, by=["date", "team"])

# 2) Team-level weekly (Friday), treating backlog as a level (last-of-week)
team_weekly = summarise_daily_panel(
    daily,
    by=["date", "team"],
    freq="W-FRI",
    resample_cum_last=("backlog_available",),  # keep as 'last' per week
)

# 3) Overall totals per day (collapse teams)
org_daily = summarise_daily_panel(daily, by=["date"])

# 4) Custom aggregation rules (e.g., use max backlog across staff instead of mean)
custom = summarise_daily_panel(
    daily, by=["date", "team"], agg_map={"backlog_available": "max"}
)
# Save events DataFrame to CSV
custom.to_csv(OUT_DIR / "Custom_Summary.csv", index=False)
print(custom)


# --- Interval Analysis: new code (non-invasive) ---
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any

y = 4  # Number of years for analysis to start with

# Meteorological seasons
_SEASON_MAP = {
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
}


@dataclass(frozen=True)

# tiny config (which months count as term; how many weeks someone is a “new starter”).
class IntervalFlags:
    term_months: Iterable[int] = (1, 4, 7, 10)
    new_starter_weeks: int = 12


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _bool(x) -> pd.Series:
    return pd.Series(x, dtype="boolean")


class IntervalAnalysis:
    """Extension utilities for case-level time interval analysis (read-only, additive)."""

    REQUIRED_COLUMNS = [
        "date",
        "staff_id",
        "team",
        "case_id",
        "case_type",
        "concern_type",
        "status",
        "dt_alloc_invest",
        "dt_pg_signoff",
        "dt_received_inv",
        "dt_alloc_team",
        "dt_close",
        "dt_sent_to_ca",
        "days_to_pg_signoff",
        "fte",
        "weighting",
        "wip",
        "wip_load",
        "time_since_last_pickup",
        "weeks_since_start",
        "is_new_starter",
        "backlog_available",
        "term_flag",
        "season",
        "dow",
        "bank_holiday",
        "event_newcase",
        "event_legal",
        "event_court",
        "event_pg_signoff",
        "event_sent_to_ca",
        "event_flagged",
        "backlog",  # <-- numeric backlog count per date
    ]

    @staticmethod
    def build_interval_frame(
        raw: pd.DataFrame,
        *,
        backlog_series: Optional[
            pd.DataFrame
        ] = None,  # expects cols ['date','backlog'] if provided
        bank_holidays: Optional[Iterable[pd.Timestamp | str]] = None,
        flags: IntervalFlags = IntervalFlags(),
        default_date_from: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Construct a dataframe matching the requested schema, with a numeric 'backlog' column.
        This DOES NOT modify existing notebook functions; it can be used alongside them.
        If a backlog series is not provided, it is computed per observed 'date' as:
            backlog(date) = #cases with (dt_received_inv <= date) and (dt_close isna or dt_close > date)
        """
        df = raw.copy()

        date_cols = [
            "date",
            "dt_alloc_invest",
            "dt_pg_signoff",
            "dt_received_inv",
            "dt_alloc_team",
            "dt_close",
            "dt_sent_to_ca",
        ]
        df = _ensure_columns(
            df,
            date_cols
            + [
                "fte",
                "weighting",
                "status",
                "case_type",
                "concern_type",
                "team",
                "staff_id",
                "case_id",
            ],
        )

        for c in date_cols:
            df[c] = _to_date(df[c])

        # Observation date default
        if "date" not in raw.columns or df["date"].isna().all():
            df["date"] = df["dt_alloc_invest"]
        fallback = df["dt_received_inv"].where(df["date"].isna(), df["date"])
        df["date"] = df["date"].fillna(df["dt_alloc_invest"]).fillna(fallback)

        # Numeric defaults
        df["fte"] = pd.to_numeric(df["fte"], errors="coerce").fillna(1.0)
        df["weighting"] = pd.to_numeric(df["weighting"], errors="coerce").fillna(1.0)

        # Primary interval(s)
        df["days_to_pg_signoff"] = (
            ((df["dt_pg_signoff"] - df["dt_alloc_invest"]).dt.days)
            .astype("float")
            .replace({np.inf: np.nan, -np.inf: np.nan})
        )

        # WIP flag
        df["wip"] = _bool(
            (df["dt_alloc_invest"].notna())
            & (df["date"].notna())
            & (df["date"] >= df["dt_alloc_invest"])
            & (df["dt_close"].isna() | (df["date"] < df["dt_close"]))
        )
        df["wip_load"] = (
            df["fte"] * df["weighting"] * df["wip"].fillna(False).astype(float)
        ).astype(float)

        # Inter-pickup (gap between allocations) per staff
        if df["staff_id"].notna().any():
            df = df.sort_values(["staff_id", "dt_alloc_invest"])
            df["time_since_last_pickup"] = (
                df.groupby("staff_id")["dt_alloc_invest"].diff().dt.days.astype("float")
            )
        else:
            df["time_since_last_pickup"] = np.nan

        # Weeks since start
        start_date = (
            _to_date(pd.Series(pd.Timestamp(default_date_from))).iloc[0]
            if default_date_from
            else df["date"].min()
        )
        df["weeks_since_start"] = ((df["date"] - start_date).dt.days / 7.0).astype(
            float
        )

        # New starter flag (weeks from first allocation)
        if df["staff_id"].notna().any():
            first_alloc = df.groupby("staff_id")["dt_alloc_invest"].transform("min")
            weeks_from_first = ((df["date"] - first_alloc).dt.days / 7.0).astype(float)
            df["is_new_starter"] = _bool(
                weeks_from_first <= float(flags.new_starter_weeks)
            )
        else:
            df["is_new_starter"] = _bool(False)

        # Backlog availability flag
        df["backlog_available"] = _bool(
            (df["dt_received_inv"].notna())
            & (df["date"].notna())
            & (df["date"] >= df["dt_received_inv"])
            & (df["dt_close"].isna() | (df["date"] < df["dt_close"]))
        )

        # Term/seasonality
        df["term_flag"] = _bool(
            df["date"].dt.month.isin(set(int(m) for m in flags.term_months))
        )
        df["season"] = df["date"].dt.month.map(_SEASON_MAP).astype("string")
        df["dow"] = df["date"].dt.day_name().astype("string")

        # Bank holiday
        if bank_holidays is None:
            df["bank_holiday"] = _bool(False)
        else:
            bh = (
                pd.to_datetime(pd.Series(list(bank_holidays)), errors="coerce")
                .dt.normalize()
                .dropna()
                .unique()
            )
            df["bank_holiday"] = _bool(df["date"].isin(bh))

        # Event flags
        status_text = (
            df["status"].astype("string").str.lower().fillna("")
            + " "
            + df["concern_type"].astype("string").str.lower().fillna("")
            + " "
            + df["case_type"].astype("string").str.lower().fillna("")
        )
        df["event_newcase"] = _bool(df["date"].eq(df["dt_received_inv"]))
        df["event_pg_signoff"] = _bool(df["date"].eq(df["dt_pg_signoff"]))
        df["event_sent_to_ca"] = _bool(df["date"].eq(df["dt_sent_to_ca"]))
        df["event_legal"] = _bool(
            status_text.str.contains(r"\\blegal\\b|solicitor|attorney|advice")
        )
        df["event_court"] = _bool(
            status_text.str.contains(r"\\bcourt\\b|hearing|tribunal")
        )
        df["event_flagged"] = _bool(
            status_text.str.contains(r"\\bflag|priority|escalat")
        )

        # --- Backlog numeric column ---
        if backlog_series is not None and {"date", "backlog"}.issubset(
            set(map(str.lower, backlog_series.columns.str.lower()))
        ):
            # Standardise columns and merge on date
            bs = backlog_series.copy()
            # normalise headers
            cols_lower = {c: c.lower() for c in bs.columns}
            bs.rename(columns={c: c.lower() for c in bs.columns}, inplace=True)
            # ensure types
            bs["date"] = _to_date(bs["date"])
            bs["backlog"] = pd.to_numeric(bs["backlog"], errors="coerce")
            df = df.merge(
                bs[["date", "backlog"]].drop_duplicates("date"), on="date", how="left"
            )
        else:
            # Compute per observed 'date' (count of outstanding cases)
            # backlog(date) = sum( dt_received_inv <= date and (dt_close isna or dt_close > date) )
            # We'll compute on the set of dates that appear in df['date'].
            dates = df["date"].dropna().sort_values().unique()
            # Pre-calc arrays for vectorised comparison
            recv = df["dt_received_inv"].values
            close = df["dt_close"].values
            # For memory safety on very large data, fall back to a groupby boolean sum.
            # Here we try a straightforward loop over unique dates.
            bmap = {}
            for d in dates:
                # mask = (recv <= d) & (np.isnan(close) | (close > d))
                mask = (recv <= d) & (pd.isna(close) | (close > d))
                bmap[d] = int(mask.sum())
            df["backlog"] = df["date"].map(bmap).astype("float")

        # Ensure all required columns exist & order
        df = _ensure_columns(df, IntervalAnalysis.REQUIRED_COLUMNS)
        df = df[IntervalAnalysis.REQUIRED_COLUMNS].copy()

        # Dtypes
        for c in [
            "wip",
            "is_new_starter",
            "backlog_available",
            "term_flag",
            "bank_holiday",
            "event_newcase",
            "event_legal",
            "event_court",
            "event_pg_signoff",
            "event_sent_to_ca",
            "event_flagged",
        ]:
            df[c] = df[c].astype("boolean")
        for c in [
            "days_to_pg_signoff",
            "fte",
            "weighting",
            "wip_load",
            "time_since_last_pickup",
            "weeks_since_start",
            "backlog",
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in [
            "staff_id",
            "team",
            "case_id",
            "case_type",
            "concern_type",
            "status",
            "season",
            "dow",
        ]:
            df[c] = df[c].astype("string")
        for c in [
            "date",
            "dt_alloc_invest",
            "dt_pg_signoff",
            "dt_received_inv",
            "dt_alloc_team",
            "dt_close",
            "dt_sent_to_ca",
        ]:
            df[c] = _to_date(df[c])

        return df

    # ---- Analysis helpers (year focus) ----
    @staticmethod
    def filter_year(
        df: pd.DataFrame, y: int = y, anchor: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        if anchor is None:
            anchor = pd.Timestamp.today().normalize()
        start = anchor - pd.Timedelta(days=y * 365)
        return df[df["date"].between(start, anchor, inclusive="both")].copy()

    @staticmethod
    def interval_columns_available(df: pd.DataFrame) -> Dict[str, pd.Series]:
        out = {}
        if "days_to_pg_signoff" in df.columns:
            out["days_to_pg_signoff"] = df["days_to_pg_signoff"]
        if {"dt_close", "dt_alloc_invest"}.issubset(df.columns):
            out["days_alloc_to_close"] = (
                df["dt_close"] - df["dt_alloc_invest"]
            ).dt.days.astype("float")
        if {"dt_sent_to_ca", "dt_alloc_invest"}.issubset(df.columns):
            out["days_alloc_to_sent_to_ca"] = (
                df["dt_sent_to_ca"] - df["dt_alloc_invest"]
            ).dt.days.astype("float")
        if "time_since_last_pickup" in df.columns:
            out["inter_pickup_days"] = df["time_since_last_pickup"]
        return out

    @staticmethod
    def distribution_summary(s: pd.Series) -> Dict[str, Any]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {"count": 0}
        q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        return {
            "count": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "min": float(s.min()),
            "p10": float(q.loc[0.1]),
            "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.5]),
            "p75": float(q.loc[0.75]),
            "p90": float(q.loc[0.9]),
            "max": float(s.max()),
        }

    @staticmethod
    def analyse_interval_distributions(
        df: pd.DataFrame,
        *,
        anchor: Optional[pd.Timestamp] = None,
        by: Optional[list[str]] = None,
    ) -> Dict[str, Any]:

        dfl = IntervalAnalysis.filter_year(df, y=y, anchor=anchor)
        metrics = IntervalAnalysis.interval_columns_available(dfl)

        if not by:
            return {
                name: IntervalAnalysis.distribution_summary(series)
                for name, series in metrics.items()
            }

        # --- FIX: make group keys safe for dropna=False ---
        import pandas as pd

        safe_keys = []
        for c in by:
            s = dfl[c].astype("object")  # avoid pandas "string" NA semantics
            s = s.where(pd.notna(s), "__NA__")  # sentinel for missing category
            safe_keys.append(s)

        grouped = dfl.groupby(safe_keys, dropna=False)
        # -----------------------------------------------

        out = {}
        for name, series in metrics.items():
            blocks = {}
            for gkey, idx in grouped.groups.items():
                # normalise key to tuple and map sentinel back to None for readability
                gkey = gkey if isinstance(gkey, tuple) else (gkey,)
                gkey = tuple(None if x == "__NA__" else x for x in gkey)

                subset = series.loc[idx]  # subset the precomputed Series by index
                blocks[gkey] = IntervalAnalysis.distribution_summary(subset)
            out[name] = blocks
        return out

    @staticmethod
    def monthly_trend(
        df: pd.DataFrame,
        metric: str = "days_to_pg_signoff",
        *,
        anchor: Optional[pd.Timestamp] = None,
        agg: str = "median",
        by: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        dfl = IntervalAnalysis.filter_year(df, y=y, anchor=anchor)
        if metric not in dfl.columns:
            raise KeyError(f"Metric '{metric}' not in dataframe.")
        dfl = dfl.copy()
        dfl["yyyymm"] = dfl["date"].dt.to_period("M").astype(str)

        def _aggfunc(x):
            if isinstance(agg, str) and agg.startswith("p") and agg[1:].isdigit():
                q = int(agg[1:]) / 100.0
                return x.quantile(q)
            return getattr(x, agg)() if hasattr(x, agg) else x.median()

        if by:
            grp = (
                dfl.groupby(by + ["yyyymm"])[metric]
                .apply(_aggfunc)
                .reset_index(name=metric)
            )
            grp = grp.sort_values(by + ["yyyymm"]).assign(
                mom_delta=lambda g: g.groupby(by)[metric].diff()
            )
        else:
            grp = (
                dfl.groupby(["yyyymm"])[metric].apply(_aggfunc).reset_index(name=metric)
            )
            grp = grp.sort_values(["yyyymm"]).assign(
                mom_delta=lambda g: g[metric].diff()
            )
        return grp

    @staticmethod
    def volatility_score(
        df: pd.DataFrame,
        metric: str = "days_to_pg_signoff",
        *,
        anchor: Optional[pd.Timestamp] = None,
        freq: str = "W",
        by: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        dfl = IntervalAnalysis.filter_year(df, y=y, anchor=anchor)
        if metric not in dfl.columns:
            raise KeyError(f"Metric '{metric}' not in dataframe.")
        dfl = dfl.set_index("date")

        if by:
            pieces = []
            for keys, g in dfl.groupby(by, dropna=False):
                bucket = g[metric].resample(freq).median()
                vol = bucket.std()
                row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
                row.update({"metric": metric, "freq": freq, "volatility": vol})
                pieces.append(row)
            return pd.DataFrame(pieces)
        else:
            bucket = dfl[metric].resample(freq).median()
            return pd.DataFrame(
                {"metric": [metric], "freq": [freq], "volatility": [bucket.std()]}
            )


# --- Usage (examples) ---
# NOTE: Examples are commented out to avoid altering notebooks' execution flow.
# You can un-comment and run after your usual pipeline steps.
#
# engineered = engineer(raw_df, colmap)                     # existing function
# daily = build_daily_panel(engineered)                     # existing function
# backlog_series = build_backlog_series(engineered)         # existing function (date, backlog)
#
# df_interval = IntervalAnalysis.build_interval_frame(
#     engineered, backlog_series=backlog_series, bank_holidays=None
# )
# summaries = IntervalAnalysis.analyse_interval_distributions(df_interval, by=["team"])
# trend = IntervalAnalysis.monthly_trend(df_interval, metric="days_to_pg_signoff", agg="median", by=["team"])
# vol = IntervalAnalysis.volatility_score(df_interval, metric="inter_pickup_days", freq="W", by=["staff_id"])


# --- Demo: Interval analysis by team (safe to run multiple times) ---

engineered = typed  # existing function
backlog_series = backlog  # existing function (date, backlog)

df_interval = IntervalAnalysis.build_interval_frame(
    engineered, backlog_series=backlog_series, bank_holidays=None
)

# IntervalAnalysis.analyse_interval_distributions = staticmethod(
#     _analyse_interval_distributions_fixed
# )

# Summaries by team
summ = IntervalAnalysis.analyse_interval_distributions(df_interval, by=["case_type"])

# Flatten to a quick table
rows = []
for metric, groups in summ.items():
    for gkey, stats in groups.items():
        case_type = gkey[0] if isinstance(gkey, tuple) else gkey
        row = {
            "metric": metric,
            "case_type": case_type,
            "count": stats.get("count", np.nan),
            "mean": stats.get("mean", np.nan),
            "std": stats.get("std", np.nan),
            "median": stats.get("p50", np.nan),
            "p90": stats.get("p90", np.nan),
        }
        rows.append(row)
summary_table = (
    pd.DataFrame(rows).sort_values(["metric", "case_type"]).reset_index(drop=True)
)
display(summary_table)

# Monthly trend (median) with MoM delta for days_to_pg_signoff by team
trend = IntervalAnalysis.monthly_trend(
    df_interval, metric="days_to_pg_signoff", agg="median", by=["case_type"]
)
print("\nMonthly median trend with MoM delta for 'days_to_pg_signoff' by case type:")
display(trend.tail(24))


# let volatility_score accept aliases / computed metrics
def _volatility_score_safe(
    df, metric: str = "days_to_pg_signoff", *, anchor=None, freq: str = "W", by=None
):
    dfl = IntervalAnalysis.filter_year(df, y=y, anchor=anchor)

    # Allow aliases or compute-on-the-fly metrics
    if metric not in dfl.columns:
        if metric == "inter_pickup_days" and "time_since_last_pickup" in dfl.columns:
            dfl = dfl.assign(inter_pickup_days=dfl["time_since_last_pickup"])
        elif metric == "days_alloc_to_close" and {
            "dt_close",
            "dt_alloc_invest",
        }.issubset(dfl.columns):
            dfl = dfl.assign(
                days_alloc_to_close=(
                    dfl["dt_close"] - dfl["dt_alloc_invest"]
                ).dt.days.astype("float")
            )
        elif metric == "days_alloc_to_sent_to_ca" and {
            "dt_sent_to_ca",
            "dt_alloc_invest",
        }.issubset(dfl.columns):
            dfl = dfl.assign(
                days_alloc_to_sent_to_ca=(
                    dfl["dt_sent_to_ca"] - dfl["dt_alloc_invest"]
                ).dt.days.astype("float")
            )
        else:
            raise KeyError(f"Metric '{metric}' not in dataframe and cannot be derived.")
    dfl = dfl.set_index("date")

    if by:
        pieces = []
        for keys, g in dfl.groupby(by, dropna=False):
            bucket = g[metric].resample(freq).median()
            vol = bucket.std()
            row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
            row.update({"metric": metric, "freq": freq, "volatility": vol})
            pieces.append(row)
        return pd.DataFrame(pieces)
    else:
        bucket = dfl[metric].resample(freq).median()
        return pd.DataFrame(
            {"metric": [metric], "freq": [freq], "volatility": [bucket.std()]}
        )


# Apply monkey patch
IntervalAnalysis.volatility_score = staticmethod(_volatility_score_safe)
print("Patched IntervalAnalysis.volatility_score to support metric aliases.")

vol = IntervalAnalysis.volatility_score(
    df_interval, metric="inter_pickup_days", freq="W", by=["staff_id"]
)
print("\nInterval Analysis:")
display(vol.tail(24))


# --- Plot monthly trend for days_to_pg_signoff by team ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

eng = typed
backlog_series = backlog
di = IntervalAnalysis.build_interval_frame(
    eng, backlog_series=backlog_series, bank_holidays=None
)

# Compute last-year monthly trend (median) with MoM delta by case_type
# Monthly trend since 2022-01 (median) with MoM delta by case_type
trend = IntervalAnalysis.monthly_trend(
    di, metric="days_to_pg_signoff", agg="median", by=["case_type"]
).copy()

trend["month"] = pd.to_datetime(trend["yyyymm"] + "-01")
# trend = IntervalAnalysis.monthly_trend(di, metric="days_to_pg_signoff", agg="median", by=["case_type"]).copy()

# Parse yyyymm into datetime for plotting
trend["month"] = pd.to_datetime(trend["yyyymm"] + "-01")

# Pivot for plotting
piv = trend.pivot(
    index="month", columns="case_type", values="days_to_pg_signoff"
).sort_index()
piv_delta = trend.pivot(
    index="month", columns="case_type", values="mom_delta"
).sort_index()

# Create output directory
outdir = Path("data/out/plot/plots")
outdir.mkdir(parents=True, exist_ok=True)

# 1) Monthly median lines
plt.figure(figsize=(16, 9))
for col in piv.columns:
    plt.plot(piv.index, piv[col], label=str(col))
plt.title("Monthly median: days_to_pg_signoff by case_type")
plt.xlabel("Month")
plt.ylabel("Days to PG signoff (median)")
plt.xticks(rotation=45)
# Add space at the edges:
plt.subplots_adjust(right=0.85, top=0.92, bottom=0.15)
# Put the legend below the chart:
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
# plt.legend()
plot1 = outdir / "monthly_trend_days_to_pg_signoff_by_case_type.png"
plt.savefig(plot1, bbox_inches="tight", dpi=150)
plt.show()

# 2) Month-over-month delta lines
plt.figure(figsize=(16, 9))
for col in piv_delta.columns:
    plt.plot(piv_delta.index, piv_delta[col], label=str(col))
plt.title("Monthly MoM delta: days_to_pg_signoff by case_type")
plt.xlabel("Month")
plt.ylabel("MoM delta (days)")
plt.xticks(rotation=45)
plt.legend()
plot2 = outdir / "monthly_mom_delta_days_to_pg_signoff_by_case_type.png"
plt.savefig(plot2, bbox_inches="tight", dpi=150)
plt.show()


# --- Overall ("all case types") monthly trend & MoM delta ---

# The try: path gives the true overall median across all cases per month.
# The fallback uses the median of case-type medians (good enough if counts per type are similar).
# The two “overlay” plots help compare each case type against the bold “ALL case types” line.
# Try to get a no-split trend directly; fall back to aggregating the case_type trends.
try:
    trend_all = IntervalAnalysis.monthly_trend(
        di, metric="days_to_pg_signoff", agg="median"  # no 'by' -> overall
    ).copy()
    trend_all["month"] = pd.to_datetime(trend_all["yyyymm"] + "-01")
except Exception:
    # Fallback: median across case-type medians (approximation if counts differ)
    trend_all = trend.groupby("yyyymm", as_index=False).agg(
        days_to_pg_signoff=("days_to_pg_signoff", "median"),
        mom_delta=("mom_delta", "median"),
    )
    trend_all["month"] = pd.to_datetime(trend_all["yyyymm"] + "-01")

s_all = trend_all.set_index("month")["days_to_pg_signoff"].sort_index()
s_all_delta = trend_all.set_index("month")["mom_delta"].sort_index()

# 3) Standalone: All case types — monthly median
plt.figure(figsize=(16, 9))
plt.plot(s_all.index, s_all.values, marker="o")
plt.title("Monthly median: days_to_pg_signoff — ALL case types")
plt.xlabel("Month")
plt.ylabel("Days to PG signoff (median)")
plt.xticks(rotation=45)
plot3 = outdir / "monthly_trend_days_to_pg_signoff_ALL.png"
plt.savefig(plot3, bbox_inches="tight", dpi=150)
plt.show()

# 4) Standalone: All case types — MoM delta
plt.figure(figsize=(16, 9))
plt.plot(s_all_delta.index, s_all_delta.values, marker="o")
plt.title("Monthly MoM delta: days_to_pg_signoff — ALL case types")
plt.xlabel("Month")
plt.ylabel("MoM delta (days)")
plt.xticks(rotation=45)
plot4 = outdir / "monthly_mom_delta_days_to_pg_signoff_ALL.png"
plt.savefig(plot4, bbox_inches="tight", dpi=150)
plt.show()

# (Optional) Overlay the ALL line on your existing multi-line charts for quick comparison
plt.figure(figsize=(16, 9))
for col in piv.columns:
    plt.plot(piv.index, piv[col], alpha=0.6, label=str(col))
plt.plot(s_all.index, s_all.values, linewidth=3, label="ALL case types")
plt.title("Monthly median: days_to_pg_signoff by case_type + ALL")
plt.xlabel("Month")
plt.ylabel("Days to PG signoff (median)")
plt.xticks(rotation=45)
plt.legend(ncol=2, fontsize=8)
plot5 = outdir / "monthly_trend_days_to_pg_signoff_by_case_type_with_ALL.png"
plt.savefig(plot5, bbox_inches="tight", dpi=150)
plt.show()

plt.figure(figsize=(16, 9))
for col in piv_delta.columns:
    plt.plot(piv_delta.index, piv_delta[col], alpha=0.6, label=str(col))
plt.plot(s_all_delta.index, s_all_delta.values, linewidth=3, label="ALL case types")
plt.title("Monthly MoM delta: days_to_pg_signoff by case_type + ALL")
plt.xlabel("Month")
plt.ylabel("MoM delta (days)")
plt.xticks(rotation=45)
plt.legend(ncol=2, fontsize=8)
plot6 = outdir / "monthly_mom_delta_days_to_pg_signoff_by_case_type_with_ALL.png"
plt.savefig(plot6, bbox_inches="tight", dpi=150)
plt.show()

print("Saved plots to:", plot3, plot4, plot5, "and", plot6)

# === END NOTEBOOK CODE ===
