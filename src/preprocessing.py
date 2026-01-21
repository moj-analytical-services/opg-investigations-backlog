# """Data preprocessing / manipulation / missing-data (re-exports from notebook)."""
# from .notebook_code import (
#     normalise_col, parse_date_series, hash_id, month_to_season, is_term_month,
#     load_raw, col, engineer
# )
# __all__ = ['normalise_col','parse_date_series','hash_id','month_to_season','is_term_month','load_raw','col','engineer']

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

        # --- handle Excel serial dates like 45786 -> 2025-05-09 ---
        if isinstance(x, (int, float)) and not pd.isna(x):
            # Excel serial (Windows 1900 date system)
            if x > 1000:  # guard against small integers that aren't dates
                return pd.to_datetime(int(x), unit="D", origin="1899-12-30")

        # Also handle numeric strings e.g. "45786" or "45786.0"
        if isinstance(x, str):
            x_strip = x.strip()
            x_num = pd.to_numeric(x_strip, errors="coerce")
            if pd.notna(x_num) and x_num > 1000:
                return pd.to_datetime(int(x_num), unit="D", origin="1899-12-30")
        # --- end of the block ---

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
            return pd.to_datetime(xs, dayfirst=True, errors="coerce")

    # Apply the parser to each element of the Series
    return s.apply(_p)

    if s is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0))

    # If numeric-like (possible Excel serials), try converting via pandas
    s_num = pd.to_numeric(s, errors="coerce")
    has_numeric = s_num.notna().any()

    # First pass: assume strings with day-first ambiguity handled later
    dt1 = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True)

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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

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
# Helper: application_type_from_case_id()
# -------------------------------------------------------------
def application_type_from_case_id(s: pd.Series) -> pd.Series:
    """
    Derive application type (0 = LPA, 1 = deputyship) from the
    'LPA or DeputyID' field.

    Logic:
      - If the value contains an LPA-style registration number:
            4 digits, '-' or space, 4 digits, '-' or space, 4 digits
        e.g. '7001-6571-3350', '7001 6571 3350'
        → classify as LPA (0).

      - Everything else (including MERIS-style numeric IDs
        with 7–8 digits, COP case numbers, 'PFA ...', 'AiA', 'SRA',
        'Multiple\\n12304198', etc.) is treated as a deputyship
        identifier → classify as 1.

    Missing / blank values are returned as <NA>.
    """
    # Normalise to string and trim whitespace
    s = s.astype("string").str.strip()

    # Identify missing values
    is_missing = s.isna() | (s == "")

    # LPA-style numbers: 4 digits, '-' OR space, 4 digits, '-' OR space, 4 digits
    lpa_pattern = r"\b\d{4}[- ]\d{4}[- ]\d{4}\b"
    has_lpa_number = s.str.contains(lpa_pattern, regex=True, na=False)

    # 0 = LPA, 1 = deputyship (Meris, COP order IDs, etc.)
    app_type = pd.Series(
        np.where(has_lpa_number, 0, 1),
        index=s.index,
        dtype="Int64",
        name="application_type",
    )

    # Put back proper missing values
    app_type[is_missing] = pd.NA

    return app_type


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

    # Derive application type from the case-ID-like field
    # 0 = LPA, 1 = deputyship
    lpa_dep_series = col(
        df, colmap, "LPA or DeputyID"
    )  # or col(df, colmap, "ID") if that's your case ID
    out["application_type"] = application_type_from_case_id(lpa_dep_series)

    # Parse and standardise relevant date columns
    out["dt_received_inv"] = parse_date_series(
        col(df, colmap, "Date Received in Investigations")
    )
    out["dt_alloc_invest"] = parse_date_series(
        col(df, colmap, "Date allocated to current investigator")
    )
    out["days_to_alloc"] = (out["dt_alloc_invest"] - out["dt_received_inv"]).dt.days

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

    # --- Legal review request date & indicator ---
    # 1) Pull the raw date column from the real dataset
    #    (change the string below if your header is slightly different)
    raw_legal_date = col(df, colmap, "Date of Legal Review Request 1")

    # 2) Parse to datetime, same as other date fields
    out["dt_legal_review_req1"] = parse_date_series(raw_legal_date)
    # out["dt_legal_req_1"] = pd.to_datetime(out["dt_legal_review_req1"], unit="D", origin="1899-12-30", errors="coerce")

    # 3) Create a binary 0/1 indicator: 1 if any legal review request date present, else 0
    out["legal_review"] = out["dt_legal_review_req1"].notna().astype("int8")

    # # 4) Optionally make it a categorical type (still 0/1 values)
    # out["legal_review"] = out["legal_review"].astype("category")
    out["legal_review_cat"] = out["legal_review"].astype("category")

    # --- Investigation Status ---
    # Pull the raw Status column from the real dataset
    # Adjust the string if your raw header is different, e.g. "Investigation Status"
    raw_status = col(df, colmap, "Status")
    # Tidy up whitespace
    status_clean = raw_status.astype("string").str.strip()

    # Map any variants to the exact categories we want
    status_clean = status_clean.replace(
        {
            "To be allocated": "To be allocated",
            "Awaiting investigator": "Awaiting investigator",
            "Investigation Phase": "Investigation Phase",
            "Closed": "Closed",
            "No further action": "No further action",
            "Further action": "Further action",
            # Add any other raw spellings / capitalisation here if needed
            # "Awaiting Investigator": "Awaiting investigator",
            # "INVESTIGATION PHASE": "Investigation Phase",
        }
    )

    # Categorical status with a fixed, ordered set of levels
    out["status"] = pd.Categorical(
        status_clean,
        categories=[
            "To be allocated",
            "Awaiting investigator",
            "Investigation Phase",
            "Closed",
            "No further action",
            "Further action",
        ],
        ordered=True,  # set to False if you don't care about ordering
    )

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

    # Compute days_to_alloc if wholly missing but dates exist
    if (
        out["days_to_alloc"].isna().all()
        and ("dt_alloc_invest" in out)
        and ("dt_received_inv" in out)
    ):
        diff = (out["dt_alloc_invest"] - out["dt_received_inv"]).dt.days
        out["days_to_alloc"] = pd.to_numeric(diff, errors="coerce")

    # --- Derive a clean boolean, then optionally filter ---
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


__all__ = [
    "RAW_PATH",
    "OUT_DIR",
    "NULL_STRINGS",
    "normalise_col",
    "parse_date_series",
    "hash_id",
    "month_to_season",
    "is_term_month",
    "load_raw",
    "col",
    "engineer",
    "date_horizon",
]

# Data Manipulation and Processing
# ## date_horizon
# - We need a date window (a start and end date) to analyze. By policy:
#     - Start comes from the earliest “date received in investigations” (dt_received_inv).
#     - End comes from the latest “PG sign-off date” (dt_pg_signoff) plus a small buffer (pad_days) to capture tail activity.
#     - If those columns are missing/empty, we can fall back to the earliest and latest across any dt_… date columns.
#     - If that still fails, we default to “last 30 days up to today (+ padding)”.

# - If either start or end is still missing and we’re allowed to fall back:
#     - Collect all columns whose names start with dt_.
#     - Stack them together, drop missing values.
#     - Use the earliest date as start and latest date as end if needed.
# - This keeps our analysis consistent and prevents accidental trimming when some dates are missing.
# - Keeps the meaning of the analysis window aligned with process reality (received → sign-off).
# - Robust to missing data thanks to fallbacks and sensible defaults.
