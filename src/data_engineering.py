import pandas as pd

# --- fabricate a tiny typed dataset ---
typed = pd.DataFrame({
    'case_id': ['C1','C2'],
    'investigator': ['Alice','Bob'],
    'team': ['T1','T1'],
    'role': ['',''],
    'fte': [1.0, 0.8],
    'staff_id': ['S1','S2'],

    # key dates
    'dt_received_inv': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02')],
    'dt_alloc_invest': [pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-03')],
    'dt_alloc_team': [pd.NaT, pd.NaT],
    'dt_pg_signoff': [pd.NaT, pd.Timestamp('2025-01-08')],
    'dt_close': [pd.Timestamp('2025-01-06'), pd.NaT],

    # events
    'dt_legal_req_1': [pd.NaT, pd.Timestamp('2025-01-04')],
    'dt_legal_req_2': [pd.NaT, pd.NaT],
    'dt_legal_req_3': [pd.NaT, pd.NaT],
    'dt_legal_approval': [pd.NaT, pd.NaT],
    'dt_date_of_order': [pd.NaT, pd.NaT],
    'dt_flagged': [pd.NaT, pd.NaT],
})

# --- run horizon, events, wip, and panel ---
start, end = date_horizon(typed, pad_days=3)
daily, backlog, events = build_daily_panel(typed, start, end)

print("Start/End:", start.date(), end.date())
print("Daily shape:", daily.shape)
print("Backlog shape:", backlog.shape)
print("Events shape:", events.shape)

print("\nDaily head:\n", daily.head())
print("\nBacklog tail:\n", backlog.tail())
print("\nEvents:\n", events.sort_values(['date','staff_id','event']))


# -----------------------------
# 🧹 DATA PRE-PROCESSING SECTION
# -----------------------------

import re
import hashlib
import pandas as pd

# Define a set of string patterns that represent missing or null values.
# These strings will be treated as equivalent to NaN during cleaning.
NULL_STRINGS = {
    '', 'na', 'n/a', 'none', 'null', '-', '--', 'unknown',
    'not completed', 'not complete', 'tbc', 'n\\a'
}


def normalise_col(c: str) -> str:
    """
    Normalize a column name for consistency.

    This function cleans up and standardizes column names by:
    - Converting to lowercase
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
    return re.sub(r'\s+', ' ', str(c).strip().lower())


def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Parse and clean a pandas Series of date strings.

    This function:
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
        xs = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', xs).replace('legel', 'legal')

        # Try strict parsing, then flexible fallback
        try:
            return pd.to_datetime(xs, dayfirst=True, errors='raise')
        except Exception:
            return pd.to_datetime(xs, infer_datetime_format=True, dayfirst=True, errors='coerce')

    # Apply the parser to each element of the Series
    return s.apply(_p)


def hash_id(t: str) -> str:
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
        An anonymized hash string prefixed with 'S', e.g., 'S1a2b3c4d'.
    """
    # Return empty string for null or blank input
    if pd.isna(t) or str(t).strip() == '':
        return ''

    # Create SHA1 hash and take first 8 characters for compact ID
    return 'S' + hashlib.sha1(str(t).encode('utf-8')).hexdigest()[:8]


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
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    }[int(m)]


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
# 🧩 DATA LOADING AND FEATURE ENGINEERING
# -------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import re
import hashlib

# -------------------------------------------------------------
# Function: load_raw()
# -------------------------------------------------------------
def load_raw(p: Path, force_encoding: str | None = None):
    """
    Load a CSV or Excel file into a pandas DataFrame with robust encoding handling.

    This function attempts to open and read raw data files safely, even when
    character encodings vary or are unknown. It tries multiple encodings in order
    until one succeeds.

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
    import io

    # Check file existence
    if not p.exists():
        raise FileNotFoundError(p)

    # Excel files typically do not have encoding issues
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p, dtype=str)
    else:
        tried = []
        # Build list of encodings to try
        encodings_to_try = (
            [force_encoding] if force_encoding else
            ["utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]
        )

        df = None
        last_err = None

        # Try to read using multiple encodings
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(
                    p, dtype=str, sep=None, engine="python",
                    encoding=enc, encoding_errors="strict"
                )
                break
            except UnicodeDecodeError as e:
                tried.append(enc)
                last_err = e
            except Exception as e:
                # Continue trying other encodings
                tried.append(enc)
                last_err = e

        # Fallback: attempt to decode with cp1252 and replace bad bytes
        if df is None:
            try:
                df = pd.read_csv(
                    p, dtype=str, sep=None, engine="python",
                    encoding="cp1252", encoding_errors="replace"
                )
                print(f"[load_raw] WARNING: used cp1252 with replacement after failed encodings: {tried}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read CSV. Tried encodings {tried}. Last error: {last_err}"
                ) from e

    # Strip whitespace from all string values
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Create mapping of normalized column names → original names
    colmap = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in df.columns}

    return df, colmap


# -------------------------------------------------------------
# Function: col()
# -------------------------------------------------------------
def col(df: pd.DataFrame, colmap: dict, name: str) -> pd.Series:
    """
    Retrieve a column from a DataFrame by fuzzy name matching.

    This function normalizes the requested column name and searches the column map
    for an exact or partial match. Returns a Series of NaNs if not found.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.
    colmap : dict
        Mapping of normalized column names to original names.
    name : str
        Column name to look up.

    Returns
    -------
    pd.Series
        The column data if found, otherwise a Series of NaN values.
    """
    k = normalise_col(name)

    # Exact match first
    if k in colmap:
        return df[colmap[k]]

    # Partial match fallback
    for kk, v in colmap.items():
        if k in kk or kk in k:
            return df[v]

    # Default: return empty column of NaNs
    return pd.Series([np.nan] * len(df))


# -------------------------------------------------------------
# Function: engineer()
# -------------------------------------------------------------
def engineer(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """
    Engineer standardized and typed columns from raw investigation data.

    This function extracts and converts the key variables such as case IDs, investigators,
    FTEs, and multiple date columns from the raw file using reusable helper functions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_raw().
    colmap : dict
        Column name mapping from load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered dataframe ready for downstream modeling.
    """
    out = pd.DataFrame({
        'case_id': col(df, colmap, 'ID'),
        'investigator': col(df, colmap, 'Investigator'),
        'team': col(df, colmap, 'Team'),
        'fte': pd.to_numeric(col(df, colmap, 'Investigator FTE'), errors='coerce')
    })

    # Parse and standardize all relevant date columns
    out['dt_received_inv'] = parse_date_series(col(df, colmap, 'Date Received in Investigations'))
    out['dt_alloc_invest'] = parse_date_series(col(df, colmap, 'Date allocated to current investigator'))
    out['dt_alloc_team'] = parse_date_series(col(df, colmap, 'Date allocated to team'))
    out['dt_pg_signoff'] = parse_date_series(col(df, colmap, 'PG Sign off date'))
    out['dt_close'] = parse_date_series(col(df, colmap, 'Closure Date'))
    out['dt_legal_req_1'] = parse_date_series(col(df, colmap, 'Date of Legal Review Request 1'))
    out['dt_legal_rej_1'] = parse_date_series(col(df, colmap, 'Date Legal Rejects 1'))
    out['dt_legal_req_2'] = parse_date_series(col(df, colmap, 'Date of Legal Review Request 2'))
    out['dt_legal_rej_2'] = parse_date_series(col(df, colmap, 'Date Legal Rejects 2'))
    out['dt_legal_req_3'] = parse_date_series(col(df, colmap, 'Date of Legel Review Request 3'))
    out['dt_legal_approval'] = parse_date_series(col(df, colmap, 'Legal Approval Date'))
    out['dt_date_of_order'] = parse_date_series(col(df, colmap, 'Date Of Order'))
    out['dt_flagged'] = parse_date_series(col(df, colmap, 'Flagged Date'))

    # Fill missing FTEs with 1.0, hash investigator names for anonymization, and add placeholders
    out['fte'] = out['fte'].fillna(1.0)
    out['staff_id'] = out['investigator'].apply(hash_id)
    out['role'] = ''

    return out


# -------------------------------------------------------------
# Function: date_horizon()
# -------------------------------------------------------------
def date_horizon(typed: pd.DataFrame, pad_days: int = 14):
    """
    Determine the overall start and end date horizon of the dataset.

    Combines several date columns to find the earliest and latest dates,
    applying a configurable padding period at the end.

    Parameters
    ----------
    typed : pd.DataFrame
        Feature-engineered dataset with standardized date columns.
    pad_days : int, default=14
        Number of days to extend the end horizon.

    Returns
    -------
    tuple of pd.Timestamp
        (start, end) normalized date range.

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
    start = pd.concat([typed['dt_received_inv'], typed['dt_alloc_invest'], typed['dt_alloc_team']]).min()
    end = pd.concat([typed['dt_close'], typed['dt_pg_signoff'], typed['dt_date_of_order']]).max()

    if pd.isna(start):
        start = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
    if pd.isna(end):
        end = pd.Timestamp.today().normalize()

    end = end + pd.Timedelta(days=pad_days)
    return start.normalize(), end.normalize()


def build_event_log(typed: pd.DataFrame) -> pd.DataFrame:
    """
    Construct an event log from feature-engineered investigation data.

    For each case, this function creates dated event records (e.g., new case pickup,
    legal requests/approvals, court orders) at the staff-day level.

    Parameters
    ----------
    typed : pd.DataFrame
        Must include:
        ['staff_id','team','fte','case_id',
         'dt_alloc_invest','dt_legal_req_1','dt_legal_req_2','dt_legal_req_3',
         'dt_legal_approval','dt_date_of_order'].

    Returns
    -------
    pd.DataFrame
        ['date','staff_id','team','fte','case_id','event','meta'].

    Examples
    --------
    >>> import pandas as pd
    >>> typed = pd.DataFrame({
    ...     'staff_id':['S1'], 'team':['A'], 'fte':[1.0], 'case_id':['C1'],
    ...     'dt_alloc_invest':[pd.Timestamp('2025-01-10')],
    ...     'dt_legal_req_1':[pd.NaT], 'dt_legal_req_2':[pd.NaT], 'dt_legal_req_3':[pd.NaT],
    ...     'dt_legal_approval':[pd.Timestamp('2025-01-20')],
    ...     'dt_date_of_order':[pd.NaT],
    ... })
    >>> ev = build_event_log(typed)
    >>> sorted(ev['event'].unique().tolist())
    ['legal_approval', 'newcase']
    >>> set(ev.columns) >= {'date','staff_id','team','fte','case_id','event','meta'}
    True
    """
    rec = []
    for _, r in typed.iterrows():
        sid, team, fte, cid = r['staff_id'], r['team'], r['fte'], r['case_id']

        def add(dt, etype):
            if pd.isna(dt):
                return
            rec.append({
                'date': dt.normalize(),
                'staff_id': sid,
                'team': team,
                'fte': fte,
                'case_id': cid,
                'event': etype,
                'meta': ''
            })

        add(r['dt_alloc_invest'], 'newcase')
        add(r['dt_legal_req_1'], 'legal_request')
        add(r['dt_legal_req_2'], 'legal_request')
        add(r['dt_legal_req_3'], 'legal_request')
        add(r['dt_legal_approval'], 'legal_approval')
        add(r['dt_date_of_order'], 'court_order')

    ev = pd.DataFrame.from_records(rec)
    return ev if not ev.empty else pd.DataFrame(
        columns=['date', 'staff_id', 'team', 'fte', 'case_id', 'event', 'meta']
    )


def build_wip_series(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Build a Work-In-Progress (WIP) daily series per staff member.

    A case is in WIP from allocation to earliest of (closure, PG sign-off, end).

    Parameters
    ----------
    typed : pd.DataFrame
        ['staff_id','team','dt_alloc_invest','dt_close','dt_pg_signoff'].
    start : pd.Timestamp
    end : pd.Timestamp

    Returns
    -------
    pd.DataFrame
        ['date','staff_id','team','wip'].

    Examples
    --------
    >>> import pandas as pd
    >>> typed = pd.DataFrame({
    ...     'staff_id':['S1','S1'], 'team':['A','A'],
    ...     'dt_alloc_invest':[pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-05')],
    ...     'dt_close':[pd.Timestamp('2025-01-03'), pd.NaT],
    ...     'dt_pg_signoff':[pd.NaT, pd.Timestamp('2025-01-07')],
    ... })
    >>> wip = build_wip_series(typed, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-10'))
    >>> set(wip.columns) == {'date','staff_id','team','wip'}
    True
    >>> wip['wip'].ge(0).all()
    True
    """
    end_dt = typed['dt_close'].fillna(typed['dt_pg_signoff']).fillna(end)
    intervals = pd.DataFrame({
        'staff_id': typed['staff_id'],
        'team': typed['team'],
        'start': typed['dt_alloc_invest'],
        'end': end_dt
    }).dropna()

    deltas = []
    for _, r in intervals.iterrows():
        s = r['start'].normalize()
        e = r['end'].normalize()
        if s > end or e < start:
            continue
        s = max(s, start)
        e = min(e, end)
        deltas.append((r['staff_id'], r['team'], s, 1))
        deltas.append((r['staff_id'], r['team'], e + pd.Timedelta(days=1), -1))

    if not deltas:
        return pd.DataFrame(columns=['date', 'staff_id', 'team', 'wip'])

    deltas = pd.DataFrame(deltas, columns=['staff_id', 'team', 'date', 'delta'])
    all_dates = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})

    rows = []
    for (sid, team), g in deltas.groupby(['staff_id', 'team']):
        gg = g.groupby('date', as_index=False)['delta'].sum()
        grid = all_dates.merge(gg, on='date', how='left').fillna({'delta': 0})
        grid['wip'] = grid['delta'].cumsum()
        grid['staff_id'] = sid
        grid['team'] = team
        rows.append(grid[['date', 'staff_id', 'team', 'wip']])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['date', 'staff_id', 'team', 'wip']
    )


def build_backlog_series(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Build a daily backlog series (accepted minus allocated cumulative totals).

    Parameters
    ----------
    typed : pd.DataFrame
        ['dt_received_inv','dt_alloc_invest'].
    start : pd.Timestamp
    end : pd.Timestamp

    Returns
    -------
    pd.DataFrame
        ['date','backlog_available'].

    Examples
    --------
    >>> import pandas as pd
    >>> typed = pd.DataFrame({
    ...     'dt_received_inv':[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-03')],
    ...     'dt_alloc_invest':[pd.Timestamp('2025-01-02'), pd.NaT],
    ... })
    >>> start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-05')
    >>> backlog = build_backlog_series(typed, start, end)
    >>> list(backlog.columns)
    ['date', 'backlog_available']
    >>> backlog.iloc[-1]['backlog_available']  # 2 received, 1 allocated -> 1
    1
    """
    accepted = (
        typed[['dt_received_inv']]
        .dropna()
        .assign(date=lambda d: d['dt_received_inv'].dt.normalize())['date']
        .value_counts()
        .sort_index()
    )
    allocated = (
        typed[['dt_alloc_invest']]
        .dropna()
        .assign(date=lambda d: d['dt_alloc_invest'].dt.normalize())['date']
        .value_counts()
        .sort_index()
    )

    idx = pd.date_range(start, end, freq='D')
    acc = accepted.reindex(idx, fill_value=0).cumsum()
    allo = allocated.reindex(idx, fill_value=0).cumsum()
    backlog = (acc - allo).rename('backlog_available').to_frame()
    backlog.index.name = 'date'
    return backlog.reset_index()



# --- fabricate a tiny typed dataset ---
typed = pd.DataFrame({
    'case_id': ['C1','C2'],
    'investigator': ['Alice','Bob'],
    'team': ['T1','T1'],
    'role': ['',''],
    'fte': [1.0, 0.8],
    'staff_id': ['S1','S2'],

    # key dates
    'dt_received_inv': [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02')],
    'dt_alloc_invest': [pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-03')],
    'dt_alloc_team': [pd.NaT, pd.NaT],
    'dt_pg_signoff': [pd.NaT, pd.Timestamp('2025-01-08')],
    'dt_close': [pd.Timestamp('2025-01-06'), pd.NaT],

    # events
    'dt_legal_req_1': [pd.NaT, pd.Timestamp('2025-01-04')],
    'dt_legal_req_2': [pd.NaT, pd.NaT],
    'dt_legal_req_3': [pd.NaT, pd.NaT],
    'dt_legal_approval': [pd.NaT, pd.NaT],
    'dt_date_of_order': [pd.NaT, pd.NaT],
    'dt_flagged': [pd.NaT, pd.NaT],
})

# --- run horizon, events, wip, and panel ---
start, end = date_horizon(typed, pad_days=3)
daily, backlog, events = build_daily_panel(typed, start, end)

print("Start/End:", start.date(), end.date())
print("Daily shape:", daily.shape)
print("Backlog shape:", backlog.shape)
print("Events shape:", events.shape)

print("\nDaily head:\n", daily.head())
print("\nBacklog tail:\n", backlog.tail())
print("\nEvents:\n", events.sort_values(['date','staff_id','event']))