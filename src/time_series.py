import pandas as pd

from preprocessing import (
    month_to_season,
    is_term_month,
    date_horizon,
)  # or other helpers if needed


# ## build_event_log
# - We need a timeline of events (one row per date × staff × case × event) to understand what happened and when.
# This function:
#     - Computes the analysis window using date_horizon.
#     - Scans each case for milestone dates (received, allocated, legal steps, sign-off, closed, etc.).
#     - Writes an event row for every milestone date within the horizon.
#     - Adds a compact meta JSON with context (weighting, case type, status, etc.) so we can analyze later without bloating columns.
# - Produces a long-format event timeline that’s ideal for plotting, counting, and modelling (“how many sign-offs per week?”, “who picked up cases today?”).
# - The meta JSON keeps useful context for each event without making the table very wide and repetitive.
# - Restricting to the computed horizon ensures the log always aligns with your agreed analysis window.


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


# - Build a day-by-day series showing how many cases each investigator has “in progress” (WIP), and an optional workload measure that accounts for case complexity and staff FTE.

# - A case is counted as WIP from the day it’s allocated to an investigator until the earliest of:
#     - it is closed, it gets PG sign-off, or we reach the reporting end date.

# - We want a daily time series showing, for each staff member, how many cases they are actively working (WIP = Work In Progress) and a simple workload measure that adjusts for case complexity and staff capacity.
#     - A case counts as WIP from the day it is allocated to an investigator (dt_alloc_invest) until the earliest of:
#         - the case is closed (dt_close), or
#         - it receives PG sign-off (dt_pg_signoff), or
#         - we reach the reporting end date.

# - Output is one row per date × staff member × team, with:
#     - wip (how many cases they have on the go) and wip_load (a proxy for workload = weighting ÷ FTE), summed over their active cases.
#         - A complex case (higher weighting) increases load.
#         - A part-time FTE increases load (same case is a bigger share of their time).

# - If you don’t provide the start and end dates, the function works them out automatically using your project rules:
#     - Start horizon comes from the earliest dt_received_inv;
#     - End horizon comes from the latest dt_pg_signoff, plus a padding window.

# - It uses your official milestones (dt_alloc_invest, dt_close, dt_pg_signoff) to decide when a case is actively being worked.

# - Fast & scalable: It uses a delta method (add +1 at the start date, −1 after the end date) so it can efficiently build daily WIP counts even for thousands of cases.

# - It gives both a count (wip) and a load (wip_load = weighting ÷ FTE) so you can see not just how many cases someone has, but how heavy that workload likely is.

# - If some dates are missing, it falls back sensibly (e.g., if a case never closes, it stays WIP until the end horizon).


# ## A tiny mental model
# - Think of each case as a bar on a timeline (from allocation to close/signoff).
# - We lay all bars for a person on top of each other.
# - For any given day, how many bars overlap? That’s wip.
# - If some bars are “heavier” (higher weighting) or the staff member has lower FTE, the overlap total becomes wip_load.

# ## Common edge cases handled
# - Open cases with no close/signoff → they count as WIP until the report end date.
# - Missing weighting/FTE → sensible defaults keep the math stable.
# - No cases for a person → they simply won’t appear in the output (or will have zeros after merge/accumulation).

# ## Tiny visual example (intuition)
# If a case runs from Jan 2 to Jan 5:
# - We add +1 on Jan 2.
# - We add −1 on Jan 6 (the day after it finishes).
# - Cumulative sum across days produces:
# Jan 1: 0
# Jan 2: 1
# Jan 3: 1
# Jan 4: 1
# Jan 5: 1
# Jan 6: 0
# Now imagine multiple cases overlapping—WIP is just the sum of overlaps each day.

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


# - Build a daily time series showing the size of the allocation backlog:
# - Calculate ow many cases have been received into Investigations but not yet allocated to an investigator.

# - It builds a timeline of the allocation backlog — how many cases have arrived in Investigations but haven’t yet been allocated to an investigator — day by day (or week by week).

# - It counts Received (cases entering the queue) and Allocated (cases leaving to a person) per day.
# - It then takes a running total (cumulative) of each and computes:
#     - Backlog = Total Received so far − Total Allocated so far.
# - optionally:
#     - Exclude weekends/holidays to focus on working days only.
#     - Resample weekly or monthly, keeping the last cumulative value per period (the correct way for running totals).
#     - Compute a weighted backlog (if some cases are heavier/more complex) using a weighting column.

# - Received means dt_received_inv (case enters the Investigations queue).
# - Allocated means dt_alloc_invest (case leaves the queue and goes to a person).
# - Backlog available (on any day) = total received so far − total allocated so far.
# - If we don’t provide a reporting window, the function figures it out using your rules:
#     - Start from the earliest dt_received_inv.
#     - End at the latest dt_pg_signoff, with a padding window added.

# - There’s also an optional weighted backlog, which treats some cases as “heavier” based on weighting (e.g., complexity).
# - Matches our operational definition of backlog (waiting to be allocated to an investigator).
# - Operationally accurate: matches the definition of backlog (awaiting allocation).
# - Transparent: shows both cumulative inputs (received/allocated) and the resulting backlog; we publish cumulative received and allocated alongside the backlog so you can audit the numbers.
# - Robust: it works even if some days have no activity; it also can clip the backlog at zero to avoid confusing negatives; prevents negative backlog and handles days with no activity cleanly.
# - Flexible & practical: business-day filtering and weekly/monthly views match how teams actually review performance; it can compute a weighted version if you want a complexity-aware measure.


# - Builds the daily picture of staff activity and backlog pressure across the investigation process.
# - Each row in the output shows, for each investigator on each date:
#     - how many cases they were working on (wip)
#     - how heavy that workload was (wip_load)
#     - what events happened that day (e.g., new case, legal step, PG sign-off)
#     - how long since they last picked up a case
#     - whether they are new in post (less than 4 weeks)
#     - what the system backlog looked like that day
#     - day-of-week, term, season, and bank holiday context
# - The result feeds directly into forecasting models, dashboards, or simulation inputs.
# - Combines everything: merges workload, case flow, and events into a single daily dataset.
# - Flexible: supports working-day calendars, holiday exclusions, and weekly backlog summaries.
# - Transparent: every part comes from separate, auditable builder functions, nothing hidden.
# - Scalable: runs efficiently even for many staff over long periods.

# - Step-by-step logic
#     1. Determine the date range (start/end) using date_horizon().
#     2.  Build core inputs:
#         - events = timeline of case milestones.
#         - wip = ongoing cases per staff/day.
#         - backlog = unallocated cases per day (received − allocated).
#     3. Create a daily grid for all staff and all working dates.
#     4. Merge in WIP and events, turning event names into flag columns (0/1).
#     5. Compute features:
#         - time since last new case pickup
#         - week, season, term, holiday, and new starter status
#     6. Join backlog context to every day’s record.
#     7. Return three consistent datasets for downstream modelling.


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

    # 4) Build staff-date grid (all combinations)
    staff = typed[["staff_id", "team"]].drop_duplicates()
    grid = (
        staff.assign(_k=1)
        .merge(pd.DataFrame({"date": date_index}).assign(_k=1), on="_k", how="outer")
        .drop(columns=["_k"])
    )

    dates = pd.DataFrame({"date": date_index, "_k": 1})
    grid = staff.merge(dates, how="cross")

    # grid = (
    #     pd.DataFrame({"staff_id": staff_id, "_k": 1})
    #     .merge(dates, on="_k")
    #     .drop(columns="_k")
    # )

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

    tmp = (
        grid.groupby("staff_id", group_keys=False)["event_newcase"]
        .apply(_days_since_last_pickup)
        .fillna("__MISSING__")  # or "MISSING" #.fillna(99) #
        # .astype("Int64")
    )
    grid["time_since_last_pickup"] = tmp.infer_objects(copy=False)

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


# # Summerise
# - Rolls up the detailed daily staff panel into team-level (or any custom grouping) time series, to quickly see trends like “total WIP per team per day/week” or “how many new cases did Team A pick up last month?”.

# - Practical: real reporting/forecasting often needs team- or org-level time series, not just staff-level detail.
# - Correct aggregation: it sums “flow” metrics (e.g., events, WIP cases) and treats stateful metrics (like backlog levels) correctly when resampling by taking the last value per period (the right way to downsample cumulative/state variables).
# - Flexible: you pick the grouping keys, the resampling frequency, and can override the aggregation rules if needed.

# - How it works (step-by-step)
#     1. Choose the grouping
#        By default it groups by date and team. You can change by to include role, or collapse to just date for an overall total.
#     2. Aggregate daily
#        It sums WIP and WIP load across staff, sums events, takes the median time since last pickup (typical day for staff), and counts distinct staff on duty.
#     3. (Optional) Resample to weekly/monthly
#        If you pass freq='W-FRI' (weekly Fridays) or 'MS' (month-start), it:
#        - Sums the “flow” fields within each period (e.g., total new cases in that week/month).
#        - Takes the last value for stateful/level fields (e.g., backlog_available) so the weekly/monthly series reflects the end-of-period level.
#     4. Return a tidy frame
#        With columns like: wip_sum, wip_load_sum, event_*_sum, backlog_available_mean (daily means) and, when resampled, last values for backlog-like metrics (you can change the list via resample_cum_last).


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


# # 1) Team-level daily
# team_daily = summarise_daily_panel(daily, by=['date','team'])

# # 2) Team-level weekly (Friday), treating backlog as a level (last-of-week)
# team_weekly = summarise_daily_panel(
#     daily,
#     by=['date','team'],
#     freq='W-FRI',
#     resample_cum_last=('backlog_available',)  # keep as 'last' per week
# )

# # 3) Overall totals per day (collapse teams)
# org_daily = summarise_daily_panel(daily, by=['date'])

# # 4) Custom aggregation rules (e.g., use max backlog across staff instead of mean)
# custom = summarise_daily_panel(
#     daily,
#     by=['date','team'],
#     agg_map={'backlog_available': 'max'}
# )
# # Save events DataFrame to CSV
# custom.to_csv(OUT_DIR / "Custom_Summary.csv", index=False)
# print(custom)
