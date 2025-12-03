# --- Interval Analysis: ---
# - Goal: build a tidy table of case activity (with a backlog count) and give simple tools to measure how key time intervals have changed over the last year, overall and by team/staff.
# - Why we added it: leaders and investigators want to see how fast things move (e.g., “days to PG sign-off”), whether that’s improving, and where volatility lives.
# - Input: engineered cases table, plus optional daily backlog_series (date + backlog).
# - Output: A dataframe with the selected columns (e.g., date, staff_id, team, days_to_pg_signoff, time_since_last_pickup, numeric backlog, seasonal flags, event flags), plus helper functions for last-year summaries, trends, and volatility.

# - Introduces a small utility class and helper functions to (a) construct a case-level
# dataframe aligned to your requested schema — including a `backlog` numeric column — and
# (b) analyse the distribution of *time-interval changes/fluctuation since last year*.

# **Key capabilities:**
# - Build a dataframe with the columns:
#   `date, staff_id, team, case_id, case_type, concern_type, status, dt_alloc_invest, dt_pg_signoff, dt_received_inv, dt_alloc_team, dt_close, dt_sent_to_ca, days_to_pg_signoff, fte, weighting, wip, wip_load, time_since_last_pickup, weeks_since_start, is_new_starter, backlog_available, term_flag, season, dow, bank_holiday, event_newcase, event_legal, event_court, event_pg_signoff, event_sent_to_ca, event_flagged, backlog`.
# - If you already compute a backlog series with an existing function (e.g. `build_backlog_series`),
#   you can pass it in and it will be merged. If not, a per-date backlog is computed as the
#   number of cases with `dt_received_inv <= date` and (`dt_close` is null or `dt_close > date`).
# - Last-year filters, grouped distribution summaries, month-over-month trend with deltas,
#   and simple volatility scores.

# > Usage examples are provided in comments at the bottom of the new code cell.

from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
import matplotlib.pyplot as plt


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


def plot_pg_signoff_monthly_trends(
    di: pd.DataFrame,
    outdir: Path | str = "data/out/plot/plots",
) -> dict:
    """
    Plot monthly median and month-over-month delta of `days_to_pg_signoff`
    by case_type, plus overall ("ALL case types") trends.
    Saves PNGs and also displays the plots.

    Parameters
    ----------
    di : pd.DataFrame
        Interval frame from IntervalAnalysis.build_interval_frame(...).
    outdir : Path or str, default "data/out/plot/plots"
        Directory where PNG plots will be saved.

    Returns
    -------
    dict
        {"trend": ..., "trend_all": ..., "plots": {...}}
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[plot_pg_signoff_monthly_trends] outdir =", outdir)

    # --- Compute monthly trend by case_type ---
    trend = IntervalAnalysis.monthly_trend(
        di,
        metric="days_to_pg_signoff",
        agg="median",
        by=["case_type"],
    ).copy()
    trend["month"] = pd.to_datetime(trend["yyyymm"] + "-01")

    piv = trend.pivot(
        index="month", columns="case_type", values="days_to_pg_signoff"
    ).sort_index()
    piv_delta = trend.pivot(
        index="month", columns="case_type", values="mom_delta"
    ).sort_index()

    # 1) Monthly median lines
    plt.figure(figsize=(16, 9))
    for col in piv.columns:
        plt.plot(piv.index, piv[col], label=str(col))
    plt.title("Monthly median: days_to_pg_signoff by case_type")
    plt.xlabel("Month")
    plt.ylabel("Days to PG signoff (median)")
    plt.xticks(rotation=45)
    plt.subplots_adjust(right=0.85, top=0.92, bottom=0.15)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
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
    try:
        trend_all = IntervalAnalysis.monthly_trend(
            di, metric="days_to_pg_signoff", agg="median"  # no 'by' -> overall
        ).copy()
        trend_all["month"] = pd.to_datetime(trend_all["yyyymm"] + "-01")
    except Exception:
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

    # 5) Overlay with ALL (median)
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

    # 6) Overlay with ALL (delta)
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

    print("[plot_pg_signoff_monthly_trends] Saved plots to:")
    print(" ", plot1)
    print(" ", plot2)
    print(" ", plot3)
    print(" ", plot4)
    print(" ", plot5)
    print(" ", plot6)

    return {
        "trend": trend,
        "trend_all": trend_all,
        "plots": {
            "by_case_type": plot1,
            "by_case_type_delta": plot2,
            "all": plot3,
            "all_delta": plot4,
            "by_case_type_with_all": plot5,
            "by_case_type_delta_with_all": plot6,
        },
    }
