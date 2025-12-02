from .preprocessing import load_raw, engineer
from .time_series import build_event_log, build_wip_series, build_backlog_series, build_daily_panel
from .interval_analysis import IntervalAnalysis
from .eda_opg import EDAConfig, OPGInvestigationEDA


# ------------------------------------------------------------
# High-level orchestrator class combining pipeline + EDA tools
# ------------------------------------------------------------


class InvestigationBacklogProject:
    """
    High-level helper that wires together the core engineering
    pipeline from ``Build_Investigator_Daily_from_Raw_14_11_25``
    and the EDA toolkit from ``eda.ipynb``.

    Typical usage
    -------------
    >>> from pathlib import Path
    >>> project = InvestigationBacklogProject()
    >>> outputs = project.run_full_pipeline()
    >>> daily = outputs["daily"]
    >>> daily.head()
    """

    def __init__(self, raw_path: "Path" = None, out_dir: "Path" = None):
        """
        Parameters
        ----------
        raw_path:
            Path to the CSV extract with investigations data. If not
            provided, uses the global RAW_PATH defined in the
            engineering module.
        out_dir:
            Directory where derived CSVs and plots will be written.
            If not provided, uses the global OUT_DIR.

        Example
        -------
        >>> from pathlib import Path
        >>> project = InvestigationBacklogProject(
        ...     raw_path=Path("data/raw/raw.csv"),
        ...     out_dir=Path("data/out"),
        ... )
        """
        from pathlib import Path as _Path

        # Fall back to module-level defaults if not supplied
        global RAW_PATH, OUT_DIR
        self.raw_path = _Path(raw_path) if raw_path is not None else RAW_PATH
        self.out_dir = _Path(out_dir) if out_dir is not None else OUT_DIR

    # -------------------------
    # 1) Core engineering steps
    # -------------------------

    def load_raw(self):
        """
        Load the raw investigations extract via :func:`load_raw`.

        Returns
        -------
        raw : pandas.DataFrame
        colmap : dict

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> raw, colmap = project.load_raw()
        >>> raw.shape  # doctest: +SKIP
        """
        return load_raw(self.raw_path)

    def engineer(self, raw=None, colmap=None, only_reallocated: bool = False):
        """
        Run the :func:`engineer` step to create a clean, typed case-level table.

        Parameters
        ----------
        raw, colmap:
            Optionally pass in the raw DataFrame and column map. If omitted
            they are loaded from ``self.raw_path`` using :func:`load_raw`.
        only_reallocated:
            See the underlying :func:`engineer` docstring.

        Returns
        -------
        typed : pandas.DataFrame

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> typed = project.engineer()
        >>> typed.columns.tolist()[:5]  # doctest: +SKIP
        """
        if raw is None or colmap is None:
            raw, colmap = self.load_raw()
        return engineer(raw, colmap, only_reallocated=only_reallocated)

    def build_event_log(self, typed=None):
        """
        Build an investigator-centric event log from the engineered table.

        Parameters
        ----------
        typed:
            Engineered DataFrame. If omitted it is computed via :meth:`engineer`.

        Returns
        -------
        events : pandas.DataFrame

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> typed = project.engineer()
        >>> events = project.build_event_log(typed)
        >>> events.head()  # doctest: +SKIP
        """
        if typed is None:
            typed = self.engineer()
        return build_event_log(typed)

    def build_wip_series(self, events=None):
        """
        Build the Work-In-Progress (WIP) daily time series.

        Parameters
        ----------
        events:
            Event log from :meth:`build_event_log`. If omitted it is built on the fly.

        Returns
        -------
        wip : pandas.DataFrame

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> wip = project.build_wip_series()
        >>> wip.head()  # doctest: +SKIP
        """
        if events is None:
            events = self.build_event_log()
        return build_wip_series(events)

    def build_backlog_series(self, events=None):
        """
        Build the backlog daily time series.

        Parameters
        ----------
        events:
            Event log from :meth:`build_event_log`. If omitted it is built on the fly.

        Returns
        -------
        backlog : pandas.DataFrame

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> backlog = project.build_backlog_series()
        >>> backlog.head()  # doctest: +SKIP
        """
        if events is None:
            events = self.build_event_log()
        return build_backlog_series(events)

    def build_daily_panel(self, typed=None, wip=None, backlog=None):
        """
        Build a daily, investigator-level panel suitable for modelling.

        Parameters
        ----------
        typed, wip, backlog:
            Optional intermediate tables. If omitted they are all recomputed.

        Returns
        -------
        daily : pandas.DataFrame

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> daily = project.build_daily_panel()
        >>> daily.head()  # doctest: +SKIP
        """
        if typed is None:
            typed = self.engineer()
        if wip is None:
            wip = self.build_wip_series()
        if backlog is None:
            backlog = self.build_backlog_series()
        return build_daily_panel(typed, wip, backlog)

    def run_full_pipeline(self):
        """
        Convenience method: run the entire core pipeline in one call.

        Returns
        -------
        outputs : dict
            A dictionary with keys:
            ``raw``, ``colmap``, ``typed``, ``events``,
            ``wip``, ``backlog``, ``daily``.

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> outputs = project.run_full_pipeline()
        >>> sorted(outputs.keys())  # doctest: +SKIP
        """
        raw, colmap = self.load_raw()
        typed = self.engineer(raw, colmap)
        events = self.build_event_log(typed)
        wip = self.build_wip_series(events)
        backlog = self.build_backlog_series(events)
        daily = self.build_daily_panel(typed, wip, backlog)
        return {
            "raw": raw,
            "colmap": colmap,
            "typed": typed,
            "events": events,
            "wip": wip,
            "backlog": backlog,
            "daily": daily,
        }

    # -------------------------
    # 2) Interval analysis and EDA
    # -------------------------

    def build_interval_frame(self, typed=None, backlog_series=None, bank_holidays=None):
        """
        Build the interval-analysis-ready frame using :class:`IntervalAnalysis`.

        Parameters
        ----------
        typed:
            Engineered table (output of :meth:`engineer`).
        backlog_series:
            Backlog time series (output of :meth:`build_backlog_series`).
        bank_holidays:
            Optional list/Series of bank-holiday dates.

        Returns
        -------
        di : pandas.DataFrame

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> outputs = project.run_full_pipeline()
        >>> di = project.build_interval_frame(outputs["typed"], outputs["backlog"])
        >>> di.head()  # doctest: +SKIP
        """
        if typed is None or backlog_series is None:
            outputs = self.run_full_pipeline()
            typed = outputs["typed"]
            backlog_series = outputs["backlog"]
        return IntervalAnalysis.build_interval_frame(
            typed, backlog_series=backlog_series, bank_holidays=bank_holidays
        )

    def monthly_trend(self, di, metric="days_to_pg_signoff", agg="median", by=None):
        """
        Thin wrapper around :meth:`IntervalAnalysis.monthly_trend`.

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> outputs = project.run_full_pipeline()
        >>> di = project.build_interval_frame(outputs["typed"], outputs["backlog"])
        >>> trend = project.monthly_trend(di, metric="days_to_pg_signoff")
        >>> trend.head()  # doctest: +SKIP
        """
        return IntervalAnalysis.monthly_trend(di, metric=metric, agg=agg, by=by)

    def make_eda(self, df=None, config=None):
        """
        Construct an :class:`OPGInvestigationEDA` instance.

        Parameters
        ----------
        df:
            DataFrame to analyse. If omitted uses the engineered case-level table.
        config:
            :class:`EDAConfig` defining the column mapping. If omitted, a
            minimal sensible default is inferred.

        Returns
        -------
        eda : OPGInvestigationEDA

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> outputs = project.run_full_pipeline()
        >>> eda = project.make_eda(outputs["typed"])
        >>> overview = eda.quick_overview()
        """

        if df is None:
            outputs = self.run_full_pipeline()
            df = outputs["typed"]
        if config is None:
            # Fallback minimal config assuming the engineered column names

            config = EDAConfig(
                id_col="case_id",
                date_received="dt_received_opg",
                date_allocated="dt_alloc_invest",
                date_signed_off="dt_pg_signoff",
            )
        return OPGInvestigationEDA(df, config)

    # -------------------------
    # 3) Collective demo
    # -------------------------

    def demo_all(self):
        """
        Run an end-to-end demo:

        1. Run the full engineering pipeline.
        2. Build the interval frame and monthly trend KPIs.
        3. Construct an EDA object and compute a quick overview.
        4. Generate and save monthly PG sign-off trend plots.

        This mirrors the demonstration cells from the original notebooks
        but in a single callable method.

        Example
        -------
        >>> project = InvestigationBacklogProject()
        >>> demo_outputs = project.demo_all()  # doctest: +SKIP
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # 1) Core pipeline
        outputs = self.run_full_pipeline()
        typed = outputs["typed"]
        backlog = outputs["backlog"]

        # 2) Interval frame + monthly trend
        di = self.build_interval_frame(typed, backlog)
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

        # 3) Simple EDA overview
        eda = self.make_eda(typed)
        overview = eda.quick_overview()

        # 4) Plots (saved into self.out_dir)
        outdir = self.out_dir
        outdir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(16, 9))
        for col in piv.columns:
            plt.plot(piv.index, piv[col], alpha=0.6, label=str(col))
        plt.title("Monthly median days_to_pg_signoff by case_type")
        plt.xlabel("Month")
        plt.ylabel("Median days to PG sign-off")
        plt.xticks(rotation=45)
        plt.legend(ncol=2, fontsize=8)
        plot1 = outdir / "monthly_median_days_to_pg_signoff_by_case_type.png"
        plt.savefig(plot1, bbox_inches="tight", dpi=150)
        plt.close()

        plt.figure(figsize=(16, 9))
        for col in piv_delta.columns:
            plt.plot(piv_delta.index, piv_delta[col], alpha=0.6, label=str(col))
        plt.title("Monthly MoM delta: days_to_pg_signoff by case_type")
        plt.xlabel("Month")
        plt.ylabel("MoM delta (days)")
        plt.xticks(rotation=45)
        plt.legend(ncol=2, fontsize=8)
        plot2 = outdir / "monthly_mom_delta_days_to_pg_signoff_by_case_type.png"
        plt.savefig(plot2, bbox_inches="tight", dpi=150)
        plt.close()

        print("Demo complete.")
        print("Key outputs:")
        print("  - Engineered table shape:", typed.shape)
        print("  - Daily panel shape:", outputs["daily"].shape)
        print("  - Interval frame shape:", di.shape)
        print("  - Overview columns:", list(overview.columns))
        print("  - Plots saved to:", plot1, "and", plot2)

        return {
            **outputs,
            "interval_frame": di,
            "trend": trend,
            "overview": overview,
            "plots": {"median_trend": plot1, "delta_trend": plot2},
        }