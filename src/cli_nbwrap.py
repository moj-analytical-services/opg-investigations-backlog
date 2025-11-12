# --- RUN-ALL PIPELINE (prep → intervals → team demo → distributions) ---
@cli.command(name="run-all")
@click.option("--raw", "raw_csv", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to the raw investigations CSV.")
@click.option("--outbase", type=click.Path(file_okay=False), default=".",
              help="Base folder under which outputs will be written (data/processed, reports).")
@click.option("--interval-col", default="days_to_alloc",
              help="Interval to analyse in distributions (e.g., 'days_to_alloc' or 'days_to_pg_signoff').")
@click.option("--group", default="case_type", help="Grouping column for distributions (default: case_type).")
def run_all(raw_csv: str, outbase: str, interval_col: str, group: str):
    """
    Run the full sequence using your original notebook logic, without changing it:
      1) DATA PRE-PROCESSING / MANIPULATION / IMPUTATION (engineered.csv)
      2) INTERVALS: event_log.csv, backlog_series.csv, daily_panel.csv, daily_panel_summary.csv
      3) DEMO: last-year median days_to_pg_signoff by team (reports/last_year_by_team.csv)
      4) DISTRIBUTIONS: annual_stats.csv, yoy_change.csv for the chosen interval/group

    Results are written to:
      - {outbase}/data/processed/
      - {outbase}/reports/
    """
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    outbase = Path(outbase)
    data_dir = outbase / "data" / "processed"
    rep_dir = outbase / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) PREP (uses your notebook functions exactly) ----
    raw, colmap = load_raw(Path(raw_csv))         # from preprocessing.py (re-exports your notebook)
    eng = engineer(raw, colmap)                   # your missing-data & typing logic stays intact
    eng_path = data_dir / "engineered.csv"
    eng.to_csv(eng_path, index=False)

    # ---- 2) INTERVALS (uses your notebook functions exactly) ----
    events = build_event_log(eng);            events_path = data_dir / "event_log.csv";         events.to_csv(events_path, index=False)
    backlog = build_backlog_series(eng);      backlog_path = data_dir / "backlog_series.csv";   backlog.to_csv(backlog_path, index=False)
    daily = build_daily_panel(eng);           daily_path = data_dir / "daily_panel.csv";        daily.to_csv(daily_path, index=False)
    dsum = summarise_daily_panel(eng);        dsum_path = data_dir / "daily_panel_summary.csv"; dsum.to_csv(dsum_path, index=False)

    # ---- 3) DEMO: last-year interval analysis by team (non-invasive) ----
    try:
        trend = last_year_by_team(eng_df=eng, backlog_series=backlog, bank_holidays=None)
        trend_path = rep_dir / "last_year_by_team.csv"
        trend.to_csv(trend_path, index=False)
    except Exception as e:
        trend_path = None
        click.echo(f"[warn] last_year_by_team skipped ({e})")

    # ---- 4) DISTRIBUTIONS: per-case_type over years for your chosen interval ----
    res = interval_change_distribution(eng, interval_col=interval_col, group=group)
    annual_path = rep_dir / "annual_stats.csv"
    yoy_path = rep_dir / "yoy_change.csv"
    res["annual_stats"].to_csv(annual_path, index=False)
    res["yoy_change"].to_csv(yoy_path, index=False)

    # ---- 5) Minimal run summary (Markdown) ----
    summary = rep_dir / "run_all_summary.md"
    start = pd.to_datetime(eng.get("date_received_opg") if "date_received_opg" in eng.columns else eng.get("date")).min()
    end = pd.to_datetime(eng.get("date_received_opg") if "date_received_opg" in eng.columns else eng.get("date")).max()
    summary.write_text(
        f"# Run Summary ({datetime.utcnow().isoformat()}Z)\n\n"
        f"- Raw input: `{raw_csv}`\n"
        f"- Engineered rows: {len(eng):,}\n"
        f"- Date span: {start.date() if pd.notna(start) else 'n/a'} to {end.date() if pd.notna(end) else 'n/a'}\n"
        f"- Outputs:\n"
        f"  - {events_path}\n  - {backlog_path}\n  - {daily_path}\n  - {dsum_path}\n"
        f"  - {trend_path if trend_path else '(trend step skipped)'}\n"
        f"  - {annual_path}\n  - {yoy_path}\n",
        encoding="utf-8"
    )

    click.echo(f"✅ Completed. Outputs in:\n  {data_dir}\n  {rep_dir}")
