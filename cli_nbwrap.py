# src/g7_assessment/cli_nbwrap.py
# Wraps your existing notebook functions into small, task-focused CLI commands
# without changing the underlying logic.

from __future__ import annotations
from pathlib import Path
import click
import pandas as pd

# Import ONLY from the wrappers that re-export your original notebook logic
from .preprocessing import load_raw, engineer
from .intervals import (
    build_event_log,
    build_backlog_series,
    build_daily_panel,
    summarise_daily_panel,
)
from .analysis_demo import last_year_by_team
from .distributions import interval_change_distribution


@click.group()
def cli():
    """Task runner for OPG investigations (uses your original notebook logic)."""
    pass


@cli.command("prep")
@click.option(
    "--raw",
    "raw_csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the raw investigations CSV.",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False),
    default="data/processed",
    help="Where engineered outputs will be written.",
)
def prep(raw_csv: str, outdir: str):
    """
    DATA PRE-PROCESSING / DATA MANIPULATION / MISSING DATA IMPUTATION
    ---------------------------------------------------------------
    Uses your notebook's load_raw + engineer functions and writes engineered.csv
    (no code changes, just calling your functions).
    """
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load raw + column map (exactly as your notebook’s function expects)
    raw, colmap = load_raw(Path(raw_csv))

    # 2) Engineer typed/clean dataset (your missing-data handling remains intact)
    typed = engineer(raw, colmap)

    # 3) Persist engineered dataset for downstream tasks
    out_file = out_dir / "engineered.csv"
    typed.to_csv(out_file, index=False)
    click.echo(f"Engineered dataset written: {out_file}")


@cli.command("intervals")
@click.option(
    "--eng",
    "eng_csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to engineered.csv produced by `prep`.",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False),
    default="data/processed",
    help="Where event_log.csv / backlog_series.csv / daily_panel.csv will be written.",
)
def intervals_cmd(eng_csv: str, outdir: str):
    """
    INTERVAL ANALYSIS + PANELS
    --------------------------
    Builds event log, backlog/WIP series, daily panel and a summary table.
    """
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(eng_csv, parse_dates=True)

    # 1) Event log (your existing notebook function)
    events = build_event_log(df)
    events_out = out_dir / "event_log.csv"
    events.to_csv(events_out, index=False)

    # 2) Backlog/WIP/Daily panel (your existing notebook functions)
    backlog = build_backlog_series(df)
    backlog_out = out_dir / "backlog_series.csv"
    backlog.to_csv(backlog_out, index=False)
    daily = build_daily_panel(df)
    daily_out = out_dir / "daily_panel.csv"
    daily.to_csv(daily_out, index=False)

    # 3) Summary table (unchanged logic)
    summary = summarise_daily_panel(df)
    summary_out = out_dir / "daily_panel_summary.csv"
    summary.to_csv(summary_out, index=False)

    click.echo(f"Wrote: {events_out}, {backlog_out}, {daily_out}, {summary_out}")


@cli.command("trend-demo")
@click.option(
    "--eng",
    "eng_csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to engineered.csv.",
)
@click.option(
    "--backlog",
    "backlog_csv",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Optional: backlog_series.csv if your IntervalAnalysis uses it.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="reports/last_year_by_team.csv",
    help="Where to write the trend CSV.",
)
def trend_demo(eng_csv: str, backlog_csv: str | None, out: str):
    """
    DEMO: Last-year interval analysis by team (non-invasive)
    -------------------------------------------------------
    Calls your IntervalAnalysis monthly_trend_last_year(...) via the wrapper.
    """
    eng = pd.read_csv(eng_csv, parse_dates=True)
    backlog = pd.read_csv(backlog_csv) if backlog_csv else None
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trend = last_year_by_team(eng_df=eng, backlog_series=backlog, bank_holidays=None)
    trend.to_csv(out_path, index=False)
    click.echo(f"Trend written: {out_path}")


@cli.command("interval-distribution")
@click.option(
    "--eng",
    "eng_csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to engineered.csv.",
)
@click.option(
    "--interval-col",
    default="days_to_alloc",
    help="Interval column to analyse (e.g., 'days_to_alloc' or 'days_to_pg_signoff').",
)
@click.option(
    "--group", default="case_type", help="Grouping column (default: case_type)."
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False),
    default="reports",
    help="Where annual_stats.csv and yoy_change.csv will be written.",
)
def interval_distribution(eng_csv: str, interval_col: str, group: str, outdir: str):
    """
    DISTRIBUTION OF INTERVAL CHANGES OVER YEARS (by case_type or others)
    --------------------------------------------------------------------
    Summarises per-year distributions and year-on-year changes.
    """
    eng = pd.read_csv(eng_csv, parse_dates=True)
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = interval_change_distribution(eng, interval_col=interval_col, group=group)
    (out_dir / "annual_stats.csv").write_text(res["annual_stats"].to_csv(index=False))
    (out_dir / "yoy_change.csv").write_text(res["yoy_change"].to_csv(index=False))

    click.echo(f"Wrote: {out_dir/'annual_stats.csv'}, {out_dir/'yoy_change.csv'}")


if __name__ == "__main__":
    cli()
