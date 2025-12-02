# run a preprocessing + time-series demo

from pathlib import Path

from preprocessing import load_raw, engineer
from time_series import (
    build_event_log,
    build_wip_series,
    build_backlog_series,
    build_daily_panel,
    summarise_daily_panel,
)

# 1) Load raw data and engineer typed table
raw_path = Path("data/raw/raw.csv")  # adjust if needed
raw, colmap = load_raw(raw_path)
typed = engineer(raw, colmap)

# 2) Build core time-series artefacts
events = build_event_log(typed)
wip = build_wip_series(typed)
backlog = build_backlog_series(typed)

daily, backlog_ts, events_ts = build_daily_panel(
    typed,
    start=None,
    end=None,
    exclude_weekends=True,
    holidays=None,
    pad_days=14,
    backlog_freq="W-FRI",
)

# 3) Aggregate to team-level daily and weekly
team_daily = summarise_daily_panel(daily, by=["date", "team"])
team_weekly = summarise_daily_panel(daily, by=["date", "team"], freq="W-FRI")

print(team_daily.head())
print(team_weekly.head())

