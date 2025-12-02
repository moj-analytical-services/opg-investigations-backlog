from pathlib import Path
from data import (
    generate_synthetic,
    basic_clean,
    engineer_intervals,
    daily_backlog_series,
)


def test_generate_and_engineer(tmp_path: Path):
    df = generate_synthetic(500, seed=1)
    assert {"date_received_opg", "date_allocated_investigator"}.issubset(df.columns)
    df = engineer_intervals(basic_clean(df))
    assert "days_to_alloc" in df.columns
    daily = daily_backlog_series(df)
    assert set(["date", "backlog"]).issubset(daily.columns)
    assert len(daily) > 0
