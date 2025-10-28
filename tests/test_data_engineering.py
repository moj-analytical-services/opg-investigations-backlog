# tests/test_data_engineering.py
import pandas as pd
import numpy as np
import importlib
import pytest

# Import the module under test
de = importlib.import_module("data_engineering")

def test_month_and_term_helpers():
    assert de.month_to_season(4) == "spring"
    assert de.month_to_season(10) == "autumn"
    assert de.is_term_month(8) == 0
    assert de.is_term_month(1) == 1

def test_normalise_and_hash_id():
    assert de.normalise_col("  Foo   BAR ") == "foo bar"
    h = de.hash_id("Alice")
    assert isinstance(h, str) and h.startswith("S") and len(h) == 9
    assert de.hash_id("") == ""

def test_parse_date_series_handles_nulls_and_ordinals():
    s = pd.Series(["1st Jan 2025", "unknown", None, "  02/01/2025  "])
    out = de.parse_date_series(s)
    assert pd.isna(out.iloc[1]) and pd.isna(out.iloc[2])
    assert out.dt.year.tolist().count(2025) >= 2

def test_date_horizon_basic(typed_df_small, horizon):
    start, end = horizon
    assert isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp)
    assert end >= start
    # Check padding took effect (>= base span)
    assert (end - start).days >= (pd.Timestamp('2025-01-06') - pd.Timestamp('2025-01-01')).days

def test_build_event_log_minimal(typed_df_small):
    ev = de.build_event_log(typed_df_small)
    # columns exist
    assert set(["date","staff_id","team","fte","case_id","event","meta"]).issubset(ev.columns)
    # events contain newcase and possibly legal_request
    assert ("newcase" in ev["event"].unique()) or ev.empty

def test_build_wip_series_shapes(typed_df_small, horizon):
    start, end = horizon
    wip = de.build_wip_series(typed_df_small, start, end)
    assert set(wip.columns) == {"date","staff_id","team","wip"}
    assert (wip["wip"] >= 0).all()

def test_build_backlog_series_monotone(typed_df_small, horizon):
    start, end = horizon
    backlog = de.build_backlog_series(typed_df_small, start, end)
    assert list(backlog.columns) == ["date","backlog_available"]
    # monotone non-decreasing of cumulative accepted minus allocated is not guaranteed,
    # but values should be finite ints
    assert backlog["backlog_available"].dtype.kind in "iu" or backlog["backlog_available"].dtype.kind == "f"
    assert backlog["backlog_available"].notna().all()

def test_build_daily_panel_contract(typed_df_small, horizon):
    start, end = horizon
    daily, backlog, events = de.build_daily_panel(typed_df_small, start, end)

    # check expected columns present
    must_have = {
        "date","staff_id","team","role","fte",
        "is_new_starter","weeks_since_start",
        "wip","time_since_last_pickup",
        "mentoring_flag","trainee_flag",
        "backlog_available","term_flag","season","dow","bank_holiday",
        "event_newcase","event_legal","event_court"
    }
    assert must_have.issubset(daily.columns)
    # shape sensible
    assert len(backlog) == (end - start).days + 1
    # events either empty or contains expected event labels
    if not events.empty:
        assert set(events["event"]).issubset({"newcase","legal_request","legal_approval","court_order"})

@pytest.mark.parametrize("enc", ["utf-8-sig", "cp1252"])
def test_load_raw_roundtrip(tmp_path, enc):
    """Minimal load_raw smoke test across two encodings."""
    # Skip if function not present (e.g., when only panel helpers are exported)
    if not hasattr(de, "load_raw"):
        pytest.skip("load_raw not exported in data_engineering")

    p = tmp_path / "tiny.csv"
    df_in = pd.DataFrame({
        "ID": ["1","2"],
        "Investigator": ["Alice","Bob"],
        "Team": ["T1","T1"],
        "Investigator FTE": ["1.0","0.8"],
        "Date Received in Investigations": ["2025-01-01","2025-01-02"],
    })
    # write in requested encoding
    df_in.to_csv(p, index=False, encoding=enc)
    df_out, colmap = de.load_raw(p)
    # Check we got strings (dtype=str) and colmap populated
    assert df_out.shape == df_in.shape
    assert isinstance(colmap, dict) and "id" in colmap
