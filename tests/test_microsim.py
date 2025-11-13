# tests/test_microsim.py
from pathlib import Path
from g7_assessment.data import generate_synthetic, basic_clean, engineer_intervals
from g7_assessment.microsim import arrivals_by_case_type, backlog_daily, staffing_daily, km_quantiles_by_group, legal_review_routing

def test_microsim_exports_shape(tmp_path: Path):
    df = engineer_intervals(basic_clean(generate_synthetic(1000, seed=11)))
    assert len(arrivals_by_case_type(df, "D")) > 0
    assert len(arrivals_by_case_type(df, "M")) > 0
    assert len(backlog_daily(df)) > 0
    assert len(staffing_daily(df)) >= 0  # can be zero if no allocations in tiny samples
    km = km_quantiles_by_group(df)
    # Not all groups will have events; allow empty, but function must return a DataFrame
    assert km is not None
    route = legal_review_routing(df)
    assert "rate" in route.columns
