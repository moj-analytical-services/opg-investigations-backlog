# tests/conftest.py
import pandas as pd
import pytest


@pytest.fixture
def typed_df_small():
    """Tiny, coherent 'typed' dataframe for smoke tests."""
    return pd.DataFrame(
        {
            "case_id": ["C1", "C2"],
            "investigator": ["Alice", "Bob"],
            "team": ["T1", "T1"],
            "role": ["", ""],
            "fte": [1.0, 0.8],
            "staff_id": ["S1", "S2"],
            # key dates
            "dt_received_inv": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
            "dt_alloc_invest": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")],
            "dt_alloc_team": [pd.NaT, pd.NaT],
            "dt_pg_signoff": [pd.NaT, pd.Timestamp("2025-01-08")],
            "dt_close": [pd.Timestamp("2025-01-06"), pd.NaT],
            # events
            "dt_legal_req_1": [pd.NaT, pd.Timestamp("2025-01-04")],
            "dt_legal_req_2": [pd.NaT, pd.NaT],
            "dt_legal_req_3": [pd.NaT, pd.NaT],
            "dt_legal_approval": [pd.NaT, pd.NaT],
            "dt_date_of_order": [pd.NaT, pd.NaT],
            "dt_flagged": [pd.NaT, pd.NaT],
        }
    )


@pytest.fixture
def horizon(typed_df_small):
    """Compute a short padded horizon for the tiny dataset."""
    from data_engineering import date_horizon

    start, end = date_horizon(typed_df_small, pad_days=3)
    return start, end
