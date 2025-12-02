# Unit tests using pytest (tests/test_data_quality.py)
# ------------------------------------
import pytest
import pandas as pd
from datetime import datetime
from .src.data_quality.data_quality import DataQualityChecks


@pytest.fixture
def sample_df(tmp_path):
    # create a small sample CSV for testing
    data = {
        "case_no": ["A1", None, "A3"],
        "unique_id": ["u1", "u2", "u3"],
        "registrationdate": ["2020-01-01", "2020-05-05", "notadate"],
        "date_received_in_opg": ["2020-02-01", "2020-05-01", "2021-01-01"],
        "casesubtype": ["pfa", "pfa", "xyz"],
        "concern_type": ["Financial", "Both", "Health and Welfare"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_validate_dates(sample_df):
    dq = DataQualityChecks(sample_df)
    df2 = dq.validate_dates(["registrationdate", "date_received_in_opg"])
    assert df2["registrationdate"].isna().sum() == 1  # 'notadate'
    assert df2["date_received_in_opg"].isna().sum() == 0


def test_compute_delay(sample_df):
    dq = DataQualityChecks(sample_df)
    dq.validate_dates(["registrationdate", "date_received_in_opg"])
    df2 = dq.compute_delay("registrationdate", "date_received_in_opg")
    # one delay negative or NA should be dropped
    assert all(df2["delay_days"] >= 0)


def test_impute_delays(sample_df):
    dq = DataQualityChecks(sample_df)
    dq.validate_dates(["registrationdate", "date_received_in_opg"])
    dq.compute_delay("registrationdate", "date_received_in_opg")
    df3 = dq.impute_delays()
    # no NaNs remain
    assert df3["delay_days"].isna().sum() == 0


def test_remove_duplicates(sample_df):
    dq = DataQualityChecks(sample_df)
    dq.validate_dates(["registrationdate", "date_received_in_opg"])
    dq.compute_delay("registrationdate", "date_received_in_opg")
    dq.derive_keys(("case_no", "unique_id"))
    # duplicate creation
    df_dup = dq.df.append(dq.df.iloc[0])
    dq.df = df_dup
    df4 = dq.remove_duplicates()
    assert df4.shape[0] < df_dup.shape[0]


def test_parse_month_valid():
    # Unquoted, quoted, extra whitespace
    assert DataQualityChecks.parse_month("2020-01") == datetime(2020, 1, 1)
    assert DataQualityChecks.parse_month("'2021-12'") == datetime(2021, 12, 1)
    assert DataQualityChecks.parse_month(" 2022-05 ") == datetime(2022, 5, 1)


def test_parse_month_invalid():
    # Wrong format should raise ValueError
    with pytest.raises(ValueError):
        DataQualityChecks.parse_month("2020/01")
    with pytest.raises(ValueError):
        DataQualityChecks.parse_month("Jan-2020")


def test_generate_month_list_valid():
    months = DataQualityChecks.generate_month_list("2020-01", "2020-03")
    expected = [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)]
    assert months == expected


def test_generate_month_list_invalid_range():
    # start after end
    with pytest.raises(ValueError):
        DataQualityChecks.generate_month_list("2020-04", "2020-02")


def test_last_day_of_month():
    # Feb non-leap, Feb leap, Apr
    assert DataQualityChecks.last_day_of_month(datetime(2021, 2, 15)) == "2021-02-28"
    assert DataQualityChecks.last_day_of_month(datetime(2020, 2, 10)) == "2020-02-29"
    assert DataQualityChecks.last_day_of_month(datetime(2021, 4, 5)) == "2021-04-30"


# Additional cross-column and consistency tests


def test_flag_invalid_delays(sample_df):
    # Should flag rows where registrationdate is NaT or after receipt
    dq = DataQualityChecks(sample_df)
    dq.validate_dates(["registrationdate", "date_received_in_opg"])
    mask = dq.flag_invalid_delays("registrationdate", "date_received_in_opg")
    # One row has invalid registrationdate parsing → True
    assert mask.sum() == 1
    # Ensure flagged row corresponds to sample index 2
    assert mask.iloc[2]


def test_derive_keys_unique(sample_df):
    # Ensure derived_id column is created and unique
    dq = DataQualityChecks(sample_df)
    dq.validate_dates(["registrationdate", "date_received_in_opg"])
    df2 = dq.derive_keys(("case_no", "unique_id"))
    assert "derived_id" in df2.columns
    # No duplicates in derived_id
    assert df2["derived_id"].is_unique


def test_remove_duplicates_keeps_first(sample_df):
    # Simulate duplicates
    dq = DataQualityChecks(sample_df)
    dq.validate_dates(["registrationdate", "date_received_in_opg"])
    dq.compute_delay("registrationdate", "date_received_in_opg")
    dq.derive_keys(("case_no", "unique_id"))
    df_dup = pd.concat([dq.df, dq.df.iloc[[0]]], ignore_index=True)
    dq.df = df_dup
    before = dq.df.shape[0]
    after = dq.remove_duplicates().shape[0]
    assert after == before - 1


def test_summary_statistics_contains_expected():
    dq = DataQualityChecks(sample_df)
    desc = dq.summary_statistics()
    # summary should include delay_days if computed later, but at least include registrationdate col
    assert "registrationdate" in desc.columns
    assert "date_received_in_opg" in desc.columns


def test_registration_before_receipt_consistency(sample_df):
    # After validation, registrationdate should be <= receipt date or NaT
    dq = DataQualityChecks(sample_df)
    df2 = dq.validate_dates(["registrationdate", "date_received_in_opg"])
    # All non-null registrations are <= receipt
    valid = df2[df2["registrationdate"].notna()]
    assert (valid["registrationdate"] <= valid["date_received_in_opg"]).all()
