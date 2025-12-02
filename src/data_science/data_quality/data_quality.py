# data_quality.py
# ------------------------------------
import pandas as pd
from typing import List, Tuple, Optional

class DataQualityChecks:
    """
    A suite of data quality checks and resolution methods.
    """
    def __init__(self, filepath: str):
        """
        Load the CSV file into a DataFrame.
        :param filepath: Path to the CSV data file
        """
        # Read CSV, allow large fields
        self.df = pd.read_csv(filepath, low_memory=False)

    def summary_statistics(self) -> pd.DataFrame:
        """
        Returns summary statistics and info on missing values.
        """
        # DataFrame info
        info_buf = []
        self.df.info(buf=info_buf)
        info = ''.join(info_buf)
        # Describe all columns
        desc = self.df.describe(include='all')
        return desc

    def validate_dates(self, date_cols: List[str]) -> pd.DataFrame:
        """
        Ensure specified columns are proper dates, coerce invalids to NaT.
        :param date_cols: List of column names to convert
        """
        for col in date_cols:
            # Convert column to datetime, coerces errors to NaT
            self.df[col] = pd.to_datetime(
                self.df[col], errors='coerce', dayfirst=True
            )
        return self.df
    
    def parse_month(month_str: str) -> datetime:
        """Strip quotes/whitespace and parse 'YYYY-MM' → datetime."""
        cleaned = month_str.strip().strip("'\"")
        return datetime.strptime(cleaned, "%Y-%m")

    def generate_month_list(start_month: str, end_month: str):
        """
        Return a list of datetime objects for each month-start
        from start_month to end_month inclusive.
        """
        start_dt = parse_month(start_month)
        end_dt = parse_month(end_month)
        if start_dt > end_dt:
            raise ValueError(f"Start month ({start_month}) is after end month ({end_month})")

        months = []
        current = start_dt
        while current <= end_dt:
            months.append(current)
            current += relativedelta(months=1)
        return months

    def last_day_of_month(dt: datetime) -> str:
        """
        Return the last day of dt's month as 'YYYY-MM-DD'.
        """
        day = calendar.monthrange(dt.year, dt.month)[1]
        return dt.replace(day=day).strftime("%Y-%m-%d")

    def flag_invalid_delays(self, start_col: str, end_col: str) -> pd.Series:
        """
        Flag rows where registration > receipt or either date is missing.
        Returns a boolean mask of invalid rows.
        """
        mask = (
            self.df[start_col].isna() |
            self.df[end_col].isna() |
            (self.df[start_col] > self.df[end_col])
        )
        return mask

    def compute_delay(self, start_col: str, end_col: str, drop_neg: bool=True) -> pd.DataFrame:
        """
        Compute the number of days between two date columns, 
        assign NaN for invalid ones, optionally drop negatives.

        For each row:
          1. Subtract registration (start_col) from receipt (end_col).
          2. Assign negative or invalid results to NaN.
          3. Optionally drop those rows if drop_neg=True.

        :param start_col: name of the registration-date column
        :param end_col:   name of the receipt-date column
        :param drop_neg:  whether to drop rows with negative delays
        :returns:         the DataFrame with a new 'delay_days' column
        """
        # Calculate raw delta in days
        self.df['delay_days'] = (
            self.df[end_col] - self.df[start_col]
        ).dt.days

        # Invalidate rows where dates are wrong
        invalid = self.flag_invalid_delays(start_col, end_col)
        self.df.loc[invalid, 'delay_days'] = pd.NA

        # Drop negative or missing delays
        if drop_neg:
            self.df = self.df[
                self.df['delay_days'].notna() & (self.df['delay_days'] >= 0)
            ].copy()
        return self.df
    
    def impute_delays(self, date_col: str='registrationdate') -> pd.DataFrame:
        """
        Fill missing delays with group-year mean, fallback to overall mean.
        """
        # Determine delay year: reg year or receipt year
        self.df['delay_year'] = (
            self.df['registrationdate'].dt.year
            .fillna(self.df['date_received_in_opg'].dt.year)
            .astype(int)
        )
        # Group-wise fill
        self.df['delay_days'] = self.df.groupby('delay_year')['delay_days']
            .transform(lambda s: s.fillna(s.mean()))
        # Fill any remaining with overall mean
        overall = self.df['delay_days'].mean()
        self.df['delay_days'] = self.df['delay_days'].fillna(overall)
        # Clean up
        self.df.drop(columns=['delay_year'], inplace=True)
        return self.df

    def derive_keys(self, id_cols: Tuple[str,str]) -> pd.DataFrame:
        """
        Build a hybrid derived_id: (case_no + date) or unique_id fallback.
        :param id_cols: Tuple of (case_no_col, unique_id_col)
        """
        c_no, u_id = id_cols
        def make_id(row):
            if pd.notna(row[c_no]) and str(row[c_no]).strip():
                date_str = row['date_received_in_opg'].strftime('%Y%m%d')
                return f"{row[c_no]}_{date_str}"
            return str(row[u_id])
        self.df['derived_id'] = self.df.apply(make_id, axis=1)
        return self.df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Drop duplicate rows based on 'derived_id', keep first occurrence.
        """
        self.df = self.df.drop_duplicates(subset='derived_id', keep='first')
        return self.df

    def run_all_checks(self) -> pd.DataFrame:
        """
        Executes full pipeline of validation, delay compute, impute, dedup.
        """
        # 1. Validate date formats
        self.validate_dates(['registrationdate','date_received_in_opg'])
        # 2. Compute and clean delays
        self.compute_delay('registrationdate','date_received_in_opg')
        # 3. Impute missing delays
        self.impute_delays()
        # 4. Derive keys & remove duplicates
        self.derive_keys(('case_no','unique_id'))
        self.remove_duplicates()
        return self.df



# Unit tests using pytest (tests/test_data_quality.py)
# ------------------------------------
# import pytest
# from data_quality import DataQualityChecks
#
# @pytest.fixture
# def sample_df(tmp_path):
#     # create a small sample CSV for testing
#     data = {
#         'case_no': ['A1', None, 'A3'],
#         'unique_id': ['u1','u2','u3'],
#         'registrationdate': ['2020-01-01','2020-05-05','notadate'],
#         'date_received_in_opg': ['2020-02-01','2020-05-01','2021-01-01'],
#         'casesubtype': ['pfa','pfa','xyz'],
#         'concern_type': ['Financial','Both','Health and Welfare']
#     }
#     df = pd.DataFrame(data)
#     path = tmp_path / 'sample.csv'
#     df.to_csv(path, index=False)
#     return str(path)
#
# def test_validate_dates(sample_df):
#     dq = DataQualityChecks(sample_df)
#     df2 = dq.validate_dates(['registrationdate','date_received_in_opg'])
#     assert df2['registrationdate'].isna().sum() == 1  # 'notadate'
#     assert df2['date_received_in_opg'].isna().sum() == 0
#
# def test_compute_delay(sample_df):
#     dq = DataQualityChecks(sample_df)
#     dq.validate_dates(['registrationdate','date_received_in_opg'])
#     df2 = dq.compute_delay('registrationdate','date_received_in_opg')
#     # one delay negative or NA should be dropped
#     assert all(df2['delay_days'] >= 0)
#
# def test_impute_delays(sample_df):
#     dq = DataQualityChecks(sample_df)
#     dq.validate_dates(['registrationdate','date_received_in_opg'])
#     dq.compute_delay('registrationdate','date_received_in_opg')
#     df3 = dq.impute_delays()
#     # no NaNs remain
#     assert df3['delay_days'].isna().sum() == 0
#
# def test_remove_duplicates(sample_df):
#     dq = DataQualityChecks(sample_df)
#     dq.validate_dates(['registrationdate','date_received_in_opg'])
#     dq.compute_delay('registrationdate','date_received_in_opg')
#     dq.derive_keys(('case_no','unique_id'))
#     # duplicate creation
#     df_dup = dq.df.append(dq.df.iloc[0])
#     dq.df = df_dup
#     df4 = dq.remove_duplicates()
#     assert df4.shape[0] < df_dup.shape[0]
```
