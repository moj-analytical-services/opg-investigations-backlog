# data_quality.py
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
from typing import List, Tuple, Dict


class DataQualityChecks:
    """
    A suite of data quality checks and resolution methods
    for PFA investigation datasets.

    This class provides:
      - Validation of date formats
      - Delay computation and imputation
      - Key derivation and duplicate removal
      - Stub for custom consistency rules via `check_consistency()`

    Example:
        dq = DataQualityChecks('data.csv')
        dq.validate_dates(['registrationdate', 'date_received_in_opg'])
        dq.compute_delay('registrationdate', 'date_received_in_opg')
        dq.impute_delays()
        # Apply business-specific consistency checks
        issues = dq.check_consistency()
        if issues:
            print(issues)
        df_clean = dq.run_all_checks()
    """

    def __init__(self, filepath: str):
        # Load CSV into DataFrame
        self.df = pd.read_csv(filepath, low_memory=False)

    def summary_statistics(self) -> pd.DataFrame:
        """
        Returns descriptive stats for all columns.
        """
        return self.df.describe(include="all")

    def validate_dates(self, date_cols: List[str]) -> pd.DataFrame:
        """
        Convert specified columns to datetime, coercing invalids to NaT.
        """
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce", dayfirst=True)
        return self.df

    @staticmethod
    def parse_month(month_str: str) -> datetime:
        """
        Parse 'YYYY-MM' (with optional quotes/whitespace) -> first day of month.
        """
        cleaned = month_str.strip().strip("'\"")
        return datetime.strptime(cleaned, "%Y-%m")

    @staticmethod
    def generate_month_list(start_month: str, end_month: str) -> List[datetime]:
        """
        Generate list of month-start datetimes between two 'YYYY-MM' strings.
        """
        start_dt = DataQualityChecks.parse_month(start_month)
        end_dt = DataQualityChecks.parse_month(end_month)
        if start_dt > end_dt:
            raise ValueError(f"Start month {start_month} is after {end_month}")
        months = []
        current = start_dt
        while current <= end_dt:
            months.append(current)
            current += relativedelta(months=1)
        return months

    @staticmethod
    def last_day_of_month(dt: datetime) -> str:
        """
        Return last calendar day of dt's month as 'YYYY-MM-DD'.
        """
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        return dt.replace(day=last_day).strftime("%Y-%m-%d")

    def flag_invalid_delays(self, start_col: str, end_col: str) -> pd.Series:
        """
        Boolean mask for rows where registration > receipt or dates missing.
        """
        return (
            self.df[start_col].isna()
            | self.df[end_col].isna()
            | (self.df[start_col] > self.df[end_col])
        )

    def compute_delay(
        self, start_col: str, end_col: str, drop_neg: bool = True
    ) -> pd.DataFrame:
        """
        Compute 'delay_days' = receipt - registration.
        Sets invalid or negative delays to NaN (or drops if drop_neg=True).
        """
        # Calculate raw difference
        self.df["delay_days"] = (self.df[end_col] - self.df[start_col]).dt.days
        # Invalidate incorrect
        invalid = self.flag_invalid_delays(start_col, end_col)
        self.df.loc[invalid, "delay_days"] = pd.NA
        # Optionally drop negatives/missing
        if drop_neg:
            self.df = self.df[
                self.df["delay_days"].notna() & (self.df["delay_days"] >= 0)
            ].copy()
        return self.df

    def impute_delays(self) -> pd.DataFrame:
        """
        Impute missing 'delay_days' by group-year mean, fallback overall mean.
        """
        # Decide grouping year
        self.df["delay_year"] = (
            self.df["registrationdate"]
            .dt.year.fillna(self.df["date_received_in_opg"].dt.year)
            .astype(int)
        )
        # Group mean imputation
        self.df["delay_days"] = self.df.groupby("delay_year")["delay_days"].transform(
            lambda s: s.fillna(s.mean())
        )
        # Fallback
        overall = self.df["delay_days"].mean()
        self.df["delay_days"] = self.df["delay_days"].fillna(overall)
        self.df.drop(columns=["delay_year"], inplace=True)
        return self.df

    def derive_keys(self, id_cols: Tuple[str, str]) -> pd.DataFrame:
        """
        Build 'derived_id' = 'case_no_YYYYMMDD' or fallback 'unique_id'.
        """
        case_col, uniq_col = id_cols

        def make_id(row):
            if pd.notna(row[case_col]) and str(row[case_col]).strip():
                date_str = row["date_received_in_opg"].strftime("%Y%m%d")
                return f"{row[case_col]}_{date_str}"
            return str(row[uniq_col])

        self.df["derived_id"] = self.df.apply(make_id, axis=1)
        return self.df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Drop rows with duplicate 'derived_id', keep first.
        """
        self.df = self.df.drop_duplicates(subset="derived_id", keep="first")
        return self.df

    def check_consistency(self) -> Dict[str, pd.Series]:
        """
        Stub for business rule consistency checks.
        Returns a dict mapping rule names to boolean masks of violations.
        Example rule: registrationdate <= date_received_in_opg.
        """
        masks: Dict[str, pd.Series] = {}
        # Example consistency rule
        masks["reg_before_recv"] = (
            self.df["registrationdate"] <= self.df["date_received_in_opg"]
        ) | self.df["registrationdate"].isna()
        # TODO: add more rules here, e.g. valid 'concern_type' values, postcode formats, etc.
        return masks

    def run_all_checks(self) -> pd.DataFrame:
        """
        Execute full pipeline: validate -> compute -> impute -> derive -> dedup.
        """
        self.validate_dates(["registrationdate", "date_received_in_opg"])
        self.compute_delay("registrationdate", "date_received_in_opg")
        self.impute_delays()
        self.derive_keys(("case_no", "unique_id"))
        self.remove_duplicates()
        return self.df
