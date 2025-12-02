# preprocessing.py
# Utilities for data loading, cleaning, feature engineering, and time-series prep.
# Uses pandas, numpy, scikit-learn. Charts use matplotlib only.

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_csv(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace in string columns, unify missing markers
    df = df.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
    return df


def drop_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique(dropna=False)
    to_drop = [c for c, u in nunique.items() if u <= 1]
    return df.drop(columns=to_drop)


def iqr_clip(df: pd.DataFrame, cols: List[str], k: float = 1.5) -> pd.DataFrame:
    """Clip extreme outliers using IQR rule per column."""
    df = df.copy()
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def build_tabular_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    impute_numeric: str = "median",
    impute_categorical: str = "most_frequent",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    num_steps = []
    if impute_numeric:
        num_steps.append(("imputer", SimpleImputer(strategy=impute_numeric)))
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_categorical)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )
    return pre


def make_timeseries(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str = "MS", agg: str = "sum"
) -> pd.Series:
    """Return a regular time series indexed by date at given freq with chosen aggregation."""
    s = (
        df[[date_col, value_col]]
        .dropna()
        .assign(**{date_col: pd.to_datetime(df[date_col])})
        .set_index(date_col)[value_col]
        .resample(freq)
    )
    if agg == "sum":
        return s.sum().astype(float)
    elif agg == "mean":
        return s.mean().astype(float)
    else:
        return getattr(s, agg)().astype(float)


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    df = df.copy()
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["dow"] = d.dt.dayofweek
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    return df


def lag_features(s: pd.Series, lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = s.shift(k)
    return df


def train_test_split_time(s: pd.Series, test_size: int):
    """Split keeping temporal order; test_size is number of periods."""
    return s.iloc[:-test_size], s.iloc[-test_size:]
