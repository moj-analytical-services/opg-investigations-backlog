from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICALS = ["team", "case_type", "risk", "reallocation"]
NUMERICS = ["weighting", "days_to_alloc"]

def build_design_matrix(df: pd.DataFrame):
    X_cat = [c for c in CATEGORICALS if c in df.columns]
    X_num = [c for c in NUMERICS if c in df.columns]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), X_cat),
            ("num", StandardScaler(), X_num),
        ]
    )
    return pre

def derive_case_level_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "reallocation" in out.columns and out["reallocation"].dtype != bool:
        out["reallocation"] = out["reallocation"].astype(bool)
    return out
