# src/features.py
# Impute/scale numerics, OHE categoricals, optional numeric interactions, and K-Fold target encoding for high-card vars.

from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .encoders import KFoldTargetEncoder

DEFAULT_NUM = ["weighting", "days_to_alloc"]
DEFAULT_CAT = ["team", "case_type", "risk", "reallocation"]
DEFAULT_HIGH = ["occupation"]  # if present

def build_preprocessor(df: pd.DataFrame, y_name: str = "needs_legal_review", numeric_interactions: bool = True):
    num = [c for c in DEFAULT_NUM if c in df.columns]
    cat = [c for c in DEFAULT_CAT if c in df.columns]
    high = [c for c in DEFAULT_HIGH if c in df.columns]

    num_steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if numeric_interactions and len(num) >= 2:
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    te_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("te", KFoldTargetEncoder(cols=high, target_col=y_name, n_splits=5, smoothing=10.0)),
    ]) if high else "drop"

    pre = ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", cat_pipe, cat),
        ("high", te_pipe, high),
    ], remainder="drop", verbose_feature_names_out=False)
    return pre
