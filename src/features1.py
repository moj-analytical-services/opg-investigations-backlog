# src/features.py
# Purpose: Central place to build leak-safe preprocessing pipelines
#          (impute/scale numerics, one-hot encode categoricals, target encode high-card).

from __future__ import annotations
from typing import List, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .encoders import KFoldTargetEncoder

# Default feature lists (will be subsetted by presence in df)
DEFAULT_NUM = ["weighting", "days_to_alloc"]
DEFAULT_CAT = ["team", "case_type", "risk", "reallocation"]
# Potential high-card columns; if missing in df, they’re ignored.
DEFAULT_HIGH_CARD = ["occupation"]

def build_preprocessor(
    df: pd.DataFrame,
    y_name: str = "needs_legal_review",
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    high_card_cols: Optional[List[str]] = None,
    numeric_interactions: bool = True,
):
    """
    Returns a ColumnTransformer that:
      - imputes + scales numerics (and optionally adds interaction-only polynomials),
      - one-hot encodes medium-card categoricals,
      - K-Fold target-encodes high-card categoricals (requires y at fit time).

    Notes
    -----
    - ColumnTransformer will pass y through to Transformers (our KFoldTargetEncoder uses it).
    - Any column that doesn't exist is automatically dropped by the subset below.
    """
    # Resolve column lists by presence
    num = [c for c in (num_cols or DEFAULT_NUM) if c in df.columns]
    cat = [c for c in (cat_cols or DEFAULT_CAT) if c in df.columns]
    high = [c for c in (high_card_cols or DEFAULT_HIGH_CARD) if c in df.columns]

    # Numeric pipeline: impute median -> scale -> (optional) pairwise interactions
    num_steps = [
        ("impute", SimpleImputer(strategy="median")),   # robust to skew/outliers
        ("scale", StandardScaler()),
    ]
    if numeric_interactions and len(num) >= 2:
        # interaction_only=True to avoid squared terms unless you want curvature
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)))
    num_pipe = Pipeline(num_steps)

    # Categorical one-hot: impute most frequent -> OHE with reference level dropped
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    # High-card target encoding (e.g., occupation): leak-safe via our K-Fold encoder
    # This transformer expects DF with those cols and receives y in fit().
    te_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("te", KFoldTargetEncoder(cols=high, target_col=y_name, n_splits=5, smoothing=10.0)),
    ]) if high else "drop"

    # Build the ColumnTransformer (order matters only for interpretability)
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num),
            ("cat", cat_pipe, cat),
            ("high", te_pipe, high),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre
