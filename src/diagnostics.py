# src/diagnostics.py
# Purpose: Lightweight diagnostics: VIF and correlations for redundancy checks.

from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm

def vif_for_numeric(df: pd.DataFrame, cols):
    """
    Compute VIF for a set of numeric columns (after dropping NA rows).
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[cols].dropna()
    X = sm.add_constant(X)
    vifs = []
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        vifs.append((c, variance_inflation_factor(X.values, i)))
    return pd.DataFrame(vifs, columns=["feature", "vif"])

def corr_table(df: pd.DataFrame, cols):
    """
    Pearson correlation table (pairwise) for a column subset.
    """
    return df[cols].corr(method="pearson")
