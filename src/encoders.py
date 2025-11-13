# src/encoders.py
# Purpose: A scikit-learn compatible K-fold target encoder to handle
#          high-cardinality categorical variables without target leakage.

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    K-Fold target mean encoder for high-cardinality categoricals.

    - Fits fold-by-fold mappings (train folds only) to avoid target leakage.
    - Applies smoothing toward the global prior mean to stabilise rare levels.
    - During transform(), uses a full-data smoothed mapping (seen categories)
      and falls back to the global prior for unseen categories.

    Parameters
    ----------
    cols : list[str]
        Categorical columns to encode (must exist in X).
    target_col : str
        Name of the target column in y (binary 0/1 recommended).
    n_splits : int
        Number of folds for out-of-fold estimates.
    smoothing : float
        Larger -> more weight on global prior; smaller -> more weight on raw mean.
    random_state : Optional[int]
        For deterministic fold splits.
    """
    def __init__(
        self,
        cols: List[str],
        target_col: str,
        n_splits: int = 5,
        smoothing: float = 10.0,
        random_state: Optional[int] = 42,
    ):
        self.cols = cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_: Optional[float] = None
        self.full_maps_: dict = {}   # mapping per column (Series: category -> encoded float)

    def _smooth_mean(self, count: pd.Series, mean: pd.Series, prior: float) -> pd.Series:
        # Bayesian-style smoothing toward prior mean
        return (count * mean + self.smoothing * prior) / (count + self.smoothing)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        # X is the subset with specified columns; y is the full target Series/array.
        if y is None:
            raise ValueError("KFoldTargetEncoder requires y to be provided to fit().")
        X = pd.DataFrame(X, columns=self.cols) if not isinstance(X, pd.DataFrame) else X[self.cols]
        y = pd.Series(y, name=self.target_col).astype(float)

        # Prior (global) mean
        self.global_mean_ = float(y.mean())

        # Build full-data smoothed maps for transform()
        self.full_maps_ = {}
        for c in self.cols:
            tmp = pd.DataFrame({c: X[c].astype("category"), self.target_col: y.values})
            grp = tmp.groupby(c)[self.target_col].agg(["mean", "count"])
            sm = self._smooth_mean(grp["count"], grp["mean"], self.global_mean_)
            self.full_maps_[c] = sm

        return self

    def transform(self, X):
        # Map categories to smoothed means; unseen -> global prior
        X = pd.DataFrame(X, columns=self.cols) if not isinstance(X, pd.DataFrame) else X[self.cols]
        out = pd.DataFrame(index=X.index)
        for c in self.cols:
            m = self.full_maps_.get(c, pd.Series(dtype=float))
            out[f"{c}_te"] = X[c].map(m).fillna(self.global_mean_)
        return out.values  # return numpy for ColumnTransformer compatibility
