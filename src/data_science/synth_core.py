# synth_core.py
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

EPS = 1e-9


def _rank_gaussian(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    ranks = np.argsort(np.argsort(x)) + 1
    u = (ranks - 3 / 8) / (len(x) + 1 / 4)
    z = norm.ppf(np.clip(u, EPS, 1 - EPS))
    z = (z - np.mean(z)) / (np.std(z) + EPS)
    return z


def _inv_rank_gaussian(z: np.ndarray, ref: np.ndarray) -> np.ndarray:
    order = np.argsort(z)
    sorted_ref = np.sort(ref)
    out = np.empty_like(ref, dtype=float)
    out[order] = sorted_ref
    return out


class GaussianCopulaSynthesizer:
    def __init__(self, categorical_cols: Optional[List[str]] = None):
        self.categorical_cols = categorical_cols or []
        self.numeric_cols: List[str] = []
        self.num_means = None
        self.num_cov = None
        self.cat_probs: Dict[str, pd.Series] = {}
        self.fitted = False
        self._ref_numeric = None

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        self.numeric_cols = [c for c in df.columns if c not in self.categorical_cols]
        num = df[self.numeric_cols].select_dtypes(include=[np.number]).copy()
        for c in self.numeric_cols:
            if not np.issubdtype(df[c].dtype, np.number):
                num[c] = pd.to_numeric(df[c], errors="coerce")
        num = num.fillna(num.median())

        Z = []
        for c in self.numeric_cols:
            z = _rank_gaussian(num[c].values)
            Z.append(z)
        if len(Z) == 0:
            self.num_means = None
            self.num_cov = None
        else:
            Z = np.column_stack(Z)
            self.num_means = Z.mean(axis=0)
            self.num_cov = np.cov(Z, rowvar=False) + np.eye(Z.shape[1]) * 1e-6
        self._ref_numeric = num

        for c in self.categorical_cols:
            v = df[c].astype("category")
            probs = v.value_counts(dropna=False) + 1
            probs = probs / probs.sum()
            self.cat_probs[c] = probs
        self.fitted = True
        return self

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        assert self.fitted, "Call fit() first."
        rng = np.random.default_rng(seed)

        if self.num_means is not None:
            Zsyn = rng.multivariate_normal(self.num_means, self.num_cov, size=n)
            num_syn = {}
            for j, c in enumerate(self.numeric_cols):
                num_syn[c] = _inv_rank_gaussian(Zsyn[:, j], self._ref_numeric[c].values)
            df_num = pd.DataFrame(num_syn)
        else:
            df_num = pd.DataFrame(index=range(n))

        df_cat = pd.DataFrame(index=range(n))
        for c, probs in self.cat_probs.items():
            cats = probs.index.tolist()
            p = probs.values
            idx = rng.choice(len(cats), size=n, p=p)
            df_cat[c] = [cats[i] for i in idx]

        return pd.concat([df_num, df_cat], axis=1)


def stratified_synthesis(
    df: pd.DataFrame,
    cat_cols: List[str],
    min_per_stratum: int = 50,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit per-stratum synthesizers on cross of cat_cols when counts >= min_per_stratum.
    Back-off to global synthesizer otherwise. Helps preserve group structure without overfitting tiny groups.
    """
    rng = np.random.default_rng(seed)
    if not cat_cols:
        gc = GaussianCopulaSynthesizer(categorical_cols=[]).fit(df)
        return gc.sample(len(df), seed=int(rng.integers(0, 1e9)))
    # build strata
    strata = df.groupby(cat_cols)
    pieces = []
    for keys, sub in strata:
        if len(sub) >= min_per_stratum:
            cats_in = [c for c in df.columns if c in cat_cols]
            gc = GaussianCopulaSynthesizer(categorical_cols=cats_in).fit(sub)
            pieces.append(gc.sample(len(sub), seed=int(rng.integers(0, 1e9))))
        else:
            # backoff: sample from global
            pass
    # backoff global for all
    if len(pieces) < len(df):
        gc_global = GaussianCopulaSynthesizer(categorical_cols=cat_cols).fit(df)
        need = len(df) - sum(len(p) for p in pieces)
        pieces.append(gc_global.sample(need, seed=int(rng.integers(0, 1e9))))
    return pd.concat(pieces, ignore_index=True)
