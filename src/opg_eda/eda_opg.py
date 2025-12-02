from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter

    _HAS_LIFELINES = True
except Exception:
    _HAS_LIFELINES = False

try:

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


@dataclass
class EDAConfig:
    id_col: str
    date_received: str
    date_allocated: str
    date_signed_off: str
    target_col: Optional[str] = None
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None
    time_index_col: Optional[str] = None
    team_col: Optional[str] = None
    risk_col: Optional[str] = None
    case_type_col: Optional[str] = None
    region_col: Optional[str] = None
    resample_rule: str = "D"
    lag_list: Tuple[int, ...] = (1, 7, 14)


class OPGInvestigationEDA:
    def __init__(self, df: pd.DataFrame, config: EDAConfig) -> None:
        self.df = df.copy()
        self.cfg = config
        self._derive_standard_fields()

    def _derive_standard_fields(self) -> None:
        for col in [
            self.cfg.date_received,
            self.cfg.date_allocated,
            self.cfg.date_signed_off,
        ]:
            if col not in self.df.columns:
                raise KeyError(f"Expected date column missing: {col}")
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        self.df["days_to_alloc"] = (
            self.df[self.cfg.date_allocated] - self.df[self.cfg.date_received]
        ).dt.days
        self.df.loc[self.df["days_to_alloc"] < 0, "days_to_alloc"] = np.nan
        self.df["event_alloc"] = self.df[self.cfg.date_allocated].notna().astype(int)

        self.df["days_to_signoff"] = (
            self.df[self.cfg.date_signed_off] - self.df[self.cfg.date_received]
        ).dt.days
        self.df.loc[self.df["days_to_signoff"] < 0, "days_to_signoff"] = np.nan
        self.df["event_signoff"] = self.df[self.cfg.date_signed_off].notna().astype(int)

    def quick_overview(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        out["shape"] = self.df.shape
        out["dtypes"] = self.df.dtypes.astype(str).to_dict()
        out["missing_pct"] = self.df.isna().mean().sort_values(ascending=False)
        out["duplicate_ids"] = (
            int(self.df.duplicated(subset=[self.cfg.id_col]).sum())
            if self.cfg.id_col in self.df
            else None
        )
        coverage = {}
        for col in [
            self.cfg.date_received,
            self.cfg.date_allocated,
            self.cfg.date_signed_off,
        ]:
            coverage[col] = (self.df[col].min(), self.df[col].max())
        out["date_ranges"] = coverage
        if self.cfg.target_col and self.cfg.target_col in self.df:
            out["target_counts"] = (
                self.df[self.cfg.target_col].value_counts(dropna=False).to_dict()
            )
            out["target_share"] = (
                self.df[self.cfg.target_col]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            )
        return out

    def missingness_matrix(self, cols: Optional[List[str]] = None) -> pd.Series:
        cols = cols or list(self.df.columns)
        return self.df[cols].isna().mean().sort_values(ascending=False)

    def missing_vs_target(self, feature: str) -> Optional[pd.Series]:
        if not self.cfg.target_col or self.cfg.target_col not in self.df:
            return None
        return (
            self.df.assign(_miss=self.df[feature].isna().astype(int))
            .groupby(self.cfg.target_col)["_miss"]
            .mean()
            .sort_index()
        )

    def iqr_outliers(self, col: str) -> Dict[str, object]:
        ser = self.df[col].dropna()
        q1, q3 = ser.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (self.df[col] < lo) | (self.df[col] > hi)
        return {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lo),
            "upper_bound": float(hi),
            "n_outliers": int(mask.sum()),
            "outlier_rows": self.df.loc[mask, [self.cfg.id_col, col]].head(10),
        }

    def group_summary(
        self, by: List[str], metrics: Dict[str, Tuple[str, str]]
    ) -> pd.DataFrame:
        agg_spec = {
            k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in metrics.items()
        }
        out = self.df.groupby(by).agg(**agg_spec).reset_index()
        return out

    def numeric_correlations(self, method: str = "spearman") -> pd.DataFrame:
        if not self.cfg.numeric_cols:
            raise ValueError("No numeric_cols configured for correlation.")
        return self.df[self.cfg.numeric_cols].corr(method=method)

    @staticmethod
    def cramers_v(x: pd.Series, y: pd.Series) -> float:
        if not _HAS_SCIPY:
            warnings.warn("scipy not available: returning NaN for Cramér’s V")
            return float("nan")
        tbl = pd.crosstab(x, y)
        from scipy.stats import chi2_contingency

        chi2 = chi2_contingency(tbl)[0]
        n = tbl.to_numpy().sum()
        phi2 = chi2 / n
        r, k = tbl.shape
        phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rcorr = r - (r - 1) ** 2 / (n - 1)
        kcorr = k - (k - 1) ** 2 / (n - 1)
        return float(np.sqrt(phi2corr / max(1e-12, min(kcorr - 1, rcorr - 1))))

    def redundancy_drop_list(
        self, cols: Optional[List[str]] = None, thresh: float = 0.90
    ) -> List[str]:
        cols = cols or (self.cfg.numeric_cols or [])
        corr = self.df[cols].corr().abs()
        upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
        return [c for c in upper.columns if (upper[c] > thresh).any()]

    def vif_report(self, cols: Optional[List[str]] = None) -> Optional[pd.Series]:
        if not _HAS_STATSMODELS:
            warnings.warn("statsmodels not available: VIF report skipped.")
            return None
        cols = cols or (self.cfg.numeric_cols or [])
        X = self.df[cols].dropna()
        X = sm.add_constant(X)
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
            name="VIF",
        )
        return vif.sort_values(ascending=False)

    def imbalance_summary(self) -> Optional[Dict[str, float]]:
        if not self.cfg.target_col or self.cfg.target_col not in self.df:
            return None
        vc = self.df[self.cfg.target_col].value_counts(dropna=False)
        pos_share = float(self.df[self.cfg.target_col].mean())
        return {
            "n0": int(vc.get(0, 0)),
            "n1": int(vc.get(1, 0)),
            "pos_share": round(pos_share, 4),
        }

    def leakage_scan(self, suspicious_keywords: Iterable[str]) -> List[str]:
        keys = [k.lower() for k in suspicious_keywords]
        hits = []
        for c in self.df.columns:
            low = c.lower()
            if any(k in low for k in keys):
                hits.append(c)
        return hits

    def binned_interaction_rate(
        self, num_col: str, cat_col: str, target: Optional[str] = None, q: int = 5
    ) -> pd.DataFrame:
        target = target or self.cfg.target_col
        if not target:
            raise ValueError("Target column required for interaction rates.")
        tmp = self.df[[num_col, cat_col, target]].dropna(subset=[num_col, cat_col])
        tmp["__bin__"] = pd.qcut(
            tmp[num_col], q=min(q, tmp[num_col].nunique()), duplicates="drop"
        )
        out = tmp.groupby([cat_col, "__bin__"])[target].mean().unstack()
        return out

    def resample_time_series(self, metrics: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
        if not self.cfg.time_index_col:
            raise ValueError("Set time_index_col in EDAConfig to resample.")
        ts = self.df.set_index(
            pd.to_datetime(self.df[self.cfg.time_index_col], errors="coerce")
        ).sort_index()
        agg_spec = {
            k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in metrics.items()
        }
        out = ts.resample(self.cfg.resample_rule).agg(**agg_spec)
        if self.cfg.resample_rule.upper() == "D":
            for k in list(metrics.keys()):
                out[f"{k}_7d"] = out[k].rolling(7, min_periods=3).mean()
        return out

    def lag_correlations(
        self, s1: pd.Series, s2: pd.Series, lags: Optional[Iterable[int]] = None
    ) -> pd.Series:
        lags = list(lags or self.cfg.lag_list)
        out = {}
        for k in lags:
            out[f"lag_{k}"] = float(s1.corr(s2.shift(k)))
        return pd.Series(out)

    def km_quantiles_by_group(
        self,
        duration: str,
        event: str,
        group: str,
        probs: Iterable[float] = (0.25, 0.5, 0.75),
    ) -> pd.DataFrame:
        res = []
        for g, dfg in self.df.groupby(group):
            if _HAS_LIFELINES:
                km = KaplanMeierFitter()
                km.fit(durations=dfg[duration], event_observed=dfg[event])
                row = {"group": g}
                for p in probs:
                    row[f"q{int(p*100)}"] = float(km.quantile(p))
                res.append(row)
            else:
                warnings.warn(
                    "lifelines not available; using naive quantiles (censoring ignored)."
                )
                row = {"group": g}
                for p in probs:
                    row[f"q{int(p*100)}"] = float(dfg[duration].quantile(p))
                res.append(row)
        return pd.DataFrame(res).sort_values("group").reset_index(drop=True)

    def monthly_kpis(self) -> pd.DataFrame:
        if not self.cfg.team_col:
            raise ValueError("team_col must be configured for monthly KPIs.")
        month = self.df[self.cfg.date_received].dt.to_period("M").dt.to_timestamp()
        tmp = self.df.assign(__month=month)
        out = (
            tmp.groupby([self.cfg.team_col, "__month"])
            .agg(
                backlog=(
                    ("backlog", "last") if "backlog" in tmp.columns else ("id", "count")
                ),
                median_alloc=("days_to_alloc", "median"),
                legal_rate=(
                    (self.cfg.target_col, "mean")
                    if self.cfg.target_col
                    else (self.cfg.id_col, "count")
                ),
            )
            .reset_index()
        )
        return out.sort_values([self.cfg.team_col, "__month"])
