# eda_opg.py
# Advanced, object-oriented EDA utilities tailored to OPG Investigation Backlog data.
# We estimate time to PG sign-off with a Kaplan–Meier curve so we can use both completed and still-open cases without bias. From the survival curve we read median and tail quantiles (P80/P90). Those feed capacity planning, SLAs, and discrete-event simulation. For example, High-risk cases show a longer P90, so adding experienced reviewers there reduces the tail and the visible backlog. We verify group differences with a log-rank test, and we export quantiles by case type as inputs to the microsimulation.

# For each investigation case we care about “How long from when OPG receives the concern until PG signs it off?”. Many cases are still open on the day you analyse the data. Those open cases are right-censored: we know they’ve already taken at least X days, but we don’t yet know the final total. If you simply drop open cases or pretend they finished today, you’ll bias results (usually underestimating true times).

from __future__ import (
    annotations,
)  # ensures forward refs in type hints work in Python <3.11

from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)  # precise type hints for maintainability and IDE help
import warnings  # to warn (not crash) when optional deps are missing

import numpy as np  # numerical work (corr, quantiles)
import pandas as pd  # core dataframe operations
from dataclasses import dataclass  # dataclass for a clear, typed configuration object


# Optional scientific/statistical packages.
try:
    from lifelines import (
        KaplanMeierFitter,
    )  # survival analysis (censoring-aware) - non-parametric stats

    _HAS_LIFELINES = True  # flag for availability
except Exception:
    _HAS_LIFELINES = False  # if not installed, we degrade gracefully

try:
    from scipy.stats import chi2_contingency  # for Cramér’s V (categorical association)

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import statsmodels.api as sm  # for VIF (variance inflation factor) to remove multicolinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


# -------------------------------
# 1) Configuration for the EDA run
# -------------------------------


# eda_opg.py
# Advanced, object-oriented EDA utilities tailored to OPG Investigation Backlog data.
# We estimate time to PG sign-off with a Kaplan–Meier curve so we can use both completed and still-open cases without bias. From the survival curve we read median and tail quantiles (P80/P90). Those feed capacity planning, SLAs, and discrete-event simulation. For example, High-risk cases show a longer P90, so adding experienced reviewers there reduces the tail and the visible backlog. We verify group differences with a log-rank test, and we export quantiles by case type as inputs to the microsimulation.

# For each investigation case we care about “How long from when OPG receives the concern until PG signs it off?”. Many cases are still open on the day you analyse the data. Those open cases are right-censored: we know they’ve already taken at least X days, but we don’t yet know the final total. If you simply drop open cases or pretend they finished today, you’ll bias results (usually underestimating true times).


# -------------------------------
# 1) Configuration for the EDA run
# -------------------------------


@dataclass
class EDAConfig:
    """
    Configuration object declaring column names and options explicitly.
    Make structure explicit to avoid 'magic strings' spread in code.
    """

    id_col: str  # unique case identifier column
    date_received: str  # date case received by OPG
    date_allocated: str  # date case allocated to investigator (may be missing)
    date_signed_off: str  # date case signed off (may be missing)
    target_col: Optional[str] = None  # optional target (e.g. 'legal_review' 0/1)
    numeric_cols: Optional[List[str]] = None  # numeric feature columns
    categorical_cols: Optional[List[str]] = None  # categorical feature columns
    time_index_col: Optional[str] = (
        None  # column to use as time index for resampling (e.g. 'date_received')
    )
    team_col: Optional[str] = None  # team field for KPI grouping
    risk_col: Optional[str] = None  # risk band
    case_type_col: Optional[str] = None  # case type
    region_col: Optional[str] = None  # region (optional)
    # Defaults for time-series resampling and lag analysis
    resample_rule: str = "D"  # daily by default
    lag_list: Tuple[int, ...] = (1, 7, 14)  # lags to compute correlations at


# -------------------------------
# 2) Main EDA class
# -------------------------------


class OPGInvestigationEDA:
    """
    An object-oriented EDA toolkit for OPG investigations backlog problems.
    Provides validated, reproducible, unit-testable methods for exploratory analysis.
    """

    def __init__(self, df: pd.DataFrame, config: EDAConfig) -> None:
        """
        Store the data and config, and immediately derive standard fields (durations + censor flags).
        """
        self.df = df.copy()  # do not mutate the caller's DataFrame
        self.cfg = config  # keep typed configuration
        self._derive_standard_fields()  # add days_to_alloc + censor flags up-front

    # ---------------------------
    # Core derivations and checks
    # ---------------------------

    def _derive_standard_fields(self) -> None:
        """
        Derive commonly used interval variables and censor flags.
        These are used across many EDA tasks in backlog analysis.
        """
        # Ensure the three date columns exist and are datetime
        for col in [
            self.cfg.date_received,
            self.cfg.date_allocated,
            self.cfg.date_signed_off,
        ]:
            if col not in self.df.columns:
                raise KeyError(
                    f"Expected date column missing: {col}"
                )  # fail early with a clear message
            self.df[col] = pd.to_datetime(
                self.df[col], errors="coerce"
            )  # coerce invalid strings to NaT

        # Derive time-to-sign-off (days). For NaT (not signed_off yet), result is NaN.
        self.df["days_to_signoff"] = (
            self.df[self.cfg.date_signed_off] - self.df[self.cfg.date_received]
        ).dt.days

        # Negative durations indicate data issues (signed_off before received). Set to NaN (to be investigated).
        self.df.loc[self.df["days_to_signoff"] < 0, "days_to_signoff"] = np.nan

        # Censor flag for signed_off event: 1 if signed_off exists, else 0.
        self.df["event_signoff"] = self.df[self.cfg.date_signed_off].notna().astype(int)

        # Derive time-to-allocate(days) similarly; not always used, but often requested.
        self.df["days_to_alloc"] = (
            self.df[self.cfg.date_allocated] - self.df[self.cfg.date_received]
        ).dt.days
        self.df.loc[self.df["days_to_alloc"] < 0, "days_to_alloc"] = np.nan

        # Censor flag for allocate: 1 if allocated date exists.
        self.df["event_alloc"] = self.df[self.cfg.date_allocated].notna().astype(int)

    # ---------------------------
    # 0) Quick structural summary
    # ---------------------------

    def quick_overview(self) -> Dict[str, object]:
        """
        Provide a compact dict of shape, dtypes, missingness, duplicate id counts, and time coverage.
        """
        out: Dict[str, object] = {}  # container for multiple small facts
        out["shape"] = self.df.shape  # (rows, cols)
        # .astype(int) → Converts True → 1 and False → 0.
        out["dtypes"] = self.df.dtypes.astype(
            str
        ).to_dict()  # map column -> dtype string
        # mean(): When called on a boolean DataFrame (isna()), True is treated as 1 and False as 0. By default, mean() operates column-wise (axis=0), so it computes the fraction of missing values in each column.
        # .sort_values(ascending=False): Sorts the resulting Series in descending order, so columns with the highest percentage of missing values appear first.
        out["missing_pct"] = (
            self.df.isna().mean().sort_values(ascending=False)
        )  # missingness percentage per column

        # Duplicate ID counts (only if id column is provided)
        if self.cfg.id_col in self.df.columns:
            out["duplicate_ids"] = int(
                self.df.duplicated(subset=[self.cfg.id_col]).sum()
            )
        else:
            out["duplicate_ids"] = None

        # Time coverage for the 3 key date columns
        coverage = {}
        for col in [
            self.cfg.date_received,
            self.cfg.date_allocated,
            self.cfg.date_signed_off,
        ]:
            coverage[col] = (self.df[col].min(), self.df[col].max())
        out["date_ranges"] = coverage

        # Class balance if target present (e.g. legal_review)
        if self.cfg.target_col and self.cfg.target_col in self.df:
            # Count occurrences of each class, including NaN
            out["target_counts"] = (
                self.df[self.cfg.target_col].value_counts(dropna=False).to_dict()
            )
            # returns percentage-like values (fractions of total count). Useful when you want distribution instead of absolute counts. You can multiply by 100 to get percentages:
            out["target_share"] = (
                self.df[self.cfg.target_col]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            )

        return out

    # ---------------------------
    # 1) Missing data profiling
    # ---------------------------

    def missingness_matrix(self, cols: Optional[List[str]] = None) -> pd.Series:
        """
        Return missingness fraction by column, optionally restricted to a subset list.
        """
        cols = cols or list(self.df.columns)  # if no list provided, use all columns
        return (
            self.df[cols].isna().mean().sort_values(ascending=False)
        )  # fraction missing per column

    def missing_vs_target(self, feature: str) -> Optional[pd.Series]:
        """
        Check whether missingness in a feature relates to the target (if any).
        Returns target-wise mean missingness (proportions) or None if no target.
        """
        if not self.cfg.target_col or self.cfg.target_col not in self.df:
            return None  # cannot compare without a target
        return (  # compute the mean proportion of missing values for a given column (feature) grouped by the target column
            # assign(_miss=...) → Adds a temporary column _miss to the DataFrame without modifying the original. containing those 0/1 values.
            # .astype(int) → Converts True → 1 and False → 0.
            self.df.assign(
                _miss=self.df[feature].isna().astype(int)
            )  # 1 if missing else 0
            # Calculates the mean of _miss in each group → proportion of missing values.
            .groupby(self.cfg.target_col)["_miss"]
            .mean()  # average missingness by target
            .sort_index()  # Sorts the result by the group labels.
        )

    # ---------------------------
    # 2) Distribution & outliers
    # ---------------------------

    def iqr_outliers(self, col: str) -> Dict[str, object]:
        """
        Classic IQR rule to flag outliers for a numeric column (robust to skew).
        """
        # By default, dropna() works row-wise (axis=0) and drops rows where at least one value is missing.
        # If you only want to drop rows where all values are missing, you can use:self.df[cols].dropna(how='all')
        # If you want to drop rows with NaN only in specific columns, you can pass subset=cols.
        ser = self.df[
            col
        ].dropna()  # ignore NaN and drops rows where at least one value is missing.
        q1, q3 = ser.quantile([0.25, 0.75])  # first and third quartiles
        iqr = q3 - q1  # interquartile range
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # Tukey's rule bounds
        mask = (self.df[col] < lo) | (self.df[col] > hi)  # boolean mask for outliers
        return {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lo),
            "upper_bound": float(hi),
            "n_outliers": int(mask.sum()),
            "outlier_rows": self.df.loc[mask, [self.cfg.id_col, col]].head(
                10
            ),  # sample some IDs to inspect
        }

    # ---------------------------
    # 3) Categorical summaries
    # ---------------------------

    def group_summary(
        self, by: List[str], metrics: Dict[str, Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        General grouped summary: pass a list of group columns and metric spec dict:
        metrics = {"n": ("id", "count"), "legal_rate": ("legal_review", "mean")}

        Returns sorted grouped table.
        """
        # Build an agg dict in the signature pandas expects
        # building a dynamic aggregation specification for a Pandas groupby using pd.NamedAgg.
        agg_spec = {
            k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in metrics.items()
        }
        # agg(**agg_spec): Expands the dictionary into keyword arguments for named aggregation.
        out = (
            self.df.groupby(by, observed=False).agg(**agg_spec).reset_index()
        )  # perform the groupby aggregation
        return out  # leave sorting to the caller

    # ---------------------------
    # 4) Correlations & redundancy
    # ---------------------------

    def numeric_correlations(self, method: str = "spearman") -> pd.DataFrame:
        """
        Return numeric-numeric correlation matrix (Spearman default for robustness to skew and outliers).
        """
        if not self.cfg.numeric_cols:
            raise ValueError("No numeric_cols configured for correlation.")
        return self.df[self.cfg.numeric_cols].corr(method=method)  # correlation matrix

    @staticmethod
    def cramers_v(x: pd.Series, y: pd.Series) -> float:
        """
        Cramér’s V between two categorical variables (bias-corrected).
        Requires scipy; returns np.nan if unavailable.
        Define a function that takes two pandas Series (two categorical columns)
        and returns a single number (the V value).
        Cramér’s V is a 0–1 strength-of-association measure between two
        categorical variables (0 = no association, 1 = very strong).
        It’s like a correlation for categories.
        Use it in EDA to spot redundant categorical features
        (e.g., case_type vs risk_band) so the model isn’t learning the same information twice.
        Cramér’s V (important with small samples or many categories).
        Notes the SciPy dependency.
        Feature redundancy: If case_type and risk_band have a high V (say ≥ 0.5),
        they’re strongly associated. You might:
        keep both but be cautious about interpreting coefficients,
        or drop/merge one to simplify the model and reduce multicollinearity risk.
        Data understanding: High associations can reveal operational patterns
        (e.g., some teams disproportionately handle certain case types).
        Model stability: Redundant predictors can make models unstable or harder to interpret;
        removing redundancy improves robustness and clarity.
        Rules of thumb for interpretation (context-dependent):
        V < 0.1 ≈ negligible, 0.1–0.3 weak, 0.3–0.5 moderate, > 0.5 strong.
        Important: Cramér’s V is an association strength, not a causal measure and not
        a significance test by itself (the chi-square test provides a p-value; V provides effect size).
        Missing values: pd.crosstab ignores NaNs; decide whether to impute or drop beforehand.
        High-cardinality categories: Many levels can complicate interpretation; consider grouping rare levels.
        Huge sample sizes: Chi-square p-values will be almost always “significant”;
        look at V as an effect size to judge practical significance.
        Ordinal categories: If categories are truly ordered (e.g., severity bands),
        you may also analyse with Spearman’s correlation on numeric codes (with care).
        """
        if not _HAS_SCIPY:  # if scipy not installed, degrade gracefully, don’t crash.
            # Warn the user and return NaN. Thiskeeps CI and quick environments stable even without optional deps.
            warnings.warn("scipy not available: returning NaN for Cramér’s V")
            return float("nan")
        # Build a contingency table of counts for every combination of x category and y category.
        # rows = case_type (LPA/Deputyship/Other), cols = risk_band (Low/Medium/High).
        tbl = pd.crosstab(x, y)  # contingency table
        # Run the chi-square test of independence on that table.
        # The Chi-square test looks at the pattern of observations and will tell us if certain combinations of the categories occur more frequently than we would expect by chance, given the total number of times each category occurred. The chi-square statistic tells you how much difference exists between the observed count in each table cell to the counts you would expect if there were no relationship at all in the population. Thus, low p-values (p< .05) indicate a likely difference between the theoretical population and the collected sample. You can conclude that a relationship exists between the categorical variables.
        # chi2_contingency returns (chi2 statistic, p-value, dof, expected counts).
        # We take index [0]: the statistic.
        # Intuition: larger chi-square ⇒ bigger deviation from independence.
        chi2 = chi2_contingency(tbl)[0]  # chi-squared test statistic
        # Total number of observations in the table, needed to turn chi-square into an effect size.
        n = tbl.to_numpy().sum()  # sample size
        # Compute phi-squared (χ² divided by sample size). This scales the statistic by how much data we have.
        phi2 = chi2 / n  # raw effect size
        # Number of row categories (r) and column categories (k). Needed for bias correction.
        r, k = tbl.shape  # rows, cols
        # Bias correction (Bergsma 2013)
        # These three lines implement a bias correction (Bergsma & Wicher, 2013), which adjusts the effect size when sample sizes are small or category counts are high. phi2corr subtracts a small-sample “expected inflation” term from phi2. rcorr and kcorr are corrected counts of categories used in the denominator to keep the scale proper (so V stays within [0, 1] more reliably).
        phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rcorr = r - (r - 1) ** 2 / (n - 1)
        kcorr = k - (k - 1) ** 2 / (n - 1)
        # max(1e-12, ...) is a safety guard so we never divide by zero (e.g., degenerate tables).The square root puts the metric on the 0–1 scale. Casting to float makes sure you get a plain Python float (handy for printing/logging).
        return float(
            np.sqrt(phi2corr / max(1e-12, min(kcorr - 1, rcorr - 1)))
        )  # guard against div-by-zero

    def redundancy_drop_list(
        self, cols: Optional[List[str]] = None, thresh: float = 0.90
    ) -> List[str]:
        """
        Identify numeric columns to drop because they are highly correlated with others (abs(r) > thresh).
        """
        cols = cols or (self.cfg.numeric_cols or [])
        corr = self.df[cols].corr().abs()  # absolute Pearson by default
        upper = corr.where(
            np.triu(np.ones_like(corr, dtype=bool), k=1)
        )  # upper triangle without diagonal, while masking the rest with NaN.
        return [
            c for c in upper.columns if (upper[c] > thresh).any()
        ]  # columns with any high corr

    # ---------------------------
    # 5) Multicollinearity (VIF)
    # ---------------------------

    def vif_report(self, cols: Optional[List[str]] = None) -> Optional[pd.Series]:
        """
        Multicolinearity detection using Variance Inflation Factor for a set of numeric predictors.
        Returns None if statsmodels is not available.
        Interpreting VIF: VIF = 1 → No correlation with other variables.
        1 < VIF ≤ 5 → Moderate correlation (usually acceptable).
        VIF > 5 or 10 → High multicollinearity; consider removing or transforming the variable.
        Common Pitfall: If you run your original code without add_constant,
        you might get artificially low or high VIF values.
        """
        if not _HAS_STATSMODELS:
            warnings.warn("statsmodels not available: VIF report skipped.")
            return None
        cols = cols or (self.cfg.numeric_cols or [])
        # Removes any rows that contain NaN (missing values) in any of those selected columns
        # By default, dropna() works row-wise (axis=0) and drops rows where at least one value is missing.
        # If you only want to drop rows where all values are missing, you can use:self.df[cols].dropna(how='all')
        # If you want to drop rows with NaN only in specific columns, you can pass subset=cols.

        # X containing only the independent variables (no target column).
        X = self.df[cols].dropna()  # VIF requires no missing values
        # import statsmodels.api as sm
        # It adds an intercept column (a column of 1s) to your dataset X.
        # This is important for regression models in statsmodels,
        # because by default they do not automatically include an intercept term.
        # from statsmodels.stats.outliers_influence import variance_inflation_factor
        # from statsmodels.tools.tools import add_constant
        X = sm.add_constant(
            X
        )  # add constant term, otherwise, VIF values can be misleading
        vif = pd.Series(  # variance_inflation_factor is imported from statsmodels.stats.outliers_influence.
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
            name="VIF",
        )
        return vif.sort_values(ascending=False)  # higher VIF -> more collinear

    # ---------------------------
    # 6) Class imbalance & leakage
    # ---------------------------

    def imbalance_summary(self) -> Optional[Dict[str, float]]:
        """
        Return positive share and counts for a binary target (if present).
        Calculate the positive class share from a Pandas DataFrame and
        then builds a dictionary with counts for each class (n0, n1)
        and the positive share (pos_share).
        """
        if not self.cfg.target_col or self.cfg.target_col not in self.df:
            return None
        # Counting the frequency of each unique value in a column, including NaN values.
        vc = self.df[self.cfg.target_col].value_counts(dropna=False)
        # Safely compute positive share
        # pos_share = self.df[self.cfg.target_col].mean()  # This is already a float scalar
        # Build the result dictionary with safe int casting
        # return {
        #    "n0": int(vc.get(0, 0)),  # Count of class 0
        #    "n1": int(vc.get(1, 0)),  # Count of class 1
        #    "pos_share": round(float(pos_share), 4)  # Ensure float before rounding
        # }
        pos_share = float(self.df[self.cfg.target_col].mean())
        return {
            "n0": int(vc.get(0, 0)),
            "n1": int(vc.get(1, 0)),
            "pos_share": round(pos_share, 4),
        }

    def leakage_scan(self, suspicious_keywords: Iterable[str]) -> List[str]:
        """
        Heuristically scan columns for likely post-treatment/leakage fields
        (e.g., 'post', 'signed', 'decision').
        Catch post-treatment / leakage columns (e.g., anything created
        after allocation/sign-off) before modeling.
        Leakage makes models look unrealistically good.
        Pre-allocation models must not use columns like date_pg_signoff,
        signed_status, legal_decision—these leak future info. This quick scan is a sanity net.
        Example: sus = eda.leakage_scan(["post", "signed", "decision", "pg_signoff"])
        print(sus)  # e.g. ['date_pg_signoff', 'signed_off_flag', 'legal_decision_code']
        """
        # normalise for case-insensitive check
        keys = [
            k.lower() for k in suspicious_keywords
        ]  # 1) Lower-case the keywords so matching is case-insensitive.

        hits = []  # 2) Start an empty list to store suspicious column names.
        for c in self.df.columns:  # 3) Look at every column in the DataFrame.
            low = c.lower()  # 4) Lower-case the column name (case-insensitive compare).
            if any(
                k in low for k in keys
            ):  # 5) If *any* keyword is a substring of the column name…
                hits.append(c)  # 6) …flag it by appending to hits.
        return hits  # 7) Return the list of suspicious columns.

    # ---------------------------
    # 7) Interactions (binned)
    # ---------------------------

    def binned_interaction_rate(
        self, num_col: str, cat_col: str, target: Optional[str] = None, q: int = 5
    ) -> pd.DataFrame:
        """
        Compute target mean across bins of a numeric column and levels of a categorical column.
        Useful to screen interactions (e.g., risk_band × days_to_alloc -> legal_review rate).
        quickly see if a numeric feature and a categorical feature interact to change the
        target rate (e.g., legal review rate varies by risk_band and days_to_alloc bins).
        Example: You can spot patterns like
        “High risk + long days_to_alloc → much higher legal-review rate”,
        justifying an interaction term or different triage rules.
        tab = eda.binned_interaction_rate("days_to_alloc", "risk_band", target="legal_review", q=5)
        print(tab)  # a table of legal_review rate by risk_band (rows) and days_to_alloc bins (columns)
        """

        target = (
            target or self.cfg.target_col
        )  # 1) Use provided target or the configured default target.
        if not target:
            raise ValueError(
                "Target column required for interaction rates."
            )  # 2) Must know which target to average.
        # 3) Keep only rows that have both the numeric and categorical values (target may still be NaN).
        tmp = self.df[[num_col, cat_col, target]].dropna(subset=[num_col, cat_col])
        # 4) Bin the numeric feature into ~q quantile bins (robust to skew).
        tmp["__bin__"] = pd.qcut(
            tmp[num_col],
            q=min(
                q, tmp[num_col].nunique()
            ),  # Do not create more bins than we have unique values.
            duplicates="drop",  #    If some bins would be identical, 'duplicates="drop"' merges them cleanly.
        )
        # 5) For each category×bin cell, compute the mean target (e.g., legal review rate).
        # Pivot to a matrix: rows=category, cols=bins.
        out = tmp.groupby([cat_col, "__bin__"])[target].mean().unstack()
        return out  # 6) Return the matrix for inspection/plotting.

    # ---------------------------
    # 8) Time-series EDA
    # ---------------------------

    def resample_time_series(self, metrics: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
        """
        Resample to daily/weekly, aggregating metrics.
        Example metrics:
            {"backlog": ("backlog", "last"), "inv_mean": ("investigators_on_duty", "mean")}
        Turn event rows into daily/weekly time series KPIs
        (e.g., daily backlog last value, average investigators on duty),
        optionally add a 7-day smoother.
        Clean daily KPIs (backlog, staffing) to plot trends, check seasonality,
        and feed forecasting models (SARIMAX, etc.).
        Example:
        daily = eda.resample_time_series({
        "backlog": ("backlog", "last"),
        "inv_mean": ("investigators_on_duty", "mean"),
        })
        print(daily.tail())
        """
        # 1) Need to know which column is the time index (e.g., 'date_received_opg').
        if not self.cfg.time_index_col:
            raise ValueError("Set time_index_col in EDAConfig to resample.")

        # 2) Create a DateTimeIndex from the configured column, coercing bad values to NaT; sort chronologically.
        ts = self.df.set_index(
            pd.to_datetime(self.df[self.cfg.time_index_col], errors="coerce")
        ).sort_index()

        # 3) Build an aggregation spec, e.g., 'backlog' uses ('backlog', 'last'),
        # 'inv_mean' uses ('investigators_on_duty', 'mean').
        agg_spec = {
            k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in metrics.items()
        }

        # 4) Resample by the rule ('D' for daily, 'W' for weekly) and aggregate using the spec.
        out = ts.resample(self.cfg.resample_rule).agg(
            **agg_spec
        )  # astericks expand the dic to keywords

        if self.cfg.resample_rule.upper() == "D":
            for k in list(metrics.keys()):
                out[f"{k}_7d"] = out[k].rolling(7, min_periods=3).mean()
        # 5) Convenience: if daily, add a 7-day rolling mean per metric (smoother, handles weekends/spikes).

        return out
        # 6) Return the resampled KPI frame (indexed by date).

    def lag_correlations(
        self, s1: pd.Series, s2: pd.Series, lags: Optional[Iterable[int]] = None
    ) -> pd.Series:
        """
        Compute correlation between s1[t] and s2[t - k] for specified lags (default from config).
        quick check if changes in one series lead or follow another
        (e.g., does staffing today correlate with backlog a week later?).
        If corr(backlog, inv_mean shifted by 7) is negative and large in magnitude,
        more staff tends to reduce backlog about a week later.
        It’s not causality proof, but a strong operational hint for modeling and simulation.
        Example:
        corrs = eda.lag_correlations(daily["backlog"], daily["inv_mean"], lags=[1,7,14])
        print(corrs)
        """
        lags = list(
            lags or self.cfg.lag_list
        )  # 1) Use provided lags or defaults (e.g., [1, 7, 14]).
        out = {}  # 2) Result holder: lag name -> correlation number.
        for k in lags:  # 3) For each lag k…
            out[f"lag_{k}"] = float(
                s1.corr(s2.shift(k))
            )  # 4) …compute corr between s1[t] and s2 shifted by k (i.e., s2 at t-k).
        return pd.Series(out)  # 5) Return as a small Series for easy reading/plotting.

    # ---------------------------
    # 9) Survival / interval analysis
    # ---------------------------

    def km_quantiles_by_group(
        self,
        duration: str,
        event: str,
        group: str,
        probs: Iterable[float] = (0.25, 0.5, 0.75),
    ) -> pd.DataFrame:
        """
        Compute Kaplan–Meier quantiles by group if lifelines is installed.
        Fallback: naive quantiles (ignores censoring) with a warning.
        Kaplan–Meier (censor-aware) service-time quantiles (median, P75, P90…)
        by group (e.g., by risk_band).
        Falls back to naive quantiles if lifelines is missing.
        “High-risk median to PG sign-off is 45 days (P90=110) vs
        Low-risk median 28 (P90=70).”
        That directly informs SLAs, case prioritisation, and DES service-time inputs.
        Example:
        km_tab = eda.km_quantiles_by_group("days_to_signoff", "event_signoff", "risk_band", probs=(0.5, 0.8, 0.9))
        """
        res = []  # 1) Collect per-group rows here.
        for g, dfg in self.df.groupby(
            group
        ):  # 2) For each group (e.g., each risk band)…
            if _HAS_LIFELINES:
                km = KaplanMeierFitter()  # 3) Create a KM estimator.
                km.fit(
                    durations=dfg[duration], event_observed=dfg[event]
                )  # 4) Fit with durations + censor flags.
                row = {"group": g}  # 5) Start a result row with the group name.
                for p in probs:
                    row[f"q{int(p*100)}"] = float(
                        km.quantile(p)
                    )  # 6) Read censor-aware quantiles (e.g., q50=median).
                res.append(row)  # 7) Save this group’s row.
            else:
                warnings.warn(
                    "lifelines not available; using naive quantiles (censoring ignored)."
                )
                row = {"group": g}
                for p in probs:
                    row[f"q{int(p*100)}"] = float(
                        dfg[duration].quantile(p)
                    )  # 8) Fallback: naive quantiles.
                res.append(row)
        return (
            pd.DataFrame(res).sort_values("group").reset_index(drop=True)
        )  # 9) Return a tidy table.

    # ---------------------------
    # 10) KPI tables for stakeholders
    # ---------------------------

    def monthly_kpis(self) -> pd.DataFrame:
        """
        Produce a stakeholder-friendly monthly KPI table by team (if team_col set):
        - backlog (last of month),
        - median days_to_alloc,
        - legal review rate.
        Produce a stakeholder-ready monthly table per team: backlog,
        median time to allocation, and legal-review rate.
        It turns raw rows into an executive KPI view over time:
        “Team B’s median days to allocation rose in Q3 while
        backlog last-of-month climbed; legal review rate stable.”
        Perfect for dashboards and monthly packs.
        Example:
        kpis = eda.monthly_kpis()
        print(kpis.head())
        return out.sort_values([self.cfg.team_col, "__month"])
        """
        if not self.cfg.team_col:
            raise ValueError("team_col must be configured for monthly KPIs.")
        # 1) Need the team column to group by.

        month = self.df[self.cfg.date_received].dt.to_period("M").dt.to_timestamp()
        # 2) Convert received date to a month period, then back to a timestamp (first day of that month).

        tmp = self.df.assign(__month=month)
        # 3) Add a helper column '__month' for grouping.

        out = (
            tmp.groupby([self.cfg.team_col, "__month"])
            .agg(
                backlog=(
                    ("backlog", "last") if "backlog" in tmp.columns else ("id", "count")
                ),
                # 4) Backlog: the last value in each month (if available). Otherwise, fallback to a count.
                median_alloc=("days_to_alloc", "median"),
                # 5) Median time-to-allocation (robust to skew).
                legal_rate=(
                    (self.cfg.target_col, "mean")
                    if self.cfg.target_col
                    else (self.cfg.id_col, "count")
                ),
                # 6) If target exists (e.g., legal_review), take mean (i.e., rate). Else, just count.
            )
            .reset_index()
        )
        return out.sort_values([self.cfg.team_col, "__month"])
        # 7) Return a tidy table sorted by team and month.


__all__ = ["EDAConfig", "OPGInvestigationEDA"]
