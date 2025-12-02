#!/usr/bin/env python3
"""
analyzers.py

Extended module providing object‑oriented classes for:
 - Exploratory Data Analysis (EDA)
 - Feature Engineering
 - Model Training & Evaluation
 - Statistical Analysis (Correlation, MR, Causality)
     - Correlation (Pearson, Spearman)
     - Mendelian Randomization (2‑stage least squares)
     - Causality tests (Conditional Mutual Information, Transfer Entropy, Granger Causality)

Usage (from shell):
    python src/analyzers.py --help

Or import in a notebook:
    from src.analyzers import (
        EDAAnalyzer,
        FeatureEngineer,
        ModelTrainer,
        CorrelationAnalyzer,
        RandomizationAnalyzer,
        CausalityAnalyzer,
    )
"""


import numpy as np                                  # Numerical operations
import pandas as pd                                 # DataFrame handling
from sklearn.preprocessing import KBinsDiscretizer   # Discretize continuous data
from statsmodels.tsa.stattools import grangercausalitytests  # Granger causality tests

class BaseAnalyzer:
    """
    Base class for all analyzers.
    Holds a pandas DataFrame and provides common functionality.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a DataFrame.
        :param df: DataFrame with data to analyze.
        """
        self.df = df.copy()


class EDAAnalyzer(BaseAnalyzer):
    """
    Exploratory Data Analysis: summary statistics, missing values,
    correlation matrix, and basic visualizations.
    """
    def summary(self) -> pd.DataFrame:
        """
        Return descriptive statistics for numeric columns.
        """
        return self.df.describe(include='all')

    def missing_summary(self) -> pd.DataFrame:
        """
        Return count and percentage of missing values per column.
        """
        missing = self.df.isna().sum()
        pct = 100 * missing / len(self.df)
        return pd.DataFrame({'missing_count': missing, 'missing_pct': pct})

    def correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix for numeric features.
        :param method: 'pearson', 'spearman', or 'kendall'
        """
        return self.df.select_dtypes(include=[np.number]).corr(method=method)


class FeatureEngineer(BaseAnalyzer):
    """
    Automated feature engineering: encoding, scaling, and splitting.
    """
    def get_features_and_target(self, target_col: str):
        """
        Separate DataFrame into feature matrix X and target vector y.
        """
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        return X, y

    def one_hot_encode(self, X: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Return DataFrame with one-hot encoding for specified categorical columns.
        """
        return pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    def scale_numeric(self, X: pd.DataFrame, numeric_cols: list):
        """
        Scale numeric columns to zero mean and unit variance.
        """
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        return X

    def train_test_split(self, X: pd.DataFrame, y: pd.Series,
                         test_size: float = 0.2, random_state: int = 42):
        """
        Wrapper around sklearn.model_selection.train_test_split.
        """
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


class ModelTrainer:
    """
    Model training, evaluation, and hyperparameter tuning.
    """
    def __init__(self, estimator, param_grid: dict = None):
        """
        :param estimator: an sklearn-like estimator or Pipeline
        :param param_grid: dict for GridSearchCV hyperparameters
        """
        self.estimator = estimator
        self.param_grid = param_grid

    def cross_validate(self, X, y, cv: int = 5, scoring: str = 'roc_auc'):
        """
        Perform cross-validation and return mean score.
        """
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.estimator, X, y, cv=cv, scoring=scoring)
        return scores

    def fit(self, X_train, y_train):
        """
        Fit the estimator on training data.
        If param_grid provided, runs a grid search.
        """
        if self.param_grid:
            from sklearn.model_selection import GridSearchCV
            grid = GridSearchCV(self.estimator, self.param_grid,
                                cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            self.best_estimator_ = grid.best_estimator_
            return grid
        else:
            self.estimator.fit(X_train, y_train)
            return self.estimator

    def evaluate(self, X_test, y_test):
        """
        Evaluate the fitted model on test data. Returns a dict of metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            classification_report,
            confusion_matrix,
        )
        est = getattr(self, 'best_estimator_', self.estimator)
        y_pred = est.predict(X_test)
        y_proba = est.predict_proba(X_test)[:, 1] if hasattr(est, "predict_proba") else None

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        return results


# Classes for Correlation, Randomization, Causality follow...

class CorrelationAnalyser(BaseAnalyser):
    """
    Analyser for computing correlations between two variables in the DataFrame.
    Inherits from BaseAnalyser.
    """
    def pearson(self, x: str, y: str) -> float:
        """
        Compute Pearson correlation coefficient between columns x and y.

        :param x: name of the first numeric column
        :param y: name of the second numeric column
        :return: Pearson r (float)
        """
        # Select the two columns and compute the correlation matrix,
        # then extract the off-diagonal element at (0,1)
        return self.df[[x, y]].corr(method='pearson').iloc[0, 1]

    def spearman(self, x: str, y: str) -> float:
        """
        Compute Spearman rank correlation between columns x and y.

        :param x: name of the first column
        :param y: name of the second column
        :return: Spearman rho (float)
        """
        return self.df[[x, y]].corr(method='spearman').iloc[0, 1]


class RandomizationAnalyser(BaseAnalyser):
    """
    Analyser for Mendelian (instrumental-variable) randomization.
    Implements a two-stage least squares procedure.
    """
    def mendelian_randomization(self, exposure: str, outcome: str, instrument: str):
        """
        Perform two-stage least squares:
         1) Regress exposure on instrument
         2) Regress outcome on predicted exposure from stage 1

        :param exposure: name of the exposure column
        :param outcome: name of the outcome column
        :param instrument: name of the genetic instrument column
        :return: statsmodels RegressionResults of stage‑2 regression
        """
        import statsmodels.api as sm

        # Drop rows with missing data in any of the three columns
        data = self.df.dropna(subset=[exposure, outcome, instrument])

        # Stage 1: fit exposure ~ instrument + intercept
        inst = sm.add_constant(data[instrument])       # add constant term
        model1 = sm.OLS(data[exposure], inst).fit()    # OLS regression
        exp_hat = model1.predict(inst)                 # predicted exposure

        # Stage 2: fit outcome ~ predicted exposure + intercept
        inst2 = sm.add_constant(exp_hat)               
        model2 = sm.OLS(data[outcome], inst2).fit()
        return model2  # return fitted model object


class CausalityAnalyser(BaseAnalyser):
    """
    Analyser for various causality metrics:
     - Conditional Mutual Information (CMI)
     - Transfer Entropy (TE)
     - Granger Causality (GC)
    """
    def conditional_mutual_information(self, x: str, y: str, z: str, n_bins: int = 10) -> float:
        """
        Estimate I(X; Y | Z) by discretizing X, Y, Z into bins.

        :param x: name of variable X
        :param y: name of variable Y
        :param z: name of conditioning variable Z
        :param n_bins: number of bins for discretization
        :return: estimated conditional mutual information
        """
        # Select and drop rows with missing values
        data = self.df[[x, y, z]].dropna()

        # Discretize each variable into integer bins [0..n_bins-1]
        disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        Xd, Yd, Zd = disc.fit_transform(data).astype(int).T
        n = len(Xd)

        # Count joint and marginal frequencies
        from collections import Counter
        p_xyz = Counter(zip(Xd, Yd, Zd))
        p_xz  = Counter(zip(Xd, Zd))
        p_yz  = Counter(zip(Yd, Zd))
        p_z   = Counter(Zd)

        # Compute CMI sum_{x,y,z} p(x,y,z) * log( (p(x,y,z)*p(z)) / (p(x,z)*p(y,z)) )
        cmi = 0.0
        for (xi, yi, zi), count in p_xyz.items():
            p_xyz_val = count / n
            p_xz_val  = p_xz[(xi, zi)] / n
            p_yz_val  = p_yz[(yi, zi)] / n
            p_z_val   = p_z[zi] / n
            cmi += p_xyz_val * np.log((p_xyz_val * p_z_val) / (p_xz_val * p_yz_val) + 1e-12)
        return cmi

    def transfer_entropy(self, source: str, target: str, lag: int = 1, n_bins: int = 10) -> float:
        """
        Estimate Transfer Entropy TE(source→target) ≈ I(source_{t-lag}; target_t | target_{t-lag})

        :param source: name of source time series
        :param target: name of target time series
        :param lag: lag order
        :param n_bins: number of bins for discretization
        :return: estimated transfer entropy
        """
        # Prepare lagged variables
        df = self.df[[source, target]].dropna()
        df['target_lag'] = df[target].shift(lag)
        df['source_lag'] = df[source].shift(lag)
        df = df.dropna()

        # Compute conditional mutual information for TE
        return self.conditional_mutual_information('source_lag', target, 'target_lag', n_bins=n_bins)

    def granger_causality(self, source: str, target: str, maxlag: int = 1, **kwargs):
        """
        Perform Granger causality test: does `source` help predict `target`?

        :param source: name of source series
        :param target: name of target series
        :param maxlag: maximum lag to test
        :return: dictionary of test results per lag
        """
        data = self.df[[target, source]].dropna()
        # Format: array [[target, source], ...]
        arr = data.values
        results = grangercausalitytests(arr, maxlag=maxlag, verbose=False)
        return results

if __name__ == "__main__":
    print("This module defines EDAAnalyzer, FeatureEngineer, ModelTrainer, "
          "CorrelationAnalyzer, RandomizationAnalyzer, CausalityAnalyzer")

from src.analyzers import (
    EDAAnalyzer, FeatureEngineer, ModelTrainer,
    CorrelationAnalyzer, RandomizationAnalyzer, CausalityAnalyzer
)

# 1) EDA
eda = EDAAnalyzer(df_sample)
print(eda.summary())
print(eda.missing_summary())

# 2) Features
fe = FeatureEngineer(df_sample)
X, y = fe.get_features_and_target('outcome')
X = fe.one_hot_encode(X, ['instrument'])
X = fe.scale_numeric(X, ['exposure'])
X_train, X_test, y_train, y_test = fe.train_test_split(X, y)

# 3) Modeling
from sklearn.ensemble import RandomForestClassifier
mt = ModelTrainer(RandomForestClassifier(random_state=42),
                  param_grid={'n_estimators': [50,100], 'max_depth': [3,5]})
print("CV scores:", mt.cross_validate(X_train, y_train))
grid = mt.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Test eval:", mt.evaluate(X_test, y_test))

# 4) Stats
corr = CorrelationAnalyzer(df_sample)
print("Pearson X/Y:", corr.pearson('X','Y'))
    
    # Example CLI: python src/analysers.py --input data.csv --mode pearson --x col1 --y col2
    import argparse

    parser = argparse.ArgumentParser(description="Run statistical analysers on a CSV file")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input CSV file")
    parser.add_argument("--mode", "-m", required=True,
                        choices=["pearson", "spearman", "mr", "cmi", "te", "gc"],
                        help="Analysis mode")
    parser.add_argument("--x", help="Column X (for correlation, CMI, TE, GC)")
    parser.add_argument("--y", help="Column Y (for correlation, CMI, TE, GC)")
    parser.add_argument("--z", help="Column Z (for CMI)")
    parser.add_argument("--instrument", help="Instrument column (for MR)")
    parser.add_argument("--exposure", help="Exposure column (for MR)")
    parser.add_argument("--outcome", help="Outcome column (for MR)")
    parser.add_argument("--lag", type=int, default=1, help="Lag for TE/GC")
    parser.add_argument("--bins", type=int, default=10, help="Bins for discretization")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    if args.mode in ["pearson", "spearman"]:
        corr = CorrelationAnalyser(df)
        func = corr.pearson if args.mode == "pearson" else corr.spearman
        print(f"{args.mode}({args.x}, {args.y}) =", func(args.x, args.y))

    elif args.mode == "mr":
        rnd = RandomizationAnalyser(df)
        model = rnd.mendelian_randomization(args.exposure, args.outcome, args.instrument)
        print(model.summary())

    elif args.mode == "cmi":
        caus = CausalityAnalyser(df)
        print("CMI:", caus.conditional_mutual_information(args.x, args.y, args.z, n_bins=args.bins))

    elif args.mode == "te":
        caus = CausalityAnalyser(df)
        print("TE:", caus.transfer_entropy(args.x, args.y, lag=args.lag, n_bins=args.bins))

    elif args.mode == "gc":
        caus = CausalityAnalyser(df)
        res = caus.granger_causality(args.x, args.y, maxlag=args.lag)
        print("Granger Causality results:", res)



# :
# df = pd.read_csv('your_data.csv')
# corr = CorrelationAnalyser(df)
# print("Pearson r:", corr.pearson('X', 'Y'))
# rnd = RandomizationAnalyser(df)
# mr_model = rnd.mendelian_randomization('exposure', 'outcome', 'instrument')
# print(mr_model.summary())
# caus = CausalityAnalyser(df)
# print("Conditional MI:", caus.conditional_mutual_information('X','Y','Z'))
# print("Transfer Entropy:", caus.transfer_entropy('X','Y'))
# print("Granger Causality:", caus.granger_causality('X','Y', maxlag=3))
