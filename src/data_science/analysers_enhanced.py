#!/usr/bin/env python3
"""
analysers_enhanced.py

An extension of src/analysers.py to include a HeatstrokeAnalyser that:
  - Loads and preprocesses the heatstroke dataset
  - Computes correlation and causation metrics in a loop for each feature
  - Visualises results via heatmap, scatter plots, and a DAG (Directed Acyclic Graph)
  - Identifies and interprets the most influential risk factors

Requires:
  - pandas, numpy, matplotlib, scipy, graphviz, statsmodels
  - Existing analysers: CorrelationAnalyser, CausalityAnalyser from src/analysers.py

Usage:
    from analyzers_enhanced import HeatstrokeAnalyser
    hsa = HeatstrokeAnalyser(csv_path)
    hsa.run_full_analysis()
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from scipy.stats import pointbiserialr, chi2_contingency
from src.analysers import (
    CausalityAnalyser,
)  # reuse existing analysers


class HeatstrokeAnalyser:
    """
    Class to analyse risk factors for heatstroke death.
    Methods:
      - load_and_clean: load CSV, encode target, compute BMI, drop NaNs
      - test_correlations: loop numeric and categorical features to compute
        point-biserial (numeric) or Cramér's V (categorical) vs binary death outcome
      - test_causation: use Conditional Mutual Information for all features
      - visualize_heatmap: plot correlation matrix as heatmap
      - visualize_scatter: scatter plot numeric features against outcome probability
      - build_dag: create a simple DAG of top causal features
      - run_full_analysis: orchestrate all steps and interpret results
    """

    def __init__(self, csv_path: str):
        """
        Initialize with path to heatstroke dataset CSV.
        """
        self.csv_path = csv_path
        self.df = None
        self.results = {"correlation": {}, "causation": {}}

    def load_and_clean(self):
        """
        Load data, encode target, compute BMI, drop missing or infinite values.
        """
        df = pd.read_csv(self.csv_path)
        # Encode binary target
        df["Death"] = df["Death From Heatstroke"].astype(int)
        # Drop missing values in critical columns
        df = df.dropna(subset=["Age", "Height", "Weight", "Death"])
        # Compute BMI
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
        # Remove infinities and further NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        self.df = df.reset_index(drop=True)

    @staticmethod
    def cramers_v(x, y) -> float:
        """
        Compute Cramér's V for two categorical variables.
        """
        confusion = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion)[0]
        n = confusion.sum().sum()
        k = min(confusion.shape)
        return np.sqrt(chi2 / (n * (k - 1) + 1e-12))

    def test_correlations(self):
        """
        For each feature, compute correlation with Death:
          - Numeric: point-biserial
          - Categorical: Cramér's V
        Store absolute values in self.results['correlation'].
        """
        df = self.df
        target = "Death"
        for col in df.columns:
            if col == target:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                # point biserial for numeric vs binary
                corr, _ = pointbiserialr(series, df[target])
            else:
                # categorical
                corr = self.cramers_v(series, df[target])
            self.results["correlation"][col] = abs(corr)

    def test_causation(self, bins=10):
        """
        For each feature, estimate conditional mutual information I(feature; Death | feature_lag)
        using CausalityAnalyser. Store in self.results['causation'].
        """
        ca = CausalityAnalyser(self.df)
        for col in self.df.columns:
            if col == "Death":
                continue
            try:
                # use the feature itself as instrument via small lag (non-time series approx)
                cmi = ca.conditional_mutual_information(col, "Death", col, n_bins=bins)
            except Exception:
                cmi = np.nan
            self.results["causation"][col] = cmi

    def visualize_heatmap(self):
        """
        Plot a heatmap of the correlation matrix among top features.
        """
        # build full correlation matrix for numeric cols
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[num_cols].corr()
        plt.figure(figsize=(8, 6))
        plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Pearson r")
        plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
        plt.yticks(range(len(num_cols)), num_cols)
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def visualize_scatter(self):
        """
        For top numeric features by correlation, scatter vs Death probability.
        """
        # rank numeric by corr
        corr_sorted = {
            k: v
            for k, v in sorted(
                self.results["correlation"].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        top_nums = [
            k for k in corr_sorted if pd.api.types.is_numeric_dtype(self.df[k])
        ][:3]
        for col in top_nums:
            plt.figure()
            plt.scatter(self.df[col], self.df["Death"], alpha=0.3)
            plt.xlabel(col)
            plt.ylabel("Death (0/1)")
            plt.title(f"Scatter of {col} vs Heatstroke Death")
            plt.show()

    def build_dag(self, top_k=5):
        """
        Construct a simple DAG of top_k causal features pointing to Death.
        """
        # pick top causation
        sorted_causal = sorted(
            self.results["causation"].items(),
            key=lambda item: item[1] if not np.isnan(item[1]) else -1,
            reverse=True,
        )
        top_feats = [feat for feat, _ in sorted_causal[:top_k]]
        dot = graphviz.Digraph(comment="Heatstroke DAG")
        # add nodes
        for feat in top_feats:
            dot.node(feat, feat)
        dot.node("Death", "Death")
        # add edges
        for feat in top_feats:
            dot.edge(feat, "Death")
        display(dot)

    def run_full_analysis(self):
        """
        Orchestrate loading, testing, visualizing, DAG building, and interpretation.
        """
        print("Loading and cleaning data...")
        self.load_and_clean()
        print("Testing correlations...")
        self.test_correlations()
        print("Testing causation...")
        self.test_causation()
        print("Correlation results (top 5):")
        for feat, val in sorted(
            self.results["correlation"].items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"  {feat}: {val:.3f}")
        print("Causation results (top 5):")
        for feat, val in sorted(
            self.results["causation"].items(),
            key=lambda x: (x[1] if not np.isnan(x[1]) else -1),
            reverse=True,
        )[:5]:
            print(f"  {feat}: {val:.3f}")
        print("Generating heatmap...")
        self.visualize_heatmap()
        print("Generating scatter plots...")
        self.visualize_scatter()
        print("Building DAG...")
        self.build_dag()
        print(
            "Analysis complete. Interpret the printed results and visualizations to identify the most influential factors."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heatstroke risk factor analysis")
    parser.add_argument("csv_path", help="Path to G7 summer dataset CSV")
    args = parser.parse_args()
    hsa = HeatstrokeAnalyser(args.csv_path)
    hsa.run_full_analysis()
