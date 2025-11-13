# src/modeling.py
# Purpose: Models including advanced logistic pipeline for legal-review propensity
#          using an elastic-net regularised logistic regression with CV.

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .features import build_preprocessor

def fit_legal_review_logit_enet(
    df: pd.DataFrame,
    y_name: str = "needs_legal_review",
    random_state: int = 42,
):
    """
    Fit an elastic-net logistic regression on engineered features using a leak-safe preprocessor.

    Steps
    -----
    1) Build ColumnTransformer with imputation, scaling, OHE, and K-Fold target encoding.
    2) Split data (stratified) into train/test.
    3) Fit LogisticRegressionCV with elastic-net penalty (solver='saga').
    4) Return fitted pipeline and metrics dict (AUC, AP, Brier, calibration).

    Returns
    -------
    pipe : sklearn Pipeline
        Preprocessor + classifier combined, ready for .predict_proba().
    metrics : dict
        AUC, average precision, brier score, and calibration arrays.
    X_test, y_test : pd.DataFrame, pd.Series
        Hold-out split for downstream plotting/reporting.
    """
    # Drop rows without target
    df = df.dropna(subset=[y_name]).copy()
    y = df[y_name].astype(int)
    X = df.drop(columns=[y_name])

    # Build leak-safe preprocessor (ColumnTransformer). It will receive y during fit().
    pre = build_preprocessor(df=df, y_name=y_name)

    # Elastic-Net logistic with CV: balances sparsity (L1) and stability (L2)
    logit = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.5, 0.9],          # explore L1/L2 mixes
        Cs=20,                               # inverse regularisation strength grid
        cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
        scoring="roc_auc",
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",             # robust when class skew exists
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", logit),
    ])

    # Hold-out split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Fit end-to-end (ColumnTransformer gets both X_tr and y_tr)
    pipe.fit(X_tr, y_tr)

    # Evaluate on the hold-out set
    y_prob = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    ap = average_precision_score(y_te, y_prob)
    brier = brier_score_loss(y_te, y_prob)
    frac_pos, mean_pred = calibration_curve(y_te, y_prob, n_bins=10, strategy="uniform")

    metrics = {
        "auc": float(auc),
        "average_precision": float(ap),
        "brier": float(brier),
        "calibration_frac_pos": frac_pos,
        "calibration_mean_pred": mean_pred,
    }
    return pipe, metrics, X_te, y_te

def save_calibration_plot(mean_pred, frac_pos, out_png: Path):
    """
    Save a simple calibration curve plot.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0,1], [0,1], "--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)

def export_model_and_report(pipe, metrics: dict, out_dir: Path):
    """
    Persist the fitted pipeline and key metrics for reporting.
    """
    from joblib import dump
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(pipe, out_dir / "legal_review_enet.joblib")
    pd.Series(metrics).to_csv(out_dir / "legal_review_enet_metrics.csv")
