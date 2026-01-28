# src/g7_assessment/modeling.py
# Legal review propensity (elastic-net); backlog GLM (Poisson/NegBin) with scenarios; survival for intervals.

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .features import build_preprocessor


# ---------- LEGAL REVIEW: elastic-net logistic ----------
def fit_legal_review_enet(
    df: pd.DataFrame, y_name: str = "needs_legal_review", random_state: int = 42
):
    """Fit an elastic-net logistic with leak-safe preprocessing (OHE + target encoding if present)."""
    df = df.dropna(subset=[y_name]).copy()
    y = df[y_name].astype(int)
    X = df.drop(columns=[y_name])

    pre = build_preprocessor(
        df, y_name=y_name
    )  # impute, scale, OHE, TE (if high-card present)
    clf = LogisticRegressionCV(  # CV over C and l1_ratio for elastic-net
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.5, 0.9],
        Cs=20,
        cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
        scoring="roc_auc",
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    pipe.fit(Xtr, ytr)

    yprob = pipe.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, yprob)
    ap = average_precision_score(yte, yprob)
    brier = brier_score_loss(yte, yprob)
    frac_pos, mean_pred = calibration_curve(yte, yprob, n_bins=10, strategy="uniform")

    metrics = {
        "auc": float(auc),
        "average_precision": float(ap),
        "brier": float(brier),
        "calibration_frac_pos": frac_pos,
        "calibration_mean_pred": mean_pred,
    }
    return pipe, metrics


# ---------- BACKLOG DRIVERS: GLM with over-dispersion check ----------
def fit_backlog_glm(daily_backlog: pd.DataFrame, staffing: pd.DataFrame):
    """Poisson GLM with investigators_on_duty; fallback to NegBin on over-dispersion."""
    df = daily_backlog.merge(staffing, on="date", how="left").fillna(
        {"investigators_on_duty": 0, "n_allocations": 0}
    )
    df["t"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["date"]).min()).dt.days
    X = sm.add_constant(df[["investigators_on_duty", "n_allocations", "t"]])
    y = df["backlog"].astype(int)

    poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    overdisp = (y.var() - y.mean()) / y.mean() if y.mean() > 0 else 0
    model = (
        poisson
        if overdisp <= 1.0
        else sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    )
    return model, df


def scenario_staffing_effect(model, design_df: pd.DataFrame, delta: int):
    """Counterfactual: step-change investigators_on_duty by `delta`; return predicted series & mean effect."""
    df = design_df.copy()
    df["investigators_on_duty"] = np.maximum(0, df["investigators_on_duty"] + delta)
    X = sm.add_constant(df[["investigators_on_duty", "n_allocations", "t"]])
    return model.predict(X)


# ---------- SURVIVAL: time-to-PG-signoff ----------
def fit_survival_pg(df: pd.DataFrame):
    """Cox PH on time to PG signoff; returns lifelines model (with OHE for cats)."""
    sdf = (
        df[["days_to_pg_signoff", "event_pg_signoff", "case_type", "risk", "weighting"]]
        .dropna(subset=["days_to_pg_signoff"])
        .copy()
    )
    sdf = sdf.rename(
        columns={"days_to_pg_signoff": "duration", "event_pg_signoff": "event"}
    )
    sdf = pd.get_dummies(sdf, columns=["case_type", "risk"], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(sdf, duration_col="duration", event_col="event", show_progress=False)
    return cph
