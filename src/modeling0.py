from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .features import build_design_matrix, derive_case_level_features

def fit_backlog_glm(daily_backlog: pd.DataFrame, staffing: pd.DataFrame):
    df = daily_backlog.merge(staffing, on="date", how="left").fillna({"investigators_on_duty": 0, "n_allocations": 0})
    df["t"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["date"]).min()).dt.days
    X = sm.add_constant(df[["investigators_on_duty", "n_allocations", "t"]])
    y = df["backlog"].astype(int)
    poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    overdisp = (y.var() - y.mean()) / y.mean() if y.mean() > 0 else 0
    model = poisson
    if overdisp > 1.0:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    return model, df

def forecast_backlog(model, df_template: pd.DataFrame, days: int = 90):
    last_date = pd.to_datetime(df_template["date"]).max()
    future_days = pd.date_range(last_date + pd.Timedelta(days=1), periods=days, freq="D")
    med_staff = df_template["investigators_on_duty"].median()
    med_alloc = df_template["n_allocations"].median()
    base_t0 = (pd.to_datetime(df_template["date"]) - pd.to_datetime(df_template["date"]).min()).dt.days.max()
    Xf = pd.DataFrame({
        "const": 1.0,
        "investigators_on_duty": med_staff,
        "n_allocations": med_alloc,
        "t": base_t0 + np.arange(1, days+1),
    })
    mu = model.predict(Xf)
    out = pd.DataFrame({"date": future_days, "pred_backlog": np.asarray(mu)})
    return out

def fit_legal_review_classifier(df: pd.DataFrame):
    df = derive_case_level_features(df)
    X_cols = ["team", "case_type", "risk", "reallocation", "weighting", "days_to_alloc"]
    X = df[X_cols]
    y = df["needs_legal_review"].astype(int)
    pre = build_design_matrix(df[X_cols + ["needs_legal_review"]])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200, solver="liblinear"))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    yprob = pipe.predict_proba(Xte)[:, 1]
    return pipe, (yte, yprob)

def fit_survival_model(df: pd.DataFrame):
    sdf = df[["days_to_pg_signoff", "event_pg_signoff", "case_type", "risk", "weighting"]].dropna(subset=["days_to_pg_signoff"])
    if sdf.empty:
        raise ValueError("No survival data found.")
    sdf = sdf.rename(columns={"days_to_pg_signoff": "duration", "event_pg_signoff": "event"})
    sdf = pd.get_dummies(sdf, columns=["case_type", "risk"], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(sdf, duration_col="duration", event_col="event", show_progress=False)
    return cph

def scenario_apply_staffing(df_template: pd.DataFrame, delta_investigators: int, model):
    df_scn = df_template.copy()
    df_scn["investigators_on_duty"] = np.maximum(0, df_scn["investigators_on_duty"] + delta_investigators)
    X = sm.add_constant(df_scn[["investigators_on_duty", "n_allocations", "t"]])
    pred = model.predict(X)
    return pred
