# src/timeseries.py
# SARIMAX forecasting with exogenous drivers (staffing, case mix)

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_backlog_sarimax(hist_df: pd.DataFrame, exog_cols=("investigators_on_duty","n_allocations"), order=(1,1,1), seasonal_order=(1,0,1,7)):
    """Fit SARIMAX to daily backlog with weekly seasonality; returns fitted model and design."""
    df = hist_df.sort_values("date").set_index("date").copy()
    exog = df[list(exog_cols)].fillna(0)
    endog = df["backlog"].astype(float)
    m = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return m, df

def forecast_sarimax(model, df_design, days=90):
    last = df_design.index.max()
    fut = pd.date_range(last + pd.Timedelta(days=1), periods=days, freq="D")
    # Hold exogenous to recent median (simple, stable approach; plug scenarios here if needed)
    med = df_design.median(numeric_only=True)
    exog_fut = np.tile(med.values, (days,1))
    pred = model.get_forecast(steps=days, exog=exog_fut)
    out = pd.DataFrame({"date": fut, "pred_backlog": pred.predicted_mean.values})
    return out
