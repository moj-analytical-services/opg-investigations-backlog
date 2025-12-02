# ts_forecasting.py
# Naive, Holt-Winters (ETS), SARIMA; ACF/PACF; Ljung-Box; unit-root tests.

import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


def plot_acf_pacf(series: pd.Series, lags: int = 24):
    fig1 = plot_acf(series.dropna(), lags=lags)
    fig1.tight_layout()
    fig2 = plot_pacf(series.dropna(), lags=lags, method="ywm")
    fig2.tight_layout()
    return fig1, fig2


def naive_forecast(
    series: pd.Series, horizon: int = 1, seasonal_periods: int = None
) -> pd.Series:
    if seasonal_periods:
        return series.shift(seasonal_periods).dropna().iloc[-horizon:]
    else:
        return pd.Series(
            [series.iloc[-1]] * horizon,
            index=pd.date_range(
                start=series.index[-1], periods=horizon + 1, freq=series.index.freq
            )[1:],
        )


def holt_winters_forecast(
    series: pd.Series,
    seasonal: str = "add",
    seasonal_periods: int = 12,
    horizon: int = 12,
):
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit = model.fit()
    fcst = fit.forecast(horizon)
    return fit, fcst


def sarima_forecast(
    series: pd.Series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), horizon: int = 12
):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fcst = fit.get_forecast(steps=horizon)
    mean = fcst.predicted_mean
    ci = fcst.conf_int()
    return fit, mean, ci


def ljung_box_residuals(residuals: pd.Series, lags: int = 24):
    lb = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
    return lb


def adf_test(series: pd.Series):
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "ADF_stat": result[0],
        "pvalue": result[1],
        "lags": result[2],
        "nobs": result[3],
    }


def stl_decompose(series: pd.Series, seasonal: int = 7):
    from statsmodels.tsa.seasonal import STL

    res = STL(series, period=seasonal).fit()
    return res
