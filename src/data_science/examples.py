# examples.py
# Minimal runnable examples on synthetic data showing how to call the utilities.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import train_test_split_time
from ts_forecasting import (
    plot_acf_pacf,
    holt_winters_forecast,
    sarima_forecast,
    naive_forecast,
)
from metrics import regression_errors
from dimensionality import pca_fit_transform, plot_explained_variance

# --- Synthetic time-series ---
np.random.seed(42)
date_idx = pd.date_range("2018-01-01", periods=96, freq="MS")
trend = np.linspace(0, 20, len(date_idx))
season = 10 + 3 * np.sin(2 * np.pi * date_idx.month / 12)
noise = np.random.normal(0, 1.5, len(date_idx))
y = trend + season + noise
s = pd.Series(y, index=date_idx)

# Train/test split
train, test = train_test_split_time(s, test_size=12)

# ACF/PACF plots
plot_acf_pacf(train, lags=24)

# Holt-Winters forecast
fit_hw, fc_hw = holt_winters_forecast(
    train, seasonal="add", seasonal_periods=12, horizon=len(test)
)

# SARIMA forecast
fit_sa, mean_sa, ci_sa = sarima_forecast(
    train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), horizon=len(test)
)

# Naive seasonal forecast
fc_naive = naive_forecast(train, horizon=len(test), seasonal_periods=12)

# Errors
errs_hw = regression_errors(test.values, fc_hw.values)
errs_sa = regression_errors(test.values, mean_sa.values)
errs_nv = regression_errors(test.values, fc_naive.values)

print("HW errors:", errs_hw)
print("SARIMA errors:", errs_sa)
print("Naive errors:", errs_nv)

# --- PCA example ---
X = np.column_stack(
    [np.random.normal(size=200), np.random.normal(size=200), np.random.normal(size=200)]
)
pca, Z = pca_fit_transform(X, n_components=2, scale=True)
plot_explained_variance(pca)
plt.show()
