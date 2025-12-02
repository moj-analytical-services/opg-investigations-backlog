# ts_gen.py
import numpy as np
import pandas as pd
from typing import Optional, List
from statsmodels.tsa.seasonal import STL


def ar1_noise(n, phi=0.5, sigma=1.0, seed=None):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def seasonal_trend_from_real(y: pd.Series, period: int):
    """Estimate level, trend per step, and seasonality amplitude via STL decomposition."""
    res = STL(y, period=period, robust=True).fit()
    trend = res.trend
    level = float(trend.dropna().iloc[0])
    trend_per_step = float(
        (trend.dropna().iloc[-1] - trend.dropna().iloc[0])
        / max(1, len(trend.dropna()) - 1)
    )
    season_amp = float(res.seasonal.abs().median() * 1.5)
    return level, trend_per_step, season_amp


def synth_count_series(
    idx,
    level,
    trend_per_step,
    season_amp,
    season_period,
    ar_phi=0.3,
    noise_sigma=1.0,
    seed=None,
):
    t = np.arange(len(idx))
    season = season_amp * np.sin(2 * np.pi * t / season_period)
    ar = ar1_noise(len(idx), phi=ar_phi, sigma=noise_sigma, seed=seed)
    y = np.maximum(0, level + trend_per_step * t + season + ar).round().astype(int)
    return pd.Series(y, index=idx)


def aggregate_counts(
    df: pd.DataFrame, date_col: str, group_cols: Optional[List[str]], freq: str = "MS"
):
    dt = pd.to_datetime(df[date_col])
    df = df.copy()
    df[date_col] = dt
    if group_cols:
        grp = (
            df.groupby([pd.Grouper(key=date_col, freq=freq)] + group_cols)
            .size()
            .rename("count")
        )
        return grp.reset_index()
    else:
        grp = df.groupby(pd.Grouper(key=date_col, freq=freq)).size().rename("count")
        return grp.reset_index()
