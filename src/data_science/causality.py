# causality.py
# Granger causality tests using statsmodels.
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def granger_test(df: pd.DataFrame, x_col: str, y_col: str, maxlag: int = 4):
    """
    Test whether x_col Granger-causes y_col. DataFrame must contain both columns ordered by time.
    Returns the full statsmodels result dict.
    """
    data = df[[y_col, x_col]].dropna()
    return grangercausalitytests(data.values, maxlag=maxlag, verbose=False)
