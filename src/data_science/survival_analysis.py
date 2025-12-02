# survival_analysis.py
# Templates using lifelines; may require: pip install lifelines
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    import pandas as pd
except ImportError:
    pass

kmf_template = """
# pip install lifelines
from lifelines import KaplanMeierFitter
import pandas as pd

def fit_km(df, duration_col: str, event_col: str):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[duration_col], event_observed=df[event_col])
    return kmf
"""

cox_template = """
# pip install lifelines
from lifelines import CoxPHFitter
import pandas as pd

def fit_cox(df, duration_col: str, event_col: str, covariates: list):
    cph = CoxPHFitter()
    cph.fit(df[covariates + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)
    return cph
"""
