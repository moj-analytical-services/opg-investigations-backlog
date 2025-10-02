import pandas as pd
from lifelines import CoxPHFitter

def fit_cox(df: pd.DataFrame, formula: str = 'triage_delay + wip + staffing + risk_band'):
    cph=CoxPHFitter(); cph.fit(df,duration_col='duration',event_col='event',formula=formula); return cph

def hazard_multiplier(cph: CoxPHFitter, row: pd.Series):
    coefs=cph.params_.reindex(row.index).fillna(0.0); return float((coefs*row).sum())
