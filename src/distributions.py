"""Distributions of interval changes over multiple years by case_type."""
import pandas as pd
import numpy as np

def interval_change_distribution(interval_df, interval_col='days_to_alloc', group='case_type'):
    df = interval_df.copy()
    if 'date_received_opg' in df.columns:
        df['year'] = pd.to_datetime(df['date_received_opg']).dt.year
    elif 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
    else:
        raise ValueError("Provide 'date_received_opg' or 'date' to derive year.")
    df = df.dropna(subset=[interval_col])
    summary = (df.groupby([group,'year'])[interval_col]
                 .agg(median='median', p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75),
                      mean='mean', n='size')
                 .reset_index()
                 .sort_values([group,'year']))
    yoy = (summary.pivot(index='year', columns=group, values='median')
                 .diff().stack().reset_index().rename(columns={0:'yoy_median_change'}))
    return {'annual_stats': summary, 'yoy_change': yoy}
