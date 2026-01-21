# Synthesized data
# demo_eda.py
# Small, self-contained demo that exercises key methods on synthetic OPG-like data.

import numpy as np  # numerical work (corr, quantiles)
import pandas as pd  # core dataframe operations
from eda_opg import EDAConfig, OPGInvestigationEDA

# ----- 1) Create a small synthetic dataset for demonstration -----
rng = np.random.default_rng(42)
n = 2000

# Base dates
start = pd.Timestamp("2024-01-01")
recv_dates = start + pd.to_timedelta(rng.integers(0, 300, size=n), unit="D")

# Allocation occurs for ~85% within 1-30 days; else censored (NaT)
alloc_delays = rng.integers(1, 31, size=n)
allocated_mask = rng.random(size=n) < 0.85
alloc_dates = pd.Series(recv_dates) + pd.to_timedelta(alloc_delays, unit="D")
alloc_dates = alloc_dates.where(allocated_mask, pd.NaT)

# Sign-off for ~70% within 20-120 days from received; else censored
signoff_delays = rng.integers(20, 121, size=n)
so_mask = rng.random(size=n) < 0.70
signoff_dates = pd.Series(recv_dates) + pd.to_timedelta(signoff_delays, unit="D")
signoff_dates = signoff_dates.where(so_mask, pd.NaT)

# Categorical fields
case_types = rng.choice(["LPA", "Deputyship", "Other"], size=n, p=[0.6, 0.3, 0.1])
risk_band = rng.choice(["Low", "Medium", "High"], size=n, p=[0.5, 0.35, 0.15])
teams = rng.choice(["Team A", "Team B", "Team C"], size=n, p=[0.4, 0.4, 0.2])
region = rng.choice(["North", "Midlands", "South"], size=n)

# Daily ops fields
investigators_on_duty = rng.integers(8, 20, size=n)  # rough proxy
allocations = rng.integers(0, 25, size=n)  # allocated on that day
backlog = np.maximum(
    0, 500 + rng.normal(0, 60, size=n).astype(int)
)  # evolving backlog proxy

# Target: legal review ~5%, with higher odds for High risk and longer allocation delay
# We'll simulate it based on logits to mimic a real signal
base_logit = -3.0 + 0.02 * np.nan_to_num(alloc_dates - recv_dates).astype(
    "timedelta64[D]"
).astype(float)
risk_bump = np.select(
    [risk_band == "High", risk_band == "Medium"], [1.2, 0.4], default=0.0
)
logit = base_logit + risk_bump
prob = 1 / (1 + np.exp(-logit))
legal_review = (rng.random(size=n) < prob).astype(int)

# Assemble DataFrame
df = pd.DataFrame(
    {
        "id": np.arange(1, n + 1),
        "date_received_opg": recv_dates,
        "date_allocated_investigator": alloc_dates,
        "date_pg_signoff": signoff_dates,
        "case_type": case_types,
        "risk_band": risk_band,
        "team": teams,
        "region": region,
        "investigators_on_duty": investigators_on_duty,
        "allocations": allocations,
        "backlog": backlog,
        "legal_review": legal_review,
    }
)

# ----- 2) Configure columns and instantiate the EDA toolkit -----
cfg = EDAConfig(
    id_col="id",
    date_received="date_received_opg",
    date_allocated="date_allocated_investigator",
    date_signed_off="date_pg_signoff",
    target_col="legal_review",
    numeric_cols=[
        "days_to_alloc",
        "days_to_signoff",
        "investigators_on_duty",
        "allocations",
        "backlog",
    ],
    categorical_cols=["case_type", "risk_band", "team", "region"],
    time_index_col="date_received_opg",
    team_col="team",
    risk_col="risk_band",
    case_type_col="case_type",
)

eda = OPGInvestigationEDA(df, cfg)

# ----- 3) Run a few core EDA tasks (print or log these in practice) -----
print("\n== QUICK OVERVIEW ==")
print(eda.quick_overview())

print("\n== MISSINGNESS ==")
print(eda.missingness_matrix().head(10))
print(
    "Missing 'days_to_signoff' vs target:\n", eda.missing_vs_target("days_to_signoff")
)

print("\n== OUTLIERS (days_to_signoff) ==")
print(eda.iqr_outliers("days_to_signoff"))

print("\n== CATEGORICAL SUMMARY (case_type × risk_band) ==")
summary = eda.group_summary(
    by=["case_type", "risk_band"],
    metrics={
        "n": ("id", "count"),
        "legal_rate": ("legal_review", "mean"),
        "med_alloc": ("days_to_signoff", "median"),
    },
)
print(summary.head(12))

print("\n== NUMERIC CORRELATIONS (Spearman) ==")
print(eda.numeric_correlations("spearman"))

print("\n== REDUNDANCY DROP LIST (|r|>0.9) ==")
print(eda.redundancy_drop_list())

print("\n== CLASS IMBALANCE ==")
print(eda.imbalance_summary())

print("\n== LEAKAGE SCAN ==")
print(eda.leakage_scan(["post", "signed", "decision", "outcome"]))

print("\n== INTERACTION: risk_band × binned days_to_signoff -> legal_review rate ==")
print(eda.binned_interaction_rate("days_to_signoff", "risk_band"))

print("\n== RESAMPLED TIME SERIES (daily) ==")
ts = eda.resample_time_series(
    {
        "backlog": ("backlog", "last"),
        "inv_mean": ("investigators_on_duty", "mean"),
    }
)
print(ts.tail())

print("\n== LAG CORRELATIONS: backlog vs inv_mean ==")
print(eda.lag_correlations(ts["backlog"], ts["inv_mean"]))

print("\n== KM QUANTILES by risk_band (signoff) ==")
print(eda.km_quantiles_by_group("days_to_signoff", "event_signed_off", "risk_band"))

print("\n== MONTHLY KPIs by team ==")
print(eda.monthly_kpis().head(12))
