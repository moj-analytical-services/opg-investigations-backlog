import numpy as np
import pandas as pd
import pytest

from opg_eda import EDAConfig, OPGInvestigationEDA

pytestmark = [pytest.mark.smoke]


def make_synth(n=300, seed=123):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    recv = start + pd.to_timedelta(rng.integers(0, 60, size=n), unit="D")
    alloc_delay = rng.integers(1, 21, size=n)
    mask_alloc = rng.random(size=n) < 0.85
    alloc = pd.Series(recv) + pd.to_timedelta(alloc_delay, unit="D")
    alloc = alloc.where(mask_alloc, pd.NaT)

    so_delay = rng.integers(20, 61, size=n)
    mask_so = rng.random(size=n) < 0.70
    so = pd.Series(recv) + pd.to_timedelta(so_delay, unit="D")
    so = so.where(mask_so, pd.NaT)

    risk = rng.choice(["Low", "Medium", "High"], size=n, p=[0.5, 0.35, 0.15])
    case_type = rng.choice(["LPA", "Deputyship", "Other"], size=n, p=[0.6, 0.3, 0.1])
    team = rng.choice(["Team A", "Team B", "Team C"], size=n, p=[0.4, 0.4, 0.2])
    region = rng.choice(["North", "Midlands", "South"], size=n)

    inv = rng.integers(8, 20, size=n)
    allocs = rng.integers(0, 25, size=n)
    backlog = np.maximum(0, 300 + rng.normal(0, 40, size=n).astype(int))

    base_logit = -3.0 + 0.02 * np.nan_to_num((alloc - recv)).astype(
        "timedelta64[D]"
    ).astype(float)
    risk_bump = np.select([risk == "High", risk == "Medium"], [1.0, 0.3], default=0.0)
    prob = 1 / (1 + np.exp(-(base_logit + risk_bump)))
    legal = (rng.random(size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "date_received_opg": recv,
            "date_allocated_investigator": alloc,
            "date_pg_signoff": so,
            "case_type": case_type,
            "risk_band": risk,
            "team": team,
            "region": region,
            "investigators_on_duty": inv,
            "allocations": allocs,
            "backlog": backlog,
            "legal_review": legal,
        }
    )
    return df


def cfg():
    return EDAConfig(
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


def test_smoke():
    eda = OPGInvestigationEDA(make_synth(), cfg())

    ov = eda.quick_overview()
    assert isinstance(ov["shape"], tuple)

    miss = eda.missingness_matrix()
    assert isinstance(miss, pd.Series)

    out = eda.iqr_outliers("days_to_alloc")
    assert "n_outliers" in out and isinstance(out["n_outliers"], int)

    gs = eda.group_summary(["case_type", "risk_band"], {"n": ("id", "count")})
    assert {"case_type", "risk_band", "n"}.issubset(gs.columns)

    corr = eda.numeric_correlations("spearman")
    assert "backlog" in corr.columns

    drops = eda.redundancy_drop_list()
    assert isinstance(drops, list)

    imb = eda.imbalance_summary()
    assert imb is None or "pos_share" in imb

    inter = eda.binned_interaction_rate("days_to_alloc", "risk_band")
    assert inter.shape[0] > 0

    ts = eda.resample_time_series(
        {"backlog": ("backlog", "last"), "inv_mean": ("investigators_on_duty", "mean")}
    )
    assert {"backlog", "inv_mean"}.issubset(ts.columns)

    lags = eda.lag_correlations(ts["backlog"], ts["inv_mean"])
    assert "lag_7" in lags.index

    kmq = eda.km_quantiles_by_group("days_to_alloc", "event_alloc", "risk_band")
    assert kmq.shape[0] > 0

    kpis = eda.monthly_kpis()
    assert {"team", "__month"}.issubset(kpis.columns)
