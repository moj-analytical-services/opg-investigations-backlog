# src/synth.py
# Synthetic investigations generator for nightly runs.
# Produces a realistic-shaped CSV with the columns the pipeline expects.
#
# Columns generated (subset; add more if you like):
# - id (int), case_type (str), risk (str), weighting (float), reallocation (int),
# - date_received_opg (YYYY-MM-DD),
# - date_allocated_investigator (YYYY-MM-DD or NaT),
# - days_to_alloc (int; implied by dates; your engineer() may re-derive),
# - investigator (str),
# - needs_legal_review (0/1),
# - days_to_pg_signoff (int), event_pg_signoff (0/1), date_pg_signoff (date)
#
# NOTE: Staffing daily metrics in the code are derived from the per-day unique
#       investigators appearing in allocations, so we generate 'investigator'.

from __future__ import annotations
from typing import Sequence, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import click


CASE_TYPES_DEFAULT: Sequence[str] = ("LPA", "Deputyship", "EPA", "Complaint", "Other")
RISKS_DEFAULT: Sequence[str] = ("Low", "Medium", "High")


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def generate_synthetic(
    n_rows: int = 20_000,
    start_date: str = "2022-01-01",
    days_span: int = 1_200,  # ~3.3y
    case_types: Sequence[str] = CASE_TYPES_DEFAULT,
    risks: Sequence[str] = RISKS_DEFAULT,
    n_investigators: int = 180,
    seed: Optional[int] = 7,
) -> pd.DataFrame:
    """
    Create a synthetic OPG-like investigations dataset with sensible relationships.

    - Concerns (arrivals) are uniformly spread across [start_date, start_date+days_span).
    - Allocation delay ~ Gamma shaped by (case_type, risk). Some remain unallocated.
    - PG sign-off delay ~ Gamma; some are censored (open) -> event_pg_signoff=0.
    - Legal review propensity depends on case_type & risk.
    - Reallocation rare event with slightly longer allocation delay.

    Returns
    -------
    DataFrame with columns listed at the top of this file.
    """
    rng = _rng(seed)
    start = pd.to_datetime(start_date)

    # ID
    ids = np.arange(1, n_rows + 1, dtype=int)

    # Case mix
    ct_weights = np.array([0.45, 0.25, 0.05, 0.15, 0.10])[: len(case_types)]
    ct_weights = ct_weights / ct_weights.sum()
    case_type = rng.choice(case_types, size=n_rows, p=ct_weights)

    risk_weights = np.array([0.55, 0.33, 0.12])[: len(risks)]
    risk_weights = risk_weights / risk_weights.sum()
    risk = rng.choice(risks, size=n_rows, p=risk_weights)

    # Arrivals (date_received_opg)
    offsets = rng.integers(0, max(days_span, 1), size=n_rows)
    date_received = (start + pd.to_timedelta(offsets, unit="D")).normalize()

    # Investigator assignment at allocation
    inv_ids = np.array([f"INV_{i:03d}" for i in range(1, n_investigators + 1)])
    investigator = rng.choice(inv_ids, size=n_rows)

    # Reallocation indicator (rare)
    reallocation = rng.binomial(1, 0.12, size=n_rows)

    # Allocation delay (days) by case_type & risk (Gamma; strictly positive)
    # Base means (tune as needed)
    base_means = {"LPA": 8, "Deputyship": 14, "EPA": 10, "Complaint": 6, "Other": 9}
    risk_mult = {"Low": 0.8, "Medium": 1.0, "High": 1.25}

    mean_alloc = np.array(
        [
            base_means.get(ct, 10) * risk_mult.get(rk, 1.0)
            for ct, rk in zip(case_type, risk)
        ]
    )
    # Reallocation adds friction
    mean_alloc = mean_alloc * (1.10 + 0.35 * reallocation)

    # Gamma(k, theta) with k fixed, theta=mean/k
    k_alloc = 2.0
    theta_alloc = mean_alloc / k_alloc
    alloc_delay = rng.gamma(shape=k_alloc, scale=theta_alloc).astype(int)

    # Some cases not yet allocated (open backlog)
    open_alloc = rng.binomial(1, 0.06, size=n_rows).astype(bool)
    date_alloc = date_received + pd.to_timedelta(alloc_delay, unit="D")
    date_alloc = date_alloc.astype("datetime64[ns]")
    date_alloc[open_alloc] = pd.NaT

    # Needs legal review probability by (case_type, risk)
    base_legal = {
        "LPA": 0.08,
        "Deputyship": 0.14,
        "EPA": 0.10,
        "Complaint": 0.05,
        "Other": 0.07,
    }
    risk_legal = {"Low": -0.01, "Medium": 0.0, "High": +0.07}
    p_legal = np.clip(
        np.array(
            [
                base_legal.get(ct, 0.08) + risk_legal.get(rk, 0.0)
                for ct, rk in zip(case_type, risk)
            ]
        ),
        0.01,
        0.9,
    )
    needs_legal_review = rng.binomial(1, p_legal).astype(int)

    # PG sign-off delay (Gamma), with heavier tails; censored if long
    base_pg = {"LPA": 20, "Deputyship": 35, "EPA": 24, "Complaint": 12, "Other": 18}
    mean_pg = np.array(
        [
            base_pg.get(ct, 20) * risk_mult.get(rk, 1.0)
            for ct, rk in zip(case_type, risk)
        ]
    )
    mean_pg = mean_pg * (1.05 + 0.20 * needs_legal_review)
    k_pg = 1.6
    theta_pg = mean_pg / k_pg
    pg_delay = rng.gamma(shape=k_pg, scale=theta_pg).astype(int)

    # Some are censored (no sign-off yet)
    cens = rng.binomial(1, 0.18, size=n_rows).astype(bool)
    date_pg = date_received + pd.to_timedelta(pg_delay, unit="D")
    date_pg[cens] = pd.NaT
    event_pg = (~cens).astype(int)

    # Weighting in [1..5] skewed upward with risk
    weighting = np.clip(
        rng.normal(
            2.8
            + np.array([{"Low": -0.3, "Medium": 0.0, "High": +0.4}[rk] for rk in risk]),
            0.6,
        ),
        1.0,
        5.0,
    )

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "case_type": case_type,
            "risk": risk,
            "weighting": weighting.round(2),
            "reallocation": reallocation.astype(int),
            "date_received_opg": date_received,
            "date_allocated_investigator": date_alloc,
            "investigator": investigator,
            "needs_legal_review": needs_legal_review,
            "days_to_pg_signoff": pg_delay,
            "event_pg_signoff": event_pg,
            "date_pg_signoff": date_pg,
        }
    )

    # Derive days_to_alloc from dates (may be recomputed by your engineer())
    df["days_to_alloc"] = (
        df["date_allocated_investigator"] - df["date_received_opg"]
    ).dt.days
    return df


@click.command()
@click.option(
    "--rows",
    type=int,
    default=20000,
    show_default=True,
    help="Number of synthetic cases.",
)
@click.option(
    "--start",
    "start_date",
    type=str,
    default="2022-01-01",
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--span",
    "days_span",
    type=int,
    default=1200,
    show_default=True,
    help="Days span to spread arrivals.",
)
@click.option("--seed", type=int, default=7, show_default=True, help="Random seed.")
@click.option(
    "--out",
    "out_csv",
    type=click.Path(dir_okay=False),
    default="data/raw/synthetic_investigations.csv",
    show_default=True,
)
def main(rows, start_date, days_span, seed, out_csv):
    """CLI: write a fresh synthetic dataset to CSV each run."""
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic(
        n_rows=rows, start_date=start_date, days_span=days_span, seed=seed
    )
    df.to_csv(out, index=False)
    click.echo(f"Wrote synthetic data: {out} (rows={len(df):,})")


if __name__ == "__main__":
    main()
