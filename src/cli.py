import click
from pathlib import Path
import pandas as pd
from joblib import dump
from .config import REPORTS_DIR, MODELS_DIR
from .data import (
    generate_synthetic,
    load_data,
    basic_clean,
    engineer_intervals,
    daily_backlog_series,
    aggregate_staffing,
)
from .modeling import (
    fit_backlog_glm,
    forecast_backlog,
    fit_legal_review_classifier,
    fit_survival_model,
    scenario_apply_staffing,
)
from .evaluation import classification_metrics
from .visualisation import plot_daily_backlog, plot_forecast


@click.group()
def cli():
    pass


@cli.command(name="generate-data")
@click.option("--rows", type=int, default=8000, help="Number of synthetic rows")
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="data/raw/synthetic_investigations.csv",
)
def generate_data(rows, out):
    df = generate_synthetic(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    click.echo(f"Wrote {len(df):,} rows to {out}")


@cli.command()
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
def eda(csv):
    df = load_data(Path(csv))
    df = basic_clean(df)
    df = engineer_intervals(df)
    daily = daily_backlog_series(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig1 = plot_daily_backlog(daily)
    p1 = REPORTS_DIR / "daily_backlog.png"
    fig1.savefig(p1, dpi=160, bbox_inches="tight")
    click.echo(f"Saved {p1}")
    df.describe(include="all").to_csv(REPORTS_DIR / "summary_describe.csv")
    click.echo("EDA done.")


@cli.command()
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
def train(csv):
    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    daily = daily_backlog_series(df)
    staff = aggregate_staffing(df)
    glm, glm_df = fit_backlog_glm(daily, staff)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(glm, MODELS_DIR / "backlog_glm.joblib")
    glm_df.to_csv(MODELS_DIR / "backlog_glm_design.csv", index=False)
    fc = forecast_backlog(glm, glm_df, days=30)
    fc.to_csv(MODELS_DIR / "forecast_30d.csv", index=False)
    clf, (yte, yprob) = fit_legal_review_classifier(df)
    dump(clf, MODELS_DIR / "legal_review_clf.joblib")

    metrics = classification_metrics(yte, yprob)
    pd.Series(metrics).to_csv(MODELS_DIR / "legal_review_metrics.csv")
    try:
        cph = fit_survival_model(df)
        dump(cph, MODELS_DIR / "survival_cph.joblib")
    except Exception as e:
        click.echo(f"Survival model skipped: {e}")
    click.echo("Training complete.")


@cli.command()
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--days", type=int, default=90)
def forecast(csv, days):
    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    daily = daily_backlog_series(df)
    staff = aggregate_staffing(df)
    glm, glm_df = fit_backlog_glm(daily, staff)
    fc = forecast_backlog(glm, glm_df, days=days)
    fc.to_csv(REPORTS_DIR / f"forecast_{days}d.csv", index=False)
    fig = plot_forecast(glm_df[["date", "backlog"]], fc)
    fig.savefig(REPORTS_DIR / f"forecast_{days}d.png", dpi=160, bbox_inches="tight")
    click.echo(f"Saved forecast to reports/forecast_{days}d.csv/.png")


@cli.command()
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--delta-investigators", type=int, default=5, help="Change in investigators on duty"
)
def simulate(csv, delta_investigators):
    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    daily = daily_backlog_series(df)
    staff = aggregate_staffing(df)
    glm, glm_df = fit_backlog_glm(daily, staff)
    base_pred = glm.fittedvalues
    scn_pred = scenario_apply_staffing(glm_df, delta_investigators, glm)
    effect = (scn_pred - base_pred).mean()
    out = pd.DataFrame(
        {"date": glm_df["date"], "baseline": base_pred, "scenario": scn_pred}
    )
    out.to_csv(REPORTS_DIR / "scenario_vs_baseline.csv", index=False)
    click.echo(
        f"Mean change in backlog with Δinvestigators={delta_investigators}: {effect:.2f} cases/day (negative is good)."
    )


@cli.command()
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
def diagnostics(csv):
    """
    Run quick diagnostics: VIF (numerics) and correlation snapshot.
    Saves to reports/.
    """
    from .config import REPORTS_DIR
    from .data import load_data, basic_clean, engineer_intervals
    from .diagnostics import vif_for_numeric, corr_table

    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Choose numeric columns that generally exist in the project
    num_cols = [c for c in ["weighting", "days_to_alloc"] if c in df.columns]
    if num_cols:
        vif_df = vif_for_numeric(df, num_cols)
        vif_df.to_csv(REPORTS_DIR / "vif_numeric.csv", index=False)

    # Correlation among a small stable set
    corr_cols = [c for c in ["weighting", "days_to_alloc"] if c in df.columns]
    if len(corr_cols) >= 2:
        corr = corr_table(df, corr_cols)
        corr.to_csv(REPORTS_DIR / "corr_numeric.csv")

    click.echo("Diagnostics written to reports/")


@cli.command(name="logit-advanced")
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
def logit_advanced(csv):
    """
    Train advanced elastic-net logistic model for legal review propensity.
    Saves pipeline + metrics + calibration plot into models/.
    """
    from .config import MODELS_DIR, REPORTS_DIR
    from .data import load_data, basic_clean, engineer_intervals
    from .modeling import (
        fit_legal_review_logit_enet,
        save_calibration_plot,
        export_model_and_report,
    )

    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    pipe, metrics, Xte, yte = fit_legal_review_logit_enet(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_model_and_report(pipe, metrics, MODELS_DIR)

    # Save calibration curve
    cal_png = REPORTS_DIR / "legal_review_enet_calibration.png"
    save_calibration_plot(
        metrics["calibration_mean_pred"], metrics["calibration_frac_pos"], cal_png
    )
    click.echo("Advanced logistic model, metrics, and calibration saved.")


@cli.command()
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
def diagnostics(csv):
    """Write VIF & correlation diagnostics to reports/."""
    from .config import REPORTS_DIR
    from .data import load_data, basic_clean, engineer_intervals
    from .diagnostics import vif_for_numeric, corr_table

    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in ["weighting", "days_to_alloc"] if c in df.columns]
    if num_cols:
        vif_for_numeric(df, num_cols).to_csv(
            REPORTS_DIR / "vif_numeric.csv", index=False
        )
        corr_table(df, num_cols).to_csv(REPORTS_DIR / "corr_numeric.csv")
    click.echo("Diagnostics saved to reports/.")


@cli.command(name="legal-review-advanced")
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
def legal_review_advanced(csv):
    """Train elastic-net legal review model; save pipeline, metrics, calibration curve."""
    from .config import REPORTS_DIR, MODELS_DIR
    from .data import load_data, basic_clean, engineer_intervals
    from .modeling import fit_legal_review_enet
    import matplotlib.pyplot as plt

    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    pipe, metrics = fit_legal_review_enet(df)
    from joblib import dump

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipe, MODELS_DIR / "legal_review_enet.joblib")
    pd.Series(metrics).to_csv(MODELS_DIR / "legal_review_enet_metrics.csv")
    # Calibration curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        metrics["calibration_mean_pred"], metrics["calibration_frac_pos"], marker="o"
    )
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("Calibration")
    ax.set_xlabel("Mean predicted prob")
    ax.set_ylabel("Fraction positives")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "legal_review_enet_calibration.png", dpi=160)
    click.echo("Saved legal review model & metrics.")


@cli.command(name="backlog-drivers")
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--delta",
    type=int,
    default=5,
    help="Step change in investigators_on_duty for scenario.",
)
def backlog_drivers(csv, delta):
    """Fit GLM drivers for backlog and run a staffing scenario; write scenario CSV."""
    from .config import REPORTS_DIR
    from .data import (
        load_data,
        basic_clean,
        engineer_intervals,
        daily_backlog_series,
        aggregate_staffing,
    )
    from .modeling import fit_backlog_glm, scenario_staffing_effect

    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    daily = daily_backlog_series(df)  # uses receipt→allocation interval
    staff = aggregate_staffing(df)
    glm, design = fit_backlog_glm(daily, staff)

    base = glm.fittedvalues
    scen = scenario_staffing_effect(glm, design, delta)
    out = pd.DataFrame({"date": design["date"], "baseline": base, "scenario": scen})
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS_DIR / f"glm_staffing_scenario_delta{delta}.csv", index=False)
    click.echo("Backlog drivers & scenario saved.")


@cli.command(name="tsa-forecast")
@click.option("--csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--days", type=int, default=90)
def tsa_forecast(csv, days):
    """SARIMAX forecast of backlog with exogenous drivers."""
    from .config import REPORTS_DIR
    from .data import (
        load_data,
        basic_clean,
        engineer_intervals,
        daily_backlog_series,
        aggregate_staffing,
    )
    from .timeseries import fit_backlog_sarimax, forecast_sarimax

    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    daily = daily_backlog_series(df)
    staff = aggregate_staffing(df)
    hist = daily.merge(staff, on="date", how="left").fillna(
        {"investigators_on_duty": 0, "n_allocations": 0}
    )
    m, design = fit_backlog_sarimax(hist)
    fc = forecast_sarimax(m, design, days=days)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_csv(REPORTS_DIR / f"sarimax_forecast_{days}d.csv", index=False)
    click.echo("Forecast saved.")


@cli.command(name="microsim-export")
@click.option(
    "--csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to investigations CSV (real or synthetic).",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False),
    default="data/processed/microsim_inputs",
    help="Output directory for simulation inputs.",
)
def microsim_export(csv, outdir):
    """
    Build micro-simulation input bundle:
      - arrivals_daily.csv / arrivals_monthly.csv
      - backlog_daily.csv
      - staffing_daily.csv
      - service_time_quantiles_pg_signoff.csv
      - routing_legal_review.csv
      - metadata.json
    """
    from pathlib import Path
    from .data import load_data, basic_clean, engineer_intervals
    from .microsim import write_microsim_bundle

    # 1) Load & clean (lower-cased columns, dates parsed, booleans normalised)
    df = load_data(Path(csv))
    df = basic_clean(df)
    df = engineer_intervals(
        df
    )  # adds days_to_* and event flags; derives needs_legal_review if dates present

    # 2) Write bundle
    meta = write_microsim_bundle(df, Path(outdir))
    click.echo(f"Micro-simulation inputs written to: {outdir}")
    click.echo(f"Summary: {meta}")


if __name__ == "__main__":
    cli()
