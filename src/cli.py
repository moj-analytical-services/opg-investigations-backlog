import click
from pathlib import Path
import pandas as pd
from joblib import dump
from .config import REPORTS_DIR, MODELS_DIR
from .data import generate_synthetic, load_data, basic_clean, engineer_intervals, daily_backlog_series, aggregate_staffing
from .modeling import fit_backlog_glm, forecast_backlog, fit_legal_review_classifier, fit_survival_model, scenario_apply_staffing
from .evaluation import classification_metrics
from .visualisation import plot_daily_backlog, plot_forecast

@click.group()
def cli():
    pass

@cli.command(name="generate-data")
@click.option("--rows", type=int, default=8000, help="Number of synthetic rows")
@click.option("--out", type=click.Path(dir_okay=False), default="data/raw/synthetic_investigations.csv")
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
    from .evaluation import classification_metrics
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
@click.option("--delta-investigators", type=int, default=5, help="Change in investigators on duty")
def simulate(csv, delta_investigators):
    df = engineer_intervals(basic_clean(load_data(Path(csv))))
    daily = daily_backlog_series(df)
    staff = aggregate_staffing(df)
    glm, glm_df = fit_backlog_glm(daily, staff)
    base_pred = glm.fittedvalues
    scn_pred = scenario_apply_staffing(glm_df, delta_investigators, glm)
    effect = (scn_pred - base_pred).mean()
    out = pd.DataFrame({"date": glm_df["date"], "baseline": base_pred, "scenario": scn_pred})
    out.to_csv(REPORTS_DIR / "scenario_vs_baseline.csv", index=False)
    click.echo(f"Mean change in backlog with Δinvestigators={delta_investigators}: {effect:.2f} cases/day (negative is good).")
    
if __name__ == "__main__":
    cli()
