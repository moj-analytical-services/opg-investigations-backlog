# User Guide

## 1. Installation
See the Quickstart in README. Use a virtual environment and `pip install -r requirements.txt`.

## 2. Inputs
- CSV exported from the Investigations DB with columns aligned to the data dictionary.
- Or generate synthetic data:
  ```bash
  python -m g7_assessment.cli generate-data --rows 8000 --out data/raw/synthetic_investigations.csv
  ```

## 3. Commands
- `generate-data` – create a synthetic dataset consistent with the schema.
- `eda` – summary profiles, missingness, correlations, and key plots saved in `reports/`.
- `train` – fits:
  - Backlog GLM (Poisson/NegBin) vs staffing & case mix
  - Legal review classifier
  - Time-to-PG-signoff survival model (Cox PH)
  - Daily backlog forecaster (ETS)
- `forecast` – produces 90-day backlog forecast.
- `simulate` – applies staffing deltas to assess backlog impact.

## 4. Outputs
- `models/` with fitted artefacts (joblib) and `reports/` with plots/metrics.
- `data/processed/` contains engineered features ready for modelling.
