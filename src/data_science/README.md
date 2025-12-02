# Investigation Synthetic Data Kit (Config-Driven)

This kit tailors synthetic-data generation to **investigation datasets** via a simple YAML config.
It supports:
- **Tabular** synthetic data (Gaussian copula + smoothed categorical sampling),
- **Time-series** synthetic counts by date and group (seasonal trend + AR noise),
- **k-anonymity style coarsening** before synthesis,
- **Privacy indicators** (nearest-neighbour gap, exact-match rate on quasi-IDs),
- **Utility checks** (distribution, correlation, simple predictive TSTR/TRTS if a target is provided),
- A **schema/data contract** export for shareable code.

## Quick start
1. Edit `config/investigation_schema.yml` to match your columns.
2. Put your real CSV somewhere accessible (keep it private).
3. Run:
   ```bash
   python build_tabular.py --real_csv /secure/path/investigations.csv --out_dir ./out
   python build_timeseries.py --real_csv /secure/path/investigations.csv --out_dir ./out
   python checks.py --real_csv /secure/path/investigations.csv --syn_csv ./out/synthetic_tabular.csv
   python data_contract.py --syn_csv ./out/synthetic_tabular.csv --out ./out/schema_contract.json
   ```
4. Share **code + synthetic + contract**, not the real CSV.

> Libraries used: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `matplotlib` (plots optional).
> No seaborn. Advanced DP is out-of-scope here; a minimal 1-D DP demo is included elsewhere in your previous bundle.
