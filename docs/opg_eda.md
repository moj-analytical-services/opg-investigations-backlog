# OPG EDA (Exploratory Data Analysis) — Repo Skeleton

## Install
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Run tests (smoke)
```bash
pytest -q -m "smoke"
```

## CI wrapper
We use `scripts/ci.sh` as a single source of truth. GitHub Actions calls it in `.github/workflows/ci.yml`.

## Package layout
```
src/opg_eda/
  __init__.py
  eda_opg.py
tests/
  test_eda_opg.py
scripts/
  ci.sh
```

## Usage
See `tests/test_eda_opg.py` for a synthetic example of constructing `EDAConfig` and `OPGInvestigationEDA` and calling methods.
