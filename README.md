# OPG Investigations Backlog
Operational analytics, forecasting, and **transparent micro‑simulation + AI‑assisted optimisation** for the Office of the Public Guardian (OPG) investigations backlog.

> **Classification:** OFFICIAL (when populated). Do **not** commit personal data.
> **Security:** Use environment variables / GitHub Secrets for credentials.

## What’s included
- **Linked data & metrics** pipeline (design) for backlog measurement (size, age, throughput).
- **Micro‑simulation (DES)** engine (transparent) for staffing & policy scenarios.
- **AI helpers**: Gaussian‑Process **emulator** and **Bayesian optimisation** to search policy space fast; **causal survival** scaffold for CoP escalation.
- **CI/CD**: tests, lint, type checks, security scans, docs deploy, releases.
- **Docs & governance**: DPIA template, Model Card, QA checklist, strategy pack.
- **No‑code UI**: `Streamlit` scenario runner for stakeholders.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
make lint && make test
streamlit run app.py
mkdocs serve
```

## Repository structure
```

opg-investigations-backlog/
├─ src/opg_backlog_sim/        # Package: DES + AI helpers
├─ configs/                    # YAML inputs & scenarios
├─ docs/                       # MkDocs site (project spec, playbook, AI methods)
├─ .github/workflows/          # CI/CD pipelines
├─ tests/                      # Unit tests
├─ notebooks/                  # Exploration (outputs stripped)
├─ data/{raw,interim,processed}/  # (placeholders only)
└─ ...

```

## New in this version 02 October 2025
- **AI‑driven hybrid**: emulator (GP), Bayesian optimisation and causal survival scaffold.
- **Streamlit UI** for non‑coders.
- Extra **CI** (CodeQL, Dependabot, pip‑audit, stale, release‑please).
- Expanded docs on **AI & Optimisation** and non‑black‑box governance.

## Licensing
- **Code**: MIT (`LICENSE`).
- **Docs & non‑code**: Open Government Licence v3 (`DATA_LICENSE.md`).

See `docs/OPG_Investigations_Backlog_Documentation.md` for the full specification.
