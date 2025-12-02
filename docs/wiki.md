# Repository and Git Setup
## goal
**Deliver a reproducible, explainable investigations pipeline that quantifies backlog drivers, forecasts workload, prioritises high-risk cases, and exports simulation-ready inputs, with strong governance (privacy, ethics), CI/CD, and collaboration.**

- **Branching**: feature branches (feat/, fix/, refactor/), protected main.

- **CI gates**: ruff + black + pytest run on every push/PR via GitHub Actions.

- **PR hygiene**: PR template, CODEOWNERS, issue templates, semantic commit summaries.

- **Project management**: create a GitHub Project board (Backlog → In Progress → Review → Done), tag issues (data, model, infra, docs).

- **Quality & testing**: schema checks, unit & property tests, reproducibility (seeds), pre-commit hooks, doc pages for QA and ethics.
All of the above is pre-wired in the repo so you can demonstrate collaborative, production-ready habits.


- **“We use a CI wrapper (Makefile/script) so devs and CI run the same steps — no ‘works on my machine’ drift.”**

- **“We gate PRs with smoke tests: a tiny synthetic run proves the critical path creates the key artefacts.”**

- **“Devs get fast feedback and stable reviews; heavier tests run nightly on a fresh synthetic data build.”**

- **“We keep a feature catalogue linking code lineage, business definitions, and plots — auditable and policy-relevant.”**

- **“We export simulation-ready inputs (arrival rates, service-time quantiles, routing) to power micro-simulation.”**

- **We use a CI wrapper so devs and CI run the same steps. We gate PRs with smoke tests—tiny, end-to-end checks that a synthetic dataset runs through our pipeline and produces key artefacts. It keeps feedback fast and dependable; deeper tests run nightly.**

- A **CI wrapper** (a **Makefile** target or scripts/ci.sh) makes the exact same command work locally and in **GitHub Actions**.

- “Our **CI wrapper** ensures the *same* steps run locally and in CI—no ‘works on my machine’ drift.”

- “We gate PRs with **smoke tests**: a tiny synthetic run proves the critical path creates the key artefacts.”

- “Devs get **fast feedback** and stable reviews; heavier tests run **nightly** on a fresh synthetic dataset.”


## 1) Repository structure/layout (RAP-friendly) 
```
opg-investigations-backlog/
├── configs/                 # YAML inputs & scenarios
├── data/                    # raw & processed data (git-ignored) (placeholders only)
│   ├── raw/                 # original, untouched data, never commit PII
│   └── processed/           # cleaned or transformed data outputs
│   └── out/                 # processed data outputs
├── notebooks/               # exploratory and analysis notebooks
│   └── data-analysis.ipynb  # data analysis notebook
│   └── Leila-yousefi.ipynb
│   └── corr_cause.ipynb
├── docs/                    # User guide, QA, ethics, PM, interview Q&A, MkDocs site (schema, SIMUL8 wiring, model cards)
├── src/                     # Python modules/scripts, production code (pip-installable pkg)
│   ├── __init__.py          # marks this folder as a package
│   └── data_processing.py   # reusable data-loading and cleaning functions
│   └── data_quality.py      # Data Quality Checks class
│   └── notebook_code.py     # verbatim code extracted from your notebook (magics commented)
│   └── preprocessing.py     # re-exports: normalise_col, parse_date_series, engineer, ...
│   └── intervals.py         # re-exports: build_event_log, build_wip_series, build_backlog_series, ...
│   └── analysis_demo.py     # last-year interval analysis by team (non-invasive)
│   └── distributions.py     # distribution of interval changes by case_type across years
│   └── cli_nbwrap.py        # ETL, feature eng, models, viz, CLI (Click)
│   └── encoders.py          # K-Fold target encoder
│   └── features.py          # preprocessor with TE support
│   └── modeling.py          # adds advanced logistic pipeline + helpers
│   └── synth.py             # synthetic data generator (no PII)
│   └── cli.py               # add two new commands: diagnostics, logit-advanced
│   └── cli_nbwrap.py        # CLI wrapper orchestrating notebook logic (no logic changes)
│   └── diagnostics.py       # VIF, correlations, over-dispersion
├── tests/                   # Automated unit testing, Pytest unit tests (data & modeling)
│   └── test_data_quality.py # pytest tests for Data Quality
│   └── test_logit_pipeline.py # pytest tests for Data Quality
├─ .github/
│   └─ workflows/
│       └─ ci.yml            # Lint + tests on PRs/commits to main
│       └─ ci-cd.yml         # multi-stage pipeline(build→test→ deploy-qa), needs: enforce ordering, ties QA deploy to protected qa environment requires  approval.
├── .pre-commit-config.yaml  # black, ruff, nbstripout, etc.
├── .gitignore               # Configure to exclude from Git /data/, environment folders, caches, and any large files.
├── reports/                 # derived CSV/figures for stakeholders
├── scripts/ci.sh            # CI wrapper (local == CI)
├── README.md                # quickstart + governance pointers, project overview and setup instructions
├── docs/                    # MkDocs site (schema, SIMUL8 wiring, model cards)
├── Makefile                 # make setup | run-all | docs | synth
└── requirements.txt         # pinned Python dependencies / freezed library versions to ensure consistent environments across machines.
└── ...
```

## 2) Version control & collaboration
- Protected main: no direct pushes; all changes via PR.
- Branch naming: feat/…, fix/…, exp/… (for experiments), docs/…, chore/….
- Conventional commits for clean changelogs (feat:, fix:, docs:).
- PR template requires: problem, approach, evidence (metrics/plots), risks, rollout.
- CODEOWNERS maps folders to reviewers (data eng / modelling / policy).
- Project board: Backlog → In progress → Review → Done (weekly demo cadence).
- Issue tracking: GitHub Issues as the **SoT = “Source of Truth”**; optionally mirror to JIRA if needed. The single, authoritative place where the team records work, decisions, status and acceptance criteria. Everyone looks there first; if something isn’t in Issues, it’s not part of the plan.
    - Alignment: one canonical backlog → fewer conflicting lists (JIRA, spreadsheets, Slack).
    - Traceability: Issues ↔ PRs ↔ commits ↔ releases → clear audit trail.
    - Accountability: owners, due dates, acceptance criteria in one place.
- How to make GitHub Issues your SoT (quick rules)
    - Every piece of work has an Issue (no orphan PRs).
    - Link PRs to Issues with keywords (Closes #123) and Project/Milestone tags.
    - Use templates so Issues have problem, scope, acceptance criteria, data ethics/privacy notes.
    - Labels for triage (e.g., priority:P1, area:backlog, type:bug/feat/model).
    - Project board (Backlog → In Progress → Review → Done) reflects Issue status—not a separate spreadsheet.
    - Decisions (ADRs, key choices) are linked from the Issue.
    - If another tool (e.g., JIRA) must exist, one is SoT; the other mirrors it (sync or summary), never both as masters.
- PR description
```bash
Closes #123
- Implements backlog GLM with NegBin fallback
- Adds smoke test for daily backlog generation
```
- Issue template (excerpt)
```
## Problem
## Approach
## Acceptance Criteria
- [ ] Metric X computed for case_type
- [ ] Data ethics check recorded
## Links
- Design / ADR:
- Related PRs:
```

## 3) Documentation: what, where, how
- README.md: quickstart, data ethics, how to run pipeline, links to docs.
- MkDocs + Pages for:
    - Microsim schema (docs/microsim_schema.json)
    - SIMUL8 wiring guide (docs/how_to_wire_simul8.md)
    - Working Agreement (ways of working, quality gates)
    - Model cards (purpose, data, metrics, fairness, caveats)
    - Feature catalogue (definitions, lineage, plots)

- Notebooks: narrative analysis only; converted to code via wrappers. Use nbstripout + jupytext for diffs.
```yaml
# mkdocs.yml (snippet)
nav:
  - Home: docs/index.md
  - Microsim Schema: docs/microsim_schema.json
  - How to wire SIMUL8: docs/how_to_wire_simul8.md
  - Working Agreement: docs/working_agreement.md
```

## 4) Reproducibility & environments
- Pin Python & packages in requirements.txt (or environment.yml).
- Determinism: set seeds in modeling; synthetic generator provides stable tests.
- Data contracts: validate inputs with a schema (e.g., Pandera).
```python
# pandera input schema (example)
import pandera as pa
from pandera import Column, Check
EngineeredSchema = pa.DataFrameSchema({
    "id": Column(int),
    "case_type": Column(str),
    "risk": Column(str, checks=Check.isin(["Low","Medium","High"])),
    "date_received_opg": Column(pa.DateTime),
    "date_allocated_investigator": Column(pa.DateTime, nullable=True),
    "days_to_alloc": Column(float, nullable=True, checks=Check.ge(0)),
})
```
## 5) Data ethics, privacy, and sharing
- No PII in repo; use synthetic or anonymised data for examples and CI.
- Access controls: real data only via approved secure stores; bucket ACLs not repo-tracked.
- DPIA alignment; Model cards include fairness, limitations, and intended use.
- Metadata on every artifact (data snapshot hash, code version, timestamp).

## 6) Code quality (pre-commit)
```python
# .pre-commit-config.yaml (essentials)
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks: [ { id: black } ]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: ["--fix"]
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks: [ { id: nbstripout } ]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - { id: check-ast }
      - { id: check-yaml }
      - { id: end-of-file-fixer }
      - { id: trailing-whitespace }
```

## 7) CI wrapper & smoke tests (fast PR gates)
- “Our CI wrapper ensures the same steps run locally and in CI — no ‘works on my machine’ drift.”
```bash
# scripts/ci.sh
#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip
pip install -r requirements.txt
ruff check .
black --check .
pytest -q -m "smoke" --maxfail=1
```

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: "pip" }
      - run: scripts/ci.sh
```

```ini
# pytest.ini
[pytest]
markers =
    smoke: very fast checks for CI gating
```

```python
# tests/test_pipeline_smoke.py
import pytest
from pathlib import Path
@pytest.mark.smoke
def test_tiny_synth_pipeline(tmp_path: Path):
    from synth import generate_synthetic
    from preprocessing import load_raw, engineer
    from intervals import build_event_log, build_backlog_series
    df = generate_synthetic(n_rows=400, seed=42)
    raw_csv = tmp_path / "raw.csv"
    df.to_csv(raw_csv, index=False)
    raw, colmap = load_raw(raw_csv)
    eng = engineer(raw, colmap)
    events = build_event_log(eng)
    backlog = build_backlog_series(eng)
    assert len(eng) > 0
    assert events is not None and backlog is not None
```

- Pattern
    - Every PR: wrapper + smokes → quick green/red.
    - Nightly: full pipeline on fresh synthetic dataset (+ heavier tests/backtests/schema checks).

## 8) Nightly runs & artifacts
- “Heavier tests run nightly on a fresh synthetic data build.”
```yaml
# .github/workflows/nightly-run-all.yml
name: Nightly Run-All (Synthetic)
on:
  schedule: [ { cron: "0 3 * * *" } ]
  workflow_dispatch: {}
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: "pip" }
      - run: |
          python -m pip install -U pip
          pip install -r requirements.txt
          python -m g7_assessment.synth --rows 25000 --out data/raw/synth.csv --seed $RANDOM
          python -m g7_assessment.cli_nbwrap run-all --raw data/raw/synth.csv --outbase .
      - uses: actions/upload-artifact@v4
        with:
          name: nightly-artifacts
          path: |
            reports/**
            data/processed/**
          if-no-files-found: warn
```

## 9) Model evaluation, explainability & feature governance
- Results repo layout
    - models/ (fitted pipelines, params, hash of training data)
    - model_eval/ (notebooks/CSV: metrics, calibration curves)
    - reports/ (stakeholder tables & figures)

- Explainability
    - SHAP for global/local; eli5 permutation importance; keep seeds constant.
    - Feature catalogue in docs/feature_catalogue.md: definition, lineage (file+func), business meaning, plots.
```python
# SHAP example (binary classifier)
import shap
shap.explainers.Permutation # optional for model-agnostic
explainer = shap.Explainer(model.predict_proba, X_background)
shap_values = explainer(X_eval)
shap.plots.bar(shap_values, max_display=15)      # global
shap.plots.waterfall(shap_values[0])             # local
```
- Dimensionality reduction
    - PCA/clustering for diagnostics & communication (document in docs; plots in reports).
```python
# PCA quick diagnostic
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Xz = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xz)
coords = pca.transform(Xz)  # 2D for stakeholder plots
```

## 10) Backlog analytics: core DS patterns (ready to discuss)
- GLM (Poisson→NegBin) for backlog drivers: effect of investigators_on_duty, case_type, reallocation.
- Cox PH for time-to-allocation / PG sign-off (censoring-aware).
- Elastic-net logistic for legal review propensity (stable, explainable).
- SARIMAX for daily backlog forecasts with exogenous staffing.

**All are scripted via cli_nbwrap.py commands that call the existing notebook logic — ensuring policy-grade reproducibility.**

## 11) Continuous delivery (optional)
- Tag releases vMAJOR.MINOR.PATCH.
- Release notes summarise user-visible changes (plain English for policy).
- Optionally publish docs via GitHub Pages on every merge to main.
```yaml
# .github/workflows/docs.yml (build MkDocs)
name: Docs
on: { push: { branches: [ main ] } }
permissions: { contents: write }
jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: |
          pip install mkdocs mkdocs-material
          mkdocs build --strict
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

## 12) Dashboards & stakeholder loop
- Power-BI: connect to reports/ CSVs (non-PII) with scheduled refresh; embed issue links back to GitHub for feedback.
- JIRA/GitHub linking: PRs reference Issue IDs; dashboards link to the current release tag & model card.

## 13) Security & dependency hygiene
- Dependabot for pip updates.
- Secret scanning enabled; never commit .env or keys.
- pip caching in Actions; verify hashes if desired.
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule: { interval: "weekly" }
```

## 14) Minimal command cheatsheet (for live demo)
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
# Generate synth & run the full pipeline (no real data needed)
python -m synth --rows 8000 --out data/raw/synth.csv
python -m cli_nbwrap run-all --raw data/raw/synth.csv --outbase .
# See outputs
ls data/processed/    # engineered.csv, event_log.csv, backlog_series.csv, …
ls reports/           # last_year_by_team.csv, annual_stats.csv, yoy_change.csv, run_all_summary.md
```

## 15) Anti-patterns to avoid (and say so)
- Duplicated CI logic (README vs Make vs Workflow) → drift.
- Monolithic tests that do everything → slow feedback, devs stop running them.
- Network-dependent smokes → flaky CI (prefer synthetic).
- Hidden local .env magic → not reproducible (encode in scripts/CI).