# OPG Investigations Backlog
Operational analytics, forecasting, and **transparent micro‑simulation + AI‑assisted optimisation** for the Office of the Public Guardian (OPG) investigations backlog.

> **Classification:** OFFICIAL (when populated). Do **not** commit personal data.
> **Security:** Use environment variables / GitHub Secrets for credentials.


&nbsp;
#==============================================================================
# @author: Dr. Leila Yousefi 
#==============================================================================
&nbsp;



# Contents

* What the  for the Model does
* Why the for the Model is useful
* How users can get started with the  for the Model
* Where users can get help with  for the Model
* Who maintains and contributes to the  for the Model


&nbsp;
# [Repository and Git Setup](#setup) 
&nbsp;
# [Quality Assurance](#qa) 
&nbsp;
# [Project management & CI](#pm-ci)
&nbsp;
# [Methods and Model](#summ) - summary of the quantitative methods used for modelling.  
&nbsp;   
# [Inputs](#inputs)
&nbsp;
# [Outputs](#outputs)
&nbsp;
# [Higher-level Process flow Diagrams](#high-process-flow) - proccess flow diagrames of main inputs/outputs for the Model.
&nbsp;
# [Aim](#aim) 
&nbsp;
# [Objectives](#objectives)
&nbsp;   
# [Background Knowledge](#Background)
&nbsp;
# [Control Assumptions and Sensitivity Analysis](#control-assumptions)
&nbsp;
# [Data Sources](#data-sources)
&nbsp;
# [Model Calculation](#calc-model) - details of model calculation
&nbsp;
# [Feature Engineering and Data Preparation](#preprocessing)
&nbsp;
# [Implementing the Demand Forecasting for LPA Model](#model) - details of model scripts
&nbsp;
# [Future Work](#future)
&nbsp;
# [Licencing](#licence)
&nbsp; 

&nbsp; 


<a name="setup"></a>
# Repository and Git Setup
- **Branching**: feature branches (feat/, fix/, refactor/), protected main.

- **CI gates**: ruff + black + pytest run on every push/PR via GitHub Actions.

- **PR hygiene**: PR template, CODEOWNERS, issue templates, semantic commit summaries.

- **Project management**: create a GitHub Project board (Backlog → In Progress → Review → Done), tag issues (data, model, infra, docs).

- **Quality & testing**: schema checks, unit & property tests, reproducibility (seeds), pre-commit hooks, doc pages for QA and ethics.
All of the above is pre-wired in the repo so you can demonstrate collaborative, production-ready habits.

- **We use a CI wrapper so devs and CI run the same steps. We gate PRs with smoke tests—tiny, end-to-end checks that a synthetic dataset runs through our pipeline and produces key artefacts. It keeps feedback fast and dependable; deeper tests run nightly.**
- A **CI wrapper** (a **Makefile** target or scripts/ci.sh) makes the exact same command work locally and in **GitHub Actions**.
* “Our **CI wrapper** ensures the *same* steps run locally and in CI—no ‘works on my machine’ drift.”
* “We gate PRs with **smoke tests**: a tiny synthetic run proves the critical path creates the key artefacts.”
* “Devs get **fast feedback** and stable reviews; heavier tests run **nightly** on a fresh synthetic dataset.”


## Repository structure / layout
```
opg-investigations-backlog/
├── configs/                 # YAML inputs & scenarios
├── data/                    # raw & processed data (git-ignored) (placeholders only)
│   ├── raw/                 # original, untouched data
│   └── processed/           # cleaned or transformed data outputs
│   └── out/                 # processed data outputs
├── notebooks/               # exploratory and analysis notebooks
│   └── data-analysis.ipynb  # data analysis notebook
│   └── Leila-yousefi.ipynb
│   └── corr_cause.ipynb
├── docs/                    # User guide, QA, ethics, PM, interview Q&A
├── src/                     # Python modules and scripts
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
│   └── cli.py               # add two new commands: diagnostics, logit-advanced
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
├── README.md                # project overview and setup instructions
└── requirements.txt         # pinned Python dependencies / freezed library versions to ensure consistent environments across machines.
└── ...
```

## How to use the modules (examples)
```python
# PREPROCESSING / MANIPULATION / IMPUTATION
from preprocessing import load_raw, engineer
raw, colmap = load_raw("data/raw/raw.csv")
typed = engineer(raw, colmap)  # uses your notebook’s exact logic

# INTERVAL ANALYSIS
from intervals import build_event_log, build_backlog_series, summarise_daily_panel
events = build_event_log(typed)
backlog = build_backlog_series(typed)
daily = summarise_daily_panel(typed)

# DEMO: Last-year interval analysis by team (non-invasive)
from analysis_demo import last_year_by_team
trend = last_year_by_team(eng_df=typed, backlog_series=backlog, bank_holidays=None)

# DISTRIBUTIONS over years by case_type
from distributions import interval_change_distribution
res = interval_change_distribution(interval_df=typed, interval_col="days_to_alloc", group="case_type")
annual_stats = res["annual_stats"]  # per year x case_type: median, p25, p75, mean, n
yoy_change   = res["yoy_change"]    # year-on-year change in medians by case_type

```


### How to run (end-to-end)
```bash
# 0) Activate your venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) PREP: run your original preprocessing/engineering (unchanged logic)
python -m cli_nbwrap prep --raw data/raw/raw.csv --outdir data/processed

# 2) INTERVALS: event log + backlog + panels (uses your notebook functions)
python -m cli_nbwrap intervals --eng data/processed/engineered.csv --outdir data/processed

# 3) DEMO: last-year team trend (non-invasive)
python -m cli_nbwrap trend-demo --eng data/processed/engineered.csv \
                                              --backlog data/processed/backlog_series.csv \
                                              --out reports/last_year_by_team.csv

# 4) DISTRIBUTIONS: per-case_type over years (default uses days_to_alloc)
python -m cli_nbwrap interval-distribution --eng data/processed/engineered.csv \
                                                         --interval-col days_to_alloc \
                                                         --group case_type \
                                                         --outdir reports

```


## Quickstart
```bash
# 1) Create env and install
# venv & deps (if not already set up)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline in one go (choose your CSV and output base folder)
python -m cli_nbwrap run-all \
  --raw data/raw/raw.csv \
  --outbase . \
  --interval-col days_to_alloc \
  --group case_type

# 2) Generate synthetic data (replace with your real CSV when ready)
python -m cli generate-data --rows 8000 --out data/raw/synthetic_investigations.csv

# 3) EDA (saves plots & tables to ./reports)
python -m cli eda --csv data/raw/synthetic_investigations.csv

# 4) Train all models (saves into ./models)
python -m cli train --csv data/raw/synthetic_investigations.csv

# 5) Forecast 90-day backlog + plot
python -m cli forecast --csv data/raw/synthetic_investigations.csv --days 90

# 6) Staffing scenario: add 10 investigators
python -m cli simulate --csv data/raw/synthetic_investigations.csv --delta-investigators 10


make lint && make test
streamlit run app.py
mkdocs serve
```

### What each command does (and how it uses your code)
prep → DATA PRE-PROCESSING / MANIPULATION / IMPUTATION
Calls your load_raw and engineer functions, writes data/processed/engineered.csv.

intervals → INTERVAL ANALYSIS
Calls your build_event_log, build_backlog_series, build_daily_panel, summarise_daily_panel. Writes four CSVs.

trend-demo → Demo: last-year interval analysis by team
Calls your IntervalAnalysis.monthly_trend_last_year(...) through the wrapper.

interval-distribution → Distributions across years by case_type
Computes per-year stats and YoY deltas for days_to_alloc (or any interval column you pick).

## implementation (Push to GitHub)
```bash
git init
git add .
git commit -m "feat: initial backlog project"
git branch -M main
git remote add origin https://github.com/moj-analytical-services/opg-investigations-backlog.git
git push -u origin main

flake8 src/ tests/

```

## Run tests locally
```bash
# Discover & run all tests in tests/
pytest --maxfail=1 --disable-warnings -q

```
or from Jupyter lab notebook: 
```python
!pytest -q
```

## Automate via CI (GitHub Actions)
Create a file .github/workflows/ci.yml:

## Discover & run all tests in tests/
pytest --maxfail=1 --disable-warnings -q
-q gives you a concise report.

On success you’ll see something like === 10 passed in 0.5s ===.

On failure you’ll get full assertion tracebacks
## Workflow

- data: place raw files in data/raw/.
- data: place cleaned / transformed data files in data/processed/.

- notebooks: work in notebooks/data_analysis.ipynb.

- modularise: once stable, move functions into src/.

- version: commit early & often:

- review: open a PR against main when ready.

```bash
git add .
git commit -m "EDA: initial missing-value summary"
git push origin main
```

- Version Control Workflow
    - Branching: Create a feature branch, e.g. feature/LY-analysis-01.
    - Commits: After each major step—EDA, modeling, evaluation—commit with a clear message:
    ```bash
    git add notebooks/01_analysis_template.ipynb src/data_processing.py
    git commit -m "EDA: added missing-value visualization"
    ```
    - Pull Request: When complete, open a PR against main and tag reviewers.

    - commit code + tests + requirements.
    
    - **CI**: add GitHub Actions workflow to enforce tests on every push.
    - Verify on GitHub: Navigate to the Actions tab of your repository on GitHub. You should see the “CI” workflow queued or running.
- Sharing with the Panel
    - Push your branch to GitHub: git push origin feature/analysis-X.
    - At discussion time, share the GitHub link to your notebook so the panel can view the commit history, code annotations, and outputs live.
    
- create the file if it doesn't exist
```bash
touch .gitignore
# Remove data/ from the index (but leave the files on disk)
git rm -r --cached data/
# Commit the change
git commit -m "Remove data folder from tracking per .gitignore"

python src/analyzers.py
```

## GitHub Actions setup to schedule CI to run the full pipline
### Nightly CI to run the full pipeline
a one-liner Makefile target (make run-all) or wire this into your GitHub Actions so the pipeline runs on a nightly schedule against a sample dataset.
1. Makefile (repo root) 
2. Nightly CI to run the full pipeline (.github/workflows/nightly-run-all.yml)
- Notes
If your real raw data cannot live in the repo, uncomment the “Retrieve raw data” step and pull from a secure store using GitHub Secrets.
Otherwise, commit a small data/raw/sample_raw.csv and the workflow will use it.

### Nightly micro-simulation export
As already added the microsim-export command earlier, you can schedule it too.
- .github/workflows/nightly-microsim.yml

### Running the full pipeline (help the team to run it easily)
```bash
make setup
make run-all RAW=data/raw/raw.csv
# or, if you committed a small sample:
make run-all RAW=data/raw/sample_raw.csv
```

#### Output files
data/processed/engineered.csv, event_log.csv, backlog_series.csv, daily_panel.csv, daily_panel_summary.csv
reports/last_year_by_team.csv, annual_stats.csv, yoy_change.csv, run_all_summary.md

### Nightly CI (example)
- A scheduled workflow (.github/workflows/nightly-run-all.yml) runs the pipeline at 03:00 UTC and uploads artifacts.
---

## Sanity checklist

- [ ] You’ve already added the **wrappers** and **CLI** (`src/g7_assessment/cli_nbwrap.py`) I provided earlier.  
- [ ] Add this **Makefile** at repo root.  
- [ ] Add the **nightly** workflows under `.github/workflows/`.  
- [ ] Ensure **`data/raw/raw.csv`** (real) or **`data/raw/sample_raw.csv`** (toy) is present or retrievable.  
- [ ] Push to GitHub → check **Actions** → artifacts contain `reports/` & `data/processed/`.

## Synthetic data generator + an updated nightly workflow
- This self-generates data, runs the pipeline against it, and publishes artifacts, keeping everything reproducible and policy-safe without relying on a checked-in sample.

1. creates a fresh dataset every night and then runs your full pipeline.
    - src/synth.py
2. Add a CLI command to produce the CSV (optional but handy)
    - src/cli_nbwrap.py
3. Update the nightly workflow to use the generator
4. Replace: .github/workflows/nightly-run-all.yml with this:
```yaml
name: Nightly Run-All (Synthetic)

on:
  schedule:
    - cron: "0 3 * * *"   # 03:00 UTC daily
  workflow_dispatch: {}

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Generate fresh synthetic dataset
        run: |
          # Option A: call the synth module directly
          python -m synth --rows 25000 --start 2022-01-01 --span 1400 --seed $RANDOM --out data/raw/synthetic_investigations.csv
          # Option B (equivalent): via the wrapper CLI
          # python -m cli_nbwrap gen-synth --rows 25000 --out data/raw/synthetic_investigations.csv

      - name: Run full pipeline on synthetic data
        run: |
          python -m cli_nbwrap run-all \
            --raw data/raw/synthetic_investigations.csv \
            --outbase .

      - name: Upload artifacts (reports + processed)
        uses: actions/upload-artifact@v4
        with:
          name: nightly-artifacts
          path: |
            reports/**
            data/processed/**
          if-no-files-found: warn

```

4. (Optional) Makefile helper
    - Append to your root Makefile:
    ```
    synth:
    	. $(VENV)/bin/activate && $(PY) -m g7_assessment.synth --rows 20000 --out data/raw/synthetic_investigations.csv
    
    ```

    - Run locally with:
    ```bash
    make setup
    make synth
    make run-all RAW=data/raw/synthetic_investigations.csv
    ```

&nbsp; 

&nbsp;

<a name="setup"></a>
# Quality Assurance
- use pre-commit for local QA, but merges are gated by CI (lint, tests, schema checks) on GitHub.
- use protected branches, required reviews via CODEOWNERS, and a Projects board for shared visibility.
- for data quality, run Pandera/Great Expectations in CI.
- document with model cards and stakeholder-friendly release notes, and publish docs automatically. This setup lets analysts, engineers, and policy colleagues collaborate with confidence and traceability.
- Quality starts locally with pre-commit (Black/Ruff/nbstripout).
- On GitHub, gate merges with protected branches, required reviews, and Actions CI (lint, tests, data checks).
- manage work with Projects, issue/PR templates, CODEOWNERS, and semantic releases.
- for data, use schema contracts (Pandera/Great Expectations), reproducible environments, and artifact versioning. We communicate findings via model cards, docs, and stakeholder-focused release notes.

&nbsp; 

## QA on GitHub (beyond pre-commit)

### Repository rules
- **Protected branches (main)**: require PRs, forbid force pushes, require up-to-date branch before merge.
- **Required status checks**: CI must pass (lint, unit tests, data-contract tests).
- **CODEOWNERS**: auto-request specific reviewers (e.g., data engineering for ETL, policy for docs).

### Automated checks (Actions)
- **Lint/format**: ruff check ., black --check .
- **Tests**: pytest -q, coverage thresholds
- **Type checks (optional)**: mypy or pyright
- **Security**: CodeQL code scanning, secret scanning, Dependabot (PRs for vulnerable deps)
- **Data contracts**: run Pandera or Great Expectations suite in CI; fail the build if schema breaks
- **Notebook hygiene**: nbstripout in pre-commit; optionally fail CI if outputs are present

### Data & experiments
- **Reproducibility**: pinned **requirements.txt** (or poetry.lock), make targets, seeds, deterministic tests
- **Artifact versioning**: models/ + metadata; consider DVC (or Git LFS) for large data/models

### Regression & reliability
- Property-based tests (Hypothesis) for data transforms
- Golden/master tests for key reports/metrics (compare to tolerance bands)
- Smoke tests on sample data for faster feedback

## Peer review & collaboration
- **PR process**
    - Templates: “Problem → Approach → Evidence (metrics/plots) → Risks → Rollout/monitoring”
    - Small PRs (≤ 300 lines) with before/after screenshots for plots/reports
    - Enforce Conventional Commits (feat/fix/chore) for clean changelogs

- **Reviews**
    - Require ≥1–2 approvals; block self-merge; dismiss stale reviews after changes
    - Checklists: tests added, docs updated, data contract updated, performance/fairness checked

- **Documentation**
    - README.md (how to run), docs/ (user guide, QA, ethics, model card)
    - Autopublish docs with MkDocs + GitHub Pages on merge to main

- **Stakeholder comms**
    - **Write release notes in plain language (what changed, expected impact, caveats)**
    - **Link PRs to Issues with acceptance criteria; add labels (data/model/infra/policy)**

- **Cross-discipline**
    - Discussions for RFCs; ADRs (Architecture Decision Records) for durable choices
    - Projects (Kanban) with swimlanes by workstream (Data Eng / Modelling / Policy / Comms)

&nbsp; 

## Quality Assurance using pre-commit
**We keep a pinned .pre-commit-config.yaml in the repo so every commit is auto-formatted and linted locally, and CI enforces the same rules. Changing that file changes the rules everyone runs—pre-commit auto-installs the new versions, so we update via pre-commit autoupdate and fix any formatting in a single PR for consistent, reproducible quality.”**

- pre-commit is a framework that runs checks/formatters before a Git commit (the “pre-commit” hook). If a check fails, the commit is blocked so bad code doesn’t enter the repo.
- .pre-commit-config.yaml is the project’s manifest that lists which checks to run (e.g., Black, Ruff, nbstripout, trailing whitespace), which versions to use, and any args/filters.

### Benefit of pre-commit
- Consistency & hygiene: automatic formatting, linting, and file fixes.
- Catch issues early: broken YAML/JSON, syntax errors, big diffs in notebooks, etc.
- Less review noise: PRs focus on logic, not whitespace and commas.
- Reproducible tooling: versions are pinned in the config, so every dev and CI run the same tools.

### Set pre-commit up (project & developer)
```
pip install pre-commit
pre-commit install          # adds the Git hook in .git/hooks/pre-commit

```

- On commit: runs automatically and may modify files (e.g., Black). If it edits files, re-add (git add -A) and commit again.
- Run on all files (useful after adding a new hook):
```
pre-commit run --all-files

```
- New/updated hooks take effect on the next run. pre-commit will auto-install tool versions declared under rev into a local cache.
- Everyone pulling the change will start using the updated hooks the next time they commit (no reinstall needed—pre-commit install is only to place the Git hook).
- CI will also use the new config (if your pipeline calls pre-commit or runs the same linters).
- If you bump versions (e.g., Black/Ruff), formatting or lint rules might change → expect some files to be reformatted/fixed.
- If you tighten rules (e.g., more Ruff checks), commits may start failing until code is fixed.
- To safely upgrade pinned versions, use:
```
pre-commit autoupdate   # updates revs to latest tags
pre-commit run -a       # fix everything locally
git commit -m "chore: bump pre-commit hook versions"

```
- Limit scope: in a hook you can set files:/exclude: regex to target only certain paths (e.g., avoid running Black on generated code).
- Stages: most hooks run at pre-commit; there are also pre-push, commit-msg hooks if you configure them.
- Bypass (discouraged): git commit --no-verify skips hooks locally, but CI should still run them so policy can’t be bypassed.
- Clean cache: pre-commit gc if caches get large.

### Typical pitfalls (and fixes) for pre-commit
- Notebooks: use nbstripout so outputs/metadata don’t bloat diffs. If you need cell IDs, add a hook that inserts them before stripout or run a separate notebook normaliser.
- Large diffs after upgrades: run pre-commit run -a and commit the mechanical changes in a separate “chore” PR.
- Tool mismatch across devs: keep versions pinned under rev and update them centrally via pre-commit autoupdate.


&nbsp; 

&nbsp; 

<a name="pm-ci"></a>
# Project management & CI
- Shared goals: Projects + Issues make scope/priority visible to non-engineers.
- Predictability: CI catches breakages early; protected branches keep main always deployable.
- Trust & influence: clear PRs, evidence (metrics/plots), and release notes help leaders/policy colleagues understand trade-offs.
- **Branch protection**: Settings → Branches → Protect main (require PRs, CI checks, 1–2 reviews).
- **Actions CI**: add a ci.yml with steps for install, lint, tests, data-contracts, coverage.
- **CODEOWNERS**: map folders to teams (e.g., src/etl/ @data-eng, docs/ @policy-team).
- **Issue & PR templates**: add .github/ISSUE_TEMPLATE/*.md and PULL_REQUEST_TEMPLATE.md.
- **Projects**: create board; columns Backlog → In Progress → Review → Done; automate issue → PR → Done transitions.
- **Docs site**: MkDocs workflow to publish on main; link from repo header.
- **Release flow**: Conventional Commits + Release Drafter or semantic-release to auto-generate changelogs.

## Documentation site
After pushing to GitHub:
1. Ensure Actions run for the **Docs** workflow.
2. In **Settings → Pages**, set Source to **Deploy from a branch** and select the `gh-pages` branch, `/ (root)`.
3. Your site will publish at: `https://moj-analytical-services.github.io/opg-investigations-backlog/`.


&nbsp; 

&nbsp; 

<a name="summ"></a>
# Methods and Model 

## What’s included
- **Linked data & metrics** pipeline (design) for backlog measurement (size, age, throughput).
- **Micro‑simulation (DES)** engine (transparent) for staffing & policy scenarios.
- **AI helpers**: Gaussian‑Process **emulator** and **Bayesian optimisation** to search policy space fast; **causal survival** scaffold for CoP escalation.
- **CI/CD**: tests, lint, type checks, security scans, docs deploy, releases.
- **Docs & governance**: DPIA template, Model Card, QA checklist, strategy pack.
- **No‑code UI**: `Streamlit` scenario runner for stakeholders.

&nbsp; 




## Scoping success
Three questions: backlog drivers, time-to-sign-off, and legal-review propensity. KPIs: MAE for forecasts, C-index for survival, AUC/Brier for classifier. Prioritised interpretability and policy levers (staffing, case mix).

## Why Poisson/Negative Binomial for backlog?
Backlog is a count; start with Poisson GLM; if variance >> mean, switch to NegBin. Coefficients stay interpretable as log-rate ratios.

## Does staffing reduce backlog? (Feasibility test)
Include investigators_on_duty in GLM, control for time and case-type mix; examine coefficient, partial dependence, and run counterfactual scenarios via the simulator.

## Multicollinearity handling
Check VIF/correlations; use reference categories, drop redundant features, or regularise (ridge/elastic-net) if needed.

## Missing data strategy
Parse all dates; derive intervals; explicit Unknown for categoricals; median/model-based imputation inside CV folds.

## Survival model choice
Cox PH: handles censoring and yields hazard ratios for risk/case-type; sanity-check PH assumptions, report C-index.

## Ethics & governance?
Minimise features, document lawful basis, publish model cards, subgroup performance checks, no personal data in repo; see docs/ethics.md.

## Risk/backlog & interval analysis
engineered days_to_alloc and days_to_pg_signoff; backlog GLM + survival analysis.

## High-risk/legal review identification
classifier for needs_legal_review using case type/risk/weighting/reallocation.

## Feasibility of staffing changes
scenario tool measuring backlog change per Δ investigators.

## Data prep & QA
cleaning, imputation, feature pipelines, tests, CI, and collaboration workflow.

## Further work with time
Hierarchical modelling across teams, richer calendar effects, Bayesian uncertainty propagation, DES microsimulation integration.


&nbsp; 

&nbsp; 


**Aim:**

&nbsp; 

&nbsp; 

<a name="objectives"></a>
# Objectives

1. **Backlog drivers** Measure how investigator staffing over time and the time-to-allocation interval (receipt → allocation) influence daily, monthly, and annual backlog, overall and by case type/risk.

2. **Timing analysis** Use statistical and time-series methods to understand which factors lengthen or shorten key intervals (e.g., time to allocation; time to PG sign-off).

3. **High-risk triage** Identify applications most likely to generate concerns so verification can be prioritised upstream.

4. **Legal review propensity** Estimate which case types (and related factors) are more likely to require legal review, to plan capacity and reduce rework.

5. **Feasibility testing** Test whether adding investigators is the dominant lever versus alternatives (case-mix shifts, reallocation, process changes), including “step-change” scenarios and expected backlog reduction.

6. **Simulation handoff** Produce simulation-ready inputs (arrival rates by case type, service-time distributions with censoring, routing probabilities to legal review, staffing profiles) to power a micro-simulation of the investigation pathway.

7. **Policy impact** Provide recommendations to reduce backlog and improve donor experience, with quantified trade-offs, uncertainty, and subgroup fairness checks.

8.


## Bite-size Objectives
1. **Feature engineering & preprocessing (leak-safe)**
    - Impute + scale numerics, encode categoricals, add optional numeric interactions.
    - Handle high-cardinality categories (e.g., occupation) via K-Fold target encoding (no leakage).

2. **Advanced logistic model for legal-review propensity**
    - Elastic-Net Logistic (CV) for stable selection; optional domain interactions.
    - Proper split, metrics (AUC/AP/Brier), probability calibration curve.

3. **Diagnostics**
    - VIF & correlation snapshot (redundancy).
    - Optional over-dispersion check (GLM viewpoint).

4. **CLI tasks**
    - New commands to run diagnostics and an advanced logistic pipeline end-to-end.

5. **Tests**
    - Smoke tests so CI stays green.


## A. Break the problem into small objectives
### A1) **Data layer & clear definitions**
- Goal: create leak-safe features and time series that policy colleagues can trust.
- Intervals: days_to_alloc (receipt → investigator allocation), days_to_pg_signoff (receipt → PG signoff).
- Backlog: daily count of cases not yet allocated (D level), plus M and Y aggregates; case-type slices.
- Exogenous drivers: per-day investigators_on_duty, n_allocations, case-mix shares, reallocation rate.
- Outputs: tidy CSVs under data/processed/ for downstream models & simulation.

### A2) Drivers of backlog (explanatory model)
- Goal: quantify how staffing, case mix, and time affect backlog.
- Model: Poisson GLM; auto-switch to NegBin on over-dispersion.
- Effects: elasticities/partial dependence for (investigators_on_duty, case_type, risk, reallocation).
- Feasibility: “what if” scenarios (step change in staff; counterfactual contributions of other factors).

### A3) Time-to-allocation / PG-signoff (interval outcomes)
- Goal: which factors accelerate or delay the process?
- Model: Cox PH survival (handles censoring); check PH visually; hazard ratios by case_type, risk, weighting.

### A4) Time series of backlog (operational forecasting)
- Goal: short-horizon, explainable forecasts for ops planning.
- Model: SARIMAX with exogenous (investigators_on_duty, case-mix shares).
- Backtest: rolling origin; report MAE/MAPE; produce 90-day forecast.

### A5) High-risk applications likely to trigger concerns (prioritisation)

- Goal: score incoming applications to triage verification.
- Model: (if your upstream app dataset is available) balanced elastic-net logistic; if not yet available, wire a placeholder CLI expecting a CSV with raised_concern target. (Keeps the contract ready.)

### A6) Legal review propensity
- Goal: which cases are likely to require legal review?
- Model: elastic-net logistic with target encoding for high-card fields (occupation/team, if needed).
- Deliverables: AUC/PR-AUC, calibration plot, subgroup fairness by case_type.

### A7) Micro-simulation handoff
- Goal: parameters for DES/micro-sim.
- Exports: arrivals per day by case_type; routing probs (→ legal review); service-time distributions (from survival); staffing as a controllable resource; write microsim_inputs/*.csv.
- 
&nbsp; 

&nbsp; 


<a name="preprocessing"></a>
# Feature Engineering and Data Preparation
### DATA LOADING AND FEATURE ENGINEERING
- Imports and environment setup, data loading, joining/merging datasets, aggregation/grouping, pivot/reshape, data cleaning, sorting, feature engineering, exporting outputs

This function tidies the raw spreadsheet into a clean table the rest of the pipeline can use. It turns text into numbers where needed (like FTE and weighting), converts date columns into actual dates, creates an anonymised staff_id for privacy, and adds a simple is_reallocated flag based on the “Reallocated Case” column.
You can choose whether to keep only reallocated cases (only_reallocated=False) or all cases (False). Either way, the is_reallocated indicator is included so you can filter or analyse by it later.

- only_reallocated parameter (default True): lets you reuse the same function both for a reallocated-only analysis and for whole-population runs (set False).
- is_reallocated column: a clear boolean you can use downstream for filtering, stratifying, or auditing — even when you choose not to filter.
- Role handling: if a Role column exists upstream, it’s used; otherwise an empty role is set (keeps pipelines stable).
- Safety: if days_to_pg_signoff is entirely missing but relevant dates exist, it’s computed from dt_alloc_invest → dt_pg_signoff.


- load_raw: 
    - Files can be saved with different text encodings.
    - This loader tries common encodings automatically, tidies the text, and builds a dictionary that lets us find columns even if their names have odd spacing or capitalisation.
- col:
    - Column headers can vary slightly (extra spaces, different cases). The col helper looks up a column by a cleaned version of its name and tolerates small differences so pipelines keep working across files.
- engineer:
    - This turns the raw spreadsheet into a tidy table the models can use:
        - consistent column names and types (numbers are numbers, dates are dates)
        - adds missing-but-important fields (e.g., an anonymised staff ID)
        - keeps only the cases marked as Not *reallocated*

### Data Manipulation and Processing
#### date_horizon
- We need a date window (a start and end date) to analyze. By policy:
    - Start comes from the earliest “date received in investigations” (dt_received_inv).
    - End comes from the latest “PG sign-off date” (dt_pg_signoff) plus a small buffer (pad_days) to capture tail activity.
    - If those columns are missing/empty, we can fall back to the earliest and latest across any dt_… date columns.
    - If that still fails, we default to “last 30 days up to today (+ padding)”.

- If either start or end is still missing and we’re allowed to fall back:
    - Collect all columns whose names start with dt_.
    - Stack them together, drop missing values.
    - Use the earliest date as start and latest date as end if needed.
- This keeps our analysis consistent and prevents accidental trimming when some dates are missing.
- Keeps the meaning of the analysis window aligned with process reality (received → sign-off).
- Robust to missing data thanks to fallbacks and sensible defaults.
- The padding helps catch late events around the sign-off boundary.

#### build_event_log
- We need a timeline of events (one row per date × staff × case × event) to understand what happened and when.
This function:
    - Computes the analysis window using date_horizon.
    - Scans each case for milestone dates (received, allocated, legal steps, sign-off, closed, etc.).
    - Writes an event row for every milestone date within the horizon.
    - Adds a compact meta JSON with context (weighting, case type, status, etc.) so we can analyze later without bloating columns.
- Produces a long-format event timeline that’s ideal for plotting, counting, and modelling (“how many sign-offs per week?”, “who picked up cases today?”).
- The meta JSON keeps useful context for each event without making the table very wide and repetitive.
- Restricting to the computed horizon ensures the log always aligns with your agreed analysis window.


### TIME SERIES ANALYSIS
- Build a day-by-day series showing how many cases each investigator has “in progress” (WIP), and an optional workload measure that accounts for case complexity and staff FTE.

- A case is counted as WIP from the day it’s allocated to an investigator until the earliest of:
    - it is closed, it gets PG sign-off, or we reach the reporting end date.

- We want a daily time series showing, for each staff member, how many cases they are actively working (WIP = Work In Progress) and a simple workload measure that adjusts for case complexity and staff capacity.
    - A case counts as WIP from the day it is allocated to an investigator (dt_alloc_invest) until the earliest of:
        - the case is closed (dt_close), or
        - it receives PG sign-off (dt_pg_signoff), or
        - we reach the reporting end date.

- Output is one row per date × staff member × team, with:
    - wip (how many cases they have on the go) and wip_load (a proxy for workload = weighting ÷ FTE), summed over their active cases.
        - A complex case (higher weighting) increases load.
        - A part-time FTE increases load (same case is a bigger share of their time).

- If you don’t provide the start and end dates, the function works them out automatically using your project rules:
    - Start horizon comes from the earliest dt_received_inv;
    - End horizon comes from the latest dt_pg_signoff, plus a padding window.

- It uses your official milestones (dt_alloc_invest, dt_close, dt_pg_signoff) to decide when a case is actively being worked.

- Fast & scalable: It uses a delta method (add +1 at the start date, −1 after the end date) so it can efficiently build daily WIP counts even for thousands of cases.

- It gives both a count (wip) and a load (wip_load = weighting ÷ FTE) so you can see not just how many cases someone has, but how heavy that workload likely is.

- If some dates are missing, it falls back sensibly (e.g., if a case never closes, it stays WIP until the end horizon).


#### A tiny mental model
- Think of each case as a bar on a timeline (from allocation to close/signoff).
- We lay all bars for a person on top of each other.
- For any given day, how many bars overlap? That’s wip.
- If some bars are “heavier” (higher weighting) or the staff member has lower FTE, the overlap total becomes wip_load.

#### Common edge cases handled
- Open cases with no close/signoff → they count as WIP until the report end date.
- Missing weighting/FTE → sensible defaults keep the math stable.
- No cases for a person → they simply won’t appear in the output (or will have zeros after merge/accumulation).

#### Tiny visual example (intuition)
If a case runs from Jan 2 to Jan 5:
- We add +1 on Jan 2.
- We add −1 on Jan 6 (the day after it finishes).
- Cumulative sum across days produces:
Jan 1: 0
Jan 2: 1
Jan 3: 1
Jan 4: 1
Jan 5: 1
Jan 6: 0
Now imagine multiple cases overlapping—WIP is just the sum of overlaps each day.


- Build a daily time series showing the size of the allocation backlog: 
- Calculate ow many cases have been received into Investigations but not yet allocated to an investigator.

- It builds a timeline of the allocation backlog — how many cases have arrived in Investigations but haven’t yet been allocated to an investigator — day by day (or week by week).

- It counts Received (cases entering the queue) and Allocated (cases leaving to a person) per day.
- It then takes a running total (cumulative) of each and computes:
    - Backlog = Total Received so far − Total Allocated so far.
- optionally:
    - Exclude weekends/holidays to focus on working days only.
    - Resample weekly or monthly, keeping the last cumulative value per period (the correct way for running totals).
    - Compute a weighted backlog (if some cases are heavier/more complex) using a weighting column.

- Received means dt_received_inv (case enters the Investigations queue).
- Allocated means dt_alloc_invest (case leaves the queue and goes to a person).
- Backlog available (on any day) = total received so far − total allocated so far.
- If we don’t provide a reporting window, the function figures it out using your rules:
    - Start from the earliest dt_received_inv.
    - End at the latest dt_pg_signoff, with a padding window added.

- There’s also an optional weighted backlog, which treats some cases as “heavier” based on weighting (e.g., complexity).
- Matches our operational definition of backlog (waiting to be allocated to an investigator).
- Operationally accurate: matches the definition of backlog (awaiting allocation).
- Transparent: shows both cumulative inputs (received/allocated) and the resulting backlog; we publish cumulative received and allocated alongside the backlog so you can audit the numbers.
- Robust: it works even if some days have no activity; it also can clip the backlog at zero to avoid confusing negatives; prevents negative backlog and handles days with no activity cleanly.
- Flexible & practical: business-day filtering and weekly/monthly views match how teams actually review performance; it can compute a weighted version if you want a complexity-aware measure.

- Builds the daily picture of staff activity and backlog pressure across the investigation process.
- Each row in the output shows, for each investigator on each date:
    - how many cases they were working on (wip)
    - how heavy that workload was (wip_load)
    - what events happened that day (e.g., new case, legal step, PG sign-off)
    - how long since they last picked up a case
    - whether they are new in post (less than 4 weeks)
    - what the system backlog looked like that day
    - day-of-week, term, season, and bank holiday context
- The result feeds directly into forecasting models, dashboards, or simulation inputs.
- Combines everything: merges workload, case flow, and events into a single daily dataset.
- Flexible: supports working-day calendars, holiday exclusions, and weekly backlog summaries.
- Transparent: every part comes from separate, auditable builder functions, nothing hidden.
- Scalable: runs efficiently even for many staff over long periods.

- Step-by-step logic
    1. Determine the date range (start/end) using date_horizon().
    2.  Build core inputs:
        - events = timeline of case milestones.
        - wip = ongoing cases per staff/day.
        - backlog = unallocated cases per day (received − allocated).
    3. Create a daily grid for all staff and all working dates.
    4. Merge in WIP and events, turning event names into flag columns (0/1).
    5. Compute features:
        - time since last new case pickup
        - week, season, term, holiday, and new starter status
    6. Join backlog context to every day’s record.
    7. Return three consistent datasets for downstream modelling.

### Summerise
- Rolls up the detailed daily staff panel into team-level (or any custom grouping) time series, to quickly see trends like “total WIP per team per day/week” or “how many new cases did Team A pick up last month?”.

- Practical: real reporting/forecasting often needs team- or org-level time series, not just staff-level detail.
- Correct aggregation: it sums “flow” metrics (e.g., events, WIP cases) and treats stateful metrics (like backlog levels) correctly when resampling by taking the last value per period (the right way to downsample cumulative/state variables).
- Flexible: you pick the grouping keys, the resampling frequency, and can override the aggregation rules if needed.

- How it works (step-by-step)
    1. Choose the grouping
       By default it groups by date and team. You can change by to include role, or collapse to just date for an overall total.
    2. Aggregate daily
       It sums WIP and WIP load across staff, sums events, takes the median time since last pickup (typical day for staff), and counts distinct staff on duty.
    3. (Optional) Resample to weekly/monthly
       If you pass freq='W-FRI' (weekly Fridays) or 'MS' (month-start), it:
       - Sums the “flow” fields within each period (e.g., total new cases in that week/month).
       - Takes the last value for stateful/level fields (e.g., backlog_available) so the weekly/monthly series reflects the end-of-period level.
    4. Return a tidy frame
       With columns like: wip_sum, wip_load_sum, event_*_sum, backlog_available_mean (daily means) and, when resampled, last values for backlog-like metrics (you can change the list via resample_cum_last).

       
## 1) Executive Summary
This notebook/pipeline constructs a daily-level dataset of investigations by investigator, role, and team, and prepares it for Bayesian forecasting (PyMC/Bambi). Outputs support backlog management, staffing decisions, and monitoring.

## 2) Data Science Process (CRISP-DM)
- **Business Understanding:** Reduce and manage backlog; evaluate staffing/process changes.
- **Data Understanding:** Inspect coverage, timing, and consistency for logs/assignments/roster.
- **Data Preparation:** Parse dates, clean IDs, join tables, create daily roll-ups.
- **Modelling:** Hierarchical Negative Binomial with calendar effects and random intercepts.
- **Evaluation:** Backtesting, calibration (PPC), accuracy (MAE/MAPE), operational value.
- **Deployment:** Export datasets/forecasts; schedule refresh; QA checks.
- **Monitoring:** Data drift and forecast accuracy over time.


## 3) Data Engineering
**Inputs:** case logs, assignments, investigator roster, calendars, reference mappings.  
**Pipeline:** load → clean → join → derive daily panel → compute metrics → quality checks → export.

## 4) Features for Modelling
Calendar features (DOW/holidays), operational features (availability/FTE), demand proxies (inflow/backlog), trend/seasonality, and hierarchical IDs (investigator→team→role).


## 5) Bayesian Predictive Modelling (overview)
Counts are modelled with **Negative Binomial** (NB) to allow overdispersion. The linear predictor includes fixed effects (weekday/holiday) and random intercepts for investigator/team/role (**partial pooling**). Posterior draws provide 90‑day forecasts with credible intervals.


&nbsp; 

&nbsp; 



## Data Pre-processing, Manipulation, Time Series Analysis, Interval Analysis
- (prep → intervals → demo → distributions)
1. Ensure these wrapper modules are present
    - src/notebook_code.py
    - src/preprocessing.py
    - src/intervals.py
    - src/analysis_demo.py
    - src/distributions.py
2. Add the CLI wrapper (if not already added)
    - src/cli_nbwrap.py
4. Append this new “all-in-one” command to src/cli_nbwrap.py
5. output should generate follwoings:
   - data/processed/engineered.csv, event_log.csv, backlog_series.csv, daily_panel.csv, daily_panel_summary.csv
   - reports/last_year_by_team.csv (if your IntervalAnalysis API is available)
   - reports/annual_stats.csv, reports/yoy_change.csv
   - reports/run_all_summary.md (quick audit note)
6. Tiny CI smoke test
   - tests/test_run_all_imports.py
    ```python
    def test_cli_nbwrap_imports():
        import cli_nbwrap as cli
        assert hasattr(cli, "run_all")
    ```
7. What each module does
- notebook_code.py — a straight lift of all code cells from your Build_Investigator_Daily_from_Raw_12_11_25.ipynb (no changes to logic; only %/! lines commented).
- preprocessing.py — “DATA PRE-PROCESSING / DATA MANIPULATION / MISSING DATA IMPUTATION” section; simply imports and re-exports your functions such as normalise_col, parse_date_series, hash_id, month_to_season, is_term_month, load_raw, col, engineer.
- intervals.py — “INTERVAL ANALYSIS” section; re-exports your build_event_log, build_wip_series, build_backlog_series, build_daily_panel, summarise_daily_panel.
- analysis_demo.py — “Demo: last-year interval analysis by team (non-invasive)”; provides a last_year_by_team(...) function that calls your existing IntervalAnalysis API (if present).
- distributions.py — “For each casetype and for all of them, find the distribution of time-interval changes over a few years”; gives interval_change_distribution(...) to summarise medians and YoY deltas per case_type using your engineered dates/intervals.

- timeseries.py — SARIMAX backlog forecasting with exogenous drivers (staffing, case mix)
- timeseries.py —
## Variable-selection workflow for logistic regression 
- **domain-led → leak-safe preprocessing → shrinkage-based selection → interaction sanity checks → robust inference refit → calibration & subgroup evidence.**
  
- **variable Selection methodology:**
I started with a domain-informed set and encoded them in a leak-safe pipeline. I removed/combined redundant predictors (VIF/correlation), added a short, pre-declared list of interactions, and used elastic-net logistic with cross-validation for shrinkage and selection. I retained features that were non-zero and sign-stable across folds and improved CV AUC/calibration. For high-cardinality variables like occupation, I used target encoding or random effects. Finally, I refit a compact GLM for explainability and reported odds ratios with robust SEs, plus calibration and subgroup checks. 
It balances predictive performance with interpretability and gives concrete keep/drop rules.

0) **Frame it first**
- Target & metric: binary outcome (e.g., legal review), primary metric AUC/PR-AUC; secondary: calibration (Brier/reliability), subgroup fairness.

- Split: time-based split if predicting the future; otherwise stratified train/valid/test. All choices below happen inside CV on the training data (no leakage).

1) **Build a candidate set (domain + EDA)**
- Start from a domain-informed list (must-include features).
- Add plausible proxies and a shortlist of interactions (e.g., risk × case_type, workload × team); apply the hierarchical principle (include main effects if you include an interaction).

**Guardrails:**
- Dates ⇒ derive intervals (days) not raw dates.
- Create missingness flags (e.g., risk_missing) for important fields.

2) **Make the design matrix (leak-safe)**
- Numeric: median impute → standardise.
- Categorical: One-Hot with reference level (drop='first') or target encoding (nested CV) if very high cardinality.
- Keep the imputer/encoder inside a Pipeline so they’re fit on train folds only.

3) **Screen for redundancy (don’t overdo it)**
- Correlation/VIF: if two variables tell the same story (e.g., casetype & weighting), drop one or combine (e.g., priority).
- If you keep both, consider residualising one on the other (orthogonalise) or rely on ridge/elastic-net to stabilise.

4) **Baseline, then regularised model**
- Fit a baseline (main effects only) to anchor calibration.
- Fit Elastic Net logistic (penalty='elasticnet', solver='saga') with CV to shrink and select features (especially after one-hot/interaction expansion).
- **Keep/drop rules (prediction-first):**
    - Keep variables with non-zero coefficients selected in most CV folds (stability).
    - Prefer Elastic Net over pure Lasso when features are correlated.
    - If occupation/team has many levels, prefer random effects (mixed model) or target encoding; if you must one-hot, use ridge/elastic-net.

5) **Interactions & nonlinearity**
- Add a small, pre-declared set of interactions/polynomials.
- Retain only if they improve CV AUC/PR-AUC and remain sign-stable across folds. (You can also use ALE interaction plots as evidence.)

6) **Over-dispersion / clustering checks**
- In logistic GLM, check Pearson χ² / df and residual clustering by team/occupation.
- If inflated SEs: report cluster-robust SEs or fit a mixed-effects logistic with a random intercept for the grouping factor.

7) **Final “reporting model”**
- For explanation, refit a parsimonious GLM (main effects ± a few interactions you kept) on the full training set:
    - Report odds ratios with 95% CIs (cluster-robust if needed).
    - Provide calibration (reliability curve) and subgroup metrics.
- Lock hyperparameters; evaluate once on the held-out test.

- **Regularised, prediction-oriented selection (scikit-learn):**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV

num = ["weighting", "days_to_alloc"]
cat = ["risk", "case_type"]  # keep high-card variables separate or target-encode

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    # Add sparse interactions among numerics if plausible
    ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", num_pipe, num),
    ("cat", cat_pipe, cat),
], remainder="drop")

logit = LogisticRegressionCV(
    penalty="elasticnet", solver="saga", l1_ratios=[0.1,0.5,0.9],
    Cs=20, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc", max_iter=5000, n_jobs=-1
)

pipe = Pipeline([("pre", pre), ("clf", logit)]).fit(X_train, y_train)

```

- **Explainable refit (statsmodels, after selecting variables):**
```python
import statsmodels.formula.api as smf

# Example: keep main effects + one interaction that proved useful
m = smf.glm(
    "y ~ weighting + days_to_alloc + risk + case_type + risk:case_type",
    data=train_df, family=smf.families.Binomial()
).fit(cov_type="HC3")  # or cluster={"groups": train_df["occupation"]}
print((m.params).apply(lambda b: (b, np.exp(b))))  # log-odds and odds ratios

```




&nbsp; 


- encoders.py — A leak-safe K-Fold target encoder for high-cardinality categoricals (e.g., occupation).
    - It computes means on each training fold only and maps them to the validation fold; at inference it uses a smoothed global mapping.

- features.py — Build a ColumnTransformer that:
    - imputes/scales numerics and optionally adds numeric interactions,
    - one-hot encodes moderate-card categoricals, and
    - target-encodes high-card columns (e.g., occupation if present).

- modeling.py — Add an advanced logistic pipeline (Elastic-Net), plus helpers for calibration and coefficient export.
(Existing GLM/backlog/survival code can remain—this adds to it.)
  
- diagnostics.py — Quick VIF & correlation helpers (run from CLI; VIF on numerics; correlation on selected fields).
    - **VIF/correlation to address multicollinearity; you can drop/combine features if VIFs are high.**

- test_logit_pipeline.py — Keeps CI happy and proves the pipeline fits and predicts.

How to run (end-to-end)
```bash
# After you replace/add files above and install requirements:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# Generate data (or use your real CSV)
python -m cli generate-data --rows 8000 --out data/raw/synthetic_investigations.csv

# Diagnostics (VIF/correlation)
python -m cli diagnostics --csv data/raw/synthetic_investigations.csv

# Advanced logistic model
python -m cli logit-advanced --csv data/raw/synthetic_investigations.csv

# Check outputs
ls reports/   # calibration plot + diagnostics
ls models/    # joblib + metrics csv
pytest -q     # quick tests
```

## C. How each task is addressed (techniques & code)

### C1) Backlog drivers (explanatory; feasibility)
- Technique: GLM (Poisson→NegBin) with investigators_on_duty, n_allocations, time trend t.
- Why: interpretable log-rate ratios; NegBin handles over-dispersion.
- Feasibility: run step change scenarios (e.g., Δ=+5 investigators). Compare baseline vs scenario series (CSV written).
- Run it
python -m g7_assessment.cli backlog-drivers --csv data/raw/synthetic_investigations.csv --delta 10

### C2) Time-to-PG-signoff (interval)
- Technique: Cox PH with OHE for case_type, risk, weighting.
- Why: handles censoring (open cases); hazard ratios are policy-friendly (e.g., “High risk ↗ hazard by 30%”).
- Where: function fit_survival_pg in modeling.py.

### C3) Forecasting backlog (daily → 90 days)
- Technique: SARIMAX with weekly seasonality (7) and exog drivers (investigators_on_duty, n_allocations).
- Why: captures autocorrelation/seasonality + allows scenarios on exogenous paths.
- Run
python -m g7_assessment.cli tsa-forecast --csv data/raw/synthetic_investigations.csv --days 90

### C4) High-risk applications (triage)
- Technique: elastic-net logistic with leak-safe preprocessing and target encoding for high-card variables (e.g., occupation).
- Why: balances sparsity & stability; TE avoids exploding one-hot columns and leakage.
- Run (legal review model as example)
python -m g7_assessment.cli legal-review-advanced --csv data/raw/synthetic_investigations.csv

### C5) Diagnostics & multicollinearity
- Technique: VIF on numerics and pairwise correlations; drop/merge highly redundant predictors (or keep both but rely on ridge/elastic-net).
- Run
python -m g7_assessment.cli diagnostics --csv data/raw/synthetic_investigations.csv

### C6) Micro-simulation handoff (what to export)
From the code above, you already get:
- Arrivals/backlog: reports/ forecast & historical daily counts (input for arrival processes).
- Service-time: survival model outputs (export duration quantiles/hazard by case_type/risk for DES).
- Routing: legal-review probability by slice (case_type, risk).
- Staffing: aggregate_staffing daily series (resource availability).


## D. Line-by-line code comments (example: elastic-net legal review)
```python
# Build preprocessing (impute/scale/OHE/TE) and fit a regularised logistic classifier
pre = build_preprocessor(df, y_name=y_name)      # <-- ColumnTransformer: numerics + cats + KFold target encoding
clf = LogisticRegressionCV(                      # <-- Cross-validated logistic regression
    penalty="elasticnet", solver="saga",         #     Use SAGA to support elastic-net
    l1_ratios=[0.1,0.5,0.9], Cs=20,              #     Grid over L1/L2 mix and regularisation strength
    cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
    scoring="roc_auc", max_iter=5000, n_jobs=-1, class_weight="balanced"  # <-- robust to class imbalance
)
pipe = Pipeline([("pre", pre), ("clf", clf)])    # <-- Single pipeline = leak-safe training & inference

Xtr, Xte, ytr, yte = train_test_split(           # <-- Hold-out test for unbiased evaluation
    X, y, test_size=0.2, stratify=y, random_state=random_state
)
pipe.fit(Xtr, ytr)                                # <-- Fits transformers on Xtr only (no leakage)

yprob = pipe.predict_proba(Xte)[:, 1]            # <-- Probabilities for metrics & thresholding
auc = roc_auc_score(yte, yprob)                  # <-- Discrimination
ap = average_precision_score(yte, yprob)         # <-- Precision-recall (good for rare outcomes)
brier = brier_score_loss(yte, yprob)             # <-- Calibration
frac_pos, mean_pred = calibration_curve(yte, yprob, n_bins=10, strategy="uniform")  # <-- Reliability curve
```

## E. Suggested talking points (policy-friendly)
- Staffing is influential, but not alone. GLM shows the marginal effect of investigators; we also quantify effects from case mix and reallocation. Feasibility scenarios show expected reduction per Δ staff.

- Interval reduction levers. Cox PH indicates which case types/risks prolong allocation/sign-off; those become process improvement targets.

- Operational forecasts. SARIMAX gives 90-day backlog trajectories with CI; planners can test what-ifs on staffing.

- Triage for verification. Elastic-net model identifies high-risk applications (and legal review propensity) to prioritise checks before cases become backlog.

- Simulation-ready. We export rates, distributions, and routing probabilities for a micro-simulation to test policy packages end-to-end.
&nbsp; 

## What this adds to the simulation model as input
- We export a small, consistent set of CSVs (under data/processed/microsim_inputs/) that your simulation can consume:
    - Arrivals of new concerns (investigation cases) by day and by month, split by case_type.
    - Backlog history by day (for initial conditions / validation).
    - Staffing series by day (investigators_on_duty, n_allocations).
    - Service-time distributions (time from receipt → PG sign-off) as Kaplan–Meier quantiles by case_type×risk (handles censoring).
    - Routing probabilities to legal review by case_type×risk.
    - A small metadata.json (timestamp, record counts) for auditability.

- microsim.py —
    - Purpose: Create CSV inputs for a downstream discrete-event / micro-simulation.
    - Exports arrivals, backlog, staffing, service-time quantiles (KM), and legal-review routing.

- test_microsim.py
### How to run
```bash
# 0) Install deps and pre-commit (if not already)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# 1) Generate synthetic data (or point to your real CSV)
python -m cli generate-data --rows 8000 --out data/raw/synthetic_investigations.csv

# 2) Export micro-sim inputs
python -m cli microsim-export --csv data/raw/synthetic_investigations.csv --outdir data/processed/microsim_inputs

# 3) Inspect outputs
ls data/processed/microsim_inputs
# arrivals_daily.csv, arrivals_monthly.csv, backlog_daily.csv, staffing_daily.csv,
# service_time_quantiles_pg_signoff.csv, routing_legal_review.csv, metadata.json

# 4) (Optional) Run tests
pytest -q
```

### Why these choices (to explain to stakeholders)

- Arrivals & backlog: provide the base demand and starting WIP for the simulator, sliced by case_type to capture mix.

- Staffing: a controllable resource in DES; the time series allows you to replay historical staffing or inject scenarios.

- Service-time distributions (KM): robust to right-censoring (open cases), so quantiles aren’t biased low.

- Routing to legal review: turns into a simple branch probability in the simulator; stratifying by case_type×risk keeps it policy-relevant.

- Metadata: makes runs auditable.

&nbsp; 


# 1) Forecasting incoming investigations (medium–long term)

**Short answer:** Yes—both can be used, but for medium–long-term forecasts of incoming investigations, a **Bayesian predictive model (with an explicit lag component inside it)** is the stronger primary approach. A “pure” lag-distribution is a great ingredient, not the whole meal.

## What each method does

### Lag-distribution models (two common uses)

* **Delay/convolution kernel:** today’s investigations are earlier *exposures* convolved with a delay distribution:
  ( y_t \approx \sum_{k=0}^{K} p_k \cdot \text{exposure}_{t-k} ), with (p_k \ge 0, \sum p_k = 1).
* **Distributed-lag regression:** regress ( y_t ) on several lagged covariates (seasonality, drivers) over multiple months.

**Strengths:** simple, interpretable delays; fast; great for nowcasting when a single upstream flow dominates.
**Limitations:** relies on separate forecasting of upstream flows; brittle with structural change; struggles with multiple interacting drivers.

### Bayesian predictive modelling (hierarchical dynamic counts)

Model arrivals directly as a probabilistic time series with covariates:

* **Likelihood:** Negative Binomial (over-dispersed counts) or Poisson-Gamma.
* **Linear predictor:**
  [
  \log \lambda_{t,g}
  = \underbrace{\text{local trend/level}}_{\text{state space}}

  * \underbrace{\text{seasonality (weekly, annual)}}_{\text{Fourier}}
  * \beta^\top X_{t,g}
  * \underbrace{\sum_{k=0}^{K}\gamma_k,\text{exposure}*{t-k}}*{\text{lag kernel}}
  * \varepsilon_{t,g}
    ]
    with (g) for team/case type; (X_{t,g}) can include concern types, pipeline signals, risk flags, staffing, legal markers, etc.

**Why it fits:** handles multiple drivers and their lags; yields calibrated uncertainty; partial pooling stabilises small groups; supports scenario analysis.

## Which is better for this objective?

> **Bayesian hierarchical dynamic count model with an embedded lag kernel** (primary choice).

Use the lag distribution **inside** the Bayesian model to capture realistic delays (e.g., concern → investigation).

## Quick decision guide

* **Strong upstream exposure + stable delay + short horizon?** Lag kernel baseline is fine.
* **Medium/long horizon, factor attribution, multiple teams/case types, need uncertainty?** Bayesian hierarchical DGLM with lag kernel.

## Practical next steps

1. Define weekly (y_{t,g}) = new investigations by team/case type.
2. Build upstream exposure series (E_t) (e.g., concerns/referrals).
3. Fit hierarchical NegBin state-space with Fourier seasonality, covariates (X_{t,g}), and regularised lag kernel on (E_{t-k}).
4. Validate with rolling-origin backtests (MASE/CRPS) + posterior predictive checks.
5. Deliver factor effects, forecast distributions, and scenario runs.
6. Feed predictive draws into your backlog micro-simulation alongside separately estimated duration/transition hazards.

**Bottom line:** treat the **lag distribution as a component**, and the **Bayesian model as the container**—that combination best matches your data richness and planning needs.

---

# 2) Breaking the forecast down by case type (risk/complexity)

**Short answer:** Both methods can work, but for forecasting **by case type**, a **Bayesian hierarchical model (optionally with lag components)** is the better primary choice. A “pure” lag-distribution is best kept as a feature or baseline.

## Why Bayesian wins for case-type breakdowns

* **Partial pooling for sparse/rare types:** shares strength across types/teams so small categories are stable.
* **Coherent splits:** joint modelling (e.g., logistic-normal or Dirichlet-multinomial) can ensure type-level forecasts sum to the total.
* **Richer drivers & interactions:** include seasonality, policy/staffing effects, upstream signals, and per-type lag kernels.
* **Better uncertainty:** credible intervals for both totals and types; calibratable via posterior predictive checks.
* **Robust to drift/structural change:** state-space components and priors cushion redefinitions of “risk/complexity.”

## When a lag-distribution is enough

Use it if you have **reliable, leading per-type exposure series** and a **stable delay** from exposure → investigation. It’s transparent and fast for near-term nowcasting, but:

* struggles with sparse types,
* is brittle under classification changes or shifting delays, and
* doesn’t guarantee totals = sum of categories without extra reconciliation.

## Modelling patterns (pick one)

### Top-down, coherent composition *(recommended for reporting)*

1. Forecast total arrivals (N_t) with a Bayesian dynamic NegBin.
2. Forecast category shares (\boldsymbol{\pi}*t) with a **logistic-normal** or **Dirichlet-multinomial** regression:
   (\text{logit}(\pi*{t,c}) = \alpha_c + s_c(t) + \beta_c^\top X_t + \sum_{k=0}^{K}\gamma_{c,k} E_{t-k} + \epsilon_{t,c}).
3. Draw (\mathbf{y}_t \sim \text{Multinomial}(N_t, \boldsymbol{\pi}_t)) for coherent splits.

### Bottom-up, hierarchical counts *(simple to fit)*

For each type (c): (y_{t,c} \sim \text{NegBin}(\lambda_{t,c})),
(\log \lambda_{t,c} = \mu_c + s_c(t) + \beta_c^\top X_t + \sum_{k}\gamma_{c,k} E_{t-k} + \eta_{t,c}),
with hierarchical priors over (\mu_c, \beta_c, \gamma_{c,\cdot}).
Sum draws for totals (optionally reconcile to enforce coherence).

**Include lag kernels** (\gamma_{c,k}) if you have upstream signals; otherwise trend/seasonality + covariates still perform well.

## Decision rule

* **Medium–long horizon, sparse categories, need calibrated intervals and coherent splits?** → **Bayesian hierarchical** (top-down composition or bottom-up with reconciliation), with lagged exposures as features.
* **Strong per-type leading indicators + stable delays + short horizon?** → Lag-distribution baseline; keep a Bayesian model for robustness.

**Bottom line:** use the lag distribution as a **feature**; the **Bayesian hierarchical framework** is the right container for accurate, coherent, and well-calibrated forecasts by risk/complexity.

&nbsp; 
<a name="future"></a>
# Future Work
- **AI‑driven hybrid**: emulator (GP), Bayesian optimisation and causal survival scaffold.
- **Streamlit UI** for non‑coders.
- Extra **CI** (CodeQL, Dependabot, pip‑audit, stale, release‑please).
- Expanded docs on **AI & Optimisation** and non‑black‑box governance.

&nbsp; 

&nbsp; 

<a name="licence"></a>
# Licensing
- **Code**: MIT (`LICENSE`).
- **Docs & non‑code**: Open Government Licence v3 (`docs/DATA_LICENSE.md`).

See `docs/OPG_Investigations_Backlog_Documentation.md` for the full specification.