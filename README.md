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

&nbsp; 

&nbsp; 


<a name="setup"></a>
# Repository and Git Setup
- Branching: feature branches (feat/, fix/, refactor/), protected main.

- CI gates: ruff + black + pytest run on every push/PR via GitHub Actions.

- PR hygiene: PR template, CODEOWNERS, issue templates, semantic commit summaries.

- Project management: create a GitHub Project board (Backlog → In Progress → Review → Done), tag issues (data, model, infra, docs).

- Quality & testing: schema checks, unit & property tests, reproducibility (seeds), pre-commit hooks, doc pages for QA and ethics.
All of the above is pre-wired in the repo so you can demonstrate collaborative, production-ready habits.

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
│   └── g7_assessment/       # ETL, feature eng, models, viz, CLI (Click)
├── tests/                   # Automated unit testing, Pytest unit tests (data & modeling)
│   └── test_data_quality.py # pytest tests for Data Quality
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

## Quickstart
```bash
# 1) Create env and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pip install -U pip
pip install -e ".[dev]"

pre-commit install


# 2) Generate synthetic data (replace with your real CSV when ready)
python -m g7_assessment.cli generate-data --rows 8000 --out data/raw/synthetic_investigations.csv

# 3) EDA (saves plots & tables to ./reports)
python -m g7_assessment.cli eda --csv data/raw/synthetic_investigations.csv

# 4) Train all models (saves into ./models)
python -m g7_assessment.cli train --csv data/raw/synthetic_investigations.csv

# 5) Forecast 90-day backlog + plot
python -m g7_assessment.cli forecast --csv data/raw/synthetic_investigations.csv --days 90

# 6) Staffing scenario: add 10 investigators
python -m g7_assessment.cli simulate --csv data/raw/synthetic_investigations.csv --delta-investigators 10


make lint && make test
streamlit run app.py
mkdocs serve

```
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

&nbsp; 

&nbsp;

<a name="setup"></a>
# Quality Assurance
Quality starts locally with pre-commit (Black/Ruff/nbstripout). On GitHub, we gate merges with protected branches, required reviews, and Actions CI (lint, tests, data checks). We manage work with Projects, issue/PR templates, CODEOWNERS, and semantic releases. For data, we use schema contracts (Pandera/Great Expectations), reproducible environments, and artifact versioning. We communicate findings via model cards, docs, and stakeholder-focused release notes.
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

**Objective:**
1. 

2. 


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



# Future Work
- **AI‑driven hybrid**: emulator (GP), Bayesian optimisation and causal survival scaffold.
- **Streamlit UI** for non‑coders.
- Extra **CI** (CodeQL, Dependabot, pip‑audit, stale, release‑please).
- Expanded docs on **AI & Optimisation** and non‑black‑box governance.

## Licensing
- **Code**: MIT (`LICENSE`).
- **Docs & non‑code**: Open Government Licence v3 (`docs/DATA_LICENSE.md`).

See `docs/OPG_Investigations_Backlog_Documentation.md` for the full specification.