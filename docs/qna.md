# Q&A

## Approach
**Q:** How did you scope the problem and define success?
**A (short):** I decomposed the brief into three measurable questions: (1) factors that drive daily backlog, (2) factors that drive time-to-PG-signoff, and (3) probability of legal review. I defined KPIs as MAE for backlog forecasts, concordance for survival, and AUC for legal review. I prioritised interpretability and policy-actionability before complexity.

**Q:** Why Poisson/Negative Binomial for backlog?
**A:** Backlog counts per day are non-negative integers with overdispersion. I start with a Poisson GLM; if variance exceeds mean materially, I switch to Negative Binomial. Coefficients remain interpretable as log-rate ratios for policy levers (e.g., staffing).

**Q:** How do you test the “staffing reduces backlog” hypothesis?
**A:** Include `investigators_on_duty` as a covariate, test significance and effect size, check partial dependence, and run counterfactual simulations (+/- staff). Also guard against confounding with time and case mix by adding calendar terms and case-type proportions.

**Q:** How do you address multicollinearity?
**A:** Correlation heatmaps, VIF, and penalisation (ridge). Where variables encode similar constructs (e.g., team and case type), I use one-hot with reference categories and drop highly collinear features.

**Q:** Missing data & imputation?
**A:** Date fields: derive intervals and use event indicators; for categorical NULLs, use explicit `"Unknown"` category. For numeric gaps, median or model-based imputation scoped within train folds.

**Q:** Fairness & ethics?
**A:** Report subgroup performance; avoid features that proxy protected attributes; document risks and mitigations in **docs/ethics.md**.

**Q:** What further work with more time?
**A:** Joint hierarchical model for teams; calendar effects with holidays; richer simulation (DES); Bayesian models for full uncertainty propagation to policy scenarios.

## Findings (from synthetic run)
- Staffing increases are associated with reduced backlog levels; effect is strongest in high case-load weeks.
- Case types such as `TPO` and `Fraud` show longer time-to-signoff and higher legal review propensity.
(Replace with real findings when run on live data.)

## Techniques
- Count regression (Poisson/NegBin), Cox PH survival, logistic classifier, ETS forecasting.
- Feature selection via univariate screening + regularised GLMs.
- CI/CD, testing, and documentation for production readiness.

# CI Wrapper & Smoke Tests — What *Devs* Get and Why It’s Useful

## Why devs love a CI wrapper

A **CI wrapper** (a Makefile target or `scripts/ci.sh`) makes the *exact same command* work **locally** and **in GitHub Actions**.

### Concrete benefits for developers

* **Local ↔ CI parity:** One source of truth for install → lint → tests → (tiny) run. Fewer “works on my machine” bugs.
* **Fast onboarding:** New joiner runs `./scripts/ci.sh` or `make setup && make test` and is instantly aligned with the team.
* **Fewer config snowflakes:** No re-teaching “which linter flags?” or “which Python?”—it’s all encoded.
* **Reviewable automation:** PR reviewers can run the wrapper locally before reviewing; less back-and-forth.
* **Reproducible debugging:** When CI fails, devs re-run the *same* wrapper to reproduce and fix quickly.
* **Portable between repos:** Copy the wrapper to new projects; teams converge on a common dev experience.

### Small enhancements devs typically add

* **Caching:** In Actions, enable pip cache to speed installs.
* **Matrix testing:** Add Python versions (e.g., 3.10, 3.12) without changing the wrapper.
* **Selective stages:** Flags like `FAST=1` to skip slow checks locally but run them in CI nightly.

```yaml
# Example pip cache in GitHub Actions
- uses: actions/setup-python@v5
  with: { python-version: "3.12", cache: "pip" }
- run: scripts/ci.sh
```

---

## How smoke tests help devs (and reviewers)

**Smoke tests** are tiny, <60-second checks that exercise the **critical path** (imports, synthetics, key outputs).

### Concrete benefits for developers

* **Rapid feedback loop:** Run on every push → immediate red/green before you context-switch.
* **Early regression catchers:** Breaks from refactors (renamed columns, CLI args) show up instantly.
* **Safe refactoring:** When smokes stay green, you’re confident to proceed; deeper tests can run later.
* **Stable code reviews:** Reviewers see a quick ✅ and focus on design, not plumbing.
* **Low friction in notebooks-to-code flow:** Smokes prove your notebook-wrapped pipeline still runs end-to-end.

### What to include in smoke tests (data-science flavoured)

* **Imports:** Modules import cleanly (no circular deps).
* **Tiny synthetic run:** 200–500 rows → run `prep → intervals → write CSVs`.
* **Key artefacts exist:** `engineered.csv`, `backlog_series.csv`, `annual_stats.csv`.
* **Fast-only marker:** `@pytest.mark.smoke` so CI can run just these in PRs; run the full suite nightly.

```bash
# PR CI: fast signal
pytest -q -m "smoke" --maxfail=1
# Nightly: everything (including slow/backtests)
pytest -q
```

---

## Developer workflow scenarios (how it helps day-to-day)

| Scenario             | Without wrapper + smokes           | With wrapper + smokes                                |
| -------------------- | ---------------------------------- | ---------------------------------------------------- |
| New dev sets up      | Asks for docs; mismatched tools    | `make setup && ./scripts/ci.sh` → aligned in minutes |
| Refactor a function  | Breaks CI later; hard to reproduce | Smokes fail locally first; fix before push           |
| Review a PR          | “Does this even run?”              | CI wrapper green; reviewer focuses on design         |
| Fix intermittent bug | Works locally but not in CI        | Same command both places → easier repro              |
| Add a new check      | Update many places                 | Update wrapper once; local + CI stay in sync         |

---

## Practical tips for dev teams

* **Keep smokes tiny:** Aim for **<60 seconds** total on a laptop & in CI.
* **Mark slow tests:** `@pytest.mark.slow` and run them nightly.
* **Use synthetic data:** Avoid PII, avoid flaky network calls, ensure determinism via fixed seeds.
* **Fail fast:** Lint first (Ruff/Black), then smokes. Fast red saves time.
* **Pre-commit + CI wrapper:** Pre-commit catches formatting locally; wrapper enforces it in CI.

---

## Anti-patterns to avoid

* **Duplicating CI logic** (different steps in README, Makefile, workflow) → drift and confusion.
* **Monolithic “tests” that do everything** → slow feedback; devs stop running them.
* **Network-dependent smokes** (APIs, S3) → flaky CI; mock or generate synthetics instead.
* **Hidden env state** (local `.env` magic) → won’t reproduce in CI; encode env in the wrapper.

---

## Ready-made snippets (drop into your repo)

**CI wrapper:** `scripts/ci.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip
pip install -r requirements.txt
ruff check .
black --check .
pytest -q -m "smoke" --maxfail=1
```

**Smoke marker:** `pytest.ini`

```ini
[pytest]
markers =
    smoke: very fast checks for CI gating
```

**Example smoke tests:**
`tests/test_imports_smoke.py` and `tests/test_pipeline_smoke.py` (from my previous message) exercise imports and a tiny synthetic run producing the key artefacts.

---

## One-liners you can use in the interview

* “Our **CI wrapper** ensures the *same* steps run locally and in CI—no ‘works on my machine’ drift.”
* “We gate PRs with **smoke tests**: a tiny synthetic run proves the critical path creates the key artefacts.”
* “Devs get **fast feedback** and stable reviews; heavier tests run **nightly** on a fresh synthetic dataset.”


# CI Wrapper & Smoke Tests — Copy-Ready Guide

## What’s a CI wrapper?

A **CI wrapper** is a thin, reusable script/target that standardises what CI runs (install, lint, tests, tiny demo run). Instead of hard-coding many steps in your GitHub Action, you call **one thing** that devs can run locally too. That gives you **one source of truth** for quality checks.

### Why it’s useful

* Same commands **locally and in CI** → fewer “works on my machine” surprises.
* Easier to maintain: update the wrapper once; CI and devs both pick it up.
* Predictable, fast feedback for reviewers and policy colleagues (green/red gate).

---

## Drop-in CI wrapper (script + workflow)

**Add this script:** `scripts/ci.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
pip install -r requirements.txt

# Lint/format (fail fast)
ruff check .
black --check .

# Fast tests (smoke only)
pytest -q -m "smoke" --maxfail=1
```

> Make it executable:

```bash
chmod +x scripts/ci.sh
```

**Call it from GitHub Actions:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: scripts/ci.sh
```

**Run locally the same way:**

```bash
./scripts/ci.sh
```

> If you prefer **Makefiles**, your `make lint` / `make test` targets can serve as the CI wrapper—have Actions call `make` instead of the shell script.

---

## What are smoke tests?

**Smoke tests** are ultra-fast checks that the **critical path** works at a basic level (imports, a tiny synthetic run, key files created). They don’t prove full correctness; they **catch breakages early** in <60 seconds.

### Why they’re useful

* Keep PRs moving: quick green/red signal.
* Catch obvious regressions (bad import, missing column, CLI arg rename).
* Cheap to run on every push; run deeper tests nightly.

---

## Drop-in smoke tests for your repo

**Create:** `pytest.ini`

```ini
[pytest]
markers =
    smoke: very fast checks for CI gating
```

**Create:** `tests/test_imports_smoke.py`

```python
import pytest

@pytest.mark.smoke
def test_wrappers_import():
    import g7_assessment.preprocessing as p
    import g7_assessment.intervals as itv
    import g7_assessment.analysis_demo as demo
    import g7_assessment.distributions as dist
    assert hasattr(p, "engineer")
    assert hasattr(itv, "build_backlog_series")
    assert hasattr(demo, "last_year_by_team")
    assert hasattr(dist, "interval_change_distribution")
```

**Create:** `tests/test_pipeline_smoke.py`

```python
import pytest
from pathlib import Path
import pandas as pd

@pytest.mark.smoke
def test_tiny_synth_prep_and_intervals(tmp_path: Path):
    # 1) Generate a tiny synthetic dataset in-memory (fast)
    from g7_assessment.synth import generate_synthetic
    df = generate_synthetic(n_rows=500, seed=123)
    raw_csv = tmp_path / "raw.csv"
    df.to_csv(raw_csv, index=False)

    # 2) Run prep using your existing logic (via wrappers)
    from g7_assessment.preprocessing import load_raw, engineer
    raw, colmap = load_raw(raw_csv)
    eng = engineer(raw, colmap)
    assert len(eng) > 0
    assert "date_received_opg" in eng.columns

    # 3) Build minimal interval artefacts
    from g7_assessment.intervals import build_event_log, build_backlog_series
    events = build_event_log(eng)
    backlog = build_backlog_series(eng)

    # Sanity: not empty (may be small)
    assert events is not None
    assert backlog is not None
```

**Create:** `tests/test_cli_smoke.py`

```python
import pytest, subprocess, sys
from pathlib import Path

@pytest.mark.smoke
def test_cli_run_all_smoke(tmp_path: Path):
    raw_csv = tmp_path / "raw.csv"

    # Generate tiny synthetic file via CLI (fast)
    cmd = [sys.executable, "-m", "g7_assessment.synth",
           "--rows", "400", "--seed", "101", "--out", str(raw_csv)]
    subprocess.run(cmd, check=True)

    # Run the full wrapper pipeline with minimal outputs
    cmd = [sys.executable, "-m", "g7_assessment.cli_nbwrap",
           "run-all", "--raw", str(raw_csv), "--outbase", str(tmp_path),
           "--interval-col", "days_to_alloc", "--group", "case_type"]
    subprocess.run(cmd, check=True)

    # Check key artefacts were created
    assert (tmp_path / "data/processed/engineered.csv").exists()
    assert (tmp_path / "data/processed/backlog_series.csv").exists()
    assert (tmp_path / "reports/annual_stats.csv").exists()
```

---

## How to use both together (pattern)

* On **every push/PR**: CI runs the **wrapper** → runs **smoke tests** → fast signal.
* **Nightly**: run the **full pipeline** on a fresh synthetic dataset (use your nightly workflow), plus any heavier tests (backtests, schema validations).

**Optional add-on:** mark heavy tests with `@pytest.mark.slow` and:

* PR CI: `pytest -q -m "smoke"`
* Nightly: `pytest -q` (everything)

---

## TL;DR (interview-ready)

> “We use a **CI wrapper** so devs and CI run the **same** steps. We gate PRs with **smoke tests**—tiny, end-to-end checks that a synthetic dataset runs through our pipeline and produces key artefacts. It keeps feedback fast and dependable; deeper tests run nightly.”


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