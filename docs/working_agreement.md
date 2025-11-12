# Working Agreement — Data Science on GitHub (One Page)

**Purpose**  
Enable fast, safe, and transparent collaboration across analysts, data scientists, data engineers, and policy colleagues.

## 1) Ways of Working
- **Single source of truth:** `main` is always releasable. All changes via Pull Request (PR).
- **Work tracking:** Every change links to a GitHub Issue with clear acceptance criteria.
- **Definition of Done (DoD):** CI green, reviewer approval, updated docs, and release notes written in plain language.

## 2) Roles
- **Author:** builds change, keeps PR small, writes tests and docs.
- **Reviewer(s):** ensure correctness, clarity, and user impact. At least one reviewer from the relevant area (data eng/model/policy).
- **Maintainer:** merges after gates pass; responsible for release tagging and changelog.

## 3) Branching & PRs
- **Branch naming:** `feat/`, `fix/`, `docs/`, `chore/`.
- **Small, independent PRs:** ≤ ~300 LOC where possible; include before/after screenshots for plots/reports.
- **Commit style:** Conventional Commits (`feat:`, `fix:`) for clean changelogs.
- **Protected `main`:** require PR, passing CI, and ≥1–2 approvals; block force-push.

## 4) Quality Gates (CI & Local)
- **Local pre-commit:** Black, Ruff, nbstripout, YAML checks before commit.
- **CI (required):** install → lint → tests → data schema checks (Pandera/GE) → coverage threshold → (optional) type checks.
- **Artifacts:** models/reports carry metadata (data snapshot, code version, timestamp).

## 5) Data Handling & Ethics
- **No personal data in repo.** Use synthetic/anonymised data for examples.
- **Data contracts:** validate input tables against a schema; fail CI if contract breaks.
- **Model cards:** record purpose, data, metrics, fairness checks, caveats, and owners.

## 6) Reviews & Documentation
- **PR template:** Problem → Approach → Evidence (metrics/plots) → Risks → Rollout.
- **CODEOWNERS:** auto-request area experts; reviewers respond within 1–2 working days.
- **Docs:** Update `README.md` and `docs/` for user-visible changes; publish via GitHub Pages.

## 7) Releases & Environments
- **Versioning:** semantic (MAJOR.MINOR.PATCH). Tag releases.
- **Changelog:** auto-generated from Conventional Commits; add stakeholder-friendly summary.
- **Deployment:** only from `main`; roll back by reverting PRs or redeploying prior tag.

## 8) Communication Cadence
- **Stand-ups (brief):** blockers, risks, decisions needed.
- **Weekly demo:** show working increments to stakeholders; collect feedback.
- **Decisions:** log in short ADRs (Architecture Decision Records) within `/docs/adr/`.

## 9) Escalation & Risk
- **Red builds:** PR author fixes or reverts within the day.
- **Breaking data contract:** stop-the-line; hotfix PR with tests.
- **Security:** secret scanning on; rotate keys immediately on leaks.

---

## First-Week Rollout Checklist
1. **Enable branch protection** on `main` (require PR, CI checks, 1–2 reviews).
2. **Add CODEOWNERS** to map folders to teams (data eng / modelling / policy).
3. **Create CI** workflow (`.github/workflows/ci.yml`): lint, tests, schema checks, coverage.
4. **Install pre-commit** and run `pre-commit autoupdate`; share `make setup` or `README` steps.
5. **Create Issue & PR templates** and a **Project board** (Backlog → In Progress → Review → Done).
6. **Publish docs** via MkDocs + GitHub Pages (optional but recommended).
7. **Define metrics** to monitor (AUC/PR-AUC, calibration, latency, cost) and add a smoke test dataset.

## Interview Sound Bite
“Quality is built-in: we use pre-commit locally and gate merges with CI (lint, tests, schema checks). We work in small PRs against protected `main`, with CODEOWNERS and a clear PR template. A shared Project board gives visibility; releases are tagged with plain-language notes for policy colleagues. Model cards and data contracts make our work explainable and auditable.”
