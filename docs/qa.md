# Quality Assurance & Testing

- **Unit tests** with `pytest` for data transforms and pipelines.
- **Property-based tests** with `hypothesis` for robustness (e.g., date ordering, idempotent cleaning).
- **Static checks**: ruff (lint), black (format).
- **CI** runs lint & tests on every PR/commit to `main`.
- **Data contracts**: schema checks (non-null, allowed categories) in `tests/`.
- **Reproducibility**: random seeds, pinned dependencies in `requirements.txt`.
- **Quality & testing**: schema checks, unit & property tests, reproducibility (seeds), pre-commit hooks, doc pages for QA and ethics.
All of the above is pre-wired in the repo so you can demonstrate collaborative, production-ready habits.
- **We use a CI wrapper so devs and CI run the same steps. We gate PRs with smoke tests—tiny, end-to-end checks that a synthetic dataset runs through our pipeline and produces key artefacts. It keeps feedback fast and dependable; deeper tests run nightly.**


# RACI, Backlog & Gantt — Practical Guide (with your migration project as the example)

## What is RACI (in plain English)?

RACI is a lightweight way to make ownership unmistakable:

* **R — Responsible:** The doer(s) who complete the task or deliverable.
* **A — Accountable:** Exactly one person who signs off and is ultimately answerable.
* **C — Consulted:** People whose input you actively seek before decisions (two-way).
* **I — Informed:** People you keep in the loop after decisions/updates (one-way).

> Keep it light (one line per deliverable) and visible (on the board and timeline).

---

## How you set up a lightweight RACI, a shared backlog, and a simple Gantt

### 1) Lightweight RACI (baked into your ways of working)

* **Where:** GitHub Projects (or JIRA/Trello) with a custom **RACI** field and a required **Owner** field.
* **How:** For each epic/issue, set:

  * **Owner (R):** a single named person (e.g., `@jane.deng`).
  * **RACI field:** `R: Jane | A: You | C: DS Engineer, Model Owner | I: SAS Exit Lead, Governance`.
  * **Blocked?** Add a red `BLOCKER` label, a short note, and @mention the blocker’s owner.
* **Why it worked:** Everyone could see who does, who decides, who advises, and who’s just kept updated—no guessing.

#### Example RACI for your key migration deliverables

| Deliverable / Decision                            | Responsible (R)        | Accountable (A)                   | Consulted (C)                   | Informed (I)                   |
| ------------------------------------------------- | ---------------------- | --------------------------------- | ------------------------------- | ------------------------------ |
| Legacy → Python/Athena data ingestion             | Data Engineer (MoJ DS) | **You (Senior OR Analyst)**       | Former model holder; Data Owner | SAS Discontinuation Lead; Team |
| Transform & feature layer (Python)                | **You**                | **You**                           | Senior OR Adviser; DS Engineer  | Governance Board Secretariat   |
| Parallel run (SAS vs Python) & tolerance criteria | You + Junior Analyst   | **You**                           | Model Owner; QA reviewer        | Wider OR & DS communities      |
| QA pack (AQuA-aligned) + unit tests               | Junior Analyst         | **You**                           | DS Engineer; QA lead            | Governance Board               |
| Documentation & lineage                           | Junior Analyst         | **You**                           | Previous post holder; Analysts  | Downstream forecasting team    |
| Governance sign-off & release                     | **You**                | Deputy Director (Gov Board Chair) | DS Lead; Finance/Model Owner    | Programme Board; SAS Exit Lead |

> Tip: **Accountable** is always one person (often you), even if several people are **Responsible**.

---

### 2) Shared backlog (owners, blockers, deadlines stay obvious)

* **Columns:** Backlog → In Progress → In Review → Done (plus a **Blockers** swimlane).
* **Required fields per ticket:** Owner (R), RACI string, Due date, Dependency link (`blocked by #142`).
* **Labels:** `BLOCKER`, `RISK`, `GOV-SIGNOFF`, `PARALLEL-RUN`, `QA`, `DOCS`.

**Example backlog items**

1. **EPIC:** Stand up Athena ingestion pipeline

   * **R:** DS Engineer | **A:** You | **C:** Data Owner | **I:** SAS Exit Lead
   * **Due:** 15 Mar
   * **Blocked by:** IAM access (#131)

2. **Story:** Implement unit tests for transform functions

   * **R:** Junior Analyst | **A:** You | **C:** DS Engineer | **I:** Governance
   * **Due:** 22 Mar
   * **Notes:** Start with happy paths; take edge cases from legacy Excel AQA log.

3. **Story:** Parallel run: variance report (SAS vs Python)

   * **R:** You | **A:** You | **C:** Senior OR Adviser; Model Owner | **I:** Wider team
   * **Due:** 29 Mar
   * **Acceptance:** < 2% variance on KPIs for 3 consecutive runs.

4. **Task:** Lineage & assumptions doc (living)

   * **R:** Junior Analyst | **A:** You | **C:** Previous post holder | **I:** Forecasting team
   * **Due:** 25 Mar
   * **Tip:** Link to DAG sketch and data contracts.

5. **Decision ticket:** Tolerance thresholds for sign-off

   * **R:** You | **A:** You | **C:** Governance Chair; DS Lead | **I:** Programme Board
   * **Due:** 01 Apr
   * **Outcome:** Recorded in decision log; attached to release tag.

> Also mirror **R** in GitHub **CODEOWNERS** for `/etl/`, `/qa/`, `/docs/` to make reviews automatic.

---

### 3) Simple Gantt (expose deadlines & dependencies without ceremony)

Keep it lean (spreadsheet or Project “timeline” view):

* **Rows:** Epics/milestones only (5–8 lines).
* **Columns:** Start, End, Owner (R), Dependency, Status, Risk (RAG).
* **Dependencies:** Arrow or a “Depends on” column.

**Minimal Gantt snapshot (illustrative)**

| Milestone                      | Owner (R)      | Start  | End        | Depends on      | Risk (RAG) |
| ------------------------------ | -------------- | ------ | ---------- | --------------- | ---------- |
| Data ingestion ready (Athena)  | DS Engineer    | 25 Feb | **15 Mar** | IAM access      | Amber      |
| Transform layer (Python)       | You            | 10 Mar | **24 Mar** | Ingestion       | Green      |
| Unit tests & QA pack           | Junior Analyst | 12 Mar | **22 Mar** | Transform stubs | Green      |
| Parallel run & variance report | You            | 18 Mar | **29 Mar** | QA pack         | Amber      |
| Governance sign-off            | You            | —      | **01 Apr** | Variance report | Green      |
| Release & handover             | You            | —      | **05 Apr** | Sign-off        | Green      |

> When **IAM access** slipped, you applied the `BLOCKER` label on the ticket and Gantt, @mentioned the platform team, and added a mitigation (temporary read-only creds) to keep transform work moving.

---

## How this combo reduced resistance and sped delivery

* **Clarity reduces conflict:** Clear **Accountable** owners and visible due dates made decisions faster and kinder. The staged parallel-run was accepted because sign-off roles and tolerance criteria were explicit.
* **Inclusion with structure:** RACI made it safe for a **neurodivergent colleague** and juniors to own visible pieces (docs/QA). Previous model holders were on the **Consulted** list from the start.
* **Blockers surface early:** A red `BLOCKER` label plus dated note prompted help from the right people—no long email chains.
* **Governance loves evidence:** Clean RACI per deliverable, short Gantt, backlog with QA/variance gates → approvals felt low-risk.

---

## Quick templates you can reuse

**RACI line (paste into an issue)**

```text
R: <name> | A: <name> | C: <roles/names> | I: <roles/names> | Due: <date> | Depends on: #<id>
```

**Decision ticket (for things people argue about)**

```text
Title: <Decision>
Context: <why this matters now>
Options considered: <A / B / C>
Decision: <chosen option> | Accountable: <name>
Evidence: <links to runs, variance report, QA>
Effective from: <release tag / date>
```

**Backlog columns**

```text
Backlog | In Progress | In Review | Done | (swimlane: Blockers)
```

**Common labels**

```text
BLOCKER, RISK, GOV-SIGNOFF, PARALLEL-RUN, QA, DOCS
```

---

## Extra tips (from your project)

* Use **show-and-tell** demos to invite critique early; capture decisions in living docs.
* Run a **SWOT** session to turn resistance into co-ownership of the solution.
* Keep **Accountable** to one person per deliverable; avoid shared accountability.
* Summarise key messages, inputs, risks, and assumptions on **one page**; publish on the intranet to widen access.
