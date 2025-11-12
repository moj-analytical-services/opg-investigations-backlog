# OPG Investigations Backlog – End-to-End Documentation

**Date:** 2025-10-27

---
## Step-by-step guidance

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

## 6) Forecasting the Next 90 Days
Create a future calendar by unit (investigator×team×role), attach features, draw from the posterior predictive, and aggregate to team/role/org totals.

## 7) Validation & Reporting
Assess accuracy and calibration, inspect PPCs, and report credible intervals and service‑level implications (e.g., backlog burn‑down).

## 8) Running in JupyterLab
Install `pandas`, `numpy`, `pymc`, `bambi`, `arviz`, and plotting libs. Run the notebook top‑to‑bottom; outputs write to `/mnt/data/forecasts/` by default.

## 9) Glossary
**Investigated cases:** cases worked/completed that day. **WIP:** active, not yet closed. **Backlog:** unassigned/unworked awaiting allocation. **Partial pooling:** shares information across groups while allowing differences.

---

## 10) Why Poisson–Gamma (Negative Binomial) for daily case counts?

Daily counts usually show **overdispersion** (variance > mean) and heterogeneity (availability, complexity, outages). A **Poisson–Gamma mixture** captures this and is equivalent to the **Negative Binomial** likelihood we use.

### For data scientists (math/stats)
- **Conditional model:** \(y \mid \lambda \sim \text{Poisson}(\lambda)\).  
- **Heterogeneity:** \(\lambda \sim \text{Gamma}(r, \beta)\) (shape \(r\), rate \(\beta\)).  
- **Marginal:** \(y \sim \text{NegBin}(r, \tfrac{\beta}{\beta+1})\). With mean \(\mu = r/\beta\):  
  \[ \operatorname{Var}(y) = \mu + \alpha \mu^2, \quad \alpha = 1/r. \]
  This quadratic mean–variance relation matches operational counts better than Poisson (Var=Mean).
- **Regression link:** \(\log \mu_i = x_i^\top \beta + b_{\text{inv}[i]} + b_{\text{team}[i]} + b_{\text{role}[i]} + \cdots\).  
  NB with overdispersion \(\alpha\) equals Poisson–Gamma under a log link. Random intercepts give **partial pooling**.
- **Conjugacy (intuition):** In simple cases, \(\lambda \mid y \sim \text{Gamma}(r{+}y,\beta{+}1)\). HMC/NUTS fits the regression efficiently.

**Alternatives:** Poisson–lognormal, Zero‑Inflated NB. Start with NB; extend if PPC/coverage suggests.

### For non‑experts (plain English)
- We model **how many cases** are worked each day. Real life is noisy: staff, holidays, and tricky cases vary.
- Poisson assumes the **spread equals the average**—too optimistic. NB allows extra variability so forecasts aren’t over‑confident.
- Think: each day/team has a **speed setting** that can change. NB captures that and gives **realistic ranges** to plan against.

*Practical note:* In the code this appears as `NegativeBinomial` with a log link and an estimated **overdispersion** parameter (`alpha`/`r`). Larger overdispersion ⇒ wider intervals.

---

## 11) Why Bayesian (PyMC & Bambi)?

### For data scientists (math/stats)
- Bayesian inference estimates the **full posterior** \(p(\theta \mid y)\), not just a point estimate—supporting **credible intervals** and decision‑making under uncertainty.
- **Hierarchical priors** implement **partial pooling**, improving estimates for sparse units and reducing overfitting.
- Priors act as **regularisation**, stabilising weekday/holiday and group effects; **posterior predictive checks** assess calibration.
- Implemented with **PyMC** (NUTS/HMC) or **Bambi** (formula interface on top of PyMC).

### For non‑experts (plain English)
- Shares information across teams/people so small groups don’t swing wildly.
- Produces **ranges** (not just single numbers) so you can plan for best/typical/worst cases.
- Transparent and extensible: easy to add drivers like backlog, inflow, or leave patterns.
