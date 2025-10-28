# OPG Investigations Backlog – End-to-End Documentation

**Date:** 2025-10-27

---

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
