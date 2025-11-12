# Interview Q&A (G7 Data Science Assessment)

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
