# Quarterly Demand and Modelling Planning Working Group  
**Technical Report, Insights & AI Recommendations**  
**Date of Meeting:** 6 August 2025

---

## 1. Executive Summary

This report brings together quantitative and qualitative evidence from recent working group sessions, forecasting models, and the latest demand and investigations data. It evaluates current and projected trends for Lasting Power of Attorney (LPA) receipts, deputyship caseload, and investigations. The impact of imminent fee changes and media interventions (notably Martin Lewis broadcasts) are examined, and next-step recommendations are made with a focus on further leveraging AI and data science to improve forecasting and operational planning.

---

## 2. LPA Receipts: Forecasts, Drivers, and Fee Impacts

### 2.1. Current Forecasts

- **Average Daily Receipts (2025/26):**
  - **Forecast Range:** 5,600 – 5,800 per day, with the central estimate rising to 5,700–5,800 (from 5,600 in the previous quarter):contentReference[oaicite:0]{index=0}.
  - **Quarterly Growth:** Q1 2025/26 receipts were 4.2% higher than Q1 2024/25, with every month in Q1 surpassing the prior year’s equivalent month.

- **Demand Drivers:**
  - The recent rise is closely linked to the **Martin Lewis broadcast** in March 2025. Previous experience suggests such spikes are typically not sustained: 2024 saw a similar surge followed by gradual decline over 12 months.

### 2.2. Impact of Fee Increase

- **Short-Term:**  
  - The planned LPA fee increase (announcement in August, implementation in November 2025) is expected to **induce a spike in demand** as applicants seek to "beat" the increase.
  - Historical analogy: The 2011 fee rise produced a notable short-term spike (+7%) followed by a return to trend.
  - If **media amplification** (Martin Lewis) occurs, the demand spike could be larger and more persistent, lifting annual averages closer to 5,800/day:contentReference[oaicite:1]{index=1}.

- **Long-Term:**  
  - International and domestic evidence (e.g., divorce application fees) suggests **long-term demand is relatively inelastic** to fee changes if fees are a small proportion of total costs and the need is strong.
  - Nonetheless, perception matters: If the fee increase is salient or seen as burdensome, a modest longer-term dip in demand is possible:contentReference[oaicite:2]{index=2}.

### 2.3. Financial Modelling and Risk

- **Receipts and Revenue:**  
  - The average receipts assumption for financial planning is 5,600–5,700/day. A move to 5,800/day is possible with sustained media effect:contentReference[oaicite:3]{index=3}.
  - **Estimated additional income** from a £10 fee rise in Q4 2025/26: up to £2 million, but actual impact will depend on both the spike and post-implementation drop-off:contentReference[oaicite:4]{index=4}.

- **Risk Approach:**  
  - The group agreed to take a **cautious baseline** (5,700/day), treating higher demand as "opportunity" to be released only if it sustains over several weeks.
  - Ongoing monitoring of the demand trajectory, particularly in response to media events, is required.

---

## 3. Deputyship Forecasts: Caseload, Data, and Long-Term Change

### 3.1. Caseload Dynamics

- **Forecasting Approach:**  
  - The group now relies on **clearly defined active caseloads** at fixed points in time, avoiding unreliable separate counts for new orders and terminations due to legacy data issues:contentReference[oaicite:5]{index=5}.

- **Key Figures:**  
  - **April 2025 Caseload:** Used as baseline.
  - **Example forecast for April 2026:**
    - 66,525 (if new orders and terminations remain as in 2025)
    - 64,662 (if new orders drop 20%)
    - 68,388 (if new orders rise 20%):contentReference[oaicite:6]{index=6}.
  - Trend: Caseload is likely to **remain stable or increase** unless there is a significant sustained fall in new orders.

### 3.2. Drivers and Demographics

- **Age Distribution:**  
  - Marked **decline in new deputyships for ages 80–95**, potentially linked to increased LPA take-up in that cohort.
  - Areas with low LPA uptake show higher deputyship rates, suggesting **substitution effects**.

- **Implication:**  
  - As LPA take-up grows, especially among older people, new deputyships may plateau or decline, but this is uneven regionally and demographically.

### 3.3. Forecasting Limitations and Improvements

- **Forecast Horizon:**  
  - Current models are short-term and rely on crude % assumptions for new orders.
  - **Longer-term improvements** needed: linking LPA trends, population ageing, and possibly integrating AI predictive modelling to forecast by region, age, and risk factors.

---

## 4. Investigations: Demand Forecasting and Model Enhancements

### 4.1. Modelling Overview

- **Current Logic:**  
  - The **number of living LPA donors** is the primary driver for investigations demand.
  - Modelling splits donors into **70+ and under 70**, reflecting their different growth rates and risk profiles:contentReference[oaicite:7]{index=7}.

- **Trends:**
  - Donors aged 70+: Expected to increase by 2 million in five years (60% growth).
  - Donors under 70: Expected to increase by 1 million (82% growth).
  - **90% of investigations involve donors aged 70+**.

- **Investigation Rates:**  
  - Since 2021, rates show month-to-month variability but broadly follow a random walk with trend, particularly among older donors.
  - The **pandemic period** (2020) and changes in safeguarding strategy (2018) created one-off surges and distortions; post-2021 data is preferred for forecasting:contentReference[oaicite:8]{index=8}.

### 4.2. Data and Model Challenges

- **Key Limitations:**
  - Modelled estimates for living donors rely on assumptions around mortality and loss of capacity.
  - Some safeguarding concerns relate to non-LPA holders or pre-registration events—these are **not captured** by current per-LPA-holder rates.

- **Workshop Actions:**  
  - Explore **data linkage** to improve mortality and capacity loss tracking.
  - **Segment investigations** by characteristics (number of attorneys, care home status, donor- vs attorney-led applications).
  - Direct calculation of **age-specific investigation rates** for more accurate forecasting.

---

## 5. Technical and Analytical Recommendations

### 5.1. AI and Data Science for Enhanced Forecasting

- **Short-Term:**
  - Use time series models (ARIMA, exponential smoothing) for short-run demand and receipt forecasting, incorporating external events (media broadcasts, fee change announcements) as exogenous variables.
  - Implement **automated anomaly detection** for identifying real-time demand shifts.

- **Medium- to Long-Term:**
  - Develop **machine learning models** (e.g., gradient boosting, random forests) to forecast new deputyship orders and investigation rates by region, age, and socioeconomic status, leveraging LPA uptake data and demographic shifts.
  - Integrate **natural language processing** on free-text safeguarding concerns to identify emerging risk themes.

- **Data Infrastructure:**
  - Prioritise the development of a **linked dataset** (LPA, deputyship, safeguarding, and mortality records).
  - Regularly review and update key model assumptions based on new data and policy changes.

### 5.2. Policy and Operational Actions

- **Fee Change Communication:**
  - Prepare for further media-driven surges in demand by scenario planning for 5,600, 5,700, and 5,800 average daily receipts.
  - Monitor application volumes daily following fee announcement and implement flexible operational resourcing.

- **Deputyship and LPA Integration:**
  - Study local variation in LPA take-up and deputyship to identify areas where interventions could reduce unnecessary deputyships.
  - Consider integrating deputyship and LPA forecasting for a unified long-term caseload planning model.

- **Investigations:**
  - Move towards **real-time risk dashboards** for investigations, updating forecasts monthly and stratifying by age, attorney type, and care setting.
  - Collaborate with data scientists to refine models as more granular data becomes available.

---

## 6. Next Steps (Agreed and Recommended)

### Immediate (Next Quarter)

- **Monitor demand:** Daily/weekly tracking of LPA application volumes in response to fee increase announcement and media coverage.
- **Scenario review:** Hold a short working group session if average receipts exceed 5,700/day for several sustained weeks.
- **Finance:** Recalibrate income and resourcing models if a step-change in demand is detected.

### Short-Medium Term

- **Data Projects:** Begin feasibility analysis for data linkage (mortality, capacity, safeguarding, and deputyship).
- **Model Enhancement:** Pilot AI/ML-based demand forecasting with scenario analysis.
- **Communication:** Prepare communications for applicants, anticipating potential bottlenecks pre- and post-fee change.

### Medium-Long Term

- **Forecasting Strategy:** Develop a five-year integrated demand model covering LPA, deputyship, and investigations.
- **Policy Alignment:** Align forecasting outputs with fee-setting, operational planning, and safeguarding strategy reviews.

---

## 7. Appendices

- **A. Slide Decks:** [LPA Demand, Deputyship Forecast, Investigations Modelling] (Summarised in report)  
- **B. Meeting Transcript:** [Key quotes and decisions included]
- **C. Data Tables and Scenario Charts:** Available in attached presentations

---

## 8. References

- LPA Demand Forecasts August 2025:contentReference[oaicite:9]{index=9}
- Deputyship Forecast August 2025:contentReference[oaicite:10]{index=10}
- Forecasting Investigations August 2025:contentReference[oaicite:11]{index=11}
- Quarterly Working Group Meeting Transcript, 6 August 2025:contentReference[oaicite:12]{index=12}

---

> **This report can be further tailored with additional AI-driven scenario modelling or interactive dashboards upon request.**  
> Let me know if you want scenario charts, Python code for model enhancement, or an executive summary version for senior leaders!
