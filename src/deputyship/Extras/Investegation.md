# Investigation

To analyse the reasons for the step reduction in the investigation rate following the pandemic using time series analysis, you can follow these steps:

1. **Data Collection**:
   - Gather data on investigation rates from the Office of Public Guardian over a significant period, including pre-pandemic and post-pandemic periods.
   - Collect demographic information, attorney and donor information, and any other relevant variables.

2. **Data Preprocessing**:
   - Clean the data to handle missing values, outliers, and inconsistencies.
   - Normalize or standardize the data if necessary.

3. **Exploratory Data Analysis (EDA)**:
   - Visualize the investigation rates over time using line plots to identify trends, seasonality, and any abrupt changes.
   - Use histograms, box plots, and scatter plots to understand the distribution and relationships between variables.

4. **Time Series Decomposition**:
   - Decompose the time series data into trend, seasonal, and residual components to better understand underlying patterns.

5. **Modelling**:
   - Apply time series models such as ARIMA (AutoRegressive Integrated Moving Average) to model the investigation rates.
   - Use regression analysis to identify the impact of demographic information, attorney and donor information on investigation rates.

6. **Hypothesis Testing**:
   - Formulate hypotheses about potential reasons for the reduction in investigation rates (e.g., changes in policy, resource allocation, public behavior).
   - Use statistical tests to validate these hypotheses.

7. **Causal Analysis**:
   - Apply causal inference techniques such as Granger causality tests or difference-in-differences analysis to identify causal relationships.

8. **Interpretation and Reporting**:
   - Interpret the results to understand the factors contributing to the reduction in investigation rates.
   - Prepare a report summarising the findings, including visualisations and statistical evidence.

Here's the summary in Markdown format:

### Summary of Analysis and Thoughts on Deputyship Termination Rates and LPA Mortality Rates

1. **Mortality Rates and Caseload Reductions:**
   - You believe the actual impact on caseload reductions lies between the 25% and 5% marks.
   - The drop in investigations during COVID-19 was likely due to the pandemic's immediate impact, causing confusion and a temporary halt in investigations.

2. **Impact of COVID-19 on LPA Holders:**
   - Comparing the number of registered LPAs before and after COVID-19 could reveal changes in caseload ages.
   - For example, if the number of LPAs for ages 18-60 increased from 25,000 in 2020 to 150,000 in 2023, this could explain part of the 50% reduction in investigations.

3. **Termination Rates Analysis:**
   - Termination rates for deputyships were around 4% higher during the pandemic (October 2020 to October 2021) compared to 2018.
   - This increase equates to approximately 500 additional deaths when scaled to the 2018 caseload size.
   - Excess mortality for the general population from 2020 to 2023 was estimated at 180,000 (10% more deaths).

4. **Higher Termination Rates for Deputyships:**
   - Deputyship termination rates are higher than the mortality rates used in the LPA model.
   - This is expected due to pre-existing health conditions among those requiring deputyships.
   - The question arises whether deputyship termination rates are a better proxy for LPA donor mortality rates than general population mortality rates.

5. **Impact of COVID-19 on LPA Holders:**
   - Applying deputyship termination rates to LPA donors reduces the expected number of living LPA holders to around 2.3 million by the end of 2023, compared to 3.8 million using national mortality rates.
   - For those aged 70+, the expected number of living LPA holders would be around 1.4 million.
   - Assuming 50% of LPAs are actively in use, around 700,000 LPAs among donors aged 70+ would be in use by 2023.
   - Excess mortality could have reduced the number of living LPA holders aged 70+ by around 26%.

6. **Conclusions:**
   - The pandemic may have reduced the number of living LPA holders aged 70+ and demand for investigations by around 26%, or around half of the observed 50% reduction in investigation rates.
   - Without certain assumptions, the impact of excess mortality might have reduced the investigation rate by around 5%.
   - The pandemic is only a partial explanation for the reduction in investigation rates.

7. **Trends and Data Issues:**
   - The recent trend in younger donors is unlikely to explain the reduction in investigation rates since 2020.
   - Historical data issues, including duplicates in Sirius data, have been identified and corrected.

8. **Investigation Rate Trends:**
   - The overall investigation rate increased sharply in 2018 and 2019, driven by property and finance cases.
   - Health and Welfare case investigation rates have remained stable since 2013.

### Questions for Further Analysis:
1. Why did the volume of finance and property investigations increase sharply in 2018 and 2019?
2. Why has the investigation rate not returned to pre-pandemic levels?

### Hypotheses:
- The reduction post-pandemic may simply return the investigation rate to pre-2018 levels.
- Excess mortality among donors could partially explain the lower post-pandemic investigation rate.
- The increase in investigations in 2018 and 2019 may have brought forward demand, reducing post-pandemic cases.


# Driving investigation demand in 2018-19:
Regarding the data generated from the SQL code in the previous example as linked investigation and Lasting Power of Attorney (LPA) data, can you suggest some analysis to answer some questions below  was set for a planned workshop with the stakeholders (OPG) to help understand what is driving investigation demand?

You’ll see on the attached image that I have been trying to establish why there was a spike in Property and Finance cases investigated in 2018 and 2019 which seemed to then disappear following the pandemic. My best guess (and it is a guess) is that this may have been caused by a change in the criteria for investigation in 2018 which essentially made it a lot easier to investigate cases, before this applicants had to provide a lot of supporting documentation which have been difficult to provide and therefore been a barrier to making an investigation. If so then the spike in 2018 and 2019 might be a backlog of cases (or pent up demand) released when the ability to submit an application became much easier. 

If this is due to a backlog of cases then this would suggest that investigations in 2018 and 2019 had taken much longer to submit than usual. I was hoping that we might see this is in the data if we looked at the time periods in the linked data from registration to the date of investigation, which I would expect to be much longer on average in 2018 and 2019  than in the years before or after these dates.  
1. Why did property and finance investigations surge from 2018 onwards? 
2. Could the Spike in Property and Finance Cases be due to possible pent up demand (backlog?) of Concerns that the  simplification of criteria for investigation in 2018 made easier to investigate? 
3. Did change in Strategy in 2018 also result in more referrals from external agencies? Is there any evidence of an increase in external referrals? Could pent up demand come from external agencies? 
4. Why didn’t Health and Welfare cases follow the same pattern? 
4. Could the change in Triage from May 2016 onwards which caused a spike in investigations be a factor? if so how?  
5. Why have property and finance investigations returned to the longer term trend following the pandemic? If we assume the previous spike was caused by pent up demand , is it reasonable to assume this demand has now diminished?  
6. Have there been any other changes since 2018 in the criteria for selecting investigations from concerns raised? If so what changed?  
7. Did the rise in DIY and Online LPA Submissions from 2015 onwards (without legal advice) lead to more errors and misunderstandings about the responsibilities of attorneys, increasing the risk of unintentional or deliberate misuse? 
8.  Could the online tool have lead therefore to a pent up rise in concerns, that simplification of the investigation criteria in  2018 released?  
9.  Are there any other external reasons that your aware of that would help explain the increase in property and finance cases? 

## Solutions:
1. Trend‐break detection & intervention analysis

- Segmented time-series regression on monthly counts of Property & Finance (P&F) investigations to see whether there’s a statistically significant “jump” in early-2018 and again in early-2020. Estimate the change in slope and level at:
    - April 2018 (criteria change)
    - May 2016 (triage change)
    - March 2020 (pandemic lockdown)
- Chow-test or equivalent to verify whether the post-2018 slope is different from pre-2018 and whether post-2020 reversion is significant.
- Chow-test Purpose: The Chow test helps identify structural breaks in data, which are points where the relationship between variables changes. 
- How it works: It compares the residual sum of squares (RSS) from a pooled regression model (using all data) with the RSS from separate regressions for each subset of the data. 
- Null Hypothesis: The null hypothesis of the Chow test is that the coefficients are the same across the different data subsets (or time periods). 
- Alternative Hypothesis: The alternative hypothesis is that the coefficients are different, indicating a structural break. 
- Interpretation: If the calculated F-statistic from the Chow test is greater than the critical F-value, the null hypothesis is rejected, suggesting a structural break. 
- Applications:
    - Econometrics: Analysing economic time series data to see if relationships between variables have changed over time, such as after a policy change or economic shock. 
    - Panel Data Analysis: Determining if the relationships between variables are consistent across different groups or individuals in a panel dataset. 
    - Regression Discontinuity Designs: Assessing whether there is a structural break at a specific cutoff point in a regression discontinuity design. 
- Example: Imagine analyzing the relationship between advertising spend and sales. A Chow test could be used to see if the relationship is the same before and after a major change in marketing strategy. 
2. Registration-to-Investigation lead times
- Compute, for each case,
    - days_to_investigation = investigation_start_date – registration_date
- Compare the distribution (mean, median, IQR) of these lead times by cohort year (2015, 2016, … , 2022).
    - Expect higher mean/median in 2018–19 if pent-up cases took longer.
    - Plot box-and-whisker by year.

3. Referral source mix over time
- Does your data capture an “origin” field—e.g. external agency vs public vs internal?
- Trend the counts of “external referrals” by month/year.
    - Look for a bump post-2018 in external referrals.
    - Calculate the proportion of P&F investigations coming from external agencies vs other sources, by year.
- If there is a spike in external referrals in 2018, that supports the “agency pent-up” hypothesis.

4. Contrast P&F vs Health & Welfare
- Repeat both the time-series and lead-time analyses for H&W cases.
- If H&W shows no level change in 2018, that suggests the criteria change applied mainly to P&F investigations.

5. Triaging change May 2016
- Overlay a vertical marker at May 2016 on your P&F count chart and on the lead-time boxplots.
- See whether there is a small uptick right after May 2016, and whether cases registered around then took longer.
- Also test a 2016 “intervention” in your segmented regression.

6. Pandemic reversion in 2020
- Compare Q2 2020 onward to pre-2018 trend:
    - Did counts simply return to the long-run trend line you’d project from 2011–2015?
    - If pent-up demand was exhausted, you’d expect counts to revert to baseline rather than undershoot.

7. Criteria changes since 2018
- If you have metadata on “policy version” or a date when selection criteria were tweaked again, you can mark those on your time-series.
- Even if don’t, ask stakeholders for any “minor” updates post-2018 and check for small local bumps.

8. DIY & Online LPA channel analysis
- Tag applications by channel (DIY/online vs paper/with-lawyer).
- Trend the monthly counts of DIY/online submissions from 2015 onward.
- Cross-tabulate: what proportion of DIY submissions result in later investigations? Is it higher in 2018–19 than previously?
- If DIY clients misunderstand responsibilities, might see a higher error-rate flag or shorter registration-to-investigation times (i.e. they trigger investigations sooner).

9. Alternative external drivers
- Socio-economic indicators: overlay house-price indices or debt-statistics to see if there’s a macro-economic driver.
- Media events or major public inquiries around estate mismanagement in 2018–19.
- Staffing levels / backlog: if you have internal data on investigator headcounts or back-office capacity, a sudden hire or process improvement could temporarily clear a backlog.

## Next Steps
- Pull your master table of cases with fields:
    - registration_date
    - investigation_start_date
    - case_type (P&F vs H&W)
    - referral_source
    - submission_channel (DIY vs lawyer)
- Compute lead times and classify each case by year-of-registration and year-of-investigation.

- Build a small dashboard with:
    - Monthly count lines for P&F and H&W (with vertical markers at May 2016, Apr 2018, Mar 2020).
    - Boxplots of days-to-investigate by registration year.
    - Stacked-bars for referral source share by year.
    - Line for DIY channel growth.
- That suite of charts and statistical tests should equip your stakeholders to see whether 2018’s spike was:
    - A genuine surge in new concerns,
    - A release of a pent-up backlog,
    - Driven by external agencies or DIY clients, or
    - Simply a data artefact that reverted post-pandemic.