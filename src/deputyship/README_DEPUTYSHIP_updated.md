# OPG deputyship demand forecasting model

## What this work is

This repository contains a Python based workflow for deputyship caseload analysis and forecasting. The workflow is implemented as a Jupyter notebook that can be run from top to bottom to produce extracts, flow measures, a stop flow forecast by age, and a set of charts and written insights.

The key idea is to treat the active caseload as a stock that changes over time through two flows. People enter the caseload when a new deputyship becomes active. People leave the caseload when an existing deputyship is no longer active. Once we can measure those flows consistently from the data, we can build a transparent forecast.

## The main notebook

The main working notebook is:

`Deputyship_Data_extraction_StopFlow_model_Jan26.ipynb`

A fully documented version is also provided:

`Deputyship_Data_extraction_StopFlow_model_Jan26_with_documentation.ipynb`

The documented version adds a plain English explanation cell before every code cell. The Python code is unchanged.

## What the notebook produces

When you run the notebook you should expect four classes of output in the `output` folder.

1. Monthly extracts. One csv per month plus an Excel workbook with one sheet per month, and one combined csv for the whole range.
2. Flow summaries. Month level and year level tables that count active clients and the number of people who entered and exited.
3. Age specific rate table. A month by age table with active counts, entry counts, termination counts, and derived rates.
4. Forecast outputs and visuals. A combined historical and forecast table, plus charts and an insights markdown file.

## Data source and definitions

### Data source

The extraction step queries Sirius data in the `opg_sirius_prod` schema through `pydbtools`. The query is designed to be a point in time snapshot by filtering every table to the same `glueexporteddate`.

To run the notebook you need:

1. Network access to the data environment that hosts the Sirius export tables.
2. A working `pydbtools` configuration that can run SQL against `opg_sirius_prod`.

If you do not have access, the notebook will fail at the first query. In that case, request the required permissions and confirm you can run a simple test query in the same environment.

### Core entities

The extract returns one row per case record number for the snapshot date, along with supporting attributes such as supervision level and risk score.

The notebook uses two related counting concepts.

Active orders. This is the number of rows in a snapshot or a grouped table. It is a proxy for the number of active orders recorded on that day.

Active clients. This is the number of unique case numbers in a snapshot or a grouped table. It is a proxy for the number of distinct clients in the active caseload.

Whether case number is a perfect person identifier depends on upstream business rules. In this notebook it is treated as the practical unit for client level counting.

### Month end snapshots

All month based analysis in this notebook uses month end snapshots. For a month like `2025-01` the snapshot date is the last calendar day of that month.

Month end snapshots reduce within month noise and match the typical reporting cadence.

## Flow measures and the math behind them

### Month to month flow

For a month m, define A(m) as the set of active case numbers observed at month end.

Then:

Active count in month m equals |A(m)|.

Entered in month m equals the number of case numbers that are in A(m) but not in A(m−1).

Exited in month m equals the number of case numbers that are in A(m−1) but not in A(m).

This is implemented using set arithmetic, which avoids double counting and is computationally efficient.

### Year on year flow

The forecasting method used later relies on a seasonal comparison. For that reason the notebook also calculates a year on year view of flow.

For each month m, define the previous year comparison month as m−12 and define A(m−12) as the set of active case numbers in that month end snapshot.

Then:

Entered in month m equals |A(m) ∖ A(m−12)|.

Exited in month m equals |A(m−12) ∖ A(m)|.

This measure tells you how the caseload changed relative to the same point in the calendar one year earlier.

## Age groups, rates, and missing age handling

### Age calculation

The extract computes age in years as an integer approximation:

age_years = round( days_between(date_of_birth, created_date) ÷ 365.25 )

If the computed value is negative it is set to zero. The 365.25 factor accounts for leap years on average.

### Age grouping

Age is grouped into single year bands from 0 up to 106. Each band represents an interval of one year.

If age is missing or out of range, the row is labelled Unknown.

### Age specific entry and termination rates

For each month m and each age group g the notebook calculates:

active(g, m) as the number of unique case numbers in the base population for that age and month.

entered(g, m) as the number of unique case numbers that entered for that age and month.

terminations(g, m) as the number of unique case numbers that exited for that age and month.

Termination rate is defined as:

termination_rate(g, m) = terminations(g, m) ÷ active(g, m) when active(g, m) is greater than zero.

Retention rate is defined as:

retention_rate(g, m) = 1 − termination_rate(g, m)

These rates are used for descriptive insight and as inputs to the stop flow forecast.

### Optional redistribution of Unknown ages

If `redistribute_unknown_age` is enabled, Unknown age rows are reassigned into concrete age groups to avoid a separate Unknown category in the forecast.

The method is proportional and deterministic.

1. Count known cases by age group and convert counts into proportions.
2. Multiply proportions by the number of Unknown rows to get target fractional allocations.
3. Convert targets into integers by taking the floor, then allocate the remaining units to the largest fractional remainders. This is Hamilton apportionment.
4. Assign Unknown rows in a stable order so repeated runs produce the same allocation.

This is an imputation. It improves continuity of age series but it does not add new information about the true ages of Unknown cases.

## Stop flow forecasting

### The forecasting recurrence

The forecast uses a stop flow recurrence applied separately to each age group:

active(t) = active(t−1) + entered(t−12) − terminations(t−12)

active(t) is the forecast active count for a month t.
active(t−1) is the previous month forecast count.
entered(t−12) and terminations(t−12) are the historical counts observed one year earlier in the same calendar month.

The seasonal assumption is that entries and terminations repeat their pattern from the same month in the previous year.

A floor at zero is applied so active counts never become negative.

### What the forecast returns

The forecast produces a table by month and age group plus month level and year level totals across ages. These outputs are then combined with historical data to create a single table suitable for reporting and charts.

## Visuals and insight notes

The notebook includes a function that produces a consistent set of charts and a short insights markdown file. It standardises column names, ensures month is treated as a date, and saves outputs into the `output` folder.

For the totals chart it also adds a rough 95 percent uncertainty band using a Poisson style approximation for counts:

lower = max(0, x − z √(phi x))
upper = x + z √(phi x)

where z is 1.96 and phi is a dispersion factor.

This band is an indicative visual aid, not a full probabilistic forecast interval.

## How to run the notebook

### Recommended run order

1. Open the notebook in Jupyter.
2. Run the dependency set up cell first so imports work.
3. Run the function definition cells so all helpers are in memory.
4. In the first run block, set `start_month`, `end_month`, and `output_base`, then run the cell to generate extracts and `ages_df`.
5. In the forecasting run block, set the forecast horizon `periods`, then run the cell to generate the combined historical and forecast table plus charts.

### Parameters you will usually change

`start_month` and `end_month` control the historical extraction window. They should be strings formatted like `YYYY-MM`, for example `2024-12`.

`periods` controls how many future months are forecast.

`output_base` controls where files are written.

### Output folder safety

The run block calls a helper that deletes all files inside the output folder before writing new results. Do not point `output_base` at any folder that contains work you need to keep.

## Output files and where to find them

All outputs are written under the `output` folder.

### Extract outputs

For each month you will get:

1. A csv file in a month folder such as `output/2025-01/cases_2025-01.csv`
2. An Excel workbook named like `cases_2024-12_to_2025-12.xlsx` with one sheet per month
3. A combined csv named like `all_cases_2024-12_to_2025-12.csv`

### Flow outputs

1. `monthly_flow_2024-12_to_2025-12.csv`
2. `yearonyear_flows_2024-12_to_2025-12.csv`

### Age rate output

`termination_and_entry_rates_by_age_2024-12_to_2025-12.csv`

### Forecast and reporting outputs

1. A combined historical and forecast csv named like `final_deputyship_historical_forecasts_2022_2025.csv`
2. A set of png charts and an insights markdown file in the `output` folder

## Known quirks and practical tips

### Database access and performance

If a query fails, first confirm your `pydbtools` connection works with a trivial query in the same session.

Large date ranges can take time because the notebook queries once per month. If you need a long history, consider running in smaller chunks and then combining outputs.

### Monthly active cases helper

The `calculate_monthly_active_cases` function currently does not filter the input DataFrame down to each month inside its loop. The comments show an earlier intention to fetch per month. If you rely on its outputs, confirm that the inputs are filtered in the way you expect.

### Reproducibility

The Unknown age redistribution is deterministic given a fixed input DataFrame because it assigns rows in sorted index order. If you change upstream sorting, the exact row level reassignment can change even if the aggregated allocations stay the same.

## How to maintain and extend this work

### Updating the extract

If you need additional fields, add them in the SQL query inside `fetch_cases_for_date`. Keep the `glueexporteddate` filter consistent across joined tables to preserve point in time integrity.

### Changing age bins

The year on year rate function accepts `age_bins` and `age_labels`. You can use this to move from single year groups to broader bands. If you do, update the visualisation settings so age ordering remains sensible.

### Improving the forecast

The stop flow method is a transparent baseline. If you want a richer model, common next steps include:

1. Modelling entered and terminations as separate time series with seasonal components.
2. Allowing age progression by moving cohorts forward through age groups.
3. Adding scenario controls that scale entry or termination rates.

Any extension should keep the same clear separation between extraction, feature engineering, forecasting, and reporting so results remain auditable.

## Ownership

This notebook and documentation were prepared for the OPG forecasting workstream. If you are taking over, start by running a short date window end to end, inspect the output tables, and then expand the window once you are confident the definitions match the reporting need.
