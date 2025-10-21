
# Build Investigator Daily Panel (from OPG raw extract)

**Script:** `build_investigator_daily_from_raw.py`

## Usage
```bash
python build_investigator_daily_from_raw.py --in /path/to/raw.csv --outdir /path/to/out
```

- Accepts CSV or Excel (.xlsx/.xls) with columns like the ones shown in your screenshot.
- Handles messy UK dates (e.g. `11/10/2024`, `Not Completed` -> blank).
- Derives daily **WIP per investigator**, event flags (**new case**, **legal**, **court order**), and a **backlog** series.

## Outputs
- `investigator_daily.csv` – daily panel ready for the Bayesian model
- `backlog_series.csv` – approximate central backlog (accepted minus allocations)
- `event_log.csv` – tidy event rows (one per case/date)

## Notes
- Role is not present in the raw table; left blank.
- `bank_holiday` is set to 0 (you can merge a holiday calendar later).
- `term_flag` uses a simple month proxy (Aug=0, otherwise 1).
- End-of-case is Closure Date > PG Sign off date > horizon end.

