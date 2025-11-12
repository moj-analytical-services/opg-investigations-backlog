# Function Usage Examples & Patch Notes

These examples were executed against your files:
- Notebook: `Build_Investigator_Daily_from_Raw_30_10_25.ipynb`
- Data: `data/raw/raw.csv`

I did **not change any of your functions**. To run end‑to‑end here, I temporarily added a small runtime shim so `json.dumps` can serialize NumPy scalars (affects `build_event_log`). Below are precise one‑line patches you can add to the notebook if you prefer a source fix.

## Quickstart (end‑to‑end)

```python
from pathlib import Path

raw, colmap = load_raw(Path('data/raw/raw.csv'))
typed = engineer(raw, colmap)
start, end = date_horizon(typed)  # returns (start, end)

# Works after applying the JSON shim or Patch 1 below
events = build_event_log(typed)

wip = build_wip_series(typed, start, end)
backlog = build_backlog_series(typed, start, end)

# build_daily_panel currently needs Patch 2
# daily, backlog2, events2 = build_daily_panel(typed, start, end)
# summary = summarise_daily_panel(daily)
```

## Individual function examples

### `normalise_col(c: str) -> str`

```python
normalise_col(' Investigator Name ')  # -> 'investigator_name'
```

### `parse_date_series(s: pd.Series) -> pd.Series`

```python
dates = parse_date_series(col(raw, colmap, 'received_date'))  # example
```

### `hash_id(t: str) -> str`

```python
hash_id('C1')  # stable hashed identifier
```

### `month_to_season(m: int) -> str`

```python
month_to_season(3)  # 'spring' (UK seasons)
```

### `is_term_month(m: int) -> int`

```python
is_term_month(3)  # returns 1/0
```

### `load_raw(p: Path, force_encoding: str | None = None)`
Returns a tuple `(raw_df, colmap)`.

```python
raw, colmap = load_raw(Path('/mnt/data/raw.csv'))
```

### `col(df: pd.DataFrame, colmap: dict, name: str) -> pd.Series`
Resolves a canonical column to the actual header.

```python
investigator_series = col(raw, colmap, 'investigator')
```

### `engineer(df: pd.DataFrame, colmap: dict) -> pd.DataFrame`

```python
typed = engineer(raw, colmap)
```

### `date_horizon(typed: pd.DataFrame, pad_days: int = 14, fallback_to_all_dates: bool = True)`
Returns `(start, end)`.

```python
start, end = date_horizon(typed)
```

### `build_event_log(typed: pd.DataFrame, pad_days: int = 14, fallback_to_all_dates: bool = True) -> pd.DataFrame`

```python
events = build_event_log(typed)  # uses defaults
```

### `build_wip_series(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ...) -> pd.DataFrame`

```python
wip = build_wip_series(typed, start, end)
```

### `build_backlog_series(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, freq: str | None = None, ...) -> pd.DataFrame`

```python
backlog = build_backlog_series(typed, start, end)
```

### `build_daily_panel(typed: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ...)`
Returns `(daily, backlog, events)`.
**Note:** needs Patch 2 below.

```python
daily, backlog2, events2 = build_daily_panel(typed, start, end)
```

### `summarise_daily_panel(daily: pd.DataFrame, by: list[str] = [...], extras: list[str] = ('backlog_available',)) -> pd.DataFrame`

```python
summary = summarise_daily_panel(daily)
```


---

## Patches to make everything run without shims

### Patch 1 — `build_event_log`: JSON serialization of NumPy scalars

**Problem:** `meta_json = json.dumps(meta_dict, ensure_ascii=False)` can fail when `meta_dict` contains NumPy scalars (e.g., `np.int64`).

**Fix (one line):** replace that line with:

```python
meta_json = json.dumps(meta_dict, ensure_ascii=False, default=lambda o: getattr(o, 'item', lambda: o)())
```

**Where:** Inside `build_event_log`, at the line where `meta_json` is computed (around the loop emitting events).


### Patch 2 — `build_daily_panel`: ensure `grid` is defined before merging

**Problem:** `UnboundLocalError: cannot access local variable 'grid' where it is not associated with a value`.

**Fix (insert directly after `dates = pd.DataFrame({'date': date_index})`):

```python
# 4) Build staff-date grid (all combinations)
staff = typed[['staff_id', 'team']].drop_duplicates()
grid = (
    staff.assign(_k=1)
         .merge(pd.DataFrame({'date': date_index}).assign(_k=1), on='_k', how='outer')
         .drop(columns=['_k'])
)
```

This defines the staff-by-date grid so subsequent merges (WIP, events, backlog) work.


---

## Files written (from this run)

- `/mnt/data/typed_engineered.csv`
- `/mnt/data/event_log.csv`
- `/mnt/data/wip_series.csv`
- `/mnt/data/backlog_series.csv`