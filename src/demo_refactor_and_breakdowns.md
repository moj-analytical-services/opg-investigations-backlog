# demo.ipynb – refactor interval breakdowns + expand legal review analysis

This is written to match what you asked for:

1) **Refactor** the repeated “Interval distributions by case_type” code (roughly your line ~171 to ~495) into **one function**.
2) Make the same outputs work for **any** grouping column (e.g. `case_type` *or* `application_type`).
3) Add the extra outputs you want:
   - **New case start gaps** (e.g. `inter_pickup_days`) broken down by `case_type`, `risk_band`, `application_type`, and `legal_review`.
   - **Allocated → PG sign-off** time broken down the same way (use `days_to_signoff`).
   - **% legal review** broken down by case type and other characteristics.
4) Fix the runtime errors you’ve hit (Path, NAType, category mean).

---

## A) Fix `UnboundLocalError: Path` (blocking error)

That error happens when *inside `demo_all()`* you accidentally assign to `Path` (e.g. `Path = ...`), which makes it a local variable.

**Fastest fix:** add this as the **first line inside `demo_all()`**:

```python
from pathlib import Path
```

**Better fix (if you find an assignment):** rename any local variable called `Path` to `out_path` / `path_obj`.

---

## B) Fix the plotting + aggregation errors you hit

### B1) `float() argument must be a string or a real number, not 'NAType'`

Before plotting bars (or imshow), force numeric and drop missing:

```python
rules_low = rules_low.copy()
rules_low["prob_new_case"] = pd.to_numeric(rules_low["prob_new_case"], errors="coerce")
rules_low = rules_low.dropna(subset=["prob_new_case"])
ax.bar(rules_low["gap_band"].astype(str), rules_low["prob_new_case"])
```

### B2) `category dtype does not support aggregation 'mean'`

Your `legal_review` is a *category* (strings), so `.mean()` fails.

Create a numeric flag once (after `typed = engineer(...)`):

```python
typed = typed.copy()
typed["legal_review_flag"] = typed["legal_review"].astype("string").isin(["1", "True", "true", "Y", "Yes"]).astype(int)
```

Then use `legal_review_flag` for rates.

---

## C) Refactor the repeated “Interval distributions by case_type” section

### C1) Where to change

In the **first code cell**, inside `demo_all()`, locate the section that begins with something like:

```python
# Interval distributions by case_type
interval_dists_by_case_type = IntervalAnalysis.analyse_interval_distributions(di, by=["case_type"])
```

…and then has many repeated blocks converting dicts to DataFrames and plotting.

You will:

1) **Paste the helper functions below** just above that section.
2) **Replace** the repeated block with **two function calls**.

### C2) Paste these helpers inside `demo_all()` (just above the old block)

```python
DEFAULT_INTERVAL_METRICS = [
    # “new case start” gap
    "inter_pickup_days",
    # alloc → PG sign-off (already engineered in preprocessing.py)
    "days_to_signoff",
    # received → PG sign-off
    "days_to_pg_signoff",
    # alloc → close
    "days_alloc_to_close",
    # received/alloc → legal review request
    "days_recieved_to_legal_review",
    "days_alloc_to_req_legal_review",
    # received → alloc
    "days_to_alloc",
]

def _describe_days(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                "p10": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p90": np.nan, "max": np.nan}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }

def interval_distributions_by(di: pd.DataFrame, by: list[str], metrics: list[str] | None = None) -> dict[str, pd.DataFrame]:
    metrics = metrics or [m for m in DEFAULT_INTERVAL_METRICS if m in di.columns]
    out: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        tbl = (
            di.groupby(by, dropna=False)[metric]
              .apply(_describe_days)
              .apply(pd.Series)
              .reset_index()
        )
        out[metric] = tbl
    return out

def save_and_plot_interval_breakdowns(
    di: pd.DataFrame,
    by: list[str],
    outdir: Path,
    label: str | None = None,
    metrics: list[str] | None = None,
    max_categories: int = 25,
) -> dict[str, pd.DataFrame]:

    label = label or "_".join(by)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tables = interval_distributions_by(di, by=by, metrics=metrics)

    for metric, df in tables.items():
        # Save tidy table
        df.to_csv(outdir / f"interval_dists_by_{label}__{metric}.csv", index=False)

        # Bar chart of mean days
        plot_df = df.copy()
        plot_df["mean"] = pd.to_numeric(plot_df["mean"], errors="coerce")
        plot_df = plot_df.dropna(subset=["mean"])

        if len(by) == 1 and plot_df.shape[0] > max_categories:
            plot_df = plot_df.sort_values("count", ascending=False).head(max_categories)

        if len(by) == 1:
            x = plot_df[by[0]].astype(str)
        else:
            x = plot_df[by].astype(str).agg(" | ".join, axis=1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x, plot_df["mean"])
        ax.set_ylabel(f"Mean {metric} (days)")
        ax.set_title(f"Mean {metric} by {label}")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"bar_mean_{metric}_by_{label}.png", dpi=150)
        plt.close(fig)

    return tables
```

### C3) Replace the whole repeated case_type block with this

```python
# Interval distributions by case_type (refactored)
interval_tables_case_type = save_and_plot_interval_breakdowns(
    di,
    by=["case_type"],
    outdir=outplots,
    label="case_type",
    metrics=[
        "inter_pickup_days",
        "days_to_signoff",
        "days_alloc_to_close",
        "days_to_pg_signoff",
        "days_recieved_to_legal_review",
        "days_alloc_to_req_legal_review",
    ],
)

# Same outputs by application_type
interval_tables_app_type = save_and_plot_interval_breakdowns(
    di,
    by=["application_type"],
    outdir=outplots,
    label="application_type",
    metrics=[
        "inter_pickup_days",
        "days_to_signoff",
        "days_alloc_to_close",
        "days_to_pg_signoff",
        "days_recieved_to_legal_review",
        "days_alloc_to_req_legal_review",
    ],
)
```

That gives you the same style of tables/plots, but now switching the grouping is just changing `by=[...]`.

---

## D) Add the requested “4-way” breakdown outputs

### D1) Ensure `risk_band` exists

If you already have `risk_band`, skip this.

If you don’t, you need a proxy. If `weighting` is your best proxy, one simple banding is:

```python
if "risk_band" not in di.columns and "weighting" in di.columns:
    w = pd.to_numeric(di["weighting"], errors="coerce")
    di["risk_band"] = pd.cut(
        w,
        bins=[-np.inf, 1, 2, 3, np.inf],
        labels=["Low", "Medium", "High", "Very high"],
    )
```

### D2) New case starts: gap between pickups (`inter_pickup_days`)

```python
save_and_plot_interval_breakdowns(
    di,
    by=["case_type", "risk_band", "application_type", "legal_review"],
    outdir=outcsv,
    label="case_type_risk_app_legal",
    metrics=["inter_pickup_days"],
)
```

### D3) Allocation → PG sign-off (`days_to_signoff`)

```python
save_and_plot_interval_breakdowns(
    di,
    by=["case_type", "risk_band", "application_type", "legal_review"],
    outdir=outcsv,
    label="case_type_risk_app_legal",
    metrics=["days_to_signoff"],
)
```

This produces tidy CSVs like:
- `interval_dists_by_case_type_risk_app_legal__inter_pickup_days.csv`
- `interval_dists_by_case_type_risk_app_legal__days_to_signoff.csv`

---

## E) Expand “% cases needing legal review”

### E1) Helper function

Paste inside `demo_all()` (after `typed` exists):

```python
def legal_review_rate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["legal_review_flag"] = tmp["legal_review"].astype("string").isin(["1", "True", "true", "Y", "Yes"]).astype(int)

    out = (
        tmp.groupby(by, dropna=False)
           .agg(
               n_cases=("case_id", "count"),
               legal_rate=("legal_review_flag", "mean"),
               median_days_to_signoff=("days_to_signoff", "median"),
               mean_days_to_signoff=("days_to_signoff", "mean"),
           )
           .reset_index()
           .sort_values("legal_rate", ascending=False)
    )
    return out
```

### E2) Required breakdowns

```python
legal_by_case_type = legal_review_rate(typed, by=["case_type"])
legal_by_case_type.to_csv(outcsv / "legal_review_rate_by_case_type.csv", index=False)

legal_by_case_risk = legal_review_rate(typed, by=["case_type", "risk_band"])
legal_by_case_risk.to_csv(outcsv / "legal_review_rate_by_case_type_risk.csv", index=False)

legal_by_case_app = legal_review_rate(typed, by=["case_type", "application_type"])
legal_by_case_app.to_csv(outcsv / "legal_review_rate_by_case_type_application_type.csv", index=False)

legal_by_case_app_risk = legal_review_rate(typed, by=["case_type", "application_type", "risk_band"])
legal_by_case_app_risk.to_csv(outcsv / "legal_review_rate_by_case_type_application_type_risk.csv", index=False)
```

Optional bar chart:

```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(legal_by_case_type["case_type"].astype(str), legal_by_case_type["legal_rate"])
ax.set_ylabel("Legal review rate")
ax.set_title("Legal review rate by case type")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(outplots / "legal_review_rate_by_case_type.png", dpi=150)
plt.close(fig)
```

---

## F) Risk × age (and “legal review more likely as time increases”)

### F1) Risk × age heatmap (if you have an age column)

If your linked data includes `donor_age` (or similar), bin it:

```python
if "donor_age" in typed.columns:
    tmp = typed.copy()
    tmp["legal_review_flag"] = tmp["legal_review"].astype("string").isin(["1"]).astype(int)

    tmp["age_band"] = pd.cut(
        pd.to_numeric(tmp["donor_age"], errors="coerce"),
        bins=[0, 30, 40, 50, 60, 70, 80, 90, 120],
        right=False,
    )

    tbl = (
        tmp.groupby(["risk_band", "age_band"], dropna=False)
           .agg(n=("case_id", "count"), legal_rate=("legal_review_flag", "mean"))
           .reset_index()
    )

    pivot = tbl.pivot(index="risk_band", columns="age_band", values="legal_rate")
    mat = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_title("Legal review rate: Risk × Age band")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outplots / "heatmap_legal_rate_risk_x_age.png", dpi=150)
    plt.close(fig)
```

### F2) Why does legal review appear more likely as time increases?

Two mechanisms usually coexist:

1) **Confounding by complexity**: complex cases both take longer and are more likely to need legal input.
2) **Legal review adds time**: requesting/rejecting/iterating legal review can extend the overall duration.

You can test the second mechanism directly:

```python
# Compare alloc→signoff durations by legal review
comp = (
    typed.assign(legal_review_flag=typed["legal_review"].astype("string").isin(["1"]).astype(int))
         .groupby("legal_review_flag")["days_to_signoff"]
         .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
)
print(comp)

# How late does legal review typically occur?
if "days_alloc_to_req_legal_review" in di.columns:
    print(di["days_alloc_to_req_legal_review"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
```

If `days_alloc_to_req_legal_review` is often **late**, it’s consistent with “the hard cases run long, then need legal input later”.

---

## G) Legal review rate vs additional characteristics (linked data)

Once you’ve merged in the linked attributes (names may differ), do:

```python
candidate_cols = [
    "application_type",
    "case_type",
    "concern_type",
    "risk_band",
    "sex",                 # or donor_sex
    "n_attorneys",         # or attorney_count
    "days_since_registered",
]

present = [c for c in candidate_cols if c in typed.columns]
for col in present:
    out = legal_review_rate(typed, by=[col])
    out.to_csv(outcsv / f"legal_review_rate_by_{col}.csv", index=False)
```

For numeric predictors (age, #attorneys, time since registered), **bin first** (quantiles or business cut points), then feed the binned column to `legal_review_rate()`.

---

## H) Fuzzy inference for micro-sim (quick guidance)

Fuzzy inference can be a good fit if you need:
- **Interpretability** (rules you can show to senior stakeholders)
- **Smooth thresholds** (not hard cut-offs)
- A way to combine **data + expert judgement**

A pragmatic pattern:

1) Build a discrete-event micro-sim (cases arrive, investigators pick up/close cases).
2) Use **fuzzy rules** for decisions (pickup propensity, legal-review propensity), but initialise rule strengths using empirical rates from this analysis.
3) Calibrate membership functions/rule weights to match observed:
   - backlog trajectory
   - WIP distribution
   - interval distributions (pickup gaps / alloc→signoff)
   - legal review rate by segment

If you want a strong baseline for comparison, fit a simple **logistic model** for legal-review propensity and a **hazard model** for time-to-close, then decide whether fuzzy rules add value (interpretability / policy testing).
