# build_timeseries.py
import argparse
import numpy as np
import pandas as pd
from ts_gen import aggregate_counts, seasonal_trend_from_real, synth_count_series


def load_config(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", required=True)
    ap.add_argument("--config", default="config/investigation_schema.yml")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = load_config(args.config)

    date_col = cfg["primary_date_col"]
    group_cols = cfg.get("ts_group_cols", [])
    freq = cfg.get("ts_freq", "MS")
    period = cfg.get("ts_season_period", 12)

    df = pd.read_csv(args.real_csv, parse_dates=[date_col])
    counts = aggregate_counts(df, date_col=date_col, group_cols=group_cols, freq=freq)

    # For each group (or overall), estimate parameters and generate synthetic counts
    out_rows = []
    rng = np.random.default_rng(args.seed)
    if group_cols:
        for keys, sub in counts.groupby(group_cols):
            idx = pd.date_range(
                start=sub[date_col].min(), end=sub[date_col].max(), freq=freq
            )
            s = sub.set_index(date_col)["count"].reindex(idx).fillna(0)
            level, trend, amp = seasonal_trend_from_real(s, period=period)
            syn = synth_count_series(
                idx,
                level,
                trend,
                amp,
                season_period=period,
                ar_phi=0.4,
                noise_sigma=max(1.0, s.std() / 2),
                seed=int(rng.integers(0, 1e9)),
            )
            row = sub[group_cols].iloc[0].to_dict()
            out_rows.append(pd.DataFrame({date_col: idx, "count": syn, **row}))
    else:
        idx = pd.date_range(
            start=counts[date_col].min(), end=counts[date_col].max(), freq=freq
        )
        s = counts.set_index(date_col)["count"].reindex(idx).fillna(0)
        level, trend, amp = seasonal_trend_from_real(s, period=period)
        syn = synth_count_series(
            idx,
            level,
            trend,
            amp,
            season_period=period,
            ar_phi=0.4,
            noise_sigma=max(1.0, s.std() / 2),
            seed=int(rng.integers(0, 1e9)),
        )
        out_rows.append(pd.DataFrame({date_col: idx, "count": syn}))

    syn_df = pd.concat(out_rows, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = f"{args.out_dir}/synthetic_timeseries.csv"
    syn_df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    import os

    main()
