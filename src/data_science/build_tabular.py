# build_tabular.py
import argparse
import json
import pandas as pd
from synth_core import GaussianCopulaSynthesizer, stratified_synthesis
from coarsen import (
    top_code_numeric,
    bin_numeric,
    collapse_rare_categories,
    apply_k_anonymity,
)


def load_config(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", required=True)
    ap.add_argument("--config", default="config/investigation_schema.yml")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.real_csv, parse_dates=cfg.get("parse_dates", []))

    # Drop direct identifiers
    for col in cfg.get("drop_identifiers", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Coarsen numeric & categorical
    df = top_code_numeric(
        df,
        cfg.get("topcode_numeric", []),
        upper_quantile=cfg.get("topcode_quantile", 0.99),
    )
    df = bin_numeric(df, cfg.get("bin_numeric", {}))
    for c in cfg.get("collapse_rare_categoricals", []):
        if c in df.columns:
            df = collapse_rare_categories(
                df, c, min_count=cfg.get("rare_min_count", 50)
            )
    df = apply_k_anonymity(
        df, cfg.get("quasi_identifiers", []), k=cfg.get("k_anonymity_k", 10)
    )

    # Synthesis
    cat_cols = cfg.get("categorical_cols", [])
    strata_cols = cfg.get("strata_for_preserving", [])
    if strata_cols:
        syn = stratified_synthesis(
            df,
            cat_cols=strata_cols,
            min_per_stratum=cfg.get("min_per_stratum", 50),
            seed=args.seed,
        )
    else:
        gc = GaussianCopulaSynthesizer(categorical_cols=cat_cols).fit(df)
        syn = gc.sample(len(df), seed=args.seed)

    # Reorder columns to original order where possible
    syn = syn[
        [c for c in df.columns if c in syn.columns]
        + [c for c in syn.columns if c not in df.columns]
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = f"{args.out_dir}/synthetic_tabular.csv"
    syn.to_csv(out_csv, index=False)

    # Simple report
    report = {
        "rows": int(len(syn)),
        "cols": list(syn.columns),
        "note": "Synthetic tabular generated with Gaussian copula baseline and smoothed categoricals.",
    }
    with open(f"{args.out_dir}/tabular_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    import os

    main()
