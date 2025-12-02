# checks.py
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def nearest_neighbour_gap(df_real: pd.DataFrame, df_syn: pd.DataFrame, cols):
    R = df_real[cols].to_numpy(dtype=float)
    S = df_syn[cols].to_numpy(dtype=float)
    out = []
    for r in R:
        dists = np.linalg.norm(S - r, axis=1)
        out.append(dists.min())
    return np.array(out)


def exact_match_rate(df_real: pd.DataFrame, df_syn: pd.DataFrame, quasi_cols):
    real_tuples = set(tuple(x) for x in df_real[quasi_cols].astype(str).values.tolist())
    syn_tuples = set(tuple(x) for x in df_syn[quasi_cols].astype(str).values.tolist())
    inter = real_tuples.intersection(syn_tuples)
    return len(inter) / max(1, len(real_tuples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", required=True)
    ap.add_argument("--syn_csv", required=True)
    ap.add_argument("--config", default="config/investigation_schema.yml")
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    import yaml
    import os

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    real = pd.read_csv(args.real_csv)
    syn = pd.read_csv(args.syn_csv)

    os.makedirs(args.out_dir, exist_ok=True)
    report = {}

    # Stats diffs numeric
    num = [
        c for c in cfg.get("numeric_cols", []) if c in real.columns and c in syn.columns
    ]
    if num:
        report["means_diff"] = (syn[num].mean() - real[num].mean()).to_dict()
        report["vars_diff"] = (syn[num].var() - real[num].var()).to_dict()
        # Corr L1 diff
        common_num = [
            c for c in num if real[c].dtype.kind in "if" and syn[c].dtype.kind in "if"
        ]
        if len(common_num) >= 2:
            report["corr_diff_L1"] = float(
                (syn[common_num].corr() - real[common_num].corr()).abs().values.mean()
            )

    # Privacy indicators
    qi = [
        c
        for c in cfg.get("quasi_identifiers", [])
        if c in real.columns and c in syn.columns
    ]
    if qi:
        report["exact_match_rate_qi"] = float(exact_match_rate(real, syn, qi))

    nn_cols = [
        c
        for c in cfg.get("nn_gap_numeric", [])
        if c in real.columns and c in syn.columns
    ]
    if nn_cols:
        g = nearest_neighbour_gap(real, syn, nn_cols)
        report["nn_gap_mean"] = float(g.mean())
        report["nn_gap_min"] = float(g.min())

    # Utility TSTR/TRTS for simple binary target
    tgt = cfg.get("target_col")
    if tgt and tgt in real.columns and tgt in syn.columns:
        Xs = syn.drop(columns=[tgt]).select_dtypes(include=["number"]).fillna(0).values
        ys = syn[tgt].values
        Xr = real.drop(columns=[tgt]).select_dtypes(include=["number"]).fillna(0).values
        yr = real[tgt].values
        try:
            m = LogisticRegression(max_iter=1000, class_weight="balanced")
            m.fit(Xs, ys)
            report["TSTR_AUC"] = float(roc_auc_score(yr, m.predict_proba(Xr)[:, 1]))
            m2 = LogisticRegression(max_iter=1000, class_weight="balanced")
            m2.fit(Xr, yr)
            report["TRTS_AUC"] = float(roc_auc_score(ys, m2.predict_proba(Xs)[:, 1]))
        except Exception as e:
            report["tstr_trts_error"] = str(e)

    with open(
        f"{args.out_dir}/privacy_utility_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)
    print("Wrote:", f"{args.out_dir}/privacy_utility_report.json")


if __name__ == "__main__":
    main()
