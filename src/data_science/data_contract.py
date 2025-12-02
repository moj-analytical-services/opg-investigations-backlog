# data_contract.py
import argparse
import json
import pandas as pd


def infer_schema(df: pd.DataFrame):
    schema = {}
    for c in df.columns:
        dt = str(df[c].dtype)
        sample = df[c].dropna()
        entry = {"dtype": dt}
        if "float" in dt or "int" in dt:
            entry["min"] = float(sample.min()) if not sample.empty else None
            entry["max"] = float(sample.max()) if not sample.empty else None
        else:
            entry["unique_values"] = sorted(list(map(str, sample.unique()[:50])))
            entry["cardinality"] = int(sample.nunique())
        schema[c] = entry
    return schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.syn_csv)
    schema = infer_schema(df)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"schema": schema}, f, indent=2)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
