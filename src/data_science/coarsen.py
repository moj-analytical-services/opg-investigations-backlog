# coarsen.py
import pandas as pd
from typing import List, Dict


def top_code_numeric(df: pd.DataFrame, cols: List[str], upper_quantile: float = 0.99):
    for c in cols:
        if c in df.columns:
            q = df[c].quantile(upper_quantile)
            df[c] = df[c].clip(upper=q)
    return df


def bin_numeric(df: pd.DataFrame, bins: Dict[str, list]):
    """Bin numeric columns into provided edges; replaces with categorical labels."""
    for c, edges in bins.items():
        if c in df.columns:
            df[c] = pd.cut(df[c], bins=edges, include_lowest=True).astype(str)
    return df


def collapse_rare_categories(
    df: pd.DataFrame, col: str, min_count: int = 50, other_label: str = "Other"
):
    vc = df[col].astype(str).value_counts()
    rare = vc[vc < min_count].index
    df[col] = df[col].astype(str).where(~df[col].astype(str).isin(rare), other_label)
    return df


def apply_k_anonymity(
    df: pd.DataFrame, quasi_id_cols: List[str], k: int = 10, other_label: str = "Other"
):
    if not quasi_id_cols:
        return df
    vc = df.groupby(quasi_id_cols).size().reset_index(name="n")
    rare = vc[vc["n"] < k][quasi_id_cols]
    if rare.empty:
        return df
    # Collapse last quasi-ID by labeling rare combos in the last column
    target_col = quasi_id_cols[-1]
    df = df.copy()
    mask = (
        df.merge(rare, on=quasi_id_cols, how="left", indicator=True)["_merge"] == "both"
    )
    df.loc[mask, target_col] = other_label
    return df
