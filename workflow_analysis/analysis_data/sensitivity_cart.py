
#!/usr/bin/env python3
"""
sensitivity_cart.py — Critical-path I/O sensitivity analysis with CART region summaries.

Inputs
------
A CSV like your workflow_makespan_stageorder.csv containing:
- total
- per-stage storage assignments: <stage>_stor (or *_store)
- numeric candidates: read_<stage>, write_<stage>, in_<stage>, out_<stage>
- critical_path: "stage_in_A->read_B->write_B->stage_out_C" style tokens
- nodes

Outputs
-------
- workflow_sensitivity_regions_exec_beegfs.csv
- workflow_sensitivity_regions_exec_local.csv
- workflow_sensitivity_regions_movement.csv
- workflow_rowwise_sensitivities.csv

“Strict wildcard” rule:
A column shows '*' in a region only if, conditional on the fixed columns in that
region, the region spans ALL values of that column present in the input data.
"""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

def _as_float(x):
    try:
        if isinstance(x, str) and x.strip() == "-":
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def detect_store_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_stor") or c.endswith("_store")]
    if not cols:
        raise RuntimeError("No *_stor or *_store columns found in the input CSV.")
    return cols

def stage_name_from_store_col(c: str) -> str:
    if c.endswith("_stor"):
        return c[:-5]
    if c.endswith("_store"):
        return c[:-6]
    return c

def parse_critical_tokens(s: str) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    return [t for t in s.split("->") if t]

def trimmed_stats(arr, trim_pct: float) -> Tuple[float, float]:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if trim_pct <= 0 or trim_pct * 2 >= 1.0:
        return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0)
    xs = np.sort(x)
    k = int(np.floor(len(xs) * trim_pct))
    k = min(k, (len(xs) - 1) // 2)
    t = xs[k: len(xs)-k] if k > 0 else xs
    return float(np.mean(t)), float(np.std(t, ddof=1) if t.size > 1 else 0.0)

def strict_wildcard_summary(df_all: pd.DataFrame,
                            df_labeled: pd.DataFrame,
                            store_cols: List[str],
                            target_col: str,
                            trim_pct: float = 0.0) -> pd.DataFrame:
    rows = []
    for reg, grp in df_labeled.groupby("region", sort=False):
        fixed_cols = [c for c in store_cols + ["nodes"] if grp[c].nunique() == 1]
        fixed_vals = {c: grp[c].iloc[0] for c in fixed_cols}

        cond = pd.Series(True, index=df_all.index)
        for c, v in fixed_vals.items():
            cond &= (df_all[c] == v)
        df_cond = df_all[cond]

        row = {}
        for c in store_cols:
            present = set(map(str, grp[c].unique()))
            if len(present) == 1:
                row[c] = next(iter(present))
            else:
                domain = set(map(str, df_cond[c].unique()))
                if len(domain) > 1 and present == domain:
                    row[c] = "*"
                else:
                    row[c] = ",".join(sorted(present))

        mean, std = trimmed_stats(grp[target_col].values, trim_pct=trim_pct)
        cv = (std / mean) if (mean and mean != 0) else np.nan
        nodes_str = ",".join(map(str, sorted(map(int, pd.unique(grp["nodes"])))))

        rows.append({
            "region": reg,
            "size": len(grp),
            "mean": mean,
            "std": std,
            "cv": cv,
            **row,
            "nodes": nodes_str
        })

    out = pd.DataFrame(rows)
    col_order = ["region", "size", "mean", "std", "cv"] + store_cols + ["nodes"]
    return out[col_order]

def fit_tree_and_label(df: pd.DataFrame,
                       store_cols: List[str],
                       target_col: str,
                       nodes_as_feature: bool = True,
                       target_regions: int = 12,
                       max_depth: int = 8,
                       min_leaf: int = 5,
                       criterion: str = "squared_error"):
    feats = store_cols + (["nodes"] if nodes_as_feature else [])
    X = df[feats].copy()
    y = df[target_col].values.astype(float)

    cat_cols = store_cols
    num_cols = ["nodes"] if nodes_as_feature else []

    try:
        # scikit-learn >= 1.4
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn < 1.4
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ct = ColumnTransformer(
        transformers=[("cat", ohe, cat_cols)],
        remainder="passthrough" if num_cols else "drop"
    )
    X_enc = ct.fit_transform(X)

    base = DecisionTreeRegressor(random_state=0, criterion=criterion)
    base.fit(X_enc, y)
    path = base.cost_complexity_pruning_path(X_enc, y)
    alphas = np.unique(path.ccp_alphas)
    alphas = np.insert(alphas[alphas > 0], 0, 0.0)

    best = None
    for a in alphas:
        t = DecisionTreeRegressor(
            random_state=0,
            criterion=criterion,
            ccp_alpha=a,
            max_depth=max_depth,
            min_samples_leaf=min_leaf
        )
        t.fit(X_enc, y)
        leaves = np.unique(t.apply(X_enc)).size
        cand = (abs(leaves - target_regions), -leaves, a, t)
        if best is None or cand < (best[0], best[1], best[2], best[3]):
            best = cand
    _, _, best_alpha, best_tree = best

    leaf_ids = best_tree.apply(X_enc)
    df_labeled = df.copy()
    df_labeled["region"] = [int(l) for l in leaf_ids]
    return df_labeled, best_tree, ct, best_alpha

def compute_rowwise_sensitivities(df: pd.DataFrame) -> pd.DataFrame:
    store_cols = detect_store_cols(df)
    stage_names = [stage_name_from_store_col(c) for c in store_cols]
    stores = sorted({v for c in store_cols for v in df[c].dropna().unique()})
    local_stores = [s for s in stores if s != "beegfs"]

    # Ensure numeric candidates are floats
    for s in stage_names:
        for base in ("read_", "write_", "in_", "out_"):
            col = base + s
            if col in df.columns:
                df[col] = df[col].apply(_as_float)

    # Initialize sensitivity columns
    for s in stores:
        df[f"lam_read_{s}"]  = 0.0
        df[f"lam_write_{s}"] = 0.0
    df["lam_in_total"]  = 0.0
    df["lam_out_total"] = 0.0

    # Attribute along the critical path
    for idx, row in df.iterrows():
        tokens = parse_critical_tokens(row.get("critical_path", ""))
        store_of = {stage_name_from_store_col(c): row[c] for c in store_cols}

        for tok in tokens:
            m = re.match(r"^(read|write|stage_in|stage_out)_(.+)$", tok)
            if not m:
                continue
            kind, stg = m.group(1), m.group(2).strip()

            if kind == "read":
                col = f"read_{stg}"
                val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store in stores:
                    df.at[idx, f"lam_read_{store}"] += val

            elif kind == "write":
                col = f"write_{stg}"
                val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store in stores:
                    df.at[idx, f"lam_write_{store}"] += val

            elif kind == "stage_in":
                col = f"in_{stg}"
                val = row[col] if col in df.columns else 0.0
                df.at[idx, "lam_in_total"] += val

            elif kind == "stage_out":
                col = f"out_{stg}"
                val = row[col] if col in df.columns else 0.0
                df.at[idx, "lam_out_total"] += val

    # Normalize by total
    eps = 1e-12
    if "beegfs" in stores:
        df["target_exec_beegfs_share"] = (df["lam_read_beegfs"] + df["lam_write_beegfs"]) / (df["total"] + eps)
    else:
        df["target_exec_beegfs_share"] = 0.0
    if local_stores:
        df["target_exec_local_share"]  = (
            sum(df[f"lam_read_{s}"] + df[f"lam_write_{s}"] for s in local_stores) / (df["total"] + eps)
        )
    else:
        df["target_exec_local_share"] = 0.0
    df["target_movement_share"]    = (df["lam_in_total"] + df["lam_out_total"]) / (df["total"] + eps)

    for c in ["target_exec_beegfs_share", "target_exec_local_share", "target_movement_share"]:
        df[c] = df[c].fillna(0.0).clip(lower=0.0, upper=1.0)

    return df, store_cols

def main():
    ap = argparse.ArgumentParser(description="Critical-path I/O sensitivity analysis with CART region summaries.")
    ap.add_argument("--input", required=True, help="Path to workflow_makespan_stageorder.csv")
    ap.add_argument("--outdir", default="sens_out", help="Directory to write outputs")
    ap.add_argument("--target-regions", type=int, default=12, help="Approximate number of CART regions")
    ap.add_argument("--max-depth", type=int, default=8, help="Max depth for CART")
    ap.add_argument("--min-leaf", type=int, default=5, help="Minimum samples per leaf")
    ap.add_argument("--trim-pct", type=float, default=0.0, help="Trimmed mean/std percentage (0..0.49)")
    args = ap.parse_args()

    outdir = pd.Path = args.outdir

    df = pd.read_csv(args.input)

    # Compute row-wise sensitivities
    df_sens, store_cols = compute_rowwise_sensitivities(df.copy())

    # Save row-wise sensitivities
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rowwise_path = str(Path(args.outdir) / "workflow_rowwise_sensitivities.csv")
    keep_cols = ["nodes", "total"] + store_cols + [c for c in df_sens.columns if c.startswith("lam_")] + \
                ["target_exec_beegfs_share", "target_exec_local_share", "target_movement_share"]
    df_sens[keep_cols].to_csv(rowwise_path, index=False)

    targets = [
        ("target_exec_beegfs_share", "workflow_sensitivity_regions_exec_beegfs.csv"),
        ("target_exec_local_share",  "workflow_sensitivity_regions_exec_local.csv"),
        ("target_movement_share",    "workflow_sensitivity_regions_movement.csv"),
    ]

    # Fit CART and summarize for each target
    for target_col, out_name in targets:
        labeled, tree, ct, alpha = fit_tree_and_label(
            df=df_sens,
            store_cols=store_cols,
            target_col=target_col,
            nodes_as_feature=True,
            target_regions=args.target_regions,
            max_depth=args.max_depth,
            min_leaf=args.min_leaf,
            criterion="squared_error"
        )

        summary = strict_wildcard_summary(
            df_all=df_sens,
            df_labeled=labeled,
            store_cols=store_cols,
            target_col=target_col,
            trim_pct=args.trim_pct
        )
        out_path = Path(args.outdir) / out_name
        summary.to_csv(out_path, index=False)

    print("Wrote:")
    print(" -", rowwise_path)
    for _, out_name in targets:
        print(" -", Path(args.outdir) / out_name)

if __name__ == "__main__":
    main()
