# Build a "total-first" sensitivity workflow:
# 1) Partition the config space by TOTAL using CART.
# 2) Summarize "best" (low-total) regions with strict wildcards.
# 3) For each top-k region, analyze within-region sensitivity of TOTAL to any store column
#    that still varies inside the region (simple per-level deltas).
#
# This script will:
# - Reuse rowwise sensitivity computation (lam_* and shares).
# - Fit a CART for TOTAL to get regions.
# - Save two CSVs:
#     * regions_by_total.csv (ranked by mean_total asc; includes share means)
#     * top_regions_store_effects.csv (per-region, per-column value effects on total)
# - Produce a quick bar plot of the best regions by mean_total.
#
import pandas as pd
import numpy as np
from pathlib import Path
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------- helpers from prior script ----------
def _as_float(x):
    try:
        if isinstance(x, str) and x.strip() == "-":
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def detect_store_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c.endswith("_stor") or c.endswith("_store")]
    if not cols:
        raise RuntimeError("No *_stor or *_store columns found.")
    return cols

def stage_name_from_store_col(c: str):
    if c.endswith("_stor"):
        return c[:-5]
    if c.endswith("_store"):
        return c[:-6]
    return c

def parse_critical_tokens(s):
    if not isinstance(s, str) or not s:
        return []
    return [t for t in s.split("->") if t]

def compute_rowwise_sensitivities(df: pd.DataFrame):
    store_cols = detect_store_cols(df)
    stage_names = [stage_name_from_store_col(c) for c in store_cols]
    stores = sorted({v for c in store_cols for v in df[c].dropna().unique()})
    local_stores = [s for s in stores if s != "beegfs"]

    # numeric coercion for candidate columns
    for s in stage_names:
        for base in ("read_", "write_", "in_", "out_"):
            col = base + s
            if col in df.columns:
                df[col] = df[col].apply(_as_float)

    for s in stores:
        df[f"lam_read_{s}"]  = 0.0
        df[f"lam_write_{s}"] = 0.0
    df["lam_in_total"]  = 0.0
    df["lam_out_total"] = 0.0

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

    eps = 1e-12
    if "beegfs" in stores:
        df["exec_beegfs_share"] = (df["lam_read_beegfs"] + df["lam_write_beegfs"]) / (df["total"] + eps)
    else:
        df["exec_beegfs_share"] = 0.0
    non_beegfs = [s for s in stores if s != "beegfs"]
    if non_beegfs:
        df["exec_local_share"]  = sum(df[f"lam_read_{s}"] + df[f"lam_write_{s}"] for s in non_beegfs) / (df["total"] + eps)
    else:
        df["exec_local_share"] = 0.0
    df["movement_share"]    = (df["lam_in_total"] + df["lam_out_total"]) / (df["total"] + eps)

    for c in ["exec_beegfs_share", "exec_local_share", "movement_share"]:
        df[c] = df[c].fillna(0.0).clip(lower=0.0, upper=1.0)

    return df, store_cols

def strict_wildcard_summary(df_all: pd.DataFrame,
                            df_labeled: pd.DataFrame,
                            store_cols,
                            stat_cols_map,
                            additional_cols=None):
    """
    stat_cols_map: dict output_name -> (source_col, agg)
       e.g., {"mean_total": ("total","mean"), "cv_total": ("total","cv"), ...}
    """
    additional_cols = additional_cols or []
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

        def stat(series, how):
            x = series.dropna().astype(float).values
            if x.size == 0:
                return np.nan
            if how == "mean":
                return float(np.mean(x))
            if how == "std":
                return float(np.std(x, ddof=1) if x.size > 1 else 0.0)
            if how == "cv":
                m = float(np.mean(x))
                s = float(np.std(x, ddof=1) if x.size > 1 else 0.0)
                return (s / m) if m else np.nan
            if how == "q25":
                return float(np.quantile(x, 0.25))
            if how == "median":
                return float(np.quantile(x, 0.5))
            if how == "q75":
                return float(np.quantile(x, 0.75))
            if how == "min":
                return float(np.min(x))
            if how == "max":
                return float(np.max(x))
            return np.nan

        out_stats = {}
        for out_name, (src, how) in stat_cols_map.items():
            out_stats[out_name] = stat(grp[src], how)

        nodes_str = ",".join(map(str, sorted(map(int, pd.unique(grp["nodes"])))))
        base = {
            "region": reg,
            "size": len(grp),
            **out_stats,
            **row,
            "nodes": nodes_str
        }
        for col in additional_cols:
            base[col] = stat(grp[col], "mean")
        rows.append(base)

    # prepare column order
    head_cols = ["region", "size"] + list(stat_cols_map.keys())
    mid_cols = store_cols
    tail_cols = ["nodes"] + additional_cols
    df_out = pd.DataFrame(rows)[head_cols + mid_cols + tail_cols]
    return df_out

def fit_cart_for_total(df: pd.DataFrame, store_cols, target_regions=12, max_depth=8, min_leaf=5):
    feats = store_cols + ["nodes"]
    X = df[feats].copy()
    y = df["total"].astype(float).values

    # OHE with version-agnostic flag
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    ct = ColumnTransformer(
        transformers=[("cat", ohe, store_cols)],
        remainder="passthrough"
    )
    X_enc = ct.fit_transform(X)

    base = DecisionTreeRegressor(random_state=0)
    base.fit(X_enc, y)
    path = base.cost_complexity_pruning_path(X_enc, y)
    alphas = np.unique(path.ccp_alphas)
    alphas = np.insert(alphas[alphas > 0], 0, 0.0)

    best = None
    for a in alphas:
        t = DecisionTreeRegressor(
            random_state=0,
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
    labeled = df.copy()
    labeled["region"] = [int(l) for l in leaf_ids]
    return labeled, best_tree, ct, best_alpha

# ---------- run on available CSV ----------
in_path = "/mnt/data/workflow_makespan_stageorder.csv"
df_raw = pd.read_csv(in_path)

# Compute rowwise sensitivities once
df_sens, store_cols = compute_rowwise_sensitivities(df_raw.copy())

# Fit CART on TOTAL
labeled, tree, ct, alpha = fit_cart_for_total(df_sens, store_cols, target_regions=12, max_depth=8, min_leaf=5)

# Region summary ranked by TOTAL first, then include mean shares
stat_cols = {
    "mean_total":   ("total", "mean"),
    "std_total":    ("total", "std"),
    "cv_total":     ("total", "cv"),
    "q25_total":    ("total", "q25"),
    "median_total": ("total", "median"),
    "q75_total":    ("total", "q75"),
    "min_total":    ("total", "min"),
    "max_total":    ("total", "max"),
}

regions_total = strict_wildcard_summary(
    df_all=df_sens,
    df_labeled=labeled,
    store_cols=store_cols,
    stat_cols_map=stat_cols,
    additional_cols=["exec_beegfs_share", "exec_local_share", "movement_share"]
)

# Rank ascending by mean_total
regions_total = regions_total.sort_values(["mean_total", "cv_total", "size"], ascending=[True, True, False]).reset_index(drop=True)
regions_path = "/mnt/data/regions_by_total.csv"
regions_total.to_csv(regions_path, index=False)

# For the top-k regions, analyze within-region effects of any column that varies
top_k = 5 if len(regions_total) >= 5 else len(regions_total)
top_region_ids = regions_total["region"].head(top_k).tolist()

effect_rows = []
for rid in top_region_ids:
    grp = labeled[labeled["region"] == rid]
    for col in store_cols:
        if grp[col].nunique() >= 2:
            means = grp.groupby(col)["total"].mean().sort_values()
            best = float(means.iloc[0])
            for val, m in means.items():
                effect_rows.append({
                    "region": rid,
                    "column": col,
                    "value": val,
                    "mean_total": float(m),
                    "delta_from_best": float(m - best),
                    "count": int((grp[col] == val).sum())
                })

effects_df = pd.DataFrame(effect_rows)
effects_path = "/mnt/data/top_regions_store_effects.csv"
effects_df.to_csv(effects_path, index=False)

# Quick plot of top regions by mean_total
import matplotlib.pyplot as plt
plt.figure()
plt.bar(regions_total["region"].astype(str).head(10), regions_total["mean_total"].head(10))
plt.title("Best regions by TOTAL (lower is better)")
plt.xlabel("Region")
plt.ylabel("Mean TOTAL")
plot_path = "/mnt/data/regions_by_total_top10.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Regions by TOTAL (ranked)", regions_total)
if not effects_df.empty:
    caas_jupyter_tools.display_dataframe_to_user("Top regions: per-store effects on TOTAL", effects_df)

regions_path, effects_path, plot_path
