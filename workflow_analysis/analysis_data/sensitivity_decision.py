
#!/usr/bin/env python3
"""
sensitivity_decision.py — Decision-guiding sensitivity for HPC workflows.

This script extends the CART-based analysis to be *decision-oriented*:
1) Discover regions by TOTAL first (CART on `total`).
2) Explain those regions with strict wildcards + average I/O shares.
3) Within the best (top-K) regions, quantify:
   - per-store residual effects on TOTAL (delta_from_best),
   - node-scaling curves: ΔT/Δn, efficiency E(n), and n* (diminishing returns),
   - pairwise placement synergy (interaction index),
   - regret (region- and config-level) vs the global best.

Inputs
------
A CSV produced by your pipeline, e.g., workflow_makespan_stageorder.csv
Required columns:
- total
- nodes
- <stage>_stor (or *_store) — categorical storage assignments
- read_<stage>, write_<stage>, in_<stage>, out_<stage>
- critical_path — tokens "stage_in_S->read_T->..."; used to attribute critical barriers

Outputs (in --outdir)
---------------------
- regions_by_total.csv (ranked by mean_total; includes strict wildcards + mean shares)
- node_scaling_by_region.csv (per region, per nodes: mean, ΔT, efficiency, n* flag)
- top_regions_store_effects.csv (per best region: per-store value deltas on TOTAL)
- synergy_by_region.csv (pairwise store interactions per region; most negative are best synergies)
- regret_per_region.csv (region-level regret vs global best)
- regret_per_config.csv (row-level regret vs global best)
- workflow_rowwise_sensitivities.csv (lam_* columns + shares, for inspection)

Usage
-----
python sensitivity_decision.py \
  --input 1kgenome/workflow_makespan_stageorder.csv \
  --outdir sens_out \
  --target-regions 12 --max-depth 8 --min-leaf 5 \
  --top-k 5 --nstar-eps 0.05 --pair-limit 12

Notes
-----
- Strict wildcard: a column shows '*' only if, conditional on the region's fixed columns,
  the region spans ALL values of that column present in the data.
- OneHotEncoder compatibility: handles sklearn >=1.4 (sparse_output) and <1.4 (sparse).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re
from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------------------------- helpers ----------------------------

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
        raise RuntimeError("No *_stor or *_store columns found.")
    return cols

def stage_name_from_store_col(c: str) -> str:
    return c[:-5] if c.endswith("_stor") else (c[:-6] if c.endswith("_store") else c)

def parse_critical_tokens(s: str) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    return [t for t in s.split("->") if t]

def compute_rowwise_sensitivities(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Attach lam_* and share columns to df."""
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

    # initialize lambdas
    for s in stores:
        df[f"lam_read_{s}"]  = 0.0
        df[f"lam_write_{s}"] = 0.0
    df["lam_in_total"]  = 0.0
    df["lam_out_total"] = 0.0

    # attribute along the critical path
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

    # shares
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
    df["movement_share"] = (df["lam_in_total"] + df["lam_out_total"]) / (df["total"] + eps)

    for c in ["exec_beegfs_share", "exec_local_share", "movement_share"]:
        df[c] = df[c].fillna(0.0).clip(lower=0.0, upper=1.0)

    return df, store_cols

def strict_wildcard_summary(df_all: pd.DataFrame,
                            df_labeled: pd.DataFrame,
                            store_cols: List[str],
                            stat_cols_map: dict,
                            additional_cols: List[str] = None) -> pd.DataFrame:
    """Summarize leaves with strict wildcard policy and provided stats."""
    additional_cols = additional_cols or []
    rows = []
    for reg, grp in df_labeled.groupby("region", sort=False):
        fixed_cols = [c for c in store_cols + ["nodes"] if grp[c].nunique() == 1]
        fixed_vals = {c: grp[c].iloc[0] for c in fixed_cols}

        cond = pd.Series(True, index=df_all.index)
        for c, v in fixed_vals.items():
            cond &= (df_all[c] == v)
        df_cond = df_all[cond]

        # wildcard logic
        row_wc = {}
        for c in store_cols:
            present = set(map(str, grp[c].unique()))
            if len(present) == 1:
                row_wc[c] = next(iter(present))
            else:
                domain = set(map(str, df_cond[c].unique()))
                if len(domain) > 1 and present == domain:
                    row_wc[c] = "*"
                else:
                    row_wc[c] = ",".join(sorted(present))

        # stats
        def stat(series, how):
            x = series.dropna().astype(float).values
            if x.size == 0:
                return np.nan
            if how == "mean":
                return float(np.mean(x))
            if how == "std":
                return float(np.std(x, ddof=1) if x.size > 1 else 0.0)
            if how == "cv":
                m = float(np.mean(x)); s = float(np.std(x, ddof=1) if x.size > 1 else 0.0)
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

        stats_out = {name: stat(grp[src], how) for name, (src, how) in stat_cols_map.items()}

        nodes_str = ",".join(map(str, sorted(map(int, pd.unique(grp["nodes"])))))
        base = {
            "region": reg,
            "size": len(grp),
            **stats_out,
            **row_wc,
            "nodes": nodes_str
        }
        for col in additional_cols:
            base[f"mean_{col}"] = stat(grp[col], "mean")
            base[f"cv_{col}"]   = stat(grp[col], "cv")
        rows.append(base)

    head = ["region", "size"] + list(stat_cols_map.keys())
    mid = store_cols
    tail = ["nodes"] + sum(([f"mean_{c}", f"cv_{c}"] for c in (additional_cols or [])), [])
    return pd.DataFrame(rows)[head + mid + tail]

def fit_cart_for_total(df: pd.DataFrame, store_cols: List[str],
                       target_regions=12, max_depth=8, min_leaf=5):
    feats = store_cols + ["nodes"]
    X = df[feats].copy()
    y = df["total"].astype(float).values

    # sklearn compatibility
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

# ---------------------- decision-guiding analyses ----------------------

def per_store_effects_in_region(labeled: pd.DataFrame, store_cols: List[str], region_id: int) -> pd.DataFrame:
    grp = labeled[labeled["region"] == region_id]
    rows = []
    for col in store_cols:
        if grp[col].nunique() >= 2:
            means = grp.groupby(col)["total"].mean().sort_values()
            best = float(means.iloc[0])
            for val, m in means.items():
                rows.append({
                    "region": region_id,
                    "column": col,
                    "value": val,
                    "mean_total": float(m),
                    "delta_from_best": float(m - best),
                    "count": int((grp[col] == val).sum())
                })
    return pd.DataFrame(rows)

def node_scaling_by_region(labeled: pd.DataFrame, region_id: int, nstar_eps: float = 0.05) -> pd.DataFrame:
    grp = labeled[labeled["region"] == region_id].copy()
    if grp.empty:
        return pd.DataFrame(columns=["region","nodes","count","mean_total","std_total","delta_from_prev","rel_delta","efficiency","is_nstar"])
    tbl = grp.groupby("nodes")["total"].agg(["count","mean","std"]).reset_index().rename(columns={"mean":"mean_total","std":"std_total"})
    tbl = tbl.sort_values("nodes").reset_index(drop=True)

    # deltas and efficiency vs min-n
    deltas = [np.nan]
    rel_deltas = [np.nan]
    nodes_list = tbl["nodes"].tolist()
    means_list = tbl["mean_total"].tolist()
    for i in range(1, len(tbl)):
        d = means_list[i] - means_list[i-1]
        deltas.append(float(d))
        rel_deltas.append(float(abs(d) / means_list[i]) if means_list[i] else np.nan)

    n0 = nodes_list[0]
    t0 = means_list[0]
    eff = [1.0 if nodes_list[i]==n0 else (t0 / ((nodes_list[i]/n0) * means_list[i]) if means_list[i] and n0 else np.nan) for i in range(len(tbl))]

    tbl["delta_from_prev"] = deltas
    tbl["rel_delta"] = rel_deltas
    tbl["efficiency"] = eff

    # n* = first node where rel_delta < eps (diminishing returns)
    nstar_flag = [False]*len(tbl)
    idx_nstar = next((i for i, rd in enumerate(rel_deltas) if (i>0 and rd is not None and not np.isnan(rd) and rd < nstar_eps)), None)
    if idx_nstar is not None:
        nstar_flag[idx_nstar] = True
    tbl["is_nstar"] = nstar_flag
    tbl.insert(0, "region", region_id)
    return tbl

def synergy_by_region(labeled: pd.DataFrame, store_cols: List[str], region_id: int, pair_limit: int = 12) -> pd.DataFrame:
    grp = labeled[labeled["region"] == region_id].copy()
    rows = []
    # pick columns with at least 2 values
    var_cols = [c for c in store_cols if grp[c].nunique() >= 2]
    # bound number of pairs
    pairs = []
    for i in range(len(var_cols)):
        for j in range(i+1, len(var_cols)):
            pairs.append((var_cols[i], var_cols[j]))
    pairs = pairs[:pair_limit]

    overall = grp["total"].mean() if not grp.empty else np.nan
    for a, b in pairs:
        # marginal means
        mA = grp.groupby(a)["total"].mean()
        mB = grp.groupby(b)["total"].mean()
        # joint means
        mAB = grp.groupby([a,b])["total"].mean()

        # compute interaction I(a,b) = mAB - mA - mB + mOverall for every present (va,vb)
        for (va, vb), mab in mAB.items():
            ia = mA.get(va, np.nan)
            ib = mB.get(vb, np.nan)
            if any(pd.isna([mab, ia, ib, overall])):
                continue
            I = float(mab - ia - ib + overall)
            cnt = int(((grp[a] == va) & (grp[b] == vb)).sum())
            rows.append({
                "region": region_id,
                "colA": a, "valA": va,
                "colB": b, "valB": vb,
                "interaction": I,
                "mean_total_combo": float(mab),
                "mean_total_overall": float(overall),
                "mean_total_A": float(ia),
                "mean_total_B": float(ib),
                "count": cnt
            })
    out = pd.DataFrame(rows)
    # sort: most negative interaction (super-additive improvement) first
    if not out.empty:
        out = out.sort_values(["region","interaction","count"], ascending=[True, True, False])
    return out

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Decision-guiding sensitivity for HPC workflows (TOTAL-first + node & placement analysis).")
    ap.add_argument("--input", required=True, help="Path to workflow_makespan_stageorder.csv")
    ap.add_argument("--outdir", default="sens_out", help="Directory to write outputs")
    ap.add_argument("--target-regions", type=int, default=12, help="Approximate number of CART regions")
    ap.add_argument("--max-depth", type=int, default=8, help="Max depth for CART")
    ap.add_argument("--min-leaf", type=int, default=5, help="Minimum samples per leaf")
    ap.add_argument("--top-k", type=int, default=5, help="How many best (lowest TOTAL) regions to analyze deeply")
    ap.add_argument("--nstar-eps", type=float, default=0.05, help="Diminishing-returns threshold for n* detection")
    ap.add_argument("--pair-limit", type=int, default=12, help="Max number of (store,store) pairs for synergy per region")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input)
    if "nodes" not in df_raw.columns or "total" not in df_raw.columns:
        raise RuntimeError("Input CSV must contain 'nodes' and 'total' columns.")

    # 1) Compute rowwise sensitivities (lam_* + shares)
    df_sens, store_cols = compute_rowwise_sensitivities(df_raw.copy())
    rowwise_path = outdir / "workflow_rowwise_sensitivities.csv"
    keep_cols = ["nodes", "total"] + store_cols + [c for c in df_sens.columns if c.startswith("lam_")] + \
                ["exec_beegfs_share", "exec_local_share", "movement_share"]
    df_sens[keep_cols].to_csv(rowwise_path, index=False)

    # 2) CART on TOTAL (regions)
    labeled, tree, ct, alpha = fit_cart_for_total(
        df=df_sens, store_cols=store_cols,
        target_regions=args.target_regions, max_depth=args.max_depth, min_leaf=args.min_leaf
    )

    # 3) Strict-wildcard region summary + shares
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
    regions = strict_wildcard_summary(
        df_all=df_sens, df_labeled=labeled, store_cols=store_cols,
        stat_cols_map=stat_cols, additional_cols=["exec_beegfs_share", "exec_local_share", "movement_share"]
    )
    regions = regions.sort_values(["mean_total","cv_total","size"], ascending=[True, True, False]).reset_index(drop=True)
    regions_path = outdir / "regions_by_total.csv"
    regions.to_csv(regions_path, index=False)

    # 4) Regret (region-level vs global best; config-level too)
    global_best_total = df_sens["total"].min()
    reg_region = regions[["region","mean_total","cv_total","size"]].copy()
    reg_region["regret_vs_global_best"] = reg_region["mean_total"] - global_best_total
    reg_region.to_csv(outdir / "regret_per_region.csv", index=False)

    reg_cfg = labeled[["region","nodes","total"] + store_cols].copy()
    reg_cfg["regret_vs_global_best"] = reg_cfg["total"] - global_best_total
    reg_cfg.to_csv(outdir / "regret_per_config.csv", index=False)

    # 5) Deep-dive the top-K regions
    top_regions = regions["region"].head(args.top_k).tolist()

    # 5a) Residual per-store effects (delta from best per value)
    effects_all = []
    for rid in top_regions:
        eff = per_store_effects_in_region(labeled, store_cols, rid)
        if not eff.empty:
            effects_all.append(eff)
    if effects_all:
        effects_df = pd.concat(effects_all, ignore_index=True)
    else:
        effects_df = pd.DataFrame(columns=["region","column","value","mean_total","delta_from_best","count"])
    effects_df.to_csv(outdir / "top_regions_store_effects.csv", index=False)

    # 5b) Node scaling in each region (ΔT/Δn, efficiency, n* flag)
    scale_rows = []
    for rid in labeled["region"].unique():
        tbl = node_scaling_by_region(labeled, rid, nstar_eps=args.nstar_eps)
        if not tbl.empty:
            scale_rows.append(tbl)
    node_scale_df = pd.concat(scale_rows, ignore_index=True) if scale_rows else pd.DataFrame()
    node_scale_df.to_csv(outdir / "node_scaling_by_region.csv", index=False)

    # 5c) Placement synergy (pairwise interaction index)
    syn_rows = []
    for rid in top_regions:
        syn = synergy_by_region(labeled, store_cols, rid, pair_limit=args.pair_limit)
        if not syn.empty:
            syn_rows.append(syn)
    synergy_df = pd.concat(syn_rows, ignore_index=True) if syn_rows else pd.DataFrame(columns=[
        "region","colA","valA","colB","valB","interaction","mean_total_combo","mean_total_overall","mean_total_A","mean_total_B","count"
    ])
    synergy_df.to_csv(outdir / "synergy_by_region.csv", index=False)

    print("Wrote:")
    for p in [rowwise_path, regions_path, outdir / "regret_per_region.csv", outdir / "regret_per_config.csv",
              outdir / "top_regions_store_effects.csv", outdir / "node_scaling_by_region.csv", outdir / "synergy_by_region.csv"]:
        print(" -", p)

if __name__ == "__main__":
    main()
