#!/usr/bin/env python3
"""
sensitivity_decision.py — Decision-guiding sensitivity for HPC workflows.

This script extends CART-based analysis to be *decision-oriented*:
1) Discover regions by TOTAL first (CART on `total`) using a tree over
   (categorical *_stor columns + numeric nodes). Regions are the tree leaves.
2) Explain those regions with strict wildcards + average I/O shares:
   - For each *_stor column, if a region has a single value ⇒ show that value,
     otherwise show '*' (wildcard).
   - For shares, we attribute along the *critical_path* and compute:
       exec_beegfs_share, exec_local_share, movement_share.
     (FIXED: we strip parenthetical counts from critical_path tokens.)
3) Within the best (top-K) regions, quantify:
   - per-store residual effects on TOTAL (delta_from_best),
   - node-scaling curves: ΔT from previous node-level, relative change,
     efficiency E(n) against ideal linear speedup, and n* (diminishing returns),
   - pairwise placement synergy (simple interaction index),
   - regret (region- and config-level) vs the global best.

Outputs
-------
- workflow_rowwise_sensitivities.csv
    Original rows + lam_* columns + share columns + region id.
- regions_by_total.csv
    Per-region stats (size, total stats), strict-wildcard columns for *_stor,
    nodes list covered, and mean/CV of share columns. (Header corrected:
    synergy report uses 'mean_total_combo', not a truncated label.)
- regret_per_region.csv
- regret_per_config.csv
- top_regions_store_effects.csv
- node_scaling_by_region.csv
- synergy_by_region.csv
- critical_path_prevalence_by_region.csv

Example
-------
python sensitivity_decision.py \
  --input 1kgenome/workflow_makespan_stageorder.csv \
  --outdir sens_out \
  --target-regions 12 --max-depth 8 --min-leaf 5 \
  --top-k 5 --nstar-eps 0.05 --pair-limit 12
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple
from itertools import combinations
import re

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
    s = s or ""
    return [t for t in s.split("->") if t]

# ----------------- rowwise attribution & shares ------------------

def compute_rowwise_sensitivities(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Attach lam_* and share columns to df (FIXED token parsing)."""
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
            m = re.match(r"^(read|write|stage_in|stage_out)_(.+)$", tok.strip())
            if not m:
                continue
            kind, stg_raw = m.group(1), m.group(2).strip()
            # FIX: strip trailing parenthetical counts like "(4)"
            stg = re.sub(r"\(.*\)$", "", stg_raw)

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

    if local_stores:
        df["exec_local_share"]  = sum(df[f"lam_read_{s}"] + df[f"lam_write_{s}"] for s in local_stores) / (df["total"] + eps)
    else:
        df["exec_local_share"] = 0.0

    df["movement_share"] = (df["lam_in_total"] + df["lam_out_total"]) / (df["total"] + eps)

    for c in ["exec_beegfs_share", "exec_local_share", "movement_share"]:
        df[c] = df[c].fillna(0.0).clip(lower=0.0, upper=1.0)

    return df, store_cols

# --------------------------- CART -----------------------------

def fit_cart_and_label_regions(df: pd.DataFrame, store_cols: List[str],
                               target_regions: int, max_depth: int, min_leaf: int) -> pd.Series:
    """Fit CART on total; return a 0..(n_leaves-1) region label per row."""
    X_cat = store_cols
    X_num = ["nodes"] if "nodes" in df.columns else []
    X_all = X_cat + X_num

    # Build transformer
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), X_cat)],
                           remainder="passthrough")
    X = ct.fit_transform(df[X_all])
    y = df["total"].values

    # Decision tree
    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_leaf_nodes=target_regions if target_regions else None,
        random_state=42
    )
    tree.fit(X, y)

    # Leaf ids -> compact 0..K-1
    leaf_ids = tree.apply(X)
    # Map leaf id to consecutive region ids sorted by mean total ascending
    df_tmp = pd.DataFrame({"leaf": leaf_ids, "total": y})
    leaf_mean = df_tmp.groupby("leaf")["total"].mean().sort_values().index.tolist()
    leaf_to_region = {leaf: i for i, leaf in enumerate(leaf_mean)}
    regions = pd.Series([leaf_to_region[l] for l in leaf_ids], index=df.index, name="region")
    return regions

# ----------------------- Region summaries ---------------------

def strict_wildcard_summary(df_all: pd.DataFrame,
                            df_labeled: pd.DataFrame,
                            store_cols: List[str],
                            stat_cols_map: dict,
                            additional_cols: List[str] = None) -> pd.DataFrame:
    """Summarize leaves with strict wildcard policy and provided stats."""
    additional_cols = additional_cols or []
    rows = []
    for reg, grp in df_labeled.groupby("region"):
        # strict wildcard: if region has 1 unique value for col -> that value, else '*'
        row_wc = {}
        for c in store_cols:
            vals = grp[c].dropna().unique().tolist()
            row_wc[c] = vals[0] if len(vals) == 1 else "*"

        # stats on totals
        def stat(series, how):
            x = series.to_numpy(dtype=float)
            if how == "n":
                return int(x.size)
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
            return x.size

        stats_out = {name: stat(grp[src], how) for name, (src, how) in stat_cols_map.items()}

        nodes_str = ",".join(map(str, sorted(map(int, pd.unique(grp["nodes"]))))) if "nodes" in grp else ""
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
    out = pd.DataFrame(rows).sort_values("mean_total", ascending=True).reset_index(drop=True)
    return out

# -------------------- Per-region deep dives -------------------

def compute_store_effects(df: pd.DataFrame, store_cols: List[str], top_regions: List[int]) -> pd.DataFrame:
    rows = []
    for reg in top_regions:
        grp = df[df["region"] == reg]
        if grp.empty: continue
        for c in store_cols:
            for val, sub in grp.groupby(c):
                m = sub["total"].mean()
                rows.append({"region": reg, "column": c, "value": val, "mean_total": m, "count": len(sub)})
    eff = pd.DataFrame(rows)
    if eff.empty:
        return eff
    eff["delta_from_best"] = 0.0
    eff = eff.sort_values(["region","column","mean_total"])
    def add_delta(g):
        g = g.sort_values("mean_total").copy()
        best = g["mean_total"].iloc[0]
        g["delta_from_best"] = g["mean_total"] - best
        return g
    eff = eff.groupby(["region","column"]).apply(add_delta).reset_index(drop=True)
    return eff[["region","column","value","mean_total","delta_from_best","count"]]

def compute_node_scaling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reg, grp in df.groupby("region"):
        if "nodes" not in grp.columns: 
            continue
        g = grp.groupby("nodes")["total"].agg(["count","mean","std"]).reset_index().sort_values("nodes")
        prev_mean = None
        base_nodes = int(g["nodes"].iloc[0])
        base_mean  = float(g["mean"].iloc[0])
        for _, r in g.iterrows():
            nodes = int(r["nodes"]); mean = float(r["mean"]); std = float(r["std"]) if not np.isnan(r["std"]) else 0.0
            if prev_mean is None:
                rows.append({"region": reg, "nodes": nodes, "count": int(r["count"]), "mean_total": mean, "std_total": std,
                             "delta_from_prev": np.nan, "rel_delta": np.nan,
                             "efficiency": 1.0, "is_nstar": False})
            else:
                delta = mean - prev_mean  # expect negative if improving
                rel   = abs(delta) / mean if mean != 0 else np.nan
                # Efficiency vs ideal linear: E(n) = T(n0) / (T(n) * (n/n0))
                eff = base_mean / (mean * (nodes / base_nodes)) if mean != 0 else np.nan
                rows.append({"region": reg, "nodes": nodes, "count": int(r["count"]), "mean_total": mean, "std_total": std,
                             "delta_from_prev": delta, "rel_delta": rel,
                             "efficiency": eff, "is_nstar": False})
            prev_mean = mean
    return pd.DataFrame(rows)

def mark_nstar(df_ns: pd.DataFrame, nstar_eps: float) -> pd.DataFrame:
    if df_ns.empty:
        return df_ns
    out = df_ns.copy()
    out["is_nstar"] = out["rel_delta"].abs() < float(nstar_eps)
    return out

def compute_regret(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_best = df["total"].min()
    cfg = df[["region","nodes","total"] + detect_store_cols(df)].copy()
    cfg["regret_vs_global_best"] = cfg["total"] - global_best
    reg = df.groupby("region")["total"].mean().reset_index().rename(columns={"total":"mean_total"})
    reg["regret_vs_global_best"] = reg["mean_total"] - global_best
    return reg, cfg

def compute_synergy(df: pd.DataFrame, store_cols: List[str], pair_limit: int) -> pd.DataFrame:
    rows = []
    for reg, grp in df.groupby("region"):
        if len(store_cols) < 2 or grp.empty:
            continue
        overall = grp["total"].mean()
        pairs = list(combinations(store_cols, 2))[:pair_limit]
        for a,b in pairs:
            ga = grp.groupby(a)["total"].mean()
            gb = grp.groupby(b)["total"].mean()
            gab = grp.groupby([a,b])["total"].mean().reset_index().rename(columns={"total":"mean_total_combo"})
            for _, r in gab.iterrows():
                va, vb = r[a], r[b]
                mean_ab = float(r["mean_total_combo"])
                mean_a = float(ga.get(va, np.nan))
                mean_b = float(gb.get(vb, np.nan))
                interaction = mean_ab - mean_a - mean_b + overall
                cnt = int(len(grp[(grp[a]==va) & (grp[b]==vb)]))
                rows.append({
                    "region": reg,
                    "colA": a, "valA": va,
                    "colB": b, "valB": vb,
                    "interaction": interaction,
                    "mean_total_combo": mean_ab,
                    "mean_total_overall": overall,
                    "mean_total_A": mean_a,
                    "mean_total_B": mean_b,
                    "count": cnt
                })
    return pd.DataFrame(rows)

def critical_path_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reg, grp in df.groupby("region"):
        size = len(grp)
        counts = grp["critical_path"].value_counts()
        prevalent = counts.index[0] if not counts.empty else ""
        for path, cnt in counts.items():
            rep = grp[grp["critical_path"] == path].iloc[0]
            rows.append({
                "region": reg,
                "region_size": size,
                "prevalent_critical_path": prevalent,
                "critical_path": path,
                "count": int(cnt),
                "fraction": float(cnt/size) if size else 0.0,
                "config_nodes": int(rep["nodes"]) if "nodes" in rep else None,
                **{c.replace("_stor","").replace("_store","store"): rep[c] for c in detect_store_cols(df)}
            })
    out = pd.DataFrame(rows).sort_values(["region","count"], ascending=[True, False])
    return out

# ------------------------------ MAIN -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Decision-guided sensitivity & region analysis for workflow totals.")
    ap.add_argument("--input", required=True, help="Input CSV (e.g., workflow_makespan_stageorder.csv)")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs")
    ap.add_argument("--target-regions", type=int, default=12, help="Approximate number of CART regions (max leaves)")
    ap.add_argument("--max-depth", type=int, default=8, help="Max depth for CART")
    ap.add_argument("--min-leaf", type=int, default=5, help="Minimum samples per leaf")
    ap.add_argument("--top-k", type=int, default=5, help="How many best (lowest TOTAL) regions to analyze deeply")
    ap.add_argument("--nstar-eps", type=float, default=0.05, help="Diminishing-returns threshold for n* detection")
    ap.add_argument("--pair-limit", type=int, default=12, help="Max number of (store,store) pairs for synergy per region")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    # 1) Rowwise sensitivities (lam_* and shares)
    df_rw, store_cols = compute_rowwise_sensitivities(df.copy())

    # 2) CART regions
    regions = fit_cart_and_label_regions(df_rw, store_cols, args.target_regions, args.max_depth, args.min_leaf)
    df_rw["region"] = regions

    # Save rowwise with region
    rowwise_path = outdir / "workflow_rowwise_sensitivities.csv"
    df_rw.to_csv(rowwise_path, index=False)

    # 3) Region summaries (strict wildcard + share stats)
    stat_cols_map = {
        "mean_total": ("total", "mean"),
        "std_total":  ("total", "std"),
        "cv_total":   ("total", "cv"),
        "q25_total":  ("total", "q25"),
        "median_total": ("total", "median"),
        "q75_total":  ("total", "q75"),
        "min_total":  ("total", "min"),
        "max_total":  ("total", "max"),
    }
    regions_df = strict_wildcard_summary(
        df_all=df_rw, df_labeled=df_rw, store_cols=store_cols, stat_cols_map=stat_cols_map,
        additional_cols=["exec_beegfs_share","exec_local_share","movement_share"]
    )
    regions_path = outdir / "regions_by_total.csv"
    regions_df.to_csv(regions_path, index=False)

    # 4) Regret
    regret_region, regret_config = compute_regret(df_rw)
    regret_region.to_csv(outdir / "regret_per_region.csv", index=False)
    regret_config.to_csv(outdir / "regret_per_config.csv", index=False)

    # 5) Top-K store effects (by lowest mean_total regions)
    top_regs = regions_df.sort_values("mean_total").head(args.top_k)["region"].tolist()
    eff_df = compute_store_effects(df_rw, store_cols, top_regs)
    eff_df.to_csv(outdir / "top_regions_store_effects.csv", index=False)

    # 6) Node scaling
    ns_df = compute_node_scaling(df_rw)
    ns_df = mark_nstar(ns_df, args.nstar_eps)
    ns_df.to_csv(outdir / "node_scaling_by_region.csv", index=False)

    # 7) Synergy
    syn_df = compute_synergy(df_rw, store_cols, args.pair_limit)
    syn_df.to_csv(outdir / "synergy_by_region.csv", index=False)

    # 8) Critical-path prevalence per region
    cp_df = critical_path_prevalence(df_rw)
    cp_df.to_csv(outdir / "critical_path_prevalence_by_region.csv", index=False)

    print("Wrote:")
    for p in [rowwise_path, regions_path, outdir / "regret_per_region.csv", outdir / "regret_per_config.csv",
              outdir / "top_regions_store_effects.csv", outdir / "node_scaling_by_region.csv",
              outdir / "synergy_by_region.csv", outdir / "critical_path_prevalence_by_region.csv"]:
        print(" -", p)

if __name__ == "__main__":
    main()
