#!/usr/bin/env python
# cart_summary.py
# -----------------------------------------------------------
# CART region discovery with:
#  • schema-driven stage order (workflow/*.json or --schema)
#  • *_store and *_stor detection
#  • full CCP-path leaf targeting + residual fallback
#  • NEW: variance-controlled leaf refinement (max CV / std)
#  • optional trimmed statistics in output
# -----------------------------------------------------------
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------- encoders & helpers ----------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def detect_store_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c.endswith("_store") or c.endswith("_stor")]
    if not cols:
        raise ValueError("No stage↔storage columns found (expected *_store or *_stor).")
    return cols

def load_stage_order_map(workflow_dir: Path, store_cols: list[str], schema_path: str | None):
    def extract_from_dict(d: dict) -> dict[str,int]:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict) and "stage_order" in v:
                s = str(k)
                if f"{s}_store" in store_cols or f"{s}_stor" in store_cols:
                    try:
                        out[s] = int(v["stage_order"])
                    except Exception:
                        pass
        return out
    if schema_path:
        p = Path(schema_path)
        with p.open() as f: data = json.load(f)
        return extract_from_dict(data)
    best_map, best_overlap = {}, -1
    for p in workflow_dir.glob("*.json"):
        try:
            with p.open() as f: data = json.load(f)
            m = extract_from_dict(data)
            if len(m) > best_overlap:
                best_map, best_overlap = m, len(m)
        except Exception:
            continue
    return best_map

def order_store_cols(store_cols: list[str], stage_order_map: dict[str,int]) -> list[str]:
    ordered_stage_names = sorted(stage_order_map.keys(), key=lambda s: (stage_order_map[s], s))
    ordered = []
    for s in ordered_stage_names:
        for suf in ("_store", "_stor"):
            c = s + suf
            if c in store_cols:
                ordered.append(c); break
    for c in store_cols:
        if c not in ordered:
            ordered.append(c)
    return ordered

def build_encoder(store_cols: list[str], include_nodes: bool):
    feats = [("cat", make_ohe(), store_cols)]
    if include_nodes:
        feats.append(("num", "passthrough", ["nodes"]))
    return ColumnTransformer(feats)

def encode(enc, X):
    Xenc = enc.fit_transform(X)
    return Xenc.toarray() if hasattr(Xenc, "toarray") else Xenc

def choose_tree_by_target_leaves(Xenc, y, criterion, max_depth, min_leaf, target_leaves):
    base = DecisionTreeRegressor(random_state=0, criterion=criterion)
    base.fit(Xenc, y)
    path = base.cost_complexity_pruning_path(Xenc, y)
    alphas = np.unique(path.ccp_alphas)
    alphas = np.insert(alphas[alphas > 0], 0, 0.0)
    best = None
    for a in alphas:
        t = DecisionTreeRegressor(random_state=0, criterion=criterion,
                                  ccp_alpha=a, max_depth=max_depth,
                                  min_samples_leaf=min_leaf)
        t.fit(Xenc, y)
        leaves = np.unique(t.apply(Xenc)).size
        cand = (abs(leaves - target_leaves), -leaves, a, t)
        if best is None or cand < (best[0], best[1], best[2], best[3]):
            best = cand
    _, _, best_alpha, best_tree = best
    return best_alpha, best_tree

# ---------- stats & summaries ----------
def trimmed_stats(x: np.ndarray, trim_pct: float):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    if trim_pct <= 0 or trim_pct*2 >= 1.0:
        return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0)
    x_sorted = np.sort(x)
    k = int(np.floor(len(x_sorted) * trim_pct))
    if k*2 >= len(x_sorted):
        return float(np.mean(x_sorted)), float(np.std(x_sorted, ddof=1) if len(x_sorted) > 1 else 0.0)
    t = x_sorted[k: len(x_sorted)-k]
    return float(np.mean(t)), float(np.std(t, ddof=1) if t.size > 1 else 0.0)

def summarise_regions_strict(df_all: pd.DataFrame,
                             df_regioned: pd.DataFrame,
                             ordered_store_cols: list[str],
                             trim_pct: float):
    """
    Strict wildcard policy:
      * Only use '*' for a column c if, conditional on the region's fixed columns,
        the region contains ALL possible assignments of c that appear in df_all.
      * If c is constant inside the region, print that value.
      * Otherwise print the comma-separated values observed in the region.
    """
    def trimmed_stats(x: np.ndarray, trim_pct: float):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.nan, np.nan
        if trim_pct <= 0 or trim_pct * 2 >= 1.0:
            return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0)
        x_sorted = np.sort(x)
        k = int(np.floor(len(x_sorted) * trim_pct))
        k = min(k, (len(x_sorted) - 1) // 2)
        t = x_sorted[k: len(x_sorted) - k] if k > 0 else x_sorted
        return float(np.mean(t)), float(np.std(t, ddof=1) if t.size > 1 else 0.0)

    rows = []
    for reg, grp in df_regioned.groupby("region"):
        # Fixed columns in this region (including nodes if constant)
        fixed_cols = [c for c in ordered_store_cols + ["nodes"] if grp[c].nunique() == 1]
        fixed_vals = {c: grp[c].iat[0] for c in fixed_cols}

        # Full conditional slice from the original data
        cond = pd.Series(True, index=df_all.index)
        for c, v in fixed_vals.items():
            cond &= (df_all[c] == v)
        df_cond = df_all[cond]

        # Build the row with strict wildcard logic
        row = {}
        for c in ordered_store_cols:
            present = set(grp[c].astype(str).unique())
            if grp[c].nunique() == 1:
                # Constant inside region → print the value, not '*'
                row[c] = next(iter(present))
            else:
                domain = set(df_cond[c].astype(str).unique())
                # Wildcard only if region covers the full conditional domain (>1 level)
                if len(domain) > 1 and present == domain:
                    row[c] = "*"
                else:
                    row[c] = ",".join(sorted(present))

        # Stats (optionally trimmed)
        mean, std = trimmed_stats(grp["total"].values, trim_pct)
        cv = (std / mean) if (mean and mean != 0) else np.nan
        nodes_str = ",".join(map(str, sorted(map(int, pd.unique(grp["nodes"])))))

        rows.append(dict(
            region=int(reg) if isinstance(reg, (int, np.integer)) else reg,
            size=len(grp),
            mean=mean,
            std=std,
            cv=cv,
            **row,
            nodes=nodes_str
        ))

    out = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    # Enforce column order: [region,size,mean,std,cv, stages..., nodes]
    return out[["region", "size", "mean", "std", "cv"] + ordered_store_cols + ["nodes"]]

# ---------- refinement ----------
def refine_regions(df: pd.DataFrame,
                   ordered_store_cols: list[str],
                   include_nodes: bool,
                   min_leaf: int,
                   refine_depth: int,
                   refine_passes: int,
                   max_rel_std: float | None,
                   max_std: float | None,
                   criterion_refine: str = "absolute_error"):
    """
    Iteratively split any region whose variability exceeds thresholds.
    Splits are performed using a small local tree on that region only.
    """
    df = df.copy()
    for _ in range(refine_passes):
        splits = 0
        for reg, g in list(df.groupby("region")):
            vals = g["total"].values
            if len(vals) < 2 * min_leaf:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            too_noisy = False
            if max_rel_std is not None and mean > 0:
                if std / mean > max_rel_std:
                    too_noisy = True
            if max_std is not None and std > max_std:
                too_noisy = True
            if not too_noisy:
                continue

            # Fit a small local tree on this region
            enc_local = build_encoder(ordered_store_cols, include_nodes)
            Xg = g[ordered_store_cols + (["nodes"] if include_nodes else [])]
            yg = g["total"].values
            Xg_enc = encode(enc_local, Xg)
            tree = DecisionTreeRegressor(random_state=0,
                                         criterion=criterion_refine,
                                         ccp_alpha=0.0,
                                         max_depth=refine_depth,
                                         min_samples_leaf=min_leaf)
            tree.fit(Xg_enc, yg)
            leaf_ids = tree.apply(Xg_enc)
            uniq = np.unique(leaf_ids)
            if uniq.size <= 1:
                continue  # no useful split
            # assign new composite region labels
            new_labels = [f"{reg}-{int(l)}" for l in leaf_ids]
            df.loc[g.index, "region"] = new_labels
            splits += (uniq.size - 1)
        if splits == 0:
            break
    return df

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="CART region summariser with schema order and variance-controlled refinement.")
    # workflow-aware I/O
    ap.add_argument("--workflow", type=str, help="Workflow dir")
    ap.add_argument("--input", type=str, help="Override input CSV")
    ap.add_argument("--output", type=str, help="Override output CSV")
    ap.add_argument("--schema", type=str, default=None, help="Explicit schema JSON (else auto-detect in --workflow)")
    # model controls
    ap.add_argument("--target-regions", type=int, default=12)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-leaf", type=int, default=5)
    ap.add_argument("--criterion", default="squared_error",
                    choices=["squared_error","absolute_error","poisson"])
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--ignore-nodes", action="store_true", help="Exclude 'nodes' as a predictor")
    ap.add_argument("--fallback-min-regions", type=int, default=6,
                    help="If initial result yields < this many regions, residualize by nodes and refit")
    # variance & summary controls
    ap.add_argument("--max-rel-std", type=float, default=0.15,
                    help="Max coefficient of variation per region (std/mean). Set <=0 to disable.")
    ap.add_argument("--max-std", type=float, default=None,
                    help="Max absolute std per region (optional).")
    ap.add_argument("--refine-depth", type=int, default=2,
                    help="Depth of local refinement trees (default 2)")
    ap.add_argument("--refine-passes", type=int, default=2,
                    help="Max refinement iterations (default 2)")
    ap.add_argument("--trim-pct", type=float, default=0.0,
                    help="Trim fraction for mean/std (e.g., 0.05 trims 5%% off each tail)")
    args = ap.parse_args()

    # Paths
    if not args.input or not args.output:
        if not args.workflow:
            ap.error("Provide --workflow OR both --input and --output.")
        wf = Path(args.workflow)
        args.input = args.input or str(wf / "workflow_makespan_stageorder.csv")
        args.output = args.output or str(wf / "cart_regions.csv")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load & schema
    df = pd.read_csv(args.input)
    store_cols_all = detect_store_cols(df)
    wf_dir = Path(args.workflow) if args.workflow else Path(args.input).parent
    stage_order_map = load_stage_order_map(wf_dir, store_cols_all, args.schema)
    ordered_store_cols = order_store_cols(store_cols_all, stage_order_map)

    # Target
    y_col = "total"
    if args.log_target:
        df = df.copy()
        df["_y"] = np.log(df["total"].astype(float) + 1e-9)
        y_col = "_y"

    include_nodes = (not args.ignore_nodes)

    def fit_tree_on(df_in: pd.DataFrame, target_col: str, tag: str = ""):
        enc = build_encoder(ordered_store_cols, include_nodes)
        X = df_in[ordered_store_cols + (["nodes"] if include_nodes else [])]
        y = df_in[target_col].values
        Xenc = encode(enc, X)
        _, tree = choose_tree_by_target_leaves(
            Xenc, y, args.criterion, args.max_depth, args.min_leaf, args.target_regions
        )
        leaf = tree.apply(Xenc)
        out = df_in.copy()
        out["region"] = [f"{tag}{int(l)}" for l in leaf]
        return out

    # Pass 1: original target
    assigned = fit_tree_on(df, y_col)
    regions1 = summarise_regions_strict(df, assigned, ordered_store_cols, args.trim_pct)
    if len(regions1) < args.fallback_min_regions:
        # residualize by nodes (linear space)
        df_res = df.copy()
        per_node_mean = df_res.groupby("nodes")["total"].transform("mean")
        df_res["_resid"] = df_res["total"] - per_node_mean
        assigned = fit_tree_on(df_res, "_resid", tag="R:")
        # note: we still evaluate refinement & summaries using 'total'

    # Refinement to enforce variance caps (evaluate on 'total')
    assigned_refined = refine_regions(
        df=assigned.assign(total=df["total"].values),  # ensure 'total' present
        ordered_store_cols=ordered_store_cols,
        include_nodes=include_nodes,
        min_leaf=args.min_leaf,
        refine_depth=args.refine_depth,
        refine_passes=args.refine_passes,
        max_rel_std=(args.max_rel_std if args.max_rel_std and args.max_rel_std > 0 else None),
        max_std=args.max_std,
        criterion_refine="absolute_error",
    )

    # Final summary (trimmed stats optional)
    summary = summarise_regions_strict(df, assigned_refined, ordered_store_cols, args.trim_pct)
    summary.to_csv(args.output, index=False)
    print(f"Written: {args.output}")
    print(f"Regions: {len(summary)} | refine_passes={args.refine_passes} | "
          f"max_rel_std={args.max_rel_std} | trim_pct={args.trim_pcnt if hasattr(args,'trim_pcnt') else args.trim_pct}")

if __name__ == "__main__":
    main()
