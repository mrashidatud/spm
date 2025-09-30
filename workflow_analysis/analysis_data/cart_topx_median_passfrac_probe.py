#!/usr/bin/env python3
# cart_topx_median_passfrac_probe.py
"""
CART pruning-path debug with TOP-X median separation using a PASS FRACTION (no bootstrap)

What this does
--------------
- Fits a CART model (stores one-hot + optional nodes).
- Walks the cost–complexity pruning path (from many leaves → few).
- For each subtree:
    • Order leaves by **median(total)** ascending (lower is better).
    • Focus on the **top-X** leaves (X smallest medians).
    • For each adjacent pair among those top-X leaves (i, i+1), mark a **pass**
      if:  median_{i+1} >= (1 + Y) * median_{i}  (Y is a relative gap, e.g., 0.10 for 10%).
    • Compute the **pass fraction**: (#passes) / (topX_pairs).
- A subtree is **OK** if pass_frac_top ≥ conf_frac (e.g., 0.80) and all leaves have at least `min_leaf` rows.
- Select the **first** OK subtree along the path (most detailed that meets the rule).

No bootstrap. No overlap checks. Purely deterministic median gaps.

Outputs (in --outdir)
---------------------
- cart_topx_median_passfrac_candidates.csv
  Columns per pruning step (alpha):
    alpha_idx, ccp_alpha,
    n_leaves, min_size, med_size, max_size,
    topX_n, pairs_topX, n_pass_topX, n_fail_topX, pass_frac_topX,
    min_adjacent_ratio_minus1_topX, ok_topX

Usage example
-------------
python cart_topx_median_passfrac_probe.py \
  --input 1kgenome/workflow_makespan_stageorder.csv \
  --outdir sens_out \
  --top-x 3 --y-sep 0.10 --conf-frac 0.90 \
  --min-leaf 3 --criterion absolute_error
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# ---------------- basics ----------------
def detect_store_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_stor") or c.endswith("_store")]
    if not cols:
        raise RuntimeError("No *_stor or *_store columns found in input.")
    return cols

def ensure_dense(X):
    try:
        return X.toarray()
    except Exception:
        return np.asarray(X)

def leaf_stats_by_median(y: np.ndarray, leaf_ids: np.ndarray) -> pd.DataFrame:
    """Per-leaf stats ordered by **median(total)** ascending."""
    df = pd.DataFrame({"leaf": leaf_ids, "y": y})
    g = df.groupby("leaf")["y"].agg(
        size="size", mean="mean", median="median", std="std", min="min", max="max"
    ).reset_index()
    return g.sort_values("median").reset_index(drop=True)


# ------------- top-X median pass-fraction summary -------------
def topx_median_passfrac_summary(
    y: np.ndarray,
    leaves: np.ndarray,
    top_x: int,
    y_sep: float,
    min_leaf: int,
    conf_frac: float,
) -> dict:
    """
    Among leaves ordered by **median(total)**, check only the top-X leaves.
    Adjacent pair (i,i+1) passes if median_{i+1} >= (1 + y_sep) * median_{i}.
    Return diagnostics for selection.
    """
    stats = leaf_stats_by_median(y, leaves)  # ordered by median
    n_leaves = len(stats)

    # Leaf-size guardrail (global)
    all_minleaf = bool((stats["size"] >= min_leaf).all())

    # How many leaves are available for top-X
    topX_n = int(min(top_x, n_leaves))
    if topX_n <= 1:
        # No comparisons to make
        return {
            "n_leaves": n_leaves,
            "min_size": int(stats["size"].min()) if n_leaves else 0,
            "med_size": int(stats["size"].median()) if n_leaves else 0,
            "max_size": int(stats["size"].max()) if n_leaves else 0,
            "topX_n": topX_n,
            "pairs_topX": 0,
            "n_pass_topX": 0,
            "n_fail_topX": 0,
            "pass_frac_topX": 1.0,  # vacuously passes
            "min_adjacent_ratio_minus1_topX": np.nan,
            "ok_topX": True and all_minleaf,
        }

    # Adjacent checks within top-X
    top_stats = stats.iloc[:topX_n, :].copy()
    medians = top_stats["median"].to_numpy(dtype=float)

    n_pairs = topX_n - 1
    n_pass = 0
    min_ratio_minus1 = np.inf

    for i in range(n_pairs):
        m_i = medians[i]
        m_j = medians[i + 1]

        if m_i <= 0:
            # Totals should be positive; if not, treat carefully
            ratio_minus1 = np.inf if m_j > 0 else 0.0
            passes = (m_j >= 0.0)
        else:
            ratio = m_j / m_i
            ratio_minus1 = ratio - 1.0
            passes = (ratio_minus1 >= y_sep)

        if passes:
            n_pass += 1
        if np.isfinite(ratio_minus1):
            min_ratio_minus1 = min(min_ratio_minus1, ratio_minus1)

    if not np.isfinite(min_ratio_minus1):
        min_ratio_minus1 = np.nan

    pass_frac = n_pass / n_pairs
    ok = (pass_frac >= conf_frac) and all_minleaf

    return {
        "n_leaves": n_leaves,
        "min_size": int(stats["size"].min()),
        "med_size": int(stats["size"].median()),
        "max_size": int(stats["size"].max()),
        "topX_n": topX_n,
        "pairs_topX": n_pairs,
        "n_pass_topX": int(n_pass),
        "n_fail_topX": int(n_pairs - n_pass),
        "pass_frac_topX": float(pass_frac),
        "min_adjacent_ratio_minus1_topX": float(min_ratio_minus1) if np.isfinite(min_ratio_minus1) else np.nan,
        "ok_topX": bool(ok),
    }


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="CART top-X median separation using pass-fraction (no bootstrap).")
    ap.add_argument("--input", required=True, help="Input CSV with 'total' and *_stor(e) columns (+ optional 'nodes').")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs.")
    # Separation rule
    ap.add_argument("--top-x", type=int, default=5, help="Number of top leaves (by median) to check.")
    ap.add_argument("--y-sep", type=float, default=0.10, help="Required adjacent **median** separation (e.g., 0.10 = 10%).")
    ap.add_argument("--conf-frac", type=float, default=0.80,
                    help="Required fraction of passing adjacent pairs among top-X (e.g., 0.80 for 80%%).")
    # Tree controls
    ap.add_argument("--min-leaf", type=int, default=3, help="CART min_samples_leaf.")
    ap.add_argument("--max-depth", type=int, default=None, help="Optional CART max_depth.")
    ap.add_argument("--criterion", choices=["squared_error", "absolute_error"], default="absolute_error",
                    help="Tree split criterion. 'absolute_error' is more median-like (robust).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    # Build X with OneHot on stores + passthrough nodes (if present)
    store_cols = detect_store_cols(df)
    X_num = ["nodes"] if "nodes" in df.columns else []
    X_all = store_cols + X_num

    # OneHot (dense), compatible with old/new sklearn
    try:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), store_cols)],
            remainder="passthrough",
            sparse_threshold=0.0
        )
    except TypeError:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), store_cols)],
            remainder="passthrough",
            sparse_threshold=0.0
        )

    X = ensure_dense(ct.fit_transform(df[X_all]))
    y = df["total"].to_numpy()

    # Fit base tree
    try:
        base = DecisionTreeRegressor(
            criterion=args.criterion,
            min_samples_leaf=args.min_leaf,
            max_depth=args.max_depth,
            random_state=42
        )
        base.fit(X, y)
    except ValueError:
        # Fallback if 'absolute_error' not available
        base = DecisionTreeRegressor(
            criterion="squared_error",
            min_samples_leaf=args.min_leaf,
            max_depth=args.max_depth,
            random_state=42
        )
        base.fit(X, y)

    # Pruning path
    path = base.cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)  # ascending; smaller alpha = more detailed

    # Evaluate each alpha from detailed → simpler
    rows = []
    chosen_idx = None
    for idx, a in enumerate(alphas):
        try:
            tree = DecisionTreeRegressor(
                criterion=args.criterion,
                min_samples_leaf=args.min_leaf,
                max_depth=args.max_depth,
                random_state=42,
                ccp_alpha=float(a)
            )
            tree.fit(X, y)
        except ValueError:
            tree = DecisionTreeRegressor(
                criterion="squared_error",
                min_samples_leaf=args.min_leaf,
                max_depth=args.max_depth,
                random_state=42,
                ccp_alpha=float(a)
            )
            tree.fit(X, y)

        leaves = tree.apply(X)

        summ = topx_median_passfrac_summary(
            y, leaves,
            top_x=args.top_x,
            y_sep=args.y_sep,
            min_leaf=args.min_leaf,
            conf_frac=args.conf_frac
        )
        rows.append({
            "alpha_idx": idx,
            "ccp_alpha": float(a),
            **summ
        })

        if chosen_idx is None and summ["ok_topX"]:
            chosen_idx = idx  # first OK = most detailed that passes

    cand_df = pd.DataFrame(rows)
    out_path = outdir / "cart_topx_median_passfrac_candidates.csv"
    cand_df.to_csv(out_path, index=False)

    print("\n=== CART top-X median pass-fraction (first 12 rows) ===")
    print(cand_df.head(12).to_string(index=False))
    print(f"\nSaved full table: {out_path}")

    if chosen_idx is not None:
        row = cand_df.iloc[chosen_idx]
        print(
            f"\nFirst subtree meeting TOP-{row['topX_n']} median Y={args.y_sep:.2f} with pass-fraction ≥ {args.conf_frac:.2f}: "
            f"alpha_idx={int(row['alpha_idx'])}, ccp_alpha={row['ccp_alpha']:.6g}, "
            f"n_leaves={int(row['n_leaves'])}, pass_frac_topX={row['pass_frac_topX']:.3f}, "
            f"min_adjacent_ratio_minus1_topX={row['min_adjacent_ratio_minus1_topX']:.6g}"
        )
    else:
        print(
            "\nNo subtree satisfies the TOP-X median pass-fraction rule.\n"
            "Try lowering --y-sep (e.g., 0.07), lowering --conf-frac (e.g., 0.60), "
            "raising --min-leaf to stabilize medians, or increasing --max-depth."
        )


if __name__ == "__main__":
    main()
