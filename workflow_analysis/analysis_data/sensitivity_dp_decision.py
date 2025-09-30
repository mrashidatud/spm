#!/usr/bin/env python3
"""
Decision-Guiding Sensitivity for HPC Workflows
Constrained 1-D Segmentation (maximize regions under median gap + min size)

Motivation
----------
We want as many interpretable, decision-ready “regions” as possible, with a
guarantee that neighboring regions are practically different, not just noisy.
Instead of relying on CART structure, we:
  1) map each configuration to a single scalar score (ordering key),
  2) sort rows by that key (lower is better),
  3) run *global* dynamic programming to partition the ordered rows into the
     maximum number of contiguous segments such that:
       • each segment has >= min-seg-size rows, and
       • adjacent segments satisfy median_{i+1} >= (1 + y-sep) * median_i.

Policy (deterministic; no bootstrap)
------------------------------------
• Ordering key: default is a robust tree prediction (absolute_error); or '--order-by total'
• DP grid: cutpoints restricted to a coarse grid for tractability. Step = min-seg-size * grid-step.
• Objective: maximize segment count; ties broken by using the path that ends at the last index.

Outputs
-------
 - workflow_rowwise_sensitivities.csv
 - regions_by_total.csv
 - regret_per_region.csv
 - regret_per_config.csv
 - critical_path_prevalence_by_region.csv
 - region_merge_map.csv            (segment interval -> region id)
 - segmentation_adjacent_audit.csv (adjacent median ratios for the chosen segmentation)

Notes
-----
• No ANOVA/overlap/bootstrap tests.
• Robust to outliers via medians and absolute_error ordering.
• If you want stricter or looser regions: tune --y-sep, --min-seg-size, --grid-step.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# ---------------------------- basic utils ----------------------------

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
        raise RuntimeError("No *_stor or *_store columns found in input.")
    return cols

def stage_name_from_store_col(c: str) -> str:
    return c[:-5] if c.endswith("_stor") else (c[:-6] if c.endswith("_store") else c)

def parse_critical_tokens(s: str) -> List[str]:
    s = s or ""
    return [t.strip() for t in s.split("->") if t.strip()]

def _ensure_dense(X):
    try:
        return X.toarray()
    except Exception:
        return np.asarray(X)


# --------------- rowwise attribution & shares (robust parsing) ---------------

def compute_rowwise_sensitivities(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build lam_* components and share columns per row using the 'critical_path' tokens.
    Robust to tokens like 'read_stage(3)' by stripping parenthetical suffixes.
    """
    store_cols = detect_store_cols(df)
    stage_names = [stage_name_from_store_col(c) for c in store_cols]
    stores = sorted({v for c in store_cols for v in df[c].dropna().unique()})
    local_stores = [s for s in stores if s != "beegfs"]

    # numeric coercion for potential timing columns
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
            kind, stg_raw = m.group(1), m.group(2).strip()
            stg = re.sub(r"\(.*\)$", "", stg_raw)  # strip "(...)"

            if kind == "read":
                col = f"read_{stg}"
                val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store is not None:
                    df.at[idx, f"lam_read_{store}"] += val

            elif kind == "write":
                col = f"write_{stg}"
                val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store is not None:
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


# --------------------------- ordering key ---------------------------

def build_design_matrix(df: pd.DataFrame, store_cols: List[str]):
    X_cat = store_cols
    X_num = ["nodes"] if "nodes" in df.columns else []
    X_all = X_cat + X_num
    # OneHot dense; compatible with old/new sklearn
    try:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X_cat)],
            remainder="passthrough",
            sparse_threshold=0.0
        )
    except TypeError:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), X_cat)],
            remainder="passthrough",
            sparse_threshold=0.0
        )
    X = _ensure_dense(ct.fit_transform(df[X_all]))
    return X, X_all, ct

def predict_order_score(
    df: pd.DataFrame,
    store_cols: List[str],
    order_by: str = "pred",
    max_depth: Optional[int] = 8,
    min_leaf: Optional[int] = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Return a scalar score per row used to sort configurations (lower = better).
    order_by='pred': DecisionTreeRegressor w/ absolute_error (robust).
    order_by='total': use observed total directly.
    """
    if order_by == "total":
        return df["total"].to_numpy(dtype=float)

    X, _, _ = build_design_matrix(df, store_cols)
    y = df["total"].to_numpy(dtype=float)

    # robust tree; fallback if absolute_error unavailable
    try:
        tree = DecisionTreeRegressor(
            criterion="absolute_error",
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=random_state
        )
        tree.fit(X, y)
    except ValueError:
        tree = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=random_state
        )
        tree.fit(X, y)
    return tree.predict(X)


# -------------------- constrained segmentation DP --------------------

def build_grid_indices(n: int, min_seg_size: int, grid_step: int) -> List[int]:
    """
    Build monotone grid cut indices P: 0, step, 2*step, ..., n  (ensure n included).
    step = min_seg_size * grid_step (>= min_seg_size).
    """
    step = max(min_seg_size, min_seg_size * max(1, int(grid_step)))
    P = list(range(0, n, step))
    if P[-1] != n:
        P.append(n)
    return P

def precompute_interval_medians(y_sorted: np.ndarray, P: List[int]) -> np.ndarray:
    """
    Precompute medians for intervals defined by grid cuts.
    Return an upper-triangular matrix M where M[a, b] = median of y_sorted[P[a]:P[b]] for a<b, else nan.
    """
    m = len(P)
    M = np.full((m, m), np.nan, dtype=float)
    for a in range(m - 1):
        start = P[a]
        for b in range(a + 1, m):
            end = P[b]
            seg = y_sorted[start:end]
            if seg.size > 0:
                M[a, b] = float(np.median(seg))
    return M

def constrained_segmentation_max_regions(
    y_sorted: np.ndarray,
    P: List[int],
    y_sep: float,
    min_seg_size: int
) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    """
    Dynamic programming over grid P to maximize region count with constraints:
      - segments are contiguous [P[a]:P[b]), b>a,
      - each has size >= min_seg_size (guaranteed by the grid),
      - adjacent medians satisfy   median(b,c) >= (1+y_sep)*median(a,b).

    Returns:
      - list of (start_idx, end_idx) in original *row order* (indices in [0,n]),
      - matrix of precomputed medians M (for auditing).
    """
    n = len(y_sorted)
    m = len(P)
    if m < 2:
        return [(0, n)], np.array([[np.nan]])

    # Precompute medians for all grid intervals
    M = precompute_interval_medians(y_sorted, P)

    # DP arrays
    NEG_INF = -10**9
    dp = np.full((m, m), NEG_INF, dtype=int)  # dp[a,b] = max segments ending with [a,b]
    prev_idx = np.full((m, m), -1, dtype=int) # prev a-index for [a,b]

    # Initialize first segments starting at 0 (a=0)
    for b in range(1, m):
        dp[0, b] = 1
        prev_idx[0, b] = -1  # start

    # For each anchor 'a' (start of current segment), we need quick queries over dp[p,a]
    # indexed by median(p,a), to check threshold <= M[a,b]/(1+y_sep).
    for a in range(1, m - 0):  # a from 1..m-1; when a==m-1 no outgoing
        # Collect predecessors [p,a] with p < a
        meds_prev = []
        dps_prev = []
        idx_prev = []
        for p in range(0, a):
            if dp[p, a] > NEG_INF and np.isfinite(M[p, a]):
                meds_prev.append(M[p, a])
                dps_prev.append(dp[p, a])
                idx_prev.append(p)
        if len(meds_prev) > 0:
            order = np.argsort(meds_prev)
            meds_prev = np.array(meds_prev, dtype=float)[order]
            dps_prev = np.array(dps_prev, dtype=int)[order]
            idx_prev = np.array(idx_prev, dtype=int)[order]
            # prefix max over dps to recover argmax quickly
            pref_best = np.empty_like(dps_prev)
            pref_arg  = np.empty_like(idx_prev)
            best_val = NEG_INF
            best_arg = -1
            for i in range(len(dps_prev)):
                if dps_prev[i] > best_val:
                    best_val = dps_prev[i]
                    best_arg = idx_prev[i]
                pref_best[i] = best_val
                pref_arg[i] = best_arg
        else:
            meds_prev = np.array([], dtype=float)
            dps_prev  = np.array([], dtype=int)
            pref_best = np.array([], dtype=int)
            pref_arg  = np.array([], dtype=int)

        # Now extend to segments [a,b] where b>a
        for b in range(a + 1, m):
            med_ab = M[a, b]
            if not np.isfinite(med_ab):
                continue
            # Candidate 1: start at 0 (only if a==0, already handled)
            best_val = dp[a, b]  # initial NEG_INF
            best_p   = -1

            # Candidate 2: follow some [p,a] whose median <= threshold
            if meds_prev.size > 0:
                T = med_ab / (1.0 + y_sep)
                # rightmost index where meds_prev[idx] <= T
                k = np.searchsorted(meds_prev, T, side="right") - 1
                if k >= 0:
                    val = pref_best[k] + 1
                    if val > best_val:
                        best_val = val
                        best_p = pref_arg[k]

            if best_val > dp[a, b]:
                dp[a, b] = best_val
                prev_idx[a, b] = best_p

    # Choose best path that ends exactly at the last cut m-1 (i.e., covers whole prefix)
    # Among all a < m-1 with dp[a,m-1] finite, pick the one with the largest segment count.
    end_b = m - 1
    best_a = -1
    best_val = NEG_INF
    for a in range(0, end_b):
        if dp[a, end_b] > best_val:
            best_val = dp[a, end_b]
            best_a = a

    if best_a == -1:
        # Fallback: a single segment covering all
        return [(0, n)], M

    # Reconstruct path backwards
    segments = []
    cur_a = best_a
    cur_b = end_b
    while cur_a >= 0:
        start = P[cur_a]
        end = P[cur_b]
        segments.append((start, end))
        prev_a = prev_idx[cur_a, cur_b]
        if prev_a == -1:
            # reached start segment; it must start at 0
            break
        cur_b = cur_a
        cur_a = prev_a

    segments = list(reversed(segments))  # in order
    # Ensure full coverage from 0..n; if the first segment doesn't start at 0 (edge-rare), prepend it.
    if segments and segments[0][0] != 0:
        segments = [(0, segments[0][0])] + segments
    # If the last segment ends before n (edge-rare), append tail.
    if segments and segments[-1][1] != n:
        segments = segments + [(segments[-1][1], n)]

    return segments, M


# ----------------------- summaries & analytics -----------------------

def strict_wildcard_summary_joint_unique(
    df_all: pd.DataFrame,
    df_labeled: pd.DataFrame,
    store_cols: List[str],
    stat_cols_map: dict,
    additional_cols: List[str] | None = None,
    include_nodes: bool = True,
) -> pd.DataFrame:
    """
    Strict wildcard with JOINT uniqueness against succeeding regions.

    A column c in a region may show '*' iff:
      (A) (Conditional-domain) Given the region's fixed columns (single-valued in the region),
          the region spans ALL values of c present in df_all under those fixed values, AND
      (B) (Joint uniqueness) If we star a set S of columns, then for NO later region (by ascending
          median_total) is the tuple of per-column value-sets on all columns except S EQUAL to this
          region's tuple on those columns. If there is a clash, iteratively UNSTAR a column from S
          (chosen to remove the most clashes; tie-broken by shorter rendering), and re-check until
          no clashes remain.

    • 'nodes' is included as a wildcard column when include_nodes=True.
    • Values render as: single value, comma-joined list (if not star-eligible), or '*'.
    """
    additional_cols = additional_cols or []

    # Which columns can show wildcard in the summary?
    wc_cols = list(store_cols)
    if include_nodes and ("nodes" in df_all.columns):
        wc_cols = wc_cols + ["nodes"]

    # Utility: deterministic, comparable signature of a set of values
    def _vals_set(series_like) -> tuple:
        vals = pd.Series(series_like).dropna().astype(str).unique().tolist()
        return tuple(sorted(vals))

    # 1) Order regions by ascending median_total
    reg_order = (
        df_labeled.groupby("region")["total"]
        .agg(median_total="median")
        .reset_index()
        .sort_values("median_total", ascending=True)
    )
    order = reg_order["region"].tolist()
    pos = {r: i for i, r in enumerate(order)}

    # 2) Precompute per-region per-column PRESENT sets (in the labeled data)
    reg_present: dict[int, dict[str, tuple]] = {}
    for reg, grp in df_labeled.groupby("region"):
        reg_present[reg] = {c: _vals_set(grp[c]) for c in wc_cols}

    rows = []
    for reg in order:
        grp = df_labeled[df_labeled["region"] == reg]

        # Fixed columns (single value in region) among wc_cols
        fixed_cols = [c for c in wc_cols if grp[c].nunique(dropna=True) == 1]
        fixed_vals = {c: grp[c].iloc[0] for c in fixed_cols}

        # Conditional slice of df_all under fixed values
        cond = pd.Series(True, index=df_all.index)
        for c, v in fixed_vals.items():
            cond &= (df_all[c] == v)
        df_cond = df_all[cond]

        # Compute: per-column present sets (this region) and conditional domain sets
        present_sets = {c: set(map(str, grp[c].dropna().unique())) for c in wc_cols}
        domain_sets  = {c: set(map(str, df_cond[c].dropna().unique())) for c in wc_cols}

        # Initial star eligibility by conditional-domain rule (A)
        star_eligible = {
            c: (len(present_sets[c]) > 1 and              # region really spans >1 value
                len(domain_sets[c]) > 1 and               # non-trivial domain
                present_sets[c] == domain_sets[c])        # spans the entire conditional domain
            for c in wc_cols
        }
        star_set = {c for c, ok in star_eligible.items() if ok}

        # Build per-region signatures of OTHER columns' value-sets for all later regions
        my_pos = pos[reg]
        later_regs = order[my_pos + 1:]

        # Helper: signature excluding a set S of columns, for a given region id
        def sig_exc(region_id: int, exclude_cols: set[str]) -> tuple:
            # tuple of (colname, tuple(sorted_vals)) for all columns not in exclude_cols, sorted by colname
            parts = []
            for k in wc_cols:
                if k in exclude_cols:
                    continue
                parts.append((k, reg_present[region_id][k]))
            return tuple(sorted(parts, key=lambda x: x[0]))

        # JOINT uniqueness check (B): greedily reduce star_set to avoid clashes with any later region
        def count_clashes(S: set[str]) -> int:
            """Number of later regions whose signature excluding S equals ours."""
            my_sig = sig_exc(reg, S)
            clashes = 0
            for r2 in later_regs:
                if sig_exc(r2, S) == my_sig:
                    clashes += 1
            return clashes

        # Greedy unstar loop
        S = set(star_set)  # working copy
        clashes = count_clashes(S)
        if clashes > 0 and len(S) > 0:
            # iteratively unstar the column that removes the most clashes; tie-breaker favors shorter rendering
            while clashes > 0 and len(S) > 0:
                best_c = None
                best_clash = clashes
                # evaluate each candidate to unstar
                for c in list(S):
                    S_try = set(S); S_try.remove(c)
                    c_clashes = count_clashes(S_try)
                    if c_clashes < best_clash:
                        best_clash = c_clashes
                        best_c = c
                    elif c_clashes == best_clash:
                        # tie-break: prefer un-starring the column with the smallest present set (shorter cell)
                        if best_c is None or len(present_sets[c]) < len(present_sets[best_c]):
                            best_c = c
                if best_c is None:
                    break
                S.remove(best_c)
                clashes = best_clash

        # Now render cells
        row_wc: dict[str, str] = {}
        for c in wc_cols:
            if len(present_sets[c]) == 1:
                row_wc[c] = next(iter(present_sets[c]))
            elif c in S:
                row_wc[c] = "*"
            else:
                row_wc[c] = ",".join(sorted(present_sets[c]))

        # ---- stats block (same as your existing summary) ----
        def stat(series: pd.Series, how: str):
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
                return float(np.min(x)) if x.size else np.nan
            if how == "max":
                return float(np.max(x)) if x.size else np.nan
            return x.size

        stats_out = {name: stat(grp[src], how) for name, (src, how) in stat_cols_map.items()}
        for col in additional_cols:
            stats_out[f"mean_{col}"] = stat(grp[col], "mean")
            stats_out[f"cv_{col}"]   = stat(grp[col], "cv")

        rows.append({
            "region": int(reg),
            "size": int(len(grp)),
            **stats_out,
            **row_wc
        })

    out = pd.DataFrame(rows).sort_values("median_total", ascending=True).reset_index(drop=True)
    return out

def compute_regret(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_best = df["total"].min()
    cfg = df[["region", "nodes", "total"] + detect_store_cols(df)].copy()
    cfg["regret_vs_global_best"] = cfg["total"] - global_best
    reg = df.groupby("region")["total"].mean().reset_index().rename(columns={"total": "mean_total"})
    reg["regret_vs_global_best"] = reg["mean_total"] - global_best
    return reg, cfg

def critical_path_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stores = detect_store_cols(df)
    for reg, grp in df.groupby("region"):
        size = len(grp)
        counts = grp["critical_path"].value_counts()
        prevalent = counts.index[0] if not counts.empty else ""
        for path, cnt in counts.items():
            rep = grp[grp["critical_path"] == path].iloc[0]
            base = {
                "region": reg,
                "region_size": size,
                "prevalent_critical_path": prevalent,
                "critical_path": path,
                "count": int(cnt),
                "fraction": float(cnt / size) if size else 0.0,
                "config_nodes": int(rep["nodes"]) if "nodes" in rep else None,
            }
            for c in stores:
                base[c] = rep[c]
            rows.append(base)
    out = pd.DataFrame(rows).sort_values(["region", "count"], ascending=[True, False])
    return out


# -------------------------------- MAIN --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sensitivity & region analysis via constrained 1-D segmentation.")
    ap.add_argument("--input", required=True, help="Input CSV (e.g., workflow_makespan_stageorder.csv)")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs")

    # Segmentation constraints
    ap.add_argument("--y-sep", type=float, default=0.10,
                    help="Required adjacent **median** separation (e.g., 0.10 = 10%).")
    ap.add_argument("--min-seg-size", type=int, default=3,
                    help="Minimum rows per segment (region).")
    ap.add_argument("--grid-step", type=int, default=1,
                    help="Coarsening factor for DP grid (>=1). Effective step = min-seg-size * grid-step.")

    # Ordering key options
    ap.add_argument("--order-by", choices=["pred", "total"], default="pred",
                    help="Sort rows by a robust prediction ('pred') or raw 'total'.")
    ap.add_argument("--max-depth", type=int, default=8, help="Depth for the ordering tree (if order-by='pred').")
    ap.add_argument("--order-min-leaf", type=int, default=5, help="Min leaf for ordering tree (if order-by='pred').")

    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load & enrich
    df0 = pd.read_csv(args.input)
    df, store_cols = compute_rowwise_sensitivities(df0.copy())

    # Ordering key and sort (lower is better)
    order_score = predict_order_score(
        df, store_cols,
        order_by=args.order_by,
        max_depth=args.max_depth,
        min_leaf=args.order_min_leaf,
        random_state=42
    )
    df["_order_score"] = order_score
    df["_ord_idx"] = np.argsort(order_score)
    df_sorted = df.sort_values("_order_score", kind="mergesort").reset_index(drop=True)

    # Build grid and run constrained segmentation on sorted rows
    y_sorted = df_sorted["total"].to_numpy(dtype=float)
    n = len(y_sorted)

    if n < args.min_seg_size:
        # not enough data for multiple segments
        segments = [(0, n)]
        M = np.array([[np.nan]])
    else:
        P = build_grid_indices(n, min_seg_size=args.min_seg_size, grid_step=args.grid_step)
        segments, M = constrained_segmentation_max_regions(
            y_sorted=y_sorted, P=P, y_sep=args.y_seP if hasattr(args, 'y_seP') else args.y_sep,
            min_seg_size=args.min_seg_size
        )

    # Assign region ids (0..K-1) by segment order
    region_ids = np.full(n, -1, dtype=int)
    for rid, (s, e) in enumerate(segments):
        region_ids[s:e] = rid
    df_sorted["region"] = region_ids

    # Map back to original row order
    df_out = df_sorted.sort_values("_ord_idx").drop(columns=["_ord_idx"]).reset_index(drop=True)

    # Build a readable mapping for segments
    seg_map_rows = []
    for rid, (s, e) in enumerate(segments):
        seg_map_rows.append({
            "region": rid,
            "sorted_start_idx": s,
            "sorted_end_idx": e,
            "size": int(e - s),
            "median_total": float(np.median(y_sorted[s:e])),
        })
    seg_map = pd.DataFrame(seg_map_rows).sort_values("region")
    seg_map.to_csv(outdir / "region_merge_map.csv", index=False)

    # Segmentation adjacent audit (medians & ratios)
    audit_rows = []
    for i in range(len(segments) - 1):
        s1, e1 = segments[i]
        s2, e2 = segments[i+1]
        med1 = float(np.median(y_sorted[s1:e1])) if e1 > s1 else np.nan
        med2 = float(np.median(y_sorted[s2:e2])) if e2 > s2 else np.nan
        ratio_minus1 = (med2 / med1 - 1.0) if med1 > 0 else np.inf
        audit_rows.append({
            "left_region": i,
            "right_region": i+1,
            "median_left": med1,
            "median_right": med2,
            "ratio_minus1": ratio_minus1,
            "passes": (ratio_minus1 >= args.y_sep)
        })
    pd.DataFrame(audit_rows).to_csv(outdir / "segmentation_adjacent_audit.csv", index=False)

    # Save rowwise output
    df_out.to_csv(outdir / "workflow_rowwise_sensitivities.csv", index=False)

    # Region summaries (strict wildcard + share stats), ordered by median_total
    stat_cols_map = {
        "mean_total":   ("total", "mean"),
        "median_total": ("total", "median"),
        "std_total":    ("total", "std"),
        "cv_total":     ("total", "cv"),
        "q25_total":    ("total", "q25"),
        "q75_total":    ("total", "q75"),
        "min_total":    ("total", "min"),
        "max_total":    ("total", "max"),
    }
    regions_df = strict_wildcard_summary_joint_unique(
        df_all=df,                      # the full input table (before/after labeling; use full input domain)
        df_labeled=df_out,               # the labeled rows with region column
        store_cols=store_cols,
        stat_cols_map=stat_cols_map,
        additional_cols=["exec_beegfs_share", "exec_local_share", "movement_share"],
        include_nodes=True,             # <-- nodes included in wildcard logic
    )

    regions_df.to_csv(outdir / "regions_by_total.csv", index=False)

    # Regret (merged regions)
    regret_region, regret_config = compute_regret(df_out)
    regret_region.to_csv(outdir / "regret_per_region.csv", index=False)
    regret_config.to_csv(outdir / "regret_per_config.csv", index=False)

    # Critical-path prevalence per region
    cp_df = critical_path_prevalence(df_out)
    cp_df.to_csv(outdir / "critical_path_prevalence_by_region.csv", index=False)

    print("Wrote:")
    for p in [
        outdir / "workflow_rowwise_sensitivities.csv",
        outdir / "regions_by_total.csv",
        outdir / "regret_per_region.csv",
        outdir / "regret_per_config.csv",
        outdir / "critical_path_prevalence_by_region.csv",
        outdir / "region_merge_map.csv",
        outdir / "segmentation_adjacent_audit.csv",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
