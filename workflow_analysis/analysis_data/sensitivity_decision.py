#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity & Region Analysis for Workflow Makespans
CART with Repeated-CV Statistical Separation (All Leaves) + MAE-aware Alpha Selection
-------------------------------------------------------------------------------------

Motivation
----------
We want interpretable, axis-aligned regions (CART leaves) that are *meaningfully different*,
not just low-impurity. Instead of selecting the tree purely by error or a bespoke "top-X"
rule, we select the pruning level (ccp_alpha) by a **statistical separation metric across
all adjacent leaves** (ordered by median total), and we **factor in MAE** to avoid picking
a model that separates well but predicts poorly.

What this script does
---------------------
1) Builds row-wise sensitivities (λ terms) from the critical path and computes execution/
   movement shares.
2) Builds a CART design matrix (One-Hot for storage columns; passes through `nodes`).
3) Creates a cost-complexity **alpha grid** and runs **Repeated K-fold CV**. For each alpha:
   • Train on K−1 folds (optionally with an **inner split-acceptance gate** that disallows
     complexity increases that harm MAE or separation on an inner 80/20 validation slice).
   • Evaluate on outer folds using a **statistical separation metric over ALL adjacent leaves**:
       - Order leaves by median(total).
       - For each adjacent pair (i,i+1), compute **Hedges’ g** (absolute) when both sides
         have at least `--n-min-pair` rows; count it "significant" if |g| ≥ `--g-min`.
       - Aggregate into a single fold score via:
            avg_effect  := weighted mean |g| among significant pairs;
            frac_sig    := weighted fraction of significant adjacencies;
            product     := avg_effect × frac_sig  (recommended default).
         Weights are either min(n_i,n_j) or the harmonic mean 2/(1/n_i+1/n_j).
   • Also record **outer-fold MAE** on totals.
4) **Alpha selection** (across repeats) using your choice of:
   • pass_only      : maximize separation only (sep_med).
   • constrained    : maximize separation among alphas whose median MAE ≤ (1+mae_budget)*best.
   • weighted       : maximize blended score J = (1−w)*sep_norm + w*(1−mae_norm).
5) Final fit at the chosen α; (Optional) **post-merge** adjacent leaves to union them into
   regions if a δ/τ gap or quantile guard fails (never re-splits CART).
6) Summarizes regions with **joint-unique strict wildcard** notation (including nodes)
   and writes standard diagnostics (regions, regret, critical-path prevalence, CV candidates).

Key options (new/changed)
-------------------------
 --separation-metric {stat_g_all, pass_wfrac}    # default stat_g_all (all adjacencies; Hedges' g)
 --g-min FLOAT (default 0.30)                    # minimum |g| to count as "different"
 --n-min-pair INT (default 5)                    # minimum rows per leaf on both sides to evaluate a pair
 --stat-score {avg_effect, frac_sig, product}    # fold-level separation score (default product)

 --select-mode {constrained, pass_only, weighted}  # how to combine separation & MAE (default constrained)
 --mae-budget FLOAT (default 0.02)                 # for constrained: within +2% of best median MAE
 --mae-weight FLOAT (default 0.30)                 # for weighted blend

The previous top-X pass-fraction metric is still available via --separation-metric pass_wfrac,
but it is no longer the default. When using stat_g_all, top-frac/max-topN are ignored.

Outputs
-------
 - workflow_rowwise_sensitivities.csv
 - regions_by_total.csv               (joint-unique wildcard, includes nodes)
 - regret_per_region.csv
 - regret_per_config.csv
 - critical_path_prevalence_by_region.csv
 - region_merge_map.csv               (CART leaf → ordered region id)
 - cart_cv_candidates.csv             (per-α × fold diagnostics; sep_val + mae)
 - cart_postmerge_adjacent_audit.csv  (if --postmerge)
 - cart_postmerge_map.csv             (if --postmerge)

Example
-------
python sensitivity_decision.py \
  --input 1kgenome/workflow_makespan_stageorder.csv \
  --outdir sens_out \
  --kfolds 5 --repeats 5 \
  --separation-metric stat_g_all --g-min 0.30 --n-min-pair 5 --stat-score product \
  --pair-weight hmean --robust-quantiles \
  --min-leaf 3 --max-depth 12 --criterion absolute_error \
  --select-mode constrained --mae-budget 0.02 \
  --gate-splits \
  --postmerge --min-region-size 3 --abs-gap-tau 0.0 --min-guard-n 10
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import re
import math

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, train_test_split

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

def _cv_value(arr: np.ndarray, estimator: str = "std") -> float:
    """Return CV = sd/mean or (1.4826*MAD)/median (robust) with guards."""
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return np.inf
    if estimator == "mad":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sd = 1.4826 * mad
        mu = med
    else:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        mu = float(np.mean(x))
    if not np.isfinite(mu) or mu <= 0:
        return np.inf
    return sd / mu

def _pooled_cv(cv_i: float, cv_j: float) -> float:
    if not np.isfinite(cv_i) or not np.isfinite(cv_j):
        return np.inf
    return math.sqrt(0.5 * (cv_i * cv_i + cv_j * cv_j))

def _pair_g_threshold(delta: Optional[float],
                      cv_pooled: float,
                      g_min_fixed: float,
                      use_auto: bool,
                      g_floor: float,
                      g_cap: float) -> float:
    """Return the g threshold to use for this pair."""
    if use_auto and (delta is not None) and np.isfinite(cv_pooled) and cv_pooled > 0:
        g_auto = delta / cv_pooled
        g_thr = max(g_floor, min(g_cap, g_auto))
        return float(g_thr)
    return float(g_min_fixed)

# --------------- rowwise attribution & shares ---------------

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
                col = f"read_{stg}"; val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store is not None:
                    df.at[idx, f"lam_read_{store}"] += val
            elif kind == "write":
                col = f"write_{stg}"; val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store is not None:
                    df.at[idx, f"lam_write_{store}"] += val
            elif kind == "stage_in":
                col = f"in_{stg}"; val = row[col] if col in df.columns else 0.0
                df.at[idx, "lam_in_total"] += val
            elif kind == "stage_out":
                col = f"out_{stg}"; val = row[col] if col in df.columns else 0.0
                df.at[idx, "lam_out_total"] += val

    # shares
    eps = 1e-12
    if "beegfs" in stores:
        df["exec_beegfs_share"] = (df["lam_read_beegfs"] + df["lam_write_beegfs"]) / (df["total"] + eps)
    else:
        df["exec_beegfs_share"] = 0.0
    if local_stores:
        df["exec_local_share"] = sum(df[f"lam_read_{s}"] + df[f"lam_write_{s}"] for s in local_stores) / (df["total"] + eps)
    else:
        df["exec_local_share"] = 0.0
    df["movement_share"] = (df["lam_in_total"] + df["lam_out_total"]) / (df["total"] + eps)

    for c in ["exec_beegfs_share", "exec_local_share", "movement_share"]:
        df[c] = df[c].fillna(0.0).clip(lower=0.0, upper=1.0)

    return df, store_cols

# --------------------------- CART design matrix ---------------------------

def build_cart_design(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]):
    all_cols = cat_cols + num_cols
    try:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
            remainder="passthrough", sparse_threshold=0.0
        )
    except TypeError:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)],
            remainder="passthrough", sparse_threshold=0.0
        )
    X = _ensure_dense(ct.fit_transform(df[all_cols]))
    return X, ct, all_cols

# --------------------------- quantiles (robust option) ---------------------------

def qhat(x: np.ndarray, q: float, robust: bool = False) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0:
        return float("nan")
    if not robust:
        return float(np.quantile(x, q))
    try:
        return float(np.quantile(x, q, method="hazen"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="linear"))

# --------------------------- legacy top-X metric (optional) ---------------------------

def _leaf_stats_by_median(y: np.ndarray, leaf_ids: np.ndarray, robust_q: bool) -> pd.DataFrame:
    df = pd.DataFrame({"leaf": leaf_ids, "y": y})
    rows = []
    for leaf, grp in df.groupby("leaf"):
        arr = grp["y"].to_numpy(dtype=float)
        rows.append({
            "leaf": int(leaf),
            "size": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": qhat(arr, 0.5, robust=robust_q),
            "q25": qhat(arr, 0.25, robust=robust_q),
            "q75": qhat(arr, 0.75, robust=robust_q),
        })
    g = pd.DataFrame(rows)
    return g.sort_values("median").reset_index(drop=True)

def _pair_weight(n_i: int, n_j: int, mode: str = "min") -> float:
    if mode == "hmean":
        if n_i <= 0 or n_j <= 0:
            return 0.0
        return 2.0 / (1.0 / n_i + 1.0 / n_j)
    return float(min(n_i, n_j))

def topfrac_pass_metric(
    y: np.ndarray,
    leaf_ids: np.ndarray,
    top_frac: float = 0.25,
    max_topN: int = 12,
    y_sep: float = 0.10,
    use_overlap_guard: bool = True,
    robust_q: bool = False,
    pair_weight: str = "min",
) -> Dict[str, float]:
    stats = _leaf_stats_by_median(y, leaf_ids, robust_q=robust_q)
    L = len(stats)
    if L <= 1:
        return dict(skip=True, n_leaves=L, topN=min(L,1), pairs=0,
                    pass_pairs=0, pass_frac=float("nan"),
                    w_num=0.0, w_den=0.0, pass_wfrac=float("nan"),
                    min_gap_ratio_minus1=float("nan"))
    topN = max(3, int(np.ceil(top_frac * L)))
    topN = min(topN, max_topN, L)
    if topN < 2:
        return dict(skip=True, n_leaves=L, topN=topN, pairs=0,
                    pass_pairs=0, pass_frac=float("nan"),
                    w_num=0.0, w_den=0.0, pass_wfrac=float("nan"),
                    min_gap_ratio_minus1=float("nan"))
    s = stats.iloc[:topN].reset_index(drop=True)

    pass_pairs = 0; w_num = 0.0; w_den = 0.0; min_gap = np.inf
    for i in range(topN - 1):
        m_i = float(s.loc[i, "median"]); m_j = float(s.loc[i+1, "median"])
        q75_i = float(s.loc[i, "q75"]);  q25_j = float(s.loc[i+1, "q25"])
        n_i   = int(s.loc[i, "size"]);   n_j   = int(s.loc[i+1, "size"])

        ratio_minus1 = np.inf if m_i <= 0 else (m_j / m_i) - 1.0
        gap_ok = (ratio_minus1 >= y_sep)
        ov_ok  = (q75_i * (1.0 + 0.5*y_sep) <= q25_j) if use_overlap_guard else True
        ok     = (gap_ok and ov_ok)

        w = _pair_weight(n_i, n_j, mode=pair_weight)
        w_den += w
        if ok:
            pass_pairs += 1; w_num += w
        if np.isfinite(ratio_minus1):
            min_gap = min(min_gap, ratio_minus1)

    pass_frac = (pass_pairs / (topN - 1)) if (topN - 1) > 0 else float("nan")
    pass_wfrac = (w_num / w_den) if w_den > 0 else float("nan")
    if not np.isfinite(min_gap):
        min_gap = float("nan")

    return dict(skip=False, n_leaves=L, topN=topN, pairs=topN-1,
                pass_pairs=int(pass_pairs), pass_frac=float(pass_frac),
                w_num=float(w_num), w_den=float(w_den), pass_wfrac=float(pass_wfrac),
                min_gap_ratio_minus1=float(min_gap))

# --------------------------- NEW: effect size metric over ALL leaves ---------------------------

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased standardized mean difference (Hedges' g)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    denom = (nx + ny - 2)
    if denom <= 0:
        return np.nan
    s_p = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / denom) if denom > 0 else np.nan
    if not np.isfinite(s_p) or s_p == 0:
        return np.nan
    d = (my - mx) / s_p
    J = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0)) if (nx + ny) > 2 else 1.0
    return float(J * d)

def _leaf_stats_with_rows(y: np.ndarray, leaf_ids: np.ndarray, robust_q: bool):
    df = pd.DataFrame({"leaf": leaf_ids, "y": y, "_row": np.arange(len(y))})
    rows = []
    for leaf, grp in df.groupby("leaf"):
        arr = grp["y"].to_numpy(dtype=float)
        rows.append({
            "leaf": int(leaf),
            "size": int(arr.size),
            "median": qhat(arr, 0.5, robust=robust_q),
            "rows": tuple(grp["_row"].to_numpy())
        })
    g = pd.DataFrame(rows)
    return g.sort_values("median").reset_index(drop=True)

def adjacent_effect_metric_all(
    y: np.ndarray,
    leaf_ids: np.ndarray,
    g_min: float = 0.30,
    n_min_pair: int = 5,
    pair_weight: str = "hmean",
    robust_q: bool = False,
    score_mode: str = "product",
    # NEW knobs:
    g_min_auto: bool = False,
    delta: Optional[float] = None,
    cv_estimator: str = "std",
    g_floor: float = 0.0,
    g_cap: float = 1.0,
) -> dict:
    stats = _leaf_stats_with_rows(y, leaf_ids, robust_q=robust_q)
    L = len(stats)
    if L <= 1:
        return {"skip": True, "n_leaves": L, "pairs_eval": 0, "pairs_sig": 0,
                "avg_effect": 0.0, "frac_sig": 0.0, "score": float("nan")}

    sum_g_sig, W, Ws = 0.0, 0.0, 0.0
    pairs_eval, pairs_sig = 0, 0

    for i in range(L - 1):
        rows_i = stats.loc[i, "rows"]; rows_j = stats.loc[i+1, "rows"]
        n_i, n_j = len(rows_i), len(rows_j)
        if min(n_i, n_j) < n_min_pair:
            continue

        yi = y[list(rows_i)]; yj = y[list(rows_j)]
        g = abs(hedges_g(yi, yj))
        if not np.isfinite(g):
            continue

        # --- NEW: adaptive threshold based on δ and pooled CV ---
        cv_i = _cv_value(yi, estimator=cv_estimator)
        cv_j = _cv_value(yj, estimator=cv_estimator)
        cv_p = _pooled_cv(cv_i, cv_j)
        g_thr = _pair_g_threshold(delta=delta, cv_pooled=cv_p,
                                  g_min_fixed=g_min, use_auto=g_min_auto,
                                  g_floor=g_floor, g_cap=g_cap)
        # ---------------------------------------------------------

        w = _pair_weight(n_i, n_j, mode=pair_weight)
        W += w; pairs_eval += 1
        if g >= g_thr:
            Ws += w; pairs_sig += 1; sum_g_sig += g * w

    avg_effect = (sum_g_sig / Ws) if Ws > 0 else 0.0
    frac_sig   = (Ws / W) if W > 0 else 0.0
    if score_mode == "avg_effect":
        score = avg_effect
    elif score_mode == "frac_sig":
        score = frac_sig
    else:
        score = avg_effect * frac_sig

    return {"skip": False, "n_leaves": L, "pairs_eval": pairs_eval, "pairs_sig": pairs_sig,
            "avg_effect": float(avg_effect), "frac_sig": float(frac_sig), "score": float(score)}

# ----------------------- CCP alpha grid + CV selection (with repeats & MAE) -----------------------

def cart_alpha_grid(X: np.ndarray, y: np.ndarray,
                    min_leaf: int, max_depth: Optional[int],
                    criterion: str, random_state: int = 42) -> np.ndarray:
    try:
        base = DecisionTreeRegressor(
            criterion=criterion, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state
        )
        base.fit(X, y)
    except ValueError:
        base = DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state
        )
        base.fit(X, y)
    path = base.cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)
    if alphas.size == 0:
        return np.array([0.0], dtype=float)
    return alphas

def inner_gate_alpha(
    X_tr: np.ndarray, y_tr: np.ndarray,
    max_depth: Optional[int], min_leaf: int, criterion: str,
    # separation metric selector and kwargs
    sep_kind: str,
    sep_kwargs: dict,
    gate_seed: int, val_frac: float = 0.2,
    tol_mae: float = 0.0, tol_sep: float = 0.0
) -> Tuple[float, pd.DataFrame]:
    """
    Split-acceptance gate on an inner 80/20 split of the *training* fold.
    Traverse α from simple→complex (large→small); accept a step only if
    MAE_val does not increase (<= tol_mae) AND separation does not decrease (>= -tol_sep).
    Return accepted α (most complex allowed) and audit DataFrame.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=val_frac, random_state=gate_seed, shuffle=True
    )

    try:
        base = DecisionTreeRegressor(
            criterion=criterion, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=gate_seed
        )
        base.fit(X_fit, y_fit)
    except ValueError:
        base = DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=gate_seed
        )
    path = base.cost_complexity_pruning_path(X_fit, y_fit)
    alphas = np.unique(path.ccp_alphas)
    alphas = np.sort(alphas)[::-1]  # simple→complex

    audit = []
    prev_mae = float("inf")
    prev_sep = -float("inf")
    accepted_alpha = alphas[0]

    def compute_sep(tree, Xv, yv) -> float:
        leaves_v = tree.apply(Xv)
        if sep_kind == "stat_g_all":
            m = adjacent_effect_metric_all(y=yv, leaf_ids=leaves_v, **sep_kwargs)
            return float(m["score"])
        else:
            # legacy pass_wfrac path
            m = topfrac_pass_metric(y=yv, leaf_ids=leaves_v, **sep_kwargs)
            return float(m["pass_wfrac"])

    for a in alphas:
        try:
            tree = DecisionTreeRegressor(
                criterion=criterion, min_samples_leaf=min_leaf,
                max_depth=max_depth, random_state=gate_seed, ccp_alpha=float(a)
            )
            tree.fit(X_fit, y_fit)
        except ValueError:
            tree = DecisionTreeRegressor(
                criterion="squared_error", min_samples_leaf=min_leaf,
                max_depth=max_depth, random_state=gate_seed, ccp_alpha=float(a)
            )
            tree.fit(X_fit, y_fit)

        pred = tree.predict(X_val)
        mae = float(np.mean(np.abs(pred - y_val)))
        sep = compute_sep(tree, X_val, y_val)

        accept = (mae <= prev_mae + tol_mae) and (sep >= prev_sep - tol_sep)
        audit.append({"alpha": float(a), "mae_val": mae, "sep_val": sep, "accept": bool(accept)})

        if accept:
            accepted_alpha = float(a)
            prev_mae = mae
            prev_sep = sep
        else:
            break

    return accepted_alpha, pd.DataFrame(audit)

def cv_select_alpha_cart_repeated(
    df: pd.DataFrame,
    store_cols: List[str],
    # separation metric config
    separation_metric: str,
    top_frac: float,
    max_topN: int,
    y_sep: float,
    g_min: float,
    n_min_pair: int,
    stat_score: str,
    pair_weight: str,
    robust_q: bool,
    # >>> NEW: adaptive g controls (threaded) <<<
    g_min_auto: bool = False,
    delta: Optional[float] = None,
    cv_estimator: str = "std",
    g_floor: float = 0.0,
    g_cap: float = 1.0,
    # CV controls
    kfolds: int = 5,
    repeats: int = 5,
    min_leaf: int = 5,
    max_depth: Optional[int] = None,
    criterion: str = "absolute_error",
    use_overlap_guard: bool = True,  # only used by legacy pass_wfrac
    # gating
    gate_splits: bool = False,
    gate_val_frac: float = 0.2,
    gate_mae_tol: float = 0.0,
    gate_sep_tol: float = 0.0,
    seed: int = 42,
    # selection blend
    select_mode: str = "constrained",
    mae_budget: float = 0.02,
    mae_weight: float = 0.30,
) -> Tuple[float, pd.DataFrame]:
    """
    Repeated-K-fold alpha selection using a separation metric (stat_g_all or pass_wfrac)
    and outer-fold MAE. Aggregation across repeats by medians.
    """
    num_cols = ["nodes"] if "nodes" in df.columns else []
    X, _, _ = build_cart_design(df, store_cols, num_cols)
    y = df["total"].to_numpy(dtype=float)

    alphas = cart_alpha_grid(X, y, min_leaf, max_depth, criterion, random_state=seed)
    if alphas.size == 0:
        alphas = np.array([0.0], dtype=float)

    # metric kwargs per kind
    if separation_metric == "stat_g_all":
        sep_kwargs = dict(
            g_min=g_min,
            n_min_pair=n_min_pair,
            pair_weight=pair_weight,
            robust_q=robust_q,
            score_mode=stat_score,
            # >>> NEW: pass adaptive-g knobs down <<<
            g_min_auto=g_min_auto,
            delta=delta,
            cv_estimator=cv_estimator,
            g_floor=g_floor,
            g_cap=g_cap,
        )
    else:
        sep_kwargs = dict(
            top_frac=top_frac,
            max_topN=max_topN,
            y_sep=y_sep,
            use_overlap_guard=use_overlap_guard,
            robust_q=robust_q,
            pair_weight=pair_weight,
        )

    rows = []
    for r in range(repeats):
        rep_seed = seed + 137 * r
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=rep_seed)
        gated_alpha_per_fold: Dict[int, float] = {}

        for ai, a in enumerate(alphas):
            leaves_sum = 0.0
            pairs_sum  = 0
            skipped_folds = 0
            sep_vals = []
            mae_num = 0.0; mae_den = 0

            for fidx, (tr, va) in enumerate(kf.split(X)):
                Xt, Xv = X[tr], X[va]
                yt, yv = y[tr], y[va]

                if gate_splits and fidx not in gated_alpha_per_fold:
                    a_gate, _ = inner_gate_alpha(
                        Xt, yt, max_depth=max_depth, min_leaf=min_leaf, criterion=criterion,
                        sep_kind=separation_metric, sep_kwargs=sep_kwargs,
                        gate_seed=rep_seed + fidx,
                        val_frac=gate_val_frac, tol_mae=gate_mae_tol, tol_sep=gate_sep_tol
                    )
                    gated_alpha_per_fold[fidx] = float(a_gate)
                a_eff = float(max(a, gated_alpha_per_fold.get(fidx, a)))

                try:
                    tree = DecisionTreeRegressor(
                        criterion=criterion, min_samples_leaf=min_leaf,
                        max_depth=max_depth, random_state=rep_seed + fidx, ccp_alpha=a_eff
                    ); tree.fit(Xt, yt)
                except ValueError:
                    tree = DecisionTreeRegressor(
                        criterion="squared_error", min_samples_leaf=min_leaf,
                        max_depth=max_depth, random_state=rep_seed + fidx, ccp_alpha=a_eff
                    ); tree.fit(Xt, yt)

                leaves_v = tree.apply(Xv)
                # separation metric on OUTER validation
                if separation_metric == "stat_g_all":
                    m = adjacent_effect_metric_all(y=yv, leaf_ids=leaves_v, **sep_kwargs)
                    sep_val = float(m["score"])
                    leaves_here = int(m["n_leaves"])
                    pairs_here  = int(m["pairs_eval"])
                    skipped = bool(m["skip"])
                else:
                    m = topfrac_pass_metric(y=yv, leaf_ids=leaves_v, **sep_kwargs)
                    sep_val = float(m["pass_wfrac"])
                    leaves_here = int(m["n_leaves"])
                    pairs_here  = int(m["pairs"])
                    skipped = bool(m["skip"])

                # MAE
                pred = tree.predict(Xv)
                err = np.abs(pred - yv)
                mae_fold = float(np.mean(err))
                mae_num += float(err.sum()); mae_den += int(len(yv))

                if skipped:
                    skipped_folds += 1
                else:
                    leaves_sum += leaves_here
                    pairs_sum  += pairs_here
                    sep_vals.append(sep_val)

                rows.append({
                    "repeat": r, "alpha_idx": ai, "ccp_alpha": float(a),
                    "fold": fidx, "n_val": int(len(yv)),
                    "n_leaves_val": int(leaves_here),
                    "pairs_val": int(pairs_here),
                    "sep_val": float(sep_val),
                    "skipped": bool(skipped),
                    "a_eff": float(a_eff),
                    "mae_val": float(mae_fold),
                })

            cv_sep = float(np.nanmedian(sep_vals)) if len(sep_vals) else float("nan")
            cv_mae = float(mae_num / mae_den) if mae_den > 0 else float("nan")

            rows.append({
                "repeat": r, "alpha_idx": ai, "ccp_alpha": float(a),
                "fold": "CV-agg", "n_val": int(len(y)),
                "n_leaves_val": float(leaves_sum / max(1, (kfolds - skipped_folds))) if (kfolds - skipped_folds) > 0 else float("nan"),
                "pairs_val": int(pairs_sum),
                "sep_val": float(cv_sep),
                "skipped": (skipped_folds > 0),
                "a_eff": np.nan,
                "mae_val": float(cv_mae),
            })

    df_diag = pd.DataFrame(rows)

    # ---- Final α across repeats: use MEDIANS across repeats per α ----
    by_alpha = (
        df_diag[df_diag["fold"].eq("CV-agg")]
        .groupby("ccp_alpha", as_index=False)
        .agg(
            sep_med=("sep_val", "median"),
            mae_med=("mae_val", "median"),
            n_leaves_mean=("n_leaves_val", "mean"),
        )
        .sort_values("ccp_alpha")
    )

    if select_mode == "pass_only":
        by_alpha = by_alpha.sort_values(["sep_med", "n_leaves_mean", "ccp_alpha"],
                                        ascending=[False, False, True]).reset_index(drop=True)
        chosen_alpha = float(by_alpha.iloc[0]["ccp_alpha"])

    elif select_mode == "constrained":
        best_mae_med = by_alpha["mae_med"].min()
        eligible = by_alpha[by_alpha["mae_med"] <= (1.0 + mae_budget) * best_mae_med].copy()
        if eligible.empty:
            eligible = by_alpha.nsmallest(1, "mae_med").copy()
        # TIP: if you want to prefer simpler models on ties, swap sort orders:
        # ascending=[False, True, False]
        eligible = eligible.sort_values(["sep_med", "n_leaves_mean", "ccp_alpha"],
                                        ascending=[False, False, True]).reset_index(drop=True)
        chosen_alpha = float(eligible.iloc[0]["ccp_alpha"])

    else:  # 'weighted'
        a = by_alpha["sep_med"].to_numpy()
        b = by_alpha["mae_med"].to_numpy()
        a_min, a_max = np.nanmin(a), np.nanmax(a)
        b_min, b_max = np.nanmin(b), np.nanmax(b)
        a_norm = np.zeros_like(a) if not np.isfinite(a_min + a_max) or (a_max - a_min) == 0 else (a - a_min) / (a_max - a_min)
        b_norm = np.zeros_like(b) if not np.isfinite(b_min + b_max) or (b_max - b_min) == 0 else (b - b_min) / (b_max - b_min)
        J = (1.0 - mae_weight) * a_norm + mae_weight * (1.0 - b_norm)
        by_alpha = by_alpha.assign(J=J)
        by_alpha = by_alpha.sort_values(["J", "n_leaves_mean", "ccp_alpha"],
                                        ascending=[False, False, True]).reset_index(drop=True)
        chosen_alpha = float(by_alpha.iloc[0]["ccp_alpha"])

    return chosen_alpha, df_diag

# ---------------------------- Final fit + labeling ----------------------------

def _leaf_stats_by_median_simple(y: np.ndarray, leaf_ids: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"leaf": leaf_ids, "y": y})
    g = df.groupby("leaf")["y"].agg(size="size", mean="mean", median="median").reset_index()
    return g.sort_values("median").reset_index(drop=True)

def fit_cart_and_label(
    df: pd.DataFrame, store_cols: List[str],
    ccp_alpha: float, min_leaf: int, max_depth: Optional[int],
    criterion: str = "absolute_error", random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, int], DecisionTreeRegressor]:
    num_cols = ["nodes"] if "nodes" in df.columns else []
    X, _, _ = build_cart_design(df, store_cols, num_cols)
    y = df["total"].to_numpy(dtype=float)
    try:
        tree = DecisionTreeRegressor(
            criterion=criterion, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state, ccp_alpha=float(ccp_alpha)
        ); tree.fit(X, y)
    except ValueError:
        tree = DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state, ccp_alpha=float(ccp_alpha)
        ); tree.fit(X, y)

    leaves = tree.apply(X)
    stats = _leaf_stats_by_median_simple(y, leaves)  # order by median
    ordered = stats["leaf"].tolist()
    leaf_to_region = {int(ordered[i]): i for i in range(len(ordered))}
    regions = np.array([leaf_to_region[int(l)] for l in leaves], dtype=int)

    out = df.copy()
    out["region_provisional"] = pd.Series(leaves, index=df.index).astype(str)
    out["region"] = regions
    prov_to_final = {str(k): int(v) for k, v in leaf_to_region.items()}
    return out, prov_to_final, tree

# -------------------- Optional post-merge: union of adjacent leaves --------------------

def _leaf_order_stats(y: np.ndarray, leaf_ids: np.ndarray, robust_q: bool) -> pd.DataFrame:
    df = pd.DataFrame({"leaf": leaf_ids, "y": y, "_row": np.arange(len(y))})
    rows = []
    for leaf, grp in df.groupby("leaf"):
        arr = grp["y"].to_numpy(dtype=float)
        rows.append({
            "leaf": int(leaf),
            "size": int(arr.size),
            "median": qhat(arr, 0.5, robust=robust_q),
            "q25": qhat(arr, 0.25, robust=robust_q),
            "q75": qhat(arr, 0.75, robust=robust_q),
            "rows": tuple(grp["_row"].to_numpy())
        })
    g = pd.DataFrame(rows)
    return g.sort_values("median").reset_index(drop=True)

def _pool_stats(y_pool: np.ndarray, robust_q: bool) -> Tuple[float, float, float]:
    return (qhat(y_pool, 0.5, robust=robust_q),
            qhat(y_pool, 0.25, robust=robust_q),
            qhat(y_pool, 0.75, robust=robust_q))

def cart_leaf_postmerge_union(
    y: np.ndarray,
    leaf_ids: np.ndarray,
    y_sep: float = 0.10,
    abs_tau: float = 0.0,
    min_region_size: int = 3,
    use_overlap_guard: bool = True,
    min_guard_n: int = 10,
    robust_q: bool = False,
) -> Tuple[Dict[int,int], pd.DataFrame]:
    stats = _leaf_order_stats(y, leaf_ids, robust_q=robust_q)
    if len(stats) == 0:
        return {}, pd.DataFrame()
    leaf_rows = {int(r.leaf): list(r.rows) for r in stats.itertuples(index=False)}

    blocks = []
    cur_leaves = [int(stats.loc[0, "leaf"])]
    cur_idxs = leaf_rows[cur_leaves[0]].copy()
    cur_n = len(cur_idxs)
    cur_med = float(stats.loc[0, "median"])
    cur_q25 = float(stats.loc[0, "q25"])
    cur_q75 = float(stats.loc[0, "q75"])

    audit_rows = []
    for k in range(1, len(stats)):
        leaf_k = int(stats.loc[k, "leaf"]); rows_k = leaf_rows[leaf_k]
        med_k = float(stats.loc[k, "median"]); q25_k = float(stats.loc[k, "q25"]); q75_k = float(stats.loc[k, "q75"])
        n_k = len(rows_k)

        must_merge = (cur_n < min_region_size)
        rel_ok = (med_k >= (1.0 + y_sep) * cur_med)
        abs_ok = ((med_k - cur_med) >= abs_tau)
        gap_ok = (rel_ok or abs_ok)

        ov_ok = True
        if use_overlap_guard and (cur_n >= min_guard_n) and (n_k >= min_guard_n):
            ov_ok = (cur_q75 * (1.0 + 0.5*y_sep) <= q25_k)

        decision = "start_new" if (not must_merge and gap_ok and ov_ok) else "merge_into_prev"
        audit_rows.append({
            "prev_block_size": cur_n, "prev_block_median": cur_med, "prev_block_q75": cur_q75,
            "leaf_k": leaf_k, "leaf_n": n_k, "leaf_median": med_k, "leaf_q25": q25_k,
            "rel_gap_ok": bool(rel_ok), "abs_gap_ok": bool(abs_ok), "gap_ok": bool(gap_ok),
            "overlap_ok": bool(ov_ok), "must_merge": bool(must_merge), "decision": decision
        })

        if decision == "start_new":
            blocks.append({"leaves": cur_leaves.copy(), "rows": cur_idxs.copy(),
                           "n": cur_n, "median": cur_med, "q25": cur_q25, "q75": cur_q75})
            cur_leaves = [leaf_k]
            cur_idxs = rows_k.copy()
            cur_n = len(cur_idxs)
            cur_med, cur_q25, cur_q75 = med_k, q25_k, q75_k
        else:
            cur_leaves.append(leaf_k)
            cur_idxs.extend(rows_k)
            cur_n = len(cur_idxs)
            med_new, q25_new, q75_new = _pool_stats(np.asarray(y)[cur_idxs], robust_q=robust_q)
            cur_med, cur_q25, cur_q75 = med_new, q25_new, q75_new

    blocks.append({"leaves": cur_leaves, "rows": cur_idxs,
                   "n": cur_n, "median": cur_med, "q25": cur_q25, "q75": cur_q75})

    leaf_to_region = {}
    for rid, b in enumerate(blocks):
        for lf in b["leaves"]:
            leaf_to_region[int(lf)] = rid

    audit = pd.DataFrame(audit_rows)
    return leaf_to_region, audit

# ----------------------- Joint-unique strict wildcard summary -----------------------

def strict_wildcard_summary_joint_unique(
    df_all: pd.DataFrame,
    df_labeled: pd.DataFrame,
    store_cols: List[str],
    stat_cols_map: Dict[str, Tuple[str, str]],
    additional_cols: List[str] | None = None,
    include_nodes: bool = True,
) -> pd.DataFrame:
    additional_cols = additional_cols or []
    wc_cols = list(store_cols)
    if include_nodes and ("nodes" in df_all.columns):
        wc_cols.append("nodes")

    def _vals_set(series_like) -> tuple:
        vals = pd.Series(series_like).dropna().astype(str).unique().tolist()
        return tuple(sorted(vals))

    reg_order = (
        df_labeled.groupby("region")["total"]
        .agg(median_total="median").reset_index()
        .sort_values("median_total", ascending=True)
    )
    order = reg_order["region"].tolist()
    pos = {r: i for i, r in enumerate(order)}

    reg_present: Dict[int, Dict[str, tuple]] = {}
    for reg, grp in df_labeled.groupby("region"):
        reg_present[reg] = {c: _vals_set(grp[c]) for c in wc_cols}

    rows = []
    for reg in order:
        grp = df_labeled[df_labeled["region"] == reg]

        fixed_cols = [c for c in wc_cols if grp[c].nunique(dropna=True) == 1]
        cond = pd.Series(True, index=df_all.index)
        for c in fixed_cols:
            cond &= (df_all[c] == grp[c].iloc[0])
        df_cond = df_all[cond]

        present_sets = {c: set(map(str, grp[c].dropna().unique())) for c in wc_cols}
        domain_sets  = {c: set(map(str, df_cond[c].dropna().unique())) for c in wc_cols}

        eligible = {c: (len(present_sets[c]) > 1 and len(domain_sets[c]) > 1 and present_sets[c] == domain_sets[c]) for c in wc_cols}
        S = {c for c, ok in eligible.items() if ok}

        later = order[pos[reg] + 1:]

        def sig_exc(region_id: int, exclude: set[str]) -> tuple:
            parts = []
            for k in wc_cols:
                if k in exclude:
                    continue
                parts.append((k, reg_present[region_id][k]))
            return tuple(sorted(parts, key=lambda x: x[0]))

        def clashes(Sset: set[str]) -> int:
            my = sig_exc(reg, Sset)
            return sum(sig_exc(r2, Sset) == my for r2 in later)

        ccount = clashes(S)
        while ccount > 0 and len(S) > 0:
            drop = None; best = ccount
            for c in list(S):
                S_try = S - {c}
                cc = clashes(S_try)
                if cc < best or (cc == best and len(present_sets[c]) < len(present_sets.get(drop, []))):
                    best = cc; drop = c
            if drop is None:
                break
            S.remove(drop); ccount = best

        wc_out = {}
        for c in wc_cols:
            if len(present_sets[c]) == 1:
                wc_out[c] = next(iter(present_sets[c]))
            elif c in S:
                wc_out[c] = "*"
            else:
                wc_out[c] = ",".join(sorted(present_sets[c]))

        def stat(series: pd.Series, how: str):
            x = series.to_numpy(dtype=float)
            if how == "n": return int(x.size)
            if how == "mean": return float(np.mean(x))
            if how == "std": return float(np.std(x, ddof=1) if x.size > 1 else 0.0)
            if how == "cv":
                m = float(np.mean(x)); s = float(np.std(x, ddof=1) if x.size > 1 else 0.0)
                return (s / m) if m else np.nan
            if how == "q25": return float(np.quantile(x, 0.25))
            if how == "median": return float(np.quantile(x, 0.5))
            if how == "q75": return float(np.quantile(x, 0.75))
            if how == "min": return float(np.min(x)) if x.size else np.nan
            if how == "max": return float(np.max(x)) if x.size else np.nan
            return np.nan

        stats_out = {name: stat(grp[src], how) for name, (src, how) in stat_cols_map.items()}
        for col in additional_cols:
            stats_out[f"mean_{col}"] = stat(grp[col], "mean")
            stats_out[f"cv_{col}"]   = stat(grp[col], "cv")

        rows.append({"region": int(reg), "size": int(len(grp)), **stats_out, **wc_out})

    out = pd.DataFrame(rows).sort_values("median_total", ascending=True).reset_index(drop=True)
    return out

# -------------------------------- Regret & Prevalence --------------------------------

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
        counts = grp["critical_path"].value_counts(dropna=False)
        prevalent = counts.index[0] if not counts.empty else ""
        for path, cnt in counts.items():
            rep = grp[grp["critical_path"] == path].iloc[0] if path in grp["critical_path"].values else grp.iloc[0]
            base = {
                "region": reg,
                "region_size": size,
                "prevalent_critical_path": prevalent,
                "critical_path": path if pd.notna(path) else "",
                "count": int(cnt),
                "fraction": float(cnt / size) if size else 0.0,
                "config_nodes": int(rep["nodes"]) if "nodes" in rep else None,
            }
            for c in stores:
                base[c] = rep[c]
            rows.append(base)
    out = pd.DataFrame(rows).sort_values(["region", "count"], ascending=[True, False])
    return out

# ----------------------------------------- MAIN -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CART regions with repeated-CV statistical separation over all leaves and MAE-aware alpha selection.")
    ap.add_argument("--input", required=True, help="Input CSV (e.g., workflow_makespan_stageorder.csv)")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs")

    # Separation metric controls
    ap.add_argument("--separation-metric",
                    choices=["stat_g_all", "pass_wfrac"], default="stat_g_all",
                    help="stat_g_all: Hedges' g on ALL adjacent leaves (default). pass_wfrac: legacy top-fraction pass metric.")
    ap.add_argument("--delta", type=float, default=0.10,
                help="Target relative gap δ (e.g., 0.10 for 10%). Used when --g-min-auto is on.")
    ap.add_argument("--g-min", type=float, default=0.30, help="Min |Hedges' g| to count a pair as significantly different (effect size).")
    ap.add_argument("--g-min-auto", action="store_true",
                    help="Use per-pair adaptive threshold g_thresh = max(g_floor, min(g_cap, δ / CV_pooled)).")
    ap.add_argument("--cv-estimator", choices=["std","mad"], default="mad",
                    help="How to estimate per-leaf CV for g_min auto: 'std' (sd/mean) or 'mad' (1.4826*MAD/median).")
    ap.add_argument("--g-floor", type=float, default=0.0,
                    help="Lower bound on per-pair g threshold when using --g-min-auto.")
    ap.add_argument("--g-cap", type=float, default=1.0,
                    help="Upper cap on per-pair g threshold when using --g-min-auto.")
    ap.add_argument("--n-min-pair", type=int, default=5, help="Minimum rows required in each leaf to evaluate a pair.")
    ap.add_argument("--stat-score", choices=["avg_effect", "frac_sig", "product"], default="product",
                    help="Fold-level score using effect-size metric: avg_effect, frac_sig, or product = avg_effect*frac_sig.")
    ap.add_argument("--pair-weight", choices=["min","hmean"], default="hmean", help="Pair weighting scheme.")
    ap.add_argument("--robust-quantiles", action="store_true", help="Use Hazen-style small-n smoothing for quantiles (medians/qtiles).")

    # Legacy pass_wfrac knobs (ignored for stat_g_all)
    ap.add_argument("--top-frac", type=float, default=0.25, help="(pass_wfrac only) top fraction of leaves to evaluate.")
    ap.add_argument("--max-topN", type=int, default=12, help="(pass_wfrac only) ceiling for evaluated leaves; floor=3.")
    ap.add_argument("--y-sep", type=float, default=0.10, help="(pass_wfrac only) required adjacent median gap (relative).")
    ap.add_argument("--no-overlap-guard", action="store_true", help="(pass_wfrac only) disable Q75/Q25 overlap guard.")

    # CV controls
    ap.add_argument("--kfolds", type=int, default=5, help="Number of CV folds for alpha selection.")
    ap.add_argument("--repeats", type=int, default=5, help="Repeated K-fold CV for α stability.")

    # Tree controls
    ap.add_argument("--min-leaf", type=int, default=8, help="Minimum samples per leaf.")
    ap.add_argument("--max-depth", type=int, default=12, help="Maximum depth of the tree.")
    ap.add_argument("--criterion", choices=["squared_error", "absolute_error"], default="absolute_error",
                    help="Tree impurity. 'absolute_error' (MAE) is robust and median-friendly.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Inner split-acceptance gate
    ap.add_argument("--gate-splits", action="store_true", help="Enable split acceptance gate on inner 80/20 of training folds.")
    ap.add_argument("--gate-val-frac", type=float, default=0.2, help="Validation fraction for split gating.")
    ap.add_argument("--gate-mae-tol", type=float, default=0.0, help="Non-increase tolerance for MAE in gating.")
    ap.add_argument("--gate-sep-tol", type=float, default=0.0, help="Non-decrease tolerance for separation score in gating.")

    # Alpha selection mode blending separation and MAE
    ap.add_argument("--select-mode", choices=["pass_only","constrained","weighted"], default="constrained",
                    help="How to combine separation and MAE when choosing alpha.")
    ap.add_argument("--mae-budget", type=float, default=0.02, help="Relative MAE slack for constrained mode (e.g., 0.02 = within 2% of best).")
    ap.add_argument("--mae-weight", type=float, default=0.30, help="Weight on MAE in weighted mode.")

    # Post-merge options
    ap.add_argument("--postmerge", action="store_true", help="Enable post-hoc union-of-leaves merging.")
    ap.add_argument("--min-region-size", type=int, default=3, help="Minimum rows per final region.")
    ap.add_argument("--abs-gap-tau", type=float, default=0.0, help="Absolute gap τ for near-zero protection in post-merge.")
    ap.add_argument("--min-guard-n", type=int, default=10, help="Min n on both sides to apply overlap guard in post-merge.")

    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load & enrich
    df0 = pd.read_csv(args.input)
    df, store_cols = compute_rowwise_sensitivities(df0.copy())

    # Select α via Repeated K-fold CV with chosen separation metric + MAE
    chosen_alpha, cv_diag = cv_select_alpha_cart_repeated(
        df=df, store_cols=store_cols,
        separation_metric=args.separation_metric,
        top_frac=args.top_frac, max_topN=args.max_topN, y_sep=args.y_sep,
        g_min=args.g_min, n_min_pair=args.n_min_pair, stat_score=args.stat_score,
        pair_weight=args.pair_weight, robust_q=args.robust_quantiles,
        # >>> NEW: adaptive g flags <<<
        g_min_auto=args.g_min_auto,
        delta=args.delta,
        cv_estimator=args.cv_estimator,
        g_floor=args.g_floor,
        g_cap=args.g_cap,
        # CV & tree controls
        kfolds=args.kfolds, repeats=args.repeats, min_leaf=args.min_leaf,
        max_depth=args.max_depth, criterion=args.criterion,
        use_overlap_guard=not args.no_overlap_guard,  # legacy path only
        gate_splits=args.gate_splits, gate_val_frac=args.gate_val_frac,
        gate_mae_tol=args.gate_mae_tol, gate_sep_tol=args.gate_sep_tol,
        seed=args.seed,
        # selection blend
        select_mode=args.select_mode, mae_budget=args.mae_budget, mae_weight=args.mae_weight
    )
    cv_diag.to_csv(outdir / "cart_cv_candidates.csv", index=False)

    # Final fit & labeling at chosen α
    df_cf, prov_to_final, tree = fit_cart_and_label(
        df=df, store_cols=store_cols, ccp_alpha=chosen_alpha,
        min_leaf=args.min_leaf, max_depth=args.max_depth,
        criterion=args.criterion, random_state=args.seed
    )

    # Optional post-merge (union of leaves only; respects CART boundaries)
    if args.postmerge:
        leaf_key = df_cf["region"].to_numpy(dtype=int)
        y = df_cf["total"].to_numpy(dtype=float)
        leaf_to_final, pm_audit = cart_leaf_postmerge_union(
            y=y, leaf_ids=leaf_key, y_sep=args.y_sep, abs_tau=args.abs_gap_tau,
            min_region_size=args.min_region_size, use_overlap_guard=not args.no_overlap_guard,
            min_guard_n=args.min_guard_n, robust_q=args.robust_quantiles
        )
        df_cf["leaf_region"] = df_cf["region"].astype(int)
        df_cf["region"] = df_cf["leaf_region"].map(leaf_to_final).astype(int)
        pm_audit.to_csv(outdir / "cart_postmerge_adjacent_audit.csv", index=False)
        pd.DataFrame([{"leaf_region": k, "merged_region": v} for k, v in leaf_to_final.items()]) \
            .to_csv(outdir / "cart_postmerge_map.csv", index=False)

    # Save rowwise + mapping
    df_cf.to_csv(outdir / "workflow_rowwise_sensitivities.csv", index=False)
    pd.DataFrame([{"region_provisional": k, "region": v} for k, v in prov_to_final.items()]) \
        .to_csv(outdir / "region_merge_map.csv", index=False)

    # Region summary with joint-unique wildcard
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
        df_all=df, df_labeled=df_cf, store_cols=store_cols,
        stat_cols_map=stat_cols_map,
        additional_cols=["exec_beegfs_share", "exec_local_share", "movement_share"],
        include_nodes=True
    )
    regions_df.to_csv(outdir / "regions_by_total.csv", index=False)

    # Regret metrics
    regret_region, regret_config = compute_regret(df_cf)
    regret_region.to_csv(outdir / "regret_per_region.csv", index=False)
    regret_config.to_csv(outdir / "regret_per_config.csv", index=False)

    # Critical path prevalence
    cp_df = critical_path_prevalence(df_cf)
    cp_df.to_csv(outdir / "critical_path_prevalence_by_region.csv", index=False)

    print("Chosen α (ccp_alpha):", chosen_alpha)
    print("Wrote:")
    for p in [
        outdir / "workflow_rowwise_sensitivities.csv",
        outdir / "regions_by_total.csv",
        outdir / "regret_per_region.csv",
        outdir / "regret_per_config.csv",
        outdir / "critical_path_prevalence_by_region.csv",
        outdir / "region_merge_map.csv",
        outdir / "cart_cv_candidates.csv",
    ]:
        print(" -", p)
    if args.postmerge:
        print(" -", outdir / "cart_postmerge_adjacent_audit.csv")
        print(" -", outdir / "cart_postmerge_map.csv")

if __name__ == "__main__":
    main()