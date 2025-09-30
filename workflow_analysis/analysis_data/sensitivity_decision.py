#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity & Region Analysis for Workflow Makespans
CART with Repeated-CV Top-Performer Separation, Split-Gating, and Optional Post-Merge
-------------------------------------------------------------------------------------

Motivation
----------
We want interpretable regions (CART leaves) that are as granular as possible *and*
statistically meaningful among the best performers. Instead of tuning tree size by
generic error alone, we select the pruning level (ccp_alpha) out-of-sample based on
a decision-aligned metric and stabilize it with repeats. We also keep CART as the
region engine while allowing an optional post-merge step that *only merges* adjacent
leaves when the separation rule isn’t met—never re-splitting CART.

What this script does
---------------------
1) Builds row-wise sensitivities (λ terms) and execution/movement shares.
2) Fits CART models across a cost-complexity pruning path.
3) Selects α via **repeated K-fold CV** using a **top-fraction median-gap** metric:
   • Order leaves by median(total); consider the **top-π** leaves (adaptive: floor=3,
     ceiling=--max-topN).
   • For each adjacent top pair (i→j), require a **relative gap**: median_j ≥ (1+δ)·median_i,
     with optional **quantile overlap guard** (Q75(i)·(1+δ/2) ≤ Q25(j)).
   • Weight each pair by either **min(n_i,n_j)** or **harmonic mean** (2/(1/n_i+1/n_j)).
   • Aggregate the **size-weighted pass fraction** out-of-sample; repeat CV with
     different seeds and choose the α with the **median CV score** (tie-break by
     larger average leaf count).
4) (Optional) **Split acceptance gate** inside each CV training fold:
   grow complexity only if it **does not worsen MAE** and **does not reduce** the
   held-out fold’s top-set pass metric (inner 80/20 train/val split). This caps the
   maximum allowed complexity per fold.
5) Final fit at the chosen α; (Optional) **post-merge** adjacent CART leaves into
   final regions (unions of leaves) to enforce δ-gap / quantile-guard / min-size
   without undermining CART’s discovered boundaries.
6) Summarizes regions with **joint-unique strict wildcard** columns so that ‘*’ is
   used only when the region truly spans the conditional domain **and** no later
   region has the same signature on other columns.
7) Writes standard analytics (regions_by_total, regret, critical path prevalence).

Outputs
-------
 - workflow_rowwise_sensitivities.csv      (row-wise λ-attributions, shares, final regions)
 - regions_by_total.csv                    (joint-unique wildcard summary, stats incl nodes)
 - regret_per_region.csv                   (mean regret vs global best, by region)
 - regret_per_config.csv                   (per-row regret vs global best)
 - critical_path_prevalence_by_region.csv
 - region_merge_map.csv                    (CART provisional leaf → ordered region id)
 - cart_cv_candidates.csv                  (per-α × fold diagnostics + CV aggregates)
 - cart_postmerge_adjacent_audit.csv       (if --postmerge)
 - cart_postmerge_map.csv                  (if --postmerge)

Key knobs
---------
 --repeats N           : Repeated K-fold CV for α stability (default 5).
 --top-frac π          : Fraction of best leaves to evaluate (default 0.25).
 --max-topN            : Ceiling for evaluated leaves (default 12; floor is 3).
 --y-sep δ             : Required relative median gap (default 0.10).
 --pair-weight         : 'min' or 'hmean' for pair weighting.
 --robust-quantiles    : Use small-n-smoother (Hazen) for Q25/Q75/median.
 --gate-splits         : Enable split-acceptance gating with inner 80/20 val.
 --postmerge           : Merge adjacent CART leaves when separation fails; regions
                         become unions of leaves (never re-split CART).

Example
-------
python sensitivity_decision.py \
  --input 1kgenome/workflow_makespan_stageorder.csv \
  --outdir sens_out \
  --kfolds 5 --repeats 5 \
  --top-frac 0.25 --max-topN 12 --y-sep 0.10 \
  --pair-weight hmean --robust-quantiles \
  --min-leaf 3 --max-depth 12 --criterion absolute_error \
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
    # Hazen's method as a small-n smoother (HD-like behavior without SciPy)
    try:
        return float(np.quantile(x, q, method="hazen"))
    except TypeError:
        # Fallback for older NumPy
        return float(np.quantile(x, q, interpolation="linear"))


# --------------------------- top-fraction metric ---------------------------

def _leaf_stats_by_median(y: np.ndarray, leaf_ids: np.ndarray, robust_q: bool) -> pd.DataFrame:
    """Per-leaf stats ordered by median(total) ascending."""
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
        # Harmonic mean; guard tiny edge cases
        if n_i <= 0 or n_j <= 0:
            return 0.0
        return 2.0 / (1.0 / n_i + 1.0 / n_j)
    # default: min
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
    """
    Size-weighted pass fraction among adjacent pairs in the top fraction of leaves (by median).

    Adaptive top set: topN = clamp(ceil(π·L), min=3, max=max_topN). If topN < 2, mark skip.
    Returns dict with keys: skip, n_leaves, topN, pairs, pass_pairs, pass_frac, w_num, w_den, pass_wfrac, min_gap_ratio_minus1
    """
    stats = _leaf_stats_by_median(y, leaf_ids, robust_q=robust_q)
    L = len(stats)
    if L <= 1:
        return dict(skip=True, n_leaves=L, topN=min(L, 1), pairs=0,
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

    pass_pairs = 0
    w_num = 0.0
    w_den = 0.0
    min_gap = np.inf

    for i in range(topN - 1):
        m_i = float(s.loc[i, "median"])
        m_j = float(s.loc[i+1, "median"])
        q75_i = float(s.loc[i, "q75"])
        q25_j = float(s.loc[i+1, "q25"])
        n_i   = int(s.loc[i, "size"])
        n_j   = int(s.loc[i+1, "size"])

        ratio_minus1 = np.inf if m_i <= 0 else (m_j / m_i) - 1.0
        gap_ok = (ratio_minus1 >= y_sep)
        ov_ok  = (q75_i * (1.0 + 0.5 * y_sep) <= q25_j) if use_overlap_guard else True
        ok     = (gap_ok and ov_ok)

        w = _pair_weight(n_i, n_j, mode=pair_weight)
        w_den += w
        if ok:
            pass_pairs += 1
            w_num += w

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


# ----------------------- CCP alpha grid + CV selection (with repeats) -----------------------

def cart_alpha_grid(X: np.ndarray, y: np.ndarray,
                    min_leaf: int, max_depth: Optional[int],
                    criterion: str, random_state: int = 42) -> np.ndarray:
    """Get a stable α grid from a base tree fit on the full data."""
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
    metric_kwargs: dict, gate_seed: int, val_frac: float = 0.2,
    tol_mae: float = 0.0, tol_pass: float = 0.0
) -> Tuple[float, pd.DataFrame]:
    """
    Split-acceptance gate on an inner 80/20 split of the *training* fold.
    Traverse α from simple→complex (large→small); accept a step only if
    MAE_val does not increase (<= tol) AND pass_wfrac does not decrease (>= -tol).
    Return accepted α (most complex allowed) and audit DataFrame.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=val_frac, random_state=gate_seed, shuffle=True
    )

    # Build α path from the fitting subset
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
        base.fit(X_fit, y_fit)
    path = base.cost_complexity_pruning_path(X_fit, y_fit)
    alphas = np.unique(path.ccp_alphas)
    # Walk from large→small α (simple→complex)
    alphas = np.sort(alphas)[::-1]

    audit = []
    # Start with the simplest model (largest α)
    prev_mae = float("inf")
    prev_pass = -float("inf")
    accepted_alpha = alphas[0]

    for a in alphas:
        # Fit at this α on X_fit, evaluate on X_val
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
        leaves_v = tree.apply(X_val)
        m = topfrac_pass_metric(y=y_val, leaf_ids=leaves_v, **metric_kwargs)

        # Some folds may "skip" due to too-few leaves; treat pass as NaN -> neutral
        pass_w = m["pass_wfrac"]
        if np.isnan(pass_w):
            pass_w = prev_pass

        # Accept if mae <= prev_mae + tol and pass_w >= prev_pass - tol
        accept = (mae <= prev_mae + tol_mae) and (pass_w >= prev_pass - tol_pass)
        audit.append({"alpha": float(a), "mae_val": mae, "pass_wfrac_val": pass_w, "accept": bool(accept)})

        if accept:
            accepted_alpha = float(a)
            prev_mae = mae
            prev_pass = pass_w
        else:
            # stop early: further complexity unlikely to be accepted
            break

    return accepted_alpha, pd.DataFrame(audit)


def cv_select_alpha_cart_repeated(
    df: pd.DataFrame,
    store_cols: List[str],
    top_frac: float,
    max_topN: int,
    y_sep: float,
    kfolds: int = 5,
    repeats: int = 5,
    min_leaf: int = 5,
    max_depth: Optional[int] = None,
    criterion: str = "absolute_error",
    use_overlap_guard: bool = True,
    robust_q: bool = False,
    pair_weight: str = "min",
    gate_splits: bool = False,
    gate_val_frac: float = 0.2,
    gate_mae_tol: float = 0.0,
    gate_pass_tol: float = 0.0,
    seed: int = 42
) -> Tuple[float, pd.DataFrame]:
    """
    Repeated-K-fold α selection for CART using the top-fraction size-weighted pass metric.
    For each repeat (seed), run a K-fold CV. If gate_splits=True, compute an inner
    acceptance α per fold to cap complexity. Return (chosen_alpha, diagnostics_df).
    """
    num_cols = ["nodes"] if "nodes" in df.columns else []
    X, _, _ = build_cart_design(df, store_cols, num_cols)
    y = df["total"].to_numpy(dtype=float)

    alphas = cart_alpha_grid(X, y, min_leaf, max_depth, criterion, random_state=seed)
    if alphas.size == 0:
        alphas = np.array([0.0], dtype=float)

    metric_kwargs = dict(
        top_frac=top_frac, max_topN=max_topN, y_sep=y_sep,
        use_overlap_guard=use_overlap_guard, robust_q=robust_q, pair_weight=pair_weight
    )

    rows = []
    repeat_summaries = []

    for r in range(repeats):
        rep_seed = seed + 137 * r
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=rep_seed)

        gated_alpha_per_fold: Dict[int, float] = {}

        for ai, a in enumerate(alphas):
            agg_w_num = 0.0
            agg_w_den = 0.0
            leaves_sum = 0.0
            pairs_sum = 0
            skipped_folds = 0

            for fidx, (tr, va) in enumerate(kf.split(X)):
                Xt, Xv = X[tr], X[va]
                yt, yv = y[tr], y[va]

                if gate_splits and fidx not in gated_alpha_per_fold:
                    a_gate, _audit = inner_gate_alpha(
                        Xt, yt, max_depth=max_depth, min_leaf=min_leaf, criterion=criterion,
                        metric_kwargs=metric_kwargs, gate_seed=rep_seed + fidx,
                        val_frac=gate_val_frac, tol_mae=gate_mae_tol, tol_pass=gate_pass_tol
                    )
                    gated_alpha_per_fold[fidx] = float(a_gate)
                a_eff = float(max(a, gated_alpha_per_fold.get(fidx, a)))  # larger α = simpler tree

                try:
                    tree = DecisionTreeRegressor(
                        criterion=criterion, min_samples_leaf=min_leaf,
                        max_depth=max_depth, random_state=rep_seed + fidx, ccp_alpha=a_eff
                    )
                    tree.fit(Xt, yt)
                except ValueError:
                    tree = DecisionTreeRegressor(
                        criterion="squared_error", min_samples_leaf=min_leaf,
                        max_depth=max_depth, random_state=rep_seed + fidx, ccp_alpha=a_eff
                    )
                    tree.fit(Xt, yt)

                leaves_v = tree.apply(Xv)
                m = topfrac_pass_metric(y=yv, leaf_ids=leaves_v, **metric_kwargs)

                if m["skip"]:
                    skipped_folds += 1
                else:
                    agg_w_num += m["w_num"]
                    agg_w_den += m["w_den"]
                    leaves_sum += m["n_leaves"]
                    pairs_sum  += m["pairs"]

                rows.append({
                    "repeat": r, "alpha_idx": ai, "ccp_alpha": float(a),
                    "fold": fidx, "n_val": int(len(yv)),
                    "n_leaves_val": int(m["n_leaves"]),
                    "topN_val": int(m["topN"]),
                    "pairs_val": int(m["pairs"]),
                    "pass_pairs_val": int(m["pass_pairs"]),
                    "pass_frac_val": float(m["pass_frac"]),
                    "pass_wfrac_val": float(m["pass_wfrac"]),
                    "w_num_val": float(m["w_num"]), "w_den_val": float(m["w_den"]),
                    "min_gap_ratio_minus1_val": float(m["min_gap_ratio_minus1"]),
                    "skipped": bool(m["skip"]),
                    "a_eff": float(a_eff),
                })

            cv_pass_wfrac = (agg_w_num / agg_w_den) if agg_w_den > 0 else float("nan")
            rows.append({
                "repeat": r, "alpha_idx": ai, "ccp_alpha": float(a),
                "fold": "CV-agg", "n_val": int(len(y)),
                "n_leaves_val": float(leaves_sum / max(1, (kfolds - skipped_folds))) if (kfolds - skipped_folds) > 0 else float("nan"),
                "topN_val": np.nan, "pairs_val": int(pairs_sum),
                "pass_pairs_val": np.nan, "pass_frac_val": np.nan,
                "pass_wfrac_val": float(cv_pass_wfrac),
                "w_num_val": float(agg_w_num), "w_den_val": float(agg_w_den),
                "min_gap_ratio_minus1_val": np.nan,
                "skipped": (skipped_folds > 0),
                "a_eff": np.nan,
            })

        diag = pd.DataFrame(rows)
        diag_rep = diag[diag["fold"].eq("CV-agg") & diag["repeat"].eq(r)] \
            .sort_values(["pass_wfrac_val", "n_leaves_val", "ccp_alpha"], ascending=[False, False, True]) \
            .reset_index(drop=True)
        best_row = diag_rep.iloc[0]
        repeat_summaries.append({
            "repeat": r,
            "chosen_alpha": float(best_row["ccp_alpha"]),
            "pass_wfrac_val": float(best_row["pass_wfrac_val"]),
            "n_leaves_val": float(best_row["n_leaves_val"]),
        })

    df_diag = pd.DataFrame(rows)
    df_rep = pd.DataFrame(repeat_summaries)

    # Final α: median CV score per α across repeats; tie-break by larger avg leaves then smaller α
    by_alpha = df_diag[df_diag["fold"].eq("CV-agg")].groupby("ccp_alpha", as_index=False).agg(
        pass_wfrac_med=("pass_wfrac_val", "median"),
        n_leaves_mean=("n_leaves_val", "mean")
    ).sort_values(["pass_wfrac_med", "n_leaves_mean", "ccp_alpha"], ascending=[False, False, True]).reset_index(drop=True)

    chosen_alpha = float(by_alpha.iloc[0]["ccp_alpha"])
    return chosen_alpha, df_diag


# ---------------------------- Final fit + labeling ----------------------------

def _leaf_stats_by_median_simple(y: np.ndarray, leaf_ids: np.ndarray) -> pd.DataFrame:
    """Per-leaf stats ordered by median(total) ascending (simple quantiles)."""
    df = pd.DataFrame({"leaf": leaf_ids, "y": y})
    g = df.groupby("leaf")["y"].agg(
        size="size", mean="mean", median="median"
    ).reset_index()
    return g.sort_values("median").reset_index(drop=True)

def fit_cart_and_label(
    df: pd.DataFrame, store_cols: List[str],
    ccp_alpha: float, min_leaf: int, max_depth: Optional[int],
    criterion: str = "absolute_error", random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, int], DecisionTreeRegressor]:
    """Fit tree on full data at chosen α and return labeled df + provisional→final region map + tree object."""
    num_cols = ["nodes"] if "nodes" in df.columns else []
    X, _, _ = build_cart_design(df, store_cols, num_cols)
    y = df["total"].to_numpy(dtype=float)
    try:
        tree = DecisionTreeRegressor(
            criterion=criterion, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state, ccp_alpha=float(ccp_alpha)
        )
        tree.fit(X, y)
    except ValueError:
        tree = DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state, ccp_alpha=float(ccp_alpha)
        )
        tree.fit(X, y)

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
    """Per-leaf stats ordered by median(y) ascending, with member row indices."""
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
    """
    Merge *adjacent* CART leaves (by median ascending) to produce final regions
    that satisfy: min size, (rel OR abs) gap, and an optional Q75/Q25 guard.

    Returns:
      leaf_to_region: mapping leaf_id -> final_region_id (0..R-1, in median order)
      audit: per-adjacent decision table for the chosen segmentation
    """
    # 1) Per-leaf ordered stats
    stats = _leaf_order_stats(y, leaf_ids, robust_q=robust_q)
    if len(stats) == 0:
        return {}, pd.DataFrame()
    # Convenience accessors
    leaf_rows = {int(r.leaf): list(r.rows) for r in stats.itertuples(index=False)}

    # 2) Greedy single-pass union-of-leaves (never splits CART)
    blocks = []
    # start first block
    cur_leaves = [int(stats.loc[0, "leaf"])]
    cur_idxs = leaf_rows[cur_leaves[0]].copy()
    cur_n = len(cur_idxs)
    cur_med = float(stats.loc[0, "median"])
    cur_q25 = float(stats.loc[0, "q25"])
    cur_q75 = float(stats.loc[0, "q75"])

    audit_rows = []

    for k in range(1, len(stats)):
        leaf_k = int(stats.loc[k, "leaf"])
        rows_k = leaf_rows[leaf_k]
        med_k = float(stats.loc[k, "median"])
        q25_k = float(stats.loc[k, "q25"])
        q75_k = float(stats.loc[k, "q75"])
        n_k = len(rows_k)

        # If current block is undersized, we must merge to satisfy min size
        must_merge = (cur_n < min_region_size)

        # Evaluate separation if we were to start a new region at leaf_k
        rel_ok = (med_k >= (1.0 + y_sep) * cur_med)
        abs_ok = ((med_k - cur_med) >= abs_tau)
        gap_ok = (rel_ok or abs_ok)

        ov_ok = True
        if use_overlap_guard and (cur_n >= min_guard_n) and (n_k >= min_guard_n):
            ov_ok = (cur_q75 * (1.0 + 0.5*y_sep) <= q25_k)

        start_new_ok = (gap_ok and ov_ok)

        decision = "start_new" if (not must_merge and start_new_ok) else "merge_into_prev"
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

    # Build mapping leaf -> final region id in order
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
    """
    Strict wildcard with JOINT uniqueness against succeeding regions.

    A column c in a region may show '*' iff:
      (A) (Conditional-domain) Given the region's fixed columns (single-valued in the region),
          the region spans ALL values of c present in df_all under those fixed values, AND
      (B) (Joint uniqueness) If we star a set S of columns, then for NO later region (by ascending
          median_total) is the tuple of per-column value-sets on all columns except S EQUAL to this
          region's tuple on those columns. If there is a clash, iteratively UNSTAR columns to break ties.

    • 'nodes' is included as a wildcard column when include_nodes=True.
    """
    additional_cols = additional_cols or []

    wc_cols = list(store_cols)
    if include_nodes and ("nodes" in df_all.columns):
        wc_cols.append("nodes")

    def _vals_set(series_like) -> tuple:
        vals = pd.Series(series_like).dropna().astype(str).unique().tolist()
        return tuple(sorted(vals))

    # region order by median_total
    reg_order = (
        df_labeled.groupby("region")["total"]
        .agg(median_total="median").reset_index()
        .sort_values("median_total", ascending=True)
    )
    order = reg_order["region"].tolist()
    pos = {r: i for i, r in enumerate(order)}

    # per-region, per-column value sets in labeled data
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

        eligible = {
            c: (len(present_sets[c]) > 1 and len(domain_sets[c]) > 1 and present_sets[c] == domain_sets[c])
            for c in wc_cols
        }
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

        # Greedy unstar to remove joint clashes
        ccount = clashes(S)
        while ccount > 0 and len(S) > 0:
            drop = None
            best = ccount
            for c in list(S):
                S_try = S - {c}
                cc = clashes(S_try)
                if cc < best or (cc == best and len(present_sets[c]) < len(present_sets.get(drop, []))):
                    best = cc; drop = c
            if drop is None:
                break
            S.remove(drop)
            ccount = best

        # render wildcard cells
        wc_out = {}
        for c in wc_cols:
            if len(present_sets[c]) == 1:
                wc_out[c] = next(iter(present_sets[c]))
            elif c in S:
                wc_out[c] = "*"
            else:
                wc_out[c] = ",".join(sorted(present_sets[c]))

        # stats block
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
    ap = argparse.ArgumentParser(description="CART regions with repeated-CV top-performer separation, split-gating, and optional post-merge.")
    ap.add_argument("--input", required=True, help="Input CSV (e.g., workflow_makespan_stageorder.csv)")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs")

    # CV + metric controls
    ap.add_argument("--kfolds", type=int, default=5, help="Number of CV folds for alpha selection.")
    ap.add_argument("--repeats", type=int, default=5, help="Repeated K-fold CV for α stability.")
    ap.add_argument("--top-frac", type=float, default=0.25, help="Top fraction of leaves to evaluate (0–1).")
    ap.add_argument("--max-topN", type=int, default=12, help="Ceiling for evaluated leaves; floor=3.")
    ap.add_argument("--y-sep", type=float, default=0.10, help="Required adjacent median gap (relative).")
    ap.add_argument("--pair-weight", choices=["min", "hmean"], default="min", help="Pair weighting scheme.")
    ap.add_argument("--robust-quantiles", action="store_true", help="Use Hazen-style small-n smoothing for quantiles.")
    ap.add_argument("--no-overlap-guard", action="store_true", help="Disable Q75/Q25 overlap guard in metric.")

    # Tree controls
    ap.add_argument("--min-leaf", type=int, default=8, help="Minimum samples per leaf (stability).")
    ap.add_argument("--max-depth", type=int, default=12, help="Maximum depth of the tree.")
    ap.add_argument("--criterion", choices=["squared_error", "absolute_error"], default="absolute_error",
                    help="Tree impurity. 'absolute_error' is robust and median-friendly.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Split-acceptance gate (inner validation) controls
    ap.add_argument("--gate-splits", action="store_true", help="Enable split acceptance gate on inner 80/20 of training folds.")
    ap.add_argument("--gate-val-frac", type=float, default=0.2, help="Validation fraction for split gating.")
    ap.add_argument("--gate-mae-tol", type=float, default=0.0, help="Non-increase tolerance for MAE in gating.")
    ap.add_argument("--gate-pass-tol", type=float, default=0.0, help="Non-decrease tolerance for pass_wfrac in gating.")

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

    # Select α via Repeated K-fold CV on the separation metric (no leakage)
    chosen_alpha, cv_diag = cv_select_alpha_cart_repeated(
        df=df, store_cols=store_cols,
        top_frac=args.top_frac, max_topN=args.max_topN, y_sep=args.y_sep,
        kfolds=args.kfolds, repeats=args.repeats, min_leaf=args.min_leaf,
        max_depth=args.max_depth, criterion=args.criterion,
        use_overlap_guard=not args.no_overlap_guard,
        robust_q=args.robust_quantiles, pair_weight=args.pair_weight,
        gate_splits=args.gate_splits, gate_val_frac=args.gate_val_frac,
        gate_mae_tol=args.gate_mae_tol, gate_pass_tol=args.gate_pass_tol,
        seed=args.seed
    )
    cv_diag.to_csv(outdir / "cart_cv_candidates.csv", index=False)

    # Final fit & labeling at the chosen α
    df_cf, prov_to_final, tree = fit_cart_and_label(
        df=df, store_cols=store_cols, ccp_alpha=chosen_alpha,
        min_leaf=args.min_leaf, max_depth=args.max_depth,
        criterion=args.criterion, random_state=args.seed
    )

    # Optional post-merge (unions of leaves only; respects CART boundaries)
    if args.postmerge:
        # Use current ordered leaf-region ids as the leaf key for merging
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

    # Save rowwise output & mapping
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
