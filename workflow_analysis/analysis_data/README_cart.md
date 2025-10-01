# CART Region Formation (Modular)

This suite refactors the original `sensitivity_decision.py` into modular files, while preserving functionality and making the controls safer and less redundant.

## Files

- **`cart_main.py`** – Entry point (CLI → pipeline).
- **`cart_cli.py`** – Argument parsing + **redundancy removal** (ties certain knobs to `--min-leaf` unless explicitly set).
- **`cart_pipeline.py`** – Orchestration: load data → CV alpha selection → final fit → (optional) post-merge → summaries & outputs.
- **`cart_features.py`** – Row-wise sensitivity attribution (`lam_*`) + shares; store column detection.
- **`cart_tree.py`** – CART design matrix, CCP alpha grid, split-gate, final fit & label, post-merge (union-of-leaves).
- **`cart_metrics.py`** – Quantiles, Hedges’ g, **all-leaves** separation metric with fixed/adaptive `g_min`, CV helpers.
- **`cart_summaries.py`** – Joint-unique strict wildcard summary (includes `nodes`), regret metrics, critical-path prevalence.

## What it does (high-level)

1. **Row-wise sensitivities** (`lam_*`) from `critical_path` + share features.
2. **CART** (OneHot + nodes) with **cost-complexity pruning**; build an α grid.
3. **Repeated K-fold CV**:
   - For each α: fit on train folds (optional **split gate** on inner 80/20), evaluate on the **outer** fold:
     - **Separation** = adjacent-pair **Hedges’ g** score across **all leaves** (ordered by median total).
       - Pair is “different” if \(|g| ≥ g_{\text{thr}}\).
       - \(g_{\text{thr}}\) can be **fixed** (`--g-min`) **or adaptive** (`--g-min-auto`: \(g_{\text{thr}}=\max(g_{\text{floor}}, \min(g_{\text{cap}}, \delta / \text{CV}_{\text{pooled}}))\)).
     - **MAE** of totals.
   - Aggregate across folds to a CV-agg row per α; aggregate across repeats by **medians**.
4. **Select α**:
   - `constrained` (default): maximize separation among α whose MAE ≤ (1+`--mae-budget`)*best. **Tie-break** prefers simpler models (fewer leaves, larger α).
   - `weighted`: blended objective (normalized separation and MAE).
   - `pass_only`: separation only.
5. **Final fit** at chosen α → **ordered leaves → regions**.
6. **Optional post-merge** (union of adjacent leaves only) enforcing δ / τ gap and optional overlap guard.
7. **Summaries & output**:
   - `workflow_rowwise_sensitivities.csv`
   - `regions_by_total.csv` (joint-unique wildcard; includes `nodes`)
   - `regret_per_region.csv`, `regret_per_config.csv`
   - `critical_path_prevalence_by_region.csv`
   - `region_merge_map.csv`
   - `cart_cv_candidates.csv`
   - (if `--postmerge`) `cart_postmerge_adjacent_audit.csv`, `cart_postmerge_map.csv`

## Redundancy removed

To reduce confusing knobs:
- If **not provided**, these inherit `--min-leaf`:
  - `--n-min-pair` (min rows per side for pair evaluation)
  - `--min-region-size` (min rows per final region in post-merge)
  - `--min-guard-n` (min n on both sides to apply quantile overlap guard)

If you **do** pass them, your values are respected.

## Key controls

- **Accuracy vs. Separation**
  - `--select-mode constrained --mae-budget 0.02` → pick best separation within +2% MAE of the best.
  - `--stat-score product` (default) → balances “how many pairs are significant” and “how big they are”.
- **Stability**
  - `--repeats 5` smooths α selection.
  - `--min-leaf 8–10` yields more reliable leaf medians/effect sizes.
- **Adaptive g**
  - `--g-min-auto --delta 0.10 --cv-estimator mad --g-floor 0.20 --g-cap 1.00`
  - Interprets your “10% matters” rule under the current noise level (via pooled CV).

## Example

```bash
python cart_main.py \
  --input 1kgenome/workflow_makespan_stageorder.csv \
  --outdir sens_out \
  --kfolds 5 --repeats 5 \
  --separation-metric stat_g_all --g-min-auto --stat-score avg_effect \
  --pair-weight hmean --robust-quantiles \
  --min-leaf 3 --max-depth 12 --criterion absolute_error \
  --select-mode weighted --mae-weight 0.5 --debug 
