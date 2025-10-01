#!/usr/bin/env python3
"""
Scatter subplots by node scale using workflow_rowwise_sensitivities.csv

- Uses sensitivities file (with explicit per-row `region`, `nodes`, and *_stor).
- Joins to workflow_makespan_stageorder.csv via the intersection of *_stor columns
  to get the true `total` per configuration.
- For each node scale (2, 5, 10 if present):
    * Sort configurations by ascending total.
    * X-axis: configuration index (1..N), show every 10th tick (minor ticks at 1).
    * Y-axis: total (shared across all subplots for easy comparison).
    * Color points by **region** with a color order that follows the **numeric
      order of region IDs** (0,1,2,...,10,11,...) — not lexicographic.

Outputs
-------
- Figure: ./sens_out/regions_scatter_true_labels.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from typing import List, Any

# --------- Paths ---------
SENS_CSV    = "./sens_out/workflow_rowwise_sensitivities.csv"
CONFIGS_CSV = "./pyflxtrkr/pyflex_s9_48f/workflow_makespan_stageorder.csv"
OUT_PNG     = "./sens_out/regions_scatter_true_labels.png"

REGION_COL = "region"
NODES_COL  = "nodes"
TOTAL_COL  = "total"

# --------- Helpers ---------
def _norm_str(s: pd.Series) -> pd.Series:
    """Normalize string-ish columns for reliable joining."""
    return s.astype(str).str.strip().str.lower()

def build_config_label(df: pd.DataFrame, stor_cols: List[str]) -> pd.Series:
    """Create a stable configuration label from *_stor columns (normalized)."""
    parts = []
    for c in sorted(stor_cols):
        base = c[:-5]  # drop '_stor'
        vals = _norm_str(df[c])
        parts.append(vals.map(lambda v: f"{base}:{v}"))
    return pd.Series([" | ".join(vals) for vals in zip(*parts)], index=df.index)

def series_mode(s: pd.Series):
    """Return the first mode if multiple."""
    m = s.mode(dropna=False)
    return m.iloc[0] if not m.empty else np.nan

def _region_sort_key(x: Any):
    """Sort key that prefers numeric ordering when possible."""
    try:
        return (0, float(x))  # numeric first, ascending
    except Exception:
        return (1, str(x))    # then non-numeric, lexicographic

# --------- Load ---------
sdf = pd.read_csv(SENS_CSV)     # must contain region, nodes, *_stor columns
cdf = pd.read_csv(CONFIGS_CSV)  # must contain nodes, *_stor columns, total

# Identify overlapping *_stor columns to build a common configuration key
stor_cols_sens = sorted([c for c in sdf.columns if c.endswith("_stor")])
stor_cols_cfg  = sorted([c for c in cdf.columns if c.endswith("_stor")])
stor_cols_join = [c for c in stor_cols_sens if c in stor_cols_cfg]
if not stor_cols_join:
    raise ValueError("No overlapping '*_stor' columns between sensitivities and configs to build configuration key.")

# Normalize *_stor columns before building labels
for c in stor_cols_join:
    sdf[c] = _norm_str(sdf[c])
    cdf[c] = _norm_str(cdf[c])

# Build configuration labels and align node types
sdf = sdf.copy()
cdf = cdf.copy()
sdf["configuration"] = build_config_label(sdf, stor_cols_join)
cdf["configuration"] = build_config_label(cdf, stor_cols_join)

sdf[NODES_COL] = pd.to_numeric(sdf[NODES_COL], errors="coerce").astype("Int64")
cdf[NODES_COL] = pd.to_numeric(cdf[NODES_COL], errors="coerce").astype("Int64")
cdf[TOTAL_COL] = pd.to_numeric(cdf.get(TOTAL_COL, np.nan), errors="coerce")

# --- Join to get region + total per (nodes, configuration) ---
joined = pd.merge(
    sdf[[REGION_COL, NODES_COL, "configuration"]],
    cdf[[NODES_COL, "configuration", TOTAL_COL]],
    on=[NODES_COL, "configuration"],
    how="inner"
)

# --- Aggregate per configuration (keep a single region label per config via mode) ---
joined_agg = (
    joined
    .groupby([NODES_COL, "configuration"], as_index=False)
    .agg(
        total_mean=(TOTAL_COL, "mean"),
        region=(REGION_COL, series_mode)
    )
)
joined_agg.rename(columns={"total_mean": TOTAL_COL}, inplace=True)

# --- Node scales to plot ---
desired_scales = [2, 4, 8]
scales_present = [s for s in desired_scales if s in joined_agg[NODES_COL].dropna().unique().tolist()]
if not scales_present:
    scales_present = sorted(joined_agg[NODES_COL].dropna().unique().tolist())

# --- Build per-scale, sorted datasets and global Y range ---
per_scale_points = {}
all_totals = []

for s in scales_present:
    df_s = joined_agg[joined_agg[NODES_COL] == s].copy()
    df_s = df_s.sort_values(TOTAL_COL, ascending=True).reset_index(drop=True)
    per_scale_points[s] = df_s
    all_totals.extend(df_s[TOTAL_COL].tolist())

if not all_totals:
    raise ValueError("No totals available after joining sensitivities with configs.")

ymin = float(np.nanmin(all_totals))
ymax = float(np.nanmax(all_totals))
yrange = ymax - ymin if ymax > ymin else (abs(ymax) if ymax != 0 else 1.0)
pad = 0.08 * yrange
ymin_plot = ymin - pad
ymax_plot = ymax + pad

# --- Build a numeric-ordered region→code mapping for consistent colors across subplots ---
all_regions = pd.unique(joined_agg[REGION_COL])
region_order = sorted(all_regions, key=_region_sort_key)   # numeric order if possible
region_to_code = {r: i for i, r in enumerate(region_order)}
code_to_region = {i: r for i, r in enumerate(region_order)}
# Use a fixed normalization so colorbar covers the full region set regardless of subplot contents
norm = plt.Normalize(vmin=0, vmax=len(region_order) - 1)

# --- Figure ---
fig, axes = plt.subplots(1, len(scales_present), figsize=(5.8 * len(scales_present), 4.9), sharey=True)
if len(scales_present) == 1:
    axes = [axes]

last_sc = None
for ax, s in zip(axes, scales_present):
    dfp = per_scale_points[s]
    if dfp.empty:
        ax.text(0.5, 0.5, f"No data for nodes={s}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    x = np.arange(1, len(dfp) + 1)
    y = dfp[TOTAL_COL].values
    r_codes = dfp[REGION_COL].map(region_to_code).values

    last_sc = ax.scatter(x, y, c=r_codes, s=28, norm=norm, cmap="viridis")
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_title(f"Nodes = {s}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

# Axis labels
axes[0].set_ylabel("Total cost")
fig.supxlabel("Configuration index (ascending per subplot)")

# Colorbar with true region labels, in numeric order
cbar = fig.colorbar(last_sc, ax=axes, orientation="vertical", fraction=0.035, pad=0.02)
cbar.set_ticks(list(code_to_region.keys()))
cbar.set_ticklabels([str(code_to_region[k]) for k in code_to_region.keys()])
cbar.set_label("Region")

# Manual spacing (no tight_layout)
fig.subplots_adjust(wspace=0.18, left=0.07, right=0.86, bottom=0.12, top=0.90)

# --- Save figure ---
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print("Saved figure:", OUT_PNG)
