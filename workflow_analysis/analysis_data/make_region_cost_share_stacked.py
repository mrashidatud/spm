#!/usr/bin/env python3
"""
Region stacked shares + overlaid avg total makespan (single-row legend on top),
USING workflow_rowwise_sensitivities.csv (not regions_by_total.csv).

What this does
--------------
1) Reads:
   - workflow_rowwise_sensitivities.csv  (has 'region', 'nodes', *_stor assignments per row)
   - workflow_makespan_stageorder.csv    (per-configuration totals and per-stage costs)

2) Builds a join key "configuration" in BOTH files from the intersection of *_stor columns,
   then inner-joins on ['nodes','configuration'] to get (region, nodes, costs).

3) For each joined row:
   shared_exec = Σ(read_<stage> + write_<stage>) for stages with <stage>_stor == 'beegfs'
   local_exec  = Σ(read_<stage> + write_<stage>) for stages with <stage>_stor in {'ssd','tmpfs',others}
   movement    = Σ(in_<stage> + out_<stage>) for all stages
   denom       = critical_path if > 0 else (shared_exec + local_exec + movement)
   shares      = components / denom

4) Aggregates by (nodes, region):
   - mean_shared, mean_local, mean_movement  (renormalized to sum to 1)
   - avg_total = mean(total)

5) Plots THREE SUBPLOTS (one per node scale: 2, 5, 10 if present):
   - X-axis: region #
   - Left Y-axis: stacked bars of the three shares (sum=1)
   - Right Y-axis: overlaid dashed red line of avg_total, with the SAME y2 range across subplots
   - Regions are sorted within each subplot by ascending avg_total.
   - A single, centered, top legend in one row.

Output
------
- region_share_stacked_with_makespan_from_sensitivities.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# --------- Paths ---------
SENS_CSV    = "./sens_out/workflow_rowwise_sensitivities.csv"
CONFIGS_CSV = "./ddmd/workflow_makespan_stageorder.csv"
OUT_PNG     = "./sens_out/region_share_stacked_with_makespan_from_sensitivities.png"

# --------- Node scales to show ---------
DESIRED_SCALES = [1, 2, 4]

# --------- Colors (high contrast) ---------
COLOR_SHARED = "#1f77b4"  # blue
COLOR_LOCAL  = "#ff7f0e"  # orange
COLOR_MOVE   = "#2ca02c"  # green
COLOR_LINE   = "#d62728"  # red (dashed)

# --------- Helpers ---------
def _to_num(s: pd.Series) -> pd.Series:
    """Convert to numeric, coercing errors to 0.0."""
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _stor_class(val: str) -> str:
    """Classify storage into 'shared' (beegfs) or 'local' (ssd/tmpfs/other)."""
    s = str(val).strip().lower()
    return "shared" if s == "beegfs" else "local"

def _build_config_label(df: pd.DataFrame, stor_cols: List[str]) -> pd.Series:
    """Create a deterministic configuration label from *_stor columns."""
    parts = []
    for c in sorted(stor_cols):
        base = c[:-5]  # drop '_stor'
        parts.append(df[c].astype(str).map(lambda v: f"{base}:{v}"))
    return pd.Series([" | ".join(vals) for vals in zip(*parts)], index=df.index)

# --------- Load data ---------
sdf = pd.read_csv(SENS_CSV)     # contains 'region', 'nodes', *_stor columns
cdf = pd.read_csv(CONFIGS_CSV)  # totals and stage costs

# Identify *_stor columns for joining
stor_cols_sens = sorted([c for c in sdf.columns if c.endswith("_stor")])
stor_cols_cfg  = sorted([c for c in cdf.columns if c.endswith("_stor")])
stor_cols_join = [c for c in stor_cols_sens if c in stor_cols_cfg]
if not stor_cols_join:
    raise ValueError("No overlapping '*_stor' columns between sensitivities and makespan CSVs.")

# Build configuration label in both dataframes
sdf = sdf.copy()
cdf = cdf.copy()
sdf["configuration"] = _build_config_label(sdf, stor_cols_join)
cdf["configuration"] = _build_config_label(cdf, stor_cols_join)

# Coerce nodes to integers
sdf["nodes"] = pd.to_numeric(sdf["nodes"], errors="coerce").astype("Int64")
cdf["nodes"] = pd.to_numeric(cdf["nodes"], errors="coerce").astype("Int64")

# Ensure numeric totals/critical path
cdf["total"] = _to_num(cdf.get("total", 0.0))
cdf["critical_path"] = _to_num(cdf.get("critical_path", 0.0))

# Build per-row components in configs
stages = [c[:-5] for c in stor_cols_cfg]  # base stage names
read_by_stage  = {st: _to_num(cdf.get(f"read_{st}", 0.0))  for st in stages}
write_by_stage = {st: _to_num(cdf.get(f"write_{st}", 0.0)) for st in stages}
in_by_stage    = {st: _to_num(cdf.get(f"in_{st}", 0.0))    for st in stages}
out_by_stage   = {st: _to_num(cdf.get(f"out_{st}", 0.0))   for st in stages}

shared_exec = pd.Series(0.0, index=cdf.index)
local_exec  = pd.Series(0.0, index=cdf.index)
movement    = pd.Series(0.0, index=cdf.index)

# For splitting exec cost by stor class, use *_stor columns from cdf
for st, stor_col in zip(stages, stor_cols_cfg):
    stor_kind = cdf[stor_col].map(_stor_class)
    exec_cost = read_by_stage[st] + write_by_stage[st]
    move_cost = in_by_stage[st] + out_by_stage[st]
    movement += move_cost
    shared_exec += exec_cost.where(stor_kind.eq("shared"), 0.0)
    local_exec  += exec_cost.where(stor_kind.eq("local"),  0.0)

cdf["shared_exec"] = shared_exec
cdf["local_exec"]  = local_exec
cdf["movement"]    = movement

# Join sensitivities rows (to read region labels) with per-row costs (to compute shares)
joined = pd.merge(
    sdf[["region", "nodes", "configuration"]],
    cdf[["nodes", "configuration", "total", "critical_path", "shared_exec", "local_exec", "movement"]],
    on=["nodes", "configuration"],
    how="inner"
)

# Compute per-row shares
denom = joined["critical_path"].copy()
fallback = denom <= 0
denom.loc[fallback] = (joined.loc[fallback, "shared_exec"]
                       + joined.loc[fallback, "local_exec"]
                       + joined.loc[fallback, "movement"])
denom = denom.replace(0, np.nan)

joined["share_shared"]   = (joined["shared_exec"] / denom).fillna(0.0)
joined["share_local"]    = (joined["local_exec"]  / denom).fillna(0.0)
joined["share_movement"] = (joined["movement"]    / denom).fillna(0.0)

# Aggregate per (nodes, region)
agg = joined.groupby(["nodes", "region"], as_index=False).agg(
    mean_shared=("share_shared", "mean"),
    mean_local=("share_local", "mean"),
    mean_movement=("share_movement", "mean"),
    avg_total=("total", "mean"),
)

# Renormalize shares to sum to 1
s_sum = (agg["mean_shared"] + agg["mean_local"] + agg["mean_movement"]).replace(0, np.nan)
agg["mean_shared"]   = (agg["mean_shared"]   / s_sum).fillna(0.0)
agg["mean_local"]    = (agg["mean_local"]    / s_sum).fillna(0.0)
agg["mean_movement"] = (agg["mean_movement"] / s_sum).fillna(0.0)

# Scales to plot
scales_present = [s for s in DESIRED_SCALES if s in agg["nodes"].dropna().unique().tolist()]
if not scales_present:
    scales_present = sorted(agg["nodes"].dropna().unique().tolist())

# Sort regions within each scale by ascending avg_total
per_scale: Dict[int, pd.DataFrame] = {}
for scale in scales_present:
    df_s = agg[agg["nodes"] == scale].copy()
    try:
        df_s["_rnum"] = pd.to_numeric(df_s["region"], errors="coerce")
        df_s = df_s.sort_values(["avg_total", "_rnum"], ascending=[True, True]).drop(columns=["_rnum"])
    except Exception:
        df_s = df_s.sort_values(["avg_total", "region"], ascending=[True, True])
    per_scale[scale] = df_s.reset_index(drop=True)

# Global y2 (makespan) range shared across subplots
all_avg_totals = pd.concat([per_scale[s]["avg_total"] for s in scales_present], ignore_index=True) if scales_present else pd.Series([], dtype=float)
y2_min = 0.0 if all_avg_totals.empty else min(0.0, float(np.nanmin(all_avg_totals)))
y2_max = 1.05 * float(np.nanmax(all_avg_totals)) if not all_avg_totals.empty else 1.0

# Plot: 1 row, up to 3 columns
nplots = len(scales_present)
fig, axes = plt.subplots(1, nplots, figsize=(6.6*nplots, 5.2), sharey=True)
if nplots == 1:
    axes = [axes]

# We'll gather handles once (from the first subplot) for a single-row legend on top
legend_handles = None
legend_labels  = None
line_handle_for_legend = None

for ax, scale in zip(axes, scales_present):
    df_s = per_scale[scale]
    if df_s.empty:
        ax.text(0.5, 0.5, f"No data for nodes={scale}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    x = np.arange(len(df_s))

    # Stacked bars on left y-axis
    b1 = ax.bar(x, df_s["mean_shared"].values, width=0.8, color=COLOR_SHARED, label="Shared exec")
    b2 = ax.bar(x, df_s["mean_local"].values,  width=0.8,
                bottom=df_s["mean_shared"].values, color=COLOR_LOCAL, label="Local exec")
    bottom2 = df_s["mean_shared"].values + df_s["mean_local"].values
    b3 = ax.bar(x, df_s["mean_movement"].values, width=0.8,
                bottom=bottom2, color=COLOR_MOVE, label="Data movement")

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Share of critical cost (stacked)")
    ax.set_xlabel("Region #")
    ax.set_xticks(x)
    ax.set_xticklabels(df_s["region"].astype(str).tolist(), rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Overlay dashed red line on right y-axis (consistent range)
    ax2 = ax.twinx()
    (line_handle,) = ax2.plot(
        x, df_s["avg_total"].values,
        linestyle="--", color=COLOR_LINE, marker="o", linewidth=2, markersize=4,
        label="Avg total makespan",
    )
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel("Avg total makespan")

    # Save legend handles/labels from first subplot only
    if legend_handles is None:
        legend_handles = [b1, b2, b3, line_handle]
        legend_labels  = ["Shared exec", "Local exec", "Data movement", "Avg total makespan"]

    ax.set_title(f"Nodes = {scale}")

# Single-row, centered legend on top (outside axes, above)
fig.legend(
    legend_handles, legend_labels,
    loc="upper center", ncol=4, frameon=False,
    bbox_to_anchor=(0.5, 0.97), borderaxespad=0.0, columnspacing=1.8, handlelength=3.0
)

# Adjust layout to make room for top legend
fig.subplots_adjust(wspace=0.20, left=0.07, right=0.93, bottom=0.18, top=0.88)

fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
