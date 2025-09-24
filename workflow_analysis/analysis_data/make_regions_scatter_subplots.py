#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

regions_path = "./sens_out/regions_by_total.csv"          # path to regions CSV
configs_path = "./1kgenome/workflow_makespan_stageorder.csv"  # path to full configs CSV
out_png = "./sens_out/regions_scatter_true_labels.png"
out_map = "./sens_out/regions_scatter_index_mapping.csv"

region_col = "region"
nodes_col_r = "nodes"
total_col_c = "total"
nodes_col_c = "nodes"

# --- Load ---
rdf = pd.read_csv(regions_path)
cdf = pd.read_csv(configs_path)

# --- Find overlapping *_stor columns to match on ---
stor_cols_r = sorted([c for c in rdf.columns if c.endswith("_stor")])
stor_cols_c = sorted([c for c in cdf.columns if c.endswith("_stor")])
stor_cols = [c for c in stor_cols_r if c in stor_cols_c]
if not stor_cols:
    raise ValueError("No overlapping '*_stor' columns between the two CSVs to map regions to configurations.")

# --- Build human-readable configuration label in the configs CSV ---
def make_config_label(row):
    parts = []
    for c in sorted(stor_cols_c):
        parts.append(f"{c[:-5]}:{row[c]}")
    return " | ".join(parts)

cdf = cdf.copy()
cdf["configuration"] = cdf.apply(make_config_label, axis=1)
cdf[nodes_col_c] = cdf[nodes_col_c].astype(int)

# --- Parse possible multi-valued 'nodes' entries like '5,10' in regions CSV ---
def parse_nodes(val):
    s = str(val)
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except:
            pass
    return out if out else [np.nan]

rdf = rdf.copy()
rdf["nodes_list"] = rdf[nodes_col_r].apply(parse_nodes)
rdf = rdf.explode("nodes_list", ignore_index=True)
rdf = rdf.dropna(subset=["nodes_list"])
rdf["nodes_list"] = rdf["nodes_list"].astype(int)
rdf[nodes_col_r] = rdf["nodes_list"]
rdf = rdf.drop(columns=["nodes_list"])

# --- node scales to plot ---
desired_scales = [2, 5, 10]
scales_present = [s for s in desired_scales if s in cdf[nodes_col_c].unique() and s in rdf[nodes_col_r].unique()]
if not scales_present:
    scales_present = sorted(set(cdf[nodes_col_c].unique()).intersection(set(rdf[nodes_col_r].unique())))

# --- helper: interpret region stor entries with wildcards (*) and comma-separated choices ---
def allowed_set(val):
    s = str(val).strip()
    if s == "*" or s.lower() == "any":
        return None  # wildcard: accept any value
    opts = [p.strip() for p in s.split(",") if p.strip()]
    return set(opts) if opts else None

# --- Build per-scale points by mapping each region to matching configurations, then reading true totals ---
per_scale_points = {}
for s in scales_present:
    rdf_s = rdf[rdf[nodes_col_r] == s]
    cdf_s = cdf[cdf[nodes_col_c] == s]

    rows = []
    for _, r in rdf_s.iterrows():
        allow_map = {c: allowed_set(r[c]) for c in stor_cols}
        mask = np.ones(len(cdf_s), dtype=bool)
        for c in stor_cols:
            if allow_map[c] is None:
                continue
            mask &= cdf_s[c].isin(allow_map[c])
        matches = cdf_s[mask]
        if matches.empty:
            continue
        g = matches.groupby("configuration", as_index=False)[total_col_c].mean()
        g[region_col] = r[region_col]
        rows.append(g)

    points = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["configuration", total_col_c, region_col])
    points = points.sort_values(total_col_c, ascending=True).reset_index(drop=True)
    per_scale_points[s] = points

# --- Shared Y range ---
all_vals = []
for s, dfp in per_scale_points.items():
    all_vals.extend(dfp[total_col_c].tolist())
if not all_vals:
    raise ValueError("No matching configuration totals found when mapping regions to configurations.")

ymin = float(np.nanmin(all_vals))
ymax = float(np.nanmax(all_vals))
yrange = ymax - ymin if ymax > ymin else (abs(ymax) if ymax != 0 else 1.0)
pad = 0.08 * yrange
ymin_plot = ymin - pad
ymax_plot = ymax + pad

# --- Region label → numeric code for colormap; will display true labels on colorbar ---
all_region_labels = sorted(pd.unique(rdf[region_col].astype(str)))
region_to_code = {r:i for i, r in enumerate(all_region_labels)}
code_to_region = {i:r for r,i in region_to_code.items()}

# --- Figure ---
fig, axes = plt.subplots(1, len(scales_present), figsize=(5.8*len(scales_present), 4.9), sharey=True)
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
    y = dfp[total_col_c].values
    r_codes = dfp[region_col].astype(str).map(region_to_code).values
    last_sc = ax.scatter(x, y, c=r_codes, s=28)  # let matplotlib pick colormap

    ax.xaxis.set_major_locator(MultipleLocator(10))   # show every 10th config index
    ax.xaxis.set_minor_locator(MultipleLocator(1))    # unlabeled minor ticks
    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_title(f"Nodes = {s}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

axes[0].set_ylabel("Total cost")
fig.supxlabel("Configuration index (ascending per subplot)")

# True region labels on colorbar
cbar = fig.colorbar(last_sc, ax=axes, orientation="vertical", fraction=0.035, pad=0.02)
cbar.set_ticks(list(code_to_region.keys()))
cbar.set_ticklabels([code_to_region[k] for k in code_to_region.keys()])
cbar.set_label("Region")

# No tight_layout; adjust manually
fig.subplots_adjust(wspace=0.18, left=0.07, right=0.86, bottom=0.12, top=0.90)

fig.savefig(out_png, dpi=200, bbox_inches="tight")

# Save per-subplot mapping for traceability
summary_rows = []
for s in scales_present:
    dfp = per_scale_points[s].copy()
    dfp.insert(0, "index_in_subplot", np.arange(1, len(dfp)+1))
    dfp.insert(1, "nodes", s)
    summary_rows.append(dfp.rename(columns={total_col_c: "total"}))
summary = pd.concat(summary_rows, ignore_index=True)
summary.to_csv(out_map, index=False)

print("Saved figure:", out_png)
print("Saved mapping:", out_map)
