#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator   # <-- added

csv_path = "./ddmd/workflow_makespan_stageorder.csv"
desired_scales = [1, 2, 4]
out_png = "./ddmd/total_cost_comparison_subplots_numbers.png"
mapping_csv = "./ddmd/config_number_mapping.csv"

df = pd.read_csv(csv_path)
nodes_col = "nodes"
total_col = "total"

stor_cols = sorted([c for c in df.columns if c.endswith("_stor")])
if not stor_cols:
    raise ValueError("No columns ending with '_stor' found to define configurations.")

def make_config_label(row):
    parts = []
    for c in stor_cols:
        base = c[:-5]
        parts.append(f"{base}:{row[c]}")
    return " | ".join(parts)

df = df.copy()
df["configuration"] = df.apply(make_config_label, axis=1)
df[nodes_col] = df[nodes_col].astype(int)

df_sub = df[df[nodes_col].isin(desired_scales)].copy()
agg = df_sub.groupby(["configuration", nodes_col], as_index=False)[total_col].mean()

totals_by_scale = agg.groupby(nodes_col)[total_col].sum().sort_values()
lowest_scale = int(totals_by_scale.index[0])

order_df = agg[agg[nodes_col] == lowest_scale].sort_values(total_col, ascending=True)
config_order = list(order_df["configuration"])

scales_present = [s for s in desired_scales if s in agg[nodes_col].unique()]
plot_data = {}
global_vals = []
for s in scales_present:
    s_vals = agg[agg[nodes_col] == s].set_index("configuration")[total_col].reindex(config_order)
    plot_data[s] = s_vals
    global_vals.extend(s_vals.dropna().tolist())

config_numbers = {cfg: i+1 for i, cfg in enumerate(config_order)}
mapping_df = pd.DataFrame({"#": [config_numbers[c] for c in config_order], "configuration": config_order})
mapping_df.to_csv(mapping_csv, index=False)

ymin = min(global_vals)
ymax = max(global_vals)
yrange = ymax - ymin if ymax > ymin else (abs(ymax) if ymax != 0 else 1.0)
pad = 0.08 * yrange
ymin_plot = ymin - pad
ymax_plot = ymax + pad

fig, axes = plt.subplots(1, len(scales_present), figsize=(5.2*len(scales_present), 4.6), sharey=True)
if len(scales_present) == 1:
    axes = [axes]

legend_proxy = [Patch(label="Total cost")]

for ax, s in zip(axes, scales_present):
    vals = plot_data[s]
    x = np.arange(1, len(config_order)+1)
    ax.bar(x, vals.values)
    # show major tick every 10th configuration number
    ax.xaxis.set_major_locator(MultipleLocator(5))
    # optional: minor ticks for all bars (unlabeled)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_title(f"Nodes = {s}")
    ax.set_ylim(ymin_plot, ymax_plot)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_proxy, loc="upper left", frameon=False)

axes[0].set_ylabel("Total cost")
fig.supxlabel("Configuration index (ascending per subplot)")
fig.tight_layout()
fig.savefig(out_png, dpi=200, bbox_inches="tight")

print("Saved:", out_png)
print("Mapping CSV:", mapping_csv)
