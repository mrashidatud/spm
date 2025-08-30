# plotting.py
import matplotlib.pyplot as plt, numpy as np, pandas as pd, os
from constants import OUT_DIR

def fmt_label(r):
    parts = [f"{c.replace('_stor','')}={r[c]}" 
                for c in r.index if c.endswith('_stor')]
    return "|".join(parts)

def plots(df, thresh=None):
    tag = f"_lt{thresh}" if thresh else ""
    nodes = sorted(df.nodes.unique())
    fig, axes = plt.subplots(len(nodes), 1,
                             figsize=(10,3*len(nodes)), dpi=120, sharey=False)
    if len(nodes)==1: axes=[axes]
    for ax,n in zip(axes,nodes):
        sub = df[df.nodes==n]
        x = np.arange(len(sub))
        ax.scatter(x, sub.total)
        ax.set_title(f"Nodes={n}{' (filtered)' if thresh else ''}")
        ax.set_ylabel("Makespan")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.apply(fmt_label, axis=1),
                       rotation=90, fontsize=6)
    axes[-1].set_xlabel("Configuration")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/scatter_by_nodes{tag}.png")
    # best-per-node
    best = df.loc[df.groupby('nodes').total.idxmin()]
    plt.figure(figsize=(5,4), dpi=120)
    plt.plot(best.nodes, best.total, marker='o')
    for _,r in best.iterrows():
        plt.annotate(f'N{r.nodes}', (r.nodes, r.total),
                     textcoords='offset points', xytext=(5,5), fontsize=8)
    plt.xlabel("Nodes"); plt.ylabel("Best makespan")
    plt.title("Best per node")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/best_per_node{tag}.png")
