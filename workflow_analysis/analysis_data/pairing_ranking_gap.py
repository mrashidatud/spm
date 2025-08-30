#!/usr/bin/env python3
"""
Top‑K stage‑storage‑node pairing recommender.

Algorithm
---------
1. *Per‑node grouping*  (gap‑growth rule)
   • For each node‑count, sort runtimes ascending.
   • g[i] = t[i+1] − t[i].       Start a new group if
       g[i] > (1 + GAP_THR) * g[i−1]        (default 20 % growth).
2. *Cross‑node aggregation*
   • Groups sharing the **same storage pattern** are merged
     when their mean runtimes differ ≤ AGG_THR (default 20 %).
3. *Wildcard consolidation*
   • A pattern with “*” absorbs a more specific sibling
     (e.g. `mutation: *` absorbs `mutation: tmpfs`) as long as
     their means are within AGG_THR.
4. Rank by fastest representative runtime; report slowdown vs. best.

Author: ChatGPT
"""
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
PRETTY = {                          # readable stage names
    "individuals_stor":       "individuals",
    "individuals_merge_stor": "merge",
    "merge_stor":             "merge",
    "sifting_stor":           "sifting",
    "mutation_overlap_stor":  "mutation",
    "mutation_stor":          "mutation",
    "frequency_stor":         "frequency",
}


# ──────────────────────  helpers  ────────────────────────────────────────────
def gap_groups(df_sorted, thr: float, col="total"):
    """Yield consecutive slices split by >thr gap‑growth rule."""
    times = df_sorted[col].values
    if len(times) <= 1:
        yield df_sorted
        return
    gaps, start = np.diff(times), 0
    for i in range(1, len(gaps)):
        if gaps[i] > gaps[i - 1] * (1 + thr):
            yield df_sorted.iloc[start : i + 1]
            start = i + 1
    yield df_sorted.iloc[start:]


def stage_pattern(rows: pd.DataFrame, store_cols):
    parts = []
    for c in store_cols:
        stage = PRETTY.get(c, c.replace("_stor", ""))
        vals = set(rows[c])
        parts.append(f"{stage}: {vals.pop() if len(vals)==1 else '*'}")
    return ", ".join(parts)


def pat_to_dict(pattern: str):
    return {k.strip(): v.strip()
            for k, v in (item.split(": ", 1) for item in pattern.split(", "))}


# ──────────────────────  wildcard merge  ─────────────────────────────────────
def wildcard_merge(groups: list[dict], agg_thr: float):
    """
    Further consolidate groups when a wildcard pattern ('*') covers
    a more specific sibling and their means differ ≤ agg_thr.
    """
    merged = []
    while groups:
        g = groups.pop(0)
        g_dict = pat_to_dict(g["pattern"])

        i = 0
        while i < len(groups):          # use index because we'll pop inside loop
            h = groups[i]
            h_dict = pat_to_dict(h["pattern"])

            # does g's '*' cover h's concrete values?  (ignore node field)
            covers = all(g_dict[k] in ("*", h_dict[k]) for k in h_dict if k != "node")

            # runtimes close enough?
            close = (max(g["mean"], h["mean"])
                     <= min(g["mean"], h["mean"]) * (1 + agg_thr))

            if covers and close:
                # merge h into g
                g["nodes"].update(h["nodes"])
                g["rows"] = pd.concat([g["rows"], h["rows"]], ignore_index=True)
                g["mean"] = (g["mean"] + h["mean"]) / 2
                g["rep"]  = min(g["rep"], h["rep"])
                groups.pop(i)           # remove h (do NOT advance i)
            else:
                i += 1                  # advance only when nothing is removed
        merged.append(g)
    return merged


# ──────────────────────  main processing  ────────────────────────────────────
def two_order_grouping(csv_path: Path, gap_thr=0.20, agg_thr=0.20):
    df = pd.read_csv(csv_path)
    store_cols = [c for c in df.columns if c.endswith("_stor")]
    node_col, total_col = "nodes", "total"

    node_groups = []
    # ① per-node gap‑based groups
    for node, df_node in df.groupby(node_col):
        best_cfg = (df_node.groupby(store_cols, as_index=False)[total_col]
                    .min()
                    .sort_values(total_col)
                    .reset_index(drop=True))
        for rows in gap_groups(best_cfg, thr=gap_thr, col=total_col):
            node_groups.append({
                "pattern": stage_pattern(rows, store_cols),
                "nodes":   {int(node)},
                "rows":    rows,
                "mean":    rows[total_col].mean(),
                "rep":     rows[total_col].min(),
            })

    # ② merge identical patterns across nodes (within agg_thr)
    patt_map = {}
    for g in node_groups:
        patt_map.setdefault(g["pattern"], []).append(g)

    merged = []
    for patt, lst in patt_map.items():
        lst.sort(key=lambda x: x["mean"])       # ascending mean
        cur = lst[0]
        for nxt in lst[1:]:
            if nxt["mean"] <= cur["mean"] * (1 + agg_thr):
                cur["nodes"].update(nxt["nodes"])
                cur["rows"] = pd.concat([cur["rows"], nxt["rows"]], ignore_index=True)
                cur["mean"] = (cur["mean"] + nxt["mean"]) / 2
                cur["rep"]  = min(cur["rep"], nxt["rep"])
            else:
                merged.append(cur);  cur = nxt
        merged.append(cur)

    # ③ wildcard consolidation
    final_groups = wildcard_merge(merged, agg_thr=agg_thr)
    final_groups.sort(key=lambda x: x["rep"])
    return final_groups


def build_table(groups: list[dict], top_k: int = 10):
    best = groups[0]["rep"]
    rows = []
    for rank, g in enumerate(groups[:top_k], 1):
        node_part = f"node: {next(iter(g['nodes']))}" if len(g["nodes"]) == 1 else "node: *"
        rows.append({
            "Rank":  rank,
            "Pairing": f"{g['pattern']}, {node_part}",
            "Representative runtime (s)": round(g["rep"], 3),
            "Slowdown vs best": f"{g['rep']/best:.2f}×",
        })
    return pd.DataFrame(rows)


# ──────────────────────  CLI  ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Recommend top‑K stage‑storage‑node pairings.")
    ap.add_argument("csv", nargs="?", default="workflow_makespan_stageorder.csv",
                    help="CSV with per‑run runtimes")
    ap.add_argument("--top", type=int, default=10, help="how many pairings to show")
    ap.add_argument("--gap_thr", type=float, default=0.20,
                    help="gap growth threshold inside a node‑count (20 %% default)")
    ap.add_argument("--agg_thr", type=float, default=0.20,
                    help="mean‑runtime threshold for merging groups (20 %% default)")
    args = ap.parse_args()

    groups = two_order_grouping(Path(args.csv), args.gap_thr, args.agg_thr)
    print(build_table(groups, args.top).to_string(index=False))


if __name__ == "__main__":
    main()
