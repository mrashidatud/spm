#!/usr/bin/env python
# -----------------------------------------------------------
# cluster_summary_param.py
# -----------------------------------------------------------
# Ward hierarchical clustering with user‑controlled parameters
# * Either set a fixed --k  OR  search a wider --k-range
# -----------------------------------------------------------
import argparse, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--k", type=int, help="Force exact number of clusters")
p.add_argument("--k-range", type=str, default="2-10",
               help="Range for silhouette search, e.g. 3-12")
args = p.parse_args()

# ---------- 1. Load & encode ----------
df = pd.read_csv("workflow_makespan_stageorder.csv")
cat_cols = [c for c in df.columns if c.endswith("_store")]
num_cols = ["nodes"]
enc = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
     ("num", "passthrough", num_cols)]
).fit_transform(df[cat_cols + num_cols])
dense = enc.toarray() if hasattr(enc, "toarray") else enc

# ---------- 2. Pick k ----------
if args.k:
    best_k = args.k
else:
    lo, hi = map(int, args.k_range.split("-"))
    best_k, best_score = None, -1
    for k in range(lo, hi + 1):
        lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(dense)
        sc  = silhouette_score(dense, lbl)
        if sc > best_score:
            best_k, best_score = k, sc
    print(f"Chose k = {best_k} (silhouette = {best_score:.3f})")

# ---------- 3. Final clustering ----------
labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(dense)
df["cluster"] = labels

# ---------- 4. Summaries ----------
summ = []
for cl, g in df.groupby("cluster"):
    commons = {c: (g[c].iat[0] if g[c].nunique() == 1 else "*") for c in cat_cols}
    commons["nodes"] = ",".join(str(n) for n in sorted(g["nodes"].unique()))
    summ.append(dict(cluster=cl,
                     size=len(g),
                     mean=g.total.mean(),
                     std=g.total.std(),
                     **commons))
pd.DataFrame(summ).sort_values("mean").to_csv("cluster_regions.csv", index=False)
print("Written: cluster_regions.csv")
