"""Generate the 8 graphs + results.json for the Hierarchical Clustering blog.
Run from the project root:  python generate_graphs.py
Dataset: data/Mall_Customers.csv (Kaggle: vjchoudhary7/customer-segmentation-tutorial-in-python)
"""
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

SEED = 42
np.random.seed(SEED)
OUT = Path("assets/images"); OUT.mkdir(parents=True, exist_ok=True)
INC, SPD = "Annual Income (k$)", "Spending Score (1-100)"
PAL = ["#1f6f8b", "#d1495b", "#edae49", "#66a182", "#6c4f9e", "#8d6e63", "#3d405b", "#00798c"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110})

# ---- 1. Load + inspect -------------------------------------------------------
df = pd.read_csv("data/Mall_Customers.csv")
print("shape:", df.shape, "| columns:", list(df.columns), "| missing:", int(df.isna().sum().sum()))
FEATURES = [INC, SPD]                      # CustomerID is an identifier -> excluded
X = df[FEATURES].to_numpy(float)
Xs = StandardScaler().fit_transform(X)     # z-score: (x - mean) / std
R = {"n_rows": len(df), "columns": list(df.columns), "missing": int(df.isna().sum().sum()),
     "features": FEATURES, "seed": SEED}

def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- Graph 1: raw distribution ----------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(df[INC], df[SPD], s=38, color="#4a5568", alpha=.75, edgecolor="white", lw=.5)
ax.set(xlabel="Annual Income (k$)  - x-axis feature", ylabel="Spending Score (1-100)  - y-axis feature",
       title="Mall Customers before clustering (n = 200)")
ax.text(.02, .97, "Both axes are the clustering features.\nNo colour = no labels known in advance.",
        transform=ax.transAxes, va="top", fontsize=9.5, bbox=dict(fc="#f1f5f9", ec="#cbd5e1", boxstyle="round"))
save(fig, "customer-data-distribution.png")

# ---- Graph 2: Euclidean vs Manhattan (math illustration) ---------------------
a, b = np.array([1., 1.]), np.array([5., 4.])
eu = float(np.linalg.norm(a - b)); ma = float(np.abs(a - b).sum())
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(*zip(a, b), color=PAL[0], lw=3, label=f"Euclidean = sqrt(4$^2$+3$^2$) = {eu:.0f}")
ax.plot([a[0], b[0], b[0]], [a[1], a[1], b[1]], color=PAL[1], lw=3, ls="--",
        label=f"Manhattan = |4| + |3| = {ma:.0f}")
ax.scatter(*zip(a, b), s=90, color="black", zorder=5)
ax.annotate("A (1, 1)", a, xytext=(-8, -22), textcoords="offset points")
ax.annotate("B (5, 4)", b, xytext=(6, 6), textcoords="offset points")
ax.text(3, .6, "Δx = 4", ha="center", color=PAL[1]); ax.text(5.15, 2.5, "Δy = 3", color=PAL[1])
ax.set(xlim=(0, 6.6), ylim=(0, 5), xlabel="Feature 1", ylabel="Feature 2",
       title="Euclidean vs Manhattan distance (mathematical illustration, not customer data)")
ax.legend(loc="upper left"); ax.set_aspect("equal")
save(fig, "hierarchical-distance-comparison.png")
R["g2"] = {"euclidean": eu, "manhattan": ma}

# ---- Graph 3: distance matrix on 12 sampled customers -----------------------
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(df), 12, replace=False))
D = squareform(pdist(Xs[idx], "euclidean"))
lab = [f"#{df.CustomerID[i]}" for i in idx]
fig, ax = plt.subplots(figsize=(8.2, 6.6))
im = ax.imshow(D, cmap="viridis_r")
ax.set_xticks(range(12), lab, rotation=45); ax.set_yticks(range(12), lab)
for i, j in itertools.product(range(12), range(12)):
    ax.text(j, i, f"{D[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
            color="white" if D[i, j] > D.max() * .55 else "black")
fig.colorbar(im, label="Euclidean distance (standardised units)")
ax.set_title("Pairwise distances: 12 randomly sampled customers (seed 42)")
save(fig, "hierarchical-distance-matrix.png")
iu = np.triu_indices(12, 1); k = np.argmin(np.where(D[iu] == 0, np.inf, D[iu]))
R["g3"] = {"ids": [int(df.CustomerID[i]) for i in idx], "min": float(D[iu][k]), "max": float(D.max()),
           "closest_pair": [lab[iu[0][k]], lab[iu[1][k]]]}

# ---- Linkage models (Euclidean everywhere; Ward requires Euclidean) ---------
methods = ["single", "complete", "average", "ward"]
Z = {m: linkage(Xs, method=m, metric="euclidean") for m in methods}
R["cophenetic"] = {m: float(cophenet(Z[m], pdist(Xs))[0]) for m in methods}

# ---- Graph 7 data first (used to pick k for Graphs 4, 6, 8) -----------------
ks = list(range(2, 11)); sil, db, sizes = [], [], {}
for kk in ks:
    lbl = AgglomerativeClustering(n_clusters=kk, linkage="ward").fit_predict(Xs)
    sil.append(silhouette_score(Xs, lbl)); db.append(davies_bouldin_score(Xs, lbl))
    sizes[kk] = np.bincount(lbl).tolist()
K = ks[int(np.argmax(sil))]            # chosen from the data, not forced
R.update(k_chosen=K, silhouette={str(k): float(s) for k, s in zip(ks, sil)},
         davies_bouldin={str(k): float(s) for k, s in zip(ks, db)})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
ax1.plot(ks, sil, "o-", color=PAL[0], lw=2)
ax1.scatter([K], [max(sil)], s=200, facecolor="none", edgecolor=PAL[1], lw=2.5, zorder=5)
for x, y in zip(ks, sil): ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
ax1.set(xlabel="Number of clusters (k)", ylabel="Mean silhouette score", xticks=ks,
        title="Silhouette score by k (Ward, scaled features)", ylim=(min(sil) - .04, max(sil) + .06))
ax2.bar([f"C{i}" for i in range(K)], sizes[K], color=PAL[:K])
for i, v in enumerate(sizes[K]): ax2.text(i, v + 1, str(v), ha="center")
ax2.set(title=f"Cluster sizes at k = {K}", ylabel="Customers")
save(fig, "hierarchical-cluster-evaluation.png")

# ---- Graph 4: dendrogram with cut -------------------------------------------
Zw = Z["ward"]
cut = (Zw[-K + 1, 2] + Zw[-K, 2]) / 2            # between the (K-1)th and Kth-from-last merges
fig, ax = plt.subplots(figsize=(12, 6))
dn = dendrogram(Zw, truncate_mode="lastp", p=30, ax=ax, color_threshold=cut, above_threshold_color="#6b7280",
                show_leaf_counts=True, leaf_rotation=90, leaf_font_size=9)
ax.axhline(cut, color="black", ls="--", lw=1.6)
ax.text(ax.get_xlim()[1] * .99, cut * 1.02, f"cut at height {cut:.2f} -> {K} clusters", ha="right", fontweight="bold")
# annotate the heights of the merges that survive above the cut
for h in Zw[-K + 1:, 2]:
    ax.annotate(f"{h:.2f}", (ax.get_xlim()[1] * .02, h), fontsize=8.5, color="#374151", xytext=(0, 3), textcoords="offset points")
ax.set(title="Ward dendrogram (Euclidean, scaled features), last 30 merges shown",
       xlabel="Leaves: (n) = number of customers merged into that leaf", ylabel="Merge distance (Ward criterion)")
save(fig, "hierarchical-dendrogram-cut.png")
R["g4"] = {"cut_height": float(cut), "top_merge_heights": [float(h) for h in Zw[-6:, 2]]}

# ---- Graph 5: linkage comparison --------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for ax, m in zip(axs.ravel(), methods):
    dendrogram(Z[m], truncate_mode="lastp", p=30, ax=ax, no_labels=True, color_threshold=0,
               above_threshold_color=PAL[methods.index(m)])
    ax.set_title(f"{m.capitalize()} linkage (cophenetic r = {R['cophenetic'][m]:.3f})")
    ax.set_ylabel("Merge distance")
fig.suptitle("Same data, same scaling, same Euclidean metric - different linkage (note different y-scales)", fontweight="bold")
fig.tight_layout(); save(fig, "hierarchical-linkage-comparison.png")
R["linkage_sizes_k5"] = {m: np.bincount(fcluster(Z[m], 5, "maxclust")).tolist()[1:] for m in methods}

# ---- Graph 6: final clusters -------------------------------------------------
model = AgglomerativeClustering(n_clusters=K, linkage="ward", metric="euclidean")
labels = model.fit_predict(Xs); df["Cluster"] = labels
fig, ax = plt.subplots(figsize=(8.5, 5.8))
for c in range(K):
    m = labels == c
    ax.scatter(df[INC][m], df[SPD][m], s=40, color=PAL[c], edgecolor="white", lw=.5,
               label=f"Cluster {c} (n={m.sum()})")
ax.set(xlabel="Annual Income (k$)", ylabel="Spending Score (1-100)",
       title=f"Agglomerative clustering, Ward linkage, k = {K}")
ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "hierarchical-customer-clusters.png")
R["final"] = {"silhouette": float(silhouette_score(Xs, labels)), "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
              "sizes": np.bincount(labels).tolist()}

# ---- Graph 8: profiles (medians of ORIGINAL features) ------------------------
prof = df.groupby("Cluster")[[INC, SPD, "Age"]].median()
prof["Size"] = df.Cluster.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4.8))
norm = (prof[[INC, SPD, "Age"]] - prof[[INC, SPD, "Age"]].min()) / (prof[[INC, SPD, "Age"]].max() - prof[[INC, SPD, "Age"]].min())
im = ax.imshow(norm.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(3), ["Median income (k$)", "Median spending score", "Median age (not used for fitting)"])
ax.set_yticks(range(K), [f"Cluster {c} (n={int(prof.Size[c])})" for c in range(K)])
for i in range(K):
    for j, col in enumerate([INC, SPD, "Age"]):
        ax.text(j, i, f"{prof[col].iloc[i]:.0f}", ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if norm.iloc[i, j] > .6 else "black")
fig.colorbar(im, label="Relative level within column (0 = lowest, 1 = highest)")
ax.set_title("Cluster profiles: medians of original (unscaled) features")
save(fig, "hierarchical-cluster-profiles.png")
R["profiles"] = prof.round(1).reset_index().to_dict("records")
Path("results.json").write_text(json.dumps(R, indent=2))
print(json.dumps({k: R[k] for k in ["k_chosen", "silhouette", "cophenetic", "final", "g4", "g3", "linkage_sizes_k5"]}, indent=1))
print(prof)
