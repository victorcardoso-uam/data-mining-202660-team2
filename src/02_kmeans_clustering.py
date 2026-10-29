"""
02_kmeans_clustering.py
Business question. How many types of bottleneck scenarios exist in the plant?

The script follows these steps.
  1. Selects the 7 operational variables from the proposal (Milestone 1).
  2. Applies log1p to the skewed variables (WIP, overtime, idle) and then StandardScaler
     (K-Means uses Euclidean distances, so without scaling overtime would dominate everything).
  3. Sweeps k = 2..10 and computes inertia (elbow), Silhouette and Davies-Bouldin.
  4. Trains the chosen k and profiles each cluster (including its non-compliance rate).

Leakage note. The target variable and actual_productivity are NOT used in the clustering,
they are only used AFTERWARDS to describe the clusters.

Run it with  python src/02_kmeans_clustering.py
"""
from pathlib import Path

import matplotlib  # pyright: ignore[reportMissingModuleSource]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "garments_clean.csv"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SKEWED = ["wip", "over_time", "idle_time", "idle_men"]
OTHER = ["smv", "no_of_workers", "no_of_style_change"]
FEATURES = SKEWED + OTHER
RANDOM_STATE = 42

prep = ColumnTransformer([
    ("log_scale", Pipeline([("log", FunctionTransformer(np.log1p)),
                            ("scale", StandardScaler())]), SKEWED),
    ("scale", StandardScaler(), OTHER),
])

df = pd.read_csv(DATA)
X = prep.fit_transform(df[FEATURES])

# ---- k sweep --------------------------------------------------------------
rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    rows.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X, km.labels_),
        "davies_bouldin": davies_bouldin_score(X, km.labels_),
    })
scan = pd.DataFrame(rows)
print(scan.round(3).to_string(index=False))
scan.to_csv(ROOT / "reports" / "kmeans_k_scan.csv", index=False)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(scan["k"], scan["inertia"], "o-", color="#1f4e79", label="Inertia (elbow)")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (sum of squared distances)")
ax2 = ax1.twinx()
ax2.plot(scan["k"], scan["silhouette"], "s-", color="#c0504d", label="Silhouette")
ax2.axhline(0.35, ls="--", color="#c0504d", alpha=0.5, label="Criterion ≥ 0.35")
ax2.set_ylabel("Silhouette score")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, loc="center right")
ax1.set_title("Choosing k, elbow method vs. Silhouette")
fig.tight_layout()
fig.savefig(FIG / "kmeans_elbow_silhouette.png", dpi=200)
plt.close(fig)

# ---- Final model ----------------------------------------------------------
# Chosen k is the one with the best Silhouette between 3 and 6 (k=2 only separates
# the departments and tells the manager nothing new, and above 6 the clusters
# stop being interpretable).
candidates = scan[(scan["k"] >= 3) & (scan["k"] <= 6)]
K = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(X)
df["cluster"] = km.labels_
print(f"\nChosen k = {K} | Silhouette = {silhouette_score(X, km.labels_):.3f} "
      f"| Davies-Bouldin = {davies_bouldin_score(X, km.labels_):.3f}")

profile = df.groupby("cluster").agg(
    registros=("team", "size"),
    pct_sewing=("department", lambda s: (s == "sewing").mean()),
    smv=("smv", "mean"),
    wip=("wip", "mean"),
    over_time=("over_time", "mean"),
    incentive=("incentive", "mean"),
    idle_time=("idle_time", "mean"),
    idle_men=("idle_men", "mean"),
    style_change=("no_of_style_change", "mean"),
    workers=("no_of_workers", "mean"),
    tasa_incumplimiento=("missed_target", "mean"),
).round(2)
print("\nCluster profile\n", profile.to_string())
profile.to_csv(ROOT / "reports" / "kmeans_cluster_profile.csv")
df.to_csv(ROOT / "data" / "processed" / "garments_clustered.csv", index=False)

# Chart showing the non-compliance rate by cluster
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(profile.index.astype(str), profile["tasa_incumplimiento"] * 100,
              color="#1f4e79")
ax.axhline(df["missed_target"].mean() * 100, ls="--", color="gray",
           label=f"Plant average ({df['missed_target'].mean():.1%})")
for b, n in zip(bars, profile["registros"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"n={n}",
            ha="center", fontsize=9)
ax.set_xlabel("Cluster")
ax.set_ylabel("% of records that miss the target")
ax.set_title("Risk of missing the target by operating scenario (K-Means)")
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "kmeans_risk_by_cluster.png", dpi=200)
plt.close(fig)
print("\nFigures saved to reports/figures/")
