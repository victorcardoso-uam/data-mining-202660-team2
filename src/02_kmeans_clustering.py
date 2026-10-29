"""
02_kmeans_clustering.py
Pregunta de negocio: ¿cuántos tipos de escenario de cuello de botella existen en la planta?

Pasos:
  1. Selecciona las 7 variables operativas de la propuesta (Milestone 1).
  2. log1p a las variables sesgadas (WIP, overtime, idle) y StandardScaler
     (K-Means usa distancias euclidianas: sin escalar, overtime dominaría todo).
  3. Barre k = 2..10 y calcula inercia (codo), Silhouette y Davies-Bouldin.
  4. Entrena el k elegido y perfila cada clúster (incluida su tasa de incumplimiento).

Nota anti-fuga: la variable objetivo y actual_productivity NO entran al clustering;
solo se usan DESPUÉS para describir los clústeres.

Uso:  python src/02_kmeans_clustering.py
"""
from pathlib import Path

import matplotlib

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

# ---- Barrido de k ---------------------------------------------------------
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
ax1.plot(scan["k"], scan["inertia"], "o-", color="#1f4e79", label="Inercia (codo)")
ax1.set_xlabel("Número de clústeres (k)")
ax1.set_ylabel("Inercia (suma de distancias²)")
ax2 = ax1.twinx()
ax2.plot(scan["k"], scan["silhouette"], "s-", color="#c0504d", label="Silhouette")
ax2.axhline(0.35, ls="--", color="#c0504d", alpha=0.5, label="Criterio ≥ 0.35")
ax2.set_ylabel("Silhouette score")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, loc="center right")
ax1.set_title("Selección de k: método del codo vs. Silhouette")
fig.tight_layout()
fig.savefig(FIG / "kmeans_elbow_silhouette.png", dpi=200)
plt.close(fig)

# ---- Modelo final ---------------------------------------------------------
# k elegido: el de mejor Silhouette entre 3 y 6 (k=2 solo separa departamentos
# y no le dice nada nuevo al gerente; > 6 deja de ser interpretable).
candidates = scan[(scan["k"] >= 3) & (scan["k"] <= 6)]
K = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(X)
df["cluster"] = km.labels_
print(f"\nk elegido = {K} | Silhouette = {silhouette_score(X, km.labels_):.3f} "
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
print("\nPerfil de clústeres:\n", profile.to_string())
profile.to_csv(ROOT / "reports" / "kmeans_cluster_profile.csv")
df.to_csv(ROOT / "data" / "processed" / "garments_clustered.csv", index=False)

# Gráfica: tasa de incumplimiento por clúster
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(profile.index.astype(str), profile["tasa_incumplimiento"] * 100,
              color="#1f4e79")
ax.axhline(df["missed_target"].mean() * 100, ls="--", color="gray",
           label=f"Promedio planta ({df['missed_target'].mean():.1%})")
for b, n in zip(bars, profile["registros"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"n={n}",
            ha="center", fontsize=9)
ax.set_xlabel("Clúster")
ax.set_ylabel("% de registros que no alcanzan la meta")
ax.set_title("Riesgo de incumplimiento por escenario operativo (K-Means)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "kmeans_risk_by_cluster.png", dpi=200)
plt.close(fig)
print("\nFiguras guardadas en reports/figures/")
