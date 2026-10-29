import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score

# Configure headless execution for safely saving plots in MLOps pipelines
plt.switch_backend('Agg')

# Dynamic path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'garments_worker_productivity.csv')
)
FIGURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'reports', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Data Ingestion & Preprocessing
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    df = pd.read_csv('garments_worker_productivity.csv')

# Normalize known department typos (e.g., 'sweing ' -> 'sewing')
df['department'] = df['department'].str.strip().str.lower()
df['department'] = df['department'].replace({'sweing': 'sewing'})

# IMMEX Imputation: Finishing WIP is physically 0 (avoiding mean/median imputation)
df['missing_wip'] = df['wip'].isnull().astype(int)
df['wip'] = df['wip'].fillna(0)

# Feature Engineering
df['gap'] = df['actual_productivity'] - df['targeted_productivity']
df['reaches_goal'] = (df['gap'] >= 0).astype(int)

print("=" * 65)
print(" GARMENT IMMEX WORKER PRODUCTIVITY - MILESTONE 2 PIPELINE")
print("=" * 65)
print(f"Total Ingested Records    : {df.shape[0]} (Min 1,000+ Requirement Met)")
print(f"Failed Records (gap < 0) : {(1 - df['reaches_goal'].mean()) * 100:.2f}%")
print("=" * 65)

# 2. Model 1: Decision Tree Classifier (Risk Classification)
features_tree = ['smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers']
X = df[features_tree]
y = df['reaches_goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Pruned Decision Tree to prevent overfitting and guarantee plant interpretability
tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

# Export Confusion Matrix plot using pure Matplotlib
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=14, fontweight='bold', color='black')

# Fixed tick locations to prevent UserWarning
ax.set_xticks([0, 1])
ax.set_xticklabels(['Miss Goal', 'Meet Goal'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Miss Goal', 'Meet Goal'])

ax.set_title('Decision Tree Confusion Matrix (Garment IMMEX)', fontsize=12, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Label', fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix_tree.png'), dpi=300)
plt.close(fig)

# 3. Model 2: K-Means Clustering (Bottleneck Scenarios)
features_kmeans = ['smv', 'wip', 'over_time', 'idle_time', 'no_of_workers']
X_kmeans = df[features_kmeans]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_kmeans)
sil_score = silhouette_score(X_kmeans, df['cluster'])

print(f"\nK-Means Cohesion Score (Silhouette) : {sil_score:.4f}")

# Export Bottleneck Clusters plot using pure Matplotlib
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(df['wip'], df['smv'], c=df['cluster'], cmap='Set1', alpha=0.8, edgecolors='k')
ax.set_title('K-Means Bottleneck Clusters (WIP vs SMV)', fontsize=12, fontweight='bold')
ax.set_xlabel('Work In Process (WIP)', fontweight='bold')
ax.set_ylabel('Standard Minute Value (SMV)', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'kmeans_bottleneck_clusters.png'), dpi=300)
plt.close(fig)

print("\n--- PIPELINE COMPLETE: DIAGNOSTIC PLOTS SAVED TO reports/figures/ ---")
