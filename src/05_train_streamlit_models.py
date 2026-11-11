"""
05_train_streamlit_models.py
Milestone 3 (Activity 23): trains and serializes every model the Streamlit app loads.

Outputs (all relative to the repository root):
  data/team_dataset.csv                 clean dataset with the target `line_stoppage_flag`
  models/decision_tree_model.joblib     pruned Decision Tree pipeline (One-Hot + CART)
  models/ann_mlp_model.joblib           Neural Network pipeline (scaler + One-Hot + MLPClassifier)
  models/logistic_regression_model.joblib  classical linear baseline
  models/kmeans_model.joblib            K-Means pipeline (log1p + StandardScaler + KMeans k=6)
  models/standard_scaler.joblib         fitted feature preprocessor used by the ANN (scaling + encoding)
  models/metadata.json                  feature lists, test indices, thresholds, cluster names, CV scores

Target: line_stoppage_flag = 1 when actual_productivity < targeted_productivity
(the same definition used in Milestones 1-2, renamed as in the Activity 23 client table).

Anti-leakage: actual_productivity and date never enter X; the split is done before any fitting;
every scaler/encoder lives inside a Pipeline, so it is fitted on training data only.

Usage (from the repository root):  python src/05_train_streamlit_models.py
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "garments_worker_productivity.csv"
DATA_OUT = ROOT / "data" / "team_dataset.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42

TARGET = "line_stoppage_flag"
CATEGORICAL = ["department", "day", "quarter", "team"]
NUMERIC = ["targeted_productivity", "smv", "wip", "over_time", "incentive",
           "idle_time", "idle_men", "no_of_style_change", "no_of_workers"]
SKEWED = ["wip", "over_time", "incentive", "idle_time", "idle_men"]
REGULAR = [c for c in NUMERIC if c not in SKEWED]
CLUSTER_FEATURES = ["wip", "over_time", "idle_time", "idle_men",
                    "smv", "no_of_workers", "no_of_style_change"]
LEAKY = ["actual_productivity", "date"]
THRESHOLDS = {"yellow": 0.30, "red": 0.60}   # validated out-of-fold in Milestone 2


# ---------------------------------------------------------------- 1. clean data
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["department"] = df["department"].str.strip().replace({"sweing": "sewing"})
    df["quarter"] = df["quarter"].str.strip()
    df["day"] = df["day"].str.strip()
    # The 506 WIP nulls are exactly the Finishing records: no work in progress there.
    df.loc[df["department"] == "finishing", "wip"] = df.loc[
        df["department"] == "finishing", "wip"].fillna(0)
    df[TARGET] = (df["actual_productivity"] < df["targeted_productivity"]).astype(int)
    assert df[NUMERIC + CATEGORICAL].isna().sum().sum() == 0, "Nulls remain"
    return df


raw = pd.read_csv(RAW)
df = clean(raw)
df.to_csv(DATA_OUT, index=False)
print(f"Clean dataset: {df.shape} | {TARGET} rate = {df[TARGET].mean():.1%} -> {DATA_OUT.relative_to(ROOT)}")

X = df[CATEGORICAL + NUMERIC]
y = df[TARGET]
assert not set(LEAKY) & set(X.columns), "Data leakage: forbidden column in X"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------- 2. models
def scaled_preprocessor() -> ColumnTransformer:
    """One-Hot for categories, log1p + StandardScaler for skewed counts, StandardScaler for the rest."""
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("log_scale", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                ("scale", StandardScaler())]), SKEWED),
        ("scale", StandardScaler(), REGULAR),
    ])


models = {
    # Hyperparameters from the Milestone 2 GridSearchCV (stratified 5-fold, F1-macro)
    "Decision Tree": Pipeline([
        ("prep", ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC)])),
        ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, ccp_alpha=0.00126,
                                         random_state=RANDOM_STATE)),
    ]),
    "Logistic Regression": Pipeline([
        ("prep", scaled_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    # Best of a small CV scan: (16,), (32,16), (64,32) x alpha in {0.001, 0.01, 0.1}
    "Neural Network (ANN)": Pipeline([
        ("prep", scaled_preprocessor()),
        ("model", MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.01,
                                max_iter=800, early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=25, random_state=RANDOM_STATE)),
    ]),
}
files = {"Decision Tree": "decision_tree_model.joblib",
         "Logistic Regression": "logistic_regression_model.joblib",
         "Neural Network (ANN)": "ann_mlp_model.joblib"}

metrics = {}
for name, pipe in models.items():
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba > 0.5).astype(int)   # same rule as model.predict(): a 0.50 tie counts as "meets target"
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 (macro)": f1_score(y_test, pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV F1 (train, 5-fold)": float(cv_f1.mean()),
    }
    joblib.dump(pipe, MODELS / files[name])
    print(f"{name:22s} test F1-macro={metrics[name]['F1 (macro)']:.3f} "
          f"ROC-AUC={metrics[name]['ROC-AUC']:.3f}  CV F1={cv_f1.mean():.3f} -> models/{files[name]}")

ann = models["Neural Network (ANN)"].named_steps["model"]
print(f"ANN converged in {ann.n_iter_} epochs (early stopping)")

# Standalone scaler artifact: the fitted preprocessor the ANN uses (scaling + encoding)
joblib.dump(models["Neural Network (ANN)"].named_steps["prep"], MODELS / "standard_scaler.joblib")


# ---------------------------------------------------------------- 3. K-Means
kmeans = Pipeline([
    ("prep", ColumnTransformer([
        ("log_scale", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                ("scale", StandardScaler())]), ["wip", "over_time", "idle_time", "idle_men"]),
        ("scale", StandardScaler(), ["smv", "no_of_workers", "no_of_style_change"]),
    ])),
    ("model", KMeans(n_clusters=6, n_init=10, random_state=RANDOM_STATE)),
])
labels = kmeans.fit_predict(df[CLUSTER_FEATURES])
sil = silhouette_score(kmeans.named_steps["prep"].transform(df[CLUSTER_FEATURES]), labels)
joblib.dump(kmeans, MODELS / "kmeans_model.joblib")

prof = df.assign(cluster=labels).groupby("cluster").agg(
    n=(TARGET, "size"), rate=(TARGET, "mean"), sewing=("department", lambda s: (s == "sewing").mean()),
    idle_time=("idle_time", "mean"), idle_men=("idle_men", "mean"), over_time=("over_time", "mean"),
    style=("no_of_style_change", "mean"))


def name_cluster(r) -> str:
    if r.sewing == 0:
        return "Finishing teams"
    if r.idle_time > 50:
        return "Long stoppage"
    if r.idle_men > 5:
        return "Idle workers + style change"
    if r.over_time < 100:
        return "No overtime, high incentive"
    if r["style"] > 0.5:
        return "Sewing + style change"
    return "Stable sewing"


cluster_info = {int(c): {"name": name_cluster(r), "n": int(r.n), "miss_rate": float(r.rate)}
                for c, r in prof.iterrows()}
print(f"K-Means k=6 silhouette={sil:.3f} -> models/kmeans_model.joblib")
for c, v in cluster_info.items():
    print(f"   cluster {c}: {v['name']:28s} n={v['n']:4d}  miss rate={v['miss_rate']:.0%}")


# ---------------------------------------------------------------- 4. metadata
meta = {
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "sklearn_version": sklearn.__version__,
    "target": TARGET,
    "categorical": CATEGORICAL, "numeric": NUMERIC, "cluster_features": CLUSTER_FEATURES,
    "model_files": files, "kmeans_file": "kmeans_model.joblib", "scaler_file": "standard_scaler.joblib",
    "test_index": [int(i) for i in X_test.index],
    "thresholds": THRESHOLDS,
    "metrics_at_training": metrics,
    "kmeans": {"k": 6, "silhouette": float(sil), "clusters": cluster_info},
    "ann": {"hidden_layers": [32, 16], "epochs": int(ann.n_iter_)},
}
(MODELS / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Saved models/metadata.json")
