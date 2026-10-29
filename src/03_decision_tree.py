"""
03_decision_tree.py
Business question. Given the operating state of a team, will it reach its target today
and which variable explains it?

The script follows these steps.
  1. Stratified 80/20 split BEFORE any fitting (prevents data leakage).
  2. Pipeline = ColumnTransformer (One-Hot for categorical variables) + DecisionTreeClassifier.
     The encoder is fitted only on the training data inside each cross-validation fold.
  3. Unpruned tree (baseline) to show the overfitting.
  4. Stratified 5-fold GridSearchCV over max_depth, min_samples_leaf, ccp_alpha
     and class_weight, optimizing F1-macro.
  5. Test evaluation with F1-macro, ROC-AUC, confusion matrix, ROC and P-R curves,
     feature importance and readable IF-THEN rules.

Leakage note. 'actual_productivity' defines the target variable, so it is EXCLUDED from X.
'date' is a time identifier, so it is also excluded.

Run it with  python src/03_decision_tree.py
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "garments_clean.csv"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FIG.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42

CATEGORICAL = ["department", "day", "quarter", "team"]
NUMERIC = ["targeted_productivity", "smv", "wip", "over_time", "incentive",
           "idle_time", "idle_men", "no_of_style_change", "no_of_workers"]
TARGET = "missed_target"
LEAKY = ["actual_productivity", "date"]  # these never enter the model

df = pd.read_csv(DATA)
X = df[CATEGORICAL + NUMERIC]
y = df[TARGET]
assert not set(LEAKY) & set(X.columns), "Data leakage, forbidden column in X"

# ---- 1. Stratified split --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Train shape {X_train.shape} | Test shape {X_test.shape}")
print(f"Non-compliance rate, train={y_train.mean():.3f}  test={y_test.mean():.3f}\n")

# ---- 2. Pipeline ------------------------------------------------------------
# Trees do not need scaling, we only encode the categorical variables.
prep = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     CATEGORICAL),
    ("num", "passthrough", NUMERIC),
], verbose_feature_names_out=False)


def make_pipe(**tree_params):
    return Pipeline([
        ("prep", prep),
        ("tree", DecisionTreeClassifier(random_state=RANDOM_STATE, **tree_params)),
    ])


# ---- 3. Baseline, unpruned tree (overfitting) -------------------------------
base = make_pipe().fit(X_train, y_train)
base_tree = base.named_steps["tree"]
print("UNPRUNED TREE (baseline)")
print(f"  leaves={base_tree.get_n_leaves()}  depth={base_tree.get_depth()}")
print(f"  accuracy train={accuracy_score(y_train, base.predict(X_train)):.3f}  "
      f"test={accuracy_score(y_test, base.predict(X_test)):.3f}")
print(f"  F1-macro test={f1_score(y_test, base.predict(X_test), average='macro'):.3f}\n")

# ---- 4. Hyperparameter search with 5-fold CV --------------------------------
alphas = base_tree.cost_complexity_pruning_path(
    base.named_steps["prep"].transform(X_train), y_train).ccp_alphas
alpha_grid = np.unique(np.round(np.quantile(alphas, np.linspace(0, 0.95, 12)), 5))

grid = {
    "tree__max_depth": [3, 4, 5, 6, 8],
    "tree__min_samples_leaf": [5, 10, 20, 30],
    "tree__ccp_alpha": alpha_grid,
    "tree__class_weight": [None, "balanced"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = GridSearchCV(make_pipe(), grid, scoring="f1_macro", cv=cv, n_jobs=-1,
                      return_train_score=True)
search.fit(X_train, y_train)
best = search.best_estimator_
best_tree = best.named_steps["tree"]
print("BEST HYPERPARAMETERS (5-fold CV, F1-macro)")
for k, v in search.best_params_.items():
    print(f"  {k.replace('tree__', '')} = {v}")
cv_row = search.cv_results_
i = search.best_index_
print(f"  F1-macro CV = {cv_row['mean_test_score'][i]:.3f} "
      f"± {cv_row['std_test_score'][i]:.3f}  (train {cv_row['mean_train_score'][i]:.3f})")
print(f"  leaves={best_tree.get_n_leaves()}  depth={best_tree.get_depth()}\n")

# ---- 5. Test evaluation -----------------------------------------------------
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy_train": accuracy_score(y_train, best.predict(X_train)),
    "accuracy_test": accuracy_score(y_test, y_pred),
    "f1_macro_test": f1_score(y_test, y_pred, average="macro"),
    "roc_auc_test": roc_auc_score(y_test, y_proba),
}
print("TEST METRICS (never seen data)")
for k, v in metrics.items():
    print(f"  {k} = {v:.3f}")
print("\n", classification_report(y_test, y_pred,
                                  target_names=["Meets target", "Misses target"]))
pd.Series(metrics).round(4).to_csv(ROOT / "reports" / "tree_metrics.csv")

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Meets", "Misses"], cmap="Blues", ax=ax)
ax.set_title("Confusion matrix, test set (n=240)")
ax.set_xlabel("Model prediction")
ax.set_ylabel("Actual value")
fig.tight_layout()
fig.savefig(FIG / "tree_confusion_matrix.png", dpi=200)
plt.close(fig)

# ROC and Precision-Recall
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name="Pruned tree")
axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
axes[0].set_title("ROC curve")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate (Recall)")
axes[0].legend(loc="lower right")
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1],
                                        name="Pruned tree")
axes[1].axhline(y_test.mean(), ls="--", color="gray",
                label=f"Baseline ({y_test.mean():.2f})")
axes[1].set_title("Precision vs. Recall (class 'Misses target')")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "tree_roc_pr_curves.png", dpi=200)
plt.close(fig)

# Overfitting, depth vs. F1 (train vs. validation)
res = pd.DataFrame(cv_row)
depth_curve = res.groupby("param_tree__max_depth")[["mean_train_score",
                                                    "mean_test_score"]].max()
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(depth_curve.index, depth_curve["mean_train_score"], "o-", label="Train")
ax.plot(depth_curve.index, depth_curve["mean_test_score"], "s-",
        label="Validation (CV)")
ax.set_xlabel("max_depth")
ax.set_ylabel("F1-macro")
ax.set_title("Generalization gap by tree depth")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "tree_depth_validation_curve.png", dpi=200)
plt.close(fig)

# Feature importance
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best_tree.feature_importances_, index=names)
imp = imp[imp > 0].sort_values()
print("Feature importance\n", imp.sort_values(ascending=False).round(3).to_string())
imp.sort_values(ascending=False).to_csv(ROOT / "reports" / "tree_feature_importance.csv")
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot.barh(ax=ax, color="#1f4e79")
imp.plot.barh(ax=ax, color="#1f4e79")
ax.set_xlabel("Importance (Gini reduction)")
ax.set_title("Which variables explain non-compliance?")
fig.tight_layout()
fig.savefig(FIG / "tree_feature_importance.png", dpi=200)
plt.close(fig)

# Tree plot and IF-THEN rules
# Only the first 3 levels, since that is what the supervisor reads as a rule.
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(best_tree, feature_names=list(names), class_names=["Meets", "Misses"],
          filled=True, rounded=True, fontsize=10, impurity=False, proportion=True,
          max_depth=3, ax=ax)
ax.set_title("Pruned tree, first 3 levels (rules for the supervisor)")
fig.tight_layout()
fig.savefig(FIG / "tree_structure.png", dpi=200)
plt.close(fig)
rules = export_text(best_tree, feature_names=list(names), show_weights=True)
(ROOT / "reports" / "tree_rules.txt").write_text(rules, encoding="utf-8")
print("\nTREE RULES\n", rules)

# Save the model and the test predictions for the business policy
joblib.dump(best, MODELS / "decision_tree.joblib")
out = X_test.copy()
out["missed_target"] = y_test.values
out["risk_score"] = y_proba
out.to_csv(ROOT / "data" / "processed" / "test_predictions.csv", index=False)
print("Model saved to models/decision_tree.joblib")
