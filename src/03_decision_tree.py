"""
03_decision_tree.py
Pregunta de negocio: dado el estado operativo de un equipo, ¿alcanzará su meta hoy
y qué variable lo explica?

Pasos:
  1. Split estratificado 80/20 ANTES de cualquier ajuste (evita fuga de datos).
  2. Pipeline = ColumnTransformer (One-Hot para categóricas) + DecisionTreeClassifier.
     El encoder se ajusta solo con train dentro de cada fold de la validación cruzada.
  3. Árbol sin podar (línea base) para mostrar el sobreajuste.
  4. GridSearchCV 5-fold estratificado sobre max_depth, min_samples_leaf, ccp_alpha
     y class_weight, optimizando F1-macro.
  5. Evaluación en test: F1-macro, ROC-AUC, matriz de confusión, curvas ROC y P-R,
     importancia de variables y reglas IF-THEN legibles.

Anti-fuga: 'actual_productivity' define la variable objetivo -> se EXCLUYE de X.
'date' es un identificador temporal -> se excluye.

Uso:  python src/03_decision_tree.py
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
LEAKY = ["actual_productivity", "date"]  # nunca entran al modelo

df = pd.read_csv(DATA)
X = df[CATEGORICAL + NUMERIC]
y = df[TARGET]
assert not set(LEAKY) & set(X.columns), "Fuga de datos: columna prohibida en X"

# ---- 1. Split estratificado ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Tasa de incumplimiento  train={y_train.mean():.3f}  test={y_test.mean():.3f}\n")

# ---- 2. Pipeline ------------------------------------------------------------
# Los árboles no necesitan escalado; solo codificamos las categóricas.
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


# ---- 3. Línea base: árbol sin podar (sobreajuste) ----------------------------
base = make_pipe().fit(X_train, y_train)
base_tree = base.named_steps["tree"]
print("ÁRBOL SIN PODAR (línea base)")
print(f"  hojas={base_tree.get_n_leaves()}  profundidad={base_tree.get_depth()}")
print(f"  accuracy train={accuracy_score(y_train, base.predict(X_train)):.3f}  "
      f"test={accuracy_score(y_test, base.predict(X_test)):.3f}")
print(f"  F1-macro test={f1_score(y_test, base.predict(X_test), average='macro'):.3f}\n")

# ---- 4. Búsqueda de hiperparámetros con CV 5-fold ---------------------------
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
print("MEJORES HIPERPARÁMETROS (CV 5-fold, F1-macro):")
for k, v in search.best_params_.items():
    print(f"  {k.replace('tree__', '')} = {v}")
cv_row = search.cv_results_
i = search.best_index_
print(f"  F1-macro CV = {cv_row['mean_test_score'][i]:.3f} "
      f"± {cv_row['std_test_score'][i]:.3f}  (train {cv_row['mean_train_score'][i]:.3f})")
print(f"  hojas={best_tree.get_n_leaves()}  profundidad={best_tree.get_depth()}\n")

# ---- 5. Evaluación en test --------------------------------------------------
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy_train": accuracy_score(y_train, best.predict(X_train)),
    "accuracy_test": accuracy_score(y_test, y_pred),
    "f1_macro_test": f1_score(y_test, y_pred, average="macro"),
    "roc_auc_test": roc_auc_score(y_test, y_proba),
}
print("MÉTRICAS EN TEST (datos nunca vistos):")
for k, v in metrics.items():
    print(f"  {k} = {v:.3f}")
print("\n", classification_report(y_test, y_pred,
                                  target_names=["Cumple meta", "No cumple"]))
pd.Series(metrics).round(4).to_csv(ROOT / "reports" / "tree_metrics.csv")

# Matriz de confusión
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Cumple", "No cumple"], cmap="Blues", ax=ax)
ax.set_title("Matriz de confusión — test (n=240)")
ax.set_xlabel("Predicción del modelo")
ax.set_ylabel("Valor real")
fig.tight_layout()
fig.savefig(FIG / "tree_confusion_matrix.png", dpi=200)
plt.close(fig)

# ROC y Precision-Recall
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name="Árbol podado")
axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Azar")
axes[0].set_title("Curva ROC")
axes[0].set_xlabel("Tasa de falsos positivos")
axes[0].set_ylabel("Tasa de verdaderos positivos (Recall)")
axes[0].legend(loc="lower right")
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1],
                                        name="Árbol podado")
axes[1].axhline(y_test.mean(), ls="--", color="gray",
                label=f"Línea base ({y_test.mean():.2f})")
axes[1].set_title("Precision vs. Recall (clase 'No cumple')")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "tree_roc_pr_curves.png", dpi=200)
plt.close(fig)

# Sobreajuste: profundidad vs. F1 (train vs. validación)
res = pd.DataFrame(cv_row)
depth_curve = res.groupby("param_tree__max_depth")[["mean_train_score",
                                                    "mean_test_score"]].max()
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(depth_curve.index, depth_curve["mean_train_score"], "o-", label="Train")
ax.plot(depth_curve.index, depth_curve["mean_test_score"], "s-",
        label="Validación (CV)")
ax.set_xlabel("max_depth")
ax.set_ylabel("F1-macro")
ax.set_title("Brecha de generalización según la profundidad del árbol")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "tree_depth_validation_curve.png", dpi=200)
plt.close(fig)

# Importancia de variables
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best_tree.feature_importances_, index=names)
imp = imp[imp > 0].sort_values()
print("Importancia de variables:\n", imp.sort_values(ascending=False).round(3).to_string())
imp.sort_values(ascending=False).to_csv(ROOT / "reports" / "tree_feature_importance.csv")
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot.barh(ax=ax, color="#1f4e79")
ax.set_xlabel("Importancia (reducción de Gini)")
ax.set_title("¿Qué variables explican el incumplimiento?")
fig.tight_layout()
fig.savefig(FIG / "tree_feature_importance.png", dpi=200)
plt.close(fig)

# Árbol dibujado + reglas IF-THEN
# Solo los 3 primeros niveles: es lo que el supervisor lee como regla.
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(best_tree, feature_names=list(names), class_names=["Cumple", "No cumple"],
          filled=True, rounded=True, fontsize=10, impurity=False, proportion=True,
          max_depth=3, ax=ax)
ax.set_title("Árbol podado — primeros 3 niveles (reglas para el supervisor)")
fig.tight_layout()
fig.savefig(FIG / "tree_structure.png", dpi=200)
plt.close(fig)
rules = export_text(best_tree, feature_names=list(names), show_weights=True)
(ROOT / "reports" / "tree_rules.txt").write_text(rules, encoding="utf-8")
print("\nREGLAS DEL ÁRBOL:\n", rules)

# Guardar modelo y predicciones de test para la política de negocio
joblib.dump(best, MODELS / "decision_tree.joblib")
out = X_test.copy()
out["missed_target"] = y_test.values
out["risk_score"] = y_proba
out.to_csv(ROOT / "data" / "processed" / "test_predictions.csv", index=False)
print("Modelo guardado en models/decision_tree.joblib")
