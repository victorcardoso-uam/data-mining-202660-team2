# --- Library ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- team ---
CSV_FILE = "data/processed/pruning_team2_semiconductor.csv"
TARGET = "wafer_defect_flag"

df = pd.read_csv(CSV_FILE)

# Quick review of the data
print("Filas y columnas:", df.shape)
print(df.dtypes)
print(df.isna().sum())

# Quitar filas sin respuesta: no sirven para entrenar ni para evaluar
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])   # las 4 variables que usa el árbol
y = df[TARGET]                  # lo que queremos predecir

# Si alguna columna de X tiene texto, convertirla a números (0/1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,     # 25% se guarda para la prueba final
    random_state=42,    # misma división cada vez que corras el script
    stratify=y          # misma proporción de clases en ambos grupos
)

# Rellenar datos faltantes con la mediana DEL ENTRENAMIENTO
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

ALPHAS = {
    "alpha_1 (Unconstrained)": 0.001,
    "alpha_2 (Optimal Pruned)": 0.015,
    "alpha_3 (Over-Pruned)": 0.080,
}
rows = []
for name, alpha in ALPHAS.items():
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)

    leaves = clf.get_n_leaves()                     # |T|: número de hojas
    depth = clf.get_depth()                         # D_max: niveles de preguntas
    r_train = 1.0 - clf.score(X_train, y_train)     # error en entrenamiento
    r_test = 1.0 - clf.score(X_test, y_test)        # error en prueba
    total_cost = r_test + alpha * leaves            # R_alpha(T), paso 5

    rows.append({
        "Candidate": name,
        "ccp_alpha": alpha,
        "Leaves |T|": leaves,
        "Max Depth": depth,
        "Train Error %": round(r_train * 100, 2),
        "Test Error %": round(r_test * 100, 2),
        "Total Cost": round(total_cost, 4),
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv("activity_13_results.csv", index=False)   # log de salida

best_cost = results.loc[results["Total Cost"].idxmin()]
best_test = results.loc[results["Test Error %"].idxmin()]
print("\nMinimum total cost:", best_cost["Candidate"], "->", best_cost["Total Cost"])
print("Lowest test error: ", best_test["Candidate"], "->", best_test["Test Error %"], "%")