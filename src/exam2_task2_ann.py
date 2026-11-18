from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Paths & Dataset Loading (80/20 split matching Task 1)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "exam2_team2_semiconductor.csv"
MODELS_DIR = ROOT / "models"
FIG_PATH = ROOT / "reports" / "figures" / "exam2_task2_ann_loss_curve.png"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = "etch_rate_nm_min"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. Mandatory StandardScaler Normalization (Mean=0, Std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit MLPRegressor with 2 hidden layers (64, 32), ReLU, and early_stopping=True
ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)
ann_model.fit(X_train_scaled, y_train)

# 4. Evaluate on Holdout Test Set
y_pred = ann_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== TASK 2: ARTIFICIAL NEURAL NETWORK (MLPRegressor) RESULTS ===")
print(f"Architecture        : Hidden Layers (64, 32) | Activation: ReLU | Early Stopping: True")
print(f"Epochs Converged    : {ann_model.n_iter_}")
print(f"Best Validation Score (R2): {ann_model.best_validation_score_:.4f}")
print(f"Holdout Test R2     : {r2:.4f}")
print(f"Holdout Test MAE    : {mae:.4f} nm/min")
print(f"Holdout Test RMSE   : {rmse:.4f} nm/min")

# 5. Serialize Winning Model & Scaler into /models
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

joblib.dump(ann_model, BEST_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nSerialized artifacts saved to /models:")
print(f" -> {BEST_MODEL_PATH}")
print(f" -> {SCALER_PATH}")

# 6. Plot Training Loss Convergence Curve (loss_curve_) across Epochs
plt.figure(figsize=(9, 5))
plt.plot(
    range(1, len(ann_model.loss_curve_) + 1),
    ann_model.loss_curve_,
    color="#E8641C",
    linewidth=2.2,
    label="Training Loss (MSE / 2)",
)
plt.title(
    "Task 2: ANN Training Loss Convergence Curve (MLPRegressor 64-32, ReLU, Early Stopping)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Training Epochs", fontsize=11)
plt.ylabel("Training Loss", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
print(f"Saved loss convergence plot to: {FIG_PATH}")
plt.show()
