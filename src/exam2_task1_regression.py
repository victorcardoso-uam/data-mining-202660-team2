from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.stattools import durbin_watson

# 1. Load dataset (80% train / 20% test)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "exam2_team2_semiconductor.csv"
FIG_PATH = ROOT / "reports" / "figures" / "exam2_task1_residual_diagnostics.png"
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = "etch_rate_nm_min"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

def adjusted_r2(r2: float, n: int, p: int) -> float:
    return 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))

n_test = len(y_test)
p_orig = X_test.shape[1]

# Model 1: Baseline OLS Linear
ols_base = LinearRegression()
ols_base.fit(X_train, y_train)
pred_base = ols_base.predict(X_test)

# Model 2: Log-Transformed Target OLS (z = ln(y + 1), back-transformed: exp(z) - 1)
ols_log = LinearRegression()
ols_log.fit(X_train, np.log1p(y_train))
pred_log = np.expm1(ols_log.predict(X_test))

# Model 3: Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
ols_poly = LinearRegression()
ols_poly.fit(X_train_poly, y_train)
pred_poly = ols_poly.predict(X_test_poly)

# Metrics Comparison Table
results = []
models_info = [
    ("1. Baseline OLS Linear", pred_base, p_orig),
    ("2. Log-Transformed Target OLS", pred_log, p_orig),
    ("3. Polynomial Regression (Deg 2)", pred_poly, X_test_poly.shape[1]),
]

for name, preds, p_count in models_info:
    r2 = r2_score(y_test, preds)
    adj = adjusted_r2(r2, n_test, p_count)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append({
        "Model": name,
        "R2": round(r2, 4),
        "Adjusted_R2": round(adj, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
    })

results_df = pd.DataFrame(results)
print("\n=== TASK 1: HOLDOUT TEST SET REGRESSION COMPARISON ===")
print(results_df.to_string(index=False))

# 4-Panel Classical OLS Residual Health Checks (using best OLS model: Baseline OLS)
residuals = y_test.values - pred_base
std_residuals = (residuals - residuals.mean()) / residuals.std(ddof=1)
dw_stat = durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic (Baseline OLS Residuals): {dw_stat:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Task 1: 4-Panel Classical OLS Residual Diagnostics (Baseline OLS - Semiconductor Etch Rate)",
    fontsize=13,
    fontweight="bold",
)

# Panel 1: Linearity (Residuals vs Fitted)
axes[0, 0].scatter(pred_base, residuals, alpha=0.65, color="#1f77b4", edgecolors="k", linewidth=0.4)
axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.3)
axes[0, 0].set_title("1. Linearity: Residuals vs. Fitted Values")
axes[0, 0].set_xlabel("Fitted Etch Rate (nm/min)")
axes[0, 0].set_ylabel("Residuals (nm/min)")
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Homoscedasticity (Scale-Location: sqrt(|Standardized Residuals|) vs Fitted)
axes[0, 1].scatter(pred_base, np.sqrt(np.abs(std_residuals)), alpha=0.65, color="#2ca02c", edgecolors="k", linewidth=0.4)
axes[0, 1].set_title("2. Homoscedasticity: Scale-Location Plot")
axes[0, 1].set_xlabel("Fitted Etch Rate (nm/min)")
axes[0, 1].set_ylabel(r"$\sqrt{|Standardized\ Residuals|}$")
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Independence (Residuals vs Observation Order + Durbin-Watson)
axes[1, 0].plot(range(len(residuals)), residuals, marker="o", markersize=4, linestyle="-", alpha=0.7, color="#9467bd")
axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=1.3)
axes[1, 0].set_title(f"3. Independence: Residuals vs. Order (DW = {dw_stat:.2f})")
axes[1, 0].set_xlabel("Test Observation Index")
axes[1, 0].set_ylabel("Residuals (nm/min)")
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Normality (Normal Q-Q Plot)
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("4. Normality: Normal Q-Q Plot of Residuals")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG_PATH, dpi=300)
print(f"\nSaved 4-panel diagnostic figure to: {FIG_PATH}")
plt.show()
