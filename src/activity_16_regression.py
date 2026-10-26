import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure headless execution for saving plots safely
plt.switch_backend('Agg')

# Dynamic path resolution to guarantee the dataset file is found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'regression_team2_semiconductor.csv')
)

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    df = pd.read_csv('regression_team2_semiconductor.csv')

y = df['wafer_yield_pct']
X = df.drop(columns=['wafer_yield_pct'])

print("=" * 65)
print(" EXECUTIVE DATA INGESTION & EXPLORATION REPORT (TEAM 2)")
print("=" * 65)
print(f"Dataset File Path      : data/processed/regression_team2_semiconductor.csv")
print(f"Total Telemetry Rows   : {df.shape[0]}")
print(f"Total Predictor Pool   : {X.shape[1]} features")
print(f"Target Variable (y)    : wafer_yield_pct (Continuous % Yield)")
print(f"Missing Values Checked : {df.isnull().sum().sum()} null cells found\n")

print("--- PREDICTOR FEATURES POOL (X) ---")
for i, col in enumerate(X.columns, 1):
    print(f" {i}. {col}")

print("\n--- SUMMARY STATISTICS OF TARGET (wafer_yield_pct) ---")
print(f" Mean Yield : {y.mean():.2f}%")
print(f" Min Yield  : {y.min():.2f}%")
print(f" Max Yield  : {y.max():.2f}%")
print(f" Std Dev    : {y.std():.2f}%")

# Step 2: 80/20 Train/Test Split with mandatory random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\n" + "=" * 65)
print(" TRAIN-TEST SPLIT BENCHMARKING GATE (80/20)")
print("=" * 65)
print(f"Training Set (X_train): {X_train.shape[0]} records (80%)")
print(f"Testing Set (X_test)  : {X_test.shape[0]} records (20%)")
print(f"Mandatory Random State: 42 (Guaranteed Benchmark Consistency)")
print("=" * 65 + "\n")

# Step 3: Candidate Models Feature Definition
feature_subsets = {
    "Model 1": ["rf_power_watts"],
    "Model 2": ["rf_power_watts", "chamber_pressure_torr"],
    "Model 3": ["rf_power_watts", "chamber_pressure_torr", "etching_time_sec"],
    "Model 4": ["rf_power_watts", "chamber_pressure_torr", "etching_time_sec", "gas_flow_sccm"],
    "Model 5": ["rf_power_watts", "chamber_pressure_torr", "etching_time_sec", "gas_flow_sccm", "wafer_temp_c"]
}

fitted_models = {}
predictions = {}

print("=" * 65)
print(" STEP 3: FITTING ALL 5 CANDIDATE REGRESSION MODELS")
print("=" * 65)

for model_name, features in feature_subsets.items():
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    
    y_train_pred = model.predict(X_train[features])
    y_test_pred = model.predict(X_test[features])
    
    fitted_models[model_name] = model
    predictions[model_name] = {
        "train_pred": y_train_pred,
        "test_pred": y_test_pred,
        "features": features
    }
    print(f" Successfully fitted: {model_name} ({len(features)} features)")

# Step 4: Compute Fit Statistics & Error Metrics
metrics_results = []
n_test = len(y_test)

for model_name, data in predictions.items():
    features = data["features"]
    p = len(features)
    
    y_train_pred = data["train_pred"]
    y_test_pred = data["test_pred"]
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Adjusted R2 Formula calculation
    adj_r2 = 1 - ((1 - test_r2) * (n_test - 1) / (n_test - p - 1))
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    metrics_results.append({
        "Model": model_name,
        "Features Count": p,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Adjusted R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse
    })

df_metrics = pd.DataFrame(metrics_results)

print("\n" + "=" * 65)
print(" STEP 4: 5-MODEL BENCHMARK COMPARISON TABLE")
print("=" * 65)
for _, row in df_metrics.iterrows():
    print(f" {row['Model']:<10} | Features: {row['Features Count']:<2} | Train R2: {row['Train R2']:.4f} | Test R2: {row['Test R2']:.4f} | Adj R2: {row['Adjusted R2']:.4f} | MAE: {row['MAE']:.4f} | RMSE: {row['RMSE']:.4f}")
print("=" * 65 + "\n")

# Step 5: Visual Residual Diagnostics (Health Checks)
FIGURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'reports', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

top_models = ["Model 4", "Model 5"]

print("=" * 65)
print(" STEP 5: GENERATING RESIDUAL DIAGNOSTIC DASHBOARDS")
print("=" * 65)

for model_name in top_models:
    y_pred = predictions[model_name]["test_pred"]
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Fitted Values vs. Residuals Scatter Plot
    axes[0].scatter(y_pred, residuals, alpha=0.7, color='navy', edgecolors='k')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'{model_name}: Fitted Values vs Residuals', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Fitted Values (Predicted Wafer Yield %)', fontsize=10)
    axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Panel B: Residual Distribution Histogram & Normal Curve
    axes[1].hist(residuals, bins=15, density=True, alpha=0.6, color='teal', edgecolor='black')
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = axes[1].get_xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    p_axis = stats.norm.pdf(x_axis, mu, std)
    axes[1].plot(x_axis, p_axis, 'r', linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')
    axes[1].set_title(f'{model_name}: Residual Error Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residual Error', fontsize=10)
    axes[1].set_ylabel('Density', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, f'residual_diagnostics_{model_name.lower().replace(" ", "")}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    
    print(f" Saved diagnostic dashboard: {plot_path}")

print("\n--- STEP 5 EXECUTION COMPLETE: ALL RESIDUAL DASHBOARDS GENERATED ---")
