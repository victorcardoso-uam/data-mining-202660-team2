"""
04_business_policy.py
Traduce el risk_score del árbol (probabilidad de NO cumplir la meta) en un semáforo
operativo por equipo y estima el impacto anual para el gerente de planta IMMEX.

Uso:  python src/04_business_policy.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
pred = pd.read_csv(ROOT / "data" / "processed" / "test_predictions.csv")
FIG = ROOT / "reports" / "figures"
y, p = pred["missed_target"], pred["risk_score"]

# ---- 1. Barrido de umbrales -------------------------------------------------
rows = []
for t in np.arange(0.10, 0.91, 0.05):
    yhat = (p >= t).astype(int)
    rows.append({"umbral": round(t, 2),
                 "alertas_%": yhat.mean() * 100,
                 "recall": recall_score(y, yhat, zero_division=0),
                 "precision": precision_score(y, yhat, zero_division=0)})
sweep = pd.DataFrame(rows)
print(sweep.round(2).to_string(index=False))
sweep.to_csv(ROOT / "reports" / "threshold_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sweep["umbral"], sweep["recall"], "o-", label="Recall (incumplimientos detectados)")
ax.plot(sweep["umbral"], sweep["precision"], "s-", label="Precision (alertas correctas)")
ax.plot(sweep["umbral"], sweep["alertas_%"] / 100, "^--", color="gray",
        label="% de turnos con alerta")
for t, c in [(0.30, "#e6a700"), (0.60, "#c0504d")]:
    ax.axvline(t, color=c, ls=":", lw=2)
ax.set_xlabel("Umbral de risk score")
ax.set_ylabel("Proporción")
ax.set_title("Trade-off de umbrales para el semáforo")
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "policy_threshold_tradeoff.png", dpi=200)
plt.close(fig)

# ---- 2. Semáforo -------------------------------------------------------------
YELLOW, RED = 0.30, 0.60
pred["semaforo"] = pd.cut(p, [-0.01, YELLOW, RED, 1.01],
                          labels=["Verde", "Amarillo", "Rojo"])
light = pred.groupby("semaforo", observed=False).agg(
    turnos=("missed_target", "size"),
    incumplimientos=("missed_target", "sum"),
)
light["tasa_real_incumplimiento"] = light["incumplimientos"] / light["turnos"]
light["%_de_todos_los_incumplimientos"] = light["incumplimientos"] / y.sum()
print("\nSEMÁFORO EN TEST:\n", light.round(3).to_string())
light.to_csv(ROOT / "reports" / "traffic_light_test.csv")

# ---- 3. Impacto anual (fórmula de la propuesta) -----------------------------
ANNUAL_EVENTS = 904                      # 12 equipos x 280 días x 26.9 %
caught = light.loc[["Amarillo", "Rojo"], "incumplimientos"].sum() / y.sum()
TARGET_REDUCTION = 0.175                 # meta de la propuesta (15-20 %)
needed_effectiveness = TARGET_REDUCTION / caught
print(f"\nCobertura del semáforo (amarillo + rojo): {caught:.1%} de los incumplimientos")
print(f"Para lograr -17.5 % de eventos, las intervenciones deben corregir "
      f"{needed_effectiveness:.1%} de los casos alertados")
print(f"Eventos evitados por año (meta): {ANNUAL_EVENTS * TARGET_REDUCTION:.0f}")

# Sensibilidad al costo por evento (SUPUESTO: el equipo debe validar este valor)
for cost in [5_000, 10_000, 20_000]:
    print(f"  Si cada evento cuesta ${cost:,} MXN -> ahorro anual "
          f"${ANNUAL_EVENTS * TARGET_REDUCTION * cost:,.0f} MXN")


# Resumen del impacto para la diapositiva 2
impact = pd.DataFrame({"costo_por_evento_mxn": [5_000, 10_000, 20_000]})
impact["ahorro_anual_mxn"] = impact["costo_por_evento_mxn"] * ANNUAL_EVENTS * TARGET_REDUCTION
impact.to_csv(ROOT / "reports" / "impact_summary.csv", index=False)
print("\nGuardado: reports/impact_summary.csv")
