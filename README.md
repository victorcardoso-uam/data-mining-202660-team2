# Lean Bottleneck Analysis — Garment IMMEX (Team 2)

Clasificación del riesgo de incumplimiento de metas de productividad por equipo en una planta de confección IMMEX.
Data Mining & Modern AI Systems (IIND4417) · Universidad Anáhuac Mayab · Otoño 2026

**Cliente:** Gerente de Operaciones de Planta IMMEX · **Meta:** reducir 17.5 % de los ~904 eventos anuales de incumplimiento.

## Dataset
Productivity Prediction of Garment Employees (UCI, DOI 10.24432/C51S6D, CC BY 4.0) — 1,197 filas × 15 columnas.

## Estructura
```
data/raw/          CSV original (inmutable)
data/processed/    salidas limpias (se regeneran)
src/               pipeline en orden 01 → 04
reports/figures/   gráficas para la presentación
models/            modelo entrenado (.joblib)
```

## Cómo correrlo
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/01_data_cleaning.py
python src/02_kmeans_clustering.py
python src/03_decision_tree.py
python src/04_business_policy.py
```

## Resultados (Milestone 2)
| Modelo | Métrica | Criterio | Resultado |
|---|---|---|---|
| K-Means (k=6) | Silhouette | ≥ 0.35 | 0.719 |
| Árbol de decisión podado | F1-macro (test) | ≥ 0.70 | 0.742 |
| Árbol de decisión podado | ROC-AUC (test) | — | 0.789 |

## Prevención de fuga de datos
- `actual_productivity` define la variable objetivo, así que se excluye de los predictores.
- El split 80/20 estratificado se hace antes de ajustar cualquier transformación; el One-Hot vive dentro del `Pipeline`, así que en la CV se ajusta solo con los folds de entrenamiento.
- WIP de Finishing se imputa con 0 por regla de negocio, no con estadísticos del dataset completo.
