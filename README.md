# Lean Bottleneck Analysis (Garment IMMEX, Team 2)

Classification of the risk of missing productivity targets by team in an IMMEX garment plant.
Data Mining & Modern AI Systems (IIND4417) · Universidad Anáhuac Mayab · Fall 2026

**Client.** Plant Operations Manager at an IMMEX facility · **Goal.** Reduce 17.5% of the ~904 yearly non-compliance events.

## Dataset
Productivity Prediction of Garment Employees (UCI, DOI 10.24432/C51S6D, CC BY 4.0), 1,197 rows × 15 columns.

## Structure
```
data/raw/          original CSV (immutable)
data/processed/    clean outputs (regenerated)
src/               pipeline in order 01 → 04
reports/figures/   charts for the presentation
models/            trained model (.joblib)
```

## How to run it
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/01_data_cleaning.py
python src/02_kmeans_clustering.py
python src/03_decision_tree.py
python src/04_business_policy.py
```

## Results (Milestone 2)
| Model | Metric | Criterion | Result |
|---|---|---|---|
| K-Means (k=6) | Silhouette | ≥ 0.35 | 0.719 |
| Pruned decision tree | F1-macro (test) | ≥ 0.70 | 0.742 |
| Pruned decision tree | ROC-AUC (test) | n/a | 0.789 |

## Data leakage prevention
- `actual_productivity` defines the target variable, so it is excluded from the predictors.
- The stratified 80/20 split is done before fitting any transformation. One-Hot encoding lives inside the `Pipeline`, so during cross-validation it is fitted only on the training folds.
- Finishing WIP is imputed with 0 as a business rule, not with statistics computed from the full dataset.

## Team

| Team member | Role in Milestone 2 |
| --- | --- |
| Brenda Elisa Cabrera Cruz | Documentation and code review |
| Valeria Hurtado Ruiz | Pipeline integration |
| Jorge David López Almanza | K-Means clustering |
| Mauricio Lugo Sánchez | Decision tree |
| Oliver Edier Pacheco Ku | Business policy and impact |
| Héctor Ramírez Lerma | Repository hygiene |
