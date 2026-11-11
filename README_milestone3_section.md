## Milestone 3 · Streamlit dashboard
```bash
pip install -r requirements.txt
python src/05_train_streamlit_models.py   # trains and saves models/*.joblib + data/team_dataset.csv
streamlit run app.py                      # opens http://localhost:8501
```
| Tab | Content |
|---|---|
| 📊 Data Explorer & EDA | sample records, summary statistics, histograms colored by `line_stoppage_flag` |
| ⚙️ Preprocessing & Correlation | WIP imputation, label fixes, serialized scaler (raw vs. scaled), One-Hot matrix, Pearson heatmap |
| 🤖 Multi-Model Benchmark | Decision Tree vs. Logistic Regression vs. Neural Network (ANN) + K-Means: metrics on 240 test records, confusion matrices, ROC curves, ANN loss curve |
| 🔮 Real-Time What-If | SMV, WIP, incentive and shift sliders → live probability, traffic-light policy alert and K-Means scenario |

Serialized artifacts in `models/`: `decision_tree_model.joblib`, `ann_mlp_model.joblib`, `logistic_regression_model.joblib`, `kmeans_model.joblib`, `standard_scaler.joblib`, `metadata.json`.
