"""
Garment IMMEX · Line Stoppage Risk Dashboard (Team 2 · Milestone 3)
Streamlit app that loads the serialized models trained by src/05_train_streamlit_models.py.

Run from the repository root:
    python src/05_train_streamlit_models.py   # once, to create models/*.joblib
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "team_dataset.csv"
RAW = ROOT / "data" / "raw" / "garments_worker_productivity.csv"
MODELS = ROOT / "models"
TARGET = "line_stoppage_flag"
COLORS = {0: "#2E9E5B", 1: "#E8641C"}
LABELS = {0: "Meets target (0)", 1: "Line stoppage risk (1)"}

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(page_title="Garment IMMEX · Line Stoppage Risk", page_icon="🧵",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .main-header {font-size: 2.1rem; font-weight: 700; margin-bottom: .1rem;}
  .sub-header {font-size: 1.02rem; color: #ADB5BD; margin-bottom: 1.2rem;}
  div[data-testid="stMetric"] {background: #1E2125; border: 1px solid #343A40;
                               border-radius: 10px; padding: 12px 16px;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. CACHED DATA & SERIALIZED MODEL LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Clean team dataset written by the training script (target = line_stoppage_flag)."""
    return pd.read_csv(DATA)


@st.cache_data
def load_raw():
    """Original UCI file, used only to show the cleaning steps in Tab 2."""
    return pd.read_csv(RAW) if RAW.exists() else None


@st.cache_resource
def load_artifacts():
    """Loads every .joblib artifact once per server session."""
    meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    models = {name: joblib.load(MODELS / f) for name, f in meta["model_files"].items()}
    kmeans = joblib.load(MODELS / meta["kmeans_file"])
    scaler = joblib.load(MODELS / meta["scaler_file"])
    return meta, models, kmeans, scaler


if not (DATA.exists() and (MODELS / "metadata.json").exists()):
    st.error("Models not found. From the repository root run:  python src/05_train_streamlit_models.py")
    st.stop()

df = load_data()
meta, models, kmeans, scaler = load_artifacts()
FEATURES = meta["categorical"] + meta["numeric"]
test_df = df.loc[meta["test_index"]]
X_test, y_test = test_df[FEATURES], test_df[TARGET]


@st.cache_data
def test_probabilities():
    """Probabilities of every loaded model on the 240 held-out test records."""
    return {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}


@st.cache_data
def benchmark(threshold: float) -> pd.DataFrame:
    rows = []
    for name, p in test_probabilities().items():
        pred = (p > threshold).astype(int)
        rows.append({"Model": name, "Family": "Neural network" if "ANN" in name else "Classical",
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred, zero_division=0),
                     "F1 (macro)": f1_score(y_test, pred, average="macro"),
                     "ROC-AUC": roc_auc_score(y_test, p),
                     "R²": np.nan, "Silhouette": np.nan})
    rows.append({"Model": "K-Means (k = 6)", "Family": "Unsupervised", "Accuracy": np.nan,
                 "Precision": np.nan, "Recall": np.nan, "F1 (macro)": np.nan, "ROC-AUC": np.nan,
                 "R²": np.nan, "Silhouette": meta["kmeans"]["silhouette"]})
    return pd.DataFrame(rows)


# =============================================================================
# 3. SIDEBAR CONTROLS (stored in st.session_state through widget keys)
# =============================================================================
st.sidebar.title("🧵 Garment IMMEX")
st.sidebar.caption("Team 2 · Data Mining & Modern AI Systems (IIND4417)")
st.sidebar.divider()
st.sidebar.subheader("🎛️ Inference settings")
active_model = st.sidebar.selectbox("Active model", list(models.keys()), index=0, key="active_model")
threshold = st.sidebar.slider("Classification cutoff (benchmark)", 0.05, 0.95, 0.50, 0.05, key="threshold")
yellow, red = st.sidebar.slider("Traffic light cutoffs (yellow, red)", 0.05, 0.95,
                                (meta["thresholds"]["yellow"], meta["thresholds"]["red"]), 0.05,
                                key="cutoffs")
st.sidebar.divider()
st.sidebar.subheader("💰 Business case")
cost_event = st.sidebar.number_input("Cost per line-stoppage event (MXN)", 1000, 100000, 10000, 1000,
                                     key="cost_event")
st.sidebar.caption(f"Models trained {meta['trained_at']} · scikit-learn {meta['sklearn_version']}")

# =============================================================================
# 4. HEADER & TOP KPI CARDS
# =============================================================================
st.markdown('<div class="main-header">🧵 Line Stoppage Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IMMEX Maquiladora Plant Operations Manager · '
            'Will this team miss its productivity target today?</div>', unsafe_allow_html=True)

bench = benchmark(threshold)
active_row = bench[bench["Model"] == active_model].iloc[0]
rate = df[TARGET].mean()
events = round(12 * 280 * rate)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Team-day records", f"{len(df):,}", "12 teams · Jan–Mar 2015")
k2.metric("Line stoppage rate", f"{rate:.1%}", f"≈ {events:,} events / year", delta_color="inverse")
k3.metric("Savings at −17.5%", f"MXN {events * 0.175 * cost_event / 1e6:,.2f} M",
          f"{round(events * 0.175)} events avoided")
k4.metric(f"{active_model} · ROC-AUC", f"{active_row['ROC-AUC']:.3f}",
          f"F1 {active_row['F1 (macro)']:.3f} on 240 test records")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer & EDA", "⚙️ Preprocessing & Correlation",
                                  "🤖 Multi-Model Benchmark", "🔮 Real-Time What-If Inference"])

# =============================================================================
# TAB 1: DATA EXPLORER & EDA
# =============================================================================
with tab1:
    st.subheader("Team-day production records")
    depts = st.multiselect("Department", sorted(df["department"].unique()),
                           default=sorted(df["department"].unique()), key="eda_depts")
    view = df[df["department"].isin(depts)]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(view.head(200), use_container_width=True, height=330)
    with c2:
        st.write("**Summary statistics**")
        st.dataframe(view[meta["numeric"] + ["actual_productivity"]].describe().T[["mean", "std", "min", "50%", "max"]]
                     .round(2), use_container_width=True, height=330)
    st.divider()
    feat = st.selectbox("Feature to explore", meta["numeric"], index=meta["numeric"].index("smv"), key="eda_feat")
    plot_df = view.assign(status=view[TARGET].map(LABELS))
    fig = px.histogram(plot_df, x=feat, color="status", barmode="overlay", nbins=40, opacity=.75,
                       color_discrete_map={LABELS[0]: COLORS[0], LABELS[1]: COLORS[1]},
                       title=f"Distribution of {feat} by line stoppage status")
    fig.update_layout(legend_title_text="", yaxis_title="Records")
    st.plotly_chart(fig, use_container_width=True)
    by_dept = view.groupby("department")[TARGET].mean().mul(100).round(1).rename("% line stoppage risk")
    st.caption("Share of records below target by department: " +
               " · ".join(f"{d} {v}%" for d, v in by_dept.items()))

# =============================================================================
# TAB 2: PREPROCESSING & CORRELATION
# =============================================================================
with tab2:
    st.subheader("Cleaning steps applied before training")
    raw = load_raw()
    a, b, c = st.columns(3)
    if raw is not None:
        a.metric("WIP nulls (raw → clean)", f"{int(raw['wip'].isna().sum())} → {int(df['wip'].isna().sum())}",
                 "all in Finishing → imputed 0")
        b.metric("Department labels (raw → clean)", f"{raw['department'].nunique()} → {df['department'].nunique()}",
                 "'sweing' and 'finishing ' fixed")
    c.metric("Leaky columns excluded", "2", "actual_productivity, date")
    st.info("WIP is imputed with **0**, not the median: every null belongs to Finishing, where garments are "
            "already assembled. The split (80/20, stratified) happens **before** any scaler or encoder is fitted, "
            "and both live inside the saved Pipelines.")

    st.divider()
    st.subheader("Feature scaling with the serialized scaler")
    names = list(scaler.get_feature_names_out())
    num_feat = st.selectbox("Numeric feature", meta["numeric"], index=meta["numeric"].index("over_time"),
                            key="scale_feat")
    col_name = next(n for n in names if n.split("__")[-1] == num_feat)
    scaled = pd.DataFrame(scaler.transform(df[FEATURES]), columns=names)
    s1, s2 = st.columns(2)
    s1.plotly_chart(px.histogram(df, x=num_feat, nbins=40, title=f"Raw {num_feat}",
                                 color_discrete_sequence=["#5B8DEF"]), use_container_width=True)
    s2.plotly_chart(px.histogram(scaled, x=col_name, nbins=40,
                                 title=f"Scaled ({'log1p + ' if col_name.startswith('log_scale') else ''}StandardScaler)",
                                 color_discrete_sequence=["#E8641C"]), use_container_width=True)
    st.caption(f"After scaling: mean = {scaled[col_name].mean():.2f}, std = {scaled[col_name].std():.2f}. "
               f"The encoded feature matrix has {len(names)} columns "
               f"({sum(n.startswith('onehot') for n in names)} One-Hot + {len(meta['numeric'])} numeric).")
    with st.expander("See the encoded feature matrix (first 10 rows)"):
        st.dataframe(scaled.head(10).round(2), use_container_width=True)

    st.divider()
    st.subheader("Pearson correlation matrix")
    corr = df[meta["numeric"] + [TARGET]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Numeric features vs. line_stoppage_flag")
    st.plotly_chart(fig_corr, use_container_width=True)

# =============================================================================
# TAB 3: MULTI-MODEL BENCHMARK
# =============================================================================
with tab3:
    st.subheader("Classical models vs. Neural Network · 240 unseen test records")
    st.dataframe(bench.style.format(precision=3, na_rep="—"), use_container_width=True, hide_index=True)
    st.caption(f"Classification cutoff = {threshold:.2f} (sidebar). R² does not apply: the task is "
               "classification, not regression. Silhouette applies only to K-Means.")

    probs = test_probabilities()
    st.subheader("Confusion matrices")
    cols = st.columns(len(models))
    for col, (name, p) in zip(cols, probs.items()):
        cm = confusion_matrix(y_test, (p > threshold).astype(int))
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Oranges",
                           x=["Pred. meets", "Pred. stoppage"], y=["Actual meets", "Actual stoppage"],
                           title=name)
        fig_cm.update_layout(coloraxis_showscale=False, height=330, margin=dict(t=50, b=10))
        col.plotly_chart(fig_cm, use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        fig_roc = go.Figure()
        for name, p in probs.items():
            fpr, tpr, _ = roc_curve(y_test, p)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{name} (AUC {roc_auc_score(y_test, p):.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                     line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title="ROC curves", xaxis_title="False positive rate",
                              yaxis_title="True positive rate", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_roc, use_container_width=True)
    with r2:
        ann = models["Neural Network (ANN)"].named_steps["model"]
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=ann.loss_curve_, mode="lines", name="Training loss (log-loss)"))
        if getattr(ann, "validation_scores_", None) is not None:
            fig_loss.add_trace(go.Scatter(y=ann.validation_scores_, mode="lines", yaxis="y2",
                                          name="Validation accuracy"))
        fig_loss.update_layout(title=f"ANN loss convergence · {ann.n_iter_} epochs (early stopping)",
                               xaxis_title="Epoch", yaxis_title="Log-loss",
                               yaxis2=dict(title="Validation accuracy", overlaying="y", side="right"),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_loss, use_container_width=True)
    st.caption("Reading the loss curve: the training loss drops fast and flattens; early stopping ends training "
               "when validation accuracy stops improving for 25 epochs, which prevents overfitting.")

    st.subheader("K-Means: bottleneck scenarios")
    kinfo = pd.DataFrame(meta["kmeans"]["clusters"]).T
    kinfo["miss_rate"] = kinfo["miss_rate"].astype(float) * 100
    fig_k = px.bar(kinfo.sort_values("miss_rate"), x="miss_rate", y="name", orientation="h", text_auto=".0f",
                   color_discrete_sequence=["#E8641C"],
                   title=f"% line stoppage risk by scenario · Silhouette {meta['kmeans']['silhouette']:.3f}")
    fig_k.update_layout(xaxis_title="% of records below target", yaxis_title="")
    st.plotly_chart(fig_k, use_container_width=True)

# =============================================================================
# TAB 4: REAL-TIME WHAT-IF INFERENCE
# =============================================================================
NORMAL = dict(w_department="sewing", w_team=4, w_day="Wednesday", w_quarter="Quarter2", w_target=0.80,
              w_smv=22.5, w_wip=1100, w_over_time=6900, w_incentive=50, w_workers=56,
              w_idle_time=0, w_idle_men=0, w_style=0)
HIGH_RISK = dict(NORMAL, w_incentive=0, w_wip=600)


def load_preset(preset: dict):
    for k, v in preset.items():
        st.session_state[k] = v


def on_department_change():
    # Business rule from the cleaning step: Finishing holds no work in progress.
    if st.session_state["w_department"] == "finishing":
        st.session_state["w_wip"] = 0


for k, v in NORMAL.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("history", [])

with tab4:
    st.subheader("🔮 What-if: will this team stop the line today?")
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load normal shift", on_click=load_preset, args=(NORMAL,), use_container_width=True)
    b2.button("Load high-risk shift", on_click=load_preset, args=(HIGH_RISK,), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 🎛️ Shift parameters")
        g1, g2 = st.columns(2)
        g1.selectbox("Department", ["sewing", "finishing"], key="w_department", on_change=on_department_change)
        g2.selectbox("Team", list(range(1, 13)), key="w_team")
        g1.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"], key="w_day")
        g2.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"], key="w_quarter")
        st.slider("SMV · standard minutes per garment", 2.0, 55.0, step=0.5, key="w_smv")
        st.slider("WIP · work in progress (units)", 0, 3000, step=10, key="w_wip",
                  disabled=st.session_state["w_department"] == "finishing",
                  help="Finishing always has WIP = 0 (business rule).")
        st.slider("Incentive (BDT)", 0, 150, step=1, key="w_incentive")
        st.slider("Overtime (minutes)", 0, 15000, step=60, key="w_over_time")
        st.slider("Number of workers", 2, 90, step=1, key="w_workers")
        st.slider("Targeted productivity", 0.35, 0.80, step=0.05, key="w_target")
        with st.expander("Disruptions (idle time, idle workers, style changes)"):
            st.slider("Idle time (minutes)", 0, 300, step=5, key="w_idle_time")
            st.slider("Idle workers", 0, 45, step=1, key="w_idle_men")
            st.slider("Style changes", 0, 2, step=1, key="w_style")

    s = st.session_state
    scenario = pd.DataFrame([{
        "department": s.w_department, "day": s.w_day, "quarter": s.w_quarter, "team": s.w_team,
        "targeted_productivity": s.w_target, "smv": s.w_smv,
        "wip": 0 if s.w_department == "finishing" else s.w_wip,
        "over_time": s.w_over_time, "incentive": s.w_incentive, "idle_time": s.w_idle_time,
        "idle_men": s.w_idle_men, "no_of_style_change": s.w_style, "no_of_workers": s.w_workers,
    }])[FEATURES]

    with right:
        st.markdown(f"#### 🎯 Live inference · {active_model}")
        all_p = {name: float(m.predict_proba(scenario)[0, 1]) for name, m in models.items()}
        prob = all_p[active_model]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": "Probability of missing the target"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#F8F9FA"},
                   "steps": [{"range": [0, yellow * 100], "color": "#2E9E5B"},
                             {"range": [yellow * 100, red * 100], "color": "#E0A100"},
                             {"range": [red * 100, 100], "color": "#C8372D"}]}))
        gauge.update_layout(height=260, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)
        st.progress(prob)

        if prob >= red:
            st.error(f"🛑 HIGH RISK ALERT · LINE STOPPAGE ({prob:.0%} ≥ {red:.2f})")
            st.markdown("**Floor manager policy:** send the technical supervisor before mid-shift, rebalance "
                        "operators across stations, and authorize overtime only if WIP justifies it.")
        elif prob >= yellow:
            st.warning(f"⚠️ WATCH · ELEVATED RISK ({prob:.0%})")
            st.markdown("**Floor manager policy:** check the team's incentive; if it is zero, evaluate assigning "
                        "it for today and re-check at mid-shift.")
        else:
            st.success(f"✅ NORMAL OPERATION ({prob:.0%} < {yellow:.2f})")
            st.markdown("**Floor manager policy:** no intervention, standard monitoring.")

        cluster = int(kmeans.predict(scenario[meta["cluster_features"]])[0])
        info = meta["kmeans"]["clusters"][str(cluster)]
        st.info(f"**K-Means scenario:** {info['name']} · historically {info['miss_rate']:.0%} of "
                f"{info['n']} similar shifts missed the target.")

        st.markdown("**All loaded models on this scenario**")
        st.dataframe(pd.DataFrame({"Model": list(all_p), "Probability": list(all_p.values())})
                     .style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)

        if st.button("➕ Save scenario to comparison log", use_container_width=True):
            s.history.append({"SMV": s.w_smv, "WIP": int(scenario["wip"].iloc[0]), "Incentive": s.w_incentive,
                              "Dept": s.w_department, "Model": active_model, "Risk": f"{prob:.0%}",
                              "Light": "🔴" if prob >= red else "🟡" if prob >= yellow else "🟢"})
        if s.history:
            st.dataframe(pd.DataFrame(s.history), use_container_width=True, hide_index=True)

    with st.expander("Scenario as seen by the models (raw → scaled by the serialized scaler)"):
        st.dataframe(scenario, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(scaler.transform(scenario), columns=scaler.get_feature_names_out()).round(2),
                     use_container_width=True, hide_index=True)
