from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(
    page_title="Semiconductor Fab · Etch Rate & Line Halt AI (Team 2)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 2.05rem; font-weight: 700; margin-bottom: .1rem;}
    .sub-header {font-size: 1.02rem; color: #ADB5BD; margin-bottom: 1.2rem;}
    div[data-testid="stMetric"] {background: #1E2125; border: 1px solid #343A40; border-radius: 10px; padding: 12px 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "exam2_team2_semiconductor.csv"
BEST_MODEL_PATH = ROOT / "models" / "best_model.joblib"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
TARGET = "etch_rate_nm_min"

# 2. CACHED DATA & SERIALIZED MODEL LOADING
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_artifacts():
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

df = load_data()
ann_model, scaler = load_artifacts()

# Train/test split matching Task 1 & Task 2 for live multi-model benchmark
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Sidebar status & metadata
with st.sidebar:
    st.header("⚙️ Production Artifacts")
    st.success("Loaded `models/best_model.joblib`")
    st.success("Loaded `models/scaler.joblib`")
    st.markdown("---")
    st.markdown("**Team 2 · Semiconductor Fab**")
    st.caption(f"Dataset: `exam2_team2_semiconductor.csv` ({len(df)} wafers)")
    st.caption("Target: `etch_rate_nm_min` (nm/min)")
    st.caption("Architecture: MLPRegressor (64, 32) · ReLU · Early Stopping")

st.markdown('<div class="main-header">🔬 Semiconductor Plasma Etching · Line Halt AI Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Team 2 · Midterm Exam Part 2 · Dynamic Serialized ANN & OLS Regression Suite</div>',
    unsafe_allow_html=True,
)

# 4-TAB EXECUTIVE DASHBOARD ARCHITECTURE
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Data Explorer & EDA",
    "2️⃣ Preprocessing & Scaling Proof",
    "3️⃣ Multi-Model Benchmark Suite",
    "4️⃣ Real-Time What-If Inference (Line Halt)",
])

# ==============================================================================
# TAB 1: DATA EXPLORER & EDA
# ==============================================================================
with tab1:
    st.subheader("Wafer Telemetry Overview & Statistical Profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Wafer Runs", f"{len(df):,}")
    c2.metric("Input Features", f"{X.shape[1]}")
    c3.metric("Mean Etch Rate", f"{y.mean():.2f} nm/min")
    c4.metric("Max Etch Rate", f"{y.max():.2f} nm/min")

    st.dataframe(df.head(15), use_container_width=True)
    st.markdown("#### Descriptive Statistics (`df.describe()`)")
    st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            df, x=TARGET, nbins=30, marginal="box",
            title="Target Distribution: Etch Rate (nm/min)",
            color_discrete_sequence=["#E8641C"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        fig_scat = px.scatter(
            df, x="rf_power_watts", y=TARGET, color="gas_flow_sccm",
            title="RF Power vs. Etch Rate (Colored by Gas Flow sccm)",
            color_continuous_scale="Turbo",
        )
        st.plotly_chart(fig_scat, use_container_width=True)

# ==============================================================================
# TAB 2: PREPROCESSING PIPELINE & SCALING PROOF
# ==============================================================================
with tab2:
    st.subheader("StandardScaler Normalization & Feature Correlation Matrix")
    st.info("Mandatory ANN Rule: `models/scaler.joblib` transforms raw physical units (Watts, mTorr, sccm, °C, s) to Mean = 0, Std = 1.")

    X_scaled_df = pd.DataFrame(scaler.transform(X), columns=X.columns)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Raw Feature Scales (Before Scaling)**")
        raw_stats = pd.DataFrame({"Mean": X.mean(), "Std Dev": X.std(), "Min": X.min(), "Max": X.max()})
        st.dataframe(raw_stats.style.format("{:.2f}"), use_container_width=True)
    with c_right:
        st.markdown("**Normalized Scales (Loaded from `scaler.joblib`)**")
        scaled_stats = pd.DataFrame({"Mean": X_scaled_df.mean(), "Std Dev": X_scaled_df.std(), "Min": X_scaled_df.min(), "Max": X_scaled_df.max()})
        st.dataframe(scaled_stats.style.format("{:.3f}"), use_container_width=True)

    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature & Target Pearson Correlation Heatmap",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# TAB 3: MULTI-MODEL BENCHMARK SUITE
# ==============================================================================
with tab3:
    st.subheader("Holdout Test Set Performance Comparison (OLS vs. Poly vs. Serialized ANN)")

    def adj_r2(r2, n, p):
        return 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))

    n_test, p_orig = len(y_test), X_test.shape[1]

    # 1. Baseline OLS
    m_ols = LinearRegression().fit(X_train, y_train)
    p_ols = m_ols.predict(X_test)

    # 2. Log-Transformed OLS
    m_log = LinearRegression().fit(X_train, np.log1p(y_train))
    p_log = np.expm1(m_log.predict(X_test))

    # 3. Polynomial Deg 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_tr_p = poly.fit_transform(X_train)
    X_te_p = poly.transform(X_test)
    m_poly = LinearRegression().fit(X_tr_p, y_train)
    p_poly = m_poly.predict(X_te_p)

    # 4. Serialized ANN (loaded from best_model.joblib + scaler.joblib)
    p_ann = ann_model.predict(scaler.transform(X_test))

    bench = []
    for name, preds, p_cnt in [
        ("Baseline OLS Linear", p_ols, p_orig),
        ("Log-Transformed Target OLS", p_log, p_orig),
        ("Polynomial Regression (Deg 2)", p_poly, X_te_p.shape[1]),
        ("Serialized ANN MLPRegressor (64, 32)", p_ann, p_orig),
    ]:
        r2 = r2_score(y_test, preds)
        bench.append({
            "Model": name,
            "Test R²": round(r2, 4),
            "Adjusted R²": round(adj_r2(r2, n_test, p_cnt), 4),
            "MAE (nm/min)": round(mean_absolute_error(y_test, preds), 4),
            "RMSE (nm/min)": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        })

    bench_df = pd.DataFrame(bench)
    st.dataframe(bench_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_pred = px.scatter(
            x=y_test, y=p_ann,
            labels={"x": "Actual Etch Rate (nm/min)", "y": "ANN Predicted Etch Rate (nm/min)"},
            title="Serialized ANN (`best_model.joblib`): Actual vs. Predicted",
            color_discrete_sequence=["#2E9E5B"],
        )
        fig_pred.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                           line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_pred, use_container_width=True)

    with col2:
        fig_loss = px.line(
            x=list(range(1, len(ann_model.loss_curve_) + 1)),
            y=ann_model.loss_curve_,
            labels={"x": "Epoch", "y": "Training Loss"},
            title=f"ANN Training Loss Curve ({ann_model.n_iter_} Epochs Converged)",
        )
        fig_loss.update_traces(line_color="#E8641C")
        st.plotly_chart(fig_loss, use_container_width=True)

# ==============================================================================
# TAB 4: REAL-TIME WHAT-IF SCENARIO INFERENCE (LINE HALT POLICY)
# ==============================================================================
with tab4:
    st.subheader("Real-Time Plasma Chamber Control · What-If Line Halt Simulator")
    st.markdown("Adjust **Gas Flow (`gas_flow_sccm`)** and **RF Power (`rf_power_watts`)** to simulate live wafer etching and evaluate line-halt safety thresholds.")

    s_col1, s_col2 = st.columns([1.1, 1.3])

    with s_col1:
        st.markdown("#### 🎛️ Chamber Process Controls")
        rf_power = st.slider(
            "RF Power (Watts) [Primary Control]",
            min_value=float(df["rf_power_watts"].min()),
            max_value=float(df["rf_power_watts"].max()),
            value=750.0,
            step=5.0,
        )
        gas_flow = st.slider(
            "Gas Flow (sccm) [Primary Control]",
            min_value=float(df["gas_flow_sccm"].min()),
            max_value=float(df["gas_flow_sccm"].max()),
            value=170.0,
            step=2.0,
        )
        pressure = st.slider(
            "Chamber Pressure (mTorr)",
            min_value=float(df["chamber_pressure_mTorr"].min()),
            max_value=float(df["chamber_pressure_mTorr"].max()),
            value=45.0,
            step=0.5,
        )
        wafer_temp = st.slider(
            "Wafer Temperature (°C)",
            min_value=float(df["wafer_temp_c"].min()),
            max_value=float(df["wafer_temp_c"].max()),
            value=175.0,
            step=1.0,
        )
        etch_time = st.slider(
            "Etch Time (s)",
            min_value=float(df["etch_time_s"].min()),
            max_value=float(df["etch_time_s"].max()),
            value=105.0,
            step=1.0,
        )

    # Build real-time input vector matching exact column order
    input_df = pd.DataFrame([{
        "rf_power_watts": rf_power,
        "chamber_pressure_mTorr": pressure,
        "gas_flow_sccm": gas_flow,
        "wafer_temp_c": wafer_temp,
        "etch_time_s": etch_time,
    }])

    # Scale with scaler.joblib and predict with best_model.joblib
    input_scaled = scaler.transform(input_df)
    pred_etch_rate = float(ann_model.predict(input_scaled)[0])

    # Safe operating window: 120 <= etch_rate <= 265 nm/min
    LOW_LIMIT = 120.0
    HIGH_LIMIT = 265.0
    is_line_halt = (pred_etch_rate > HIGH_LIMIT) or (pred_etch_rate < LOW_LIMIT)

    with s_col2:
        st.markdown("#### 🚨 Live Inference & Line Halt Decision")
        st.metric(
            label="Predicted Etch Rate (`best_model.joblib`)",
            value=f"{pred_etch_rate:.2f} nm/min",
            delta=f"{pred_etch_rate - y.mean():+.2f} nm/min vs. fab baseline",
        )

        # Gauge visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_etch_rate,
            title={"text": "Etch Rate Safety Envelope (120 – 265 nm/min)"},
            gauge={
                "axis": {"range": [0, 400]},
                "bar": {"color": "#E8641C" if is_line_halt else "#2E9E5B"},
                "steps": [
                    {"range": [0, LOW_LIMIT], "color": "rgba(232, 100, 28, 0.25)"},
                    {"range": [LOW_LIMIT, HIGH_LIMIT], "color": "rgba(46, 158, 91, 0.25)"},
                    {"range": [HIGH_LIMIT, 400], "color": "rgba(232, 100, 28, 0.35)"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": HIGH_LIMIT},
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if is_line_halt:
            reason = (
                f"Over-etching thermal/plasma runaway risk (>{HIGH_LIMIT} nm/min)"
                if pred_etch_rate > HIGH_LIMIT
                else f"Under-etching incomplete pattern transfer (<{LOW_LIMIT} nm/min)"
            )
            st.error(
                f"🛑 **HIGH-RISK ALERT · TRIGGER LINE HALT**\n\n"
                f"Predicted Etch Rate **{pred_etch_rate:.2f} nm/min** breached safe wafer bounds "
                f"({LOW_LIMIT}–{HIGH_LIMIT} nm/min) due to **{reason}**. "
                f"Immediate plasma chamber interlock engaged."
            )
        else:
            st.success(
                f"✅ **NORMAL OPERATION · LINE CLEARED**\n\n"
                f"Predicted Etch Rate **{pred_etch_rate:.2f} nm/min** is within nominal specification "
                f"({LOW_LIMIT}–{HIGH_LIMIT} nm/min). Wafer lot approved to continue etching."
            )
            