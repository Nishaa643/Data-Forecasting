import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backend.analyzer import analyze_dataset_and_recommend
from backend.trainer import run_transformer_forecast, compare_all_models

# 1. Page Configuration
st.set_page_config(
    page_title="Data Forecasting - AI Intelligence Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Dynamic Editorial Style Engine
# Dark Mode
bg_main = "#07080C"
card_bg = "#121520"
text_primary = "#F8FAFC"
text_muted = "#CBD5E1"
border_color = "#39445A"
plotly_template = "plotly_dark"
chart_text = "#F8FAFC"
line_hist = "#A8B4C7"
line_forecast = "#38BDF8"
line_base = "#FF6685"
hero_bg = "linear-gradient(135deg, #181514 0%, #292321 50%, #3D2314 100%)"
hero_border = "rgba(252, 211, 77, 0.4)"
hero_title = "#FCD34D"
hero_subtitle = "#FDE68A"
hero_badge_bg = "rgba(255, 255, 255, 0.08)"
hero_badge_border = "rgba(252, 211, 77, 0.35)"
hero_shadow = "0 15px 35px rgba(0, 0, 0, 0.3)"
uploader_button_bg = "#273247"
uploader_button_hover_bg = "#334155"
uploader_button_text = "#F8FAFC"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Overrides */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_main} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}
    
    [data-testid="stHeader"] {{ background: transparent !important; }}

    /* Improve contrast for native controls in both themes without changing their layout. */
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea,
    [data-testid="stAppViewContainer"] [data-baseweb="select"] > div,
    [data-testid="stAppViewContainer"] [data-baseweb="input"] > div,
    [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] {{
        color: {text_primary} !important;
        background-color: {card_bg} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stAppViewContainer"] input::placeholder,
    [data-testid="stAppViewContainer"] textarea::placeholder {{
        color: {text_muted} !important;
        opacity: 1 !important;
    }}
    [data-testid="stAppViewContainer"] [data-baseweb="select"] svg,
    [data-testid="stAppViewContainer"] [data-baseweb="input"] svg {{
        fill: {text_primary} !important;
        color: {text_primary} !important;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border-color: {border_color} !important;
    }}

    /* Universal Text Color Lock (Prevents fading on hover/cursor movement) */
    [data-testid="stAppViewContainer"] p, 
    [data-testid="stAppViewContainer"] span:not(.brand-badge):not(.card-value), 
    [data-testid="stAppViewContainer"] label, 
    [data-testid="stTabs"] button,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {{
        color: {text_primary} !important;
        fill: {text_primary} !important;
        opacity: 1 !important;
    }}

    /* Lock Expander and Container States Across Hover/Focus */
    [data-testid="stExpander"], 
    [data-testid="stExpander"] details, 
    [data-testid="stExpander"] summary {{
        background-color: {card_bg} !important;
        border-color: {border_color} !important;
    }}

    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus {{
        background-color: {card_bg} !important;
        color: {text_primary} !important;
    }}

    [data-testid="stExpander"] summary:hover span,
    [data-testid="stExpander"] summary:hover p {{
        color: {text_primary} !important;
    }}

    /* Modern App Metric Cards */
    .app-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.04);
    }}
    .card-label {{
        color: {text_muted};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }}
    .card-value {{
        color: {text_primary};
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'Cinzel Decorative', serif;
        margin-top: 4px;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}

    /* Action Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%) !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        border: none !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.8rem !important;
        box-shadow: 0 8px 25px rgba(217, 119, 6, 0.35) !important;
        transition: all 0.25s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(217, 119, 6, 0.5) !important;
    }}

    /* File Uploader Container */
    [data-testid="stFileUploader"] {{
        background-color: {bg_main} !important;
        border: 2px dashed {border_color} !important;
        border-radius: 16px !important;
        padding: 16px !important;
    }}

    /* Keep the native Upload button readable in default, hover, and focus states. */
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploader"] section button p,
    [data-testid="stFileUploader"] section button span,
    [data-testid="stFileUploader"] section button svg {{
        color: {uploader_button_text} !important;
        fill: {uploader_button_text} !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploader"] section button {{
        background-color: {uploader_button_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    [data-testid="stFileUploader"] section button:hover,
    [data-testid="stFileUploader"] section button:focus,
    [data-testid="stFileUploader"] section button:focus-visible,
    [data-testid="stFileUploader"] section button:active {{
        background-color: {uploader_button_hover_bg} !important;
        color: {uploader_button_text} !important;
        border-color: {border_color} !important;
        box-shadow: 0 0 0 2px {border_color} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# 4. Compact App Hero Header
st.markdown(f"""
    <div style="background: {hero_bg}; border: 1px solid {hero_border}; border-radius: 18px; padding: 18px 28px; margin-bottom: 24px; box-shadow: {hero_shadow}; display: flex; align-items: center; gap: 20px; width: 100%; position: relative; overflow: hidden;">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0;">
            <path d="M3 20H21" stroke="{hero_title}" stroke-width="2.8" stroke-linecap="round"/>
            <path d="M3 14L8 9L12 13L18 7L21 10" stroke="#38BDF8" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="18" cy="7" r="3" fill="{hero_title}"/>
            <circle cx="8" cy="9" r="3" fill="#38BDF8"/>
        </svg>
        <div style="flex-grow: 1; overflow: hidden;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 style="font-family: 'Cinzel Decorative', serif !important; font-size: 2.1rem !important; font-weight: 900 !important; color: {hero_title} !important; margin: 0 !important; line-height: 1.1 !important; white-space: nowrap !important; letter-spacing: 1.5px; text-shadow: 0 2px 10px rgba(0,0,0,0.16);">Data Forecasting</h1>
                    <div style="font-family: 'Cormorant Garamond', serif !important; font-size: 1rem !important; font-style: italic !important; font-weight: 700 !important; color: {hero_subtitle} !important; letter-spacing: 1px; margin-top: 2px;">Universal Time-Series & Deep Learning Prediction Platform</div>
                </div>
                <span style="background: {hero_badge_bg}; color: {hero_title}; font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; padding: 4px 10px; border-radius: 100px; border: 1px solid {hero_badge_border}; backdrop-filter: blur(8px);">STUDIO PRO</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar: data and model configuration
st.sidebar.markdown("### 1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"], key="dataset_uploader")
if uploaded_file is not None:
    try:
        temp_df = pd.read_csv(uploaded_file)
        for col in temp_df.select_dtypes(include="object").columns:
            cleaned = temp_df[col].astype(str).str.replace(r"[\$,]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() > 0.5 * len(temp_df):
                temp_df[col] = converted
        st.session_state["df_original"] = temp_df.copy()
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")

if "df_original" not in st.session_state:
    st.info("👈 Upload a CSV time-series dataset from the sidebar to initialize the forecasting workspace.")
    st.stop()

original_df = st.session_state["df_original"]
df = original_df.copy()
st.sidebar.markdown("### 2. Row Filtering")
filter_column = st.sidebar.selectbox(
    "Filter column (optional)", ["None"] + original_df.columns.tolist(), index=0,
    help="Leave as None for a continuous time series. Filtering can break chronological continuity."
)
if filter_column != "None":
    filter_values = original_df[filter_column].dropna().unique().tolist()
    selected_values = st.sidebar.multiselect(f"Values in {filter_column}", filter_values, default=[])
    if selected_values:
        df = original_df[original_df[filter_column].isin(selected_values)].copy()
st.sidebar.caption(f"Showing {len(df):,} of {len(original_df):,} rows")

with st.expander("🔍 Dataset Explorer", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

st.sidebar.markdown("### 3. Forecast Configuration")
col_names = df.columns.tolist()
time_col = st.sidebar.selectbox("Time/Date Axis", [None] + col_names, index=0)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numerical columns found in the uploaded dataset.")
    st.stop()
target_col = st.sidebar.selectbox("Target Variable", numeric_cols)
model_type = st.sidebar.selectbox("Transformer Architecture", ["Informer", "PatchTST", "FEDformer", "ETSformer", "Crossformer"])
lookback_len = st.sidebar.slider("Lookback Window", 12, 168, 24)
pred_len = st.sidebar.slider("Prediction Horizon", 6, 72, 12)
epochs = st.sidebar.slider("Training Epochs", 1, 30, 3)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

run_col, compare_col = st.sidebar.columns(2)
run_clicked = run_col.button("Run Model", type="primary", use_container_width=True)
compare_clicked = compare_col.button("Compare All", use_container_width=True)

if run_clicked:
    try:
        with st.spinner(f"Training {model_type} and evaluating on the held-out test set..."):
            from backend.trainer import run_transformer_forecast
            st.session_state["forecast_results"] = run_transformer_forecast(
                df, target_col, model_type, lookback_len, pred_len,
                time_col=time_col, epochs=epochs, batch_size=batch_size
            )
            st.session_state["forecast_config"] = {
                "model": model_type, "target": target_col,
                "lookback": lookback_len, "horizon": pred_len,
                "epochs": epochs, "batch_size": batch_size,
            }
        st.success(f"{model_type} training and test evaluation completed.")
    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")
        st.exception(exc)

if compare_clicked:
    try:
        from backend.trainer import compare_all_models
        with st.spinner("Training and evaluating all five models. This may take several minutes..."):
            rows, details = compare_all_models(
                df, target_col, lookback_len, pred_len,
                time_col=time_col, epochs=epochs, batch_size=batch_size
            )
            st.session_state["comparison_rows"] = rows
            st.session_state["comparison_details"] = details
        st.success("All five models have been evaluated.")
    except Exception as exc:
        st.error(f"Model comparison failed: {exc}")
        st.exception(exc)

# Dataset diagnostics can be viewed even before training.
from backend.analyzer import analyze_dataset_and_recommend
try:
    analysis = analyze_dataset_and_recommend(df, target_col)
except Exception as exc:
    analysis = None
    st.warning(f"Dataset analysis could not be completed: {exc}")

results = st.session_state.get("forecast_results")
comparison_rows = st.session_state.get("comparison_rows")

if results is not None:
    config = st.session_state.get("forecast_config", {})
    st.markdown("## Forecast Overview")
    a, b, c = st.columns(3)
    a.metric("Active Architecture", config.get("model", model_type))
    b.metric("Evaluated Target Records", f"{analysis['dataset_summary']['total_samples']:,}" if analysis else "—")
    variability = analysis["dataset_summary"]["variability_score"] if analysis else float("nan")
    c.metric("Variability Coefficient", f"{variability:.3f}" if np.isfinite(variability) else "N/A")

    metrics = results["metrics"]
    st.markdown("### Held-out Test Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:.4f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.4f}")
    m3.metric("MAPE", f"{metrics['MAPE']:.2f}%" if np.isfinite(metrics["MAPE"]) else "N/A")
    m4.metric("R² Score", f"{metrics['R2']:.4f}" if np.isfinite(metrics["R2"]) else "N/A")
    st.caption(f"Training time: {results['training_time']:.2f}s · Device: {results['device']} · Test windows: {results['split_info']['total_rows'] - results['split_info']['val_end']:,} test rows")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Forecast Horizon", "🎯 Actual vs Predicted", "📉 Model Comparison",
        "🔍 Dataset Analysis", "💡 Neural Recommendations", "⚙️ Training Insights"
    ])

    with tab1:
        fig = go.Figure()
        x_hist = list(range(len(results["history"])))
        x_future = list(range(len(x_hist), len(x_hist) + len(results["future_forecast"])))
        fig.add_trace(go.Scatter(x=x_hist, y=results["history"], mode="lines", name="Historical Input", line=dict(color=line_hist, width=2.5)))
        fig.add_trace(go.Scatter(x=x_future, y=results["future_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_future, y=results["future_lower"], mode="lines", fill="tonexty", fillcolor="rgba(56,189,248,0.15)", line=dict(width=0), name="Empirical 90% error band"))
        fig.add_trace(go.Scatter(x=x_future, y=results["future_forecast"], mode="lines+markers", name=f"{results['model']} Forecast", line=dict(color=line_forecast, width=3)))
        fig.update_layout(title="Historical Input and Future Forecast", xaxis_title="Time Steps", yaxis_title=target_col, template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_text), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The shaded band is an empirical error band derived from the 90th percentile of absolute held-out test residuals. It is an approximate uncertainty indicator, not a formal calibrated prediction interval.")

        base = go.Figure()
        base.add_trace(go.Scatter(y=results["actual_test"], mode="lines", name="Actual test values"))
        base.add_trace(go.Scatter(y=results["naive_test"], mode="lines", name="Naive test forecast", line=dict(dash="dash")))
        base.update_layout(title="Persistence Baseline on Test Set", xaxis_title="Flattened Test Forecast Points", yaxis_title=target_col, template=plotly_template)
        st.plotly_chart(base, use_container_width=True)

    with tab2:
        actual_fig = go.Figure()
        actual_fig.add_trace(go.Scatter(y=results["actual_test"], mode="lines", name="Actual", line=dict(color=line_hist, width=2)))
        actual_fig.add_trace(go.Scatter(y=results["predicted_test"], mode="lines", name="Predicted", line=dict(color=line_forecast, width=2)))
        actual_fig.update_layout(title="Actual vs Predicted on Held-out Test Windows", xaxis_title="Flattened Test Forecast Points", yaxis_title=target_col, template=plotly_template)
        st.plotly_chart(actual_fig, use_container_width=True)
        st.caption("Each test input window predicts a full horizon. Overlapping horizons are retained in this evaluation, so some timestamps appear more than once.")
        error = np.asarray(results["actual_test"]) - np.asarray(results["predicted_test"])
        err_fig = go.Figure(go.Histogram(x=error, nbinsx=35, name="Prediction Error"))
        err_fig.update_layout(title="Test Prediction Error Distribution", xaxis_title="Actual − Predicted", yaxis_title="Frequency", template=plotly_template)
        st.plotly_chart(err_fig, use_container_width=True)

    with tab3:
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}", "R2": "{:.4f}", "Training Time (s)": "{:.2f}", "Naive RMSE": "{:.4f}"}), use_container_width=True)
            best = comparison_df.iloc[0]
            st.success(f"Lowest test RMSE: {best['Model']} ({best['RMSE']:.4f})")
            bar = go.Figure(go.Bar(x=comparison_df["Model"], y=comparison_df["RMSE"], name="Test RMSE"))
            bar.update_layout(title="Test RMSE by Transformer Architecture (lower is better)", xaxis_title="Model", yaxis_title="RMSE", template=plotly_template)
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("Click **Compare All** in the sidebar to train and evaluate FEDformer, PatchTST, Informer, ETSformer and Crossformer on the same chronological split.")

    with tab4:
        if analysis:
            summary = analysis["dataset_summary"]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Mean", f"{summary['mean']:.4f}")
            s2.metric("Median", f"{summary['median']:.4f}")
            s3.metric("Std. Deviation", f"{summary['std']:.4f}")
            s4.metric("IQR Outliers", f"{summary['outlier_count']}")
            missing_df = pd.DataFrame({"Column": list(summary["missing_values"].keys()), "Missing Values": list(summary["missing_values"].values())})
            st.markdown("#### Missing values by column")
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
            st.write(f"Duplicate rows: **{summary['duplicate_rows']:,}** · Estimated linear trend: **{summary['trend']}**")

        target_series = pd.to_numeric(df[target_col], errors="coerce")
        hist_fig = go.Figure(go.Histogram(x=target_series.dropna(), nbinsx=40, name="Distribution"))
        hist_fig.update_layout(title=f"Distribution of {target_col}", xaxis_title=target_col, yaxis_title="Frequency", template=plotly_template)
        st.plotly_chart(hist_fig, use_container_width=True)

        rolling = target_series.rolling(window=max(3, min(24, len(target_series) // 5 or 3)), min_periods=1).mean()
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(y=target_series, mode="lines", name="Observed", opacity=0.45))
        trend_fig.add_trace(go.Scatter(y=rolling, mode="lines", name="Rolling Mean", line=dict(width=3)))
        trend_fig.update_layout(title="Target Trend and Rolling Mean", xaxis_title="Time Steps", yaxis_title=target_col, template=plotly_template)
        st.plotly_chart(trend_fig, use_container_width=True)

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr(numeric_only=True)
            heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1, colorbar_title="Correlation"))
            heat.update_layout(title="Numerical Feature Correlation", template=plotly_template)
            st.plotly_chart(heat, use_container_width=True)
        box = go.Figure(go.Box(y=target_series.dropna(), name=target_col, boxpoints="outliers"))
        box.update_layout(title=f"Outlier Inspection: {target_col}", yaxis_title=target_col, template=plotly_template)
        st.plotly_chart(box, use_container_width=True)

    with tab5:
        if analysis:
            for rec in analysis["recommendations"]:
                with st.expander(f"✨ {rec['model']}"):
                    st.write(rec["reason"])
        if comparison_rows:
            st.markdown("#### Evidence from model evaluation")
            st.write(f"The lowest test RMSE in the latest comparison was **{comparison_rows[0]['Model']}**. This is a result for the selected dataset and configuration, not a universal ranking of Transformer architectures.")
        else:
            st.info("Run Compare All to add recommendations based on measured test performance.")

    with tab6:
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=results["training_loss"], mode="lines+markers", name="Training Loss"))
        loss_fig.add_trace(go.Scatter(y=results["validation_loss"], mode="lines+markers", name="Validation Loss"))
        loss_fig.update_layout(title="Training vs Validation MSE", xaxis_title="Epoch", yaxis_title="MSE Loss", template=plotly_template)
        st.plotly_chart(loss_fig, use_container_width=True)
        details = pd.DataFrame({"Setting": ["Architecture", "Lookback", "Prediction Horizon", "Epochs", "Batch Size", "Device", "Training Time (seconds)"], "Value": [results["model"], config.get("lookback"), config.get("horizon"), config.get("epochs"), config.get("batch_size"), results["device"], round(results["training_time"], 2)]})
        st.dataframe(details, use_container_width=True, hide_index=True)

else:
    st.markdown("## Ready to Forecast")
    st.info("Choose your target, Transformer architecture and hyperparameters in the sidebar, then click **Run Model**. Use **Compare All** to benchmark all five architectures.")
    if analysis:
        st.markdown("### Quick Dataset Profile")
        summary = analysis["dataset_summary"]
        p1, p2, p3 = st.columns(3)
        p1.metric("Valid Target Rows", f"{summary['total_samples']:,}")
        p2.metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")
        p3.metric("Outlier Candidates", f"{summary['outlier_count']:,}")
